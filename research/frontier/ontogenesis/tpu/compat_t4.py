"""Compatibility, correctness and throughput test for the Gated DeltaNet conversion route on Kaggle T4 x2.
Runs as a Kaggle script (no weights downloaded; corpus slice from the attached dataset). Writes /kaggle/working/compat.json.
Every stage is independent: a failure is recorded with its traceback and the next stage still runs. Only non-sensitive
facts are recorded (versions, device names, numbers); environment variables are never dumped.

Stages
  install      pip install flash-linear-attention / fla-core without touching torch
  env          torch, CUDA, triton, fla versions; per-GPU name, compute capability, memory; bf16 support
  import_fla   the kernels and layer the conversion would use
  correct      chunk_gated_delta_rule forward and backward against a float32 reference (the delta rule as written in
               arXiv 2412.06464), in float32 and float16, errors relative to the reference's largest value
  bench_gdn    24-layer, width-896 language model (Qwen2.5-0.5B-shaped, 151,936 vocabulary) with Gated DeltaNet mixers,
  bench_attn   the same with softmax attention (torch SDPA): fp16 autocast + GradScaler, AdamW, micro-batch 4 x 1024,
               peak memory per GPU, throughput after warm-up, and throughput as a fraction of the T4's 65 TFLOPS fp16
  stability    300 optimisation steps of the GDN model on the corpus slice: loss, GradScaler scale, skipped steps, NaNs
  allreduce    2-GPU NCCL all-reduce of 256 MB over PCIe (gradient synchronisation cost for data parallelism)
  cache_io     read speed for a top-32 teacher-logit cache (int32 index + fp16 value per entry)
"""
import os, sys, json, time, subprocess, traceback, glob, math

R = {"stages": {}}
OUT = "/kaggle/working/compat.json"


def save():
    json.dump(R, open(OUT, "w"), indent=1, default=str)


def stage(name):
    def deco(fn):
        t0 = time.time()
        try:
            R["stages"][name] = {"ok": True, "result": fn()}
        except Exception as e:
            R["stages"][name] = {"ok": False, "error": repr(e)[:800], "trace": traceback.format_exc()[-1800:]}
        R["stages"][name]["seconds"] = round(time.time() - t0, 1)
        save(); print(name, "ok" if R["stages"][name]["ok"] else "FAILED " + R["stages"][name]["error"][:200], flush=True)
        return fn
    return deco


@stage("install")
def _():
    out = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention", "fla-core"], capture_output=True, text=True)
    return {"rc": out.returncode, "tail": (out.stdout + out.stderr)[-500:]}


@stage("env")
def _():
    import torch, importlib.metadata as md
    d = {"torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": torch.cuda.device_count(), "bf16_supported": torch.cuda.is_bf16_supported()}
    d["gpus"] = [{"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i)),
                  "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 2 ** 30, 1)} for i in range(torch.cuda.device_count())]
    for p in ("triton", "fla-core", "flash-linear-attention", "transformers", "einops"):
        try: d[p] = md.version(p)
        except Exception: d[p] = None
    return d


@stage("import_fla")
def _():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.layers import GatedDeltaNet
    return {"imported": ["fla.ops.gated_delta_rule.chunk_gated_delta_rule", "fla.layers.GatedDeltaNet"]}


import torch
import torch.nn as nn
import torch.nn.functional as F


def ref_gdn(q, k, v, g, beta, scale):
    """Delta rule with scalar gate (arXiv 2412.06464): S <- exp(g) S; S <- S + k (beta (v - S^T k))^T; o = S^T (scale q).
    q, k: B T H K; v: B T H V; g, beta: B T H; float32."""
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1]); outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kv = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], (v[:, t] - kv) * beta[:, t][..., None])
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(outs, 1)


def rel(a, b):
    return float(((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item())


def correctness(dtype, B=2, T=256, H=4, K=64, V=64):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    torch.manual_seed(0); dev = "cuda"
    q = F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1); k = F.normalize(torch.randn(B, T, H, K, device=dev), dim=-1)
    v = torch.randn(B, T, H, V, device=dev); g = F.logsigmoid(torch.randn(B, T, H, device=dev) + 2.0); beta = torch.rand(B, T, H, device=dev)
    ins = [q.to(dtype), k.to(dtype), v.to(dtype), g, beta.to(dtype)]
    kin = [x.detach().clone().requires_grad_(True) for x in ins]
    rin = [x.detach().clone().float().requires_grad_(True) for x in ins]
    scale = K ** -0.5
    ko = chunk_gated_delta_rule(*kin, scale=scale); ko = ko[0] if isinstance(ko, tuple) else ko
    ro = ref_gdn(*rin, scale)
    w = torch.randn_like(ro)
    (ko.float() * w).sum().backward(); (ro * w).sum().backward()
    res = {"dtype": str(dtype), "output_rel_err": rel(ko.float(), ro), "output_has_nan": bool(torch.isnan(ko).any())}
    for nm, a, b in zip("qkvgb", kin, rin):
        res["grad_" + nm + "_rel_err"] = rel(a.grad.float(), b.grad); res["grad_" + nm + "_nan"] = bool(torch.isnan(a.grad).any())
    return res


@stage("correct")
def _():
    return [correctness(torch.float32), correctness(torch.float16)]


class Block(nn.Module):
    def __init__(s, d, kind, ffn, heads):
        super().__init__(); s.kind = kind; s.n1 = nn.RMSNorm(d); s.n2 = nn.RMSNorm(d); s.heads = heads
        if kind == "gdn":
            from fla.layers import GatedDeltaNet
            s.mix = GatedDeltaNet(hidden_size=d, num_heads=heads, head_dim=d // heads, expand_v=1, mode="chunk")
        else:
            s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False)
        s.gu = nn.Linear(d, 2 * ffn, bias=False); s.dn = nn.Linear(ffn, d, bias=False)

    def forward(s, x):
        h = s.n1(x)
        if s.kind == "gdn":
            y = s.mix(h); y = y[0] if isinstance(y, tuple) else y
        else:
            B, T, d = h.shape
            q, k, v = s.qkv(h).view(B, T, 3, s.heads, d // s.heads).permute(2, 0, 3, 1, 4)
            y = s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d))
        x = x + y
        g, u = s.gu(s.n2(x)).chunk(2, -1)
        return x + s.dn(F.silu(g) * u)


class LM(nn.Module):
    def __init__(s, kind, d=896, L=24, ffn=4864, heads=7, vocab=151936):
        super().__init__(); s.emb = nn.Embedding(vocab, d); s.blocks = nn.ModuleList([Block(d, kind, ffn, heads) for _ in range(L)]); s.nf = nn.RMSNorm(d)
        nn.init.normal_(s.emb.weight, std=0.02)

    def loss(s, ids, ckpt=False, chunk=1024):
        x = s.emb(ids[:, :-1])
        for b in s.blocks:
            x = torch.utils.checkpoint.checkpoint(b, x, use_reentrant=False) if ckpt else b(x)
        x = s.nf(x).reshape(-1, x.shape[-1]); y = ids[:, 1:].reshape(-1); tot = 0.0
        for i in range(0, x.shape[0], chunk):
            tot = tot + torch.utils.checkpoint.checkpoint(lambda xc, yc: F.cross_entropy((xc @ s.emb.weight.T).float(), yc, reduction="sum"), x[i:i + chunk], y[i:i + chunk], use_reentrant=False)
        return tot / y.numel()


def get_batch(data, B, T, rng):
    import numpy as np
    ix = rng.integers(0, len(data) - T - 1, B)
    return torch.from_numpy(np.stack([data[i:i + T + 1].astype("int64") for i in ix])).cuda() % 151936


def bench(kind, B=4, T=1024, steps=20, ckpt=False, data=None):
    import numpy as np
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = LM(kind).cuda(); n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True); scaler = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(0)
    if data is None: data = rng.integers(0, 32768, 10_000_000).astype("uint16")
    for i in range(steps + 5):
        if i == 5: torch.cuda.synchronize(); t0 = time.time()
        ids = get_batch(data, B, T, rng)
        with torch.autocast("cuda", dtype=torch.float16): l = m.loss(ids, ckpt)
        scaler.scale(l).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dt = (time.time() - t0) / steps; tps = B * T / dt
    res = {"kind": kind, "params_M": round(n_params / 1e6, 1), "micro_batch": f"{B}x{T}", "layer_checkpointing": ckpt, "sec_per_step": round(dt, 3),
           "tokens_per_s": round(tps), "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           "model_flops_utilisation_vs_65TF": round(6 * n_params * tps / 65e12, 3), "last_loss": round(float(l), 3)}
    del m, opt; torch.cuda.empty_cache(); return res


def bench_all(kind):
    out = []
    for B, ck in ((4, False), (2, False), (4, True), (2, True)):
        try:
            r = bench(kind, B=B, ckpt=ck); out.append(r)
            if not ck: break
        except torch.cuda.OutOfMemoryError as e:
            out.append({"kind": kind, "micro_batch": f"{B}x1024", "layer_checkpointing": ck, "oom": True}); torch.cuda.empty_cache()
    return out


@stage("bench_gdn")
def _():
    return bench_all("gdn")


@stage("bench_attn")
def _():
    return bench_all("attn")


@stage("stability")
def _():
    import numpy as np
    files = glob.glob("/kaggle/input/**/train.bin", recursive=True)
    data = np.memmap(files[0], dtype=np.uint16, mode="r") if files else None
    torch.manual_seed(0); m = LM("gdn").cuda(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    scaler = torch.amp.GradScaler("cuda"); rng = np.random.default_rng(1); log = []; skipped = 0; nan = 0
    B = 2 if any(r.get("oom") and r["micro_batch"] == "4x1024" and not r["layer_checkpointing"] for r in R["stages"].get("bench_gdn", {}).get("result", [])) else 4
    for step in range(300):
        for gp in opt.param_groups: gp["lr"] = 3e-4 * min(1, (step + 1) / 50)
        ids = get_batch(data, B, 1024, rng) if data is not None else torch.randint(0, 32768, (B, 1025)).cuda()
        with torch.autocast("cuda", dtype=torch.float16): l = m.loss(ids, ckpt=False)
        if not torch.isfinite(l): nan += 1
        scaler.scale(l).backward(); sc0 = scaler.get_scale(); scaler.unscale_(opt)
        gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)); scaler.step(opt); scaler.update()
        if scaler.get_scale() < sc0: skipped += 1
        opt.zero_grad(set_to_none=True)
        if step % 25 == 0 or step == 299: log.append([step, round(float(l), 3), round(gn, 3), scaler.get_scale()])
    return {"used_corpus_slice": data is not None, "steps": 300, "nonfinite_loss_steps": nan, "scaler_skipped_steps": skipped, "log[step,loss,gradnorm,scale]": log,
            "loss_start": log[0][1], "loss_end": log[-1][1], "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}


AR = '''
import os, time, json, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(r)
x = torch.randn(128 * 2 ** 20, device="cuda", dtype=torch.float16)
for _ in range(3): dist.all_reduce(x)
torch.cuda.synchronize(); t = time.time()
for _ in range(10): dist.all_reduce(x)
torch.cuda.synchronize(); dt = (time.time() - t) / 10
if r == 0: open("/kaggle/working/allreduce.json", "w").write(json.dumps({"size_MiB": 256, "seconds": round(dt, 4), "GB_per_s": round(0.268 / dt, 2)}))
'''


@stage("allreduce")
def _():
    open("/kaggle/working/ar.py", "w").write(AR)
    p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "/kaggle/working/ar.py"], capture_output=True, text=True, timeout=300)
    r = json.load(open("/kaggle/working/allreduce.json")) if os.path.exists("/kaggle/working/allreduce.json") else {"failed": (p.stdout + p.stderr)[-600:]}
    r["gradient_bytes_for_0.5B_fp16_MiB"] = 953; return r


@stage("cache_io")
def _():
    import numpy as np
    n = 1_000_000; path = "/kaggle/working/cache.bin"
    arr = np.zeros((n, 32), dtype=[("i", "<i4"), ("v", "<f2")]); arr.tofile(path)
    t = time.time(); a = np.fromfile(path, dtype=arr.dtype); seq = time.time() - t
    mm = np.memmap(path, dtype=arr.dtype, mode="r", shape=(n, 32)); rng = np.random.default_rng(0); t = time.time()
    for _ in range(200): s = int(rng.integers(0, n - 4096)); _x = np.asarray(mm[s:s + 4096])
    rnd = time.time() - t; os.remove(path)
    mb = n * 32 * 6 / 2 ** 20
    return {"cache_MiB_per_1M_tokens": round(mb), "sequential_read_MiB_per_s": round(mb / seq), "random_4096_token_reads_per_s": round(200 / rnd),
            "tokens_per_s_supported_at_random_reads": round(200 / rnd * 4096)}


print(json.dumps({k: v.get("ok") for k, v in R["stages"].items()}))
