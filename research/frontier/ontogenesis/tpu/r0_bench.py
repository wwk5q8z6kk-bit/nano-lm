"""R0: bounded measurement of the Gated DeltaNet training path on one rented GPU (RESEARCH-PLAN.md section 20.6).
Approved by the owner on 2026-11-24 as one run with a hard US$6 cap including compute, storage and transfer.

It measures, on the workload that stage C would train (24 layers, width 896, 7 heads of 128, SwiGLU 4,864, vocabulary 151,936,
547M parameters, AdamW with fp32 master weights and autocast), not just a compatibility check. For the recurrent model and a
softmax-attention baseline under the same conditions it records:
  - environment, and the GPU's measured ceilings (large-GEMM TFLOPS and copy bandwidth) next to the datasheet figures
  - kernel correctness against a float32 reference (the recurrence of arXiv 2412.06464)
  - for a grid of micro-batch x sequence configurations: first-step time (compilation and warm-up), steady-state step time
    (mean and spread over 30 steps), forward, backward and optimiser time separately, tokens per second, peak memory, and
    achieved FLOPs as a fraction of both the measured GEMM ceiling and the datasheet peak
  - the teacher-signal path: a cached top-32 teacher-logit file read through a prefetching loader and used in a sparse
    distillation loss, against the same step without it, and the time the training step spent waiting for data
  - a profiler breakdown of device time (GEMM, recurrent kernels, attention, optimiser, other) and how much of the wall time
    the device was idle, which locates the bottleneck instead of assuming one
  - a short stability run in the intended precision
Phases stop being started once R0_BUDGET_S seconds have elapsed. Every phase is isolated: a failure is recorded and the run
continues. Nothing sensitive is recorded (no environment dump).
"""
import os, sys, json, time, math, subprocess, traceback, threading, queue
import numpy as np

OUT = os.environ.get("R0_OUT", "/workspace/out/r0.json")
BUDGET = float(os.environ.get("R0_BUDGET_S", "1500"))
DT = os.environ.get("R0_DTYPE", "bfloat16")
SMALL = os.environ.get("R0_SMALL") == "1"          # dry run on a small GPU
T_START = time.time()
R = {"dtype": DT, "small_mode": SMALL, "phases": {}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def save(): json.dump(R, open(OUT, "w"), indent=1, default=str)


def log(*a): print(f"[{time.time() - T_START:7.1f}s]", *a, flush=True)


def phase(name, need_s=0):
    def deco(fn):
        left = BUDGET - (time.time() - T_START)
        if left < need_s:
            R["phases"][name] = {"ok": None, "skipped": f"budget: {left:.0f}s left, needs {need_s}s"}; save(); log(name, "skipped"); return fn
        t0 = time.time()
        try: R["phases"][name] = {"ok": True, "result": fn()}
        except Exception as e: R["phases"][name] = {"ok": False, "error": repr(e)[:600], "trace": traceback.format_exc()[-1500:]}
        R["phases"][name]["seconds"] = round(time.time() - t0, 1); save()
        log(name, "ok" if R["phases"][name]["ok"] else "FAILED " + R["phases"][name].get("error", "")[:200]); return fn
    return deco


if os.environ.get("R0_INSTALL", "1") == "1":
    t0 = time.time()
    out = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention", "fla-core"], capture_output=True, text=True)
    out2 = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "einops"], capture_output=True, text=True)
    R["install_seconds"] = round(time.time() - t0, 1); R["install_rc"] = [out.returncode, out2.returncode]; save(); log("install", R["install_rc"], R["install_seconds"])

import torch
import torch.nn as nn
import torch.nn.functional as F

CAST = torch.bfloat16 if DT == "bfloat16" else torch.float16
DEV = "cuda"
SPEC = {"H100 80GB HBM3": (989.0, 3350.0), "H100 PCIe": (756.0, 2000.0), "H100 NVL": (835.0, 3900.0), "A100-SXM4-80GB": (312.0, 2039.0),
        "A100 80GB PCIe": (312.0, 1935.0), "Tesla T4": (65.0, 320.0), "NVIDIA L40S": (362.0, 864.0)}      # datasheet dense tensor TFLOPS, GB/s
GPU = torch.cuda.get_device_name(0)
SPEC_TF, SPEC_BW = next((v for k, v in SPEC.items() if k in GPU), (None, None))


@phase("env")
def _():
    import importlib.metadata as md
    d = {"gpu": GPU, "capability": list(torch.cuda.get_device_capability(0)), "memory_GiB": round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1),
         "sm_count": torch.cuda.get_device_properties(0).multi_processor_count, "torch": torch.__version__, "cuda": torch.version.cuda,
         "datasheet_peak_TFLOPS_dense_16bit": SPEC_TF, "datasheet_bandwidth_GBps": SPEC_BW, "cpu_count": os.cpu_count()}
    for p in ("triton", "fla-core", "flash-linear-attention", "transformers"):
        try: d[p] = md.version(p)
        except Exception: d[p] = None
    return d


@phase("ceilings", 60)
def _():
    n = 8192 if not SMALL else 4096
    a = torch.randn(n, n, device=DEV, dtype=CAST); b = torch.randn(n, n, device=DEV, dtype=CAST)
    for _ in range(5): a @ b
    torch.cuda.synchronize(); t = time.time(); it = 30 if not SMALL else 10
    for _ in range(it): a @ b
    torch.cuda.synchronize(); gem = 2 * n ** 3 * it / (time.time() - t) / 1e12
    x = torch.empty(2 ** 30 // 2 if not SMALL else 2 ** 28 // 2, dtype=torch.float16, device=DEV); y = torch.empty_like(x)
    for _ in range(3): y.copy_(x)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20): y.copy_(x)
    torch.cuda.synchronize(); bw = 2 * x.numel() * 2 * 20 / (time.time() - t) / 1e9
    del a, b, x, y; torch.cuda.empty_cache()
    return {"measured_gemm_TFLOPS": round(gem, 1), "measured_copy_bandwidth_GBps": round(bw, 1), "gemm_matrix": n}


GEMM_PEAK = (R["phases"].get("ceilings", {}).get("result") or {}).get("measured_gemm_TFLOPS")


def ref_gdn(q, k, v, g, beta, scale):
    B, T, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1]); outs = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kv = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], (v[:, t] - kv) * beta[:, t][..., None])
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(outs, 1)


@phase("correctness", 120)
def _():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    res = []
    for dtype in (torch.float32, CAST):
        torch.manual_seed(0); B, T, H, K = 2, 128, 4, 64
        q = F.normalize(torch.randn(B, T, H, K, device=DEV), dim=-1); k = F.normalize(torch.randn(B, T, H, K, device=DEV), dim=-1)
        v = torch.randn(B, T, H, K, device=DEV); g = F.logsigmoid(torch.randn(B, T, H, device=DEV) + 2); beta = torch.rand(B, T, H, device=DEV)
        ins = [q.to(dtype), k.to(dtype), v.to(dtype), g, beta.to(dtype)]
        kin = [x.detach().clone().requires_grad_(True) for x in ins]; rin = [x.detach().clone().float().requires_grad_(True) for x in ins]
        ko = chunk_gated_delta_rule(*kin, scale=K ** -0.5); ko = ko[0] if isinstance(ko, tuple) else ko
        ro = ref_gdn(*rin, K ** -0.5); w = torch.randn_like(ro)
        (ko.float() * w).sum().backward(); (ro * w).sum().backward()
        rel = lambda a, b: float(((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item())
        res.append({"dtype": str(dtype), "output_rel_err": rel(ko.float(), ro), **{f"grad_{n}_rel_err": rel(a.grad.float(), b.grad) for n, a, b in zip("qkvgb", kin, rin)}})
    return res


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
        x = x + y; g, u = s.gu(s.n2(x)).chunk(2, -1)
        return x + s.dn(F.silu(g) * u)


class LM(nn.Module):
    def __init__(s, kind, d, L, ffn, heads, vocab):
        super().__init__(); s.emb = nn.Embedding(vocab, d); s.blocks = nn.ModuleList([Block(d, kind, ffn, heads) for _ in range(L)]); s.nf = nn.RMSNorm(d)
        nn.init.normal_(s.emb.weight, std=0.02)

    def hidden(s, ids):
        x = s.emb(ids)
        for b in s.blocks: x = b(x)
        return s.nf(x).reshape(-1, x.shape[-1])

    def loss(s, ids, kd=None, chunk=2048):
        x = s.hidden(ids[:, :-1]); y = ids[:, 1:].reshape(-1); W = s.emb.weight; tot = 0.0
        for i in range(0, x.shape[0], chunk):
            if kd is None:
                f = lambda xc, yc: F.cross_entropy((xc @ W.T).float(), yc, reduction="sum"); args = (x[i:i + chunk], y[i:i + chunk])
            else:
                f = lambda xc, yc, ti, tv: sparse_kd((xc @ W.T).float(), yc, ti, tv); args = (x[i:i + chunk], y[i:i + chunk], kd[0].reshape(-1, kd[0].shape[-1])[i:i + chunk], kd[1].reshape(-1, kd[1].shape[-1])[i:i + chunk])
            tot = tot + torch.utils.checkpoint.checkpoint(f, *args, use_reentrant=False)
        return tot / y.numel()


def sparse_kd(logits, y, tidx, tval):
    """0.5 * cross-entropy on the data + 0.5 * KL(teacher top-k renormalised || student) over the cached top-k entries."""
    lse = torch.logsumexp(logits, -1, keepdim=True)
    ce = (lse[:, 0] - logits.gather(1, y[:, None])[:, 0]).sum()
    tp = torch.softmax(tval.float(), -1); slp = logits.gather(1, tidx.long()) - lse
    kl = (tp * (torch.log(tp.clamp_min(1e-9)) - slp)).sum()
    return 0.5 * ce + 0.5 * kl


class Loader:
    """Prefetching reader of a cached teacher-signal file: per token a top-k index (int32) and value (fp16)."""
    def __init__(s, path, n_tokens, B, T, k, vocab, depth=3):
        s.idx = np.memmap(path + ".idx", dtype=np.int32, mode="r", shape=(n_tokens, k)); s.val = np.memmap(path + ".val", dtype=np.float16, mode="r", shape=(n_tokens, k))
        s.n, s.B, s.T, s.k, s.vocab, s.q = n_tokens, B, T, k, vocab, queue.Queue(depth); s.rng = np.random.default_rng(0); s.stop = False; s.wait = 0.0; s.read = 0.0
        s.th = threading.Thread(target=s.run, daemon=True); s.th.start()

    def run(s):
        while not s.stop:
            t = time.time(); st = s.rng.integers(0, s.n - s.T - 1, s.B)
            ids = torch.from_numpy(s.rng.integers(0, s.vocab, (s.B, s.T)).astype(np.int64))
            ti = torch.from_numpy(np.stack([s.idx[i:i + s.T - 1] for i in st])); tv = torch.from_numpy(np.stack([s.val[i:i + s.T - 1] for i in st]))
            s.read += time.time() - t; s.q.put((ids.pin_memory(), ti.pin_memory(), tv.pin_memory()))

    def get(s):
        t = time.time(); item = s.q.get(); s.wait += time.time() - t
        return [x.to(DEV, non_blocking=True) for x in item]


def make_model(kind, d=896, L=24, ffn=4864, heads=7, vocab=151936):
    if SMALL: L, vocab = 6, 32768
    return LM(kind, d, L, ffn, heads, vocab).to(DEV)


def train_bench(kind, B, T, kd=False, steps=30, warm=4, loader=None):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m = make_model(kind); n_params = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True); use_scaler = CAST == torch.float16; scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    vocab = m.emb.num_embeddings; gen = torch.Generator(device="cpu").manual_seed(0); times = {"first": None, "warm": [], "fwd": [], "bwd": [], "opt": [], "step": []}

    def one(timed):
        if loader is not None: ids, ti, tv = loader.get(); kdv = (ti, tv)
        else: ids = torch.randint(0, vocab, (B, T), generator=gen).to(DEV); kdv = None
        e = [torch.cuda.Event(enable_timing=True) for _ in range(4)]; e[0].record()
        with torch.autocast("cuda", dtype=CAST): l = m.loss(ids, kdv)
        e[1].record(); scaler.scale(l).backward(); e[2].record(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); e[3].record(); torch.cuda.synchronize()
        if timed:
            times["fwd"].append(e[0].elapsed_time(e[1])); times["bwd"].append(e[1].elapsed_time(e[2])); times["opt"].append(e[2].elapsed_time(e[3])); times["step"].append(e[0].elapsed_time(e[3]))
        return float(l)

    t = time.time(); one(False); torch.cuda.synchronize(); times["first"] = time.time() - t
    for _ in range(warm - 1):
        t = time.time(); one(False); times["warm"].append(round(time.time() - t, 3))
    if loader is not None: loader.wait = 0.0
    t_all = time.time()
    for _ in range(steps): last = one(True)
    wall = (time.time() - t_all) / steps; ms = lambda v: [round(float(np.mean(v)), 1), round(float(np.std(v)), 1)]
    tps = B * T / wall; fl = 6 * n_params * tps / 1e12
    res = {"kind": kind, "micro_batch": f"{B}x{T}", "kd_cached_teacher": kd, "params_M": round(n_params / 1e6, 1), "first_step_s_incl_compile": round(times["first"], 2), "warmup_steps_s": times["warm"],
           "steady_step_s_mean": round(wall, 4), "steady_step_ms_fwd_bwd_opt_mean_sd": {"forward": ms(times["fwd"]), "backward": ms(times["bwd"]), "optimizer": ms(times["opt"]), "device_step": ms(times["step"])},
           "tokens_per_s": round(tps), "achieved_model_TFLOPS_6N": round(fl, 1), "fraction_of_measured_gemm_ceiling": round(fl / GEMM_PEAK, 3) if GEMM_PEAK else None,
           "fraction_of_datasheet_peak": round(fl / SPEC_TF, 3) if SPEC_TF else None, "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "last_loss": round(last, 3)}
    if loader is not None: res["loader_wait_s_per_step"] = round(loader.wait / steps, 4); res["loader_read_s_total"] = round(loader.read, 2)
    del m, opt; torch.cuda.empty_cache(); return res


def grid():
    return [(2, 1024)] if SMALL else [(4, 1024), (8, 1024), (16, 1024), (8, 2048)]


def run_grid(kind):
    out = []
    for B, T in grid():
        try: out.append(train_bench(kind, B, T))
        except torch.cuda.OutOfMemoryError: out.append({"kind": kind, "micro_batch": f"{B}x{T}", "oom": True}); torch.cuda.empty_cache()
    return out


@phase("bench_gdn", 240)
def _(): return run_grid("gdn")


@phase("bench_attention", 200)
def _(): return run_grid("attn")


def best_cfg(name):
    rows = [r for r in R["phases"].get(name, {}).get("result", []) if "tokens_per_s" in r]
    return max(rows, key=lambda r: r["tokens_per_s"]) if rows else None


@phase("teacher_cache_path", 200)
def _():
    b = best_cfg("bench_gdn"); B, T = (int(x) for x in (b["micro_batch"] if b else "4x1024").split("x")); k = 32; n = 4_000_000 if not SMALL else 400_000
    path = "/tmp/r0_cache" if os.path.exists("/tmp") else "r0_cache"; vocab = 151936 if not SMALL else 32768; rng = np.random.default_rng(1)
    t = time.time()
    np.memmap(path + ".idx", dtype=np.int32, mode="w+", shape=(n, k))[:] = rng.integers(0, vocab, (n, k)); np.memmap(path + ".val", dtype=np.float16, mode="w+", shape=(n, k))[:] = rng.normal(size=(n, k)).astype(np.float16)
    create_s = time.time() - t; ld = Loader(path, n, B, T, k, vocab)
    res = {"cache_file_MiB": round(n * k * 6 / 2 ** 20), "cache_creation_s": round(create_s, 1), "without_cache": best_cfg("bench_gdn")}
    res["with_cache"] = train_bench("gdn", B, T - 1 + 1, kd=True, loader=ld); ld.stop = True
    return res


@phase("profile", 150)
def _():
    from torch.profiler import profile, ProfilerActivity
    b = best_cfg("bench_gdn"); B, T = (int(x) for x in (b["micro_batch"] if b else "4x1024").split("x")); m = make_model("gdn"); opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=CAST == torch.float16); ids = torch.randint(0, m.emb.num_embeddings, (B, T), device=DEV)

    def step():
        with torch.autocast("cuda", dtype=CAST): l = m.loss(ids)
        scaler.scale(l).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    for _ in range(4): step()
    torch.cuda.synchronize(); w0 = time.time()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(5): step()
        torch.cuda.synchronize()
    wall = time.time() - w0
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None)
        if t is None: t = getattr(e, "cuda_time_total", 0)
        if t and t > 0: rows.append((e.key, t, e.count))
    rows = [r for r in rows if not r[0].startswith(("cudaLaunch", "aten::", "cuda", "Memcpy", "Memset")) or "Memcpy" in r[0]]
    tot = sum(t for _, t, _ in rows)
    cats = {"gemm": ("gemm", "cutlass", "cublas", "xmma", "nvjet", "sm80_", "sm90_", "ampere_", "turing_", "volta_"), "attention_flash": ("flash", "fmha", "attention"),
            "recurrent_triton": ("chunk", "gated_delta", "delta_rule", "fused_recurrent", "recompute_w_u", "fwd_prepare", "solve_tril", "l2norm", "kkt", "conv"), "optimizer": ("adam", "multi_tensor", "foreach"),
            "norm_activation_elementwise": ("rms", "norm", "silu", "elementwise", "vectorized", "mul", "add", "copy", "cat", "index")}
    share = {c: 0.0 for c in cats}; share["other"] = 0.0
    for k, t, _ in rows:
        kl = k.lower(); hit = next((c for c, ws in cats.items() if any(w in kl for w in ws)), "other"); share[hit] += t
    return {"micro_batch": f"{B}x{T}", "profiled_steps": 5, "wall_s_per_step": round(wall / 5, 4), "device_busy_s_per_step": round(tot / 5 / 1e6, 4), "device_busy_fraction_of_wall": round(tot / 1e6 / wall, 3),
            "device_time_share": {c: round(v / tot, 3) for c, v in share.items()}, "top_kernels": [{"name": k[:90], "share": round(t / tot, 3), "calls_per_step": c // 5} for k, t, c in sorted(rows, key=lambda r: -r[1])[:14]]}


@phase("stability", 120)
def _():
    m = make_model("gdn"); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1, fused=True); scaler = torch.amp.GradScaler("cuda", enabled=CAST == torch.float16)
    b = best_cfg("bench_gdn"); B, T = (int(x) for x in (b["micro_batch"] if b else "2x1024").split("x")); g = torch.Generator(device="cpu").manual_seed(2); log_ = []; nonfinite = 0
    for s in range(100):
        for gp in opt.param_groups: gp["lr"] = 3e-4 * min(1, (s + 1) / 20)
        ids = torch.randint(0, m.emb.num_embeddings, (B, T), generator=g).to(DEV)
        with torch.autocast("cuda", dtype=CAST): l = m.loss(ids)
        nonfinite += int(not torch.isfinite(l)); scaler.scale(l).backward(); scaler.unscale_(opt); gn = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if s % 20 == 0 or s == 99: log_.append([s, round(float(l), 3), round(gn, 3)])
    return {"steps": 100, "nonfinite_loss_steps": nonfinite, "log[step,loss,gradnorm]": log_, "note": "random tokens: checks numerical stability only, not learning"}


R["total_seconds"] = round(time.time() - T_START, 1); R["done"] = True; save()
print("R0_RESULT_BEGIN"); print(json.dumps(R, default=str)); print("R0_RESULT_END", flush=True)
