"""STAGE A2: the registered stage-A script (convert_stage_a.py, unchanged logic and seeds) plus an exploratory extension at the end:
leave-one-out layer importance, placement comparisons with five random draws, and the inherit-trained layer weights saved.
Its per-layer numbers also serve as a rerun-to-rerun variation check of stage A. The registered criteria are in convert_stage_a.py.

Stage A (execution and correctness) of the inheritance route: convert the attention layers of SmolLM2-135M
(Apache 2.0; 30 layers, width 576, 9 query heads, 3 key-value heads, head size 64) into Gated DeltaNet layers.
RESEARCH-PLAN.md sections 8 and 20.5. Registered before the first run.

For every layer i the teacher's attention sublayer is the target: given the sublayer's input (the layer-normed hidden
state) the replacement must reproduce the sublayer's output. Per layer, three arms, same data, same 300 steps:
  inherit   a Gated DeltaNet layer whose q, k, v and o projections are copied from the teacher's attention
            (the 3 key-value heads are repeated to 9 to match), trained to match the teacher's output
  random    the same layer with default initialisation, same training (the inheritance control)
  untrained both initialisations before any training (the "untrained layer" of the criterion)
Metric: relative squared error  ||y - t||^2 / ||t||^2  on held-out sequences (0 = exact, 1 = no better than zero).

Registered pass criteria (not changed after seeing results):
  C1  at least 90% of layers (27 of 30) have trained-inherit error <= 0.1 x the untrained-random error
  C2  no non-finite loss in any training run
  C3  measured conversion-training throughput (tokens/s) and peak memory are reported
Descriptive, no pass/fail: language-model loss (nats per token on held-out text, teacher tokenizer, no end-to-end
training) of six models: the teacher; all 30 layers converted (inherit, trained); the 22 converted layers with the lowest
error and 8 attention layers kept (data-driven placement); a random 22/8 split (placement control); all 30 converted from
random initialisation, trained; all 30 inherit-initialised but untrained.

Runs on a Kaggle T4 (fla kernels) or, with NANO_LOCAL=1, on CPU with a naive reference recurrence for debugging only.
"""
import os, sys, json, time, glob, math, random, subprocess, traceback
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

LOCAL = os.environ.get("NANO_LOCAL") == "1"
OUT = os.environ.get("NANO_OUT", "/kaggle/working/stage_a.json")
DEV = "cpu" if LOCAL else "cuda"
T = 64 if LOCAL else 1024
N_TRAIN, N_VAL = (8, 4) if LOCAL else (64, 16)
STEPS = 3 if LOCAL else 300
BS = 4 if LOCAL else 8
R = {"config": {"T": T, "n_train": N_TRAIN, "n_val": N_VAL, "steps": STEPS, "batch": BS, "local": LOCAL}, "layers": [], "lm": {}}


def save(): json.dump(R, open(OUT, "w"), indent=1, default=str)


if not LOCAL:
    out = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention", "fla-core"], capture_output=True, text=True)
    R["pip_rc"] = out.returncode


def ref_gdn(q, k, v, g, beta, scale):
    B, T_, H, K = q.shape
    S = q.new_zeros(B, H, K, v.shape[-1]); outs = []
    for t in range(T_):
        S = S * g[:, t].exp()[..., None, None]
        kv = torch.einsum("bhk,bhkv->bhv", k[:, t], S)
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, t], (v[:, t] - kv) * beta[:, t][..., None])
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))
    return torch.stack(outs, 1)


class NaiveGDN(nn.Module):
    """CPU debugging stand-in with the same projection names as fla's GatedDeltaNet. Not used on the GPU."""
    def __init__(s, hidden_size, num_heads, head_dim, **kw):
        super().__init__(); s.h, s.d = num_heads, head_dim; kd = num_heads * head_dim
        s.q_proj = nn.Linear(hidden_size, kd, bias=False); s.k_proj = nn.Linear(hidden_size, kd, bias=False); s.v_proj = nn.Linear(hidden_size, kd, bias=False)
        s.a_proj = nn.Linear(hidden_size, num_heads, bias=False); s.b_proj = nn.Linear(hidden_size, num_heads, bias=False); s.o_proj = nn.Linear(kd, hidden_size, bias=False)

    def forward(s, x, **kw):
        B, T_, _ = x.shape; sh = lambda z: z.view(B, T_, s.h, s.d)
        q, k, v = F.normalize(sh(s.q_proj(x)), dim=-1), F.normalize(sh(s.k_proj(x)), dim=-1), sh(s.v_proj(x))
        g = F.logsigmoid(s.a_proj(x) + 3.0); beta = torch.sigmoid(s.b_proj(x))
        return s.o_proj(ref_gdn(q, k, v, g, beta, s.d ** -0.5).reshape(B, T_, -1)), None, None


def make_gdn(hidden, heads, head_dim):
    if LOCAL: return NaiveGDN(hidden, heads, head_dim).to(DEV)
    from fla.layers import GatedDeltaNet
    return GatedDeltaNet(hidden_size=hidden, num_heads=heads, head_dim=head_dim, expand_v=1, mode="chunk").to(DEV)


def inherit(gdn, attn, heads, kv_heads, head_dim):
    """Copy the teacher's projections. Query heads map onto key-value heads in consecutive groups (grouped-query attention)."""
    rep = heads // kv_heads
    with torch.no_grad():
        gdn.q_proj.weight.copy_(attn.q_proj.weight)
        for name in ("k_proj", "v_proj"):
            w = getattr(attn, name).weight.view(kv_heads, head_dim, -1).repeat_interleave(rep, dim=0).reshape(heads * head_dim, -1)
            getattr(gdn, name).weight.copy_(w)
        gdn.o_proj.weight.copy_(attn.o_proj.weight)


def rel_err(y, t): return float(((y.float() - t.float()) ** 2).sum() / (t.float() ** 2).sum())


def build_data(tok_teacher):
    from tokenizers import Tokenizer
    root = os.environ.get("NANO_DATA") or (glob.glob("/kaggle/input/**/train.bin", recursive=True) or [""])[0].rsplit("/", 1)[0]
    ours = Tokenizer.from_file(f"{root}/tokenizer_32k.json"); data = np.memmap(f"{root}/train.bin", dtype=np.uint16, mode="r")
    need = (N_TRAIN + N_VAL) * T; ids = []; pos = 5_000_000 % (len(data) - 10 ** 6)
    while len(ids) < need + T:
        text = ours.decode(data[pos:pos + 1500].astype("int64").tolist()); pos += 1500
        ids += tok_teacher(text, add_special_tokens=False)["input_ids"] + [tok_teacher.eos_token_id]
    arr = torch.tensor(ids[:need]).view(N_TRAIN + N_VAL, T)
    return arr[:N_TRAIN], arr[N_TRAIN:]


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = os.environ.get("NANO_TEACHER", "HuggingFaceTB/SmolLM2-135M")
    tok = AutoTokenizer.from_pretrained(name); teacher = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(DEV).eval()
    cfg = teacher.config; H, KV, hd, D, L = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers
    R["teacher"] = {"name": name, "layers": L, "hidden": D, "heads": H, "kv_heads": KV, "head_dim": hd, "vocab": cfg.vocab_size}
    tr, va = build_data(tok); R["config"]["train_tokens"] = int(tr.numel()); R["config"]["val_tokens"] = int(va.numel())
    cache = {"in": [[] for _ in range(L)], "out": [[] for _ in range(L)]}; split = {"cur": "train"}

    def mk(i):
        def hook(mod, args, kwargs, out):
            h = kwargs.get("hidden_states", args[0] if args else None); o = out[0] if isinstance(out, tuple) else out
            cache["in"][i].append((split["cur"], h.detach().half().cpu())); cache["out"][i].append((split["cur"], o.detach().half().cpu()))
        return hook
    hs = [teacher.model.layers[i].self_attn.register_forward_hook(mk(i), with_kwargs=True) for i in range(L)]
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=not LOCAL):
        for nm, d in (("train", tr), ("val", va)):
            split["cur"] = nm
            for b in range(0, d.shape[0], 4): teacher(d[b:b + 4].to(DEV), use_cache=False)
    for h in hs: h.remove()
    get = lambda i, k, nm: torch.cat([x for s, x in cache[k][i] if s == nm])
    R["teacher_cached"] = True; save()

    def train_layer(i, init):
        torch.manual_seed(1000 + i)
        gdn = make_gdn(D, H, hd)
        if init == "inherit": inherit(gdn, teacher.model.layers[i].self_attn, H, KV, hd)
        xt, yt, xv, yv = get(i, "in", "train"), get(i, "out", "train"), get(i, "in", "val"), get(i, "out", "val")

        def val_err():
            gdn.eval(); num = den = 0.0
            with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=not LOCAL):
                for b in range(0, xv.shape[0], 4):
                    y = gdn(xv[b:b + 4].to(DEV).float())[0]; t = yv[b:b + 4].to(DEV).float()
                    num += float(((y.float() - t) ** 2).sum()); den += float((t ** 2).sum())
            gdn.train(); return num / den
        e0 = val_err(); opt = torch.optim.AdamW(gdn.parameters(), lr=1e-3, weight_decay=0.0); scaler = torch.amp.GradScaler(DEV, enabled=not LOCAL)
        g = torch.Generator().manual_seed(i); nonfinite = 0; t0 = time.time()
        for s in range(STEPS):
            for gp in opt.param_groups: gp["lr"] = 1e-3 * min(1.0, (s + 1) / 20) * (0.5 * (1 + math.cos(math.pi * s / STEPS)) * 0.9 + 0.1)
            ix = torch.randint(0, xt.shape[0], (BS,), generator=g)
            x, t = xt[ix].to(DEV).float(), yt[ix].to(DEV).float()
            with torch.autocast(DEV, dtype=torch.float16, enabled=not LOCAL):
                y = gdn(x)[0]
            loss = ((y.float() - t) ** 2).sum() / (t ** 2).sum()
            if not torch.isfinite(loss): nonfinite += 1
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(gdn.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if not LOCAL: torch.cuda.synchronize()
        dt = time.time() - t0
        return gdn, {"err_untrained": e0, "err_trained": val_err(), "nonfinite_steps": nonfinite, "train_seconds": round(dt, 1), "train_tokens_per_s": round(STEPS * BS * T / dt)}

    mods = {"inherit": {}, "random": {}}; rows = []
    for i in range(L):
        row = {"layer": i}
        for init in ("inherit", "random"):
            try:
                gdn, r = train_layer(i, init); mods[init][i] = gdn; row[init] = r
            except Exception as e:
                row[init] = {"error": repr(e)[:300], "trace": traceback.format_exc()[-900:]}
        rows.append(row); R["layers"] = rows; save(); print("layer", i, {k: (round(v["err_untrained"], 3), round(v["err_trained"], 3)) if "err_trained" in v else "ERR" for k, v in row.items() if k != "layer"}, flush=True)

    ok = [r for r in rows if "err_trained" in r.get("inherit", {}) and "err_trained" in r.get("random", {})]
    thr = [r["inherit"]["err_trained"] <= 0.1 * r["random"]["err_untrained"] for r in ok]
    R["criteria"] = {"C1_fraction_layers_ratio_le_0.1": round(sum(thr) / max(1, len(ok)), 3), "C1_pass": bool(len(ok) == L and sum(thr) >= 0.9 * L),
                     "C2_nonfinite_total": sum(r[k]["nonfinite_steps"] for r in ok for k in ("inherit", "random")),
                     "C3_median_train_tokens_per_s": int(np.median([r["inherit"]["train_tokens_per_s"] for r in ok])) if ok else None,
                     "C3_peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if not LOCAL else None,
                     "median_err": {"inherit_untrained": float(np.median([r["inherit"]["err_untrained"] for r in ok])), "inherit_trained": float(np.median([r["inherit"]["err_trained"] for r in ok])),
                                    "random_untrained": float(np.median([r["random"]["err_untrained"] for r in ok])), "random_trained": float(np.median([r["random"]["err_trained"] for r in ok]))}}
    save(); print(json.dumps(R["criteria"]), flush=True)

    class Wrap(nn.Module):
        def __init__(s, gdn): super().__init__(); s.gdn = gdn
        def forward(s, hidden_states=None, *a, **k): return s.gdn(hidden_states)[0], None

    orig = {i: teacher.model.layers[i].self_attn for i in range(L)}

    @torch.no_grad()
    def lm_loss(replace):
        for i in range(L): teacher.model.layers[i].self_attn = Wrap(replace[i]) if i in replace else orig[i]
        tot = n = 0.0
        with torch.autocast(DEV, dtype=torch.float16, enabled=not LOCAL):
            for b in range(0, va.shape[0], 4):
                ids = va[b:b + 4].to(DEV); lg = teacher(ids, use_cache=False).logits.float()
                tot += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")); n += ids[:, 1:].numel()
        for i in range(L): teacher.model.layers[i].self_attn = orig[i]
        return tot / n

    errs = {r["layer"]: r["inherit"]["err_trained"] for r in ok}; keep_hi = set(sorted(errs, key=errs.get, reverse=True)[:max(1, round(0.27 * L))])
    keep_rand = set(random.Random(0).sample(range(L), len(keep_hi)))
    untr = {}
    for i in range(L):
        g = make_gdn(D, H, hd); inherit(g, orig[i], H, KV, hd); untr[i] = g
    sets = {"teacher": {}, "all_inherit_trained": mods["inherit"],
            "keep_highest_error_attention": {i: m for i, m in mods["inherit"].items() if i not in keep_hi},
            "keep_random_attention": {i: m for i, m in mods["inherit"].items() if i not in keep_rand},
            "all_random_trained": mods["random"], "all_inherit_untrained": untr}
    for k, rep in sets.items():
        try: R["lm"][k] = {"nats_per_token": round(lm_loss(rep), 4), "converted_layers": len(rep)}
        except Exception as e: R["lm"][k] = {"error": repr(e)[:300], "trace": traceback.format_exc()[-900:]}
        save(); print("lm", k, R["lm"][k], flush=True)
    R["kept_attention_layers"] = {"highest_error": sorted(keep_hi), "random": sorted(keep_rand)}; save()

    # ---- exploratory extension (stage A2; not part of the registered criteria) ----
    base = R["lm"]["teacher"]["nats_per_token"]
    imp = {i: round(lm_loss({i: mods["inherit"][i]}) - base, 4) for i in range(L)}       # convert ONE layer, measure the loss change
    R["layer_importance_delta_nats"] = imp; save()
    order = sorted(imp, key=imp.get); n_keep = len(keep_hi)
    place = {"keep_most_harmful_to_convert": set(order[-n_keep:]), "keep_least_harmful_to_convert": set(order[:n_keep])}
    for d in range(5): place[f"keep_random_draw{d}"] = set(random.Random(100 + d).sample(range(L), n_keep))
    R["placement"] = {}
    for nm, keep in place.items():
        R["placement"][nm] = {"kept_attention_layers": sorted(keep), "nats_per_token": round(lm_loss({i: m for i, m in mods["inherit"].items() if i not in keep}), 4)}
        save(); print("placement", nm, R["placement"][nm]["nats_per_token"], flush=True)
    if not LOCAL: torch.save({f"inherit_{i}": m.state_dict() for i, m in mods["inherit"].items()}, "/kaggle/working/inherit_layers.pt")


try:
    main()
except Exception as e:
    R["fatal"] = {"error": repr(e)[:500], "trace": traceback.format_exc()[-1500:]}; save(); print("FATAL", repr(e)[:300])
