"""JAX/TPU 40.38M / 57.2M Untied-Head x ValueRes x Equal-Parameter Tied Factorial Runner.

Tests whether untying the output LM head (`lm_head` != `emb.T`) resolves the tied-embedding output
rotation bottleneck identified in Draft PR #214 (`lambda_7 = 0.123`, `alpha_7 = 0.073`), improves
both held-out BPB and 7-task zero-shot accuracy at near-zero extra FLOPs, and reduces the marginal
gain and Layer-7 skip concentration of `ValueRes`:

  1. `dense_tied`      : `d=512, L=8, ffn=1408, H=8, KV=4`, tied `emb` (`40,379,904` params)
  2. `dense_untied`    : `d=512, L=8, ffn=1408, H=8, KV=4`, untied `lm_head [512, 32768]` (`57,157,120` params, `0%` extra block FLOPs)
  3. `dense_untied_vr` : `dense_untied` + `+7` learned `lam_1..lam_7` `ValueRes` scalars (`57,157,127` params)
  4. `dense_tied_57m`  : `d=640, L=8, ffn=1728, H=10, KV=5`, tied `emb` (`57,355,904` params, `+54.2%` block FLOPs)

Optimizer contract (`optax.contrib.muon` matching `train_jax.py` & `valueres_42m_jax.py`):
  - Both `emb` and `lm_head` map to `None` in `muon_weight_dimension_numbers` (updated by AdamW `b1=0.9, b2=0.95, wd=0.1`,
    never orthogonalized by Newton-Schulz), while 2D Transformer block projections use Muon (`lr=4e-3, wd=0.1, consistent_rms=0.2`).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

VOCAB = 32768
EOS = 1
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ARMS = ("dense_tied", "dense_untied", "dense_untied_vr", "dense_tied_57m")
SIZES = {
    "T": dict(d=64, layers=4, ffn=176, heads=4, kv=2),
    "T_WIDE": dict(d=80, layers=4, ffn=224, heads=4, kv=2),
    "M": dict(d=512, layers=8, ffn=1408, heads=8, kv=4),
    "M_WIDE": dict(d=640, layers=8, ffn=1728, heads=10, kv=5),
}
CHANCE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "piqa": 0.50,
    "winogrande": 0.50,
    "openbookqa": 0.25,
    "lambada": 0.0,
}


def resolve_cfg(size: str, arm: str) -> dict:
    if arm == "dense_tied_57m":
        return SIZES[f"{size}_WIDE"]
    return SIZES[size]


def sha256_file(path: str | Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    rem = max_bytes
    with open(path, "rb") as f:
        while True:
            to_read = (1 << 20) if rem is None else min(1 << 20, rem)
            if to_read <= 0:
                break
            b = f.read(to_read)
            if not b:
                break
            h.update(b)
            if rem is not None:
                rem -= len(b)
    return h.hexdigest()


def init_params(key, cfg: dict, arm: str, vocab: int = VOCAB) -> dict:
    assert arm in ARMS, f"Unknown arm {arm}"
    d, L, ffn, H, KV = cfg["d"], cfg["layers"], cfg["ffn"], cfg["heads"], cfg["kv"]
    hd = d // H
    # Preserve exact PRNG stream for `emb` and `blocks` so `dense_tied`, `dense_untied`, and `dense_untied_vr`
    # start from identical `emb` and block matrices per seed!
    ks = iter(jax.random.split(key, 1 + 6 * L))
    n = lambda shape, scale=1.0: 0.02 * scale * jax.random.normal(next(ks), shape, jnp.float32)
    res = 1.0 / math.sqrt(2 * L)
    one = lambda k: jnp.ones((k,), jnp.float32)
    blocks = []
    for i in range(L):
        blk = dict(
            n1=one(d),
            n2=one(d),
            qn=one(hd),
            kn=one(hd),
            q=n((d, H * hd)),
            k=n((d, KV * hd)),
            v=n((d, KV * hd)),
            o=n((H * hd, d), res),
            gu=n((d, 2 * ffn)),
            dn=n((ffn, d), res),
        )
        if arm == "dense_untied_vr" and i > 0:
            blk["lam"] = jnp.array(0.5, dtype=jnp.float32)
        blocks.append(blk)
    out = dict(emb=n((vocab, d)), nf=one(d), blocks=blocks)
    if arm in ("dense_untied", "dense_untied_vr"):
        k_head = jax.random.fold_in(key, 999_999)
        out["lm_head"] = 0.02 * jax.random.normal(k_head, (d, vocab), jnp.float32)
    return out


def n_params(params: dict) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def rms(x, w, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), -1, keepdims=True) + eps) * w.astype(jnp.float32)).astype(x.dtype)


def rope_tables(T: int, hd: int, theta: float = 10000.0):
    inv = 1.0 / (theta ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    f = jnp.outer(jnp.arange(T, dtype=jnp.float32), inv)
    return jnp.cos(f), jnp.sin(f)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos.astype(x.dtype), sin.astype(x.dtype)
    return jnp.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).reshape(x.shape)


def forward_with_diag(
    params: dict,
    x,
    cfg: dict,
    arm: str,
    dtype=jnp.bfloat16,
    ablation: str = "none",
    perm_idx=None,
    return_diag: bool = False,
):
    B, T = x.shape
    H, KV = cfg["heads"], cfg["kv"]
    hd = cfg["d"] // H
    L = cfg["layers"]
    rep = H // KV
    cos, sin = rope_tables(T, hd)
    causal = jnp.tril(jnp.ones((T, T), jnp.bool_))

    h = params["emb"].astype(dtype)[x]
    v0 = None
    layer_diags = []

    for i, b in enumerate(params["blocks"]):
        a = rms(h, b["n1"])
        q = apply_rope(rms((a @ b["q"].astype(dtype)).reshape(B, T, H, hd), b["qn"]).swapaxes(1, 2), cos, sin)
        k = apply_rope(rms((a @ b["k"].astype(dtype)).reshape(B, T, KV, hd), b["kn"]).swapaxes(1, 2), cos, sin)
        v_raw = (a @ b["v"].astype(dtype)).reshape(B, T, KV, hd).swapaxes(1, 2)

        if i == 0:
            v0 = v_raw
            v = v_raw
        elif arm == "dense_untied_vr":
            lam = b["lam"].astype(jnp.float32)
            if ablation == "no_skip" or (ablation == "ablate_late_6_7" and i in (L - 2, L - 1)) or (ablation == "ablate_mid_3_4" and i in (L // 2 - 1, L // 2)):
                lam = jnp.array(1.0, dtype=jnp.float32)
            v0_ref = v0[:, :, perm_idx, :] if (ablation == "perm_v0" and perm_idx is not None) else v0
            v_cur32 = v_raw.astype(jnp.float32)
            v0_32 = v0_ref.astype(jnp.float32)
            part_cur = lam * v_cur32
            part_v0 = (1.0 - lam) * v0_32
            v = (part_cur + part_v0).astype(dtype)
            if return_diag:
                n_raw = jnp.sqrt(jnp.mean(jnp.square(v_cur32)) + 1e-12)
                n_v0 = jnp.sqrt(jnp.mean(jnp.square(v0_32)) + 1e-12)
                n_pcur = jnp.sqrt(jnp.mean(jnp.square(part_cur), axis=-1) + 1e-12)
                n_pv0 = jnp.sqrt(jnp.mean(jnp.square(part_v0), axis=-1) + 1e-12)
                eff_v0_share = jnp.mean(n_pv0 / (n_pcur + n_pv0 + 1e-12))
                dot = jnp.sum(v_cur32 * v0_32, axis=-1)
                denom = jnp.sqrt(jnp.sum(jnp.square(v_cur32), axis=-1) * jnp.sum(jnp.square(v0_32), axis=-1) + 1e-12)
                cos_sim = jnp.mean(dot / denom)
                layer_diags.append((lam, n_raw, n_v0, eff_v0_share, cos_sim))
        else:
            v = v_raw

        qg = q.reshape(B, KV, rep, T, hd)
        s = jnp.einsum("bkrth,bksh->bkrts", qg, k, preferred_element_type=jnp.float32) * (hd ** -0.5)
        s = jnp.where(causal, s, jnp.finfo(jnp.float32).min)
        p = jax.nn.softmax(s, -1).astype(dtype)
        att = jnp.einsum("bkrts,bksh->bkrth", p, v).reshape(B, H, T, hd).swapaxes(1, 2).reshape(B, T, H * hd)
        h = h + att @ b["o"].astype(dtype)

        gu = rms(h, b["n2"]) @ b["gu"].astype(dtype)
        g, u = jnp.split(gu, 2, -1)
        h = h + (jax.nn.silu(g) * u) @ b["dn"].astype(dtype)

    h_norm = rms(h, params["nf"])
    w_out = params["lm_head"].astype(dtype) if "lm_head" in params else params["emb"].astype(dtype).T
    logits = h_norm @ w_out
    if return_diag:
        return logits, layer_diags
    return logits


def make_muon_optimizer(lr: float, warmup: int, total_steps: int, lr_floor: float = 0.01, wd: float = 0.1):
    total = max(1, total_steps)
    wu = max(1, warmup)
    sched = lambda n: lr * jnp.minimum(1.0, (n + 1) / wu) * (
        lr_floor + (1.0 - lr_floor) * 0.5 * (1.0 + jnp.cos(jnp.pi * jnp.minimum(1.0, (n + 1) / total)))
    )
    dn = optax.contrib.MuonDimensionNumbers()

    def dims(p):
        d_tree = dict(
            emb=None,
            nf=None,
            blocks=[
                {k: (dn if getattr(v, "ndim", 0) == 2 else None) for k, v in blk.items()}
                for blk in p["blocks"]
            ],
        )
        if "lm_head" in p:
            d_tree["lm_head"] = None
        return d_tree

    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.contrib.muon(
            sched,
            weight_decay=wd,
            adam_b1=0.9,
            adam_b2=0.95,
            adam_weight_decay=wd,
            consistent_rms=0.2,
            muon_weight_dimension_numbers=dims,
        ),
    )


def save_ckpt(path: Path, params: dict, arm: str) -> str:
    flat = {"emb": np.asarray(params["emb"]), "nf": np.asarray(params["nf"]), "arm": np.array(arm)}
    if "lm_head" in params:
        flat["lm_head"] = np.asarray(params["lm_head"])
    for i, b in enumerate(params["blocks"]):
        for k, v in b.items():
            flat[f"b{i}_{k}"] = np.asarray(v)
    np.savez(path, **flat)
    return sha256_file(path)


def eval_crops(arr: np.ndarray, seq: int = 512, n_batches: int = 20, batch_size: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    starts = rng.integers(0, len(arr) - (seq + 1), size=(n_batches, batch_size))
    out = np.empty((n_batches, batch_size, seq + 1), dtype=np.int32)
    for b in range(n_batches):
        for r in range(batch_size):
            s = int(starts[b, r])
            out[b, r] = arr[s : s + seq + 1]
    return out


def score_7tasks(params: dict, cfg: dict, arm: str, tasks_path: Path, dev, batch_size: int = 16, seq_len: int = 512) -> dict:
    data = json.loads(tasks_path.read_text())

    @jax.jit
    def score_fn(p, x, y):
        lp = jax.nn.log_softmax(forward_with_diag(p, x, cfg, arm, jnp.bfloat16).astype(jnp.float32), -1)
        return jnp.take_along_axis(lp, y[..., None], -1)[..., 0], jnp.argmax(lp, -1) == y

    ents = []
    for task, rows in data.items():
        for e, r in enumerate(rows):
            pairs = [(c, r["cont"]) for c in r["ctxs"]] if task == "winogrande" else [(r["ctx"], k) for k in r["conts"]]
            for j, (ctx, cont) in enumerate(pairs):
                seq = ([EOS] + ctx + cont)[-(seq_len + 1) :]
                n_ctx = len(seq) - len(cont)
                ents.append((task, e, j, seq, n_ctx - 1, len(seq) - 1))
    ents.sort(key=lambda x: len(x[3]))

    S = {}
    for i in range(0, len(ents), batch_size):
        b = ents[i : i + batch_size]
        L = min(seq_len, 64 * math.ceil((max(len(x[3]) for x in b) - 1) / 64))
        X = np.zeros((batch_size, L), np.int32)
        Y = np.zeros((batch_size, L), np.int32)
        for r, x in enumerate(b):
            s = np.asarray(x[3], np.int32)
            X[r, : len(s) - 1] = s[:-1]
            Y[r, : len(s) - 1] = s[1:]
        lp, ok = score_fn(params, jax.device_put(X, dev), jax.device_put(Y, dev))
        lp, ok = np.asarray(lp), np.asarray(ok)
        for r, x in enumerate(b):
            S[x[:3]] = (float(lp[r, x[4] : x[5]].sum()), bool(ok[r, x[4] : x[5]].all()))

    res = {}
    for task, rows in data.items():
        if task == "lambada":
            ll = np.array([S[(task, e, 0)][0] for e in range(len(rows))])
            res[task] = {
                "acc": round(float(np.mean([S[(task, e, 0)][1] for e in range(len(rows))])), 4),
                "ppl": round(float(np.exp(-ll.mean())), 2),
                "n": len(rows),
            }
            continue
        acc, accn = [], []
        for e, r in enumerate(rows):
            k = len(r["ctxs"]) if task == "winogrande" else len(r["conts"])
            sc = np.array([S[(task, e, j)][0] for j in range(k)])
            acc.append(int(np.argmax(sc)) == r["gold"])
            if task != "winogrande":
                lens = np.array(r["cont_chars"])
                accn.append(int(np.argmax(sc / np.maximum(lens, 1))) == r["gold"])
        res[task] = {"acc": round(float(np.mean(acc)), 4), "n": len(rows), "chance": CHANCE[task]}
        if accn:
            res[task]["acc_norm"] = round(float(np.mean(accn)), 4)
    return res


def gcs_upload(src: Path, dst_prefix: str):
    if not dst_prefix:
        return
    dst = f"{dst_prefix.rstrip('/')}/{src.name}"
    for _ in range(3):
        r = subprocess.run(["gcloud", "storage", "cp", str(src), dst, "--quiet"], capture_output=True, text=True)
        if r.returncode == 0:
            return
        time.sleep(3)


def run_single_job(
    arm: str,
    seed: int,
    size: str,
    data_dir: Path,
    val_dir: Path,
    out_dir: Path,
    tasks_path: Path | None,
    gcs_prefix: str,
    device_idx: int,
    train_tokens: int,
    batch_size: int = 64,
    micro: int = 0,
    seq: int = 512,
    lr: float = 4e-3,
    wd: float = 0.1,
    lr_floor: float = 0.01,
    warmup: int = 500,
    chunk_steps: int = 200,
    eval_every: int = 4000,
    eval_batches: int = 20,
    provenance_base: dict | None = None,
) -> dict:
    dev = jax.devices()[device_idx]
    cfg = resolve_cfg(size, arm)
    micro = micro or batch_size
    assert batch_size % micro == 0
    accum = batch_size // micro
    tok_per_step = batch_size * seq
    steps = max(chunk_steps, (train_tokens // tok_per_step // chunk_steps) * chunk_steps)

    manifest = json.loads((val_dir / "manifest.json").read_text())
    bpt = {d: manifest["domains"][d]["val"]["bytes_per_token"] for d in manifest["mix"]}
    mix_w = manifest["mix"]
    train_mm = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    vals = {
        d: eval_crops(np.memmap(val_dir / f"val_{d}.bin", dtype=np.uint16, mode="r"), seq=seq, n_batches=eval_batches)
        for d in mix_w
    }

    with jax.default_device(dev):
        params = init_params(jax.random.PRNGKey(seed), cfg, arm, vocab=VOCAB)
        param_count = n_params(params)
        opt = make_muon_optimizer(lr=lr, warmup=warmup, total_steps=steps, lr_floor=lr_floor, wd=wd)
        opt_state = opt.init(params)

    def loss_fn(p, tok):
        lg = forward_with_diag(p, tok[:, :-1], cfg, arm, jnp.bfloat16).astype(jnp.float32)
        return optax.softmax_cross_entropy_with_integer_labels(lg, tok[:, 1:]).mean()

    @jax.jit
    def run_chunk(p, st, toks):
        def one_step(carry, tok):
            p_c, st_c = carry
            if accum == 1:
                l, g = jax.value_and_grad(loss_fn)(p_c, tok)
            else:
                def acc_fn(a, tk):
                    lv, gm = jax.value_and_grad(loss_fn)(p_c, tk)
                    return (a[0] + lv / accum, jax.tree_util.tree_map(lambda x, y: x + y / accum, a[1], gm)), None
                (l, g), _ = jax.lax.scan(
                    acc_fn,
                    (jnp.zeros((), jnp.float32), jax.tree_util.tree_map(jnp.zeros_like, p_c)),
                    tok.reshape(accum, micro, tok.shape[1]),
                )
            gn = optax.global_norm(g)
            u, st_next = opt.update(g, st_c, p_c)
            return (optax.apply_updates(p_c, u), st_next), (l, gn)

        (p_out, st_out), (losses, gnorms) = jax.lax.scan(one_step, (p, st), toks)
        return p_out, st_out, losses[-1], gnorms[-1]

    @jax.jit
    def eval_batch_fn(p, x, y):
        lg = forward_with_diag(p, x, cfg, arm, jnp.bfloat16).astype(jnp.float32)
        lp = jax.nn.log_softmax(lg, -1)
        nll = -jnp.take_along_axis(lp, y[..., None], -1)[..., 0]
        gold_lg = jnp.take_along_axis(lg, y[..., None], -1)
        rank0 = jnp.sum(lg > gold_lg, axis=-1)
        return nll, rank0

    def make_ablation_eval(ablation_mode: str):
        @jax.jit
        def fn(p, x, y, p_idx):
            lg = forward_with_diag(p, x, cfg, arm, jnp.bfloat16, ablation=ablation_mode, perm_idx=p_idx).astype(jnp.float32)
            lp = jax.nn.log_softmax(lg, -1)
            return -jnp.mean(jnp.take_along_axis(lp, y[..., None], -1))
        return fn

    @jax.jit
    def diag_batch_fn(p, x):
        _, diags = forward_with_diag(p, x, cfg, arm, jnp.bfloat16, return_diag=True)
        return diags

    def evaluate_full(p, include_ranks: bool = False) -> tuple[dict, dict | None]:
        dom_bpb = {}
        rank_acc = {"r1": 0.0, "r2_8": 0.0, "r9_64": 0.0, "r65p": 0.0, "top1": 0.0, "top64": 0.0}
        for d, batches in vals.items():
            tot_nll = 0.0
            r1 = r28 = r964 = r65 = t1 = t64 = 0.0
            n_tok = batches.shape[0] * batches.shape[1] * seq
            for b in batches:
                nll, rk = eval_batch_fn(p, jax.device_put(b[:, :-1], dev), jax.device_put(b[:, 1:], dev))
                tot_nll += float(jnp.sum(nll))
                if include_ranks:
                    bits = nll / math.log(2.0)
                    r1 += float(jnp.sum(jnp.where(rk == 0, bits, 0.0)))
                    r28 += float(jnp.sum(jnp.where((rk >= 1) & (rk <= 7), bits, 0.0)))
                    r964 += float(jnp.sum(jnp.where((rk >= 8) & (rk <= 63), bits, 0.0)))
                    r65 += float(jnp.sum(jnp.where(rk >= 64, bits, 0.0)))
                    t1 += float(jnp.sum(rk == 0))
                    t64 += float(jnp.sum(rk < 64))
            scale_bpb = 1.0 / (n_tok * math.log(2.0) * bpt[d])
            dom_bpb[d] = round(tot_nll * scale_bpb, 4)
            if include_ranks:
                w = mix_w[d]
                denom_bpb = n_tok * bpt[d]
                rank_acc["r1"] += w * (r1 / denom_bpb)
                rank_acc["r2_8"] += w * (r28 / denom_bpb)
                rank_acc["r9_64"] += w * (r964 / denom_bpb)
                rank_acc["r65p"] += w * (r65 / denom_bpb)
                rank_acc["top1"] += w * (t1 / n_tok)
                rank_acc["top64"] += w * (t64 / n_tok)
        dom_bpb["mix"] = round(sum(mix_w[d] * dom_bpb[d] for d in mix_w), 4)
        ranks_out = (
            {
                "rank1_calib": round(rank_acc["r1"], 4),
                "rank2_8_rerank": round(rank_acc["r2_8"], 4),
                "rank9_64_mid": round(rank_acc["r9_64"], 4),
                "rank65plus_miss": round(rank_acc["r65p"], 4),
                "top1_acc": round(rank_acc["top1"], 4),
                "top64_acc": round(rank_acc["top64"], 4),
            }
            if include_ranks
            else None
        )
        return dom_bpb, ranks_out

    rng = np.random.default_rng(seed + 1)
    warm_toks = np.zeros((chunk_steps, batch_size, seq + 1), dtype=np.int32)
    _, _, warm_l, _ = run_chunk(params, opt_state, jax.device_put(warm_toks, dev))
    jax.block_until_ready(warm_l)
    _ = eval_batch_fn(params, jax.device_put(vals["web"][0][:, :-1], dev), jax.device_put(vals["web"][0][:, 1:], dev))

    curve = []
    train_sec = 0.0
    t0 = time.perf_counter()

    for step in range(chunk_steps, steps + 1, chunk_steps):
        starts = rng.integers(0, len(train_mm) - (seq + 1), size=(chunk_steps, batch_size))
        buf = np.empty((chunk_steps, batch_size, seq + 1), dtype=np.int32)
        for c_i in range(chunk_steps):
            for b_i in range(batch_size):
                s_idx = int(starts[c_i, b_i])
                buf[c_i, b_i] = train_mm[s_idx : s_idx + seq + 1]

        params, opt_state, loss_val, g_norm = run_chunk(params, opt_state, jax.device_put(buf, dev))

        if step % eval_every == 0 or step == steps:
            jax.block_until_ready(loss_val)
            train_sec += time.perf_counter() - t0
            ev_bpb, rk_out = evaluate_full(params, include_ranks=(step == steps))
            row = {
                "step": step,
                "tokens": step * tok_per_step,
                "train_sec": round(train_sec, 2),
                "tokens_per_second": round((step * tok_per_step) / max(train_sec, 1e-6)),
                "train_loss": round(float(loss_val), 4),
                "grad_norm": round(float(g_norm), 4),
                **ev_bpb,
            }
            curve.append(row)
            print(json.dumps({"arm": arm, "seed": seed, "dev": device_idx, **row}), flush=True)
            t0 = time.perf_counter()

    diagnostics = {}
    if "lm_head" in params:
        emb32 = params["emb"].astype(jnp.float32)  # [V, d]
        hd32 = params["lm_head"].astype(jnp.float32).T  # [V, d]
        dot_eh = jnp.sum(emb32 * hd32, axis=-1)
        den_eh = jnp.sqrt(jnp.sum(emb32 ** 2, axis=-1) * jnp.sum(hd32 ** 2, axis=-1) + 1e-12)
        diagnostics["mean_cos_emb_vs_lm_head"] = round(float(jnp.mean(dot_eh / den_eh)), 5)
        diagnostics["emb_rms"] = round(float(jnp.sqrt(jnp.mean(emb32 ** 2))), 5)
        diagnostics["lm_head_rms"] = round(float(jnp.sqrt(jnp.mean(hd32 ** 2))), 5)

    if arm == "dense_untied_vr":
        diagnostics["lambdas"] = [round(float(b["lam"]), 5) for b in params["blocks"][1:]]
        acc_d = [[0.0] * 5 for _ in range(cfg["layers"] - 1)]
        n_diag_b = 0
        for d in mix_w:
            for b in vals[d][:4]:
                dout = diag_batch_fn(params, jax.device_put(b[:, :-1], dev))
                for l_i, tup in enumerate(dout):
                    for k_i in range(5):
                        acc_d[l_i][k_i] += float(tup[k_i])
                n_diag_b += 1
        diagnostics["layer_value_telemetry"] = [
            {
                "layer": l_i + 1,
                "nominal_lam": round(acc_d[l_i][0] / n_diag_b, 5),
                "nominal_v0_coeff": round(1.0 - acc_d[l_i][0] / n_diag_b, 5),
                "v_cur_rms": round(acc_d[l_i][1] / n_diag_b, 5),
                "v0_rms": round(acc_d[l_i][2] / n_diag_b, 5),
                "effective_v0_norm_share": round(acc_d[l_i][3] / n_diag_b, 5),
                "cos_v_cur_v0": round(acc_d[l_i][4] / n_diag_b, 5),
            }
            for l_i in range(cfg["layers"] - 1)
        ]

    ablations = {}
    if arm == "dense_untied_vr":
        perm_rng = np.random.default_rng(999)
        perm_idx = jax.device_put(jnp.asarray(perm_rng.permutation(seq), dtype=jnp.int32), dev)
        for ab_mode in ("no_skip", "ablate_late_6_7", "ablate_mid_3_4", "perm_v0"):
            ab_fn = make_ablation_eval(ab_mode)
            ab_dom = {}
            for d, batches in vals.items():
                tot = sum(
                    float(ab_fn(params, jax.device_put(b[:, :-1], dev), jax.device_put(b[:, 1:], dev), perm_idx))
                    for b in batches
                ) / len(batches)
                ab_dom[d] = round(tot / (math.log(2.0) * bpt[d]), 4)
            ab_dom["mix"] = round(sum(mix_w[d] * ab_dom[d] for d in mix_w), 4)
            ab_dom["delta_mix_vs_intact"] = round(ab_dom["mix"] - curve[-1]["mix"], 4)
            ablations[ab_mode] = ab_dom

    ckpt_name = f"{arm}-{size}-s{seed}.npz"
    ckpt_path = out_dir / ckpt_name
    ckpt_sha = save_ckpt(ckpt_path, params, arm)

    tasks_eval = None
    if tasks_path is not None and tasks_path.exists():
        tasks_eval = score_7tasks(params, cfg, arm, tasks_path, dev, batch_size=16, seq_len=seq)

    non_embed_params = param_count - int(params["emb"].size) - (int(params["lm_head"].size) if "lm_head" in params else 0)
    result = {
        "schema": "nanolm_untied_head_57m_factorial_v1",
        "completed": True,
        "arm": arm,
        "size": size,
        "cfg": cfg,
        "seed": seed,
        "params": param_count,
        "non_embed_params": non_embed_params,
        "device_index": device_idx,
        "device_kind": dev.device_kind,
        "jax_version": jax.__version__,
        "optax_version": optax.__version__,
        "recipe": {
            "opt": "optax.contrib.muon",
            "lr": lr,
            "wd": wd,
            "wd_on": "matrices_with_adamw_on_emb_and_lm_head",
            "lr_floor": lr_floor,
            "warmup": warmup,
            "batch_size": batch_size,
            "micro": micro,
            "seq": seq,
            "steps": steps,
            "train_tokens": steps * tok_per_step,
        },
        "provenance": {
            **(provenance_base or {}),
            "code_sha256": CODE_SHA256,
            "ckpt_file": ckpt_name,
            "ckpt_sha256": ckpt_sha,
        },
        "curve": curve,
        "rank_buckets_mix": rk_out,
        "diagnostics": diagnostics,
        "causal_ablations": ablations,
        "downstream_7tasks": tasks_eval,
    }
    res_path = out_dir / f"run_{arm}_{size}_s{seed}.json"
    res_path.write_text(json.dumps(result, indent=2))
    gcs_upload(res_path, gcs_prefix)
    gcs_upload(ckpt_path, f"{gcs_prefix.rstrip('/')}/ckpt" if gcs_prefix else "")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--val-dir", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gcs", default="")
    ap.add_argument("--size", default="M", choices=("T", "M"))
    ap.add_argument("--seeds", default="1000,1001,1002")
    ap.add_argument("--arms", default="dense_tied,dense_untied,dense_untied_vr,dense_tied_57m")
    ap.add_argument("--train-tokens", type=int, default=800_000_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro", type=int, default=0)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--lr-floor", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--chunk-steps", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=4000)
    ap.add_argument("--eval-batches", type=int, default=20)
    a = ap.parse_args()

    data_dir = Path(a.data_dir)
    val_dir = Path(a.val_dir) if a.val_dir else data_dir
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = Path(a.tasks) if a.tasks else None

    prov_base = {
        "train_bin_bytes": (data_dir / "train.bin").stat().st_size,
        "train_bin_head64mb_sha256": sha256_file(data_dir / "train.bin", max_bytes=64 * 1024 * 1024),
        "manifest_sha256": sha256_file(val_dir / "manifest.json"),
        "val_sha256": {
            d: sha256_file(val_dir / f"val_{d}.bin")
            for d in ("web", "textbook", "math", "code")
        },
        "tasks_sha256": sha256_file(tasks_path) if (tasks_path and tasks_path.exists()) else None,
    }

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    n_dev = len(jax.devices())
    print(json.dumps({"event": "start_untied_factorial", "n_devices": n_dev, "seeds": seeds, "arms": arms, "prov": prov_base}), flush=True)

    all_errors = []
    for s_idx, seed in enumerate(seeds):
        q_jobs: queue.Queue = queue.Queue()
        for a_idx, arm in enumerate(arms):
            res_file = out_dir / f"run_{arm}_{a.size}_s{seed}.json"
            if res_file.exists():
                try:
                    existing = json.loads(res_file.read_text())
                    if existing.get("completed") is True:
                        print(json.dumps({"event": "skip_completed", "arm": arm, "seed": seed}), flush=True)
                        continue
                except Exception:
                    pass
            q_jobs.put((a_idx, arm))

        def worker(dev_id: int):
            while True:
                try:
                    a_idx, arm = q_jobs.get_nowait()
                except queue.Empty:
                    return
                assigned_dev = (a_idx + s_idx) % n_dev if n_dev >= len(arms) else dev_id
                try:
                    run_single_job(
                        arm=arm,
                        seed=seed,
                        size=a.size,
                        data_dir=data_dir,
                        val_dir=val_dir,
                        out_dir=out_dir,
                        tasks_path=tasks_path,
                        gcs_prefix=a.gcs,
                        device_idx=assigned_dev,
                        train_tokens=a.train_tokens,
                        batch_size=a.batch_size,
                        micro=a.micro,
                        seq=a.seq,
                        lr=a.lr,
                        wd=a.wd,
                        lr_floor=a.lr_floor,
                        warmup=a.warmup,
                        chunk_steps=a.chunk_steps,
                        eval_every=a.eval_every,
                        eval_batches=a.eval_batches,
                        provenance_base=prov_base,
                    )
                except Exception as exc:
                    err = {"event": "job_failed", "arm": arm, "seed": seed, "dev": assigned_dev, "error": repr(exc)}
                    print(json.dumps(err), flush=True)
                    all_errors.append(err)

        n_workers = min(n_dev, q_jobs.qsize())
        threads = [threading.Thread(target=worker, args=(d_i,)) for d_i in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(json.dumps({"event": "all_done", "errors": all_errors}), flush=True)
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
