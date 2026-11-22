"""Gate 1: Token-Level Bottleneck Decomposition (14M -> 40M -> 101M) + 101M Causal Cache Audit.

Pre-registered predictions (written before any execution of this script):
  B1 (101M causal in-context cache): At 101M (dense-L), exact 512-token causal cache gain on mix
     is <= 0.0012 BPB (halving from 40M's 0.0018) and < 0.0040 BPB on code, with causality check
     passing and foreign-window control within +-0.0005.
  B2 (Copyable vs Non-Copyable loss share): Non-copyable tokens (no earlier exact >=2-gram match
     emitting y_t in the 512-token prefix) account for > 85% of held-out mix BPB at 101M (> 75%
     even on code), and > 80% of the 14M -> 101M scaling gain (0.1584 BPB) comes from non-copyable tokens.
  B3 (Search/Reranking vs Parametric Miss): At 101M, y_t falls inside the model's Top-64 candidates
     on > 85% of held-out tokens, and Ranks 2..64 (where y_t is already a surfaced candidate but not
     argmax) hold > 40% of total NLL.
  B4 (Context horizon & Rare-token storage): Late context (pos 256..511) improves from 14M -> 101M
     by at least 0.02 BPB more than early context (pos 0..63) on math and code; rare corpus tokens
     (frequency rank > 8192) incur > 2x higher NLL/token than top-512 tokens.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize_scalar

VOCAB = 32768
SIZES = {
    "S": dict(d=256, layers=8, ffn=704, heads=4, kv=2),
    "M": dict(d=512, layers=8, ffn=1408, heads=8, kv=4),
    "L": dict(d=768, layers=12, ffn=2048, heads=12, kv=4),
}
ORDERS, BUCKETS = (3, 2, 1), 8

DATA_DIR = Path(os.environ.get("NANOLM_DATA_DIR", "/Volumes/Express1M2/nanolm-lab/data/v1"))
CKPT_DIR = Path(os.environ.get("NANOLM_CKPT_DIR", "/Volumes/Express1M2/nanolm-lab/ckpt"))
OUT_PATH = Path(__file__).resolve().parent / "results" / "gate1_bottleneck_results.json"


def load_params(path: Path):
    z = np.load(path)
    n = 1 + max(int(k[1:].split(".")[0]) for k in z.files if k.startswith("b"))
    return dict(
        emb=jnp.asarray(z["emb"]),
        nf=jnp.asarray(z["nf"]),
        blocks=[
            {k.split(".", 1)[1]: jnp.asarray(z[k]) for k in z.files if k.startswith(f"b{i}.")}
            for i in range(n)
        ],
    )


def rms(x, w, eps=1e-6):
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), -1, keepdims=True) + eps) * w.astype(jnp.float32)).astype(x.dtype)


def rope_tables(T, hd, theta=10000.0):
    inv = 1.0 / theta ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd)
    f = jnp.outer(jnp.arange(T, dtype=jnp.float32), inv)
    return jnp.cos(f), jnp.sin(f)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos.astype(x.dtype), sin.astype(x.dtype)
    return jnp.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).reshape(x.shape)


def forward(params, idx, c, dt=jnp.bfloat16):
    H, KV = c["heads"], c["kv"]
    p = jax.tree_util.tree_map(lambda a: a.astype(dt), params)
    B, T = idx.shape
    x = p["emb"][idx]
    d = x.shape[-1]
    hd = d // H
    rep = H // KV
    cos, sin = rope_tables(T, hd)
    mask = jnp.tril(jnp.ones((T, T), bool))
    neg = jnp.finfo(jnp.float32).min
    for b in p["blocks"]:
        h = rms(x, b["n1"])
        q = rms((h @ b["q"]).reshape(B, T, KV, rep, hd), b["qn"]).transpose(0, 2, 3, 1, 4)
        k = rms((h @ b["k"]).reshape(B, T, KV, hd), b["kn"]).transpose(0, 2, 1, 3)
        v = (h @ b["v"]).reshape(B, T, KV, hd).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        s = jnp.einsum("bgrqd,bgkd->bgrqk", q, k).astype(jnp.float32) * (hd ** -0.5)
        a = jax.nn.softmax(jnp.where(mask, s, neg), -1).astype(dt)
        y = jnp.einsum("bgrqk,bgkd->bgrqd", a, v).transpose(0, 3, 1, 2, 4).reshape(B, T, H * hd)
        x = x + y @ b["o"]
        g, u = jnp.split(rms(x, b["n2"]) @ b["gu"], 2, -1)
        x = x + (jax.nn.silu(g) * u) @ b["dn"]
    return (rms(x, p["nf"]) @ p["emb"].T).astype(jnp.float32)


def make_eval_fn(c):
    @jax.jit
    def eval_batch(params, x):
        # x: [B, T+1]
        lg = forward(params, x[:, :-1], c, jnp.bfloat16)  # [B, T, V] float32
        lp_all = jax.nn.log_softmax(lg, axis=-1)
        tgt = x[:, 1:]
        true_lp = jnp.take_along_axis(lp_all, tgt[..., None], axis=-1)[..., 0]
        true_lg = jnp.take_along_axis(lg, tgt[..., None], axis=-1)[..., 0]
        # Exact 0-indexed rank: number of tokens with strictly higher logit
        rank = jnp.sum(lg > true_lg[..., None], axis=-1).astype(jnp.int32)
        # Entropy of predictive distribution in bits
        ent_bits = -jnp.sum(jnp.exp(lp_all) * lp_all, axis=-1) / math.log(2.0)
        return true_lp, rank, ent_bits
    return eval_batch


def cache_probs_and_copy2(tgt_tokens, foreign=None):
    """Exact cache_probs from cache_exchange.py + boolean indicator for exact >=2-gram prefix match emitting y_t."""
    T = len(tgt_tokens) - 1
    s = np.zeros(T, np.float32)
    g = np.zeros(T, np.int64)
    copy2 = np.zeros(T, bool)  # True iff an earlier >=1-context (>=2-gram) in prefix emitted y_t
    seen = [dict() for _ in ORDERS]
    if foreign is not None:
        for t in range(len(foreign) - 1):
            for oi, n in enumerate(ORDERS):
                if t >= n - 1:
                    e = seen[oi].setdefault(tuple(foreign[t - n + 1 : t + 1]), [0, {}])
                    e[0] += 1
                    e[1][foreign[t + 1]] = e[1].get(foreign[t + 1], 0) + 1
    for t in range(T):
        y = tgt_tokens[t + 1]
        for oi, n in enumerate(ORDERS):
            if t >= n - 1:
                e = seen[oi].get(tuple(tgt_tokens[t - n + 1 : t + 1]))
                if e is not None:
                    s[t] = e[1].get(y, 0) / e[0]
                    g[t] = 1 + oi * BUCKETS + min(int(math.log2(e[0])), BUCKETS - 1)
                    break
        if foreign is None:
            # Check if ANY >=1-token context (i.e. >=2-gram span) occurred earlier with follower y
            for oi, n in enumerate(ORDERS):
                if t >= n - 1:
                    e = seen[oi].get(tuple(tgt_tokens[t - n + 1 : t + 1]))
                    if e is not None and e[1].get(y, 0) > 0:
                        copy2[t] = True
                        break
            for oi, n in enumerate(ORDERS):
                if t >= n - 1:
                    e = seen[oi].setdefault(tuple(tgt_tokens[t - n + 1 : t + 1]), [0, {}])
                    e[0] += 1
                    e[1][y] = e[1].get(y, 0) + 1
    return s, g, copy2


def causality_check(window, rng, trials=8):
    s0, g0, c0 = cache_probs_and_copy2(window)
    for _ in range(trials):
        t = int(rng.integers(8, len(window) - 8))
        w = list(window)
        w[t + 2 :] = [int(x) for x in rng.integers(0, VOCAB, len(w) - t - 2)]
        s1, g1, c1 = cache_probs_and_copy2(w)
        if not (np.array_equal(s0[: t + 1], s1[: t + 1]) and np.array_equal(g0[: t + 1], g1[: t + 1]) and np.array_equal(c0[: t + 1], c1[: t + 1])):
            return False
    return True


def fit_table(lp, s, g):
    lam = np.zeros(1 + len(ORDERS) * BUCKETS)
    p = np.exp(lp)
    for k in range(1, len(lam)):
        m = g == k
        if m.sum() < 50:
            continue
        lam[k] = minimize_scalar(
            lambda x: -np.mean(np.log(x * s[m] + (1 - x) * p[m] + 1e-12)),
            bounds=(0.0, 1.0),
            method="bounded",
        ).x
    return lam


def nll_mix(lp, s, g, lam):
    return -np.log(lam[g] * s + (1 - lam[g]) * np.exp(lp) + 1e-12)


def main():
    t0 = time.time()
    manifest = json.load(open(DATA_DIR / "manifest.json"))
    weights = manifest["mix"]
    T = 512

    # 1. Corpus frequency rank from training data ONLY (first 50M tokens)
    full = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    counts = np.bincount(np.asarray(full[:50_000_000], dtype=np.int64), minlength=VOCAB)
    order = np.argsort(-counts)
    freq_rank = np.empty(VOCAB, dtype=np.int32)
    freq_rank[order] = np.arange(VOCAB, dtype=np.int32)

    assert all(
        causality_check(np.asarray(full[i : i + T + 1], dtype=np.int32).tolist(), np.random.default_rng(99 + i))
        for i in (0, 1000003, 500000007)
    ), "Causality check failed!"

    # Precompute exact validation crops & prefix-copy metadata once (shared across S, M, L)
    val_crops = {}
    val_cache_meta = {}
    for dom in ("web", "textbook", "math", "code"):
        d = np.fromfile(DATA_DIR / f"val_{dom}.bin", dtype=np.uint16).astype(np.int32)
        r0 = np.random.default_rng(0)
        batches = []
        meta = []
        for _ in range(20):
            ix = r0.integers(0, len(d) - T - 1, 16)
            X = np.stack([d[i : i + T + 1] for i in ix])
            batches.append(X)
            sgc = [cache_probs_and_copy2(X[b].tolist()) for b in range(16)]
            sf = [cache_probs_and_copy2(X[b].tolist(), foreign=X[(b + 1) % 16].tolist()) for b in range(16)]
            meta.append((
                np.stack([x[0] for x in sgc]),
                np.stack([x[1] for x in sgc]),
                np.stack([x[2] for x in sgc]),
                np.stack([x[0] for x in sf]),
                np.stack([x[1] for x in sf]),
            ))
        val_crops[dom] = batches
        val_cache_meta[dom] = meta
    print(f"[{time.time()-t0:.1f}s] Precomputed validation crops & exact prefix-copy masks.")

    report = {"models": {}, "cache_L": {}}
    pos_idx = np.broadcast_to(np.arange(T, dtype=np.int32)[None, :], (16, T))

    for sz in ("S", "M", "L"):
        ts = time.time()
        ckpt_path = CKPT_DIR / f"dense-{sz}-lr0.001-s1000.npz"
        rep_json = json.load(open(CKPT_DIR / f"dense-{sz}-lr0.001-s1000.json"))["curve"][-1]
        params = load_params(ckpt_path)
        eval_fn = make_eval_fn(SIZES[sz])

        # For L, also run exact 32-batch training-window fit for cache-exchange-L!
        if sz == "L":
            rng = np.random.default_rng(54321)
            LP_fit, S_fit, G_fit, S2_fit, G2_fit = [], [], [], [], []
            for _ in range(32):
                ix = rng.integers(0, len(full) - T - 1, 32)
                X = np.stack([np.asarray(full[i : i + T + 1]) for i in ix]).astype(np.int32)
                lp1, _, _ = eval_fn(params[:16] if isinstance(params, list) else params, X[:16])
                lp2, _, _ = eval_fn(params, X[16:])
                lp = np.concatenate([np.asarray(lp1), np.asarray(lp2)], axis=0)
                LP_fit.append(lp.ravel())
                for b in range(32):
                    row = X[b].tolist()
                    s, g, _ = cache_probs_and_copy2(row)
                    S_fit.append(s)
                    G_fit.append(g)
                    sf, gf, _ = cache_probs_and_copy2(row, foreign=X[(b + 1) % 32].tolist())
                    S2_fit.append(sf)
                    G2_fit.append(gf)
            lam_L = fit_table(np.concatenate(LP_fit), np.concatenate(S_fit), np.concatenate(G_fit))
            lam_Lf = fit_table(np.concatenate(LP_fit), np.concatenate(S2_fit), np.concatenate(G2_fit))

        dom_stats = {}
        cache_L_rows = {}
        for dom in ("web", "textbook", "math", "code"):
            bpt = manifest["domains"][dom]["val"]["bytes_per_token"]
            nll_list, rank_list, ent_list, copy_list, f_rank_list, pos_list = [], [], [], [], [], []
            mix_nll_list, for_nll_list = [], []
            for X, (s, g, copy2, s2, g2) in zip(val_crops[dom], val_cache_meta[dom]):
                lp, rnk, ent = eval_fn(params, X)
                lp_np = np.asarray(lp)
                nll_list.append(-lp_np)
                rank_list.append(np.asarray(rnk))
                ent_list.append(np.asarray(ent))
                copy_list.append(copy2)
                f_rank_list.append(freq_rank[X[:, 1:]])
                pos_list.append(pos_idx)
                if sz == "L":
                    mix_nll_list.append(nll_mix(lp_np, s, g, lam_L))
                    for_nll_list.append(nll_mix(lp_np, s2, g2, lam_Lf))

            nll_arr = np.concatenate(nll_list).ravel()  # nats per token
            bits_arr = nll_arr / math.log(2.0)          # bits per token
            bpb_scale = 1.0 / bpt                       # multiply mean bits/tok by 1/bpt -> BPB
            rnk_arr = np.concatenate(rank_list).ravel()
            ent_arr = np.concatenate(ent_list).ravel()
            cpy_arr = np.concatenate(copy_list).ravel()
            frk_arr = np.concatenate(f_rank_list).ravel()
            pos_arr = np.concatenate(pos_list).ravel()

            total_bpb = float(bits_arr.mean() * bpb_scale)
            assert abs(total_bpb - rep_json[dom]) <= 0.0002, f"Mismatch {sz} {dom}: {total_bpb} vs {rep_json[dom]}"

            # Additive BPB contribution of a mask m: sum(bits[m]) / N_total / bpt (sums exactly to total_bpb!)
            contrib = lambda m: round(float(bits_arr[m].sum() / len(bits_arr) * bpb_scale), 4)
            cond_bpb = lambda m: round(float(bits_arr[m].mean() * bpb_scale), 4) if m.any() else 0.0
            share = lambda m: round(float(m.mean()), 4)

            dom_stats[dom] = {
                "total_bpb": round(total_bpb, 4),
                "reported_bpb": rep_json[dom],
                "mean_entropy_bits": round(float(ent_arr.mean()), 3),
                "copy_split": {
                    "copyable_token_share": share(cpy_arr),
                    "copyable_bpb_contrib": contrib(cpy_arr),
                    "noncopy_bpb_contrib": contrib(~cpy_arr),
                    "copyable_cond_bpb": cond_bpb(cpy_arr),
                    "noncopy_cond_bpb": cond_bpb(~cpy_arr),
                },
                "rank_split": {
                    "top1_acc": share(rnk_arr == 0),
                    "top8_acc": share(rnk_arr < 8),
                    "top64_acc": share(rnk_arr < 64),
                    "rank1_calib_bpb_contrib": contrib(rnk_arr == 0),
                    "rank2_8_rerank_bpb_contrib": contrib((rnk_arr >= 1) & (rnk_arr < 8)),
                    "rank9_64_mid_bpb_contrib": contrib((rnk_arr >= 8) & (rnk_arr < 64)),
                    "rank65plus_miss_bpb_contrib": contrib(rnk_arr >= 64),
                },
                "pos_cond_bpb": {
                    "early_0_63": cond_bpb(pos_arr < 64),
                    "mid_64_255": cond_bpb((pos_arr >= 64) & (pos_arr < 256)),
                    "late_256_511": cond_bpb(pos_arr >= 256),
                },
                "freq_split": {
                    "common_0_511_share": share(frk_arr < 512),
                    "mid_512_8191_share": share((frk_arr >= 512) & (frk_arr < 8192)),
                    "rare_8192plus_share": share(frk_arr >= 8192),
                    "common_0_511_bpb_contrib": contrib(frk_arr < 512),
                    "mid_512_8191_bpb_contrib": contrib((frk_arr >= 512) & (frk_arr < 8192)),
                    "rare_8192plus_bpb_contrib": contrib(frk_arr >= 8192),
                    "common_bits_per_tok": round(float(bits_arr[frk_arr < 512].mean()), 3),
                    "rare_bits_per_tok": round(float(bits_arr[frk_arr >= 8192].mean()), 3),
                },
            }
            if sz == "L":
                m_bpb = float(np.concatenate(mix_nll_list).mean() / math.log(2.0) / bpt)
                f_bpb = float(np.concatenate(for_nll_list).mean() / math.log(2.0) / bpt)
                cache_L_rows[dom] = {
                    "model": round(total_bpb, 4),
                    "with_cache": round(m_bpb, 4),
                    "foreign_cache": round(f_bpb, 4),
                    "gain": round(total_bpb - m_bpb, 4),
                    "foreign_gain": round(total_bpb - f_bpb, 4),
                }

        # Compute weighted mix across domains
        mix_total = round(sum(weights[d] * dom_stats[d]["total_bpb"] for d in weights), 4)
        mix_copy_c = round(sum(weights[d] * dom_stats[d]["copy_split"]["copyable_bpb_contrib"] for d in weights), 4)
        mix_noncopy_c = round(sum(weights[d] * dom_stats[d]["copy_split"]["noncopy_bpb_contrib"] for d in weights), 4)
        mix_r1 = round(sum(weights[d] * dom_stats[d]["rank_split"]["rank1_calib_bpb_contrib"] for d in weights), 4)
        mix_r2_8 = round(sum(weights[d] * dom_stats[d]["rank_split"]["rank2_8_rerank_bpb_contrib"] for d in weights), 4)
        mix_r9_64 = round(sum(weights[d] * dom_stats[d]["rank_split"]["rank9_64_mid_bpb_contrib"] for d in weights), 4)
        mix_r65 = round(sum(weights[d] * dom_stats[d]["rank_split"]["rank65plus_miss_bpb_contrib"] for d in weights), 4)
        mix_top1 = round(sum(weights[d] * dom_stats[d]["rank_split"]["top1_acc"] for d in weights), 4)
        mix_top64 = round(sum(weights[d] * dom_stats[d]["rank_split"]["top64_acc"] for d in weights), 4)
        mix_early = round(sum(weights[d] * dom_stats[d]["pos_cond_bpb"]["early_0_63"] for d in weights), 4)
        mix_late = round(sum(weights[d] * dom_stats[d]["pos_cond_bpb"]["late_256_511"] for d in weights), 4)

        dom_stats["mix"] = {
            "total_bpb": mix_total,
            "reported_bpb": rep_json["mix"],
            "copyable_bpb_contrib": mix_copy_c,
            "noncopy_bpb_contrib": mix_noncopy_c,
            "noncopy_loss_fraction": round(mix_noncopy_c / mix_total, 4),
            "top1_acc": mix_top1,
            "top64_acc": mix_top64,
            "rank1_calib_bpb_contrib": mix_r1,
            "rank2_8_rerank_bpb_contrib": mix_r2_8,
            "rank9_64_mid_bpb_contrib": mix_r9_64,
            "rank2_64_loss_fraction": round((mix_r2_8 + mix_r9_64) / mix_total, 4),
            "rank65plus_miss_bpb_contrib": mix_r65,
            "early_0_63_bpb": mix_early,
            "late_256_511_bpb": mix_late,
        }
        report["models"][sz] = dom_stats
        if sz == "L":
            for k in ("model", "with_cache", "foreign_cache", "gain", "foreign_gain"):
                cache_L_rows.setdefault("mix", {})[k] = round(sum(weights[d] * cache_L_rows[d][k] for d in weights), 4)
            report["cache_L"] = cache_L_rows
        print(f"[{time.time()-t0:.1f}s] Finished {sz} in {time.time()-ts:.1f}s | mix={mix_total:.4f} (reported {rep_json['mix']:.4f})")

    report["elapsed_seconds"] = round(time.time() - t0, 2)
    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(json.dumps({"S_mix": report["models"]["S"]["mix"], "M_mix": report["models"]["M"]["mix"], "L_mix": report["models"]["L"]["mix"], "cache_L": report["cache_L"]}, indent=2))


if __name__ == "__main__":
    main()
