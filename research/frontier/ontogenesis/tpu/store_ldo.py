"""The store test of the decision note (section 6), with the estimator corrected as recorded in section 10.

What changed, and why (both found by the independent review before the 101M checkpoint was scored):
1. Leave-document-out fitting. store_exchange.py fits the 25 mixing weights on training text with counts that leave
   out only the current position. Other copies of the same n-gram in the same document stay in, so the store looks
   more reliable there than it will on held-out text, and the fitted weights over-trust it. Here every count used
   for fitting excludes the whole current document. The fit still uses training text only, so the registered rule
   "no tuning on held-out text" is kept.
2. A control that can fail. The old uninformed-store control fitted its own weights, which all came out 0, so it
   equalled the model by construction. Here the real fitted weights are applied to the uninformed store (the store
   consulted with the context of a position 100,003 tokens away). That must make the loss worse; a gain would mean
   the evaluation leaks information.
3. Reliability diagnostic. For each gate cell: the store's mean probability of the true token under leave-position-out
   counts, under leave-document-out counts, and on held-out text.

Everything else is unchanged from store_exchange.py: the same store (4-gram counts over 48 evenly spaced 1M-token
chunks of the training file, 6% of it), the same fitting windows, the same evaluation crops, the same 25-cell table.
The registered decision rule is read on the leave-document-out estimate at 101M. Its scope is narrowed (section 10):
a gain under 0.01 closes this store as a complement, not the n-gram store line.
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import jax, jax.numpy as jnp
from train_jax import SIZES, forward
from store_exchange import Store, ctx_key, fit_table, nll, load_params, BITS, ORDERS, BUCKETS

EOS = 1


def doc_ids(tok, chunk):
    """Document index for every position: a new document after each end-of-document token and at each chunk start."""
    start = np.zeros(len(tok), bool); start[1:] = tok[:-1] == EOS; start[::chunk] = True
    return np.cumsum(start) - 1


def within_doc_counts(tok, doc):
    """For every position and order: how often its context, and its (context, next token) pair, occur in its own document."""
    out = {}
    mix = lambda a, b: (a * np.int64(-7046029254386353131)) ^ (b * np.int64(-4658895280553007687))   # hash of a pair of int64
    with np.errstate(over="ignore"):
        for n in ORDERS:
            ck = ctx_key(tok, n); ok = ck >= 0
            pk = (np.maximum(ck, 0) << BITS) | tok.astype(np.int64)
            res = []
            for key in (ck, pk):
                h = mix(doc.astype(np.int64) + 1, key)
                _, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
                c = cnt[inv].astype(np.int32); c[~ok] = 0; res.append(c)
            out[n] = tuple(res)
    return out


def lookup_counts(store, tok, sub_cx=None, sub_cxy=None, loo=0):
    """Store lookup with per-position counts removed: sub_* (per order) or a constant loo."""
    s = np.zeros(len(tok), np.float32); g = np.zeros(len(tok), np.int64); done = np.zeros(len(tok), bool)
    for oi, n in enumerate(ORDERS):
        pu, pc, cu, cc = store.t[n]; ck = ctx_key(tok, n); ok = ck >= 0
        cx = store._get(cu, cc, ck); cxy = store._get(pu, pc, (np.maximum(ck, 0) << BITS) | tok.astype(np.int64))
        if sub_cx is not None: cx = cx - sub_cx[n]; cxy = cxy - sub_cxy[n]
        else: cx = cx - loo; cxy = cxy - loo
        use = ok & ~done & (cx >= 1)
        s[use] = np.maximum(cxy[use], 0) / cx[use]
        g[use] = 1 + oi * BUCKETS + np.minimum(np.log2(cx[use]).astype(np.int64), BUCKETS - 1)
        done |= use
    return s, g


def reliability(s, g):
    return {int(k): round(float(s[g == k].mean()), 4) for k in range(1, 1 + len(ORDERS) * BUCKETS) if (g == k).sum() >= 200}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--reported", default="")
    ap.add_argument("--data", default="/Volumes/Express1M2/nanolm-lab/data/v1"); ap.add_argument("--out", default="")
    ap.add_argument("--seq", type=int, default=512); ap.add_argument("--fit-batches", type=int, default=64)
    a = ap.parse_args(); c = SIZES[a.size]; T = a.seq; t0 = time.time(); dt = jnp.bfloat16
    manifest = json.load(open(f"{a.data}/manifest.json"))
    full = np.memmap(f"{a.data}/train.bin", dtype=np.uint16, mode="r"); span = 1_000_000
    tok = np.concatenate([np.asarray(full[i:i + span]) for i in np.linspace(0, len(full) - span, 48).astype(np.int64)]).astype(np.int32)
    store = Store(tok); doc = doc_ids(tok, span); wd = within_doc_counts(tok, doc)
    s_loo, g_loo = lookup_counts(store, tok, loo=1)
    s_ldo, g_ldo = lookup_counts(store, tok, {n: wd[n][0] for n in ORDERS}, {n: wd[n][1] for n in ORDERS})
    params = load_params(a.ckpt)
    lp_one = jax.jit(lambda p, x: jnp.take_along_axis(jax.nn.log_softmax(forward(p, x[:, :-1], c, dt), -1), x[:, 1:, None], -1)[..., 0])
    lp_fn = lambda p, x: np.concatenate([np.asarray(lp_one(p, x[k:k + 8])) for k in range(0, len(x), 8)])   # 8 windows at a time: laptop memory
    rng = np.random.default_rng(12345); LP, I = [], []                              # same fitting windows as store_exchange.py
    for _ in range(a.fit_batches):
        ix = rng.integers(0, len(tok) - T - 1, 32)
        LP.append(np.asarray(lp_fn(params, np.stack([tok[i:i + T + 1] for i in ix]))).ravel())
        I.append((ix[:, None] + 1 + np.arange(T)[None, :]).ravel())
    LP, I = np.concatenate(LP), np.concatenate(I)
    lam_loo = fit_table(LP, s_loo[I], g_loo[I]); lam_ldo = fit_table(LP, s_ldo[I], g_ldo[I])
    out = {"size": a.size, "ckpt": a.ckpt, "store_tokens": int(len(tok)), "store_share_of_training_file": round(len(tok) / len(full), 4),
           "documents_in_store_text": int(doc[-1] + 1), "fit_tokens": int(len(LP)),
           "table_leave_position_out": [round(float(x), 3) for x in lam_loo], "table_leave_document_out": [round(float(x), 3) for x in lam_ldo],
           "match_rate_fit": {"leave_position_out": round(float((g_loo[I] > 0).mean()), 4), "leave_document_out": round(float((g_ldo[I] > 0).mean()), 4)},
           "reliability": {"leave_position_out": reliability(s_loo[I], g_loo[I]), "leave_document_out": reliability(s_ldo[I], g_ldo[I])}}
    rows = {}; held = []
    for dom in ("web", "textbook", "math", "code"):
        d = np.fromfile(f"{a.data}/val_{dom}.bin", dtype=np.uint16).astype(np.int32); sv, gv = store.lookup(d, 0); sv2, gv2 = store.lookup_shifted(d)
        r0 = np.random.default_rng(0); acc = {k: [] for k in ("model", "ldo", "loo", "uninformed_real_weights")}; sh, gh = [], []
        for _ in range(20):
            ix = r0.integers(0, len(d) - T - 1, 16)
            lp = np.asarray(lp_fn(params, np.stack([d[i:i + T + 1] for i in ix])))
            ss = np.stack([sv[i + 1:i + T + 1] for i in ix]); gg = np.stack([gv[i + 1:i + T + 1] for i in ix])
            ss2 = np.stack([sv2[i + 1:i + T + 1] for i in ix]); gg2 = np.stack([gv2[i + 1:i + T + 1] for i in ix])
            acc["model"].append(-lp.mean()); acc["ldo"].append(nll(lp, ss, gg, lam_ldo).mean()); acc["loo"].append(nll(lp, ss, gg, lam_loo).mean())
            acc["uninformed_real_weights"].append(nll(lp, ss2, gg2, lam_ldo).mean()); sh.append(ss.ravel()); gh.append(gg.ravel())
        held.append((np.concatenate(sh), np.concatenate(gh)))
        bpt = manifest["domains"][dom]["val"]["bytes_per_token"]
        rows[dom] = {k: round(float(np.mean(v)) / math.log(2) / bpt, 4) for k, v in acc.items()}
    w = manifest["mix"]; rows["mix"] = {k: round(sum(w[d] * rows[d][k] for d in w), 4) for k in rows["web"]}
    out["bits_per_byte"] = rows
    out["gain_leave_document_out"] = {d: round(rows[d]["model"] - rows[d]["ldo"], 4) for d in rows}
    out["gain_leave_position_out"] = {d: round(rows[d]["model"] - rows[d]["loo"], 4) for d in rows}
    out["failing_control_gain"] = {d: round(rows[d]["model"] - rows[d]["uninformed_real_weights"], 4) for d in rows}
    out["failing_control_passed"] = bool(all(out["failing_control_gain"][d] < 0 for d in rows))
    hs, hg = np.concatenate([x[0] for x in held]), np.concatenate([x[1] for x in held]); out["reliability"]["held_out"] = reliability(hs, hg)
    if a.reported:
        rep = json.load(open(a.reported))["curve"][-1]; out["max_abs_difference_from_reported"] = round(max(abs(rep[d] - rows[d]["model"]) for d in rows), 4)
    out["seconds"] = round(time.time() - t0); print(json.dumps(out))
    if a.out: json.dump(out, open(a.out, "w"), indent=1)
