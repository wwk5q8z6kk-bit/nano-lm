"""How much could exact copying from the current context add to a trained dense model?

This is the cheap, non-neural control that must come before any trained copy head (the blueprint's
"simple deterministic substring lookup must be in the test matrix"). It needs no training.

For one finished checkpoint, on the held-out evaluation crops the training runs score:
  the cache, at each position, looks back inside the same 512-token window for the longest earlier
  match of the last 3, 2 or 1 tokens and proposes the tokens that followed those matches, in proportion
  to how often each followed. It sees only tokens before the position: it is causal by construction
  and uses no outside data. The model's probability is mixed with the cache's using one weight per
  (match length, number of matches) cell, fitted on training-text windows with the model frozen.

Registered before any run of this file:
  Background: the 5M-parameter copy model (M21) kept a 0.36 bits-per-token advantage on copyable tokens
  of code; an exact n-gram store over the corpus added only 0.004 bits per byte to the trained 14M model.
  Expectation: the cache adds 0.01 to 0.03 bits per byte on code for the 14M model, less for the 40M
  model, and under 0.005 on web text.
  Decision rule, read on the 40M checkpoint:
    gain on code of 0.02 or more  -> train the copy-head arms on TPU (learned gate, constant gate,
                                     parameter-matched non-copy head) at 14M and 40M;
    under 0.01 on every domain    -> close the explicit-copy line at this scale: the dense model
                                     already copies what an exact in-context cache can supply;
    in between                    -> one trained arm at 14M only, to see whether a learned pointer
                                     finds more than the exact cache.
  Controls: weight forced to zero must reproduce the checkpoint's reported bits per byte; a cache built
  from a different window (the next crop in the batch) must give no gain.
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize_scalar
from train_jax import SIZES, forward
from store_exchange import load_params

ORDERS, BUCKETS = (3, 2, 1), 8


def cache_probs(ctx, tgt_tokens):
    """ctx: the window that feeds the cache [T+1]; tgt_tokens: the window whose next tokens are scored [T+1].
    Normally the same window. Returns, for each position t in 0..T-1, the cache's probability of tgt[t+1]
    given the last tokens of tgt[..t] matched against earlier positions of ctx, and the cell index (0 = no match)."""
    T = len(tgt_tokens) - 1; s = np.zeros(T, np.float32); g = np.zeros(T, np.int64)
    seen = [dict() for _ in ORDERS]                                   # key -> [total, {follower: count}]
    own = ctx is tgt_tokens
    if not own:                                                       # foreign cache: fill it completely first, never update
        for t in range(len(ctx) - 1):
            for oi, n in enumerate(ORDERS):
                if t >= n - 1:
                    e = seen[oi].setdefault(tuple(ctx[t - n + 1:t + 1]), [0, {}]); e[0] += 1; e[1][ctx[t + 1]] = e[1].get(ctx[t + 1], 0) + 1
    for t in range(T):
        y = tgt_tokens[t + 1]
        for oi, n in enumerate(ORDERS):
            if t >= n - 1:
                e = seen[oi].get(tuple(tgt_tokens[t - n + 1:t + 1]))
                if e is not None:
                    s[t] = e[1].get(y, 0) / e[0]; g[t] = 1 + oi * BUCKETS + min(int(math.log2(e[0])), BUCKETS - 1); break
        if own:
            for oi, n in enumerate(ORDERS):
                if t >= n - 1:
                    e = seen[oi].setdefault(tuple(tgt_tokens[t - n + 1:t + 1]), [0, {}]); e[0] += 1; e[1][y] = e[1].get(y, 0) + 1
    return s, g


def fit_table(lp, s, g):
    lam = np.zeros(1 + len(ORDERS) * BUCKETS); p = np.exp(lp)
    for k in range(1, len(lam)):
        m = g == k
        if m.sum() < 50: continue
        lam[k] = minimize_scalar(lambda x: -np.mean(np.log(x * s[m] + (1 - x) * p[m] + 1e-12)), bounds=(0.0, 1.0), method="bounded").x
    return lam


def nll(lp, s, g, lam): return -np.log(lam[g] * s + (1 - lam[g]) * np.exp(lp) + 1e-12)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--reported", default="")
    ap.add_argument("--data", default="/Volumes/Express1M2/nanolm-lab/data/v1"); ap.add_argument("--out", default="")
    ap.add_argument("--seq", type=int, default=512); ap.add_argument("--fit-batches", type=int, default=32)
    a = ap.parse_args(); c = SIZES[a.size]; T = a.seq; t0 = time.time()
    manifest = json.load(open(f"{a.data}/manifest.json")); params = load_params(a.ckpt)
    lp_fn = jax.jit(lambda p, x: jnp.take_along_axis(jax.nn.log_softmax(forward(p, x[:, :-1], c, jnp.bfloat16), -1), x[:, 1:, None], -1)[..., 0])
    full = np.memmap(f"{a.data}/train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(54321); LP, S, G, S2, G2 = [], [], [], [], []       # training windows, for fitting the weights only
    for _ in range(a.fit_batches):
        ix = rng.integers(0, len(full) - T - 1, 32); X = np.stack([np.asarray(full[i:i + T + 1]) for i in ix]).astype(np.int32)
        LP.append(np.asarray(lp_fn(params, X)).ravel())
        for b in range(len(X)):
            row = X[b].tolist(); s, g = cache_probs(row, row); S.append(s); G.append(g)
            s, g = cache_probs(X[(b + 1) % len(X)].tolist(), row); S2.append(s); G2.append(g)
    LP = np.concatenate(LP); lam = fit_table(LP, np.concatenate(S), np.concatenate(G)); lam_f = fit_table(LP, np.concatenate(S2), np.concatenate(G2))
    out = {"size": a.size, "ckpt": a.ckpt, "fit_tokens": int(len(LP)), "table": [round(float(x), 3) for x in lam]}; rows = {}; share = {}
    for dom in ("web", "textbook", "math", "code"):
        d = np.fromfile(f"{a.data}/val_{dom}.bin", dtype=np.uint16).astype(np.int32); r0 = np.random.default_rng(0); base, mix, foreign, hit = [], [], [], []
        for _ in range(20):
            ix = r0.integers(0, len(d) - T - 1, 16); X = np.stack([d[i:i + T + 1] for i in ix]); lp = np.asarray(lp_fn(params, X))
            sg = [cache_probs(X[b].tolist(), X[b].tolist()) for b in range(16)]; sf = [cache_probs(X[(b + 1) % 16].tolist(), X[b].tolist()) for b in range(16)]
            s = np.stack([x[0] for x in sg]); g = np.stack([x[1] for x in sg]); s2 = np.stack([x[0] for x in sf]); g2 = np.stack([x[1] for x in sf])
            base.append(-lp.mean()); mix.append(nll(lp, s, g, lam).mean()); foreign.append(nll(lp, s2, g2, lam_f).mean()); hit.append(float((s > 0).mean()))
        bpt = manifest["domains"][dom]["val"]["bytes_per_token"]; bits = lambda x: float(np.mean(x)) / math.log(2) / bpt
        rows[dom] = {"model": round(bits(base), 4), "with_cache": round(bits(mix), 4), "foreign_cache": round(bits(foreign), 4)}; share[dom] = round(float(np.mean(hit)), 3)
    w = manifest["mix"]; rows["mix"] = {k: round(sum(w[d] * rows[d][k] for d in w), 4) for k in ("model", "with_cache", "foreign_cache")}
    out["bits_per_byte"] = rows; out["gain"] = {d: round(rows[d]["model"] - rows[d]["with_cache"], 4) for d in rows}
    out["foreign_cache_gain"] = {d: round(rows[d]["model"] - rows[d]["foreign_cache"], 4) for d in rows}
    out["share_of_tokens_the_cache_could_copy"] = share
    if a.reported:
        rep = json.load(open(a.reported))["curve"][-1]; out["max_abs_difference_from_reported"] = round(max(abs(rep[d] - rows[d]["model"]) for d in rows), 4)
    out["seconds"] = round(time.time() - t0); print(json.dumps(out))
    if a.out: json.dump(out, open(a.out, "w"))
