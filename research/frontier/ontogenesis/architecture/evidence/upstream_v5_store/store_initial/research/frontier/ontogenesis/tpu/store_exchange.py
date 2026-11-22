"""Does an exact n-gram store still help a larger, fully trained model?

Registered in theory/RESEARCH-DECISION-20261121.md, section 6, before this file was run.

For one finished checkpoint: compute the model's probability of the true next token on the held-out
evaluation crops (the same crops train_jax.py scores) and on about 1M tokens of training text; fit the
25-entry mixing table on the training text only (store counts leave out the current position, model
frozen); report bits per byte with and without the store. The store is the pilot's: 4-gram counts over
the same 48 evenly spaced 1M-token chunks of the training file.

Controls: mixing weight forced to zero must reproduce the checkpoint's own reported bits per byte;
a store whose values are shuffled across positions must give no gain.
"""
from __future__ import annotations
import argparse, json, math, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize_scalar
from train_jax import SIZES, forward

BITS, ORDERS, BUCKETS = 15, (3, 2, 1), 8


def ctx_key(tok, n):
    k = np.zeros(len(tok), np.int64)
    for j in range(1, n + 1): k[j:] |= tok[:-j].astype(np.int64) << (BITS * (j - 1))
    k[:n] = -1; return k


class Store:                                     # identical to lm_store.py
    def __init__(self, tok):
        self.t = {}
        for n in ORDERS:
            ck = ctx_key(tok, n); ok = ck >= 0
            pu, pc = np.unique((ck[ok] << BITS) | tok[ok].astype(np.int64), return_counts=True)
            cu, cc = np.unique(ck[ok], return_counts=True)
            self.t[n] = (pu, pc.astype(np.int32), cu, cc.astype(np.int32))
        self.bytes = sum(a.nbytes for v in self.t.values() for a in v)

    @staticmethod
    def _get(keys, counts, q):
        i = np.searchsorted(keys, q); i[i >= len(keys)] = len(keys) - 1
        return np.where(keys[i] == q, counts[i], 0)

    def lookup(self, tok, loo):
        s = np.zeros(len(tok), np.float32); g = np.zeros(len(tok), np.int64); done = np.zeros(len(tok), bool)
        for oi, n in enumerate(ORDERS):
            pu, pc, cu, cc = self.t[n]; ck = ctx_key(tok, n); ok = ck >= 0
            cx = self._get(cu, cc, ck) - loo; cxy = self._get(pu, pc, (ck << BITS) | tok.astype(np.int64)) - loo
            use = ok & ~done & (cx >= 1)
            s[use] = np.maximum(cxy[use], 0) / cx[use]
            g[use] = 1 + oi * BUCKETS + np.minimum(np.log2(cx[use]).astype(np.int64), BUCKETS - 1)
            done |= use
        return s, g


def load_params(path):
    z = np.load(path); n = 1 + max(int(k[1:].split(".")[0]) for k in z.files if k.startswith("b"))
    return dict(emb=jnp.asarray(z["emb"]), nf=jnp.asarray(z["nf"]),
                blocks=[{k.split(".", 1)[1]: jnp.asarray(z[k]) for k in z.files if k.startswith(f"b{i}.")} for i in range(n)])


def fit_table(lp, s, g):
    """One mixing weight per gate value, each maximising the training log-likelihood on its own positions."""
    lam = np.zeros(1 + len(ORDERS) * BUCKETS); p = np.exp(lp)
    for k in range(1, len(lam)):
        m = g == k
        if m.sum() < 50: continue
        f = lambda x: -np.mean(np.log(x * s[m] + (1 - x) * p[m] + 1e-12))
        lam[k] = minimize_scalar(f, bounds=(0.0, 1.0), method="bounded").x
    return lam


def nll(lp, s, g, lam): return -np.log(lam[g] * s + (1 - lam[g]) * np.exp(lp) + 1e-12)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--reported", default="")
    ap.add_argument("--data", default="/Volumes/Express1M2/nanolm-lab/data/v1"); ap.add_argument("--out", default="")
    ap.add_argument("--seq", type=int, default=512); ap.add_argument("--fit-batches", type=int, default=64); ap.add_argument("--f32", action="store_true")
    a = ap.parse_args(); c = SIZES[a.size]; dt = jnp.float32 if a.f32 else jnp.bfloat16; T = a.seq
    manifest = json.load(open(f"{a.data}/manifest.json")); t0 = time.time()
    full = np.memmap(f"{a.data}/train.bin", dtype=np.uint16, mode="r"); span = 1_000_000
    tok = np.concatenate([np.asarray(full[i:i + span]) for i in np.linspace(0, len(full) - span, 48).astype(np.int64)]).astype(np.int32)
    store = Store(tok); s_tr, g_tr = store.lookup(tok, 1)
    params = load_params(a.ckpt)
    lp_fn = jax.jit(lambda p, x: jnp.take_along_axis(jax.nn.log_softmax(forward(p, x[:, :-1], c, dt), -1), x[:, 1:, None], -1)[..., 0])
    rng = np.random.default_rng(12345); LP, S, G = [], [], []                       # training text for fitting the table
    for _ in range(a.fit_batches):
        ix = rng.integers(0, len(tok) - T - 1, 32)
        LP.append(np.asarray(lp_fn(params, np.stack([tok[i:i + T + 1] for i in ix]))).ravel())
        S.append(np.stack([s_tr[i + 1:i + T + 1] for i in ix]).ravel()); G.append(np.stack([g_tr[i + 1:i + T + 1] for i in ix]).ravel())
    LP, S, G = np.concatenate(LP), np.concatenate(S), np.concatenate(G)
    lam = fit_table(LP, S, G)
    perm = np.random.default_rng(7).permutation(len(S)); lam_shuf = fit_table(LP, S[perm], G[perm])
    out = {"size": a.size, "ckpt": a.ckpt, "store_bytes": int(store.bytes), "fit_tokens": int(len(LP)), "table": [round(float(x), 3) for x in lam]}
    rows = {}
    for dom in ("web", "textbook", "math", "code"):
        d = np.fromfile(f"{a.data}/val_{dom}.bin", dtype=np.uint16).astype(np.int32); sv, gv = store.lookup(d, 0)
        r0 = np.random.default_rng(0); lp, ss, gg = [], [], []
        for _ in range(20):
            ix = r0.integers(0, len(d) - T - 1, 16)
            lp.append(np.asarray(lp_fn(params, np.stack([d[i:i + T + 1] for i in ix]))))          # [16, T], kept per batch
            ss.append(np.stack([sv[i + 1:i + T + 1] for i in ix])); gg.append(np.stack([gv[i + 1:i + T + 1] for i in ix]))
        bpt = manifest["domains"][dom]["val"]["bytes_per_token"]; bits = lambda x: float(x) / math.log(2) / bpt
        base = np.mean([-(b.mean()) for b in lp])                                                  # mean of per-batch means, as train_jax does
        mix = np.mean([nll(b, x, y, lam).mean() for b, x, y in zip(lp, ss, gg)])
        sp = np.random.default_rng(8).permutation(lp[0].size)
        shuf = np.mean([nll(b, x.ravel()[sp].reshape(x.shape), y.ravel()[sp].reshape(y.shape), lam_shuf).mean() for b, x, y in zip(lp, ss, gg)])
        rows[dom] = {"model": round(bits(base), 4), "with_store": round(bits(mix), 4), "shuffled_store": round(bits(shuf), 4)}
    w = manifest["mix"]; rows["mix"] = {k: round(sum(w[d] * rows[d][k] for d in w), 4) for k in ("model", "with_store", "shuffled_store")}
    out["bits_per_byte"] = rows; out["gain"] = {d: round(rows[d]["model"] - rows[d]["with_store"], 4) for d in rows}
    out["shuffled_gain_mix"] = round(rows["mix"]["model"] - rows["mix"]["shuffled_store"], 4)
    if a.reported:
        rep = json.load(open(a.reported))["curve"][-1]; out["reported_by_training_run"] = {d: rep[d] for d in rows}
        out["max_abs_difference_from_reported"] = round(max(abs(rep[d] - rows[d]["model"]) for d in rows), 4)
    out["seconds"] = round(time.time() - t0)
    print(json.dumps(out))
    if a.out: json.dump(out, open(a.out, "w"))
