"""Language-model pilot: does routing memorisable content to an exact store free the learned core?

Principle carried over from the routing test (routing_test.py): content a store can supply exactly
should not be pushed through the shared learned parameters. Here the store is an exact n-gram
table over the training text. The model's output is a mixture

    p(y | context) = lam * p_store(y | last 1-3 tokens) + (1 - lam) * p_core(y | context)

lam = sigmoid(table[order matched, log2 bucket of context count]); lam is 0 when nothing matches.
Only the store's probability of the TRUE next token is needed for the loss and for evaluation, so
the cost is three binary searches per position. On training text the store's counts exclude the
position itself (leave-one-out), so the store cannot simply recite the answer.

Prior art, so no novelty is implied: interpolating a neural model with a nearest-neighbour or
n-gram store at inference, and training a neural model on the residual of an n-gram model, are
both published (cited from memory, unchecked). The question here is narrower and ours: at equal
steps on the same text, does training WITH the store beat adding it afterwards, and can a core
with half the blocks plus the store match the full core alone?

Arms (one seed, pilot; all on the same 48M-token training subset, 2,000 steps, batch 32 x 512):
  base8    8 blocks, no store
  post8    the trained base8, store mixed in at evaluation only; lam table fitted afterwards on
           training text with the core frozen
  joint8   8 blocks, trained through the mixture from the start
  base4    4 blocks, no store
  joint4   4 blocks, trained through the mixture
Metric: bits per byte on the four held-out domains and the weighted mix, same evaluation batches
for every arm. Store size in bytes is reported; it is part of the system.

Predictions, written before any run:
  S1 post8 is better than base8 on the mix (the published inference-time effect).
  S2 joint8 is better than post8 by at least 0.01 bits per byte. If not, training with the store
     adds nothing beyond mixing it in afterwards, and the routing principle does not carry to text.
  S3 joint4 is at least as good as base8: half the blocks plus a store match the full core.
  S4 the gain from the store is largest on code and smallest on web.
  One seed: differences under about 0.01 are not evidence.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m21_dense_code"))
from dense import Cfg, LM  # noqa: E402

BITS, ORDERS, BUCKETS = 15, (3, 2, 1), 8          # context lengths tried, longest first


def ctx_key(tok, n):                               # pack the n tokens before each position; positions < n get -1
    k = np.zeros(len(tok), np.int64)
    for j in range(1, n + 1): k[j:] |= tok[:-j].astype(np.int64) << (BITS * (j - 1))
    k[:n] = -1; return k


class Store:
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
        """For every position: store probability of the true token, and gate index (0 = no match)."""
        s = np.zeros(len(tok), np.float32); g = np.zeros(len(tok), np.int64); done = np.zeros(len(tok), bool)
        for oi, n in enumerate(ORDERS):
            pu, pc, cu, cc = self.t[n]; ck = ctx_key(tok, n); ok = ck >= 0
            cx = self._get(cu, cc, ck) - loo; cxy = self._get(pu, pc, (ck << BITS) | tok.astype(np.int64)) - loo
            use = ok & ~done & (cx >= 1)
            s[use] = np.maximum(cxy[use], 0) / cx[use]
            g[use] = 1 + oi * BUCKETS + np.minimum(np.log2(cx[use]).astype(np.int64), BUCKETS - 1)
            done |= use
        return s, g


def mix_nll(logits, y, s, g, table):
    lp = F.log_softmax(logits.float(), -1).gather(-1, y[..., None]).squeeze(-1)
    if table is None: return -lp.mean()
    lam = torch.sigmoid(table[g]) * (g > 0)
    return -torch.log(lam * s + (1 - lam) * lp.exp() + 1e-12).mean()


def get(tok, s, g, n, seq, rng, dev):
    """x = tokens i-1 .. i+seq-2, y = tokens i .. i+seq-1; store values are aligned with y."""
    ix = rng.integers(1, len(tok) - seq - 1, n)
    T = lambda arr, off, dt: torch.from_numpy(np.stack([arr[i + off:i + off + seq] for i in ix]).astype(dt)).to(dev)
    return T(tok, -1, np.int64), T(tok, 0, np.int64), T(s, 0, np.float32), T(g, 0, np.int64)


@torch.no_grad()
def evaluate(model, table, val, manifest, seq, dev, n_batches=20, bs=16):
    out = {}; model.eval()
    for dom, (tok, s, g) in val.items():
        rng = np.random.default_rng(0); nll = 0.0
        for _ in range(n_batches):
            x, y, sb, gb = get(tok, s, g, bs, seq, rng, dev); nll += float(mix_nll(model(x), y, sb, gb, table))
        out[dom] = nll / n_batches / math.log(2) / manifest["domains"][dom]["val"]["bytes_per_token"]
    mix = manifest["mix"]; out["mix"] = sum(mix[k] * out[k] for k in mix); model.train()
    return {k: round(v, 4) for k, v in out.items()}


def train(blocks, use_store, corpus, val, manifest, a):
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); tok, s, g = corpus
    model = LM(Cfg(n_head=4, n_kv=2, seq=a.seq, layout="A" * blocks, loop=None, d=256, ffn=704)).to(a.device)
    table = torch.nn.Parameter(torch.zeros(1 + len(ORDERS) * BUCKETS, device=a.device)) if use_store else None
    opt = torch.optim.AdamW([{"params": model.parameters(), "weight_decay": 0.1}] + ([{"params": [table], "weight_decay": 0.0}] if use_store else []), lr=a.lr, betas=(0.9, 0.95))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
        for grp in opt.param_groups: grp["lr"] = lr
        x, y, sb, gb = get(tok, s, g, a.bs, a.seq, rng, a.device)
        loss = mix_nll(model(x), y, sb, gb, table); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 500 == 0: print(f"blocks {blocks} store {use_store} step {step} loss {float(loss):.4f} sec {time.time() - t0:.0f}", flush=True)
    return model, table, sum(p.numel() for p in model.parameters()), round(time.time() - t0)


def fit_table(model, corpus, a):                      # core frozen; fit only the mixing table on training text
    tok, s, g = corpus; rng = np.random.default_rng(a.seed + 1)
    table = torch.nn.Parameter(torch.zeros(1 + len(ORDERS) * BUCKETS, device=a.device)); opt = torch.optim.Adam([table], lr=0.05); model.eval()
    for _ in range(300):
        x, y, sb, gb = get(tok, s, g, a.bs, a.seq, rng, a.device)
        with torch.no_grad(): lg = model(x)
        loss = mix_nll(lg, y, sb, gb, table); opt.zero_grad(); loss.backward(); opt.step()
    model.train(); return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/Volumes/Express1M2/nanolm-lab/data/v1"); ap.add_argument("--chunks", type=int, default=48)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--device", default="mps"); ap.add_argument("--out", default="")
    ap.add_argument("--arms", default="base8,joint8,base4,joint4")
    a = ap.parse_args()
    root = Path(a.data); manifest = json.load(open(root / "manifest.json"))
    full = np.memmap(root / "train.bin", dtype=np.uint16, mode="r"); span = 1_000_000
    starts = np.linspace(0, len(full) - span, a.chunks).astype(np.int64)
    tok = np.concatenate([np.asarray(full[i:i + span]) for i in starts]).astype(np.int32)
    t0 = time.time(); store = Store(tok); s, g = store.lookup(tok, 1)
    print(json.dumps({"train_tokens": len(tok), "store_bytes": store.bytes, "build_seconds": round(time.time() - t0),
                      "train_match_rate": round(float((g > 0).mean()), 4), "train_mean_store_prob": round(float(s.mean()), 4)}), flush=True)
    val = {}
    for dom in ("web", "textbook", "math", "code"):
        v = np.asarray(np.memmap(root / f"val_{dom}.bin", dtype=np.uint16, mode="r")).astype(np.int32); vs, vg = store.lookup(v, 0); val[dom] = (v, vs, vg)
        print(json.dumps({"val": dom, "match_rate": round(float((vg > 0).mean()), 4), "mean_store_prob": round(float(vs.mean()), 4)}), flush=True)
    rows = []
    for arm in a.arms.split(","):
        blocks, use = int(arm[-1]), arm.startswith("joint")
        model, table, params, sec = train(blocks, use, (tok, s, g), val, manifest, a)
        rows.append({"arm": arm, "params": params, "seconds": sec, **evaluate(model, table, val, manifest, a.seq, a.device)}); print(json.dumps(rows[-1]), flush=True)
        if not use:
            t2 = fit_table(model, (tok, s, g), a)
            rows.append({"arm": f"post{blocks}", "params": params, "seconds": sec, **evaluate(model, t2, val, manifest, a.seq, a.device)}); print(json.dumps(rows[-1]), flush=True)
        del model
    if a.out: json.dump({"store_bytes": store.bytes, "train_tokens": len(tok), "rows": rows}, open(a.out, "w"))
