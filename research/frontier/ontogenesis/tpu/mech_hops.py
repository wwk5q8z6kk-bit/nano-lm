"""Cycle 2, exploratory: does adaptive recurrent depth buy multi-step composition per unit of compute, and do extra
iterations at test time solve more steps than were trained? (Candidate 3, RESEARCH-PLAN.md section 8.)

Task: pointer chasing. A sequence lists the C links of a random chain over C+1 distinct symbols (a->b as two tokens
"a b"), in shuffled order, then QN queries "S x k a": start at x and follow k links; a is the symbol reached. Every
answer position is supervised (version 2; version 1 supervised one answer per sequence and nothing learned).
Each hop needs one sequential lookup, so a fixed-depth model should fail once k exceeds what its depth can chain.
Train on k in 1..4; evaluate k in 1..8.

Arms (same width, optimiser, steps, data stream, seeds):
  untied-6   prelude 1 attention layer + 6 distinct attention layers           (7 layer applications)
  looped-6   prelude 1 attention layer + 1 shared attention block applied 6 times (same applications, about 1/4 the
             block parameters)
  looped-rnd same, but trained with a random iteration count in 2..8 per batch (as Huginn); evaluated at R = 6, 12
             and with residual halting (eps in {0.02, 0.05}) capped at R = 12
Compute is reported as core-block applications per token.

usage: mech_hops.py --out results/mech/hops.json [--steps 4000] [--seeds 0,1,2]
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, jax, jax.numpy as jnp, optax
import nano_core as nc

SYM, C = 40, 12                       # symbols, links per chain
S_TOK, K0 = SYM, SYM + 1              # query marker; hop-count tokens K0..K0+8
VOCAB = SYM + 1 + 9
QN = 6                                # queries per sequence (version 2)
T = 2 * C + 4 * QN                    # links, then QN x "S x k ans"


ANS = jnp.asarray([2 * C + 4 * q + 3 for q in range(QN)])   # answer positions


def batch(rng, B, kmin, kmax):
    tok = np.zeros((B, T), np.int32); ks = np.zeros(B, np.int32)
    for b in range(B):
        nodes = rng.choice(SYM, C + 1, replace=False)
        links = np.stack([nodes[:-1], nodes[1:]], 1)[rng.permutation(C)]
        tok[b, :2 * C] = links.reshape(-1)
        for q in range(QN):
            k = rng.integers(kmin, kmax + 1); start = rng.integers(0, C + 1 - k)
            tok[b, 2 * C + 4 * q: 2 * C + 4 * q + 4] = [S_TOK, nodes[start], K0 + k, nodes[start + k]]
        ks[b] = k
    return jnp.asarray(tok), ks


def cfg_of(arm):
    if arm == "untied-6":
        return dict(vocab=VOCAB, d=64, H=4, ffn=128, layers=["attn"] * 7)
    return dict(vocab=VOCAB, d=64, H=4, ffn=128, pre=["attn"], core="attn", coda=[])


def logits_of(arm, p, tok, cfg, R, eps=None):
    if arm == "untied-6":
        lg, _ = nc.forward(p, tok, cfg); return lg, jnp.asarray(6.0)
    return nc.forward_looped(p, tok, cfg, R, eps)


def run(arm, seed, steps):
    cfg = cfg_of(arm)
    p = nc.init_model(jax.random.PRNGKey(seed), cfg) if arm == "untied-6" else nc.init_looped(jax.random.PRNGKey(seed), cfg)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 2e-3, max(1, min(200, steps // 10)), steps, 2e-4), weight_decay=0.01)
    st = opt.init(p)
    steps_fn = {}

    def get_step(R):
        if R not in steps_fn:
            def loss_fn(p, tok):
                lg, _ = logits_of(arm, p, tok[:, :-1], cfg, R)
                ce = optax.softmax_cross_entropy_with_integer_labels(lg[:, ANS - 1], tok[:, ANS])
                return ce.mean()
            @jax.jit
            def step(p, st, tok):
                l, g = jax.value_and_grad(loss_fn)(p, tok); u, st = opt.update(g, st, p)
                return optax.apply_updates(p, u), st, l
            steps_fn[R] = step
        return steps_fn[R]

    rng = np.random.default_rng(2000 + seed); t0 = time.time(); curve = []
    for i in range(steps):
        tok, _ = batch(rng, 64, 1, 4)
        R = int(rng.integers(2, 9)) if arm == "looped-rnd" else 6
        p, st, l = get_step(R)(p, st, tok)
        if (i + 1) % 500 == 0: curve.append([i + 1, float(l)])
    ev = np.random.default_rng(4242); evals = {}
    settings = [("R6", 6, None)] if arm != "looped-rnd" else [("R6", 6, None), ("R12", 12, None), ("halt0.02", 12, 0.02), ("halt0.05", 12, 0.05)]
    toks = [batch(ev, 256, k, k)[0] for k in range(1, 9)]
    for name, R, eps in settings:
        f = jax.jit(lambda p, tok, R=R, eps=eps: logits_of(arm, p, tok[:, :-1], cfg, R, eps))
        accs, its = [], []
        for tok in toks:
            lg, it = f(p, tok); accs.append(float((lg[:, ANS - 1].argmax(-1) == tok[:, ANS]).mean())); its.append(float(it))
        evals[name] = {"acc_by_k": [round(a, 4) for a in accs], "mean_iters_by_k": [round(i, 2) for i in its]}
    return {"arm": arm, "seed": seed, "steps": steps, "evals": evals, "curve": curve, "seconds": round(time.time() - t0),
            "params": int(sum(x.size for x in jax.tree_util.tree_leaves(p)))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", default="0,1,2"); a = ap.parse_args(); rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in ("untied-6", "looped-6", "looped-rnd"):
            r = run(arm, seed, a.steps); rows.append(r); print(json.dumps(r), flush=True); json.dump(rows, open(a.out, "w"), indent=1)
