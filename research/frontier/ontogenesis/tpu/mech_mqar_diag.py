"""Cycle 1c, exploratory: why did the oracle memory do worse with a larger budget (cycle 1: M=8 3/3, M=16 2/3,
M=32 0/3 seeds learned)? Leading hypotheses and the arm that separates them:
  H-distract  the filler slots (most recent tokens = earlier queries, which repeat keys) act as decoys
  H-slots     more readable slots make the read layer harder to train, whatever they hold
  H-luck      optimisation is bimodal and 3 seeds cannot tell
Arms (task, model, optimiser and data stream as mech_mqar.py; all write scores supplied, no learning in the write):
  oracle-recency-M    oracle value tokens, remaining slots filled with the most recent tokens   (M = 8, 16, 32)
  oracle-random-32    oracle value tokens, remaining slots filled with random earlier tokens     (M = 32)
  oracle-only         oracle value tokens only, no filler (8 readable tokens)
Prediction if H-distract: oracle-only and oracle-random-32 learn far more often than oracle-recency-32.
If H-slots: oracle-random-32 fails like oracle-recency-32 while oracle-only succeeds. If H-luck: no arm differs beyond
the success-rate uncertainty. Every seed is reported, with the first training step whose batch accuracy passes 0.9.

usage: mech_mqar_diag.py --out f.json [--steps 4000] [--seeds 0..7]
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, jax, jax.numpy as jnp, optax
import nano_core as nc
import mech_mqar as mq


def scores(arm, orc, rng):
    B, T = orc.shape
    o = np.asarray(orc, np.float32)
    if arm.startswith("oracle-recency"): return jnp.asarray(o + 1e-3 * np.arange(T)[None])
    if arm.startswith("oracle-random"): return jnp.asarray(o + 1e-3 * rng.uniform(size=(B, T)))
    return jnp.asarray(o)                                             # oracle-only


def run(arm, seed, steps):
    M = int(arm.rsplit("-", 1)[1]) if arm[-1].isdigit() else 1000
    pol = "given-only" if arm == "oracle-only" else "given"
    cfg = mq.make_cfg("mem"); params = nc.init_model(jax.random.PRNGKey(seed), cfg)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-3, max(1, min(200, steps // 10)), steps, 3e-4), weight_decay=0.01)
    st = opt.init(params)

    def loss_fn(p, tok, ans, sc):
        logits, _ = nc.forward(p, tok[:, :-1], cfg, pol, M, sc[:, :-1])
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, tok[:, 1:]); m = ans[:, 1:]
        return (ce * m).sum() / m.sum(), ((logits.argmax(-1) == tok[:, 1:]) * m).sum() / m.sum()

    @jax.jit
    def step(p, st, tok, ans, sc):
        (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(p, tok, ans, sc); u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l, a

    rng = np.random.default_rng(1000 + seed); srng = np.random.default_rng(5000 + seed)
    first90, curve, t0 = None, [], time.time()
    for i in range(steps):
        tok, ans, orc = mq.batch(rng, 64)
        params, st, l, a = step(params, st, tok, ans, scores(arm, orc, srng))
        if first90 is None and float(a) > 0.9: first90 = i + 1
        if (i + 1) % 500 == 0: curve.append([i + 1, round(float(l), 4), round(float(a), 4)])
    ev, er = np.random.default_rng(999), np.random.default_rng(998)
    evalf = jax.jit(lambda p, tok, ans, sc: loss_fn(p, tok, ans, sc)[1]); accs = []
    for _ in range(16):
        tok, ans, orc = mq.batch(ev, 128); accs.append(float(evalf(params, tok, ans, scores(arm, orc, er))))
    acc = float(np.mean(accs))
    return {"arm": arm, "seed": seed, "M": M, "steps": steps, "eval_acc": round(acc, 4), "learned": acc > 0.9,
            "first_step_batch_acc_over_0.9": first90, "curve": curve, "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7"); a = ap.parse_args(); rows = []
    arms = ["oracle-recency-8", "oracle-recency-16", "oracle-recency-32", "oracle-random-32", "oracle-only"]
    for seed in [int(s) for s in a.seeds.split(",")]:
        for arm in arms:
            r = run(arm, seed, a.steps); rows.append(r); print(json.dumps(r), flush=True); json.dump(rows, open(a.out, "w"), indent=1)
