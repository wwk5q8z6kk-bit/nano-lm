"""Cycle 1, exploratory: can a budgeted exact memory recover the recall a recurrent model loses, and does a write
policy trained on a teacher's attention beat recency? (Candidate 2, RESEARCH-PLAN.md section 8.)

Task: multi-query associative recall (MQAR; Arora et al. 2023, "Zoology"). A sequence holds N key-value pairs
(k1 v1 k2 v2 ...), then Q queries; each query is a key from the pairs, and the model must output its value at the
next position. Keys and values are drawn without replacement per sequence. Loss and accuracy only at answer positions.

Arms (same width, depth, optimiser, steps, data stream, seeds):
  attn            [attn, attn]                    full causal attention: the recall reference and the teacher
  gdn             [gdn, gdn]                      Gated DeltaNet only: fixed state, no exact memory
  mem-recency     [gdn, mem], keep the M most recent tokens
  mem-learned     [gdn, mem], keep the M tokens with the highest learned write score; the write head is trained only
                  to match the teacher's (attn arm, same seed) later-attention mass on each token
  mem-oracle      [gdn, mem], keep the value tokens of the queried pairs (needs knowledge of future queries: an upper
                  bound, not a usable policy)
Budgets M in {8, 16, 32} with N = 16 pairs (32 pair tokens), Q = 8 queries.

usage: mech_mqar.py --out results/mech/mqar.json [--steps 3000] [--seeds 0,1,2]
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, jax, jax.numpy as jnp, optax
import nano_core as nc

N, Q, KEYS, VALS = 16, 8, 64, 64          # pilot: N=32 reached only 40% for attention in 3,000 steps
VOCAB = KEYS + VALS
T = 2 * N + 2 * Q


def batch(rng, B):
    tok = np.zeros((B, T), np.int32); ans = np.zeros((B, T), bool); orc = np.zeros((B, T), bool)
    for b in range(B):
        ks = rng.choice(KEYS, N, replace=False); vs = KEYS + rng.choice(VALS, N, replace=False)
        tok[b, 0:2 * N:2], tok[b, 1:2 * N:2] = ks, vs
        qi = rng.choice(N, Q, replace=False)
        tok[b, 2 * N::2], tok[b, 2 * N + 1::2] = ks[qi], vs[qi]
        ans[b, 2 * N + 1::2] = True                                   # predict the value token at these positions
        orc[b, 2 * qi + 1] = True                                     # the queried pairs' value tokens
    return jnp.asarray(tok), jnp.asarray(ans), jnp.asarray(orc)


def make_cfg(arm):
    layers = {"attn": ["attn", "attn"], "gdn": ["gdn", "gdn"]}.get(arm, ["gdn", "mem"])
    return dict(vocab=VOCAB, d=64, H=4, ffn=128, layers=layers)


def policy_of(arm):
    return {"mem-recency": "recency", "mem-learned": "learned", "mem-oracle": "oracle"}.get(arm)


def teacher_target(tparams, tcfg, tok):
    """Later-attention mass each position receives in the teacher's last attention layer, averaged over heads,
    normalised per sequence: the target distribution for the learned write head."""
    _, aux = nc.forward(tparams, tok, tcfg)
    w = aux["attn"][-1].mean(1)                                       # B t s
    mass = w.sum(1)                                                   # B s: attention received from later positions
    return mass / mass.sum(-1, keepdims=True)


def run(arm, seed, M, steps, teacher=None):
    cfg = make_cfg(arm); pol = policy_of(arm)
    params = nc.init_model(jax.random.PRNGKey(seed), cfg)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, 3e-3, max(1, min(200, steps // 10)), steps, 3e-4), weight_decay=0.01)
    st = opt.init(params)

    def loss_fn(p, tok, ans, orc, tgt):
        logits, aux = nc.forward(p, tok[:, :-1], cfg, pol, M, orc[:, :-1])
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, tok[:, 1:])
        m = ans[:, 1:]
        loss = (ce * m).sum() / m.sum()
        if pol == "learned":                                          # write head learns only from the teacher signal
            lw = jax.nn.log_softmax(aux["write"], -1)
            loss = loss + 1.0 * -(tgt[:, :-1] * lw).sum(-1).mean()
        acc = ((logits.argmax(-1) == tok[:, 1:]) * m).sum() / m.sum()
        return loss, acc

    @jax.jit
    def step(p, st, tok, ans, orc, tgt):
        (l, a), g = jax.value_and_grad(loss_fn, has_aux=True)(p, tok, ans, orc, tgt)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l, a

    rng = np.random.default_rng(1000 + seed)                          # same data stream for every arm with this seed
    tgt_fn = jax.jit(lambda tok: teacher_target(teacher[0], teacher[1], tok)) if pol == "learned" else None
    t0 = time.time(); curve = []
    for i in range(steps):
        tok, ans, orc = batch(rng, 64)
        tgt = tgt_fn(tok) if tgt_fn else jnp.zeros(tok.shape)
        params, st, l, a = step(params, st, tok, ans, orc, tgt)
        if (i + 1) % 500 == 0: curve.append([i + 1, float(l), float(a)])
    ev = np.random.default_rng(999); accs = []
    evalf = jax.jit(lambda p, tok, ans, orc, tgt: loss_fn(p, tok, ans, orc, tgt)[1])
    for _ in range(16):
        tok, ans, orc = batch(ev, 128)
        tgt = tgt_fn(tok) if tgt_fn else jnp.zeros(tok.shape)
        accs.append(float(evalf(params, tok, ans, orc, tgt)))
    out = {"arm": arm, "seed": seed, "M": M if pol else None, "steps": steps, "eval_acc": round(float(np.mean(accs)), 4),
           "eval_n": 16 * 128 * Q, "curve": curve, "seconds": round(time.time() - t0),
           "params": int(sum(x.size for x in jax.tree_util.tree_leaves(params)))}
    if pol == "learned":                                              # what the policy learned to keep
        tok, ans, orc = batch(np.random.default_rng(7), 64)
        _, aux = nc.forward(params, tok, cfg, pol, M, orc)
        sc = np.asarray(aux["write"]); top = np.argsort(-sc[:, :2 * N], 1)[:, :M]
        out["kept_value_token_share"] = round(float((top % 2 == 1).mean()), 4)
    return out, params


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--Ms", default="8,16,32")
    a = ap.parse_args(); rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        r, tparams = run("attn", seed, None, a.steps); rows.append(r); print(json.dumps(r), flush=True)
        r, _ = run("gdn", seed, None, a.steps); rows.append(r); print(json.dumps(r), flush=True)
        for M in [int(m) for m in a.Ms.split(",")]:
            for arm in ("mem-recency", "mem-learned", "mem-oracle"):
                r, _ = run(arm, seed, M, a.steps, teacher=(tparams, make_cfg("attn"))); rows.append(r); print(json.dumps(r), flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)
    json.dump(rows, open(a.out, "w"), indent=1)
