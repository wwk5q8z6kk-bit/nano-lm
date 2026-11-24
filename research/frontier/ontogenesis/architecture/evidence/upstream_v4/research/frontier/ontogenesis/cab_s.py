"""Structured acquisition benchmark (CAB-S) and a learner that writes by adding statistics.

Why this file exists
--------------------
1. The first acquisition benchmark (cab.py) used arbitrary facts. A plain lookup table scores
   100% on acquisition, retention, correction and (by iterating) composition there. So it cannot
   tell cumulative learning from storage. Cumulative learning needs a third property: what was
   learned earlier must make new things easier (forward transfer). That needs a world with
   structure to discover.

2. Candidate principle (derived, not claimed new): a memory can relearn its addresses without
   overwriting what it holds if
     (a) every item keeps its own address (a vector per entity), so changing one address
         touches one item, and
     (b) everything shared is a SUM of per-item statistics, so an item's contribution can be
         subtracted and added back exactly when its address changes, and
     (c) the raw facts are kept in an exact store, which is authoritative for anything seen.
   Gradient steps are neither additive nor reversible; sums of statistics are both.
   Here: fact (e, r) -> value y (one-hot). Model y ~ W_r u_e. Shared statistics per relation:
     A_r = sum_e u_e u_e^T,   B_r = sum_e u_e y_{e,r}^T,   W_r = B_r^T (A_r + lam I)^-1.
   A new entity's address is solved in closed form from its few facts given the current W.

World
-----
Z latent types, R relations, V values. rule[z, r] is a value. Each entity has a hidden type.
truth[e, r] = rule[type(e), r], except a fraction EPS of entries are arbitrary (exceptions).
Each phase NEW entities arrive; K of their R relations are shown. FIX earlier shown facts are
changed to a new arbitrary value. Nothing is shown again unless the learner keeps it.

Measured after every phase, no further fitting
  retain_gen  accuracy on all earlier shown facts using the generalising model alone
  retain      same, with the learner's exact store if it has one
  correct     accuracy on the facts changed this phase (with store if any)
  infer_new   accuracy on the NOT-shown relations of this phase's entities (forward transfer)
  infer_old   accuracy on the not-shown relations of earlier entities
Chance is 1/V. Ceiling for infer is about 1 - EPS.

Learners
  table        exact store; for unseen facts answers the most common value seen for that relation
  grad         same bilinear model, trained by Adam on the current phase only
  grad_replay  same, each step trained on a batch drawn from ALL facts seen so far
  additive     the statistics learner above, with exact store

Predictions written before the confirmation run (seeds 4001-4003; seed 4000 used for tuning
the learning rate of the gradient learners and lam of the additive learner):
  P1 table: retain 1.0, infer_new below 0.15.
  P2 grad: infer_new rises above 0.5, but retain_gen ends below 0.5 and infer_old ends at least
     0.2 below infer_new (the shared decoder drifts away from old addresses).
  P3 additive: retain 1.0 exactly, infer_new at least 0.75 late, infer_old within 0.1 of infer_new.
  P4 grad_replay comes close to additive on infer_new (within 0.1). If so, the additive write
     buys exactness and cost, not accuracy, and that must be said plainly.
  The idea fails if additive infer_new ends below 0.5, or more than 0.1 below grad_replay.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch

Z, R, V, EPS, K = 16, 8, 32, 0.10, 3


def world(seed, phases, new, fix):
    rng = np.random.default_rng(seed)
    n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n)
    truth = rule[typ]                                                     # [n, R]
    exc = rng.random((n, R)) < EPS
    truth = np.where(exc, rng.integers(V, size=(n, R)), truth)
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    return rng, truth.copy(), shown


class Table:
    def __init__(s, n, m, **kw): s.d, s.cnt = {}, np.zeros((R, V))
    def learn(s, facts, ph):
        for e, r, y in facts:
            s.d[(e, r)] = y; s.cnt[r, y] += 1
    def predict(s, e, r, store=True):
        out = s.cnt[r].argmax(-1) if np.ndim(r) else s.cnt[r].argmax()
        out = np.array(out, copy=True)
        if store:
            for i, (a, b) in enumerate(zip(e, r)):
                if (a, b) in s.d: out[i] = s.d[(a, b)]
        return out


class Grad:
    def __init__(s, n, m, lr=1e-2, steps=200, replay=False, seed=0, **kw):
        torch.manual_seed(seed)
        s.u = torch.nn.Parameter(torch.randn(n, m) * 0.1); s.W = torch.nn.Parameter(torch.randn(R, V, m) * 0.1)
        s.opt = torch.optim.Adam([s.u, s.W], lr=lr); s.steps, s.replay, s.seen, s.rng = steps, replay, {}, np.random.default_rng(seed)
    def learn(s, facts, ph):
        for e, r, y in facts: s.seen[(e, r)] = y
        cur = np.array(facts)
        for _ in range(s.steps):
            b = cur
            if s.replay:
                keys = list(s.seen); pick = s.rng.integers(len(keys), size=len(cur))
                b = np.array([(*keys[i], s.seen[keys[i]]) for i in pick])
            e, r, y = (torch.from_numpy(b[:, j]) for j in range(3))
            logits = torch.einsum("bvm,bm->bv", s.W[r], s.u[e])
            loss = torch.nn.functional.cross_entropy(logits, y)
            s.opt.zero_grad(); loss.backward(); s.opt.step()
    def predict(s, e, r, store=True):
        with torch.no_grad():
            return torch.einsum("bvm,bm->bv", s.W[torch.from_numpy(r)], s.u[torch.from_numpy(e)]).argmax(-1).numpy()


class Additive:
    def __init__(s, n, m, lam=0.1, sweeps=3, old_sample=256, seed=0, **kw):
        s.rng = np.random.default_rng(seed); s.m, s.lam, s.sweeps, s.old_sample = m, lam, sweeps, old_sample
        s.u = np.zeros((n, m)); s.A = np.zeros((R, m, m)); s.B = np.zeros((R, m, V)); s.W = s.rng.normal(0, 0.1, (R, V, m))
        s.store, s.by_e, s.in_stats = {}, {}, set()
    def _sub(s, e):
        if e in s.in_stats:
            for r, y in s.by_e[e].items(): s.A[r] -= np.outer(s.u[e], s.u[e]); s.B[r][:, y] -= s.u[e]
            s.in_stats.discard(e)
    def _add(s, e):
        for r, y in s.by_e[e].items(): s.A[r] += np.outer(s.u[e], s.u[e]); s.B[r][:, y] += s.u[e]
        s.in_stats.add(e)
    def _solve_u(s, e):
        G = s.lam * np.eye(s.m); h = np.zeros(s.m)
        for r, y in s.by_e[e].items(): G += s.W[r].T @ s.W[r]; h += s.W[r][y]
        s.u[e] = np.linalg.solve(G, h)
    def _solve_W(s):
        for r in range(R): s.W[r] = np.linalg.solve(s.A[r] + s.lam * np.eye(s.m), s.B[r]).T
    def learn(s, facts, ph):
        touched = set()
        for e, r, y in facts:
            if (e, r) in s.store: s._sub(e)                                # a correction: remove the item, change it, re-add below
            s.store[(e, r)] = y; s.by_e.setdefault(e, {})[r] = y; touched.add(e)
        for e in touched:
            s._sub(e)
            if not s.u[e].any(): s.u[e] = s.rng.normal(0, 0.1, s.m)
        for _ in range(s.sweeps):
            old = [e for e in s.by_e if e not in touched]
            if len(old) > s.old_sample: old = list(s.rng.choice(old, s.old_sample, replace=False))
            for e in list(touched) + old:
                s._sub(e); s._solve_u(e); s._add(e)
            s._solve_W()
    def predict(s, e, r, store=True):
        out = np.einsum("bvm,bm->bv", s.W[r], s.u[e]).argmax(-1)
        if store:
            for i, (a, b) in enumerate(zip(e, r)):
                if (a, b) in s.store: out[i] = s.store[(a, b)]
        return out


def run(name, seed, phases, new, fix, m, **hp):
    rng, truth, shown = world(seed, phases, new, fix)
    n = phases * new
    L = {"table": Table, "grad": Grad, "grad_replay": lambda n, m, **k: Grad(n, m, replay=True, **k), "additive": Additive}[name](n, m, seed=seed, **hp)
    acc = lambda e, r, store=True: float((L.predict(e, r, store) == truth[e, r]).mean()) if len(e) else float("nan")
    log, t0 = [], time.time()
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]
        fe = fr = np.zeros(0, int)
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False)
            fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix)
            facts += [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        L.learn(facts, ph)
        mrow = {"phase": ph}
        ne, nr = np.nonzero(~shown[ents]); mrow["infer_new"] = acc(ents[ne], nr)
        if ph:
            oe, orr = np.nonzero(shown[:ph * new])
            mrow["retain_gen"] = acc(oe, orr, False); mrow["retain"] = acc(oe, orr, True); mrow["correct"] = acc(fe, fr, True)
            ue, ur = np.nonzero(~shown[:ph * new]); mrow["infer_old"] = acc(ue, ur)
        log.append(mrow)
    q = max(1, phases // 4); keys = ["infer_new", "infer_old", "retain_gen", "retain", "correct"]
    mean = lambda rows, k: round(float(np.nanmean([r.get(k, np.nan) for r in rows])), 3)
    return {"learner": name, "seed": seed, "hp": hp, "seconds": round(time.time() - t0, 1),
            "early": {k: mean(log[1:1 + q], k) for k in keys}, "late": {k: mean(log[-q:], k) for k in keys}, "log": log}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", default="table,grad,grad_replay,additive"); ap.add_argument("--seeds", default="4000")
    ap.add_argument("--phases", type=int, default=40); ap.add_argument("--new", type=int, default=64); ap.add_argument("--fix", type=int, default=16)
    ap.add_argument("--m", type=int, default=24); ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--out", default="")
    a = ap.parse_args(); rows = []
    for name in a.learners.split(","):
        hp = {"lr": a.lr} if name.startswith("grad") else {"lam": a.lam} if name == "additive" else {}
        for seed in [int(x) for x in a.seeds.split(",")]:
            r = run(name, seed, a.phases, a.new, a.fix, a.m, **hp); rows.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "log"}), flush=True)
    if a.out: json.dump(rows, open(a.out, "w"))
