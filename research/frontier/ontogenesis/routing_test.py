"""Causal test: does keeping irregular facts out of the gradient-trained model stop it forgetting regular ones?

Background (exploratory, found after forgetting_split.py, three seeds, EPS 0.1): at the end of
training the gradient learner retained old regular facts at 100% on entities with no irregular
fact, and at 78-80% on entities that also carried an exception or a correction. That was noticed
after the fact, so it is a hypothesis, not a result. This file tests it by intervention.

Hypothesis: forgetting of regular knowledge is collateral damage. An entity's address is bent to
memorise its irregular fact; the shared decoder later drifts back toward the rules; the bent
address then decodes wrongly for all of that entity's facts.

Arms (same world, seeds 6001-6003, EPS 0.1, 40 phases, lr 0.003, 200 steps per phase):
  mixed    every shown fact trains the model (as before)
  oracle   irregular facts (exceptions, corrections) go to an exact store and never train the model
  self     no oracle. From phase 5 on: train 100 steps on everything, then any fact the model still
           gets wrong goes to the exact store and is dropped from the remaining 100 steps
Measured at the end, model alone: retention of regular facts aged 20+ phases, on entities with and
without an irregular shown fact. Also inference on unseen relations of the last ten phases' entities.

Predictions, written before any run of this file:
  R1 mixed reproduces the gap: at least 0.15 between the two kinds of entity.
  R2 oracle removes it: regular facts on irregular-carrying entities at least 0.97.
  R3 self recovers at least half the gap: at least 0.90 on irregular-carrying entities.
  R4 routing does not hurt inference: oracle and self within 0.03 of mixed or better.
  If R2 fails, the collateral-damage hypothesis is wrong.
"""
import json, sys
import numpy as np, torch
from cab_s import Z, R, V, K, Grad

def run(arm, seed, eps=0.1, phases=40, new=64, fix=16, m=24):
    rng = np.random.default_rng(seed); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n)
    truth = np.where(rng.random((n, R)) < eps, rng.integers(V, size=(n, R)), rule[typ])
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    corrected = np.zeros((n, R), bool)
    L = Grad(n, m, seed=seed, lr=0.003, steps=200); stored = 0
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False); fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True
            facts += [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        if arm == "oracle":
            keep = [f for f in facts if f[2] == rule[typ[f[0]], f[1]]]; stored += len(facts) - len(keep); L.learn(keep, ph)
        elif arm == "self" and ph >= 5:
            L.steps = 100; L.learn(facts, ph)
            a = np.array(facts); wrong = L.predict(a[:, 0], a[:, 1], False) != a[:, 2]
            keep = [f for f, w in zip(facts, wrong) if not w]; stored += int(wrong.sum()); L.learn(keep, ph); L.steps = 200
        else: L.learn(facts, ph)
    e, r = np.nonzero(shown); ok = L.predict(e, r, False) == truth[e, r]
    isrule = (truth[e, r] == rule[typ[e], r]) & ~corrected[e, r]
    irr = ((shown & ((truth != rule[typ]) | corrected)).sum(1) > 0)[e]
    old = (phases - 1) - np.repeat(np.arange(phases), new)[e] >= 20
    late = np.arange((phases - 10) * new, n); ue, ur = np.nonzero(~shown[late])
    infer = float((L.predict(late[ue], ur, False) == truth[late[ue], ur]).mean())
    return {"arm": arm, "seed": seed, "rule_old_clean": round(float(ok[isrule & old & ~irr].mean()), 4),
            "rule_old_irregular": round(float(ok[isrule & old & irr].mean()), 4), "infer_late": round(infer, 4),
            "stored_fraction": round(stored / int(shown.sum() + fix * (phases - 1)), 4)}

if __name__ == "__main__":
    arm = sys.argv[1]; rows = [run(arm, s) for s in (6001, 6002, 6003)]
    for r in rows: print(json.dumps(r), flush=True)
    json.dump(rows, open(f"results/routing/{arm}.json", "w"))
