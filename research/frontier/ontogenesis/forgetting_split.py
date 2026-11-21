"""What exactly does a gradient learner forget? A split of retention by kind of fact.

Separate investigation; does not alter cab_s.py or any registered run. Reuses its learners.

Two explanations of forgetting are set against each other.
  H-consistency  Forgetting is confined to content that later data does not agree with:
                 arbitrary exceptions and overwritten values. Content that follows the world's
                 rules is not forgotten, because every later example re-teaches the same rule.
                 On this view "retention" of regular knowledge is just inference.
  H-order        Forgetting is caused by training order itself: whatever was learned earlier
                 is eroded by later updates, regular or not.

World as in cab_s.py, with the exception rate EPS varied. Learner: the gradient learner trained
on the current phase only (lr 0.003, the value tuned earlier on seed 4000). The additive learner's
model-alone retention is recorded for reference. Seeds 5001-5003, 40 phases.

At the end, retention by the model alone is measured on shown facts, split by
  kind  rule (value equals the rule for the entity's type), exception (arbitrary from the start),
        corrected (overwritten later by an arbitrary value)
  age   phases since the fact was last shown: 0-4, 5-19, 20+

Predictions, written before any run of this file:
  C1 (H-consistency) With EPS 0.1 and 0.3, gradient retention on rule facts aged 20+ is at least
     0.95, and at least 0.30 above retention on exception facts aged 20+.
  C2 (H-consistency) With EPS 0 and no corrections, overall gradient retention is at least 0.97.
  O1 (H-order) Rule facts aged 20+ are retained at least 0.10 worse than rule facts aged 0-4.
  If C1 and C2 hold and O1 fails, forgetting in this world is a property of irregular content,
  not of learning order. If O1 holds, order matters even for regular content.
  Either way the result is confined to a world whose rules never change.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import cab_s
from cab_s import Z, R, V, K, Grad, Additive


def run(name, seed, eps, phases, new, fix, m, **hp):
    rng = np.random.default_rng(seed); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n)
    truth = np.where(rng.random((n, R)) < eps, rng.integers(V, size=(n, R)), rule[typ])
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    last = np.repeat(np.arange(phases), new)[:, None].repeat(R, 1)          # phase each fact was last shown
    corrected = np.zeros((n, R), bool)
    L = (Grad if name == "grad" else Additive)(n, m, seed=seed, **hp)
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]
        if ph and fix:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False)
            fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True; last[fe, fr] = ph
            facts += [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        L.learn(facts, ph)
    e, r = np.nonzero(shown); ok = L.predict(e, r, False) == truth[e, r]
    age = (phases - 1) - last[e, r]
    kind = np.where(corrected[e, r], 2, np.where(truth[e, r] == rule[typ[e], r], 0, 1))
    out = {"learner": name, "seed": seed, "eps": eps, "fix": fix, "overall": round(float(ok.mean()), 4)}
    for ki, kn in enumerate(("rule", "exception", "corrected")):
        for lo, hi, an in ((0, 4, "0-4"), (5, 19, "5-19"), (20, 10 ** 9, "20+")):
            sel = (kind == ki) & (age >= lo) & (age <= hi)
            out[f"{kn}_{an}"] = [round(float(ok[sel].mean()), 4) if sel.any() else None, int(sel.sum())]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", default="grad"); ap.add_argument("--eps", type=float, default=0.1); ap.add_argument("--fix", type=int, default=16)
    ap.add_argument("--seeds", default="5001,5002,5003"); ap.add_argument("--out", default="")
    a = ap.parse_args(); rows = []
    hp = {"lr": 0.003} if a.learner == "grad" else {"lam": 1.0}
    for seed in [int(x) for x in a.seeds.split(",")]:
        rows.append(run(a.learner, seed, a.eps, 40, 64, a.fix, 24, **hp)); print(json.dumps(rows[-1]), flush=True)
    if a.out: json.dump(rows, open(a.out, "w"))
