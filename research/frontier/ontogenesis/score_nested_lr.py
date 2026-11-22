"""Mechanical reading of PROTOCOL-NESTED-LR.md. Committed before any of its results.

A run "learned" if two-step accuracy (curve entry [step, seconds, {"h2": ...}]) reaches at least 0.90 at any evaluation.
1e-3 counts come from results/nested/<arm>-s<seed>.json; 3e-4 and 3e-3 from <dir>/<arm>-lr<rate>-s<seed>.json.
usage: score_nested_lr.py <dir with the 40 new files> [--out verdict.json]
"""
from __future__ import annotations
import json, os, sys

ARMS = {"A": ("looped", "dense"), "B": ("tf-4x2-d64", "tf-8x1-d64")}           # (reuse, no reuse)
SEEDS, NEW = (21, 22, 23, 24, 25), ("3e-4", "3e-3")
learned = lambda f: any(e[2]["h2"] >= 0.90 for e in json.load(open(f))["curve"])
d = sys.argv[1]; out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else ""
counts, missing = {}, []
for arm in [a for p in ARMS.values() for a in p]:
    counts[arm] = {"1e-3": sum(learned(f"results/nested/{arm}-s{s}.json") for s in SEEDS)}
    for lr in NEW:
        fs = [f"{d}/{arm}-lr{lr}-s{s}.json" for s in SEEDS]; missing += [f for f in fs if not os.path.exists(f)]
        counts[arm][lr] = sum(learned(f) for f in fs if os.path.exists(f))
if missing: raise SystemExit(f"missing {len(missing)} result files, e.g. {missing[0]}; nothing is scored")
best = {a: max(c.values()) for a, c in counts.items()}
pairs = {p: {"reuse": r, "no_reuse": n, "reuse_best": best[r], "no_reuse_best": best[n],
             "stands": best[r] >= best[n] + 2, "explained": best[n] >= best[r] - 1} for p, (r, n) in ARMS.items()}
v = {"counts": counts, "best": best, "pairs": pairs,
     "L1_dense_3e-4_ge3": counts["dense"]["3e-4"] >= 3, "L2_8x1_3e-4_ge3": counts["tf-8x1-d64"]["3e-4"] >= 3,
     "L3_3e-3_no_arm_above_1e-3": all(c["3e-3"] <= c["1e-3"] for c in counts.values())}
v["decision"] = ("reuse advantage stands" if all(p["stands"] for p in pairs.values())
                 else "explained by learning rate" if all(p["explained"] for p in pairs.values()) else "mixed")
print(json.dumps(v, indent=1))
if out: json.dump(v, open(out, "w"), indent=1)
