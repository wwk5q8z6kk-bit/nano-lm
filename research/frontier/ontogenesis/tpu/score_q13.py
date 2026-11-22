"""Mechanical reading of Question 13's predictions Q2 and Q3 (INVESTIGATION.md). Committed before the 386M run finished.

Q2: the 386M model's final held-out mix (bits per byte) lies in [0.79, 0.83].
Q3a: it beats our 211M model (results/scale1/dense-XL-...-4.2e+09...) on ARC-Easy and PIQA (plain accuracy, both).
Q3b: against Pythia-410M's official figures as registered (ARC-Easy 0.521, PIQA 0.668, LAMBADA perplexity 10.8):
     within 0.03 on ARC-Easy and on PIQA (absolute difference, plain accuracy), and worse (higher) LAMBADA perplexity.
Plain accuracy is used throughout because the registered Pythia figures are plain accuracy and the 211M comparison
in Question 12 was reported in plain accuracy. Length-normalised accuracy and the SmolLM2 references are reported
alongside, not scored. If the full-rate run diverged and the half-rate rerun finished, the rerun is scored and named.

usage: score_q13.py <run.json> <run.bench.json> <xl.bench.json> [--nocontext <run.bench-nocontext.json>] [--out verdict.json]
"""
from __future__ import annotations
import argparse, json

PYTHIA_410M = {"arc_easy": 0.521, "piqa": 0.668, "lambada_ppl": 10.8}

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("bench"); ap.add_argument("xl")
ap.add_argument("--nocontext", default=""); ap.add_argument("--out", default="")
a = ap.parse_args()
run, b, xl = json.load(open(a.run)), json.load(open(a.bench))["tasks"], json.load(open(a.xl))["tasks"]
if not run.get("finished"): raise SystemExit(f"{a.run} is not finished; nothing is scored")
mix = run["curve"][-1]["mix"]
v = {"run": run["name"], "final_step": run["curve"][-1]["step"], "final_mix": mix, "final_domains": {k: run["curve"][-1][k] for k in ("web", "textbook", "math", "code")}}
v["Q2"] = {"held": 0.79 <= mix <= 0.83, "bound": [0.79, 0.83], "value": mix}
q3a = {k: {"ours_386M": b[k]["acc"], "ours_211M": xl[k]["acc"], "beats": b[k]["acc"] > xl[k]["acc"]} for k in ("arc_easy", "piqa")}
v["Q3a"] = {"held": all(x["beats"] for x in q3a.values()), **q3a}
q3b = {k: {"ours": b[k]["acc"], "pythia_410m": PYTHIA_410M[k], "within_0.03": abs(b[k]["acc"] - PYTHIA_410M[k]) <= 0.03} for k in ("arc_easy", "piqa")}
q3b["lambada_ppl"] = {"ours": b["lambada"]["ppl"], "pythia_410m": PYTHIA_410M["lambada_ppl"], "worse": b["lambada"]["ppl"] > PYTHIA_410M["lambada_ppl"]}
v["Q3b"] = {"held": q3b["arc_easy"]["within_0.03"] and q3b["piqa"]["within_0.03"] and q3b["lambada_ppl"]["worse"], **q3b}
v["all_tasks_acc"] = {k: {"acc": t["acc"], "acc_norm": t.get("acc_norm"), "ppl": t.get("ppl")} for k, t in b.items()}
if a.nocontext:
    nc = json.load(open(a.nocontext))["tasks"]
    v["nocontext_acc"] = {k: t["acc"] for k, t in nc.items()}
print(json.dumps(v, indent=1))
if a.out: json.dump(v, open(a.out, "w"), indent=1)
