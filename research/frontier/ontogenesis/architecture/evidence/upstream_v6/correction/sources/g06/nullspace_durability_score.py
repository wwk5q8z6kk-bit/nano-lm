"""Apply predictions N1-N6 of nullspace_durability.py mechanically. Committed before any registered run.
A prediction holds only if its bound is met by the mean over seeds 8001-8003; per-seed values are printed."""
import json
import numpy as np

ARMS = ("none", "owm_1", "owm_1e-2", "owm_1e-4", "corr_replay")
D = {a: json.load(open(f"results/nullspace_durability/{a}.json")) for a in ARMS}
REC = json.load(open("results/correction_interference/constrained.json"))
m = lambda a, k: round(float(np.mean([r[k] for r in D[a]])), 4)
per = lambda a, k: [r[k] for r in D[a]]
N = {}
N["N1"] = {"held": per("none", "corrected_old") == [r["corrected_old"] for r in REC] and per("none", "corrected_now") == [r["corrected_now"] for r in REC],
           "none": per("none", "corrected_old"), "recorded": [r["corrected_old"] for r in REC]}
N["N2"] = {"held": m("owm_1e-4", "corrected_old") >= 0.80 and m("owm_1e-4", "rule_old_corrected") >= 0.95,
           "corrected_old": per("owm_1e-4", "corrected_old"), "rule_old_corrected": per("owm_1e-4", "rule_old_corrected")}
N["N3"] = {"held": abs(m("owm_1e-4", "acquire_now") - m("none", "acquire_now")) <= 0.05,
           "owm_1e-4": per("owm_1e-4", "acquire_now"), "none": per("none", "acquire_now")}
N["N4"] = {"held": abs(m("owm_1e-4", "infer_late") - m("none", "infer_late")) <= 0.05,
           "owm_1e-4": per("owm_1e-4", "infer_late"), "none": per("none", "infer_late")}
N["N5"] = {"held": m("corr_replay", "corrected_old") >= 0.90 and abs(m("corr_replay", "acquire_now") - m("none", "acquire_now")) <= 0.02,
           "corrected_old": per("corr_replay", "corrected_old"), "acquire_now": per("corr_replay", "acquire_now")}
N["N6"] = {"held": all(r["room_end"] is not None and r["room_end"] <= 0.05 for r in D["owm_1e-4"]), "room_end": {a: per(a, "room_end") for a in ARMS}}
table = {a: {k: m(a, k) for k in ("acquire_now", "corrected_now", "corrected_old", "rule_old_corrected", "rule_old_clean", "infer_late")} for a in ARMS}
json.dump({"predictions": N, "table": table}, open("results/nullspace_durability/verdict.json", "w"), indent=1)
for k, v in N.items(): print(k, "HELD" if v["held"] else "FAILED", json.dumps({a: b for a, b in v.items() if a != "held"}))
print(json.dumps(table, indent=1))
