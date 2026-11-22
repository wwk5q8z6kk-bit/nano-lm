"""Apply predictions X1-X6 of rule_change.py mechanically. Committed before any registered result existed."""
import json
import numpy as np

ARMS = ("plain", "seen2", "oracle", "oracle_dec")
D = {a: json.load(open(f"results/rule_change/{a}.json")) for a in ARMS}
m = lambda a, k: round(float(np.mean([r[k] for r in D[a]])), 4)
per = lambda a, k: [r[k] for r in D[a]]
X = {}
X["X1"] = {"held": m("plain", "noise_resisted") <= 0.30 and m("plain", "exception_kept") <= 0.85 and 0.50 <= m("plain", "revise_next5") <= 0.85,
           **{k: per("plain", k) for k in ("noise_resisted", "exception_kept", "revise_next5")}}
X["X2"] = {"held": m("seen2", "noise_resisted") >= 0.95 and m("seen2", "exception_kept") >= 0.95 and abs(m("seen2", "revise_next5") - m("plain", "revise_next5")) <= 0.10 and m("seen2", "revise_old") <= 0.70,
           **{k: per("seen2", k) for k in ("noise_resisted", "exception_kept", "revise_next5", "revise_old")}}
X["X3"] = {"held": all(m("oracle_dec", k) >= 0.95 for k in ("revise_next5", "revise_old", "exception_kept", "noise_resisted")),
           **{k: per("oracle_dec", k) for k in ("revise_next5", "revise_old", "exception_kept", "noise_resisted")}}
X["X4"] = {"held": m("oracle", "revise_old") <= 0.70, "revise_old": per("oracle", "revise_old")}
X["X5"] = {"held": m("oracle_dec", "revise_old") - m("seen2", "revise_old") >= 0.30, "gap": round(m("oracle_dec", "revise_old") - m("seen2", "revise_old"), 4)}
X["X6"] = {"held": all(m(a, "infer_stable") >= 0.90 for a in ARMS), **{a: m(a, "infer_stable") for a in ARMS}}
keys = ("acquire_now", "infer_stable", "revise_next5", "revise_next10", "revise_old", "exception_kept", "noise_resisted", "store_size")
table = {a: {k: m(a, k) for k in keys} for a in ARMS}
json.dump({"predictions": X, "table": table}, open("results/rule_change/verdict.json", "w"), indent=1)
for k, v in X.items(): print(k, "HELD" if v["held"] else "FAILED", json.dumps({a: b for a, b in v.items() if a != "held"}))
print(json.dumps(table, indent=1))
