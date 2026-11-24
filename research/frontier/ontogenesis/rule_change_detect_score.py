"""Apply predictions Y1-Y5 of rule_change_detect.py mechanically. Committed before any registered result."""
import json, sys
BASE = sys.argv[1] if len(sys.argv) > 1 else "results"   # results_v2 for the corrected reruns
import numpy as np
D = {a: json.load(open(f"{BASE}/rule_change_detect/{a}.json")) for a in ("seen2", "detect", "oracle_dec")}
m = lambda a, k: round(float(np.mean([r[k] for r in D[a]])), 4)
per = lambda a, k: [r[k] for r in D[a]]
Y = {}
Y["Y1"] = {"held": m("detect", "revise_old") >= 0.80 and m("detect", "revise_old") - m("seen2", "revise_old") >= 0.25, "detect": per("detect", "revise_old"), "seen2": per("seen2", "revise_old")}
Y["Y2"] = {"held": m("detect", "revise_next5") >= 0.85, "detect": per("detect", "revise_next5")}
Y["Y3"] = {"held": abs(m("detect", "infer_stable") - m("seen2", "infer_stable")) <= 0.02 and m("detect", "exception_kept") >= 0.97 and m("detect", "noise_resisted") >= 0.93,
           **{k: per("detect", k) for k in ("infer_stable", "exception_kept", "noise_resisted")}, "seen2_infer_stable": per("seen2", "infer_stable")}
Y["Y4"] = {"held": m("oracle_dec", "revise_old") - m("detect", "revise_old") <= 0.10, "oracle_dec": per("oracle_dec", "revise_old"), "detect": per("detect", "revise_old")}
Y["Y5"] = {"held": m("detect", "missed") <= 1, "missed": per("detect", "missed")}
keys = ("acquire_now", "infer_stable", "revise_next5", "revise_next10", "revise_old", "exception_kept", "noise_resisted", "store_size")
table = {a: {k: m(a, k) for k in keys} for a in D}
table["detect"].update({k: per("detect", k) for k in ("flags", "false_alarms", "missed", "delay_mean")})
json.dump({"predictions": Y, "table": table}, open(f"{BASE}/rule_change_detect/verdict.json", "w"), indent=1)
for k, v in Y.items(): print(k, "HELD" if v["held"] else "FAILED", json.dumps({a: b for a, b in v.items() if a != "held"}))
print(json.dumps(table, indent=1))
