"""Apply Q1-Q4 of PROTOCOL-THREEWAY-EM-COOLDOWN.md mechanically. Committed before any confirmatory result."""
import json, sys
import numpy as np
BASE = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
A = ("em_nodet", "em_narrow", "em_narrow_cd")
D = {a: json.load(open(f"{BASE}/threeway_em_cooldown/{a}.json")) for a in A}
m = lambda a, k: float(np.mean([r[k] for r in D[a]]))
pi = lambda a, k, i: D[a][i][k]
ok = lambda f: sum(1 for i in range(5) if f(i))
Q = {
 "Q1": (abs(m("em_narrow_cd", "infer_stable") - m("em_nodet", "infer_stable")) <= 0.02, ok(lambda i: abs(pi("em_narrow_cd", "infer_stable", i) - pi("em_nodet", "infer_stable", i)) <= 0.02)),
 "Q2": (m("em_narrow_cd", "revise_old") - m("em_nodet", "revise_old") >= 0.25, ok(lambda i: pi("em_narrow_cd", "revise_old", i) - pi("em_nodet", "revise_old", i) >= 0.25)),
 "Q3": (m("em_narrow_cd", "infer_stable") - m("em_narrow", "infer_stable") >= 0.02, ok(lambda i: pi("em_narrow_cd", "infer_stable", i) - pi("em_narrow", "infer_stable", i) >= 0.02)),
 "Q4": (m("em_narrow_cd", "exception_kept") >= 0.93 and abs(m("em_narrow_cd", "noise_resisted") - m("em_nodet", "noise_resisted")) <= 0.02,
        ok(lambda i: pi("em_narrow_cd", "exception_kept", i) >= 0.93 and abs(pi("em_narrow_cd", "noise_resisted", i) - pi("em_nodet", "noise_resisted", i)) <= 0.02)),
}
keys = ("infer_stable", "revise_old", "revise_next5", "exception_kept", "noise_resisted", "flags", "false_alarms", "missed", "purity", "changed_types_split")
table = {a: {k: round(m(a, k), 4) for k in keys} for a in A}
json.dump({"predictions": {k: {"held_on_mean": bool(v[0]), "seeds_meeting_bound": v[1]} for k, v in Q.items()}, "table": table}, open(f"{BASE}/threeway_em_cooldown/verdict.json", "w"), indent=1)
for k, (h, n) in Q.items(): print(k, "HELD" if h else "FAILED", f"({n} of 5 seeds)")
for a, v in table.items(): print(a, json.dumps(v))
