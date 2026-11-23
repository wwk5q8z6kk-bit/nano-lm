"""Apply C1-C5 of PROTOCOL-THREEWAY-COST.md mechanically. Committed before any confirmatory result."""
import json, sys
BASE = sys.argv[1] if len(sys.argv) > 1 else "results"   # results_v2 for the corrected reruns
import numpy as np
A = ("none", "oracle", "oracle_x16", "oracle_nochange", "threeway", "threeway_x16")
D = {a: json.load(open(f"{BASE}/threeway_cost/{a}.json")) for a in A}
m = lambda a, k: float(np.mean([r[k] for r in D[a]]))
pi = lambda a, k, i: D[a][i][k]
ok = lambda f: sum(1 for i in range(5) if f(i))
gap = lambda a, i: pi(a, "infer_stable_shared_only", i) - pi(a, "infer_stable", i)
C = {
 "C1": (m("oracle", "infer_stable_shared_only") - m("oracle", "infer_stable") >= 0.02, ok(lambda i: gap("oracle", i) >= 0.02)),
 "C2": (m("oracle_x16", "infer_stable_shared_only") - m("oracle_x16", "infer_stable") <= 0.01, ok(lambda i: gap("oracle_x16", i) <= 0.01)),
 "C3": (m("none", "infer_stable_shared_only") - m("oracle", "infer_stable_shared_only") >= 0.08, ok(lambda i: pi("none", "infer_stable_shared_only", i) - pi("oracle", "infer_stable_shared_only", i) >= 0.08)),
 "C4": (abs(m("oracle_nochange", "infer_stable_shared_only") - m("none", "infer_stable_shared_only")) <= 0.02, ok(lambda i: abs(pi("oracle_nochange", "infer_stable_shared_only", i) - pi("none", "infer_stable_shared_only", i)) <= 0.02)),
 "C5": (m("threeway_x16", "infer_stable") - m("threeway", "infer_stable") >= 0.02, ok(lambda i: pi("threeway_x16", "infer_stable", i) - pi("threeway", "infer_stable", i) >= 0.02)),
}
K = ("infer_stable", "infer_stable_shared_only", "revise_old", "exception_kept", "noise_resisted")
table = {a: {k: round(m(a, k), 4) if not np.isnan(m(a, k)) else None for k in K} for a in A}
json.dump({"predictions": {k: {"held_on_mean": bool(v[0]), "seeds_meeting_bound": v[1]} for k, v in C.items()}, "table": table}, open(f"{BASE}/threeway_cost/verdict.json", "w"), indent=1)
for k, (h, n) in C.items(): print(k, "HELD" if h else "FAILED", f"({n} of 5 seeds)")
for a, v in table.items(): print(a, json.dumps(v))
