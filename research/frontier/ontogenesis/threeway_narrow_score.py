"""Apply N1-N5 of PROTOCOL-THREEWAY-NARROW.md mechanically. Committed before any confirmatory result."""
import json
import numpy as np
A = ("none", "oracle", "oracle_s02", "oracle_s035", "threeway", "threeway_s02", "threeway_fixed")
D = {a: json.load(open(f"results/threeway_narrow/{a}.json")) for a in A}
m = lambda a, k: float(np.mean([r[k] for r in D[a]]))
pi = lambda a, k, i: D[a][i][k]
ok = lambda f: sum(1 for i in range(5) if f(i))
SO, RO, ST = "infer_stable_shared_only", "revise_old", "infer_stable"
N = {
 "N1": (m("oracle_s02", SO) - m("oracle", SO) >= 0.06, ok(lambda i: pi("oracle_s02", SO, i) - pi("oracle", SO, i) >= 0.06)),
 "N2": (m("oracle_s02", RO) >= 0.88, ok(lambda i: pi("oracle_s02", RO, i) >= 0.88)),
 "N3": (m("oracle_s035", RO) < m("oracle_s02", RO) and m("oracle_s035", SO) >= m("oracle_s02", SO),
        ok(lambda i: pi("oracle_s035", RO, i) < pi("oracle_s02", RO, i) and pi("oracle_s035", SO, i) >= pi("oracle_s02", SO, i))),
 "N4": (m("threeway_s02", SO) - m("threeway", SO) >= 0.04 and m("threeway", RO) - m("threeway_s02", RO) <= 0.10,
        ok(lambda i: pi("threeway_s02", SO, i) - pi("threeway", SO, i) >= 0.04 and pi("threeway", RO, i) - pi("threeway_s02", RO, i) <= 0.10)),
 "N5": (m("none", ST) - m("threeway_fixed", ST) <= 0.06 and m("threeway_fixed", RO) >= 0.60,
        ok(lambda i: pi("none", ST, i) - pi("threeway_fixed", ST, i) <= 0.06 and pi("threeway_fixed", RO, i) >= 0.60)),
}
K = (ST, SO, RO, "revise_next5", "exception_kept", "noise_resisted")
table = {a: {k: (round(m(a, k), 4) if not np.isnan(m(a, k)) else None) for k in K} for a in A}
json.dump({"predictions": {k: {"held_on_mean": bool(v[0]), "seeds_meeting_bound": v[1]} for k, v in N.items()}, "table": table}, open("results/threeway_narrow/verdict.json", "w"), indent=1)
for k, (h, n) in N.items(): print(k, "HELD" if h else "FAILED", f"({n} of 5 seeds)")
for a, v in table.items(): print(a, json.dumps(v))
