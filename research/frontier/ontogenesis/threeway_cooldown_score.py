"""Apply K1-K5 of PROTOCOL-THREEWAY-COOLDOWN.md mechanically. Committed before any confirmatory result."""
import json, sys
import numpy as np
BASE = sys.argv[1] if len(sys.argv) > 1 else "results_v2"
A = ("none", "threeway", "threeway_cd", "threeway_s02", "threeway_s02_cd")
D = {a: json.load(open(f"{BASE}/threeway_cooldown/{a}.json")) for a in A}
m = lambda a, k: float(np.mean([r[k] for r in D[a]]))
pi = lambda a, k, i: D[a][i][k]
ok = lambda f: sum(1 for i in range(5) if f(i))
K = {
 "K1": (m("threeway_cd", "flags") <= 0.6 * m("threeway", "flags"), ok(lambda i: pi("threeway_cd", "flags", i) <= 0.6 * pi("threeway", "flags", i))),
 "K2": (m("threeway_cd", "missed") - m("threeway", "missed") <= 0.5 and abs(m("threeway_cd", "revise_old") - m("threeway", "revise_old")) <= 0.05,
        ok(lambda i: pi("threeway_cd", "missed", i) - pi("threeway", "missed", i) <= 1 and abs(pi("threeway_cd", "revise_old", i) - pi("threeway", "revise_old", i)) <= 0.05)),
 "K3": (m("threeway_s02_cd", "infer_stable") - m("threeway", "infer_stable") >= 0.03, ok(lambda i: pi("threeway_s02_cd", "infer_stable", i) - pi("threeway", "infer_stable", i) >= 0.03)),
 "K4": (m("threeway_s02_cd", "exception_kept") - m("threeway", "exception_kept") >= 0.03, ok(lambda i: pi("threeway_s02_cd", "exception_kept", i) - pi("threeway", "exception_kept", i) >= 0.03)),
 "K5": (m("none", "infer_stable") - m("threeway_s02_cd", "infer_stable") <= 0.08, ok(lambda i: pi("none", "infer_stable", i) - pi("threeway_s02_cd", "infer_stable", i) <= 0.08)),
}
keys = ("infer_stable", "infer_stable_shared_only", "revise_old", "revise_next5", "exception_kept", "noise_resisted", "flags", "false_alarms", "missed")
table = {a: {k: round(m(a, k), 4) for k in keys} for a in A}
json.dump({"predictions": {k: {"held_on_mean": bool(v[0]), "seeds_meeting_bound": v[1]} for k, v in K.items()}, "table": table}, open(f"{BASE}/threeway_cooldown/verdict.json", "w"), indent=1)
for k, (h, n) in K.items(): print(k, "HELD" if h else "FAILED", f"({n} of 5 seeds)")
for a, v in table.items(): print(a, json.dumps(v))
