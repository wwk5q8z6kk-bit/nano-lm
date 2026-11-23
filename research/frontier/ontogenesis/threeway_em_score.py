"""Apply E1-E5 of PROTOCOL-THREEWAY-EM.md mechanically. Committed before any confirmatory result."""
import json
import numpy as np
A = ("given_det", "given_nodet", "em_det", "em_nodet")
D = {a: json.load(open(f"results/threeway_em/{a}.json")) for a in A}
m = lambda a, k: float(np.mean([r[k] for r in D[a]]))
pi = lambda a, k, i: D[a][i][k]
ok = lambda f: sum(1 for i in range(5) if f(i))
E = {
 "E1": (m("em_det", "purity") >= 0.95 and m("em_det", "changed_types_split") <= 0.20, ok(lambda i: pi("em_det", "purity", i) >= 0.95 and pi("em_det", "changed_types_split", i) <= 0.20)),
 "E2": (m("em_det", "revise_old") - m("em_nodet", "revise_old") >= 0.25, ok(lambda i: pi("em_det", "revise_old", i) - pi("em_nodet", "revise_old", i) >= 0.25)),
 "E3": (0.05 <= m("given_det", "revise_old") - m("em_det", "revise_old") <= 0.30, ok(lambda i: 0.05 <= pi("given_det", "revise_old", i) - pi("em_det", "revise_old", i) <= 0.30)),
 "E4": (m("given_det", "infer_stable") - m("em_det", "infer_stable") <= 0.10, ok(lambda i: pi("given_det", "infer_stable", i) - pi("em_det", "infer_stable", i) <= 0.10)),
 "E5": (all(m(a, "exception_kept") >= 0.93 and m(a, "noise_resisted") >= 0.90 for a in A), ok(lambda i: all(pi(a, "exception_kept", i) >= 0.93 and pi(a, "noise_resisted", i) >= 0.90 for a in A))),
}
K = ("infer_stable", "infer_stable_shared_only", "revise_old", "revise_next5", "exception_kept", "noise_resisted", "purity", "clusters_used", "changed_types_split", "flags", "false_alarms", "missed")
def mm(a, k):
    xs = [r.get(k) for r in D[a]]
    return None if any(x is None for x in xs) else round(float(np.mean(xs)), 4)
table = {a: {k: mm(a, k) for k in K} for a in A}
json.dump({"predictions": {k: {"held_on_mean": bool(v[0]), "seeds_meeting_bound": v[1]} for k, v in E.items()}, "table": table}, open("results/threeway_em/verdict.json", "w"), indent=1)
for k, (h, n) in E.items(): print(k, "HELD" if h else "FAILED", f"({n} of 5 seeds)")
for a, v in table.items(): print(a, json.dumps(v))
