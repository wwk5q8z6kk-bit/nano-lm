"""Apply predictions Z1-Z8 of PROTOCOL-THREEWAY.md mechanically. Committed before any registered result."""
import json, sys
BASE = sys.argv[1] if len(sys.argv) > 1 else "results"   # results_v2 for the corrected reruns
import numpy as np
ARMS = ("none", "private", "private_t02", "change_only", "threeway", "threeway_t02", "threeway_rev200", "threeway_privfirst", "threeway_random", "oracle")
D = {a: json.load(open(f"{BASE}/threeway/{a}.json")) for a in ARMS}
m = lambda a, k: float(np.nanmean([np.nan if r[k] is None else r[k] for r in D[a]]))
per = lambda a, k: [r[k] for r in D[a]]
def seeds_ok(f): return sum(1 for i in range(5) if f(i))
pi = lambda a, k, i: D[a][i][k] if D[a][i][k] is not None else np.nan
Z = {}
Z["Z1"] = (m("threeway", "revise_old") - m("none", "revise_old") >= 0.30, seeds_ok(lambda i: pi("threeway", "revise_old", i) - pi("none", "revise_old", i) >= 0.30))
Z["Z2"] = (m("threeway", "revise_old") - m("private", "revise_old") >= 0.30, seeds_ok(lambda i: pi("threeway", "revise_old", i) - pi("private", "revise_old", i) >= 0.30))
Z["Z3"] = (m("threeway", "exception_kept") - m("none", "exception_kept") >= 0.30, seeds_ok(lambda i: pi("threeway", "exception_kept", i) - pi("none", "exception_kept", i) >= 0.30))
Z["Z4"] = (m("threeway", "revise_old") - m("threeway_privfirst", "revise_old") >= 0.15, seeds_ok(lambda i: pi("threeway", "revise_old", i) - pi("threeway_privfirst", "revise_old", i) >= 0.15))
Z["Z5"] = (m("threeway", "exception_kept") - m("threeway", "exceptions_in_false_groups_kept") >= 0.20, seeds_ok(lambda i: pi("threeway", "exception_kept", i) - pi("threeway", "exceptions_in_false_groups_kept", i) >= 0.20))
Z["Z6"] = (m("threeway", "infer_stable_other_pairs") - m("threeway", "infer_stable_fa_pairs") >= 0.10, seeds_ok(lambda i: pi("threeway", "infer_stable_other_pairs", i) - pi("threeway", "infer_stable_fa_pairs", i) >= 0.10))
Z["Z7"] = (m("threeway", "exception_kept") - m("threeway_random", "exception_kept") >= 0.20, seeds_ok(lambda i: pi("threeway", "exception_kept", i) - pi("threeway_random", "exception_kept", i) >= 0.20))
Z["Z8"] = (m("threeway", "revise_old") - m("threeway_rev200", "revise_old") >= 0.10, seeds_ok(lambda i: pi("threeway", "revise_old", i) - pi("threeway_rev200", "revise_old", i) >= 0.10))
keys = ("acquire_now", "infer_stable", "infer_stable_fa_pairs", "infer_stable_other_pairs", "revise_next5", "revise_old", "exception_kept", "noise_resisted",
        "routed_private", "flags", "false_alarms", "missed", "exceptions_in_false_groups", "exceptions_in_false_groups_kept")
table = {a: {k: round(m(a, k), 4) for k in keys} for a in ARMS}
json.dump({"predictions": {k: {"held_on_mean": bool(v[0]), "seeds_meeting_bound": v[1]} for k, v in Z.items()}, "table": table}, open(f"{BASE}/threeway/verdict.json", "w"), indent=1)
for k, (h, n) in Z.items(): print(k, "HELD" if h else "FAILED", f"(bound met on {n} of 5 seeds)")
for a, v in table.items(): print(a, json.dumps(v))
