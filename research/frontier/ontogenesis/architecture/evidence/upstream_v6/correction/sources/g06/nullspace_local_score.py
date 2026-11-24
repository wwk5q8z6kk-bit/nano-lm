"""Apply predictions L1-L6 of nullspace_local.py mechanically. Committed before any registered result existed."""
import json
import numpy as np

ARMS = ("none_m24", "owm_all_m24", "owm_ent_m24", "owm_rel_m24", "owm_all-a10_m24", "owm_all-a100_m24", "none_m96", "owm_all_m96", "owm_ent_m96", "owm_rel_m96")
D = {a: json.load(open(f"results/nullspace_local/{a}.json")) for a in ARMS}
G = {a: json.load(open(f"results/nullspace_durability/{a}.json")) for a in ("none", "owm_1")}
m = lambda a, k, src=D: round(float(np.mean([r[k] for r in src[a]])), 4)
per = lambda a, k, src=D: [r[k] for r in src[a]]
L = {}
L["L1"] = {"held": per("none_m24", "corrected_old") == per("none", "corrected_old", G) and per("none_m24", "infer_late") == per("none", "infer_late", G)
           and abs(m("owm_all_m24", "corrected_old") - m("owm_1", "corrected_old", G)) <= 0.01 and abs(m("owm_all_m24", "infer_late") - m("owm_1", "infer_late", G)) <= 0.01,
           "none_m24": per("none_m24", "corrected_old"), "g06_none": per("none", "corrected_old", G),
           "owm_all_m24": [m("owm_all_m24", "corrected_old"), m("owm_all_m24", "infer_late")], "g06_owm_1": [m("owm_1", "corrected_old", G), m("owm_1", "infer_late", G)]}
L["L2"] = {"held": m("owm_ent_m24", "infer_late") <= 0.80, "infer_late": per("owm_ent_m24", "infer_late")}
L["L3"] = {"held": m("owm_rel_m24", "corrected_old") >= 0.90 and m("owm_rel_m24", "infer_late") >= 0.85,
           "corrected_old": per("owm_rel_m24", "corrected_old"), "infer_late": per("owm_rel_m24", "infer_late")}
L["L4"] = {"held": m("owm_rel_m96", "corrected_old") >= 0.90 and abs(m("owm_rel_m96", "infer_late") - m("none_m96", "infer_late")) <= 0.03,
           "corrected_old": per("owm_rel_m96", "corrected_old"), "infer_late": per("owm_rel_m96", "infer_late"), "none_m96_infer": per("none_m96", "infer_late")}
L["L5"] = {"held": m("owm_all_m96", "infer_late") <= 0.85, "infer_late": per("owm_all_m96", "infer_late")}
co = [m(a, "corrected_old") for a in ("owm_all_m24", "owm_all-a10_m24", "owm_all-a100_m24")]
inf = [m(a, "infer_late") for a in ("owm_all_m24", "owm_all-a10_m24", "owm_all-a100_m24")]
L["L6"] = {"held": co[0] >= co[1] >= co[2] and inf[0] <= inf[1] <= inf[2], "corrected_old_alpha_1_10_100": co, "infer_late_alpha_1_10_100": inf}
table = {a: {k: m(a, k) for k in ("acquire_now", "corrected_now", "corrected_old", "rule_old_corrected", "rule_old_clean", "infer_late")} | {"room_end": per(a, "room_end")} for a in ARMS}
json.dump({"predictions": L, "table": table}, open("results/nullspace_local/verdict.json", "w"), indent=1)
for k, v in L.items(): print(k, "HELD" if v["held"] else "FAILED", json.dumps({a: b for a, b in v.items() if a != "held"}))
print(json.dumps(table, indent=1))
