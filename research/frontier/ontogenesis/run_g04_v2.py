"""Rerun every G04 study affected by the Bugbot findings on PR #204, with the corrected code (version 2).

Arm settings are exactly those of the original registered runs. The original runs used inline runners that were never
committed (a gap Bugbot exposed by reading threeway.py's __main__ alone); this file records them. Originals stay in
results/; corrected results go to results_v2/.

usage: run_g04_v2.py STUDY      STUDY in detect, threeway, cost, narrow, em
"""
import json, os, sys
import threeway as t, rule_change_detect as d

STUDIES = {
 "detect": ([7201, 7202, 7203], {a: ("detect_module", a, {}) for a in ("seen2", "detect", "oracle_dec")}),
 "threeway": ([7301, 7302, 7303, 7304, 7305], {a: ("threeway", a, {}) for a in ("none", "private", "private_t02", "change_only", "threeway",
                "threeway_t02", "threeway_rev200", "threeway_privfirst", "threeway_random", "oracle")}),
 "cost": ([7401, 7402, 7403, 7404, 7405], {"none": ("threeway", "none", {}), "oracle": ("threeway", "oracle", {}),
          "oracle_x16": ("threeway", "oracle", {"p_slots": 262144}), "oracle_nochange": ("threeway", "oracle", {"changes_on": False}),
          "threeway": ("threeway", "threeway", {}), "threeway_x16": ("threeway", "threeway", {"p_slots": 262144})}),
 "narrow": ([7501, 7502, 7503, 7504, 7505], {"none": ("threeway", "none", {}), "oracle": ("threeway", "oracle", {}),
            "oracle_s02": ("threeway", "oracle", {"narrow": True, "share": 0.2}), "oracle_s035": ("threeway", "oracle", {"narrow": True, "share": 0.35}),
            "threeway": ("threeway", "threeway", {}), "threeway_s02": ("threeway", "threeway", {"narrow": True, "share": 0.2}),
            "threeway_fixed": ("threeway", "threeway", {"narrow": True, "share": 0.2, "p_slots": 262144})}),
 "cooldown": ([7701, 7702, 7703, 7704, 7705], {"none": ("threeway", "none", {}), "threeway": ("threeway", "threeway", {}),
              "threeway_cd": ("threeway", "threeway", {"cooldown": True}), "threeway_s02": ("threeway", "threeway", {"narrow": True, "share": 0.2}),
              "threeway_s02_cd": ("threeway", "threeway", {"narrow": True, "share": 0.2, "cooldown": True})}),
 "em": ([7601, 7602, 7603, 7604, 7605], {"given_det": ("threeway", "threeway", {"codes": "given32", "detector": True, "narrow": True, "share": 0.2}),
        "given_nodet": ("threeway", "threeway", {"codes": "given32", "detector": False, "narrow": True, "share": 0.2}),
        "em_det": ("threeway", "threeway", {"codes": "em", "detector": True, "narrow": True, "share": 0.2}),
        "em_nodet": ("threeway", "threeway", {"codes": "em", "detector": False, "narrow": True, "share": 0.2})}),
}
FOLDER = {"cooldown": "threeway_cooldown", "detect": "rule_change_detect", "threeway": "threeway", "cost": "threeway_cost", "narrow": "threeway_narrow", "em": "threeway_em"}

if __name__ == "__main__":
    study = sys.argv[1]; seeds, arms = STUDIES[study]
    out = f"results_v2/{FOLDER[study]}"; os.makedirs(out, exist_ok=True)
    for name, (mod, arm, kw) in arms.items():
        path = f"{out}/{name}.json"
        if os.path.exists(path): print(name, "exists, skipped", flush=True); continue
        run = d.run if mod == "detect_module" else t.run
        rows = [run(arm, s, **kw) for s in seeds]
        json.dump(rows, open(path, "w"), indent=1); print(study, name, "done", flush=True)
