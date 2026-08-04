#!/usr/bin/env python3
"""Stage 4 — E4 kill gate on frozen R★ under U_R★."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from trajectory.e4.gref_infer import make_gref_predict  # noqa: E402
from trajectory.e4.methods import COST_C, predict_c_m1, predict_c_m2, predict_c_m4  # noqa: E402
from trajectory.e4.utility import RECIPE, decide, evaluate_method  # noqa: E402

E4 = Path(__file__).resolve().parent
DATA = E4 / "data"
TRAJ = REPO / "trajectory"
M_ASSIGN = RECIPE["M_rubric_preassigned"]


def main():
    # Preconditions
    probe = json.loads((TRAJ / "results_e4_classical_probe.json").read_text())
    if not probe.get("in_Rstar"):
        raise SystemExit("VOID: classical probe in_Rstar=false")
    world = json.loads((DATA / "rstar_world_manifest.json").read_text())
    if not world.get("world_frozen"):
        raise SystemExit("VOID: world not frozen")

    eval_items = json.loads((DATA / "rstar_eval.json").read_text())
    print(f"eval n={len(eval_items)} venue_gref loading...", flush=True)
    gref = make_gref_predict()

    methods = {
        "C-M1": (predict_c_m1, COST_C["C-M1"], M_ASSIGN["C-M1"]),
        "C-M2": (predict_c_m2, COST_C["C-M2"], M_ASSIGN["C-M2"]),
        "C-M4": (predict_c_m4, COST_C["C-M4"], M_ASSIGN["C-M4"]),
        "G-ref": (gref, COST_C["G-ref"], M_ASSIGN["G-ref|verify-on"]),
    }

    # C: pre-assigned relative compute indices (E1-continuity O(1) scale), frozen in
    # methods.COST_C before scoring. Raw wall-clock ratios are diagnostics only —
    # using raw ratio (~1e4) as C would silently change U semantics vs E1 κ=0.05.
    sample = eval_items[:20]
    def mean_lat(fn):
        ts = []
        for i, it in enumerate(sample):
            t0 = time.perf_counter()
            fn(it, f"warm/{i}")
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2]

    lat_cm1 = mean_lat(predict_c_m1)
    lat_diag = {
        "C-M1": lat_cm1,
        "C-M2": mean_lat(predict_c_m2),
        "C-M4": mean_lat(predict_c_m4),
        "G-ref": mean_lat(gref),
    }
    cost_runtime = {k: COST_C[k] for k in ("C-M1", "C-M2", "C-M4", "G-ref")}
    print("cost_C_frozen", cost_runtime, flush=True)
    print("latency_p50_diag", {k: round(v, 6) for k, v in lat_diag.items()}, flush=True)

    rows = []
    primary = {}
    for name, (fn, _c_design, M) in methods.items():
        # For G-ref verify-off use M=0.75; verify-on use 0.80
        for verify_on in (False, True):
            M_use = M_ASSIGN["G-ref|verify-on"] if (name == "G-ref" and verify_on) else (
                M_ASSIGN["G-ref"] if name == "G-ref" else M_ASSIGN[name]
            )
            c_use = cost_runtime[name]
            key = f"{name}|verify={'on' if verify_on else 'off'}"
            print(f"eval {key}...", flush=True)
            res = evaluate_method(name, fn, eval_items, verify_on=verify_on, cost_c=c_use, M=M_use)
            out = {k: v for k, v in res.items() if k != "item_logs"}
            out["prereg"] = "trajectory/PREREG_E4_Rstar_killgate.md"
            out["execution_anchor"] = "verdict/E4-kill@6af178d"
            out["hardware"] = {
                "class": "apple-mps",
                "gref_venue": getattr(gref, "venue", "mps"),
                "C_measurement": "preassigned_relative_vs_CM1_E1_scale",
                "cost_C": cost_runtime,
                "latency_p50_diag": lat_diag,
            }
            path = TRAJ / f"results_e4_{name.replace('-', '').lower()}_v{'on' if verify_on else 'off'}.json"
            # normalize filenames
            fname = {
                "C-M1": "cm1",
                "C-M2": "cm2",
                "C-M4": "cm4",
                "G-ref": "gref",
            }[name]
            path = TRAJ / f"results_e4_{fname}_v{'on' if verify_on else 'off'}.json"
            path.write_text(json.dumps(out, indent=2) + "\n")
            (TRAJ / f"results_e4_items_{fname}_v{'on' if verify_on else 'off'}.json").write_text(
                json.dumps(res["item_logs"])
            )
            s = res["summary"]
            print(
                f"  U={s['U']:.4f} Q={s['Q']:.3f} E={s['E']:.3f} R={s['R']:.3f} "
                f"L={s['L_p50']:.4f} C={s['C']:.2f} M={s['M']:.2f} rec={s['recall']:.3f}",
                flush=True,
            )
            rows.append({"method": name, "verify_on": verify_on, **{k: s[k] for k in s if k != "U_sensitivity"}, "U_sensitivity": s["U_sensitivity"]})
            if verify_on:
                primary[name] = res

    decision = decide(primary)
    util = {
        "schema": "nano-lm.e4.utility.v1",
        "prereg": "trajectory/PREREG_E4_Rstar_killgate.md",
        "execution_anchor": "verdict/E4-kill@6af178d",
        "recipe_freeze": "trajectory/e4/recipe_freeze.json",
        "world_manifest": "trajectory/e4/data/rstar_world_manifest.json",
        "classical_probe": "trajectory/results_e4_classical_probe.json",
        "U_definition": RECIPE["U_Rstar"],
        "g_ref_recipe": RECIPE["g_ref_recipe"],
        "g_ref_meta": json.loads((E4 / "checkpoints" / "gref_nano_rstar_sft_v1.meta.json").read_text()),
        "hardware": {"class": "apple-mps", "cost_C": cost_runtime, "latency_p50_diag": {k: lat_diag[k] for k in lat_diag}},
        "rows": rows,
        "decision": decision,
        "consequences_table": "trajectory/PREREG_E4_Rstar_killgate.md §3",
        "not_nanoscribe": True,
        "not_old_task": True,
        "e3_not_human": True,
    }
    util_path = TRAJ / "results_e4_utility.json"
    util_path.write_text(json.dumps(util, indent=2) + "\n")
    print(json.dumps(decision, indent=2), flush=True)
    print(f"wrote {util_path}", flush=True)


if __name__ == "__main__":
    main()
