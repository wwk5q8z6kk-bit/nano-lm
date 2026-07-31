#!/usr/bin/env python3
"""Execute E1 kill-gate: M1–M5 (+ local M0) on m0–m4, write frozen result JSONs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO))

from trajectory.e1.common import (  # noqa: E402
    aggregate_decision,
    evaluate_method,
    load_instances,
)
from trajectory.e1 import methods as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-m0", action="store_true", help="Skip local LM M0 (faster)")
    ap.add_argument("--m0-tag", default="scale", choices=["nano", "scale"])
    ap.add_argument("--skip-train-heavy", action="store_true", help="Skip M3/M5 training methods")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=REPO / "trajectory")
    args = ap.parse_args()

    instances = load_instances()
    print(f"loaded {len(instances)} instances × {len(instances[0])} items", flush=True)

    if not args.skip_train_heavy:
        print(f"training M3/M5 on n_train={args.n_train}...", flush=True)
        M.train_m3(n_train=args.n_train)
        M.train_m5(n_train=args.n_train)
        print("train done", flush=True)

    method_fns = {
        "M1_template": M.predict_m1,
        "M2_dict_span": M.predict_m2,
        "M4_constrained": M.predict_m4,
    }
    if not args.skip_train_heavy:
        method_fns["M3_crf_lite"] = M.predict_m3
        method_fns["M5_span_clf"] = M.predict_m5

    if not args.skip_m0:
        print(f"loading M0 tag={args.m0_tag}...", flush=True)
        method_fns[f"M0_{args.m0_tag}"] = M.make_m0_predict(args.m0_tag)

    results = {}
    utility_rows = []
    for name, fn in method_fns.items():
        cost = M.COST_C.get(name, 0.1)
        for verify_on in (False, True):
            key = f"{name}|verify={'on' if verify_on else 'off'}"
            print(f"eval {key}...", flush=True)
            res = evaluate_method(name, fn, instances, verify_on=verify_on, cost_c=cost)
            # strip bulky logs from per-method file optional — keep in separate
            out = {k: v for k, v in res.items() if k != "item_logs"}
            out["prereg"] = "trajectory/PREREG_E1_nonlm_baseline.md"
            out["m0_note"] = (
                "Official M0 is max U(Pythia-160M LoRA, own-3.2B+LoRA). "
                f"Local provisional M0={args.m0_tag} when present."
            )
            path = args.out_dir / f"results_e1_nonlm_{name}_v{'on' if verify_on else 'off'}.json"
            # store item logs separately compressed-ish
            path.write_text(json.dumps(out, indent=2))
            log_path = args.out_dir / f"results_e1_items_{name}_v{'on' if verify_on else 'off'}.json"
            log_path.write_text(json.dumps(res["item_logs"]))
            print(
                f"  U={res['summary']['U']:.4f} P={res['summary']['P']:.3f} "
                f"recall={res['summary']['recall']:.3f} rho={res['summary']['rho']:.3f} "
                f"gap={res['summary']['gap_pts']:.2f}",
                flush=True,
            )
            if verify_on:
                results[name] = res
            utility_rows.append({
                "method": name,
                "verify_on": verify_on,
                **{k: res["summary"][k] for k in (
                    "U", "P", "M", "rho", "L_p50", "C", "recall", "halluc",
                    "gap_pts", "held_recall", "seen_recall", "correct_norm_rate",
                    "liability_presented_bad",
                )},
                "U_sensitivity": res["summary"]["U_sensitivity"],
            })

    m0_name = f"M0_{args.m0_tag}" if f"M0_{args.m0_tag}" in results else None
    decision = (
        aggregate_decision(results, m0_name)
        if m0_name
        else {"verdict": "INCOMPLETE", "reason": "M0 not run", "U": {k: v['summary']['U'] for k, v in results.items()}}
    )
    util = {
        "prereg": "trajectory/PREREG_E1_nonlm_baseline.md",
        "instrument": "trajectory/scribe_eval_m{0..4}.json",
        "weights_default": {"alpha": 1.0, "beta": 0.5, "gamma": 0.3, "lam": 0.02, "kappa": 0.05},
        "rows": utility_rows,
        "decision": decision,
        "note": (
            "KILL/SURVIVE vs local M0 is provisional until Pythia-160M LoRA / "
            "own-3.2B+LoRA corner U is measured on this harness."
        ),
    }
    util_path = args.out_dir / "results_e1_utility.json"
    util_path.write_text(json.dumps(util, indent=2))
    print(json.dumps(decision, indent=2), flush=True)
    print(f"wrote {util_path}", flush=True)


if __name__ == "__main__":
    main()
