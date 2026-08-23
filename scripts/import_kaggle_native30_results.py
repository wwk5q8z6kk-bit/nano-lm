#!/usr/bin/env python3
"""Merge downloaded Kaggle reval_results into the campaign summary artifact."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/campaign/manifests/native30_revalidation_wave1_v1.json"
DEFAULT_OUT = ROOT / "artifacts/campaign/native30_revalidation_summary_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory with <run_id>_train.json and eval JSON from Kaggle /kaggle/working/reval_results",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    run_ids = [r["run_id"] for r in manifest["runs"]]
    results_dir = args.results_dir

    runs: list[dict] = []
    missing: list[str] = []
    for run_id in run_ids:
        train_path = results_dir / f"{run_id}_train.json"
        if not train_path.is_file():
            missing.append(run_id)
            continue
        train = json.loads(train_path.read_text())
        eval_constrained = results_dir / f"{run_id}_constrained.json"
        eval_unconstrained = results_dir / f"{run_id}_unconstrained.json"
        runs.append(
            {
                "run_id": run_id,
                "train": train,
                "eval_constrained": json.loads(eval_constrained.read_text()) if eval_constrained.is_file() else None,
                "eval_unconstrained": json.loads(eval_unconstrained.read_text()) if eval_unconstrained.is_file() else None,
            }
        )

    complete = len(runs) == len(run_ids) and not missing
    summary = {
        "schema": "nano.campaign.native30_revalidation.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment_id": manifest["experiment_id"],
        "corpus": manifest["dataset"],
        "surface": "kaggle_gpu_t4",
        "import_script": "scripts/import_kaggle_native30_results.py",
        "results_dir": str(results_dir),
        "runs_expected": len(run_ids),
        "runs_imported": len(runs),
        "runs_missing": missing,
        "marker": "NATIVE30_REVALIDATION_DONE" if complete else "INCOMPLETE",
        "verdict": "COMPLETE" if complete else "PARTIAL",
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.out} ({len(runs)}/{len(run_ids)} runs)")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
