#!/usr/bin/env python3
"""Merge downloaded Kaggle reval_results into the campaign summary artifact."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign.native30_revalidation import (
    DEFAULT_SUMMARY_PATH,
    import_train_results,
)

MANIFEST = ROOT / "artifacts/campaign/manifests/native30_revalidation_wave1_v1.json"
DEFAULT_OUT = DEFAULT_SUMMARY_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory with <run_id>_train.json and eval JSON from Kaggle /kaggle/working/reval_results",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--surface", default="kaggle_gpu_t4")
    args = parser.parse_args()

    imported, expected = import_train_results(
        args.results_dir,
        args.out,
        surface=args.surface,
        root=ROOT,
    )
    print(f"wrote {args.out} ({imported}/{expected} runs)")
    return 0 if imported == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
