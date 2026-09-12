#!/usr/bin/env python3
"""Run Wave-1 native30 revalidation locally (CUDA / Apple MPS / CPU).

End-to-end A→Z: corpus gate → arm preflight → interleaved training → dual-mode eval
→ Wilson-interval analysis. No RunPod or Kaggle required on Apple Silicon.

Examples:
    # Full wave (~3h on MPS for 9 arms @ 1800 steps):
    python3 scripts/run_native30_revalidation.py

    # CI / pipeline smoke (one arm, 30 steps):
    python3 scripts/run_native30_revalidation.py --smoke

    # Resume after interruption (skips runs with *_train.json markers):
    python3 scripts/run_native30_revalidation.py --results-dir artifacts/campaign/reval_results

    # Import + analyze only:
    python3 scripts/run_native30_revalidation.py --import-only --analyze
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign.native30_revalidation import (
    DEFAULT_RESULTS_DIR,
    DEFAULT_SUMMARY_PATH,
    EVAL_SUITE,
    MARKER_DONE,
    interleaved_run_ids,
    import_train_results,
    analyze_and_write_summary,
    detect_surface_label,
    run_revalidation_wave,
    verify_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Per-run train/eval JSON output directory",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--smoke", action="store_true", help="One arm, 30 steps, smoke eval suite")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--suite", default=EVAL_SUITE)
    parser.add_argument("--run-id", action="append", dest="run_ids", help="Subset of run IDs")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--import-only", action="store_true")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="After import, run analyze_revalidation.py (Wilson + verdicts)",
    )
    parser.add_argument(
        "--eval-on-device",
        action="store_true",
        help="Eval on training device instead of CPU (default: CPU eval)",
    )
    parser.add_argument("--verify-corpus-only", action="store_true")
    args = parser.parse_args()

    if args.verify_corpus_only:
        info = verify_corpus()
        print(f"CORPUS_OK hash={info['content_hash'][:16]} rows={info['train_rows']}")
        return 0

    if args.smoke:
        run_ids = ("reval30_decoder_control_s0",)
        max_steps = args.max_steps or 30
        suite = "p1_contract_smoke_v1"
    else:
        run_ids = tuple(args.run_ids or interleaved_run_ids())
        max_steps = args.max_steps
        suite = args.suite

    surface = detect_surface_label()
    print(f"surface={surface} runs={len(run_ids)} results={args.results_dir}", flush=True)

    if args.import_only:
        imported, expected = import_train_results(
            args.results_dir, args.out, surface=surface, root=ROOT
        )
        print(f"imported {imported}/{expected} -> {args.out}")
        if args.analyze:
            return analyze_and_write_summary(args.results_dir, args.out, root=ROOT)
        return 0 if imported == expected else 1

    failed, skipped = run_revalidation_wave(
        args.results_dir,
        run_ids,
        max_steps=max_steps,
        suite=suite,
        skip_train=args.skip_train,
        skip_eval=args.skip_eval,
        eval_cpu=not args.eval_on_device,
        cpu_smoke=args.smoke and max_steps is not None and max_steps <= 30,
        root=ROOT,
    )

    if skipped:
        print(f"skipped (already complete): {skipped}", flush=True)
    if failed:
        print(f"RUNS_FAILED: {failed}", flush=True)

    imported, expected = import_train_results(
        args.results_dir, args.out, surface=surface, root=ROOT
    )
    print(f"imported {imported}/{expected} -> {args.out}", flush=True)

    rc = 0
    if args.analyze and imported > 0:
        rc = analyze_and_write_summary(args.results_dir, args.out, root=ROOT)

    if failed:
        return 1
    if imported == expected:
        print(MARKER_DONE, flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
