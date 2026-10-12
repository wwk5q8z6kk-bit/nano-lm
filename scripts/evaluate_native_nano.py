#!/usr/bin/env python3
"""Evaluate native Nano checkpoint — distill dev loss or P1 structured suites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign_datasets import SMOKE_SUITE_REVISION
from nanoscribe.native.config import smoke_config, NativeVariant
from nanoscribe.native.evaluate import evaluate_native_model
from nanoscribe.native.model import build_native_model
from nanoscribe.native.p1_eval import evaluate_native_p1_suite, load_native_model_for_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate native Nano")
    parser.add_argument("--run-id", default="smoke_cpu")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--variant", choices=["native_a", "native_b"], default="native_a")
    parser.add_argument("--step", type=int, default=None, help="checkpoint step (default latest.pt)")
    parser.add_argument("--suite", default="", help=f"P1 suite e.g. {SMOKE_SUITE_REVISION}")
    parser.add_argument("--cpu", action="store_true", help="force CPU eval")
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="free autoregressive generation instead of candidate selection; "
        "required for exact_gold_span to measure evidence transport rather than "
        "candidate-set construction",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.cpu_smoke or (args.run_id == "smoke_cpu" and not args.suite):
        variant = NativeVariant.NATIVE_B if args.variant == "native_b" else NativeVariant.NATIVE_A
        cfg = smoke_config(variant=variant)
        build = build_native_model(cfg)
        result = evaluate_native_model(build.model, cfg)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if not args.suite:
        parser.error("--suite required for checkpoint eval (e.g. p1_contract_smoke_v1)")

    model, cfg, ckpt_path = load_native_model_for_eval(
        args.run_id,
        step=args.step,
        cpu=args.cpu,
    )
    result = evaluate_native_p1_suite(
        model,
        cfg,
        suite=args.suite,
        checkpoint_path=ckpt_path,
        constrained=not args.unconstrained,
    )
    payload = result.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
