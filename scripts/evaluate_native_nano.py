#!/usr/bin/env python3
"""Evaluate native Nano checkpoint on disjoint dev partition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.native.config import config_for_run, smoke_config, NativeVariant
from nanoscribe.native.evaluate import evaluate_native_model
from nanoscribe.native.model import build_native_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate native Nano")
    parser.add_argument("--run-id", default="smoke_cpu")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--variant", choices=["native_a", "native_b"], default="native_a")
    args = parser.parse_args()

    if args.cpu_smoke or args.run_id == "smoke_cpu":
        variant = NativeVariant.NATIVE_B if args.variant == "native_b" else NativeVariant.NATIVE_A
        cfg = smoke_config(variant=variant)
    else:
        cfg = config_for_run(args.run_id, cpu_smoke=True)

    build = build_native_model(cfg)
    result = evaluate_native_model(build.model, cfg)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
