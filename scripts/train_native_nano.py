#!/usr/bin/env python3
"""Train native Nano model — CPU smoke or RunPod launch gate."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.native.config import config_for_run, smoke_config, NativeVariant
from nanoscribe.native.data import export_distill_train_json
from nanoscribe.native.train import cpu_smoke_train, train_native, train_run_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Train native Nano")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--variant", choices=["native_a", "native_b"], default="native_a")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-train-json", action="store_true")
    parser.add_argument("--dataset", default="artifacts/campaign/p1_distill_train_v1.json")
    parser.add_argument(
        "--purpose",
        default="architecture_screening",
        help="training purpose for corpus launch guard (unit fixture requires trainer_smoke/qlora_canary_fixture)",
    )
    args = parser.parse_args()

    if args.export_train_json:
        print(json.dumps(export_distill_train_json(), indent=2))
        return 0

    if args.cpu_smoke:
        result = cpu_smoke_train(variant=args.variant)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if not args.run_id:
        parser.error("--run-id required unless --cpu-smoke or --export-train-json")

    cfg = config_for_run(args.run_id, cpu_smoke=args.cpu_smoke)
    cfg = replace(cfg, dataset_path=args.dataset, purpose=args.purpose)
    result = train_native(cfg, resume=args.resume)
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
