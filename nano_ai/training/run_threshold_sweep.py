"""Produce the real risk-coverage curve on the calibration partition.

Mirrors `train_evidence_query._calibrate_model` exactly -- same loader, same
encoder, same shared inference authority, same gold construction -- and then
sweeps every candidate threshold instead of selecting the single point
`minimal_zero_wrong_presented_inclusive_v1` picks.

Governance: the calibration partition is where threshold selection is already
licensed (`used_for_threshold_selection` is false only for development). This
reads calibration data and a trained checkpoint; it selects nothing, changes no
frozen artifact, and writes only the curve it computes.

Usage:
    python3 -m nano_ai.training.run_threshold_sweep \
        --checkpoint artifacts/nano_h6/kaggle/results-20260805/results/seed-20260805/epoch-2.pt \
        --calibration artifacts/nano_h5/data/calibration.jsonl \
        --tokenizer sft/tokenizer.json \
        --output artifacts/nano_h6/analysis/threshold_curve_seed20260805_epoch2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nano_ai.selective import aurc, coverage_at_zero_risk, coverage_cost_of
from nano_ai.training.threshold_sweep import sweep_thresholds


def _load(checkpoint: Path, calibration: Path, tokenizer_path: Path, device: str):
    """Rebuild the exact objects `_calibrate_model` uses."""
    from nano_ai.training.evidence_query_inference import CalibrationGold
    from nano_ai.training.evidence_query_model import NanoEvidenceQueryPointerModel
    from nano_ai.training.evaluate_pointer import build_pointer_inference_inputs
    from nano_ai.adapters.state_span import parse_state_span_summary
    from nano_ai.training import replay_mixture_data
    from nano_ai.training.train_pointer import encode_pointer_partition, load_pointer_tokenizer

    # Same loader the H6 trainer uses; `calibration` is the dataset directory.
    bundle = replay_mixture_data.load_replay_mixture_dataset(calibration)
    examples = bundle.calibration
    tokenizer = load_pointer_tokenizer(tokenizer_path)
    records = encode_pointer_partition(tokenizer, examples, expected_split="train")
    inputs = build_pointer_inference_inputs(examples, records)

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    weights = state.get("model", state)
    # H6 checkpoints carry the state-conditioned residual; H3/H5 do not.
    if "state_boundary_query_offsets" in weights:
        from nano_ai.training.state_conditioned_evidence_query_model import (
            NanoStateConditionedEvidenceQueryPointerModel,
        )

        model = NanoStateConditionedEvidenceQueryPointerModel()
    else:
        model = NanoEvidenceQueryPointerModel()
    model.load_state_dict(weights)
    model.to(device).eval()

    gold = tuple(
        CalibrationGold(
            example_id=example.example_id,
            proposals=parse_state_span_summary(example.target, example.transcript),
        )
        for example in examples
    )
    return model, inputs, gold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True, help="dataset DIRECTORY (contains fit/calibration/manifest)")
    parser.add_argument("--tokenizer", type=Path, default=Path("sft/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    from nano_ai.training.evidence_query_inference import (
        batched_evidence_query_inference,
    )

    model, inputs, gold = _load(
        args.checkpoint, args.calibration, args.tokenizer, args.device
    )
    with torch.inference_mode():
        inference = batched_evidence_query_inference(
            model, inputs, device=args.device, batch_size=args.batch_size
        )

    points = sweep_thresholds(inference, gold, limit=args.limit)
    permissive = points[0]
    zero_risk = coverage_at_zero_risk(points)

    payload = {
        "schema": "nano.threshold-risk-coverage-curve.v1",
        "checkpoint": str(args.checkpoint),
        "calibration": str(args.calibration),
        "partition": "calibration (threshold selection licensed here)",
        "selects_nothing": True,
        "attempts": permissive.attempts,
        "points": [point.as_dict() for point in points],
        "aurc": aurc(points) if len(points) > 1 else None,
        "permissive": permissive.as_dict(),
        "best_zero_risk": zero_risk.as_dict() if zero_risk else None,
        "cost_of_zero_risk": (
            coverage_cost_of(zero_risk, permissive) if zero_risk else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"points: {len(points)}   attempts: {permissive.attempts}")
    print(
        f"permissive: coverage {permissive.coverage:.4f} "
        f"risk {permissive.selective_risk:.4f} retained {permissive.retained_correct}"
    )
    if zero_risk:
        cost = payload["cost_of_zero_risk"]
        print(
            f"zero-risk : coverage {zero_risk.coverage:.4f} "
            f"retained {zero_risk.retained_correct} "
            f"(gave up {cost['correct_given_up']} correct to remove "
            f"{cost['errors_removed']} errors)"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
