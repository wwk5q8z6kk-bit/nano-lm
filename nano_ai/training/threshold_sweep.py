"""Sweep the global abstention threshold to produce a risk-coverage curve.

**Why a new module.** `evidence_query_inference.py` is SHA-pinned as
`evidence_query_inference` by `training_source_paths()`, so it cannot be
edited. It already contains everything needed — `apply_global_threshold` and
`_calibration_diagnostics` are both fully threshold-parameterized — they were
simply never called in a loop. This module imports them and does exactly that.

**What it reveals.** `select_global_threshold` implements
`minimal_zero_wrong_presented_inclusive_v1`, whose entire policy is one line:

    threshold = max(wrong_confidences, default=0.0)

It takes the confidence of the worst presented-and-wrong field, applies it
once, asserts risk reached zero, and returns — evaluating exactly two points of
a curve it never plots, and never naming the coverage it spent. On H6's own
calibration data that choice discarded 2,470 of 3,060 presentations (80.7%) and
2,321 correct answers to remove 149 errors.

**Governance.** The development partition is fenced
(`used_for_threshold_selection: false`), so this runs on the *calibration*
partition, where threshold selection is already licensed. A curve computed on
development is diagnostic-only and selects nothing.
"""

from __future__ import annotations

from typing import Sequence

from nano_ai.selective import OperatingPoint
from nano_ai.training.evidence_query_inference import (
    EvidenceQueryInferenceResult,
    _calibration_diagnostics,
    apply_global_threshold,
)


def candidate_thresholds(
    inference: EvidenceQueryInferenceResult, *, limit: int | None = None
) -> list[float]:
    """Every distinct field confidence, ascending.

    These are the only values at which the presented set can change, so they
    are the complete set of interesting operating points; anything between two
    adjacent candidates yields an identical partition.
    """

    values = {
        float(confidence)
        for row in inference.field_joint_confidences
        for confidence in row
    }
    values.add(0.0)
    ordered = sorted(values)
    if limit is not None and limit > 0 and len(ordered) > limit:
        # Uniform subsample preserving both endpoints.
        step = (len(ordered) - 1) / (limit - 1)
        picked = {ordered[min(len(ordered) - 1, round(i * step))] for i in range(limit)}
        ordered = sorted(picked)
    return ordered


def _point_from_diagnostics(
    diagnostics: dict, *, attempts: int, threshold: float, label: str
) -> OperatingPoint:
    wrong = diagnostics["wrong_presented"]
    return OperatingPoint(
        attempts=attempts,
        presented=int(wrong["denominator"]),
        wrong_presented=int(wrong["numerator"]),
        threshold=threshold,
        label=label,
        per_state={
            name: dict(values)
            for name, values in diagnostics.get("slices", {}).items()
            if isinstance(values, dict)
        },
    )


def sweep_thresholds(
    inference: EvidenceQueryInferenceResult,
    gold: Sequence[object],
    *,
    limit: int | None = 200,
) -> list[OperatingPoint]:
    """Return the risk-coverage curve over all distinct confidence thresholds.

    `attempts` is the total field count — the denominator the model cannot
    shrink — so coverage is comparable across every operating point. Note that
    `_calibration_diagnostics` denominates `wrong_presented` in *presented*,
    which is exactly the quantity a threshold controls; that is why coverage
    has to be recomputed against attempts here rather than read off directly.
    """

    frozen_gold = tuple(gold)
    attempts = sum(len(row.proposals) for row in frozen_gold)
    if attempts <= 0:
        raise ValueError("calibration gold contains no fields")

    points: list[OperatingPoint] = []
    for threshold in candidate_thresholds(inference, limit=limit):
        predictions = (
            inference.predictions
            if threshold == 0.0
            else apply_global_threshold(inference, threshold)
        )
        diagnostics = _calibration_diagnostics(predictions, frozen_gold)
        points.append(
            _point_from_diagnostics(
                diagnostics,
                attempts=attempts,
                threshold=threshold,
                label=f"t={threshold:.6f}",
            )
        )
    return points
