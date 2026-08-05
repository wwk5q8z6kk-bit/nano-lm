"""A4: the threshold sweep that `minimal_zero_wrong_presented_inclusive_v1` never does.

The frozen policy is one line -- `threshold = max(wrong_confidences)` -- so it
evaluates two points of a curve it never plots and never names the coverage it
spent. These tests pin the sweep's contract on synthetic data whose curve is
verifiable by hand; the real curve is produced by the runner on the calibration
partition.
"""

from __future__ import annotations

import pytest

from nano_ai.selective import OperatingPoint, aurc, coverage_at_zero_risk, coverage_cost_of


def _point(threshold: float, presented: int, wrong: int) -> OperatingPoint:
    return OperatingPoint(
        attempts=100, presented=presented, wrong_presented=wrong, threshold=threshold
    )


def test_candidate_thresholds_are_the_only_interesting_points():
    """Between two adjacent confidences the presented set cannot change."""
    from nano_ai.training.threshold_sweep import candidate_thresholds

    class _Inference:
        field_joint_confidences = ((0.9, 0.5), (0.5, 0.2))

    values = candidate_thresholds(_Inference())
    assert values == [0.0, 0.2, 0.5, 0.9]


def test_candidate_thresholds_subsample_preserves_endpoints():
    from nano_ai.training.threshold_sweep import candidate_thresholds

    class _Inference:
        field_joint_confidences = (tuple(i / 1000 for i in range(1, 1000)),)

    values = candidate_thresholds(_Inference(), limit=10)
    assert len(values) <= 10
    assert values[0] == 0.0 and values[-1] == pytest.approx(0.999)


def test_curve_exposes_the_tradeoff_the_frozen_policy_hides():
    """A zero-risk point can cost most of the coverage; both must be visible."""
    permissive = _point(0.0, presented=80, wrong=8)
    strict = _point(0.99, presented=15, wrong=0)

    assert permissive.selective_risk == pytest.approx(0.1)
    assert strict.selective_risk == 0.0  # what the policy reports
    cost = coverage_cost_of(strict, permissive)
    assert cost["errors_removed"] == 8
    assert cost["correct_given_up"] == 57  # what the policy does not report


def test_aurc_orders_two_curves_by_ranking_quality():
    good = [_point(0.0, 100, 5), _point(0.5, 50, 0)]
    bad = [_point(0.0, 100, 5), _point(0.5, 50, 4)]
    assert aurc(good) < aurc(bad)


def test_zero_risk_selection_ignores_the_mute_operating_point():
    points = [_point(1.0, 0, 0), _point(0.8, 30, 0), _point(0.0, 90, 6)]
    best = coverage_at_zero_risk(points)
    assert best is not None and best.presented == 30
