"""Tests for risk-coverage analysis of ternary selective prediction.

The load-bearing test is `test_mute_system_is_degenerate_and_fails_the_gate`:
it pins the defect this module exists to prevent -- a system that presents
nothing scores zero selective risk and must NOT pass an admission gate.
"""

from __future__ import annotations

import pytest

from nano_ai.selective import (
    OperatingPoint,
    SelectiveAnalysisError,
    aurc,
    coverage_at_zero_risk,
    coverage_cost_of,
    gate_is_degeneracy_safe,
    risk_coverage_curve,
)


def test_basic_quantities():
    p = OperatingPoint(attempts=1000, presented=800, wrong_presented=40)
    assert p.coverage == 0.8
    assert p.selective_risk == 0.05
    assert p.retained_correct == 760
    assert not p.is_degenerate


def test_mute_system_is_degenerate_and_fails_the_gate():
    """The defect: presenting nothing yields zero risk and must not pass.

    This is `fabric/slice.py:248`'s failure mode -- `presented_err / max(1,
    presented)` is 0.0 at zero coverage, which compares favourably against any
    raw error rate.
    """
    mute = OperatingPoint(attempts=1000, presented=0, wrong_presented=0)
    assert mute.selective_risk == 0.0  # vacuously perfect
    assert mute.is_degenerate
    ok, reason = gate_is_degeneracy_safe(mute, coverage_floor=0.5)
    assert ok is False
    assert "degenerate" in reason


def test_gate_enforces_coverage_floor_denominated_in_attempts():
    thin = OperatingPoint(attempts=1000, presented=100, wrong_presented=0)
    ok, reason = gate_is_degeneracy_safe(thin, coverage_floor=0.5)
    assert ok is False and "below floor" in reason

    healthy = OperatingPoint(attempts=1000, presented=900, wrong_presented=9)
    ok, _ = gate_is_degeneracy_safe(healthy, coverage_floor=0.5)
    assert ok is True


def test_abstention_ledger_must_close():
    with pytest.raises(SelectiveAnalysisError, match="ledger does not close"):
        OperatingPoint(
            attempts=100,
            presented=60,
            wrong_presented=0,
            abstention_benefit=10,
            abstention_cost=10,  # 20 != 40 withheld
        )


def test_fabric_ledger_closes_exactly():
    """Fabric's real numbers: every withheld field was an error."""
    p = OperatingPoint(
        attempts=935,
        presented=762,
        wrong_presented=0,
        abstention_benefit=173,
        abstention_cost=0,
        label="fabric nano v2 m1",
    )
    assert p.abstention_cost == 0
    assert round(p.coverage, 3) == 0.815


def test_coverage_cost_of_names_what_a_threshold_paid():
    """H6's calibration collapse, in canonical terms."""
    uncal = OperatingPoint(attempts=4000, presented=3060, wrong_presented=149)
    cal = OperatingPoint(attempts=4000, presented=590, wrong_presented=0)
    cost = coverage_cost_of(cal, uncal)
    assert cost["errors_removed"] == 149
    assert cost["correct_given_up"] == 2321
    assert round(cost["presented_given_up_fraction"], 3) == 0.807


def test_aurc_requires_a_curve_not_a_point():
    single = [OperatingPoint(attempts=100, presented=80, wrong_presented=4)]
    with pytest.raises(SelectiveAnalysisError, match="needs >= 2"):
        aurc(single)


def test_aurc_rewards_better_ranking():
    """A score that ranks errors last has lower AURC at matched coverage."""
    good = [
        OperatingPoint(attempts=100, presented=50, wrong_presented=0),
        OperatingPoint(attempts=100, presented=100, wrong_presented=10),
    ]
    bad = [
        OperatingPoint(attempts=100, presented=50, wrong_presented=5),
        OperatingPoint(attempts=100, presented=100, wrong_presented=10),
    ]
    assert aurc(good) < aurc(bad)


def test_coverage_at_zero_risk_excludes_the_mute_system():
    points = [
        OperatingPoint(attempts=100, presented=0, wrong_presented=0),
        OperatingPoint(attempts=100, presented=40, wrong_presented=0),
        OperatingPoint(attempts=100, presented=90, wrong_presented=3),
    ]
    best = coverage_at_zero_risk(points)
    assert best is not None and best.presented == 40


def test_curve_rejects_mismatched_attempt_counts():
    with pytest.raises(SelectiveAnalysisError, match="different attempt counts"):
        risk_coverage_curve(
            [
                OperatingPoint(attempts=100, presented=50, wrong_presented=1),
                OperatingPoint(attempts=200, presented=50, wrong_presented=1),
            ]
        )
