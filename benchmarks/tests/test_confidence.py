"""Wilson bound tests — including the properties that make the bound worth having.

Run explicitly (``benchmarks`` is not in ``pytest.ini`` testpaths):

    pytest benchmarks/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.confidence import (  # noqa: E402
    Z_95,
    failure_gate_passes,
    runs_needed_for_zero_observed,
    wilson_lower,
    wilson_upper,
)


# --- the point of the whole module ------------------------------------------


def test_zero_observed_violations_is_not_a_low_rate():
    """The misreading the bound exists to prevent.

    Twelve clean runs are consistent with a true violation rate near one in four.
    A gate written against the observed count would pass this; a gate written
    against the bound does not.
    """
    bound = wilson_upper(0, 12)
    assert bound is not None
    assert bound == pytest.approx(0.2425, abs=1e-3)
    assert failure_gate_passes(0, 12, epsilon=0.05) is False


def test_a_strict_epsilon_forces_a_large_sample():
    """Driving the bound below a strict ε is what buys the sample size."""
    assert failure_gate_passes(0, 100, epsilon=0.01) is False
    n_needed = runs_needed_for_zero_observed(0.01)
    assert n_needed > 300
    assert failure_gate_passes(0, n_needed, epsilon=0.01) is True


def test_runs_needed_agrees_with_the_bound_it_claims_to_invert():
    """The closed form is verified against wilson_upper, not trusted as algebra."""
    for epsilon in (0.5, 0.2, 0.1, 0.05, 0.01, 0.001):
        n = runs_needed_for_zero_observed(epsilon)
        assert wilson_upper(0, n) <= epsilon
        assert wilson_upper(0, n - 1) > epsilon  # minimal


# --- bound properties -------------------------------------------------------


def test_bounds_bracket_the_point_estimate():
    for successes, n in ((0, 10), (1, 10), (5, 10), (9, 10), (10, 10), (3, 7)):
        p_hat = successes / n
        lower, upper = wilson_lower(successes, n), wilson_upper(successes, n)
        assert lower is not None and upper is not None
        assert lower <= p_hat <= upper


def test_bounds_stay_inside_the_unit_interval_at_the_extremes():
    """Where Wald degenerates: p̂ = 0 and p̂ = 1."""
    assert wilson_lower(0, 5) == 0.0
    assert 0.0 < wilson_upper(0, 5) < 1.0
    assert wilson_upper(5, 5) == 1.0
    assert 0.0 < wilson_lower(5, 5) < 1.0


def test_the_interval_narrows_as_evidence_accumulates():
    widths = [
        wilson_upper(0, n) - wilson_lower(0, n)
        for n in (5, 10, 50, 100, 1000)
    ]
    assert widths == sorted(widths, reverse=True)


def test_a_wider_z_gives_a_more_conservative_upper_bound():
    one_sided_95 = wilson_upper(1, 50, z=1.645)
    two_sided_95 = wilson_upper(1, 50, z=Z_95)
    assert one_sided_95 < two_sided_95


# --- undefined vs decided ---------------------------------------------------


def test_no_observations_is_undefined_not_zero():
    assert wilson_upper(0, 0) is None
    assert wilson_lower(0, 0) is None
    assert failure_gate_passes(0, 0, epsilon=0.05) is None


def test_insufficient_data_leaves_the_gate_undecided_not_passed():
    """`None` must never be read as a pass — that is the whole hazard."""
    undecided = failure_gate_passes(0, 5, epsilon=0.5, n_min=30)
    assert undecided is None
    assert undecided is not True
    assert failure_gate_passes(0, 30, epsilon=0.5, n_min=30) is True


def test_data_sufficiency_is_independent_of_epsilon():
    """A generous ε must not let a tiny sample through."""
    assert failure_gate_passes(0, 2, epsilon=0.99) is True
    assert failure_gate_passes(0, 2, epsilon=0.99, n_min=30) is None


# --- input validation -------------------------------------------------------


@pytest.mark.parametrize(
    "successes,n",
    [(-1, 10), (5, -1), (11, 10)],
)
def test_impossible_counts_raise(successes, n):
    with pytest.raises(ValueError):
        wilson_upper(successes, n)


@pytest.mark.parametrize("epsilon", [0.0, 1.0, -0.1, 1.5])
def test_epsilon_outside_the_open_unit_interval_raises(epsilon):
    with pytest.raises(ValueError):
        runs_needed_for_zero_observed(epsilon)


# --- integration with the metric modules ------------------------------------


def test_aggregate_rate_output_feeds_the_gate_directly():
    """`aggregate_rate` returns (successes, n) precisely so this composes."""
    from benchmarks.gold_metrics import aggregate_rate

    escapes = aggregate_rate([False, False, False, True, False])
    assert escapes == (1, 5)
    successes, n = escapes
    assert failure_gate_passes(successes, n, epsilon=0.05) is False
    assert wilson_upper(successes, n) > 1 / 5  # bound exceeds the point estimate


def test_all_undefined_runs_produce_no_gate_decision():
    from benchmarks.gold_metrics import aggregate_rate

    assert aggregate_rate([None, None]) is None  # nothing to pass to the gate
