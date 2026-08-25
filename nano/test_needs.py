"""Tests for the "needed" axis of MTA-EPISTEMIC.

Same two-kind split as `test_slw.py`: property tests assert the ranker behaves
as claimed, manipulation checks assert the *scorer* would notice if it did not.
The headline claim here — "the ranking beats asking arbitrarily" — is the kind
that scores well by accident when the metric is weak, so the checks that try to
break it are doing most of the work.

No test writes to disk and nothing here loads a model.
"""

from __future__ import annotations

import pytest

from nano.needs import (
    DEFAULT_STRATEGY,
    InformationRequest,
    NeedKind,
    Strategy,
    explain_plan,
    priority_of,
    rank_needs,
)
from nano.slw import (
    SyntheticWorld,
    WorldSpec,
    build_full,
    compare_need_strategies,
    epistemic_inputs,
    paired_delta,
    score_information_need,
    tick_to_date,
    _broken_keys,
)

SMALL = WorldSpec(seed=11, n_sites=3, units_per_site=3, components_per_unit=2,
                  n_ticks=20, checkpoint_every=5)


@pytest.fixture(scope="module")
def world():
    return SyntheticWorld.generate(SMALL)


@pytest.fixture(scope="module")
def builder(world):
    return build_full(world, world.observations, world.spec.n_ticks)


@pytest.fixture(scope="module")
def inputs(builder, world):
    return epistemic_inputs(builder, tick_to_date(world.spec.n_ticks))


@pytest.fixture(scope="module")
def scored(world):
    return score_information_need(world)


# ---------------------------------------------------------------------------
# A request has to be actionable to exist
# ---------------------------------------------------------------------------

def test_a_request_without_a_reason_is_refused():
    with pytest.raises(ValueError, match="without a reason"):
        InformationRequest(subject="unit_0000", attribute="status",
                           kind=NeedKind.GAP, reason="",
                           would_resolve=("confirm status",))


def test_a_request_that_resolves_nothing_is_refused():
    with pytest.raises(ValueError, match="never be closed"):
        InformationRequest(subject="unit_0000", attribute="status",
                           kind=NeedKind.GAP, reason="no report arrived",
                           would_resolve=())


def test_every_generated_request_explains_itself(inputs):
    requests = rank_needs(**inputs)
    assert requests
    for r in requests:
        assert r.reason and r.would_resolve
        assert r.kind in NeedKind


# ---------------------------------------------------------------------------
# The plan is a plan, not an inventory
# ---------------------------------------------------------------------------

def test_ranking_is_deterministic(inputs):
    """A plan that reshuffles under an unchanged world cannot be reviewed."""
    first = [r.request_id for r in rank_needs(**inputs)]
    second = [r.request_id for r in rank_needs(**inputs)]
    assert first == second


def test_priority_is_non_increasing_down_the_list(inputs):
    priorities = [r.priority for r in rank_needs(**inputs)]
    assert priorities == sorted(priorities, reverse=True)


def test_contradictions_outrank_staleness(inputs):
    """Ordered by how little the system knows: a contradiction means it holds no
    usable value at all, staleness means it holds one that may have moved."""
    requests = rank_needs(**inputs, strategy=Strategy.KIND)
    by_kind = {}
    for r in requests:
        by_kind.setdefault(r.kind, []).append(r.priority)
    if NeedKind.CONTRADICTION in by_kind and NeedKind.STALE in by_kind:
        assert min(by_kind[NeedKind.CONTRADICTION]) > max(by_kind[NeedKind.STALE])


def test_a_truncated_plan_says_what_it_deferred(inputs):
    """Reporting only what was selected reads as though nothing was left out."""
    requests = rank_needs(**inputs)
    plan = explain_plan(requests, budget=3)
    assert plan["selected"] == 3
    assert plan["deferred"] == len(requests) - 3
    assert plan["deferred_mix"], "a truncated plan reported no deferred work"
    assert sum(plan["selected_mix"].values()) == 3
    for entry in plan["top"]:
        assert entry["why"]


def test_the_ranker_cannot_reach_ground_truth(inputs):
    """Structural guard: `rank_needs` takes plain dicts and sets. If it ever
    grew a world or ledger parameter, this fails and the leak is visible."""
    import inspect
    params = set(inspect.signature(rank_needs).parameters)
    forbidden = {"world", "truth", "ledger", "builder", "snapshot"}
    assert not (params & forbidden), f"ranker can reach {params & forbidden}"
    assert set(inputs) <= params


# ---------------------------------------------------------------------------
# The headline: does ranking beat asking arbitrarily?
# ---------------------------------------------------------------------------

def test_ranking_beats_the_arbitrary_control(scored):
    assert scored["broken_keys"] > 0, "nothing was wrong — the test is vacuous"
    assert scored["ranking_beats_control"]
    assert scored["best_precision"] > scored["control_precision"] * 2


def test_the_arbitrary_control_really_is_signal_free(scored):
    """Manipulation check on the control itself.

    If ARBITRARY accidentally carried signal, every lift measured against it
    would be understated and the comparison would be worthless. Its precision
    should sit near the base rate of broken keys.
    """
    control = scored["strategies"][Strategy.ARBITRARY.value]
    p = control["budgets"][scored["comparison_budget"]]["precision_at_k"]
    base = scored["base_rate"]
    assert abs(p - base) < 0.20, (
        f"control precision {p:.3f} is far from the base rate {base:.3f} — it "
        "is not a no-signal control")


def test_spending_the_budget_actually_removes_error(scored):
    """Precision could look good while fixing nothing. This is the check that
    the loop closes: acquire the selected information and the error must fall."""
    budget = scored["comparison_budget"]
    best = scored["strategies"][scored["best_strategy"]]["budgets"][budget]
    assert best["errors_fixed"] > 0
    assert best["errors_after"] < best["errors_before"]
    assert best["fixed_per_question"] > scored["base_rate"]


def test_an_inverted_ranking_scores_below_the_control(world, builder):
    """Manipulation check with teeth. Asking the *least* useful questions first
    must score worse than asking arbitrarily; if it does not, the metric is not
    measuring ordering at all.
    """
    inputs = epistemic_inputs(builder, tick_to_date(world.spec.n_ticks))
    broken = _broken_keys(world, builder)
    requests = rank_needs(**inputs, strategy=DEFAULT_STRATEGY)
    k = max(1, len(requests) // 4)

    def precision(rs):
        return sum(1 for r in rs[:k] if r.key in broken) / k

    forward = precision(requests)
    inverted = precision(list(reversed(requests)))
    assert inverted < forward, "reversing the plan changed nothing"
    assert inverted < precision(rank_needs(**inputs, strategy=Strategy.ARBITRARY))


# ---------------------------------------------------------------------------
# Complexity has to earn its place, one term at a time
# ---------------------------------------------------------------------------

def test_the_default_strategy_is_the_simplest_justified_one():
    """Pins a measured negative result.

    Scarcity- and age-weighting both sounded obviously useful and neither is
    distinguishable from zero. If a future change makes one of them genuinely
    pay, this fails — deliberately, so the default is moved on purpose and the
    recorded numbers get updated rather than quietly going stale.
    """
    comparison = compare_need_strategies(
        seeds=tuple(range(41, 47)),
        spec_factory=lambda seed: WorldSpec(seed=seed, n_sites=3,
                                            units_per_site=3, n_ticks=20,
                                            checkpoint_every=5))
    assert comparison["steps"]["kind - arbitrary"]["distinguishable"], (
        "cause-based ranking no longer beats the control")
    assert comparison["default_is_justified"], (
        f"default is {comparison['current_default']} but the simplest justified "
        f"strategy is {comparison['simplest_justified_strategy']}")


def test_paired_delta_reports_a_real_shift_as_distinguishable():
    a = [0.90, 0.92, 0.88, 0.91, 0.89, 0.93]
    b = [0.20, 0.22, 0.18, 0.21, 0.19, 0.23]
    result = paired_delta(a, b)
    assert result["distinguishable"]
    assert result["ci_low"] > 0


def test_paired_delta_reports_noise_as_not_distinguishable():
    """The check that stops a refinement being promoted on a mean difference
    that sits inside the noise."""
    a = [0.50, 0.62, 0.44, 0.58, 0.41, 0.61]
    b = [0.52, 0.43, 0.59, 0.40, 0.63, 0.45]
    assert not paired_delta(a, b)["distinguishable"]


def test_paired_delta_refuses_unpaired_or_tiny_series():
    with pytest.raises(ValueError, match="equal-length"):
        paired_delta([0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="equal-length"):
        paired_delta([0.1], [0.1])


def test_the_arbitrary_strategy_assigns_no_priority(inputs):
    for r in rank_needs(**inputs, strategy=Strategy.ARBITRARY):
        assert r.priority == 0.0
    request = InformationRequest(subject="u", attribute="a",
                                 kind=NeedKind.CONTRADICTION, reason="r",
                                 would_resolve=("x",))
    assert priority_of(request, Strategy.ARBITRARY) == 0.0
    assert priority_of(request, Strategy.KIND) > 0.0


# ---------------------------------------------------------------------------
# Age handling
# ---------------------------------------------------------------------------

def test_month_precision_age_is_measured_from_the_oldest_it_could_be():
    """Underestimating staleness leaves a stale value unquestioned, so a
    month-precision timestamp is aged from the first of that month."""
    from nano.needs import _days_between
    assert _days_between("2026-01", "2026-02-01") == 31
    assert _days_between("2026-01-31", "2026-02-01") == 1
    assert _days_between("2026-03-01", "2026-02-01") == 0     # never negative
    assert _days_between("not-a-date", "2026-02-01") == 0
