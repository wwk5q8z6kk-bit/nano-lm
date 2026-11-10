"""Unit pins for the "needed" axis. No world, no ledger, no model.

The ranker consumes plain dicts on purpose: if these tests had to build a
synthetic fleet to check that a request without a reason is rejected, the
guard would not get run.
"""

from __future__ import annotations

import pytest

from nano.needs import (
    DEFAULT_STRATEGY,
    InformationRequest,
    NeedKind,
    Strategy,
    explain_plan,
    rank_needs,
)


def _state(**overrides):
    base = {
        "conflicted": {("u1", "status"): ("2026-01-10", ("up", "down"))},
        "gaps": {("u2", "load")},
        "resolved": {("u2", "load"): "0.4", ("u3", "wear"): "0.1"},
        "latest_time": {("u1", "status"): "2026-01-10",
                        ("u2", "load"): "2026-01-01",
                        ("u3", "wear"): "2026-01-20"},
        "evidence_count": {("u1", "status"): 2, ("u2", "load"): 1,
                           ("u3", "wear"): 1},
        "now": "2026-01-22",
    }
    base.update(overrides)
    return base


def test_a_request_without_a_reason_is_rejected():
    with pytest.raises(ValueError, match="without a reason"):
        InformationRequest(
            subject="u1", attribute="status", kind=NeedKind.GAP,
            reason="", would_resolve=("confirm u1.status",))


def test_a_request_that_resolves_nothing_is_rejected():
    with pytest.raises(ValueError, match="names nothing"):
        InformationRequest(
            subject="u1", attribute="status", kind=NeedKind.GAP,
            reason="missing", would_resolve=())


def test_ranking_is_deterministic():
    a = [r.key for r in rank_needs(**_state())]
    b = [r.key for r in rank_needs(**_state())]
    assert a == b and a


def test_contradiction_outranks_a_gap_under_kind():
    ranked = rank_needs(**_state(), strategy=Strategy.KIND)
    kinds = [r.kind for r in ranked]
    assert kinds[0] is NeedKind.CONTRADICTION
    assert NeedKind.GAP in kinds


def test_arbitrary_control_does_not_prefer_contradictions():
    """If ARBITRARY ranked by cause it would not be a control."""
    ranked = rank_needs(**_state(), strategy=Strategy.ARBITRARY)
    # All priorities are 0, so order is lexicographic on the key.
    keys = [r.key for r in ranked]
    assert keys == sorted(keys)


def test_default_strategy_is_the_simplest_that_beat_the_control():
    assert DEFAULT_STRATEGY is Strategy.KIND


def test_explain_plan_names_what_it_defers():
    requests = rank_needs(**_state())
    plan = explain_plan(requests, budget=1)
    assert plan["selected"] == 1
    assert plan["deferred"] == len(requests) - 1
    assert plan["deferred_mix"]
