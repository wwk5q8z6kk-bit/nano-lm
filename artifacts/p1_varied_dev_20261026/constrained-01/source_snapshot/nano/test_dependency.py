"""Pins for dependency lineage and invalidation (Layer XIII)."""
from __future__ import annotations

import pytest

from nano.dependency import Dependency, DependencyGraph, Freshness


def _chain() -> DependencyGraph:
    """E1 -> C1 -> E2 -> S1 -> R1, with an independent branch E9 -> C9."""
    g = DependencyGraph()
    g.register(Dependency("C1", ("E1",), kind="claim<-evidence"))
    g.register(Dependency("E2", ("C1",), kind="event<-claim"))
    g.register(Dependency("S1", ("E2",), kind="state<-event"))
    g.register(Dependency("R1", ("S1",), kind="artifact<-state"))
    g.register(Dependency("C9", ("E9",), kind="claim<-evidence"))
    return g


def test_correction_marks_the_whole_downstream_chain() -> None:
    g = _chain()
    r = g.invalidate("E1", reason="corrected start date")
    assert set(r["direct"]) == {"C1"}
    assert set(r["transitive"]) == {"E2", "S1", "R1"}
    assert set(g.stale()) == {"E1", "C1", "E2", "S1", "R1"}


def test_invalidation_is_precise_not_blanket() -> None:
    """Over-invalidation is safe but useless; it destroys incrementality."""
    g = _chain()
    r = g.invalidate("E1", reason="corrected")
    assert "C9" in r["unaffected"]
    assert g.freshness["C9"] is Freshness.CURRENT
    assert "E9" in g.current()


def test_direct_and_transitive_staleness_are_distinguished() -> None:
    """Claiming certainty about deep descendants would be over-invalidation."""
    g = _chain()
    g.invalidate("E1", reason="corrected")
    assert g.freshness["C1"] is Freshness.STALE
    assert g.freshness["S1"] is Freshness.POSSIBLY_STALE


def test_invalidation_without_a_reason_is_rejected() -> None:
    g = _chain()
    with pytest.raises(ValueError, match="requires a reason"):
        g.invalidate("E1", reason="")


def test_every_stale_object_can_explain_itself() -> None:
    g = _chain()
    g.invalidate("E1", reason="corrected start date")
    for oid in g.stale():
        e = g.explain(oid)
        assert e["reason"], f"{oid} is stale with no recorded reason"


def test_supersession_is_recorded_distinctly_from_staleness() -> None:
    g = _chain()
    g.invalidate("E1", reason="replaced by E1prime", superseded=True)
    assert g.freshness["E1"] is Freshness.SUPERSEDED
    assert g.freshness["C1"] is Freshness.STALE


def test_recomputation_produces_a_new_id_rather_than_rewriting() -> None:
    g = _chain()
    with pytest.raises(ValueError, match="already has lineage"):
        g.register(Dependency("C1", ("E1prime",)))


def test_derived_object_needs_inputs() -> None:
    with pytest.raises(ValueError, match="is not derived"):
        Dependency("X", ())


def test_cycles_are_rejected() -> None:
    """Lineage with a cycle is not a derivation order and cannot be replayed."""
    g = DependencyGraph()
    g.register(Dependency("B", ("A",)))
    g.register(Dependency("C", ("B",)))
    with pytest.raises(ValueError, match="cycle"):
        g.register(Dependency("A", ("C",)))


def test_recompute_order_rebuilds_inputs_before_dependents() -> None:
    g = _chain()
    g.invalidate("E1", reason="corrected")
    order = g.recompute_order()
    assert order.index("C1") < order.index("E2") < order.index("S1") < order.index("R1")


def test_a_diamond_dependency_invalidates_once() -> None:
    """Two paths to the same descendant must not double-count or loop."""
    g = DependencyGraph()
    g.register(Dependency("L", ("E",)))
    g.register(Dependency("R", ("E",)))
    g.register(Dependency("M", ("L", "R")))
    r = g.invalidate("E", reason="corrected")
    assert set(r["direct"]) == {"L", "R"}
    assert r["transitive"] == ["M"]
