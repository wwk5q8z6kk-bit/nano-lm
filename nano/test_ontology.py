"""Pins for the ontology registry — expansion discipline, not reduction."""
from __future__ import annotations

import pytest

from nano.ontology import PRIMITIVES, Presence, Primitive, overloading_candidates, summary


def test_primitive_names_are_unique() -> None:
    names = [p.name for p in PRIMITIVES]
    assert len(names) == len(set(names))


def test_every_primitive_names_what_it_is_not() -> None:
    """A primitive undistinguished from anything is not a primitive."""
    for p in PRIMITIVES:
        assert p.must_not_conflate_with, p.name


def test_no_primitive_is_overloaded() -> None:
    """The expansion discipline's failure mode: one type doing two jobs.

    Proliferation is fine. A shared implementation pointer is not — it means a
    distinction exists in the vocabulary but not in the code, which is how the
    distinction gets lost downstream.
    """
    over = overloading_candidates()
    assert not over, f"one implementation carrying two primitives: {over}"


def test_presence_claims_are_backed() -> None:
    for p in PRIMITIVES:
        if p.presence in (Presence.IN_CODE, Presence.PARTIAL):
            assert p.implementation, p.name
        if p.presence is Presence.NAMED_ONLY:
            assert not p.implementation, p.name


def test_splits_reference_a_real_parent() -> None:
    """`splits_from` is earned history; it must point at a primitive that exists."""
    names = {p.name for p in PRIMITIVES}
    for p in PRIMITIVES:
        if p.splits_from:
            assert p.splits_from in names, f"{p.name} splits from unknown {p.splits_from}"


def test_undistinguished_primitive_is_rejected() -> None:
    with pytest.raises(ValueError, match="must_not_conflate_with is required"):
        Primitive(name="X", plane="Cognition", definition="d",
                  must_not_conflate_with=(), presence=Presence.NAMED_ONLY)


def test_registry_stays_open_for_expansion() -> None:
    """A guard against quietly turning this into a fixed minimal set: if the
    vocabulary ever shrinks below what is already implemented, something was
    collapsed rather than added."""
    s = summary()
    built = s["by_presence"]["IN_CODE"] + s["by_presence"]["PARTIAL"]
    assert s["total"] >= built, "vocabulary smaller than what is implemented"
    assert s["total"] >= 25, (
        "primitive count dropped sharply — expansion is the rule; if a genuine "
        "merge happened, update this floor deliberately")
