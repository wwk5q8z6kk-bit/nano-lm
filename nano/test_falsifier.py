"""Unit pins for MTA-WOULDCHANGE.

The contract is that a falsifier is *executable*: it names an observation
concrete enough to go and get. Most of these tests make the vague version
unconstructible, because "evidence to the contrary" is true of every claim ever
made and actionable for none of them.
"""

from __future__ import annotations

import pytest

from nano.falsifier import (
    Falsifier,
    FalsifierKind,
    Outcome,
    _next_day,
    broad_falsifier,
    falsifiers_for,
)

STATE = dict(
    resolved={("a", "state"): "up"},
    conflicted={("b", "state"): ("2026-02-01", ("up", "down"))},
    latest_time={("a", "state"): "2026-02-20", ("b", "state"): "2026-02-01"},
    evidence={("a", "state"): ("span_a",), ("b", "state"): ("span_b",)},
    as_of="2026-03-01")


def test_a_vague_falsifier_is_unconstructible():
    with pytest.raises(ValueError, match="concrete observation"):
        Falsifier(claim="x", entity="a", attribute="state", value="",
                  event_time="2026-02-21",
                  kind=FalsifierKind.SUPERSEDING_REPORT,
                  outcome=Outcome.NEW_VALUE, rests_on=("s",), reason="because")


def test_a_falsifier_must_say_why_it_bites():
    with pytest.raises(ValueError, match="why it bites"):
        Falsifier(claim="x", entity="a", attribute="state", value="down",
                  event_time="2026-02-21",
                  kind=FalsifierKind.SUPERSEDING_REPORT,
                  outcome=Outcome.NEW_VALUE, rests_on=("s",), reason="")


def test_a_falsifier_for_an_evidenced_claim_must_cite_what_it_rests_on():
    """A falsifier is a function of what the claim depends on. One built without
    reference to that is a guess about what might matter."""
    with pytest.raises(ValueError, match="what the claim rests on"):
        Falsifier(claim="x", entity="a", attribute="state", value="down",
                  event_time="2026-02-21",
                  kind=FalsifierKind.SUPERSEDING_REPORT,
                  outcome=Outcome.NEW_VALUE, reason="because")


def test_a_first_report_falsifier_needs_no_lineage():
    """Nothing is known, so there is nothing for the claim to rest on."""
    ok = Falsifier(claim="unknown", entity="z", attribute="state", value="any",
                   event_time="2026-03-01", kind=FalsifierKind.FIRST_REPORT,
                   outcome=Outcome.BECOMES_KNOWN, reason="nothing is known")
    assert ok.falsifier_id


def test_a_held_value_gets_both_ways_it_could_move():
    """The resolution rule admits exactly two mechanisms, so both are named:
    a later differing report replaces, a same-time differing report contests."""
    found = falsifiers_for("a", "state", **STATE)
    kinds = {f.kind for f in found}
    assert kinds == {FalsifierKind.SUPERSEDING_REPORT,
                     FalsifierKind.CONTRADICTING_REPORT}
    outcomes = {f.outcome for f in found}
    assert outcomes == {Outcome.NEW_VALUE, Outcome.BECOMES_DISPUTED}
    for f in found:
        assert f.rests_on == ("span_a",)
        assert f.value != "up"


def test_a_disputed_key_is_falsified_by_something_strictly_later():
    found = falsifiers_for("b", "state", **STATE)
    assert len(found) == 1
    assert found[0].kind is FalsifierKind.RESOLVING_REPORT
    assert found[0].event_time > "2026-02-01"


def test_an_unknown_key_is_falsified_by_any_report():
    found = falsifiers_for("z", "state", **STATE)
    assert found[0].kind is FalsifierKind.FIRST_REPORT
    assert found[0].outcome is Outcome.BECOMES_KNOWN


def test_the_superseding_time_is_unambiguously_later_than_a_month():
    """A falsifier dated inside the month it must overturn cannot be ordered
    against it — the mixed-precision defect, one more time."""
    assert _next_day("2026-01") == "2026-02-01"
    assert _next_day("2026-12") == "2027-01-01"
    assert _next_day("2026-01-05") == "2026-01-06"


def test_the_description_names_the_observation_and_the_consequence():
    text = falsifiers_for("a", "state", **STATE)[0].describe()
    assert "a.state" in text and "reported as" in text
    assert "replaced" in text


# ---------------------------------------------------------------------------
# The control's defect, pinned
# ---------------------------------------------------------------------------

def test_the_broad_control_names_the_key_but_constrains_nothing():
    """It sounds specific — it names an entity and an attribute — and the
    earliest observation satisfying its wording is a corroboration."""
    control = broad_falsifier("a", "state", **STATE)[0]
    assert control.entity == "a" and control.attribute == "state"
    assert control.value == "up"            # i.e. what is already believed
    assert control.event_time == "2026-02-20"


def test_the_control_is_more_informative_than_the_usual_hedge():
    """Not a strawman: it is strictly better than 'evidence to the contrary',
    and still under-specified in the way that matters."""
    control = broad_falsifier("a", "state", **STATE)[0]
    assert control.reason
    assert control.rests_on


def test_falsifier_generation_cannot_reach_the_world():
    import pathlib
    source = pathlib.Path(__file__).with_name("falsifier.py").read_text()
    assert "nano.slw" not in source
    assert "truth_at" not in source
