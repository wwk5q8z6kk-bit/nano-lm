"""Unit pins for the retrieval control arm.

The result of §L.8 is only worth anything if this arm is *strong*. A weak
baseline would make "retrieval is not memory" look proven while proving nothing,
so several tests here exist specifically to stop that: the index must rank
competently, must make the same three-way output decision, and must be charged
honestly for what it reads.
"""

from __future__ import annotations

import pytest

from nano.contracts import EpistemicStatus
from nano.kernel import Identity
from nano.retrieval import (
    SCAN_COST,
    SPAN_READ_COST,
    STRUCTURALLY_UNAVAILABLE,
    IndexedSpan,
    ObservationIndex,
    retrieval_answer,
)
from nano.runtime import Disposition, Question, QuestionKind

SUBJECT = Identity(subject_id="dp0", kind="subject", authority="test")


def _span(sid, ent, attr, val, event, doc):
    return IndexedSpan(span_id=sid, entity=ent, attribute=attr, value=val,
                       verbatim=f"{ent} {attr} = {val}",
                       documentation_time=doc, event_time=event)


def _index():
    return ObservationIndex(spans=[
        _span("s1", "a", "state", "up", "2026-01-05", "2026-01-05"),
        _span("s2", "a", "state", "down", "2026-02-10", "2026-02-10"),
        _span("s3", "b", "state", "up", "2026-02-01", "2026-02-01"),
        _span("s4", "b", "state", "down", "2026-02-01", "2026-02-02"),
    ])


def _q(kind, ent="", attr=""):
    return Question(kind=kind, entity=ent, attribute=attr)


# ---------------------------------------------------------------------------
# The baseline is competent
# ---------------------------------------------------------------------------

def test_ranking_is_by_event_time_not_arrival_time():
    """A delayed report about an old event must not outrank a timely report
    about a recent one. Getting this wrong would be a strawman."""
    index = ObservationIndex(spans=[
        _span("old", "a", "state", "up", "2026-01-01", "2026-03-01"),
        _span("new", "a", "state", "down", "2026-02-01", "2026-02-01")])
    assert index.search("a", "state", 4)[0].span_id == "new"


def test_the_most_recent_value_is_presented_with_its_evidence():
    a = retrieval_answer(_q(QuestionKind.CURRENT_VALUE, "a", "state"),
                         _index(), SUBJECT, k=4)
    assert a.disposition is Disposition.PRESENT
    assert a.content.endswith("down")
    assert a.evidence_span_ids == ("s2",)


def test_disagreement_at_the_latest_event_time_is_escalated():
    """The arm makes the same three-way decision. Denying it an output decision
    would make the undeclared-error comparison meaningless."""
    a = retrieval_answer(_q(QuestionKind.CURRENT_VALUE, "b", "state"),
                         _index(), SUBJECT, k=4)
    assert a.disposition is Disposition.REVIEW
    assert a.epistemic_status is EpistemicStatus.CONFLICTING
    assert "up" in a.content and "down" in a.content


def test_nothing_retrieved_abstains():
    a = retrieval_answer(_q(QuestionKind.CURRENT_VALUE, "zz", "state"),
                         _index(), SUBJECT, k=4)
    assert a.disposition is Disposition.ABSTAIN
    assert a.epistemic_status is EpistemicStatus.NOT_FOUND


def test_a_budgeted_scan_does_not_claim_nothing_exists():
    """Honest about scope: finding no conflict in k spans is not establishing
    that none exists."""
    index = ObservationIndex(spans=[
        _span("s1", "a", "state", "up", "2026-01-05", "2026-01-05")])
    a = retrieval_answer(_q(QuestionKind.WHAT_CONFLICTS), index, SUBJECT, k=1)
    assert a.disposition is Disposition.ABSTAIN
    assert "cannot establish" in a.reason


# ---------------------------------------------------------------------------
# What a fold-free index structurally cannot do
# ---------------------------------------------------------------------------

def test_the_four_stateful_question_kinds_are_declared_unavailable():
    """Not scored as wrong answers — counting a structural impossibility as a
    failed attempt would inflate the result and hide the mechanism."""
    expected = {QuestionKind.WHAT_CHANGED, QuestionKind.WHY_DIFFERENT,
                QuestionKind.WHAT_IS_STALE, QuestionKind.WHAT_WOULD_HELP}
    assert set(STRUCTURALLY_UNAVAILABLE) == expected
    for kind in expected:
        q = (_q(kind, "a", "state")
             if kind is QuestionKind.WHY_DIFFERENT else _q(kind))
        a = retrieval_answer(q, _index(), SUBJECT, k=4)
        assert a.disposition is Disposition.ABSTAIN
        assert "unavailable without persistent state" in a.reason
        assert a.cost_spent == 0.0


def test_every_unavailable_kind_states_its_mechanism():
    for kind, why in STRUCTURALLY_UNAVAILABLE.items():
        assert len(why) > 40, f"{kind.value} has no real explanation"


# ---------------------------------------------------------------------------
# Cost is measured, not asserted
# ---------------------------------------------------------------------------

def test_reading_more_spans_costs_more():
    small, large = _index(), _index()
    retrieval_answer(_q(QuestionKind.CURRENT_VALUE, "a", "state"), small,
                     SUBJECT, k=1)
    retrieval_answer(_q(QuestionKind.CURRENT_VALUE, "a", "state"), large,
                     SUBJECT, k=4)
    assert large.spans_read > small.spans_read
    assert large.cost > small.cost


def test_cost_matches_the_declared_model():
    index = _index()
    retrieval_answer(_q(QuestionKind.CURRENT_VALUE, "a", "state"), index,
                     SUBJECT, k=4)
    assert index.searches == 1
    assert index.cost == SCAN_COST + index.spans_read * SPAN_READ_COST


def test_k_zero_reads_the_whole_index():
    """The context-stuffing limit has to be reachable, or the result could be
    dismissed as a budget artifact."""
    index = _index()
    retrieval_answer(_q(QuestionKind.WHAT_CONFLICTS), index, SUBJECT, k=0)
    assert index.spans_read == len(index.spans)


# ---------------------------------------------------------------------------
# Structural guard
# ---------------------------------------------------------------------------

def test_the_index_cannot_reach_the_world_or_the_fold():
    import pathlib
    source = pathlib.Path(__file__).with_name("retrieval.py").read_text()
    assert "nano.slw" not in source
    assert "truth_at" not in source
    assert "LedgerBuilder" not in source, (
        "the retrieval arm must not be able to consult the fold it is the "
        "control for")


def test_the_index_holds_no_resolution_state():
    index = _index()
    for forbidden in ("resolved", "conflicted", "superseded", "gaps",
                      "freshness"):
        assert not hasattr(index, forbidden)


def test_building_the_index_skips_unparseable_spans():
    class FakeSpan:
        def __init__(self, sid, text):
            self.evidence_span_id = sid
            self.verbatim = text
            self.documentation_time = "2026-01-01"
            self.candidate_event_time = "2026-01-01"

    def parse(text):
        head, _, value = text.partition(" = ")
        entity, _, attribute = head.partition(" ")
        return (entity, attribute, value) if entity and attribute and value \
            else None

    index = ObservationIndex.build(
        [FakeSpan("ok", "a state = up"), FakeSpan("bad", "unparseable")], parse)
    assert [s.span_id for s in index.spans] == ["ok"]
