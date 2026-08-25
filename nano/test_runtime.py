"""Tests for the executable cognitive substrate.

Unit pins only — no world, no ledger, no model. The executor consumes a narrow
`StateView`, so its guards can be exercised directly; if these tests had to
generate a synthetic fleet to check that a conflicted key never gets PRESENTed,
the guard would not get run.

Benchmark-level scoring of the runtime lives in `test_slw.py`, next to the
other scorers.
"""

from __future__ import annotations

import pytest

from nano.contracts import DerivationMode, EpistemicStatus
from nano.dependency import Freshness
from nano.kernel import Identity, SliceStatus
from nano.runtime import (
    Answer,
    BudgetExhausted,
    CAPABILITIES,
    Disposition,
    Question,
    QuestionKind,
    StateView,
    answer,
    open_slice,
    render,
    run_slice,
    select_capability,
    verify,
)

SUBJECT = Identity(subject_id="dp0", kind="subject", authority="test")
OTHER = Identity(subject_id="dp0-other", kind="subject", authority="test")


def _view(**overrides) -> StateView:
    base = dict(
        subject=SUBJECT, as_of="2026-03-01",
        resolved={("a", "state"): "up", ("c", "state"): "down"},
        conflicted={("b", "state"): ("2026-02-01", ("up", "down"))},
        gaps=frozenset({("c", "state")}),
        evidence={("a", "state"): ("span_a",), ("b", "state"): ("span_b",),
                  ("c", "state"): ("span_c",)},
        latest_time={("a", "state"): "2026-02-20"},
        evidence_count={("a", "state"): 2},
        known_spans=frozenset({"span_a", "span_b", "span_c"}),
        prior={("a", "state"): "down"},
        freshness={"view:a": Freshness.CURRENT},
        ledger_version=7)
    base.update(overrides)
    return StateView(**base)


def _slice(budget: float = 40.0):
    return open_slice("unit test", SUBJECT, budget=budget)


# ---------------------------------------------------------------------------
# Structural guard — the executor must not be able to see the answer
# ---------------------------------------------------------------------------

def test_the_runtime_cannot_reach_the_world_or_a_scorer():
    """`nano.slw` owns `truth_at`, `_broken_keys` and `_acquire`. If the runtime
    ever imports it, an executor could grade itself and would score perfectly
    while demonstrating nothing."""
    import pathlib
    source = pathlib.Path(__file__).with_name("runtime.py").read_text()
    assert "nano.slw" not in source
    assert "truth_at" not in source
    import nano.runtime as runtime
    assert not hasattr(runtime, "SyntheticWorld")


def test_state_view_exposes_only_believed_state():
    view = _view()
    for forbidden in ("truth", "world", "changes", "observations"):
        assert not hasattr(view, forbidden)


# ---------------------------------------------------------------------------
# An answer has a standing, and the standing is derived
# ---------------------------------------------------------------------------

def test_an_answer_without_a_reason_is_refused():
    with pytest.raises(ValueError, match="why it has the standing"):
        Answer(question=Question(kind=QuestionKind.WHAT_CONFLICTS),
               disposition=Disposition.ABSTAIN, content="", reason="")


def test_presenting_without_evidence_is_refused():
    """§XXX: an ungrounded claim must abstain or escalate. It may not be
    presented, whatever the rest of the pipeline decided."""
    with pytest.raises(ValueError, match="refusing to PRESENT"):
        Answer(question=Question(kind=QuestionKind.WHAT_CONFLICTS),
               disposition=Disposition.PRESENT, content="everything is fine",
               reason="felt right", evidence_span_ids=())


def test_a_resolved_key_is_presented_with_its_evidence():
    a = answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="a",
                        attribute="state"), _view(), _slice())
    assert a.disposition is Disposition.PRESENT
    assert "up" in a.content
    assert a.evidence_span_ids == ("span_a",)
    assert a.receipt is not None and a.receipt.unsupported_count == 0


def test_a_conflicted_key_goes_to_review_and_is_never_presented():
    """The failure mode is picking. A live disagreement is escalated, and both
    values survive into the content."""
    a = answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="b",
                        attribute="state"), _view(), _slice())
    assert a.disposition is Disposition.REVIEW
    assert a.epistemic_status is EpistemicStatus.CONFLICTING
    assert "up" in a.content and "down" in a.content


def test_an_unreported_attribute_abstains_rather_than_guessing():
    """Absence-never-from-silence at the output boundary."""
    a = answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="a",
                        attribute="never_reported"), _view(), _slice())
    assert a.disposition is Disposition.ABSTAIN
    assert a.epistemic_status is EpistemicStatus.NOT_FOUND
    assert a.content == ""
    assert "no source" in a.reason


def test_a_key_with_a_known_missing_report_escalates_rather_than_presenting():
    """A value is held, but a change to it went unreported — so it may predate
    the current state. Presenting it as current would be the stale-but-confident
    failure."""
    a = answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="c",
                        attribute="state"), _view(), _slice())
    assert a.disposition is Disposition.REVIEW
    assert a.epistemic_status is EpistemicStatus.OUTDATED


def test_why_different_attributes_the_change_to_evidence():
    a = answer(Question(kind=QuestionKind.WHY_DIFFERENT, entity="a",
                        attribute="state"), _view(), _slice())
    assert a.disposition is Disposition.PRESENT
    assert "was down" in a.content and "now up" in a.content
    assert a.evidence_span_ids


def test_what_changed_abstains_without_a_prior_answer():
    """A first observation is not a change."""
    a = answer(Question(kind=QuestionKind.WHAT_CHANGED), _view(prior={}),
               _slice())
    assert a.disposition is Disposition.ABSTAIN


# ---------------------------------------------------------------------------
# Verification can only downgrade
# ---------------------------------------------------------------------------

def test_a_fabricated_citation_is_caught_and_downgraded():
    """The span the answer would cite does not resolve in the ledger. Nothing
    downstream may promote that back to PRESENT."""
    view = _view(known_spans=frozenset({"span_b", "span_c"}))   # span_a gone
    a = answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="a",
                        attribute="state"), view, _slice())
    assert a.disposition is Disposition.ABSTAIN
    assert "do not resolve" in a.reason
    assert a.content == ""


def test_verify_reports_uncited_content_as_unsupported():
    receipt = verify("a claim", (), _view(), artifact_id="art")
    assert receipt.unsupported_count == 1
    assert receipt.coverage_status == "uncited"
    assert receipt.provenance_coverage == 0.0


# ---------------------------------------------------------------------------
# Capability selection reads cost and reliability
# ---------------------------------------------------------------------------

def test_selection_takes_the_cheapest_sufficient_capability():
    q = Question(kind=QuestionKind.CURRENT_VALUE, entity="a", attribute="state")
    tool = select_capability(q, reliability_floor=0.9, budget_remaining=100)
    candidates = CAPABILITIES[QuestionKind.CURRENT_VALUE]
    assert tool.cost == min(t.cost for t in candidates if t.reliability >= 0.9)


def test_selection_refuses_when_nothing_meets_the_reliability_floor():
    q = Question(kind=QuestionKind.WHAT_WOULD_HELP)
    with pytest.raises(ValueError, match="reliability floor"):
        select_capability(q, reliability_floor=0.999, budget_remaining=100)


def test_selection_refuses_rather_than_silently_downgrading():
    q = Question(kind=QuestionKind.WHAT_WOULD_HELP)
    with pytest.raises(BudgetExhausted):
        select_capability(q, reliability_floor=0.9, budget_remaining=0.5)


def test_every_question_kind_has_a_capability():
    for kind in QuestionKind:
        assert CAPABILITIES.get(kind), f"{kind.value} has no capability"


# ---------------------------------------------------------------------------
# Budget is a real stop condition
# ---------------------------------------------------------------------------

def test_running_out_of_budget_is_reported_as_review_not_abstention():
    """A resource fact and an epistemic fact are different. Reporting a budget
    stop as an abstention would hide a capacity problem behind an
    epistemic-sounding word."""
    work = _slice(budget=1.0)
    q = Question(kind=QuestionKind.CURRENT_VALUE, entity="a", attribute="state")
    first = answer(q, _view(), work)
    assert first.disposition is Disposition.PRESENT

    second = answer(Question(kind=QuestionKind.WHAT_WOULD_HELP), _view(), work)
    assert second.disposition is Disposition.REVIEW
    assert "stopped before answering" in second.reason
    assert work.status is SliceStatus.STOPPED


def test_a_slice_must_say_how_it_stops_before_it_starts():
    work = _slice()
    assert work.stop_conditions
    assert work.compute_budget > 0


def test_run_slice_marks_itself_verified_when_it_finishes():
    work = _slice()
    answers = run_slice(work, [
        Question(kind=QuestionKind.CURRENT_VALUE, entity="a", attribute="state"),
        Question(kind=QuestionKind.WHAT_CONFLICTS)], _view())
    assert len(answers) == 2
    assert work.status is SliceStatus.VERIFIED
    assert work.spent > 0


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_running_a_slice_against_another_subjects_state_is_refused():
    work = open_slice("cross subject", OTHER)
    with pytest.raises(ValueError, match="cross-subject"):
        answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="a",
                        attribute="state"), _view(), work)


# ---------------------------------------------------------------------------
# Artifact compilation
# ---------------------------------------------------------------------------

def test_evidence_is_selected_before_anything_is_rendered():
    a = answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="a",
                        attribute="state"), _view(), _slice())
    assert a.artifact is not None
    assert a.artifact.selected_evidence
    assert a.artifact.required_claims == (a.content,)
    assert a.artifact.uncertainty_disclosure == (a.reason,)


def test_an_abstention_compiles_no_artifact():
    a = answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="a",
                        attribute="never_reported"), _view(), _slice())
    assert a.artifact is None


def test_the_rendered_form_leads_with_the_disposition():
    """A reader skimming output must not be able to miss that the system
    abstained or escalated."""
    view, work = _view(), _slice()
    abstain = render(answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="a",
                                     attribute="never_reported"), view, work))
    review = render(answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="b",
                                    attribute="state"), view, work))
    present = render(answer(Question(kind=QuestionKind.CURRENT_VALUE, entity="a",
                                     attribute="state"), view, work))
    assert abstain.startswith("[ABSTAIN]")
    assert review.startswith("[REVIEW]")
    assert not present.startswith("[")


# ---------------------------------------------------------------------------
# Derivation mode is not decorative
# ---------------------------------------------------------------------------

def test_a_planned_recommendation_is_marked_inferred_not_observed():
    """§VI: a forecast rendered as a finding is a safety failure, not a
    formatting one. The information plan is reasoning over evidence, not
    something any source reported."""
    a = answer(Question(kind=QuestionKind.WHAT_WOULD_HELP), _view(), _slice())
    assert a.derivation is DerivationMode.INFERRED
    assert a.derivation in {DerivationMode.INFERRED, DerivationMode.DERIVED}


def test_no_answer_claims_to_have_observed_what_it_computed():
    view, work = _view(), _slice(budget=200.0)
    for kind in QuestionKind:
        q = (Question(kind=kind, entity="a", attribute="state")
             if kind in {QuestionKind.CURRENT_VALUE, QuestionKind.WHAT_SUPPORTS,
                         QuestionKind.WHY_DIFFERENT}
             else Question(kind=kind))
        a = answer(q, view, work)
        if a.derivation is DerivationMode.OBSERVED:
            assert a.question.kind is QuestionKind.WHAT_SUPPORTS, (
                f"{kind.value} claims OBSERVED but it computed its answer")
