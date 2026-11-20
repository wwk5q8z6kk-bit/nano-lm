"""Either/or reports do not establish either individual alternative."""

import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, grounded_candidate_choices
from nanoscribe.scribe import scribe


@pytest.mark.parametrize("slot,field,report,rejected,accepted,uncertainty", [
    ("SEV", FieldName.SEVERITY, "The pain is either mild or severe.",
     ["mild", "severe", "either mild or severe"], [], "either mild or severe"),
    ("DUR", FieldName.DURATION, "It started either two days ago or three days ago.",
     ["two days", "three days ago"], [], "either two days ago or three days ago"),
    ("SEV", FieldName.SEVERITY, "Either the pain is 3.5/10 or it is 6.5/10.",
     ["3.5/10", "6.5/10"], [], "Either the pain is 3.5/10 or it is 6.5/10"),
    ("SEV", FieldName.SEVERITY, "The pain is either mild, moderate, or severe.",
     ["mild", "moderate", "severe"], [], "either mild, moderate, or severe"),
    ("DUR", FieldName.DURATION, "It began two days ago and the pain is either mild or severe.",
     [], ["two days ago"], None),
    ("DUR", FieldName.DURATION, "The pain is either mild or severe, but it began two days ago.",
     [], ["two days ago"], None),
    ("SEV", FieldName.SEVERITY, "Either at rest or walking, my pain is mild.",
     [], ["mild"], None),
    ("SEV", FieldName.SEVERITY, "The pain is mild either way.",
     [], ["mild"], None),
    ("SEV", FieldName.SEVERITY, "Either way, the pain is mild and I can work or rest.",
     [], ["mild"], None),
    ("DUR", FieldName.DURATION, "It lasted either two or three days.",
     ["three days"], [], "either two or three days"),
    ("MED", FieldName.MEDICATION, "I took either aspirin or ibuprofen.",
     ["aspirin", "ibuprofen"], [], None),
    ("DUR", FieldName.DURATION, "It began either Monday or Tuesday. The swelling began yesterday.",
     ["Monday", "Tuesday"], ["yesterday"], "either Monday or Tuesday"),
])
def test_scalar_reporting_alternatives(slot, field, report, rejected, accepted, uncertainty):
    transcript = "Patient: " + report
    doc = transcript_source(NanoInput("alternatives", transcript))
    menu = grounded_candidate_choices(transcript, field, set())
    english = build_grounded_english_menu(doc, field, set())
    for value in rejected + accepted:
        start = transcript.index(value)
        span = EvidenceSpan(start, start + len(value), value)
        proposal = "|".join(f"{s}:S[{value}]" if s == slot else f"{s}:M" for s in ("CC", "DUR", "SEV", "MED", "ALG"))
        note = scribe(transcript, solver=compact_solver(lambda _: proposal, binder="grounded"), context_policy="grounded")
        claim = next(c for c in note["note"]["claims"] if c["field"] == field.value)
        supported = value in accepted
        assert claim["state"] == ("supported" if supported else "uncertain")
        assert note["verification"]["passed"]
        assert _observation_kind(doc, field, span, context_policy="grounded") is (FieldState.SUPPORTED if supported else None)
        assert (f":S[{value}]" in menu) == supported
        assert (f" REPORTED[{value}]" in english) == supported

    assert {k for k in menu if k.startswith(":U[") and k != ":U[]"} == (
        {f":U[{uncertainty}]"} if uncertainty else set())
    assert {k for k in english if k.startswith(" UNCERTAIN[")} == (
        {f" UNCERTAIN[{uncertainty}]"} if uncertainty else set())


def test_disjunctive_medication_denial_does_not_verify_absence():
    transcript = "Patient: Either I took no medication or I took aspirin."
    value = transcript.removeprefix("Patient: ")
    doc = transcript_source(NanoInput("alternative-denial", transcript))
    start = transcript.index(value)
    span = EvidenceSpan(start, start + len(value), value)
    assert _observation_kind(doc, FieldName.MEDICATION, span, context_policy="grounded") is None
    assert f":A[{value}]" not in grounded_candidate_choices(transcript, FieldName.MEDICATION, set())
