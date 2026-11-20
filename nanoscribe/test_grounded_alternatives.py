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


@pytest.mark.parametrize("field,report,quote,allowed", [
    (FieldName.DURATION, "My headache has lasted either two or three days.", "either two or three days", True),
    (FieldName.DURATION, "My headache started either Monday or Tuesday.", "either Monday or Tuesday", True),
    (FieldName.SEVERITY, "My pain is either mild, moderate, or severe.", "either mild, moderate, or severe", True),
    (FieldName.SEVERITY, "Either my pain is 3.5/10 or it is 6.5/10.", "Either my pain is 3.5/10 or it is 6.5/10", True),
    (FieldName.SEVERITY, "My pain is either two or severe.", "either two or severe", False),
    (FieldName.DURATION, "My pain started either two or Tuesday.", "either two or Tuesday", False),
    (FieldName.SEVERITY, "My pain is either mild or the traffic is bad.", "either mild or the traffic is bad", False),
    (FieldName.DURATION, "The pass lasts either two or three days.", "either two or three days", False),
    (FieldName.DURATION, "It started either Monday or Tuesday.", "either Monday or Tuesday", False),
    (FieldName.SEVERITY, "I have a headache but the parking is either 3/10 or 4/10.", "either 3/10 or 4/10", False),
    (FieldName.SEVERITY, "My sister's pain is either mild or severe.", "either mild or severe", False),
])
def test_after_logistics_the_clinical_subject_must_own_each_alternative(field, report, quote, allowed):
    text = "Other: Where is the garage?\nPatient: " + report
    request = NanoInput("logistical-alternatives", text)
    doc = transcript_source(request)
    menu = grounded_candidate_choices(text, field, set())
    english = build_grounded_english_menu(doc, field, set())
    assert (f":U[{quote}]" in menu) is allowed
    assert (f" UNCERTAIN[{quote}]" in english) is allowed
    raw = "|".join(s + (f":U[{quote}]" if s == ("DUR" if field is FieldName.DURATION else "SEV") else ":M")
                   for s in ("CC", "DUR", "SEV", "MED", "ALG"))
    extraction, diagnostics = compact_solver(lambda _: raw, binder="grounded").infer_grounded(request)
    index = 1 if field is FieldName.DURATION else 2
    assert diagnostics["fields"][index]["verified"] is allowed
    assert bool(extraction.fields[index].evidence) is allowed


def test_plain_other_statement_preserves_preceding_context_for_alternatives():
    text = ("Doctor: How long did the headache last?\nOther: The umbrella is upstairs.\n"
            "Patient: It lasted either two or three days.")
    assert ":U[either two or three days]" in grounded_candidate_choices(text, FieldName.DURATION, set())
