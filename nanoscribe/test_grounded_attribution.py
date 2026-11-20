"""Verify grounded proposals, constrained choices and independent evidence checks."""
import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import format_grounded_span_token, grounded_candidate_choices
from nanoscribe.scribe import scribe


def check(source, field, value, *, supported):
    doc = transcript_source(NanoInput("attribution", source))
    start = source.index(value)
    span = EvidenceSpan(start, start + len(value), value)
    token = format_grounded_span_token(doc, span)
    slot = {FieldName.ALLERGY: "ALG", FieldName.MEDICATION: "MED", FieldName.DURATION: "DUR",
            FieldName.SEVERITY: "SEV"}[field]
    proposal = "|".join(s + (f":S[{token}]" if s == slot else ":M") for s in ("CC", "DUR", "SEV", "MED", "ALG"))
    result = scribe(source, solver=compact_solver(lambda _: proposal, binder="grounded"), context_policy="grounded")
    assert result["verification"]["passed"]
    claim = next(c for c in result["note"]["claims"] if c["field"] == field.value)
    assert claim["state"] == ("supported" if supported else "uncertain")
    assert _observation_kind(doc, field, span, context_policy="grounded") is (FieldState.SUPPORTED if supported else None)
    assert (f":S[{token}]" in grounded_candidate_choices(source, field, set())) == supported


@pytest.mark.parametrize("turn,field,value", [
    ("My chart says I am allergic to penicillin, but that is false.", FieldName.ALLERGY, "penicillin"),
    ("The chart lists an allergy to latex. That's inaccurate.", FieldName.ALLERGY, "latex"),
    ("My note claims I took aspirin, but I dispute that.", FieldName.MEDICATION, "aspirin"),
    ("According to my record, it began two weeks ago. That entry is not accurate.", FieldName.DURATION, "two weeks"),
    ('The chart says, “My pain is mild. I am allergic to latex.” I disagree.', FieldName.ALLERGY, "latex"),
    ("The chart states that the pain is severe.", FieldName.SEVERITY, "severe"),
    ("My chart says I had a rash and I took aspirin.", FieldName.MEDICATION, "aspirin"),
    ("My note reads 'It is severe. I am allergic to latex.'", FieldName.ALLERGY, "latex"),
    ("I never said I took aspirin.", FieldName.MEDICATION, "aspirin"),
    ("It is false that I am allergic to latex.", FieldName.ALLERGY, "latex"),
    ("I am allergic to latex, but that was false.", FieldName.ALLERGY, "latex"),
    ("I am allergic to latex. That was false.", FieldName.ALLERGY, "latex"),
    ("I am allergic to latex.\nPatient: That was false.", FieldName.ALLERGY, "latex"),
])
def test_embedded_or_disavowed_value_cannot_be_asserted(turn, field, value):
    check("Patient: " + turn, field, value, supported=False)


@pytest.mark.parametrize("turn,field,quoted,reported", [
    ("My chart says I took aspirin, but I took ibuprofen.", FieldName.MEDICATION, "aspirin", "ibuprofen"),
    ("I took ibuprofen, although my chart says I took aspirin.", FieldName.MEDICATION, "aspirin", "ibuprofen"),
    ("My chart says I am allergic to latex. I am allergic to wool.", FieldName.ALLERGY, "latex", "wool"),
    ("My record says it began two weeks ago, but it began three days ago.", FieldName.DURATION, "two weeks", "three days"),
])
def test_independent_report_survives_an_attributed_sibling(turn, field, quoted, reported):
    source = "Patient: " + turn
    check(source, field, quoted, supported=False)
    check(source, field, reported, supported=True)


@pytest.mark.parametrize("turn,field,value", [
    ("I am allergic to latex.", FieldName.ALLERGY, "latex"),
    ("I checked my chart. I took ibuprofen.", FieldName.MEDICATION, "ibuprofen"),
    ("I would say severe.", FieldName.SEVERITY, "severe"),
    ("I took ibuprofen. The chart is wrong about the appointment time.", FieldName.MEDICATION, "ibuprofen"),
])
def test_direct_reports_survive_incidental_record_or_speech_language(turn, field, value):
    check("Patient: " + turn, field, value, supported=True)
