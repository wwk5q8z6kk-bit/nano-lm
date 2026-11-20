"""Grounded allergy conflict evidence must refer to the same allergen scope."""
import pytest

from nano_ai.contract import EvidenceSpan, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.scribe import scribe
from nanoscribe.test_grounded_substance_ownership import ALG, observed


@pytest.mark.parametrize("denial,allergen,conflict", [
    ("I have no drug allergies.", "Latex", False),
    ("I have no drug allergies.", "Penicillin", True),
    ("I don't have any medication allergies.", "Latex", False),
    ("I am not allergic to any medications.", "Aspirin", True),
    ("I am not allergic to any medications.", "Nickel", False),
    ("I have no drug allergies.", "Zelqor", False),
    ("No material allergies.", "Latex", True),
    ("I have no allergies to medications.", "Penicillin", True),
    ("I have no allergies to medications.", "Latex", False),
    ("I have no drug allergies.", "Zinc", False),
    ("No material allergies.", "Penicillin", False),
    ("No food allergies.", "Peanuts", True),
    ("No food allergies.", "Latex", False),
    ("I have no allergies.", "Latex", True),
    ("No known allergies.", "Zelqor", True),
    ("I am not allergic to latex.", "Latex", True),
    ("I am not allergic to latex.", "Latex gloves", False),
    ("I am not allergic to latex.", "Penicillin", False),
    ("No allergies when indoors.", "Latex", False),
    ("Doctor: Any drug allergies?\nPatient: No.", "Latex", False),
    ("Doctor: Any drug allergies?\nPatient: No.", "Penicillin", True),
    ("Doctor: Any allergies?\nPatient: Nope.", "Zelqor", True),
    ("Doctor: Any food allergies?\nPatient: No.", "Peanuts", True),
])
@pytest.mark.parametrize("denial_first", [True, False])
def test_denial_only_contradicts_an_allergen_within_its_scope(denial, allergen, conflict, denial_first):
    negative = denial if denial.startswith("Doctor:") else f"Patient: {denial}"
    positive = f"Patient: {allergen} gives me hives."
    report = "\n".join((negative, positive) if denial_first else (positive, negative))
    claim, _, compact, english = observed(report, ALG, allergen)
    assert claim["state"] == ("conflicting" if conflict else "supported")
    assert claim["value"] == (None if conflict else allergen)
    assert any(c.startswith(":C[") for c in compact) == conflict
    assert any(c.startswith(" CONTRADICTED[") for c in english) == conflict
    if conflict:
        return

    assert f":S[{allergen}]" in compact and f" REPORTED[{allergen}]" in english
    reply = negative.rsplit("Patient: ", 1)[1]
    assert f":A[{reply}]" not in compact and f" DENIED[{reply}]" not in english
    start = report.index(reply)
    span = EvidenceSpan(start, start + len(reply), reply)
    doc = transcript_source(NanoInput("denial-scope", report))
    assert _observation_kind(doc, ALG, span, context_policy="grounded") is None
    raw = f"CC:M|DUR:M|SEV:M|MED:M|ALG:A[{reply}]"
    draft = scribe(report, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    assert draft["verification"]["passed"]
    assert next(c for c in draft["note"]["claims"] if c["field"] == ALG.value)["state"] != "absent"


def test_a_category_question_can_identify_an_otherwise_unrecognized_allergen():
    report = "Patient: I have no drug allergies.\nDoctor: Any medication allergies?\nPatient: Zelqor."
    claim, _, compact, english = observed(report, ALG, "Zelqor")
    assert claim["state"] == "conflicting"
    assert any(c.startswith(":C[") for c in compact)
    assert any(c.startswith(" CONTRADICTED[") for c in english)


@pytest.mark.parametrize("report,value", [
    ("Patient: I have no drug allergies.", "I have no drug allergies."),
    ("Doctor: Any allergies?\nPatient: No.", "No."),
])
def test_standalone_recognized_denial_remains_available(report, value):
    _, _, menu, english = observed(report, ALG, value)
    assert f":A[{value}]" in menu and f" DENIED[{value}]" in english


def test_same_patient_medication_report_can_ground_the_unknown_allergen_category():
    report = "Patient: My medication is Zelqor.\nPatient: No drug allergies.\nPatient: I am allergic to Zelqor."
    _, _, compact, english = observed(report, ALG, "Zelqor")
    assert any(c.startswith(":C[") for c in compact)
    assert any(c.startswith(" CONTRADICTED[") for c in english)
