"""Severity must describe the presenting symptom or answer its severity question."""
import pytest

from nano_ai.contract import FieldName
from nanoscribe.compact import compact_solver
from nanoscribe.scribe import scribe
from nanoscribe.test_grounded_followup import check


@pytest.mark.parametrize("report,value,expected", [
    ("My headache is mild.\nPatient: I have a severe penicillin allergy.", "severe", False),
    ("My headache is mild.\nPatient: I have a severe penicillin allergy.", "mild", True),
    ("I have a severe allergy to latex, but my headache is mild.", "severe", False),
    ("I have a severe allergy to latex, but my headache is mild.", "mild", True),
    ("My cough is mild and my penicillin reaction is severe.", "severe", False),
    ("My cough is mild and my penicillin reaction is severe.", "mild", True),
    ("The antibiotic side effects are 8/10.", "8/10", False),
    ("I have a penicillin allergy. It is severe.", "severe", False),
    ("I have a headache. It is severe.", "severe", True),
    ("My headache is mild after taking aspirin.", "mild", True),
    ("I took a strong dose for my headache.", "strong", False),
    ("I have a mild rash as part of a severe penicillin allergy.", "mild", True),
    ("I have a mild rash as part of a severe penicillin allergy.", "severe", False),
    ("I took aspirin for a severe headache.", "severe", True),
    ("I had a severe traffic delay on the way to the clinic.", "severe", False),
    ("Severe thunderstorms delayed me.", "Severe", False),
    ("My headache is severe after the storm.", "severe", True),
    ("The train ride was 7/10.", "7/10", False),
    ("I have a severe headache and a penicillin allergy.", "severe", True),
    ("I have severe ear ringing because of the noise.", "severe", True),
    ("The headache rates 6/10.", "6/10", True),
])
def test_selected_rating_describes_the_symptom(report, value, expected):
    source = "Patient: " + report
    check(source, "SEV", FieldName.SEVERITY, value, source.index(value), expected)


@pytest.mark.parametrize("question,answer,value,expected", [
    ("How severe was the penicillin allergy?", "Severe.", "Severe", False),
    ("How would you rate the coffee?", "7/10.", "7/10", False),
    ("How bad was the traffic?", "7/10.", "7/10", False),
    ("Any medications?", "Mild.", "Mild", False),
    ("How bad is your jaw pain?", "7/10.", "7/10", True),
    ("How bad is your headache?", "My allergy is severe.", "severe", False),
    ("How severe is the reaction?", "My headache is mild.", "mild", True),
])
def test_elliptical_rating_inherits_its_question_subject(question, answer, value, expected):
    source = f"Doctor: {question}\nPatient: {answer}"
    check(source, "SEV", FieldName.SEVERITY, value, source.rindex(value), expected)


def test_nonpresenting_allergy_rating_does_not_override_the_selected_headache():
    source = "Patient: My headache is mild.\nPatient: I have a severe penicillin allergy."
    raw = "CC:S[headache]|DUR:M|SEV:S[severe]|MED:M|ALG:M"
    note = scribe(source, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    assert note["verification"]["passed"]
    claims = {c["field"]: c for c in note["note"]["claims"]}
    assert claims["chief_complaint"]["value"] == "headache"
    assert claims["severity"]["state"] == "uncertain"
