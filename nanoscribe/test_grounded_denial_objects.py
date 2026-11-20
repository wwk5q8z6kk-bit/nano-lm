"""The negated object must authorize the asserted field across all grounded routes."""
import pytest

from nano_ai.contract import NanoInput
from nanoscribe.check_note import transcript_source
from nanoscribe.grounded import build_grounded_english_menu
from nanoscribe.test_grounded_observation_scope import ALG, MED, check


@pytest.mark.parametrize("text,denied", [
    ("I have no medication allergies.", ALG), ("I have no drug allergies.", ALG),
    ("No known medicine allergies.", ALG), ("I do not have any medication allergies.", ALG),
    ("I have no known allergies.", ALG), ("I’m not allergic to anything.", ALG),
    ("I have no allergy medication.", None), ("I have no allergy medicine.", None),
    ("I have no medication list.", None), ("No drug allergy bracelet.", None),
    ("I have no medication today was a typo.", None), ("I have no medication list for today.", None),
    ("I have no allergy test results.", None), ("I have no medicine cabinet key.", None),
    ("I never received the note saying no medications.", None),
    ("I bought no medications.", None), ("I have no record of drug allergies.", None),
    ("I took no medications.", MED), ("No remedies.", MED),
    ("I haven’t used any treatments.", MED), ("I take no medicine.", MED),
    ("I have taken no medications.", MED), ("I did not take anything today.", MED),
    ("Haven’t taken anything.", MED), ("Took nothing for it.", MED),
    ("I did not have any medication.", MED), ("I haven’t taken any medications today.", MED),
    ("I don't take any medications.", MED), ("I don't have any allergies.", ALG),
])
@pytest.mark.parametrize("field", [MED, ALG])
def test_direct_denial_binds_the_complete_predicate_and_object(text, denied, field):
    report = "Patient: " + text
    allowed = field is denied
    compact = check(report, field, "A", text, "absent" if allowed else "uncertain")
    assert (f":A[{text}]" in compact) == allowed
    source = transcript_source(NanoInput("denial-head", report))
    english = build_grounded_english_menu(source, field, set())
    assert (f" DENIED[{text}]" in english) == allowed


@pytest.mark.parametrize("denial,field,value,report", [
    ("I have no medication allergies.", MED, "aspirin", "I took aspirin today."),
    ("I have no allergy medication.", ALG, "latex", "I am allergic to latex."),
    ("I have no medication list.", MED, "aspirin", "I took aspirin today."),
])
def test_unrelated_denial_cannot_conflict_with_an_independent_report(denial, field, value, report):
    source = f"Patient: {denial}\nPatient: {report}"
    choices = check(source, field, "S", value, "supported")
    assert not any(c.startswith(":C[") for c in choices)
    assert f":S[{value}]" in choices
