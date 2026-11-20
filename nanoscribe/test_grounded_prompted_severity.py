"""A patient can answer a current symptom rating question in the question's scale."""
import pytest

from nano_ai.contract import FieldName, NanoInput
from nanoscribe.check_note import transcript_source
from nanoscribe.grounded import build_grounded_english_menu, detect_grounded_conflicts, grounded_candidate_choices
from nanoscribe.test_grounded import _grounded_claim
from nanoscribe.test_grounded_followup import check


CURRENT = "How severe is your pain now?"
SCALE = "On a scale of one to ten, how severe is your pain?"


@pytest.mark.parametrize("question, reply, quote, supported", [
    (CURRENT, "My rating is seven out of ten.", "seven out of ten", True),
    ("How would you rate this symptom right now?", "My score is moderate.", "moderate", True),
    ("What is your pain score now?", "It is at most four out of ten right now.", "at most four out of ten", True),
    ("How bad is your pain right now?", "I would rate it at least 6/10.", "at least 6/10", True),
    (CURRENT, "My rating is about 3/10.", "about 3/10", True),
    (CURRENT, "My rating is about 3/10.", "3/10", False),
    (SCALE, "I would rate it around six.", "around six", True),
    (SCALE, "I would rate it around six.", "six", False),
    (CURRENT, "My rating is at most 3/10.", "3/10", False),
    (SCALE, "It's a seven.", "seven", True),
    (SCALE.replace(",", ""), "It’s a seven.", "seven", True),
    ("Rate your pain right now from 1 to 10.", "4.", "4", True),
    ("How bad is your headache on a scale from zero to ten?", "Zero.", "Zero", True),
    (SCALE, "Zero.", "Zero", False),
    (SCALE, "11.", "11", False),
    (SCALE, "11/10.", "11/10", False),
    (SCALE, "0/10.", "0/10", False),
    (SCALE, "It is minus three.", "three", False),
    (SCALE, "−3/10.", "−3/10", False),
    (SCALE, "Minus three out of ten.", "Minus three out of ten", False),
    ("How would you rate the coffee on a scale of one to ten?", "My headache is 0/10.", "0/10", True),
    ("On a scale from one to ten, how severe is your headache?", "My knee pain is 0/10.", "0/10", True),
    ("On a scale from one to ten, how severe is your headache?", "My headache is 0/10.", "0/10", False),
    (SCALE, "Three out of five.", "Three", False),
    (SCALE, "Seven days.", "Seven", False),
    (SCALE, "Seven or nine.", "Seven", False),
    (SCALE, "It could be seven.", "seven", False),
    (SCALE, "My dose is seven.", "seven", False),
    (SCALE, "My sister's rating is seven.", "seven", False),
    (SCALE, "My rating was severe yesterday.", "severe", False),
    (SCALE, "It’s a seven. That rating was yesterday.", "seven", False),
    (CURRENT, "Seven.", "Seven", False),
    ("How bad might your pain get tomorrow on a scale from one to ten?", "Seven.", "Seven", False),
    ("How bad was your pain yesterday on a scale from one to ten?", "My rating is severe.", "severe", False),
    ("How bad is your mother's pain on a scale from one to ten?", "Seven.", "Seven", False),
    ("How would you rate the effect of the medication on your pain?", "My rating is severe.", "severe", False),
    ("Rate your pain medication on a scale of one to ten.", "Seven.", "Seven", False),
    ("Rate the wait between one and ten.", "My rating is moderate.", "moderate", False),
])
def test_requested_value_and_source(question, reply, quote, supported):
    source = f"Doctor: {question}\nPatient: {reply}"
    check(source, "SEV", FieldName.SEVERITY, quote, source.rindex(quote), supported)


@pytest.mark.parametrize("turns", [
    "Other: Seven.",
    "Unknown: Seven.",
    "Other: Can you hear me?\nPatient: Seven.",
    "Patient: I need help with my coat. Seven.",
])
def test_turn_that_supplies_the_scale(turns):
    source = f"Doctor: {SCALE}\n{turns}"
    check(source, "SEV", FieldName.SEVERITY, "Seven", source.rindex("Seven"), False)


def test_an_unresolved_pronoun_does_not_inherit_an_unrelated_subject():
    source = "Doctor: How was the parking?\nPatient: The lot was crowded.\nDoctor: How bad is it now?\nPatient: My rating is severe."
    check(source, "SEV", FieldName.SEVERITY, "severe", source.index("severe"), False)


@pytest.mark.parametrize("first, second, question, conflict", [
    ("7.", "8.", SCALE, True),
    ("It’s a seven.", "My rating is eight out of ten.", SCALE, True),
    ("My rating is 7/10.", "Eight.", "Rate your pain from one to ten.", True),
    ("7/10.", "The pain is 8/10.", SCALE, True),
    ("7.", "My pain is 8/10.", SCALE, True),
    ("7.", "My wrist pain is 8/10.", SCALE, False),
    ("My rating is mild.", "It is severe.", SCALE, True),
    ("Seven.", "7/10.", SCALE, False),
    ("About seven.", "Eight.", SCALE, False),
    ("At least six.", "Eight.", SCALE, False),
    ("Eight.", "Moderate.", SCALE, False),
    ("Eight.", "Three.", "Rate your headache from one to ten.", False),
])
def test_inherited_ratings_compared_at_same_scope(first, second, question, conflict):
    source = f"Doctor: {SCALE}\nPatient: {first}\nDoctor: {question}\nPatient: {second}"
    doc = transcript_source(NanoInput("rating", source))
    field = FieldName.SEVERITY
    pairs = detect_grounded_conflicts(doc, field)
    assert bool(pairs) == conflict
    assert any(key.startswith(":C[") for key in grounded_candidate_choices(source, field, set())) == conflict
    assert any(code == "C" for code, _ in build_grounded_english_menu(doc, field, set()).values()) == conflict
    if conflict:
        left, right = pairs[0]
        assert left.text in first and right.text in second
        for span in (left, right):
            state, evidence = _grounded_claim(source, field, f"SEV:S[{span.text}]")
            assert state == "conflicting" and evidence == [left.text, right.text]


@pytest.mark.parametrize("reply, quote, supported", [
    ("7. I was wrong.", "7", False),
    ("It's a seven. I was mistaken.", "seven", False),
    ("7.\nPatient: I was wrong.", "7", False),
    ("Seven. I was mistaken about the appointment.", "Seven", True),
    ("7.\nPatient: I took aspirin. I was wrong.", "7", True),
    ("7. I took aspirin.\nPatient: I was wrong.", "7", True),
    ("7.\nDoctor: What about your medication?\nPatient: I was wrong.", "7", True),
    ("7.\nOther: I was wrong.", "7", True),
    ("7.\nPatient: My brother was wrong.", "7", True),
    ("7.\nPatient: If I was mistaken, I would call.", "7", True),
    ("7. I was wrong.\nDoctor: Rate your pain from one to ten.\nPatient: Eight.", "Eight", True),
])
def test_withdrawing_a_prompted_rating(reply, quote, supported):
    source = f"Doctor: {SCALE}\nPatient: {reply}"
    check(source, "SEV", FieldName.SEVERITY, quote, source.rindex(quote), supported)
