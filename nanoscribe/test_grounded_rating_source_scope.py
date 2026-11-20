"""The conversational source and anatomical site determine which rating an utterance owns."""
import pytest
from nano_ai.contract import FieldName
from nanoscribe.test_grounded_followup import check


@pytest.mark.parametrize("source,quote,accepted", [
    ("Doctor: Read the sentence on the printed card aloud.\nPatient: My shaking is 5/10.", "5/10", False),
    ("Doctor: Please read the line on that card.\nPatient: My headache is 5/10.", "5/10", False),
    ("Doctor: Read the printed example silently, then tell me your actual headache rating.\nPatient: My headache is 5/10 right now.", "5/10", True),
    ("Doctor: Read the example aloud.\nPatient: I cannot see the example. My headache is 5/10 right now.", "5/10", True),
    ("Doctor: Read the example aloud.\nPatient: I can see the example. My headache is 5/10 right now.", "5/10", False),
    ("Doctor: Read the example aloud.\nPatient: My headache is 5/10 right now.", "5/10", False),
    ("Patient: My tremor is 5/10. That rating was yesterday afternoon. The tremor is gone now.", "5/10", False),
    ("Patient: My left wrist pain is 5/10. That was at work. Right now my right knee pain is 2/10.", "5/10", True),
    ("Patient: The headache is 5/10. That was at work. Right now my tremor is 2/10.", "5/10", True),
    ("Patient: My left wrist pain is 5/10. That was at work. Right now my left wrist pain is 2/10.", "5/10", False),
    ("Patient: My left wrist pain is 5/10. That was at work. Right now my right wrist pain is 2/10.", "5/10", True),
])
def test_local_statement_owner(source, quote, accepted):
    check(source, "SEV", FieldName.SEVERITY, quote, source.index(quote), accepted)


@pytest.mark.parametrize("question,accepted", [
    ("Read the example aloud. How severe is your pain now?", True),
    ("Read the card. How bad is your pain?", True),
    ("Repeat the sample, then what is your current pain score?", True),
    ("Please read the script. How severe was your pain yesterday?", False),
    ("Read the example aloud. How bad is your mother's pain?", False),
    ('Read the sentence aloud: "How severe is your pain now?"', False),
    ("Read the example aloud. The example asks how severe your pain is now.", False),
    ("How severe is your pain now? Read the example aloud.", False),
])
def test_personal_rating_question_after_reading(question, accepted):
    source = f"Doctor: {question}\nPatient: My pain is 5/10."
    check(source, "SEV", FieldName.SEVERITY, "5/10", source.index("5/10"), accepted)


@pytest.mark.parametrize("earlier,later,accepted", [
    ("My headache is about", "my right knee pain", True),
    ("My headache is about", "my headache", False),
    ("My left wrist pain feels roughly", "my right wrist pain", True),
    ("My left wrist pain feels roughly", "my left wrist pain", False),
    ("I would rate my headache at", "my knee pain", True),
    ("I would rate my headache at", "my headache", False),
    ("I would say", "my knee pain", False),
])
def test_setting_compares_the_original_named_symptom(earlier, later, accepted):
    source = f"Doctor: How bad is it now?\nPatient: {earlier} 7/10. That was at work. Right now {later} is 2/10."
    # The constrained choices preserve the modifier as part of the evidence.
    modifier = earlier.split()[-1]
    quote = f"{modifier} 7/10" if modifier in ("about", "roughly") else "7/10"
    check(source, "SEV", FieldName.SEVERITY, quote, source.index(quote), accepted)


@pytest.mark.parametrize("report,accepted", [
    ("The headache that started yesterday was 5/10 when I checked just now.", True),
    ("My wrist pain which began last Friday was 5/10 when I just rechecked it.", True),
    ("The headache that started two days ago was 5/10 when I measured it a moment ago.", True),
    ("The headache that started yesterday was 5/10 when I checked it yesterday.", False),
    ("The headache that started yesterday was 5/10 when she checked just now.", False),
    ("Yesterday on the bus the headache was 5/10.", False),
    ("Last Friday the headache that started yesterday was 5/10 when I checked just now.", False),
    ("The headache that started yesterday was 5/10 last night.", False),
])
def test_dated_onset_and_measurement(report, accepted):
    source = f"Doctor: How bad is your pain now?\nPatient: {report}"
    check(source, "SEV", FieldName.SEVERITY, "5/10", source.index("5/10"), accepted)


@pytest.mark.parametrize("question,lead,later,accepted", [
    ("How severe is your headache now?", "", "my knee pain", True),
    ("How severe is your headache now?", "", "my headache", False),
    ("How bad is your left wrist pain?", "I would say ", "my right wrist pain", True),
    ("How bad is your left wrist pain?", "I would say ", "my left wrist pain", False),
    ("How severe is the nausea?", "", "my knee pain", True),
    ("How severe is it now?", "", "my knee pain", False),
    ("How bad is the pain now?", "", "my knee pain", False),
    ("How bad is your headache?", "My wrist pain is ", "my wrist pain", False),
])
def test_bare_answer_inherits_requested_symptom(question, lead, later, accepted):
    source = f"Doctor: {question}\nPatient: {lead}7/10. That was at work. Right now {later} is 2/10."
    check(source, "SEV", FieldName.SEVERITY, "7/10", source.index("7/10"), accepted)
