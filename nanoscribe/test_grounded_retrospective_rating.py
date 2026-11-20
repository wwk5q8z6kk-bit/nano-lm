"""A patient's immediate retrospective qualification changes the time of a rating."""
import pytest

from nano_ai.contract import FieldName
from nanoscribe.test_grounded_followup import check


RATING = "Doctor: How strong is it now?\nPatient: I would say 7/10."


@pytest.mark.parametrize("question,reply,quote,accepted", [
    ("How severe is the pain now?", "Three days ago my pain was 7/10.", "7/10", False),
    ("How severe was the pain at its peak?", "7/10.", "7/10", False),
    ("How severe was the pain at its peak?", "Right now my pain is 2/10.", "2/10", True),
])
def test_reported_dates_and_question_scope(question, reply, quote, accepted):
    source = f"Doctor: {question}\nPatient: {reply}"
    check(source, "SEV", FieldName.SEVERITY, quote, source.index(quote), accepted)


@pytest.mark.parametrize("report,quote,accepted", [
    ("Patient: Wait, that was on the bus. Right now my shaking is 3/10.", "7/10", False),
    ("Patient: Wait, that was on the bus. Right now my shaking is 3/10.", "3/10", True),
    ("Patient: That score was yesterday. My tremor is 2/10 now.", "7/10", False),
    ("Patient: That score was yesterday. My tremor is 2/10 now.", "2/10", True),
    ("Patient: That rating was before the visit. At the moment my chills are 4/10.", "7/10", False),
    ("Patient: That rating was before the visit. At the moment my chills are 4/10.", "4/10", True),
    ("Other: Wait, that was on the bus. Right now my shaking is 3/10.", "7/10", True),
    ("Other: Wait, that was on the bus. Right now my shaking is 3/10.", "3/10", False),
    ("Doctor: That score was yesterday.", "7/10", True),
    ("Patient: Was that score yesterday?", "7/10", True),
    ("Patient: The bus ride was 3/10. That was yesterday.", "7/10", True),
    ("Patient: I quoted the speaker: “That score was yesterday.”", "7/10", True),
    ("Patient: My chills are 4/10 now, if I imagine them.", "4/10", False),
    ("Patient: The ticket was on the bus. My coffee is 3/10.", "7/10", True),
    ("Patient: My energy was 3/10 yesterday.", "7/10", True),
    ("Patient: My shaking was 3/10 yesterday.", "3/10", False),
    ("Patient: My shaking is 3/10 if I skip eating.", "3/10", False),
    ("Patient: If my shaking is 3/10, I can write.", "3/10", False),
    ("Patient: My sister said, 'right now my shaking is 3/10.'", "3/10", False),
    ("Patient: My sister said, 'That was on the bus.'", "7/10", True),
    ("Patient: If that score was yesterday, I would have to check.", "7/10", True),
    ("Patient: I bought a ticket. That was on the bus.", "7/10", True),
    ("Patient: Wait, that was on my packing list.", "7/10", True),
    ("Patient: My shaking washing machine is 3/10 now.", "3/10", False),
    ("Patient: My mother's shaking is 3/10 now.", "3/10", False),
])
def test_attribution_and_replacement(report, quote, accepted):
    source = RATING + "\n" + report
    check(source, "SEV", FieldName.SEVERITY, quote, source.index(quote), accepted)


def test_a_different_subject_resets_the_reference_even_in_one_sentence():
    source = ("Patient: My headache is 7/10 but the car ride was bumpy. "
              "That was on the bus.")
    check(source, "SEV", FieldName.SEVERITY, "7/10", source.index("7/10"), True)


@pytest.mark.parametrize("quote,accepted", [("8/10", False), ("4/10", True)])
def test_same_patient_turn(quote, accepted):
    source = ("Doctor: Rate your pain now.\n"
              "Patient: 8/10. That was last night. Currently my shaking is 4/10.")
    check(source, "SEV", FieldName.SEVERITY, quote, source.index(quote), accepted)


@pytest.mark.parametrize("report,quote,accepted", [
    ("Yesterday on the bus my pain was 2/10.", "2/10", False),
    ("Yesterday, my headache was 2/10.", "2/10", False),
    ("Last Friday my wrist pain was 2/10.", "2/10", False),
    ("My pain was 2/10 yesterday.", "2/10", False),
    ("My headache felt severe last night.", "severe", False),
    ("Today my headache is 2/10.", "2/10", True),
    ("My pain is 2/10 today.", "2/10", True),
    ("My pain is 2/10.", "2/10", True),
    ("Earlier I took the bus, but my headache is 2/10 now.", "2/10", True),
    ("Yesterday my pain was 7/10, but now my pain is 2/10.", "7/10", False),
    ("Yesterday my pain was 7/10, but now my pain is 2/10.", "2/10", True),
    ("My pain is 2/10 today, but yesterday my pain was 7/10.", "2/10", True),
    ("My pain is 2/10 today, but yesterday my pain was 7/10.", "7/10", False),
    ("My pain was 2/10 when I checked just now.", "2/10", True),
])
def test_dated_measurement_is_local_to_the_selected_rating(report, quote, accepted):
    source = "Doctor: How severe is the pain now?\nPatient: " + report
    check(source, "SEV", FieldName.SEVERITY, quote, source.index(quote), accepted)


@pytest.mark.parametrize("source,accepted", [
    ("Patient: My tremor is 5/10. That rating was two days ago. It is gone now.", False),
    ("Patient: My shaking is 5/10. That score was this morning. It has settled completely now.", False),
    ("Patient: My shaking is 5/10. That score was earlier today. My shaking has stopped now.", False),
    ("Doctor: Read the printed example aloud.\nPatient: My shaking is 5/10.", False),
    ("Doctor: Please repeat the sample sentence.\nPatient: My headache is 5/10.", False),
    ("Doctor: Read me your own current score.\nPatient: My headache is 5/10.", True),
    ("Patient: I am still at work. My headache is 5/10.\nPatient: That was at work. I am still here and the score has not changed.", True),
    ("Patient: My headache is 5/10. That was at school.", True),
    ("Patient: My headache is 5/10. That was at school. My coffee is 2/10 now.", True),
    ("Patient: My headache is 5/10. That was at school. Right now my nausea is 2/10.", True),
    ("Patient: My headache is 5/10. That was at school. My sister's pain is 2/10 now.", True),
    ("Patient: My headache is 5/10. That was at school. Right now it is 2/10.", False),
])
def test_context_of_timing_and_recitation(source, accepted):
    check(source, "SEV", FieldName.SEVERITY, "5/10", source.index("5/10"), accepted)


@pytest.mark.parametrize("turns,quote,accepted", [
    ("Unknown: The umbrella is upstairs.\nPatient: That score was yesterday.", "7/10", False),
    ("Other: The lobby door is closed.\nUnknown: The umbrella is upstairs.\n"
     "Patient: That rating was two days ago.", "7/10", False),
    ("Unknown: The umbrella is upstairs.\nPatient: That was at work.\n"
     "Unknown: The door is closed.\nPatient: My neck pain is 2/10 now.", "7/10", False),
    ("Unknown: The umbrella is upstairs.\nPatient: That was at work.\n"
     "Unknown: The door is closed.\nPatient: My neck pain is 2/10 now.", "2/10", True),
    ("Unknown: The umbrella is upstairs.\nPatient: That was at work.", "7/10", True),
    ("Unknown: The umbrella is upstairs.\nPatient: I took the bus. That score was yesterday.", "7/10", True),
    ("Unknown: The umbrella is upstairs.\nDoctor: When did you rate the shuttle?\n"
     "Patient: That score was yesterday.", "7/10", True),
    ("Unknown: The umbrella is upstairs.\nOther: That score was yesterday.", "7/10", True),
    ("Unknown: The umbrella is upstairs.\nOther: I gave the parking app an 8/10.\n"
     "Patient: That rating was yesterday.", "7/10", True),
    ("Unknown: The umbrella is upstairs. That was my neck pain, not the patient's.\n"
     "Patient: That score was yesterday.", "7/10", False),
    ("Unknown: When can I collect the ticket?\nPatient: That was yesterday.", "7/10", True),
    ("Unknown: Where is the lift?\nPatient: That score was yesterday.", "7/10", False),
    ("Unknown: When can I collect the ticket?\nPatient: That number was yesterday.", "7/10", True),
])
def test_rating_correction_across_logistical_asides(turns, quote, accepted):
    source = "Patient: My neck pain is 7/10.\n" + turns
    check(source, "SEV", FieldName.SEVERITY, quote, source.index(quote), accepted)
