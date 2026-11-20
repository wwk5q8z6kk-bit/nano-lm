"""A clinical interval or substance must belong to the reported event."""
import pytest

from nano_ai.contract import FieldName
from nanoscribe.test_grounded_followup import check


@pytest.mark.parametrize("question,reply,object_text,supported", [
    ("", "I took a taxi to the clinic.", "taxi", False),
    ("Which medication did you try?", "I took the train to pick up my medication.", "train", False),
    ("What brings you in?", "I tried a new commute.", "new commute", False),
    ("", "I used my calendar for medication reminders.", "calendar", False),
    ("", "I took a picture of my medication.", "picture", False),
    ("Which medicine did you use?", "I tried yoga for my headache.", "yoga", False),
    ("", "I applied for a job.", "job", False),
    ("", "I took a short walk for my headache.", "short walk", False),
    ("", "I took aspirin on the bus.", "aspirin", True),
    ("", "I used nasal spray for the headache.", "nasal spray", True),
    ("", "I took Xytal 20 mg.", "Xytal 20 mg", True),
    ("What medicine did you try?", "I tried Falpren.", "Falpren", True),
])
def test_action_with_a_reported_object(question, reply, object_text, supported):
    source = (f"Doctor: {question}\n" if question else "") + f"Patient: {reply}"
    check(source, "MED", FieldName.MEDICATION, object_text, source.rindex(object_text), supported)


@pytest.mark.parametrize("reply,interval,supported", [
    ("My headache started two days ago.\nPatient: I took aspirin for one week.", "one week", False),
    ("My headache started two days ago.\nPatient: I took aspirin for one week.", "two days ago", True),
    ("I baked a cake for four hours.", "four hours", False),
    ("My holiday lasted seven days.", "seven days", False),
    ("I spent three hours organizing receipts.", "three hours", False),
    ("I drove for four hours.", "four hours", False),
    ("I was on the train for four hours.", "four hours", False),
    ("I waited in the clinic for two hours.", "two hours", False),
    ("I was working for six hours.", "six hours", False),
    ("For two days I was driving.", "two days", False),
    ("I used my phone but my headache kept me awake for four hours.", "four hours", True),
    ("I took aspirin but the pain lasted for two days.", "two days", True),
    ("My headache started after work and lasted two days.", "two days", True),
    ("I have had a headache for two days.", "two days", True),
])
def test_interval_belongs_to_the_episode(reply, interval, supported):
    source = "Patient: " + reply
    check(source, "DUR", FieldName.DURATION, interval, source.rindex(interval), supported)


@pytest.mark.parametrize("prompt", ["Tell me what medicine you took.", "List your medicines.", "Describe your medication."])
@pytest.mark.parametrize("count,supported", [("question", True), ("two questions", False)])
def test_an_imperative_request_begins_the_next_answer(prompt, count, supported):
    source = (f"Doctor: Answer the next {count} about your child, not yourself.\n"
              "Doctor: Any allergies?\nPatient: Nickel.\n"
              f"Doctor: {prompt}\nPatient: I took ibuprofen.")
    check(source, "MED", FieldName.MEDICATION, "ibuprofen", source.rindex("ibuprofen"), supported)
