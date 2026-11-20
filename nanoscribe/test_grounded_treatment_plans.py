"""An answer to a treatment plan does not document completed or current use."""
import pytest

from nano_ai.contract import FieldState
from nanoscribe.test_grounded_substance_ownership import ALG, MED, observed


@pytest.mark.parametrize("question", [
    "What medicine will you take tomorrow?",
    "Which medication are you planning to try?",
    "Which medication do you intend to start?",
    "Which treatment might you use next?",
    "What medicine are you going to take?",
    "Which tablets would you consider taking?",
    "What medicine are you taking tomorrow?",
    "Tomorrow, which medicine are you taking?",
    "What medication do you take starting tomorrow?",
    "Which new medicine did the clinic recommend?",
    "What medication would you like prescribed?",
    "Which treatment are you asking to change?",
    "Which medicine do you plan on using?",
    "What is the plan for your next medication?",
    "Which drug are you supposed to start?",
    "Any medication planned for next week?",
    "Which medication will you be on next week?",
])
@pytest.mark.parametrize("answer", ["Aspirin.", "Aspirin for my headache."])
def test_bare_reply_to_plan_does_not_certify_use(question, answer):
    claim, direct, compact, english = observed(f"Doctor: {question}\nPatient: {answer}", MED, "Aspirin")
    assert claim["value"] is None and direct is None
    assert not any(choice.startswith((":S[", ":C[")) for choice in compact)
    assert not any(choice.startswith((" REPORTED[", " CONTRADICTED[")) for choice in english)


@pytest.mark.parametrize("question,answer,field,value", [
    ("Which medication did you take yesterday?", "Aspirin.", MED, "Aspirin"),
    ("Which medication do you currently take?", "Aspirin.", MED, "Aspirin"),
    ("Which medication are you on now?", "Aspirin.", MED, "Aspirin"),
    ("Which medication did you take before tomorrow's scan?", "Aspirin.", MED, "Aspirin"),
    ("Which medication are you taking now to prepare for next week's test?", "Aspirin.", MED, "Aspirin"),
    ("Would you tell me what medicine you took?", "Aspirin.", MED, "Aspirin"),
    ("Before we plan the next dose, what medication did you take?", "Aspirin.", MED, "Aspirin"),
    ("What medicine will you take tomorrow?", "I took aspirin yesterday.", MED, "aspirin"),
    ("Which medicine do you plan to start?", "I am currently taking ibuprofen.", MED, "ibuprofen"),
    ("Which medicine will you take tomorrow?", "My current medication is warfarin.", MED, "warfarin"),
    ("Which medication do you intend to start?", "I am allergic to latex.", ALG, "latex"),
    ("Which medication do you intend to start?", "Aspirin. I took ibuprofen this morning.", MED, "ibuprofen"),
])
def test_actual_report_has_its_own_source(question, answer, field, value):
    claim, direct, compact, english = observed(f"Doctor: {question}\nPatient: {answer}", field, value)
    assert claim["value"] == value and direct is FieldState.SUPPORTED
    assert f":S[{value}]" in compact and f" REPORTED[{value}]" in english


def test_plan_with_later_denial_does_not_invent_conflict():
    report = "Doctor: Which medication are you planning to try?\nPatient: Aspirin.\nPatient: I do not take any medication."
    claim, direct, compact, english = observed(report, MED, "Aspirin")
    assert claim["state"] != "conflicting" and direct is None
    assert not any(choice.startswith((":S[", ":C[")) for choice in compact)
    assert not any(choice.startswith((" REPORTED[", " CONTRADICTED[")) for choice in english)


@pytest.mark.parametrize("reply", [
    "I am taking aspirin tomorrow.",
    "Tomorrow I am taking aspirin.",
    "I take aspirin starting next week.",
    "I am taking aspirin for my headache tomorrow.",
    "I am taking aspirin in three days.",
])
def test_explicitly_scheduled_use_cannot_certify_any_part_of_the_future_report(reply):
    report = f"Doctor: Which medication will you take tomorrow?\nPatient: {reply}"
    for proposal in ("aspirin", reply[reply.index("aspirin"):].rstrip(".")):
        claim, direct, compact, english = observed(report, MED, proposal)
        assert claim["value"] is None and direct is None
        assert not any(x.startswith(":S[") for x in compact)
        assert not any(x.startswith(" REPORTED[") for x in english)


@pytest.mark.parametrize("reply", [
    "I am currently taking aspirin until tomorrow.",
    "I took aspirin yesterday.",
    "I take aspirin daily and I will take it tomorrow.",
    "I am taking aspirin before tomorrow's appointment.",
    "I took aspirin for my headache. I am taking ibuprofen tomorrow.",
])
def test_scheduled_action_preserves_a_separate_actual_medication(reply):
    report = f"Doctor: What medicine will you take tomorrow?\nPatient: {reply}"
    claim, direct, compact, english = observed(report, MED, "aspirin")
    assert claim["value"] == "aspirin" and direct is FieldState.SUPPORTED
    assert ":S[aspirin]" in compact and " REPORTED[aspirin]" in english
    assert not any("ibuprofen" in x for x in (*compact, *english))


@pytest.mark.parametrize("reply", [
    "I am currently taking aspirin for my procedure tomorrow.",
    "I am taking aspirin for my scheduled dental surgery next week.",
    "I am taking aspirin ahead of my appointment tomorrow.",
    "I am taking aspirin because my operation is tomorrow.",
    "I am taking aspirin for my procedure that is scheduled for tomorrow.",
    "I take aspirin for my outpatient procedure in three days.",
])
def test_future_time_can_belong_to_the_reason_for_current_use(reply):
    claim, direct, compact, english = observed(f"Patient: {reply}", MED, "aspirin")
    assert claim["value"] == "aspirin" and direct is FieldState.SUPPORTED
    assert ":S[aspirin]" in compact and " REPORTED[aspirin]" in english


@pytest.mark.parametrize("reply", [
    "I am taking aspirin tomorrow for my procedure.",
    "I am taking aspirin starting tomorrow for my appointment.",
    "I am taking aspirin for my procedure starting tomorrow.",
    "I am taking aspirin tomorrow morning before my surgery.",
])
def test_future_administration_is_still_planned_before_a_procedure(reply):
    claim, direct, compact, english = observed(f"Patient: {reply}", MED, "aspirin")
    assert claim["value"] is None and direct is None
    assert not any(x.startswith(":S[") for x in compact)
    assert not any(x.startswith(" REPORTED[") for x in english)
