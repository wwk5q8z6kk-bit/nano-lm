"""New injected request-scope cases; independent validation data stays unseen."""

import pytest

from nano_ai.contract import FieldName
from nanoscribe.context import question_field, question_targets


@pytest.mark.parametrize("question,field", [
    ("What did you come here for?", FieldName.CHIEF_COMPLAINT),
    ("What could I help you with today?", FieldName.CHIEF_COMPLAINT),
    ("What concern are you having now?", FieldName.CHIEF_COMPLAINT),
    ("Tell me what's going on.", FieldName.CHIEF_COMPLAINT),
    ("Please tell me how severe the pain is.", FieldName.SEVERITY),
    ("How long ago did the tingling first appear?", FieldName.DURATION),
    ("How long has your wrist pain been troubling you?", FieldName.DURATION),
    ("How far back does this discomfort go?", FieldName.DURATION),
    ("When was the numbness first noticeable?", FieldName.DURATION),
    ("How do you rate the itching?", FieldName.SEVERITY),
    ("How much is that problem bothering you now?", FieldName.SEVERITY),
    ("How intense does the shoulder ache seem?", FieldName.SEVERITY),
    ("Which tablets are you currently taking for the discomfort?", FieldName.MEDICATION),
    ("Are you currently using anything to help with that?", FieldName.MEDICATION),
    ("Do you use any medicines for your cough?", FieldName.MEDICATION),
    ("What are your known allergies?", FieldName.ALLERGY),
    ("Is there anything that you're allergic to?", FieldName.ALLERGY),
    ("Are there drugs you have an allergy to?", FieldName.ALLERGY),
    ("Which drugs do you react badly to?", FieldName.ALLERGY),
    ("Do any medicines cause you allergic reactions?", FieldName.ALLERGY),
])
def test_whole_request_operators_match_only_the_requested_field(question, field):
    assert question_field(question) is field
    assert question_targets(question) == frozenset({field})


@pytest.mark.parametrize("prefix", [
    "Thank you. ", "Okay, ", "Good afternoon! ", "All right. ",
    "Thanks. Now, ", "To start, ", "Hello, please ",
    "Would you please tell me ", "Could you tell me ",
])
def test_bounded_discourse_and_polite_request_composition(prefix):
    question = prefix + "how long has the ringing been present?"
    assert question_field(question) is FieldName.DURATION
    assert question_targets(question) == frozenset({FieldName.DURATION})


@pytest.mark.parametrize("question", [
    "Thanks for telling me about your aunt. How long has the ringing been present?",
    "Think about your old episode. How do you rate the itching?",
    "Thank you. How long ago did her tingling first appear?",
    "Okay, what did your wife come here for?",
    "Please repeat: What could I help you with today?",
    '"How much is that problem bothering you now?"',
    "Okay, is there anything that she is allergic to?",
    "Would you please tell me which tablets you will take tomorrow?",
    "Would you please tell me how long has the ringing been present and how intense is it?",
    "How long ago did the school trip begin?",
    "Do any medicines cause your son allergic reactions?",
    "If you reacted badly to a medicine, what would you do?",
    "What did you come here for? Which tablets did you take?",
    "How do you rate the itching or your medication?",
])
def test_new_operators_do_not_remove_scope_or_compound_hazards(question):
    assert question_field(question) is None


@pytest.mark.parametrize("question,expected", [
    ("Which medicines does your partner react badly to?", {FieldName.ALLERGY}),
    ("What drugs do you take and react badly to?", {FieldName.MEDICATION, FieldName.ALLERGY}),
    ("Is there anything your sister is allergic to?", {FieldName.ALLERGY}),
])
def test_reaction_request_targets_do_not_quarantine_medication_use(question, expected):
    assert question_field(question) is None
    assert question_targets(question) == frozenset(expected)
