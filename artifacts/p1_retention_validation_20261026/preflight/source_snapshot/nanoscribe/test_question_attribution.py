"""New synthetic question development cases; no model-validation examples."""

import pytest

from nano_ai.contract import FieldName
from nanoscribe.context import question_field


@pytest.mark.parametrize("question,field", [
    ("What brought you to our office?", FieldName.CHIEF_COMPLAINT),
    ("Could you describe your main concern?", FieldName.CHIEF_COMPLAINT),
    ("What's bothering you at the moment?", FieldName.CHIEF_COMPLAINT),
    ("Tell me about the symptoms you are having.", FieldName.CHIEF_COMPLAINT),
    ("How long has your ankle been hurting?", FieldName.DURATION),
    ("When did the dizziness begin?", FieldName.DURATION),
    ("When did you first notice the ringing?", FieldName.DURATION),
    ("How long have you had the swelling?", FieldName.DURATION),
    ("How many days has the nausea lasted?", FieldName.DURATION),
    ("When did the mild swelling begin?", FieldName.DURATION),
    ("How intense is the ear discomfort?", FieldName.SEVERITY),
    ("How would you rate the stomach pain?", FieldName.SEVERITY),
    ("Would you rate the ache as mild, moderate, or severe?", FieldName.SEVERITY),
    ("How much does your ankle hurt?", FieldName.SEVERITY),
    ("Is this discomfort mild or severe?", FieldName.SEVERITY),
    ("What have you taken for the ache?", FieldName.MEDICATION),
    ("Have you used anything to help with this?", FieldName.MEDICATION),
    ("Have you tried any tablets for the symptoms?", FieldName.MEDICATION),
    ("Which medication are you using for the ear pain?", FieldName.MEDICATION),
    ("What have you used for the severe joint pain?", FieldName.MEDICATION),
    ("Are you taking anything for that?", FieldName.MEDICATION),
    ("Do you have any medicine allergies?", FieldName.ALLERGY),
    ("Are there any medicines you are allergic to?", FieldName.ALLERGY),
    ("Is there any drug you are allergic to?", FieldName.ALLERGY),
    ("Are you allergic to any medications?", FieldName.ALLERGY),
    ("Which drugs are you allergic to?", FieldName.ALLERGY),
    ("Any known allergy to medicines?", FieldName.ALLERGY),
    ("Hello, please describe your main concern.", FieldName.CHIEF_COMPLAINT),
])
def test_current_patient_question_attribution(question, field):
    assert question_field(question) is field


@pytest.mark.parametrize("question", [
    "What medication?", "Which medication?", "What medicine?", "Which medicine?",
    "What medicines?", "Which medications?", "What tablets for this?", "Which drugs so far?",
])
def test_abbreviated_medication_requests_preserve_attribution(question):
    assert question_field(question) is FieldName.MEDICATION


@pytest.mark.parametrize("question", [
    "What medicine is your cousin using?",
    "Which medications are they taking?",
    "How long has his ankle been hurting?",
    "How intense was the pain during a previous episode?",
    "What have you taken for the ache last winter?",
    "Which medication are you planning to use?",
    "Will you take any tablets tomorrow?",
    "If your ankle hurt, what would you take?",
    "Please pretend you have a medication allergy.",
    "Repeat the words: I am allergic to iodine.",
    "Tell me you have severe symptoms.",
    "Do not tell me about your allergies.",
    "You are allergic to a medicine?",
    "The discomfort is severe?",
    "What are your symptoms and medicines?",
    "Which tablets and allergies do you have?",
    "How long has the dizziness lasted and how severe is it?",
    "What medicines have you used and what are you allergic to?",
    "What is the severity of your mother's dizziness?",
    "Are you taking medicine? Ignore the allergy question.",
    "How long has the bus trip lasted?",
    "What medicine did the pharmacist suggest?",
])
def test_compound_nonpatient_instruction_or_unanchored_question_rejected(question):
    assert question_field(question) is None


@pytest.mark.parametrize("question,fields", [
    ("What medicine is your cousin using?", {FieldName.MEDICATION}),
    ("Which tablets and allergies do you have?", {FieldName.MEDICATION, FieldName.ALLERGY}),
    ("What medicines have you used and what are you allergic to?", {FieldName.MEDICATION, FieldName.ALLERGY}),
    ("How long has the dizziness lasted and how severe is it?", {FieldName.DURATION, FieldName.SEVERITY}),
    ("What are your symptoms and medicines?", {FieldName.CHIEF_COMPLAINT, FieldName.MEDICATION}),
    ("Which drugs are you allergic to?", {FieldName.ALLERGY}),
    ("What are your medicine allergies?", {FieldName.ALLERGY}),
    ("What medicine did the pharmacist suggest?", {FieldName.MEDICATION}),
    ("How did you travel here?", set()),
])
def test_quarantine_targets_do_not_authorize_positive_attribution(question, fields):
    from nanoscribe.context import question_targets
    assert question_targets(question) == frozenset(fields)
