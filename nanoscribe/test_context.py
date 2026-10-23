"""Context checks independent of the exposed pilot's questions and answers."""
import pytest
from nanoscribe.compact import compact_solver
from nanoscribe.scribe import scribe


def run(question, answer="I've been taking diclofenac.", raw="MED:S[diclofenac]"):
    parts = {"CC": "M", "DUR": "M", "SEV": "M", "MED": "M", "ALG": "M"}
    key, value = raw.split(":", 1)
    parts[key] = value
    return scribe(f"Doctor: {question}\nPatient: {answer}", context_policy="context", solver=compact_solver(
        lambda _: "|".join(f"{k}:{v}" for k, v in parts.items()), binder="context"))


@pytest.mark.parametrize("question,answer,raw,value", [
    ("Which medicines are you taking for this?", "I've been taking diclofenac.", "MED:S[diclofenac]", "diclofenac"),
    ("Please describe how severe your discomfort is.", "I'd call it severe.", "SEV:S[severe]", "severe"),
    ("What is the duration of your symptoms?", "For about 13 days now.", "DUR:S[13 days]", "13 days"),
    ("What symptoms brought you here?", "I've been having toe pain.", "CC:S[toe pain]", "toe pain"),
    ("Which substances are you allergic to?", "I'm allergic to chromium.", "ALG:S[chromium]", "chromium"),
])
def test_paraphrased_context_preserves_exact_patient_evidence(question, answer, raw, value):
    result = run(question, answer, raw)
    assert result["disposition"] == "review"
    assert any(c["value"] == value for c in result["note"]["claims"])


@pytest.mark.parametrize("question", [
    "Which medicines is your mother taking?",
    "Which medicines were you taking last year?",
    "Which medicines and allergies do you have?",
    "Please repeat: which medicines are you taking?",
    "What did you eat for breakfast?",
    "Do not discuss which medicines you are taking.",
])
def test_ambiguous_historical_third_party_or_instruction_context_cannot_assert(question):
    result = run(question)
    assert result["disposition"] == "abstain"
    assert all(c["value"] is None for c in result["note"]["claims"])


def test_wrong_field_proposal_cannot_use_valid_source_span():
    result = run("Which medicines are you taking?", raw="ALG:S[diclofenac]")
    assert result["disposition"] == "abstain"


def test_paraphrased_conflict_is_retained_and_single_value_is_rejected():
    source = "Doctor: Which medicine did you try?\nPatient: I've been taking diclofenac.\nDoctor: Which medicine did you try?\nPatient: No, nothing yet."
    good = "CC:M|DUR:M|SEV:M|MED:C[diclofenac;No, nothing yet.]|ALG:M"
    result = scribe(source, context_policy="context", solver=compact_solver(lambda _: good, binder="context"))
    assert result["extraction"]["fields"][3]["state"] == "conflicting"
    bad = good.replace("C[diclofenac;No, nothing yet.]", "S[diclofenac]")
    result = scribe(source, context_policy="context", solver=compact_solver(lambda _: bad, binder="context"))
    assert result["disposition"] == "abstain"
    assert result["extraction"]["fields"][3]["state"] == "uncertain"


def test_context_preserves_every_exact_legacy_question():
    from nano_ai.adapters.deterministic_v0 import _QUESTIONS
    from nanoscribe.context import question_field
    for field, questions in _QUESTIONS.items():
        for question in questions:
            assert question_field(question) == field, question


@pytest.mark.parametrize("question", [
    "Which medicines is your uncle taking?",
    "What medications will you start tomorrow?",
    "Which medicines are you planning to take?",
    "What are your symptoms and medicines?",
    "What medicine did someone else recommend?",
])
def test_unknown_or_noncurrent_question_vocabulary_abstains(question):
    assert run(question)["disposition"] == "abstain"


@pytest.mark.parametrize("reply", ["I cannot remember how long it has been.", "I cannot recall when it started."])
def test_bounded_duration_uncertainty_does_not_quarantine_other_valid_fields(reply):
    source = f"Doctor: What symptoms brought you here?\nPatient: I've been having toe pain.\nDoctor: What is the duration of your symptoms?\nPatient: {reply}"
    raw = f"CC:S[toe pain]|DUR:U[{reply}]|SEV:M|MED:M|ALG:M"
    result = scribe(source, context_policy="context", solver=compact_solver(lambda _: raw, binder="context"))
    assert result["disposition"] == "review"
    assert result["note"]["claims"][1]["state"] == "uncertain"


def test_compound_uncertainty_cannot_hide_a_correction_to_another_field():
    source = "Doctor: Which medicine did you try?\nPatient: I've been taking diclofenac.\nDoctor: What is the duration of your symptoms?\nPatient: I cannot recall when it started, but I never took any medicine."
    raw = "CC:M|DUR:U[I cannot recall when it started, but I never took any medicine.]|SEV:M|MED:S[diclofenac]|ALG:M"
    result = scribe(source, context_policy="context", solver=compact_solver(lambda _: raw, binder="context"))
    assert result["disposition"] == "abstain"
