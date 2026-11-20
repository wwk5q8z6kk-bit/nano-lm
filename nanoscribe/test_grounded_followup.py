"""A source token and its subsequent qualification follow the evidence through both verifiers."""

import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import build_grounded_english_menu, format_grounded_span_token, grounded_candidate_choices
from nanoscribe.scribe import scribe


def check(source, slot, field, value, start, supported):
    doc = transcript_source(NanoInput("followup", source))
    span = EvidenceSpan(start, start + len(value), value)
    token = format_grounded_span_token(doc, span)
    proposal = "|".join(f"{s}:S[{token}]" if s == slot else f"{s}:M" for s in ("CC", "DUR", "SEV", "MED", "ALG"))
    note = scribe(source, solver=compact_solver(lambda _: proposal, binder="grounded"), context_policy="grounded")
    assert note["verification"]["passed"]
    claim = next(c for c in note["note"]["claims"] if c["field"] == field.value)
    assert claim["state"] == ("supported" if supported else "uncertain")
    assert _observation_kind(doc, field, span, context_policy="grounded") is (FieldState.SUPPORTED if supported else None)
    choices = grounded_candidate_choices(source, field, set())
    assert (f":S[{token}]" in choices) == supported
    english_spans = {s for code, spans in build_grounded_english_menu(doc, field, set()).values() if code == "S" for s in spans}
    assert (span in english_spans) == supported


@pytest.mark.parametrize("slot,field,turn,value", [
    ("MED", FieldName.MEDICATION, "I took dihydrocodeine.", "codeine"),
    ("MED", FieldName.MEDICATION, "I took hydrocortisone.", "hydro"),
    ("ALG", FieldName.ALLERGY, "I am allergic to hydrocortisone.", "cortisone"),
    ("MED", FieldName.MEDICATION, "I took co-codamol.", "codamol"),
    ("ALG", FieldName.ALLERGY, "I am allergic to sodium chloride.", "sodium"),
    ("MED", FieldName.MEDICATION, "I took calcium carbonate for the pain.", "carbonate"),
    ("ALG", FieldName.ALLERGY, "I am allergic to methylcellulose.", "cell"),
    ("MED", FieldName.MEDICATION, "I took acetaminophen/codeine.", "codeine"),
])
def test_subword_is_not_the_reported_substance(slot, field, turn, value):
    source = "Patient: " + turn
    check(source, slot, field, value, source.index(value), False)


def test_offset_tag_distinguishes_a_word_from_a_subword_in_another_turn():
    source = "Patient: The package says dihydrocodeine.\nPatient: I took codeine."
    check(source, "MED", FieldName.MEDICATION, "codeine", source.index("codeine"), False)
    check(source, "MED", FieldName.MEDICATION, "codeine", source.rindex("codeine"), True)


@pytest.mark.parametrize("slot,field,turn,value", [
    ("MED", FieldName.MEDICATION, "I took dihydrocodeine.", "dihydrocodeine"),
    ("ALG", FieldName.ALLERGY, "I am allergic to hydrocortisone.", "hydrocortisone"),
    ("MED", FieldName.MEDICATION, "I took co-codamol.", "co-codamol"),
    ("ALG", FieldName.ALLERGY, "I am allergic to sodium chloride.", "sodium chloride"),
    ("MED", FieldName.MEDICATION, "I took calcium carbonate for the pain.", "calcium carbonate"),
])
def test_complete_substance_word_is_still_supported(slot, field, turn, value):
    source = "Patient: " + turn
    check(source, slot, field, value, source.index(value), True)


@pytest.mark.parametrize("slot,field,prior,followup,value,supported", [
    ("MED", FieldName.MEDICATION, "I took ibuprofen.", "I am not sure whether I took it.", "ibuprofen", False),
    ("MED", FieldName.MEDICATION, "I took ibuprofen.", "I can't remember taking ibuprofen.", "ibuprofen", False),
    ("MED", FieldName.MEDICATION, "I took ibuprofen.", "I'm no longer sure about that.", "ibuprofen", False),
    ("ALG", FieldName.ALLERGY, "I'm allergic to latex.", "I am unsure whether I am allergic to it.", "latex", False),
    ("SEV", FieldName.SEVERITY, "The pain is mild.", "I'm not sure how severe it is.", "mild", False),
    ("DUR", FieldName.DURATION, "It started yesterday.", "I cannot recall when it started.", "yesterday", False),
    ("MED", FieldName.MEDICATION, "I took ibuprofen.", "I am not sure whether I took aspirin.", "ibuprofen", True),
    ("MED", FieldName.MEDICATION, "I took ibuprofen.", "I am not sure whether ibuprofen helped.", "ibuprofen", True),
    ("MED", FieldName.MEDICATION, "I took ibuprofen.", "I am not sure when I took it.", "ibuprofen", True),
    ("MED", FieldName.MEDICATION, "I took ibuprofen today.", "I am not sure if I took it yesterday.", "ibuprofen", True),
    ("ALG", FieldName.ALLERGY, "I'm allergic to latex.", "I'm not sure whether I am allergic to nickel.", "latex", True),
    ("SEV", FieldName.SEVERITY, "The pain is mild.", "I am unsure when it started.", "mild", True),
    ("DUR", FieldName.DURATION, "It started yesterday.", "I am unsure when it will improve.", "yesterday", True),
    ("MED", FieldName.MEDICATION, "I took ibuprofen.", "I am not sure whether my sister took it.", "ibuprofen", True),
])
@pytest.mark.parametrize("separator", [" ", "\nPatient: "])
def test_subsequent_patient_doubt_is_scoped_to_the_prior_observation(slot, field, prior, followup, value, supported, separator):
    source = "Patient: " + prior + separator + followup
    check(source, slot, field, value, source.index(value), supported)


def test_new_topic_breaks_an_unqualified_followup_reference():
    source = "Patient: I took ibuprofen.\nDoctor: Any allergies?\nPatient: I am not sure about that."
    check(source, "MED", FieldName.MEDICATION, "ibuprofen", source.index("ibuprofen"), True)


def test_confirmation_question_preserves_the_observation_under_discussion():
    source = "Patient: I took ibuprofen.\nDoctor: Are you sure about that?\nPatient: I am not sure about that."
    check(source, "MED", FieldName.MEDICATION, "ibuprofen", source.index("ibuprofen"), False)


def test_a_later_positive_occurrence_can_reaffirm_the_initial_doubt():
    source = "Patient: I took ibuprofen.\nPatient: I am not sure whether I took it.\nPatient: I took ibuprofen."
    check(source, "MED", FieldName.MEDICATION, "ibuprofen", source.index("ibuprofen"), False)
    check(source, "MED", FieldName.MEDICATION, "ibuprofen", source.rindex("ibuprofen"), True)


@pytest.mark.parametrize("statement,field,doubt", [
    ("I took no medications.", FieldName.MEDICATION, "I am unsure whether I took aspirin."),
    ("I have no allergies.", FieldName.ALLERGY, "I am not sure if I am allergic to latex."),
])
def test_field_wide_denial_is_also_qualified_by_a_later_uncertain_instance(statement, field, doubt):
    source = "Patient: " + statement + "\nPatient: " + doubt
    doc = transcript_source(NanoInput("denial-doubt", source))
    start = source.index(statement)
    span = EvidenceSpan(start, start + len(statement), statement)
    assert _observation_kind(doc, field, span, context_policy="grounded") is None
    assert f":A[{statement}]" not in grounded_candidate_choices(source, field, set())


@pytest.mark.parametrize("turn", [
    "I took aspirin earlier, but I will take ibuprofen later.",
    "I took aspirin later.",
    "I took aspirin again.",
])
def test_administration_adverb_is_outside_the_complete_substance_name(turn):
    source = "Patient: " + turn
    check(source, "MED", FieldName.MEDICATION, "aspirin", source.index("aspirin"), True)
