"""An explicit patient correction invalidates the named earlier medication occurrence."""
import pytest

from nano_ai.contract import EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import (
    build_grounded_english_menu, format_grounded_span_token, grounded_candidate_choices,
    verify_grounded_span,
)
from nanoscribe.scribe import scribe


MED = FieldName.MEDICATION


def assert_medication(transcript, named, occurrence, supported):
    source = transcript_source(NanoInput("correction", transcript))
    start = -1
    for _ in range(occurrence + 1):
        start = transcript.index(named, start + 1)
    span = EvidenceSpan(start, start + len(named), named)
    assert verify_grounded_span(source, MED, FieldState.SUPPORTED, span)[0] == supported
    state = FieldState.SUPPORTED if supported else None
    assert _observation_kind(source, MED, span, context_policy="grounded") is state
    token = format_grounded_span_token(source, span)
    proposal = f":S[{token}]"
    assert (proposal in grounded_candidate_choices(transcript, MED, set())) == supported
    menu = build_grounded_english_menu(source, MED, set())
    assert (("S", (span,)) in menu.values()) == supported
    summary = f"CC:M|DUR:M|SEV:M|MED{proposal}|ALG:M"
    note = scribe(transcript, solver=compact_solver(lambda _: summary, binder="grounded"),
                  context_policy="grounded")
    assert note["verification"]["passed"]
    medication = next(c for c in note["note"]["claims"] if c["field"] == MED.value)
    assert medication["state"] == ("supported" if supported else "uncertain")
    if supported:
        assert medication["value"] == named
    return menu


@pytest.mark.parametrize("prefix", ["Let me correct that. I did not take", "Correction: I didn’t take",
                                    "Actually, I didn't take", "Let me correct my answer: I haven't taken"])
@pytest.mark.parametrize("boundary", [" ", "\nPatient: "])
def test_withdrawal_changes_both_menus_and_free_source_verification(prefix, boundary):
    text = "Patient: I took naproxen." + boundary + prefix + " naproxen. I took aspirin."
    menu = assert_medication(text, "naproxen", 0, False)
    assert_medication(text, "naproxen", 1, False)
    assert_medication(text, "aspirin", 0, True)
    assert not any(code == "U" and len(spans) > 1 for code, spans in menu.values())


@pytest.mark.parametrize("statement", [
    "Other: Let me correct that. I did not take aspirin.",
    "Patient: If I need to correct that, I didn't take aspirin.",
    "Patient: Let me correct that. Did I not take aspirin?",
    "Patient: Let me correct that. My mother did not take aspirin.",
    "Patient: Let me correct that. I did not take aspirin for my knee.",
    "Patient: Actually, I did not take aspirin yesterday.",
    "Patient: Actually, I did not take aspirin gel.",
    "Patient: Actually, I did not take ibuprofen.",
    "Doctor: Which medicine did you take yesterday?\nPatient: Actually, I did not take aspirin.",
])
def test_unrelated_corrections_keep_original_report(statement):
    assert_medication("Patient: I took aspirin today.\n" + statement, "aspirin", 0, True)


def test_only_corrected_object_is_withdrawn_from_coadministered_names():
    text = "Patient: I took aspirin and ibuprofen.\nPatient: Actually, I didn't take ibuprofen."
    assert_medication(text, "ibuprofen", 0, False)
    assert_medication(text, "aspirin", 0, True)


def test_later_identical_reaffirmation_has_its_own_evidence():
    text = ("Patient: I took aspirin.\nPatient: Let me correct that. I did not take aspirin.\n"
            "Patient: I took aspirin today.")
    assert_medication(text, "aspirin", 0, False)
    assert_medication(text, "aspirin", 2, True)


def test_distinct_patient_reports_still_require_the_multivalue_menu():
    text = "Patient: I took aspirin.\nPatient: I took ibuprofen."
    menu = assert_medication(text, "aspirin", 0, False)
    assert " UNCERTAIN[aspirin; ibuprofen]" in menu


def test_combined_quote_cannot_reintroduce_withdrawn_member():
    text = ("Patient: I took aspirin and ibuprofen.\n"
            "Patient: Actually, I didn't take ibuprofen.")
    assert_medication(text, "aspirin and ibuprofen", 0, False)


@pytest.mark.parametrize("denial", ["500 mg of aspirin", "aspirin 500 mg", "aspirin for my ankle"])
def test_adjusting_dose_or_indication_preserves_confirmed_use(denial):
    assert_medication(f"Patient: I took aspirin.\nPatient: Actually, I didn't take {denial}.",
                      "aspirin", 0, True)


def test_explicit_same_day_correction_withdraws_only_matching_date():
    text = ("Patient: I took aspirin yesterday.\nPatient: I took aspirin today.\n"
            "Patient: Let me correct that. I didn't take aspirin today.")
    assert_medication(text, "aspirin", 0, True)
    assert_medication(text, "aspirin", 1, False)


@pytest.mark.parametrize("transition", [", but I took", ", I took", " and I took"])
def test_independent_replacement_predicate_preserves_named_withdrawal(transition):
    text = f"Patient: I took aspirin.\nPatient: Actually, I didn't take aspirin{transition} naproxen."
    assert_medication(text, "aspirin", 0, False)


@pytest.mark.parametrize("original,withdrawn", [
    ("aspirin for my ankle", "aspirin for my ankle"),
    ("aspirin 500 mg", "500 mg of aspirin"),
])
def test_same_administration_qualification_can_be_withdrawn(original, withdrawn):
    text = f"Patient: I took {original}.\nPatient: Actually, I didn't take {withdrawn}."
    assert_medication(text, "aspirin", 0, False)


@pytest.mark.parametrize("join", [" and ", " or ", ", "])
def test_named_negative_list_preserves_unmentioned_earlier_treatment(join):
    text = ("Patient: I took aspirin. I took ibuprofen. I took naproxen.\n"
            f"Patient: Let me correct that. I didn't take ibuprofen{join}aspirin.")
    assert_medication(text, "ibuprofen", 0, False)
    assert_medication(text, "aspirin", 0, False)
    assert_medication(text, "naproxen", 0, True)


def test_explanation_for_nonuse_preserves_correction():
    text = ("Patient: I took aspirin.\nPatient: Let me correct that. "
            "I didn't take aspirin because I never opened the packet. I took ibuprofen.")
    assert_medication(text, "aspirin", 0, False)
    assert_medication(text, "ibuprofen", 0, True)


def test_question_with_an_explanation_preserves_original_administration():
    text = "Patient: I took aspirin.\nPatient: Actually, I didn't take aspirin because I felt better?"
    assert_medication(text, "aspirin", 0, True)


def test_tentative_withdrawal_does_not_certify_unique_replacement():
    text = ("Patient: I took aspirin.\nPatient: Let me correct that. "
            "I didn't take aspirin, perhaps. I took ibuprofen.")
    menu = assert_medication(text, "ibuprofen", 0, False)
    assert " UNCERTAIN[aspirin; ibuprofen]" in menu


@pytest.mark.parametrize("emphasis", ["at all", "whatsoever"])
def test_emphasis_does_not_limit_denial_to_a_different_administration(emphasis):
    text = f"Patient: I took aspirin today.\nPatient: Actually, I did not take aspirin {emphasis}."
    assert_medication(text, "aspirin", 0, False)


@pytest.mark.parametrize("replacement", ["but I did take", "and I took", "but I have taken"])
def test_corrected_predicate_and_affirmed_replacement_own_separate_sources(replacement):
    text = (f"Patient: I took aspirin.\nPatient: Actually, I did not take aspirin, {replacement} "
            "ibuprofen.")
    assert_medication(text, "aspirin", 0, False)
    assert_medication(text, "ibuprofen", 0, True)


@pytest.mark.parametrize("noun", ["any drugs", "any meds", "any pills", "medication",
                                  "any treatments", "any medicines", "anything"])
@pytest.mark.parametrize("marker", ["Actually, ", "Let me correct myself. "])
def test_generic_medication_correction_withdraws_prior_named_report(noun, marker):
    text = f"Patient: I took aspirin.\nPatient: {marker}I didn't take {noun}."
    assert_medication(text, "aspirin", 0, False)


@pytest.mark.parametrize("restriction", ["any medication yesterday", "any pills for my knee",
                                        "any drugs yesterday at all", "allergy medication at all"])
def test_restricted_or_specific_denial_preserves_different_administration(restriction):
    text = f"Patient: I took aspirin today.\nPatient: Actually, I didn't take {restriction}."
    assert_medication(text, "aspirin", 0, True)


def test_category_with_same_date_withdraws_only_matching_occurrence():
    text = ("Patient: I took aspirin yesterday.\nPatient: I took aspirin today.\n"
            "Patient: Let me correct that. I did not take any drugs today at all.")
    assert_medication(text, "aspirin", 0, True)
    assert_medication(text, "aspirin", 1, False)


@pytest.mark.parametrize("day,supported", [("yesterday", True), ("today", False)])
def test_emphasis_before_calendar_qualifier_keeps_real_restriction(day, supported):
    text = f"Patient: I took aspirin today.\nPatient: Actually, I didn't take aspirin at all {day}."
    assert_medication(text, "aspirin", 0, supported)
