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
    return assert_substance(transcript, MED, named, occurrence, supported)


def assert_substance(transcript, field, named, occurrence, supported):
    source = transcript_source(NanoInput("correction", transcript))
    start = -1
    for _ in range(occurrence + 1):
        start = transcript.index(named, start + 1)
    span = EvidenceSpan(start, start + len(named), named)
    assert verify_grounded_span(source, field, FieldState.SUPPORTED, span)[0] == supported
    state = FieldState.SUPPORTED if supported else None
    assert _observation_kind(source, field, span, context_policy="grounded") is state
    token = format_grounded_span_token(source, span)
    proposal = f":S[{token}]"
    assert (proposal in grounded_candidate_choices(transcript, field, set())) == supported
    menu = build_grounded_english_menu(source, field, set())
    assert (("S", (span,)) in menu.values()) == supported
    med, alg = (proposal, ":M") if field is MED else (":M", proposal)
    summary = f"CC:M|DUR:M|SEV:M|MED{med}|ALG{alg}"
    note = scribe(transcript, solver=compact_solver(lambda _: summary, binder="grounded"),
                  context_policy="grounded")
    assert note["verification"]["passed"]
    claim = next(c for c in note["note"]["claims"] if c["field"] == field.value)
    assert claim["state"] == ("supported" if supported else "uncertain")
    if supported:
        assert claim["value"] == named
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


@pytest.mark.parametrize("denial", ["aspirin after all", "aspirin, after all",
                                    "500 mg of aspirin for my knee", "aspirin 500 mg for my knee",
                                    "500 mg of aspirin today", "aspirin 500 mg today",
                                    "500 mg of aspirin for my knee today"])
def test_explicit_matching_dose_date_and_indication_do_not_need_contiguous_source(denial):
    text = (f"Patient: I took 500 mg of aspirin today for my knee.\n"
            f"Patient: Actually, I didn't take {denial}.")
    assert_medication(text, "aspirin", 0, False)


@pytest.mark.parametrize("denial", ["500 mg of aspirin today", "200 mg of aspirin yesterday",
                                    "200 mg of aspirin for my knee"])
def test_each_supplied_administration_restriction_must_match(denial):
    text = (f"Patient: I took 200 mg of aspirin today for my ankle.\n"
            f"Patient: Actually, I didn't take {denial}.")
    assert_medication(text, "aspirin", 0, True)


@pytest.mark.parametrize("original,denial", [
    ("aspirin and ibuprofen 500 mg", "500 mg of aspirin"),
    ("500 mg of aspirin and ibuprofen", "500 mg of ibuprofen"),
    ("aspirin and ibuprofen for my knee", "aspirin for my knee"),
    ("aspirin and ibuprofen yesterday", "aspirin yesterday"),
])
def test_another_coadministered_name_cannot_supply_a_restriction(original, denial):
    text = f"Patient: I took {original}.\nPatient: Actually, I didn't take {denial}."
    source = transcript_source(NanoInput("coadministration", text))
    menu = build_grounded_english_menu(source, MED, set())
    values = [[s.text for s in spans] for code, spans in menu.values() if code == "U"]
    assert ["aspirin", "ibuprofen"] in values
    assert not any(code == "S" for code, _ in menu.values())


@pytest.mark.parametrize("denial", ["500 mg of ibuprofen", "ibuprofen 500 mg"])
def test_same_coadministered_name_can_supply_its_own_dose(denial):
    text = (f"Patient: I took aspirin and ibuprofen 500 mg.\n"
            f"Patient: Actually, I didn't take {denial}.")
    assert_medication(text, "aspirin", 0, True)
    assert_medication(text, "ibuprofen", 0, False)


@pytest.mark.parametrize("withdrawn,expected", [
    ("500 mg of aspirin and 200 mg of ibuprofen", "naproxen"),
    ("aspirin 500 mg and ibuprofen 200 mg", "naproxen"),
])
def test_coordinated_correction_compares_each_dose_with_its_target(withdrawn, expected):
    text = ("Patient: I took 500 mg of aspirin. I took ibuprofen 200 mg. I took naproxen.\n"
            f"Patient: Actually, I didn't take {withdrawn}.")
    assert_medication(text, "aspirin", 0, False)
    assert_medication(text, "ibuprofen", 0, False)
    assert_medication(text, expected, 0, True)


def test_accompaniment_remains_part_of_the_named_administration():
    text = ("Patient: I took aspirin with water.\n"
            "Patient: Actually, I didn't take aspirin with water.")
    assert_medication(text, "aspirin", 0, False)


@pytest.mark.parametrize("denial,withdrawn", [
    ("any aspirin", True), ("any of the aspirin", True),
    ("either aspirin or ibuprofen", True), ("either ibuprofen or aspirin", True),
    ("any ibuprofen", False), ("any more aspirin", False),
    ("any aspirin yesterday", False),
])
def test_named_denial_quantifier_keeps_the_correct_name_and_scope(denial, withdrawn):
    text = f"Patient: I took aspirin today.\nPatient: Actually, I didn't take {denial}."
    assert_medication(text, "aspirin", 0, not withdrawn)


@pytest.mark.parametrize("answer,supported", [
    ("and I'm not allergic to it", False), ("and I'm not allergic to aspirin", False),
    ("and I have no allergies", False), ("and I'm not allergic to latex", True),
    ("but I am allergic to it", True), ("and I took ibuprofen", True),
])
def test_independent_allergy_statement_has_its_own_polarity_and_object(answer, supported):
    text = (f"Patient: I am allergic to aspirin.\nPatient: Actually, I didn't take aspirin, {answer}.")
    assert_substance(text, FieldName.ALLERGY, "aspirin", 0, supported)


def test_reference_to_another_medication_does_not_retract_unrelated_allergen():
    text = "Patient: I am allergic to latex.\nPatient: Actually, I didn't take aspirin, and I'm not allergic to it."
    assert_substance(text, FieldName.ALLERGY, "latex", 0, True)


def test_independent_allergy_retraction_does_not_absorb_the_medication_object():
    text = "Patient: I took aspirin.\nPatient: Actually, I didn't take aspirin, and I'm not allergic to it."
    assert_medication(text, "aspirin", 0, False)


@pytest.mark.parametrize("denied,first_remains", [("yesterday", False), ("today", True)])
def test_coordinated_repeated_name_has_an_interval_for_each_occurrence(denied, first_remains):
    text = (f"Patient: I took aspirin yesterday and I took aspirin today.\n"
            f"Patient: Actually, I didn't take aspirin {denied}.")
    assert_medication(text, "aspirin", 0, first_remains)
    assert_medication(text, "aspirin", 1, not first_remains)


@pytest.mark.parametrize("allergen,scope,supported", [
    ("aspirin", "any drugs", False), ("latex", "any drugs", True),
    ("latex", "latex", False), ("aspirin", "latex", True),
])
def test_second_medication_or_allergy_clause_respects_object_category(allergen, scope, supported):
    text = (f"Patient: I am allergic to {allergen}.\n"
            f"Patient: Actually, I didn't take aspirin, and I'm not allergic to {scope}.")
    assert_substance(text, FieldName.ALLERGY, allergen, 0, supported)


@pytest.mark.parametrize("denied,supported", [("today", False), ("yesterday", True)])
def test_unqualified_direct_answer_inherits_the_medication_questions_day(denied, supported):
    text = ("Doctor: Which medicine did you take today?\nPatient: I took aspirin.\n"
            f"Patient: Actually, I didn't take any aspirin {denied}.")
    assert_medication(text, "aspirin", 0, supported)


@pytest.mark.parametrize("second", ["and I didn't take", "but I did not use", "and I haven't taken"])
def test_each_coordinated_negative_patient_predicate_retracts_its_own_name(second):
    text = (f"Patient: I took aspirin and ibuprofen. I took naproxen.\n"
            f"Patient: Actually, I didn't take aspirin, {second} ibuprofen.")
    assert_medication(text, "aspirin", 0, False)
    assert_medication(text, "ibuprofen", 0, False)
    assert_medication(text, "naproxen", 0, True)


def test_second_negative_predicate_has_its_own_date():
    text = ("Patient: I took aspirin today. I took ibuprofen yesterday.\n"
            "Patient: Actually, I didn't take aspirin today, and I didn't take ibuprofen today.")
    assert_medication(text, "aspirin", 0, False)
    assert_medication(text, "ibuprofen", 0, True)


@pytest.mark.parametrize("original,withdrawn,supported", [
    ("Monday", "on Monday", False), ("on Monday", "Monday", False),
    ("Monday morning", "on Monday morning", False), ("Tuesday", "on Monday", True),
    ("Monday evening", "on Monday morning", True), ("Monday", "before Monday", True),
    ("Monday for my ankle", "on Monday for my ankle", False),
    ("Monday for my knee", "on Monday for my ankle", True),
])
def test_calendar_preposition_belongs_to_the_date(original, withdrawn, supported):
    text = f"Patient: I took aspirin {original}.\nPatient: Actually, I didn't take aspirin {withdrawn}."
    assert_medication(text, "aspirin", 0, supported)


@pytest.mark.parametrize("clarification", [
    "but I am talking about last year", "and I mean the previous visit",
    "because I was giving a hypothetical", "but I used to say that in childhood",
])
def test_later_historical_or_hypothetical_context_cannot_retract_current_report(clarification):
    text = f"Patient: I took aspirin today.\nPatient: Actually, I didn't take aspirin, {clarification}."
    assert_medication(text, "aspirin", 0, True)


@pytest.mark.parametrize("suffix,remains", [
    (", but I might have taken ibuprofen instead.", False),
    (", and I am not sure whether I took ibuprofen.", False),
    (", but should I take ibuprofen?", False),
    (" because I might be allergic to its coating.", False),
    (" because I was not sure of the dose.", False),
    (" because the bottle might have been expired.", False),
    (", but I took ibuprofen last year.", False),
    (", but I'm not sure of that.", True),
    (", but I might be wrong about that.", True),
    (", but did I?", True),
    ("?", True),
])
def test_uncertainty_belongs_to_its_own_predicate(suffix, remains):
    text = "Patient: I took aspirin.\nPatient: Actually, I didn't take aspirin" + suffix
    assert_medication(text, "aspirin", 0, remains)


def test_hedged_second_denial_leaves_only_its_own_source_supported():
    text = ("Patient: I took aspirin. I took ibuprofen.\n"
            "Patient: Actually, I didn't take aspirin, and I didn't take ibuprofen, perhaps.")
    assert_medication(text, "aspirin", 0, False)
    assert_medication(text, "ibuprofen", 0, True)


@pytest.mark.parametrize("question,denial,supported", [
    ("Which drugs are you allergic to?", "any drugs", False),
    ("Which drugs are you allergic to?", "any foods", True),
    ("Which foods are you allergic to?", "any foods", False),
    ("Which foods are you allergic to?", "any drugs", True),
    ("What are your allergies?", "any drugs", True),
])
def test_mixed_correction_retains_typed_allergen_source(question, denial, supported):
    transcript = (f"Doctor: {question}\nPatient: zinc.\n"
                  f"Patient: Actually, I didn't take aspirin, and I'm not allergic to {denial}.")
    assert_substance(transcript, FieldName.ALLERGY, "zinc", 0, supported)


@pytest.mark.parametrize("tail,supported", [
    (", but I am answering for my mother", True),
    (", and I meant my father", True),
    (", but I am answering for myself", False),
    (" because I ran out", False),
    (" because my mother kept the bottle", False),
    (" because I was answering for my mother", True),
])
def test_retraction_requires_the_patient_to_own_the_denial(tail, supported):
    text = f"Patient: I took aspirin today.\nPatient: Actually, I didn't take aspirin{tail}."
    assert_medication(text, "aspirin", 0, supported)


@pytest.mark.parametrize("names,withdrawn,survivor", [
    ("aspirin and ibuprofen", "aspirin", "ibuprofen"),
    ("naproxen and cetirizine", "cetirizine", "naproxen"),
    ("aspirin, ibuprofen, and naproxen", "aspirin or ibuprofen", "naproxen"),
])
def test_single_surviving_coordinated_medication_can_be_selected(names, withdrawn, survivor):
    text = f"Patient: I took {names}.\nPatient: Actually, I didn't take {withdrawn}."
    menu = assert_medication(text, survivor, 0, True)
    assert not any(kind == "U" and len(spans) > 1 for kind, spans in menu.values())
