"""Only a patient's later correction can supersede their own reported observation."""
import pytest

from nano_ai.contract import FIELD_ORDER, EvidenceSpan, FieldName, FieldState, NanoInput
from nanoscribe.check_note import _observation_kind, transcript_source, uncovered_turns
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import (
    build_grounded_english_menu, format_grounded_span_token, grounded_candidate_choices,
)
from nanoscribe.scribe import scribe


CC, DUR, SEV, MED, ALG = FIELD_ORDER
REPORTED, DENIED = FieldState.SUPPORTED, FieldState.ABSENT


def check(report, field, quote, *, state=REPORTED, licensed, after=0):
    """Exercise the offset-based verifier, both menus, and free proposal through note checking."""
    request = NanoInput("speaker-boundary", report)
    source = transcript_source(request)
    start = report.index(quote, after)
    span = EvidenceSpan(start, start + len(quote), quote)
    code = "S" if state is REPORTED else "A"
    choice = f":{code}[{format_grounded_span_token(source, span)}]"
    field_index = FIELD_ORDER.index(field)
    raw = "|".join(slot + (choice if idx == field_index else ":M")
                   for idx, slot in enumerate(("CC", "DUR", "SEV", "MED", "ALG")))
    solver = compact_solver(lambda _: raw, binder="grounded")
    draft = scribe(report, solver=solver, context_policy="grounded")
    assert draft["verification"]["passed"]
    claim = draft["note"]["claims"][field_index]
    assert (claim["state"], claim["value"]) == (
        state.value if licensed else "uncertain", quote if licensed and state is REPORTED else None)

    verified = _observation_kind(source, field, span, context_policy="grounded")
    assert verified is (state if licensed else None)
    assert (choice in grounded_candidate_choices(report, field, set())) == licensed
    english = build_grounded_english_menu(source, field, set())
    assert ((code, (span,)) in english.values()) == licensed
    return {turn.text for turn in uncovered_turns(source, solver.infer_grounded(request)[0], context_policy="grounded")}


@pytest.mark.parametrize("field,reported,quote,state", [
    (CC, "I have wrist pain.", "wrist pain", REPORTED),
    (DUR, "It started six days ago.", "six days", REPORTED),
    (SEV, "The pain is 3/10.", "3/10", REPORTED),
    (MED, "I took naproxen.", "naproxen", REPORTED),
    (ALG, "I am allergic to nickel.", "nickel", REPORTED),
    (MED, "I haven't taken anything.", "I haven't taken anything.", DENIED),
    (ALG, "No known allergies.", "No known allergies.", DENIED),
])
def test_later_identified_speaker_preserves_patient_evidence_and_remains_uncovered(field, reported, quote, state):
    aside = "Caregiver: We already paid for the parking."
    uncovered = check(f"Patient: {reported}\r\n{aside}", field, quote, state=state, licensed=True)
    assert aside in uncovered


@pytest.mark.parametrize("aside", [
    "Unknown: The jacket is by the chair.",
    "uNkNoWn:  We found the parking pass. Could you please leave the pass at the desk?",
    "The umbrella is upstairs.",
    "Unknown: The main entrance closes at noon. Our taxi is by the side door.",
    "Unknown: Where could we collect the visitor passes?",
])
def test_unidentified_logistics_preserves_patient_evidence_and_remains_uncovered(aside):
    uncovered = check(f"Patient: I took naproxen.\r\n{aside}", MED, "naproxen", licensed=True)
    assert aside in uncovered


@pytest.mark.parametrize("aside", [
    "Unknown: The umbrella is upstairs. The answers marked Patient were mine.",
    "Unknown: I found the bag. But the drug name was wrong.",
    "Unknown: The parking pass is here, actually I took ibuprofen.",
    "Unknown: Sorry, I was mistaken. My coat is by the desk.",
    "Office: My coat is by the desk.",
])
def test_logistics_cannot_mask_an_unidentified_correction_or_unrecognized_label(aside):
    check(f"Patient: I took naproxen.\n{aside}", MED, "naproxen", licensed=False)


def test_unidentified_aside_preserves_a_later_patient_retraction():
    check("Patient: I took naproxen.\nUnknown: The parking pass is here.\nPatient: Actually I was wrong.",
          MED, "naproxen", licensed=False)


@pytest.mark.parametrize("aside,doubt,licensed", [
    ("The umbrella is upstairs.", "I'm not sure whether I took it.", False),
    ("Where is the garage?", "I'm not sure whether I took naproxen.", False),
    ("Where is the garage?", "I'm not sure about that.", True),
    ("The umbrella is upstairs.", "I am not sure about that. The umbrella is downstairs.", True),
    ("The umbrella is upstairs.", "I'm not sure whether I took loratadine.", True),
])
def test_logistical_asides_preserve_the_scope_of_patient_doubt(aside, doubt, licensed):
    check(f"Patient: I took naproxen.\nUnknown: {aside}\nPatient: {doubt}", MED, "naproxen", licensed=licensed)


@pytest.mark.parametrize("field,first,second", [
    (MED, "I took naproxen.", "I haven't taken anything."),
    (ALG, "No known allergies.", "I am allergic to nickel."),
    (SEV, "My neck pain is 3/10.", "My neck pain is 7/10."),
])
def test_subsequent_explicit_patient_evidence_still_exposes_the_conflict(field, first, second):
    def choices(aside):
        report = f"Patient: {first}\n{aside}Patient: {second}"
        source = transcript_source(NanoInput("speaker-boundary", report))
        return set(build_grounded_english_menu(source, field, set()))

    baseline = choices("")
    assert any(label.lstrip().startswith("CONTRADICTED[") for label in baseline)
    assert choices("Unknown: The main entrance is closed.\n") == baseline


@pytest.mark.parametrize("question,answer,field,quote,state", [
    ("What medication did you take?", "Naproxen.", MED, "Naproxen", REPORTED),
    ("What medication did you take?", "I haven't taken anything.", MED, "I haven't taken anything.", DENIED),
    ("How severe is your pain?", "7/10.", SEV, "7/10", REPORTED),
    ("How severe is your pain?", "It is mild.", SEV, "mild", REPORTED),
    ("When did it start?", "Monday.", DUR, "Monday", REPORTED),
    ("How long has it hurt?", "I have a headache. Four days.", DUR, "Four days", REPORTED),
])
def test_unidentified_logistics_cannot_supply_the_question_for_a_patient_answer(question, answer, field, quote, state):
    check(f"Doctor: {question}\nUnknown: Where is the garage?\nPatient: {answer}",
          field, quote, state=state, licensed=False)


@pytest.mark.parametrize("field,answer,quote", [
    (SEV, "My foot pain is 7/10.", "7/10"),
    (SEV, "Currently my neck pain is 2/10.", "2/10"),
    (SEV, "My headache feels severe.", "severe"),
    (DUR, "My headache has been constant for three days.", "three days"),
    (DUR, "I have had a headache for three days.", "three days"),
    (MED, "I took naproxen.", "naproxen"),
])
def test_field_specific_patient_statement_after_a_logistical_question(field, answer, quote):
    check(f"Unknown: Where is the garage?\nPatient: {answer}", field, quote, licensed=True)


@pytest.mark.parametrize("answer,field,quote,licensed", [
    ("Monday for the pass, but my headache has lasted four days.", DUR, "Monday", False),
    ("Monday for the pass, but my headache has lasted four days.", DUR, "four days", True),
    ("My headache has lasted four days, but the pass expires Monday.", DUR, "Monday", False),
    ("I took the bag with my aspirin in it.", MED, "bag", False),
    ("I took the bag with my aspirin in it.", MED, "aspirin", False),
    ("I took aspirin from the bag.", MED, "aspirin", True),
])
def test_patient_measurement_or_substance_belongs_to_the_cited_object(answer, field, quote, licensed):
    report = f"Unknown: Can you bring the bag?\nPatient: {answer}"
    check(report, field, quote, licensed=licensed, after=report.index("Patient:"))


@pytest.mark.parametrize("field,answer,quote", [
    (DUR, "It started six days ago.", "six days"),
    (SEV, "It is mild.", "mild"),
])
def test_plain_other_statement_preserves_previous_scalar_context(field, answer, quote):
    check(f"Patient: I have a headache.\nOther: The umbrella is upstairs.\nPatient: {answer}",
          field, quote, licensed=True)


@pytest.mark.parametrize("answer,field,eligible", [
    ("I am not sure.", CC, False),
    ("I am not sure.", ALG, False),
    ("I am not sure where the umbrella is.", CC, False),
    ("I am not sure where the garage is.", MED, False),
    ("I am not sure whether I took the bus with aspirin in my bag.", MED, False),
    ("I am not sure whether I am allergic to nickel.", ALG, True),
    ("I am not sure whether I took naproxen.", MED, True),
    ("I am not sure how long my headache has lasted.", DUR, True),
    ("I cannot remember when my wrist pain started.", DUR, True),
    ("I am not sure how severe my headache is.", SEV, True),
    ("I cannot tell how bad the rash is.", SEV, True),
    ("I am unsure of the duration of my headache.", DUR, True),
    ("I cannot tell how long the parking pass lasts with my headache.", DUR, False),
    ("I am unsure how bad the delay is for my headache.", SEV, False),
    ("I am not sure how severe my headache is.", DUR, False),
    ("I cannot remember when my wrist pain started.", SEV, False),
])
@pytest.mark.parametrize("aside", ["Unknown: Where is the garage?", "Unknown: The umbrella is upstairs."])
def test_uncertainty_after_logistics_requires_a_clinical_target(answer, field, eligible, aside):
    report = f"{aside}\nPatient: {answer}"
    request = NanoInput("logistics-uncertainty", report)
    source = transcript_source(request)
    candidate = f":U[{answer}]"
    index = FIELD_ORDER.index(field)
    assert (candidate in grounded_candidate_choices(report, field, set())) == eligible
    assert (f" UNCERTAIN[{answer}]" in build_grounded_english_menu(source, field, set())) == eligible
    raw = "|".join(label + (candidate if idx == index else ":M")
                   for idx, label in enumerate(("CC", "DUR", "SEV", "MED", "ALG")))
    result, diagnostics = compact_solver(lambda _: raw, binder="grounded").infer_grounded(request)
    assert diagnostics["fields"][index]["verified"] is eligible
    assert bool(result.fields[index].evidence) == eligible


@pytest.mark.parametrize("later,licensed", [
    ("Mother: I was mistaken about naproxen.", True),
    ("  pArTnEr  :  Actually I took ibuprofen instead.", True),
    ("Family member: I was mistaken about my own medication.", True),
    ("Patient’s spouse: I was mistaken about naproxen.", True),
    ("Interpreter: I need a different microphone.", True),
    ("Receptionist: Your receipt is in the envelope.", True),
    ("Billing clerk: I was mistaken about my appointment time.", True),
    ("Front-desk staff: Everything I said was about my own visit.", True),
    ("Scheduling coordinator: I will put the forms outside.", True),
    ("Office: I was mistaken about naproxen.", False),
    ("Registration status: I was mistaken about naproxen.", False),
    ("Registrar: Everything the patient said was about her husband.", False),
    ("Other: I was mistaken about naproxen.", True),
    ("Doctor: I was mistaken about naproxen.", True),
    ("Patient: I was mistaken about naproxen.", False),
    ("Patient: Actually I was wrong.", False),
    ("Patient: I was mistaken about my bus fare.", True),
    ("Companion: I took the bus.\nPatient: I was mistaken about naproxen.", False),
    ("Unknown: I was mistaken about naproxen.", False),
    ("Unattributed text about the parking.", False),
    ("Family history: My medication list was wrong.", False),
    ("I was mistaken about naproxen.", False),
    ("Correction: I was mistaken about naproxen.", False),
    ("Patient:", False),
    ("Mother: Everything I said was about my cousin.", True),
    ("Companion: Actually, all of my answers were about my grandfather.", True),
    ("Patient: Everything I said was about my cousin.", False),
    ("Unknown: Everything I said was about my cousin.", False),
    ("Family member: Everything the patient said was about her husband.", False),
    ("Caregiver: All my answers were wrong. Everything the patient said was about her husband.", False),
    ('Sister: The patient told me, "Everything I said was about my cousin."', False),
])
def test_later_correction_owner_and_scope(later, licensed):
    check("Patient: I took naproxen.\n" + later, MED, "naproxen", licensed=licensed)


@pytest.mark.parametrize("speaker", ["Companion", "Family", "Caregiver", "Receptionist", "Unknown"])
def test_nonpatient_report_is_never_patient_evidence(speaker):
    check(f"Patient: Thanks.\n{speaker}: I took cetirizine.", MED, "cetirizine", licensed=False)


def test_other_voice_breaks_question_attribution():
    check("Doctor: What medication did you take?\nCompanion: How about your sister?\nPatient: Loratadine.",
          MED, "Loratadine", licensed=False)


@pytest.mark.parametrize("aside,licensed", [
    ("All my answers were mistakenly labeled Patient.", False),
    ("What I said was recorded as the patient.", False),
    ("My statements were wrongly attributed to the patient in this transcript.", False),
    ("The speaker labeled Patient was actually me.", False),
    ("The voice tagged as 'Patient' was mine.", False),
    ("The voice tagged as Patient earlier was mine.", False),
    ("It was me answering under the Patient tag.", False),
    ("I spoke under the Patient label in this recording.", False),
    ("I spoke under the caregiver label.", True),
    ("I never spoke under the Patient label.", True),
    ("Could I speak under the Patient label?", True),
    ("I spoke under the Patient label on a separate training form.", True),
    ("The patient has not said anything yet.", False),
    ("Actually, the patient hasn't spoken yet.", False),
    ("My answers were marked Family. The patient answered separately.", True),
    ("My answers were not labeled Patient.", True),
    ("Were my answers labeled Patient?", True),
    ("My answers were about the patient.", True),
    ("I found a folder labeled Patient.", True),
    ("The speaker labeled Patient was the patient.", True),
    ("The patient has not said anything about the parking.", True),
    ("The patient has not spoken to me yet.", True),
    ("If the patient has not spoken yet, let me know.", True),
])
def test_reported_speaker_identity_warning(aside, licensed):
    check("Patient: I am allergic to nickel.\nSister: " + aside, ALG, "nickel", licensed=licensed)


@pytest.mark.parametrize("field,reported,quote,state", [
    (CC, "I have ear pain.", "ear pain", REPORTED),
    (DUR, "It started three days ago.", "three days", REPORTED),
    (SEV, "It is 7/10.", "7/10", REPORTED),
    (MED, "I took ibuprofen.", "ibuprofen", REPORTED),
    (ALG, "I am allergic to latex.", "latex", REPORTED),
    (ALG, "No known allergies.", "No known allergies.", DENIED),
])
def test_identity_warning_has_a_turn_boundary(field, reported, quote, state):
    warning = "Guardian: The patient has not spoken yet."
    source = f"Patient: {reported}\n{warning}\nPatient: {reported}"
    check(source, field, quote, state=state, licensed=False)
    check(source, field, quote, state=state, licensed=True, after=source.index(warning))


def test_global_reassignment_after_identity_warning_still_quarantines_new_answer():
    source = ("Guardian: The patient has not spoken yet.\nDoctor: What do you take?\n"
              "Patient: I take aspirin.\nPatient: Everything I said was about my sister.")
    check(source, MED, "aspirin", licensed=False)


def test_identity_warning_cannot_supply_third_party_question_scope():
    source = ("Sister: The patient has not spoken yet. What do your children take?\n"
              "Patient: I take aspirin.")
    check(source, MED, "aspirin", licensed=False)


def test_newest_identity_warning_supersedes_intervening_reports():
    source = ("Guardian: The patient has not spoken yet.\nPatient: I take aspirin.\n"
              "Sister: The speaker labeled Patient was actually me.\nPatient: I take zinc.")
    check(source, MED, "aspirin", licensed=False)
    check(source, MED, "zinc", licensed=True)


def test_identity_warning_excludes_earlier_uncertainty_and_laterality_conflicts():
    from nanoscribe.grounded import detect_grounded_conflicts, extract_field_uncertain_bundles

    first = ("Patient: It was either three days or four days.\n"
             "Patient: My right wrist hurts right now; my left wrist doesn't hurt.\n"
             "Patient: My left wrist hurts right now; my right wrist doesn't hurt.")
    last = ("\nSister: The patient has not spoken yet.\n"
            "Patient: My left wrist hurts right now; my right wrist doesn't hurt.")
    before = transcript_source(NanoInput("before-warning", first))
    after = transcript_source(NanoInput("after-warning", first + last))
    assert extract_field_uncertain_bundles(before, DUR)
    assert detect_grounded_conflicts(before, CC)
    assert not extract_field_uncertain_bundles(after, DUR)
    assert not detect_grounded_conflicts(after, CC)


@pytest.mark.parametrize("aside, licensed", [
    ("I was mislabeled as Patient, so those allergies are mine.", False),
    ("What I said was recorded as the patient, please correct the transcript.", False),
    ("The speaker labeled Patient was actually me — those were my words.", False),
    ("I was mislabeled as Patient, on a separate training form.", True),
    ("If I was mislabeled as Patient, please fix the label.", True),
    ("My answers were not labeled Patient, so the speaker is correct.", True),
])
def test_continued_identity_warning(aside, licensed):
    check("Patient: I am allergic to nickel.\nSister: " + aside, ALG, "nickel", licensed=licensed)


@pytest.mark.parametrize("field, value, quote, state", [
    (ALG, "No known allergies.", "No known allergies.", DENIED),
    (MED, "I took naproxen.", "naproxen", REPORTED),
])
@pytest.mark.parametrize("intervening, licensed", [
    ("Nurse: We already paid for parking.", False),
    ("Nurse: We paid for parking.\nOther: Thanks for the receipt.", False),
    ("Nurse: We paid for parking.\nNurse: Actually I was wrong.", False),
    ("Patient: My phone is in my bag.", True),
    ("Doctor: Has your address changed?", True),
    ("Sister: What about your husband's medicine?", True),
])
def test_patient_corrects_last_answer_across_asides(field, value, quote, state, intervening, licensed):
    check(f"Patient: {value}\n{intervening}\nPatient: Actually I was wrong.",
          field, quote, state=state, licensed=licensed)


@pytest.mark.parametrize("intervening, conflicting", [
    ("", True),
    ("Doctor: What about your pain right now?\n", True),
    ("Sister: What about your husband's pain?\n", False),
    ("Other: What about your husband's pain?\n", False),
    ("Receptionist: What about your child's pain?\n", False),
    ("Sister: What about your husband's pain?\nDoctor: Now tell me about your pain.\n", True),
])
def test_forced_laterality_conflict_requires_eligible_reply(intervening, conflicting):
    from nanoscribe.grounded import detect_grounded_conflicts
    text = ("Patient: My right wrist hurts right now; my left wrist doesn't hurt.\n"
            + intervening + "Patient: My left wrist hurts right now; my right wrist doesn't hurt.")
    source = transcript_source(NanoInput("laterality-reply", text))
    solver = compact_solver(lambda _: "CC:M|DUR:M|SEV:M|MED:M|ALG:M", binder="grounded")
    output = scribe(text, solver=solver, context_policy="grounded")
    assert output["verification"]["passed"]
    assert bool(detect_grounded_conflicts(source, CC)) == conflicting
    assert (output["note"]["claims"][0]["state"] == "conflicting") == conflicting
    assert any(choice.startswith(":C[") for choice in grounded_candidate_choices(text, CC, set())) == conflicting
    assert any(c == "C" for c, _ in build_grounded_english_menu(source, CC, set()).values()) == conflicting


@pytest.mark.parametrize("intervening, eligible", [
    ("Doctor: How long did your symptoms last?", True),
    ("Sister: How long did your husband's symptoms last?", False),
    ("Other: How long did your husband's symptoms last?", False),
])
def test_uncertainty_bundle_requires_eligible_reply(intervening, eligible):
    from nanoscribe.grounded import extract_field_uncertain_bundles
    text = intervening + "\nPatient: It was either three days or four days."
    source = transcript_source(NanoInput("uncertainty-reply", text))
    assert bool(extract_field_uncertain_bundles(source, DUR)) == eligible


def test_other_aside_mentioning_a_relative_keeps_an_independent_patient_report():
    check("Other: I left your husband's umbrella by the door.\nPatient: I take naproxen.",
          MED, "naproxen", licensed=True)


def test_neutral_aside_mentioning_time_preserves_patient_correction_adjacency():
    check("Patient: No known allergies.\nNurse: We paid for parking last month.\nPatient: Actually I was wrong.",
          ALG, "No known allergies.", state=DENIED, licensed=False)


@pytest.mark.parametrize("disclaimer, licensed", [
    ("That is my usual routine. I pay for parking every morning.", True),
    ("Naproxen is just my usual routine.", False),
    ("That medication is just my usual routine.", False),
])
def test_routine_disclaimer_needs_a_medication_referent_across_an_aside(disclaimer, licensed):
    check(f"Patient: I took naproxen.\nOther: I paid for the parking.\nPatient: {disclaimer}",
          MED, "naproxen", licensed=licensed)


@pytest.mark.parametrize("answer,quote,licensed", [
    ("I took 10 mg of rimegepant.", "rimegepant", True),
    ("I took rimegepant for my headache.", "rimegepant", True),
    ("I took the bus, but I took rimegepant for my pain.", "rimegepant", True),
    ("I took the bag holding 10 mg of rimegepant.", "rimegepant", False),
    ("I took the bag holding 10 mg of rimegepant.", "the bag", False),
    ("I took the bag with rimegepant in it.", "rimegepant", False),
    ("I took the bag with aspirin in it.", "aspirin", False),
    ("I took aspirin from the bag.", "aspirin", True),
    ("I took my parking pass and I took 10 mg of rimegepant.", "parking pass", False),
    ("I took my parking pass and I took 10 mg of rimegepant.", "rimegepant", True),
    ("I plan to take 10 mg of rimegepant.", "rimegepant", False),
])
def test_medication_after_logistics_uses_its_own_treatment_event(answer, quote, licensed):
    report = f"Other: Where is the garage?\nPatient: {answer}"
    check(report, MED, quote, licensed=licensed)


@pytest.mark.parametrize("intervening,followup,licensed", [
    ("Unknown: The umbrella is upstairs.", "My headache has resolved.", False),
    ("Other: Where is the garage?", "The headache has gone away.", False),
    ("Other: Where is the garage?", "It is upstairs. My headache has gone away.", False),
    ("Other: The umbrella is upstairs.\nUnknown: The lobby door is closed.", "My headache has stopped.", False),
    ("Other: Where is the garage?", "It is gone.", True),
    ("Other: Where is the garage?", "My rash has resolved.", True),
    ("Other: Where is the garage?", "My headache has not resolved.", True),
    ("Other: Where is the garage?", "My headache has gone away but it has returned.", True),
    ("Doctor: Did the rash clear up?", "It is gone.", True),
    ("Other: Where is the garage?", "My headache has resolved. I still have chest pain.", False),
])
def test_named_resolution_across_logistics(intervening, followup, licensed):
    report = f"Patient: I have a headache.\n{intervening}\nPatient: {followup}"
    check(report, CC, "headache", licensed=licensed)
    if "chest pain" in followup:
        check(report, CC, "chest pain", licensed=True)


@pytest.mark.parametrize("field,answer,excerpt,eligible", [
    (MED, "I am not sure whether I took naproxen.", "I am not sure whether I took naproxen", True),
    (MED, "To be honest, I am not sure whether I took naproxen.", "not sure whether I took naproxen", True),
    (MED, "I am not sure whether I took naproxen, to be honest.", "not sure whether I took naproxen", True),
    (MED, "I am not sure whether I took naproxen, well.", "not sure whether I took naproxen", True),
    (MED, "I am not sure whether I took naproxen well.", "not sure whether I took naproxen", False),
    (MED, "I am not sure whether I took naproxen. My bus is late.", "not sure whether I took naproxen", True),
    (MED, "I am not sure whether I took naproxen.", "I am not sure", False),
    (MED, "I am not sure whether I took naproxen.", "I took naproxen", False),
    (MED, "I am not sure whether I took naproxen.", "I am not sure whether I took naprox", False),
    (MED, "I am not sure whether I took naproxen after lunch.", "not sure whether I took naproxen", False),
    (ALG, "To be honest, I am unsure if I have drug allergies.", "unsure if I have drug allergies", True),
    (ALG, "I am unsure if I have drug allergies.", "drug allergies", False),
    (DUR, "Honestly, I cannot remember when my wrist pain began.", "cannot remember when my wrist pain began", True),
    (SEV, "Honestly, I am unsure how severe my headache is.", "unsure how severe my headache is", True),
    (SEV, "I am unsure how long the delay is. My headache is severe.", "I am unsure how long the delay is", False),
    (SEV, "My sister is unsure how severe my headache is.", "unsure how severe my headache is", False),
    (ALG, "I am unsure whether I have allergies.", "whether I have allergies", False),
    (SEV, "The leaflet asks whether my headache is severe.", "whether my headache is severe", False),
    (DUR, "My headache lasted either two or three days.", "either two", False),
    (MED, "I am not sure whether I took naproxen or aspirin.", "not sure whether I took naproxen", False),
    (MED, "I am not sure whether I took naproxen.", "I am not sure whether I took naproxen||naproxen", False),
])
def test_uncertain_excerpt_retains_hedge_and_clinical_target(field, answer, excerpt, eligible):
    text = "Other: Where is the garage?\nPatient: " + answer
    request = NanoInput("uncertain-excerpt", text)
    index = FIELD_ORDER.index(field)
    raw = "|".join(label + (f":U[{excerpt}]" if idx == index else ":M")
                   for idx, label in enumerate(("CC", "DUR", "SEV", "MED", "ALG")))
    extraction, trace = compact_solver(lambda _: raw, binder="grounded").infer_grounded(request)
    assert trace["fields"][index]["verified"] is eligible
    assert [span.text for span in extraction.fields[index].evidence] == ([excerpt] if eligible else [])
