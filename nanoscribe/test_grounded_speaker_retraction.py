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
