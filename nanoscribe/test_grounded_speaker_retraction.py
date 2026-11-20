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


def check(report, field, quote, *, state=REPORTED, licensed):
    """Exercise the offset-based verifier, both menus, and free proposal through note checking."""
    request = NanoInput("speaker-boundary", report)
    source = transcript_source(request)
    start = report.index(quote)
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


@pytest.mark.parametrize("speaker", ["Companion", "Family", "Caregiver", "Unknown"])
def test_nonpatient_report_is_never_patient_evidence(speaker):
    check(f"Patient: Thanks.\n{speaker}: I took cetirizine.", MED, "cetirizine", licensed=False)


def test_other_voice_breaks_question_attribution():
    check("Doctor: What medication did you take?\nCompanion: How about your sister?\nPatient: Loratadine.",
          MED, "Loratadine", licensed=False)
