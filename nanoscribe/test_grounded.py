"""Tests for the opt-in evidence-grounded co-design policy (`evidence-grounded-v1`)."""

from __future__ import annotations

import pytest

from nano_ai.contract import FieldName
from nanoscribe.compact import compact_solver
from nanoscribe.grounded import grounded_candidate_choices
from nanoscribe.scribe import scribe


def test_natural_clinician_phrasing_retains_grounded_claims_where_regex_abstains():
    transcript = (
        "Doctor: Walk me through what brought you to urgent care.\n"
        "Patient: Throat soreness.\n"
        "Doctor: Roughly how much time has passed since that started?\n"
        "Patient: Around 4 days.\n"
        "Doctor: Where would that land on mild, moderate, or severe?\n"
        "Patient: Moderate.\n"
        "Doctor: Anything from the pharmacy cabinet you've swallowed for relief?\n"
        "Patient: Lozenges, no — I took acetaminophen.\n"
        "Doctor: Any drug sensitivities on your chart?\n"
        "Patient: No allergies."
    )
    # Note: turn 4 has 'no' inside clause, so 'acetaminophen' must be quarantined while clean turns survive!
    clean_transcript = (
        "Doctor: Walk me through what brought you to urgent care.\n"
        "Patient: Throat soreness.\n"
        "Doctor: Roughly how much time has passed since that started?\n"
        "Patient: Around 4 days.\n"
        "Doctor: Where would that land on mild, moderate, or severe?\n"
        "Patient: Moderate.\n"
        "Doctor: Anything from the pharmacy cabinet you swallowed for relief?\n"
        "Patient: I took acetaminophen.\n"
        "Doctor: Any drug sensitivities on your chart?\n"
        "Patient: No allergies."
    )
    raw = "CC:S[Throat soreness]|DUR:S[4 days]|SEV:S[Moderate]|MED:S[acetaminophen]|ALG:A[No allergies.]"

    regex_draft = scribe(
        clean_transcript,
        solver=compact_solver(lambda _: raw, binder="context"),
        context_policy="context",
    )
    assert regex_draft["disposition"] == "abstain"

    grounded_draft = scribe(
        clean_transcript,
        solver=compact_solver(lambda _: raw, binder="grounded"),
        context_policy="grounded",
    )
    assert grounded_draft["status"] == "draft"
    assert grounded_draft["disposition"] == "review"
    assert grounded_draft["verification"]["passed"] is True
    claims = {c["field"]: c for c in grounded_draft["note"]["claims"]}
    assert claims["chief_complaint"]["state"] == "supported"
    assert claims["chief_complaint"]["value"] == "Throat soreness"
    assert claims["duration"]["value"] == "4 days"
    assert claims["severity"]["value"] == "Moderate"
    assert claims["medication"]["value"] == "acetaminophen"
    assert claims["allergy"]["state"] == "absent"

    # Verify mixed negation clause in turn 4 of first transcript is quarantined locally without killing CC/DUR/SEV/ALG.
    mixed_draft = scribe(
        transcript,
        solver=compact_solver(lambda _: raw, binder="grounded"),
        context_policy="grounded",
    )
    mixed_claims = {c["field"]: c for c in mixed_draft["note"]["claims"]}
    assert mixed_claims["medication"]["state"] == "uncertain"
    assert mixed_claims["chief_complaint"]["state"] == "supported"
    assert mixed_claims["allergy"]["state"] == "absent"


def test_independent_slot_recovery_preserves_valid_assertions_on_partial_malformed_free_output():
    transcript = (
        "Doctor: Tell me why you came in.\n"
        "Patient: Ear ringing.\n"
        "Doctor: How intense is the ringing?\n"
        "Patient: Severe."
    )
    malformed_raw = "CC:S[Ear ringing]|DUR:BROKEN_SYNTAX|SEV:S[Severe]|MED:M|ALG:M"
    result = scribe(
        transcript,
        solver=compact_solver(lambda _: malformed_raw, binder="grounded"),
        context_policy="grounded",
    )
    assert result["status"] == "draft"
    assert result["disposition"] == "review"
    claims = {c["field"]: c for c in result["note"]["claims"]}
    assert claims["chief_complaint"]["value"] == "Ear ringing"
    assert claims["duration"]["state"] == "uncertain"
    assert claims["severity"]["value"] == "Severe"


@pytest.mark.parametrize(
    "transcript,raw",
    [
        (
            "Doctor: What medicines does your mother take?\nPatient: Lisinopril.",
            "CC:M|DUR:M|SEV:M|MED:S[Lisinopril]|ALG:M",
        ),
        (
            "Doctor: What happened during your previous episode last year?\nPatient: Severe dizziness.",
            "CC:S[dizziness]|DUR:M|SEV:S[Severe]|MED:M|ALG:M",
        ),
        (
            "Doctor: Did you take anything?\nPatient: I stopped ibuprofen yesterday.",
            "CC:M|DUR:M|SEV:M|MED:S[ibuprofen]|ALG:M",
        ),
        (
            "Doctor: Any symptoms?\nPatient: Maybe ankle swelling, I am not sure.",
            "CC:S[ankle swelling]|DUR:M|SEV:M|MED:M|ALG:M",
        ),
        (
            "Doctor: Any medicines?\nPatient: I took naproxen.\nPatient: Actually, I was mistaken about naproxen.",
            "CC:M|DUR:M|SEV:M|MED:S[naproxen]|ALG:M",
        ),
        (
            "Doctor: What symptoms?\nPatient: A dry cough.",
            "CC:S[A dry cough]|DUR:M|SEV:M|MED:M|ALG:M",
        ),
        (
            "Doctor: What symptoms?\nPatient: Wrist pain.\nPatient: All of my answers were about my brother.",
            "CC:S[Wrist pain]|DUR:M|SEV:M|MED:M|ALG:M",
        ),
    ],
)
def test_adversarial_hazards_never_emit_presented_assertions(transcript: str, raw: str):
    result = scribe(
        transcript,
        solver=compact_solver(lambda _: raw, binder="grounded"),
        context_policy="grounded",
    )
    assert result["disposition"] == "abstain"
    if result["note"] is not None:
        for claim in result["note"]["claims"]:
            assert claim["state"] not in ("supported", "absent")


def test_grounded_candidate_menu_offers_clean_spans_without_determiners_or_third_party():
    transcript = (
        "Doctor: Walk me through what brought you here.\n"
        "Patient: A dry cough.\n"
        "Doctor: What does your sister take?\n"
        "Patient: Cetirizine.\n"
        "Doctor: What medication did you take today?\n"
        "Patient: Dextromethorphan."
    )
    cc_choices = grounded_candidate_choices(transcript, FieldName.CHIEF_COMPLAINT, set())
    assert ":S[dry cough]" in cc_choices
    assert ":S[A dry cough]" not in cc_choices

    med_choices = grounded_candidate_choices(transcript, FieldName.MEDICATION, set())
    assert ":S[Dextromethorphan]" in med_choices
    assert ":S[Cetirizine]" not in med_choices


def test_grounded_conflicts_and_denials_respect_clause_scope():
    scenarios = [
        ("Patient: I took aspirin. I haven't taken anything.",
         "CC:M|DUR:M|SEV:M|MED:A[I haven't taken anything.]|ALG:M", "medication", "conflicting"),
        ("Doctor: How severe is your pain?\nPatient: Mild.\nDoctor: How severe is your mother's pain?\nPatient: Severe.",
         "CC:M|DUR:M|SEV:S[Mild]|MED:M|ALG:M", "severity", "supported"),
        ("Patient: I'm allergic to latex.\nPatient: I'm not allergic to penicillin.",
         "CC:M|DUR:M|SEV:M|MED:M|ALG:S[latex]", "allergy", "supported"),
        ("Patient: I'm not allergic to penicillin.",
         "CC:M|DUR:M|SEV:M|MED:M|ALG:A[I'm not allergic to penicillin.]", "allergy", "uncertain"),
        ("Patient: I haven't taken any aspirin.",
         "CC:M|DUR:M|SEV:M|MED:A[I haven't taken any aspirin.]|ALG:M", "medication", "uncertain"),
    ]
    for transcript, raw, field, expected in scenarios:
        draft = scribe(transcript, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
        assert draft["verification"]["passed"]
        claim = next(c for c in draft["note"]["claims"] if c["field"] == field)
        assert claim["state"] == expected, (transcript, claim)
