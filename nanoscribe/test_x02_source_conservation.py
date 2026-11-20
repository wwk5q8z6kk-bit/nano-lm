"""Tests for X02 source-conservation lineage and the seven mandatory Section 8.1 failure probes."""

from __future__ import annotations

from nanoscribe.compact import compact_solver
from nanoscribe.grounded import grounded_candidate_choices, trace_proposal_conservation
from nanoscribe.scribe import scribe
from nano_ai.contract import FieldName


def test_x02_probe_1_denial_is_not_absence_and_supersedes_earlier_context():
    transcript = (
        "Doctor: Did the triage note mention aspirin?\n"
        "Patient: I do not take aspirin.\n"
        "Doctor: Have you taken any medications for this headache?\n"
        "Patient: No medications."
    )
    raw = "CC:M|DUR:M|SEV:M|MED:A[No medications.]|ALG:M"
    res = scribe(transcript, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    claims = {c["field"]: c for c in res["note"]["claims"]}
    assert claims["medication"]["state"] == "absent"
    assert claims["medication"]["value"] is None


def test_x02_probe_2_proposal_lineage_detects_silent_drops_and_traces_conservation():
    transcript = (
        "Doctor: What brings you in?\n"
        "Patient: Throat soreness.\n"
        "Doctor: How long?\n"
        "Patient: Three days.\n"
        "Doctor: How severe?\n"
        "Patient: Moderate.\n"
        "Doctor: Any medicines?\n"
        "Patient: I haven't taken any medications.\n"
        "Doctor: Allergies?\n"
        "Patient: Penicillin."
    )
    valid_raw = "CC:S[Throat soreness]|DUR:S[Three days]|SEV:S[Moderate]|MED:A[I haven't taken any medications.]|ALG:S[Penicillin]"
    trace_ok = trace_proposal_conservation(transcript, valid_raw, context_policy="grounded")
    assert all(row["conserved"] is True and row["loss_boundary"] is None for row in trace_ok)

    # Inject a partial malformed slot and verify exact boundary localization without losing valid fields
    broken_raw = "CC:S[Throat soreness]|DUR:BROKEN|SEV:S[Moderate]|MED:M|ALG:S[Penicillin]"
    trace_broken = {r["field"]: r for r in trace_proposal_conservation(transcript, broken_raw, context_policy="grounded")}
    assert trace_broken["chief_complaint"]["conserved"] is True
    assert trace_broken["duration"]["conserved"] is False
    assert trace_broken["duration"]["loss_boundary"] == "slot_parse_or_offset"
    assert trace_broken["severity"]["conserved"] is True
    assert trace_broken["allergy"]["conserved"] is True


def test_x02_probe_3_incompatible_medication_claims_preserve_conflict():
    transcript = (
        "Doctor: Have you taken anything for it?\n"
        "Patient: I took ibuprofen.\n"
        "Doctor: Wait, let me double check what you took.\n"
        "Patient: I haven't taken anything."
    )
    raw = "CC:M|DUR:M|SEV:M|MED:S[ibuprofen]|ALG:M"
    res = scribe(transcript, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    ext = {f["field"]: f for f in res["extraction"]["fields"]}
    assert ext["medication"]["state"] == "conflicting"
    assert len(ext["medication"]["evidence"]) >= 2


def test_x02_probe_4_isolated_uncertainty_retains_supported_assertions():
    transcript = (
        "Doctor: What brings you in?\n"
        "Patient: Ear ringing.\n"
        "Doctor: How severe?\n"
        "Patient: I'm not sure whether it's mild or moderate."
    )
    raw = "CC:S[Ear ringing]|DUR:M|SEV:U[I'm not sure whether it's mild or moderate.]|MED:M|ALG:M"
    res = scribe(transcript, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    assert res["disposition"] == "review"
    claims = {c["field"]: c for c in res["note"]["claims"]}
    assert claims["chief_complaint"]["state"] == "supported"
    assert claims["chief_complaint"]["value"] == "Ear ringing"
    assert claims["severity"]["state"] == "uncertain"


def test_x02_probe_5_novel_rare_single_occurrence_drug_exact_byte_preservation():
    transcript = (
        "Doctor: What brings you in?\n"
        "Patient: Rash on forearm.\n"
        "Doctor: Did you try any medicine?\n"
        "Patient: I took Xyzalorfen-420mg."
    )
    raw = "CC:S[Rash on forearm]|DUR:M|SEV:M|MED:S[Xyzalorfen-420mg]|ALG:M"
    res = scribe(transcript, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    claims = {c["field"]: c for c in res["note"]["claims"]}
    assert claims["medication"]["state"] == "supported"
    assert claims["medication"]["value"] == "Xyzalorfen-420mg"


def test_x02_probe_6_missing_irrelevant_transcript_abstains_without_fabrication():
    transcript = (
        "Doctor: Good morning.\n"
        "Patient: Sorry I am late, traffic was terrible."
    )
    raw = "CC:S[headache]|DUR:M|SEV:M|MED:M|ALG:M"
    res = scribe(transcript, solver=compact_solver(lambda _: raw, binder="grounded"), context_policy="grounded")
    assert res["disposition"] == "abstain"
    if res["note"] is not None:
        assert all(c["state"] not in ("supported", "absent") for c in res["note"]["claims"])


def test_x02_probe_7_medication_timestamp_does_not_leak_into_symptom_duration():
    transcript = (
        "Doctor: What would you like to discuss?\n"
        "Patient: Stomach cramps, starting the evening before last.\n"
        "Doctor: How would you rate them?\n"
        "Patient: Somewhere between mild and moderate; I can't choose one.\n"
        "Doctor: What have you tried for the cramps?\n"
        "Patient: I took ibuprofen and an antacid for them yesterday.\n"
        "Doctor: Any allergies?\n"
        "Patient: No allergies."
    )
    dur_choices = grounded_candidate_choices(transcript, FieldName.DURATION, set())
    assert ":S[the evening before last]" in dur_choices
    assert ":S[yesterday]" not in dur_choices

    raw = (
        "CC:S[Stomach cramps]|DUR:S[the evening before last]|"
        "SEV:U[Somewhere between mild and moderate; I can't choose one.]|"
        "MED:U[ibuprofen||an antacid]|ALG:A[No allergies.]"
    )
    trace = {r["field"]: r for r in trace_proposal_conservation(transcript, raw, context_policy="grounded")}
    assert trace["duration"]["conserved"] is True
    assert trace["duration"]["binder_state"] == "supported"
    assert trace["duration"]["note_rendered_value"] == "the evening before last"


def test_volunteered_treatment_conflicts_with_blanket_denial():
    source = "Doctor: Tell me more.\nPatient: Aspirin for the headache.\nPatient: No medications."
    trace = trace_proposal_conservation(source, "CC:M|DUR:M|SEV:M|MED:A[No medications.]|ALG:M")
    assert trace[3]["binder_state"] == "conflicting"
    for unasserted in ("I could take aspirin for the headache.", "I did not take aspirin for the headache."):
        negative = source.replace("Aspirin for the headache.", unasserted)
        assert trace_proposal_conservation(negative, "CC:M|DUR:M|SEV:M|MED:A[No medications.]|ALG:M")[3]["binder_state"] == "absent"


def test_mixed_clause_duration_uses_the_nearest_event():
    source = "Doctor: Tell me more.\nPatient: I took ibuprofen yesterday because my back has hurt for three days."
    raw = "CC:M|DUR:S[three days]|SEV:M|MED:M|ALG:M"
    assert trace_proposal_conservation(source, raw)[1]["conserved"]
    choices = grounded_candidate_choices(source, FieldName.DURATION, set())
    assert ":S[three days]" in choices
    assert ":S[yesterday]" not in choices


def test_context_lineage_retains_added_support():
    source = "Doctor: What brings you in?\nPatient: Headache.\nDoctor: What brings you in?\nPatient: A headache."
    row = trace_proposal_conservation(source, "CC:S[Headache]|DUR:M|SEV:M|MED:M|ALG:M", context_policy="context")[0]
    assert len(row["binder_spans"]) > len(row["raw_spans"])
    assert row["conserved"]


def test_grounded_actual_fact_and_planned_treatment_in_same_sentence():
    examples = [
        ("I am allergic to penicillin, so I will take aspirin.", FieldName.ALLERGY, "penicillin", "aspirin"),
        ("I took aspirin earlier, but I will take ibuprofen later.", FieldName.MEDICATION, "aspirin", "ibuprofen"),
        ("I will take ibuprofen later, but I took aspirin earlier.", FieldName.MEDICATION, "aspirin", "ibuprofen"),
    ]
    for reply, field, actual, planned in examples:
        source = "Doctor: Tell me more.\nPatient: " + reply
        label, ordinal = ("ALG", 4) if field == FieldName.ALLERGY else ("MED", 3)
        template = "CC:M|DUR:M|SEV:M|MED:M|ALG:M"
        row = trace_proposal_conservation(source, template.replace(label + ":M", f"{label}:S[{actual}]"))[ordinal]
        assert row["note_rendered_value"] == actual
        planned_row = trace_proposal_conservation(source, template.replace("MED:M", f"MED:S[{planned}]"))[3]
        assert planned_row["note_rendered_state"] == "uncertain"
        assert not any(planned in choice for choice in grounded_candidate_choices(source, FieldName.MEDICATION, set()) if choice.startswith(":S["))


def test_ambiguous_on_requires_medication_context():
    for reply in ("I am on my period with cramps.", "I am on edge."):
        source = "Doctor: What brings you in?\nPatient: " + reply
        assert any(c.startswith(":S[") for c in grounded_candidate_choices(source, FieldName.CHIEF_COMPLAINT, set()))
        assert not any(c.startswith(":S[") for c in grounded_candidate_choices(source, FieldName.MEDICATION, set()))
    reply = "Doctor: What medication are you taking?\nPatient: I am on cetirizine."
    assert ":S[cetirizine]" in grounded_candidate_choices(reply, FieldName.MEDICATION, set())


def test_taken_requires_treatment_context_in_both_grounded_menus_and_verification():
    from nano_ai.contract import NanoInput
    from nanoscribe.check_note import transcript_source
    from nanoscribe.grounded import build_grounded_english_menu

    transcript = "Doctor: What brings you in?\nPatient: I have taken a fall and my wrist hurts."
    source = transcript_source(NanoInput("fall", transcript))
    chief = grounded_candidate_choices(transcript, FieldName.CHIEF_COMPLAINT, set())
    assert any("my wrist hurts" in c for c in chief if c.startswith(":S["))
    repeated = transcript.replace("What brings you in?", "Have you taken a fall?")
    assert any(c.startswith(":S[") for c in grounded_candidate_choices(repeated, FieldName.CHIEF_COMPLAINT, set()))
    english = build_grounded_english_menu(source, FieldName.CHIEF_COMPLAINT, set())
    assert any("my wrist hurts" in c for c in english if c.startswith(" REPORTED["))
    assert not any(c.startswith(":S[") for c in grounded_candidate_choices(transcript, FieldName.MEDICATION, set()))
    assert not any(c.startswith(" REPORTED[") for c in build_grounded_english_menu(source, FieldName.MEDICATION, set()))
    raw = "CC:M|DUR:M|SEV:M|MED:S[fall]|ALG:M"
    assert trace_proposal_conservation(transcript, raw)[3]["note_rendered_state"] == "uncertain"


def test_taken_with_a_treatment_question_or_dose_offers_the_medication():
    for question, reply, value in (
        ("What medicines have you tried?", "I have taken ibuprofen.", "ibuprofen"),
        ("Tell me more.", "I've taken ibuprofen 200 mg yesterday.", "ibuprofen 200 mg"),
        ("Tell me more.", "I have taken aspirin for my headache.", "aspirin"),
    ):
        source = f"Doctor: {question}\nPatient: {reply}"
        assert f":S[{value}]" in grounded_candidate_choices(source, FieldName.MEDICATION, set())
        assert not any(c.startswith(":S[") for c in grounded_candidate_choices(source, FieldName.CHIEF_COMPLAINT, set()))
        assert ":S[yesterday]" not in grounded_candidate_choices(source, FieldName.DURATION, set())
        raw = f"CC:M|DUR:M|SEV:M|MED:S[{value}]|ALG:M"
        assert trace_proposal_conservation(source, raw)[3]["note_rendered_value"] == value
