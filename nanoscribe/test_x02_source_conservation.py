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


def test_open_treatment_indication_offers_bare_medication_answers_in_both_menus():
    from nano_ai.contract import NanoInput
    from nanoscribe.check_note import transcript_source
    from nanoscribe.grounded import build_grounded_english_menu

    for indication, answer in (("the cramps", "Ibuprofen"), ("nausea", "Ondansetron"),
                               ("these symptoms", "Acetaminophen"), ("your allergies", "Cetirizine")):
        transcript = f"Doctor: What have you taken for {indication}?\nPatient: {answer}."
        source = transcript_source(NanoInput("treatment", transcript))
        assert f":S[{answer}]" in grounded_candidate_choices(transcript, FieldName.MEDICATION, set())
        assert f" REPORTED[{answer}]" in build_grounded_english_menu(source, FieldName.MEDICATION, set())
        assert not any(c.startswith(":S[") for c in grounded_candidate_choices(transcript, FieldName.ALLERGY, set()))

    for question in ("What have you taken for granted?", "Have you taken a trip for the holidays?"):
        transcript = f"Doctor: {question}\nPatient: Travel."
        source = transcript_source(NanoInput("other", transcript))
        assert not any(c.startswith(":S[") for c in grounded_candidate_choices(transcript, FieldName.MEDICATION, set()))
        assert not any(c.startswith(" REPORTED[") for c in build_grounded_english_menu(source, FieldName.MEDICATION, set()))


def test_reported_treatment_indication_preserves_named_substance_in_both_menus():
    from nano_ai.contract import NanoInput
    from nanoscribe.check_note import transcript_source
    from nanoscribe.grounded import build_grounded_english_menu

    for value, indication in (("aspirin", "my cramps"), ("ondansetron", "my nausea"),
                              ("acetaminophen", "soreness")):
        transcript = f"Doctor: Tell me more.\nPatient: I have taken {value} for {indication}."
        source = transcript_source(NanoInput("treatment", transcript))
        assert f":S[{value}]" in grounded_candidate_choices(transcript, FieldName.MEDICATION, set())
        assert f" REPORTED[{value}]" in build_grounded_english_menu(source, FieldName.MEDICATION, set())
        row = trace_proposal_conservation(transcript, f"CC:M|DUR:M|SEV:M|MED:S[{value}]|ALG:M")[3]
        assert row["note_rendered_value"] == value
        name = indication.split()[-1]
        assert trace_proposal_conservation(transcript, f"CC:M|DUR:M|SEV:M|MED:S[{name}]|ALG:M")[3]["note_rendered_state"] == "uncertain"

    for reply, proposed in (("I have taken a trip for my birthday.", "trip"),
                            ("I have taken it for granted.", "it"),
                            ("I have taken time off for my cramps.", "time off"),
                            ("I have taken a break for my headache.", "break"),
                            ("I have taken a break for two days.", "break")):
        transcript = f"Doctor: Tell me more.\nPatient: {reply}"
        assert not any(c.startswith(":S[") for c in grounded_candidate_choices(transcript, FieldName.MEDICATION, set()))
        assert trace_proposal_conservation(transcript, f"CC:M|DUR:M|SEV:M|MED:S[{proposed}]|ALG:M")[3]["note_rendered_state"] == "uncertain"


def test_duration_after_answer_belongs_to_the_separate_patient_report():
    from nano_ai.contract import NanoInput
    from nanoscribe.check_note import transcript_source
    from nanoscribe.grounded import build_grounded_english_menu

    for question, field, value, reply, duration in (
        ("Any medicines?", FieldName.MEDICATION, "Aspirin", "Aspirin. My headache has persisted for three days.", "three days"),
        ("Any medicines?", FieldName.MEDICATION, "Ibuprofen", "Ibuprofen; the pressure has returned every day for two weeks.", "two weeks"),
        ("Any medicines?", FieldName.MEDICATION, "Aspirin", "Aspirin. Three days ago my headache returned.", "Three days ago"),
        ("Any allergies?", FieldName.ALLERGY, "Penicillin", "Penicillin. My throat has felt scratchy for four days.", "four days"),
    ):
        transcript = f"Doctor: {question}\nPatient: {reply}"
        source = transcript_source(NanoInput("separate", transcript))
        assert f":S[{duration}]" in grounded_candidate_choices(transcript, FieldName.DURATION, set())
        english = build_grounded_english_menu(source, FieldName.DURATION, set())
        spans = english[f" REPORTED[{duration}]"][1]
        assert [(s.start, s.end) for s in spans] == [(transcript.index(duration), transcript.index(duration) + len(duration))]
        assert f":S[{value}]" in grounded_candidate_choices(transcript, field, set())
        rows = trace_proposal_conservation(transcript, f"CC:M|DUR:S[{duration}]|SEV:M|MED:M|ALG:M")
        assert rows[1]["conserved"] and rows[1]["note_rendered_value"] == duration


def test_medication_answer_timing_stays_with_the_treatment():
    for reply, duration in (("Aspirin yesterday.", "yesterday"),
                            ("Aspirin. I took it yesterday.", "yesterday"),
                            ("Aspirin. Just yesterday.", "yesterday"),
                            ("Aspirin. I stopped taking it yesterday.", "yesterday"),
                            ("Aspirin. Yesterday I took it because my back has hurt for three days.", "Yesterday"),
                            ("Aspirin. For two days.", "two days"),
                            ("Aspirin. My last dose was two days ago.", "two days ago")):
        transcript = f"Doctor: Any medicines?\nPatient: {reply}"
        assert f":S[{duration}]" not in grounded_candidate_choices(transcript, FieldName.DURATION, set())
        row = trace_proposal_conservation(transcript, f"CC:M|DUR:S[{duration}]|SEV:M|MED:M|ALG:M")[1]
        assert row["note_rendered_state"] == "uncertain"


def test_conservation_checks_the_actual_conflict_evidence_at_the_binder():
    text = "Patient: I took aspirin.\nPatient: No medications.\nPatient: I took ibuprofen."
    raw = "CC:M|DUR:M|SEV:M|MED:C[{}]|ALG:M"
    replaced = trace_proposal_conservation(text, raw.format("aspirin;ibuprofen"))[3]
    assert replaced["binder_state"] == replaced["raw_state"] == "conflicting"
    assert not replaced["conserved"] and replaced["loss_boundary"] == "binder_rejected"
    assert [s["text"] for s in replaced["binder_spans"]] == ["I took aspirin.", "No medications."]
    for pair in ("I took aspirin.;No medications.", "No medications.;I took aspirin."):
        retained = trace_proposal_conservation(text, raw.format(pair))[3]
        assert retained["conserved"] and retained["loss_boundary"] is None
        assert retained["raw_spans"] == retained["binder_spans"]


def test_coordinated_taken_names_keep_the_exact_coexisting_treatment_evidence():
    from nano_ai.contract import NanoInput
    from nanoscribe.check_note import transcript_source
    from nanoscribe.grounded import build_grounded_english_menu

    for text in ("aspirin for my headache and ibuprofen for my cramps",
                 "aspirin and ibuprofen for my cramps"):
        transcript = f"Doctor: Tell me more.\nPatient: I have taken {text}."
        source = transcript_source(NanoInput("taken", transcript))
        menu = grounded_candidate_choices(transcript, FieldName.MEDICATION, set())
        assert ":U[aspirin||ibuprofen]" in menu
        assert not any(item.startswith(":S[") for item in menu)
        english = build_grounded_english_menu(source, FieldName.MEDICATION, set())
        assert any([s.text for s in spans] == ["aspirin", "ibuprofen"] for code, spans in english.values() if code == "U")
        for name in ("aspirin", "ibuprofen"):
            result = trace_proposal_conservation(transcript, f"CC:M|DUR:M|SEV:M|MED:S[{name}]|ALG:M")[3]
            assert result["binder_state"] == "uncertain"
            assert [(s["start"], s["end"], s["text"]) for s in result["binder_spans"]] == [
                (transcript.index(v), transcript.index(v) + len(v), v) for v in ("aspirin", "ibuprofen")]

    transcript = "Patient: I have taken aspirin for my headache and sore throat."
    assert ":S[aspirin]" in grounded_candidate_choices(transcript, FieldName.MEDICATION, set())
    for proposed in ("headache", "sore throat"):
        result = trace_proposal_conservation(transcript, f"CC:M|DUR:M|SEV:M|MED:S[{proposed}]|ALG:M")[3]
        assert result["note_rendered_state"] == "uncertain"


def test_symptom_duration_does_not_inherit_time_requested_for_a_treatment():
    from nano_ai.contract import NanoInput
    from nanoscribe.check_note import transcript_source
    from nanoscribe.grounded import build_grounded_english_menu

    intro = "Doctor: What medication are you taking now?\nPatient: levothyroxine\n"
    for question, reply, timing in (
        ("When did you start that medication?", "on Monday", "Monday"),
        ("When was your last dose?", "Yesterday morning.", "Yesterday"),
        ("How long have you been taking it?", "about forty minutes", "about forty minutes"),
        ("What date did you stop taking the tablets?", "since 10/02/2026", "since 10/02/2026"),
        ("When did you start levothyroxine?", "I started it on Tuesday.", "Tuesday"),
    ):
        transcript = intro + f"Doctor: {question}\nPatient: {reply}"
        source = transcript_source(NanoInput("drug-timing", transcript))
        assert not any(t.startswith(":S[") for t in grounded_candidate_choices(transcript, FieldName.DURATION, set()))
        assert not any(t.startswith(" REPORTED[") for t in build_grounded_english_menu(source, FieldName.DURATION, set()))
        result = trace_proposal_conservation(transcript, f"CC:M|DUR:S[{timing}]|SEV:M|MED:M|ALG:M")[1]
        assert result["binder_state"] == "uncertain"

    for question, reply, timing in (
        ("When did your headache start after you took the tablets?", "on Monday", "Monday"),
        ("When did you start feeling dizzy on that medication?", "on Monday", "Monday"),
        ("When did you start that medication?", "On Monday my jaw tightness returned.", "Monday"),
        ("How long has the pressure lasted?", "about forty minutes", "about forty minutes"),
        ("When did your symptoms start?", "since 10/02/2026", "since 10/02/2026"),
        ("When was your last dose?", "On Monday. My headache has persisted for three days.", "three days"),
        ("When did you take your medication?", "My jaw tightness has persisted for two days.", "two days"),
    ):
        transcript = intro + f"Doctor: {question}\nPatient: {reply}"
        assert f":S[{timing}]" in grounded_candidate_choices(transcript, FieldName.DURATION, set())
        assert trace_proposal_conservation(transcript, f"CC:M|DUR:S[{timing}]|SEV:M|MED:M|ALG:M")[1]["conserved"]


def test_administration_adjuncts_do_not_displace_reported_medications():
    from nano_ai.contract import NanoInput
    from nanoscribe.check_note import transcript_source
    from nanoscribe.grounded import build_grounded_english_menu

    for adjunct in ("water", "a full glass of water", "a sip of juice", "food", "my breakfast"):
        for medication in ("aspirin", "aspirin and ibuprofen"):
            transcript = f"Doctor: Tell me more.\nPatient: I have taken {medication} with {adjunct} for my headache."
            source = transcript_source(NanoInput("adjunct", transcript))
            choices = grounded_candidate_choices(transcript, FieldName.MEDICATION, set())
            english = build_grounded_english_menu(source, FieldName.MEDICATION, set())
            result = trace_proposal_conservation(transcript, "CC:M|DUR:M|SEV:M|MED:S[aspirin]|ALG:M")[3]
            if medication == "aspirin":
                assert ":S[aspirin]" in choices and " REPORTED[aspirin]" in english
                assert result["conserved"] and result["note_rendered_value"] == "aspirin"
            else:
                assert ":U[aspirin||ibuprofen]" in choices
                assert any([s.text for s in spans] == ["aspirin", "ibuprofen"] for code, spans in english.values() if code == "U")
            assert all(s["text"] not in adjunct for s in result["binder_spans"])
            rejected = trace_proposal_conservation(transcript, f"CC:M|DUR:M|SEV:M|MED:S[{adjunct}]|ALG:M")[3]
            assert not rejected["conserved"]

    for coordinated in ("aspirin with ibuprofen", "aspirin with milk thistle"):
        transcript = f"Patient: I have taken {coordinated} for my cramps."
        assert ":U[" + coordinated.replace(" with ", "||") + "]" in grounded_candidate_choices(transcript, FieldName.MEDICATION, set())
