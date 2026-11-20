# Adversarial pins for constrained evidence selection + span evaluation.
# Run: python3 nanoscribe/test_evidence_transport.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.encounter import (
    AssertionState,
    AtomType,
    Certainty,
    ClinicalAtom,
    EncounterError,
    EncounterRecord,
    EvidenceSpan,
    Experiencer,
    Speaker,
    TemporalState,
    Temporality,
    UnresolvedItem,
    assemble_source,
)
from nanoscribe.evaluate import (
    PredictedAtom,
    PredictedEncounter,
    SupportRelation,
    VerifierResult,
    atom_result,
    evaluate,
)
from nanoscribe.select import ConstrainedSelector, copy_span, match_count, relocate, snap_relocate


def _source():
    return assemble_source(
        "src-1",
        (
            (Speaker.CLINICIAN, "What brings you in today?"),
            (Speaker.PATIENT, "My neck has been hurting."),
            (Speaker.CLINICIAN, "I think this is cervical strain."),
            (Speaker.PATIENT, "No allergies."),
            (Speaker.PATIENT, "I used to have migraines years ago."),
        ),
    )


def _gold():
    source = _source()
    neck = relocate(source, "neck", evidence_id="ev-neck")
    deny = relocate(source, "No allergies.", evidence_id="ev-deny")
    hist = relocate(source, "migraines", evidence_id="ev-hist")
    assess = relocate(source, "cervical strain", evidence_id="ev-assess")
    assert neck and deny and hist and assess
    return EncounterRecord(
        encounter_id="enc-1",
        sources=(source,),
        evidence=(neck, deny, hist, assess),
        atoms=(
            ClinicalAtom(
                atom_id="atom-neck",
                atom_type=AtomType.SYMPTOM,
                raw_value="neck",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                evidence_ids=("ev-neck",),
            ),
            ClinicalAtom(
                atom_id="atom-alg",
                atom_type=AtomType.ALLERGY,
                raw_value="allergies",
                assertion_state=AssertionState.DENIED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                evidence_ids=("ev-deny",),
            ),
            ClinicalAtom(
                atom_id="atom-hist",
                atom_type=AtomType.SYMPTOM,
                raw_value="migraines",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.HISTORICAL),
                certainty=Certainty.STATED,
                evidence_ids=("ev-hist",),
            ),
            ClinicalAtom(
                atom_id="atom-assess",
                atom_type=AtomType.ASSESSMENT,
                raw_value="cervical strain",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.CLINICIAN,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                evidence_ids=("ev-assess",),
            ),
        ),
        unresolved=(
            UnresolvedItem(
                unresolved_id="u-med",
                topic="medication",
                reason="silence",
                review_required=True,
            ),
        ),
    )


def _pred_from_gold(gold: EncounterRecord, atom_id: str, **overrides) -> PredictedAtom:
    atom = gold.atom(atom_id)
    spans = tuple(gold.span(evidence_id) for evidence_id in atom.evidence_ids)
    payload = dict(
        atom_id=atom.atom_id,
        atom_type=atom.atom_type,
        raw_value=atom.raw_value,
        assertion_state=atom.assertion_state,
        speaker=atom.speaker,
        experiencer=atom.experiencer,
        temporality=atom.temporality,
        certainty=atom.certainty,
        evidence_ids=atom.evidence_ids,
        spans=spans,
        review_required=atom.review_required,
    )
    payload.update(overrides)
    return PredictedAtom(**payload)


def expect(code: str, fn) -> None:
    try:
        fn()
    except EncounterError as exc:
        assert exc.code == code, f"expected {code}, got {exc.code}: {exc}"
        return
    raise AssertionError(f"expected EncounterError {code}")


def test_copy_span_requires_exact_source_substring() -> None:
    source = _source()
    selector = ConstrainedSelector()
    start = source.text.index("neck")
    span = selector.copy_span(source, start, start + 4, evidence_id="ev-1")
    assert span.text == "neck"
    assert source.text[span.start : span.end] == "neck"
    assert span.speaker is Speaker.PATIENT
    expect(
        "evidence_text_mismatch",
        lambda: selector.copy_span(source, 0, 4, evidence_id="ev-x", text="neck"),
    )


def test_copy_span_rejects_cross_turn_and_invented_text() -> None:
    source = _source()
    selector = ConstrainedSelector()
    t1, t2 = source.turns[1], source.turns[2]
    expect(
        "evidence_crosses_turn",
        lambda: selector.copy_span(source, t1.start, t2.end, evidence_id="ev-cross"),
    )
    # Offsets that would claim text not at that location.
    expect(
        "evidence_text_mismatch",
        lambda: selector.copy_span(source, 0, 4, evidence_id="ev-fake", text="pain"),
    )


def test_relocate_unique_exact_quote_including_clinician() -> None:
    source = _source()
    patient = relocate(source, "neck", evidence_id="ev-neck")
    clinician = relocate(source, "cervical strain", evidence_id="ev-assess")
    assert patient is not None and patient.speaker is Speaker.PATIENT
    assert clinician is not None and clinician.speaker is Speaker.CLINICIAN
    assert relocate(source, "paraphrased pain", evidence_id="ev-none") is None


def test_relocate_abstains_on_ambiguous_or_missing_quote() -> None:
    source = assemble_source(
        "src-amb",
        (
            (Speaker.PATIENT, "It is mild today."),
            (Speaker.CLINICIAN, "Was it mild last week too?"),
        ),
    )
    assert match_count(source, "mild") == 2
    assert relocate(source, "mild", evidence_id="ev-mild") is None
    assert relocate(source, "severe", evidence_id="ev-sev") is None


def test_snap_relocates_surface_variation_not_paraphrase() -> None:
    source = _source()
    snapped = snap_relocate(source, "NECK", evidence_id="ev-neck")
    assert snapped is not None
    assert snapped.text == "neck"
    # Word change is semantic; must not snap.
    assert snap_relocate(source, "cervicalgia", evidence_id="ev-para") is None
    # Unicode quote / dash surface form against an exact clinician span.
    quoted = snap_relocate(source, "cervical  strain", evidence_id="ev-ws")
    assert quoted is not None
    assert quoted.text == "cervical strain"


def test_selector_cannot_emit_non_source_text() -> None:
    source = _source()
    selector = ConstrainedSelector()
    span = selector.select_quote(source, "neck", evidence_id="ev-1")
    assert span is not None
    assert "neck" in source.text
    assert selector.select_quote(source, "the patient has cervicalgia", evidence_id="ev-2") is None


def test_evaluate_exact_gold_span_and_char_f1() -> None:
    gold = _gold()
    pred = PredictedEncounter(atoms=(_pred_from_gold(gold, "atom-neck"),))
    report = evaluate(gold, pred)
    neck = atom_result(report, "atom-neck")
    assert report.exact_gold_span == 1
    assert report.span_character_f1 == 1.0
    assert report.wrong_source == 0
    assert report.wrong_mention == 0
    assert report.assertion_state_correct == 1
    assert report.support_direct_exact == 1
    assert neck.support_relation is SupportRelation.DIRECT_EXACT
    assert not hasattr(report, "support_correct")


def test_evaluate_wrong_mention_is_not_wrong_source() -> None:
    gold = _gold()
    source = gold.sources[0]
    hurting = relocate(source, "hurting", evidence_id="ev-hurt")
    assert hurting is not None
    pred = PredictedEncounter(
        atoms=(
            _pred_from_gold(
                gold,
                "atom-neck",
                raw_value="hurting",
                evidence_ids=("ev-hurt",),
                spans=(hurting,),
            ),
        )
    )
    report = evaluate(gold, pred)
    neck = atom_result(report, "atom-neck")
    assert report.exact_gold_span == 0
    assert report.wrong_mention == 1
    assert report.wrong_source == 0
    assert neck.wrong_mention
    assert not neck.wrong_source
    assert 0.0 <= report.span_character_f1 < 1.0


def test_evaluate_state_support_omission_and_abstention() -> None:
    gold = _gold()
    pred = PredictedEncounter(
        atoms=(
            _pred_from_gold(gold, "atom-neck"),
            _pred_from_gold(gold, "atom-alg"),
            PredictedAtom(atom_id="atom-hist", abstained=True),
            PredictedAtom(atom_id="medication", abstained=True),
        )
    )
    report = evaluate(gold, pred)
    assert report.assertion_state_correct >= 2
    assert report.unnecessary_abstention == 1  # hist
    assert report.omission == 2  # hist abstained + assess missing
    assert report.correct_abstention == 1  # medication silence
    assert atom_result(report, "atom-alg").support_relation is SupportRelation.REVIEW_REQUIRED


def test_evaluate_malformed_invented_and_critical() -> None:
    gold = _gold()
    source = gold.sources[0]
    invented = EvidenceSpan(
        evidence_id="ev-fake",
        source_id=source.source_id,
        turn_id=source.turns[1].turn_id,
        speaker=Speaker.PATIENT,
        start=source.turns[1].start,
        end=source.turns[1].start + 4,
        text="pain",
    )
    pred = PredictedEncounter(
        atoms=(
            PredictedAtom(atom_id="atom-neck", malformed=True),
            _pred_from_gold(gold, "atom-assess", evidence_ids=("ev-fake",), spans=(invented,)),
        )
    )
    report = evaluate(gold, pred)
    assert report.malformed >= 1
    assert report.critical_error >= 1
    assert report.invalid_span >= 1
    assert atom_result(report, "atom-assess").invalid_span


def test_evaluate_ambiguity_is_not_a_guessed_span() -> None:
    source = assemble_source(
        "src-amb",
        (
            (Speaker.PATIENT, "Pain is mild."),
            (Speaker.CLINICIAN, "Still mild?"),
        ),
    )
    assert match_count(source, "mild") == 2
    pred = PredictedEncounter(
        atoms=(
            PredictedAtom(
                atom_id="atom-sev",
                assertion_state=AssertionState.ASSERTED,
                quote="mild",
                abstained=True,
            ),
        )
    )
    gold = EncounterRecord(
        encounter_id="enc-amb",
        sources=(source,),
        evidence=(),
        atoms=(),
        unresolved=(
            UnresolvedItem(
                unresolved_id="u-sev",
                topic="severity",
                reason="ambiguity",
                review_required=True,
            ),
        ),
    )
    report = evaluate(gold, pred, source_for_quotes=source)
    assert report.ambiguity == 1


def test_evaluate_reports_coverage_latency_memory() -> None:
    gold = _gold()
    pred = PredictedEncounter(
        atoms=(_pred_from_gold(gold, "atom-neck"),),
        latency_s=0.012,
        memory_bytes=2048,
    )
    report = evaluate(gold, pred)
    assert 0.0 < report.coverage <= 1.0
    assert report.latency_s == 0.012
    assert report.memory_bytes == 2048


def test_wider_containing_span_is_direct_exact_not_gold_match() -> None:
    gold = _gold()
    source = gold.sources[0]
    turn = source.turns[1]
    wide = copy_span(source, turn.start, turn.end, evidence_id="ev-wide")
    assert wide.text == "My neck has been hurting."
    pred = PredictedEncounter(
        atoms=(
            _pred_from_gold(
                gold,
                "atom-neck",
                evidence_ids=("ev-wide",),
                spans=(wide,),
            ),
        )
    )
    report = evaluate(gold, pred)
    neck = atom_result(report, "atom-neck")
    assert neck.exact_gold_span is False
    assert neck.support_relation is SupportRelation.DIRECT_EXACT
    assert report.exact_gold_span == 0
    assert report.support_direct_exact == 1


def test_normalized_support_is_not_direct_exact() -> None:
    gold = _gold()
    pred = PredictedEncounter(atoms=(_pred_from_gold(gold, "atom-neck", raw_value="Neck"),))
    report = evaluate(gold, pred)
    neck = atom_result(report, "atom-neck")
    assert neck.support_relation is SupportRelation.NORMALIZED
    assert report.support_normalized == 1
    assert report.support_direct_exact == 0
    assert neck.exact_gold_span is True


def test_wrong_source_uses_a_different_source_id() -> None:
    gold = _gold()
    other = assemble_source("src-2", ((Speaker.PATIENT, "My neck has been hurting."),))
    other_neck = relocate(other, "neck", evidence_id="ev-other-neck")
    assert other_neck is not None
    record = EncounterRecord(
        encounter_id="enc-2src",
        sources=gold.sources + (other,),
        evidence=gold.evidence + (other_neck,),
        atoms=gold.atoms,
        unresolved=gold.unresolved,
    )
    pred = PredictedEncounter(
        atoms=(
            _pred_from_gold(
                record,
                "atom-neck",
                evidence_ids=("ev-other-neck",),
                spans=(other_neck,),
            ),
        )
    )
    report = evaluate(record, pred)
    neck = atom_result(report, "atom-neck")
    assert neck.wrong_source
    assert not neck.wrong_mention
    assert report.wrong_source == 1
    assert report.wrong_mention == 0


def test_uncertain_state_is_evaluated() -> None:
    source = assemble_source(
        "src-u",
        ((Speaker.PATIENT, "I am not sure if I take anything."),),
    )
    span = relocate(source, "I am not sure if I take anything.", evidence_id="ev-unsure")
    assert span is not None
    gold = EncounterRecord(
        encounter_id="enc-u",
        sources=(source,),
        evidence=(span,),
        atoms=(
            ClinicalAtom(
                atom_id="atom-med",
                atom_type=AtomType.MEDICATION,
                raw_value="anything",
                assertion_state=AssertionState.UNCERTAIN,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.UNCERTAIN,
                evidence_ids=("ev-unsure",),
            ),
        ),
    )
    report = evaluate(gold, PredictedEncounter(atoms=(_pred_from_gold(gold, "atom-med"),)))
    med = atom_result(report, "atom-med")
    assert med.assertion_state_correct
    assert med.support_relation is SupportRelation.DIRECT_EXACT
    assert report.assertion_state_correct == 1


def test_conflicting_evaluates_both_spans() -> None:
    source = assemble_source(
        "src-c",
        (
            (Speaker.PATIENT, "Pain is mild."),
            (Speaker.PATIENT, "Pain is severe."),
        ),
    )
    mild = relocate(source, "mild", evidence_id="ev-mild")
    severe = relocate(source, "severe", evidence_id="ev-severe")
    assert mild and severe
    gold = EncounterRecord(
        encounter_id="enc-c",
        sources=(source,),
        evidence=(mild, severe),
        atoms=(
            ClinicalAtom(
                atom_id="atom-sev",
                atom_type=AtomType.SYMPTOM,
                raw_value="mild",
                assertion_state=AssertionState.CONFLICTING,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.UNKNOWN,
                evidence_ids=("ev-mild", "ev-severe"),
            ),
        ),
    )
    report = evaluate(gold, PredictedEncounter(atoms=(_pred_from_gold(gold, "atom-sev"),)))
    item = atom_result(report, "atom-sev")
    assert item.exact_gold_span
    assert item.span_character_f1 == 1.0
    assert item.support_relation is SupportRelation.REVIEW_REQUIRED
    assert report.support_review_required == 1


def test_span_order_and_second_span_are_not_first_span_artifacts() -> None:
    gold = _gold()
    source = gold.sources[0]
    hurting = relocate(source, "hurting", evidence_id="ev-hurt")
    assert hurting is not None
    multi = EncounterRecord(
        encounter_id="enc-multi",
        sources=gold.sources,
        evidence=gold.evidence + (hurting,),
        atoms=(
            ClinicalAtom(
                atom_id="atom-neck",
                atom_type=AtomType.SYMPTOM,
                raw_value="neck",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                evidence_ids=("ev-neck", "ev-hurt"),
            ),
        ),
    )
    swapped = PredictedEncounter(
        atoms=(
            _pred_from_gold(
                multi,
                "atom-neck",
                evidence_ids=("ev-hurt", "ev-neck"),
                spans=(hurting, multi.span("ev-neck")),
            ),
        )
    )
    swapped_report = evaluate(multi, swapped)
    assert atom_result(swapped_report, "atom-neck").exact_gold_span

    turn = source.turns[1]
    wide = copy_span(source, turn.start, turn.end, evidence_id="ev-wide-first")
    first_differs = PredictedEncounter(
        atoms=(
            _pred_from_gold(
                multi,
                "atom-neck",
                evidence_ids=("ev-wide-first", "ev-hurt"),
                spans=(wide, hurting),
            ),
        )
    )
    partial = evaluate(multi, first_differs)
    item = atom_result(partial, "atom-neck")
    assert item.exact_gold_span is False
    assert item.span_character_f1 > 0.0
    assert item.wrong_mention


def test_duplicate_predicted_ids_are_malformed() -> None:
    gold = _gold()
    pred = PredictedEncounter(
        atoms=(
            _pred_from_gold(gold, "atom-neck"),
            _pred_from_gold(gold, "atom-neck"),
        )
    )
    report = evaluate(gold, pred)
    assert report.malformed >= 1
    assert atom_result(report, "atom-neck").malformed


def test_nonexistent_source_is_critical_and_does_not_crash() -> None:
    gold = _gold()
    ghost = EvidenceSpan(
        evidence_id="ev-ghost",
        source_id="no-such-source",
        turn_id="no-such-turn",
        speaker=Speaker.PATIENT,
        start=0,
        end=4,
        text="neck",
    )
    pred = PredictedEncounter(
        atoms=(
            _pred_from_gold(
                gold,
                "atom-neck",
                evidence_ids=("ev-ghost",),
                spans=(ghost,),
            ),
        )
    )
    report = evaluate(gold, pred)
    neck = atom_result(report, "atom-neck")
    assert report.critical_error >= 1
    assert neck.critical_error
    assert not neck.wrong_source


def test_wrong_turn_and_speaker_are_invalid_spans() -> None:
    gold = _gold()
    source = gold.sources[0]
    neck = gold.span("ev-neck")
    wrong_turn = EvidenceSpan(
        evidence_id="ev-bad-turn",
        source_id=neck.source_id,
        turn_id=source.turns[0].turn_id,
        speaker=neck.speaker,
        start=neck.start,
        end=neck.end,
        text=neck.text,
    )
    wrong_speaker = EvidenceSpan(
        evidence_id="ev-bad-spk",
        source_id=neck.source_id,
        turn_id=neck.turn_id,
        speaker=Speaker.CLINICIAN,
        start=neck.start,
        end=neck.end,
        text=neck.text,
    )
    turn_report = evaluate(
        gold,
        PredictedEncounter(
            atoms=(_pred_from_gold(gold, "atom-neck", evidence_ids=("ev-bad-turn",), spans=(wrong_turn,)),)
        ),
    )
    speaker_report = evaluate(
        gold,
        PredictedEncounter(
            atoms=(
                _pred_from_gold(gold, "atom-neck", evidence_ids=("ev-bad-spk",), spans=(wrong_speaker,)),
            )
        ),
    )
    assert atom_result(turn_report, "atom-neck").invalid_span
    assert atom_result(speaker_report, "atom-neck").invalid_span
    assert turn_report.critical_error >= 1
    assert speaker_report.critical_error >= 1


def test_ungrounded_raw_value_is_malformed() -> None:
    gold = _gold()
    pred = PredictedEncounter(atoms=(_pred_from_gold(gold, "atom-neck", raw_value="diabetes"),))
    report = evaluate(gold, pred)
    neck = atom_result(report, "atom-neck")
    assert neck.malformed
    assert neck.support_relation is None
    assert neck.support_relation is not SupportRelation.SEMANTICALLY_SUPPORTED
    assert report.malformed >= 1
    assert report.support_direct_exact == 0
    assert report.support_unsupported == 0


def test_uncertain_without_evidence_is_malformed() -> None:
    gold = _gold()
    pred = PredictedEncounter(
        atoms=(
            _pred_from_gold(
                gold,
                "atom-neck",
                assertion_state=AssertionState.UNCERTAIN,
                certainty=Certainty.UNCERTAIN,
                evidence_ids=(),
                spans=(),
            ),
        )
    )
    report = evaluate(gold, pred)
    assert atom_result(report, "atom-neck").malformed
    assert report.malformed >= 1


def test_nonauthoritative_without_review_is_malformed() -> None:
    source = assemble_source("src-o", ((Speaker.OTHER, "He has diabetes."),))
    span = relocate(source, "diabetes", evidence_id="ev-o")
    assert span is not None
    gold = EncounterRecord(
        encounter_id="enc-o",
        sources=(source,),
        evidence=(span,),
        atoms=(
            ClinicalAtom(
                atom_id="atom-dm",
                atom_type=AtomType.DIAGNOSIS_STATEMENT,
                raw_value="diabetes",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.OTHER,
                experiencer=Experiencer.OTHER,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                evidence_ids=("ev-o",),
                review_required=True,
            ),
        ),
    )
    pred = PredictedEncounter(
        atoms=(_pred_from_gold(gold, "atom-dm", review_required=False),)
    )
    report = evaluate(gold, pred)
    assert atom_result(report, "atom-dm").malformed
    assert report.malformed >= 1


def _all_gold_preds(gold: EncounterRecord) -> tuple[PredictedAtom, ...]:
    return tuple(_pred_from_gold(gold, atom.atom_id) for atom in gold.atoms)


def _extra_atom(gold: EncounterRecord, atom_id: str) -> PredictedAtom:
    neck = gold.span("ev-neck")
    return PredictedAtom(
        atom_id=atom_id,
        atom_type=AtomType.DIAGNOSIS_STATEMENT,
        raw_value="neck",
        assertion_state=AssertionState.ASSERTED,
        speaker=Speaker.PATIENT,
        experiencer=Experiencer.PATIENT,
        temporality=TemporalState(kind=Temporality.CURRENT),
        certainty=Certainty.STATED,
        evidence_ids=("ev-neck",),
        spans=(neck,),
    )


def _report_is_clean(report) -> bool:
    return (
        report.omission == 0
        and report.malformed == 0
        and report.critical_error == 0
        and report.spurious_atom == 0
        and report.wrong_source == 0
        and report.wrong_mention == 0
        and report.invalid_span == 0
    )


def test_one_valid_extra_atom_is_spurious() -> None:
    gold = _gold()
    pred = PredictedEncounter(
        atoms=_all_gold_preds(gold) + (_extra_atom(gold, "atom-hallucinated-cancer"),)
    )
    report = evaluate(gold, pred)
    extra = atom_result(report, "atom-hallucinated-cancer")
    assert report.spurious_atom == 1
    assert extra.spurious_atom
    assert not extra.malformed


def test_ten_extra_atoms_cannot_keep_a_perfect_report() -> None:
    gold = _gold()
    extras = tuple(_extra_atom(gold, f"atom-hallucinated-{index}") for index in range(10))
    clean = evaluate(gold, PredictedEncounter(atoms=_all_gold_preds(gold)))
    assert _report_is_clean(clean)
    assert clean.exact_gold_span == len(gold.atoms)
    report = evaluate(gold, PredictedEncounter(atoms=_all_gold_preds(gold) + extras))
    assert report.spurious_atom == 10
    assert report.exact_gold_span == len(gold.atoms)
    assert not _report_is_clean(report)


def test_duplicate_unknown_prediction_ids_are_malformed() -> None:
    gold = _gold()
    extra = _extra_atom(gold, "atom-hallucinated-diagnosis")
    pred = PredictedEncounter(atoms=_all_gold_preds(gold) + (extra, extra))
    report = evaluate(gold, pred)
    assert report.malformed >= 1
    assert report.spurious_atom == 0
    assert atom_result(report, "atom-hallucinated-diagnosis").malformed


def test_unresolved_abstention_is_not_spurious() -> None:
    gold = _gold()
    pred = PredictedEncounter(
        atoms=_all_gold_preds(gold)
        + (PredictedAtom(atom_id="medication", abstained=True),)
    )
    report = evaluate(gold, pred)
    assert report.spurious_atom == 0
    assert report.correct_abstention == 1
    assert _report_is_clean(report)


def test_prediction_cannot_self_declare_semantic_support() -> None:
    gold = _gold()
    try:
        PredictedAtom(  # type: ignore[call-arg]
            atom_id="atom-alg",
            verifier_support=SupportRelation.SEMANTICALLY_SUPPORTED,
        )
    except TypeError:
        pass
    else:
        raise AssertionError("PredictedAtom must not accept verifier_support")
    report = evaluate(gold, PredictedEncounter(atoms=(_pred_from_gold(gold, "atom-alg"),)))
    assert atom_result(report, "atom-alg").support_relation is SupportRelation.REVIEW_REQUIRED
    assert report.support_semantically_supported == 0


def test_independent_verifier_can_declare_semantic_support() -> None:
    gold = _gold()
    pred = PredictedEncounter(atoms=(_pred_from_gold(gold, "atom-alg"),))
    report = evaluate(
        gold,
        pred,
        verifier_results=(
            VerifierResult("atom-alg", SupportRelation.SEMANTICALLY_SUPPORTED),
        ),
    )
    assert atom_result(report, "atom-alg").support_relation is SupportRelation.SEMANTICALLY_SUPPORTED
    assert report.support_semantically_supported == 1
    mapping_report = evaluate(
        gold,
        pred,
        verifier_results={"atom-alg": SupportRelation.CONTRADICTED},
    )
    assert (
        atom_result(mapping_report, "atom-alg").support_relation
        is SupportRelation.CONTRADICTED
    )


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"evidence transport pins: {len(fns)}/{len(fns)} PASS")
