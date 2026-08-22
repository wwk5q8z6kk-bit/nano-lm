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
from nanoscribe.evaluate import PredictedAtom, PredictedEncounter, evaluate
from nanoscribe.select import ConstrainedSelector, match_count, relocate, snap_relocate


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
    pred = PredictedEncounter(
        atoms=(
            PredictedAtom(
                atom_id="atom-neck",
                assertion_state=AssertionState.ASSERTED,
                spans=(gold.span("ev-neck"),),
            ),
        )
    )
    report = evaluate(gold, pred)
    assert report.correct_gold_span == 1
    assert report.char_span_f1 == 1.0
    assert report.wrong_source_span == 0
    assert report.state_correct == 1
    assert report.support_correct == 1


def test_evaluate_wrong_source_span_and_partial_f1() -> None:
    gold = _gold()
    source = gold.sources[0]
    hurting = relocate(source, "hurting", evidence_id="ev-hurt")
    assert hurting is not None
    pred = PredictedEncounter(
        atoms=(
            PredictedAtom(
                atom_id="atom-neck",
                assertion_state=AssertionState.ASSERTED,
                spans=(hurting,),
            ),
        )
    )
    report = evaluate(gold, pred)
    assert report.correct_gold_span == 0
    assert report.wrong_source_span == 1
    assert 0.0 <= report.char_span_f1 < 1.0


def test_evaluate_state_support_omission_and_abstention() -> None:
    gold = _gold()
    pred = PredictedEncounter(
        atoms=(
            PredictedAtom(
                atom_id="atom-neck",
                assertion_state=AssertionState.ASSERTED,
                spans=(gold.span("ev-neck"),),
            ),
            PredictedAtom(
                atom_id="atom-alg",
                assertion_state=AssertionState.DENIED,
                spans=(gold.span("ev-deny"),),
            ),
            PredictedAtom(atom_id="atom-hist", abstained=True),
            PredictedAtom(atom_id="medication", abstained=True),
        )
    )
    report = evaluate(gold, pred)
    assert report.state_correct >= 2
    assert report.unnecessary_abstention == 1  # hist
    assert report.omission == 2  # hist abstained + assess missing
    assert report.correct_abstention == 1  # medication silence


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
            PredictedAtom(
                atom_id="atom-assess",
                assertion_state=AssertionState.ASSERTED,
                spans=(invented,),
            ),
        )
    )
    report = evaluate(gold, pred)
    assert report.malformed >= 1
    assert report.critical_error >= 1


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
        atoms=(
            PredictedAtom(
                atom_id="atom-neck",
                assertion_state=AssertionState.ASSERTED,
                spans=(gold.span("ev-neck"),),
            ),
        ),
        latency_s=0.012,
        memory_bytes=2048,
    )
    report = evaluate(gold, pred)
    assert 0.0 < report.coverage <= 1.0
    assert report.latency_s == 0.012
    assert report.memory_bytes == 2048


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"evidence transport pins: {len(fns)}/{len(fns)} PASS")
