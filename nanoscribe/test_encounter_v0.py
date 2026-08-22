# Adversarial contract pins for Encounter Representation v0.
# Run: python3 nanoscribe/test_encounter_v0.py
# Stdlib only — no model, no pytest, no PHI.
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.encounter import (
    SCHEMA_VERSION,
    AssertionState,
    AtomType,
    Certainty,
    ClinicalAtom,
    Conflict,
    EncounterError,
    EncounterRecord,
    EvidenceSpan,
    Experiencer,
    Speaker,
    TemporalState,
    Temporality,
    UnresolvedItem,
    apply_normalization,
    assemble_source,
)

TRANSFORM = "strip_articles_nfkc"


def _source():
    return assemble_source(
        "src-1",
        (
            (Speaker.CLINICIAN, "What brings you in today?"),
            (Speaker.PATIENT, "My neck has been hurting."),
            (Speaker.CLINICIAN, "How long?"),
            (Speaker.PATIENT, "About 6 days."),
            (Speaker.PATIENT, "My mother had breast cancer."),
            (Speaker.CLINICIAN, "I think this is cervical strain."),
            (Speaker.CLINICIAN, "Start ibuprofen and follow up in a week."),
            (Speaker.PATIENT, "I am not sure if I take anything."),
            (Speaker.PATIENT, "No allergies."),
            (Speaker.PATIENT, "I used to have migraines years ago."),
        ),
    )


def _span(source, turn_index: int, text: str, evidence_id: str) -> EvidenceSpan:
    turn = source.turns[turn_index]
    rel = turn.text.index(text)
    start = turn.start + rel
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source.source_id,
        turn_id=turn.turn_id,
        speaker=turn.speaker,
        start=start,
        end=start + len(text),
        text=text,
    )


def _atom(**kwargs) -> ClinicalAtom:
    defaults = dict(
        atom_id="atom-1",
        atom_type=AtomType.SYMPTOM,
        raw_value="neck",
        assertion_state=AssertionState.ASSERTED,
        speaker=Speaker.PATIENT,
        experiencer=Experiencer.PATIENT,
        temporality=TemporalState(kind=Temporality.CURRENT),
        certainty=Certainty.STATED,
        evidence_ids=("ev-neck",),
        review_required=False,
    )
    defaults.update(kwargs)
    return ClinicalAtom(**defaults)


def _record(*atoms: ClinicalAtom, source=None, extra_spans=(), **kwargs) -> EncounterRecord:
    source = source or _source()
    spans = [
        _span(source, 1, "neck", "ev-neck"),
        _span(source, 3, "6 days", "ev-dur"),
        _span(source, 4, "breast cancer", "ev-mom"),
        _span(source, 5, "cervical strain", "ev-assess"),
        _span(source, 6, "Start ibuprofen and follow up in a week.", "ev-plan"),
        _span(source, 7, "I am not sure if I take anything.", "ev-unsure"),
        _span(source, 8, "No allergies.", "ev-deny"),
        _span(source, 9, "migraines", "ev-hist"),
        *extra_spans,
    ]
    defaults = dict(
        encounter_id="enc-1",
        sources=(source,),
        evidence=tuple(spans),
        atoms=atoms,
    )
    defaults.update(kwargs)
    return EncounterRecord(**defaults)


def expect(code: str, fn) -> None:
    try:
        fn()
    except EncounterError as exc:
        assert exc.code == code, f"expected {code}, got {exc.code}: {exc}"
        return
    raise AssertionError(f"expected EncounterError {code}")


def test_exact_offsets_and_text_match() -> None:
    source = _source()
    span = _span(source, 1, "neck", "ev-neck")
    assert source.text[span.start : span.end] == "neck"
    rec = _record(_atom())
    rec.validate()


def test_wrong_offsets_rejected() -> None:
    source = _source()
    bad = EvidenceSpan(
        evidence_id="ev-bad",
        source_id=source.source_id,
        turn_id=source.turns[1].turn_id,
        speaker=Speaker.PATIENT,
        start=0,
        end=4,
        text="neck",
    )
    expect("evidence_text_mismatch", lambda: _record(_atom(evidence_ids=("ev-bad",)), extra_spans=(bad,)))


def test_wrong_evidence_text_rejected() -> None:
    source = _source()
    turn = source.turns[1]
    start = turn.start + turn.text.index("neck")
    bad = EvidenceSpan(
        evidence_id="ev-bad",
        source_id=source.source_id,
        turn_id=turn.turn_id,
        speaker=Speaker.PATIENT,
        start=start,
        end=start + 4,
        text="pain",
    )
    expect("evidence_text_mismatch", lambda: _record(_atom(evidence_ids=("ev-bad",)), extra_spans=(bad,)))


def test_wrong_speaker_rejected() -> None:
    source = _source()
    turn = source.turns[1]
    start = turn.start + turn.text.index("neck")
    bad = EvidenceSpan(
        evidence_id="ev-bad",
        source_id=source.source_id,
        turn_id=turn.turn_id,
        speaker=Speaker.CLINICIAN,
        start=start,
        end=start + 4,
        text="neck",
    )
    expect("evidence_speaker_mismatch", lambda: _record(_atom(evidence_ids=("ev-bad",)), extra_spans=(bad,)))


def test_cross_turn_span_rejected() -> None:
    source = _source()
    t1, t2 = source.turns[1], source.turns[2]
    bad = EvidenceSpan(
        evidence_id="ev-cross",
        source_id=source.source_id,
        turn_id=t1.turn_id,
        speaker=t1.speaker,
        start=t1.start,
        end=t2.end,
        text=source.text[t1.start : t2.end],
    )
    expect(
        "evidence_crosses_turn",
        lambda: _record(_atom(evidence_ids=("ev-cross",)), extra_spans=(bad,)),
    )


def test_duplicate_ids_rejected() -> None:
    source = _source()
    dup = _span(source, 1, "hurting", "ev-neck")
    expect("duplicate_id", lambda: _record(_atom(), extra_spans=(dup,)))


def test_denial_requires_explicit_evidence() -> None:
    rec = _record(
        _atom(
            atom_id="atom-alg",
            atom_type=AtomType.ALLERGY,
            raw_value="allergies",
            assertion_state=AssertionState.DENIED,
            evidence_ids=("ev-deny",),
        )
    )
    rec.validate()


def test_absence_from_silence_rejected() -> None:
    expect(
        "denied_without_evidence",
        lambda: _record(
            _atom(
                atom_id="atom-alg",
                atom_type=AtomType.ALLERGY,
                raw_value="allergies",
                assertion_state=AssertionState.DENIED,
                evidence_ids=(),
            )
        ),
    )


def test_uncertain_statement_is_not_denial() -> None:
    rec = _record(
        _atom(
            atom_id="atom-med",
            atom_type=AtomType.MEDICATION,
            raw_value="anything",
            assertion_state=AssertionState.UNCERTAIN,
            certainty=Certainty.UNCERTAIN,
            evidence_ids=("ev-unsure",),
        )
    )
    rec.validate()
    expect(
        "denied_without_denial_state",
        lambda: _record(
            _atom(
                atom_id="atom-med",
                atom_type=AtomType.MEDICATION,
                raw_value="anything",
                assertion_state=AssertionState.DENIED,
                evidence_ids=("ev-unsure",),
            )
        ),
    )


def test_conflicting_statements_need_distinct_evidence() -> None:
    source = _source()
    extra = _span(source, 1, "hurting", "ev-neck-2")
    rec = _record(
        _atom(
            assertion_state=AssertionState.CONFLICTING,
            evidence_ids=("ev-neck", "ev-neck-2"),
        ),
        extra_spans=(extra,),
        conflicts=(
            Conflict(
                conflict_id="c-1",
                atom_ids=("atom-1",),
                evidence_ids=("ev-neck", "ev-neck-2"),
            ),
        ),
    )
    rec.validate()
    expect(
        "insufficient_conflict_evidence",
        lambda: _record(_atom(assertion_state=AssertionState.CONFLICTING, evidence_ids=("ev-neck",))),
    )


def test_patient_and_clinician_attribution() -> None:
    rec = _record(
        _atom(),
        _atom(
            atom_id="atom-assess",
            atom_type=AtomType.ASSESSMENT,
            raw_value="cervical strain",
            speaker=Speaker.CLINICIAN,
            experiencer=Experiencer.PATIENT,
            evidence_ids=("ev-assess",),
        ),
        _atom(
            atom_id="atom-plan",
            atom_type=AtomType.PLAN,
            raw_value="Start ibuprofen and follow up in a week.",
            speaker=Speaker.CLINICIAN,
            temporality=TemporalState(kind=Temporality.FUTURE),
            evidence_ids=("ev-plan",),
        ),
    )
    rec.validate()
    assert rec.atom("atom-assess").speaker is Speaker.CLINICIAN
    assert rec.atom("atom-plan").speaker is Speaker.CLINICIAN


def test_family_member_experiencer() -> None:
    rec = _record(
        _atom(
            atom_id="atom-fh",
            atom_type=AtomType.HISTORY,
            raw_value="breast cancer",
            experiencer=Experiencer.OTHER,
            temporality=TemporalState(kind=Temporality.HISTORICAL),
            evidence_ids=("ev-mom",),
        )
    )
    rec.validate()
    assert rec.atom("atom-fh").experiencer is Experiencer.OTHER
    expect(
        "experiencer_mismatch",
        lambda: _record(
            _atom(
                atom_id="atom-fh",
                atom_type=AtomType.HISTORY,
                raw_value="breast cancer",
                experiencer=Experiencer.PATIENT,
                evidence_ids=("ev-mom",),
            )
        ),
    )


def test_historical_vs_current() -> None:
    rec = _record(
        _atom(),
        _atom(
            atom_id="atom-old",
            atom_type=AtomType.SYMPTOM,
            raw_value="migraines",
            temporality=TemporalState(kind=Temporality.HISTORICAL, time_expression_raw="years ago"),
            evidence_ids=("ev-hist",),
        ),
    )
    rec.validate()
    assert rec.atom("atom-1").temporality.kind is Temporality.CURRENT
    assert rec.atom("atom-old").temporality.kind is Temporality.HISTORICAL
    assert rec.atom("atom-old").temporality.time_expression_raw == "years ago"


def test_future_plans() -> None:
    rec = _record(
        _atom(
            atom_id="atom-plan",
            atom_type=AtomType.PLAN,
            raw_value="Start ibuprofen and follow up in a week.",
            speaker=Speaker.CLINICIAN,
            temporality=TemporalState(kind=Temporality.FUTURE, time_expression_raw="in a week"),
            evidence_ids=("ev-plan",),
        )
    )
    rec.validate()
    assert rec.atom("atom-plan").temporality.kind is Temporality.FUTURE


def test_deterministic_normalization_does_not_overwrite_raw() -> None:
    raw = "a neck"
    normalized = apply_normalization(TRANSFORM, raw)
    assert normalized == "neck"
    rec = _record(
        _atom(
            raw_value=raw,
            normalized_value=normalized,
            normalization_transform=TRANSFORM,
        )
    )
    rec.validate()
    assert rec.atom("atom-1").raw_value == raw
    assert rec.atom("atom-1").normalized_value == "neck"
    expect(
        "normalization_mismatch",
        lambda: _record(
            _atom(
                raw_value=raw,
                normalized_value="cervicalgia",
                normalization_transform=TRANSFORM,
            )
        ),
    )
    expect(
        "normalized_without_transform",
        lambda: _record(_atom(raw_value=raw, normalized_value="neck")),
    )


def test_json_round_trip() -> None:
    rec = _record(
        _atom(
            raw_value="a neck",
            normalized_value="neck",
            normalization_transform=TRANSFORM,
        )
    )
    encoded = rec.to_json()
    restored = EncounterRecord.from_json(encoded)
    assert restored == rec
    assert encoded == json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert rec.schema_version == SCHEMA_VERSION


def test_malformed_json_rejected() -> None:
    expect("invalid_json", lambda: EncounterRecord.from_json("{not json"))
    expect(
        "duplicate_key",
        lambda: EncounterRecord.from_json('{"encounter_id":"a","encounter_id":"b"}'),
    )


def test_unknown_enum_rejected() -> None:
    rec = _record(_atom())
    payload = rec.to_dict()
    payload["atoms"][0]["assertion_state"] = "maybe"
    expect("invalid_enum", lambda: EncounterRecord.from_dict(payload))
    payload = rec.to_dict()
    payload["atoms"][0]["atom_type"] = "chief_complaint"
    expect("invalid_enum", lambda: EncounterRecord.from_dict(payload))


def test_multiple_evidence_spans() -> None:
    source = _source()
    extra = _span(source, 1, "hurting", "ev-hurt")
    rec = _record(_atom(evidence_ids=("ev-neck", "ev-hurt")), extra_spans=(extra,))
    rec.validate()
    assert rec.atom("atom-1").evidence_ids == ("ev-neck", "ev-hurt")


def test_one_span_can_support_two_atoms() -> None:
    rec = _record(
        _atom(atom_id="atom-a", raw_value="neck"),
        _atom(atom_id="atom-b", raw_value="neck", atom_type=AtomType.SYMPTOM),
    )
    rec.validate()
    assert rec.atom("atom-a").evidence_ids == rec.atom("atom-b").evidence_ids


def test_stable_identifiers() -> None:
    rec = _record(_atom())
    rec.validate()
    assert rec.encounter_id == "enc-1"
    assert rec.sources[0].source_id == "src-1"
    assert rec.evidence[0].evidence_id == "ev-neck"
    assert rec.atom("atom-1").atom_id == "atom-1"
    again = EncounterRecord.from_json(rec.to_json())
    assert again.encounter_id == rec.encounter_id
    assert again.sources[0].turns[0].turn_id == rec.sources[0].turns[0].turn_id


def test_silence_is_unresolved_not_a_negative_fact() -> None:
    rec = _record(
        _atom(),
        unresolved=(
            UnresolvedItem(
                unresolved_id="u-med",
                topic="medication",
                reason="silence",
                review_required=True,
            ),
        ),
    )
    rec.validate()
    assert rec.unresolved[0].reason == "silence"
    assert rec.unresolved[0].review_required


def test_clinician_evidence_is_allowed() -> None:
    source = _source()
    clinician_turn = source.turns[5]
    assert clinician_turn.speaker is Speaker.CLINICIAN
    rec = _record(
        _atom(
            atom_id="atom-assess",
            atom_type=AtomType.ASSESSMENT,
            raw_value="cervical strain",
            speaker=Speaker.CLINICIAN,
            evidence_ids=("ev-assess",),
        )
    )
    rec.validate()
    span = rec.span("ev-assess")
    assert span.speaker is Speaker.CLINICIAN
    assert source.text[span.start : span.end] == "cervical strain"


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"encounter v0 pins: {len(fns)}/{len(fns)} PASS")
