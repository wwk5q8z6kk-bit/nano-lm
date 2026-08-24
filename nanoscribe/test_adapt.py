# Adversarial pins for model adapter + baseline bridge.
# Run: python3 nanoscribe/test_adapt.py
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.adapt import (
    CANDIDATE_SCHEMA_VERSION,
    AdaptError,
    CandidateAtom,
    CandidateEvidenceRequest,
    ModelCandidate,
    ModelCandidateBatch,
    ModelInput,
    adapt,
    adapt_json,
    adapt_span_port_line,
    candidate_from_span_port_line,
    evidence_requests,
    format_label_answer,
    parse_label_and_quotes,
    run_pipeline,
)
from nanoscribe.adapters import (
    FixtureSpanPortAdapter,
    ModelAdapter,
    Qwen25BaselineAdapter,
    default_baseline_specs,
    default_qwen_fixture_adapter,
)
from nanoscribe.encounter import (
    AssertionState,
    AtomType,
    Certainty,
    ClinicalAtom,
    EncounterRecord,
    Experiencer,
    Speaker,
    TemporalState,
    Temporality,
    UnresolvedItem,
    assemble_source,
)
from nanoscribe.evaluate import atom_result, evaluate
from nanoscribe.select import relocate


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


def _model_input(source=None) -> ModelInput:
    source = source or _source()
    return ModelInput(source=source, encounter_id="enc-1")


def expect(code: str, fn) -> None:
    try:
        fn()
    except AdaptError as exc:
        assert exc.code == code, f"expected {code}, got {exc.code}: {exc}"
        return
    raise AssertionError(f"expected AdaptError {code}")


def test_parse_label_and_quotes() -> None:
    assert parse_label_and_quotes('STATED: "neck"') == ("STATED", ("neck",))
    assert parse_label_and_quotes("NOT_MENTIONED") == ("NOT_MENTIONED", ())
    assert parse_label_and_quotes("maybe later") == (None, ())
    assert format_label_answer("DENIED", ("No allergies.",)) == 'DENIED: "No allergies."'


def test_candidate_json_rejects_trusted_evidence_keys() -> None:
    payload = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "atoms": [
            {
                "atom_id": "atom-neck",
                "atom_type": "symptom",
                "raw_value": "neck",
                "assertion_state": "asserted",
                "speaker": "patient",
                "experiencer": "patient",
                "temporality": {
                    "kind": "current",
                    "onset_raw": None,
                    "duration_raw": None,
                    "time_expression_raw": None,
                },
                "certainty": "stated",
                "quotes": ["neck"],
                "review_required": False,
                "abstained": False,
                "malformed": False,
                "start": 0,
            }
        ],
    }
    expect("forbidden_key", lambda: ModelCandidate.from_dict(payload))


def test_candidate_json_round_trip() -> None:
    candidate = ModelCandidate(
        atoms=(
            CandidateAtom(
                atom_id="atom-neck",
                atom_type=AtomType.SYMPTOM,
                raw_value="neck",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                quotes=("neck",),
            ),
        )
    )
    restored = ModelCandidate.from_json(json.dumps(candidate.to_dict()))
    assert restored == candidate


def test_span_port_line_to_candidate() -> None:
    atom = candidate_from_span_port_line(
        atom_id="atom-neck",
        atom_type=AtomType.SYMPTOM,
        raw_value="neck",
        raw_line='STATED: "neck"',
    )
    assert atom.assertion_state is AssertionState.ASSERTED
    assert atom.quotes == ("neck",)
    abstain = candidate_from_span_port_line(
        atom_id="atom-med",
        atom_type=AtomType.MEDICATION,
        raw_value="medication",
        raw_line="NOT_MENTIONED",
    )
    assert abstain.abstained


def test_adapt_binds_quotes_through_selector() -> None:
    model_input = _model_input()
    pred = adapt(
        model_input,
        ModelCandidate(
            atoms=(
                CandidateAtom(
                    atom_id="atom-neck",
                    atom_type=AtomType.SYMPTOM,
                    raw_value="neck",
                    assertion_state=AssertionState.ASSERTED,
                    speaker=Speaker.PATIENT,
                    experiencer=Experiencer.PATIENT,
                    temporality=TemporalState(kind=Temporality.CURRENT),
                    certainty=Certainty.STATED,
                    quotes=("neck",),
                ),
            )
        ),
    )
    assert len(pred.atoms) == 1
    atom = pred.atoms[0]
    assert atom.spans
    assert model_input.source.text[atom.spans[0].start : atom.spans[0].end] == "neck"


def test_paraphrase_abstains_instead_of_inventing_evidence() -> None:
    model_input = _model_input()
    pred = adapt_span_port_line(
        model_input,
        atom_id="atom-neck",
        atom_type=AtomType.SYMPTOM,
        raw_value="neck",
        raw_line='STATED: "cervicalgia"',
    )
    assert pred.abstained
    assert not pred.spans


def test_span_port_baseline_through_evaluator() -> None:
    gold = _gold()
    model_input = _model_input(gold.sources[0])
    preds = [
        adapt_span_port_line(
            model_input,
            atom_id="atom-neck",
            atom_type=AtomType.SYMPTOM,
            raw_value="neck",
            raw_line='STATED: "neck"',
        ),
        adapt_span_port_line(
            model_input,
            atom_id="atom-alg",
            atom_type=AtomType.ALLERGY,
            raw_value="allergies",
            raw_line='DENIED: "No allergies."',
        ),
        adapt_span_port_line(
            model_input,
            atom_id="atom-hist",
            atom_type=AtomType.SYMPTOM,
            raw_value="migraines",
            raw_line='STATED: "migraines"',
        ),
        adapt_span_port_line(
            model_input,
            atom_id="atom-assess",
            atom_type=AtomType.ASSESSMENT,
            raw_value="cervical strain",
            raw_line='STATED: "cervical strain"',
            speaker=Speaker.CLINICIAN,
            experiencer=Experiencer.PATIENT,
        ),
        adapt_span_port_line(
            model_input,
            atom_id="medication",
            atom_type=AtomType.MEDICATION,
            raw_value="medication",
            raw_line="NOT_MENTIONED",
        ),
    ]
    from nanoscribe.evaluate import PredictedEncounter

    report = evaluate(gold, PredictedEncounter(atoms=tuple(preds)))
    assert atom_result(report, "atom-neck").exact_gold_span
    assert atom_result(report, "atom-neck").assertion_state_correct
    assert report.support_direct_exact >= 2
    assert report.correct_abstention == 1
    assert report.spurious_atom == 0


def test_adapt_json_end_to_end() -> None:
    model_input = _model_input()
    raw = json.dumps(
        {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "atoms": [
                {
                    "atom_id": "atom-neck",
                    "atom_type": "symptom",
                    "raw_value": "neck",
                    "assertion_state": "asserted",
                    "speaker": "patient",
                    "experiencer": "patient",
                    "temporality": {
                        "kind": "current",
                        "onset_raw": None,
                        "duration_raw": None,
                        "time_expression_raw": None,
                    },
                    "certainty": "stated",
                    "quotes": ["neck"],
                    "review_required": False,
                    "abstained": False,
                    "malformed": False,
                }
            ],
        }
    )
    pred = adapt_json(raw, model_input)
    report = evaluate(_gold(), pred)
    assert atom_result(report, "atom-neck").exact_gold_span


def test_abstained_atom_without_quotes_parses() -> None:
    batch = ModelCandidate.from_dict(
        {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "atoms": [
                {
                    "atom_id": "medication",
                    "atom_type": "medication",
                    "raw_value": "medication",
                    "assertion_state": "uncertain",
                    "speaker": "patient",
                    "experiencer": "patient",
                    "temporality": {"kind": "current"},
                    "certainty": "uncertain",
                    "review_required": False,
                    "abstained": True,
                    "malformed": False,
                }
            ],
        }
    )
    assert len(batch.atoms) == 1
    assert batch.atoms[0].abstained
    assert batch.atoms[0].quotes == ()


def test_model_cannot_emit_encounter_schema() -> None:
    expect(
        "schema_version",
        lambda: ModelCandidate.from_dict(
            {"schema_version": "nano.encounter.v0", "atoms": []}
        ),
    )


def test_evidence_requests_from_quotes() -> None:
    atom = CandidateAtom(atom_id="a", quotes=("neck", "hurting"))
    reqs = evidence_requests(atom)
    assert reqs == (
        CandidateEvidenceRequest(quote="neck"),
        CandidateEvidenceRequest(quote="hurting"),
    )


def test_fixture_adapter_implements_protocol() -> None:
    adapter = default_qwen_fixture_adapter()
    assert isinstance(adapter, ModelAdapter)
    batch = adapter.propose(_model_input(), default_baseline_specs())
    assert isinstance(batch, ModelCandidateBatch)
    assert len(batch.atoms) == 5


def test_qwen_skeleton_falls_back_to_fixture() -> None:
    adapter = Qwen25BaselineAdapter(fixture_lines={"atom-neck": 'STATED: "neck"'})
    batch = adapter.propose(
        _model_input(),
        (default_baseline_specs()[0],),
    )
    assert batch.atoms[0].quotes == ("neck",)


def test_run_pipeline_smoke_with_encounter_probe() -> None:
    gold = _gold()
    model_input = _model_input(gold.sources[0])
    batch = FixtureSpanPortAdapter(
        lines={
            "atom-neck": 'STATED: "neck"',
            "atom-alg": 'DENIED: "No allergies."',
            "atom-hist": 'STATED: "migraines"',
            "atom-assess": 'STATED: "cervical strain"',
            "medication": "NOT_MENTIONED",
        }
    ).propose(model_input, default_baseline_specs())
    predicted, report = run_pipeline(model_input, batch, gold=gold)
    assert report is not None
    assert len(predicted.atoms) == 5
    assert atom_result(report, "atom-neck").exact_gold_span
    assert report.correct_abstention == 1


def test_failure_layers_on_clean_baseline() -> None:
    from nanoscribe.decompose import classify_report

    gold = _gold()
    model_input = _model_input(gold.sources[0])
    batch = default_qwen_fixture_adapter().propose(model_input, default_baseline_specs())
    _, report = run_pipeline(model_input, batch, gold=gold)
    assert report is not None
    layers = classify_report(report)
    assert layers["layers"]["malformed"] == 0
    assert layers["layers"]["commission"] == 0
    assert layers["layers"]["transport"] == 0
    assert layers["support_mix"]["direct_exact"] == 3


def test_run_eval_fixture_only_smoke() -> None:
    from nanoscribe.run_eval import run_baseline_eval

    saved = os.environ.pop("NANOSCIBE_QWEN_WEIGHTS", None)
    try:
        result = run_baseline_eval(fixture_only=True)
    finally:
        if saved is not None:
            os.environ["NANOSCIBE_QWEN_WEIGHTS"] = saved

    assert result["fixture_only"] is True
    assert set(result) >= {"aggregate", "layers", "per_atom"}
    assert result["layers"]["layers"]["malformed"] == 0
    assert result["layers"]["layers"]["commission"] == 0
    assert result["aggregate"]["correct_abstention"] == 1
    assert result["per_atom"]["atom-neck"]["exact_gold_span"]


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"adapt pins: {len(fns)}/{len(fns)} PASS")
