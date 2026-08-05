from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.contract import FIELD_ORDER, FieldState
from nano_ai.training import evaluate_evidence_query_h6
from nano_ai.training.evaluate_evidence_query_h6 import (
    EvidenceQueryCandidate,
    EvidenceQueryEvaluationError,
    EvidenceQueryH6PointerSolver,
)
from nano_ai.training.evidence_query_model import NanoEvidenceQueryPointerModel
from nano_ai.training.pointer_model import NANO_TRUNK_PARAMETER_COUNT
from nano_ai.training.state_conditioned_evidence_query_model import (
    ARCHITECTURE_VERSION,
    NanoStateConditionedEvidenceQueryPointerModel,
)
from nano_ai.training.state_conditioned_evidence_query_model import (
    EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6 as EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
)
from nano_ai.training.state_conditioned_evidence_query_model import (
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6 as NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
)
from nano_ai.training.state_span_data import canonical_json_bytes
from nano_ai.training.train_evidence_query_h6 import (
    H6_TRAINING_RECIPE_VERSION,
    H6_TRAINING_REPORT_SCHEMA_VERSION,
    PRESERVED_H3_SOURCE_SHA256,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_h6_verifier_solver_descriptor_uses_h6_identity() -> None:
    solver = EvidenceQueryH6PointerSolver(
        lambda _transcript: (),
        artifact_bytes=1_234,
    )

    assert solver.descriptor.to_dict() == {
        "solver_id": "candidate/nano-evidence-query-h6-pointer-v1",
        "kind": "hybrid",
        "version": ARCHITECTURE_VERSION,
        "parameter_count": NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
        "artifact_bytes": 1_234,
    }


def _bucket(numerator: int, denominator: int) -> dict[str, float | int]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
    }


def _gate_metrics() -> dict[str, dict[str, float | int]]:
    return {
        "overall": _bucket(3_041, 5_000),
        "held_value": _bucket(2_167, 2_987),
        "missing_target": _bucket(219, 250),
        "absence": _bucket(383, 413),
        "conflict_target": _bucket(236, 250),
        "uncertain_target": _bucket(228, 250),
        "failures": _bucket(10, 1_000),
        "false_presented": _bucket(0, 100),
    }


def _phase(
    *,
    absence: int,
    missing: int,
    uncertain: int,
    conflicting: int,
    overall: int,
    wrong_presented: int,
) -> dict[str, object]:
    slices = {
        "overall": _bucket(overall, 4_000),
        "absence": _bucket(absence, 330),
        "missing_target": _bucket(missing, 200),
        "uncertain_target": _bucket(uncertain, 200),
        "conflicting_target": _bucket(conflicting, 200),
    }
    return {
        "slices": slices,
        "selection": {
            "macro_joint": sum(
                slices[name]["rate"]
                for name in (
                    "absence",
                    "missing_target",
                    "uncertain_target",
                    "conflicting_target",
                )
            )
            / 4.0,
            "overall_joint": slices["overall"]["rate"],
        },
        "wrong_presented": _bucket(wrong_presented, 500),
    }


def _calibration(*, uncalibrated_overall: int) -> dict[str, object]:
    return {
        "uncalibrated": _phase(
            absence=165,
            missing=100,
            uncertain=100,
            conflicting=100,
            overall=uncalibrated_overall,
            wrong_presented=4,
        ),
        "global_threshold": 0.42,
        "calibrated": _phase(
            absence=330,
            missing=200,
            uncertain=200,
            conflicting=200,
            overall=4_000,
            wrong_presented=0,
        ),
        "threshold_policy": evaluate_evidence_query_h6.CALIBRATION_THRESHOLD_POLICY,
    }


def _runtime() -> dict[str, object]:
    h3 = evaluate_evidence_query_h6.h3_evaluation
    return {
        "python": h3.platform.python_version(),
        "torch": h3.torch.__version__,
        "tokenizers": getattr(__import__("tokenizers"), "__version__", None),
        "cuda": h3.torch.version.cuda,
        "gpu": None,
        "cublas_workspace_config": None,
        "platform": h3.platform.platform(),
        "seconds": 1.0,
    }


def _candidate(
    *,
    seed: int,
    epoch: int,
    macro: float,
    overall: float,
    checkpoint_digest: str,
    report_digest: str,
) -> EvidenceQueryCandidate:
    return EvidenceQueryCandidate(
        seed=seed,
        epoch=epoch,
        path=Path(f"seed-{seed}.pt"),
        sha256=checkpoint_digest,
        artifact_bytes=100,
        report_sha256=report_digest,
        global_threshold=0.5,
        macro_joint=macro,
        overall_joint=overall,
        report={"device": "cpu", "runtime": _runtime()},
    )


def _proposal_rows(state: FieldState) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(field=field_name, state=state) for field_name in FIELD_ORDER
    )


def test_h6_hyperparameter_validator_requires_exact_residual_mechanism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        evaluate_evidence_query_h6.h3_evaluation,
        "_validate_hyperparameters",
        lambda value: captured.append(value),
    )
    value = {
        "h5_frozen_setting": 1,
        **evaluate_evidence_query_h6._H6_HYPERPARAMETER_EXTENSION,
    }

    evaluate_evidence_query_h6._validate_h6_hyperparameters(value)

    assert captured == [{"h5_frozen_setting": 1}]
    malformed = dict(value)
    malformed["state_posterior_detached"] = False
    with pytest.raises(EvidenceQueryEvaluationError, match="recipe changed"):
        evaluate_evidence_query_h6._validate_h6_hyperparameters(malformed)


def test_h6_evaluator_rejects_a_self_consistent_non_h5_replay_bundle() -> None:
    identities = evaluate_evidence_query_h6.H5_DATASET_FILE_IDENTITY
    expected = {role: row["sha256"] for role, row in identities.items()}
    bundle = SimpleNamespace(
        manifest_sha256=expected["manifest"],
        input_sha256=expected,
    )

    evaluate_evidence_query_h6._require_frozen_h5_training_bundle(
        expected["manifest"],
        bundle,
    )

    bundle.input_sha256 = {**expected, "fit": "0" * 64}
    with pytest.raises(EvidenceQueryEvaluationError, match="byte-exact"):
        evaluate_evidence_query_h6._require_frozen_h5_training_bundle(
            expected["manifest"],
            bundle,
        )


def test_h6_checkpoint_loader_is_strict_for_query_residual_state(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    model = NanoStateConditionedEvidenceQueryPointerModel()
    checkpoint = tmp_path / "h6.pt"
    torch.save(model.state_dict(), checkpoint)
    snapshot = checkpoint.read_bytes()
    candidate = _candidate(
        seed=20260805,
        epoch=1,
        macro=0.5,
        overall=0.5,
        checkpoint_digest=_sha256(snapshot),
        report_digest="a" * 64,
    )
    candidate = EvidenceQueryCandidate(
        seed=candidate.seed,
        epoch=candidate.epoch,
        path=checkpoint,
        sha256=candidate.sha256,
        artifact_bytes=len(snapshot),
        report_sha256=candidate.report_sha256,
        global_threshold=candidate.global_threshold,
        macro_joint=candidate.macro_joint,
        overall_joint=candidate.overall_joint,
        report=candidate.report,
    )

    loaded = evaluate_evidence_query_h6._load_evidence_query_model(
        candidate,
        device="cpu",
    )

    assert isinstance(loaded, NanoStateConditionedEvidenceQueryPointerModel)
    assert set(loaded.state_dict()) == set(model.state_dict())

    old_checkpoint = tmp_path / "h5.pt"
    torch.save(NanoEvidenceQueryPointerModel().state_dict(), old_checkpoint)
    old_snapshot = old_checkpoint.read_bytes()
    old_candidate = EvidenceQueryCandidate(
        seed=candidate.seed,
        epoch=candidate.epoch,
        path=old_checkpoint,
        sha256=_sha256(old_snapshot),
        artifact_bytes=len(old_snapshot),
        report_sha256=candidate.report_sha256,
        global_threshold=candidate.global_threshold,
        macro_joint=candidate.macro_joint,
        overall_joint=candidate.overall_joint,
        report=candidate.report,
    )
    with pytest.raises(EvidenceQueryEvaluationError, match="does not match"):
        evaluate_evidence_query_h6._load_evidence_query_model(
            old_candidate,
            device="cpu",
        )


def test_h6_quality_gates_bind_stricter_held_value_floor() -> None:
    passing = _gate_metrics()

    assert (
        evaluate_evidence_query_h6._quality_gates(
            passing, require_zero_wrong_presented=True
        )["all_quality_gates_passed"]
        is True
    )
    raw_wrong = copy.deepcopy(passing)
    raw_wrong["false_presented"] = _bucket(7, 100)
    assert (
        evaluate_evidence_query_h6._quality_gates(
            raw_wrong, require_zero_wrong_presented=False
        )["all_quality_gates_passed"]
        is True
    )
    below_held_floor = copy.deepcopy(passing)
    below_held_floor["held_value"] = _bucket(2_166, 2_987)
    evidence = evaluate_evidence_query_h6._quality_gates(
        below_held_floor, require_zero_wrong_presented=False
    )
    assert evidence["gate_evidence"]["held_value"]["minimum_numerator"] == 2_167
    assert evidence["all_quality_gates_passed"] is False


def test_uncalibrated_modal_state_boundary_accepts_949_and_rejects_950() -> None:
    passing_predictions = tuple(
        SimpleNamespace(
            error=None,
            proposals=_proposal_rows(
                FieldState.SUPPORTED if index < 949 else FieldState.ABSENT
            ),
        )
        for index in range(1_000)
    )
    failing_predictions = tuple(
        SimpleNamespace(
            error=None,
            proposals=_proposal_rows(
                FieldState.SUPPORTED if index < 950 else FieldState.ABSENT
            ),
        )
        for index in range(1_000)
    )

    passing = evaluate_evidence_query_h6._uncalibrated_state_balance(
        passing_predictions
    )
    failing = evaluate_evidence_query_h6._uncalibrated_state_balance(
        failing_predictions
    )

    assert passing["passed"] is True
    assert all(row["modal_count"] == 949 for row in passing["fields"].values())
    assert failing["passed"] is False
    assert all(row["modal_count"] == 950 for row in failing["fields"].values())


def test_uncalibrated_admission_combines_semantics_and_state_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collapsed = {
        "development_rows": 1_000,
        "maximum_modal_count": 949,
        "fields": {},
        "passed": False,
    }
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_uncalibrated_state_balance",
        lambda _predictions: collapsed,
    )

    gate, balance = evaluate_evidence_query_h6._uncalibrated_admission(
        _gate_metrics(), ()
    )

    assert gate["semantic_and_retention_passed"] is True
    assert gate["uncalibrated_state_balance_passed"] is False
    assert gate["all_quality_gates_passed"] is False
    assert balance is collapsed


def test_primary_selection_requires_two_fixed_seeds_and_uses_frozen_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = {
        "first.json": _candidate(
            seed=20260805,
            epoch=3,
            macro=0.8,
            overall=0.7,
            checkpoint_digest="c" * 64,
            report_digest="a" * 64,
        ),
        "second.json": _candidate(
            seed=20260806,
            epoch=2,
            macro=0.8,
            overall=0.7,
            checkpoint_digest="d" * 64,
            report_digest="b" * 64,
        ),
    }
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "load_candidate_from_training_report",
        lambda path, **_kwargs: candidates[Path(path).name],
    )
    reports = (("first.json", "a" * 64), ("second.json", "b" * 64))
    kwargs = {
        "expected_manifest_sha256": "e" * 64,
        "training_data_dir": Path("training"),
        "training_bundle": object(),
    }

    primary, authenticated = evaluate_evidence_query_h6.authenticate_and_select_primary(
        reports, **kwargs
    )

    assert [candidate.seed for candidate in authenticated] == [20260805, 20260806]
    assert primary.seed == 20260806
    candidates["first.json"] = _candidate(
        seed=20260805,
        epoch=2,
        macro=0.8,
        overall=0.7,
        checkpoint_digest="c" * 64,
        report_digest="a" * 64,
    )
    primary, _ = evaluate_evidence_query_h6.authenticate_and_select_primary(
        reports, **kwargs
    )
    assert primary.seed == 20260805
    with pytest.raises(EvidenceQueryEvaluationError, match="exactly two"):
        evaluate_evidence_query_h6.authenticate_and_select_primary(
            reports[:1], **kwargs
        )


def test_report_loader_authenticates_h6_checkpoint_and_source_classes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = b"selected H6 replay state"
    checkpoint_path = tmp_path / "epoch-2.pt"
    checkpoint_path.write_bytes(checkpoint)
    checkpoints = [
        {
            "filename": f"epoch-{epoch}.pt",
            "sha256": str(epoch) * 64,
            "bytes": 100 + epoch,
        }
        for epoch in range(1, 4)
    ]
    checkpoints[1] = {
        "filename": checkpoint_path.name,
        "sha256": _sha256(checkpoint),
        "bytes": len(checkpoint),
    }
    calibrations = [
        _calibration(uncalibrated_overall=overall) for overall in (1_000, 3_000, 2_000)
    ]
    epochs = [
        {
            "epoch": epoch,
            "train_loss": 1.0,
            "state_loss": 0.5,
            "pointer_loss": 0.5,
            "seconds": 1.0,
            "checkpoint": checkpoints[epoch - 1],
            "calibration": calibrations[epoch - 1],
        }
        for epoch in range(1, 4)
    ]
    changed = {"model": "d" * 64, "training": "e" * 64}
    report = {
        "schema_version": H6_TRAINING_REPORT_SCHEMA_VERSION,
        "recipe": H6_TRAINING_RECIPE_VERSION,
        "status": "complete",
        "seed": 20260805,
        "device": "cpu",
        "architecture_version": ARCHITECTURE_VERSION,
        "architecture_identity": FROZEN_NANO_V01.architecture_identity,
        "parameter_count": NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
        "trunk_parameter_count": NANO_TRUNK_PARAMETER_COUNT,
        "evidence_query_head_parameter_count": EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
        "base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
        "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
        "dataset_manifest_sha256": "f" * 64,
        "dataset": {},
        "hyperparameters": dict(
            evaluate_evidence_query_h6._H6_HYPERPARAMETER_EXTENSION
        ),
        "epochs": epochs,
        "candidate": {"epoch": 2, **checkpoints[1]},
        "calibration": {"selected_epoch": 2, **calibrations[1]},
        "dev_used_for_selection": False,
        "legacy_record_artifact_accessed": False,
        "fresh_v1_accessed": False,
        "preserved_source_sha256": dict(PRESERVED_H3_SOURCE_SHA256),
        "changed_source_sha256": changed,
        "runtime": {},
        "selection_note": evaluate_evidence_query_h6._TRAINING_SELECTION_NOTE,
    }
    report_path = tmp_path / "training-report.json"
    report_snapshot = canonical_json_bytes(report)
    report_path.write_bytes(report_snapshot)
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_validate_report_dataset_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_preserved_source_hashes",
        lambda: dict(PRESERVED_H3_SOURCE_SHA256),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6, "_changed_source_hashes", lambda: changed
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6.h3_evaluation,
        "_validate_hyperparameters",
        lambda _value: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6.h3_evaluation,
        "_validate_runtime",
        lambda _value, *, device: None,
    )

    candidate = evaluate_evidence_query_h6.load_candidate_from_training_report(
        report_path,
        expected_report_sha256=_sha256(report_snapshot),
        expected_manifest_sha256="f" * 64,
        training_data_dir=tmp_path,
        training_bundle=object(),
    )

    assert candidate.epoch == 2
    assert candidate.sha256 == _sha256(checkpoint)
    checkpoint_path.write_bytes(b"tampered")
    with pytest.raises(EvidenceQueryEvaluationError, match="SHA-256 mismatch"):
        evaluate_evidence_query_h6.load_candidate_from_training_report(
            report_path,
            expected_report_sha256=_sha256(report_snapshot),
            expected_manifest_sha256="f" * 64,
            training_data_dir=tmp_path,
            training_bundle=object(),
        )


def test_report_dataset_identity_binds_replay_files_and_value_limitation(
    tmp_path: Path,
) -> None:
    snapshots = {
        "manifest": b"manifest",
        "fit": b"fit rows",
        "calibration": b"calibration rows",
    }
    for name, payload in snapshots.items():
        filename = "manifest.json" if name == "manifest" else f"{name}.jsonl"
        (tmp_path / filename).write_bytes(payload)
    overlap = {
        "calibration_open_value_literal_substring_occurrence_is_eligibility_rule": (
            False
        ),
        "all_hard_intersections_zero": True,
        "calibration_records_modified": False,
        "calibration_open_value_literal_substring_occurrence": {
            "policy": "expected_recorded_nonblocking",
            "candidate_worlds": 2_800,
            "literal_substring_disjoint_worlds": 1_053,
            "worlds_with_literal_substring_occurrence": 1_747,
            "exact_value_identity_not_claimed": True,
        },
    }
    restrictions = {
        "legacy_record_artifact_read": False,
        "development_read": False,
        "benchmark_read": False,
        "sealed_confirmation_read": False,
    }
    manifest = {
        "generator_sha256": "a" * 64,
        "normalization": {"identity": "normalization"},
        "training_identity": {"identity": "training"},
        "partitions": {
            "fit": {"identity": "fit"},
            "calibration": {"identity": "calibration"},
        },
        "sources": {"legacy": {}, "surface": {}},
        "overlap_audit": overlap,
        "restrictions": restrictions,
    }
    bundle = SimpleNamespace(
        manifest=manifest,
        input_sha256={name: _sha256(payload) for name, payload in snapshots.items()},
    )
    value = {
        "schema_version": evaluate_evidence_query_h6.DATASET_SCHEMA_VERSION,
        "generator": evaluate_evidence_query_h6.replay_mixture_data.GENERATOR_VERSION,
        "generator_sha256": manifest["generator_sha256"],
        "selection_policy": (
            evaluate_evidence_query_h6.replay_mixture_data.SELECTION_POLICY_VERSION
        ),
        "target_grammar": evaluate_evidence_query_h6.TARGET_GRAMMAR_VERSION,
        "normalization": manifest["normalization"],
        "training_identity": manifest["training_identity"],
        "source_manifest": {
            "filename": "manifest.json",
            "bytes": len(snapshots["manifest"]),
            "sha256": _sha256(snapshots["manifest"]),
        },
        "source_fit": {
            "filename": "fit.jsonl",
            "bytes": len(snapshots["fit"]),
            "sha256": _sha256(snapshots["fit"]),
            "records": 11_200,
            "worlds": 2_800,
            "legacy_worlds": 1_400,
            "surface_worlds": 1_400,
            "gradient_bearing": True,
        },
        "source_calibration": {
            "filename": "calibration.jsonl",
            "bytes": len(snapshots["calibration"]),
            "sha256": _sha256(snapshots["calibration"]),
            "records": 800,
            "worlds": 200,
            "namespace": "train-calibration",
            "gradient_bearing": False,
            "reused_unchanged_from_h4": True,
        },
        "fit": {
            "identity": "fit",
            "state_class_counts": evaluate_evidence_query_h6._state_class_counts(
                evaluate_evidence_query_h6.FIT_STATE_CLASS_COUNTS
            ),
        },
        "calibration": {
            "identity": "calibration",
            "state_class_counts": evaluate_evidence_query_h6._state_class_counts(
                evaluate_evidence_query_h6.CALIBRATION_STATE_CLASS_COUNTS
            ),
        },
        "sources": manifest["sources"],
        "overlap_audit": overlap,
        "restrictions": restrictions,
    }

    assert set(value) == evaluate_evidence_query_h6.REPORT_DATASET_KEYS

    evaluate_evidence_query_h6._validate_report_dataset_metadata(
        value,
        training_data_dir=tmp_path,
        training_bundle=bundle,
        expected_manifest_sha256=_sha256(snapshots["manifest"]),
    )

    malformed = copy.deepcopy(value)
    malformed["overlap_audit"]["calibration_open_value_literal_substring_occurrence"][
        "literal_substring_disjoint_worlds"
    ] = 1_054
    with pytest.raises(EvidenceQueryEvaluationError, match="identity changed"):
        evaluate_evidence_query_h6._validate_report_dataset_metadata(
            malformed,
            training_data_dir=tmp_path,
            training_bundle=bundle,
            expected_manifest_sha256=_sha256(snapshots["manifest"]),
        )


def test_authentication_precedes_development_and_collapse_stops_later_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    forbidden_calls: list[str] = []
    candidate = _candidate(
        seed=20260805,
        epoch=1,
        macro=0.8,
        overall=0.7,
        checkpoint_digest="c" * 64,
        report_digest="a" * 64,
    )
    training_bundle = SimpleNamespace(
        manifest_sha256="e" * 64,
        manifest={},
        input_sha256={
            "manifest": "e" * 64,
            "fit": "f" * 64,
            "calibration": "a" * 64,
        },
        fit=(object(),),
        calibration=(object(),),
    )
    example = SimpleNamespace(world_id="dev-world")
    development_bundle = SimpleNamespace(
        manifest_sha256="d" * 64,
        dev_sha256="b" * 64,
        examples=(example,),
    )
    inference_input = SimpleNamespace(example_id="dev-1")
    inference = SimpleNamespace(example_ids=("dev-1",), predictions=(object(),))
    collapsed = {
        "development_rows": 1_000,
        "maximum_modal_count": 949,
        "fields": {},
        "passed": False,
    }

    def load_training(*_args, **_kwargs):
        events.append("load-training")
        return training_bundle

    def authenticate(*_args, **_kwargs):
        events.append("authenticate-and-select")
        return candidate, (candidate,)

    def load_development(*_args, **kwargs):
        events.append("load-development")
        assert kwargs["expected_manifest_sha256"] == "d" * 64
        return development_bundle

    def forbidden(stage: str):
        def call(*_args, **_kwargs):
            forbidden_calls.append(stage)
            raise AssertionError(f"{stage} must not run")

        return call

    monkeypatch.setattr(evaluate_evidence_query_h6, "_evaluation_source_hashes", dict)
    monkeypatch.setattr(
        evaluate_evidence_query_h6.replay_mixture_data,
        "load_replay_mixture_dataset",
        load_training,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6, "authenticate_and_select_primary", authenticate
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_canonical_evaluation_runtime",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_require_training_data_unchanged",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_require_frozen_h5_training_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6, "load_development_bundle", load_development
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "load_pointer_tokenizer",
        lambda *_args, **_kwargs: events.append("load-tokenizer") or object(),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "encode_pointer_partition",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "build_pointer_inference_inputs",
        lambda *_args, **_kwargs: (inference_input,),
    )
    monkeypatch.setattr(evaluate_evidence_query_h6, "_fixture_cases", lambda *_args: ())
    monkeypatch.setattr(evaluate_evidence_query_h6, "_seed_evaluation", lambda: None)
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_load_evidence_query_model",
        lambda *_args, **_kwargs: events.append("load-model") or object(),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "batched_evidence_query_inference",
        lambda *_args, **_kwargs: inference,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "raw_pointer_diagnostics",
        lambda *_args, **_kwargs: {"acceptance": {"metrics": _gate_metrics()}},
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_uncalibrated_state_balance",
        lambda _predictions: collapsed,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "apply_global_threshold",
        forbidden("threshold"),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_evaluate_predictions",
        forbidden("verifier"),
    )

    result = evaluate_evidence_query_h6.evaluate_development(
        training_data_dir=tmp_path / "training",
        training_manifest_sha256="e" * 64,
        development_data_dir=tmp_path / "development",
        development_manifest_sha256="d" * 64,
        tokenizer_path=tmp_path / "tokenizer.json",
        training_reports=(("a", "a" * 64), ("b", "b" * 64)),
        output_path=tmp_path / "result.json",
    )

    assert events == [
        "load-training",
        "authenticate-and-select",
        "load-model",
        "load-tokenizer",
        "load-development",
    ]
    assert forbidden_calls == []
    assert result["schema_version"] == (
        "nano.evidence-query-h6-development-evaluation.v1"
    )
    assert (
        result["training_data"]["recipe"]
        == evaluate_evidence_query_h6.replay_mixture_data.TRAINING_RECIPE_VERSION
    )
    assert result["calibrated_raw"] is None
    assert result["verifier_final"] is None
    assert result["protocol"]["threshold_applied"] is False
    assert result["protocol"]["calibration_scope"] == {
        "partition_reused_unchanged_from_h4": True,
        "familiar_legacy_values_present": True,
        "unseen_h4_surfaces_and_templates": True,
        "open_value_disjoint": False,
        "selection_role": "surface_and_template_transfer_ranking_only",
        "quality_decision_partition": "known_development",
    }
    assert result["decision"]["uncalibrated_semantic_and_retention_passed"] is True
    assert result["decision"]["uncalibrated_state_balance_passed"] is False
    assert result["decision"]["quality_gate_passed"] is False
    assert (tmp_path / "result.json").is_file()


def test_digest_valid_malformed_checkpoint_is_rejected_before_development(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "malformed-h6.pt"
    torch.save({"not_an_h6_parameter": torch.zeros(1)}, checkpoint_path)
    snapshot = checkpoint_path.read_bytes()
    candidate = EvidenceQueryCandidate(
        seed=20260805,
        epoch=1,
        path=checkpoint_path,
        sha256=_sha256(snapshot),
        artifact_bytes=len(snapshot),
        report_sha256="a" * 64,
        global_threshold=0.5,
        macro_joint=0.8,
        overall_joint=0.7,
        report={"device": "cpu", "runtime": _runtime()},
    )
    training_bundle = SimpleNamespace(
        manifest_sha256="e" * 64,
        manifest={},
        input_sha256={
            "manifest": "e" * 64,
            "fit": "f" * 64,
            "calibration": "a" * 64,
        },
    )
    development_opened = False

    def load_development(*_args, **_kwargs):
        nonlocal development_opened
        development_opened = True
        raise AssertionError("known development must remain unopened")

    monkeypatch.setattr(evaluate_evidence_query_h6, "_evaluation_source_hashes", dict)
    monkeypatch.setattr(
        evaluate_evidence_query_h6.replay_mixture_data,
        "load_replay_mixture_dataset",
        lambda *_args, **_kwargs: training_bundle,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_require_frozen_h5_training_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "authenticate_and_select_primary",
        lambda *_args, **_kwargs: (candidate, (candidate,)),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_canonical_evaluation_runtime",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_require_training_data_unchanged",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6, "load_development_bundle", load_development
    )

    with pytest.raises(
        EvidenceQueryEvaluationError,
        match=f"does not match {ARCHITECTURE_VERSION}",
    ):
        evaluate_evidence_query_h6.evaluate_development(
            training_data_dir=tmp_path / "training",
            training_manifest_sha256="e" * 64,
            development_data_dir=tmp_path / "development",
            development_manifest_sha256="d" * 64,
            tokenizer_path=tmp_path / "tokenizer.json",
            training_reports=(("a", "a" * 64), ("b", "b" * 64)),
            output_path=tmp_path / "result.json",
        )

    assert development_opened is False
    assert not (tmp_path / "result.json").exists()


def test_calibrated_quality_failure_stops_before_verifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stages: list[str] = []
    candidate = _candidate(
        seed=20260805,
        epoch=1,
        macro=0.8,
        overall=0.7,
        checkpoint_digest="c" * 64,
        report_digest="a" * 64,
    )
    training_bundle = SimpleNamespace(
        manifest_sha256="e" * 64,
        manifest={},
        input_sha256={
            "manifest": "e" * 64,
            "fit": "f" * 64,
            "calibration": "a" * 64,
        },
        fit=(object(),),
        calibration=(object(),),
    )
    development_bundle = SimpleNamespace(
        manifest_sha256="d" * 64,
        dev_sha256="b" * 64,
        examples=(SimpleNamespace(world_id="dev-world"),),
    )
    inference_input = SimpleNamespace(example_id="dev-1")
    inference = SimpleNamespace(example_ids=("dev-1",), predictions=(object(),))
    passing_balance = {
        "development_rows": 1_000,
        "maximum_modal_count": 949,
        "fields": {},
        "passed": True,
    }
    calibrated_failure = copy.deepcopy(_gate_metrics())
    calibrated_failure["false_presented"] = _bucket(1, 100)
    metrics = iter((_gate_metrics(), calibrated_failure))

    monkeypatch.setattr(evaluate_evidence_query_h6, "_evaluation_source_hashes", dict)
    monkeypatch.setattr(
        evaluate_evidence_query_h6.replay_mixture_data,
        "load_replay_mixture_dataset",
        lambda *_args, **_kwargs: training_bundle,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "authenticate_and_select_primary",
        lambda *_args, **_kwargs: (candidate, (candidate,)),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_canonical_evaluation_runtime",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_require_training_data_unchanged",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_require_frozen_h5_training_bundle",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "load_development_bundle",
        lambda *_args, **_kwargs: development_bundle,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "load_pointer_tokenizer",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "encode_pointer_partition",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "build_pointer_inference_inputs",
        lambda *_args, **_kwargs: (inference_input,),
    )
    monkeypatch.setattr(evaluate_evidence_query_h6, "_fixture_cases", lambda *_args: ())
    monkeypatch.setattr(evaluate_evidence_query_h6, "_seed_evaluation", lambda: None)
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_load_evidence_query_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "batched_evidence_query_inference",
        lambda *_args, **_kwargs: inference,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "raw_pointer_diagnostics",
        lambda *_args, **_kwargs: {"acceptance": {"metrics": next(metrics)}},
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6,
        "_uncalibrated_state_balance",
        lambda _predictions: passing_balance,
    )

    def apply_threshold(*_args, **_kwargs):
        stages.append("threshold")
        return (object(),)

    def forbidden_verifier(*_args, **_kwargs):
        stages.append("verifier")
        raise AssertionError("verifier must not run after calibrated failure")

    monkeypatch.setattr(
        evaluate_evidence_query_h6, "apply_global_threshold", apply_threshold
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h6, "_evaluate_predictions", forbidden_verifier
    )

    result = evaluate_evidence_query_h6.evaluate_development(
        training_data_dir=tmp_path / "training",
        training_manifest_sha256="e" * 64,
        development_data_dir=tmp_path / "development",
        development_manifest_sha256="d" * 64,
        tokenizer_path=tmp_path / "tokenizer.json",
        training_reports=(("a", "a" * 64), ("b", "b" * 64)),
        output_path=tmp_path / "result.json",
    )

    assert stages == ["threshold"]
    assert result["calibrated_raw"] is not None
    assert result["calibrated_quality"]["all_quality_gates_passed"] is False
    assert result["verifier_final"] is None
    assert result["verifier_final_quality"] is None
    assert result["decision"]["verifier_final_quality_passed"] is None
    assert result["decision"]["quality_gate_passed"] is False


def test_evaluator_imports_no_benchmark_or_sealed_partition_module() -> None:
    source = Path(evaluate_evidence_query_h6.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert all("benchmark" not in module for module in modules)
    assert all("fresh" not in module for module in modules)
    assert "train.jsonl" not in source
    assert "fresh_v1_partition" not in source
