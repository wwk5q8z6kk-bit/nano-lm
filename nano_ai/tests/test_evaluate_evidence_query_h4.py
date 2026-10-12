from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.training import evaluate_evidence_query_h4
from nano_ai.training.evaluate_evidence_query_h4 import (
    EvidenceQueryCandidate,
    EvidenceQueryEvaluationError,
)
from nano_ai.training.evidence_query_model import (
    ARCHITECTURE_VERSION,
    EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
)
from nano_ai.training.pointer_model import NANO_TRUNK_PARAMETER_COUNT
from nano_ai.training.state_span_data import canonical_json_bytes
from nano_ai.training.train_evidence_query_h4 import (
    H4_TRAINING_RECIPE_VERSION,
    H4_TRAINING_REPORT_SCHEMA_VERSION,
    PRESERVED_H3_SOURCE_SHA256,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bucket(numerator: int, denominator: int) -> dict[str, float | int]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
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
        "threshold_policy": evaluate_evidence_query_h4.CALIBRATION_THRESHOLD_POLICY,
    }


def _runtime() -> dict[str, object]:
    h3 = evaluate_evidence_query_h4.h3_evaluation
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


def _gate_metrics() -> dict[str, dict[str, float | int | None]]:
    return {
        "overall": _bucket(3_041, 5_000),
        "held_value": _bucket(1_905, 2_987),
        "missing_target": _bucket(219, 250),
        "absence": _bucket(383, 413),
        "conflict_target": _bucket(236, 250),
        "uncertain_target": _bucket(228, 250),
        "failures": _bucket(10, 1_000),
        "false_presented": _bucket(0, 100),
    }


def test_h4_reuses_exact_h3_calibration_and_quality_floors() -> None:
    row = {"calibration": _calibration(uncalibrated_overall=2_000)}

    macro, overall, threshold = evaluate_evidence_query_h4._epoch_calibration(row)

    assert (macro, overall, threshold) == (0.5, 0.5, 0.42)
    passing = _gate_metrics()
    assert (
        evaluate_evidence_query_h4._quality_gates(
            passing, require_zero_wrong_presented=True
        )["all_quality_gates_passed"]
        is True
    )
    wrong_but_admissible = copy.deepcopy(passing)
    wrong_but_admissible["false_presented"] = _bucket(5, 100)
    assert (
        evaluate_evidence_query_h4._quality_gates(
            wrong_but_admissible, require_zero_wrong_presented=False
        )["all_quality_gates_passed"]
        is True
    )
    below_floor = copy.deepcopy(passing)
    below_floor["uncertain_target"] = _bucket(227, 250)
    assert (
        evaluate_evidence_query_h4._quality_gates(
            below_floor, require_zero_wrong_presented=True
        )["all_quality_gates_passed"]
        is False
    )


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
        evaluate_evidence_query_h4,
        "load_candidate_from_training_report",
        lambda path, **_kwargs: candidates[Path(path).name],
    )
    reports = (("first.json", "a" * 64), ("second.json", "b" * 64))
    kwargs = {
        "expected_manifest_sha256": "e" * 64,
        "training_data_dir": Path("training"),
        "training_bundle": object(),
    }

    primary, authenticated = evaluate_evidence_query_h4.authenticate_and_select_primary(
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
    primary, _ = evaluate_evidence_query_h4.authenticate_and_select_primary(
        reports, **kwargs
    )
    assert primary.seed == 20260805
    with pytest.raises(EvidenceQueryEvaluationError, match="exactly two"):
        evaluate_evidence_query_h4.authenticate_and_select_primary(
            reports[:1], **kwargs
        )


def test_report_loader_authenticates_checkpoint_and_both_source_classes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = b"selected H4 state mapping"
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
    changed = {"data_generator": "d" * 64, "training": "e" * 64}
    report = {
        "schema_version": H4_TRAINING_REPORT_SCHEMA_VERSION,
        "recipe": H4_TRAINING_RECIPE_VERSION,
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
        "hyperparameters": {},
        "epochs": epochs,
        "candidate": {"epoch": 2, **checkpoints[1]},
        "calibration": {"selected_epoch": 2, **calibrations[1]},
        "dev_used_for_selection": False,
        "fresh_v1_accessed": False,
        "preserved_source_sha256": dict(PRESERVED_H3_SOURCE_SHA256),
        "changed_source_sha256": changed,
        "runtime": {},
        "selection_note": evaluate_evidence_query_h4._TRAINING_SELECTION_NOTE,
    }
    report_path = tmp_path / "training-report.json"
    report_snapshot = canonical_json_bytes(report)
    report_path.write_bytes(report_snapshot)
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "_validate_report_dataset_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "_preserved_source_hashes",
        lambda: dict(PRESERVED_H3_SOURCE_SHA256),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4, "_changed_source_hashes", lambda: changed
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4.h3_evaluation,
        "_validate_hyperparameters",
        lambda _value: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4.h3_evaluation,
        "_validate_runtime",
        lambda _value, *, device: None,
    )

    candidate = evaluate_evidence_query_h4.load_candidate_from_training_report(
        report_path,
        expected_report_sha256=_sha256(report_snapshot),
        expected_manifest_sha256="f" * 64,
        training_data_dir=tmp_path,
        training_bundle=object(),
    )

    assert candidate.epoch == 2
    assert candidate.sha256 == _sha256(checkpoint)
    assert candidate.artifact_bytes == len(checkpoint)
    checkpoint_path.write_bytes(b"tampered")
    with pytest.raises(EvidenceQueryEvaluationError, match="SHA-256 mismatch"):
        evaluate_evidence_query_h4.load_candidate_from_training_report(
            report_path,
            expected_report_sha256=_sha256(report_snapshot),
            expected_manifest_sha256="f" * 64,
            training_data_dir=tmp_path,
            training_bundle=object(),
        )


def test_report_dataset_identity_binds_separate_h4_files(
    tmp_path: Path,
) -> None:
    snapshots = {
        "manifest": b"manifest",
        "fit": b"fit rows",
        "calibration": b"calibration rows",
    }
    (tmp_path / "manifest.json").write_bytes(snapshots["manifest"])
    (tmp_path / "fit.jsonl").write_bytes(snapshots["fit"])
    (tmp_path / "calibration.jsonl").write_bytes(snapshots["calibration"])
    manifest = {
        "partitions": {
            "fit": {"identity": "fit"},
            "calibration": {"identity": "calibration"},
        },
        "isolation": {"world_ids_disjoint": True},
    }
    bundle = SimpleNamespace(
        manifest=manifest,
        input_sha256={name: _sha256(payload) for name, payload in snapshots.items()},
    )
    value = {
        "schema_version": evaluate_evidence_query_h4.DATASET_SCHEMA_VERSION,
        "generator": evaluate_evidence_query_h4.surface_transfer_data.GENERATOR_VERSION,
        "target_grammar": evaluate_evidence_query_h4.TARGET_GRAMMAR_VERSION,
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
            "namespace": "train-fit",
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
        },
        "fit": {
            "identity": "fit",
            "state_class_counts": evaluate_evidence_query_h4._state_class_counts(
                evaluate_evidence_query_h4.FIT_STATE_CLASS_COUNTS
            ),
        },
        "calibration": {
            "identity": "calibration",
            "state_class_counts": evaluate_evidence_query_h4._state_class_counts(
                evaluate_evidence_query_h4.CALIBRATION_STATE_CLASS_COUNTS
            ),
        },
        "isolation": manifest["isolation"],
    }

    evaluate_evidence_query_h4._validate_report_dataset_metadata(
        value,
        training_data_dir=tmp_path,
        training_bundle=bundle,
        expected_manifest_sha256=_sha256(snapshots["manifest"]),
    )

    malformed = copy.deepcopy(value)
    malformed["source_calibration"]["sha256"] = "0" * 64
    with pytest.raises(EvidenceQueryEvaluationError, match="identity changed"):
        evaluate_evidence_query_h4._validate_report_dataset_metadata(
            malformed,
            training_data_dir=tmp_path,
            training_bundle=bundle,
            expected_manifest_sha256=_sha256(snapshots["manifest"]),
        )


def test_authentication_and_selection_precede_known_development_and_stop_raw_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
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
    failing_metrics = _gate_metrics()
    failing_metrics["overall"] = _bucket(3_040, 5_000)
    forbidden_calls: list[str] = []

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

    monkeypatch.setattr(evaluate_evidence_query_h4, "_evaluation_source_hashes", dict)
    monkeypatch.setattr(
        evaluate_evidence_query_h4, "load_h4_training_bundle", load_training
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "authenticate_and_select_primary",
        authenticate,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "_canonical_evaluation_runtime",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "_require_training_data_unchanged",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4, "load_development_bundle", load_development
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "load_pointer_tokenizer",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "encode_pointer_partition",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "build_pointer_inference_inputs",
        lambda *_args, **_kwargs: (inference_input,),
    )
    monkeypatch.setattr(evaluate_evidence_query_h4, "_fixture_cases", lambda *_args: ())
    monkeypatch.setattr(evaluate_evidence_query_h4, "_seed_evaluation", lambda: None)
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "_load_evidence_query_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "batched_evidence_query_inference",
        lambda *_args, **_kwargs: inference,
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "raw_pointer_diagnostics",
        lambda *_args, **_kwargs: {"acceptance": {"metrics": failing_metrics}},
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "apply_global_threshold",
        forbidden("threshold"),
    )
    monkeypatch.setattr(
        evaluate_evidence_query_h4,
        "_evaluate_predictions",
        forbidden("verifier"),
    )

    result = evaluate_evidence_query_h4.evaluate_development(
        training_data_dir=tmp_path / "training",
        training_manifest_sha256="e" * 64,
        development_data_dir=tmp_path / "development",
        development_manifest_sha256="d" * 64,
        tokenizer_path=tmp_path / "tokenizer.json",
        training_reports=(("a", "a" * 64), ("b", "b" * 64)),
        output_path=tmp_path / "result.json",
    )

    assert events == ["load-training", "authenticate-and-select", "load-development"]
    assert forbidden_calls == []
    assert result["schema_version"] == (
        "nano.evidence-query-h4-development-evaluation.v1"
    )
    assert result["training_data"]["manifest_sha256"] == "e" * 64
    assert result["partition"]["manifest_sha256"] == "d" * 64
    assert result["calibrated_raw"] is None
    assert result["verifier_final"] is None
    assert result["protocol"]["threshold_applied"] is False
    assert result["decision"]["quality_gate_passed"] is False
    assert (tmp_path / "result.json").is_file()


def test_evaluator_imports_no_benchmark_or_sealed_partition_module() -> None:
    source = Path(evaluate_evidence_query_h4.__file__).read_text(encoding="utf-8")
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
