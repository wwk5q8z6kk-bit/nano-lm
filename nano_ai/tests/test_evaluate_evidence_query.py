from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.training import evaluate_evidence_query
from nano_ai.training.evaluate_evidence_query import (
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
        "threshold_policy": evaluate_evidence_query.CALIBRATION_THRESHOLD_POLICY,
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
    runtime = {
        "python": evaluate_evidence_query.platform.python_version(),
        "torch": evaluate_evidence_query.torch.__version__,
        "tokenizers": getattr(__import__("tokenizers"), "__version__", None),
        "cuda": evaluate_evidence_query.torch.version.cuda,
        "gpu": None,
        "cublas_workspace_config": None,
        "platform": evaluate_evidence_query.platform.platform(),
        "seconds": 1.0,
    }
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
        report={"device": "cpu", "runtime": runtime},
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


def test_calibration_uses_uncalibrated_ranking_and_bucketed_wrong_presented() -> None:
    row = {"calibration": _calibration(uncalibrated_overall=2_000)}

    macro, overall, threshold = evaluate_evidence_query._epoch_calibration(row)

    assert macro == 0.5
    assert overall == 0.5
    assert threshold == 0.42
    malformed = copy.deepcopy(row)
    malformed["calibration"]["calibrated"]["wrong_presented"] = 0
    with pytest.raises(EvidenceQueryEvaluationError, match="wrong_presented"):
        evaluate_evidence_query._epoch_calibration(malformed)


def test_quality_gates_enforce_every_absolute_floor_and_zero_wrong() -> None:
    passing = _gate_metrics()

    result = evaluate_evidence_query._quality_gates(
        passing,
        require_zero_wrong_presented=True,
    )

    assert result["all_quality_gates_passed"] is True
    failing = copy.deepcopy(passing)
    failing["held_value"] = _bucket(1_904, 2_987)
    assert (
        evaluate_evidence_query._quality_gates(
            failing,
            require_zero_wrong_presented=True,
        )["all_quality_gates_passed"]
        is False
    )
    failing = copy.deepcopy(passing)
    failing["failures"] = _bucket(11, 1_000)
    assert (
        evaluate_evidence_query._quality_gates(
            failing,
            require_zero_wrong_presented=True,
        )["all_quality_gates_passed"]
        is False
    )
    failing = copy.deepcopy(passing)
    failing["false_presented"] = _bucket(1, 100)
    assert (
        evaluate_evidence_query._quality_gates(
            failing,
            require_zero_wrong_presented=True,
        )["all_quality_gates_passed"]
        is False
    )


def test_primary_selection_uses_macro_overall_epoch_then_frozen_seed(
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

    def load(path, **_kwargs):
        return candidates[Path(path).name]

    monkeypatch.setattr(
        evaluate_evidence_query,
        "load_candidate_from_training_report",
        load,
    )
    reports = (("first.json", "a" * 64), ("second.json", "b" * 64))

    primary, _authenticated = evaluate_evidence_query.authenticate_and_select_primary(
        reports,
        expected_manifest_sha256="e" * 64,
    )

    assert primary.seed == 20260806
    candidates["first.json"] = _candidate(
        seed=20260805,
        epoch=2,
        macro=0.8,
        overall=0.7,
        checkpoint_digest="c" * 64,
        report_digest="a" * 64,
    )
    primary, _authenticated = evaluate_evidence_query.authenticate_and_select_primary(
        reports,
        expected_manifest_sha256="e" * 64,
    )
    assert primary.seed == 20260805


def test_primary_authentication_requires_exactly_two_reports() -> None:
    with pytest.raises(EvidenceQueryEvaluationError, match="exactly two"):
        evaluate_evidence_query.authenticate_and_select_primary(
            (("only.json", "a" * 64),),
            expected_manifest_sha256="e" * 64,
        )


def test_report_loader_authenticates_selected_checkpoint_hash_and_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = b"selected H3 state mapping"
    checkpoint_path = tmp_path / "epoch-2.pt"
    checkpoint_path.write_bytes(checkpoint)
    checkpoints = [
        {
            "filename": f"epoch-{epoch}.pt",
            "sha256": (str(epoch) * 64),
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
    selected = checkpoints[1]
    report = {
        "schema_version": (
            evaluate_evidence_query.EVIDENCE_QUERY_TRAINING_REPORT_SCHEMA_VERSION
        ),
        "recipe": evaluate_evidence_query.EVIDENCE_QUERY_TRAINING_RECIPE_VERSION,
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
        "dataset_manifest_sha256": "e" * 64,
        "dataset": {},
        "hyperparameters": {},
        "epochs": epochs,
        "candidate": {"epoch": 2, **selected},
        "calibration": {"selected_epoch": 2, **calibrations[1]},
        "source_sha256": {"training": "f" * 64},
        "runtime": {},
        "selection_note": evaluate_evidence_query._TRAINING_SELECTION_NOTE,
        "dev_used_for_selection": False,
        "fresh_v1_accessed": False,
    }
    report_path = tmp_path / "training-report.json"
    report_snapshot = canonical_json_bytes(report)
    report_path.write_bytes(report_snapshot)
    monkeypatch.setattr(
        evaluate_evidence_query,
        "_training_source_paths",
        lambda: {"training": tmp_path / "training.py"},
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "_training_source_hashes",
        lambda: {"training": "f" * 64},
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "_validate_report_dataset_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "_validate_hyperparameters",
        lambda _value: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "_validate_runtime",
        lambda _value, *, device: None,
    )

    candidate = evaluate_evidence_query.load_candidate_from_training_report(
        report_path,
        expected_report_sha256=_sha256(report_snapshot),
        expected_manifest_sha256="e" * 64,
    )

    assert candidate.epoch == 2
    assert candidate.sha256 == _sha256(checkpoint)
    assert candidate.artifact_bytes == len(checkpoint)
    checkpoint_path.write_bytes(b"tampered")
    with pytest.raises(EvidenceQueryEvaluationError, match="SHA-256 mismatch"):
        evaluate_evidence_query.load_candidate_from_training_report(
            report_path,
            expected_report_sha256=_sha256(report_snapshot),
            expected_manifest_sha256="e" * 64,
        )


def test_primary_selection_precedes_loading_known_development(
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

    def authenticate(*_args, **_kwargs):
        events.append("authenticate-and-select")
        return candidate, (candidate,)

    def load_dev(*_args, **_kwargs):
        events.append("load-development")
        raise RuntimeError("sentinel")

    monkeypatch.setattr(
        evaluate_evidence_query,
        "_evaluation_source_hashes",
        dict,
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "authenticate_and_select_primary",
        authenticate,
    )
    monkeypatch.setattr(evaluate_evidence_query, "load_development_bundle", load_dev)

    with pytest.raises(EvidenceQueryEvaluationError, match="could not be verified"):
        evaluate_evidence_query.evaluate_development(
            data_dir=tmp_path / "data",
            manifest_sha256="e" * 64,
            tokenizer_path=tmp_path / "tokenizer.json",
            training_reports=(("a", "a" * 64), ("b", "b" * 64)),
            output_path=tmp_path / "result.json",
        )

    assert events == ["authenticate-and-select", "load-development"]


def test_canonical_evaluation_pins_batch_device_and_runtime() -> None:
    first = _candidate(
        seed=20260805,
        epoch=1,
        macro=0.8,
        overall=0.7,
        checkpoint_digest="c" * 64,
        report_digest="a" * 64,
    )
    second = _candidate(
        seed=20260806,
        epoch=1,
        macro=0.7,
        overall=0.6,
        checkpoint_digest="d" * 64,
        report_digest="b" * 64,
    )

    observed = evaluate_evidence_query._canonical_evaluation_runtime(
        first,
        (first, second),
        device="cpu",
        batch_size=32,
    )

    assert observed["gpu"] is None
    with pytest.raises(EvidenceQueryEvaluationError, match="batch_size=32"):
        evaluate_evidence_query._canonical_evaluation_runtime(
            first,
            (first, second),
            device="cpu",
            batch_size=16,
        )
    with pytest.raises(EvidenceQueryEvaluationError, match="device must match"):
        evaluate_evidence_query._canonical_evaluation_runtime(
            first,
            (first, second),
            device="mps",
            batch_size=32,
        )


def test_canonical_evaluation_requires_both_seeds_on_same_runtime() -> None:
    first = _candidate(
        seed=20260805,
        epoch=1,
        macro=0.8,
        overall=0.7,
        checkpoint_digest="c" * 64,
        report_digest="a" * 64,
    )
    second = _candidate(
        seed=20260806,
        epoch=1,
        macro=0.7,
        overall=0.6,
        checkpoint_digest="d" * 64,
        report_digest="b" * 64,
    )
    second.report["runtime"]["gpu"] = "another GPU"

    with pytest.raises(EvidenceQueryEvaluationError, match="one canonical runtime"):
        evaluate_evidence_query._canonical_evaluation_runtime(
            first,
            (first, second),
            device="cpu",
            batch_size=32,
        )


def test_failed_uncalibrated_admission_skips_threshold_and_verifier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = _candidate(
        seed=20260805,
        epoch=1,
        macro=0.8,
        overall=0.7,
        checkpoint_digest="c" * 64,
        report_digest="a" * 64,
    )
    example = SimpleNamespace(world_id="dev-world")
    bundle = SimpleNamespace(
        manifest={"train": {"sha256": "f" * 64}},
        manifest_sha256="e" * 64,
        dev_sha256="d" * 64,
        examples=(example,),
    )
    inference_input = SimpleNamespace(example_id="dev-1")
    inference = SimpleNamespace(
        example_ids=("dev-1",),
        predictions=(object(),),
    )
    failing_metrics = _gate_metrics()
    failing_metrics["overall"] = _bucket(3_040, 5_000)
    raw_calls: list[str] = []
    forbidden_calls: list[str] = []

    def raw_diagnostics(*_args, **_kwargs):
        raw_calls.append("uncalibrated")
        return {"acceptance": {"metrics": failing_metrics}}

    def forbid_threshold(*_args, **_kwargs):
        forbidden_calls.append("threshold")
        raise AssertionError("threshold must not be applied after failed admission")

    def forbid_verifier(*_args, **_kwargs):
        forbidden_calls.append("verifier")
        raise AssertionError("verifier must not run after failed admission")

    monkeypatch.setattr(evaluate_evidence_query, "_evaluation_source_hashes", dict)
    monkeypatch.setattr(
        evaluate_evidence_query,
        "authenticate_and_select_primary",
        lambda *_args, **_kwargs: (candidate, (candidate,)),
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "load_development_bundle",
        lambda *_args, **_kwargs: bundle,
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "_validate_dataset_identity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "load_pointer_tokenizer",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "encode_pointer_partition",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "build_pointer_inference_inputs",
        lambda *_args, **_kwargs: (inference_input,),
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "_fixture_cases",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(evaluate_evidence_query, "_seed_evaluation", lambda: None)
    monkeypatch.setattr(
        evaluate_evidence_query,
        "_load_evidence_query_model",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "batched_evidence_query_inference",
        lambda *_args, **_kwargs: inference,
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "raw_pointer_diagnostics",
        raw_diagnostics,
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "apply_global_threshold",
        forbid_threshold,
    )
    monkeypatch.setattr(
        evaluate_evidence_query,
        "_evaluate_predictions",
        forbid_verifier,
    )

    result = evaluate_evidence_query.evaluate_development(
        data_dir=tmp_path / "data",
        manifest_sha256="e" * 64,
        tokenizer_path=tmp_path / "tokenizer.json",
        training_reports=(("a", "a" * 64), ("b", "b" * 64)),
        output_path=tmp_path / "result.json",
    )

    assert raw_calls == ["uncalibrated"]
    assert forbidden_calls == []
    assert result["status"] == "complete"
    assert result["calibrated_raw"] is None
    assert result["calibrated_quality"] is None
    assert result["verifier_final"] is None
    assert result["verifier_final_quality"] is None
    assert result["protocol"]["threshold_applied"] is False
    assert result["decision"] == {
        "uncalibrated_semantic_admission_passed": False,
        "calibrated_raw_quality_passed": None,
        "verifier_final_quality_passed": None,
        "quality_gate_passed": False,
        "latency_assessed": False,
        "fresh_v1_assessed": False,
        "next_step": (
            "reject H3 quality candidate; do not measure latency or open fresh-v1"
        ),
    }
    assert (tmp_path / "result.json").is_file()


def test_evaluator_has_no_benchmark_or_fresh_partition_imports() -> None:
    source = Path(evaluate_evidence_query.__file__).read_text(encoding="utf-8")
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
