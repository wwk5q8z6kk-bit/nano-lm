from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nano_ai.training import train_evidence_query, train_pointer, train_state_span
from nano_ai.training.evidence_query_model import ARCHITECTURE_VERSION
from nano_ai.training.state_span_data import generate_split
from nano_ai.training.train_state_span import TrainingInputError


def _bucket(numerator: int, denominator: int) -> dict[str, float | int]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else 0.0,
    }


def _metric_set(
    rates: tuple[int, int, int, int],
    *,
    overall: int,
    wrong_presented: int,
) -> dict[str, object]:
    slices = {
        name: _bucket(numerator, 100)
        for name, numerator in zip(
            train_evidence_query.CALIBRATION_SELECTION_SLICES,
            rates,
            strict=True,
        )
    }
    slices["overall"] = _bucket(overall, 4_000)
    return {
        "slices": slices,
        "selection": {
            "macro_joint": sum(
                bucket["rate"]
                for bucket in slices.values()
                if bucket is not slices["overall"]
            )
            / 4,
            "overall_joint": slices["overall"]["rate"],
        },
        "wrong_presented": _bucket(wrong_presented, 500),
    }


def _calibration_summary(
    uncalibrated_rates: tuple[int, int, int, int],
    *,
    uncalibrated_overall: int,
    calibrated_rates: tuple[int, int, int, int] = (90, 90, 90, 90),
    calibrated_overall: int = 3_600,
) -> dict[str, object]:
    return {
        "uncalibrated": _metric_set(
            uncalibrated_rates,
            overall=uncalibrated_overall,
            wrong_presented=4,
        ),
        "global_threshold": 0.42,
        "calibrated": _metric_set(
            calibrated_rates,
            overall=calibrated_overall,
            wrong_presented=0,
        ),
        "threshold_policy": "minimal_zero_wrong_presented_inclusive_v1",
    }


def _epoch(
    number: int,
    rates: tuple[int, int, int, int],
    overall: int,
    *,
    calibrated_rates: tuple[int, int, int, int] = (90, 90, 90, 90),
) -> dict[str, object]:
    return {
        "epoch": number,
        "calibration": _calibration_summary(
            rates,
            uncalibrated_overall=overall,
            calibrated_rates=calibrated_rates,
        ),
    }


def test_h3_training_contract_reuses_h2_recipe_except_architecture_and_split() -> None:
    assert ARCHITECTURE_VERSION == "nano_evidence_query_pointer_v1"
    assert (
        train_evidence_query.EVIDENCE_QUERY_TRAINING_REPORT_SCHEMA_VERSION
        == "nano.evidence-query-training-report.v0"
    )
    assert (
        train_evidence_query.EVIDENCE_QUERY_TRAINING_RECIPE_VERSION
        == "nano-evidence-query-architecture-only-v0"
    )
    assert train_evidence_query.EPOCHS == train_state_span.EPOCHS == 3
    assert train_evidence_query.BATCH_SIZE == train_state_span.BATCH_SIZE == 32
    assert train_evidence_query.TRAINING_SEEDS == (20260805, 20260806)
    assert train_evidence_query.STATE_CLASS_COUNTS == train_pointer.STATE_CLASS_COUNTS
    assert train_evidence_query.STATE_CLASS_WEIGHTS == train_pointer.STATE_CLASS_WEIGHTS
    assert (
        tuple(
            fit + calibration
            for fit, calibration in zip(
                train_evidence_query.FIT_STATE_CLASS_COUNTS,
                train_evidence_query.CALIBRATION_STATE_CLASS_COUNTS,
                strict=True,
            )
        )
        == train_pointer.STATE_CLASS_COUNTS
    )


def test_h3_world_partition_is_exact_disjoint_and_pre_encoding() -> None:
    examples = generate_split("train")

    fit, calibration = train_evidence_query.split_fit_calibration_worlds(examples)

    assert len(fit) == 11_200
    assert len(calibration) == 800
    assert {example.world_id for example in fit} == {
        f"train-world-{index:04d}" for index in range(2_800)
    }
    assert {example.world_id for example in calibration} == {
        f"train-world-{index:04d}" for index in range(2_800, 3_000)
    }
    assert {example.world_id for example in fit}.isdisjoint(
        example.world_id for example in calibration
    )


def test_training_splits_worlds_before_either_partition_reaches_encoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    examples = generate_split("train")
    encoded_boundaries: list[tuple[int, str, str]] = []

    monkeypatch.setattr(
        train_evidence_query,
        "_resolve_evidence_query_device",
        lambda _requested: "cpu",
    )
    monkeypatch.setattr(train_evidence_query, "_source_hashes", dict)
    monkeypatch.setattr(
        train_evidence_query,
        "load_training_bundle",
        lambda _path: SimpleNamespace(train=examples),
    )
    monkeypatch.setattr(
        train_evidence_query,
        "load_pointer_tokenizer",
        lambda _path: object(),
    )

    class EncodingComplete(RuntimeError):
        pass

    def record_partition(_tokenizer, partition, *, expected_split):
        assert expected_split == "train"
        encoded_boundaries.append(
            (len(partition), partition[0].world_id, partition[-1].world_id)
        )
        if len(encoded_boundaries) == 2:
            raise EncodingComplete
        return ()

    monkeypatch.setattr(
        train_evidence_query,
        "encode_pointer_partition",
        record_partition,
    )
    output = tmp_path / "candidate"

    with pytest.raises(EncodingComplete):
        train_evidence_query.train_evidence_query_candidate(
            data_dir=tmp_path / "data",
            base_checkpoint=tmp_path / "base.pt",
            tokenizer_path=tmp_path / "tokenizer.json",
            output_dir=output,
            seed=20260805,
            device="cpu",
        )

    assert encoded_boundaries == [
        (11_200, "train-world-0000", "train-world-2799"),
        (800, "train-world-2800", "train-world-2999"),
    ]
    assert not output.exists()


def test_epoch_selection_uses_uncalibrated_macro_then_overall() -> None:
    reports = [
        _epoch(1, (40, 40, 40, 40), 3_800, calibrated_rates=(100, 100, 100, 100)),
        _epoch(2, (60, 60, 60, 60), 2_000, calibrated_rates=(99, 99, 99, 99)),
        _epoch(3, (60, 60, 60, 60), 2_400, calibrated_rates=(1, 1, 1, 1)),
    ]

    selected = train_evidence_query.select_epoch_report(reports)

    assert selected is reports[2]


def test_epoch_selection_breaks_an_exact_uncalibrated_tie_to_earlier_epoch() -> None:
    reports = [
        _epoch(1, (55, 55, 55, 55), 2_200, calibrated_rates=(1, 1, 1, 1)),
        _epoch(2, (55, 55, 55, 55), 2_200, calibrated_rates=(100, 100, 100, 100)),
        _epoch(3, (10, 10, 10, 10), 3_900),
    ]

    selected = train_evidence_query.select_epoch_report(reports)

    assert selected is reports[0]


def test_calibration_validator_requires_canonical_threshold_policy() -> None:
    summary = _calibration_summary((50, 50, 50, 50), uncalibrated_overall=2_000)
    summary["threshold_policy"] = (
        "lowest_observed_global_presented_confidence_with_zero_wrong_presented"
    )

    with pytest.raises(TrainingInputError, match="threshold policy drifted"):
        train_evidence_query._validate_calibration_summary(summary)


def test_calibration_validator_rejects_bad_boundary_or_residual_error() -> None:
    wrong_boundary = _calibration_summary(
        (50, 50, 50, 50),
        uncalibrated_overall=2_000,
    )
    wrong_boundary["uncalibrated"]["slices"]["overall"] = _bucket(2_000, 3_999)
    wrong_boundary["uncalibrated"]["selection"]["overall_joint"] = 2_000 / 3_999
    with pytest.raises(TrainingInputError, match="overall denominator drifted"):
        train_evidence_query._validate_calibration_summary(wrong_boundary)

    residual_error = _calibration_summary(
        (50, 50, 50, 50),
        uncalibrated_overall=2_000,
    )
    residual_error["calibrated"]["wrong_presented"] = _bucket(1, 500)
    with pytest.raises(TrainingInputError, match="zero-wrong-presented"):
        train_evidence_query._validate_calibration_summary(residual_error)


def test_shared_calibration_path_owns_inference_gold_and_threshold_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nano_ai.training import evidence_query_inference

    examples = tuple(
        SimpleNamespace(
            example_id=f"calibration-{index:04d}",
            target=f"target-{index}",
            transcript=f"transcript-{index}",
        )
        for index in range(800)
    )
    records = tuple(object() for _ in examples)
    inputs = object()
    inference = object()
    summary = _calibration_summary(
        (50, 60, 70, 80),
        uncalibrated_overall=2_500,
    )
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        train_evidence_query,
        "build_pointer_inference_inputs",
        lambda actual_examples, actual_records: (
            observed.update(
                {"inputs_examples": actual_examples, "inputs_records": actual_records}
            )
            or inputs
        ),
    )
    monkeypatch.setattr(
        train_evidence_query,
        "parse_state_span_summary",
        lambda target, transcript: (target, transcript),
    )
    monkeypatch.setattr(
        evidence_query_inference,
        "CalibrationGold",
        lambda *, example_id, proposals: (example_id, proposals),
    )

    def fake_inference(model, actual_inputs, *, device, batch_size):
        observed.update(
            {
                "model": model,
                "inference_inputs": actual_inputs,
                "device": device,
                "batch_size": batch_size,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return inference

    monkeypatch.setattr(
        evidence_query_inference,
        "batched_evidence_query_inference",
        fake_inference,
    )

    class Selection:
        def to_dict(self):
            observed["serialized"] = True
            return copy.deepcopy(summary)

    def fake_selector(actual_inference, gold):
        observed["selector_inference"] = actual_inference
        observed["gold"] = gold
        return Selection()

    monkeypatch.setattr(
        evidence_query_inference,
        "select_global_threshold",
        fake_selector,
    )
    model = SimpleNamespace(eval=lambda: observed.update({"eval": True}))

    result = train_evidence_query._calibrate_model(
        model,
        examples,
        records,
        device="cpu",
    )

    assert result == summary
    assert observed["inputs_examples"] is examples
    assert observed["inputs_records"] is records
    assert observed["eval"] is True
    assert observed["model"] is model
    assert observed["inference_inputs"] is inputs
    assert observed["device"] == "cpu"
    assert observed["batch_size"] == 32
    assert observed["grad_enabled"] is False
    assert observed["selector_inference"] is inference
    assert len(observed["gold"]) == 800
    assert observed["gold"][0] == (
        "calibration-0000",
        ("target-0", "transcript-0"),
    )
    assert observed["serialized"] is True


def test_epoch_checkpoint_creation_is_no_clobber(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    path = tmp_path / "epoch-1.pt"

    identity = train_evidence_query._save_checkpoint(path, model)
    snapshot = path.read_bytes()

    assert identity["filename"] == "epoch-1.pt"
    assert identity["bytes"] == len(snapshot)
    assert len(identity["sha256"]) == 64
    with pytest.raises(RuntimeError, match="could not create checkpoint"):
        train_evidence_query._save_checkpoint(path, model)
    assert path.read_bytes() == snapshot


@pytest.mark.parametrize("configuration", (":4096:8", ":16:8"))
def test_h3_cuda_requires_deterministic_cublas_configuration(
    monkeypatch: pytest.MonkeyPatch,
    configuration: str,
) -> None:
    monkeypatch.setattr(
        train_evidence_query,
        "_resolve_base_device",
        lambda _device: "cuda",
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(TrainingInputError, match="deterministic CUDA requires"):
        train_evidence_query._resolve_evidence_query_device("cuda")

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(TrainingInputError, match="deterministic CUDA requires"):
        train_evidence_query._resolve_evidence_query_device("cuda")

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", configuration)
    assert train_evidence_query._resolve_evidence_query_device("cuda") == "cuda"


def test_h3_cuda_preflight_precedes_data_model_and_output_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"
    monkeypatch.setattr(
        train_evidence_query,
        "_resolve_base_device",
        lambda _device: "cuda",
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("CUDA preflight did not fail first")

    monkeypatch.setattr(train_evidence_query, "_source_hashes", unexpected)
    monkeypatch.setattr(train_evidence_query, "load_training_bundle", unexpected)
    monkeypatch.setattr(
        train_evidence_query.NanoEvidenceQueryPointerModel,
        "from_frozen_base",
        unexpected,
    )

    with pytest.raises(TrainingInputError, match="deterministic CUDA requires"):
        train_evidence_query.train_evidence_query_candidate(
            data_dir=tmp_path / "data",
            base_checkpoint=tmp_path / "base.pt",
            tokenizer_path=tmp_path / "tokenizer.json",
            output_dir=output,
            seed=20260805,
            device="cuda",
        )

    assert not output.exists()


def test_training_source_mapping_is_complete_and_evaluator_compatible() -> None:
    paths = train_evidence_query.training_source_paths()

    assert paths == train_evidence_query.expected_training_source_paths()
    assert set(paths) == {
        "base_model",
        "data_generator",
        "evidence_query_inference",
        "evidence_query_model",
        "h1_training_loader",
        "h2_objective",
        "pointer_data",
        "pointer_decoder",
        "state_span_adapter",
        "training",
    }
    assert all(path.is_file() for path in paths.values())
