from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_ai.training import replay_mixture_data, train_evidence_query_h5
from nano_ai.training.state_span_data import VARIANTS
from nano_ai.training.train_state_span import TrainingInputError


def test_h5_contract_changes_only_fit_family_composition() -> None:
    assert (
        train_evidence_query_h5.H5_TRAINING_REPORT_SCHEMA_VERSION
        == "nano.evidence-query-h5-training-report.v1"
    )
    assert (
        train_evidence_query_h5.H5_TRAINING_RECIPE_VERSION
        == replay_mixture_data.TRAINING_RECIPE_VERSION
        == "nano-evidence-query-replay-mixture-v1"
    )
    assert train_evidence_query_h5.TRAINING_SEEDS == (20260805, 20260806)
    assert train_evidence_query_h5.BATCH_SIZE == 32
    assert train_evidence_query_h5.EPOCHS == 3
    assert train_evidence_query_h5.STEPS_PER_EPOCH == 350
    assert train_evidence_query_h5.STEPS_PER_SEED == 1_050
    assert train_evidence_query_h5.SOURCE_WORLD_COUNT == 1_400
    assert train_evidence_query_h5.SOURCE_RECORD_COUNT == 5_600
    assert train_evidence_query_h5.FIT_WORLD_COUNT == 2_800
    assert train_evidence_query_h5.FIT_RECORD_COUNT == 11_200
    assert train_evidence_query_h5.CALIBRATION_WORLD_COUNT == 200
    assert train_evidence_query_h5.CALIBRATION_RECORD_COUNT == 800


def test_h5_state_counts_preserve_h4_training_exposure() -> None:
    assert train_evidence_query_h5.FIT_STATE_CLASS_COUNTS == (
        42_980,
        4_620,
        2_800,
        2_800,
        2_800,
    )
    assert (
        tuple(
            fit + calibration
            for fit, calibration in zip(
                train_evidence_query_h5.FIT_STATE_CLASS_COUNTS,
                train_evidence_query_h5.CALIBRATION_STATE_CLASS_COUNTS,
                strict=True,
            )
        )
        == train_evidence_query_h5.STATE_CLASS_COUNTS
    )
    assert dict(replay_mixture_data.EXPECTED_SOURCE_STATE_CLASS_COUNTS) == {
        state.value: count // 2
        for state, count in zip(
            train_evidence_query_h5.STATE_ORDER,
            train_evidence_query_h5.FIT_STATE_CLASS_COUNTS,
            strict=True,
        )
    }


def test_literal_h3_source_pins_and_h5_authorities_are_separate() -> None:
    observed = train_evidence_query_h5._require_preserved_h3_sources()

    assert observed == dict(train_evidence_query_h5.PRESERVED_H3_SOURCE_SHA256)
    assert set(train_evidence_query_h5.changed_source_paths()) == {
        "data_generator",
        "training",
    }
    assert (
        train_evidence_query_h5.changed_source_paths()["data_generator"]
        == Path(replay_mixture_data.__file__).resolve()
    )


def test_step_preflight_exposes_every_fit_record_once_per_epoch() -> None:
    examples = tuple(
        SimpleNamespace(world_id=f"train-world-replay-{world:04d}")
        for world in range(train_evidence_query_h5.FIT_WORLD_COUNT)
        for _variant in VARIANTS
    )

    batches = train_evidence_query_h5._require_exact_training_steps(
        examples, seed=20260805
    )

    assert set(batches) == {1, 2, 3}
    for epoch_batches in batches.values():
        assert len(epoch_batches) == 350
        assert all(len(batch) == 32 for batch in epoch_batches)
        assert sorted(index for batch in epoch_batches for index in batch) == list(
            range(train_evidence_query_h5.FIT_RECORD_COUNT)
        )


def test_cuda_preflight_precedes_sources_data_model_and_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"

    def deterministic_cuda_failure(_requested: str) -> str:
        raise TrainingInputError("deterministic CUDA requires frozen config")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("later preflight ran before CUDA validation")

    monkeypatch.setattr(
        train_evidence_query_h5,
        "_resolve_evidence_query_device",
        deterministic_cuda_failure,
    )
    monkeypatch.setattr(
        train_evidence_query_h5, "_require_preserved_h3_sources", unexpected
    )
    monkeypatch.setattr(
        replay_mixture_data,
        "load_replay_mixture_dataset",
        unexpected,
        raising=False,
    )

    with pytest.raises(TrainingInputError, match="deterministic CUDA"):
        train_evidence_query_h5.train_evidence_query_h5_candidate(
            data_dir=tmp_path / "data",
            base_checkpoint=tmp_path / "base.pt",
            tokenizer_path=tmp_path / "tokenizer.json",
            output_dir=output,
            seed=20260805,
            device="cuda",
        )

    assert not output.exists()


def test_preserved_source_drift_fails_before_data_or_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"

    monkeypatch.setattr(
        train_evidence_query_h5,
        "_resolve_evidence_query_device",
        lambda _requested: "cpu",
    )
    monkeypatch.setattr(
        train_evidence_query_h5,
        "_require_preserved_h3_sources",
        lambda: (_ for _ in ()).throw(
            TrainingInputError("H5 preserved H3 source hash mismatch: training")
        ),
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("data was opened after source-auth failure")

    monkeypatch.setattr(
        replay_mixture_data,
        "load_replay_mixture_dataset",
        unexpected,
        raising=False,
    )

    with pytest.raises(TrainingInputError, match="source hash mismatch"):
        train_evidence_query_h5.train_evidence_query_h5_candidate(
            data_dir=tmp_path / "data",
            base_checkpoint=tmp_path / "base.pt",
            tokenizer_path=tmp_path / "tokenizer.json",
            output_dir=output,
            seed=20260805,
            device="cpu",
        )

    assert not output.exists()


def test_trainer_delegates_dataset_authentication_to_replay_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"

    monkeypatch.setattr(
        train_evidence_query_h5,
        "_resolve_evidence_query_device",
        lambda _requested: "cpu",
    )
    monkeypatch.setattr(train_evidence_query_h5, "_require_preserved_h3_sources", dict)
    monkeypatch.setattr(train_evidence_query_h5, "_changed_source_hashes", dict)
    monkeypatch.setattr(
        train_evidence_query_h5,
        "_file_sha256",
        lambda _path, *, label: (
            train_evidence_query_h5.FROZEN_NANO_V01.checkpoint_sha256
            if label == "base checkpoint"
            else train_evidence_query_h5.FROZEN_NANO_V01.tokenizer_sha256
        ),
    )

    def authenticated_loader_reached(_data_dir: Path):
        raise TrainingInputError("authenticated replay loader reached")

    monkeypatch.setattr(
        replay_mixture_data,
        "load_replay_mixture_dataset",
        authenticated_loader_reached,
        raising=False,
    )

    with pytest.raises(TrainingInputError, match="replay loader reached"):
        train_evidence_query_h5.train_evidence_query_h5_candidate(
            data_dir=tmp_path / "data",
            base_checkpoint=tmp_path / "base.pt",
            tokenizer_path=tmp_path / "tokenizer.json",
            output_dir=output,
            seed=20260805,
            device="cpu",
        )

    assert not output.exists()


def test_authenticated_input_recheck_detects_partition_mutation(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    paths = {
        "manifest": data_dir / "manifest.json",
        "fit": data_dir / "fit.jsonl",
        "calibration": data_dir / "calibration.jsonl",
        "base_checkpoint": tmp_path / "base.pt",
        "tokenizer": tmp_path / "tokenizer.json",
    }
    for name, path in paths.items():
        path.write_bytes(f"{name}\n".encode())
    expected = {
        name: train_evidence_query_h5._file_sha256(path, label=name)
        for name, path in paths.items()
    }

    train_evidence_query_h5._require_input_hashes(
        data_dir=data_dir,
        base_checkpoint=paths["base_checkpoint"],
        tokenizer_path=paths["tokenizer"],
        expected=expected,
    )
    paths["fit"].write_bytes(b"mutated\n")

    with pytest.raises(TrainingInputError, match="authenticated input changed"):
        train_evidence_query_h5._require_input_hashes(
            data_dir=data_dir,
            base_checkpoint=paths["base_checkpoint"],
            tokenizer_path=paths["tokenizer"],
            expected=expected,
        )


def test_h5_trainer_has_no_legacy_record_loader_dependency() -> None:
    source = Path(train_evidence_query_h5.__file__).read_text(encoding="utf-8")

    assert "load_legacy_reproduction" not in source
    assert "legacy_manifest_path" not in source
    assert "write_replay_mixture_dataset" not in source
