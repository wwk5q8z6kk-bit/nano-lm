from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_ai.contract import FieldState
from nano_ai.training import surface_transfer_data, train_evidence_query_h4
from nano_ai.training.state_span_data import VARIANTS, canonical_json_bytes
from nano_ai.training.train_state_span import TrainingInputError


def test_h4_contract_changes_data_only_and_preserves_exposure() -> None:
    assert (
        train_evidence_query_h4.H4_TRAINING_REPORT_SCHEMA_VERSION
        == "nano.evidence-query-h4-training-report.v1"
    )
    assert (
        train_evidence_query_h4.H4_TRAINING_RECIPE_VERSION
        == surface_transfer_data.TRAINING_RECIPE_VERSION
        == "nano-evidence-query-data-only-v1"
    )
    assert train_evidence_query_h4.TRAINING_SEEDS == (20260805, 20260806)
    assert train_evidence_query_h4.BATCH_SIZE == 32
    assert train_evidence_query_h4.EPOCHS == 3
    assert train_evidence_query_h4.STEPS_PER_EPOCH == 350
    assert train_evidence_query_h4.STEPS_PER_SEED == 1_050
    assert train_evidence_query_h4.FIT_RECORD_COUNT == 11_200
    assert train_evidence_query_h4.CALIBRATION_RECORD_COUNT == 800


def test_generated_h4_bundle_authenticates_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "surface-transfer-v1"
    surface_transfer_data.write_dataset(
        data_dir,
        tokenizer_sha256=train_evidence_query_h4.FROZEN_NANO_V01.tokenizer_sha256,
        base_checkpoint_sha256=(
            train_evidence_query_h4.FROZEN_NANO_V01.checkpoint_sha256
        ),
    )

    bundle = train_evidence_query_h4.load_h4_training_bundle(data_dir)

    assert len(bundle.fit) == train_evidence_query_h4.FIT_RECORD_COUNT
    assert (
        len(bundle.calibration)
        == train_evidence_query_h4.CALIBRATION_RECORD_COUNT
    )
    for partition in ("fit", "calibration"):
        quality = bundle.manifest["partitions"][partition]["quality"]
        assert quality["all_transcripts_unique"] is True
        assert quality["all_supported_conflicts_deranged"] is True


def test_literal_h3_source_pins_match_the_preserved_authorities() -> None:
    observed = train_evidence_query_h4._require_preserved_h3_sources()

    assert observed == dict(train_evidence_query_h4.PRESERVED_H3_SOURCE_SHA256)
    assert set(observed) == {
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
    assert set(train_evidence_query_h4.changed_source_paths()) == {
        "data_generator",
        "training",
    }


@pytest.mark.parametrize("partition", ("fit", "calibration"))
def test_world_family_validator_enforces_namespaces_and_four_variants(
    partition: str,
) -> None:
    records = surface_transfer_data.generate_partition(partition, worlds=5)

    train_evidence_query_h4._validate_world_families(
        records,
        partition=partition,
        expected_worlds=5,
        expected_records=20,
    )

    corrupted = list(records)
    corrupted[-1] = replace(records[-1], example_id=records[0].example_id)
    with pytest.raises(TrainingInputError, match="record namespace drifted"):
        train_evidence_query_h4._validate_world_families(
            corrupted,
            partition=partition,
            expected_worlds=5,
            expected_records=20,
        )


def test_partition_loader_rejects_duplicate_keys_and_noncanonical_json() -> None:
    duplicate = b'{"split":"train","split":"train"}\n'
    with pytest.raises(TrainingInputError, match="invalid H4 JSON"):
        train_evidence_query_h4._load_partition(
            duplicate, filename="fit.jsonl"
        )

    record = surface_transfer_data.generate_partition("fit", worlds=5)[0]
    noncanonical = json.dumps(record.to_dict()).encode("utf-8") + b"\n"
    assert noncanonical != canonical_json_bytes(record.to_dict())
    with pytest.raises(TrainingInputError, match="not canonical JSONL"):
        train_evidence_query_h4._load_partition(
            noncanonical, filename="fit.jsonl"
        )


def test_state_field_quotas_sum_to_frozen_class_counts() -> None:
    for quota, class_counts in (
        (
            train_evidence_query_h4.FIT_STATE_FIELD_QUOTA,
            train_evidence_query_h4.FIT_STATE_CLASS_COUNTS,
        ),
        (
            train_evidence_query_h4.CALIBRATION_STATE_FIELD_QUOTA,
            train_evidence_query_h4.CALIBRATION_STATE_CLASS_COUNTS,
        ),
    ):
        observed = []
        for state in train_evidence_query_h4.STATE_ORDER:
            observed.append(
                sum(
                    count
                    for key, count in quota.items()
                    if key.endswith(f":{state.value}")
                )
            )
        assert tuple(observed) == class_counts
    assert tuple(
        fit + calibration
        for fit, calibration in zip(
            train_evidence_query_h4.FIT_STATE_CLASS_COUNTS,
            train_evidence_query_h4.CALIBRATION_STATE_CLASS_COUNTS,
            strict=True,
        )
    ) == train_evidence_query_h4.STATE_CLASS_COUNTS


def test_step_preflight_requires_exactly_350_full_batches_per_epoch() -> None:
    examples = tuple(
        SimpleNamespace(world_id=f"train-world-fit-{world:04d}")
        for world in range(2_800)
        for _variant in VARIANTS
    )

    batches = train_evidence_query_h4._require_exact_training_steps(
        examples, seed=20260805
    )

    assert set(batches) == {1, 2, 3}
    assert all(len(epoch_batches) == 350 for epoch_batches in batches.values())
    assert all(
        len(batch) == 32
        for epoch_batches in batches.values()
        for batch in epoch_batches
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
        train_evidence_query_h4,
        "_resolve_evidence_query_device",
        deterministic_cuda_failure,
    )
    monkeypatch.setattr(
        train_evidence_query_h4, "_require_preserved_h3_sources", unexpected
    )
    monkeypatch.setattr(
        train_evidence_query_h4, "load_h4_training_bundle", unexpected
    )

    with pytest.raises(TrainingInputError, match="deterministic CUDA"):
        train_evidence_query_h4.train_evidence_query_h4_candidate(
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
        train_evidence_query_h4,
        "_resolve_evidence_query_device",
        lambda _requested: "cpu",
    )
    monkeypatch.setattr(
        train_evidence_query_h4,
        "_require_preserved_h3_sources",
        lambda: (_ for _ in ()).throw(
            TrainingInputError("H4 preserved H3 source hash mismatch: training")
        ),
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("data was opened after source-auth failure")

    monkeypatch.setattr(
        train_evidence_query_h4, "load_h4_training_bundle", unexpected
    )

    with pytest.raises(TrainingInputError, match="source hash mismatch"):
        train_evidence_query_h4.train_evidence_query_h4_candidate(
            data_dir=tmp_path / "data",
            base_checkpoint=tmp_path / "base.pt",
            tokenizer_path=tmp_path / "tokenizer.json",
            output_dir=output,
            seed=20260805,
            device="cpu",
        )

    assert not output.exists()


def test_mutation_state_identity_uses_the_frozen_state_variants() -> None:
    assert set(VARIANTS) == {"normal", "missing", "uncertain", "conflicting"}
    assert set(surface_transfer_data.STATE_VARIANTS.values()) == {
        FieldState.MISSING,
        FieldState.UNCERTAIN,
        FieldState.CONFLICTING,
    }
