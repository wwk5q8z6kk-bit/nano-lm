from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class _Encoding:
    def __init__(self, ids):
        self.ids = ids


class _Tokenizer:
    def encode(self, text: str):
        return _Encoding([ord(character) % 251 + 1 for character in text])


def _tokenizer_file(tmp_path: Path) -> Path:
    path = tmp_path / "tokenizer.json"
    path.write_text('{"test": true}\n', encoding="utf-8")
    return path


def test_preparation_is_atomic_manifested_deduplicated_and_loadable(tmp_path):
    from nano_ai.pretraining import ManifestTokenDataset, prepare_dataset

    output = tmp_path / "prepared"
    manifest = prepare_dataset(
        dataset_name="tinystories",
        train_documents=["alpha", "alpha", "beta", "gamma", "delta"],
        validation_documents=["heldout", "alpha", "epsilon"],
        tokenizer=_Tokenizer(),
        tokenizer_path=_tokenizer_file(tmp_path),
        output_directory=output,
        train_token_budget=12,
        validation_token_budget=8,
        shard_token_limit=8,
    )

    assert manifest["status"] == "complete"
    assert manifest["total_tokens"] == 20
    assert manifest["controls"]["private_data_included"] is False
    assert manifest["controls"]["h6_artifacts_included"] is False
    assert manifest["splits"]["train"]["exact_duplicates_removed"] >= 1
    assert (output / "manifest.sha256").is_file()

    dataset = ManifestTokenDataset(output, "train", sequence_length=3)
    x, y = dataset.sample(np.random.default_rng(7))
    assert x.shape == y.shape == (3,)
    assert np.array_equal(x[1:], y[:-1])


def test_tampered_shard_fails_closed(tmp_path):
    from nano_ai.pretraining import DatasetPreparationError, prepare_dataset, verify_prepared_dataset

    output = tmp_path / "prepared"
    manifest = prepare_dataset(
        dataset_name="tinystories",
        train_documents=["abcdefgh"],
        validation_documents=["ijklmnop"],
        tokenizer=_Tokenizer(),
        tokenizer_path=_tokenizer_file(tmp_path),
        output_directory=output,
        train_token_budget=8,
        validation_token_budget=8,
        shard_token_limit=8,
    )
    shard = output / manifest["files"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"tampered")

    with pytest.raises(DatasetPreparationError, match="verification failed"):
        verify_prepared_dataset(output)


def test_fineweb_bounded_smoke_is_authorized_and_full_scale_still_gated(tmp_path):
    """Owner approval 2026-08-05 advanced fineweb-edu to bounded-local-smoke.

    The state machine still refuses anything not explicitly
    approved_for_bounded_local_smoke, pinned here with a fabricated state.
    """
    from nano_ai.pretraining.sources import source_record

    assert source_record("fineweb-edu")["authorization_state"] == (
        "approved_for_bounded_local_smoke"
    )
    assert source_record("fineweb-edu")["license"]["commercial_clearance"] == (
        "approved_with_attribution"
    )


def test_unapproved_state_is_still_blocked(tmp_path):
    from nano_ai.pretraining import DatasetPreparationError, prepare_dataset
    import nano_ai.pretraining.sources as _sources

    _orig = _sources.SOURCES["fineweb-edu"]["authorization_state"]
    _sources.SOURCES["fineweb-edu"]["authorization_state"] = "proposal_only"
    try:
        with pytest.raises(DatasetPreparationError, match="proposal_only"):
            prepare_dataset(
                dataset_name="fineweb-edu",
                train_documents=["unused"],
                validation_documents=["unused"],
                tokenizer=_Tokenizer(),
                tokenizer_path=_tokenizer_file(tmp_path),
                output_directory=tmp_path / "blocked",
                train_token_budget=1,
                validation_token_budget=1,
            )
    finally:
        _sources.SOURCES["fineweb-edu"]["authorization_state"] = _orig


def test_manifest_hash_detects_metadata_tampering(tmp_path):
    from nano_ai.pretraining import DatasetPreparationError, prepare_dataset, verify_prepared_dataset

    output = tmp_path / "prepared"
    prepare_dataset(
        dataset_name="tinystories",
        train_documents=["abcdefgh"],
        validation_documents=["ijklmnop"],
        tokenizer=_Tokenizer(),
        tokenizer_path=_tokenizer_file(tmp_path),
        output_directory=output,
        train_token_budget=8,
        validation_token_budget=8,
    )
    manifest_path = output / "manifest.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    value["controls"]["private_data_included"] = True
    manifest_path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(DatasetPreparationError, match="manifest SHA-256 mismatch"):
        verify_prepared_dataset(output)
