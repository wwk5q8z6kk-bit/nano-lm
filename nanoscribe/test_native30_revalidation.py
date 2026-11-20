"""Tests for native30 revalidation orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoscribe.campaign.native30_revalidation import (
    interleaved_run_ids,
    import_train_results,
    preflight_arms,
    run_revalidation_wave,
    verify_corpus,
)


def test_interleaved_run_ids_count_and_order() -> None:
    ids = interleaved_run_ids()
    assert len(ids) == 9
    assert ids[0] == "reval30_decoder_control_s0"
    assert ids[1] == "reval30_evidence_bottleneck_s0"
    assert ids[2] == "reval30_span_port_s0"
    assert ids[3] == "reval30_decoder_control_s1"


def test_verify_corpus_matches_manifest() -> None:
    info = verify_corpus()
    assert info["train_rows"] == 19194
    assert len(info["content_hash"]) == 64


def test_preflight_all_arms() -> None:
    preflight_arms(interleaved_run_ids())


def test_smoke_wave_writes_train_and_summary(tmp_path: Path, monkeypatch) -> None:
    # Isolate the checkpoint tree. Without this the smoke wave trains the REAL
    # reval30_decoder_control_s0 for 2 steps and overwrites its latest.pt in
    # artifacts/native_checkpoints/ — so merely running the test suite destroyed
    # a wave artifact (recoverable only because step_001800.pt survives beside
    # it). Observed twice on 2026-08-25.
    monkeypatch.setenv("NANO_NATIVE_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    # Use the checked-in trainer fixture for this two-step smoke. The full
    # screen partition is generated separately for research runs.
    from nanoscribe.native import config

    monkeypatch.setattr(config, "REVALIDATION_DATASET", "artifacts/campaign/p1_distill_train_v1.json")
    results = tmp_path / "reval_results"
    out = tmp_path / "summary.json"
    failed, _ = run_revalidation_wave(
        results,
        ("reval30_decoder_control_s0",),
        max_steps=2,
        suite="p1_contract_smoke_v1",
        eval_cpu=True,
        cpu_smoke=True,
    )
    assert not failed
    train_marker = results / "reval30_decoder_control_s0_train.json"
    assert train_marker.is_file()
    payload = json.loads(train_marker.read_text())
    assert payload["smoke"] and payload["device"] == "cpu"
    assert payload["steps_completed"] == 2

    imported, expected = import_train_results(
        results, out, surface="test", run_ids=("reval30_decoder_control_s0",)
    )
    assert imported == 1
    assert expected == 1
    summary = json.loads(out.read_text())
    assert summary["runs_imported"] == 1
