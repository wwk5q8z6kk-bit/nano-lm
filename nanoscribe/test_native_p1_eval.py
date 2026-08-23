# Native P1 eval and weight-pull tests.
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.campaign_datasets import SMOKE_SUITE_REVISION, campaign_cases
from nanoscribe.native.config import smoke_config
from nanoscribe.native.inference import generate_target_line
from nanoscribe.native.model import build_native_model
from nanoscribe.native.p1_eval import evaluate_native_p1_suite
from nanoscribe.native.tokenize import detokenize, hash_tokens


def test_hash_tokenize_detokenize_roundtrip_chars() -> None:
    text = "ASSERTED: neck"
    ids = hash_tokens(text, 4098)
    back = detokenize(ids, 4098)
    assert len(back) == len(text)


def test_generate_target_line_smoke() -> None:
    cfg = smoke_config()
    build = build_native_model(cfg)
    line = generate_target_line(build.model, "prompt text", cfg, max_new_tokens=8)
    assert isinstance(line, str)


def test_p1_smoke_eval_cpu(tmp_path: Path) -> None:
    from dataclasses import replace

    from nanoscribe.native.checkpoint import save_checkpoint
    from nanoscribe.native.train import train_native

    cfg = replace(smoke_config(), checkpoint_dir=str(tmp_path), max_steps=1, batch_size=2)
    train_native(cfg)
    build = build_native_model(cfg)
    from nanoscribe.native.checkpoint import load_checkpoint

    load_checkpoint(cfg, build.model)
    result = evaluate_native_p1_suite(build.model, cfg, suite=SMOKE_SUITE_REVISION)
    assert result.n_cases == 3
    assert "assertion_state_correct_count" in result.suite_metrics
    assert result.suite_metrics["n_cases"] == 3


def test_smoke_suite_cases_disjoint_from_train() -> None:
    from nanoscribe.distill_train_suite import distill_train_cases

    train_ids = {c.encounter_id for c in distill_train_cases()}
    smoke_ids = {c.encounter_id for c in campaign_cases(SMOKE_SUITE_REVISION)}
    assert smoke_ids.isdisjoint(train_ids)


def test_verify_weights_script(tmp_path: Path) -> None:
    from scripts.pull_native_weights import MIN_WEIGHT_BYTES, verify_weights

    run_id = "test_run"
    bad = tmp_path / run_id / "latest.pt"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"x" * 10)
    ok, msg = verify_weights(run_id, tmp_path)
    assert not ok
    good = tmp_path / run_id / "latest.pt"
    good.write_bytes(b"x" * (MIN_WEIGHT_BYTES + 1))
    ok, msg = verify_weights(run_id, tmp_path)
    assert ok
