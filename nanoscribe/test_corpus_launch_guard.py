"""Test corpus launch guard blocks unit fixture on GPU ranking."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.native.corpus.registry import (
    CorpusLaunchGuardError,
    assert_corpus_launch_allowed,
)


def test_unit_fixture_allowed_for_trainer_smoke_on_cpu() -> None:
    assert_corpus_launch_allowed(
        "artifacts/campaign/p1_distill_train_v1.json",
        purpose="trainer_smoke",
        cpu_smoke=True,
    )


def test_unit_fixture_blocks_gpu_architecture_screening() -> None:
    with pytest.raises(CorpusLaunchGuardError, match="NATIVE_UNIT_OVERFIT_FIXTURE"):
        assert_corpus_launch_allowed(
            "artifacts/campaign/p1_distill_train_v1.json",
            purpose="architecture_screening",
            cpu_smoke=False,
        )


def test_legacy_dataset_id_resolves() -> None:
    assert_corpus_launch_allowed(
        "p1_distill_train_v1",
        purpose="qlora_canary_fixture",
        cpu_smoke=False,
    )
