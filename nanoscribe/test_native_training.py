# Native training acceptance tests.
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.campaign_datasets import partition_cases, validate_campaign_partitions
from nanoscribe.dataset_leakage import DatasetPartition, validate_partition_disjointness
from nanoscribe.distill_train_suite import distill_train_cases
from nanoscribe.native.config import NativeVariant, config_for_run, default_loss_weights, smoke_config
from nanoscribe.native.data import NativeBatchIterator, export_distill_train_json, load_train_examples
from nanoscribe.native.checkpoint import load_checkpoint, save_checkpoint
from nanoscribe.native.evaluate import dev_cases_from_train, evaluate_native_model
from nanoscribe.native.factorial import FACTORIAL_CELLS
from nanoscribe.native.model import build_native_model, estimate_param_count
from nanoscribe.native.train import cpu_smoke_train, train_native
from nanoscribe.screening_suite import screening_core_cases


def test_native_a_config_defaults() -> None:
    cfg = config_for_run("C_s0", cpu_smoke=True)
    assert cfg.variant == NativeVariant.NATIVE_A
    assert cfg.loss_weights.evidence_align == 0.0


def test_native_b_config_evidence_weights() -> None:
    cfg = config_for_run("A_s0", cpu_smoke=True)
    assert cfg.variant == NativeVariant.NATIVE_B
    assert cfg.loss_weights.evidence_align > 0.0


def test_model_param_count_in_budget() -> None:
    cfg = config_for_run("A_s0", cpu_smoke=True)
    count = estimate_param_count(cfg)
    assert 5_000_000 <= count <= 120_000_000


def test_cpu_smoke_forward() -> None:
    cfg = smoke_config()
    build = build_native_model(cfg)
    import torch

    x = torch.randint(0, cfg.vocab_size, (1, 8))
    logits = build.model(x)
    assert logits.shape[-1] == cfg.vocab_size


def test_data_loader_batches() -> None:
    examples = load_train_examples()
    it = NativeBatchIterator(examples, batch_size=4, seed=0)
    batches = list(it)
    assert batches
    assert all(len(batch) <= 4 for batch in batches)


def test_loss_finite_cpu_smoke() -> None:
    result = cpu_smoke_train(variant="native_a")
    assert result.steps_completed >= 1
    assert result.final_loss >= 0.0


def test_train_one_step_cpu() -> None:
    from dataclasses import replace

    cfg = replace(smoke_config(), max_steps=1, batch_size=2)
    result = train_native(cfg)
    assert result.steps_completed == 1


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    from dataclasses import replace

    cfg = replace(smoke_config(), checkpoint_dir=str(tmp_path))
    build = build_native_model(cfg)
    import torch

    opt = torch.optim.AdamW(build.model.parameters(), lr=1e-3)
    path = save_checkpoint(build.model, cfg, step=1, optimizer=opt)
    assert path.is_file()
    load_checkpoint(cfg, build.model, step=1, optimizer=opt)


def test_resume_training(tmp_path: Path) -> None:
    from dataclasses import replace

    cfg = replace(smoke_config(), checkpoint_dir=str(tmp_path), max_steps=2)
    first = train_native(cfg)
    second = train_native(cfg, resume=True)
    assert second.steps_completed >= first.steps_completed


def test_evaluate_dev_set() -> None:
    cfg = smoke_config()
    build = build_native_model(cfg)
    result = evaluate_native_model(build.model, cfg)
    assert result.n_examples > 0


def test_native_a_vs_b_differ() -> None:
    a = default_loss_weights(NativeVariant.NATIVE_A)
    b = default_loss_weights(NativeVariant.NATIVE_B)
    assert a.evidence_align != b.evidence_align


def test_disjoint_train_no_eval_leak() -> None:
    train_ids = {case.encounter_id for case in distill_train_cases()}
    eval_ids = {case.encounter_id for case in screening_core_cases()}
    assert train_ids.isdisjoint(eval_ids)


def test_leakage_validator_passes_partitions() -> None:
    validate_campaign_partitions()


def test_leakage_validator_fails_on_overlap() -> None:
    train = partition_cases(DatasetPartition.TRAIN)
    eval_cases = partition_cases(DatasetPartition.FROZEN_SCREENING_EVAL)
    violations = validate_partition_disjointness(
        {
            DatasetPartition.TRAIN: train,
            DatasetPartition.FROZEN_SCREENING_EVAL: train[:1] + eval_cases,
        }
    )
    assert violations


def test_factorial_manifest_ready() -> None:
    assert len(FACTORIAL_CELLS) == 4
    assert all(len(cell.seeds) == 2 for cell in FACTORIAL_CELLS)


def test_no_cuda_required_smoke() -> None:
    result = cpu_smoke_train()
    assert result.device == "cpu"


def test_trainer_manifest_status() -> None:
    from nanoscribe.native.trainer import trainer_manifest

    manifest = trainer_manifest()
    assert manifest["status"] == "READY_FOR_GPU_LAUNCH"
    assert manifest["n_runs"] == 8


def test_export_distill_train_json(tmp_path: Path) -> None:
    payload = export_distill_train_json(tmp_path / "train.json")
    assert payload["revision"] == "p1_distill_train_v1"
    assert payload["n_cases"] == 96
