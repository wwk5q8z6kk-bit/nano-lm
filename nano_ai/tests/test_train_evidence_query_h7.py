from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nano_ai.training import train_evidence_query_h6, train_evidence_query_h7
from nano_ai.training.pointer_data import (
    encode_pointer_partition,
    load_pointer_tokenizer,
)
from nano_ai.training.pointer_model import PointerModelOutput
from nano_ai.training.state_span_data import generate_split
from nano_ai.training.train_pointer import collate_pointer_batch


def _batch_and_output() -> tuple[object, PointerModelOutput]:
    tokenizer_path = Path(__file__).resolve().parents[2] / "sft" / "tokenizer.json"
    tokenizer = load_pointer_tokenizer(tokenizer_path)
    examples = generate_split("dev", worlds=5)[:4]
    records = encode_pointer_partition(tokenizer, examples, expected_split="dev")
    batch = collate_pointer_batch(records, device="cpu")
    batch_size, sequence_length = batch.token_ids.shape
    output = PointerModelOutput(
        state_logits=torch.zeros(
            batch_size, 5, 5, dtype=torch.float32, requires_grad=True
        ),
        start_logits=torch.zeros(
            batch_size,
            sequence_length,
            5,
            2,
            dtype=torch.float32,
            requires_grad=True,
        ),
        end_logits=torch.zeros(
            batch_size,
            sequence_length,
            5,
            2,
            dtype=torch.float32,
            requires_grad=True,
        ),
        hidden_states=torch.zeros(batch_size, sequence_length, 192),
    )
    return batch, output


def test_h7_changes_only_the_preregistered_objective_contract() -> None:
    shared_names = (
        "TRAINING_SEEDS",
        "BATCH_SIZE",
        "EPOCHS",
        "STEPS_PER_EPOCH",
        "STEPS_PER_SEED",
        "FIT_RECORD_COUNT",
        "FIT_WORLD_COUNT",
        "CALIBRATION_RECORD_COUNT",
        "CALIBRATION_WORLD_COUNT",
        "H5_DATASET_FILE_IDENTITY",
        "FIT_STATE_CLASS_COUNTS",
        "CALIBRATION_STATE_CLASS_COUNTS",
        "PEAK_LEARNING_RATE",
        "WARMUP_FRACTION",
        "COSINE_FLOOR",
        "WEIGHT_DECAY",
        "GRADIENT_CLIP",
        "ADAM_BETAS",
        "ADAM_EPSILON",
    )
    for name in shared_names:
        assert getattr(train_evidence_query_h7, name) == getattr(
            train_evidence_query_h6, name
        )

    assert train_evidence_query_h7.BOUNDARY_MARGIN == 1.0
    assert train_evidence_query_h7.BOUNDARY_RADIUS == 2
    assert train_evidence_query_h7.BOUNDARY_LOSS_WEIGHT == 0.25


def test_h7_objective_adds_exact_weighted_start_end_margin() -> None:
    batch, output = _batch_and_output()

    loss = train_evidence_query_h7.h7_objective(output, batch)

    assert loss.boundary.item() == pytest.approx(
        0.5 * (loss.boundary_start.item() + loss.boundary_end.item())
    )
    assert loss.total.item() == pytest.approx(
        loss.base.total.item() + 0.25 * loss.boundary.item()
    )
    assert loss.start_comparisons > 0
    assert loss.end_comparisons > 0
    loss.total.backward()
    assert output.start_logits.grad is not None
    assert output.end_logits.grad is not None


def test_h7_authenticates_h6_recipe_model_objective_and_trainer() -> None:
    paths = train_evidence_query_h7.changed_source_paths()

    assert set(paths) == {
        "boundary_objective",
        "h6_recipe",
        "model",
        "training",
    }
    assert paths["h6_recipe"] == Path(train_evidence_query_h6.__file__).resolve()
    assert all(path.is_file() for path in paths.values())


def test_h7_trainer_cannot_open_evaluation_partitions() -> None:
    source = Path(train_evidence_query_h7.__file__).read_text(encoding="utf-8")

    assert "h7_boundary_screen_data" not in source
    assert "artifacts/nano_h7/screen-v1" not in source
    assert "load_legacy_reproduction" not in source
    assert "legacy_manifest_path" not in source
    assert "write_replay_mixture_dataset" not in source
