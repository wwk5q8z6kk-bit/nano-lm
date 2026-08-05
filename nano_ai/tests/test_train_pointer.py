from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.nn import functional as torch_functional

from nano_ai.training import train_pointer
from nano_ai.training.pointer_data import (
    STATE_ORDER,
    encode_pointer_partition,
    load_pointer_tokenizer,
)
from nano_ai.training.pointer_model import PointerModelOutput
from nano_ai.training.state_span_data import generate_split
from nano_ai.training.train_pointer import (
    STATE_CLASS_WEIGHTS,
    TrainingInputError,
    collate_pointer_batch,
    pointer_objective,
)


def _records():
    tokenizer_path = Path(__file__).resolve().parents[2] / "sft" / "tokenizer.json"
    tokenizer = load_pointer_tokenizer(tokenizer_path)
    examples = generate_split("dev", worlds=5)[:4]
    return encode_pointer_partition(tokenizer, examples, expected_split="dev")


def _zero_output(batch) -> PointerModelOutput:
    batch_size, sequence_length = batch.token_ids.shape
    return PointerModelOutput(
        state_logits=torch.zeros(batch_size, 5, len(STATE_ORDER)),
        start_logits=torch.zeros(batch_size, sequence_length, 5, 2),
        end_logits=torch.zeros(batch_size, sequence_length, 5, 2),
        hidden_states=torch.zeros(batch_size, sequence_length, 192),
    )


def test_collate_right_pads_without_moving_pointer_labels() -> None:
    records = _records()
    batch = collate_pointer_batch(records, device="cpu")

    assert batch.token_ids.shape[0] == len(records)
    assert batch.state_labels.shape == (len(records), 5)
    assert batch.span_starts.shape == (len(records), 5, 2)
    assert batch.span_mask.shape == (len(records), 5, 2)
    for row, record in enumerate(records):
        length = len(record.token_ids)
        assert batch.token_ids[row, :length].tolist() == list(record.token_ids)
        assert batch.attention_mask[row, :length].all()
        assert not batch.attention_mask[row, length:].any()
        assert batch.pointer_mask[row, :length].tolist() == list(record.pointer_mask)
        assert not batch.pointer_mask[row, length:].any()


def test_pointer_objective_masks_non_patient_logits_and_backpropagates() -> None:
    records = _records()
    batch = collate_pointer_batch(records, device="cpu")
    batch_size, sequence_length = batch.token_ids.shape
    state_logits = torch.full(
        (batch_size, 5, len(STATE_ORDER)), -4.0, requires_grad=True
    )
    start_logits = torch.full(
        (batch_size, sequence_length, 5, 2), -4.0, requires_grad=True
    )
    end_logits = torch.full_like(start_logits, -4.0, requires_grad=True)
    with torch.no_grad():
        for row in range(batch_size):
            for field in range(5):
                state_logits[row, field, batch.state_labels[row, field]] = 4.0
                for slot in range(2):
                    if batch.span_mask[row, field, slot]:
                        start = batch.span_starts[row, field, slot]
                        end = batch.span_ends[row, field, slot]
                        start_logits[row, start, field, slot] = 4.0
                        end_logits[row, end, field, slot] = 4.0
        start_logits.masked_fill_(~batch.pointer_mask[:, :, None, None], 1_000.0)
        end_logits.masked_fill_(~batch.pointer_mask[:, :, None, None], 1_000.0)

    loss = pointer_objective(
        PointerModelOutput(
            state_logits=state_logits,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=torch.zeros(batch_size, sequence_length, 192),
        ),
        batch,
    )
    loss.total.backward()

    assert math.isfinite(float(loss.total.item()))
    assert loss.total.item() < 0.1
    assert loss.state_count == batch_size * 5
    assert loss.pointer_count == int(batch.span_mask.sum().item())
    assert start_logits.grad is not None
    assert end_logits.grad is not None
    assert not start_logits.grad[
        ~batch.pointer_mask[:, :, None, None].expand_as(start_logits)
    ].any()
    assert not end_logits.grad[
        ~batch.pointer_mask[:, :, None, None].expand_as(end_logits)
    ].any()


def test_pointer_objective_rejects_active_label_outside_patient_mask() -> None:
    records = _records()
    batch = collate_pointer_batch(records, device="cpu")
    invalid_starts = batch.span_starts.clone()
    row, field, slot = batch.span_mask.nonzero()[0].tolist()
    invalid_starts[row, field, slot] = 0
    invalid = replace(batch, span_starts=invalid_starts)
    output = _zero_output(batch)

    with pytest.raises(TrainingInputError, match="escape the Patient token mask"):
        pointer_objective(output, invalid)


def test_pointer_objective_uses_frozen_inverse_frequency_state_weights() -> None:
    batch = collate_pointer_batch(_records(), device="cpu")
    torch.manual_seed(7)
    output = _zero_output(batch)
    state_logits = torch.randn_like(output.state_logits)
    output = replace(output, state_logits=state_logits)

    loss = pointer_objective(output, batch)
    weights = torch.tensor(STATE_CLASS_WEIGHTS, dtype=state_logits.dtype)
    expected = torch_functional.cross_entropy(
        state_logits.reshape(-1, len(STATE_ORDER)),
        batch.state_labels.reshape(-1),
        weight=weights,
    )

    assert loss.state.item() == pytest.approx(expected.item())
    assert loss.state_weight_sum == pytest.approx(
        weights[batch.state_labels.reshape(-1)].sum().item()
    )


def test_pointer_objective_rejects_state_arity_mismatch() -> None:
    batch = collate_pointer_batch(_records(), device="cpu")
    row, field = next(
        (row, field)
        for row in range(batch.span_mask.shape[0])
        for field in range(batch.span_mask.shape[1])
        if int(batch.span_mask[row, field].sum().item()) == 1
    )
    invalid_states = batch.state_labels.clone()
    invalid_states[row, field] = STATE_ORDER.index(
        next(state for state in STATE_ORDER if state.value == "missing")
    )
    invalid = replace(batch, state_labels=invalid_states)

    with pytest.raises(TrainingInputError, match="pointer-slot arity"):
        pointer_objective(_zero_output(batch), invalid)


def test_pointer_objective_rejects_a_mask_hole_inside_an_active_span() -> None:
    tokenizer_path = Path(__file__).resolve().parents[2] / "sft" / "tokenizer.json"
    tokenizer = load_pointer_tokenizer(tokenizer_path)
    records = encode_pointer_partition(
        tokenizer,
        generate_split("dev", worlds=5),
        expected_split="dev",
    )
    batch = collate_pointer_batch(records, device="cpu")
    row, field, slot = next(
        (row, field, slot)
        for row in range(batch.span_mask.shape[0])
        for field in range(batch.span_mask.shape[1])
        for slot in range(batch.span_mask.shape[2])
        if batch.span_mask[row, field, slot]
        and batch.span_ends[row, field, slot] - batch.span_starts[row, field, slot] >= 2
    )
    invalid_mask = batch.pointer_mask.clone()
    start = int(batch.span_starts[row, field, slot].item())
    end = int(batch.span_ends[row, field, slot].item())
    invalid_mask[row, start + (end - start) // 2] = False
    invalid = replace(batch, pointer_mask=invalid_mask)

    with pytest.raises(TrainingInputError, match="escape the Patient token mask"):
        pointer_objective(_zero_output(batch), invalid)


@pytest.mark.parametrize("configuration", (":4096:8", ":16:8"))
def test_pointer_cuda_requires_deterministic_cublas_configuration(
    monkeypatch: pytest.MonkeyPatch,
    configuration: str,
) -> None:
    monkeypatch.setattr(train_pointer, "_resolve_base_device", lambda _device: "cuda")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with pytest.raises(TrainingInputError, match="deterministic CUDA requires"):
        train_pointer._resolve_pointer_device("cuda")

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")
    with pytest.raises(TrainingInputError, match="deterministic CUDA requires"):
        train_pointer._resolve_pointer_device("cuda")

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", configuration)
    assert train_pointer._resolve_pointer_device("cuda") == "cuda"


@pytest.mark.parametrize("resolved", ("cpu", "mps"))
def test_pointer_non_cuda_device_does_not_require_cublas_configuration(
    monkeypatch: pytest.MonkeyPatch,
    resolved: str,
) -> None:
    monkeypatch.setattr(train_pointer, "_resolve_base_device", lambda _device: resolved)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    assert train_pointer._resolve_pointer_device(resolved) == resolved


def test_pointer_cuda_preflight_fails_before_data_model_or_output_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "candidate"
    monkeypatch.setattr(train_pointer, "_resolve_base_device", lambda _device: "cuda")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("CUDA preflight did not fail first")

    monkeypatch.setattr(train_pointer, "load_training_bundle", unexpected)
    monkeypatch.setattr(train_pointer.NanoPointerModel, "from_frozen_base", unexpected)

    with pytest.raises(TrainingInputError, match="deterministic CUDA requires"):
        train_pointer.train_pointer_candidate(
            data_dir=tmp_path / "data",
            base_checkpoint=tmp_path / "base.pt",
            tokenizer_path=tmp_path / "tokenizer.json",
            output_dir=output,
            seed=20260805,
            device="cuda",
        )

    assert not output.exists()
