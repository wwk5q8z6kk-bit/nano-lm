from __future__ import annotations

from pathlib import Path

import pytest


def _frozen_checkpoint() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "checkpoints"
        / "anchors"
        / "nano_v01_scribe.pt"
    )


def test_pointer_model_retains_exact_nano_trunk_and_bounded_head_geometry():
    pytest.importorskip("torch")
    from nano_ai.training.pointer_model import (
        NANO_POINTER_PARAMETER_COUNT,
        NANO_TRUNK_PARAMETER_COUNT,
        POINTER_HEAD_PARAMETER_COUNT,
        NanoPointerModel,
    )

    model = NanoPointerModel()

    assert model.trunk.config.parameter_count == NANO_TRUNK_PARAMETER_COUNT
    assert model.head_parameter_count == POINTER_HEAD_PARAMETER_COUNT == 8_665
    assert model.parameter_count == NANO_POINTER_PARAMETER_COUNT == 3_157_273
    assert tuple(model.state_head.weight.shape) == (25, 192)
    assert tuple(model.state_head.bias.shape) == (25,)
    assert tuple(model.start_head.weight.shape) == (10, 192)
    assert model.start_head.bias is None
    assert tuple(model.end_head.weight.shape) == (10, 192)
    assert model.end_head.bias is None


@pytest.mark.parametrize("sequence_length", [1, 7, 31])
def test_pointer_forward_exposes_typed_shape_checked_outputs(sequence_length):
    torch = pytest.importorskip("torch")
    from nano_ai.training.model import NANO_MODEL_CONFIG
    from nano_ai.training.pointer_model import NanoPointerModel, PointerModelOutput

    model = NanoPointerModel()
    token_ids = torch.randint(
        NANO_MODEL_CONFIG.vocabulary_size,
        (2, sequence_length),
        dtype=torch.long,
    )

    output = model(token_ids)

    assert isinstance(output, PointerModelOutput)
    assert tuple(output.hidden_states.shape) == (2, sequence_length, 192)
    assert tuple(output.state_logits.shape) == (2, 5, 5)
    assert tuple(output.start_logits.shape) == (2, sequence_length, 5, 2)
    assert tuple(output.end_logits.shape) == (2, sequence_length, 5, 2)
    (
        output.state_logits.mean()
        + output.start_logits.mean()
        + output.end_logits.mean()
    ).backward()
    assert model.trunk.emb.weight.grad is not None
    assert model.state_head.weight.grad is not None
    assert model.start_head.weight.grad is not None
    assert model.end_head.weight.grad is not None


def test_right_padding_mask_selects_each_examples_terminal_token():
    torch = pytest.importorskip("torch")
    from nano_ai.training.pointer_model import NanoPointerModel

    model = NanoPointerModel()
    token_ids = torch.randint(4_098, (2, 5), dtype=torch.long)
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]],
        dtype=torch.long,
    )

    output = model(token_ids, attention_mask=attention_mask)
    expected_terminal = torch.stack(
        (output.hidden_states[0, 2], output.hidden_states[1, 4])
    )
    expected_state_logits = model.state_head(expected_terminal).reshape(2, 5, 5)

    torch.testing.assert_close(output.state_logits, expected_state_logits)


@pytest.mark.parametrize(
    "attention_mask,match",
    [
        ([[1, 1]], "shape"),
        ([[0, 0, 0], [1, 1, 1]], "at least one"),
        ([[1, 0, 1], [1, 1, 1]], "contiguous right padding"),
        ([[1, 2, 0], [1, 1, 1]], "zero or one"),
    ],
)
def test_pointer_forward_rejects_invalid_attention_masks(attention_mask, match):
    torch = pytest.importorskip("torch")
    from nano_ai.training.pointer_model import NanoPointerModel

    model = NanoPointerModel()
    token_ids = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(ValueError, match=match):
        model(token_ids, attention_mask=torch.tensor(attention_mask))


def test_pointer_output_rejects_inconsistent_tensor_shapes():
    torch = pytest.importorskip("torch")
    from nano_ai.training.pointer_model import PointerModelOutput

    with pytest.raises(ValueError, match="state_logits"):
        PointerModelOutput(
            state_logits=torch.zeros((2, 5, 4)),
            start_logits=torch.zeros((2, 3, 5, 2)),
            end_logits=torch.zeros((2, 3, 5, 2)),
            hidden_states=torch.zeros((2, 3, 192)),
        )


def test_base_state_dict_load_is_strict_and_unprefixed():
    pytest.importorskip("torch")
    from nano_ai.training.model import NanoGPT
    from nano_ai.training.pointer_model import NanoPointerModel

    base = NanoGPT()
    model = NanoPointerModel()
    state_dict = base.state_dict()

    model.load_base_state_dict(state_dict)

    assert model.trunk.state_dict().keys() == state_dict.keys()
    broken_state_dict = dict(state_dict)
    broken_state_dict.pop("nf.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_base_state_dict(broken_state_dict)


def test_registered_frozen_checkpoint_loads_when_available():
    torch = pytest.importorskip("torch")
    from nano_ai.training.pointer_model import NanoPointerModel

    checkpoint = _frozen_checkpoint()
    if not checkpoint.is_file():
        pytest.skip("frozen release checkpoint is not present in this checkout")

    model = NanoPointerModel.from_frozen_base(checkpoint)

    frozen_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.testing.assert_close(model.trunk.emb.weight, frozen_state["emb.weight"])


def test_registered_frozen_checkpoint_loader_rejects_wrong_hash(tmp_path):
    from nano_ai.training.model import FrozenNanoCheckpointError
    from nano_ai.training.pointer_model import NanoPointerModel

    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"not Nano v0.1")

    with pytest.raises(FrozenNanoCheckpointError, match="SHA-256 mismatch"):
        NanoPointerModel.from_frozen_base(checkpoint)
