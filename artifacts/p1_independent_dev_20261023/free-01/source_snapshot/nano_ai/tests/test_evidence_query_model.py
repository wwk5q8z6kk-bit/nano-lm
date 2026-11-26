from __future__ import annotations

import pytest


def test_h3_retains_nano_trunk_with_bounded_versioned_head():
    pytest.importorskip("torch")
    from nano_ai.training.evidence_query_model import (
        ARCHITECTURE_VERSION,
        EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
        NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
        NanoEvidenceQueryPointerModel,
    )
    from nano_ai.training.pointer_model import NANO_TRUNK_PARAMETER_COUNT

    model = NanoEvidenceQueryPointerModel()

    assert model.architecture_version == ARCHITECTURE_VERSION
    assert ARCHITECTURE_VERSION == "nano_evidence_query_pointer_v1"
    assert model.trunk.config.parameter_count == NANO_TRUNK_PARAMETER_COUNT
    assert model.head_parameter_count == EVIDENCE_QUERY_HEAD_PARAMETER_COUNT == 137_861
    assert model.parameter_count == NANO_EVIDENCE_QUERY_PARAMETER_COUNT == 3_286_469
    assert model.head_parameter_count / NANO_TRUNK_PARAMETER_COUNT < 0.05


@pytest.mark.parametrize("sequence_length", [1, 7, 31])
def test_h3_forward_shapes_and_gradients(sequence_length):
    torch = pytest.importorskip("torch")
    from nano_ai.training.evidence_query_model import (
        NanoEvidenceQueryPointerModel,
    )
    from nano_ai.training.model import NANO_MODEL_CONFIG
    from nano_ai.training.pointer_model import PointerModelOutput

    model = NanoEvidenceQueryPointerModel()
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
    assert model.field_embeddings.weight.grad is not None
    assert model.evidence_query.weight.grad is not None
    assert model.shared_state_head.weight.grad is not None
    assert model.boundary_key.weight.grad is not None
    assert model.start_query.weight.grad is not None
    assert model.end_query.weight.grad is not None


def test_h3_mask_excludes_padding_from_readout_and_boundary_scores():
    torch = pytest.importorskip("torch")
    from nano_ai.training.evidence_query_model import (
        NanoEvidenceQueryPointerModel,
    )

    torch.manual_seed(17)
    model = NanoEvidenceQueryPointerModel().eval()
    first = torch.tensor([[11, 12, 13, 101, 102]], dtype=torch.long)
    second = torch.tensor([[11, 12, 13, 201, 202]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        first_output = model(first, attention_mask=attention_mask)
        second_output = model(second, attention_mask=attention_mask)

    torch.testing.assert_close(first_output.state_logits, second_output.state_logits)
    torch.testing.assert_close(
        first_output.start_logits[:, :3],
        second_output.start_logits[:, :3],
    )
    torch.testing.assert_close(
        first_output.end_logits[:, :3],
        second_output.end_logits[:, :3],
    )
    floor = torch.finfo(first_output.start_logits.dtype).min
    assert bool(torch.all(first_output.start_logits[:, 3:] == floor))
    assert bool(torch.all(first_output.end_logits[:, 3:] == floor))


def test_h3_uses_one_state_semantics_map_across_fields():
    torch = pytest.importorskip("torch")
    from nano_ai.training.evidence_query_model import (
        NanoEvidenceQueryPointerModel,
    )

    model = NanoEvidenceQueryPointerModel()
    one_representation = torch.randn(2, 1, 192)
    identical_field_states = one_representation.expand(-1, 5, -1).clone()

    logits = model.classify_states(identical_field_states)

    assert tuple(model.shared_state_head.weight.shape) == (5, 192)
    assert tuple(model.shared_state_head.bias.shape) == (5,)
    for field_index in range(1, 5):
        torch.testing.assert_close(logits[:, 0], logits[:, field_index])


def test_h3_later_context_can_change_an_early_boundary_score():
    torch = pytest.importorskip("torch")
    from nano_ai.training.evidence_query_model import (
        NanoEvidenceQueryPointerModel,
    )

    torch.manual_seed(23)
    model = NanoEvidenceQueryPointerModel().eval()
    first = torch.tensor([[31, 32, 33, 41, 42]], dtype=torch.long)
    second = torch.tensor([[31, 32, 33, 51, 52]], dtype=torch.long)

    with torch.no_grad():
        first_output = model(first)
        second_output = model(second)

    # Nano's trunk is causal: the shared prefix representation itself does not
    # see the changed suffix.  The H3 boundary query does see the complete text.
    torch.testing.assert_close(
        first_output.hidden_states[:, :3],
        second_output.hidden_states[:, :3],
    )
    assert not torch.allclose(
        first_output.start_logits[:, 0],
        second_output.start_logits[:, 0],
    )
    assert not torch.allclose(
        first_output.end_logits[:, 0],
        second_output.end_logits[:, 0],
    )


@pytest.mark.parametrize(
    "attention_mask,match",
    [
        ([[1, 1]], "shape"),
        ([[0, 0, 0], [1, 1, 1]], "at least one"),
        ([[1, 0, 1], [1, 1, 1]], "contiguous right padding"),
        ([[1, 2, 0], [1, 1, 1]], "zero or one"),
    ],
)
def test_h3_rejects_invalid_attention_masks(attention_mask, match):
    torch = pytest.importorskip("torch")
    from nano_ai.training.evidence_query_model import (
        NanoEvidenceQueryPointerModel,
    )

    model = NanoEvidenceQueryPointerModel()
    token_ids = torch.zeros((2, 3), dtype=torch.long)

    with pytest.raises(ValueError, match=match):
        model(token_ids, attention_mask=torch.tensor(attention_mask))
