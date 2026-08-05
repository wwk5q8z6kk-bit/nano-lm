from __future__ import annotations

import inspect

import pytest


def test_h6_is_an_exact_one_parameter_tensor_extension_of_h5():
    torch = pytest.importorskip("torch")
    from nano_ai.training.evidence_query_model import (
        NanoEvidenceQueryPointerModel,
    )
    from nano_ai.training.pointer_model import NANO_TRUNK_PARAMETER_COUNT
    from nano_ai.training.state_conditioned_evidence_query_model import (
        ARCHITECTURE_VERSION,
        EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6,
        NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6,
        STATE_CONDITIONING_PARAMETER_COUNT,
        NanoStateConditionedEvidenceQueryPointerModel,
    )

    torch.manual_seed(17)
    h5 = NanoEvidenceQueryPointerModel()
    torch.manual_seed(17)
    h6 = NanoStateConditionedEvidenceQueryPointerModel()

    h5_state = h5.state_dict()
    h6_state = h6.state_dict()
    assert set(h6_state).difference(h5_state) == {"state_boundary_query_offsets"}
    assert all(torch.equal(value, h6_state[name]) for name, value in h5_state.items())
    assert bool(torch.all(h6.state_boundary_query_offsets == 0))

    assert ARCHITECTURE_VERSION == "nano_evidence_query_state_boundary_residual_v1"
    assert h6.architecture_version == ARCHITECTURE_VERSION
    assert h6.trunk.config.parameter_count == NANO_TRUNK_PARAMETER_COUNT
    assert STATE_CONDITIONING_PARAMETER_COUNT == 640
    assert tuple(h6.state_boundary_query_offsets.shape) == (5, 2, 64)
    assert h6.head_parameter_count == EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6
    assert EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6 == 138_501
    assert h6.parameter_count == NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6
    assert NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6 == 3_287_109


def test_h6_constructor_consumes_no_rng_beyond_h5():
    torch = pytest.importorskip("torch")
    from nano_ai.training.evidence_query_model import NanoEvidenceQueryPointerModel
    from nano_ai.training.state_conditioned_evidence_query_model import (
        NanoStateConditionedEvidenceQueryPointerModel,
    )

    torch.manual_seed(19)
    NanoEvidenceQueryPointerModel()
    h5_rng_state = torch.random.get_rng_state()

    torch.manual_seed(19)
    NanoStateConditionedEvidenceQueryPointerModel()
    h6_rng_state = torch.random.get_rng_state()

    assert torch.equal(h6_rng_state, h5_rng_state)


def test_h6_zero_offsets_are_functionally_identical_to_h5():
    torch = pytest.importorskip("torch")
    from nano_ai.training.evidence_query_model import NanoEvidenceQueryPointerModel
    from nano_ai.training.state_conditioned_evidence_query_model import (
        NanoStateConditionedEvidenceQueryPointerModel,
    )

    torch.manual_seed(23)
    h5 = NanoEvidenceQueryPointerModel().eval()
    torch.manual_seed(23)
    h6 = NanoStateConditionedEvidenceQueryPointerModel().eval()
    token_ids = torch.tensor([[11, 12, 13, 101, 102]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        expected = h5(token_ids, attention_mask=attention_mask)
        observed = h6(token_ids, attention_mask=attention_mask)

    assert torch.equal(observed.hidden_states, expected.hidden_states)
    assert torch.equal(observed.state_logits, expected.state_logits)
    assert torch.equal(observed.start_logits, expected.start_logits)
    assert torch.equal(observed.end_logits, expected.end_logits)


def test_h6_forced_posteriors_select_boundary_specific_query_offsets():
    torch = pytest.importorskip("torch")
    from nano_ai.training.state_conditioned_evidence_query_model import (
        NanoStateConditionedEvidenceQueryPointerModel,
    )

    model = NanoStateConditionedEvidenceQueryPointerModel()
    with torch.no_grad():
        values = torch.arange(5 * 2 * 64, dtype=torch.float32).reshape(5, 2, 64)
        model.state_boundary_query_offsets.copy_(values)

    selected_states = torch.tensor([4, 2, 0, 3, 1])
    state_logits = torch.full((1, 5, 5), -torch.inf)
    state_logits[0, torch.arange(5), selected_states] = 0
    observed = model._state_conditioned_query_offsets(state_logits)
    expected = values[selected_states][None, :, :, :]

    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert not torch.equal(observed[:, :, 0], observed[:, :, 1])


def test_h6_boundary_loss_trains_offsets_but_not_state_head_through_new_path():
    torch = pytest.importorskip("torch")
    from nano_ai.training.state_conditioned_evidence_query_model import (
        NanoStateConditionedEvidenceQueryPointerModel,
    )

    torch.manual_seed(31)
    model = NanoStateConditionedEvidenceQueryPointerModel()
    output = model(torch.randint(4098, (2, 7), dtype=torch.long))

    (output.start_logits.square().mean() + output.end_logits.square().mean()).backward()

    assert model.state_boundary_query_offsets.grad is not None
    assert bool(torch.any(model.state_boundary_query_offsets.grad != 0))
    assert model.shared_state_head.weight.grad is None
    assert model.shared_state_head.bias.grad is None


def test_h6_forward_preserves_output_shapes_and_padding_mask():
    torch = pytest.importorskip("torch")
    from nano_ai.training.state_conditioned_evidence_query_model import (
        NanoStateConditionedEvidenceQueryPointerModel,
    )

    model = NanoStateConditionedEvidenceQueryPointerModel().eval()
    token_ids = torch.tensor([[11, 12, 13, 101, 102]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    with torch.no_grad():
        output = model(token_ids, attention_mask=attention_mask)

    assert tuple(output.state_logits.shape) == (1, 5, 5)
    assert tuple(output.start_logits.shape) == (1, 5, 5, 2)
    assert tuple(output.end_logits.shape) == (1, 5, 5, 2)
    floor = torch.finfo(output.start_logits.dtype).min
    assert bool(torch.all(output.start_logits[:, 3:] == floor))
    assert bool(torch.all(output.end_logits[:, 3:] == floor))
    assert tuple(inspect.signature(model.forward).parameters) == (
        "token_ids",
        "attention_mask",
    )


@pytest.mark.parametrize(
    "slots,states,match",
    [
        ((1, 5, 192), (1, 5, 5), "slot_states"),
        ((1, 5, 2, 192), (1, 5), "state_logits"),
        ((2, 5, 2, 192), (1, 5, 5), "batch sizes"),
    ],
)
def test_h6_rejects_invalid_coupling_shapes(slots, states, match):
    torch = pytest.importorskip("torch")
    from nano_ai.training.state_conditioned_evidence_query_model import (
        NanoStateConditionedEvidenceQueryPointerModel,
    )

    model = NanoStateConditionedEvidenceQueryPointerModel()
    batch = slots[0]
    hidden_states = torch.zeros((batch, 3, 192))
    valid_mask = torch.ones((batch, 3), dtype=torch.bool)
    with pytest.raises(ValueError, match=match):
        model._state_conditioned_boundary_logits(
            hidden_states,
            torch.zeros(slots),
            torch.zeros(states),
            valid_mask,
        )
