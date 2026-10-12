from __future__ import annotations

import pytest
import torch

from nano_ai.training.boundary_margin import local_boundary_margin_loss


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.zeros((1, 6, 1, 1), dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[[3]]], dtype=torch.long)
    active = torch.tensor([[[True]]])
    valid = torch.tensor([[False, True, True, True, True, True]])
    return logits, targets, active, valid


def _reference_boundary_margin_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    active: torch.Tensor,
    valid: torch.Tensor,
    *,
    margin: float,
    radius: int,
) -> tuple[torch.Tensor, int]:
    """Small direct implementation used only to prove vectorized equivalence."""

    slot_terms: list[list[torch.Tensor]] = [
        [] for _ in range(logits.shape[-1])
    ]
    for batch_index, field_index, slot_index in active.nonzero(as_tuple=False).tolist():
        target = int(targets[batch_index, field_index, slot_index])
        gold = logits[batch_index, target, field_index, slot_index]
        for neighbor in range(
            max(0, target - radius), min(logits.shape[1], target + radius + 1)
        ):
            if neighbor == target or not bool(valid[batch_index, neighbor]):
                continue
            candidate = logits[batch_index, neighbor, field_index, slot_index]
            slot_terms[slot_index].append(
                torch.relu(logits.new_tensor(margin) - (gold - candidate))
            )
    populated = [torch.stack(terms).mean() for terms in slot_terms if terms]
    return torch.stack(populated).mean(), sum(len(terms) for terms in slot_terms)


def test_boundary_margin_compares_only_local_valid_neighbors() -> None:
    logits, targets, active, valid = _fixture()
    result = local_boundary_margin_loss(
        logits,
        targets,
        active,
        valid,
        margin=1.0,
        radius=2,
    )

    assert result.comparisons == 4
    assert result.loss.item() == pytest.approx(1.0)
    result.loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 3, 0, 0].item() < 0
    assert logits.grad[0, 1, 0, 0].item() > 0
    assert logits.grad[0, 5, 0, 0].item() > 0


def test_boundary_margin_is_zero_once_gold_clears_margin() -> None:
    logits, targets, active, valid = _fixture()
    with torch.no_grad():
        logits[0, 3, 0, 0] = 2.0

    result = local_boundary_margin_loss(logits, targets, active, valid, radius=2)

    assert result.comparisons == 4
    assert result.loss.item() == pytest.approx(0.0)


def test_boundary_margin_weights_start_and_end_slots_equally() -> None:
    logits = torch.zeros((1, 5, 1, 2), dtype=torch.float32)
    targets = torch.tensor([[[0, 2]]], dtype=torch.long)
    active = torch.tensor([[[True, True]]])
    valid = torch.tensor([[True, True, True, True, True]])
    logits[0, 0, 0, 0] = 1.0  # start-slot loss is zero with two neighbors
    logits[0, 2, 0, 1] = 0.0  # end-slot loss is one with four neighbors

    result = local_boundary_margin_loss(
        logits,
        targets,
        active,
        valid,
        margin=1.0,
        radius=2,
    )

    assert result.comparisons == 6
    assert result.loss.item() == pytest.approx(0.5)


def test_boundary_margin_rejects_target_outside_patient_mask() -> None:
    logits, targets, active, valid = _fixture()
    valid[0, 3] = False

    with pytest.raises(ValueError, match="outside the valid token mask"):
        local_boundary_margin_loss(logits, targets, active, valid)


def test_boundary_margin_requires_a_local_negative() -> None:
    logits = torch.zeros((1, 1, 1, 1))
    targets = torch.tensor([[[0]]], dtype=torch.long)
    active = torch.tensor([[[True]]])
    valid = torch.tensor([[True]])

    with pytest.raises(ValueError, match="no valid local negative"):
        local_boundary_margin_loss(logits, targets, active, valid)


def test_vectorized_boundary_margin_matches_direct_reference() -> None:
    generator = torch.Generator().manual_seed(712)
    logits = torch.randn(
        (3, 11, 4, 2), generator=generator, dtype=torch.float64, requires_grad=True
    )
    targets = torch.randint(2, 9, (3, 4, 2), generator=generator)
    active = torch.rand((3, 4, 2), generator=generator) > 0.3
    valid = torch.rand((3, 11), generator=generator) > 0.25
    valid.scatter_(1, targets.reshape(3, -1), True)

    observed = local_boundary_margin_loss(
        logits, targets, active, valid, margin=0.8, radius=2
    )
    reference, reference_comparisons = _reference_boundary_margin_loss(
        logits, targets, active, valid, margin=0.8, radius=2
    )

    assert observed.comparisons == reference_comparisons
    assert observed.loss.item() == pytest.approx(reference.item())
    observed_gradient = torch.autograd.grad(observed.loss, logits, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(reference, logits)[0]
    torch.testing.assert_close(observed_gradient, reference_gradient)
