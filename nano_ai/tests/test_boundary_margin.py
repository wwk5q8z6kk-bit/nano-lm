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
