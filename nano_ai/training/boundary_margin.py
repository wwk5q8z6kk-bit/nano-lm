"""Local hard-negative objective for exact evidence boundaries.

This module is an H7 preflight primitive.  It does not load H6 artifacts or
change inference.  For each supervised start/end boundary it ranks the gold
token above nearby valid Patient tokens.  Ordinary pointer cross-entropy stays
responsible for global localization; this auxiliary term isolates the local
boundary precision mechanism diagnosed after terminal H6.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as torch_functional

BOUNDARY_MARGIN_OBJECTIVE_VERSION = "nano.boundary-margin.v1"


@dataclass(frozen=True, slots=True)
class BoundaryMarginLoss:
    """Mean hinge loss and its exact number of local comparisons."""

    loss: Tensor
    comparisons: int


def local_boundary_margin_loss(
    logits: Tensor,
    targets: Tensor,
    active_mask: Tensor,
    valid_token_mask: Tensor,
    *,
    margin: float = 1.0,
    radius: int = 2,
) -> BoundaryMarginLoss:
    """Rank each gold boundary above valid neighbors within ``radius`` tokens.

    ``logits`` has shape ``[batch, tokens, fields, slots]``.  ``targets`` and
    ``active_mask`` have shape ``[batch, fields, slots]``; inactive target
    values are ignored.  ``valid_token_mask`` has shape ``[batch, tokens]`` and
    should be the same Patient-only mask used by the pointer objective.
    """

    if logits.ndim != 4:
        raise ValueError("boundary logits must have shape [batch, tokens, fields, slots]")
    batch_size, token_count, field_count, slot_count = logits.shape
    expected_targets = (batch_size, field_count, slot_count)
    if tuple(targets.shape) != expected_targets or tuple(active_mask.shape) != expected_targets:
        raise ValueError("boundary targets and active mask have incompatible shapes")
    if tuple(valid_token_mask.shape) != (batch_size, token_count):
        raise ValueError("valid token mask has an incompatible shape")
    if targets.device != logits.device or active_mask.device != logits.device:
        raise ValueError("boundary tensors must share a device")
    if valid_token_mask.device != logits.device:
        raise ValueError("valid token mask and logits must share a device")
    if active_mask.dtype != torch.bool or valid_token_mask.dtype != torch.bool:
        raise TypeError("boundary masks must use torch.bool")
    if targets.dtype != torch.long:
        raise TypeError("boundary targets must use torch.long")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 1:
        raise ValueError("radius must be a positive integer")
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or margin <= 0:
        raise ValueError("margin must be positive")

    active_targets = targets[active_mask]
    if active_targets.numel() < 1:
        raise ValueError("boundary margin requires active supervision")
    if bool(torch.any(active_targets < 0) or torch.any(active_targets >= token_count)):
        raise ValueError("active boundary targets are out of bounds")

    position_logits = logits.permute(0, 2, 3, 1)
    safe_targets = targets.clamp(0, token_count - 1)
    gold_valid = valid_token_mask.gather(
        1, safe_targets.reshape(batch_size, -1)
    ).reshape_as(targets)
    if bool(torch.any(active_mask & ~gold_valid)):
        raise ValueError("active boundary target is outside the valid token mask")
    gold_logits = position_logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)

    loss_sums = logits.new_zeros(slot_count)
    comparison_counts = torch.zeros(
        slot_count, dtype=torch.long, device=logits.device
    )
    margin_tensor = logits.new_tensor(float(margin))
    for offset in range(-radius, radius + 1):
        if offset == 0:
            continue
        neighbor_targets = targets + offset
        in_bounds = (neighbor_targets >= 0) & (neighbor_targets < token_count)
        safe_neighbors = neighbor_targets.clamp(0, token_count - 1)
        neighbor_valid = valid_token_mask.gather(
            1, safe_neighbors.reshape(batch_size, -1)
        ).reshape_as(targets)
        comparisons_mask = active_mask & in_bounds & neighbor_valid
        neighbor_logits = position_logits.gather(
            -1, safe_neighbors.unsqueeze(-1)
        ).squeeze(-1)
        terms = torch_functional.relu(
            margin_tensor - (gold_logits - neighbor_logits)
        )
        loss_sums = loss_sums + (terms * comparisons_mask).sum(dim=(0, 1))
        comparison_counts = comparison_counts + comparisons_mask.sum(dim=(0, 1))

    populated_slots = comparison_counts > 0
    comparisons = int(comparison_counts.sum().item())
    if not bool(torch.any(populated_slots)):
        raise ValueError("boundary margin found no valid local negative")
    return BoundaryMarginLoss(
        loss=(loss_sums[populated_slots] / comparison_counts[populated_slots]).mean(),
        comparisons=comparisons,
    )


__all__ = [
    "BOUNDARY_MARGIN_OBJECTIVE_VERSION",
    "BoundaryMarginLoss",
    "local_boundary_margin_loss",
]
