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

    terms_by_slot: list[list[Tensor]] = [[] for _ in range(slot_count)]
    active_indices = active_mask.nonzero(as_tuple=False)
    for batch_index, field_index, slot_index in active_indices.tolist():
        target = int(targets[batch_index, field_index, slot_index].item())
        if not bool(valid_token_mask[batch_index, target]):
            raise ValueError("active boundary target is outside the valid token mask")
        gold_logit = logits[batch_index, target, field_index, slot_index]
        lower = max(0, target - radius)
        upper = min(token_count, target + radius + 1)
        for neighbor in range(lower, upper):
            if neighbor == target or not bool(valid_token_mask[batch_index, neighbor]):
                continue
            neighbor_logit = logits[batch_index, neighbor, field_index, slot_index]
            terms_by_slot[slot_index].append(
                torch_functional.relu(
                    logits.new_tensor(float(margin)) - (gold_logit - neighbor_logit)
                )
            )

    slot_losses = [torch.stack(terms).mean() for terms in terms_by_slot if terms]
    comparisons = sum(len(terms) for terms in terms_by_slot)
    if not slot_losses:
        raise ValueError("boundary margin found no valid local negative")
    return BoundaryMarginLoss(
        loss=torch.stack(slot_losses).mean(),
        comparisons=comparisons,
    )


__all__ = [
    "BOUNDARY_MARGIN_OBJECTIVE_VERSION",
    "BoundaryMarginLoss",
    "local_boundary_margin_loss",
]
