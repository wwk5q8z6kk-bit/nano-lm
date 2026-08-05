"""Explicit from-scratch weight initialization for Nano trunks.

**Why this is a separate module.** `nano_ai/training/model.py` is SHA-pinned as
`base_model` by the frozen H5 and H6 recipes
(`train_evidence_query_h6.py` raises `H6 preserved H3 source hash mismatch:
base_model` on any edit). Adding a function there — even a purely additive one
that no constructor calls — changes the file hash and invalidates the pin,
which is a frozen scientific record and must not move. So the initializer lives
here instead. This is what "additive-forward" means in practice: a new file,
never an edit to a pinned one.

**What it fixes.** `NanoGPT.__init__` performs no initialization at all. That is
invisible today because every current path warm-starts from an anchor
checkpoint and `load_state_dict` overwrites PyTorch's defaults — but a
from-scratch rung-1 pretrain would silently train from those defaults instead
of the scheme `pretrain/AUDIT.md` records ("init 0.02 depth-scaled", with a
step-1 loss of 8.35 ≈ ln(4096) as its sanity check).

**Explicit and opt-in on purpose.** Nothing calls this automatically, so no
warm-start path, no frozen recipe, and no pinned module changes behaviour.
It also visits only `nn.Linear` and `nn.Embedding`, never raw parameters, so
deliberately zero-initialized tensors — notably H6's
`state_boundary_query_offsets`, on which its step-0 identity argument rests —
are preserved regardless of constructor ordering.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def initialize_weights(
    model: nn.Module, *, generator: torch.Generator | None = None
) -> nn.Module:
    """Apply normal(0, 0.02) with depth-scaled residual projections.

    Residual output projections (`o` in attention, `dn` in the feed-forward)
    are additionally scaled by ``1 / sqrt(2 * layer_count)`` so residual-stream
    variance does not grow with depth.

    Pass a seeded ``generator`` for reproducible initialization; the training
    scripts that adopt this must record that seed alongside the recipe.
    """

    config = getattr(model, "config", None)
    layer_count = getattr(config, "layer_count", None)
    if not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError("model must expose config.layer_count to depth-scale init")
    depth_scale = (2.0 * layer_count) ** -0.5

    def _normal(tensor: Tensor, std: float) -> None:
        with torch.no_grad():
            tensor.normal_(mean=0.0, std=std, generator=generator)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            _normal(module.weight, 0.02)
            if module.bias is not None:
                with torch.no_grad():
                    module.bias.zero_()
        elif isinstance(module, nn.Embedding):
            _normal(module.weight, 0.02)

    for block in getattr(model, "blocks", []):
        for name in ("o", "dn"):
            projection = getattr(block, name, None)
            if isinstance(projection, nn.Linear):
                _normal(projection.weight, 0.02 * depth_scale)

    return model
