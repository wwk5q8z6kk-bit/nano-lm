"""Inference definitions from lab/arch/gen4_physics_math.py at 10e14108. See PROVENANCE.json."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def apply_data_dependent_rotor(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Apply block-2x2 symplectic rotation `R(phase)` to `x` (`[B, H, T, hd]`, `phase`: `[B, H, T, hd//2]`)."""
    cos = phase.cos()
    sin = phase.sin()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


class IdentityConv3(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d, 1, 3))
        with torch.no_grad():
            self.w[:, 0, -1] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(F.pad(x.transpose(1, 2), (2, 0)), self.w.to(x.dtype), groups=x.size(-1)).transpose(1, 2).contiguous()
