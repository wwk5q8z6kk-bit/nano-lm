"""Inference definitions from nanolm/kernels/fast_delta.py at 10e14108. See PROVENANCE.json."""

from __future__ import annotations

import torch
from torch import Tensor


def _block2x2_unit_tril_inv(A: Tensor) -> Tensor:
    """Numerically stable O(1)-bounded Divide-and-Conquer Block-2x2 unit lower-triangular inverse (`9e-6` err at beta=1.95)."""
    n = A.shape[-1]
    if n == 1:
        return torch.ones_like(A)
    h = n // 2
    # Invert both diagonal blocks in ONE batched recursive call!
    tb = torch.cat([A[..., :h, :h], A[..., h:, h:]], dim=0)
    inv_tb = _block2x2_unit_tril_inv(tb)
    L11_inv, L22_inv = inv_tb.chunk(2, dim=0)
    L21 = A[..., h:, :h]
    BL = -(L22_inv @ L21 @ L11_inv)
    top = torch.cat([L11_inv, torch.zeros_like(BL)], dim=-1)
    bot = torch.cat([BL, L22_inv], dim=-1)
    return torch.cat([top, bot], dim=-2)


def nilpotent_wy_inverse(Akk: Tensor) -> Tensor:
    """Invert unit lower-triangular `(I + Akk)` stably across Householder `beta in (0, 2)` without binomial blowup."""
    C = Akk.shape[-1]
    A_strict = Akk.tril(-1)
    if C <= 1:
        return torch.ones_like(A_strict)
    if A_strict.is_cuda or A_strict.device.type == "cpu":
        eye = torch.eye(C, device=A_strict.device, dtype=torch.float32)
        return torch.linalg.solve_triangular(eye + A_strict.float(), eye.expand_as(A_strict), upper=False).to(Akk.dtype)
    if (C & (C - 1)) == 0:
        return _block2x2_unit_tril_inv(A_strict.float()).to(Akk.dtype)
    eye = torch.eye(C, device=A_strict.device, dtype=torch.float32)
    return torch.linalg.solve_triangular(eye + A_strict.float(), eye.expand_as(A_strict), upper=False).to(Akk.dtype)
