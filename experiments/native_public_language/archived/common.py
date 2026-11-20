"""Inference definitions from lab/arch/model.py at 10e14108. See PROVENANCE.json."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.w.to(x.dtype), self.eps)


def rope_cache(T, hd, theta, device):
    inv = 1.0 / theta ** (torch.arange(0, hd, 2, device=device).float() / hd)
    f = torch.outer(torch.arange(T, device=device).float(), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):  # x: B,H,T,hd
    x1, x2 = x[..., ::2], x[..., 1::2]
    c, s = cos[None, None, : x.shape[2]], sin[None, None, : x.shape[2]]
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2)


class ShortConv(nn.Module):
    def __init__(self, ch, k):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(ch, 1, k) / math.sqrt(k))

    def forward(self, x):  # B,T,C
        x = F.pad(x.transpose(1, 2), (self.k - 1, 0))
        return F.silu(F.conv1d(x, self.w, groups=x.shape[1])).transpose(1, 2)
