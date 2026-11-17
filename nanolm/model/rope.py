"""Rotary Positional Embeddings (RoPE) with YaRN frequency scaling and per-token position IDs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RotaryEmbedding(nn.Module):
    """RoPE with optional YaRN temperature/wavelength scaling and arbitrary position_ids."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32_768,
        theta: float = 500_000.0,
        yarn_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.yarn_scale_factor = yarn_scale_factor

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        if yarn_scale_factor > 1.0:
            # Smooth NTK-by-parts / YaRN wavelength interpolation
            low_freq_wavelen = max_position_embeddings / 1.0
            high_freq_wavelen = max_position_embeddings / 8.0
            wavelens = 2 * math.pi / inv_freq
            scaled = inv_freq / yarn_scale_factor
            ramp = ((wavelens - high_freq_wavelen) / max(1e-6, low_freq_wavelen - high_freq_wavelen)).clamp(0.0, 1.0)
            inv_freq = (1.0 - ramp) * inv_freq + ramp * scaled

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        attn_scale = 1.0 if yarn_scale_factor <= 1.0 else (0.1 * math.log(yarn_scale_factor) + 1.0)
        self.attn_scale = attn_scale

    def forward(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """Compute (cos, sin) of shape [batch, 1, seq_len, dim//2] for position_ids [batch, seq_len]."""
        inv = self.inv_freq.to(device=position_ids.device, dtype=torch.float32)
        freqs = position_ids.float().unsqueeze(-1) * inv.unsqueeze(0).unsqueeze(0)  # [B, T, D/2]
        cos = (freqs.cos() * self.attn_scale).to(dtype).unsqueeze(1)
        sin = (freqs.sin() * self.attn_scale).to(dtype).unsqueeze(1)
        return cos, sin


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding to x of shape [batch, heads, seq_len, head_dim]."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)
