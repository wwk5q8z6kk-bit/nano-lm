"""Shared sequence operators, pointer heads, diagnostics, and verification helpers.

All modules operate strictly causally on CPU float32 for deterministic,
zero-cost Cycle 1 mechanism experiments.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T_CRITICAL_95 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.5705818366147395,
    9: 2.2621571627409915,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def summarize_seeds(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ci95_halfwidth": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    if n == 1:
        return {"n": 1, "mean": mean, "sd": 0.0, "ci95_halfwidth": 0.0, "min": mean, "max": mean}
    sd = float(np.std(arr, ddof=1))
    t_crit = T_CRITICAL_95.get(n - 1, 1.96)
    ci95 = t_crit * sd / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci95_halfwidth": float(ci95),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def make_adamw_no_embed_decay(model: nn.Module, lr: float, weight_decay: float = 0.01) -> torch.optim.AdamW:
    """Apply weight decay only to >=2D non-embedding matrices (prevents sparse-vocab shrinkage)."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "embed" in name or p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def effective_rank_and_cosine(keys: torch.Tensor) -> dict[str, float]:
    """Compute effective covariance rank and off-diagonal cosine of unit keys [B, T, H, d_h]."""
    with torch.no_grad():
        k = F.normalize(keys.detach().float(), dim=-1)
        b, t, h, d_h = k.shape
        # Mean off-diagonal cosine across time pairs per (b, h)
        k_bh = k.permute(0, 2, 1, 3).reshape(b * h, t, d_h)  # [BH, T, d_h]
        gram = torch.bmm(k_bh, k_bh.transpose(-1, -2))  # [BH, T, T]
        eye = torch.eye(t, device=gram.device, dtype=torch.bool).unsqueeze(0)
        off = gram.masked_select(~eye)
        mean_cos = float(off.mean().item()) if off.numel() > 0 else 0.0
        mean_abs_cos = float(off.abs().mean().item()) if off.numel() > 0 else 0.0

        # Effective rank via covariance eigenvalues (sum of singular values entropy)
        cov = torch.bmm(k_bh.transpose(-1, -2), k_bh) / max(1, t)  # [BH, d_h, d_h]
        eigvals = torch.linalg.eigvalsh(cov).clamp(min=1e-12)
        probs = eigvals / eigvals.sum(dim=-1, keepdim=True)
        entropy = -(probs * probs.log()).sum(dim=-1)
        eff_rank = float(entropy.exp().mean().item())
        return {
            "eff_rank": eff_rank,
            "max_rank": float(d_h),
            "mean_offdiag_cos": mean_cos,
            "mean_abs_offdiag_cos": mean_abs_cos,
        }


def verify_causality(model: nn.Module, vocab_size: int, seq_len: int = 32, tol: float = 1e-6) -> dict[str, Any]:
    """Verify strict causality by perturbing x[:, p:] for every boundary p in 1..seq_len-1."""
    model.eval()
    g = torch.Generator().manual_seed(12345)
    x = torch.randint(0, vocab_size, (2, seq_len), generator=g)
    with torch.no_grad():
        base_out = model(x)
        base_logits = base_out[0] if isinstance(base_out, tuple) else base_out
        max_diff = 0.0
        worst_p = -1
        for p in range(1, seq_len):
            x_pert = x.clone()
            x_pert[:, p:] = torch.randint(0, vocab_size, (2, seq_len - p), generator=g)
            # Guarantee perturbation actually alters suffix when vocab > 1
            if vocab_size > 1:
                x_pert[:, p:] = (x[:, p:] + 1) % vocab_size
            pert_out = model(x_pert)
            pert_logits = pert_out[0] if isinstance(pert_out, tuple) else pert_out
            diff = float((base_logits[:, :p] - pert_logits[:, :p]).abs().max().item())
            if diff > max_diff:
                max_diff = diff
                worst_p = p
    return {
        "seq_len": seq_len,
        "max_abs_diff": max_diff,
        "worst_boundary": worst_p,
        "passed": max_diff <= tol,
    }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class CausalDepthwiseConv1d(nn.Module):
    """Strict causal depthwise convolution supporting sequence forward and single-step updates."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        xt = x.transpose(1, 2)
        padded = F.pad(xt, (self.kernel_size - 1, 0))
        return self.conv(padded).transpose(1, 2)

    def step(self, x_t: torch.Tensor, ring: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        # x_t: [B, C], ring: [B, k-1, C]
        b, c = x_t.shape
        if self.kernel_size <= 1:
            w = self.conv.weight.view(c)
            out = x_t * w + self.conv.bias
            return out, x_t.new_zeros(b, 0, c)
        if ring is None:
            ring = x_t.new_zeros(b, self.kernel_size - 1, c)
        window = torch.cat([ring, x_t.unsqueeze(1)], dim=1)  # [B, k, C]
        w = self.conv.weight.view(c, self.kernel_size).transpose(0, 1).unsqueeze(0)  # [1, k, C]
        out = (window * w).sum(dim=1) + self.conv.bias
        next_ring = window[:, 1:, :]
        return out, next_ring


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def _apply_rope(x: torch.Tensor, offset: int = 0, theta: float = 10000.0) -> torch.Tensor:
    # x: [B, T, H, d_h]
    _, t, _, d_h = x.shape
    half = d_h // 2
    freqs = theta ** (-torch.arange(0, half, device=x.device, dtype=x.dtype) / half)
    pos = torch.arange(offset, offset + t, device=x.device, dtype=x.dtype)
    ang = pos.unsqueeze(-1) * freqs.unsqueeze(0)  # [T, half]
    cos = ang.cos().view(1, t, 1, half)
    sin = ang.sin().view(1, t, 1, half)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalAttentionMixer(nn.Module):
    """Standard RoPE Causal Attention baseline with local key shift (k=3) & QK-RMSNorm."""

    def __init__(self, d_model: int, n_heads: int, conv_kernel: int = 3) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.last_keys: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q = self.q_norm(self.q_proj(x).view(b, t, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_conv(self.k_proj(x)).view(b, t, self.n_heads, self.head_dim))
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim)
        self.last_keys = k.detach()
        q = _apply_rope(q).transpose(1, 2)  # [B, H, T, d_h]
        k = _apply_rope(k).transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, d)
        return self.o_proj(out)


class GatedLinearMixer(nn.Module):
    """Un-erased Gated Linear / Hebbian matrix recurrence S_t = alpha_t S_{t-1} + k_t v_t^T."""

    def __init__(self, d_model: int, n_heads: int, conv_kernel: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.a_proj.bias, 2.0)
        self.g_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.last_keys: torch.Tensor | None = None

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor | None] | None = None,
        step_mode: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None]]:
        b, t, d = x.shape
        h, dh = self.n_heads, self.head_dim
        if state is None:
            s = x.new_zeros(b, h, dh, dh)
            k_ring = None
        else:
            s, k_ring = state[0].clone(), state[1]
        if step_mode:
            xk, k_ring = self.k_conv.step(self.k_proj(x[:, 0]), k_ring)
            xk = xk.unsqueeze(1)
        else:
            xk = self.k_conv(self.k_proj(x))
        q = F.normalize(F.silu(self.q_proj(x)).view(b, t, h, dh), dim=-1, eps=1e-6)
        k = F.normalize(F.silu(xk).view(b, t, h, dh), dim=-1, eps=1e-6)
        v = self.v_proj(x).view(b, t, h, dh)
        self.last_keys = k.detach()
        alpha = torch.sigmoid(self.a_proj(x))  # [B, T, H]
        gate = torch.sigmoid(self.g_proj(x)).view(b, t, h, dh)

        outs = []
        for i in range(t):
            ki = k[:, i]  # [B, H, dh]
            vi = v[:, i]
            qi = q[:, i]
            ai = alpha[:, i].view(b, h, 1, 1)
            s = ai * s + torch.einsum("bhi,bhj->bhij", ki, vi)
            yi = torch.einsum("bhi,bhij->bhj", qi, s)
            outs.append(yi)
        y = torch.stack(outs, dim=1)
        y = self.o_norm(y) * gate
        return self.o_proj(y.reshape(b, t, d)), (s, k_ring)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_state(x)
        return y

    def step(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor | None] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None]]:
        y, next_state = self.forward_with_state(x_t.unsqueeze(1), state=state, step_mode=True)
        return y.squeeze(1), next_state


class GatedDeltaMixer(nn.Module):
    """Householder Gated Delta recurrence supporting SiLU-orthant vs Full-Sphere and SFOR."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        full_sphere: bool = True,
        open_gate: bool = True,
        decoupled_ew: bool = True,
        conv_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.full_sphere = full_sphere
        self.open_gate = open_gate
        self.decoupled_ew = decoupled_ew

        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.a_proj.bias, 2.0)

        self.be_proj = nn.Linear(d_model, n_heads, bias=True)
        if decoupled_ew:
            self.bw_proj = nn.Linear(d_model, n_heads, bias=True)
        else:
            self.bw_proj = None

        self.g_proj = nn.Linear(d_model, d_model, bias=True)
        if open_gate:
            nn.init.constant_(self.g_proj.bias, 1.5)
        self.o_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.last_keys: torch.Tensor | None = None

    def _compute_qkv(
        self,
        x: torch.Tensor,
        q_ring: torch.Tensor | None = None,
        k_ring: torch.Tensor | None = None,
        step_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        b, t, d = x.shape
        h, dh = self.n_heads, self.head_dim
        if step_mode:
            assert t == 1
            xq, q_ring = self.q_conv.step(self.q_proj(x[:, 0]), q_ring)
            xk, k_ring = self.k_conv.step(self.k_proj(x[:, 0]), k_ring)
            xq = xq.unsqueeze(1)
            xk = xk.unsqueeze(1)
        else:
            xq = self.q_conv(self.q_proj(x))
            xk = self.k_conv(self.k_proj(x))
        if not self.full_sphere:
            xq = F.silu(xq)
            xk = F.silu(xk)
        q = F.normalize(xq.view(b, t, h, dh), dim=-1, eps=1e-6)
        k = F.normalize(xk.view(b, t, h, dh), dim=-1, eps=1e-6)
        v = self.v_proj(x).view(b, t, h, dh)
        return q, k, v, q_ring, k_ring

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None,
        step_mode: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        b, t, d = x.shape
        h, dh = self.n_heads, self.head_dim
        if state is None:
            s = x.new_zeros(b, h, dh, dh)
            q_ring, k_ring = None, None
        else:
            s, q_ring, k_ring = state[0].clone(), state[1], state[2]

        q, k, v, q_ring, k_ring = self._compute_qkv(x, q_ring, k_ring, step_mode=step_mode)
        self.last_keys = k.detach()
        alpha = torch.sigmoid(self.a_proj(x))
        be = 2.0 * torch.sigmoid(self.be_proj(x))
        bw = 2.0 * torch.sigmoid(self.bw_proj(x)) if self.bw_proj is not None else be
        gate = torch.sigmoid(self.g_proj(x)).view(b, t, h, dh)

        outs = []
        for i in range(t):
            ki = k[:, i]  # [B, H, dh]
            vi = v[:, i]
            qi = q[:, i]
            ai = alpha[:, i].view(b, h, 1, 1)
            bei = be[:, i].view(b, h, 1)
            bwi = bw[:, i].view(b, h, 1)
            # s <- alpha * (s - be * k (k^T s)) + bw * k v^T
            k_s = torch.einsum("bhi,bhij->bhj", ki, s)  # [B, H, dh]
            s = ai * (s - torch.einsum("bhi,bhj->bhij", ki, bei * k_s)) + torch.einsum("bhi,bhj->bhij", ki, bwi * vi)
            outs.append(torch.einsum("bhi,bhij->bhj", qi, s))
        y = torch.stack(outs, dim=1)
        y = self.o_norm(y) * gate
        return self.o_proj(y.reshape(b, t, d)), (s, q_ring, k_ring)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_state(x)
        return y

    def step(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        y, next_state = self.forward_with_state(x_t.unsqueeze(1), state=state, step_mode=True)
        return y.squeeze(1), next_state


def right_householder_matrix(u: torch.Tensor) -> torch.Tensor:
    """Return exact orthogonal reflection matrix I - 2 u u^T for unit vector u [..., d]."""
    u_unit = F.normalize(u, dim=-1, eps=1e-6)
    d = u_unit.shape[-1]
    eye = torch.eye(d, device=u.device, dtype=u.dtype)
    return eye - 2.0 * torch.einsum("...i,...j->...ij", u_unit, u_unit)


def _cayley_woodbury_factors(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = (a * b).sum(dim=-1, keepdim=True)
    al = (a * a).sum(dim=-1, keepdim=True)
    be = (b * b).sum(dim=-1, keepdim=True)
    inv2 = 2.0 / (1.0 - c * c + al * be)
    u1 = (1.0 - c) * b + be * a
    u2 = al * b - (1.0 + c) * a
    return inv2, u1, u2


def skew_cayley_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return exact SO(d) rotation (I - A)(I + A)^{-1} for rank-2 skew A = a b^T - b a^T via Woodbury."""
    d = a.shape[-1]
    inv2, u1, u2 = _cayley_woodbury_factors(a, b)
    eye = torch.eye(d, device=a.device, dtype=a.dtype)
    return eye - inv2.unsqueeze(-1) * (
        torch.einsum("...i,...j->...ij", a, u1) + torch.einsum("...i,...j->...ij", b, u2)
    )


def apply_skew_cayley_right(
    s: torch.Tensor, a: torch.Tensor, b: torch.Tensor, inv2: torch.Tensor, u1: torch.Tensor, u2: torch.Tensor
) -> torch.Tensor:
    """Apply s @ R_cayley in O(dh^2) without materializing R."""
    sa = torch.einsum("...ij,...j->...i", s, a)
    sb = torch.einsum("...ij,...j->...i", s, b)
    return s - inv2.unsqueeze(-1) * (
        torch.einsum("...i,...j->...ij", sa, u1) + torch.einsum("...i,...j->...ij", sb, u2)
    )


class BSVDRightValueMixer(nn.Module):
    """Cartan-Dieudonne right-value Householder transport + left-key erasure/write."""

    def __init__(self, d_model: int, n_heads: int, conv_kernel: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.u_proj = nn.Linear(d_model, d_model, bias=False)  # right-value reflection vector
        self.a_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.a_proj.bias, 2.5)
        self.be_proj = nn.Linear(d_model, n_heads, bias=True)
        self.bw_proj = nn.Linear(d_model, n_heads, bias=True)
        self.r_gate = nn.Linear(d_model, n_heads, bias=True)  # reflection angle/blend 0 or full reflection
        self.g_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.g_proj.bias, 1.5)
        self.o_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.last_keys: torch.Tensor | None = None

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None,
        step_mode: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        b, t, d = x.shape
        h, dh = self.n_heads, self.head_dim
        if state is None:
            s = x.new_zeros(b, h, dh, dh)
            q_ring, k_ring = None, None
        else:
            s, q_ring, k_ring = state[0].clone(), state[1], state[2]

        if step_mode:
            xq, q_ring = self.q_conv.step(self.q_proj(x[:, 0]), q_ring)
            xk, k_ring = self.k_conv.step(self.k_proj(x[:, 0]), k_ring)
            xq, xk = xq.unsqueeze(1), xk.unsqueeze(1)
        else:
            xq = self.q_conv(self.q_proj(x))
            xk = self.k_conv(self.k_proj(x))

        q = F.normalize(xq.view(b, t, h, dh), dim=-1, eps=1e-6)
        k = F.normalize(xk.view(b, t, h, dh), dim=-1, eps=1e-6)
        self.last_keys = k.detach()
        v = self.v_proj(x).view(b, t, h, dh)
        u = F.normalize(self.u_proj(x).view(b, t, h, dh), dim=-1, eps=1e-6)
        alpha = torch.sigmoid(self.a_proj(x))
        be = 2.0 * torch.sigmoid(self.be_proj(x))
        bw = 2.0 * torch.sigmoid(self.bw_proj(x))
        # Continuous orthogonal Householder-Cayley blend via 2*theta or pure reflection:
        # Use exact reflection R_t = I - 2 * u u^T blended through a second reflection I - 2 * u2 u2^T
        # or orthogonal Givens-like 2-reflection product; here gated reflection strength 2*sigmoid in (0,2):
        # For strict norm-preserving Cartan-Dieudonne at full strength, rho=2*sigmoid; note rho=2 => R^T R = I.
        rho = 2.0 * torch.sigmoid(self.r_gate(x))
        gate = torch.sigmoid(self.g_proj(x)).view(b, t, h, dh)

        outs = []
        for i in range(t):
            ki, vi, qi, ui = k[:, i], v[:, i], q[:, i], u[:, i]
            ai = alpha[:, i].view(b, h, 1, 1)
            bei = be[:, i].view(b, h, 1)
            bwi = bw[:, i].view(b, h, 1)
            rhoi = rho[:, i].view(b, h, 1)
            # Right-value action: S_rot = S - rho * (S u) u^T
            s_u = torch.einsum("bhij,bhj->bhi", s, ui)  # [B, H, dh]
            s_rot = s - torch.einsum("bhi,bhj->bhij", s_u, rhoi * ui)
            # Left-key erasure + write
            k_s = torch.einsum("bhi,bhij->bhj", ki, s_rot)
            s = ai * (s_rot - torch.einsum("bhi,bhj->bhij", ki, bei * k_s)) + torch.einsum("bhi,bhj->bhij", ki, bwi * vi)
            outs.append(torch.einsum("bhi,bhij->bhj", qi, s))
        y = torch.stack(outs, dim=1)
        y = self.o_norm(y) * gate
        return self.o_proj(y.reshape(b, t, d)), (s, q_ring, k_ring)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_state(x)
        return y

    def step(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        y, next_state = self.forward_with_state(x_t.unsqueeze(1), state=state, step_mode=True)
        return y.squeeze(1), next_state


class SkewCayleyMixer(nn.Module):
    """Exact SO(d_h) right-value Cayley rotation S_t = alpha_t S_{t-1} R_t + beta_w k_t v_t^T."""

    def __init__(self, d_model: int, n_heads: int, conv_kernel: int = 3) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.ra_proj = nn.Linear(d_model, d_model, bias=False)
        self.rb_proj = nn.Linear(d_model, d_model, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.a_proj.bias, 2.5)
        self.be_proj = nn.Linear(d_model, n_heads, bias=True)
        self.bw_proj = nn.Linear(d_model, n_heads, bias=True)
        self.g_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.g_proj.bias, 1.5)
        self.o_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.last_keys: torch.Tensor | None = None

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None,
        step_mode: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        b, t, d = x.shape
        h, dh = self.n_heads, self.head_dim
        if state is None:
            s = x.new_zeros(b, h, dh, dh)
            q_ring, k_ring = None, None
        else:
            s, q_ring, k_ring = state[0].clone(), state[1], state[2]

        if step_mode:
            xq, q_ring = self.q_conv.step(self.q_proj(x[:, 0]), q_ring)
            xk, k_ring = self.k_conv.step(self.k_proj(x[:, 0]), k_ring)
            xq, xk = xq.unsqueeze(1), xk.unsqueeze(1)
        else:
            xq = self.q_conv(self.q_proj(x))
            xk = self.k_conv(self.k_proj(x))

        q = F.normalize(xq.view(b, t, h, dh), dim=-1, eps=1e-6)
        k = F.normalize(xk.view(b, t, h, dh), dim=-1, eps=1e-6)
        self.last_keys = k.detach()
        v = self.v_proj(x).view(b, t, h, dh)
        ra = torch.tanh(self.ra_proj(x).view(b, t, h, dh))
        rb = torch.tanh(self.rb_proj(x).view(b, t, h, dh))
        inv2, u1, u2 = _cayley_woodbury_factors(ra, rb)
        alpha = torch.sigmoid(self.a_proj(x))
        be = 2.0 * torch.sigmoid(self.be_proj(x))
        bw = 2.0 * torch.sigmoid(self.bw_proj(x))
        gate = torch.sigmoid(self.g_proj(x)).view(b, t, h, dh)

        outs = []
        for i in range(t):
            ki, vi, qi = k[:, i], v[:, i], q[:, i]
            ai = alpha[:, i].view(b, h, 1, 1)
            bei = be[:, i].view(b, h, 1)
            bwi = bw[:, i].view(b, h, 1)
            s_rot = apply_skew_cayley_right(s, ra[:, i], rb[:, i], inv2[:, i], u1[:, i], u2[:, i])
            k_s = torch.einsum("bhi,bhij->bhj", ki, s_rot)
            s = ai * (s_rot - torch.einsum("bhi,bhj->bhij", ki, bei * k_s)) + torch.einsum("bhi,bhj->bhij", ki, bwi * vi)
            outs.append(torch.einsum("bhi,bhij->bhj", qi, s))
        y = torch.stack(outs, dim=1)
        y = self.o_norm(y) * gate
        return self.o_proj(y.reshape(b, t, d)), (s, q_ring, k_ring)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_state(x)
        return y

    def step(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        y, next_state = self.forward_with_state(x_t.unsqueeze(1), state=state, step_mode=True)
        return y.squeeze(1), next_state


class HDCDMixer(nn.Module):
    """Heterogeneous Multi-Timescale Dual-Action Cayley-Delta (Novel Cycle 1 Primitive).

    Maintains strictly iso-state-size S_t [B, H, dh, dh] while partitioning heads into
    3 complementary functional guilds (for H >= 3; default H=4):
      - Guild 0 (Fast Syntax, head 0): short causal conv + fast decay prior (a_bias=-1.0)
      - Guild 1 (Slow Associative Revision, middle heads): split lag-0/lag-1 key shift,
        full-sphere keys, slow decay prior (a_bias=+2.5), decoupled erase/write
      - Guild 2 (Non-Abelian Orthogonal Transport, last head): exact SO(dh) right-value
        Cayley action S_{t-1} R_t + decoupled erase/write + slow decay prior (a_bias=+3.0)
    """

    def __init__(self, d_model: int, n_heads: int = 4, conv_kernel: int = 3) -> None:
        super().__init__()
        assert n_heads >= 3, "HDCDMixer requires at least 3 heads for fast/slow/group guilds"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rot_heads = 1  # Guild 2 size (last head)

        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel)
        # Initialize Guild-1 keys with split lag-0 / lag-1 induction prior
        with torch.no_grad():
            mid_start = self.head_dim
            mid_end = (n_heads - 1) * self.head_dim
            half_mid = mid_start + (mid_end - mid_start) // 2
            self.k_conv.conv.weight.zero_()
            self.k_conv.conv.bias.zero_()
            # Default lag-0 identity for all channels except second half of Guild 1 (lag-1)
            self.k_conv.conv.weight[:, 0, -1] = 1.0
            if conv_kernel >= 2 and half_mid < mid_end:
                self.k_conv.conv.weight[half_mid:mid_end, 0, -1] = 0.0
                self.k_conv.conv.weight[half_mid:mid_end, 0, -2] = 1.0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # Rank-2 skew Cayley + gated Householder reflection ONLY on Guild 2 (last head -> O(dh^2) SO(dh)/O(dh)!)
        self.ra_proj = nn.Linear(d_model, self.rot_heads * self.head_dim, bias=False)
        self.rb_proj = nn.Linear(d_model, self.rot_heads * self.head_dim, bias=False)
        self.u_proj = nn.Linear(d_model, self.rot_heads * self.head_dim, bias=False)
        self.r_gate = nn.Linear(d_model, self.rot_heads, bias=True)

        self.a_proj = nn.Linear(d_model, n_heads, bias=True)
        with torch.no_grad():
            self.a_proj.bias[0] = -1.0  # Fast syntax guild
            self.a_proj.bias[1 : n_heads - 1] = 2.5  # Slow associative revision guild
            self.a_proj.bias[n_heads - 1] = 3.0  # Persistent non-Abelian group guild

        self.be_proj = nn.Linear(d_model, n_heads, bias=True)
        self.bw_proj = nn.Linear(d_model, n_heads, bias=True)
        self.g_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.g_proj.bias, 1.5)
        self.o_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.last_keys: torch.Tensor | None = None

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None,
        step_mode: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        b, t, d = x.shape
        h, dh = self.n_heads, self.head_dim
        if state is None:
            s = x.new_zeros(b, h, dh, dh)
            q_ring, k_ring = None, None
        else:
            s, q_ring, k_ring = state[0].clone(), state[1], state[2]

        if step_mode:
            xq, q_ring = self.q_conv.step(self.q_proj(x[:, 0]), q_ring)
            xk, k_ring = self.k_conv.step(self.k_proj(x[:, 0]), k_ring)
            xq, xk = xq.unsqueeze(1), xk.unsqueeze(1)
        else:
            xq = self.q_conv(self.q_proj(x))
            xk = self.k_conv(self.k_proj(x))

        q = F.normalize(xq.view(b, t, h, dh), dim=-1, eps=1e-6)
        k = F.normalize(xk.view(b, t, h, dh), dim=-1, eps=1e-6)
        self.last_keys = k.detach()
        v = self.v_proj(x).view(b, t, h, dh)

        ra = torch.tanh(self.ra_proj(x).view(b, t, self.rot_heads, dh))
        rb = torch.tanh(self.rb_proj(x).view(b, t, self.rot_heads, dh))
        inv2, u1, u2 = _cayley_woodbury_factors(ra, rb)
        u_hh = F.normalize(self.u_proj(x).view(b, t, self.rot_heads, dh), dim=-1, eps=1e-6)
        rho_hh = 2.0 * torch.sigmoid(self.r_gate(x))  # [B, T, 1]

        alpha = torch.sigmoid(self.a_proj(x))
        be = 2.0 * torch.sigmoid(self.be_proj(x))
        bw = 2.0 * torch.sigmoid(self.bw_proj(x))
        gate = torch.sigmoid(self.g_proj(x)).view(b, t, h, dh)

        outs = []
        for i in range(t):
            ki, vi, qi = k[:, i], v[:, i], q[:, i]
            ai = alpha[:, i].view(b, h, 1, 1)
            bei = be[:, i].view(b, h, 1)
            bwi = bw[:, i].view(b, h, 1)

            # Apply exact right-value Woodbury SO(dh) + gated Householder reflection ONLY on Guild 2
            s_fast_slow = s[:, :-1]
            s_cay = apply_skew_cayley_right(
                s[:, -1:], ra[:, i], rb[:, i], inv2[:, i], u1[:, i], u2[:, i]
            )
            ui = u_hh[:, i]
            rhoi = rho_hh[:, i].view(b, 1, 1)
            s_u = torch.einsum("bhij,bhj->bhi", s_cay, ui)
            s_g2 = s_cay - torch.einsum("bhi,bhj->bhij", s_u, rhoi * ui)
            s_rot = torch.cat([s_fast_slow, s_g2], dim=1)

            k_s = torch.einsum("bhi,bhij->bhj", ki, s_rot)
            s = ai * (s_rot - torch.einsum("bhi,bhj->bhij", ki, bei * k_s)) + torch.einsum("bhi,bhj->bhij", ki, bwi * vi)
            outs.append(torch.einsum("bhi,bhij->bhj", qi, s))
        y = torch.stack(outs, dim=1)
        y = self.o_norm(y) * gate
        return self.o_proj(y.reshape(b, t, d)), (s, q_ring, k_ring)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.forward_with_state(x)
        return y

    def step(
        self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]]:
        y, next_state = self.forward_with_state(x_t.unsqueeze(1), state=state, step_mode=True)
        return y.squeeze(1), next_state


class CausalCopyPointerHead(nn.Module):
    """Strictly causal index-based copy pointer over past pairs (key state h_j -> value x_{j+1}).

    For query position t predicting x_{t+1}:
      - Candidate sources are strictly j in {0, ..., t - 1}, so target token x_{j+1} satisfies
        j + 1 <= t (zero future leakage).
      - Matching uses normalized projections of h_t against h_j (so unseen open-vocab token
        embeddings in x_{j+1} do NOT corrupt key matching).
    """

    def __init__(self, d_model: int, decouple_backbone: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.decouple_backbone = decouple_backbone
        self.ptr_q = nn.Linear(d_model, d_model, bias=False)
        self.ptr_k = nn.Linear(d_model, d_model, bias=False)
        self.ptr_scale = nn.Parameter(torch.tensor(math.log(8.0)))
        self.recency_param = nn.Parameter(torch.tensor(-2.0))
        self.gate_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.gate_proj.bias, -1.0)

    def forward(
        self,
        h: torch.Tensor,
        x_tokens: torch.Tensor,
        vocab_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # h: [B, T, D], x_tokens: [B, T], vocab_logits: [B, T, V]
        b, t, _ = h.shape
        v_size = vocab_logits.shape[-1]
        p_vocab = F.softmax(vocab_logits, dim=-1)
        if t < 2:
            return {
                "log_probs": torch.log(p_vocab.clamp(min=1e-12)),
                "vocab_logits": vocab_logits,
                "copy_gate": h.new_zeros(b, t, 1),
            }

        # When decoupled, stop-grad backbone features into pointer gate/matching so copy tokens
        # only rely on backbone representations shaped by auxiliary or non-copy objectives.
        h_ptr = h.detach() if self.decouple_backbone else h
        q = F.normalize(self.ptr_q(h_ptr), dim=-1, eps=1e-6)
        k = F.normalize(self.ptr_k(h_ptr[:, :-1, :]), dim=-1, eps=1e-6)  # j in 0..t-2 (predicting up to t-1)
        scale = self.ptr_scale.exp().clamp(max=32.0)

        # Attention [B, T, T-1]: row t attends only to j < t (i.e. j <= t - 1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, T, T-1]
        row_idx = torch.arange(t, device=h.device).unsqueeze(1)
        col_idx = torch.arange(t - 1, device=h.device).unsqueeze(0)
        valid = col_idx < row_idx  # [T, T-1]; row 0 has all False!
        # Strictly causal relative recency bias prefers latest overwrite on identical keys
        recency_slope = F.softplus(self.recency_param)
        scores = scores + recency_slope * (col_idx - row_idx).to(scores.dtype).unsqueeze(0)
        scores = scores.masked_fill(~valid.unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        # Zero out row 0 where no valid source exists
        has_src = valid.any(dim=-1).view(1, t, 1)
        attn = attn * has_src.to(attn.dtype)

        # Scatter attention mass onto copied token IDs x_{j+1} (all strictly <= t!)
        copy_targets = x_tokens[:, 1:]  # [B, T-1]
        p_copy = h.new_zeros(b, t, v_size)
        p_copy.scatter_add_(2, copy_targets.unsqueeze(1).expand(b, t, t - 1), attn)

        g = torch.sigmoid(self.gate_proj(h_ptr)) * has_src.to(h.dtype)
        p_mix = (1.0 - g) * p_vocab + g * p_copy
        return {
            "log_probs": torch.log(p_mix.clamp(min=1e-12)),
            "vocab_logits": vocab_logits,
            "copy_gate": g,
        }


def build_mixer(kind: str, d_model: int, n_heads: int) -> nn.Module:
    if kind == "attn":
        return CausalAttentionMixer(d_model, n_heads)
    if kind == "linear":
        return GatedLinearMixer(d_model, n_heads)
    if kind == "delta_silu":
        return GatedDeltaMixer(d_model, n_heads, full_sphere=False, open_gate=False, decoupled_ew=False)
    if kind == "delta_sfor":
        return GatedDeltaMixer(d_model, n_heads, full_sphere=True, open_gate=True, decoupled_ew=True)
    if kind == "bsvd":
        return BSVDRightValueMixer(d_model, n_heads)
    if kind == "cayley":
        return SkewCayleyMixer(d_model, n_heads)
    if kind == "hdcd":
        return HDCDMixer(d_model, n_heads)
    raise ValueError(f"Unknown mixer kind: {kind!r}")


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, mixer_kind: str) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = build_mixer(mixer_kind, d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Cycle1SeqModel(nn.Module):
    """Unified multi-layer sequence model for controlled mechanism & pointer comparisons."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        mixer_kind: str = "hdcd",
        pointer_mode: str = "none",  # 'none' | 'standard' | 'decoupled'
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mixer_kind = mixer_kind
        self.pointer_mode = pointer_mode
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, mixer_kind) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model)
        if pointer_mode == "standard":
            self.pointer: CausalCopyPointerHead | None = CausalCopyPointerHead(d_model, decouple_backbone=False)
        elif pointer_mode == "decoupled":
            self.pointer = CausalCopyPointerHead(d_model, decouple_backbone=True)
        else:
            self.pointer = None

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_f(h)
        vocab_logits = F.linear(h, self.embed.weight)
        if self.pointer is None:
            out = vocab_logits
            if return_aux:
                return out, {"vocab_logits": vocab_logits, "log_probs": F.log_softmax(vocab_logits, dim=-1)}
            return out
        ptr_out = self.pointer(h, x, vocab_logits)
        if return_aux:
            return ptr_out["log_probs"], ptr_out
        return ptr_out["log_probs"]

    def mixer_key_diagnostics(self) -> list[dict[str, float]]:
        diags = []
        for idx, blk in enumerate(self.blocks):
            lk = getattr(blk.mixer, "last_keys", None)
            if lk is not None:
                d = effective_rank_and_cosine(lk)
                d["layer"] = idx
                diags.append(d)
        return diags


def match_d_ff_for_target_params(
    target_non_emb_params: int,
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    mixer_kind: str,
    pointer_mode: str = "none",
) -> int:
    """Choose SwiGLU d_ff so non-embedding parameter count matches target within tight tolerance."""
    probe = Cycle1SeqModel(vocab_size, d_model, n_heads, n_layers, d_ff=16, mixer_kind=mixer_kind, pointer_mode=pointer_mode)
    probe_non_emb = count_trainable_params(probe) - probe.embed.weight.numel()
    # Each increment of 1 in d_ff adds n_layers * 3 * d_model parameters
    per_dff = n_layers * 3 * d_model
    d_ff = max(16, round(16 + (target_non_emb_params - probe_non_emb) / per_dff))
    # Round to multiple of 4 for clean alignment
    return max(16, int(round(d_ff / 4) * 4))
