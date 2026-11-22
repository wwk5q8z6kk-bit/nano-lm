"""PyTorch sequence mixer reference cells for X03–X05 (CPU/MPS/CUDA).

Provides parameter-matched causal mixers with exact sequence-vs-step parity and
spectral / full-Jacobian diagnostics:
  - DenseGQACell (AX01 / AX07: standard GQA and Differential GQA)
  - ContractionDeltaCell (beta in (0, 1) standard Gated DeltaNet baseline)
  - HouseholderDeltaCell (beta in (0, 2) rank-1 or rank-r Householder reflection)
  - GDN2IndependentGateCell (AX02: decoupled channelwise erase & write gates)
  - ComplexRotoSSMCell (AX03: Mamba-3-inspired block-SO(2) complex rotation SSM)
  - GeoEraseWriteCell (Nano Sec. 4.3: decoupled decay, reflection & delta write)
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _apply_rope_1d(x: Tensor, offset: int = 0) -> Tensor:
    """Apply standard rotary position embeddings to `[B, H, T, d_h]` (even `d_h`)."""
    _, _, seq_len, hd = x.shape
    half = hd // 2
    inv_freq = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=x.device, dtype=torch.float32) / max(1, half)
    )
    pos = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [T, half]
    cos = freqs.cos().to(x.dtype)[None, None, :, :]
    sin = freqs.sin().to(x.dtype)[None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class DenseGQACell(nn.Module):
    """Causal GQA attention cell with optional Differential Attention (`AX07`)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        *,
        use_diff_attn: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        if d_model % n_heads != 0 or n_heads % self.n_kv_heads != 0:
            raise ValueError("Invalid d_model / n_heads / n_kv_heads divisibility")
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if use_diff_attn and self.head_dim % 4 != 0:
            raise ValueError("Differential GQA requires head_dim divisible by 4")
        self.use_diff_attn = use_diff_attn
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        if use_diff_attn:
            self.diff_lambda = nn.Parameter(torch.full((n_heads,), 0.5))
            self.diff_subnorm = nn.RMSNorm(self.head_dim // 2)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> dict[str, Tensor]:
        return {
            "k": torch.empty(batch_size, self.n_kv_heads, 0, self.head_dim, device=device, dtype=dtype),
            "v": torch.empty(batch_size, self.n_kv_heads, 0, self.head_dim, device=device, dtype=dtype),
        }

    def cache_bytes_per_token(self, dtype_bytes: int = 4) -> int:
        return 2 * self.n_kv_heads * self.head_dim * dtype_bytes

    def forward(
        self, x: Tensor, state: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        bsz, seq_len, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        offset = state["k"].shape[2] if state is not None else 0
        # Each differential attention map needs complete rotary pairs within its own head.
        if self.use_diff_attn:
            q = torch.cat([_apply_rope_1d(part, offset) for part in q.chunk(2, dim=-1)], dim=-1)
            k = torch.cat([_apply_rope_1d(part, offset) for part in k.chunk(2, dim=-1)], dim=-1)
        else:
            q = _apply_rope_1d(q, offset)
            k = _apply_rope_1d(k, offset)

        if state is not None and state["k"].shape[2] > 0:
            k_full = torch.cat([state["k"], k], dim=2)
            v_full = torch.cat([state["v"], v], dim=2)
        else:
            k_full, v_full = k, v
        next_state = {"k": k_full, "v": v_full}

        rep = self.n_heads // self.n_kv_heads
        k_exp = k_full.repeat_interleave(rep, dim=1) if rep > 1 else k_full
        v_exp = v_full.repeat_interleave(rep, dim=1) if rep > 1 else v_full

        total_len = k_full.shape[2]
        q_pos = torch.arange(offset, offset + seq_len, device=x.device)[:, None]
        k_pos = torch.arange(0, total_len, device=x.device)[None, :]
        mask = k_pos <= q_pos  # [T, total_len]

        if not self.use_diff_attn:
            scores = (q @ k_exp.transpose(-1, -2)) * (self.head_dim ** -0.5)
            scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = attn @ v_exp
        else:
            half = self.head_dim // 2
            q1, q2 = q[..., :half], q[..., half:]
            k1, k2 = k_exp[..., :half], k_exp[..., half:]
            scale = half ** -0.5
            s1 = (q1 @ k1.transpose(-1, -2) * scale).masked_fill(~mask[None, None, :, :], float("-inf"))
            s2 = (q2 @ k2.transpose(-1, -2) * scale).masked_fill(~mask[None, None, :, :], float("-inf"))
            a1 = F.softmax(s1, dim=-1)
            a2 = F.softmax(s2, dim=-1)
            lam = self.diff_lambda.view(1, self.n_heads, 1, 1)
            v1, v2 = v_exp[..., :half], v_exp[..., half:]
            out1 = self.diff_subnorm(a1 @ v1 - lam * (a2 @ v1))
            out2 = self.diff_subnorm(a1 @ v2 - lam * (a2 @ v2))
            out = torch.cat([out1, out2], dim=-1)

        out = out.transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        return self.o_proj(out), next_state


class _BaseAssociativeDeltaCell(nn.Module):
    """Shared associative state mixer for Contraction, Householder, and GeoEraseWrite."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        beta_max: float,
        num_reflections: int = 1,
        decoupled_write: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.beta_max = float(beta_max)
        self.num_reflections = num_reflections
        self.decoupled_write = decoupled_write

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=True)
        self.b_proj = nn.Linear(d_model, n_heads * num_reflections, bias=True)
        if num_reflections > 1:
            self.u_extra_proj = nn.Linear(d_model, d_model * (num_reflections - 1), bias=False)
        if decoupled_write:
            self.u_reflect_proj = nn.Linear(d_model, d_model, bias=False)
            self.eta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = nn.RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        with torch.no_grad():
            nn.init.constant_(self.a_proj.bias, 1.5)  # start near high retention rho ~ 0.82
            if beta_max > 1.0:
                nn.init.constant_(self.b_proj.bias, -math.log(3.0))  # 2*sigmoid(-ln 3)=0.5
            else:
                nn.init.zeros_(self.b_proj.bias)
            if decoupled_write:
                nn.init.zeros_(self.eta_proj.bias)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.zeros(batch_size, self.n_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)

    def cache_bytes_per_token(self, dtype_bytes: int = 4) -> int:
        # Fixed recurrent state (zero marginal per-token KV growth)
        return 0

    def state_bytes(self, dtype_bytes: int = 4) -> int:
        return self.n_heads * self.head_dim * self.head_dim * dtype_bytes

    def _compute_step_params(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None, Tensor, Tensor | None]:
        bsz, seq_len, _ = x.shape
        h, hd = self.n_heads, self.head_dim
        q_raw, k_raw, v_raw = self.qkv_proj(x).split(self.d_model, dim=-1)
        q = F.normalize(q_raw.view(bsz, seq_len, h, hd), dim=-1).transpose(1, 2) * (hd ** -0.5)
        k = F.normalize(k_raw.view(bsz, seq_len, h, hd), dim=-1).transpose(1, 2)
        v = v_raw.view(bsz, seq_len, h, hd).transpose(1, 2)
        rho = torch.sigmoid(self.a_proj(x)).transpose(1, 2)  # [B, H, T]
        beta = (self.beta_max * torch.sigmoid(self.b_proj(x))).view(bsz, seq_len, h, self.num_reflections).permute(0, 2, 1, 3)
        eta = torch.sigmoid(self.eta_proj(x)).transpose(1, 2) if self.decoupled_write else None
        u0 = (
            F.normalize(self.u_reflect_proj(x).view(bsz, seq_len, h, hd), dim=-1).transpose(1, 2)
            if self.decoupled_write
            else k
        )
        if self.num_reflections > 1:
            u_extra = F.normalize(
                self.u_extra_proj(x).view(bsz, seq_len, h, self.num_reflections - 1, hd), dim=-1
            ).permute(0, 2, 1, 3, 4)  # [B, H, T, r-1, hd]
        else:
            u_extra = None
        return q, k, v, rho, beta, eta, u0, u_extra

    def step_matrix_operator(self, x_step: Tensor, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Single-step left-acting transition `[B, d]` + state `[B, H, hd, hd]`; returns `A_t`."""
        q, k, v, rho, beta, eta, u0, u_extra = self._compute_step_params(x_step.unsqueeze(1))
        qt, kt, vt = q[:, :, 0], k[:, :, 0], v[:, :, 0]
        rt = rho[:, :, 0]
        bt = beta[:, :, 0]  # [B, H, r]
        u0_t = u0[:, :, 0]
        eye = torch.eye(self.head_dim, device=x_step.device, dtype=state.dtype)

        # Left-acting transition operator A_t = rho_t * prod_j (I - beta_{t,j} u_{t,j}^T u_{t,j})
        A_t = (eye - bt[:, :, 0, None, None] * (u0_t.unsqueeze(-1) @ u0_t.unsqueeze(-2)))
        if u_extra is not None:
            for r_idx in range(self.num_reflections - 1):
                ur = u_extra[:, :, 0, r_idx]
                A_t = (eye - bt[:, :, r_idx + 1, None, None] * (ur.unsqueeze(-1) @ ur.unsqueeze(-2))) @ A_t
        A_t = rt[..., None, None] * A_t

        S_rot = A_t @ state
        if self.decoupled_write:
            assert eta is not None
            et = eta[:, :, 0].unsqueeze(-1)
            err = vt - torch.einsum("bhd,bhde->bhe", kt, state)
            next_S = S_rot + torch.einsum("bhd,bhe->bhde", kt, et * err)
        else:
            next_S = S_rot + torch.einsum("bhd,bhe->bhde", kt, bt[:, :, :1] * vt)
        y_t = torch.einsum("bhd,bhde->bhe", qt, next_S)
        return y_t, next_S, A_t

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        S = self.init_state(bsz, x.device, x.dtype) if state is None else state
        q, k, v, rho, beta, eta, u0, u_extra = self._compute_step_params(x)
        ys: list[Tensor] = []
        for t in range(seq_len):
            qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
            rt = rho[:, :, t].unsqueeze(-1).unsqueeze(-1)
            bt = beta[:, :, t]
            u0_t = u0[:, :, t]

            # Left reflection sequence: S_rot = (I - beta u^T u) S = S - u^T (beta (u S))
            proj0 = torch.einsum("bhd,bhde->bhe", u0_t, S)
            S_rot = S - bt[:, :, 0, None, None] * torch.einsum("bhd,bhe->bhde", u0_t, proj0)
            if u_extra is not None:
                for r_idx in range(self.num_reflections - 1):
                    ur = u_extra[:, :, t, r_idx]
                    proj_r = torch.einsum("bhd,bhde->bhe", ur, S_rot)
                    S_rot = S_rot - bt[:, :, r_idx + 1, None, None] * torch.einsum("bhd,bhe->bhde", ur, proj_r)
            S_rot = rt * S_rot

            if self.decoupled_write:
                assert eta is not None
                et = eta[:, :, t].unsqueeze(-1)
                err = vt - torch.einsum("bhd,bhde->bhe", kt, S)
                S = S_rot + torch.einsum("bhd,bhe->bhde", kt, et * err)
            else:
                S = S_rot + torch.einsum("bhd,bhe->bhde", kt, bt[:, :, :1] * vt)
            ys.append(torch.einsum("bhd,bhde->bhe", qt, S))

        y = torch.stack(ys, dim=2)
        y_normed = self.out_norm(y.transpose(1, 2)).reshape(bsz, seq_len, self.d_model)
        out = self.o_proj(y_normed * F.silu(self.g_proj(x)))
        return out, S


class ContractionDeltaCell(_BaseAssociativeDeltaCell):
    """Standard contraction-only Gated DeltaNet (`beta_t in (0, 1)`)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__(d_model, n_heads, beta_max=1.0, num_reflections=1, decoupled_write=False)


class HouseholderDeltaCell(_BaseAssociativeDeltaCell):
    """Eigenvalue-expanded Householder Delta cell (`beta_t in (0, 2)`, optional rank-r reflections)."""

    def __init__(self, d_model: int, n_heads: int, *, num_reflections: int = 1) -> None:
        super().__init__(d_model, n_heads, beta_max=2.0, num_reflections=num_reflections, decoupled_write=False)


class GeoEraseWriteCell(_BaseAssociativeDeltaCell):
    """Nano Sec. 4.3 decoupled decay `rho_t`, reflection `beta_t in (0, 2)`, and write `eta_t`."""

    def __init__(self, d_model: int, n_heads: int, *, num_reflections: int = 1) -> None:
        super().__init__(d_model, n_heads, beta_max=2.0, num_reflections=num_reflections, decoupled_write=True)


class GDN2IndependentGateCell(nn.Module):
    """Gated DeltaNet-2 inspired cell (`AX02`): channelwise erase `e_t` + independent write `eta_t`."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.erase_proj = nn.Linear(d_model, d_model, bias=True)
        self.write_proj = nn.Linear(d_model, d_model, bias=True)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = nn.RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        with torch.no_grad():
            # Start with low channelwise forget (retention ~ 0.88) and moderate write
            nn.init.constant_(self.erase_proj.bias, -2.0)
            nn.init.zeros_(self.write_proj.bias)
            nn.init.zeros_(self.beta_proj.bias)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.zeros(batch_size, self.n_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)

    def cache_bytes_per_token(self, dtype_bytes: int = 4) -> int:
        return 0

    def state_bytes(self, dtype_bytes: int = 4) -> int:
        return self.n_heads * self.head_dim * self.head_dim * dtype_bytes

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        h, hd = self.n_heads, self.head_dim
        S = self.init_state(bsz, x.device, x.dtype) if state is None else state

        q_raw, k_raw, v_raw = self.qkv_proj(x).split(self.d_model, dim=-1)
        q = F.normalize(q_raw.view(bsz, seq_len, h, hd), dim=-1).transpose(1, 2) * (hd ** -0.5)
        k = F.normalize(k_raw.view(bsz, seq_len, h, hd), dim=-1).transpose(1, 2)
        v = v_raw.view(bsz, seq_len, h, hd).transpose(1, 2)
        retain = (1.0 - torch.sigmoid(self.erase_proj(x))).view(bsz, seq_len, h, hd).transpose(1, 2)
        write_g = torch.sigmoid(self.write_proj(x)).view(bsz, seq_len, h, hd).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)

        ys: list[Tensor] = []
        for t in range(seq_len):
            qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
            ret_t = retain[:, :, t]
            wg_t = write_g[:, :, t]
            bt = beta[:, :, t]

            S_erased = S * ret_t.unsqueeze(-1)
            pred_v = torch.einsum("bhd,bhde->bhe", kt, S_erased)
            err = vt - pred_v
            # Rank-1 delta erase on S_erased plus channelwise gated write
            S = (S_erased - bt[..., None, None] * torch.einsum("bhd,bhe->bhde", kt, pred_v)) + torch.einsum(
                "bhd,bhe->bhde", kt, wg_t * err
            )
            ys.append(torch.einsum("bhd,bhde->bhe", qt, S))

        y = torch.stack(ys, dim=2)
        y_normed = self.out_norm(y.transpose(1, 2)).reshape(bsz, seq_len, self.d_model)
        return self.o_proj(y_normed * F.silu(self.g_proj(x))), S


class ComplexRotoSSMCell(nn.Module):
    """Mamba-3-inspired selective complex/block-SO(2) rotational state mixer (`AX03`)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("ComplexRotoSSMCell requires even head_dim")
        self.pairs = self.head_dim // 2

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.radius_proj = nn.Linear(d_model, n_heads * self.pairs, bias=True)
        self.theta_proj = nn.Linear(d_model, n_heads * self.pairs, bias=True)
        self.write_proj = nn.Linear(d_model, n_heads, bias=True)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = nn.RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        with torch.no_grad():
            nn.init.constant_(self.radius_proj.bias, 4.0)  # start near unitary rotation r ~ 0.982
            nn.init.zeros_(self.theta_proj.bias)
            nn.init.zeros_(self.write_proj.bias)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.zeros(batch_size, self.n_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)

    def cache_bytes_per_token(self, dtype_bytes: int = 4) -> int:
        return 0

    def state_bytes(self, dtype_bytes: int = 4) -> int:
        return self.n_heads * self.head_dim * self.head_dim * dtype_bytes

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        bsz, seq_len, _ = x.shape
        h, hd, p = self.n_heads, self.head_dim, self.pairs
        S = self.init_state(bsz, x.device, x.dtype) if state is None else state

        q_raw, k_raw, v_raw = self.qkv_proj(x).split(self.d_model, dim=-1)
        q = F.normalize(q_raw.view(bsz, seq_len, h, hd), dim=-1).transpose(1, 2) * (hd ** -0.5)
        k = F.normalize(k_raw.view(bsz, seq_len, h, hd), dim=-1).transpose(1, 2)
        v = v_raw.view(bsz, seq_len, h, hd).transpose(1, 2)
        rad = torch.sigmoid(self.radius_proj(x)).view(bsz, seq_len, h, p).transpose(1, 2)
        theta = (math.pi * torch.tanh(self.theta_proj(x))).view(bsz, seq_len, h, p).transpose(1, 2)
        eta = torch.sigmoid(self.write_proj(x)).transpose(1, 2)

        ys: list[Tensor] = []
        for t in range(seq_len):
            qt, kt, vt = q[:, :, t], k[:, :, t], v[:, :, t]
            r_t = rad[:, :, t]
            th_t = theta[:, :, t]
            et = eta[:, :, t].unsqueeze(-1)
            c_t = r_t * th_t.cos()
            s_t = r_t * th_t.sin()

            # Block-2x2 complex rotation on pairs of rows of S: [B, H, P, 2, hd]
            S_pairs = S.view(bsz, h, p, 2, hd)
            s_re, s_im = S_pairs[..., 0, :], S_pairs[..., 1, :]
            rot_re = c_t.unsqueeze(-1) * s_re - s_t.unsqueeze(-1) * s_im
            rot_im = s_t.unsqueeze(-1) * s_re + c_t.unsqueeze(-1) * s_im
            S_rot = torch.stack([rot_re, rot_im], dim=-2).reshape(bsz, h, hd, hd)

            # Pure selective rotational SSM update (additive drive; avoids delta reset fighting rotation)
            S = S_rot + torch.einsum("bhd,bhe->bhde", kt, et * vt)
            ys.append(torch.einsum("bhd,bhde->bhe", qt, S))

        y = torch.stack(ys, dim=2)
        y_normed = self.out_norm(y.transpose(1, 2)).reshape(bsz, seq_len, self.d_model)
        return self.o_proj(y_normed * F.silu(self.g_proj(x))), S


def build_mixer_cell(name: str, d_model: int, n_heads: int) -> nn.Module:
    if name == "dense_gqa":
        return DenseGQACell(d_model, n_heads, max(1, n_heads // 2), use_diff_attn=False)
    if name == "diff_gqa":
        return DenseGQACell(d_model, n_heads, max(1, n_heads // 2), use_diff_attn=True)
    if name == "contraction_delta":
        return ContractionDeltaCell(d_model, n_heads)
    if name == "householder_delta":
        return HouseholderDeltaCell(d_model, n_heads, num_reflections=1)
    if name == "householder_rank2":
        return HouseholderDeltaCell(d_model, n_heads, num_reflections=2)
    if name == "gdn2":
        return GDN2IndependentGateCell(d_model, n_heads)
    if name == "complex_roto_ssm":
        return ComplexRotoSSMCell(d_model, n_heads)
    if name == "geo_erase_write":
        return GeoEraseWriteCell(d_model, n_heads)
    raise ValueError(f"Unknown mixer cell {name!r}")


class ToySequenceModel(nn.Module):
    """Lightweight causal sequence model supporting homogeneous or hybrid layer stacks."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        layer_names: tuple[str, ...],
        num_classes: int,
        *,
        use_ffn: bool = True,
        learnable_init_state: bool = False,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in layer_names])
        self.mixers = nn.ModuleList([build_mixer_cell(name, d_model, n_heads) for name in layer_names])
        self.learnable_init_state = learnable_init_state
        self.init_states = nn.ParameterDict()
        if learnable_init_state:
            hd = d_model // n_heads
            for i, mixer in enumerate(self.mixers):
                if not isinstance(mixer, DenseGQACell):
                    self.init_states[str(i)] = nn.Parameter(torch.randn(1, n_heads, hd, hd) * 0.25)
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn_norms = nn.ModuleList([nn.RMSNorm(d_model) for _ in layer_names])
            self.ffns = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(d_model, 2 * d_model, bias=False),
                        nn.SiLU(),
                        nn.Linear(2 * d_model, d_model, bias=False),
                    )
                    for _ in layer_names
                ]
            )
        self.final_norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        bsz = tokens.shape[0]
        h = self.emb(tokens)
        for i, mixer in enumerate(self.mixers):
            key = str(i)
            s0 = self.init_states[key].expand(bsz, -1, -1, -1) if key in self.init_states else None
            delta, _ = mixer(self.norms[i](h), s0)
            h = h + delta
            if self.use_ffn:
                h = h + self.ffns[i](self.ffn_norms[i](h))
        return self.head(self.final_norm(h))

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def memory_profile_bytes(self, seq_len: int, dtype_bytes: int = 4) -> dict[str, int]:
        kv_per_tok = sum(m.cache_bytes_per_token(dtype_bytes) for m in self.mixers)
        state_b = sum(getattr(m, "state_bytes", lambda _b=4: 0)(dtype_bytes) for m in self.mixers)
        kv_total = kv_per_tok * seq_len
        return {
            "kv_cache_bytes": kv_total,
            "recurrent_state_bytes": state_b,
            "total_sequence_memory_bytes": kv_total + state_b,
        }


def compute_transition_and_jacobian_spectrum(
    cell: _BaseAssociativeDeltaCell | GDN2IndependentGateCell | ComplexRotoSSMCell,
    x_step: Tensor,
    state: Tensor,
) -> dict[str, float]:
    """Compare homogeneous operator spectral norm ||A_t||_2 against actual full-step Jacobian."""
    _, _, hd, _ = state.shape
    flat_dim = hd * hd
    s_in = state[:1].detach().clone().requires_grad_(True)
    x_in = x_step[:1].detach()

    y_out, s_next = cell(x_in.unsqueeze(1), s_in)
    # Full head-0 Jacobian d(S_{t,0})/d(S_{t-1,0})
    s_next_flat = s_next[0, 0].reshape(-1)
    jac_rows = []
    for i in range(flat_dim):
        grad_i = torch.autograd.grad(
            s_next_flat[i], s_in, retain_graph=(i + 1 < flat_dim), allow_unused=True
        )[0]
        row = grad_i[0, 0].reshape(-1) if grad_i is not None else torch.zeros(flat_dim)
        jac_rows.append(row)
    jac = torch.stack(jac_rows, dim=0)
    jac_svals = torch.linalg.svdvals(jac)

    out: dict[str, Any] = {
        "full_step_jacobian_sigma_max": float(jac_svals[0].item()),
        "full_step_jacobian_sigma_min": float(jac_svals[-1].item()),
        "state_output_norm": float(y_out.norm().item()),
    }
    if isinstance(cell, _BaseAssociativeDeltaCell):
        with torch.no_grad():
            _, _, A_t = cell.step_matrix_operator(x_in, s_in)
            a_svals = torch.linalg.svdvals(A_t[0, 0])
            eigvals = torch.linalg.eigvals(A_t[0, 0]).real
            out["homogeneous_A_sigma_max"] = float(a_svals[0].item())
            out["homogeneous_A_min_real_eig"] = float(eigvals.min().item())
            out["homogeneous_A_det"] = float(torch.linalg.det(A_t[0, 0]).item())
    return out
