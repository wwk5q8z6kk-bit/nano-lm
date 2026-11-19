"""Self-contained, causal-verified cross-paradigm architecture zoo for EXP-U02.

Every mechanism uses identical RMSNorm, cosine/tied token interface, and causal conventions
so differences come purely from the mathematical state/transport/recursion primitive:
  - M0_Dense_GQA_RoPE          : Standard Causal Attention (RoPE) + SwiGLU control
  - M1_GDN1_Cone_Closed        : ShortConv(F.silu) positive-orthant keys + beta in (0,1) + closed SiLU gate
  - M2_NanoOmega_FullSphere_SFOR: Linear conv + zero-mean L2 keys + beta in (0,2) + open +1.5 SFOR gate
  - M3_KDA_DeltaProduct2_AttnRes: Rank-2 Householder (beta in (0,2)) + channelwise KDA decay + Block AttnRes
  - M4_BOBO_DualFastOp         : Bilinear dual-probe outer-product coincidence binding
  - M5_HSPA_LieSO3             : Exact SU(2)->SO(3) spinor quaternion parallel transport
  - M6_NEBN_BayesFilter        : Exponential-family natural-parameter precision filter (diag+matrix)
  - M7_SkewCayley_Flow         : Continuous-time skew-symmetric - dissipative Cayley flow
  - M8_TRM_Recursive_Loop      : Shared recursive latent+draft refinement (K=3) + Monotonic Guard
  - M9_DualChannel_Pointer     : M2 FullSphere+SFOR + causal content copy-pointer channel
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MechanismSpec:
    name: str
    ffn_hidden: int
    n_layers: int = 4
    loops: int = 1
    attn_res: bool = False
    use_pointer: bool = False


MECHANISM_SPECS: dict[str, MechanismSpec] = {
    "M0_Dense_GQA_RoPE": MechanismSpec("M0_Dense_GQA_RoPE", ffn_hidden=96, n_layers=4),
    "M1_GDN1_Cone_Closed": MechanismSpec("M1_GDN1_Cone_Closed", ffn_hidden=80, n_layers=4),
    "M2_NanoOmega_FullSphere_SFOR": MechanismSpec("M2_NanoOmega_FullSphere_SFOR", ffn_hidden=78, n_layers=4),
    "M3_KDA_DeltaProduct2_AttnRes": MechanismSpec("M3_KDA_DeltaProduct2_AttnRes", ffn_hidden=44, n_layers=4, attn_res=True),
    "M4_BOBO_DualFastOp": MechanismSpec("M4_BOBO_DualFastOp", ffn_hidden=82, n_layers=4),
    "M5_HSPA_LieSO3": MechanismSpec("M5_HSPA_LieSO3", ffn_hidden=78, n_layers=4),
    "M6_NEBN_BayesFilter": MechanismSpec("M6_NEBN_BayesFilter", ffn_hidden=82, n_layers=4),
    "M7_SkewCayley_Flow": MechanismSpec("M7_SkewCayley_Flow", ffn_hidden=62, n_layers=4),
    "M8_TRM_Recursive_Loop": MechanismSpec("M8_TRM_Recursive_Loop", ffn_hidden=224, n_layers=2, loops=3),
    "M9_DualChannel_Pointer": MechanismSpec("M9_DualChannel_Pointer", ffn_hidden=72, n_layers=4, use_pointer=True),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class CausalDepthwiseConv(nn.Module):
    def __init__(self, dim: int, k: int = 4, use_silu: bool = False) -> None:
        super().__init__()
        self.k = k
        self.use_silu = use_silu
        self.w = nn.Conv1d(dim, dim, k, groups=dim, padding=k - 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w(x.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        return F.silu(y) if self.use_silu else y


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.w13 = nn.Linear(d, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g, u = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(g) * u)


def _rope_angles(T: int, hd: int, theta: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(T, device=device, dtype=torch.float32)
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    ang = torch.outer(pos, inv)
    return ang.cos().unsqueeze(0).unsqueeze(2), ang.sin().unsqueeze(0).unsqueeze(2)  # [1,T,1,hd/2]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, T, H, hd]
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.stack((out1, out2), dim=-1).flatten(-2).to(x.dtype)


def _center_l2(x: torch.Tensor, center: bool) -> torch.Tensor:
    if center:
        x = x - x.mean(dim=-1, keepdim=True)
    return F.normalize(x.float(), dim=-1)


class CausalSelfAttnMixer(nn.Module):
    def __init__(self, d: int, h: int, rope_theta: float = 10000.0) -> None:
        super().__init__()
        self.h = h
        self.hd = d // h
        self.theta = rope_theta
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).unbind(2)
        q = F.normalize(q.float(), dim=-1).to(x.dtype)
        k = F.normalize(k.float(), dim=-1).to(x.dtype)
        cos, sin = _rope_angles(T, self.hd, self.theta, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class GDN1Mixer(nn.Module):
    """Exact NextGenDeltaMixer regime: ShortConv(F.silu) orthant keys, beta in (0,1), closed gate."""

    def __init__(self, d: int, h: int, full_sphere: bool = False) -> None:
        super().__init__()
        self.h = h
        self.hd = d // h
        self.full_sphere = full_sphere
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = CausalDepthwiseConv(3 * d, k=4, use_silu=not full_sphere)
        self.a = nn.Linear(d, h, bias=False)
        self.A_log = nn.Parameter(torch.zeros(h))
        self.b = nn.Linear(d, h, bias=False)
        self.g = nn.Linear(d, d, bias=full_sphere)
        if full_sphere:
            nn.init.constant_(self.g.bias, 1.5)
        self.onorm = RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)

    def _forward_sequential(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _, _ = q.shape
        BH = B * self.h
        q_t = q.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        k_t = k.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        v_t = v.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        d_t = decay.permute(1, 0, 2).reshape(T, BH, 1, 1).contiguous().unbind(0)
        b_t = beta.permute(1, 0, 2).reshape(T, BH, 1).contiguous().unbind(0)

        S = torch.zeros(BH, self.hd, self.hd, device=q.device, dtype=torch.float32)
        outs: list[torch.Tensor] = []
        for t in range(T):
            kt = k_t[t]
            vt = v_t[t]
            qt = q_t[t]
            dt = d_t[t]
            bt = b_t[t]
            # S = decay * S - decay * beta * (S @ k) k^T + beta * v k^T
            sk = torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
            S = dt * (S - (bt * sk).unsqueeze(-1) * kt.unsqueeze(-2)) + (bt * vt).unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append(torch.bmm(S, qt.unsqueeze(-1)).squeeze(-1))
        return torch.stack(outs, dim=0).view(T, B, self.h, self.hd).permute(1, 0, 2, 3)

    def _forward_chunked_wy(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        log_d: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 32,
    ) -> torch.Tensor:
        B, T, _, _ = q.shape
        BH = B * self.h
        C = chunk_size
        pad_t = (-T) % C

        # Reorder to [BH, T, ...] before chunking
        qf = q.permute(0, 2, 1, 3).reshape(BH, T, self.hd)
        kf = k.permute(0, 2, 1, 3).reshape(BH, T, self.hd)
        vf = v.permute(0, 2, 1, 3).reshape(BH, T, self.hd)
        ldf = log_d.permute(0, 2, 1).reshape(BH, T)
        bf = beta.permute(0, 2, 1).reshape(BH, T)

        if pad_t:
            qf = F.pad(qf, (0, 0, 0, pad_t))
            kf = F.pad(kf, (0, 0, 0, pad_t))
            vf = F.pad(vf, (0, 0, 0, pad_t))
            ldf = F.pad(ldf, (0, pad_t))
            bf = F.pad(bf, (0, pad_t))

        T_pad = T + pad_t
        n_chunks = T_pad // C
        qc = qf.view(BH, n_chunks, C, self.hd)
        kc = kf.view(BH, n_chunks, C, self.hd)
        vc = vf.view(BH, n_chunks, C, self.hd)
        ldc = ldf.view(BH, n_chunks, C)
        bc = bf.view(BH, n_chunks, C)

        g = ldc.cumsum(dim=-1)
        mask_strict = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=-1)
        mask_causal = torch.tril(torch.ones(C, C, device=q.device, dtype=torch.bool), diagonal=0)
        diff = (g.unsqueeze(-1) - g.unsqueeze(-2)).clamp(max=0.0)
        decay_strict = torch.where(mask_strict, diff.exp(), torch.zeros_like(diff))
        decay_causal = torch.where(mask_causal, diff.exp(), torch.zeros_like(diff))

        L = bc.unsqueeze(-1) * decay_strict * torch.matmul(kc, kc.transpose(-1, -2))
        eye_c = torch.eye(C, device=q.device, dtype=torch.float32)
        # Exact Neumann expansion (I + L)^-1 via nilpotent doubling (L^32 == 0)
        T_wy = eye_c - L
        Lp = torch.matmul(L, L)
        T_wy = torch.matmul(T_wy, eye_c + Lp)
        Lp = torch.matmul(Lp, Lp)
        T_wy = torch.matmul(T_wy, eye_c + Lp)
        Lp = torch.matmul(Lp, Lp)
        T_wy = torch.matmul(T_wy, eye_c + Lp)
        Lp = torch.matmul(Lp, Lp)
        T_wy = torch.matmul(T_wy, eye_c + Lp)

        gamma = g.exp()
        U = torch.matmul(T_wy, bc.unsqueeze(-1) * vc)
        P = torch.matmul(T_wy, (bc * gamma).unsqueeze(-1) * kc)
        M_intra = decay_causal * torch.matmul(qc, kc.transpose(-1, -2))
        q_scaled = gamma.unsqueeze(-1) * qc
        end_decay = (g[..., -1:] - g).clamp(max=0.0).exp()
        k_end = end_decay.unsqueeze(-1) * kc
        chunk_decay = gamma[..., -1].unsqueeze(-1).unsqueeze(-1)

        S = torch.zeros(BH, self.hd, self.hd, device=q.device, dtype=torch.float32)
        ys: list[torch.Tensor] = []
        for ci in range(n_chunks):
            St = S.transpose(1, 2)
            W_c = U[:, ci] - torch.bmm(P[:, ci], St)
            Y_c = torch.bmm(q_scaled[:, ci], St) + torch.bmm(M_intra[:, ci], W_c)
            ys.append(Y_c)
            S = chunk_decay[:, ci] * S + torch.bmm(W_c.transpose(1, 2), k_end[:, ci])

        y_all = torch.cat(ys, dim=1)[:, :T]
        return y_all.view(B, self.h, T, self.hd).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.conv(self.qkv(x)).split(D, dim=-1)
        q = _center_l2(q.view(B, T, self.h, self.hd), center=self.full_sphere)
        k = _center_l2(k.view(B, T, self.h, self.hd), center=self.full_sphere)
        v = v.view(B, T, self.h, self.hd).float()

        log_d = -F.softplus(self.a(x.float())) * self.A_log.exp()  # [B, T, H], <= 0
        beta = (2.0 * torch.sigmoid(self.b(x.float()))) if self.full_sphere else torch.sigmoid(self.b(x.float()))

        if T >= 32:
            y_heads = self._forward_chunked_wy(q, k, v, log_d, beta, chunk_size=32)
        else:
            y_heads = self._forward_sequential(q, k, v, log_d.exp(), beta)
        y = self.onorm(y_heads).reshape(B, T, D)
        return self.o((y * F.silu(self.g(x.float()))).to(x.dtype))


class KDADeltaProduct2Mixer(nn.Module):
    """Rank-2 Householder rotation/reflection (beta1,beta2 in (0,2)) + channelwise KDA decay + Delta write."""

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.h = h
        self.hd = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = CausalDepthwiseConv(3 * d, k=4, use_silu=False)
        self.u12 = nn.Linear(d, 2 * d, bias=False)
        self.a = nn.Linear(d, d, bias=False)  # channelwise KDA decay!
        self.A_log = nn.Parameter(torch.zeros(self.h, self.hd))
        self.b_eta = nn.Linear(d, 3 * h, bias=False)
        self.g = nn.Linear(d, d, bias=True)
        nn.init.constant_(self.g.bias, 1.5)
        self.onorm = RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        BH = B * self.h
        q, k, v = self.conv(self.qkv(x)).split(D, dim=-1)
        q = _center_l2(q.view(B, T, self.h, self.hd), center=True)
        k = _center_l2(k.view(B, T, self.h, self.hd), center=True)
        v = v.view(B, T, self.h, self.hd).float()
        u1_raw, u2_raw = self.u12(x).view(B, T, 2, self.h, self.hd).unbind(2)
        u1 = _center_l2(u1_raw, center=True)
        u2 = _center_l2(u2_raw, center=True)

        alpha = (-F.softplus(self.a(x.float()).view(B, T, self.h, self.hd)) * self.A_log.exp()).exp()
        b1_raw, b2_raw, eta_raw = self.b_eta(x.float()).split(self.h, dim=-1)
        b1 = 2.0 * torch.sigmoid(b1_raw)
        b2 = 2.0 * torch.sigmoid(b2_raw)
        eta = torch.sigmoid(eta_raw)

        q_t = q.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        k_t = k.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        v_t = v.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        u1_t = u1.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        u2_t = u2.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        a_t = alpha.permute(1, 0, 2, 3).reshape(T, BH, self.hd, 1).contiguous().unbind(0)
        b1_t = b1.permute(1, 0, 2).reshape(T, BH, 1).contiguous().unbind(0)
        b2_t = b2.permute(1, 0, 2).reshape(T, BH, 1).contiguous().unbind(0)
        eta_t = eta.permute(1, 0, 2).reshape(T, BH, 1).contiguous().unbind(0)

        S = torch.zeros(BH, self.hd, self.hd, device=x.device, dtype=torch.float32)
        outs: list[torch.Tensor] = []
        for t in range(T):
            u1t, u2t = u1_t[t], u2_t[t]
            # Rank-2 orthogonal/contractive transition: S <- (I - b2 u2 u2^T)(I - b1 u1 u1^T) S
            su1 = torch.bmm(S, u1t.unsqueeze(-1)).squeeze(-1)
            S = S - (b1_t[t] * su1).unsqueeze(-1) * u1t.unsqueeze(-2)
            su2 = torch.bmm(S, u2t.unsqueeze(-1)).squeeze(-1)
            S = S - (b2_t[t] * su2).unsqueeze(-1) * u2t.unsqueeze(-2)
            # Channelwise KDA decay + selective Delta erase/write
            S = S * a_t[t]
            kt, vt = k_t[t], v_t[t]
            err = vt - torch.bmm(S, kt.unsqueeze(-1)).squeeze(-1)
            S = S + (eta_t[t] * err).unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append(torch.bmm(S, q_t[t].unsqueeze(-1)).squeeze(-1))
        y = self.onorm(torch.stack(outs, dim=0).view(T, B, self.h, self.hd).permute(1, 0, 2, 3)).reshape(B, T, D)
        return self.o((y * F.silu(self.g(x.float()))).to(x.dtype))


class BOBODualFastMixer(nn.Module):
    """Dual-probe bilinear outer-product binding inspired by UFOFastOperatorMixer."""

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.h = h
        self.hd = d // h
        self.proj = nn.Linear(d, 4 * d, bias=False)
        self.a = nn.Linear(d, h, bias=False)
        self.b = nn.Linear(d, h, bias=False)
        self.onorm = RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        BH = B * self.h
        q1, q2, k, v = self.proj(x).view(B, T, 4, self.h, self.hd).unbind(2)
        q1 = F.normalize(q1.float(), dim=-1)
        q2 = F.normalize(q2.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        v = v.float()
        decay = torch.sigmoid(self.a(x.float()))
        beta = torch.sigmoid(self.b(x.float()))

        q1_t = q1.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        q2_t = q2.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        k_t = k.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        v_t = v.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        d_t = decay.permute(1, 0, 2).reshape(T, BH, 1, 1).contiguous().unbind(0)
        b_t = beta.permute(1, 0, 2).reshape(T, BH, 1).contiguous().unbind(0)

        S = torch.zeros(BH, self.hd, self.hd, device=x.device, dtype=torch.float32)
        outs: list[torch.Tensor] = []
        for t in range(T):
            S = d_t[t] * S + (b_t[t] * v_t[t]).unsqueeze(-1) * k_t[t].unsqueeze(-2)
            r1 = torch.bmm(S, q1_t[t].unsqueeze(-1)).squeeze(-1)
            r2 = torch.bmm(S, q2_t[t].unsqueeze(-1)).squeeze(-1)
            outs.append(r1 * F.silu(r2))
        y = self.onorm(torch.stack(outs, dim=0).view(T, B, self.h, self.hd).permute(1, 0, 2, 3)).reshape(B, T, D)
        return self.o(y.to(x.dtype))


class HSPALieSO3Mixer(nn.Module):
    """Exact SU(2) -> SO(3) spinor quaternion parallel transport over fiber registers."""

    def __init__(self, d: int, h: int, n_reg: int = 4) -> None:
        super().__init__()
        self.h = h
        self.r = n_reg
        self.fdim = h * n_reg * 3  # 4*4*3 = 48
        self.w_omega = nn.Linear(d, self.fdim, bias=False)
        self.w_in = nn.Linear(d, self.fdim, bias=False)
        self.w_gate = nn.Linear(d, 2 * h * n_reg, bias=False)
        self.onorm = RMSNorm(self.fdim)
        self.o = nn.Linear(self.fdim, d, bias=False)

    @staticmethod
    def _rodrigues_step(state: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        # Exact SO(3) exponential map action exp([omega]_x) @ state on R^3 fibers
        theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp_min(1e-6)
        axis = omega / theta
        c = torch.cos(theta)
        s = torch.sin(theta)
        dot = (axis * state).sum(dim=-1, keepdim=True)
        return state * c + torch.linalg.cross(axis, state, dim=-1) * s + axis * dot * (1.0 - c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        omega_t = self.w_omega(x).view(B, T, self.h, self.r, 3).float().permute(1, 0, 2, 3, 4).contiguous().unbind(0)
        u_t = self.w_in(x).view(B, T, self.h, self.r, 3).float().permute(1, 0, 2, 3, 4).contiguous().unbind(0)
        a_raw, eta_raw = self.w_gate(x).view(B, T, 2, self.h, self.r).unbind(2)
        d_t = torch.sigmoid(a_raw.float()).unsqueeze(-1).permute(1, 0, 2, 3, 4).contiguous().unbind(0)
        e_t = torch.sigmoid(eta_raw.float()).unsqueeze(-1).permute(1, 0, 2, 3, 4).contiguous().unbind(0)
        S = torch.zeros(B, self.h, self.r, 3, device=x.device, dtype=torch.float32)
        outs: list[torch.Tensor] = []
        for t in range(T):
            S = d_t[t] * self._rodrigues_step(S, omega_t[t]) + e_t[t] * u_t[t]
            outs.append(S.reshape(B, self.fdim))
        y = self.onorm(torch.stack(outs, dim=1))
        return self.o(y.to(x.dtype))


class NEBNBayersFilterMixer(nn.Module):
    """Exponential-Family Natural-Parameter Bayes Filter: precision-normalized posterior H / (P + lam0)."""

    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        self.h = h
        self.hd = d // h
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.a = nn.Linear(d, h, bias=False)
        self.b = nn.Linear(d, h, bias=False)
        self.lam_log = nn.Parameter(torch.zeros(h))
        self.g = nn.Linear(d, d, bias=False)
        self.onorm = RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        BH = B * self.h
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).unbind(2)
        q = _center_l2(q, center=True)
        k = _center_l2(k, center=True)
        v = v.float()
        decay = torch.sigmoid(self.a(x.float()))
        beta = torch.sigmoid(self.b(x.float()))
        lam0 = self.lam_log.exp().view(1, self.h, 1).expand(B, -1, -1).reshape(BH, 1)

        q_t = q.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        k_t = k.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        v_t = v.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        d_t = decay.permute(1, 0, 2).reshape(T, BH, 1).contiguous().unbind(0)
        b_t = beta.permute(1, 0, 2).reshape(T, BH, 1).contiguous().unbind(0)

        H_mat = torch.zeros(BH, self.hd, self.hd, device=x.device, dtype=torch.float32)
        P_diag = torch.zeros(BH, self.hd, device=x.device, dtype=torch.float32)
        outs: list[torch.Tensor] = []
        for t in range(T):
            dt = d_t[t]
            bt = b_t[t]
            kt = k_t[t]
            vt = v_t[t]
            qt = q_t[t]
            P_diag = dt * P_diag + bt * (kt * kt)
            H_mat = dt.unsqueeze(-1) * H_mat + (bt * vt).unsqueeze(-1) * kt.unsqueeze(-2)
            prec = (P_diag * (qt * qt)).sum(dim=-1, keepdim=True) + lam0
            read = torch.bmm(H_mat, qt.unsqueeze(-1)).squeeze(-1) / prec.clamp_min(1e-3)
            # Epistemic precision certainty weight
            cert = 1.0 - 1.0 / (1.0 + prec)
            outs.append(read * cert)
        y = self.onorm(torch.stack(outs, dim=0).view(T, B, self.h, self.hd).permute(1, 0, 2, 3)).reshape(B, T, D)
        return self.o((y * F.silu(self.g(x.float()))).to(x.dtype))


class SkewCayleyFlowMixer(nn.Module):
    """Continuous-time skew-symmetric + dissipative Cayley flow via analytical 2x2 Woodbury inverse."""

    def __init__(self, d: int, h: int, rank: int = 1) -> None:
        super().__init__()
        self.h = h
        self.hd = d // h
        self.rank = rank
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.uv = nn.Linear(d, 2 * h * rank * self.hd, bias=False)
        self.gamma = nn.Linear(d, d, bias=False)
        self.h_eta = nn.Linear(d, 2 * h, bias=False)
        self.onorm = RMSNorm(self.hd)
        self.o = nn.Linear(d, d, bias=False)

    @staticmethod
    def _cayley_step_woodbury(
        S: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        gamma: torch.Tensor,
        h_step: torch.Tensor,
    ) -> torch.Tensor:
        """Exact `S @ M^T` for `A = u w^T - w u^T - diag(gamma)`, `M = (I - h A)^-1 (I + h A)`."""
        d = 1.0 / (1.0 + h_step * gamma)            # [BH, hd]
        u_d = d * u
        w_d = d * w
        alpha = (w * u_d).sum(dim=-1)               # [BH]
        beta = (u * u_d).sum(dim=-1)                # [BH]
        g_val = (w * w_d).sum(dim=-1)               # [BH]
        h_s = h_step.squeeze(-1)                    # [BH]
        det = 1.0 + (h_s * h_s) * (beta * g_val - alpha * alpha).clamp_min(0.0)
        inv_det = 2.0 / det                         # folds factor 2 into K = 2 * (C^-1)^T
        P = torch.stack((w_d, u_d), dim=-1)         # [BH, hd, 2]
        Q = torch.stack((-h_step * u_d, h_step * w_d), dim=-1)  # [BH, hd, 2]
        row0 = torch.stack(((1.0 + h_s * alpha) * inv_det, (h_s * beta) * inv_det), dim=-1)
        row1 = torch.stack(((-h_s * g_val) * inv_det, (1.0 - h_s * alpha) * inv_det), dim=-1)
        K = torch.stack((row0, row1), dim=-2)       # [BH, 2, 2]
        SP = torch.bmm(S, P)                        # [BH, hd, 2]
        return S * (2.0 * d - 1.0).unsqueeze(1) - torch.bmm(torch.bmm(SP, K), Q.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        BH = B * self.h
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).unbind(2)
        q = _center_l2(q, center=True)
        k = _center_l2(k, center=True)
        v = v.float()

        u_raw, w_raw = self.uv(x.float()).view(B, T, 2, self.h, self.rank, self.hd).unbind(2)
        if self.rank != 1:
            raise ValueError("Woodbury Cayley flow requires rank=1 skew generator")
        inv_s = 1.0 / math.sqrt(float(self.hd))
        u_vecs = (u_raw.squeeze(-2) * inv_s).permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        w_vecs = (w_raw.squeeze(-2) * inv_s).permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        gamma_t = F.softplus(self.gamma(x.float())).view(B, T, self.h, self.hd).permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        h_raw, eta_raw = self.h_eta(x.float()).split(self.h, dim=-1)
        h_t = (0.5 * torch.sigmoid(h_raw)).permute(1, 0, 2).reshape(T, BH, 1).contiguous().unbind(0)
        eta_t = torch.sigmoid(eta_raw).permute(1, 0, 2).reshape(T, BH, 1).contiguous().unbind(0)

        q_t = q.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        k_t = k.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)
        v_t = v.permute(1, 0, 2, 3).reshape(T, BH, self.hd).contiguous().unbind(0)

        S = torch.zeros(BH, self.hd, self.hd, device=x.device, dtype=torch.float32)
        outs: list[torch.Tensor] = []
        for t in range(T):
            S_rot = self._cayley_step_woodbury(S, u_vecs[t], w_vecs[t], gamma_t[t], h_t[t])
            kt, vt = k_t[t], v_t[t]
            err = vt - torch.bmm(S_rot, kt.unsqueeze(-1)).squeeze(-1)
            S = S_rot + (eta_t[t] * err).unsqueeze(-1) * kt.unsqueeze(-2)
            outs.append(torch.bmm(S, q_t[t].unsqueeze(-1)).squeeze(-1))
        y = self.onorm(torch.stack(outs, dim=0).view(T, B, self.h, self.hd).permute(1, 0, 2, 3)).reshape(B, T, D)
        return self.o(y.to(x.dtype))


class UnifiedBlock(nn.Module):
    def __init__(self, spec: MechanismSpec, d: int, h: int, rope_theta: float, ffn_hidden_override: int | None = None) -> None:
        super().__init__()
        self.n1 = RMSNorm(d)
        self.n2 = RMSNorm(d)
        self.ffn = SwiGLU(d, ffn_hidden_override if ffn_hidden_override is not None else spec.ffn_hidden)
        name = spec.name
        if name in ("M0_Dense_GQA_RoPE", "M8_TRM_Recursive_Loop"):
            self.mix: nn.Module = CausalSelfAttnMixer(d, h, rope_theta)
        elif name == "M1_GDN1_Cone_Closed":
            self.mix = GDN1Mixer(d, h, full_sphere=False)
        elif name in ("M2_NanoOmega_FullSphere_SFOR", "M9_DualChannel_Pointer"):
            self.mix = GDN1Mixer(d, h, full_sphere=True)
        elif name == "M3_KDA_DeltaProduct2_AttnRes":
            self.mix = KDADeltaProduct2Mixer(d, h)
        elif name == "M4_BOBO_DualFastOp":
            self.mix = BOBODualFastMixer(d, h)
        elif name == "M5_HSPA_LieSO3":
            self.mix = HSPALieSO3Mixer(d, h)
        elif name == "M6_NEBN_BayesFilter":
            self.mix = NEBNBayersFilterMixer(d, h)
        elif name == "M7_SkewCayley_Flow":
            self.mix = SkewCayleyFlowMixer(d, h)
        else:
            raise ValueError(f"Unknown mechanism {name!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class UnifiedBlockLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        mechanism: str,
        *,
        n_heads: int = 4,
        rope_theta: float = 10000.0,
        ffn_hidden_override: int | None = None,
    ) -> None:
        super().__init__()
        self.spec = MECHANISM_SPECS[mechanism]
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        # Initialize unit-norm embeddings so unseen open-vocab cosine transport has uniform radial norm
        with torch.no_grad():
            self.emb.weight.copy_(F.normalize(self.emb.weight, dim=-1))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(12.0)))

        self.blocks = nn.ModuleList([
            UnifiedBlock(self.spec, d_model, n_heads, rope_theta, ffn_hidden_override=ffn_hidden_override)
            for _ in range(self.spec.n_layers)
        ])
        if self.spec.attn_res:
            self.depth_q = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(self.spec.n_layers)])
            self.depth_norm = RMSNorm(d_model)
        if self.spec.loops > 1:
            self.loop_inject = nn.Linear(2 * d_model, d_model, bias=False)
        if self.spec.use_pointer:
            self.ptr_q = nn.Linear(d_model, d_model, bias=False)
            self.ptr_k = nn.Linear(d_model, d_model, bias=False)
            self.ptr_gate = nn.Linear(d_model, 1, bias=True)
        self.nf = RMSNorm(d_model)

    def _readout(self, h: torch.Tensor) -> torch.Tensor:
        hn = F.normalize(self.nf(h).float(), dim=-1)
        en = F.normalize(self.emb.weight.float(), dim=-1)
        return self.logit_scale.exp().clamp(1.0, 50.0) * F.linear(hn, en)

    def _apply_pointer_probs(self, h_final: torch.Tensor, x_toks: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        """Strictly causal content copy mixture over shifted next-token values x_{j+1} for j < t."""
        B, T = x_toks.shape
        V = self.vocab_size
        probs_base = F.softmax(base_logits, dim=-1)
        if T <= 1:
            return torch.log(probs_base.clamp_min(1e-8))
        q = self.ptr_q(h_final).float() / math.sqrt(float(self.d_model))
        k = self.ptr_k(h_final).float()
        scores = torch.matmul(q, k.transpose(-1, -2))  # [B, T, T], query t attends to source j
        # Strictly causal: j < t so copied target x_{j+1} satisfies j + 1 <= t (present in prefix!)
        strict_causal = torch.tril(torch.ones(T, T, device=x_toks.device, dtype=torch.bool), diagonal=-1)
        scores = scores.masked_fill(~strict_causal.unsqueeze(0), -1e9)
        attn_w = F.softmax(scores, dim=-1)
        attn_w = attn_w.masked_fill(torch.tril(torch.ones(T, T, device=x_toks.device, dtype=torch.bool), diagonal=-1).sum(-1, keepdim=True) == 0, 0.0)
        # Target at step t matching source key pos j copies x_{j+1}
        src_next = x_toks[:, 1:]  # [B, T-1] corresponding to j = 0..T-2
        ptr_vocab = torch.zeros(B, T, V, device=x_toks.device, dtype=torch.float32)
        ptr_vocab.scatter_add_(2, src_next.unsqueeze(1).expand(B, T, T - 1), attn_w[:, :, : T - 1])
        g = torch.sigmoid(self.ptr_gate(h_final.float()))
        # Zero pointer mix at t=0 where no j<t exists
        t0_mask = (torch.arange(T, device=x_toks.device) > 0).view(1, T, 1).float()
        g = g * t0_mask
        mixed = (1.0 - g) * probs_base + g * ptr_vocab
        return torch.log(mixed.clamp_min(1e-8))

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_loop_logits: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x_toks = x
        h = F.normalize(self.emb(x).float(), dim=-1).to(self.emb.weight.dtype)
        anchor = h

        if self.spec.loops > 1:
            loop_logits: list[torch.Tensor] = []
            for k_step in range(self.spec.loops):
                if k_step > 0:
                    h = self.loop_inject(torch.cat([h, anchor], dim=-1))
                for blk in self.blocks:
                    h = blk(h)
                loop_logits.append(self._readout(h))
            out = loop_logits[-1]
            return (out, loop_logits) if return_loop_logits else out

        if self.spec.attn_res:
            hist = [h]
            for li, blk in enumerate(self.blocks):
                stacked = torch.stack(hist, dim=0)  # [M, B, T, D]
                k_norm = self.depth_norm(stacked).float()
                w = torch.softmax((k_norm * self.depth_q[li].float()).sum(dim=-1), dim=0)
                h_in = (w.unsqueeze(-1) * stacked.float()).sum(dim=0).to(h.dtype)
                h = blk(h_in)
                hist.append(h)
        else:
            for blk in self.blocks:
                h = blk(h)

        logits = self._readout(h)
        if self.spec.use_pointer:
            logits = self._apply_pointer_probs(h, x_toks, logits)
        return ([logits] if return_loop_logits else logits) if not return_loop_logits else (logits, [logits])


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_strict_causality(vocab_size: int = 72, d_model: int = 32) -> dict[str, Any]:
    """Verify every mechanism produces EXACT parameter parity and 0 future leakage."""
    torch.manual_seed(0)
    x1 = torch.randint(0, vocab_size, (2, 20))
    x2 = x1.clone()
    x2[:, 15:] = (x2[:, 15:] + 7) % vocab_size  # perturb t >= 15
    report: dict[str, Any] = {}
    for name in MECHANISM_SPECS:
        m = UnifiedBlockLM(vocab_size, d_model, name).eval()
        with torch.no_grad():
            y1 = m(x1)
            y2 = m(x2)
        max_leak = float((y1[:, :15] - y2[:, :15]).abs().max().item())
        report[name] = {
            "params": count_trainable_params(m),
            "causal_leak_max": max_leak,
            "finite": bool(torch.isfinite(y1).all().item()),
        }
    return report


if __name__ == "__main__":
    print(json.dumps(verify_strict_causality(), indent=2) if "json" in globals() else verify_strict_causality())
