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

import heapq
import math
from dataclasses import dataclass
from typing import Any, Literal

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
    pointer_mode: str = "none"  # "none" | "dense" | "sparse_ow" | "sparse_evid" | "sparse_hybrid"
    scaled_residual: bool = False
    use_concept_slots: bool = False

    @property
    def has_pointer(self) -> bool:
        return self.pointer_mode != "none" or self.use_pointer

    @property
    def resolved_pointer_mode(self) -> str:
        if self.pointer_mode != "none":
            return self.pointer_mode
        return "dense" if self.use_pointer else "none"


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
    "M9_DualChannel_Pointer": MechanismSpec("M9_DualChannel_Pointer", ffn_hidden=72, n_layers=4, use_pointer=True, pointer_mode="dense"),
    "M10_HSPA_DensePointer": MechanismSpec("M10_HSPA_DensePointer", ffn_hidden=72, n_layers=4, use_pointer=True, pointer_mode="dense"),
    "M11_FullSphere_SparseOWPointer": MechanismSpec("M11_FullSphere_SparseOWPointer", ffn_hidden=72, n_layers=4, use_pointer=True, pointer_mode="sparse_ow"),
    "M12_HSPA_SparseOWPointer": MechanismSpec("M12_HSPA_SparseOWPointer", ffn_hidden=72, n_layers=4, use_pointer=True, pointer_mode="sparse_ow"),
    "M13_HSPA_EvidCosSparsePtr": MechanismSpec("M13_HSPA_EvidCosSparsePtr", ffn_hidden=70, n_layers=4, use_pointer=True, pointer_mode="sparse_evid"),
    "M14_GaugeSU2_Pure": MechanismSpec("M14_GaugeSU2_Pure", ffn_hidden=44, n_layers=4),
    "M15_GaugeSU2_EvidCosSparsePtr": MechanismSpec("M15_GaugeSU2_EvidCosSparsePtr", ffn_hidden=35, n_layers=4, use_pointer=True, pointer_mode="sparse_evid"),
    "M16_FullSphereTri_EvidCosSparsePtr": MechanismSpec("M16_FullSphereTri_EvidCosSparsePtr", ffn_hidden=68, n_layers=4, use_pointer=True, pointer_mode="sparse_evid"),
    "M17_HSPA_MultiScale_EvidCosSparsePtr": MechanismSpec("M17_HSPA_MultiScale_EvidCosSparsePtr", ffn_hidden=68, n_layers=4, use_pointer=True, pointer_mode="sparse_evid"),
    "M18_HSPA_HybridSemLexPtr": MechanismSpec("M18_HSPA_HybridSemLexPtr", ffn_hidden=70, n_layers=4, use_pointer=True, pointer_mode="sparse_hybrid"),
    "M19_HSPA_ScaledNorm_EvidPtr": MechanismSpec("M19_HSPA_ScaledNorm_EvidPtr", ffn_hidden=70, n_layers=4, use_pointer=True, pointer_mode="sparse_evid", scaled_residual=True),
    "M20_HSPA_ScaledAttnRes_EvidPtr": MechanismSpec("M20_HSPA_ScaledAttnRes_EvidPtr", ffn_hidden=70, n_layers=4, attn_res=True, use_pointer=True, pointer_mode="sparse_evid", scaled_residual=True),
    "M21_HSPA_ConceptSlot_EvidPtr": MechanismSpec("M21_HSPA_ConceptSlot_EvidPtr", ffn_hidden=67, n_layers=4, use_pointer=True, pointer_mode="sparse_evid", scaled_residual=True, use_concept_slots=True),
    "M22_HSPA_FullCore_U09": MechanismSpec("M22_HSPA_FullCore_U09", ffn_hidden=67, n_layers=4, attn_res=True, use_pointer=True, pointer_mode="sparse_evid", scaled_residual=True, use_concept_slots=True),
}


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x.float(), (x.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class CenteredRMSNorm(nn.Module):
    """Zero-mean RMSNorm removing isotropic DC residual drift while preserving O(1) variance."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xc = xf - xf.mean(dim=-1, keepdim=True)
        return F.rms_norm(xc, (xc.shape[-1],), self.weight.float(), self.eps).to(x.dtype)


class ConceptSlotMemory(nn.Module):
    """O(T log T * C * D) strictly causal differentiable concept-slot overwrite memory.

    Pairs contextual key h_{s-1} with value embedding emb_in_s (s >= 1) via associative overwrite:
        alpha_{s, c} = sigmoid(slot_write(h_{s-1})) * softmax(slot_k(h_{s-1}))_c
        M_s[c] = (1 - alpha_{s, c}) * M_{s-1}[c] + alpha_{s, c} * emb_in_s
    Query at step t reads M_{t-1} (strictly s < t; zero at t <= 1) and mixes via norm-aware gate.
    """

    def __init__(self, d_model: int, n_slots: int = 16) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.slot_k = nn.Linear(d_model, n_slots, bias=False)
        self.slot_q = nn.Linear(d_model, n_slots, bias=False)
        self.slot_write = nn.Linear(d_model, 1, bias=True)
        self.slot_gate = nn.Linear(d_model, 1, bias=True)
        with torch.no_grad():
            self.slot_write.bias.fill_(-1.0)
            self.slot_gate.bias.fill_(-1.0)

    @staticmethod
    def _scan_sequential(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        B, L, C, D = b.shape
        a_steps = a.unbind(dim=1)
        b_steps = b.unbind(dim=1)
        m = torch.zeros(B, C, D, device=b.device, dtype=torch.float32)
        outs: list[torch.Tensor] = []
        for t in range(L):
            m = a_steps[t] * m + b_steps[t]
            outs.append(m)
        return torch.stack(outs, dim=1)

    @staticmethod
    def _scan_parallel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        L = b.shape[1]
        if L == 0:
            return b
        aa = a
        ss = b
        stride = 1
        while stride < L:
            a_prev, s_prev = aa[:, :-stride], ss[:, :-stride]
            a_curr, s_curr = aa[:, stride:], ss[:, stride:]
            aa = torch.cat((aa[:, :stride], a_curr * a_prev), dim=1)
            ss = torch.cat((ss[:, :stride], a_curr * s_prev + s_curr), dim=1)
            stride <<= 1
        return ss

    def forward(
        self,
        h_n: torch.Tensor,
        probs_base: torch.Tensor,
        emb_in: torch.Tensor,
        en: torch.Tensor,
        logit_scale: torch.Tensor,
        *,
        force_sequential: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = h_n.shape
        if T <= 2:
            g_zero = torch.zeros(B, T, 1, device=h_n.device, dtype=torch.float32)
            return probs_base, g_zero

        h_nf = h_n.float()
        h_w = h_nf[:, :-2]
        v_w = emb_in[:, 1:-1].float()

        w_gate = torch.sigmoid(self.slot_write(h_w))
        w_slot = F.softmax(self.slot_k(h_w), dim=-1)
        alpha = w_gate * w_slot
        a = (1.0 - alpha).unsqueeze(-1)
        b = alpha.unsqueeze(-1) * v_w.unsqueeze(2)

        L = T - 2
        if not force_sequential and L >= 8:
            M_scan = self._scan_parallel(a, b)
        else:
            M_scan = self._scan_sequential(a, b)

        zero_prefix = torch.zeros(B, 2, self.n_slots, D, device=h_n.device, dtype=torch.float32)
        M_read = torch.cat((zero_prefix, M_scan), dim=1)

        q_slot = F.softmax(self.slot_q(h_nf), dim=-1)
        m_t = torch.einsum("btc,btcd->btd", q_slot, M_read)

        m_sq = (m_t * m_t).sum(dim=-1, keepdim=True)
        has_mem = (m_sq > 1e-10).float()
        m_norm = torch.sqrt(m_sq + 1e-8)
        m_dir = (m_t / m_norm) * has_mem

        scale = logit_scale.exp().clamp(1.0, 50.0)
        logits_slot = scale * F.linear(m_dir, en)
        probs_slot = F.softmax(logits_slot, dim=-1)

        g_slot = torch.sigmoid(self.slot_gate(h_nf)) * (m_norm * has_mem).clamp(0.0, 1.0)
        mixed = (1.0 - g_slot) * probs_base + g_slot * probs_slot
        return mixed, g_slot


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
        eye_c = torch.eye(C, device=q.device, dtype=torch.float32).view(1, 1, C, C).expand_as(L)
        # Exact unit lower-triangular solve (I + L)^-1 avoids Neumann |L^k| float32 cancellation
        T_wy = torch.linalg.solve_triangular(eye_c + L, eye_c, upper=False, unitriangular=True)

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


def _axis_angle_to_quat(omega: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt((omega * omega).sum(dim=-1, keepdim=True) + 1e-12)
    half = 0.5 * r
    sinc_half = torch.where(half < 1e-3, 1.0 - (half * half) / 6.0, torch.sin(half) / half.clamp_min(1e-6))
    return torch.cat((torch.cos(half), 0.5 * sinc_half * omega), dim=-1)


def _quat_mul(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    pw, px, py, pz = p.unbind(-1)
    qw, qx, qy, qz = q.unbind(-1)
    return torch.stack((
        pw * qw - px * qx - py * qy - pz * qz,
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
    ), dim=-1)


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_rot(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.linalg.cross(u, v, dim=-1)
    return v + w * t + torch.linalg.cross(u, t, dim=-1)


def _quat_to_so3_6d(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    yy, zz, xx = y * y, z * z, x * x
    xy, wz, xz, wy, yz, wx = x * y, w * z, x * z, w * y, y * z, w * x
    return torch.stack((
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy + wz),
        2.0 * (xz - wy),
        2.0 * (xy - wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz + wx),
    ), dim=-1)


def _gauge_su2_scan_state(
    dq: torch.Tensor,
    u_in: torch.Tensor,
    decay: torch.Tensor,
    eta: torch.Tensor,
    h: int,
    r: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact O(log T) gauge-decoupled SU(2) holonomy scan + co-moving scalar affine body scan."""
    B, T = dq.shape[:2]
    BHR = B * h * r
    q = dq.permute(0, 2, 3, 1, 4).reshape(BHR, T, 4).contiguous()
    stride = 1
    while stride < T:
        q = torch.cat((q[:, :stride], _quat_mul(q[:, stride:], q[:, :-stride])), dim=1)
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-6)
        stride <<= 1

    s = _quat_rot(
        _quat_conj(q),
        (eta * u_in).permute(0, 2, 3, 1, 4).reshape(BHR, T, 3).contiguous(),
    )
    a = decay.permute(0, 2, 3, 1, 4).reshape(BHR, T, 1).contiguous()
    stride = 1
    while stride < T:
        a_prev, s_prev = a[:, :-stride], s[:, :-stride]
        a_curr, s_curr = a[:, stride:], s[:, stride:]
        a = torch.cat((a[:, :stride], a_curr * a_prev), dim=1)
        s = torch.cat((s[:, :stride], a_curr * s_prev + s_curr), dim=1)
        stride <<= 1

    S = _quat_rot(q, s)
    return q, S


class HSPALieSO3Mixer(nn.Module):
    """Exact SU(2) -> SO(3) spinor quaternion parallel transport over fiber registers."""

    def __init__(
        self, d: int, h: int, n_reg: int | Literal["full_head"] = 4, multiscale: bool = False
    ) -> None:
        super().__init__()
        if n_reg == "full_head":
            if h < 1 or d < 1 or d % h:
                raise ValueError("full_head requires a positive, integral head dimension")
            # Three coordinates per register; preserve the four-register floor.
            n_reg = max(4, (d // h + 2) // 3)
        if type(n_reg) is not int or n_reg < 1:
            raise ValueError("n_reg must be a positive integer or 'full_head'")
        self.h = h
        self.r = n_reg
        self.multiscale = multiscale
        self.fdim = h * n_reg * 3
        self.conv = CausalDepthwiseConv(d, k=4, use_silu=False) if multiscale else None
        if self.conv is not None:
            nn.init.normal_(self.conv.w.weight, mean=0.0, std=0.05)
        self.w_omega = nn.Linear(d, self.fdim, bias=False)
        self.w_in = nn.Linear(d, self.fdim, bias=False)
        self.w_gate = nn.Linear(d, 2 * h * n_reg, bias=multiscale)
        if multiscale and self.w_gate.bias is not None:
            with torch.no_grad():
                a_bias = torch.linspace(0.0, 4.5, n_reg).view(1, n_reg).expand(h, n_reg).reshape(-1)
                self.w_gate.bias[: h * n_reg].copy_(a_bias)
                self.w_gate.bias[h * n_reg :].zero_()
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

    def _forward_sequential(
        self,
        omega: torch.Tensor,
        u_in: torch.Tensor,
        decay: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        B, T = omega.shape[:2]
        omega_t = omega.permute(1, 0, 2, 3, 4).contiguous().unbind(0)
        u_t = u_in.permute(1, 0, 2, 3, 4).contiguous().unbind(0)
        d_t = decay.permute(1, 0, 2, 3, 4).contiguous().unbind(0)
        e_t = eta.permute(1, 0, 2, 3, 4).contiguous().unbind(0)
        S = torch.zeros(B, self.h, self.r, 3, device=omega.device, dtype=torch.float32)
        outs: list[torch.Tensor] = []
        for t in range(T):
            S = d_t[t] * self._rodrigues_step(S, omega_t[t]) + e_t[t] * u_t[t]
            outs.append(S.reshape(B, self.fdim))
        return torch.stack(outs, dim=1)

    def _forward_parallel_scan(
        self,
        omega: torch.Tensor,
        u_in: torch.Tensor,
        decay: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        """Exact O(log T) gauge-decoupled SU(2) quaternion scan + scalar body-frame scan."""
        B, T = omega.shape[:2]
        dq = _axis_angle_to_quat(omega)
        _, S = _gauge_su2_scan_state(dq, u_in, decay, eta, self.h, self.r)
        return S.view(B, self.h, self.r, T, 3).permute(0, 3, 1, 2, 4).reshape(B, T, self.fdim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xc = (x + self.conv(x)) if self.conv is not None else x
        omega = self.w_omega(xc).view(B, T, self.h, self.r, 3).float()
        u_in = self.w_in(xc).view(B, T, self.h, self.r, 3).float()
        a_raw, eta_raw = self.w_gate(xc.float()).view(B, T, 2, self.h, self.r).unbind(2)
        decay = torch.sigmoid(a_raw.float()).unsqueeze(-1)
        eta = torch.sigmoid(eta_raw.float()).unsqueeze(-1)
        raw = (
            self._forward_parallel_scan(omega, u_in, decay, eta)
            if T >= 8
            else self._forward_sequential(omega, u_in, decay, eta)
        )
        y = self.onorm(raw)
        return self.o(y.to(x.dtype))


class GaugeSU2HolonomyMixer(nn.Module):
    """Gauge-decoupled SU(2) holonomy scan (6D SO(3) group frame + multi-timescale co-moving state)."""

    _axis_angle_to_quat = staticmethod(_axis_angle_to_quat)
    _quat_mul = staticmethod(_quat_mul)
    _quat_conj = staticmethod(_quat_conj)
    _quat_rot = staticmethod(_quat_rot)
    _quat_to_so3_6d = staticmethod(_quat_to_so3_6d)

    def __init__(self, d: int, h: int, n_reg: int = 4) -> None:
        super().__init__()
        self.h = h
        self.r = n_reg
        self.fdim = h * n_reg * 3
        self.hdim = h * n_reg * 6
        self.out_dim = h * n_reg * 9
        self.conv = CausalDepthwiseConv(d, k=4, use_silu=False)
        nn.init.normal_(self.conv.w.weight, mean=0.0, std=0.05)
        self.w_omega = nn.Linear(d, self.fdim, bias=False)
        self.w_in = nn.Linear(d, self.fdim, bias=False)
        self.w_gate = nn.Linear(d, 2 * h * n_reg, bias=True)
        with torch.no_grad():
            a_bias = torch.linspace(0.0, 4.5, n_reg).view(1, n_reg).expand(h, n_reg).reshape(-1)
            self.w_gate.bias[: h * n_reg].copy_(a_bias)
            self.w_gate.bias[h * n_reg :].zero_()
        self.onorm = RMSNorm(self.out_dim)
        self.o = nn.Linear(self.out_dim, d, bias=False)

    def _forward_sequential(
        self,
        dq: torch.Tensor,
        u_in: torch.Tensor,
        decay: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        B, T = dq.shape[:2]
        q = dq.new_zeros((B, self.h, self.r, 4))
        q[..., 0] = 1.0
        S = dq.new_zeros((B, self.h, self.r, 3))
        dq_t = dq.unbind(1)
        u_t = (eta * u_in).unbind(1)
        d_t = decay.unbind(1)
        outs: list[torch.Tensor] = []
        for t in range(T):
            q = _quat_mul(dq_t[t], q)
            q = q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-6)
            S = d_t[t] * _quat_rot(dq_t[t], S) + u_t[t]
            outs.append(torch.cat((_quat_to_so3_6d(q), S), dim=-1).reshape(B, self.out_dim))
        return torch.stack(outs, dim=1)

    def _forward_parallel_scan(
        self,
        dq: torch.Tensor,
        u_in: torch.Tensor,
        decay: torch.Tensor,
        eta: torch.Tensor,
    ) -> torch.Tensor:
        """Exact O(log T) gauge-decoupled SU(2) quaternion scan + scalar body-frame scan."""
        B, T = dq.shape[:2]
        q, S = _gauge_su2_scan_state(dq, u_in, decay, eta, self.h, self.r)
        return (
            torch.cat((_quat_to_so3_6d(q), S), dim=-1)
            .view(B, self.h, self.r, T, 9)
            .permute(0, 3, 1, 2, 4)
            .reshape(B, T, self.out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        xc = x + self.conv(x)
        omega = self.w_omega(xc).view(B, T, self.h, self.r, 3).float()
        u_in = self.w_in(xc).view(B, T, self.h, self.r, 3).float()
        a_raw, eta_raw = self.w_gate(xc.float()).view(B, T, 2, self.h, self.r).unbind(2)
        decay = torch.sigmoid(a_raw.float()).unsqueeze(-1)
        eta = torch.sigmoid(eta_raw.float()).unsqueeze(-1)
        dq = _axis_angle_to_quat(omega)
        raw = (
            self._forward_parallel_scan(dq, u_in, decay, eta)
            if T >= 8
            else self._forward_sequential(dq, u_in, decay, eta)
        )
        return self.o(self.onorm(raw).to(x.dtype))


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
    def __init__(
        self, spec: MechanismSpec, d: int, h: int, rope_theta: float, ffn_hidden_override: int | None = None,
        *, hspa_registers: int | Literal["full_head"] = 4,
    ) -> None:
        super().__init__()
        norm_cls = CenteredRMSNorm if spec.scaled_residual else RMSNorm
        self.n1 = norm_cls(d)
        self.n2 = norm_cls(d)
        self.ffn = SwiGLU(d, ffn_hidden_override if ffn_hidden_override is not None else spec.ffn_hidden)
        name = spec.name
        if name in ("M0_Dense_GQA_RoPE", "M8_TRM_Recursive_Loop"):
            self.mix: nn.Module = CausalSelfAttnMixer(d, h, rope_theta)
        elif name == "M1_GDN1_Cone_Closed":
            self.mix = GDN1Mixer(d, h, full_sphere=False)
        elif name in ("M2_NanoOmega_FullSphere_SFOR", "M9_DualChannel_Pointer", "M11_FullSphere_SparseOWPointer", "M16_FullSphereTri_EvidCosSparsePtr"):
            self.mix = GDN1Mixer(d, h, full_sphere=True)
        elif name == "M3_KDA_DeltaProduct2_AttnRes":
            self.mix = KDADeltaProduct2Mixer(d, h)
        elif name == "M4_BOBO_DualFastOp":
            self.mix = BOBODualFastMixer(d, h)
        elif name in (
            "M5_HSPA_LieSO3",
            "M10_HSPA_DensePointer",
            "M12_HSPA_SparseOWPointer",
            "M13_HSPA_EvidCosSparsePtr",
            "M18_HSPA_HybridSemLexPtr",
            "M19_HSPA_ScaledNorm_EvidPtr",
            "M20_HSPA_ScaledAttnRes_EvidPtr",
            "M21_HSPA_ConceptSlot_EvidPtr",
            "M22_HSPA_FullCore_U09",
        ):
            self.mix = HSPALieSO3Mixer(d, h, n_reg=hspa_registers, multiscale=False)
        elif name == "M17_HSPA_MultiScale_EvidCosSparsePtr":
            self.mix = HSPALieSO3Mixer(d, h, n_reg=hspa_registers, multiscale=True)
        elif name in ("M14_GaugeSU2_Pure", "M15_GaugeSU2_EvidCosSparsePtr"):
            self.mix = GaugeSU2HolonomyMixer(d, h)
        elif name == "M6_NEBN_BayesFilter":
            self.mix = NEBNBayersFilterMixer(d, h)
        elif name == "M7_SkewCayley_Flow":
            self.mix = SkewCayleyFlowMixer(d, h)
        else:
            raise ValueError(f"Unknown mechanism {name!r}")

        if hspa_registers != 4 and not isinstance(self.mix, HSPALieSO3Mixer):
            raise ValueError("hspa_registers only applies to an HSPA mixer")

        if spec.scaled_residual:
            scale = 1.0 / math.sqrt(2.0 * float(spec.n_layers))
            with torch.no_grad():
                if hasattr(self.mix, "o") and hasattr(self.mix.o, "weight"):
                    self.mix.o.weight.mul_(scale)
                self.ffn.w2.weight.mul_(scale)

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
        hspa_registers: int | Literal["full_head"] = 4,
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

        norm_cls = CenteredRMSNorm if self.spec.scaled_residual else RMSNorm
        self.blocks = nn.ModuleList([
            UnifiedBlock(self.spec, d_model, n_heads, rope_theta, ffn_hidden_override, hspa_registers=hspa_registers)
            for _ in range(self.spec.n_layers)
        ])
        if self.spec.attn_res:
            self.depth_q = nn.ParameterList([nn.Parameter(torch.zeros(d_model)) for _ in range(self.spec.n_layers)])
            self.depth_norm = norm_cls(d_model)
        if self.spec.loops > 1:
            self.loop_inject = nn.Linear(2 * d_model, d_model, bias=False)
        if self.spec.has_pointer:
            self.ptr_q = nn.Linear(d_model, d_model, bias=False)
            self.ptr_k = nn.Linear(d_model, d_model, bias=False)
            self.ptr_gate = nn.Linear(d_model, 1, bias=True)
        pmode = self.spec.resolved_pointer_mode
        if pmode in ("sparse_ow", "sparse_evid", "sparse_hybrid"):
            self.ptr_n_exact = 8
            self.ptr_n_sem = 8 if pmode == "sparse_hybrid" else 0
            self.ptr_n_local = 16 if pmode == "sparse_hybrid" else 24
            exact_bias = [2.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5]
            tail_bias = ([0.5] * 8 + [0.0] * 16) if pmode == "sparse_hybrid" else ([0.0] * 24)
            self.ptr_slot_bias = nn.Parameter(torch.tensor(exact_bias + tail_bias, dtype=torch.float32))
            self.ptr_dist_decay = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        if pmode in ("sparse_evid", "sparse_hybrid"):
            self.ptr_norm = norm_cls(d_model)
            self.ptr_lex_conv = CausalDepthwiseConv(d_model, k=4, use_silu=False)
            with torch.no_grad():
                taps = torch.tensor([0.125, 0.25, 0.5, 1.0], dtype=self.ptr_lex_conv.w.weight.dtype)
                self.ptr_lex_conv.w.weight.copy_(taps.view(1, 1, 4).expand_as(self.ptr_lex_conv.w.weight))
            self.ptr_lex_proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.ptr_lex_proj.weight)
            self.ptr_tau_ctx = nn.Parameter(torch.tensor(math.log(6.0), dtype=torch.float32))
            self.ptr_tau_lex = nn.Parameter(torch.tensor(math.log(6.0), dtype=torch.float32))
            self.ptr_ev_gate = nn.Linear(6, 1, bias=False)
            nn.init.zeros_(self.ptr_ev_gate.weight)
        if pmode == "sparse_hybrid":
            self.ptr_salience = nn.Linear(d_model, 1, bias=False)
            self.ptr_sal_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.concept_slots: ConceptSlotMemory | None = (
            ConceptSlotMemory(d_model, n_slots=16) if self.spec.use_concept_slots else None
        )
        self.nf = norm_cls(d_model)

    @staticmethod
    def _build_sparse_pointer_candidates_hash(
        x_toks: torch.Tensor,
        n_exact: int = 8,
        n_local: int = 24,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact O(B * T * K) CPU hash-list sparse candidate builder (used for T > 1024)."""
        B, T = x_toks.shape
        K = n_exact + n_local
        x_list = x_toks.detach().cpu().tolist()
        idx_all = [[[0] * K for _ in range(T)] for _ in range(B)]
        val_all = [[[False] * K for _ in range(T)] for _ in range(B)]
        for b, row in enumerate(x_list):
            last_pos: dict[int, list[int]] = {}
            idx_b = idx_all[b]
            val_b = val_all[b]
            for t, tok in enumerate(row):
                used: set[int] = set()
                hist = last_pos.get(tok)
                if hist:
                    n_e = min(n_exact, len(hist))
                    i_t = idx_b[t]
                    v_t = val_b[t]
                    for rank in range(n_e):
                        j = hist[-1 - rank]
                        i_t[rank] = j
                        v_t[rank] = True
                        used.add(j)
                i_t = idx_b[t]
                v_t = val_b[t]
                for w in range(n_local):
                    j = t - 1 - w
                    if j < 0:
                        break
                    if j not in used:
                        slot = n_exact + w
                        i_t[slot] = j
                        v_t[slot] = True
                last_pos.setdefault(tok, []).append(t)
        return (
            torch.tensor(idx_all, dtype=torch.long, device=x_toks.device),
            torch.tensor(val_all, dtype=torch.bool, device=x_toks.device),
        )

    @staticmethod
    def _build_sparse_pointer_candidates(
        x_toks: torch.Tensor,
        n_exact: int = 8,
        n_local: int = 24,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Strictly causal sparse candidate index: top-`n_exact` recency-ordered matches + deduplicated `n_local`."""
        B, T = x_toks.shape
        if T > 1024:
            return UnifiedBlockLM._build_sparse_pointer_candidates_hash(x_toks, n_exact, n_local)
        dev = x_toks.device
        t_pos = torch.arange(T, device=dev)
        j_pos = torch.arange(T, device=dev)
        strict = j_pos.unsqueeze(0) < t_pos.unsqueeze(1)
        eq = (x_toks.unsqueeze(2) == x_toks.unsqueeze(1)) & strict.unsqueeze(0)
        rec_key = torch.where(eq, j_pos.view(1, 1, T).expand(B, T, T), x_toks.new_full((B, T, T), -1))
        k_e = min(n_exact, T)
        top_vals, _ = torch.topk(rec_key, k=k_e, dim=-1, largest=True, sorted=True)
        exact_valid = top_vals >= 0
        exact_idx = top_vals.clamp_min(0)
        if k_e < n_exact:
            pad = n_exact - k_e
            exact_idx = F.pad(exact_idx, (0, pad), value=0)
            exact_valid = F.pad(exact_valid, (0, pad), value=False)

        w = torch.arange(n_local, device=dev).view(1, 1, n_local)
        raw_loc = t_pos.view(1, T, 1) - 1 - w
        loc_idx = raw_loc.clamp_min(0).expand(B, T, n_local)
        dup = ((loc_idx.unsqueeze(-1) == exact_idx.unsqueeze(-2)) & exact_valid.unsqueeze(-2)).any(dim=-1)
        loc_valid = (raw_loc >= 0) & (~dup)
        loc_idx = torch.where(loc_valid, loc_idx, torch.zeros_like(loc_idx))
        return torch.cat((exact_idx, loc_idx), dim=-1), torch.cat((exact_valid, loc_valid), dim=-1)

    @staticmethod
    def _select_causal_salience_slots(
        sal_key: torch.Tensor,
        excl_idx: torch.Tensor,
        excl_valid: torch.Tensor,
        n_sem: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Strictly causal (j < t) top-n_sem salient landmark selection, deduplicated against excl_idx."""
        B, T = sal_key.shape
        dev = sal_key.device
        if n_sem <= 0 or T <= 1:
            return (
                torch.zeros((B, T, n_sem), dtype=torch.long, device=dev),
                torch.zeros((B, T, n_sem), dtype=torch.bool, device=dev),
            )
        if T <= 1024:
            t_pos = torch.arange(T, device=dev)
            j_pos = torch.arange(T, device=dev)
            strict = (j_pos.unsqueeze(0) < t_pos.unsqueeze(1)).unsqueeze(0)  # [1, T, T]
            excl_mask = torch.zeros((B, T, T + 1), device=dev, dtype=torch.bool)
            safe_slot = torch.where(excl_valid, excl_idx, torch.full_like(excl_idx, T))
            excl_mask.scatter_(2, safe_slot, True)
            blocked = (~strict) | excl_mask[:, :, :T]
            cand_s = sal_key.unsqueeze(1).expand(B, T, T).masked_fill(blocked, -1e9)
            k_s = min(n_sem, T)
            top_s, top_i = torch.topk(cand_s, k=k_s, dim=-1, largest=True, sorted=True)
            sem_valid = top_s > -1e8
            sem_idx = torch.where(sem_valid, top_i, torch.zeros_like(top_i))
            if k_s < n_sem:
                pad = n_sem - k_s
                sem_idx = F.pad(sem_idx, (0, pad), value=0)
                sem_valid = F.pad(sem_valid, (0, pad), value=False)
            return sem_idx, sem_valid

        # O(T) streaming bounded min-heap exact via pigeonhole (cap = n_sem + excl_width)
        cap = n_sem + excl_idx.shape[-1]
        sal_cpu = sal_key.detach().cpu().tolist()
        ex_i = excl_idx.detach().cpu().tolist()
        ex_v = excl_valid.detach().cpu().tolist()
        sem_i = [[[0] * n_sem for _ in range(T)] for _ in range(B)]
        sem_v = [[[False] * n_sem for _ in range(T)] for _ in range(B)]
        for b in range(B):
            heap: list[tuple[float, int]] = []
            for t in range(T):
                excl = {idx for idx, ok in zip(ex_i[b][t], ex_v[b][t]) if ok}
                top_sorted = sorted((j for _, j in heap if j not in excl), key=lambda j: (-sal_cpu[b][j], -j))
                for rank, j in enumerate(top_sorted[:n_sem]):
                    sem_i[b][t][rank] = j
                    sem_v[b][t][rank] = True
                item = (float(sal_cpu[b][t]), t)
                if len(heap) < cap:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        return (
            torch.tensor(sem_i, dtype=torch.long, device=dev),
            torch.tensor(sem_v, dtype=torch.bool, device=dev),
        )

    def _mix_sparse_pointer_vocab(
        self,
        x_toks: torch.Tensor,
        probs_base: torch.Tensor,
        b_idx: torch.Tensor,
        cand_idx: torch.Tensor,
        attn_w: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        B, T = x_toks.shape
        copied_toks = x_toks[b_idx, (cand_idx + 1).clamp_max(T - 1)]
        ptr_vocab = torch.zeros(B, T, self.vocab_size, device=x_toks.device, dtype=torch.float32)
        ptr_vocab.scatter_add_(2, copied_toks, attn_w)
        mixed = (1.0 - g) * probs_base + g * ptr_vocab
        return torch.log(mixed.clamp_min(1e-8))

    def _apply_sparse_pointer_probs(
        self,
        h_final: torch.Tensor,
        x_toks: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        """O(T * K) overwrite-aware causal pointer with recency slot bias + learned log-distance decay."""
        B, T = x_toks.shape
        probs_base = F.softmax(base_logits, dim=-1)
        if T <= 1:
            return torch.log(probs_base.clamp_min(1e-8))

        cand_idx, cand_valid = self._build_sparse_pointer_candidates(x_toks, self.ptr_n_exact, self.ptr_n_local)
        K = self.ptr_n_exact + self.ptr_n_local
        dev = x_toks.device

        q = self.ptr_q(h_final).float() / math.sqrt(float(self.d_model))
        k = self.ptr_k(h_final).float()
        b_idx = torch.arange(B, device=dev).view(B, 1, 1)
        ctx_scores = (q.unsqueeze(2) * k[b_idx, cand_idx]).sum(dim=-1)

        t_pos = torch.arange(T, device=dev).view(1, T, 1)
        dist = (t_pos - cand_idx).clamp_min(1).float().log()
        gamma_dist = F.softplus(self.ptr_dist_decay)
        scores = ctx_scores + self.ptr_slot_bias.view(1, 1, K) - gamma_dist * dist
        scores = scores.masked_fill(~cand_valid, -1e9)

        has_any = cand_valid.any(dim=-1, keepdim=True)
        attn_w = torch.where(has_any, F.softmax(scores, dim=-1), torch.zeros_like(scores)) * cand_valid.float()
        g = torch.sigmoid(self.ptr_gate(h_final.float())) * has_any.float()
        return self._mix_sparse_pointer_vocab(x_toks, probs_base, b_idx, cand_idx, attn_w, g)

    def _compute_sparse_evid_components(
        self,
        h_final: torch.Tensor,
        x_toks: torch.Tensor,
        probs_base: torch.Tensor,
        emb_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T = x_toks.shape
        dev = x_toks.device
        h_n = self.ptr_norm(h_final).float()

        if self.ptr_n_sem > 0:
            base_idx, base_valid = self._build_sparse_pointer_candidates(x_toks, self.ptr_n_exact, self.ptr_n_local)
            sal = self.ptr_salience(h_n).squeeze(-1)
            sal_key = sal.detach() + (0.02 * torch.randn_like(sal) if self.training else 0.0)
            sem_idx, sem_valid = self._select_causal_salience_slots(sal_key, base_idx, base_valid, self.ptr_n_sem)
            ex_idx, loc_idx = base_idx[..., : self.ptr_n_exact], base_idx[..., self.ptr_n_exact :]
            ex_val, loc_val = base_valid[..., : self.ptr_n_exact], base_valid[..., self.ptr_n_exact :]
            cand_idx = torch.cat((ex_idx, sem_idx, loc_idx), dim=-1)
            cand_valid = torch.cat((ex_val, sem_valid, loc_val), dim=-1)
        else:
            sal = None
            cand_idx, cand_valid = self._build_sparse_pointer_candidates(x_toks, self.ptr_n_exact, self.ptr_n_local)

        K = cand_idx.shape[-1]
        b_idx = torch.arange(B, device=dev).view(B, 1, 1)

        q_ctx = F.normalize(self.ptr_q(h_n), dim=-1)
        k_ctx = F.normalize(self.ptr_k(h_n), dim=-1)
        cos_ctx = (q_ctx.unsqueeze(2) * k_ctx[b_idx, cand_idx]).sum(dim=-1)

        u_lex = F.normalize(self.ptr_lex_proj(self.ptr_lex_conv(emb_in).float()), dim=-1)
        cos_lex = (u_lex.unsqueeze(2) * u_lex[b_idx, cand_idx]).sum(dim=-1)

        tau_ctx = self.ptr_tau_ctx.exp().clamp(0.5, 25.0)
        tau_lex = self.ptr_tau_lex.exp().clamp(0.5, 25.0)
        t_pos = torch.arange(T, device=dev).view(1, T, 1)
        dist = (t_pos - cand_idx).clamp_min(1).float().log()
        scores = (
            tau_ctx * cos_ctx
            + tau_lex * cos_lex
            + self.ptr_slot_bias.view(1, 1, K)
            - F.softplus(self.ptr_dist_decay) * dist
        )
        if sal is not None and self.ptr_n_sem > 0:
            sem_mask = torch.zeros((1, 1, K), device=dev, dtype=scores.dtype)
            sem_mask[:, :, self.ptr_n_exact : self.ptr_n_exact + self.ptr_n_sem] = 1.0
            scores = scores + (self.ptr_sal_scale * torch.tanh(sal[b_idx, cand_idx])) * sem_mask
        scores = scores.masked_fill(~cand_valid, -1e9)

        has_any = cand_valid.any(dim=-1, keepdim=True)
        attn_w = torch.where(has_any, F.softmax(scores, dim=-1), torch.zeros_like(scores)) * cand_valid.float()

        cos_comb = 0.5 * (cos_ctx + cos_lex)
        has_exact = cand_valid[..., 0:1].float()
        log_n_exact = torch.log1p(cand_valid[..., : self.ptr_n_exact].float().sum(dim=-1, keepdim=True)) / math.log(
            self.ptr_n_exact + 1.0
        )
        max_cos = cos_comb.masked_fill(~cand_valid, -2.0).max(dim=-1, keepdim=True).values.clamp(-1.0, 1.0)
        exact0_cos = torch.where(cand_valid[..., 0:1], cos_comb[..., 0:1], torch.zeros_like(cos_comb[..., 0:1]))
        ptr_ent = -(attn_w * attn_w.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True) / math.log(float(K))
        base_ent = (-(probs_base * probs_base.clamp_min(1e-8).log()).sum(dim=-1, keepdim=True) / math.log(float(self.vocab_size))).detach()

        ev_feats = torch.cat((has_exact, log_n_exact, max_cos, exact0_cos, ptr_ent, base_ent), dim=-1)
        g = torch.sigmoid(self.ptr_gate(h_n) + self.ptr_ev_gate(ev_feats)) * has_any.float()
        return b_idx, cand_idx, cand_valid, attn_w, g, cos_lex

    def _apply_evidential_sparse_pointer_probs(
        self,
        h_final: torch.Tensor,
        x_toks: torch.Tensor,
        base_logits: torch.Tensor,
        emb_in: torch.Tensor,
    ) -> torch.Tensor:
        """O(T * K) evidential dual-basis cosine sparse pointer (bounded logits + Bayes/MDL gate)."""
        probs_base = F.softmax(base_logits, dim=-1)
        if self.concept_slots is not None:
            h_n = self.ptr_norm(h_final).float()
            en = F.normalize(self.emb.weight.float(), dim=-1)
            probs_base, _ = self.concept_slots(h_n, probs_base, emb_in, en, self.logit_scale)
        if x_toks.shape[1] <= 1:
            return torch.log(probs_base.clamp_min(1e-8))
        b_idx, cand_idx, _, attn_w, g, _ = self._compute_sparse_evid_components(h_final, x_toks, probs_base, emb_in)
        return self._mix_sparse_pointer_vocab(x_toks, probs_base, b_idx, cand_idx, attn_w, g)

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
        return_base_logits: bool = False,
    ) -> Any:
        x_toks = x
        anchor = F.normalize(self.emb(x).float(), dim=-1).to(self.emb.weight.dtype)
        h = (math.sqrt(float(self.d_model)) * anchor) if self.spec.scaled_residual else anchor

        if self.spec.loops > 1:
            loop_logits: list[torch.Tensor] = []
            for k_step in range(self.spec.loops):
                if k_step > 0:
                    h = self.loop_inject(torch.cat([h, anchor], dim=-1))
                for blk in self.blocks:
                    h = blk(h)
                loop_logits.append(self._readout(h))
            out = loop_logits[-1]
            if return_base_logits:
                return out, out
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
        base_logits = logits
        pmode = self.spec.resolved_pointer_mode
        if pmode == "dense":
            logits = self._apply_pointer_probs(h, x_toks, logits)
        elif pmode == "sparse_ow":
            logits = self._apply_sparse_pointer_probs(h, x_toks, logits)
        elif pmode in ("sparse_evid", "sparse_hybrid"):
            logits = self._apply_evidential_sparse_pointer_probs(h, x_toks, logits, anchor)
        if return_base_logits:
            return logits, base_logits
        return (logits, [logits]) if return_loop_logits else logits


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
