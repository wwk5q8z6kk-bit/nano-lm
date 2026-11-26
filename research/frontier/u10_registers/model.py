"""M21 and M22 model components from the MIT-licensed NanoLM unified-loop register study."""
from __future__ import annotations

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
        self.mix = HSPALieSO3Mixer(d, h, n_reg=hspa_registers, multiscale=False)

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
        logits = self._apply_evidential_sparse_pointer_probs(h, x_toks, logits, anchor)
        if return_base_logits:
            return logits, base_logits
        return (logits, [logits]) if return_loop_logits else logits
