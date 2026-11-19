"""Unified Holonomic Spinor Predictive Architecture (HSPA) & Controlled Baselines.

Five iso-parameter configurations sharing embeddings, RMSNorm pre-norm, and LM head
while isolating genuinely distinct computational mechanisms:
1. ``B0_Transformer``: Standard GQA + RoPE + SwiGLU attention baseline.
2. ``B1_Abelian_SSM``: Commutative U(1) complex-phase competitive register scan + SwiGLU.
3. ``Gen1_H1_HSPA_Spinor``: Non-Abelian SU(2) quaternionic parallel transport over
   competitive spinor registers + SwiGLU (zero KV cache, O(1) persistent state).
4. ``Gen1_H2_HSPA_PredCode``: SU(2) Holonomic Spinor Field + iterative precision-weighted
   Predictive-Coding FFN cell (``PredictiveCodingCell``).
5. ``Gen2_H3_HSPA_Evolved``: Evolved generation adding causal spinor wavepacket pre-filter
   (k=4) + Kuramoto cross-register phase coupling across spinor registers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_holonomy.spinor_algebra import (
    abelian_complex_scan,
    cayley_rotor,
    complex_cayley_rotor,
    complex_conjugate,
    complex_product,
    hamilton_product,
    so3_projective_adjoint,
    spinor_conjugate,
    step_holonomy,
    su2_lie_exponential,
)
from nanolm_holonomy.triton_holonomy import holonomy_scan_fast
from nanolm_posttrain.architecture import (
    ABSOLUTE_MAX_PARAMETERS,
    ParameterBudgetExceededError,
    _apply_rope,
)


class HSPAFamily(str, Enum):
    B0_TRANSFORMER = "b0_transformer"
    B1_ABELIAN_SSM = "b1_abelian_ssm"
    GEN1_H1_HSPA_SPINOR = "gen1_h1_hspa_spinor"
    GEN1_H2_HSPA_PREDCODE = "gen1_h2_hspa_predcode"
    GEN2_H3_HSPA_EVOLVED = "gen2_h3_hspa_evolved"
    GEN3_H4_HSPA_WAVEPRED = "gen3_h4_hspa_wavepred"
    GEN4_OMEGA_SO3_PREDCODE = "gen4_omega_so3_predcode"


@dataclass(frozen=True, slots=True)
class HSPAArchConfig:
    """Configuration across HSPA architectures and baselines (strict <= 4B ceiling)."""

    name: str
    family: HSPAFamily
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int
    n_registers: int = 8
    d_fiber: int = 128  # Must be divisible by 4 (quaternions) and 2 (complex)
    pc_iters: int = 2
    conv_kernel: int = 4
    max_seq_len: int = 16_384
    rope_theta: float = 500_000.0
    rms_eps: float = 1e-6
    scan_chunk_size: int = 512

    def __post_init__(self) -> None:
        for fn, val in (
            ("vocab_size", self.vocab_size),
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("d_head", self.d_head),
            ("d_ff", self.d_ff),
            ("n_registers", self.n_registers),
            ("d_fiber", self.d_fiber),
            ("max_seq_len", self.max_seq_len),
        ):
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"{fn} must be a positive integer")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("n_heads * d_head must equal d_model")
        if self.d_head % 2 != 0:
            raise ValueError("d_head must be even for RoPE")
        if self.d_fiber % 4 != 0:
            raise ValueError("d_fiber must be divisible by 4 for quaternionic spinor fibers")
        if self.parameter_count > ABSOLUTE_MAX_PARAMETERS:
            raise ParameterBudgetExceededError(
                f"{self.name} has {self.parameter_count:,} params > 4B limit"
            )

    @property
    def n_spinors(self) -> int:
        return self.d_fiber // 4

    @property
    def n_complex_fibers(self) -> int:
        return self.d_fiber // 2

    @property
    def is_transformer(self) -> bool:
        return self.family == HSPAFamily.B0_TRANSFORMER

    @property
    def is_abelian(self) -> bool:
        return self.family == HSPAFamily.B1_ABELIAN_SSM

    @property
    def uses_predcode(self) -> bool:
        return self.family in (
            HSPAFamily.GEN1_H2_HSPA_PREDCODE,
            HSPAFamily.GEN3_H4_HSPA_WAVEPRED,
            HSPAFamily.GEN4_OMEGA_SO3_PREDCODE,
        )

    @property
    def uses_gen2(self) -> bool:
        return self.family == HSPAFamily.GEN2_H3_HSPA_EVOLVED

    @property
    def uses_gen3(self) -> bool:
        return self.family == HSPAFamily.GEN3_H4_HSPA_WAVEPRED

    @property
    def uses_gen4(self) -> bool:
        return self.family == HSPAFamily.GEN4_OMEGA_SO3_PREDCODE

    @property
    def uses_wave_conv(self) -> bool:
        return self.uses_gen2 or self.uses_gen3

    @property
    def parameter_count(self) -> int:
        d = self.d_model
        total = self.vocab_size * d + d  # tied tok_emb + final_norm
        R = self.n_registers
        G = self.n_spinors
        C_comp = self.n_complex_fibers
        K = self.conv_kernel

        for _ in range(self.n_layers):
            norms = 2 * d
            if self.is_transformer:
                mix = (
                    d * (self.n_heads * self.d_head)
                    + 2 * d * (self.n_kv_heads * self.d_head)
                    + (self.n_heads * self.d_head) * d
                    + 2 * self.d_head
                )
            elif self.is_abelian:
                mix = (
                    d * self.d_fiber  # phi_proj
                    + d * (4 * R)  # write, beta, mobility, read
                    + d * (3 * C_comp)  # theta_tok, rho_tok, q_tok (scalars per complex fiber)
                    + 3 * R * C_comp  # theta_reg, rho_reg, q_reg
                    + d * (3 * R)  # theta_mod, rho_mod, q_mod
                    + self.d_fiber * d  # o_proj
                    + self.d_fiber  # out_norm
                )
            else:
                mix = (
                    d * self.d_fiber  # phi_proj
                    + d * (4 * R)  # write, beta, mobility, read
                    + d * (9 * G)  # omega_tok (3*G), rho_tok (3*G), q_tok (3*G)
                    + 3 * R * (3 * G)  # omega_reg, rho_reg, q_reg
                    + d * (3 * R)  # omega_mod, rho_mod, q_mod
                    + self.d_fiber * d  # o_proj
                    + self.d_fiber  # out_norm
                )
                if self.uses_gen2:
                    # Causal wavepacket pre-filter on phi + Kuramoto cross-register coupling K
                    mix += self.d_fiber * K + R * R + self.d_fiber
                elif self.uses_gen3:
                    # Causal wavepacket pre-filter on phi + bifurcated unitary/episodic register biases
                    mix += self.d_fiber * K + 2 * R
                elif self.uses_gen4:
                    # SO(3) projective adjoint projection + unitary/episodic register biases + scalar highway gate
                    mix += (9 * G) * self.d_fiber + 2 * R + 1

            if self.uses_predcode:
                ffn = 3 * d * self.d_ff + d + 1  # W_enc, W_dec, W_err + err_norm + log_prec
            else:
                ffn = 3 * d * self.d_ff
            total += norms + mix + ffn
        return total

    def state_bytes_per_sequence(self, seq_len: int, bytes_per_elem: int = 2) -> int:
        """Exact runtime state memory during inference at sequence length ``seq_len``."""
        if self.is_transformer:
            return 2 * self.n_layers * self.n_kv_heads * self.d_head * bytes_per_elem * seq_len
        # Constant O(1) persistent state regardless of seq_len: [R, d_fiber] (plus conv ring (k-1) for wave conv)
        base = self.n_layers * self.n_registers * self.d_fiber * bytes_per_elem
        if self.uses_wave_conv:
            base += self.n_layers * (self.conv_kernel - 1) * self.d_fiber * bytes_per_elem
        return base

    def parameter_breakdown(self, sample_seq_len: int = 8_192) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family.value,
            "total_parameters": self.parameter_count,
            "within_4b_ceiling": self.parameter_count <= ABSOLUTE_MAX_PARAMETERS,
            "n_registers": self.n_registers,
            "d_fiber": self.d_fiber,
            "state_bytes_at_seq": self.state_bytes_per_sequence(sample_seq_len, 2),
            "sample_seq_len": sample_seq_len,
        }


def _iso_match_dff(target_params: int, base_cfg: HSPAArchConfig) -> HSPAArchConfig:
    best = base_cfg
    best_gap = abs(base_cfg.parameter_count - target_params)
    for d_ff in range(max(64, base_cfg.d_ff - 1024), base_cfg.d_ff + 1024, 16):
        try:
            cand = replace(base_cfg, d_ff=d_ff)
        except ParameterBudgetExceededError:
            continue
        gap = abs(cand.parameter_count - target_params)
        if gap < best_gap:
            best_gap = gap
            best = cand
    return best


def build_suite_configs(
    *,
    preset: str = "micro",
    vocab_size: int = 4_096,
    max_seq_len: int = 2_048,
) -> tuple[HSPAArchConfig, ...]:
    """Build tightly iso-parameter suite across B0, B1, Gen1_H1, Gen1_H2, Gen2_H3."""
    if preset == "small_x1":
        d_model, n_layers, n_heads, n_kv, d_head, d_ff, n_reg, d_fib = 192, 6, 6, 2, 32, 512, 8, 128
    elif preset == "micro":
        d_model, n_layers, n_heads, n_kv, d_head, d_ff, n_reg, d_fib = 384, 8, 8, 2, 48, 1024, 8, 192
    else:
        raise ValueError(f"unknown preset: {preset}")

    b0 = HSPAArchConfig(
        name=f"b0-transformer-{preset}",
        family=HSPAFamily.B0_TRANSFORMER,
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        d_head=d_head,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
    )
    target = b0.parameter_count
    variants = (
        ("b1-abelian-ssm", HSPAFamily.B1_ABELIAN_SSM),
        ("gen1-h1-hspa-spinor", HSPAFamily.GEN1_H1_HSPA_SPINOR),
        ("gen1-h2-hspa-predcode", HSPAFamily.GEN1_H2_HSPA_PREDCODE),
        ("gen2-h3-hspa-evolved", HSPAFamily.GEN2_H3_HSPA_EVOLVED),
        ("gen3-h4-hspa-wavepred", HSPAFamily.GEN3_H4_HSPA_WAVEPRED),
        ("gen4-omega-so3-predcode", HSPAFamily.GEN4_OMEGA_SO3_PREDCODE),
    )
    out = [b0]
    for vname, fam in variants:
        base = HSPAArchConfig(
            name=f"{vname}-{preset}",
            family=fam,
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            d_head=d_head,
            d_ff=d_ff,
            n_registers=n_reg,
            d_fiber=d_fib,
            max_seq_len=max_seq_len,
        )
        out.append(_iso_match_dff(target, base))
    return tuple(out)


# ~2B Target and ~3.85B <= 4B Flagship Scaling Configurations (Gen-3 WavePred + SU(2) Triton)
HSPA_2B_CONFIG = HSPAArchConfig(
    name="nanolm-hspa-2b",
    family=HSPAFamily.GEN3_H4_HSPA_WAVEPRED,
    vocab_size=100_352,
    d_model=2_048,
    n_layers=32,
    n_heads=16,
    n_kv_heads=4,
    d_head=128,
    d_ff=6_656,
    n_registers=16,
    d_fiber=1_024,
    max_seq_len=65_536,
)

TRANSFORMER_4B_CONFIG = HSPAArchConfig(
    name="nanolm-transformer-4b",
    family=HSPAFamily.B0_TRANSFORMER,
    vocab_size=100_352,
    d_model=2_560,
    n_layers=48,
    n_heads=20,
    n_kv_heads=4,
    d_head=128,
    d_ff=7_680,
    max_seq_len=65_536,
)

HSPA_4B_CONFIG = _iso_match_dff(
    TRANSFORMER_4B_CONFIG.parameter_count,
    HSPAArchConfig(
        name="nanolm-hspa-4b",
        family=HSPAFamily.GEN3_H4_HSPA_WAVEPRED,
        vocab_size=100_352,
        d_model=2_560,
        n_layers=48,
        n_heads=20,
        n_kv_heads=4,
        d_head=128,
        d_ff=7_296,
        n_registers=16,
        d_fiber=1_024,
        max_seq_len=65_536,
    ),
)

HSPA_OMEGA_2B_CONFIG = HSPAArchConfig(
    name="nanolm-hspa-omega-2b",
    family=HSPAFamily.GEN4_OMEGA_SO3_PREDCODE,
    vocab_size=100_352,
    d_model=2_048,
    n_layers=32,
    n_heads=16,
    n_kv_heads=4,
    d_head=128,
    d_ff=6_272,
    n_registers=16,
    d_fiber=1_024,
    max_seq_len=65_536,
)

HSPA_OMEGA_4B_CONFIG = _iso_match_dff(
    TRANSFORMER_4B_CONFIG.parameter_count,
    HSPAArchConfig(
        name="nanolm-hspa-omega-4b",
        family=HSPAFamily.GEN4_OMEGA_SO3_PREDCODE,
        vocab_size=100_352,
        d_model=2_560,
        n_layers=48,
        n_heads=20,
        n_kv_heads=4,
        d_head=128,
        d_ff=6_976,
        n_registers=16,
        d_fiber=1_024,
        max_seq_len=65_536,
    ),
)


class PredictiveCodingCell(nn.Module):
    """Iterative Precision-Weighted Free-Energy Predictive-Coding FFN Cell."""

    def __init__(self, d_model: int, d_pc: int, *, iters: int = 2, rms_eps: float = 1e-6) -> None:
        super().__init__()
        self.iters = iters
        self.w_enc = nn.Linear(d_model, d_pc, bias=False)
        self.w_dec = nn.Linear(d_pc, d_model, bias=False)
        self.w_err = nn.Linear(d_model, d_pc, bias=False)
        self.err_norm = nn.RMSNorm(d_model, eps=rms_eps)
        self.log_prec = nn.Parameter(torch.zeros(1))

    def forward(self, z: Tensor) -> Tensor:
        c = F.silu(self.w_enc(z))
        for _ in range(self.iters):
            pred = self.w_dec(c)
            err = z - pred
            err_mag = torch.sqrt(torch.mean(err.float() ** 2, dim=-1, keepdim=True) + 1e-6)
            step = torch.sigmoid(self.log_prec.float() + torch.log(err_mag)).to(z.dtype)
            c = c + step * F.silu(self.w_err(self.err_norm(err)))
        return self.w_dec(c)


class _TransformerMixer(nn.Module):
    def __init__(self, cfg: HSPAArchConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(cfg.d_head, eps=cfg.rms_eps)
        self.k_norm = nn.RMSNorm(cfg.d_head, eps=cfg.rms_eps)

    def forward(
        self,
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        state: tuple[Tensor, Tensor] | None = None,
        use_state: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        bsz, seq_len, _ = h.shape
        q = self.q_norm(self.q_proj(h).view(bsz, seq_len, self.n_heads, self.d_head)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(bsz, seq_len, self.n_kv_heads, self.d_head)).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if state is not None:
            k = torch.cat((state[0], k), dim=2)
            v = torch.cat((state[1], v), dim=2)
        next_state = (k, v) if use_state else None

        k_exp, v_exp = k, v
        if self.n_heads != self.n_kv_heads:
            rep = self.n_heads // self.n_kv_heads
            k_exp = k.repeat_interleave(rep, dim=1)
            v_exp = v.repeat_interleave(rep, dim=1)

        if state is None:
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        elif seq_len == 1:
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)
        else:
            tot = k_exp.shape[2]
            mask = torch.ones((seq_len, tot), dtype=torch.bool, device=q.device).tril(diagonal=tot - seq_len)
            attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=mask)

        out = self.o_proj(attn_out.transpose(1, 2).reshape(bsz, seq_len, self.n_heads * self.d_head))
        return out, next_state


class _AbelianRegisterMixer(nn.Module):
    """Exact U(1) Commutative Control matching SU(2) HSPA structure and state bytes."""

    def __init__(self, cfg: HSPAArchConfig) -> None:
        super().__init__()
        self.R = cfg.n_registers
        self.C = cfg.n_complex_fibers
        self.d_fiber = cfg.d_fiber
        self.chunk = cfg.scan_chunk_size

        self.phi_proj = nn.Linear(cfg.d_model, self.d_fiber, bias=False)
        self.gate_proj = nn.Linear(cfg.d_model, 4 * self.R, bias=False)
        self.tok_angles = nn.Linear(cfg.d_model, 3 * self.C, bias=False)
        self.reg_mod = nn.Linear(cfg.d_model, 3 * self.R, bias=False)

        self.theta_reg = nn.Parameter(torch.randn(self.R, self.C, 1) * 0.25)
        self.rho_reg = nn.Parameter(torch.randn(self.R, self.C, 1) * 0.25)
        self.q_reg = nn.Parameter(torch.randn(self.R, self.C, 1) * 0.25)

        self.out_norm = nn.RMSNorm(self.d_fiber, eps=cfg.rms_eps)
        self.o_proj = nn.Linear(self.d_fiber, cfg.d_model, bias=False)

    def forward(
        self,
        h: Tensor,
        _cos: Tensor,
        _sin: Tensor,
        *,
        state: Tensor | None = None,
        use_state: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        bsz, seq_len, _ = h.shape
        phi = self.phi_proj(h).view(bsz, seq_len, self.C, 2).permute(0, 2, 1, 3).unsqueeze(1)  # [B,1,C,T,2]
        w_l, b_l, m_l, r_l = self.gate_proj(h).chunk(4, dim=-1)  # each [B, T, R]
        w = F.softmax(w_l, dim=-1).transpose(1, 2).unsqueeze(2).unsqueeze(-1)  # [B, R, 1, T, 1]
        beta = torch.sigmoid(b_l).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        mob = torch.sigmoid(m_l).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        read_pi = F.softmax(r_l, dim=-1).transpose(1, 2).unsqueeze(-1)  # [B, R, T, 1]

        t_tok, r_tok, q_tok = self.tok_angles(h).chunk(3, dim=-1)  # each [B, T, C]
        tm, rm, qm = self.reg_mod(h).chunk(3, dim=-1)  # [B, T, R]

        theta = mob * (
            t_tok.transpose(1, 2).unsqueeze(1).unsqueeze(-1) * torch.tanh(tm).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
            + self.theta_reg.view(1, self.R, self.C, 1, 1)
        )
        rho = (
            r_tok.transpose(1, 2).unsqueeze(1).unsqueeze(-1) * torch.tanh(rm).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
            + self.rho_reg.view(1, self.R, self.C, 1, 1)
        )
        q_ang = (
            q_tok.transpose(1, 2).unsqueeze(1).unsqueeze(-1) * torch.tanh(qm).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
            + self.q_reg.view(1, self.R, self.C, 1, 1)
        )

        decay = (1.0 - w * beta).clamp_min(1e-4)
        g = decay * complex_cayley_rotor(theta)
        s = w * complex_product(complex_cayley_rotor(rho), phi)

        all_omega, final_state = abelian_complex_scan(g, s, init_state=state, chunk_size=self.chunk)
        q_rot = complex_cayley_rotor(q_ang)
        unbound = complex_product(complex_conjugate(q_rot), all_omega)  # [B, R, C, T, 2]
        unbound_flat = unbound.permute(0, 1, 3, 2, 4).reshape(bsz, self.R, seq_len, self.d_fiber)
        readout = (read_pi * unbound_flat).sum(dim=1)  # [B, T, d_fiber]
        out = self.o_proj(self.out_norm(readout))
        return out, (final_state if use_state else None)


class SpinorRegisterField(nn.Module):
    """Non-Abelian SU(2) Parallel-Transport Spinor Field (Gen 1, Gen 2, Gen 3, Gen 4 Omega)."""

    def __init__(self, cfg: HSPAArchConfig) -> None:
        super().__init__()
        self.R = cfg.n_registers
        self.G = cfg.n_spinors
        self.d_fiber = cfg.d_fiber
        self.chunk = cfg.scan_chunk_size
        self.use_gen2 = cfg.uses_gen2
        self.use_gen3 = cfg.uses_gen3
        self.use_gen4 = cfg.uses_gen4
        self.use_conv = cfg.uses_wave_conv
        self.conv_k = cfg.conv_kernel

        self.phi_proj = nn.Linear(cfg.d_model, self.d_fiber, bias=False)
        self.gate_proj = nn.Linear(cfg.d_model, 4 * self.R, bias=False)
        self.tok_spinors = nn.Linear(cfg.d_model, 9 * self.G, bias=False)
        self.reg_mod = nn.Linear(cfg.d_model, 3 * self.R, bias=False)

        self.omega_reg = nn.Parameter(torch.randn(self.R, self.G, 3) * 0.25)
        self.rho_reg = nn.Parameter(torch.randn(self.R, self.G, 3) * 0.25)
        self.q_reg = nn.Parameter(torch.randn(self.R, self.G, 3) * 0.25)

        if self.use_conv:
            self.wave_conv_weight = nn.Parameter(torch.randn(self.d_fiber, 1, self.conv_k) * 0.02)
        if self.use_gen2:
            self.kuramoto_coupling = nn.Parameter(torch.zeros(self.R, self.R))
            self.kuramoto_scale = nn.Parameter(torch.zeros(self.d_fiber))
        if self.use_gen3:
            # Bifurcated connection initialization: first R//2 near unitary transport (long-horizon),
            # second R//2 fast-refresh episodic binding slots.
            half_r = max(1, self.R // 2)
            beta_init = torch.cat((torch.full((half_r,), -3.5), torch.full((self.R - half_r,), 1.0)))
            mob_init = torch.cat((torch.full((half_r,), 2.0), torch.full((self.R - half_r,), -2.0)))
            self.beta_reg_bias = nn.Parameter(beta_init)
            self.mob_reg_bias = nn.Parameter(mob_init)
        if self.use_gen4:
            half_r = max(1, self.R // 2)
            beta_init = torch.cat((torch.full((half_r,), -4.5), torch.full((self.R - half_r,), 0.5)))
            mob_init = torch.cat((torch.full((half_r,), 3.0), torch.zeros(self.R - half_r)))
            self.beta_reg_bias = nn.Parameter(beta_init)
            self.mob_reg_bias = nn.Parameter(mob_init)
            self.so3_read_proj = nn.Linear(9 * self.G, self.d_fiber, bias=False)
            nn.init.normal_(self.so3_read_proj.weight, std=1.0 / math.sqrt(9 * self.G))
            with torch.no_grad():
                self.omega_reg[:half_r].zero_()

        self.out_norm = nn.RMSNorm(self.d_fiber, eps=cfg.rms_eps)
        self.o_proj = nn.Linear(self.d_fiber, cfg.d_model, bias=False)

    def _causal_conv(self, x: Tensor, conv_buf: Tensor | None) -> tuple[Tensor, Tensor]:
        xt = x.transpose(1, 2)
        if conv_buf is None:
            pad_xt = F.pad(xt, (self.conv_k - 1, 0))
        else:
            pad_xt = torch.cat((conv_buf, xt), dim=-1)
        new_buf = pad_xt[..., -(self.conv_k - 1) :].detach()
        y = F.conv1d(pad_xt, self.wave_conv_weight.to(x.dtype), groups=self.d_fiber)
        return y.transpose(1, 2).contiguous(), new_buf

    def forward(
        self,
        h: Tensor,
        _cos: Tensor,
        _sin: Tensor,
        *,
        state: Any = None,
        use_state: bool = False,
    ) -> tuple[Tensor, Any]:
        bsz, seq_len, _ = h.shape
        if self.use_conv and isinstance(state, tuple):
            scan_state, conv_buf = state
        else:
            scan_state, conv_buf = state, None

        phi_raw = self.phi_proj(h)
        if self.use_conv:
            phi_conv, new_conv_buf = self._causal_conv(phi_raw, conv_buf)
            phi_raw = phi_raw + F.silu(phi_conv)
        else:
            new_conv_buf = None

        phi = phi_raw.view(bsz, seq_len, self.G, 4).permute(0, 2, 1, 3).unsqueeze(1)  # [B, 1, G, T, 4]
        w_l, b_l, m_l, r_l = self.gate_proj(h).chunk(4, dim=-1)
        if self.use_gen3 or self.use_gen4:
            b_l = b_l + self.beta_reg_bias.to(b_l.dtype).view(1, 1, self.R)
            m_l = m_l + self.mob_reg_bias.to(m_l.dtype).view(1, 1, self.R)
        w = F.softmax(w_l, dim=-1).transpose(1, 2).unsqueeze(2).unsqueeze(-1)  # [B, R, 1, T, 1]
        beta = torch.sigmoid(b_l).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        mob = torch.sigmoid(m_l).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        read_pi = F.softmax(r_l, dim=-1).transpose(1, 2).unsqueeze(-1)  # [B, R, T, 1]

        o_tok, rho_tok, q_tok = self.tok_spinors(h).chunk(3, dim=-1)  # [B, T, 3*G]
        o_tok = o_tok.view(bsz, seq_len, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1)
        rho_tok = rho_tok.view(bsz, seq_len, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1)
        q_tok = q_tok.view(bsz, seq_len, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1)

        om, rm, qm = self.reg_mod(h).chunk(3, dim=-1)  # [B, T, R]
        om = torch.tanh(om).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        rm = torch.tanh(rm).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        qm = torch.tanh(qm).transpose(1, 2).unsqueeze(2).unsqueeze(-1)

        if self.use_gen4:
            half_r = max(1, self.R // 2)
            omega = mob * (o_tok + om * self.omega_reg.to(o_tok.dtype).view(1, self.R, self.G, 1, 3))
            rho = rho_tok + rm * self.rho_reg.to(rho_tok.dtype).view(1, self.R, self.G, 1, 3)
            q_gen = q_tok + qm * self.q_reg.to(q_tok.dtype).view(1, self.R, self.G, 1, 3)
            # Exact unitary/episodic fiber split: zero decay/additive drift on holonomy half [:half_r]
            g_holo = su2_lie_exponential(omega[:, :half_r]).to(h.dtype)
            s_holo = torch.zeros(bsz, half_r, self.G, seq_len, 4, device=h.device, dtype=h.dtype)
            decay_ep = (1.0 - w[:, half_r:] * beta[:, half_r:]).clamp_min(1e-4)
            g_ep = (decay_ep * su2_lie_exponential(omega[:, half_r:])).to(h.dtype)
            s_ep = (w[:, half_r:] * hamilton_product(su2_lie_exponential(rho[:, half_r:]), phi)).to(h.dtype)
            g = torch.cat((g_holo, g_ep), dim=1)
            s = torch.cat((s_holo, s_ep), dim=1)
            if scan_state is None:
                scan_state = torch.zeros(bsz, self.R, self.G, 4, device=h.device, dtype=h.dtype)
                scan_state[:, :half_r, :, 0] = 1.0
        else:
            omega = mob * (o_tok * om + self.omega_reg.view(1, self.R, self.G, 1, 3))
            rho = rho_tok * rm + self.rho_reg.view(1, self.R, self.G, 1, 3)
            q_gen = q_tok * qm + self.q_reg.view(1, self.R, self.G, 1, 3)

            decay = (1.0 - w * beta).clamp_min(1e-4)
            g = decay * cayley_rotor(omega)  # [B, R, G, T, 4]
            s = w * hamilton_product(cayley_rotor(rho), phi)

        if seq_len == 1 and scan_state is not None:
            next_scan = step_holonomy(g.squeeze(-2), s.squeeze(-2), scan_state)
            all_omega = next_scan.unsqueeze(-2)
        else:
            all_omega, next_scan = holonomy_scan_fast(g, s, init_state=scan_state, chunk_size=self.chunk)

        if self.use_gen4:
            half_r = max(1, self.R // 2)
            # Strictly Z2-invariant SO(3) projective adjoint on holonomy bundle (zero +/-q leak)
            adj_holo = (
                so3_projective_adjoint(all_omega[:, :half_r])
                .permute(0, 1, 3, 2, 4)
                .reshape(bsz, half_r, seq_len, 9 * self.G)
            )
            holo_read = self.so3_read_proj(adj_holo.to(h.dtype))
            ep_q = su2_lie_exponential(q_gen[:, half_r:])
            ep_unbound = hamilton_product(spinor_conjugate(ep_q.to(all_omega.dtype)), all_omega[:, half_r:])
            ep_read = ep_unbound.permute(0, 1, 3, 2, 4).reshape(bsz, self.R - half_r, seq_len, self.d_fiber).to(h.dtype)
            unbound_flat = torch.cat((holo_read, ep_read), dim=1)
        else:
            q_rot = cayley_rotor(q_gen)
            unbound = hamilton_product(spinor_conjugate(q_rot), all_omega)  # [B, R, G, T, 4]
            unbound_flat = unbound.permute(0, 1, 3, 2, 4).reshape(bsz, self.R, seq_len, self.d_fiber)
            if self.use_gen2:
                reg_norm = F.normalize(unbound_flat.float(), dim=-1).to(unbound_flat.dtype)
                coup = torch.tanh(self.kuramoto_coupling).to(unbound_flat.dtype)  # [R, R]
                sync_drive = torch.einsum("rs,bstf->brtf", coup, reg_norm)
                unbound_flat = unbound_flat + torch.tanh(self.kuramoto_scale).to(unbound_flat.dtype) * sync_drive

        readout = (read_pi * unbound_flat).sum(dim=1)  # [B, T, d_fiber]
        out = self.o_proj(self.out_norm(readout))

        if not use_state:
            return out, None
        ret_state = (next_scan, new_conv_buf) if self.use_conv else next_scan
        return out, ret_state


class _UnifiedBlock(nn.Module):
    def __init__(self, cfg: HSPAArchConfig, *, layer_idx: int = 0) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)
        if cfg.is_transformer:
            self.mixer: nn.Module = _TransformerMixer(cfg)
        elif cfg.is_abelian:
            self.mixer = _AbelianRegisterMixer(cfg)
        else:
            self.mixer = SpinorRegisterField(cfg)

        self.res_mix_gate: nn.Parameter | None = (
            nn.Parameter(torch.tensor(1.0 if layer_idx == 0 else 0.0))
            if cfg.uses_gen4
            else None
        )
        self.uses_predcode = cfg.uses_predcode
        if self.uses_predcode:
            self.ffn: nn.Module = PredictiveCodingCell(cfg.d_model, cfg.d_ff, iters=cfg.pc_iters, rms_eps=cfg.rms_eps)
        else:
            self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
            self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        state: Any = None,
        use_state: bool = False,
    ) -> tuple[Tensor, Any]:
        h_mix, next_state = self.mixer(self.attn_norm(x), cos, sin, state=state, use_state=use_state)
        if self.res_mix_gate is not None:
            x = x + self.res_mix_gate.to(h_mix.dtype) * h_mix
        else:
            x = x + h_mix
        h2 = self.ffn_norm(x)
        if self.uses_predcode:
            return x + self.ffn(h2), next_state
        return x + self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2)), next_state


class HSPAUnifiedLM(nn.Module):
    """Unified Causal Language Model across Transformer, Abelian SSM, and SU(2) HSPA."""

    def __init__(self, cfg: HSPAArchConfig) -> None:
        super().__init__()
        self.config = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(_UnifiedBlock(cfg, layer_idx=idx) for idx in range(cfg.n_layers))
        self.final_norm = nn.RMSNorm(cfg.d_model, eps=cfg.rms_eps)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope_cache(self, seq_len: int, past_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if not self.config.is_transformer:
            empty = torch.empty((1,), device=device, dtype=dtype)
            return empty, empty
        pos = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, self.config.d_head, 2, device=device, dtype=torch.float32) / self.config.d_head)
        )
        freqs = torch.outer(pos, inv_freq)
        return freqs.cos()[None, None].to(dtype), freqs.sin()[None, None].to(dtype)

    def forward(
        self,
        input_ids: Tensor,
        *,
        past_states: list[Any] | None = None,
        use_state: bool = False,
        past_len: int = 0,
        logits_to_keep: int | None = None,
    ) -> Tensor | tuple[Tensor, list[Any]]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, seq_len >= 1]")
        seq_len = input_ids.shape[1]
        if past_len + seq_len > self.config.max_seq_len:
            raise ValueError(f"total seq_len {past_len + seq_len} exceeds {self.config.max_seq_len}")

        h = self.tok_emb(input_ids)
        cos, sin = self._rope_cache(seq_len, past_len, input_ids.device, h.dtype)
        next_states: list[Any] = []
        for idx, block in enumerate(self.blocks):
            st = past_states[idx] if past_states is not None else None
            h, nxt = block(h, cos, sin, state=st, use_state=use_state)
            if use_state:
                next_states.append(nxt)
        h_sel = h[:, -logits_to_keep:, :] if logits_to_keep is not None else h
        logits = F.linear(self.final_norm(h_sel), self.tok_emb.weight)
        return (logits, next_states) if use_state else logits
