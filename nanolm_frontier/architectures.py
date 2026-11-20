"""Competing NanoLM Frontier Architectures (<= 4B Parameter Ceiling, ~2B Target).

Implements five iso-interface causal language model families spanning modern
attention, preconditioned linear-recurrence, sparse product-key cortical memory,
and contractive prelude-anchored latent equilibrium reasoning:

- ``C0_DenseGQA``: Tied-embedding GQA + QK-Norm + RoPE + SwiGLU baseline.
- ``C1_GatedDeltaHybrid``: 3:1 Gated Delta-Rule (ShortConv + rank-1 Householder
  erasure) interleaved with global GQA softmax layers.
- ``C2_DiffNSDelta``: Diagonal Gauss-Newton key-whitened Delta-Rule +
  Differential Noise-Canceling Softmax Attention (``DiffGQA``).
- ``C3_CorticalPKM``: ``C2_DiffNSDelta`` augmented with Cartesian Product-Key
  Sparse Cortical Memory (M = C^2 slots, O(sqrt(M) * d) active routing flops).
- ``C4_JCLE_Frontier``: Prelude-Core-Coda contractive damped equilibrium loop
  with prelude injection over ``C3_CorticalPKM`` and test-time Shannon entropy
  early-exit halting.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_posttrain.architecture import (
    ABSOLUTE_MAX_PARAMETERS,
    ParameterBudgetExceededError,
    _apply_rope,
)


class ArchitectureFamily(str, Enum):
    C0_DENSE_GQA = "c0_dense_gqa"
    C1_GATED_DELTA_HYBRID = "c1_gated_delta_hybrid"
    C2_DIFF_NS_DELTA = "c2_diff_ns_delta"
    C3_CORTICAL_PKM = "c3_cortical_pkm"
    C4_JCLE_FRONTIER = "c4_jcle_frontier"


def _causal_depthwise_conv(x: Tensor, weight: Tensor) -> Tensor:
    """Apply depthwise causal 1D conv along sequence dimension.

    Args:
        x: ``[batch, seq_len, channels]``
        weight: ``[channels, 1, kernel_size]``
    """
    bsz, seq_len, channels = x.shape
    kernel_size = weight.shape[-1]
    x_t = x.transpose(1, 2)  # [B, C, T]
    x_pad = F.pad(x_t, (kernel_size - 1, 0))
    y = F.conv1d(x_pad, weight, bias=None, groups=channels)
    return y.transpose(1, 2).contiguous()


def chunked_ema_key_energy(
    k: Tensor,
    gamma: Tensor,
    *,
    chunk_size: int = 32,
    eps: float = 1e-4,
) -> Tensor:
    """Compute causal EMA key energy c_t = gamma * c_{t-1} + (1 - gamma) * (k_t^2).

    Runs in O(T / chunk_size) vectorized chunk matrix-multiplications in FP32 so it
    scales cleanly without token-by-token Python overhead.

    Args:
        k: ``[batch, n_heads, seq_len, d_head]``
        gamma: ``[n_heads]`` decay in ``(0, 1)``
    Returns:
        Whitened denominator ``sqrt(c_t + eps)`` in FP32 with same shape as ``k``.
    """
    bsz, n_heads, seq_len, d_head = k.shape
    device = k.device
    k_sq = (k.float() ** 2).contiguous()  # [B, H, T, D]
    g = gamma.float().clamp(1e-4, 1.0 - 1e-4).view(1, n_heads, 1, 1)
    one_minus_g = 1.0 - g

    out_chunks: list[Tensor] = []
    # Initialize state around unit mean squared magnitude per dim (1 / d_head)
    c_prev = torch.full((bsz, n_heads, 1, d_head), 1.0 / float(d_head), device=device, dtype=torch.float32)

    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        blk = end - start
        k_blk = k_sq[:, :, start:end, :]  # [B, H, L, D]
        idx = torch.arange(blk, device=device, dtype=torch.float32)
        # W[i, j] = (1 - g) * g^(i - j) for i >= j, else 0 -> [H, L, L]
        diff = idx[:, None] - idx[None, :]
        mask = diff >= 0
        g_h = gamma.float().clamp(1e-4, 1.0 - 1e-4)[:, None, None]
        # compute in log-space for safety
        log_g = torch.log(g_h)
        weights = torch.where(
            mask[None, :, :],
            (1.0 - g_h) * torch.exp(torch.clamp(diff[None, :, :] * log_g, min=-40.0, max=0.0)),
            torch.zeros(1, blk, blk, device=device, dtype=torch.float32),
        )
        # c_prev decay powers: g^(i + 1) -> [H, L, 1]
        init_decay = torch.exp(torch.clamp((idx[None, :, None] + 1.0) * log_g, min=-40.0, max=0.0))
        c_blk = init_decay[None, :, :, :] * c_prev + torch.einsum("hij,bhjd->bhid", weights, k_blk)
        out_chunks.append(c_blk)
        c_prev = c_blk[:, :, -1:, :]

    c_all = torch.cat(out_chunks, dim=2)
    return torch.sqrt(c_all + eps)


def reference_recurrent_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    beta: Tensor,
) -> Tensor:
    """Exact token-by-token reference Gated Delta-Rule recurrence (used to verify chunkwise solver).

    Recurrence per head:
        S_t = a_t * S_{t-1} @ (I - beta_t k_t k_t^T) + beta_t * v_t @ k_t^T
        o_t = S_t @ q_t
    Shapes:
        q, k: ``[B, H, T, D_k]`` (caller scales/normalizes k as desired)
        v: ``[B, H, T, D_v]``
        a, beta: ``[B, H, T]``
    """
    bsz, n_heads, seq_len, d_k = k.shape
    d_v = v.shape[-1]
    state = torch.zeros((bsz, n_heads, d_v, d_k), device=q.device, dtype=torch.float32)
    outputs: list[Tensor] = []
    qf, kf, vf = q.float(), k.float(), v.float()
    af, bf = a.float(), beta.float()
    for t in range(seq_len):
        k_t = kf[:, :, t, :]  # [B, H, D_k]
        v_t = vf[:, :, t, :]  # [B, H, D_v]
        q_t = qf[:, :, t, :]  # [B, H, D_k]
        a_t = af[:, :, t].unsqueeze(-1).unsqueeze(-1)
        b_t = bf[:, :, t].unsqueeze(-1).unsqueeze(-1)
        s_decayed = a_t * state
        # Memory readout along k_t: [B, H, D_v]
        pred_v = torch.einsum("bhvk,bhk->bhv", s_decayed, k_t)
        u_t = b_t.squeeze(-1) * (v_t - pred_v)
        state = s_decayed + torch.einsum("bhv,bhk->bhvk", u_t, k_t)
        o_t = torch.einsum("bhvk,bhk->bhv", state, q_t)
        outputs.append(o_t)
    return torch.stack(outputs, dim=2).to(q.dtype)


def chunkwise_gated_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    log_a: Tensor,
    beta: Tensor,
    *,
    chunk_size: int = 32,
) -> Tensor:
    """Exact WY-representation chunkwise parallel Gated Delta-Rule solver.

    Solves intra-chunk Householder coupling via batched unit lower-triangular solve
    (``torch.linalg.solve_triangular``) in FP32 while running parallel over chunks of length L.
    Matches ``reference_recurrent_gated_delta_rule`` mathematically up to FP rounding.
    """
    bsz, n_heads, seq_len, d_k = k.shape
    d_v = v.shape[-1]
    qf, kf, vf = q.float(), k.float(), v.float()
    log_af = log_a.float().clamp(-6.0, -1e-6)
    bf = beta.float()

    state = torch.zeros((bsz, n_heads, d_v, d_k), device=q.device, dtype=torch.float32)
    outputs: list[Tensor] = []

    for start in range(0, seq_len, chunk_size):
        end = min(seq_len, start + chunk_size)
        blk = end - start
        qc = qf[:, :, start:end, :]  # [B, H, L, D_k]
        kc = kf[:, :, start:end, :]
        vc = vf[:, :, start:end, :]
        lac = log_af[:, :, start:end]  # [B, H, L]
        bc = bf[:, :, start:end]  # [B, H, L]

        cum_log_a = torch.cumsum(lac, dim=-1)  # [B, H, L], <= 0
        gamma = torch.exp(cum_log_a)  # [B, H, L]

        # Lower-triangular ratio matrix G_ratio[i, j] = exp(cum[i] - cum[j]) for i >= j
        log_diff = cum_log_a.unsqueeze(-1) - cum_log_a.unsqueeze(-2)  # [B, H, L, L]
        tril_inc = torch.tril(torch.ones((blk, blk), device=q.device, dtype=torch.bool))
        strict_tril = torch.tril(torch.ones((blk, blk), device=q.device, dtype=torch.bool), diagonal=-1)
        g_ratio = torch.where(tril_inc, torch.exp(log_diff.clamp(-40.0, 0.0)), torch.zeros_like(log_diff))
        g_strict = torch.where(strict_tril, torch.exp(log_diff.clamp(-40.0, 0.0)), torch.zeros_like(log_diff))

        kk = torch.matmul(kc, kc.transpose(-1, -2))  # [B, H, L, L]
        # Coupling M[i, j] = beta_i * (Gamma_i / Gamma_j) * (k_i^T k_j) for i > j
        M = bc.unsqueeze(-1) * g_strict * kk
        eye = torch.eye(blk, device=q.device, dtype=torch.float32)
        A = eye + M  # unit lower-triangular [B, H, L, L]

        # RHS: beta_i * (v_i - Gamma_i * S_prev k_i)
        s_k = torch.einsum("bhvk,bhlk->bhlv", state, kc)  # [B, H, L, D_v]
        rhs = bc.unsqueeze(-1) * (vc - gamma.unsqueeze(-1) * s_k)

        u_chunk = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)  # [B, H, L, D_v]

        # Compute outputs: o_i = Gamma_i (S_prev q_i) + sum_{j<=i} (Gamma_i/Gamma_j) * (q_i^T k_j) * u_j
        s_q = torch.einsum("bhvk,bhlk->bhlv", state, qc)
        qk = torch.matmul(qc, kc.transpose(-1, -2))  # [B, H, L, L]
        causal_weights = qk * g_ratio
        o_chunk = gamma.unsqueeze(-1) * s_q + torch.matmul(causal_weights, u_chunk)
        outputs.append(o_chunk)

        # Update chunk state: S_next = Gamma_{L-1} * S_prev + sum_{j=0}^{L-1} (Gamma_{L-1}/Gamma_j) u_j k_j^T
        end_decay = gamma[:, :, -1].unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]
        rel_to_end = g_ratio[:, :, -1, :]  # [B, H, L]
        u_scaled = u_chunk * rel_to_end.unsqueeze(-1)
        state = end_decay * state + torch.einsum("bhlv,bhlk->bhvk", u_scaled, kc)

    return torch.cat(outputs, dim=2).to(q.dtype)


@dataclass(frozen=True, slots=True)
class FrontierArchConfig:
    """Configuration across competing <= 4B NanoLM Frontier architectures."""

    name: str
    family: ArchitectureFamily
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    d_ff: int
    max_seq_len: int = 8_192
    rope_theta: float = 500_000.0
    rms_eps: float = 1e-6
    delta_every_n: int = 4  # 3 linear/delta layers followed by 1 global softmax layer
    short_conv_kernel: int = 4
    delta_chunk_size: int = 32
    pkm_side: int = 64  # M = C^2 slots
    pkm_top_k: int = 8
    pkm_d_val: int = 64
    max_loops: int = 4
    active_loops: int = 2
    entropy_halt_threshold: float = 1.25

    def __post_init__(self) -> None:
        for field_name, val in (
            ("vocab_size", self.vocab_size),
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
            ("n_heads", self.n_heads),
            ("n_kv_heads", self.n_kv_heads),
            ("d_head", self.d_head),
            ("d_ff", self.d_ff),
            ("max_seq_len", self.max_seq_len),
            ("pkm_side", self.pkm_side),
            ("pkm_top_k", self.pkm_top_k),
            ("pkm_d_val", self.pkm_d_val),
            ("max_loops", self.max_loops),
            ("active_loops", self.active_loops),
        ):
            if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.n_heads % 2 != 0 or self.n_kv_heads % 2 != 0:
            raise ValueError("n_heads and n_kv_heads must be even for paired DiffGQA heads")
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError("n_heads * d_head must equal d_model")
        if self.d_head % 2 != 0:
            raise ValueError("d_head must be even for RoPE and PKM subkeys")
        if self.active_loops > self.max_loops:
            raise ValueError("active_loops cannot exceed max_loops")
        if self.uses_jcle and self.n_layers < 4:
            raise ValueError("C4_JCLE_FRONTIER requires at least 4 layers (prelude + core + coda)")
        total = self.parameter_count
        if total > ABSOLUTE_MAX_PARAMETERS:
            raise ParameterBudgetExceededError(
                f"{self.name} ({total:,} params) exceeds NanoLM's {ABSOLUTE_MAX_PARAMETERS:,} (4B) limit"
            )

    @property
    def uses_delta(self) -> bool:
        return self.family != ArchitectureFamily.C0_DENSE_GQA

    @property
    def uses_diff_and_whitening(self) -> bool:
        return self.family in (
            ArchitectureFamily.C2_DIFF_NS_DELTA,
            ArchitectureFamily.C3_CORTICAL_PKM,
            ArchitectureFamily.C4_JCLE_FRONTIER,
        )

    @property
    def uses_pkm(self) -> bool:
        return self.family in (
            ArchitectureFamily.C3_CORTICAL_PKM,
            ArchitectureFamily.C4_JCLE_FRONTIER,
        )

    @property
    def uses_jcle(self) -> bool:
        return self.family == ArchitectureFamily.C4_JCLE_FRONTIER

    def is_delta_layer(self, layer_idx: int) -> bool:
        if not self.uses_delta:
            return False
        return (layer_idx + 1) % self.delta_every_n != 0

    @property
    def resolved_loop_span(self) -> tuple[int, int]:
        if not self.uses_jcle:
            return (0, self.n_layers)
        prelude_len = max(1, self.n_layers // 6)
        coda_len = max(1, self.n_layers // 6)
        start = prelude_len
        end = self.n_layers - coda_len
        if start >= end:
            return (1, self.n_layers - 1)
        return (start, end)

    @property
    def parameter_count(self) -> int:
        d = self.d_model
        emb = self.vocab_size * d
        final_norm = d
        total = emb + final_norm

        for idx in range(self.n_layers):
            norms = 2 * d
            if self.is_delta_layer(idx):
                attn = (
                    4 * d * d  # q, k, v, o
                    + 3 * d * self.short_conv_kernel  # causal conv q, k, v
                    + 2 * d * self.n_heads  # a_proj, b_proj
                    + d * d  # g_proj
                    + self.d_head  # out_norm per head
                )
                if self.uses_diff_and_whitening:
                    attn += 2 * self.n_heads  # ema_decay_logit + whitening_mix
            else:
                attn = (
                    d * (self.n_heads * self.d_head)  # q
                    + 2 * d * (self.n_kv_heads * self.d_head)  # k, v
                    + (self.n_heads * self.d_head) * d  # o
                    + 2 * self.d_head  # qk_norm
                )
                if self.uses_diff_and_whitening:
                    attn += 4 * self.d_head + 2 * self.d_head  # 4 lambda vecs + subhead RMSNorm(2*d_head)

            ffn = 3 * d * self.d_ff
            pkm = 0
            if self.uses_pkm:
                d_key = self.d_head
                pkm = (
                    d * d_key  # key_proj
                    + 2 * self.pkm_side * (d_key // 2)  # subkeys_1, subkeys_2
                    + (self.pkm_side * self.pkm_side) * self.pkm_d_val  # values embedding
                    + self.pkm_d_val * d  # out_proj
                    + 1  # pkm_scale
                )
            total += norms + attn + ffn + pkm

        if self.uses_jcle:
            total += (
                d  # core_in_norm
                + d * d  # prelude_inj
                + self.max_loops * d  # loop_step_emb
                + self.max_loops  # loop_alpha_logits
            )
        return total

    def _layer_pass_counts(self, loops: int) -> tuple[int, ...]:
        start, end = self.resolved_loop_span
        return tuple(loops if self.uses_jcle and start <= idx < end else 1 for idx in range(self.n_layers))

    def kv_cache_bytes_per_token(self, bytes_per_elem: int = 2) -> int:
        """Estimate K/V storage for a token at the configured fixed decoding depth."""
        passes = self._layer_pass_counts(self.active_loops)
        softmax_passes = sum(n for idx, n in enumerate(passes) if not self.is_delta_layer(idx))
        return 2 * softmax_passes * self.n_kv_heads * self.d_head * bytes_per_elem

    def recurrent_state_bytes_per_sequence(self, bytes_per_elem: int = 2) -> int:
        """Estimate storage for the recurrent delta matrices at the supplied precision."""
        passes = self._layer_pass_counts(self.active_loops)
        delta_passes = sum(n for idx, n in enumerate(passes) if self.is_delta_layer(idx))
        return delta_passes * self.n_heads * self.d_head * self.d_head * bytes_per_elem

    def active_flops_per_token(self, seq_len: int = 2048, *, loops: int | None = None) -> int:
        """Estimate active multiply-add FLOPs (2 * MACs) for a single decoding token."""
        d = self.d_model
        active_r = self.active_loops if loops is None else loops

        def _layer_flops(idx: int) -> int:
            ffn_mac = 3 * d * self.d_ff
            if self.is_delta_layer(idx):
                attn_mac = 5 * d * d + 3 * d * self.short_conv_kernel + self.n_heads * self.d_head * self.d_head
            else:
                # Each differential call pairs half the query heads with a
                # double-width value. Across both calls QK costs one product
                # per head and AV costs two.
                attention_products = 3 if self.uses_diff_and_whitening else 2
                attn_mac = (
                    2 * d * (self.n_heads * self.d_head)
                    + 2 * d * (self.n_kv_heads * self.d_head)
                    + attention_products * seq_len * self.n_heads * self.d_head
                )
            pkm_mac = 0
            if self.uses_pkm:
                d_key = self.d_head
                pkm_mac = d * d_key + 2 * self.pkm_side * (d_key // 2) + self.pkm_top_k * self.pkm_d_val + self.pkm_d_val * d
            return 2 * (attn_mac + ffn_mac + pkm_mac)

        total = sum(passes * _layer_flops(idx) for idx, passes in enumerate(self._layer_pass_counts(active_r)))
        total += 2 * d * self.vocab_size  # lm_head
        if self.uses_jcle:
            total += active_r * 2 * d * d
        return total

    def parameter_breakdown(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family.value,
            "total_parameters": self.parameter_count,
            "within_4b_ceiling": self.parameter_count <= ABSOLUTE_MAX_PARAMETERS,
            "delta_layers": sum(1 for i in range(self.n_layers) if self.is_delta_layer(i)),
            "softmax_layers": sum(1 for i in range(self.n_layers) if not self.is_delta_layer(i)),
            "pkm_total_slots_per_layer": (self.pkm_side * self.pkm_side) if self.uses_pkm else 0,
            "kv_cache_bytes_per_token_bf16": self.kv_cache_bytes_per_token(2),
            "recurrent_state_bytes_bf16": self.recurrent_state_bytes_per_sequence(2),
            "active_flops_per_token_2k": self.active_flops_per_token(2048),
        }


def _iso_match_dff(target_params: int, base_cfg: FrontierArchConfig) -> FrontierArchConfig:
    """Tune ``d_ff`` (multiple of 32) so ``base_cfg`` stays tightly iso-matched to ``target_params``."""
    best_cfg = base_cfg
    best_gap = abs(base_cfg.parameter_count - target_params)
    for d_ff in range(max(128, base_cfg.d_ff - 2048), base_cfg.d_ff + 1024, 32):
        cand = replace(base_cfg, d_ff=d_ff)
        gap = abs(cand.parameter_count - target_params)
        if gap < best_gap:
            best_gap = gap
            best_cfg = cand
    return best_cfg


# Controlled Micro Tournament (~38.0M matched iso-parameter budget across C0..C4)
C0_MICRO_CONFIG = FrontierArchConfig(
    name="c0-dense-gqa-micro",
    family=ArchitectureFamily.C0_DENSE_GQA,
    vocab_size=8_192,
    d_model=512,
    n_layers=12,
    n_heads=8,
    n_kv_heads=2,
    d_head=64,
    d_ff=1_408,
    max_seq_len=2_048,
)

_C0_MICRO_TARGET = C0_MICRO_CONFIG.parameter_count

C1_MICRO_CONFIG = _iso_match_dff(
    _C0_MICRO_TARGET,
    FrontierArchConfig(
        name="c1-gated-delta-micro",
        family=ArchitectureFamily.C1_GATED_DELTA_HYBRID,
        vocab_size=8_192,
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        d_head=64,
        d_ff=1_152,
        max_seq_len=2_048,
    ),
)

C2_MICRO_CONFIG = _iso_match_dff(
    _C0_MICRO_TARGET,
    FrontierArchConfig(
        name="c2-diff-ns-delta-micro",
        family=ArchitectureFamily.C2_DIFF_NS_DELTA,
        vocab_size=8_192,
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        d_head=64,
        d_ff=1_152,
        max_seq_len=2_048,
    ),
)

C3_MICRO_CONFIG = _iso_match_dff(
    _C0_MICRO_TARGET,
    FrontierArchConfig(
        name="c3-cortical-pkm-micro",
        family=ArchitectureFamily.C3_CORTICAL_PKM,
        vocab_size=8_192,
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        d_head=64,
        d_ff=928,
        max_seq_len=2_048,
        pkm_side=64,
        pkm_top_k=8,
        pkm_d_val=64,
    ),
)

C4_MICRO_CONFIG = _iso_match_dff(
    _C0_MICRO_TARGET,
    FrontierArchConfig(
        name="c4-jcle-frontier-micro",
        family=ArchitectureFamily.C4_JCLE_FRONTIER,
        vocab_size=8_192,
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        d_head=64,
        d_ff=896,
        max_seq_len=2_048,
        pkm_side=64,
        pkm_top_k=8,
        pkm_d_val=64,
        max_loops=4,
        active_loops=2,
    ),
)


def build_micro_tournament_configs(vocab_size: int = 8_192, max_seq_len: int = 2_048) -> tuple[FrontierArchConfig, ...]:
    base_c0 = FrontierArchConfig(
        name="c0-dense-gqa-micro",
        family=ArchitectureFamily.C0_DENSE_GQA,
        vocab_size=vocab_size,
        d_model=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        d_head=64,
        d_ff=1_408,
        max_seq_len=max_seq_len,
    )
    tgt = base_c0.parameter_count
    specs = (
        ("c1-gated-delta-micro", ArchitectureFamily.C1_GATED_DELTA_HYBRID, 1_152),
        ("c2-diff-ns-delta-micro", ArchitectureFamily.C2_DIFF_NS_DELTA, 1_152),
        ("c3-cortical-pkm-micro", ArchitectureFamily.C3_CORTICAL_PKM, 928),
        ("c4-jcle-frontier-micro", ArchitectureFamily.C4_JCLE_FRONTIER, 896),
    )
    out = [base_c0]
    for name, fam, hint_dff in specs:
        cfg = FrontierArchConfig(
            name=name,
            family=fam,
            vocab_size=vocab_size,
            d_model=512,
            n_layers=12,
            n_heads=8,
            n_kv_heads=2,
            d_head=64,
            d_ff=hint_dff,
            max_seq_len=max_seq_len,
            pkm_side=64,
            pkm_top_k=8,
            pkm_d_val=64,
            max_loops=4,
            active_loops=2,
        )
        out.append(_iso_match_dff(tgt, cfg))
    return tuple(out)


# Preferred ~2B Frontier Architecture (~2.04B params, 65,536 sparse PKM slots/layer)
FRONTIER_2B_CONFIG = FrontierArchConfig(
    name="nanolm-frontier-2b",
    family=ArchitectureFamily.C4_JCLE_FRONTIER,
    vocab_size=100_352,
    d_model=2_048,
    n_layers=32,
    n_heads=16,
    n_kv_heads=4,
    d_head=128,
    d_ff=4_224,
    max_seq_len=65_536,
    pkm_side=256,
    pkm_top_k=8,
    pkm_d_val=128,
    max_loops=4,
    active_loops=2,
)

# Upper-Bound <= 4B Frontier Configuration (~3.86B params < 4.0B hard ceiling)
FRONTIER_4B_CONFIG = FrontierArchConfig(
    name="nanolm-frontier-4b",
    family=ArchitectureFamily.C4_JCLE_FRONTIER,
    vocab_size=100_352,
    d_model=2_560,
    n_layers=48,
    n_heads=20,
    n_kv_heads=4,
    d_head=128,
    d_ff=4_864,
    max_seq_len=65_536,
    pkm_side=256,
    pkm_top_k=8,
    pkm_d_val=128,
    max_loops=4,
    active_loops=2,
)


class PKMMemory(nn.Module):
    """Cartesian Product-Key Sparse Cortical Memory Bank (M = C^2 slots, top-k sparse read)."""

    def __init__(self, config: FrontierArchConfig) -> None:
        super().__init__()
        self.side = config.pkm_side
        self.top_k = config.pkm_top_k
        self.d_key = config.d_head
        self.half_key = self.d_key // 2
        self.key_proj = nn.Linear(config.d_model, self.d_key, bias=False)
        self.subkeys_1 = nn.Parameter(torch.randn(self.side, self.half_key) * (self.half_key**-0.5))
        self.subkeys_2 = nn.Parameter(torch.randn(self.side, self.half_key) * (self.half_key**-0.5))
        self.values = nn.Embedding(self.side * self.side, config.pkm_d_val)
        nn.init.normal_(self.values.weight, std=0.02)
        self.out_proj = nn.Linear(config.pkm_d_val, config.d_model, bias=False)
        self.pkm_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        q = self.key_proj(x)
        q1 = F.normalize(q[..., : self.half_key].float(), dim=-1)
        q2 = F.normalize(q[..., self.half_key :].float(), dim=-1)
        k1 = F.normalize(self.subkeys_1.float(), dim=-1)
        k2 = F.normalize(self.subkeys_2.float(), dim=-1)

        s1 = torch.matmul(q1, k1.transpose(0, 1))  # [B, T, C]
        s2 = torch.matmul(q2, k2.transpose(0, 1))  # [B, T, C]

        cand_k = min(self.top_k, self.side)
        v1, i1 = torch.topk(s1, k=cand_k, dim=-1)  # [B, T, kc]
        v2, i2 = torch.topk(s2, k=cand_k, dim=-1)  # [B, T, kc]

        # Outer sum of top candidate pairs -> kc^2 candidates -> top-k
        pair_scores = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).reshape(bsz, seq_len, cand_k * cand_k)
        pair_ids = (i1.unsqueeze(-1) * self.side + i2.unsqueeze(-2)).reshape(bsz, seq_len, cand_k * cand_k)

        final_scores, pick = torch.topk(pair_scores, k=self.top_k, dim=-1)
        final_ids = torch.gather(pair_ids, dim=-1, index=pick)
        weights = F.softmax(final_scores, dim=-1).to(x.dtype)  # [B, T, top_k]

        gathered = self.values(final_ids).to(x.dtype)  # [B, T, top_k, d_val]
        pooled = (weights.unsqueeze(-1) * gathered).sum(dim=-2)  # [B, T, d_val]
        return torch.tanh(self.pkm_scale).to(x.dtype) * self.out_proj(pooled)


class _SoftmaxOrDiffGQABlockAttn(nn.Module):
    """Global softmax attention with optional Ye et al. Differential Noise Cancellation."""

    def __init__(self, config: FrontierArchConfig, layer_idx: int) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_head = config.d_head
        self.use_diff = config.uses_diff_and_whitening

        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, config.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.d_head, eps=config.rms_eps)
        self.k_norm = nn.RMSNorm(self.d_head, eps=config.rms_eps)

        if self.use_diff:
            self.n_diff_heads = self.n_heads // 2
            self.n_diff_kv = self.n_kv_heads // 2
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * float(layer_idx))
            self.lambda_q1 = nn.Parameter(torch.randn(self.d_head) * 0.1)
            self.lambda_k1 = nn.Parameter(torch.zeros(self.d_head))
            self.lambda_q2 = nn.Parameter(torch.randn(self.d_head) * 0.1)
            self.lambda_k2 = nn.Parameter(torch.zeros(self.d_head))
            self.subhead_norm = nn.RMSNorm(2 * self.d_head, eps=config.rms_eps)

    def forward(self, h: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        bsz, seq_len, _ = h.shape
        q = self.q_norm(self.q_proj(h).view(bsz, seq_len, self.n_heads, self.d_head)).transpose(1, 2)
        k = self.k_norm(self.k_proj(h).view(bsz, seq_len, self.n_kv_heads, self.d_head)).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if not self.use_diff:
            if self.n_heads != self.n_kv_heads:
                rep = self.n_heads // self.n_kv_heads
                k = k.repeat_interleave(rep, dim=1)
                v = v.repeat_interleave(rep, dim=1)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o_proj(attn_out.transpose(1, 2).reshape(bsz, seq_len, self.n_heads * self.d_head))

        # Differential Softmax Attention over paired heads via two Flash/SDPA calls: (A1 - lam*A2) V = A1 V - lam*(A2 V)
        q1 = q[:, 0::2, :, :]
        q2 = q[:, 1::2, :, :]
        k1 = k[:, 0::2, :, :]
        k2 = k[:, 1::2, :, :]
        # Re-pack paired V to [B, n_diff_kv, T, 2 * d_head]
        v_pair = torch.cat([v[:, 0::2, :, :], v[:, 1::2, :, :]], dim=-1)

        if self.n_diff_heads != self.n_diff_kv:
            rep = self.n_diff_heads // self.n_diff_kv
            k1 = k1.repeat_interleave(rep, dim=1)
            k2 = k2.repeat_interleave(rep, dim=1)
            v_pair = v_pair.repeat_interleave(rep, dim=1)

        o1 = F.scaled_dot_product_attention(q1, k1, v_pair, is_causal=True)
        o2 = F.scaled_dot_product_attention(q2, k2, v_pair, is_causal=True)

        lam1 = torch.exp((self.lambda_q1.float() * self.lambda_k1.float()).sum().clamp(-5.0, 5.0))
        lam2 = torch.exp((self.lambda_q2.float() * self.lambda_k2.float()).sum().clamp(-5.0, 5.0))
        lam = (lam1 - lam2 + self.lambda_init).to(o1.dtype)

        diff_out = self.subhead_norm(o1 - lam * o2) * (1.0 - self.lambda_init)
        return self.o_proj(diff_out.transpose(1, 2).reshape(bsz, seq_len, self.n_heads * self.d_head))


class _GatedDeltaBlockAttn(nn.Module):
    """Chunkwise Gated Delta-Rule with optional Diagonal Gauss-Newton Key Whitening."""

    def __init__(self, config: FrontierArchConfig) -> None:
        super().__init__()
        d = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.chunk_size = config.delta_chunk_size
        self.use_whitening = config.uses_diff_and_whitening

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.a_proj = nn.Linear(d, self.n_heads, bias=False)
        self.b_proj = nn.Linear(d, self.n_heads, bias=False)
        self.g_proj = nn.Linear(d, d, bias=False)

        self.conv_q_weight = nn.Parameter(torch.randn(d, 1, config.short_conv_kernel) * 0.02)
        self.conv_k_weight = nn.Parameter(torch.randn(d, 1, config.short_conv_kernel) * 0.02)
        self.conv_v_weight = nn.Parameter(torch.randn(d, 1, config.short_conv_kernel) * 0.02)
        self.out_norm = nn.RMSNorm(self.d_head, eps=config.rms_eps)

        if self.use_whitening:
            self.ema_decay_logit = nn.Parameter(torch.full((self.n_heads,), 2.2))  # sigmoid ~ 0.90
            self.whitening_mix = nn.Parameter(torch.zeros(self.n_heads))

    def forward(self, h: Tensor) -> Tensor:
        bsz, seq_len, d = h.shape
        q = F.silu(_causal_depthwise_conv(self.q_proj(h), self.conv_q_weight))
        k = F.silu(_causal_depthwise_conv(self.k_proj(h), self.conv_k_weight))
        v = F.silu(_causal_depthwise_conv(self.v_proj(h), self.conv_v_weight))

        q = q.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        q = F.normalize(q.float(), dim=-1).to(h.dtype) * (self.d_head**-0.5)
        k_normed = F.normalize(k.float(), dim=-1)

        if self.use_whitening:
            gamma = torch.sigmoid(self.ema_decay_logit)
            denom = chunked_ema_key_energy(k_normed, gamma, chunk_size=self.chunk_size)
            k_white = F.normalize(k_normed / denom, dim=-1)
            mix = torch.sigmoid(self.whitening_mix).view(1, self.n_heads, 1, 1)
            k_eff = F.normalize((1.0 - mix) * k_normed + mix * k_white, dim=-1)
        else:
            k_eff = k_normed

        # Forget gate log_a <= 0 and update strength beta in (0, 2)
        log_a = -F.softplus(self.a_proj(h).transpose(1, 2).float()) - 1e-4
        beta = 2.0 * torch.sigmoid(self.b_proj(h).transpose(1, 2).float())

        raw_out = chunkwise_gated_delta_rule(
            q,
            k_eff.to(h.dtype),
            v,
            log_a,
            beta,
            chunk_size=self.chunk_size,
        )
        normed_out = self.out_norm(raw_out).transpose(1, 2).reshape(bsz, seq_len, d)
        gated = normed_out * F.silu(self.g_proj(h))
        return self.o_proj(gated)


class _FrontierBlock(nn.Module):
    def __init__(self, config: FrontierArchConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_delta = config.is_delta_layer(layer_idx)
        self.attn_norm = nn.RMSNorm(config.d_model, eps=config.rms_eps)
        self.ffn_norm = nn.RMSNorm(config.d_model, eps=config.rms_eps)
        self.attn: nn.Module = (
            _GatedDeltaBlockAttn(config) if self.is_delta else _SoftmaxOrDiffGQABlockAttn(config, layer_idx)
        )
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.pkm = PKMMemory(config) if config.uses_pkm else None

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        h = self.attn_norm(x)
        if self.is_delta:
            x = x + self.attn(h)
        else:
            x = x + self.attn(h, cos, sin)
        h2 = self.ffn_norm(x)
        ffn_out = self.down_proj(F.silu(self.gate_proj(h2)) * self.up_proj(h2))
        if self.pkm is not None:
            ffn_out = ffn_out + self.pkm(h2)
        return x + ffn_out


class FrontierCausalLM(nn.Module):
    """Unified Causal LM supporting all five NanoLM Frontier architectures (<= 4B)."""

    def __init__(self, config: FrontierArchConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        self.blocks = nn.ModuleList(_FrontierBlock(config, idx) for idx in range(config.n_layers))
        self.final_norm = nn.RMSNorm(config.d_model, eps=config.rms_eps)

        if config.uses_jcle:
            self.core_in_norm = nn.RMSNorm(config.d_model, eps=config.rms_eps)
            self.prelude_inj = nn.Linear(config.d_model, config.d_model, bias=False)
            nn.init.normal_(self.prelude_inj.weight, std=0.02)
            self.loop_step_emb = nn.Parameter(torch.zeros(config.max_loops, config.d_model))
            self.loop_alpha_logits = nn.Parameter(torch.zeros(config.max_loops))
        else:
            self.register_parameter("loop_step_emb", None)
            self.register_parameter("loop_alpha_logits", None)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, self.config.d_head, 2, device=device, dtype=torch.float32) / self.config.d_head)
        )
        freqs = torch.outer(pos, inv_freq)
        return freqs.cos()[None, None].to(dtype), freqs.sin()[None, None].to(dtype)

    def _head_logits(self, h: Tensor) -> Tensor:
        return F.linear(self.final_norm(h), self.tok_emb.weight)

    def forward(
        self,
        input_ids: Tensor,
        *,
        active_loops: int | None = None,
        adaptive_entropy_halt: bool = False,
        return_loop_stats: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, seq_len >= 1]")
        seq_len = input_ids.shape[1]
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        h = self.tok_emb(input_ids)
        cos, sin = self._rope_cache(seq_len, input_ids.device, h.dtype)

        if not self.config.uses_jcle:
            for block in self.blocks:
                h = block(h, cos, sin)
            logits = self._head_logits(h)
            if return_loop_stats:
                return logits, {"mean_loops": 1.0, "early_exit_rate": 0.0, "mean_entropy": 0.0}
            return logits

        start, end = self.config.resolved_loop_span
        for idx in range(0, start):
            h = self.blocks[idx](h, cos, sin)

        h_pre = h
        inj_pre = self.prelude_inj(h_pre)
        loops = self.config.active_loops if active_loops is None else active_loops
        if loops < 1 or loops > self.config.max_loops:
            raise ValueError(f"active_loops={loops} out of [1, {self.config.max_loops}]")

        assert self.loop_step_emb is not None and self.loop_alpha_logits is not None

        if not adaptive_entropy_halt:
            for r in range(loops):
                u = self.core_in_norm(h + inj_pre + self.loop_step_emb[r].view(1, 1, -1))
                h_cand = u
                for idx in range(start, end):
                    h_cand = self.blocks[idx](h_cand, cos, sin)
                if r == 0:
                    h = h_cand
                else:
                    alpha = torch.sigmoid(self.loop_alpha_logits[r]).to(h.dtype)
                    h = (1.0 - alpha) * h + alpha * h_cand
            for idx in range(end, self.config.n_layers):
                h = self.blocks[idx](h, cos, sin)
            logits = self._head_logits(h)
            if return_loop_stats:
                return logits, {"mean_loops": float(loops), "early_exit_rate": 0.0, "mean_entropy": 0.0}
            return logits

        # Adaptive zero-overhead token-level Shannon entropy halting over r = 1..loops
        halted = torch.zeros(input_ids.shape, device=input_ids.device, dtype=torch.bool)
        loops_used = torch.full(input_ids.shape, float(loops), device=input_ids.device, dtype=torch.float32)
        final_logits: Tensor | None = None
        last_entropy = torch.zeros(input_ids.shape, device=input_ids.device, dtype=torch.float32)

        for r in range(loops):
            u = self.core_in_norm(h + inj_pre + self.loop_step_emb[r].view(1, 1, -1))
            h_cand = u
            for idx in range(start, end):
                h_cand = self.blocks[idx](h_cand, cos, sin)
            if r == 0:
                h = h_cand
            else:
                alpha = torch.sigmoid(self.loop_alpha_logits[r]).to(h.dtype)
                h = (1.0 - alpha) * h + alpha * h_cand

            h_coda = h
            for idx in range(end, self.config.n_layers):
                h_coda = self.blocks[idx](h_coda, cos, sin)
            step_logits = self._head_logits(h_coda)

            probs = F.softmax(step_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
            last_entropy = torch.where(halted, last_entropy, entropy)

            if final_logits is None:
                final_logits = step_logits
            else:
                final_logits = torch.where(halted.unsqueeze(-1), final_logits, step_logits)

            newly_halted = (~halted) & (entropy < self.config.entropy_halt_threshold)
            loops_used = torch.where(newly_halted, float(r + 1), loops_used)
            halted = halted | newly_halted
            if bool(halted.all()):
                break

        assert final_logits is not None
        if return_loop_stats:
            return final_logits, {
                "mean_loops": float(loops_used.mean().item()),
                "early_exit_rate": float((loops_used < float(loops)).float().mean().item()),
                "mean_entropy": float(last_entropy.mean().item()),
            }
        return final_logits
