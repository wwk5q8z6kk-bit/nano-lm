"""Parameter accounting, strict <4B budget enforcement, and iso-parameter suite builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any

HARD_MAX_DEPLOYED_PARAMETERS = 4_000_000_000
PREFERRED_FLAGSHIP_TARGET = 3_920_000_000


class ParameterBudgetError(ValueError):
    """Raised when deployed learned parameters violate the strict < 4B ceiling."""


class ParadigmKind(str, Enum):
    B0_TRANSFORMER = "b0_transformer"
    B1_GDN2_HYBRID = "b1_gdn2_hybrid"
    I1_PHMN = "i1_phmn"
    I2_NEBN = "i2_nebn"
    I3_HRWN = "i3_hrwn"
    G2_MPRN = "g2_mprn"
    G3_HMWR = "g3_hmwr"


@dataclass(frozen=True, slots=True)
class ArchitectureConfig:
    """Unified configuration covering baselines and newly invented architectures."""

    name: str
    paradigm: ParadigmKind
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden: int
    max_seq_len: int = 32_768
    rope_theta: float = 500_000.0
    chunk_size: int = 32
    nebn_chunk_size: int = 128
    scam_slots: int = 32
    global_attn_every: int = 4
    tie_embeddings: bool = True
    jep_head_enabled: bool = False
    # Ablation switches (used by G2_MPRN / G3_HMWR and diagnostics)
    use_symplectic_reflection: bool = True
    use_precision_norm: bool = True
    use_scam_memory: bool = True
    use_phase_bilinear: bool = True
    use_depth_routing: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0 or self.n_layers <= 0:
            raise ValueError("vocab_size, d_model, and n_layers must be positive")
        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even")
        if self.n_heads <= 0 or self.head_dim <= 0 or self.n_heads * self.head_dim != self.d_model:
            raise ValueError("n_heads * head_dim must equal d_model")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE / phase pairing")
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.ffn_hidden <= 0 or self.chunk_size <= 0 or self.scam_slots < 0:
            raise ValueError("ffn_hidden and chunk_size must be positive; scam_slots >= 0")


@dataclass(frozen=True, slots=True)
class ParameterCensus:
    """Exact analytical parameter, memory, and FLOP breakdown matching PyTorch numel()."""

    embedding_parameters: int
    unembedding_parameters: int
    operator_parameters: int
    synthesis_parameters: int
    norm_parameters: int
    total_deployed_parameters: int
    training_only_parameters: int
    state_bytes_at_512_bf16: int
    state_bytes_at_4096_bf16: int
    state_bytes_at_32768_bf16: int
    forward_flops_per_token_4k: int
    within_strict_4b_limit: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _layer_operator_params(cfg: ArchitectureConfig, layer_idx: int) -> int:
    d = cfg.d_model
    kv_d = cfg.n_kv_heads * cfg.head_dim
    h = cfg.n_heads
    p = cfg.paradigm
    if p == ParadigmKind.B0_TRANSFORMER:
        return 2 * d * d + 2 * d * kv_d
    if p == ParadigmKind.B1_GDN2_HYBRID:
        if (layer_idx + 1) % cfg.global_attn_every == 0:
            return 2 * d * d + 2 * d * kv_d
        return 4 * d * d + 4 * d + 2 * h * (d + 1)
    if p == ParadigmKind.I1_PHMN:
        return 5 * d * d + 4 * d + 3 * h * (d + 1)
    if p == ParadigmKind.I2_NEBN:
        return 4 * d * d + 4 * d + 2 * h * (d + 1) + h
    if p == ParadigmKind.I3_HRWN:
        d_scam = d // 2
        wave_params = 4 * d * d + 4 * 5 * d
        scam_params = 4 * d * d_scam + d + 1
        return wave_params + scam_params + d * d
    if p == ParadigmKind.G2_MPRN:
        total = 4 * d * d + 4 * d  # k, v, q, o + 4-tap CausalKeyShift
        if cfg.use_symplectic_reflection:
            total += d * d  # u_proj
        n_gates = 3 if cfg.use_symplectic_reflection else 2
        total += n_gates * h * (d + 1)
        if cfg.use_precision_norm:
            total += h
        if cfg.use_scam_memory and cfg.scam_slots > 0:
            d_scam = d // 2
            total += 4 * d * d_scam + d + 1
        return total
    if p == ParadigmKind.G3_HMWR:
        h_met = max(1, h // 2)
        d_met = h_met * cfg.head_dim
        d_scam = max(2, d // 4)
        depth_params = (layer_idx + 1) * (d + 1) if (cfg.use_depth_routing and layer_idx >= 1) else 0
        wave_params = 2 * d * d + 20 * d
        met_params = 4 * d + 5 * d * d_met + 3 * h_met * (d + 1)
        scam_params = 4 * d * d_scam + d + 1
        return depth_params + wave_params + met_params + scam_params + d * d
    raise ValueError(f"Unsupported paradigm: {p}")


def _layer_synthesis_params(cfg: ArchitectureConfig) -> int:
    d = cfg.d_model
    ffn = cfg.ffn_hidden
    p = cfg.paradigm
    if p in (ParadigmKind.B0_TRANSFORMER, ParadigmKind.B1_GDN2_HYBRID, ParadigmKind.I3_HRWN):
        return 3 * d * ffn
    if p == ParadigmKind.G3_HMWR:
        return 3 * d * ffn + (2 * d if cfg.use_depth_routing else 0)
    if p == ParadigmKind.I1_PHMN:
        return 2 * d * ffn + d * (d // 2) + d * d
    if p == ParadigmKind.I2_NEBN:
        return 3 * d * ffn + (cfg.n_heads + 1) * ffn
    if p == ParadigmKind.G2_MPRN:
        if cfg.use_phase_bilinear:
            base = 2 * d * ffn + d * (d // 2) + d * d
            if cfg.use_precision_norm:
                base += (cfg.n_heads + 1) * d
            return base
        return 3 * d * ffn
    raise ValueError(f"Unsupported paradigm: {p}")


def _layer_state_bytes(cfg: ArchitectureConfig, layer_idx: int, seq_len: int) -> int:
    """Persistent inference state bytes per batch element at context length `seq_len`."""
    d = cfg.d_model
    h = cfg.n_heads
    hd = cfg.head_dim
    kv_d = cfg.n_kv_heads * hd
    p = cfg.paradigm
    if p == ParadigmKind.B0_TRANSFORMER:
        return 2 * 2 * kv_d * seq_len  # K + V in bf16
    if p == ParadigmKind.B1_GDN2_HYBRID:
        if (layer_idx + 1) % cfg.global_attn_every == 0:
            return 2 * 2 * kv_d * seq_len
        return 4 * h * hd * hd + 6 * d  # float32 state matrix + 3-slot bf16 key-shift ring
    if p == ParadigmKind.I1_PHMN:
        return 4 * h * hd * hd + 6 * d
    if p == ParadigmKind.I2_NEBN:
        return 4 * h * (hd * hd + hd + 1) + 6 * d
    if p == ParadigmKind.I3_HRWN:
        # Fixed causal ring buffers for dilations (1, 4, 16, 64) kernel 5 -> max offsets (4+16+64+256=340) * 2 bytes * d + SCAM keys/vals
        wave_bytes = 2 * (4 + 16 + 64 + 256) * d
        scam_bytes = cfg.scam_slots * (2 * (d // 2) * 2 + 4)
        return wave_bytes + scam_bytes
    if p == ParadigmKind.G2_MPRN:
        state_bytes = 4 * h * hd * hd + 6 * d
        if cfg.use_precision_norm:
            state_bytes += 4 * h * hd + 4 * h
        if cfg.use_scam_memory and cfg.scam_slots > 0:
            state_bytes += cfg.scam_slots * (2 * (d // 2) * 2 + 4)
        return state_bytes
    if p == ParadigmKind.G3_HMWR:
        h_met = max(1, h // 2)
        d_scam = max(2, d // 4)
        met_bytes = 4 * h_met * hd * hd + 6 * d
        wave_bytes = 2 * (4 + 16 + 64 + 256) * d
        scam_bytes = cfg.scam_slots * (4 * d_scam + 4)
        return met_bytes + wave_bytes + scam_bytes
    raise ValueError(f"Unsupported paradigm: {p}")


def compute_parameter_census(
    cfg: ArchitectureConfig,
    *,
    enforce_strict_4b: bool = True,
) -> ParameterCensus:
    """Compute exact deployed and training-only parameter counts plus state bytes."""
    emb = cfg.vocab_size * cfg.d_model
    unemb = 0 if cfg.tie_embeddings else cfg.vocab_size * cfg.d_model
    op_total = sum(_layer_operator_params(cfg, i) for i in range(cfg.n_layers))
    syn_total = cfg.n_layers * _layer_synthesis_params(cfg)
    norms = (2 * cfg.n_layers + 1) * cfg.d_model
    deployed = emb + unemb + op_total + syn_total + norms
    if enforce_strict_4b and deployed >= HARD_MAX_DEPLOYED_PARAMETERS:
        raise ParameterBudgetError(
            f"{cfg.name} deployed parameters {deployed:,} >= strict 4B ceiling {HARD_MAX_DEPLOYED_PARAMETERS:,}"
        )
    train_only = cfg.d_model * cfg.d_model if cfg.jep_head_enabled else 0
    s512 = sum(_layer_state_bytes(cfg, i, 512) for i in range(cfg.n_layers))
    s4096 = sum(_layer_state_bytes(cfg, i, 4096) for i in range(cfg.n_layers))
    s32768 = sum(_layer_state_bytes(cfg, i, 32768) for i in range(cfg.n_layers))
    # Approximate forward FLOPs/token at T=4096 (2*MAC for linear ops + attn/scan term)
    d = cfg.d_model
    ffn = cfg.ffn_hidden
    lin_flops = 2 * (op_total + syn_total + cfg.vocab_size * d)
    if cfg.paradigm == ParadigmKind.B0_TRANSFORMER:
        ctx_flops = 4 * cfg.n_layers * d * 4096
    elif cfg.paradigm == ParadigmKind.B1_GDN2_HYBRID:
        n_attn = sum(1 for i in range(cfg.n_layers) if (i + 1) % cfg.global_attn_every == 0)
        n_rec = cfg.n_layers - n_attn
        ctx_flops = 4 * n_attn * d * 4096 + 8 * n_rec * cfg.n_heads * cfg.head_dim * cfg.head_dim
    elif cfg.paradigm in (ParadigmKind.I1_PHMN, ParadigmKind.I2_NEBN, ParadigmKind.G2_MPRN):
        ctx_flops = 8 * cfg.n_layers * cfg.n_heads * cfg.head_dim * cfg.head_dim
        if cfg.paradigm == ParadigmKind.G2_MPRN and cfg.use_scam_memory:
            ctx_flops += 4 * cfg.n_layers * (d // 2) * max(1, cfg.scam_slots)
    elif cfg.paradigm == ParadigmKind.G3_HMWR:
        h_met = max(1, cfg.n_heads // 2)
        d_scam = max(2, d // 4)
        ctx_flops = 2 * cfg.n_layers * (
            20 * d + 4 * h_met * cfg.head_dim * cfg.head_dim + 2 * d_scam * max(1, cfg.scam_slots)
        )
    else:
        ctx_flops = 2 * cfg.n_layers * (20 * d + 2 * (d // 2) * max(1, cfg.scam_slots))
    return ParameterCensus(
        embedding_parameters=emb,
        unembedding_parameters=unemb,
        operator_parameters=op_total,
        synthesis_parameters=syn_total,
        norm_parameters=norms,
        total_deployed_parameters=deployed,
        training_only_parameters=train_only,
        state_bytes_at_512_bf16=s512,
        state_bytes_at_4096_bf16=s4096,
        state_bytes_at_32768_bf16=s32768,
        forward_flops_per_token_4k=lin_flops + ctx_flops,
        within_strict_4b_limit=deployed < HARD_MAX_DEPLOYED_PARAMETERS,
    )


def _solve_ffn_hidden(
    base_cfg: ArchitectureConfig,
    target_deployed: int,
    *,
    align: int = 16,
) -> ArchitectureConfig:
    """Adjust `ffn_hidden` (multiple of `align`) to match `target_deployed` as closely as possible."""
    best_cfg = base_cfg
    best_err = abs(compute_parameter_census(base_cfg, enforce_strict_4b=False).total_deployed_parameters - target_deployed)
    # Linear solve for continuous ffn_hidden then check nearby multiples
    c0 = compute_parameter_census(replace(base_cfg, ffn_hidden=align), enforce_strict_4b=False).total_deployed_parameters
    c1 = compute_parameter_census(replace(base_cfg, ffn_hidden=2 * align), enforce_strict_4b=False).total_deployed_parameters
    slope = max(1, c1 - c0)
    approx_mult = max(1, round(align + align * (target_deployed - c0) / slope))
    lo = max(align, ((approx_mult // align) - 4) * align)
    hi = ((approx_mult // align) + 5) * align
    for cand_h in range(lo, hi + 1, align):
        cand_cfg = replace(base_cfg, ffn_hidden=cand_h)
        tot = compute_parameter_census(cand_cfg, enforce_strict_4b=False).total_deployed_parameters
        err = abs(tot - target_deployed)
        if err < best_err:
            best_err = err
            best_cfg = cand_cfg
    return best_cfg


_TIER_SPECS: dict[str, dict[str, Any]] = {
    # Compact 2-layer mechanistic microscope (~115k params, vocab=256)
    "mech": {
        "vocab_size": 256,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "n_kv_heads": 2,
        "head_dim": 16,
        "b0_ffn": 192,
        "align": 2,
        "max_seq_len": 1024,
        "chunk_size": 32,
        "scam_slots": 24,
    },
    # Small real-corpus scaling rung (~8.2M params, vocab=32768)
    "scale_8m": {
        "vocab_size": 32_768,
        "d_model": 192,
        "n_layers": 6,
        "n_heads": 6,
        "n_kv_heads": 2,
        "head_dim": 32,
        "b0_ffn": 512,
        "align": 16,
        "max_seq_len": 4096,
        "chunk_size": 32,
        "scam_slots": 32,
    },
    # Cycle 1-2 discovery rung (~19.9M params, vocab=32768)
    "discovery_18m": {
        "vocab_size": 32_768,
        "d_model": 384,
        "n_layers": 6,
        "n_heads": 12,
        "n_kv_heads": 6,
        "head_dim": 32,
        "b0_ffn": 768,
        "align": 16,
        "max_seq_len": 4096,
        "chunk_size": 32,
        "scam_slots": 32,
    },
    # Upper local M3 Max scaling rung (~42.6M params, vocab=32768)
    "scale_42m": {
        "vocab_size": 32_768,
        "d_model": 576,
        "n_layers": 10,
        "n_heads": 18,
        "n_kv_heads": 6,
        "head_dim": 32,
        "b0_ffn": 1152,
        "align": 32,
        "max_seq_len": 4096,
        "chunk_size": 32,
        "scam_slots": 32,
    },
    # Flagship ~3.89B architecture realization (strictly < 4.000B!)
    "flagship_4b": {
        "vocab_size": 65_536,
        "d_model": 2816,
        "n_layers": 32,
        "n_heads": 22,
        "n_kv_heads": 11,
        "head_dim": 128,
        "b0_ffn": 10_880,
        "align": 64,
        "max_seq_len": 32_768,
        "chunk_size": 32,
        "scam_slots": 32,
    },
}


def make_iso_param_suite(
    tier: str,
    *,
    vocab_size: int | None = None,
    jep_head_enabled: bool = False,
) -> dict[ParadigmKind, ArchitectureConfig]:
    """Return a parameter-matched (within ±0.5%) dict of configs across all paradigms."""
    if tier not in _TIER_SPECS:
        raise KeyError(f"Unknown tier {tier!r}; expected one of {sorted(_TIER_SPECS)}")
    spec = dict(_TIER_SPECS[tier])
    v = int(vocab_size) if vocab_size is not None else int(spec["vocab_size"])
    b0 = ArchitectureConfig(
        name=f"b0_transformer_{tier}",
        paradigm=ParadigmKind.B0_TRANSFORMER,
        vocab_size=v,
        d_model=spec["d_model"],
        n_layers=spec["n_layers"],
        n_heads=spec["n_heads"],
        n_kv_heads=spec["n_kv_heads"],
        head_dim=spec["head_dim"],
        ffn_hidden=spec["b0_ffn"],
        max_seq_len=spec["max_seq_len"],
        chunk_size=spec["chunk_size"],
        scam_slots=spec["scam_slots"],
        jep_head_enabled=jep_head_enabled,
    )
    target = compute_parameter_census(b0).total_deployed_parameters
    suite: dict[ParadigmKind, ArchitectureConfig] = {ParadigmKind.B0_TRANSFORMER: b0}
    for kind in (
        ParadigmKind.B1_GDN2_HYBRID,
        ParadigmKind.I1_PHMN,
        ParadigmKind.I2_NEBN,
        ParadigmKind.I3_HRWN,
        ParadigmKind.G2_MPRN,
        ParadigmKind.G3_HMWR,
    ):
        cand = replace(b0, name=f"{kind.value}_{tier}", paradigm=kind)
        matched = _solve_ffn_hidden(cand, target, align=spec["align"])
        compute_parameter_census(matched, enforce_strict_4b=True)
        suite[kind] = matched
    return suite


def retune_iso_param_variant(
    cfg: ArchitectureConfig,
    reference_cfg: ArchitectureConfig,
    *,
    name: str,
    align: int = 16,
    **overrides: Any,
) -> ArchitectureConfig:
    """Create an ablated variant of `cfg` and retune `ffn_hidden` to match `reference_cfg`."""
    target = compute_parameter_census(reference_cfg, enforce_strict_4b=False).total_deployed_parameters
    modified = replace(cfg, name=name, **overrides)
    tuned = _solve_ffn_hidden(modified, target, align=align)
    compute_parameter_census(tuned, enforce_strict_4b=True)
    return tuned
