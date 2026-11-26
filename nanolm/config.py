"""Strict parameter-budget accounting and architecture configurations for NanoLM.

Hard constraint:
- Preferred production target: ~2.0B parameters
- Permitted production range: 2.0B - 4.0B parameters
- Absolute hard ceiling: 4,000,000,000 parameters (never exceeded, including
  embeddings, output head, routed experts, shared experts, norms, and deployed
  auxiliary inference modules).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

MIN_TARGET_PARAMETERS = 2_000_000_000
PREFERRED_TARGET_PARAMETERS = 2_000_000_000
HARD_MAX_PARAMETERS = 4_000_000_000


class ParameterBudgetExceededError(ValueError):
    """Raised whenever a model configuration exceeds the strict 4.0B ceiling."""


class AttentionType(str, Enum):
    GQA = "gqa"
    HYBRID_SWA = "hybrid_swa"
    MLA = "mla"


class FFNType(str, Enum):
    SWIGLU = "swiglu"
    MOE_SWIGLU = "moe_swiglu"


@dataclass(frozen=True, slots=True)
class ParameterAccounting:
    """Complete, transparent parameter, memory, and FLOP accounting."""

    total_deployed_parameters: int
    active_parameters_per_token: int
    training_only_parameters: int
    embedding_parameters: int
    unembedding_parameters: int
    attention_parameters: int
    ffn_total_parameters: int
    ffn_active_parameters: int
    router_parameters: int
    norm_parameters: int
    mtp_parameters: int
    effective_depth: int
    kv_cache_bytes_per_token_bf16: float
    kv_cache_gb_at_4k_bf16: float
    kv_cache_gb_at_32k_bf16: float
    weight_memory_gb_bf16: float
    weight_memory_gb_int4: float
    forward_flops_per_token_4k: int
    routing_flops_per_token: int
    within_hard_4b_limit: bool
    within_production_2b_4b_band: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NanoLMConfig:
    """Unified configuration for dense, hybrid-SWA, MLA, recurrent, and MoE NanoLM."""

    name: str = "nanolm-2b-dense"
    tier: str = "production"  # "production" (2B-4B) or "research_rung" (<2B ablations)
    vocab_size: int = 65_536
    hidden_size: int = 2048
    num_layers: int = 28
    num_attention_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    intermediate_size: int = 5632
    max_position_embeddings: int = 32_768
    rope_theta: float = 500_000.0
    yarn_scale_factor: float = 1.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    qk_norm: bool = True
    logit_softcap: float | None = None
    z_loss_weight: float = 1e-4

    # Attention variant
    attention_type: AttentionType = AttentionType.GQA
    sliding_window_size: int = 1024
    global_attn_every_n_layers: int = 4  # For HYBRID_SWA: 3 local : 1 global

    # MLA-specific dimensions (used only when attention_type == MLA)
    kv_lora_rank: int = 512
    q_lora_rank: int = 1024
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # FFN / Conditional Computation (MoE)
    ffn_type: FFNType = FFNType.SWIGLU
    num_experts: int = 16
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    moe_expert_intermediate_size: int = 1408
    moe_layer_freq: int = 1  # 1 = every layer is MoE; 2 = every 2nd layer is MoE
    router_bias_update_rate: float = 1e-3  # Aux-loss-free load balancing

    # Depth recurrence / parameter sharing
    loop_start_layer: int = 0
    loop_end_layer: int = 0  # If loop_end_layer > loop_start_layer and loop_repeats > 1, repeat middle stack
    loop_repeats: int = 1

    # Multi-Token Prediction (MTP) lookahead heads
    mtp_num_heads: int = 0
    mtp_loss_weight: float = 0.15
    deploy_mtp: bool = False  # If True, MTP parameters count toward deployed 4B budget!

    # Special token IDs
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("vocab_size, hidden_size, and num_layers must be positive")
        if self.attention_type in (AttentionType.GQA, AttentionType.HYBRID_SWA):
            if self.num_attention_heads % self.num_kv_heads != 0:
                raise ValueError("num_attention_heads must be divisible by num_kv_heads")
            if self.head_dim % 2 != 0:
                raise ValueError("head_dim must be even for RoPE")
        elif self.attention_type == AttentionType.MLA:
            if self.qk_rope_head_dim % 2 != 0:
                raise ValueError("qk_rope_head_dim must be even for RoPE")
        if self.ffn_type == FFNType.MOE_SWIGLU:
            if self.num_experts < self.num_experts_per_tok or self.num_experts_per_tok < 1:
                raise ValueError("Invalid MoE expert routing counts")
        if self.loop_repeats < 1:
            raise ValueError("loop_repeats must be >= 1")
        if self.loop_repeats > 1 and not (0 <= self.loop_start_layer < self.loop_end_layer <= self.num_layers):
            raise ValueError("Invalid loop layer bounds")

        acct = self.parameter_accounting()
        if acct.total_deployed_parameters > HARD_MAX_PARAMETERS:
            raise ParameterBudgetExceededError(
                f"Configuration {self.name!r} requires {acct.total_deployed_parameters:,} "
                f"deployed parameters, exceeding NanoLM's strict 4,000,000,000 ceiling."
            )
        if self.tier == "production" and acct.total_deployed_parameters < MIN_TARGET_PARAMETERS:
            raise ValueError(
                f"Production configuration {self.name!r} has {acct.total_deployed_parameters:,} "
                f"parameters (< 2,000,000,000). Mark tier='research_rung' for sub-2B ablations."
            )

    def is_moe_layer(self, layer_idx: int) -> bool:
        if self.ffn_type != FFNType.MOE_SWIGLU:
            return False
        return (layer_idx + 1) % self.moe_layer_freq == 0

    def is_sliding_layer(self, layer_idx: int) -> bool:
        if self.attention_type != AttentionType.HYBRID_SWA:
            return False
        # Every Nth layer (e.g. 3, 7, 11, ...) is global full causal attention.
        return (layer_idx + 1) % self.global_attn_every_n_layers != 0

    def execution_layer_indices(self) -> list[int]:
        """Ordered sequence of physical layer indices executed in one forward pass."""
        if self.loop_repeats <= 1 or self.loop_end_layer <= self.loop_start_layer:
            return list(range(self.num_layers))
        prefix = list(range(0, self.loop_start_layer))
        middle = list(range(self.loop_start_layer, self.loop_end_layer)) * self.loop_repeats
        suffix = list(range(self.loop_end_layer, self.num_layers))
        return prefix + middle + suffix

    def _single_attention_params_and_norms(self) -> tuple[int, int]:
        d = self.hidden_size
        h = self.num_attention_heads
        if self.attention_type in (AttentionType.GQA, AttentionType.HYBRID_SWA):
            hd = self.head_dim
            kv = self.num_kv_heads
            linear_params = d * (h * hd) + 2 * d * (kv * hd) + (h * hd) * d
            qk_norm_params = (2 * hd) if self.qk_norm else 0
            return linear_params, qk_norm_params
        # MLA: compressed Q and KV latents + decoupled RoPE key
        q_c = self.q_lora_rank
        kv_c = self.kv_lora_rank
        d_nope = self.qk_nope_head_dim
        d_rope = self.qk_rope_head_dim
        d_v = self.v_head_dim
        q_down = d * q_c
        q_up = q_c * h * (d_nope + d_rope)
        kv_down = d * (kv_c + d_rope)
        kv_up = kv_c * h * (d_nope + d_v)
        o_proj = (h * d_v) * d
        norms = q_c + kv_c
        return q_down + q_up + kv_down + kv_up + o_proj, norms

    def _single_ffn_params(self, layer_idx: int) -> tuple[int, int, int]:
        """Return (total_ffn_params, active_ffn_params, router_params) for layer_idx."""
        d = self.hidden_size
        if not self.is_moe_layer(layer_idx):
            dense = 3 * d * self.intermediate_size
            return dense, dense, 0
        expert_p = 3 * d * self.moe_expert_intermediate_size
        shared_p = self.num_shared_experts * expert_p
        routed_total = self.num_experts * expert_p
        routed_active = self.num_experts_per_tok * expert_p
        router_p = d * self.num_experts
        return shared_p + routed_total, shared_p + routed_active, router_p

    def _single_mtp_head_params(self) -> tuple[int, int]:
        """Return (linear_and_block_params, norm_params) for one MTP lookahead head."""
        d = self.hidden_size
        # Each MTP head concatenates [norm(h_prev), norm(emb(next_tok))] -> Linear(2d, d) -> 1 lightweight GQA+SwiGLU block
        proj = 2 * d * d
        attn_lin, attn_norm = self._single_attention_params_and_norms()
        ffn_dense = 3 * d * self.intermediate_size
        norms = 3 * d + 2 * d + attn_norm  # h_norm, emb_norm, head_out_norm + block pre-norms
        return proj + attn_lin + ffn_dense, norms

    def parameter_accounting(self) -> ParameterAccounting:
        d = self.hidden_size
        emb_p = self.vocab_size * d
        unemb_p = 0 if self.tie_word_embeddings else self.vocab_size * d

        attn_lin_1, attn_norm_1 = self._single_attention_params_and_norms()
        attn_total = self.num_layers * attn_lin_1
        norm_total = self.num_layers * (2 * d + attn_norm_1) + d  # 2 pre-norms/block + final RMSNorm

        ffn_total = 0
        ffn_active_unique = 0
        router_total = 0
        for i in range(self.num_layers):
            ft, fa, rp = self._single_ffn_params(i)
            ffn_total += ft
            ffn_active_unique += fa
            router_total += rp

        mtp_lin_1, mtp_norm_1 = self._single_mtp_head_params()
        mtp_total = self.mtp_num_heads * (mtp_lin_1 + mtp_norm_1)

        base_total = emb_p + unemb_p + attn_total + ffn_total + router_total + norm_total
        deployed_total = base_total + (mtp_total if self.deploy_mtp else 0)
        training_only = 0 if self.deploy_mtp else mtp_total

        exec_indices = self.execution_layer_indices()
        effective_depth = len(exec_indices)
        active_attn = effective_depth * (attn_lin_1 + attn_norm_1)
        active_ffn = 0
        active_router = 0
        for idx in exec_indices:
            _, fa, rp = self._single_ffn_params(idx)
            active_ffn += fa
            active_router += rp
        active_norms = effective_depth * (2 * d) + d
        # During generation, both embedding lookup (1 row) and full output projection (V * d) execute:
        # Standard active-parameter convention counts all weights participating in matrix-vector products per token.
        active_unemb = self.vocab_size * d
        active_mtp = mtp_total if self.deploy_mtp else 0
        active_per_token = active_unemb + active_attn + active_ffn + active_router + active_norms + active_mtp

        # KV cache footprint in bf16 (2 bytes per element)
        def kv_bytes_for_seq(seq_len: int) -> int:
            total_bytes = 0
            for idx in exec_indices:
                if self.attention_type == AttentionType.MLA:
                    # Cache stores compressed kv_lora_rank + qk_rope_head_dim per token!
                    elems = seq_len * (self.kv_lora_rank + self.qk_rope_head_dim)
                elif self.is_sliding_layer(idx):
                    eff_len = min(seq_len, self.sliding_window_size)
                    elems = 2 * eff_len * self.num_kv_heads * self.head_dim
                else:
                    elems = 2 * seq_len * self.num_kv_heads * self.head_dim
                total_bytes += 2 * elems
            return total_bytes

        kv_4k = kv_bytes_for_seq(4096)
        kv_32k = kv_bytes_for_seq(32768)
        # Asymptotic marginal bytes/token at 32k context
        kv_per_tok = kv_32k / 32768.0

        # Analytical forward FLOPs per token at 4096 context (2 * MACs for matmuls + attention dot products)
        matmul_flops = 2 * active_per_token
        attn_ctx_flops = 0
        for idx in exec_indices:
            eff_ctx = min(4096, self.sliding_window_size) if self.is_sliding_layer(idx) else 4096
            h_dim = (self.qk_nope_head_dim + self.qk_rope_head_dim) if self.attention_type == AttentionType.MLA else self.head_dim
            attn_ctx_flops += 4 * self.num_attention_heads * h_dim * eff_ctx
        routing_flops = 2 * active_router

        return ParameterAccounting(
            total_deployed_parameters=deployed_total,
            active_parameters_per_token=active_per_token,
            training_only_parameters=training_only,
            embedding_parameters=emb_p,
            unembedding_parameters=unemb_p,
            attention_parameters=attn_total,
            ffn_total_parameters=ffn_total,
            ffn_active_parameters=active_ffn,
            router_parameters=router_total,
            norm_parameters=norm_total,
            mtp_parameters=mtp_total,
            effective_depth=effective_depth,
            kv_cache_bytes_per_token_bf16=round(kv_per_tok, 2),
            kv_cache_gb_at_4k_bf16=round(kv_4k / (1024**3), 4),
            kv_cache_gb_at_32k_bf16=round(kv_32k / (1024**3), 4),
            weight_memory_gb_bf16=round(deployed_total * 2 / (1024**3), 4),
            weight_memory_gb_int4=round(deployed_total * 0.55 / (1024**3), 4),
            forward_flops_per_token_4k=matmul_flops + attn_ctx_flops,
            routing_flops_per_token=routing_flops,
            within_hard_4b_limit=deployed_total <= HARD_MAX_PARAMETERS,
            within_production_2b_4b_band=MIN_TARGET_PARAMETERS <= deployed_total <= HARD_MAX_PARAMETERS,
        )


PRESETS: dict[str, NanoLMConfig] = {
    # ==================== PRODUCTION 2B - 4B CANDIDATES ====================
    "nanolm-2b-dense": NanoLMConfig(
        name="nanolm-2b-dense",
        tier="production",
        vocab_size=65_536,
        hidden_size=2048,
        num_layers=36,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        intermediate_size=6912,
        attention_type=AttentionType.GQA,
        ffn_type=FFNType.SWIGLU,
        qk_norm=True,
        tie_word_embeddings=True,
        mtp_num_heads=1,
        deploy_mtp=False,
    ),
    "nanolm-2b-hybrid-swa": NanoLMConfig(
        name="nanolm-2b-hybrid-swa",
        tier="production",
        vocab_size=65_536,
        hidden_size=2048,
        num_layers=36,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        intermediate_size=6912,
        attention_type=AttentionType.HYBRID_SWA,
        sliding_window_size=1024,
        global_attn_every_n_layers=4,
        ffn_type=FFNType.SWIGLU,
        qk_norm=True,
        logit_softcap=30.0,
        tie_word_embeddings=True,
        mtp_num_heads=1,
        deploy_mtp=False,
    ),
    "nanolm-2b-mla": NanoLMConfig(
        name="nanolm-2b-mla",
        tier="production",
        vocab_size=65_536,
        hidden_size=2048,
        num_layers=36,
        num_attention_heads=16,
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        intermediate_size=6912,
        attention_type=AttentionType.MLA,
        ffn_type=FFNType.SWIGLU,
        tie_word_embeddings=True,
        mtp_num_heads=1,
        deploy_mtp=False,
    ),
    "nanolm-2b-loop": NanoLMConfig(
        name="nanolm-2b-loop",
        tier="production",
        vocab_size=65_536,
        hidden_size=2048,
        num_layers=36,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        intermediate_size=6912,
        attention_type=AttentionType.HYBRID_SWA,
        sliding_window_size=1024,
        global_attn_every_n_layers=4,
        loop_start_layer=12,
        loop_end_layer=24,
        loop_repeats=2,  # 36 unique layers -> 48 effective reasoning steps!
        qk_norm=True,
        tie_word_embeddings=True,
    ),
    "nanolm-3.3b-dense": NanoLMConfig(
        name="nanolm-3.3b-dense",
        tier="production",
        vocab_size=65_536,
        hidden_size=2560,
        num_layers=40,
        num_attention_heads=20,
        num_kv_heads=4,
        head_dim=128,
        intermediate_size=8192,
        attention_type=AttentionType.HYBRID_SWA,
        sliding_window_size=2048,
        global_attn_every_n_layers=4,
        ffn_type=FFNType.SWIGLU,
        qk_norm=True,
        tie_word_embeddings=True,
        mtp_num_heads=1,
        deploy_mtp=False,
    ),
    "nanolm-3.8b-moe": NanoLMConfig(
        name="nanolm-3.8b-moe",
        tier="production",
        vocab_size=65_536,
        hidden_size=2048,
        num_layers=22,
        num_attention_heads=16,
        num_kv_heads=4,
        head_dim=128,
        intermediate_size=6912,
        attention_type=AttentionType.HYBRID_SWA,
        sliding_window_size=1024,
        global_attn_every_n_layers=4,
        ffn_type=FFNType.MOE_SWIGLU,
        num_experts=16,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_expert_intermediate_size=1536,
        moe_layer_freq=1,
        qk_norm=True,
        tie_word_embeddings=True,
        mtp_num_heads=1,
        deploy_mtp=False,
    ),
    # ==================== CONTROLLED RESEARCH RUNGS (<2B) ====================
    "nanolm-25m-dense": NanoLMConfig(
        name="nanolm-25m-dense",
        tier="research_rung",
        vocab_size=16_384,
        hidden_size=384,
        num_layers=12,
        num_attention_heads=6,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=1024,
        max_position_embeddings=2048,
        attention_type=AttentionType.GQA,
        ffn_type=FFNType.SWIGLU,
        qk_norm=True,
        tie_word_embeddings=True,
    ),
    "nanolm-25m-hybrid-swa": NanoLMConfig(
        name="nanolm-25m-hybrid-swa",
        tier="research_rung",
        vocab_size=16_384,
        hidden_size=384,
        num_layers=12,
        num_attention_heads=6,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=1024,
        max_position_embeddings=2048,
        attention_type=AttentionType.HYBRID_SWA,
        sliding_window_size=256,
        global_attn_every_n_layers=4,
        ffn_type=FFNType.SWIGLU,
        qk_norm=True,
        logit_softcap=30.0,
        tie_word_embeddings=True,
        mtp_num_heads=1,
        deploy_mtp=False,
    ),
    "nanolm-25m-mla": NanoLMConfig(
        name="nanolm-25m-mla",
        tier="research_rung",
        vocab_size=16_384,
        hidden_size=384,
        num_layers=12,
        num_attention_heads=6,
        kv_lora_rank=128,
        q_lora_rank=192,
        qk_nope_head_dim=48,
        qk_rope_head_dim=32,
        v_head_dim=64,
        intermediate_size=1024,
        max_position_embeddings=2048,
        attention_type=AttentionType.MLA,
        ffn_type=FFNType.SWIGLU,
        tie_word_embeddings=True,
    ),
    "nanolm-45m-moe": NanoLMConfig(
        name="nanolm-45m-moe",
        tier="research_rung",
        vocab_size=16_384,
        hidden_size=384,
        num_layers=12,
        num_attention_heads=6,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=1024,
        max_position_embeddings=2048,
        attention_type=AttentionType.HYBRID_SWA,
        sliding_window_size=256,
        global_attn_every_n_layers=4,
        ffn_type=FFNType.MOE_SWIGLU,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_expert_intermediate_size=320,
        qk_norm=True,
        tie_word_embeddings=True,
    ),
    "nanolm-25m-loop": NanoLMConfig(
        name="nanolm-25m-loop",
        tier="research_rung",
        vocab_size=16_384,
        hidden_size=384,
        num_layers=12,
        num_attention_heads=6,
        num_kv_heads=2,
        head_dim=64,
        intermediate_size=1024,
        max_position_embeddings=2048,
        attention_type=AttentionType.HYBRID_SWA,
        sliding_window_size=256,
        global_attn_every_n_layers=4,
        loop_start_layer=4,
        loop_end_layer=8,
        loop_repeats=2,
        qk_norm=True,
        tie_word_embeddings=True,
    ),
}


def get_preset_config(name: str) -> NanoLMConfig:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {sorted(PRESETS)}")
    return PRESETS[name]


def list_presets() -> list[str]:
    return list(PRESETS.keys())
