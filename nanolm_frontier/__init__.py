"""NanoLM Frontier Architecture & Post-Training Evolution Suite (<= 4B Ceiling, ~2B Target)."""

from nanolm_frontier.architectures import (
    ABSOLUTE_MAX_PARAMETERS,
    ArchitectureFamily,
    C0_MICRO_CONFIG,
    C1_MICRO_CONFIG,
    C2_MICRO_CONFIG,
    C3_MICRO_CONFIG,
    C4_MICRO_CONFIG,
    FRONTIER_2B_CONFIG,
    FRONTIER_4B_CONFIG,
    FrontierArchConfig,
    FrontierCausalLM,
    ParameterBudgetExceededError,
    build_micro_tournament_configs,
    chunked_ema_key_energy,
    chunkwise_gated_delta_rule,
    reference_recurrent_gated_delta_rule,
)

__all__ = [
    "ABSOLUTE_MAX_PARAMETERS",
    "ArchitectureFamily",
    "C0_MICRO_CONFIG",
    "C1_MICRO_CONFIG",
    "C2_MICRO_CONFIG",
    "C3_MICRO_CONFIG",
    "C4_MICRO_CONFIG",
    "FRONTIER_2B_CONFIG",
    "FRONTIER_4B_CONFIG",
    "FrontierArchConfig",
    "FrontierCausalLM",
    "ParameterBudgetExceededError",
    "build_micro_tournament_configs",
    "chunked_ema_key_energy",
    "chunkwise_gated_delta_rule",
    "reference_recurrent_gated_delta_rule",
]
