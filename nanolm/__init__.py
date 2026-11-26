"""NanoLM foundation language model package."""

from nanolm.config import (
    HARD_MAX_PARAMETERS,
    MIN_TARGET_PARAMETERS,
    PREFERRED_TARGET_PARAMETERS,
    AttentionType,
    FFNType,
    NanoLMConfig,
    ParameterAccounting,
    ParameterBudgetExceededError,
    get_preset_config,
    list_presets,
)

__all__ = [
    "AttentionType",
    "FFNType",
    "HARD_MAX_PARAMETERS",
    "MIN_TARGET_PARAMETERS",
    "NanoLMConfig",
    "PREFERRED_TARGET_PARAMETERS",
    "ParameterAccounting",
    "ParameterBudgetExceededError",
    "get_preset_config",
    "list_presets",
]
