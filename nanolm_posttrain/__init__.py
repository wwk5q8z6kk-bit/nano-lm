"""NanoLM — Independent 2B-4B General-Purpose Foundation Language Model Stack."""

from nanolm_posttrain.architecture import (
    ABSOLUTE_MAX_PARAMETERS,
    NANO_2B_CONFIG,
    NANO_350M_CONFIG,
    NANO_4B_CONFIG,
    NANO_MICRO_CONFIG,
    PREFERRED_TARGET_PARAMETERS,
    NanoLM,
    NanoLMConfig,
    ParameterBudgetExceededError,
)
from nanolm_posttrain.eval_suite import (
    FoundationDomain,
    FoundationTask,
    TaskVerdict,
    build_foundation_eval_suite,
    build_foundation_train_curriculum,
    verify_foundation_response,
)
from nanolm_posttrain.lora import (
    LoRAConfig,
    LoRALinear,
    apply_lora,
    build_chat_sft_tensors,
    lora_state_dict,
    train_masked_sft_step,
    trainable_parameter_summary,
)

__all__ = [
    "ABSOLUTE_MAX_PARAMETERS",
    "FoundationDomain",
    "FoundationTask",
    "LoRAConfig",
    "LoRALinear",
    "NANO_2B_CONFIG",
    "NANO_350M_CONFIG",
    "NANO_4B_CONFIG",
    "NANO_MICRO_CONFIG",
    "NanoLM",
    "NanoLMConfig",
    "PREFERRED_TARGET_PARAMETERS",
    "ParameterBudgetExceededError",
    "TaskVerdict",
    "apply_lora",
    "build_chat_sft_tensors",
    "build_foundation_eval_suite",
    "build_foundation_train_curriculum",
    "lora_state_dict",
    "train_masked_sft_step",
    "trainable_parameter_summary",
    "verify_foundation_response",
]
