"""Inference adapter entry points."""

from nanoscribe.inference.qwen3_coder import (
    Qwen3CoderInferenceAdapter,
    auto_tool_choice_enabled,
    build_tool_choice,
    normalize_openai_message,
    tool_call_parser_name,
)
from nanoscribe.inference.tool_registry import (
    agent_tool_definitions,
    inference_tool_definitions,
    resolve_inference_tool_choice,
    scribe_only_tool_definitions,
)

__all__ = [
    "Qwen3CoderInferenceAdapter",
    "agent_tool_definitions",
    "auto_tool_choice_enabled",
    "build_tool_choice",
    "inference_tool_definitions",
    "normalize_openai_message",
    "resolve_inference_tool_choice",
    "scribe_only_tool_definitions",
    "tool_call_parser_name",
]
