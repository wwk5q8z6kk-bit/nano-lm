"""Inference adapters — provider-independent normalization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from nanoscribe.capabilities.parser import CapabilityToolParser, ToolResult
from nanoscribe.tool_calling import (
    DEFAULT_VLLM_TOOL_ENV,
    ToolCallParseOutcome,
    resolve_vllm_tool_env,
)

# Qwen3-Coder / vLLM may embed tool calls in message content when parsers emit text.
_QWEN_TOOL_CALL_BLOCK = re.compile(
    r"<\|tool_call\|>\s*(\{.*?\})\s*<\|/tool_call\|>",
    re.DOTALL,
)


def auto_tool_choice_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True when ENABLE_AUTO_TOOL_CHOICE is truthy in resolved vLLM env."""
    resolved = env or resolve_vllm_tool_env()
    value = resolved.get("ENABLE_AUTO_TOOL_CHOICE", DEFAULT_VLLM_TOOL_ENV["ENABLE_AUTO_TOOL_CHOICE"])
    return str(value).lower() in {"true", "1", "yes", "on"}


def tool_call_parser_name(env: Mapping[str, str] | None = None) -> str:
    resolved = env or resolve_vllm_tool_env()
    return resolved.get("TOOL_CALL_PARSER", DEFAULT_VLLM_TOOL_ENV["TOOL_CALL_PARSER"])


def build_tool_choice(
    *,
    env: Mapping[str, str] | None = None,
    forced_tool: str | None = None,
) -> str | dict[str, Any]:
    """OpenAI tool_choice value respecting ENABLE_AUTO_TOOL_CHOICE."""
    if forced_tool is not None:
        return {"type": "function", "function": {"name": forced_tool}}
    if auto_tool_choice_enabled(env):
        return "auto"
    return "required"


def extract_qwen3_content_tool_calls(content: str) -> list[dict[str, Any]]:
    """Parse embedded <|tool_call|>{...}<|/tool_call|> blocks from Qwen3-Coder text."""
    import json

    calls: list[dict[str, Any]] = []
    for index, match in enumerate(_QWEN_TOOL_CALL_BLOCK.finditer(content)):
        payload = match.group(1).strip()
        name, arguments = _split_embedded_tool_payload(payload)
        calls.append(
            {
                "id": f"qwen_embed_{index}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    return calls


def _split_embedded_tool_payload(payload: str) -> tuple[str, str]:
    import json

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return "", payload
    if not isinstance(data, Mapping):
        return "", payload
    name = data.get("name")
    if isinstance(name, str):
        arguments = data.get("arguments", {})
        if isinstance(arguments, Mapping):
            return name, json.dumps(arguments, separators=(",", ":"))
        if isinstance(arguments, str):
            return name, arguments
        return name, json.dumps(arguments, separators=(",", ":"))
    fn = data.get("function")
    if isinstance(fn, Mapping):
        fn_name = fn.get("name")
        fn_args = fn.get("arguments", {})
        if isinstance(fn_name, str):
            if isinstance(fn_args, Mapping):
                return fn_name, json.dumps(fn_args, separators=(",", ":"))
            if isinstance(fn_args, str):
                return fn_name, fn_args
            return fn_name, json.dumps(fn_args, separators=(",", ":"))
    return "", payload


def _extract_embedded_tool_name(payload: str) -> str:
    name, _ = _split_embedded_tool_payload(payload)
    return name


def normalize_openai_message(message: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalize OpenAI / vLLM / SGLang chat message into mapping with tool_calls."""
    if isinstance(message, Mapping):
        normalized = dict(message)
    else:
        normalized = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
            "tool_calls": getattr(message, "tool_calls", None),
        }
    tool_calls = normalized.get("tool_calls")
    if tool_calls:
        return normalized
    content = str(normalized.get("content") or "")
    embedded = extract_qwen3_content_tool_calls(content)
    if embedded:
        normalized["tool_calls"] = embedded
        normalized["content"] = _strip_embedded_tool_blocks(content)
    return normalized


def _strip_embedded_tool_blocks(content: str) -> str:
    stripped = _QWEN_TOOL_CALL_BLOCK.sub("", content).strip()
    return stripped


class Qwen3CoderInferenceAdapter:
    """Normalize vLLM/SGLang Qwen3-Coder outputs for CapabilityToolParser."""

    def __init__(
        self,
        *,
        parser: CapabilityToolParser | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._parser = parser or CapabilityToolParser()
        self._env = dict(env or resolve_vllm_tool_env())

    @property
    def env(self) -> dict[str, str]:
        return dict(self._env)

    @property
    def auto_tool_choice(self) -> bool:
        return auto_tool_choice_enabled(self._env)

    @property
    def parser_name(self) -> str:
        return tool_call_parser_name(self._env)

    def parse_message(self, message: Mapping[str, Any] | Any) -> ToolResult:
        normalized = normalize_openai_message(message)
        return self._parser.parse_message(normalized)

    def parse_openai_response(self, response: Mapping[str, Any] | Any) -> ToolResult:
        if isinstance(response, Mapping):
            choices = response.get("choices") or []
            if not choices:
                return ToolResult(outcome=ToolCallParseOutcome.EMPTY)
            message = normalize_openai_message(choices[0].get("message") or {})
            return self._parser.parse_message(message)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ToolResult(outcome=ToolCallParseOutcome.EMPTY)
        return self.parse_message(choices[0].message)

    def chat_completion_kwargs(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[Any],
        tool_choice: str | Mapping[str, Any] | None = None,
        forced_tool: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0,
    ) -> dict[str, Any]:
        """Build kwargs for OpenAI-compatible chat with tool_choice policy."""
        if tool_choice is not None:
            resolved_choice = tool_choice
        else:
            resolved_choice = build_tool_choice(env=self._env, forced_tool=forced_tool)
        openai_tools = [
            tool.to_openai_tool() if hasattr(tool, "to_openai_tool") else tool for tool in tools
        ]
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": openai_tools,
        }
        if resolved_choice != "none":
            kwargs["tool_choice"] = resolved_choice
        return kwargs
