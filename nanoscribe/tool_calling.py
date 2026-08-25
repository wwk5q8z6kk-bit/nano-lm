"""Tool-calling abstractions — parse, validate, and log OpenAI-compatible tool calls."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from nanoscribe.adapt import AdaptError, CANDIDATE_SCHEMA_VERSION, ModelCandidate
from nanoscribe.encounter import EncounterError

logger = logging.getLogger(__name__)

SUBMIT_CANDIDATE_ATOMS_TOOL = "submit_candidate_atoms"

# vLLM serverless defaults (override via deploy env or manifest vllm_env).
VLLM_TOOL_CALL_PARSER = "qwen3_coder"
VLLM_ENABLE_AUTO_TOOL_CHOICE = "true"

DEFAULT_VLLM_TOOL_ENV: dict[str, str] = {
    "TOOL_CALL_PARSER": VLLM_TOOL_CALL_PARSER,
    "ENABLE_AUTO_TOOL_CHOICE": VLLM_ENABLE_AUTO_TOOL_CHOICE,
}


class ToolCallParseOutcome(str, Enum):
    SUCCESS = "success"
    MALFORMED = "malformed"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGS = "invalid_args"
    TEXT_FALLBACK = "text_fallback"
    JSON_FALLBACK = "json_fallback"
    EMPTY = "empty"


class ToolCallValidationError(ValueError):
    """A machine-classifiable tool-call validation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str = "$",
        outcome: ToolCallParseOutcome = ToolCallParseOutcome.INVALID_ARGS,
    ) -> None:
        self.code = code
        self.path = path
        self.message = message
        self.outcome = outcome
        super().__init__(f"{path}: {message} [{code}]")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Internal representation of one OpenAI-compatible tool."""

    name: str
    description: str
    parameters: Mapping[str, Any]

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One parsed tool invocation from a model response."""

    id: str | None
    name: str
    arguments_raw: str
    arguments: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Outcome of parsing a model response into CandidateAtom proposals."""

    outcome: ToolCallParseOutcome
    candidate: ModelCandidate | None = None
    tool_name: str | None = None
    error: ToolCallValidationError | None = None
    raw: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def ok(self) -> bool:
        return self.outcome is ToolCallParseOutcome.SUCCESS and self.candidate is not None


def resolve_vllm_tool_env(
    *,
    manifest_env: Mapping[str, str] | None = None,
    deploy_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve vLLM tool env with precedence: deploy_env > manifest_env > defaults."""
    resolved = dict(DEFAULT_VLLM_TOOL_ENV)
    if manifest_env:
        resolved.update({k: str(v) for k, v in manifest_env.items() if k in DEFAULT_VLLM_TOOL_ENV})
    if deploy_env:
        resolved.update({k: str(v) for k, v in deploy_env.items() if k in DEFAULT_VLLM_TOOL_ENV})
    # Process environment wins last (operator override at runtime).
    for key in DEFAULT_VLLM_TOOL_ENV:
        if key in os.environ:
            resolved[key] = os.environ[key]
    return resolved


def log_tool_call_event(
    event: str,
    *,
    outcome: ToolCallParseOutcome | None = None,
    tool_name: str | None = None,
    error_code: str | None = None,
    latency_s: float | None = None,
    atom_count: int | None = None,
) -> None:
    """Structured tool-call log line (no secrets, no raw transcript)."""
    payload: dict[str, Any] = {
        "event": event,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if outcome is not None:
        payload["outcome"] = outcome.value
    if tool_name is not None:
        payload["tool_name"] = tool_name
    if error_code is not None:
        payload["error_code"] = error_code
    if latency_s is not None:
        payload["latency_s"] = round(latency_s, 4)
    if atom_count is not None:
        payload["atom_count"] = atom_count
    logger.info(json.dumps(payload, sort_keys=True))


def _tool_call_function(call: Any) -> Mapping[str, Any] | None:
    if isinstance(call, Mapping):
        fn = call.get("function")
        return fn if isinstance(fn, Mapping) else None
    fn = getattr(call, "function", None)
    if fn is None:
        return None
    name = getattr(fn, "name", None)
    arguments = getattr(fn, "arguments", None)
    if name is None:
        return None
    return {"name": name, "arguments": arguments}


def _parse_arguments_raw(raw: str | Mapping[str, Any] | None) -> Mapping[str, Any]:
    if raw is None:
        raise ToolCallValidationError(
            "empty_tool_arguments",
            "tool call arguments were empty",
            outcome=ToolCallParseOutcome.EMPTY,
        )
    if isinstance(raw, Mapping):
        return raw
    if not str(raw).strip():
        raise ToolCallValidationError(
            "empty_tool_arguments",
            "tool call arguments were empty",
            outcome=ToolCallParseOutcome.EMPTY,
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolCallValidationError(
            "invalid_json",
            str(exc),
            outcome=ToolCallParseOutcome.INVALID_ARGS,
        ) from exc
    if not isinstance(parsed, Mapping):
        raise ToolCallValidationError(
            "type_error",
            "tool arguments must be an object",
            outcome=ToolCallParseOutcome.INVALID_ARGS,
        )
    return parsed


def _validate_candidate_mapping(mapping: Mapping[str, Any]) -> ModelCandidate:
    try:
        return ModelCandidate.from_dict(mapping)
    except AdaptError as exc:
        raise ToolCallValidationError(
            exc.code,
            exc.message,
            path=exc.path,
            outcome=ToolCallParseOutcome.INVALID_ARGS,
        ) from exc
    except EncounterError as exc:
        raise ToolCallValidationError(
            exc.code,
            exc.message,
            path=exc.path,
            outcome=ToolCallParseOutcome.INVALID_ARGS,
        ) from exc


class ToolCallParser:
    """Parse OpenAI-compatible responses into validated ModelCandidate batches."""

    submit_tool_name: str = SUBMIT_CANDIDATE_ATOMS_TOOL

    def parse_arguments(self, raw: str | Mapping[str, Any] | None) -> ModelCandidate:
        mapping = _parse_arguments_raw(raw)
        return _validate_candidate_mapping(mapping)

    def parse_tool_call(self, call: Any) -> ToolCallResult:
        fn = _tool_call_function(call)
        if fn is None:
            return ToolCallResult(
                outcome=ToolCallParseOutcome.MALFORMED,
                error=ToolCallValidationError(
                    "malformed_tool_call",
                    "tool call missing function payload",
                    outcome=ToolCallParseOutcome.MALFORMED,
                ),
            )
        name = str(fn.get("name") or "")
        arguments_raw = fn.get("arguments")
        call_id = None
        if isinstance(call, Mapping):
            call_id = call.get("id")
        else:
            call_id = getattr(call, "id", None)
        raw_str = "" if arguments_raw is None else str(arguments_raw)
        tool_call = ToolCall(id=call_id, name=name, arguments_raw=raw_str)
        if name != self.submit_tool_name:
            return ToolCallResult(
                outcome=ToolCallParseOutcome.UNKNOWN_TOOL,
                tool_name=name,
                raw=raw_str,
                tool_calls=(tool_call,),
                error=ToolCallValidationError(
                    "unknown_tool",
                    f"unexpected tool name: {name}",
                    outcome=ToolCallParseOutcome.UNKNOWN_TOOL,
                ),
            )
        try:
            mapping = _parse_arguments_raw(arguments_raw)
            candidate = _validate_candidate_mapping(mapping)
            tool_call = ToolCall(
                id=call_id,
                name=name,
                arguments_raw=raw_str,
                arguments=mapping,
            )
            return ToolCallResult(
                outcome=ToolCallParseOutcome.SUCCESS,
                candidate=candidate,
                tool_name=name,
                raw=raw_str,
                tool_calls=(tool_call,),
            )
        except ToolCallValidationError as exc:
            return ToolCallResult(
                outcome=exc.outcome,
                tool_name=name,
                raw=raw_str,
                tool_calls=(tool_call,),
                error=exc,
            )

    def parse_tool_calls(self, tool_calls: Sequence[Any] | None) -> ToolCallResult:
        if not tool_calls:
            return ToolCallResult(outcome=ToolCallParseOutcome.EMPTY)
        for call in tool_calls:
            result = self.parse_tool_call(call)
            if result.outcome is ToolCallParseOutcome.SUCCESS:
                return result
            if result.outcome is not ToolCallParseOutcome.UNKNOWN_TOOL:
                return result
        first = self.parse_tool_call(tool_calls[0])
        return first

    def parse_text_json(self, raw: str) -> ToolCallResult:
        """Backward-compat path: plain JSON text (structured_inference default)."""
        text = (raw or "").strip()
        if not text:
            return ToolCallResult(outcome=ToolCallParseOutcome.EMPTY, raw=text)
        try:
            candidate = ModelCandidate.from_json(text)
            return ToolCallResult(
                outcome=ToolCallParseOutcome.JSON_FALLBACK,
                candidate=candidate,
                raw=text,
            )
        except (AdaptError, EncounterError) as exc:
            code = exc.code if isinstance(exc, AdaptError) else getattr(exc, "code", "invalid_json")
            path = exc.path if isinstance(exc, AdaptError) else getattr(exc, "path", "$")
            message = exc.message if isinstance(exc, AdaptError) else str(exc)
            return ToolCallResult(
                outcome=ToolCallParseOutcome.INVALID_ARGS,
                raw=text,
                error=ToolCallValidationError(
                    code,
                    message,
                    path=path,
                    outcome=ToolCallParseOutcome.INVALID_ARGS,
                ),
            )

    def parse_message(self, message: Any) -> ToolCallResult:
        """Parse an OpenAI chat message (tool_calls preferred, then content JSON)."""
        if isinstance(message, Mapping):
            tool_calls = message.get("tool_calls")
            content = message.get("content")
        else:
            tool_calls = getattr(message, "tool_calls", None)
            content = getattr(message, "content", None)
        if tool_calls:
            result = self.parse_tool_calls(tool_calls)
            if result.outcome is not ToolCallParseOutcome.EMPTY:
                return result
        text = str(content or "").strip()
        if text:
            json_result = self.parse_text_json(text)
            if json_result.ok or json_result.outcome is ToolCallParseOutcome.INVALID_ARGS:
                return json_result
            return ToolCallResult(
                outcome=ToolCallParseOutcome.TEXT_FALLBACK,
                raw=text,
                error=ToolCallValidationError(
                    "text_fallback",
                    "model returned plain text instead of tool call",
                    outcome=ToolCallParseOutcome.TEXT_FALLBACK,
                ),
            )
        return ToolCallResult(outcome=ToolCallParseOutcome.EMPTY)

    def parse_openai_response(self, response: Any) -> ToolCallResult:
        if isinstance(response, Mapping):
            choices = response.get("choices") or []
            if not choices:
                return ToolCallResult(outcome=ToolCallParseOutcome.EMPTY)
            message = choices[0].get("message") or {}
            return self.parse_message(message)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ToolCallResult(outcome=ToolCallParseOutcome.EMPTY)
        message = choices[0].message
        return self.parse_message(message)

    def to_model_candidate(self, result: ToolCallResult) -> ModelCandidate:
        """Never silently invent atoms — empty batch on any non-success path."""
        if result.candidate is not None and result.outcome in {
            ToolCallParseOutcome.SUCCESS,
            ToolCallParseOutcome.JSON_FALLBACK,
        }:
            return result.candidate
        return ModelCandidate(atoms=())


def build_openai_chat_kwargs(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: Sequence[ToolDefinition],
    tool_choice: str | Mapping[str, Any] | None = "required",
    max_tokens: int = 1024,
    temperature: float = 0,
    use_json_object: bool = False,
) -> dict[str, Any]:
    """Build OpenAI chat.completions kwargs for tool or structured-json paths."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = [tool.to_openai_tool() for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
    elif use_json_object:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs
