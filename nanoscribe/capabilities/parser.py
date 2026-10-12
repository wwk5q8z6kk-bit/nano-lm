"""Capability-oriented tool parsing — converges JSON and tool calls on artifact validators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nanoscribe.adapt import AdaptError, ModelCandidate
from nanoscribe.artifacts import Artifact, ArtifactError, ArtifactMetadata
from nanoscribe.artifacts.chart_spec import ChartSpec
from nanoscribe.artifacts.diagram_spec import DiagramSpec
from nanoscribe.capabilities.registry import (
    CapabilityId,
    capability_for_tool,
    get_capability,
)
from nanoscribe.capabilities.scribing import artifact_from_scribing
from nanoscribe.capabilities.summarize import artifact_from_summary, validate_summary_payload
from nanoscribe.capabilities.table import artifact_from_table, validate_table_payload
from nanoscribe.tool_calling import (
    ToolCall,
    ToolCallParseOutcome,
    ToolCallParser,
    ToolCallResult,
    ToolCallValidationError,
    _parse_arguments_raw,
    _tool_call_function,
)


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Normalized parse outcome — artifact envelope when validation succeeds."""

    outcome: ToolCallParseOutcome
    capability_id: CapabilityId | None = None
    artifact: Artifact | None = None
    candidate: ModelCandidate | None = None
    tool_name: str | None = None
    error: ToolCallValidationError | None = None
    raw: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @property
    def ok(self) -> bool:
        return self.outcome is ToolCallParseOutcome.SUCCESS and self.artifact is not None


def _validation_error_from_adapt(exc: AdaptError) -> ToolCallValidationError:
    return ToolCallValidationError(
        exc.code,
        exc.message,
        path=exc.path,
        outcome=ToolCallParseOutcome.INVALID_ARGS,
    )


def _validation_error_from_artifact(exc: ArtifactError) -> ToolCallValidationError:
    return ToolCallValidationError(
        exc.code,
        exc.message,
        path=exc.path,
        outcome=ToolCallParseOutcome.INVALID_ARGS,
    )


def _validate_capability_payload(capability_id: CapabilityId, mapping: Mapping[str, Any]) -> Artifact:
    if capability_id is CapabilityId.SCRIBE:
        try:
            candidate = ModelCandidate.from_dict(mapping)
        except AdaptError as exc:
            raise _validation_error_from_adapt(exc) from exc
        return artifact_from_scribing(candidate)
    if capability_id is CapabilityId.SUMMARIZE:
        try:
            summary = validate_summary_payload(dict(mapping))
        except ArtifactError as exc:
            raise _validation_error_from_artifact(exc) from exc
        return artifact_from_summary(summary)
    if capability_id is CapabilityId.TABLE:
        try:
            table = validate_table_payload(dict(mapping))
        except ArtifactError as exc:
            raise _validation_error_from_artifact(exc) from exc
        return artifact_from_table(table)
    if capability_id is CapabilityId.CHART:
        try:
            chart = ChartSpec.from_dict(mapping)
        except ArtifactError as exc:
            raise _validation_error_from_artifact(exc) from exc
        spec = get_capability(CapabilityId.CHART)
        return Artifact(
            artifact_type=spec.artifact_type,
            schema_version=spec.schema_version,
            data=chart.to_dict(),
            metadata=ArtifactMetadata(capability_id=CapabilityId.CHART.value),
        )
    if capability_id is CapabilityId.DIAGRAM:
        try:
            diagram = DiagramSpec.from_dict(mapping)
        except ArtifactError as exc:
            raise _validation_error_from_artifact(exc) from exc
        spec = get_capability(CapabilityId.DIAGRAM)
        return Artifact(
            artifact_type=spec.artifact_type,
            schema_version=spec.schema_version,
            data=diagram.to_dict(),
            metadata=ArtifactMetadata(capability_id=CapabilityId.DIAGRAM.value),
        )
    raise ToolCallValidationError(
        "unsupported_capability",
        f"capability {capability_id.value} has no validator",
        outcome=ToolCallParseOutcome.UNKNOWN_TOOL,
    )


class CapabilityToolParser:
    """Parse provider tool calls into validated capability artifacts."""

    def __init__(self, *, allowed_capabilities: Sequence[CapabilityId] | None = None) -> None:
        self._allowed = frozenset(allowed_capabilities) if allowed_capabilities else None
        self._legacy = ToolCallParser()

    def parse_tool_call(self, call: Any) -> ToolResult:
        fn = _tool_call_function(call)
        if fn is None:
            return ToolResult(
                outcome=ToolCallParseOutcome.MALFORMED,
                error=ToolCallValidationError(
                    "malformed_tool_call",
                    "tool call missing function payload",
                    outcome=ToolCallParseOutcome.MALFORMED,
                ),
            )
        name = str(fn.get("name") or "")
        arguments_raw = fn.get("arguments")
        call_id = call.get("id") if isinstance(call, Mapping) else getattr(call, "id", None)
        raw_str = "" if arguments_raw is None else str(arguments_raw)
        tool_call = ToolCall(id=call_id, name=name, arguments_raw=raw_str)
        spec = capability_for_tool(name)
        if spec is None:
            return ToolResult(
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
        if self._allowed is not None and spec.capability_id not in self._allowed:
            return ToolResult(
                outcome=ToolCallParseOutcome.UNKNOWN_TOOL,
                tool_name=name,
                raw=raw_str,
                tool_calls=(tool_call,),
                error=ToolCallValidationError(
                    "capability_not_allowed",
                    f"capability not allowed: {spec.capability_id.value}",
                    outcome=ToolCallParseOutcome.UNKNOWN_TOOL,
                ),
            )
        try:
            mapping = _parse_arguments_raw(arguments_raw)
            artifact = _validate_capability_payload(spec.capability_id, mapping)
            tool_call = ToolCall(id=call_id, name=name, arguments_raw=raw_str, arguments=mapping)
            candidate = None
            if spec.capability_id is CapabilityId.SCRIBE:
                candidate = ModelCandidate.from_dict(mapping)
            return ToolResult(
                outcome=ToolCallParseOutcome.SUCCESS,
                capability_id=spec.capability_id,
                artifact=artifact,
                candidate=candidate,
                tool_name=name,
                raw=raw_str,
                tool_calls=(tool_call,),
            )
        except ToolCallValidationError as exc:
            return ToolResult(
                outcome=exc.outcome,
                capability_id=spec.capability_id,
                tool_name=name,
                raw=raw_str,
                tool_calls=(tool_call,),
                error=exc,
            )

    def parse_tool_calls(self, tool_calls: Sequence[Any] | None) -> ToolResult:
        if not tool_calls:
            return ToolResult(outcome=ToolCallParseOutcome.EMPTY)
        for call in tool_calls:
            result = self.parse_tool_call(call)
            if result.outcome is ToolCallParseOutcome.SUCCESS:
                return result
            if result.outcome is not ToolCallParseOutcome.UNKNOWN_TOOL:
                return result
        return self.parse_tool_call(tool_calls[0])

    def parse_message(self, message: Any) -> ToolResult:
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
            legacy = self._legacy.parse_text_json(text)
            if legacy.candidate is not None and legacy.outcome in (
                ToolCallParseOutcome.SUCCESS,
                ToolCallParseOutcome.JSON_FALLBACK,
            ):
                artifact = artifact_from_scribing(legacy.candidate)
                return ToolResult(
                    outcome=legacy.outcome,
                    capability_id=CapabilityId.SCRIBE,
                    artifact=artifact,
                    candidate=legacy.candidate,
                    raw=text,
                )
            if legacy.error is not None:
                return ToolResult(
                    outcome=legacy.outcome,
                    capability_id=CapabilityId.SCRIBE,
                    raw=text,
                    error=legacy.error,
                )
            return ToolResult(
                outcome=ToolCallParseOutcome.TEXT_FALLBACK,
                raw=text,
                error=ToolCallValidationError(
                    "text_fallback",
                    "model returned plain text instead of tool call",
                    outcome=ToolCallParseOutcome.TEXT_FALLBACK,
                ),
            )
        return ToolResult(outcome=ToolCallParseOutcome.EMPTY)

    def parse_openai_response(self, response: Any) -> ToolResult:
        if isinstance(response, Mapping):
            choices = response.get("choices") or []
            if not choices:
                return ToolResult(outcome=ToolCallParseOutcome.EMPTY)
            message = choices[0].get("message") or {}
            return self.parse_message(message)
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ToolResult(outcome=ToolCallParseOutcome.EMPTY)
        return self.parse_message(choices[0].message)
