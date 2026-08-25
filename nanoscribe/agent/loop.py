"""Unified Nano agent loop — model tool calls, coding tools, and capabilities."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanoscribe.artifacts import Artifact
from nanoscribe.capabilities import CapabilityId, CapabilityToolParser, capability_for_tool
from nanoscribe.coding_tools import CodingToolExecutor, coding_tool_definitions, execute_coding_tool_call
from nanoscribe.inference.qwen3_coder import Qwen3CoderInferenceAdapter, normalize_openai_message
from nanoscribe.inference.tool_registry import (
    agent_tool_definitions,
    allowed_capability_ids,
    inference_tool_definitions,
    openai_tools_payload,
    resolve_inference_tool_choice,
    scribe_only_tool_definitions,
)
from nanoscribe.tool_calling import ToolCallParseOutcome, log_tool_call_event, resolve_vllm_tool_env

logger = logging.getLogger(__name__)

DEFAULT_AGENT_SYSTEM = (
    "You are Nano, a coding and analysis agent working in a repository sandbox. "
    "Use tools to inspect the codebase, run allowlisted commands, and submit structured "
    "artifacts when the task requires scribing, summarization, or tables. "
    "When you have a final answer, respond in plain text without calling more tools. "
    "Stay within the repository sandbox."
)

_CODING_TOOL_NAMES = frozenset(tool.name for tool in coding_tool_definitions())


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "offline"
    max_steps: int = 12
    timeout_s: float = 120.0
    sandbox_root: Path | None = None
    include_coding_tools: bool = True
    include_capabilities: bool = True
    scribe_only: bool = False
    capability_ids: tuple[CapabilityId, ...] = (
        CapabilityId.SCRIBE,
        CapabilityId.SUMMARIZE,
        CapabilityId.TABLE,
    )
    vllm_env: Mapping[str, str] | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    system_prompt: str = DEFAULT_AGENT_SYSTEM


@dataclass(frozen=True, slots=True)
class AgentStepLog:
    step: int
    assistant_content: str | None
    tool_calls: tuple[Mapping[str, Any], ...]
    tool_results: tuple[Mapping[str, Any], ...]
    latency_s: float
    finish_reason: str | None = None


@dataclass
class AgentRunResult:
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[AgentStepLog] = field(default_factory=list)
    stop_reason: str = "completed"
    artifacts: list[Artifact] = field(default_factory=list)
    final_content: str | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "final_content": self.final_content,
            "elapsed_s": round(self.elapsed_s, 4),
            "n_steps": len(self.steps),
            "n_artifacts": len(self.artifacts),
            "artifacts": [
                {
                    "artifact_type": artifact.artifact_type.value,
                    "schema_version": artifact.schema_version,
                    "capability_id": artifact.metadata.capability_id,
                }
                for artifact in self.artifacts
            ],
            "steps": [
                {
                    "step": step.step,
                    "finish_reason": step.finish_reason,
                    "latency_s": round(step.latency_s, 4),
                    "tool_calls": [dict(call) for call in step.tool_calls],
                    "tool_results": [dict(result) for result in step.tool_results],
                    "assistant_content": step.assistant_content,
                }
                for step in self.steps
            ],
        }


class AgentToolExecutor:
    """Dispatch tool calls to coding sandbox or capability validators."""

    def __init__(
        self,
        *,
        sandbox_root: Path,
        capability_parser: CapabilityToolParser,
    ) -> None:
        self._sandbox_root = sandbox_root
        self._capability_parser = capability_parser
        self._coding = CodingToolExecutor(sandbox_root=sandbox_root)

    def execute(self, name: str, arguments_raw: str, *, call_id: str) -> dict[str, Any]:
        if capability_for_tool(name) is not None:
            return self._execute_capability(name, arguments_raw, call_id=call_id)
        if name in _CODING_TOOL_NAMES:
            result = execute_coding_tool_call(
                name,
                arguments_raw,
                sandbox_root=self._sandbox_root,
            )
            return result.to_dict()
        return {"ok": False, "tool": name, "error": f"unknown tool: {name}"}

    def _execute_capability(self, name: str, arguments_raw: str, *, call_id: str) -> dict[str, Any]:
        fake_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments_raw},
        }
        result = self._capability_parser.parse_tool_call(fake_call)
        if result.ok and result.artifact is not None:
            return {
                "ok": True,
                "tool": name,
                "capability_id": result.capability_id.value if result.capability_id else None,
                "artifact_type": result.artifact.artifact_type.value,
                "schema_version": result.artifact.schema_version,
                "data": result.artifact.data,
            }
        error = result.error.message if result.error else f"capability validation failed: {result.outcome.value}"
        return {
            "ok": False,
            "tool": name,
            "outcome": result.outcome.value,
            "error": error,
        }


class NanoAgent:
    """Messages → model (tool_choice auto) → parse → execute → repeat until done."""

    def __init__(self, client: Any, config: AgentConfig | None = None) -> None:
        self._client = client
        self._config = config or AgentConfig()
        self._env = dict(self._config.vllm_env or resolve_vllm_tool_env())
        allowed = self._resolve_allowed_capabilities()
        self._capability_parser = CapabilityToolParser(allowed_capabilities=allowed)
        sandbox = (self._config.sandbox_root or Path.cwd()).resolve()
        self._executor = AgentToolExecutor(
            sandbox_root=sandbox,
            capability_parser=self._capability_parser,
        )
        self._adapter = Qwen3CoderInferenceAdapter(env=self._env, parser=self._capability_parser)

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def sandbox_root(self) -> Path:
        return (self._config.sandbox_root or Path.cwd()).resolve()

    def tool_definitions(self) -> list[Any]:
        tools: list[Any] = []
        if self._config.include_coding_tools:
            tools.extend(coding_tool_definitions())
        if self._config.include_capabilities:
            if self._config.scribe_only:
                tools.extend(scribe_only_tool_definitions())
            else:
                tools.extend(
                    inference_tool_definitions(
                        include_scribe=CapabilityId.SCRIBE in self._config.capability_ids,
                        include_summarize=CapabilityId.SUMMARIZE in self._config.capability_ids,
                        include_table=CapabilityId.TABLE in self._config.capability_ids,
                    )
                )
        return tools

    def _resolve_allowed_capabilities(self) -> tuple[CapabilityId, ...]:
        if self._config.scribe_only:
            return allowed_capability_ids(include_scribe=True)
        return allowed_capability_ids(
            include_scribe=CapabilityId.SCRIBE in self._config.capability_ids,
            include_summarize=CapabilityId.SUMMARIZE in self._config.capability_ids,
            include_table=CapabilityId.TABLE in self._config.capability_ids,
        )

    def _build_chat_kwargs(self, messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        tools = self.tool_definitions()
        tool_choice = resolve_inference_tool_choice(
            None,
            self._env,
            scribe_only=self._config.scribe_only,
        )
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": [dict(message) for message in messages],
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "tools": openai_tools_payload(tools),
        }
        if tool_choice != "none":
            kwargs["tool_choice"] = tool_choice
        return kwargs

    def run(
        self,
        user_message: str,
        *,
        messages: Sequence[Mapping[str, Any]] | None = None,
    ) -> AgentRunResult:
        started = time.perf_counter()
        deadline = started + self._config.timeout_s
        transcript: list[dict[str, Any]] = [dict(item) for item in messages] if messages else []
        if not transcript or transcript[0].get("role") != "system":
            transcript.insert(0, {"role": "system", "content": self._config.system_prompt})
        transcript.append({"role": "user", "content": user_message})

        steps: list[AgentStepLog] = []
        artifacts: list[Artifact] = []
        stop_reason = "max_steps"
        final_content: str | None = None

        for step_index in range(1, self._config.max_steps + 1):
            if time.perf_counter() > deadline:
                stop_reason = "timeout"
                break

            step_started = time.perf_counter()
            response = self._client.chat.completions.create(**self._build_chat_kwargs(transcript))
            choice = response.choices[0]
            normalized = normalize_openai_message(choice.message)
            content = normalized.get("content")
            tool_calls = normalized.get("tool_calls") or []
            finish_reason = choice.finish_reason

            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls
            transcript.append(assistant_message)

            if not tool_calls:
                final_content = str(content or "").strip() or None
                stop_reason = "completed"
                steps.append(
                    AgentStepLog(
                        step=step_index,
                        assistant_content=final_content,
                        tool_calls=(),
                        tool_results=(),
                        latency_s=time.perf_counter() - step_started,
                        finish_reason=finish_reason,
                    )
                )
                break

            tool_results: list[dict[str, Any]] = []
            for call in tool_calls:
                fn = call.get("function") if isinstance(call, Mapping) else getattr(call, "function", None)
                if fn is None:
                    continue
                if isinstance(fn, Mapping):
                    name = str(fn.get("name") or "")
                    arguments_raw = str(fn.get("arguments") or "")
                else:
                    name = str(getattr(fn, "name", "") or "")
                    arguments_raw = str(getattr(fn, "arguments", "") or "")
                call_id = (
                    str(call.get("id"))
                    if isinstance(call, Mapping)
                    else str(getattr(call, "id", f"call_{step_index}"))
                )
                payload = self._executor.execute(name, arguments_raw, call_id=call_id)
                tool_results.append(payload)
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(payload, separators=(",", ":")),
                    }
                )
                if payload.get("ok") and capability_for_tool(name) is not None:
                    parse_result = self._capability_parser.parse_tool_call(
                        {
                            "id": call_id,
                            "function": {"name": name, "arguments": arguments_raw},
                        }
                    )
                    if parse_result.artifact is not None:
                        artifacts.append(parse_result.artifact)
                log_tool_call_event(
                    "agent_tool_execute",
                    tool_name=name,
                    outcome=ToolCallParseOutcome.SUCCESS if payload.get("ok") else ToolCallParseOutcome.INVALID_ARGS,
                    error_code=None if payload.get("ok") else "tool_execution_failed",
                )

            steps.append(
                AgentStepLog(
                    step=step_index,
                    assistant_content=str(content or "").strip() or None,
                    tool_calls=tuple(
                        dict(call) if isinstance(call, Mapping) else {"function": str(call)}
                        for call in tool_calls
                    ),
                    tool_results=tuple(tool_results),
                    latency_s=time.perf_counter() - step_started,
                    finish_reason=finish_reason,
                )
            )

        elapsed = time.perf_counter() - started
        log_tool_call_event(
            "agent_run_complete",
            outcome=ToolCallParseOutcome.SUCCESS if stop_reason == "completed" else ToolCallParseOutcome.EMPTY,
            latency_s=elapsed,
            atom_count=len(artifacts),
        )
        return AgentRunResult(
            messages=transcript,
            steps=steps,
            stop_reason=stop_reason,
            artifacts=artifacts,
            final_content=final_content,
            elapsed_s=elapsed,
        )


def default_agent_tools_payload(*, include_coding: bool = True) -> list[dict[str, Any]]:
    """OpenAI tools[] for agent loop — coding + scribe/summarize/table."""
    tools: list[Any] = []
    if include_coding:
        tools.extend(coding_tool_definitions())
    tools.extend(agent_tool_definitions())
    return openai_tools_payload(tools)
