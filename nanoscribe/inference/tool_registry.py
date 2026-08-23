"""Inference-time tool definitions and tool_choice resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nanoscribe.capabilities import CapabilityId
from nanoscribe.capabilities.scribing import submit_candidate_atoms_definition
from nanoscribe.capabilities.summarize import submit_summary_definition
from nanoscribe.capabilities.table import submit_table_definition
from nanoscribe.inference.qwen3_coder import auto_tool_choice_enabled, build_tool_choice
from nanoscribe.tool_calling import ToolDefinition, resolve_vllm_tool_env
from nanoscribe.tools import RUN_PYTHON_STUB_TOOL, tool_choice_submit_candidates


def inference_tool_definitions(
    *,
    include_scribe: bool = True,
    include_summarize: bool = False,
    include_table: bool = False,
    include_coding_stub: bool = False,
) -> list[ToolDefinition]:
    """OpenAI tool definitions for inference — scribe default; agent tools optional."""
    tools: list[ToolDefinition] = []
    if include_scribe:
        tools.append(submit_candidate_atoms_definition())
    if include_summarize:
        tools.append(submit_summary_definition())
    if include_table:
        tools.append(submit_table_definition())
    if include_coding_stub:
        from nanoscribe.tools import run_python_stub_tool

        stub = run_python_stub_tool()
        tools.append(
            ToolDefinition(
                name=RUN_PYTHON_STUB_TOOL,
                description=stub["function"]["description"],
                parameters=stub["function"]["parameters"],
            )
        )
    return tools


def scribe_only_tool_definitions(*, include_coding_stub: bool = False) -> list[ToolDefinition]:
    return inference_tool_definitions(
        include_scribe=True,
        include_summarize=False,
        include_table=False,
        include_coding_stub=include_coding_stub,
    )


def agent_tool_definitions(*, include_coding_stub: bool = False) -> list[ToolDefinition]:
    """Scribe + summarize + table for multi-capability agent inference."""
    return inference_tool_definitions(
        include_scribe=True,
        include_summarize=True,
        include_table=True,
        include_coding_stub=include_coding_stub,
    )


def allowed_capability_ids(
    *,
    include_scribe: bool = True,
    include_summarize: bool = False,
    include_table: bool = False,
) -> tuple[CapabilityId, ...]:
    allowed: list[CapabilityId] = []
    if include_scribe:
        allowed.append(CapabilityId.SCRIBE)
    if include_summarize:
        allowed.append(CapabilityId.SUMMARIZE)
    if include_table:
        allowed.append(CapabilityId.TABLE)
    return tuple(allowed)


def resolve_inference_tool_choice(
    tool_choice: str | Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    *,
    scribe_only: bool = True,
    forced_tool: str | None = None,
) -> str | Mapping[str, Any]:
    """Resolve tool_choice: explicit override wins; else env-driven auto/required."""
    if tool_choice is not None:
        return tool_choice
    resolved_env = dict(env or resolve_vllm_tool_env())
    if forced_tool is not None:
        return build_tool_choice(env=resolved_env, forced_tool=forced_tool)
    if auto_tool_choice_enabled(resolved_env):
        return "auto"
    if scribe_only:
        return tool_choice_submit_candidates()
    return "required"


def openai_tools_payload(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    return [tool.to_openai_tool() for tool in tools]
