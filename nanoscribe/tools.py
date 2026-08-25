"""OpenAI tools schema for P1 CandidateAtom extraction and coding stubs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nanoscribe.adapt import CANDIDATE_SCHEMA_VERSION, ModelCandidate
from nanoscribe.candidate_schema import candidate_batch_parameters_schema
from nanoscribe.tool_calling import (
    SUBMIT_CANDIDATE_ATOMS_TOOL,
    ToolCallParser,
    ToolCallResult,
    ToolCallValidationError,
    ToolDefinition,
    VLLM_ENABLE_AUTO_TOOL_CHOICE,
    VLLM_TOOL_CALL_PARSER,
    DEFAULT_VLLM_TOOL_ENV as VLLM_TOOL_ENV,
)

RUN_PYTHON_STUB_TOOL = "run_python"

__all__ = [
    "SUBMIT_CANDIDATE_ATOMS_TOOL",
    "RUN_PYTHON_STUB_TOOL",
    "VLLM_TOOL_CALL_PARSER",
    "VLLM_ENABLE_AUTO_TOOL_CHOICE",
    "VLLM_TOOL_ENV",
    "ToolCallParser",
    "ToolCallResult",
    "ToolCallValidationError",
    "execute_python_stub",
    "parse_openai_tool_calls",
    "parse_tool_arguments",
    "run_python_stub_tool",
    "scribing_tool_definitions",
    "scribing_tools",
    "submit_candidate_atoms_tool",
    "tool_choice_submit_candidates",
]


def submit_candidate_atoms_definition() -> ToolDefinition:
    return ToolDefinition(
        name=SUBMIT_CANDIDATE_ATOMS_TOOL,
        description=(
            "Submit clinical fact candidates extracted from a transcript. "
            "Quote-only evidence — never emit offsets, evidence_id, or normalized_value."
        ),
        parameters=candidate_batch_parameters_schema(),
    )


def submit_candidate_atoms_tool() -> dict[str, Any]:
    """OpenAI tools[] entry for structured CandidateAtom batch."""
    return submit_candidate_atoms_definition().to_openai_tool()


def run_python_stub_tool() -> dict[str, Any]:
    """Minimal coding tool for qwen3_coder agentic campaign workflows (stub executor)."""
    return ToolDefinition(
        name=RUN_PYTHON_STUB_TOOL,
        description=(
            "Run a short Python snippet in the campaign sandbox. "
            "Use for quick data checks — not for production PHI."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute"},
                "timeout_s": {
                    "type": "integer",
                    "description": "Wall-clock limit in seconds",
                    "default": 5,
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    ).to_openai_tool()


def scribing_tool_definitions(*, include_coding_stub: bool = False) -> list[ToolDefinition]:
    tools = [submit_candidate_atoms_definition()]
    if include_coding_stub:
        tools.append(
            ToolDefinition(
                name=RUN_PYTHON_STUB_TOOL,
                description=run_python_stub_tool()["function"]["description"],
                parameters=run_python_stub_tool()["function"]["parameters"],
            )
        )
    return tools


def scribing_tools(*, include_coding_stub: bool = False) -> list[dict[str, Any]]:
    """Canonical tool list for P1 scribing inference."""
    return [tool.to_openai_tool() for tool in scribing_tool_definitions(include_coding_stub=include_coding_stub)]


def tool_choice_submit_candidates() -> dict[str, Any]:
    """Force the submit_candidate_atoms tool (OpenAI tool_choice format)."""
    return {
        "type": "function",
        "function": {"name": SUBMIT_CANDIDATE_ATOMS_TOOL},
    }


def parse_tool_arguments(raw: str | dict[str, Any]) -> ModelCandidate:
    """Parse submit_candidate_atoms tool arguments into a ModelCandidate."""
    return ToolCallParser().parse_arguments(raw)


def parse_openai_tool_calls(tool_calls: Sequence[Any] | None) -> ModelCandidate | None:
    """Extract the first submit_candidate_atoms call from an OpenAI message."""
    result = ToolCallParser().parse_tool_calls(tool_calls)
    if result.ok:
        return result.candidate
    return None


def execute_python_stub(*, code: str, timeout_s: int = 5) -> dict[str, Any]:
    """Campaign-safe stub — does not execute arbitrary code in CI or default paths."""
    del timeout_s
    return {
        "ok": False,
        "stub": True,
        "error": "run_python is a schema stub; wire a sandbox before enabling execution",
        "code_preview": code[:200],
    }

