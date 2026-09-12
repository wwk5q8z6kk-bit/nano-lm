"""OpenAI tool-calling inference path for CandidateAtom extraction."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from nanoscribe.adapt import ModelCandidate, ModelCandidateBatch
from nanoscribe.adapters import AtomSpec
from nanoscribe.capabilities import CapabilityToolParser
from nanoscribe.capabilities.parser import ToolResult
from nanoscribe.inference.qwen3_coder import Qwen3CoderInferenceAdapter
from nanoscribe.inference.tool_registry import (
    agent_tool_definitions,
    allowed_capability_ids,
    resolve_inference_tool_choice,
    scribe_only_tool_definitions,
)
from nanoscribe.prompt import build_structured_candidate_prompt, tool_candidate_system_prompt
from nanoscribe.tool_calling import ToolCallParseOutcome, log_tool_call_event, resolve_vllm_tool_env


def _scribe_candidate_from_result(result: ToolResult) -> ModelCandidate:
    if result.candidate is not None and result.outcome in {
        ToolCallParseOutcome.SUCCESS,
        ToolCallParseOutcome.JSON_FALLBACK,
    }:
        return ModelCandidate(atoms=result.candidate.atoms)
    return ModelCandidate(atoms=())


def generate_tool_candidates(
    model_input,
    atom_specs: Sequence[AtomSpec],
    *,
    client,
    model: str,
    max_tokens: int = 1024,
    include_coding_stub: bool = False,
    include_agent_tools: bool = False,
    tool_choice: str | Mapping[str, Any] | None = None,
    vllm_env: Mapping[str, str] | None = None,
) -> tuple[ModelCandidateBatch, float, int]:
    """One batched tool-call per encounter (parallel path to structured JSON)."""
    allowed = allowed_capability_ids(
        include_scribe=True,
        include_summarize=include_agent_tools,
        include_table=include_agent_tools,
    )
    parser = CapabilityToolParser(allowed_capabilities=allowed)
    env = dict(vllm_env or resolve_vllm_tool_env())
    scribe_only = not include_agent_tools
    tools = (
        agent_tool_definitions(include_coding_stub=include_coding_stub)
        if include_agent_tools
        else scribe_only_tool_definitions(include_coding_stub=include_coding_stub)
    )
    resolved_choice = resolve_inference_tool_choice(
        tool_choice,
        env,
        scribe_only=scribe_only,
    )
    adapter = Qwen3CoderInferenceAdapter(env=env, parser=parser)
    prompt = build_structured_candidate_prompt(model_input.source, tuple(atom_specs))
    started = time.perf_counter()
    kwargs = adapter.chat_completion_kwargs(
        model=model,
        system_prompt=tool_candidate_system_prompt(),
        user_prompt=prompt,
        tools=tools,
        tool_choice=resolved_choice,
        max_tokens=max_tokens,
    )
    response = client.chat.completions.create(**kwargs)
    latency_s = time.perf_counter() - started
    result = adapter.parse_openai_response(response)
    batch = ModelCandidateBatch(atoms=_scribe_candidate_from_result(result).atoms)
    log_tool_call_event(
        "tool_inference_complete",
        outcome=result.outcome,
        tool_name=result.tool_name,
        error_code=result.error.code if result.error else None,
        latency_s=latency_s,
        atom_count=len(batch.atoms),
    )
    if result.outcome not in {
        ToolCallParseOutcome.SUCCESS,
        ToolCallParseOutcome.JSON_FALLBACK,
    }:
        log_tool_call_event(
            "tool_inference_degraded",
            outcome=result.outcome,
            tool_name=result.tool_name,
            error_code=result.error.code if result.error else None,
        )
    return batch, latency_s, 0
