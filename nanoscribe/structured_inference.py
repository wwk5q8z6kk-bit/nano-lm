"""OpenAI-compatible structured CandidateAtom generation."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from nanoscribe.adapt import ModelCandidate, ModelCandidateBatch
from nanoscribe.adapters import AtomSpec
from nanoscribe.prompt import build_structured_candidate_prompt, structured_candidate_system_prompt
from nanoscribe.tool_calling import ToolCallParser


def _parse_structured_response(raw: str) -> ModelCandidate:
    return ToolCallParser().to_model_candidate(ToolCallParser().parse_text_json(raw))


def _generate_structured(
    client,
    model: str,
    user_prompt: str,
    *,
    max_tokens: int,
    use_json_object: bool,
) -> str:
    kwargs: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": structured_candidate_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if use_json_object:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def generate_structured_candidates(
    model_input,
    atom_specs: Sequence[AtomSpec],
    *,
    client,
    model: str,
    max_tokens: int = 1024,
    use_json_object: bool = True,
    use_tools: bool = False,
    include_coding_stub: bool = False,
    include_agent_tools: bool = False,
    tool_choice: str | Mapping[str, Any] | None = None,
    vllm_env: Mapping[str, str] | None = None,
) -> tuple[ModelCandidateBatch, float, int]:
    """One batched structured call per encounter.

    Default path: ``response_format=json_object`` (backward compatible).
    When ``use_tools=True``, uses Qwen3CoderInferenceAdapter + CapabilityToolParser.
    """
    if use_tools:
        from nanoscribe.tool_inference import generate_tool_candidates

        return generate_tool_candidates(
            model_input,
            atom_specs,
            client=client,
            model=model,
            max_tokens=max_tokens,
            include_coding_stub=include_coding_stub,
            include_agent_tools=include_agent_tools,
            tool_choice=tool_choice,
            vllm_env=vllm_env,
        )

    started = time.perf_counter()
    prompt = build_structured_candidate_prompt(model_input.source, tuple(atom_specs))
    raw = _generate_structured(
        client,
        model,
        prompt,
        max_tokens=max_tokens,
        use_json_object=use_json_object,
    )
    batch = _parse_structured_response(raw)
    latency_s = time.perf_counter() - started
    return batch, latency_s, 0
