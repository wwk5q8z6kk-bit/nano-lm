"""API frontier teacher adapter — Track A capability ceiling via hosted LLM."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

from nanoscribe.adapt import ModelCandidateBatch, ModelInput, candidate_from_span_port_line
from nanoscribe.adapters import AtomSpec
from nanoscribe.egress import ExternalEgressTarget, require_external_egress
from nanoscribe.prompt import build_span_port_prompt, span_port_system_prompt

DEFAULT_API_MODEL = "gpt-4o-mini"
OPENAI_API_BASE_URL = "https://api.openai.com/v1"


def _openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package required for API teacher; pip install openai"
        ) from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key, base_url=OPENAI_API_BASE_URL)


def _generate_line(client, model: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": span_port_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=64,
    )
    text = response.choices[0].message.content or ""
    return text.strip().splitlines()[0].strip()


def generate_api_span_port_lines(
    model_input: ModelInput,
    atom_specs: Sequence[AtomSpec],
    *,
    model: str = DEFAULT_API_MODEL,
) -> tuple[dict[str, str], float, int]:
    """Run one API call per atom; return lines + latency + zero memory."""
    require_external_egress(model_input, ExternalEgressTarget.openai_api())
    client = _openai_client()
    started = time.perf_counter()
    lines: dict[str, str] = {}
    for spec in atom_specs:
        prompt = build_span_port_prompt(model_input.source, spec)
        lines[spec.atom_id] = _generate_line(client, model, prompt)
    latency_s = time.perf_counter() - started
    # Rough token cost estimate for telemetry (not billed here).
    return lines, latency_s, 0
