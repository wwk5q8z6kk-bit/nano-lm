"""OpenAI-compatible structured CandidateAtom generation."""

from __future__ import annotations

import time
from collections.abc import Sequence

from nanoscribe.adapt import AdaptError, ModelCandidate, ModelCandidateBatch
from nanoscribe.encounter import EncounterError
from nanoscribe.adapters import AtomSpec
from nanoscribe.prompt import build_structured_candidate_prompt, structured_candidate_system_prompt


def _parse_structured_response(raw: str) -> ModelCandidate:
    try:
        return ModelCandidate.from_json(raw)
    except (AdaptError, EncounterError):
        return ModelCandidate(atoms=())


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
) -> tuple[ModelCandidateBatch, float, int]:
    """One batched structured call per encounter."""
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
