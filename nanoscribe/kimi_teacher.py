"""RunPod Public Endpoint — Kimi K3 frontier teacher."""

from __future__ import annotations

import time
from collections.abc import Sequence

from nanoscribe.adapt import ModelCandidateBatch, candidate_from_span_port_line
from nanoscribe.adapters import AtomSpec
from nanoscribe.prompt import build_span_port_prompt, span_port_system_prompt
from nanoscribe.runpod_openai import (
    KIMI_K3_MODEL,
    RUNPOD_KIMI_PUBLIC_BASE,
    openai_client_for_runpod,
)
from nanoscribe.structured_inference import generate_structured_candidates
from nanoscribe.serverless_endpoint import resolve_serverless_endpoint_id
from nanoscribe.tracks import SERVERLESS_STRONG_MODEL


def kimi_preflight(*, timeout_hint_s: float = 30.0, retries: int = 2) -> dict[str, object]:
    """Tiny fail-fast Kimi K3 public endpoint probe with retries."""
    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            client = openai_client_for_runpod(RUNPOD_KIMI_PUBLIC_BASE)
            started = time.perf_counter()
            response = client.chat.completions.create(
                model=KIMI_K3_MODEL,
                messages=[
                    {"role": "user", "content": "Reply with exactly: KIMI_OK"},
                ],
                temperature=0,
                max_tokens=8,
                timeout=timeout_hint_s,
            )
            text = (response.choices[0].message.content or "").strip()
            latency_s = time.perf_counter() - started
            ok = "KIMI_OK" in text or text == "KIMI_OK"
            if ok:
                return {
                    "ok": True,
                    "model": KIMI_K3_MODEL,
                    "base_url": RUNPOD_KIMI_PUBLIC_BASE,
                    "latency_s": round(latency_s, 4),
                    "response": text,
                    "attempt": attempt,
                }
            last_error = f"unexpected response: {text!r}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(1.0 * (attempt + 1))
    return {
        "ok": False,
        "model": KIMI_K3_MODEL,
        "base_url": RUNPOD_KIMI_PUBLIC_BASE,
        "error": last_error,
        "attempts": retries + 1,
    }


def alternate_managed_frontier_probe(*, max_wait_s: float = 300.0) -> dict[str, object]:
    """Probe Qwen serverless when RUNPOD_SERVERLESS_ENDPOINT_ID is set."""
    from nanoscribe.serverless_config import fetch_health

    started = time.perf_counter()
    try:
        endpoint_id = resolve_serverless_endpoint_id(required=False)
        if not endpoint_id:
            return {
                "ok": False,
                "alternate": "serverless/qwen3.8-27b-structured",
                "error": "RUNPOD_SERVERLESS_ENDPOINT_ID not set (prior endpoint deleted)",
                "role": "managed_frontier_proxy_not_kimi",
            }
        health = fetch_health(endpoint_id)
        elapsed = time.perf_counter() - started
        if elapsed > max_wait_s:
            return {"ok": False, "reason": "timeout", "elapsed_s": round(elapsed, 2)}
        jobs = health.get("jobs", {})
        workers = health.get("workers", {})
        return {
            "ok": True,
            "alternate": "serverless/qwen3.8-27b-structured",
            "model": SERVERLESS_STRONG_MODEL,
            "endpoint_id": endpoint_id,
            "role": "managed_frontier_proxy_not_kimi",
            "health": {"jobs": jobs, "workers": workers},
            "latency_s": round(elapsed, 4),
            "note": "NOT labeled as capability ceiling — practical proxy until Kimi recovers",
        }
    except Exception as exc:
        return {
            "ok": False,
            "alternate": "serverless/qwen3.8-27b-structured",
            "error": f"{type(exc).__name__}: {exc}",
        }


def kimi_preflight_with_fallback() -> dict[str, object]:
    """Kimi retry then one alternate managed frontier probe."""
    kimi = kimi_preflight(retries=2)
    if kimi.get("ok"):
        return {"kimi": kimi, "frontier_lane": "kimi_k3", "blocked": False}
    alternate = alternate_managed_frontier_probe()
    return {
        "kimi": kimi,
        "alternate_managed_frontier": alternate,
        "frontier_lane": "qwen_proxy" if alternate.get("ok") else "blocked",
        "blocked": not alternate.get("ok"),
        "blocker": "kimi_k3_endpoint_500" if not kimi.get("ok") else None,
    }


def _generate_line(client, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=KIMI_K3_MODEL,
        messages=[
            {"role": "system", "content": span_port_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=64,
    )
    text = response.choices[0].message.content or ""
    return text.strip().splitlines()[0].strip()


def generate_kimi_span_port_lines(
    model_input,
    atom_specs: Sequence[AtomSpec],
) -> tuple[dict[str, str], float, int]:
    client = openai_client_for_runpod(RUNPOD_KIMI_PUBLIC_BASE)
    started = time.perf_counter()
    lines: dict[str, str] = {}
    for spec in atom_specs:
        prompt = build_span_port_prompt(model_input.source, spec)
        lines[spec.atom_id] = _generate_line(client, prompt)
    latency_s = time.perf_counter() - started
    return lines, latency_s, 0


def generate_kimi_structured_candidates(
    model_input,
    atom_specs: Sequence[AtomSpec],
    *,
    max_tokens: int = 512,
) -> tuple[ModelCandidateBatch, float, int]:
    client = openai_client_for_runpod(RUNPOD_KIMI_PUBLIC_BASE)
    return generate_structured_candidates(
        model_input,
        atom_specs,
        client=client,
        model=KIMI_K3_MODEL,
        max_tokens=max_tokens,
        use_json_object=True,
    )


def kimi_span_port_batch_to_candidates(
    model_input,
    atom_specs: Sequence[AtomSpec],
    lines: dict[str, str],
) -> ModelCandidateBatch:
    from nanoscribe.adapt import CandidateAtom

    atoms: list[CandidateAtom] = []
    for spec in atom_specs:
        raw_line = lines.get(spec.atom_id, "NOT_MENTIONED")
        atoms.append(
            candidate_from_span_port_line(
                atom_id=spec.atom_id,
                atom_type=spec.atom_type,
                raw_value=spec.raw_value,
                raw_line=raw_line,
                speaker=spec.speaker,
                experiencer=spec.experiencer,
                temporality=spec.temporality,
            )
        )
    return ModelCandidateBatch(atoms=tuple(atoms))
