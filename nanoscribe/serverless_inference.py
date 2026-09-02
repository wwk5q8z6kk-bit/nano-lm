"""RunPod Serverless inference — OpenAI-compatible vLLM endpoints."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from urllib.parse import urlparse

from nanoscribe.adapters import AtomSpec
from nanoscribe.egress import ExternalEgressTarget, require_external_egress
from nanoscribe.prompt import build_span_port_prompt, span_port_system_prompt


def _resolve_api_key(explicit: str | None) -> str:
    if explicit:
        return explicit
    api_key = os.environ.get("RUNPOD_API_KEY")
    if api_key:
        return api_key
    config_path = os.path.expanduser("~/.runpod/config.toml")
    if os.path.isfile(config_path):
        try:
            import tomllib

            with open(config_path, "rb") as handle:
                data = tomllib.load(handle)
            key = data.get("apiKey") or data.get("apikey")
            if key:
                return str(key)
        except OSError:
            pass
    raise RuntimeError(
        "RUNPOD_API_KEY not set and ~/.runpod/config.toml unavailable"
    )


def _validated_endpoint_id(value: str) -> str:
    if not value.isascii() or not value.isalnum():
        raise ValueError("RunPod endpoint ID must be ASCII alphanumeric")
    return value


def _endpoint_id_from_canonical_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.runpod.ai"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("RunPod Serverless URL must use canonical api.runpod.ai HTTPS")
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 4 or parts[0] != "v2" or parts[2:] != ["openai", "v1"]:
        raise ValueError("RunPod Serverless URL must be /v2/{endpoint}/openai/v1")
    return _validated_endpoint_id(parts[1])


def _resolve_endpoint_id(endpoint_id: str | None, base_url: str | None) -> str:
    configured = endpoint_id or os.environ.get("RUNPOD_SERVERLESS_ENDPOINT_ID")
    canonical_endpoint = _endpoint_id_from_canonical_url(base_url) if base_url else None
    if configured:
        configured = parse_endpoint_id(configured)
        configured = _validated_endpoint_id(configured)
    if configured and canonical_endpoint and configured != canonical_endpoint:
        raise ValueError("endpoint_id must match the canonical RunPod Serverless URL")
    endpoint = configured or canonical_endpoint
    if not endpoint:
        raise RuntimeError(
            "RUNPOD_SERVERLESS_ENDPOINT_ID not set and no base_url provided"
        )
    return endpoint


def _openai_base_url(endpoint_id: str) -> str:
    endpoint = _resolve_endpoint_id(endpoint_id, None)
    return endpoint_openai_url(endpoint)


def _openai_client(*, endpoint_id: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package required for serverless inference; pip install openai"
        ) from exc
    api_key = _resolve_api_key(None)
    return OpenAI(api_key=api_key, base_url=_openai_base_url(endpoint_id))


def _generate_line(client, model: str, user_prompt: str, *, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": span_port_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    return text.strip().splitlines()[0].strip()


def generate_serverless_span_port_lines(
    model_input,
    atom_specs: Sequence[AtomSpec],
    *,
    model: str,
    endpoint_id: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 64,
) -> tuple[dict[str, str], float, int]:
    """Run one serverless call per atom; return lines + latency + zero memory."""
    resolved_endpoint_id = _resolve_endpoint_id(endpoint_id, base_url)
    require_external_egress(
        model_input,
        ExternalEgressTarget.runpod_serverless(resolved_endpoint_id),
    )
    client = _openai_client(endpoint_id=resolved_endpoint_id)
    started = time.perf_counter()
    lines: dict[str, str] = {}
    for spec in atom_specs:
        prompt = build_span_port_prompt(model_input.source, spec)
        lines[spec.atom_id] = _generate_line(
            client, model, prompt, max_tokens=max_tokens
        )
    latency_s = time.perf_counter() - started
    return lines, latency_s, 0


def endpoint_openai_url(endpoint_id: str) -> str:
    """Canonical OpenAI-compatible base URL for a RunPod serverless endpoint."""
    return f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1"


def endpoint_native_urls(endpoint_id: str) -> dict[str, str]:
    """RunPod native invoke URLs for an endpoint."""
    base = f"https://api.runpod.ai/v2/{endpoint_id}"
    return {
        "health": f"{base}/health",
        "run": f"{base}/run",
        "runsync": f"{base}/runsync",
        "openai_v1": f"{base}/openai/v1",
    }


def parse_endpoint_id(value: str) -> str:
    """Accept bare endpoint id or full RunPod URL."""
    if "://" not in value:
        return _validated_endpoint_id(value)
    return _endpoint_id_from_canonical_url(value)
