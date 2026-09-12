"""RunPod Serverless inference — OpenAI-compatible vLLM endpoints."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from nanoscribe.adapters import AtomSpec
from nanoscribe.prompt import build_span_port_prompt, span_port_system_prompt
from nanoscribe.serverless_endpoint import (
    DELETED_QWEN_SERVERLESS_ENDPOINT_ID,
    parse_endpoint_id,
    resolve_serverless_endpoint_id,
)


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


def _openai_base_url(endpoint_id: str | None, base_url: str | None) -> str:
    if base_url:
        return base_url.rstrip("/")
    if endpoint_id == DELETED_QWEN_SERVERLESS_ENDPOINT_ID:
        endpoint_id = None
    endpoint = endpoint_id or resolve_serverless_endpoint_id()
    return f"https://api.runpod.ai/v2/{endpoint}/openai/v1"


def _openai_client(*, endpoint_id: str | None = None, base_url: str | None = None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package required for serverless inference; pip install openai"
        ) from exc
    api_key = _resolve_api_key(None)
    return OpenAI(api_key=api_key, base_url=_openai_base_url(endpoint_id, base_url))


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
    client = _openai_client(endpoint_id=endpoint_id, base_url=base_url)
    started = time.perf_counter()
    lines: dict[str, str] = {}
    for spec in atom_specs:
        prompt = build_span_port_prompt(model_input.source, spec)
        lines[spec.atom_id] = _generate_line(
            client, model, prompt, max_tokens=max_tokens
        )
    latency_s = time.perf_counter() - started
    return lines, latency_s, 0


def generate_serverless_structured_candidates(
    model_input,
    atom_specs: Sequence[AtomSpec],
    *,
    model: str,
    endpoint_id: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 512,
    use_json_object: bool = True,
    use_tools: bool = False,
    include_agent_tools: bool = False,
    tool_choice: str | dict[str, Any] | None = None,
    vllm_env: dict[str, str] | None = None,
) -> tuple:
    from nanoscribe.structured_inference import generate_structured_candidates

    client = _openai_client(endpoint_id=endpoint_id, base_url=base_url)
    return generate_structured_candidates(
        model_input,
        atom_specs,
        client=client,
        model=model,
        max_tokens=max_tokens,
        use_json_object=use_json_object,
        use_tools=use_tools,
        include_agent_tools=include_agent_tools,
        tool_choice=tool_choice,
        vllm_env=vllm_env,
    )


def generate_serverless_tool_candidates(
    model_input,
    atom_specs: Sequence[AtomSpec],
    *,
    model: str,
    endpoint_id: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 1024,
    include_coding_stub: bool = False,
    include_agent_tools: bool = False,
    tool_choice: str | dict[str, Any] | None = None,
    vllm_env: dict[str, str] | None = None,
) -> tuple:
    from nanoscribe.tool_inference import generate_tool_candidates

    client = _openai_client(endpoint_id=endpoint_id, base_url=base_url)
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
