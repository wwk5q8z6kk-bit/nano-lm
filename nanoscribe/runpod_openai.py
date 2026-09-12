"""RunPod OpenAI-compatible clients (Serverless + Public Endpoints)."""

from __future__ import annotations

import os
from pathlib import Path

RUNPOD_KIMI_PUBLIC_BASE = "https://api.runpod.ai/v2/moonshot-kimi/openai/v1"
KIMI_K3_MODEL = "kimi-k3"


def resolve_runpod_api_key() -> str:
    api_key = os.environ.get("RUNPOD_API_KEY")
    if api_key:
        return api_key
    config_path = Path.home() / ".runpod" / "config.toml"
    if config_path.is_file():
        import tomllib

        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
        key = data.get("apiKey") or data.get("apikey")
        if key:
            return str(key)
    raise RuntimeError("RUNPOD_API_KEY not set and ~/.runpod/config.toml unavailable")


def openai_client_for_runpod(base_url: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package required; pip install openai") from exc
    return OpenAI(api_key=resolve_runpod_api_key(), base_url=base_url.rstrip("/"))
