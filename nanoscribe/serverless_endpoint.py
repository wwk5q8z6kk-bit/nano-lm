"""RunPod Serverless endpoint identity — env override after teardown."""

from __future__ import annotations

import os
from urllib.parse import urlparse

DELETED_QWEN_SERVERLESS_ENDPOINT_ID = "tbnur4mac60i70"


def parse_endpoint_id(value: str) -> str:
    """Accept bare endpoint id or full RunPod URL."""
    if "://" not in value:
        return value
    path = urlparse(value).path.strip("/")
    parts = path.split("/")
    if len(parts) >= 2 and parts[0] == "v2":
        return parts[1]
    raise ValueError(f"cannot parse RunPod endpoint id from {value!r}")


def resolve_serverless_endpoint_id(*, required: bool = True) -> str | None:
    """Return active endpoint id from RUNPOD_SERVERLESS_ENDPOINT_ID."""
    raw = os.environ.get("RUNPOD_SERVERLESS_ENDPOINT_ID")
    if raw:
        return parse_endpoint_id(raw.strip())
    if required:
        raise RuntimeError(
            "RUNPOD_SERVERLESS_ENDPOINT_ID is not set. "
            f"Prior endpoint {DELETED_QWEN_SERVERLESS_ENDPOINT_ID} was deleted "
            "(RunPod refused workersMin=0 on update). "
            "Recreate Qwen3.8 Serverless via hub with --workers-min 0 before the next batch."
        )
    return None
