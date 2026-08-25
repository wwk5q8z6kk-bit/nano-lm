#!/usr/bin/env python3
"""Verify RunPod Serverless endpoint is configured and reachable."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.serverless_config import fetch_health, get_endpoint
from nanoscribe.serverless_endpoint import (
    DELETED_QWEN_SERVERLESS_ENDPOINT_ID,
    resolve_serverless_endpoint_id,
)


def main() -> int:
    try:
        endpoint_id = resolve_serverless_endpoint_id()
    except RuntimeError as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "deleted_endpoint": DELETED_QWEN_SERVERLESS_ENDPOINT_ID,
                },
                indent=2,
            )
        )
        return 1

    config = get_endpoint(endpoint_id)
    health = fetch_health(endpoint_id)
    print(
        json.dumps(
            {
                "ok": True,
                "endpoint_id": endpoint_id,
                "workers_min": config.get("workersMin"),
                "workers_max": config.get("workersMax"),
                "jobs": health.get("jobs"),
                "workers": health.get("workers"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
