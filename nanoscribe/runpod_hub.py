"""RunPod Hub discovery — live IDs, never hard-code stale template IDs."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from typing import Any


HUB_SEARCH_TARGETS = (
    "vllm",
    "sglang",
    "axolotl",
    "pytorch",
    "autoresearch",
    "parameter golf",
)


def _run_hub(args: list[str]) -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["runpodctl", "hub", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return data
    return [data] if data else []


def discover_hub_catalog() -> dict[str, Any]:
    """Search Hub for campaign-relevant repos and record live IDs."""
    listed = _run_hub(["list"])
    searches: dict[str, list[dict[str, Any]]] = {}
    for term in HUB_SEARCH_TARGETS:
        hits = _run_hub(["search", term])
        searches[term] = hits[:3]

    # Key IDs resolved from live hub list/search (2026-08-23).
    resolved = {
        "vllm": _pick_id(searches.get("vllm") or listed, "vLLM", "worker-vllm"),
        "sglang": _pick_id(searches.get("sglang") or listed, "SGLang", "worker-sglang"),
        "axolotl": _pick_id(searches.get("axolotl") or listed, "Axolotl", "axolotl"),
        "pytorch_template": "runpod-torch-v240",
    }
    return {
        "schema": "nano.runpod.hub_catalog.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "resolved": resolved,
        "searches": {
            term: [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "type": item.get("type"),
                    "repo": f"{item.get('repoOwner', '')}/{item.get('repoName', '')}",
                }
                for item in items
            ]
            for term, items in searches.items()
        },
        "listed_count": len(listed),
    }


def _pick_id(
    items: list[dict[str, Any]],
    title_substr: str,
    repo_substr: str,
) -> str | None:
    for item in items:
        title = str(item.get("title", ""))
        repo = f"{item.get('repoOwner', '')}/{item.get('repoName', '')}"
        if title_substr.lower() in title.lower() or repo_substr in repo:
            hub_id = item.get("id")
            if hub_id:
                return str(hub_id)
    return None
