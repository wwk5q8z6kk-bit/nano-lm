"""RunPod Hub discovery — stable locators + live listing IDs at launch time."""

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

STABLE_LOCATORS: dict[str, str] = {
    "vllm": "runpod-workers/worker-vllm",
    "sglang": "runpod-workers/worker-sglang",
    "axolotl": "axolotl-ai-cloud/axolotl",
    "autoresearch": "runpod/hub-autoresearch",
    "parameter_golf": "runpod/parameter-golf",
    "pytorch_template": "runpod-torch-v240",
}


def stable_locator(key: str) -> str:
    return STABLE_LOCATORS.get(key, key)


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


def _repo(item: dict[str, Any]) -> str:
    return f"{item.get('repoOwner', '')}/{item.get('repoName', '')}"


def _pick_id(
    items: list[dict[str, Any]],
    title_substr: str,
    repo_substr: str,
) -> str | None:
    for item in items:
        title = str(item.get("title", ""))
        repo = _repo(item)
        if title_substr.lower() in title.lower() or repo_substr in repo:
            hub_id = item.get("id")
            if hub_id:
                return str(hub_id)
    return None


def _resolve_from_searches(
    key: str,
    searches: dict[str, list[dict[str, Any]]],
    resolved: dict[str, str | None],
    ts: str,
) -> dict[str, Any]:
    locator = stable_locator(key)
    owner, _, name = locator.partition("/")
    listing_id = resolved.get(key)
    title = None
    listing_type = None
    for items in searches.values():
        for item in items:
            repo = _repo(item)
            if repo == locator or (
                owner and name and item.get("repoOwner") == owner and item.get("repoName") == name
            ):
                listing_id = listing_id or item.get("id")
                title = item.get("title")
                listing_type = item.get("type")
                break
    return {
        "key": key,
        "stable_locator": locator,
        "resolved_listing_id": listing_id,
        "title": title,
        "type": listing_type,
        "resolved_at": ts,
    }


def discover_hub_catalog() -> dict[str, Any]:
    """Search Hub for campaign-relevant repos and record live IDs."""
    listed = _run_hub(["list"])
    searches: dict[str, list[dict[str, Any]]] = {}
    for term in HUB_SEARCH_TARGETS:
        hits = _run_hub(["search", term])
        searches[term] = hits[:3]

    ts = datetime.now(UTC).isoformat()
    resolved = {
        "vllm": _pick_id(searches.get("vllm") or listed, "vLLM", "worker-vllm"),
        "sglang": _pick_id(searches.get("sglang") or listed, "SGLang", "worker-sglang"),
        "axolotl": _pick_id(searches.get("axolotl") or listed, "Axolotl", "axolotl"),
        "autoresearch": _pick_id(searches.get("autoresearch") or listed, "autoresearch", "autoresearch"),
        "parameter_golf": _pick_id(searches.get("parameter golf") or listed, "Parameter Golf", "parameter"),
        "pytorch_template": "runpod-torch-v240",
    }
    resolved_listings = {
        k: _resolve_from_searches(k, searches, resolved, ts)
        for k in STABLE_LOCATORS
        if k != "pytorch_template"
    }
    return {
        "schema": "nano.runpod.hub_catalog.v1",
        "timestamp": ts,
        "resolved": resolved,
        "resolved_listings": resolved_listings,
        "searches": {
            term: [
                {
                    "id": item.get("id"),
                    "title": item.get("title"),
                    "type": item.get("type"),
                    "repo": _repo(item),
                }
                for item in items
            ]
            for term, items in searches.items()
        },
        "listed_count": len(listed),
    }


def resolve_hub_listing(key: str) -> dict[str, Any]:
    catalog = discover_hub_catalog()
    return catalog["resolved_listings"].get(
        key,
        _resolve_from_searches(key, catalog["searches"], catalog["resolved"], catalog["timestamp"]),
    )
