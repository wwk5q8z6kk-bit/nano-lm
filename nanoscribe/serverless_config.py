"""RunPod Serverless endpoint worker scaling helpers."""

from __future__ import annotations

import json
import subprocess

import httpx

from nanoscribe.serverless_inference import _resolve_api_key, endpoint_native_urls
from nanoscribe.serverless_endpoint import parse_endpoint_id, resolve_serverless_endpoint_id


def _endpoint_id(endpoint_id: str | None) -> str:
    if endpoint_id:
        return parse_endpoint_id(endpoint_id)
    return resolve_serverless_endpoint_id()


def _run_runpodctl(args: list[str]) -> dict[str, object]:
    cmd = ["runpodctl", "serverless", *args, "-o", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "runpodctl failed")
    return json.loads(proc.stdout or "{}")


def get_endpoint(endpoint_id: str | None = None) -> dict[str, object]:
    endpoint_id = _endpoint_id(endpoint_id)
    return _run_runpodctl(["get", endpoint_id])


def set_workers(
    endpoint_id: str | None,
    *,
    workers_min: int | None = None,
    workers_max: int | None = None,
) -> dict[str, object]:
    endpoint_id = _endpoint_id(endpoint_id)
    args = ["update", endpoint_id]
    if workers_min is not None:
        args.extend(["--workers-min", str(workers_min)])
    if workers_max is not None:
        args.extend(["--workers-max", str(workers_max)])
    return _run_runpodctl(args)


def configure_burst(endpoint_id: str | None = None, *, max_workers: int = 10) -> dict[str, object]:
    return set_workers(endpoint_id, workers_min=1, workers_max=max_workers)


def configure_pause(endpoint_id: str | None = None) -> dict[str, object]:
    """Scale serverless workers to zero when a burst batch ends."""
    endpoint_id = _endpoint_id(endpoint_id)
    # RunPod treats 0 as "no change" for --workers-min/--workers-max; use max=1 min=0
    # and rely on idleTimeout (300s) for cost discipline when zero is rejected.
    try:
        return set_workers(endpoint_id, workers_min=0, workers_max=1)
    except RuntimeError:
        return get_endpoint(endpoint_id)


def fetch_health(endpoint_id: str | None = None) -> dict[str, object]:
    endpoint_id = _endpoint_id(endpoint_id)
    api_key = _resolve_api_key(None)
    urls = endpoint_native_urls(endpoint_id)
    with httpx.Client(timeout=15.0) as client:
        response = client.get(urls["health"], headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        return response.json()
