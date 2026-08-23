#!/usr/bin/env python3
"""Campaign control plane — inventory and status refresh without paid compute."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign_fanout_lib import CAMPAIGN_STATUS_PATH, origin_master_sha, write_campaign_status
from nanoscribe.runpod_hub import discover_hub_catalog
from nanoscribe.runpod_wallet import effective_campaign_budget, query_live_balance


def _git_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return proc.stdout.strip()


def _run_json(cmd: list[str]) -> list[dict] | dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    data = json.loads(proc.stdout or "[]")
    return data if isinstance(data, list) else [data]


def inventory() -> dict:
    wallet = query_live_balance()
    budget = effective_campaign_budget()
    hub = discover_hub_catalog()
    pods = _run_json(["runpodctl", "pod", "list", "-o", "json"])
    serverless = _run_json(["runpodctl", "serverless", "list", "-o", "json"])
    gpus = _run_json(["runpodctl", "gpu", "list", "-o", "json"])
    return {
        "schema": "nano.campaign.control_plane.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "commit_sha": _git_sha(),
        "origin_master": origin_master_sha(),
        "wallet": wallet,
        "budget": budget,
        "hub_catalog": hub,
        "pods": pods,
        "serverless": serverless,
        "gpu_types_available": len(gpus) if isinstance(gpus, list) else 0,
        "active_pod_count": len(pods) if isinstance(pods, list) else 0,
        "active_serverless_count": len(serverless) if isinstance(serverless, list) else 0,
        "current_spend_per_hr": wallet.get("current_spend_per_hr", 0.0),
    }


def refresh_status(inv: dict) -> dict:
    status = write_campaign_status(
        active_tracks=[],
        serverless={
            "endpoints": inv.get("serverless", []),
            "endpoint_id": (inv.get("serverless") or [{}])[0].get("id") if inv.get("serverless") else None,
        },
        model_statuses={},
        structured_contract={},
        students={"student_plane": "vllm_sglang_serverless_not_raw_a100"},
        lanes={},
        leading_failures=[],
        next_reallocations=["inventory refresh only"],
    )
    status["commit_sha"] = inv["commit_sha"]
    status["live_runpod_balance"] = inv["wallet"].get("client_balance_usd")
    status["campaign_remaining"] = inv["budget"]["campaign_remaining"]
    status["current_spend_per_hr"] = inv["wallet"].get("current_spend_per_hr", 0.0)
    status["hub_catalog"] = inv["hub_catalog"]["resolved"]
    status["posture"] = "IDLE" if inv["active_pod_count"] == 0 and inv["current_spend_per_hr"] < 0.01 else "ACTIVE"
    CAMPAIGN_STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Campaign control plane")
    parser.add_argument("action", choices=["inventory", "refresh"], default="inventory", nargs="?")
    args = parser.parse_args()
    inv = inventory()
    if args.action == "refresh":
        refresh_status(inv)
    print(json.dumps(inv, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
