#!/usr/bin/env python3
"""Wave v2 orchestrator — wallet-aware, Hub-first, multi-lane campaign."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign import CampaignLedger, DEFAULT_LEDGER
from nanoscribe.campaign_datasets import validate_campaign_partitions
from nanoscribe.campaign_fanout_lib import (
    CAMPAIGN_STATUS_PATH,
    build_native_ab_manifests,
    ensure_campaign_spend_start,
    origin_master_sha,
    spend_delta_since_start,
    write_campaign_status,
)
from nanoscribe.native.data import export_distill_train_json
from nanoscribe.native.factorial import FACTORIAL_CELLS, canonical_run_id
from nanoscribe.native.trainer import trainer_manifest
from nanoscribe.runpod_hub import discover_hub_catalog
from nanoscribe.runpod_gpu_preflight import block_b200_without_sm100
from nanoscribe.runpod_wallet import budget_gate_with_wallet, effective_campaign_budget, query_live_balance
from nanoscribe.verifier_eval import export_verifier_dataset, verifier_metrics, build_verifier_examples
from nanoscribe.distill_train_suite import distill_train_cases

PYTORCH_TEMPLATE_ID = "runpod-torch-v240"
PYTORCH_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
VOLUME_ID = "04himzqxbm"
NATIVE_GPU = "NVIDIA A100 80GB PCIe"
NATIVE_RATE_HR = 1.39
B200_GPU = "NVIDIA B200"
NATIVE_POD_HOURS = 0.5


def _git_sha(full: bool = True) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    sha = proc.stdout.strip()
    return sha if full else sha[:7]


def _active_resources() -> dict[str, Any]:
    pods = subprocess.run(
        ["runpodctl", "pod", "list", "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    serverless = subprocess.run(
        ["runpodctl", "serverless", "list", "-o", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "pods": json.loads(pods.stdout) if pods.returncode == 0 else [],
        "serverless": json.loads(serverless.stdout) if serverless.returncode == 0 else [],
    }


def _launch_native_pod(
    *,
    name: str,
    run_ids: list[str],
    budget: dict[str, Any],
) -> dict[str, Any]:
    est_cost = round(NATIVE_RATE_HR * NATIVE_POD_HOURS, 4)
    allowed, reason, _ = budget_gate_with_wallet(est_cost)
    if not allowed:
        return {"launched": False, "reason": reason, "run_ids": run_ids}

    block_b200_without_sm100(NATIVE_GPU, PYTORCH_TEMPLATE_ID)

    commands = " && ".join(
        [
            "cd /workspace/nano-lm || (cd /workspace && git clone https://github.com/wwk5q8z6kk-bit/nano-lm.git && cd nano-lm)",
            f"git fetch origin && git checkout {_git_sha()} || git checkout frontier/accelerated-research-campaign-v2",
            "pip install -q -r requirements.txt 2>/dev/null || true",
            "python3 -m nanoscribe.runpod_gpu_preflight || exit 1",
            "python3 scripts/train_native_nano.py --export-train-json",
        ]
        + [f"python3 scripts/train_native_nano.py --run-id {rid}" for rid in run_ids]
    )

    proc = subprocess.run(
        [
            "runpodctl",
            "pod",
            "create",
            "--name",
            name,
            "--template-id",
            PYTORCH_TEMPLATE_ID,
            "--gpu-id",
            NATIVE_GPU,
            "--network-volume-id",
            VOLUME_ID,
            "--container-disk-in-gb",
            "40",
            "--volume-mount-path",
            "/workspace",
            "--ports",
            "22/tcp",
            "--docker-args",
            f"bash -lc {shlex.quote(commands)}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"launched": False, "reason": proc.stderr.strip() or proc.stdout.strip(), "run_ids": run_ids}

    pod = json.loads(proc.stdout)
    pod_id = pod.get("id")
    ledger = CampaignLedger.load()
    entry = ledger.commit(
        "native_a100",
        f"Native round1 {name} runs={','.join(run_ids)}",
        est_cost,
        pod_id=pod_id,
        gpu=NATIVE_GPU,
        rate_per_hr=NATIVE_RATE_HR,
        notes=f"template={PYTORCH_TEMPLATE_ID}",
    )
    ledger.save()

    return {
        "launched": True,
        "pod_id": pod_id,
        "name": name,
        "run_ids": run_ids,
        "est_cost_usd": est_cost,
        "remote_commands": commands,
        "surface": "pod_template_pytorch",
        "template_id": PYTORCH_TEMPLATE_ID,
    }


def run_wave(args: argparse.Namespace) -> dict[str, Any]:
    commit_sha = _git_sha()
    wallet = query_live_balance()
    budget = effective_campaign_budget()
    resources = _active_resources()
    hub = discover_hub_catalog()

    leakage = {"status": "pass"}
    try:
        validate_campaign_partitions()
    except ValueError as exc:
        leakage = {"status": "fail", "error": str(exc)}
        if not args.force:
            return {"leakage": leakage, "aborted": True}

    export_distill_train_json()
    trainer_manifest(Path("artifacts/campaign/native_trainer_manifest.json"))
    build_native_ab_manifests()
    verifier = export_verifier_dataset()
    verifier_hard = verifier_metrics(build_verifier_examples(distill_train_cases()[:48]))

    managed_ref_result: dict[str, Any] = {}
    if args.managed_ref:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "wave1_managed_ref.py")]
            + (["--c1-only"] if args.c1_only else []),
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if proc.returncode == 0:
            managed_ref_result = json.loads(proc.stdout)
        else:
            managed_ref_result = {"error": proc.stderr.strip() or proc.stdout.strip(), "ok": False}

    native_launches: list[dict[str, Any]] = []
    if args.native and leakage["status"] == "pass":
        round1_runs = [canonical_run_id(cell, seed) for cell in FACTORIAL_CELLS[:2] for seed in cell.seeds]
        round1b_runs = [canonical_run_id(cell, seed) for cell in FACTORIAL_CELLS[2:] for seed in cell.seeds]
        native_launches.append(_launch_native_pod(name="native-a100-1", run_ids=round1_runs, budget=budget))
        if budget["campaign_remaining"] > NATIVE_RATE_HR * NATIVE_POD_HOURS * 2:
            native_launches.append(_launch_native_pod(name="native-a100-2", run_ids=round1b_runs, budget=budget))

    spend_delta = spend_delta_since_start()
    ledger = CampaignLedger.load()

    payload = {
        "schema": "nano.campaign.wave_v2_checkpoint.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "commit_sha": commit_sha,
        "wallet": wallet,
        "budget": budget,
        "hub_catalog": hub,
        "active_resources": resources,
        "leakage": leakage,
        "managed_reference": managed_ref_result,
        "native_launches": native_launches,
        "verifier": {
            "hard_set_n": verifier_hard.get("n", 0),
            "baseline_accuracy": verifier_hard.get("baseline_accuracy"),
            "discriminative": verifier_hard.get("baseline_accuracy", 1.0) < 0.95,
            "learned_train": "SKIP" if verifier_hard.get("baseline_accuracy", 1.0) >= 0.95 else "GATED",
        },
        "qlora_gate": "NOT_UNLOCKED",
        "spend": {**ledger.summary(), **spend_delta},
    }

    status = write_campaign_status(
        active_tracks=[lane for lane, on in [("managed_ref", args.managed_ref), ("native", args.native)] if on],
        serverless={"endpoint_id": None, "note": "no Qwen serverless recreate without batch"},
        model_statuses={
            "managed_reference": managed_ref_result,
            "kimi_k3": {"status": "BLOCKED_NO_RETRIES", "frontier_lane": "managed_reference_proxy"},
        },
        structured_contract=managed_ref_result.get("c2") or managed_ref_result.get("c1", {}),
        students={
            "student_plane": "vllm_sglang_serverless_not_raw_a100",
            "qlora_gate": payload["qlora_gate"],
            "serving_hub_id": hub["resolved"].get("vllm"),
            "adaptation_hub_id": hub["resolved"].get("axolotl"),
        },
        lanes={
            "lane_managed_ref": {"status": "complete" if managed_ref_result else "skipped"},
            "lane_native": {"status": "launched" if any(l.get("launched") for l in native_launches) else "pending", "launches": native_launches},
            "lane_students": {"status": "vllm_serverless_gated", "surface": "hub_vllm"},
            "lane_verifier": verifier_hard,
        },
        leading_failures=[],
        next_reallocations=[
            "monitor native pods 5min fail-fast",
            "student vLLM benchmark then C1/C2 baseline",
            "Axolotl serverless when qlora gate opens",
        ],
    )
    status["commit_sha"] = commit_sha
    status["live_runpod_balance"] = wallet.get("client_balance_usd")
    status["campaign_remaining"] = budget["campaign_remaining"]
    status["hub_catalog"] = hub["resolved"]
    status["wave"] = "paid_wave_v2_hub"
    CAMPAIGN_STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")

    checkpoint_path = ROOT / "artifacts" / "campaign" / "checkpoint_v1.json"
    checkpoint_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Campaign wave v2 orchestrator")
    parser.add_argument("--managed-ref", action="store_true", default=True)
    parser.add_argument("--no-managed-ref", action="store_false", dest="managed_ref")
    parser.add_argument("--native", action="store_true", default=False)
    parser.add_argument("--c1-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="continue despite leakage fail")
    args = parser.parse_args()
    ensure_campaign_spend_start()
    payload = run_wave(args)
    print(json.dumps(payload, indent=2))
    return 0 if not payload.get("aborted") else 1


if __name__ == "__main__":
    raise SystemExit(main())
