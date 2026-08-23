#!/usr/bin/env python3
"""Wave 1 managed reference eval — C1 canary then C2 winner-only."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign import CampaignLedger, DEFAULT_LEDGER
from nanoscribe.campaign_datasets import campaign_cases
from nanoscribe.campaign_fanout_lib import (
    build_serverless_job_specs,
    evaluate_fanout_case,
    experiment_id_for,
)
from nanoscribe.harness import HarnessResult, TrackConfig, aggregate_suite_metrics, write_results
from nanoscribe.runpod_openai import resolve_runpod_api_key
from nanoscribe.serverless_fanout import CONTRACT_VERSION, FanoutJobRecord, FanoutJobSpec
from nanoscribe.tracks import ModelTrack

MANAGED_REFS = {
    "gpt_oss_120b": {
        "base_url": "https://api.runpod.ai/v2/gpt-oss-120b/openai/v1",
        "model": "openai/gpt-oss-120b",
        "label": "MANAGED_REFERENCE_CANDIDATE",
    },
    "qwen3_32b_awq": {
        "base_url": "https://api.runpod.ai/v2/qwen3-32b-awq/openai/v1",
        "model": "Qwen/Qwen3-32B-AWQ",
        "label": "MANAGED_REFERENCE_CANDIDATE",
    },
}


def _track_for(ref_key: str, cfg: dict[str, str]) -> TrackConfig:
    return TrackConfig(
        track=ModelTrack.SERVERLESS,
        model_id=f"managed/{ref_key}-structured",
        adapter_factory=lambda: None,  # unused — fanout path only
        cost_class="managed_reference",
        notes=f"{cfg['label']} — NOT universal capability ceiling",
    )


async def _run_spec(
    client: httpx.AsyncClient,
    spec: FanoutJobSpec,
    *,
    base_url: str,
) -> FanoutJobRecord:
    record = FanoutJobRecord(
        case_id=spec.case_id,
        experiment_id=spec.experiment_id,
        contract_version=spec.contract_version,
        mode=spec.mode,
        atom_id=spec.atom_id,
        submission_time=time.time(),
    )
    started = time.perf_counter()
    try:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {resolve_runpod_api_key()}",
                "Content-Type": "application/json",
            },
            json=spec.openai_input,
            timeout=httpx.Timeout(30.0, read=180.0),
        )
        record.completion_time = time.time()
        record.execution_time_ms = int((record.completion_time - started) * 1000)
        if response.status_code >= 400:
            record.status = "failed"
            record.error = f"HTTP {response.status_code}: {response.text[:200]}"
            return record
        data = response.json()
        choices = data.get("choices") or []
        record.response = choices[0].get("message", {}).get("content", "") if choices else ""
        record.status = "completed"
    except Exception as exc:
        record.status = "failed"
        record.error = f"{type(exc).__name__}: {exc}"
        record.completion_time = time.time()
    return record


async def run_managed_ref(
    ref_key: str,
    suite: str,
    *,
    concurrency: int = 16,
) -> dict[str, Any]:
    cfg = MANAGED_REFS[ref_key]
    cases = campaign_cases(suite)
    experiment_id = experiment_id_for(f"managed_{ref_key}", suite, "structured")
    specs = build_serverless_job_specs(
        cases,
        experiment_id=experiment_id,
        mode="structured",
        model=cfg["model"],
    )
    semaphore = asyncio.Semaphore(concurrency)
    records: list[FanoutJobRecord] = []

    async with httpx.AsyncClient() as client:
        async def one(spec: FanoutJobSpec) -> FanoutJobRecord:
            async with semaphore:
                return await _run_spec(client, spec, base_url=cfg["base_url"])

        records = await asyncio.gather(*(one(spec) for spec in specs))

    track = _track_for(ref_key, cfg)
    results = [evaluate_fanout_case(track, case, records, mode="structured") for case in cases]
    completed = [r for r in records if r.status == "completed"]
    failed = [r for r in records if r.status != "completed"]
    suite_metrics = aggregate_suite_metrics(results)
    return {
        "ref_key": ref_key,
        "suite": suite,
        "base_url": cfg["base_url"],
        "model": cfg["model"],
        "label": cfg["label"],
        "n_cases": len(cases),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "metrics": suite_metrics,
        "malformed": suite_metrics.get("malformed", 0),
        "ok": len(completed) > 0 and suite_metrics.get("malformed", 0) == 0,
        "records": records,
        "results": results,
        "first_error": failed[0].error if failed else None,
    }


def _score(payload: dict[str, Any]) -> float:
    if not payload.get("ok"):
        return -1.0
    metrics = payload.get("metrics") or {}
    return (
        float(metrics.get("coverage_rate", 0))
        + 0.1 * float(metrics.get("exact_gold_span_rate", 0))
        + 0.05 * float(metrics.get("assertion_state_correct_rate", 0))
        - 0.5 * float(payload.get("malformed", 0))
    )


def _record_spend(amount: float, description: str) -> None:
    from nanoscribe.runpod_wallet import budget_gate_with_wallet

    allowed, reason, budget = budget_gate_with_wallet(amount)
    if not allowed:
        raise RuntimeError(f"budget gate blocked: {reason} (remaining=${budget['campaign_remaining']})")
    ledger = CampaignLedger.load()
    entry = ledger.commit("managed_reference", description, amount, notes="api_token_estimate")
    ledger.actualize(entry, amount)
    ledger.save()


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    c1_tasks = [run_managed_ref(key, "c1_canary", concurrency=args.concurrency) for key in MANAGED_REFS]
    c1_results = await asyncio.gather(*c1_tasks)
    c1_by_key = {item["ref_key"]: item for item in c1_results}

    scores = {key: _score(item) for key, item in c1_by_key.items()}
    winner = max(scores, key=scores.get)
    winner_label = "BEST_OPERATIONAL_MANAGED_REFERENCE"

    c2_payload: dict[str, Any] | None = None
    if not args.c1_only and scores[winner] > 0:
        c2_payload = await run_managed_ref(winner, "c2_screening", concurrency=args.concurrency)

    api_est = round(0.003 * sum(r["n_cases"] for r in c1_results) + (0.003 * 128 if c2_payload else 0), 4)
    if api_est > 0:
        _record_spend(api_est, f"Wave1 managed ref C1+C2 winner={winner}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "artifacts" / "p1_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, payload in c1_by_key.items():
        write_results(
            payload["results"],
            out_dir / f"managed_ref_{key}_c1_{timestamp}.json",
            extra={
                "ref_key": key,
                "suite": "c1_canary",
                "managed_reference_label": payload["label"],
                "not_capability_ceiling": True,
            },
        )
    if c2_payload:
        write_results(
            c2_payload["results"],
            out_dir / f"managed_ref_{winner}_c2_{timestamp}.json",
            extra={
                "ref_key": winner,
                "suite": "c2_screening",
                "managed_reference_label": winner_label,
                "not_capability_ceiling": True,
            },
        )

    return {
        "c1": {key: {k: v for k, v in item.items() if k not in {"records", "results"}} for key, item in c1_by_key.items()},
        "scores": scores,
        "winner": winner,
        "winner_label": winner_label,
        "c2": {k: v for k, v in (c2_payload or {}).items() if k not in {"records", "results"}},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Wave 1 managed reference eval")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--c1-only", action="store_true")
    args = parser.parse_args()
    payload = asyncio.run(main_async(args))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
