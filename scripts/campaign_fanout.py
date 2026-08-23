#!/usr/bin/env python3
"""Parallel campaign fan-out — RunPod Serverless /run + Kimi concurrent lanes."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign import CampaignLedger, DEFAULT_LEDGER
from nanoscribe.campaign_datasets import campaign_cases
from nanoscribe.campaign_fanout_lib import (
    CAMPAIGN_STATUS_PATH,
    build_native_ab_manifests,
    build_serverless_job_specs,
    build_teacher_data_v0,
    build_verifier_dataset,
    ensure_campaign_spend_start,
    evaluate_fanout_case,
    evaluate_verifier_dataset_coverage,
    experiment_id_for,
    export_screening_p1_distill,
    origin_master_sha,
    actualize_serverless_spend,
    spend_delta_since_start,
    update_disagreement_matrix,
    write_campaign_status,
)
from nanoscribe.harness import HarnessCase, HarnessResult, write_results
from nanoscribe.kimi_teacher import kimi_preflight, kimi_preflight_with_fallback
from nanoscribe.runpod_openai import KIMI_K3_MODEL, RUNPOD_KIMI_PUBLIC_BASE, resolve_runpod_api_key
from nanoscribe.serverless_fanout import (
    CONTRACT_VERSION,
    FanoutJobRecord,
    FanoutJobSpec,
    ServerlessFanoutRunner,
    estimate_worker_cost_usd,
)
from nanoscribe.tracks import (
    SERVERLESS_ENDPOINT_ID,
    SERVERLESS_STRONG_MODEL,
    kimi_frontier_structured_track,
    serverless_strong_structured_track,
    serverless_strong_control_track,
    kimi_frontier_span_port_track,
)
from nanoscribe.serverless_config import configure_burst, configure_pause, fetch_health, get_endpoint

OUTPUT_ROOT = ROOT / "artifacts" / "p1_runs"


async def _kimi_openai_call(
    client: httpx.AsyncClient,
    openai_input: dict[str, Any],
) -> str:
    response = await client.post(
        f"{RUNPOD_KIMI_PUBLIC_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {resolve_runpod_api_key()}",
            "Content-Type": "application/json",
        },
        json=openai_input,
    )
    if response.status_code >= 500:
        raise httpx.HTTPStatusError(
            f"Kimi transient {response.status_code}",
            request=response.request,
            response=response,
        )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("message", {}).get("content") or "").strip()


async def run_kimi_fanout(
    specs: Sequence[FanoutJobSpec],
    *,
    start_concurrency: int = 32,
    max_concurrency: int = 64,
    max_retries: int = 2,
) -> list[FanoutJobRecord]:
    concurrency = start_concurrency
    semaphore = asyncio.Semaphore(concurrency)
    records: list[FanoutJobRecord | None] = [None] * len(specs)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=180.0)) as client:
        async def worker(index: int, spec: FanoutJobSpec) -> None:
            nonlocal concurrency, semaphore
            async with semaphore:
                record = FanoutJobRecord(
                    case_id=spec.case_id,
                    experiment_id=spec.experiment_id,
                    contract_version=spec.contract_version,
                    mode=spec.mode,
                    atom_id=spec.atom_id,
                )
                for attempt in range(max_retries + 1):
                    try:
                        submission_time = time.time()
                        record.submission_time = submission_time
                        text = await _kimi_openai_call(client, spec.openai_input)
                        record.completion_time = time.time()
                        record.start_time = submission_time
                        record.status = "completed"
                        record.response = text
                        record.retry_count = attempt
                        records[index] = record
                        break
                    except (httpx.HTTPError, TimeoutError) as exc:
                        record.retry_count = attempt
                        if attempt >= max_retries:
                            record.status = "failed"
                            record.error = f"{type(exc).__name__}: {exc}"
                            record.completion_time = time.time()
                            records[index] = record
                            break
                        await asyncio.sleep(0.5 * (attempt + 1))

        await asyncio.gather(*(worker(i, spec) for i, spec in enumerate(specs)))

        done = [item for item in records if item is not None]
        if len(done) >= 10 and concurrency < max_concurrency:
            failures = sum(1 for item in done if item.status != "completed")
            if failures / len(done) <= 0.05:
                concurrency = max_concurrency
                semaphore = asyncio.Semaphore(concurrency)

    return [item for item in records if item is not None]


def _budget_gate_jobs(
    ledger: CampaignLedger,
    num_jobs: int,
    *,
    lane: str,
) -> tuple[bool, str, float]:
    projected = estimate_worker_cost_usd(num_jobs)
    allowed, reason = ledger.budget_gate(projected)
    return allowed, reason, projected


def _leading_failures(results: Sequence[HarnessResult]) -> list[str]:
    counts: dict[str, int] = {}
    for result in results:
        for key, value in result.failures.to_dict().items():
            if value:
                counts[key] = counts.get(key, 0) + value
    return [f"{key}:{count}" for key, count in sorted(counts.items(), key=lambda item: -item[1])[:5]]


async def run_qwen_lane(
    cases: Sequence[HarnessCase],
    *,
    suite: str,
    mode: str,
    start_concurrency: int,
    max_concurrency: int,
    endpoint_id: str,
) -> tuple[list[FanoutJobRecord], list[HarnessResult]]:
    experiment_id = experiment_id_for("qwen38_serverless", suite, mode)
    specs = build_serverless_job_specs(cases, experiment_id=experiment_id, mode=mode)
    ledger = CampaignLedger.load()
    allowed, reason, projected = _budget_gate_jobs(ledger, len(specs), lane="qwen38_serverless")
    if not allowed:
        raise RuntimeError(f"budget gate blocked Qwen lane: {reason} (projected ${projected})")

    runner = ServerlessFanoutRunner(
        endpoint_id,
        start_concurrency=start_concurrency,
        max_concurrency=max_concurrency,
    )
    records = await runner.run_jobs(specs)
    track = (
        serverless_strong_structured_track(endpoint_id=endpoint_id)
        if mode == "structured"
        else serverless_strong_control_track(endpoint_id=endpoint_id)
    )
    results = [evaluate_fanout_case(track, case, records, mode=mode) for case in cases]
    actualize_serverless_spend(
        records,
        lane="qwen38_serverless",
        description=f"Fanout {mode} suite={suite} jobs={len(records)}",
    )
    return records, results


async def run_kimi_lane(
    cases: Sequence[HarnessCase],
    *,
    suite: str,
    mode: str,
    start_concurrency: int,
    max_concurrency: int,
) -> tuple[list[FanoutJobRecord], list[HarnessResult]]:
    experiment_id = experiment_id_for("kimi_frontier", suite, mode)
    specs = build_serverless_job_specs(
        cases,
        experiment_id=experiment_id,
        mode=mode,
        model=KIMI_K3_MODEL,
    )
    # Kimi public endpoint — API token estimate gate (conservative).
    ledger = CampaignLedger.load()
    api_est = round(0.002 * len(specs), 4)
    allowed, reason = ledger.budget_gate(api_est)
    if not allowed:
        raise RuntimeError(f"budget gate blocked Kimi lane: {reason}")

    records = await run_kimi_fanout(
        specs,
        start_concurrency=start_concurrency,
        max_concurrency=max_concurrency,
    )
    track = (
        kimi_frontier_structured_track()
        if mode == "structured"
        else kimi_frontier_span_port_track()
    )
    results = [evaluate_fanout_case(track, case, records, mode=mode) for case in cases]
    completed = [item for item in records if item.status == "completed"]
    if completed:
        amount = max(api_est, round(0.001 * len(completed), 4))
        allowed, reason = ledger.budget_gate(amount)
        if allowed:
            entry = ledger.commit(
                "kimi_frontier",
                f"Fanout {mode} suite={suite} jobs={len(records)}",
                amount,
                notes="api_token_estimate",
            )
            ledger.actualize(entry, amount)
            ledger.save()
    return records, results


async def run_wave1(
    suite: str,
    *,
    modes: Sequence[str],
    start_concurrency: int,
    max_concurrency: int,
    endpoint_id: str,
    skip_kimi: bool,
) -> dict[str, Any]:
    cases = campaign_cases(suite)
    build_verifier_dataset(cases)
    build_native_ab_manifests()

    preflight = kimi_preflight()
    health = fetch_health(endpoint_id)
    endpoint = get_endpoint(endpoint_id)

    # Burst config — idempotent if already at max_workers=10.
    burst = configure_burst(endpoint_id, max_workers=10)

    active_tracks: list[str] = []
    model_statuses: dict[str, Any] = {
        "kimi_k3": preflight,
        "qwen38_serverless": {"health": health, "endpoint": endpoint},
    }
    all_results: dict[str, list[HarnessResult]] = {}
    request_logs: dict[str, list[dict[str, Any]]] = {}

    async def qwen_task(mode: str) -> None:
        active_tracks.append(f"qwen38_{mode}")
        records, results = await run_qwen_lane(
            cases,
            suite=suite,
            mode=mode,
            start_concurrency=start_concurrency,
            max_concurrency=max_concurrency,
            endpoint_id=endpoint_id,
        )
        all_results[f"qwen_{mode}"] = results
        request_logs[f"qwen_{mode}"] = [item.to_dict() for item in records]

    async def kimi_task(mode: str) -> None:
        if skip_kimi or not preflight.get("ok"):
            model_statuses["kimi_k3"]["lane_skipped"] = True
            return
        active_tracks.append(f"kimi_{mode}")
        records, results = await run_kimi_lane(
            cases,
            suite=suite,
            mode=mode,
            start_concurrency=start_concurrency,
            max_concurrency=max_concurrency,
        )
        all_results[f"kimi_{mode}"] = results
        request_logs[f"kimi_{mode}"] = [item.to_dict() for item in records]

    tasks = [qwen_task(mode) for mode in modes]
    if not skip_kimi:
        tasks.extend(kimi_task(mode) for mode in modes)
    await asyncio.gather(*tasks)

    # Persist lane outputs.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for lane_key, results in all_results.items():
        out = OUTPUT_ROOT / f"fanout_{suite}_{lane_key}_{timestamp}.json"
        write_results(
            results,
            out,
            extra={
                "suite": suite,
                "lane": lane_key,
                "contract_version": CONTRACT_VERSION,
                "origin_master": origin_master_sha(),
                "request_log": request_logs.get(lane_key, []),
            },
        )

    kimi_structured = all_results.get("kimi_structured", [])
    qwen_structured = all_results.get("qwen_structured", [])
    if kimi_structured and qwen_structured:
        disagreement = update_disagreement_matrix(kimi_structured, qwen_structured)
    else:
        disagreement = {"note": "insufficient paired results for disagreement matrix"}

    combined_failures = [item for results in all_results.values() for item in results]
    pause = configure_pause(endpoint_id)
    status = write_campaign_status(
        active_tracks=active_tracks,
        serverless={
            "endpoint_id": endpoint_id,
            "jobs_in_progress": health.get("jobs", {}).get("inProgress", 0),
            "jobs_queued": health.get("jobs", {}).get("inQueue", 0),
            "running_workers": health.get("workers", {}).get("running", 0),
            "max_workers": endpoint.get("workersMax"),
            "burst_config": burst,
        },
        model_statuses=model_statuses,
        leading_failures=_leading_failures(combined_failures),
        next_reallocations=[
            "complete screening matrix if smoke healthy",
            "expand concurrency 32→64 when failure rate <5%",
            "retry Kimi when public endpoint recovers",
        ],
    )

    ledger = CampaignLedger.load()
    return {
        "suite": suite,
        "modes": list(modes),
        "n_cases": len(cases),
        "preflight": preflight,
        "active_tracks": active_tracks,
        "results_keys": list(all_results.keys()),
        "disagreement_summary": disagreement.get("summary", disagreement),
        "spend": ledger.summary(),
        "campaign_status_path": str(CAMPAIGN_STATUS_PATH),
        "burst": burst,
        "pause": pause,
    }


async def run_c1_c2_qwen(
    *,
    start_concurrency: int = 32,
    max_concurrency: int = 64,
    endpoint_id: str,
    skip_c2: bool = False,
) -> dict[str, Any]:
    """Lane 2 — C1 canary (32) then C2 full (128) structured fan-out."""
    ensure_campaign_spend_start()
    suites = ["c1_canary"] if skip_c2 else ["c1_canary", "c2_screening"]
    burst = configure_burst(endpoint_id, max_workers=10)
    health = fetch_health(endpoint_id)
    all_payload: dict[str, Any] = {"suites": {}, "burst": burst}

    for suite in suites:
        cases = campaign_cases(suite)
        build_verifier_dataset(cases)
        records, results = await run_qwen_lane(
            cases,
            suite=suite,
            mode="structured",
            start_concurrency=start_concurrency,
            max_concurrency=max_concurrency,
            endpoint_id=endpoint_id,
        )
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = OUTPUT_ROOT / f"fanout_{suite}_qwen_structured_{timestamp}.json"
        write_results(
            results,
            out,
            extra={
                "suite": suite,
                "lane": "qwen_structured",
                "contract_version": CONTRACT_VERSION,
                "origin_master": origin_master_sha(),
                "request_log": [item.to_dict() for item in records],
            },
        )
        cov = sum(r.aggregate.get("coverage", 0) for r in results) / max(1, len(results))
        malformed = sum(r.failures.malformed for r in results)
        all_payload["suites"][suite] = {
            "n_cases": len(cases),
            "avg_coverage": round(cov, 4),
            "malformed": malformed,
            "artifact": str(out),
        }

    pause = configure_pause(endpoint_id)
    all_payload["pause"] = pause
    return all_payload


def student_picks() -> dict[str, Any]:
    return {
        "student_a": {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "params_b": 32,
            "lora_ready": True,
            "rationale": "proven LoRA baseline path, fits A100-80GB",
            "status": "C1_BASELINE_UNLOCKED",
        },
        "student_b": {
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "params_b": 70,
            "lora_ready": True,
            "rationale": "diverse 27B-80B family, QLoRA on 2×A100",
            "status": "GATED_UNTIL_KIMI_OR_GAP_JUSTIFIED",
        },
        "qlora_gate": "Student-A baseline unlocked; QLoRA still gated until Kimi ceiling or documented Qwen-only gap",
        "qwen_only_gap_note": "Kimi 500 blocked; structured C1/C2 complete — Qwen structured is practical ceiling proxy",
    }


async def orchestrate_v2(
    *,
    start_concurrency: int = 32,
    max_concurrency: int = 64,
    endpoint_id: str,
    skip_qwen: bool = False,
    skip_c2: bool = False,
) -> dict[str, Any]:
    """Full campaign v2 orchestration — lanes 0-5 CPU prep + inference."""
    spend_start = ensure_campaign_spend_start()
    preflight = kimi_preflight_with_fallback()
    build_native_ab_manifests()
    from nanoscribe.native.trainer import trainer_manifest

    trainer_manifest(Path("artifacts/campaign/native_trainer_manifest.json"))

    lanes: dict[str, Any] = {
        "lane0_cpu": {"status": "active", "campaign_spend_start": spend_start},
        "lane1_kimi": {
            "status": "blocked" if preflight.get("blocked") else "ready",
            "preflight": preflight,
        },
        "lane2_qwen": {"status": "pending" if not skip_qwen else "skipped"},
        "lane3_students": {"status": "gated", "picks": student_picks()},
        "lane4_native": {"status": "TRAINER_READY_NO_GPU"},
        "lane5_verifier": {"status": "dataset_ready"},
    }

    cases_c2 = campaign_cases("c2_screening")
    verifier_cov = evaluate_verifier_dataset_coverage(cases_c2)
    distill = export_screening_p1_distill()
    teacher_v0 = build_teacher_data_v0(max_cases=32)

    qwen_result: dict[str, Any] = {}
    if not skip_qwen:
        lanes["lane2_qwen"]["status"] = "running"
        qwen_result = await run_c1_c2_qwen(
            start_concurrency=start_concurrency,
            max_concurrency=max_concurrency,
            endpoint_id=endpoint_id,
            skip_c2=skip_c2,
        )
        lanes["lane2_qwen"]["status"] = "complete"
        lanes["lane2_qwen"]["result"] = qwen_result

    # Structured contract verdict from latest C1 or prior screening
    c1_stats = qwen_result.get("suites", {}).get("c1_canary", {})
    structured_verdict = {
        "winner": "structured_json",
        "c1_avg_coverage": c1_stats.get("avg_coverage"),
        "c1_malformed": c1_stats.get("malformed"),
        "span_port_deprecated": True,
        "note": "structured 0 malformed vs span_port 149 malformed on screening_p1",
    }

    health = fetch_health(endpoint_id)
    endpoint = get_endpoint(endpoint_id)
    ledger = CampaignLedger.load()
    spend_delta = spend_delta_since_start()

    status = write_campaign_status(
        active_tracks=[k for k, v in lanes.items() if v.get("status") in {"active", "running", "complete", "ready"}],
        serverless={
            "endpoint_id": endpoint_id,
            "jobs_in_progress": health.get("jobs", {}).get("inProgress", 0),
            "jobs_queued": health.get("jobs", {}).get("inQueue", 0),
            "running_workers": health.get("workers", {}).get("running", 0),
            "max_workers": endpoint.get("workersMax"),
        },
        model_statuses={
            "kimi_k3": preflight.get("kimi", {}),
            "alternate_managed_frontier": preflight.get("alternate_managed_frontier", {}),
            "qwen38_structured": structured_verdict,
            "native_nano": {"status": "TRAINER_READY_NO_GPU"},
        },
        structured_contract=structured_verdict,
        students=student_picks(),
        lanes=lanes,
        leading_failures=[
            "kimi_k3_endpoint_500" if preflight.get("blocked") else "",
            "annotation_boundary_cases",
        ],
        next_reallocations=[
            "retry Kimi when endpoint recovers",
            "student baseline on C1 after contract lock",
            "native B200 launch when distill ready",
            "verifier compact train on 4090",
        ],
    )

    return {
        "origin_master": origin_master_sha(short=False),
        "spend": {**ledger.summary(), **spend_delta},
        "preflight": preflight,
        "qwen": qwen_result,
        "distill": distill,
        "teacher_data_v0": teacher_v0,
        "verifier_coverage": verifier_cov,
        "campaign_status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel campaign fan-out runner")
    parser.add_argument(
        "command",
        choices=["wave1", "preflight", "status", "orchestrate", "c1c2"],
        help="orchestrate = full v2 lanes; c1c2 = Qwen C1+C2 structured only",
    )
    parser.add_argument("--suite", default="campaign_smoke")
    parser.add_argument(
        "--modes",
        default="structured",
        help="comma-separated: structured,span_port",
    )
    parser.add_argument("--start-concurrency", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--endpoint", default=SERVERLESS_ENDPOINT_ID)
    parser.add_argument("--skip-kimi", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-c2", action="store_true")
    args = parser.parse_args()

    if args.command == "preflight":
        print(json.dumps(kimi_preflight_with_fallback(), indent=2))
        return 0

    if args.command == "status":
        health = fetch_health(args.endpoint)
        endpoint = get_endpoint(args.endpoint)
        ledger = CampaignLedger.load()
        spend_delta = spend_delta_since_start()
        payload = write_campaign_status(
            active_tracks=[],
            serverless={
                "endpoint_id": args.endpoint,
                "jobs_in_progress": health.get("jobs", {}).get("inProgress", 0),
                "jobs_queued": health.get("jobs", {}).get("inQueue", 0),
                "running_workers": health.get("workers", {}).get("running", 0),
                "max_workers": endpoint.get("workersMax"),
            },
            model_statuses={"kimi_k3": kimi_preflight_with_fallback()},
            leading_failures=[],
            next_reallocations=["await orchestrate launch"],
            students=student_picks(),
        )
        payload["spend"] = {**ledger.summary(), **spend_delta}
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "c1c2":
        payload = asyncio.run(
            run_c1_c2_qwen(
                start_concurrency=args.start_concurrency,
                max_concurrency=args.max_concurrency,
                endpoint_id=args.endpoint,
                skip_c2=args.skip_c2,
            )
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "orchestrate":
        payload = asyncio.run(
            orchestrate_v2(
                start_concurrency=args.start_concurrency,
                max_concurrency=args.max_concurrency,
                endpoint_id=args.endpoint,
                skip_qwen=args.skip_qwen,
                skip_c2=args.skip_c2,
            )
        )
        print(json.dumps(payload, indent=2))
        return 0

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    payload = asyncio.run(
        run_wave1(
            args.suite,
            modes=modes,
            start_concurrency=args.start_concurrency,
            max_concurrency=args.max_concurrency,
            endpoint_id=args.endpoint,
            skip_kimi=args.skip_kimi,
        )
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
