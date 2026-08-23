#!/usr/bin/env python3
"""Student-A C1/C2 structured eval via ephemeral vLLM serverless."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign_datasets import campaign_cases
from nanoscribe.campaign_fanout_lib import (
    build_serverless_job_specs,
    evaluate_fanout_case,
    experiment_id_for,
    actualize_serverless_spend,
)
from nanoscribe.harness import write_results
from nanoscribe.serverless_fanout import CONTRACT_VERSION, ServerlessFanoutRunner
from nanoscribe.tracks import serverless_strong_structured_track

STUDENT_MODEL = "Qwen/Qwen2.5-32B-Instruct"


async def run_suite(
    *,
    endpoint_id: str,
    suite: str,
    model: str,
    start_concurrency: int,
    max_concurrency: int,
) -> dict[str, object]:
    cases = campaign_cases(suite)
    experiment_id = experiment_id_for("student_a", suite, "structured")
    specs = build_serverless_job_specs(
        cases,
        experiment_id=experiment_id,
        mode="structured",
        model=model,
    )
    runner = ServerlessFanoutRunner(
        endpoint_id,
        start_concurrency=start_concurrency,
        max_concurrency=max_concurrency,
    )
    records = await runner.run_jobs(specs)
    track = replace(
        serverless_strong_structured_track(endpoint_id=endpoint_id, api_model=model),
        model_id=f"student/{model}-structured",
        cost_class="student_a_serverless",
        notes="Student-A vLLM serverless baseline",
    )
    results = [evaluate_fanout_case(track, case, records, mode="structured") for case in cases]
    actualize_serverless_spend(
        records,
        lane="student_a",
        description=f"Student-A structured suite={suite} jobs={len(records)}",
    )
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "artifacts" / "p1_runs" / f"student_a_structured_{suite}_{ts}.json"
    write_results(
        results,
        out,
        extra={
            "suite": suite,
            "lane": "student_a_serverless",
            "model": model,
            "endpoint_id": endpoint_id,
            "contract_version": CONTRACT_VERSION,
        },
    )
    cov = sum(
        r.aggregate.get("coverage_rate", r.aggregate.get("coverage", 0)) for r in results
    ) / max(1, len(results))
    mal = sum(r.failures.malformed for r in results)
    asc = sum(
        r.aggregate.get("assertion_state_correct_rate", r.aggregate.get("assertion_state_correct", 0))
        for r in results
    ) / max(1, len(results))
    span = sum(
        r.aggregate.get("exact_gold_span_rate", r.aggregate.get("exact_gold_span", 0)) for r in results
    ) / max(1, len(results))
    return {
        "suite": suite,
        "n_cases": len(results),
        "avg_coverage": round(cov, 4),
        "coverage_rate": round(cov, 4),
        "assertion_state_correct_rate": round(asc, 4),
        "exact_gold_span_rate": round(span, 4),
        "malformed": mal,
        "artifact": str(out),
        "n_completed": sum(1 for r in records if r.status == "completed"),
        "n_failed": sum(1 for r in records if r.status != "completed"),
    }


async def main_async(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {"model": args.model, "endpoint_id": args.endpoint, "suites": {}}
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        payload["suites"][suite] = await run_suite(
            endpoint_id=args.endpoint,
            suite=suite,
            model=args.model,
            start_concurrency=args.start_concurrency,
            max_concurrency=args.max_concurrency,
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Student-A serverless C1/C2")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default=STUDENT_MODEL)
    parser.add_argument("--suites", default="c1_canary,c2_screening")
    parser.add_argument("--start-concurrency", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=16)
    args = parser.parse_args()
    payload = asyncio.run(main_async(args))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
