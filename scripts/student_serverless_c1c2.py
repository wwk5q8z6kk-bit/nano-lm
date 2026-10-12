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
    actualize_serverless_spend,
    build_serverless_job_specs,
    evaluate_fanout_case,
    experiment_id_for,
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
    mode: str,
) -> dict[str, object]:
    cases = campaign_cases(suite)
    experiment_id = experiment_id_for("student_a", suite, mode)
    specs = build_serverless_job_specs(
        cases,
        experiment_id=experiment_id,
        mode=mode,
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
        model_id=f"student/{model}-{mode}",
        cost_class="student_a_serverless",
        notes=f"Student-A vLLM serverless baseline mode={mode}",
    )
    results = [evaluate_fanout_case(track, case, records, mode=mode) for case in cases]
    actualize_serverless_spend(
        records,
        lane="student_a",
        description=f"Student-A {mode} suite={suite} jobs={len(records)}",
    )
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "artifacts" / "p1_runs" / f"student_a_{mode}_{suite}_{ts}.json"
    write_results(
        results,
        out,
        extra={
            "suite": suite,
            "lane": "student_a_serverless",
            "model": model,
            "mode": mode,
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
        "mode": mode,
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


def _compare_payload(structured: dict[str, object], tool: dict[str, object]) -> dict[str, object]:
    return {
        "coverage_delta": round(
            float(tool.get("coverage_rate", 0)) - float(structured.get("coverage_rate", 0)),
            4,
        ),
        "assertion_delta": round(
            float(tool.get("assertion_state_correct_rate", 0))
            - float(structured.get("assertion_state_correct_rate", 0)),
            4,
        ),
        "span_delta": round(
            float(tool.get("exact_gold_span_rate", 0))
            - float(structured.get("exact_gold_span_rate", 0)),
            4,
        ),
        "malformed_delta": int(tool.get("malformed", 0)) - int(structured.get("malformed", 0)),
    }


async def main_async(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": args.model,
        "endpoint_id": args.endpoint,
        "mode": args.mode,
        "suites": {},
    }
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    if args.mode == "compare":
        compare_suite = args.compare_suite or "c1_canary"
        structured = await run_suite(
            endpoint_id=args.endpoint,
            suite=compare_suite,
            model=args.model,
            start_concurrency=args.start_concurrency,
            max_concurrency=args.max_concurrency,
            mode="structured",
        )
        tool = await run_suite(
            endpoint_id=args.endpoint,
            suite=compare_suite,
            model=args.model,
            start_concurrency=args.start_concurrency,
            max_concurrency=args.max_concurrency,
            mode="tool",
        )
        payload["compare_suite"] = compare_suite
        payload["structured_vs_tool"] = {
            "structured": structured,
            "tool": tool,
            "delta": _compare_payload(structured, tool),
        }
        if len(suites) > 1 or (len(suites) == 1 and suites[0] != compare_suite):
            for suite in suites:
                if suite == compare_suite:
                    continue
                payload["suites"][suite] = await run_suite(
                    endpoint_id=args.endpoint,
                    suite=suite,
                    model=args.model,
                    start_concurrency=args.start_concurrency,
                    max_concurrency=args.max_concurrency,
                    mode="tool",
                )
        return payload

    mode = args.mode
    for suite in suites:
        payload["suites"][suite] = await run_suite(
            endpoint_id=args.endpoint,
            suite=suite,
            model=args.model,
            start_concurrency=args.start_concurrency,
            max_concurrency=args.max_concurrency,
            mode=mode,
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Student-A serverless C1/C2")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default=STUDENT_MODEL)
    parser.add_argument("--suites", default="c1_canary,c2_screening")
    parser.add_argument(
        "--mode",
        choices=["structured", "tool", "compare"],
        default="tool",
        help="Inference mode (default: tool). compare runs structured vs tool on C1 smoke.",
    )
    parser.add_argument(
        "--compare-suite",
        default="c1_canary",
        help="Suite for structured vs tool comparison when --mode compare",
    )
    parser.add_argument("--start-concurrency", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=16)
    args = parser.parse_args()
    payload = asyncio.run(main_async(args))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
