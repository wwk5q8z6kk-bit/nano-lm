#!/usr/bin/env python3
"""Execute accelerated research campaign phases with spend ledger updates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign import CampaignLedger, DEFAULT_LEDGER
from nanoscribe.campaign_datasets import CAMPAIGN_DATASET_REVISION, campaign_cases, suite_manifest
from nanoscribe.harness import run_matrix, write_results
from nanoscribe.tracks import api_teacher_track, serverless_strong_control_track


def _record_api_spend(ledger: CampaignLedger, amount: float, description: str) -> None:
    allowed, reason = ledger.budget_gate(amount)
    if not allowed:
        print(json.dumps({"skipped_spend": description, "reason": reason}))
        return
    entry = ledger.commit("frontier_teacher", description, amount, notes="api_token_estimate")
    ledger.actualize(entry, amount)
    ledger.save()


def run_phase_ab(
    suite: str,
    output: Path,
    *,
    api_model: str,
    record_api_cost: float,
) -> dict[str, object]:
    cases = campaign_cases(suite)
    tracks = [
        api_teacher_track(api_model),
        serverless_strong_control_track(),
    ]
    results = run_matrix(tracks, cases, capture_raw_lines=True)
    write_results(
        results,
        output,
        extra={
            "campaign_id": "accelerated_research_campaign_v1",
            "dataset_revision": CAMPAIGN_DATASET_REVISION,
            "suite": suite,
            "phases": ["A_frontier_teacher_api", "B_qwen38_serverless"],
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
    ledger = CampaignLedger.load()
    if record_api_cost > 0:
        _record_api_spend(
            ledger,
            record_api_cost,
            f"Phase A API teacher eval suite={suite}",
        )
    # Serverless: rough wall-clock cost placeholder (updated after billing reconciliation).
    serverless_est = 0.15 * len(cases)
    allowed, reason = ledger.budget_gate(serverless_est)
    if allowed:
        entry = ledger.commit(
            "qwen38_serverless",
            f"Phase B serverless eval suite={suite}",
            serverless_est,
            gpu="A100-80GB-serverless",
            rate_per_hr=1.39,
            notes="execution-time estimate per encounter",
        )
        ledger.actualize(entry, serverless_est)
        ledger.save()
    summary = ledger.summary()
    return {"output": str(output), "n_results": len(results), "spend": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description="Accelerated campaign phase runner")
    parser.add_argument(
        "--phase",
        choices=["ab", "manifest"],
        default="ab",
        help="ab = frontier API + Qwen3.8 serverless",
    )
    parser.add_argument("--suite", default="campaign_v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "p1_runs" / "campaign_phase_ab.json",
    )
    parser.add_argument("--api-model", default="gpt-4o-mini")
    parser.add_argument("--api-cost-estimate", type=float, default=0.08)
    args = parser.parse_args()

    if args.phase == "manifest":
        print(json.dumps(suite_manifest(), indent=2))
        return 0

    payload = run_phase_ab(
        args.suite,
        args.output,
        api_model=args.api_model,
        record_api_cost=args.api_cost_estimate,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
