#!/usr/bin/env python3
"""Verifier lane — export disjoint dataset, baseline metrics, compact eval."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign import CampaignLedger
from nanoscribe.verifier_eval import export_verifier_dataset, verifier_metrics, build_verifier_examples
from nanoscribe.distill_train_suite import distill_train_cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifier compact train/eval")
    parser.add_argument("--record-spend", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    export = export_verifier_dataset()
    cases = distill_train_cases()[:48]
    examples = build_verifier_examples(cases)
    metrics = verifier_metrics(examples)
    elapsed = time.perf_counter() - started

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset": export,
        "metrics": metrics,
        "deterministic_baseline_accuracy": metrics.get("baseline_accuracy"),
        "disjoint_from": "p1_screening_eval_v1",
        "wall_s": round(elapsed, 2),
        "verdict": "BASELINE_ONLY" if metrics.get("baseline_accuracy", 0) >= 0.9 else "NEEDS_LEARNED_VERIFIER",
    }

    out = ROOT / "artifacts" / "p1_runs" / f"verifier_lane_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    payload["artifact"] = str(out)

    if args.record_spend:
        hours = max(elapsed / 3600.0, 0.05)
        amount = round(0.74 * hours, 4)
        ledger = CampaignLedger.load()
        allowed, reason = ledger.budget_gate(amount)
        if allowed:
            entry = ledger.commit(
                "verifier",
                "Verifier compact eval vs deterministic baseline",
                amount,
                gpu="NVIDIA GeForce RTX 4090",
                rate_per_hr=0.74,
            )
            ledger.actualize(entry, amount)
            ledger.save()

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
