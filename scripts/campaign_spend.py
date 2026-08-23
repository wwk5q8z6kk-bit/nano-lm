#!/usr/bin/env python3
"""Campaign spend ledger CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.campaign import (
    DEFAULT_LEDGER,
    CampaignLedger,
    estimate_pod_cost,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 campaign spend tracker")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="print spend summary")

    commit = sub.add_parser("commit", help="commit estimated spend")
    commit.add_argument("--lane", required=True)
    commit.add_argument("--description", required=True)
    commit.add_argument("--amount", type=float, required=True)
    commit.add_argument("--pod-id", default=None)
    commit.add_argument("--gpu", default=None)
    commit.add_argument("--rate-hr", type=float, default=None)

    actual = sub.add_parser("actual", help="mark last matching entry actual")
    actual.add_argument("--lane", required=True)
    actual.add_argument("--amount", type=float, required=True)

    est = sub.add_parser("estimate", help="estimate pod cost")
    est.add_argument("--rate-hr", type=float, required=True)
    est.add_argument("--hours", type=float, required=True)

    gate = sub.add_parser("gate", help="check if proposed spend allowed")
    gate.add_argument("--amount", type=float, required=True)

    args = parser.parse_args()
    ledger = CampaignLedger.load(args.ledger)

    if args.cmd == "status":
        print(json.dumps(ledger.summary(), indent=2))
        return 0

    if args.cmd == "estimate":
        print(json.dumps({"estimate_usd": estimate_pod_cost(args.rate_hr, args.hours)}))
        return 0

    if args.cmd == "gate":
        allowed, reason = ledger.budget_gate(args.amount)
        print(json.dumps({"allowed": allowed, "reason": reason, "posture": ledger.posture()}))
        return 0 if allowed else 1

    if args.cmd == "commit":
        entry = ledger.commit(
            args.lane,
            args.description,
            args.amount,
            pod_id=args.pod_id,
            gpu=args.gpu,
            rate_per_hr=args.rate_hr,
        )
        ledger.save(args.ledger)
        print(json.dumps({"committed": entry.to_dict(), "summary": ledger.summary()}))
        return 0

    if args.cmd == "actual":
        matches = [e for e in ledger.entries if e.lane == args.lane and e.status == "committed"]
        if not matches:
            print(json.dumps({"error": f"no committed entry for lane {args.lane}"}))
            return 1
        entry = matches[-1]
        ledger.actualize(entry, args.amount)
        ledger.save(args.ledger)
        print(json.dumps({"actualized": entry.to_dict(), "summary": ledger.summary()}))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
