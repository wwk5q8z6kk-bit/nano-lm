#!/usr/bin/env python3
"""Stub generator for p1_screening_eval_v2 — not yet authoritative."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "p1_screening_eval_v2.json"


def main() -> int:
    payload = {
        "schema": "nano.eval.screening.v2.stub",
        "revision": "p1_screening_eval_v2_stub",
        "status": "PLACEHOLDER",
        "timestamp": datetime.now(UTC).isoformat(),
        "note": "Wave 0 stub — expand with powered axis coverage before freezing.",
        "supersedes": "p1_screening_eval_v1",
        "frozen": False,
        "n_cases": 0,
        "cases": [],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"written": str(OUT.relative_to(ROOT)), "n_cases": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
