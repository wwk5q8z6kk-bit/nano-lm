#!/usr/bin/env python3
"""Verify docs/ACTIVE_NOW.md and ACTIVE_NOW.json stay consistent."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "ACTIVE_NOW.md"
JSON = ROOT / "docs" / "ACTIVE_NOW.json"

REQUIRED_JSON_KEYS = (
    "program_execution_status",
    "capability_frontier",
    "current_gate",
    "evidence_core",
    "paid_compute",
    "training",
)


def _extract_field(md: str, field: str) -> str | None:
    # Match table rows: | `field` | value |
    m = re.search(rf"\|\s*`{re.escape(field)}`\s*\|\s*([^|]+)\|", md)
    if m:
        return m.group(1).strip()
    m = re.search(rf"\*\*{re.escape(field)}:\*\*\s*([^\n]+)", md, re.I)
    return m.group(1).strip() if m else None


def main() -> int:
    if not MD.is_file() or not JSON.is_file():
        print("MISSING docs/ACTIVE_NOW files", file=sys.stderr)
        return 2
    md = MD.read_text(encoding="utf-8")
    data = json.loads(JSON.read_text(encoding="utf-8"))
    errors: list[str] = []
    for key in REQUIRED_JSON_KEYS:
        if key not in data:
            errors.append(f"json missing key: {key}")
    # Cross-check program status if present in md table
    json_status = data.get("program_execution_status")
    if json_status and json_status not in md:
        errors.append(f"md missing program_execution_status value: {json_status}")
    json_frontier = data.get("capability_frontier")
    if json_frontier and json_frontier not in md:
        errors.append(f"md missing capability_frontier value: {json_frontier}")
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("ACTIVE_NOW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
