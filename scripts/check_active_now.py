#!/usr/bin/env python3
"""Verify docs/ACTIVE_NOW.md and ACTIVE_NOW.json agree exactly."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "docs" / "ACTIVE_NOW.md"
JSON = ROOT / "docs" / "ACTIVE_NOW.json"

# Fields that must appear in both files with identical values.
SYNCED_FIELDS = (
    "program_execution_status",
    "capability_frontier",
    "current_gate",
    "evidence_core",
    "training_backend",
    "training_status",
    "paid_compute_policy",
    "frozen_confirmatory_execution",
    "phi_on_cloud",
    "phi_or_private_data",
    "clinical_claims",
)


def parse_status_table(md: str) -> dict[str, str]:
    """Parse | `field` | value | rows under ## Status section."""
    in_section = False
    rows: dict[str, str] = {}
    keys_seen: list[str] = []
    for line in md.splitlines():
        if line.startswith("## Status"):
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if not in_section:
            continue
        m = re.match(r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|", line)
        if not m:
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        keys_seen.append(key)
        if key in rows:
            raise ValueError(f"duplicate key in ACTIVE_NOW.md table: {key}")
        rows[key] = val
    if not rows:
        raise ValueError("no status table rows parsed from ACTIVE_NOW.md")
    return rows


def main() -> int:
    if not MD.is_file() or not JSON.is_file():
        print("MISSING docs/ACTIVE_NOW files", file=sys.stderr)
        return 2
    md = MD.read_text(encoding="utf-8")
    data = json.loads(JSON.read_text(encoding="utf-8"))
    errors: list[str] = []
    try:
        table = parse_status_table(md)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    for key in SYNCED_FIELDS:
        if key not in data:
            errors.append(f"json missing key: {key}")
            continue
        if key not in table:
            errors.append(f"md table missing key: {key}")
            continue
        jv = str(data[key])
        mv = table[key]
        if jv != mv:
            errors.append(f"mismatch {key}: json={jv!r} md={mv!r}")
    extra_md = set(table) - set(SYNCED_FIELDS)
    if extra_md:
        errors.append(f"md table has unexpected keys: {sorted(extra_md)}")
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print("ACTIVE_NOW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
