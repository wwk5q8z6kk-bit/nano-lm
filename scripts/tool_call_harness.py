#!/usr/bin/env python3
"""Reusable harness — load fixtures/tool_calls and exercise ToolCallParser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.tool_calling import ToolCallParser, ToolCallParseOutcome

FIXTURES = ROOT / "fixtures" / "tool_calls"


def run_fixture(name: str) -> dict[str, object]:
    parser = ToolCallParser()
    path = FIXTURES / name
    if name.endswith(".json"):
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            result = parser.parse_tool_calls(payload)
        else:
            result = parser.parse_arguments(payload)
            return {
                "fixture": name,
                "outcome": ToolCallParseOutcome.SUCCESS.value,
                "atom_count": len(result.atoms),
            }
    else:
        result = parser.parse_text_json(path.read_text())
    return {
        "fixture": name,
        "outcome": result.outcome.value,
        "atom_count": len(result.candidate.atoms) if result.candidate else 0,
        "error_code": result.error.code if result.error else None,
    }


def main() -> int:
    arg = argparse.ArgumentParser(description="Tool call parser fixture harness")
    arg.add_argument("--fixture", help="single fixture filename under fixtures/tool_calls/")
    args = arg.parse_args()
    names = [args.fixture] if args.fixture else sorted(p.name for p in FIXTURES.iterdir() if p.is_file())
    results = [run_fixture(name) for name in names]
    print(json.dumps(results, indent=2))
    failed = [item for item in results if item["outcome"] not in {"success", "json_fallback"}]
    if args.fixture and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
