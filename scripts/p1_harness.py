#!/usr/bin/env python3
"""Run P1 three-track comparison harness.

Examples:
  python3 scripts/p1_harness.py --tracks fixture
  python3 scripts/p1_harness.py --tracks fixture,compact --weights /workspace/models/Qwen2.5-1.5B-Instruct
  python3 scripts/p1_harness.py --tracks compact,frontier --output artifacts/p1_runs/smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.harness import run_matrix, write_results
from nanoscribe.tracks import (
    COMPACT_MODEL,
    FRONTIER_MODEL,
    compact_track,
    fixture_track,
    frontier_track,
    tiny_fixture_case,
)


def _parse_tracks(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 MODEL_TRACK × P1_TEST harness")
    parser.add_argument(
        "--tracks",
        default="fixture",
        help="comma-separated: fixture, compact, frontier",
    )
    parser.add_argument(
        "--compact-weights",
        default=COMPACT_MODEL,
        help="HF id or local path for compact track",
    )
    parser.add_argument(
        "--frontier-weights",
        default=FRONTIER_MODEL,
        help="HF id or local path for frontier track",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "p1_runs" / "harness_latest.json",
    )
    parser.add_argument("--capture-raw", action="store_true")
    args = parser.parse_args()

    selected = _parse_tracks(args.tracks)
    tracks = []
    for name in selected:
        if name == "fixture":
            tracks.append(fixture_track())
        elif name == "compact":
            tracks.append(
                compact_track(args.compact_weights, device=args.device)
            )
        elif name == "frontier":
            tracks.append(
                frontier_track(args.frontier_weights, device=args.device)
            )
        else:
            raise SystemExit(f"unknown track: {name}")

    cases = [tiny_fixture_case()]
    results = run_matrix(tracks, cases, capture_raw_lines=args.capture_raw)
    write_results(
        results,
        args.output,
        extra={
            "tracks_requested": selected,
            "compact_weights": args.compact_weights,
            "frontier_weights": args.frontier_weights,
            "device": args.device,
        },
    )
    print(json.dumps({"output": str(args.output), "n_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
