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
    API_TEACHER_MODEL,
    COMPACT_MODEL,
    FRONTIER_MODEL,
    SERVERLESS_ENDPOINT_ID,
    SERVERLESS_STRONG_MODEL,
    STUDENT_MODEL,
    api_teacher_track,
    compact_track,
    fixture_track,
    frontier_track,
    serverless_strong_control_track,
    student_track,
    tiny_fixture_case,
)


def _parse_tracks(raw: str) -> list[str]:
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="P1 MODEL_TRACK × P1_TEST harness")
    parser.add_argument(
        "--tracks",
        default="fixture",
        help="comma-separated: fixture, compact, serverless, frontier, api, student",
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
    parser.add_argument(
        "--api-model",
        default=API_TEACHER_MODEL,
        help="OpenAI model id for api track",
    )
    parser.add_argument(
        "--student-weights",
        default=STUDENT_MODEL,
        help="HF id or local path for student track",
    )
    parser.add_argument(
        "--serverless-endpoint",
        default=SERVERLESS_ENDPOINT_ID,
        help="RunPod serverless endpoint id for serverless track",
    )
    parser.add_argument(
        "--serverless-model",
        default=SERVERLESS_STRONG_MODEL,
        help="OpenAI model id on the serverless endpoint",
    )
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
            tracks.append(compact_track(args.compact_weights))
        elif name == "serverless":
            tracks.append(
                serverless_strong_control_track(
                    args.serverless_endpoint,
                    args.serverless_model,
                )
            )
        elif name == "api":
            tracks.append(api_teacher_track(args.api_model))
        elif name == "student":
            tracks.append(student_track(args.student_weights))
        elif name == "frontier":
            tracks.append(frontier_track(args.frontier_weights))
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
            "student_weights": args.student_weights,
            "api_model": args.api_model,
            "serverless_endpoint": args.serverless_endpoint,
            "serverless_model": args.serverless_model,
        },
    )
    print(json.dumps({"output": str(args.output), "n_results": len(results)}, indent=2))


if __name__ == "__main__":
    main()
