#!/usr/bin/env python3
"""Baseline eval runner: adapter → pipeline → PR2 metrics + failure layers.

Run: python3 nanoscribe/run_eval.py [--fixture-only]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_repo_root = str(Path(__file__).resolve().parents[1])
_script_dir = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
sys.path[:] = [p for p in sys.path if p != _script_dir]

from nanoscribe.adapt import run_pipeline
from nanoscribe.adapters import (
    DEFAULT_BASELINE_LINES,
    Qwen25BaselineAdapter,
    default_baseline_specs,
)
from nanoscribe.decompose import classify_report
from nanoscribe.test_adapt import _gold, _model_input


@contextmanager
def _without_qwen_weights_env():
    saved = os.environ.pop("NANOSCIBE_QWEN_WEIGHTS", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["NANOSCIBE_QWEN_WEIGHTS"] = saved


def _aggregate_from_report(report) -> dict[str, Any]:
    return {
        "exact_gold_span": report.exact_gold_span,
        "span_character_f1": round(report.span_character_f1, 4),
        "assertion_state_correct": report.assertion_state_correct,
        "support_direct_exact": report.support_direct_exact,
        "support_normalized": report.support_normalized,
        "support_review_required": report.support_review_required,
        "wrong_source": report.wrong_source,
        "wrong_mention": report.wrong_mention,
        "invalid_span": report.invalid_span,
        "omission": report.omission,
        "correct_abstention": report.correct_abstention,
        "unnecessary_abstention": report.unnecessary_abstention,
        "spurious_atom": report.spurious_atom,
        "malformed": report.malformed,
        "critical_error": report.critical_error,
        "coverage": round(report.coverage, 4),
    }


def _per_atom_from_report(report) -> dict[str, dict[str, Any]]:
    return {
        item.atom_id: {
            "exact_gold_span": item.exact_gold_span,
            "span_character_f1": round(item.span_character_f1, 4),
            "support_relation": (
                item.support_relation.value if item.support_relation else None
            ),
            "assertion_state_correct": item.assertion_state_correct,
            "abstained": item.abstained,
            "malformed": item.malformed,
            "omitted": item.omitted,
        }
        for item in report.atom_results
    }


def run_baseline_eval(*, fixture_only: bool = False) -> dict[str, Any]:
    """Run the deterministic baseline encounter through adapter → eval → layers."""
    specs = default_baseline_specs()
    gold = _gold()
    model_input = _model_input(gold.sources[0])

    def _run() -> dict[str, Any]:
        adapter = Qwen25BaselineAdapter(fixture_lines=DEFAULT_BASELINE_LINES)
        batch = adapter.propose(model_input, specs)
        _, report = run_pipeline(model_input, batch, gold=gold)
        assert report is not None
        return {
            "experiment": "p1_baseline_eval_v0",
            "fixture_only": fixture_only,
            "adapter": adapter.model_id,
            "aggregate": _aggregate_from_report(report),
            "layers": classify_report(report),
            "per_atom": _per_atom_from_report(report),
        }

    if fixture_only:
        with _without_qwen_weights_env():
            return _run()
    return _run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1 baseline eval with layer decomposition")
    parser.add_argument(
        "--fixture-only",
        action="store_true",
        help="Use span-port fixture even if NANOSCIBE_QWEN_WEIGHTS is set",
    )
    args = parser.parse_args(argv)
    result = run_baseline_eval(fixture_only=args.fixture_only)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
