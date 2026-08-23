"""Baseline bridge: span-port style answers through adapt → evaluate.

Stdlib only. Simulates model output without loading weights.
Run: python3 nanoscribe/benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.adapt import run_pipeline
from nanoscribe.adapters import default_baseline_specs, default_qwen_fixture_adapter
from nanoscribe.test_adapt import _gold, _model_input


BASELINE_CASES = default_baseline_specs()


def run_baseline() -> dict[str, object]:
    gold = _gold()
    model_input = _model_input(gold.sources[0])
    adapter = default_qwen_fixture_adapter()
    batch = adapter.propose(model_input, BASELINE_CASES)
    predicted, report = run_pipeline(model_input, batch, gold=gold)
    assert report is not None
    decomposition = {
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
    per_atom = {
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
    return {
        "baseline": "span_port_quote_generation",
        "model_family": "qwen2.5-1.5b-style-one-liner",
        "note": "Software simulation of historical quote+label output; no weights loaded.",
        "aggregate": decomposition,
        "per_atom": per_atom,
    }


def main() -> None:
    result = run_baseline()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
