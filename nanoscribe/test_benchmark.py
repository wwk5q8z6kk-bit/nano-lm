from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path before imports
_repo_root = str(Path(__file__).resolve().parents[1])
_script_dir = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# Remove script_dir from path to avoid shadowing select.py
sys.path[:] = [p for p in sys.path if p != _script_dir]

from unittest.mock import MagicMock, patch

from nanoscribe.benchmark import BASELINE_CASES, run_baseline
from nanoscribe.test_adapt import _gold, _model_input
from nanoscribe.adapters import Qwen25BaselineAdapter

def test_run_baseline() -> None:
    # We mock default_qwen_fixture_adapter to avoid initializing adapters
    # that might load heavy dependencies or weights, and we mock run_pipeline
    # to return a deterministic evaluation report.
    with patch("nanoscribe.benchmark.default_qwen_fixture_adapter") as mock_adapter, \
         patch("nanoscribe.benchmark.run_pipeline") as mock_pipeline:

        # Setup mocks
        mock_adapter_instance = MagicMock(spec=Qwen25BaselineAdapter)
        mock_adapter.return_value = mock_adapter_instance

        mock_report = MagicMock()
        expected_aggregate = {
            "exact_gold_span": 1,
            "span_character_f1": 0.8217,
            "assertion_state_correct": 3,
            "support_direct_exact": 4,
            "support_normalized": 5,
            "support_review_required": 6,
            "wrong_source": 7,
            "wrong_mention": 8,
            "invalid_span": 9,
            "omission": 10,
            "correct_abstention": 11,
            "unnecessary_abstention": 12,
            "spurious_atom": 13,
            "malformed": 14,
            "critical_error": 15,
            "coverage": 0.3719,
        }
        for field, value in expected_aggregate.items():
            setattr(mock_report, field, value)

        # mock per atom results
        atom_result_mock = MagicMock()
        atom_result_mock.atom_id = "atom-neck"
        atom_result_mock.exact_gold_span = True
        atom_result_mock.span_character_f1 = 1.0
        atom_result_mock.support_relation = MagicMock()
        atom_result_mock.support_relation.value = "direct_exact"
        atom_result_mock.assertion_state_correct = True
        atom_result_mock.abstained = False
        atom_result_mock.malformed = False
        atom_result_mock.omitted = False

        mock_report.atom_results = [atom_result_mock]

        mock_pipeline.return_value = (MagicMock(), mock_report)

        with patch("nanoscribe.benchmark.classify_report") as mock_classify_report:
            mock_classify_report.return_value = {"layers": {"transport": 0, "support": 0, "state": 0, "abstention": 0, "commission": 0, "malformed": 0}}

            result = run_baseline()

            gold = _gold()
            model_input = _model_input(gold.sources[0])
            mock_adapter.assert_called_once_with()
            mock_adapter_instance.propose.assert_called_once_with(model_input, BASELINE_CASES)
            mock_pipeline.assert_called_once_with(
                model_input, mock_adapter_instance.propose.return_value, gold=gold
            )

            assert "baseline" in result
            assert result["baseline"] == "span_port_quote_generation"
            assert "model_family" in result
            assert "aggregate" in result
            assert result["aggregate"] == expected_aggregate

            assert "layers" in result
            assert "layers" in result["layers"]

            assert "per_atom" in result
            assert "atom-neck" in result["per_atom"]
            assert result["per_atom"]["atom-neck"]["support_relation"] == "direct_exact"

if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in fns:
        fn()
        print(f"  PASS {name}")
    print(f"benchmark pins: {len(fns)}/{len(fns)} PASS")
