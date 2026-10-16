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

from nanoscribe.benchmark import run_baseline
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
        mock_report.exact_gold_span = 4
        mock_report.span_character_f1 = 1.0
        mock_report.assertion_state_correct = 4
        mock_report.support_direct_exact = 3
        mock_report.support_normalized = 0
        mock_report.support_review_required = 1
        mock_report.wrong_source = 0
        mock_report.wrong_mention = 0
        mock_report.invalid_span = 0
        mock_report.omission = 0
        mock_report.correct_abstention = 1
        mock_report.unnecessary_abstention = 0
        mock_report.spurious_atom = 0
        mock_report.malformed = 0
        mock_report.critical_error = 0
        mock_report.coverage = 1.0

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

            assert mock_adapter.called
            assert mock_pipeline.called

            assert "baseline" in result
            assert result["baseline"] == "span_port_quote_generation"
            assert "model_family" in result
            assert "aggregate" in result
            assert result["aggregate"]["exact_gold_span"] == 4
            assert result["aggregate"]["coverage"] == 1.0

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
