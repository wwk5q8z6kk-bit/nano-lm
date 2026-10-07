import pytest
from unittest.mock import patch, MagicMock
from nanoscribe.benchmark import run_baseline
from nanoscribe.evaluate import EvalReport, AtomEval, SupportRelation

@patch('nanoscribe.benchmark.run_pipeline')
@patch('nanoscribe.benchmark.default_qwen_fixture_adapter')
@patch('nanoscribe.benchmark._gold')
@patch('nanoscribe.benchmark._model_input')
def test_run_baseline_with_mocks(mock_model_input, mock_gold, mock_adapter, mock_run_pipeline):
    # Setup mocks
    mock_gold.return_value.sources = ["mock_source"]
    mock_model_input_instance = MagicMock()
    mock_model_input.return_value = mock_model_input_instance

    mock_adapter_instance = MagicMock()
    mock_adapter.return_value = mock_adapter_instance
    mock_batch = MagicMock()
    mock_adapter_instance.propose.return_value = mock_batch

    mock_predicted = MagicMock()
    mock_report = MagicMock(spec=EvalReport)
    mock_report.exact_gold_span = 5
    mock_report.span_character_f1 = 0.95
    mock_report.assertion_state_correct = 4
    mock_report.support_direct_exact = 3
    mock_report.support_normalized = 1
    mock_report.support_review_required = 0
    mock_report.wrong_source = 0
    mock_report.wrong_mention = 0
    mock_report.invalid_span = 0
    mock_report.omission = 1
    mock_report.correct_abstention = 1
    mock_report.unnecessary_abstention = 0
    mock_report.spurious_atom = 0
    mock_report.malformed = 0
    mock_report.critical_error = 0
    mock_report.coverage = 0.85

    mock_atom1 = MagicMock(spec=AtomEval)
    mock_atom1.atom_id = "atom1"
    mock_atom1.exact_gold_span = True
    mock_atom1.span_character_f1 = 1.0
    mock_atom1.support_relation = SupportRelation.DIRECT_EXACT
    mock_atom1.assertion_state_correct = True
    mock_atom1.abstained = False
    mock_atom1.malformed = False
    mock_atom1.omitted = False

    mock_atom2 = MagicMock(spec=AtomEval)
    mock_atom2.atom_id = "atom2"
    mock_atom2.exact_gold_span = False
    mock_atom2.span_character_f1 = 0.0
    mock_atom2.support_relation = None
    mock_atom2.assertion_state_correct = False
    mock_atom2.abstained = True
    mock_atom2.malformed = False
    mock_atom2.omitted = False

    mock_report.atom_results = [mock_atom1, mock_atom2]

    mock_run_pipeline.return_value = (mock_predicted, mock_report)

    # Run function
    result = run_baseline()

    # Assertions
    assert result["baseline"] == "span_port_quote_generation"
    assert result["model_family"] == "qwen2.5-1.5b-style-one-liner"

    # Check aggregate
    aggregate = result["aggregate"]
    assert aggregate["exact_gold_span"] == 5
    assert aggregate["span_character_f1"] == 0.9500
    assert aggregate["assertion_state_correct"] == 4
    assert aggregate["coverage"] == 0.8500
    assert aggregate["support_direct_exact"] == 3

    # Check layers
    assert "layers" in result["layers"]
    assert "support_mix" in result["layers"]

    # Check per_atom
    per_atom = result["per_atom"]
    assert "atom1" in per_atom
    assert per_atom["atom1"]["exact_gold_span"] is True
    assert per_atom["atom1"]["support_relation"] == "direct_exact"

    assert "atom2" in per_atom
    assert per_atom["atom2"]["abstained"] is True
    assert per_atom["atom2"]["support_relation"] is None
