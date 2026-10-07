import json
from unittest.mock import patch, MagicMock

from nanoscribe.benchmark import run_baseline, main

@patch("nanoscribe.benchmark.run_pipeline")
@patch("nanoscribe.benchmark.default_qwen_fixture_adapter")
@patch("nanoscribe.benchmark._gold")
@patch("nanoscribe.benchmark._model_input")
def test_run_baseline(mock_model_input, mock_gold, mock_adapter, mock_run_pipeline):
    mock_gold.return_value.sources = ["dummy_source"]

    mock_adapter_instance = MagicMock()
    mock_adapter.return_value = mock_adapter_instance
    mock_adapter_instance.propose.return_value = "dummy_batch"

    mock_report = MagicMock()
    mock_report.exact_gold_span = 1
    mock_report.span_character_f1 = 0.95
    mock_report.assertion_state_correct = 1
    mock_report.support_direct_exact = 1
    mock_report.support_normalized = 0
    mock_report.support_review_required = 0
    mock_report.wrong_source = 0
    mock_report.wrong_mention = 0
    mock_report.invalid_span = 0
    mock_report.omission = 0
    mock_report.correct_abstention = 0
    mock_report.unnecessary_abstention = 0
    mock_report.spurious_atom = 0
    mock_report.malformed = 0
    mock_report.critical_error = 0
    mock_report.coverage = 0.9

    mock_atom = MagicMock()
    mock_atom.atom_id = "atom1"
    mock_atom.exact_gold_span = True
    mock_atom.span_character_f1 = 0.95
    mock_atom.support_relation.value = "direct_exact"
    mock_atom.assertion_state_correct = True
    mock_atom.abstained = False
    mock_atom.malformed = False
    mock_atom.omitted = False

    mock_atom2 = MagicMock()
    mock_atom2.atom_id = "atom2"
    mock_atom2.exact_gold_span = False
    mock_atom2.span_character_f1 = 0.0
    mock_atom2.support_relation = None
    mock_atom2.assertion_state_correct = False
    mock_atom2.abstained = True
    mock_atom2.malformed = False
    mock_atom2.omitted = False

    mock_report.atom_results = [mock_atom, mock_atom2]

    mock_run_pipeline.return_value = ("predicted", mock_report)

    with patch("nanoscribe.benchmark.classify_report", return_value="dummy_layers"):
        result = run_baseline()

    assert result["baseline"] == "span_port_quote_generation"
    assert result["aggregate"]["exact_gold_span"] == 1
    assert result["aggregate"]["span_character_f1"] == 0.95
    assert result["aggregate"]["coverage"] == 0.9

    assert result["per_atom"]["atom1"]["exact_gold_span"] is True
    assert result["per_atom"]["atom1"]["support_relation"] == "direct_exact"

    assert result["per_atom"]["atom2"]["exact_gold_span"] is False
    assert result["per_atom"]["atom2"]["support_relation"] is None

    assert result["layers"] == "dummy_layers"

    mock_adapter.assert_called_once()
    mock_adapter_instance.propose.assert_called_once()
    mock_run_pipeline.assert_called_once()

@patch("nanoscribe.benchmark.run_baseline")
def test_main(mock_run_baseline, capsys):
    mock_run_baseline.return_value = {"key": "value"}
    main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"key": "value"}
