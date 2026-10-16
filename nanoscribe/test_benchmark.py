import json
from unittest.mock import MagicMock, patch

from nanoscribe.benchmark import main, run_baseline


@patch("nanoscribe.benchmark.classify_report")
@patch("nanoscribe.benchmark.run_pipeline")
@patch("nanoscribe.benchmark.default_qwen_fixture_adapter")
@patch("nanoscribe.benchmark._model_input")
@patch("nanoscribe.benchmark._gold")
def test_run_baseline(
    mock_gold, mock_model_input, mock_adapter, mock_run_pipeline, mock_classify
) -> None:
    # Setup mocks
    mock_gold_instance = MagicMock()
    mock_gold_instance.sources = ["source_0"]
    mock_gold.return_value = mock_gold_instance

    mock_model_input_instance = MagicMock()
    mock_model_input.return_value = mock_model_input_instance

    mock_adapter_instance = MagicMock()
    mock_adapter.return_value = mock_adapter_instance
    mock_adapter_instance.propose.return_value = ["batch_item"]

    mock_report = MagicMock()
    mock_report.exact_gold_span = 4
    mock_report.span_character_f1 = 0.95
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
    mock_report.coverage = 0.9

    mock_atom = MagicMock()
    mock_atom.atom_id = "atom-1"
    mock_atom.exact_gold_span = True
    mock_atom.span_character_f1 = 1.0
    mock_support = MagicMock()
    mock_support.value = "direct_exact"
    mock_atom.support_relation = mock_support
    mock_atom.assertion_state_correct = True
    mock_atom.abstained = False
    mock_atom.malformed = False
    mock_atom.omitted = False
    mock_report.atom_results = [mock_atom]

    mock_run_pipeline.return_value = (MagicMock(), mock_report)

    mock_classify.return_value = {"mock": "layers"}

    # Run
    result = run_baseline()

    # Assertions
    mock_gold.assert_called_once_with()
    mock_model_input.assert_called_once_with("source_0")
    mock_adapter.assert_called_once_with()
    mock_adapter_instance.propose.assert_called_once()
    mock_run_pipeline.assert_called_once_with(
        mock_model_input_instance, ["batch_item"], gold=mock_gold_instance
    )
    mock_classify.assert_called_once_with(mock_report)

    assert result["baseline"] == "span_port_quote_generation"
    assert result["model_family"] == "qwen2.5-1.5b-style-one-liner"
    assert result["layers"] == {"mock": "layers"}

    assert result["aggregate"]["exact_gold_span"] == 4
    assert result["aggregate"]["span_character_f1"] == 0.95

    assert "atom-1" in result["per_atom"]
    assert result["per_atom"]["atom-1"]["support_relation"] == "direct_exact"
    assert result["per_atom"]["atom-1"]["exact_gold_span"] is True


@patch("nanoscribe.benchmark.run_baseline")
@patch("nanoscribe.benchmark.json.dumps")
@patch("builtins.print")
def test_main(mock_print, mock_dumps, mock_run_baseline) -> None:
    mock_run_baseline.return_value = {"test": "result"}
    mock_dumps.return_value = "json_string"

    main()

    mock_run_baseline.assert_called_once_with()
    mock_dumps.assert_called_once_with({"test": "result"}, indent=2, sort_keys=True)
    mock_print.assert_called_once_with("json_string")
