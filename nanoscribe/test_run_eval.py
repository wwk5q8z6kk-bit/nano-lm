import pytest
from unittest.mock import MagicMock
from nanoscribe.run_eval import _aggregate_from_report
from nanoscribe.evaluate import EvalReport

def test_aggregate_from_report():
    report = MagicMock(spec=EvalReport)
    report.exact_gold_span = 5
    report.span_character_f1 = 0.87654
    report.assertion_state_correct = 4
    report.support_direct_exact = 3
    report.support_normalized = 2
    report.support_review_required = 1
    report.wrong_source = 0
    report.wrong_mention = 1
    report.invalid_span = 0
    report.omission = 2
    report.correct_abstention = 1
    report.unnecessary_abstention = 0
    report.spurious_atom = 3
    report.malformed = 0
    report.critical_error = 1
    report.coverage = 0.98761

    result = _aggregate_from_report(report)

    expected = {
        "exact_gold_span": 5,
        "span_character_f1": 0.8765,
        "assertion_state_correct": 4,
        "support_direct_exact": 3,
        "support_normalized": 2,
        "support_review_required": 1,
        "wrong_source": 0,
        "wrong_mention": 1,
        "invalid_span": 0,
        "omission": 2,
        "correct_abstention": 1,
        "unnecessary_abstention": 0,
        "spurious_atom": 3,
        "malformed": 0,
        "critical_error": 1,
        "coverage": 0.9876,
    }

    assert result == expected
