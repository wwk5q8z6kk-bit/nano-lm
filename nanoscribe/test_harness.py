from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.harness import (
    FailureTaxonomy,
    HarnessCase,
    HarnessResult,
    ModelTrack,
    P1TestSet,
    TrackConfig,
    run_case,
    run_matrix,
    write_results,
)
from nanoscribe.evaluate import EvalReport

def test_failure_taxonomy_from_report():
    report = EvalReport(
        exact_gold_span=0,
        span_character_f1=0.0,
        assertion_state_correct=0,
        support_direct_exact=0,
        support_normalized=0,
        support_semantically_supported=0,
        support_unsupported=0,
        support_contradicted=0,
        support_review_required=0,
        invalid_span=1,
        wrong_source=2,
        wrong_mention=3,
        ambiguity=4,
        omission=5,
        correct_abstention=0,
        unnecessary_abstention=6,
        malformed=7,
        critical_error=8,
        spurious_atom=9,
        coverage=0.0,
        atom_results=(),
        latency_s=0.0,
        memory_bytes=0,
    )

    failures = FailureTaxonomy.from_report(report)
    assert failures.invalid_span == 1
    assert failures.wrong_source == 2
    assert failures.wrong_mention == 3
    assert failures.ambiguity == 4
    assert failures.omission == 5
    assert failures.unnecessary_abstention == 6
    assert failures.malformed == 7
    assert failures.critical_error == 8
    assert failures.spurious_atom == 9

def test_failure_taxonomy_to_dict():
    failures = FailureTaxonomy(
        invalid_span=1,
        wrong_source=2,
        wrong_mention=3,
        omission=5,
        unnecessary_abstention=6,
        malformed=7,
        critical_error=8,
        spurious_atom=9,
        ambiguity=4,
    )

    d = failures.to_dict()
    assert d == {
        "invalid_span": 1,
        "wrong_source": 2,
        "wrong_mention": 3,
        "omission": 5,
        "unnecessary_abstention": 6,
        "malformed": 7,
        "critical_error": 8,
        "spurious_atom": 9,
        "ambiguity": 4,
    }

def test_harness_result_to_dict():
    failures = FailureTaxonomy()
    res = HarnessResult(
        track=ModelTrack.FIXTURE,
        model_id="test_model",
        test_set=P1TestSet.TINY_FIXTURE,
        encounter_id="enc-1",
        cost_class="low",
        aggregate={"f1": 1.0},
        failures=failures,
        per_atom={"atom-1": {"f1": 1.0}},
        latency_s=1.23456,
        memory_bytes=1024,
        raw_lines={"atom-1": "STATED: neck"},
    )

    d = res.to_dict()
    assert d["track"] == "fixture"
    assert d["model_id"] == "test_model"
    assert d["test_set"] == "tiny_fixture"
    assert d["encounter_id"] == "enc-1"
    assert d["cost_class"] == "low"
    assert d["aggregate"] == {"f1": 1.0}
    assert d["failure_taxonomy"] == failures.to_dict()
    assert d["per_atom"] == {"atom-1": {"f1": 1.0}}
    assert d["latency_s"] == 1.2346
    assert d["memory_bytes"] == 1024
    assert d["raw_lines"] == {"atom-1": "STATED: neck"}

from unittest.mock import MagicMock

def test_run_case_aggregate():
    track_mock = MagicMock(spec=TrackConfig)
    track_mock.track = ModelTrack.FIXTURE
    track_mock.model_id = "test_model"
    track_mock.cost_class = "low"

    mock_adapter = MagicMock()
    mock_batch = MagicMock()
    mock_batch.latency_s = 1.0
    mock_batch.memory_bytes = 100
    mock_batch.atoms = []
    mock_adapter.propose.return_value = mock_batch
    track_mock.adapter_factory.return_value = mock_adapter

    case_mock = MagicMock(spec=HarnessCase)
    case_mock.test_set = P1TestSet.TINY_FIXTURE
    case_mock.encounter_id = "enc-1"
    case_mock.gold = "gold"
    case_mock.model_input = "input"
    case_mock.atom_specs = "specs"

    # We need to mock nanoscribe.harness.run_pipeline which is imported directly
    import nanoscribe.harness as harness_module

    # Backup original
    original_run_pipeline = harness_module.run_pipeline

    mock_predicted = MagicMock()
    mock_report = MagicMock(spec=EvalReport)
    # mock fields for report
    mock_report.exact_gold_span = 1
    mock_report.span_character_f1 = 0.5
    mock_report.assertion_state_correct = 1
    mock_report.support_direct_exact = 1
    mock_report.support_normalized = 0
    mock_report.support_review_required = 0
    mock_report.coverage = 1.0
    mock_report.correct_abstention = 0

    # mock fields for FailureTaxonomy
    mock_report.invalid_span = 0
    mock_report.wrong_source = 0
    mock_report.wrong_mention = 0
    mock_report.omission = 0
    mock_report.unnecessary_abstention = 0
    mock_report.malformed = 0
    mock_report.critical_error = 0
    mock_report.spurious_atom = 0
    mock_report.ambiguity = 0

    mock_report.atom_results = []

    def fake_run_pipeline(input_, batch, gold):
        return mock_predicted, mock_report

    harness_module.run_pipeline = fake_run_pipeline

    try:
        res = run_case(track_mock, case_mock)
        assert res.track == ModelTrack.FIXTURE
        assert res.model_id == "test_model"
        assert res.cost_class == "low"
        assert res.test_set == P1TestSet.TINY_FIXTURE
        assert res.encounter_id == "enc-1"
        assert res.aggregate["exact_gold_span"] == 1
        assert res.latency_s == 1.0
        assert res.memory_bytes == 100
    finally:
        harness_module.run_pipeline = original_run_pipeline

def test_run_matrix():
    track_mock = MagicMock(spec=TrackConfig)
    track_mock.track = ModelTrack.FIXTURE
    track_mock.model_id = "test_model"
    track_mock.cost_class = "low"

    mock_adapter = MagicMock()
    mock_batch = MagicMock()
    mock_batch.latency_s = 1.0
    mock_batch.memory_bytes = 100
    mock_batch.atoms = []
    mock_adapter.propose.return_value = mock_batch
    track_mock.adapter_factory.return_value = mock_adapter

    case_mock1 = MagicMock(spec=HarnessCase)
    case_mock1.test_set = P1TestSet.TINY_FIXTURE
    case_mock1.encounter_id = "enc-1"

    case_mock2 = MagicMock(spec=HarnessCase)
    case_mock2.test_set = P1TestSet.TINY_FIXTURE
    case_mock2.encounter_id = "enc-2"

    import nanoscribe.harness as harness_module
    original_run_pipeline = harness_module.run_pipeline

    mock_predicted = MagicMock()
    mock_report = MagicMock(spec=EvalReport)
    # mock fields for report
    mock_report.exact_gold_span = 1
    mock_report.span_character_f1 = 0.5
    mock_report.assertion_state_correct = 1
    mock_report.support_direct_exact = 1
    mock_report.support_normalized = 0
    mock_report.support_review_required = 0
    mock_report.coverage = 1.0
    mock_report.correct_abstention = 0

    # mock fields for FailureTaxonomy
    mock_report.invalid_span = 0
    mock_report.wrong_source = 0
    mock_report.wrong_mention = 0
    mock_report.omission = 0
    mock_report.unnecessary_abstention = 0
    mock_report.malformed = 0
    mock_report.critical_error = 0
    mock_report.spurious_atom = 0
    mock_report.ambiguity = 0

    mock_report.atom_results = []

    def fake_run_pipeline(input_, batch, gold):
        return mock_predicted, mock_report

    harness_module.run_pipeline = fake_run_pipeline

    try:
        results = run_matrix([track_mock], [case_mock1, case_mock2])
        assert len(results) == 2
        assert results[0].encounter_id == "enc-1"
        assert results[1].encounter_id == "enc-2"
    finally:
        harness_module.run_pipeline = original_run_pipeline

def test_write_results(tmp_path):
    failures = FailureTaxonomy()
    res = HarnessResult(
        track=ModelTrack.FIXTURE,
        model_id="test_model",
        test_set=P1TestSet.TINY_FIXTURE,
        encounter_id="enc-1",
        cost_class="low",
        aggregate={"f1": 1.0},
        failures=failures,
        per_atom={"atom-1": {"f1": 1.0}},
        latency_s=1.23456,
        memory_bytes=1024,
    )

    output_path = tmp_path / "results.json"
    write_results([res], output_path, extra={"key": "value"})

    assert output_path.exists()
    import json
    data = json.loads(output_path.read_text())
    assert data["schema"] == "nano.p1.harness_result.v0"
    assert data["meta"] == {"key": "value"}
    assert len(data["results"]) == 1
    assert data["results"][0]["encounter_id"] == "enc-1"

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
