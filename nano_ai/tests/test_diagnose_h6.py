from __future__ import annotations

import json
from pathlib import Path

import pytest

from nano_ai.training.diagnose_h6 import (
    CANONICAL_H6_EVALUATION_SHA256,
    CANONICAL_H6_TERMINAL_DECISION_SHA256,
    DEFAULT_EVALUATION_PATH,
    DEFAULT_TERMINAL_DECISION_PATH,
    H6DiagnosisError,
    diagnose_h6,
    load_h6_diagnosis,
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_canonical_h6_diagnosis_separates_error_axes() -> None:
    report = load_h6_diagnosis()

    assert report["source"] == {
        "evaluation_sha256": CANONICAL_H6_EVALUATION_SHA256,
        "terminal_decision_sha256": CANONICAL_H6_TERMINAL_DECISION_SHA256,
        "rows": 1_000,
        "fields": 5_000,
    }
    assert report["overall"]["joint_correct"] == 3_901
    assert report["overall"]["errors"] == {
        "state_only_error": 241,
        "binding_only_error": 771,
        "combined_error": 87,
    }
    assert report["binding_shape"]["span_error_fields"] == 858
    assert report["binding_shape"]["relations"] == {
        "count_mismatch": 86,
        "disjoint": 12,
        "gold_contains_raw": 192,
        "mixed:disjoint+exact": 1,
        "mixed:exact+gold_contains_raw": 22,
        "mixed:exact+partial_overlap": 16,
        "mixed:exact+raw_contains_gold": 6,
        "mixed:gold_contains_raw+partial_overlap": 3,
        "mixed:gold_contains_raw+raw_contains_gold": 1,
        "mixed:partial_overlap+raw_contains_gold": 4,
        "partial_overlap": 48,
        "raw_contains_gold": 467,
    }
    assert report["binding_shape"]["by_field"]["chief_complaint"] == {
        "count_mismatch": 50,
        "disjoint": 11,
        "gold_contains_raw": 58,
        "partial_overlap": 9,
        "raw_contains_gold": 366,
    }

    absent = report["by_gold_state"]["absent"]
    assert absent["errors"] == {
        "state_only_error": 184,
        "binding_only_error": 0,
        "combined_error": 30,
    }
    assert absent["dominant_error_axis"] == "state_only_error"
    assert absent["raw_state_counts"]["supported"] == 177

    uncertain = report["by_gold_state"]["uncertain"]
    assert uncertain["errors"] == {
        "state_only_error": 53,
        "binding_only_error": 0,
        "combined_error": 7,
    }
    assert uncertain["dominant_error_axis"] == "state_only_error"

    conflicting = report["by_gold_state"]["conflicting"]
    assert conflicting["errors"] == {
        "state_only_error": 0,
        "binding_only_error": 57,
        "combined_error": 50,
    }
    assert conflicting["dominant_error_axis"] == "binding_only_error"

    assert report["stages"]["threshold_calibration"] == {"assessed": False}
    assert report["stages"]["verifier_final"] == {"assessed": False}
    assert report["scientific_boundary"] == {
        "raw_error_attribution_available": True,
        "threshold_effect": "not_assessed",
        "verifier_effect": "not_assessed",
        "requires_new_model_run": False,
    }


def test_h6_diagnosis_rejects_incoherent_joint_label() -> None:
    evaluation = _load(DEFAULT_EVALUATION_PATH)
    decision = _load(DEFAULT_TERMINAL_DECISION_PATH)
    field = evaluation["uncalibrated_raw"]["item_diagnostics"][0]["fields"][0]
    field["joint_exact"] = not field["joint_exact"]

    with pytest.raises(H6DiagnosisError, match="incoherent joint_exact"):
        diagnose_h6(
            evaluation,
            evaluation_sha256=decision["evaluation"]["result_sha256"],
            terminal_decision=decision,
            terminal_decision_sha256="0" * 64,
        )


def test_h6_diagnosis_rejects_result_not_bound_to_terminal_decision() -> None:
    evaluation = _load(DEFAULT_EVALUATION_PATH)
    decision = _load(DEFAULT_TERMINAL_DECISION_PATH)

    with pytest.raises(H6DiagnosisError, match="hash does not match"):
        diagnose_h6(
            evaluation,
            evaluation_sha256="0" * 64,
            terminal_decision=decision,
            terminal_decision_sha256="1" * 64,
        )


def test_h6_diagnosis_rejects_mutated_terminal_decision(tmp_path: Path) -> None:
    decision = _load(DEFAULT_TERMINAL_DECISION_PATH)
    decision["note"] = "mutated"
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(H6DiagnosisError, match="canonical hash"):
        load_h6_diagnosis(DEFAULT_EVALUATION_PATH, decision_path)
