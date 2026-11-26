from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from nano_ai.training import evaluate_evidence_query_h7


def _field(*, span_exact: bool, state_correct: bool = True) -> dict[str, object]:
    return {
        "field": "chief_complaint",
        "span_exact": span_exact,
        "state_correct": state_correct,
    }


def _arm(
    *,
    seed: int | None,
    binding_errors: int,
    state_accuracy: float,
    span_exact: tuple[bool, ...],
    unsafe: int = 0,
    decode_failures: int = 0,
    collapse: int = 0,
) -> dict[str, object]:
    return {
        "candidate": {"seed": seed},
        "metrics": {
            "state_accuracy": state_accuracy,
            "decode_failure_items": decode_failures,
        },
        "mechanism": {
            "binding_error_fields": binding_errors,
            "disjoint_or_speaker_mismatch_fields": unsafe,
            "field_collapse_count": collapse,
        },
        "raw": {
            "item_diagnostics": [
                {
                    "example_id": f"row-{index}",
                    "fields": [_field(span_exact=value)],
                }
                for index, value in enumerate(span_exact)
            ]
        },
    }


def test_h7_comparison_uses_shared_state_cohort_and_all_preregistered_bars() -> None:
    control = _arm(
        seed=None,
        binding_errors=10,
        state_accuracy=0.90,
        span_exact=(True,) * 5 + (False,) * 5,
        unsafe=2,
    )
    treatment = _arm(
        seed=20260805,
        binding_errors=8,
        state_accuracy=0.89,
        span_exact=(True,) * 6 + (False,) * 4,
        unsafe=2,
    )

    result = evaluate_evidence_query_h7._comparison(control, treatment)

    assert result["binding_error_fields"]["relative_reduction"] == pytest.approx(0.2)
    assert result["span_exact_on_shared_state_correct"] == {
        "denominator": 10,
        "control_numerator": 5,
        "control_rate": 0.5,
        "treatment_numerator": 6,
        "treatment_rate": 0.6,
        "absolute_gain": pytest.approx(0.1),
    }
    assert all(result["bars"].values())
    assert result["all_pass_bars"] is True


def test_h7_decision_is_pass_reject_or_gray_exactly_as_preregistered() -> None:
    passing = {
        "all_pass_bars": True,
        "safety_or_state_violation": False,
        "boundary_improvement_bars_met": True,
    }
    safe_mixed = {
        "all_pass_bars": False,
        "safety_or_state_violation": False,
        "boundary_improvement_bars_met": False,
    }
    unsafe = {
        "all_pass_bars": False,
        "safety_or_state_violation": True,
        "boundary_improvement_bars_met": True,
    }

    assert evaluate_evidence_query_h7._decision((passing, passing)) == "PASS"
    assert evaluate_evidence_query_h7._decision((passing, safe_mixed)) == "GRAY"
    assert evaluate_evidence_query_h7._decision((safe_mixed, safe_mixed)) == "REJECT"
    assert evaluate_evidence_query_h7._decision((passing, unsafe)) == "REJECT"


def test_h7_control_authentication_rejects_any_noncanonical_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "control.pt"
    checkpoint.write_bytes(b"not the frozen H6 control")

    with pytest.raises(evaluate_evidence_query_h7.H7EvaluationError, match="mismatch"):
        evaluate_evidence_query_h7._control_candidate(checkpoint)


def test_h7_screen_is_opened_only_after_arm_and_tokenizer_authentication() -> None:
    source = inspect.getsource(evaluate_evidence_query_h7.evaluate_h7_screen)

    assert source.index("_control_candidate") < source.index("_load_screen")
    assert source.index("_validate_treatment_report") < source.index("_load_screen")
    assert source.index("_load_model") < source.index("_load_screen")
    assert source.index("load_pointer_tokenizer") < source.index("_load_screen")


def test_h7_evaluator_pins_the_committed_screen_bytes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    screen_root = repo_root / "artifacts" / "nano_h7" / "screen-v1"

    assert hashlib.sha256((screen_root / "manifest.json").read_bytes()).hexdigest() == (
        evaluate_evidence_query_h7.H7_SCREEN_MANIFEST_SHA256
    )
    assert hashlib.sha256((screen_root / "screen.jsonl").read_bytes()).hexdigest() == (
        evaluate_evidence_query_h7.H7_SCREEN_SHA256
    )
