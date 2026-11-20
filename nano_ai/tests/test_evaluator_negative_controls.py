"""Tests for X01 mandatory evaluator negative controls (`annotated_evaluation.py`)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from nano_ai.contract import FIELD_ORDER, FieldName, FieldState
from nano_ai.training.annotated_evaluation import verify_evaluator_negative_controls
from nano_ai.training.state_span_data import StateSpanExample


def _fixture_corpus() -> tuple[tuple[StateSpanExample, ...], dict[FieldName, dict[FieldState, int]]]:
    rows = (
        StateSpanExample(
            split="dev",
            example_id="x01-case-1",
            world_id="w-1",
            variant="mixed",
            target_field=FieldName.ALLERGY,
            target_state=FieldState.CONFLICTING,
            transcript=(
                "Doctor: What brings you in today?\n"
                "Patient: headache\n"
                "Patient: possibly mild\n"
                "Patient: no medications\n"
                "Patient: penicillin\n"
                "Patient: latex"
            ),
            target="CC:S[headache]|DUR:M|SEV:U[possibly mild]|MED:A[no medications]|ALG:C[penicillin;latex]",
        ),
        StateSpanExample(
            split="dev",
            example_id="x01-case-2",
            world_id="w-2",
            variant="supported",
            target_field=FieldName.DURATION,
            target_state=FieldState.SUPPORTED,
            transcript=(
                "Doctor: Tell me about your cough.\n"
                "Patient: dry cough\n"
                "Doctor: How long has it been?\n"
                "Patient: three days\n"
                "Doctor: How bad is it?\n"
                "Patient: moderate\n"
                "Doctor: Any allergies?\n"
                "Patient: no known allergies"
            ),
            target="CC:S[dry cough]|DUR:S[three days]|SEV:S[moderate]|MED:M|ALG:A[no known allergies]",
        ),
    )
    counts = {
        FieldName.CHIEF_COMPLAINT: {s: (2 if s is FieldState.SUPPORTED else 0) for s in FieldState},
        FieldName.DURATION: {
            FieldState.SUPPORTED: 1,
            FieldState.ABSENT: 0,
            FieldState.MISSING: 1,
            FieldState.UNCERTAIN: 0,
            FieldState.CONFLICTING: 0,
        },
        FieldName.SEVERITY: {
            FieldState.SUPPORTED: 1,
            FieldState.ABSENT: 0,
            FieldState.MISSING: 0,
            FieldState.UNCERTAIN: 1,
            FieldState.CONFLICTING: 0,
        },
        FieldName.MEDICATION: {
            FieldState.SUPPORTED: 0,
            FieldState.ABSENT: 1,
            FieldState.MISSING: 1,
            FieldState.UNCERTAIN: 0,
            FieldState.CONFLICTING: 0,
        },
        FieldName.ALLERGY: {
            FieldState.SUPPORTED: 0,
            FieldState.ABSENT: 1,
            FieldState.MISSING: 0,
            FieldState.UNCERTAIN: 0,
            FieldState.CONFLICTING: 1,
        },
    }
    return rows, counts


def test_all_six_mandatory_negative_controls_pass():
    rows, counts = _fixture_corpus()
    report = verify_evaluator_negative_controls(rows, expected_counts=counts)
    assert report.passed is True
    assert set(report.controls) == {
        "gold_oracle",
        "all_missing_negative",
        "shuffled_labels_negative",
        "wrong_patient_negative",
        "corrupted_offset_negative",
        "denial_conflict_and_abstain_controls",
    }
    for name, details in report.controls.items():
        assert details["passed"] is True, f"control failed: {name}: {details}"
    assert report.controls["gold_oracle"]["grounded_exact_field_accuracy"] == 1.0
    assert report.controls["all_missing_negative"]["quota_mismatch_failed_closed"] is True
    assert report.controls["all_missing_negative"]["all_missing_grounded_exact"] < 1.0
    assert report.controls["denial_conflict_and_abstain_controls"]["abstain_control_presented"] == 0
    assert report.controls["corrupted_offset_negative"] == {
        "passed": True, "evidence_cases_checked": 2, "synthetic_cases_checked": 0, "offset_rejections": 2,
    }


def test_doctor_only_gold_annotation_rejected_before_controls_run():
    rows, counts = _fixture_corpus()
    bad = replace(
        rows[0],
        target="CC:S[What brings you in today?]|DUR:M|SEV:U[possibly mild]|MED:A[no medications]|ALG:C[penicillin;latex]",
    )
    with pytest.raises(ValueError):
        verify_evaluator_negative_controls((bad, rows[1]), expected_counts=counts)


def test_quota_disagreement_fails_closed():
    rows, counts = _fixture_corpus()
    bad_counts = {f: dict(v) for f, v in counts.items()}
    bad_counts[FieldName.CHIEF_COMPLAINT][FieldState.SUPPORTED] = 1
    bad_counts[FieldName.CHIEF_COMPLAINT][FieldState.MISSING] = 1
    with pytest.raises(ValueError, match="counts"):
        verify_evaluator_negative_controls(rows, expected_counts=bad_counts)


def test_identical_annotations_leave_rotation_inapplicable():
    from nano_ai.tests.test_annotated_evaluation import example, quotas

    first = example()
    second = replace(first, example_id="another", world_id="other-world",
                     transcript=first.transcript + "\nDoctor: Thank you.")
    counts = {f: {s: 2 * n for s, n in cells.items()} for f, cells in quotas().items()}
    report = verify_evaluator_negative_controls((first, second), expected_counts=counts)
    assert report.passed
    assert report.controls["shuffled_labels_negative"]["cross_case_ok"] is None


@pytest.mark.parametrize("accept_mismatched_offset", (False, True))
def test_offset_control_exercises_real_evaluator_without_annotated_evidence(
    monkeypatch, accept_mismatched_offset
):
    from nano_ai.contract import ContractValidationError, EvidenceSpan

    rows, _ = _fixture_corpus()
    row = replace(rows[0], transcript="Patient: Hello.", target_state=FieldState.MISSING,
                  target="CC:M|DUR:M|SEV:M|MED:M|ALG:M")
    counts = {field: {state: int(state is FieldState.MISSING) for state in FieldState}
              for field in FIELD_ORDER}
    if accept_mismatched_offset:
        validate = EvidenceSpan.validate_against

        def accept_text_mismatch(self, transcript, **kwargs):
            try:
                validate(self, transcript, **kwargs)
            except ContractValidationError as exc:
                if exc.code != "evidence_text_mismatch":
                    raise

        monkeypatch.setattr(EvidenceSpan, "validate_against", accept_text_mismatch)

    report = verify_evaluator_negative_controls((row,), expected_counts=counts)
    offset = report.controls["corrupted_offset_negative"]
    assert offset["passed"] is (not accept_mismatched_offset)
    assert report.passed is (not accept_mismatched_offset)
    assert offset["evidence_cases_checked"] == 0
    assert offset["synthetic_cases_checked"] == 1
    assert offset["offset_rejections"] == int(not accept_mismatched_offset)
