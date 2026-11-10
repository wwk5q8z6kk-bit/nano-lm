from __future__ import annotations

import pytest

from nano_ai.adapters.state_span import (
    STATE_SPAN_DIAGNOSTICS_VERSION,
    StateSpanFormatError,
    StateSpanSolver,
    parse_state_span_summary,
)
from nano_ai.contract import FIELD_ORDER, FieldName, FieldState, NanoInput
from nano_ai.solver import InferenceFailureCategory, run_inference

MIXED_STATE_TRANSCRIPT = """Doctor: Good morning, what brings you in today?
Patient: I've been having chest pain.
Doctor: How bad would you say it is?
Patient: I cannot tell.
Doctor: Have you taken anything for it?
Patient: No, nothing yet.
Doctor: Any allergies I should know about?
Patient: I'm allergic to penicillin.
Doctor: Are you allergic to anything?
Patient: No allergies."""

MIXED_STATE_SUMMARY = (
    "CC:S[chest pain]|DUR:M|SEV:U[I cannot tell.]|"
    "MED:A[No, nothing yet.]|ALG:C[penicillin;No allergies.]"
)


def _by_field(output):
    return {field.field: field for field in output.fields}


def test_native_grammar_accepts_all_five_states_with_exact_patient_offsets():
    request = NanoInput(item_id="mixed", transcript=MIXED_STATE_TRANSCRIPT)
    solver = StateSpanSolver(lambda transcript: MIXED_STATE_SUMMARY)

    output, diagnostics = solver.infer_with_state_diagnostics(request)
    fields = _by_field(output)

    assert [field.state for field in output.fields] == [
        FieldState.SUPPORTED,
        FieldState.MISSING,
        FieldState.UNCERTAIN,
        FieldState.ABSENT,
        FieldState.CONFLICTING,
    ]
    assert fields[FieldName.CHIEF_COMPLAINT].value == "chest pain"
    assert [span.text for span in fields[FieldName.ALLERGY].evidence] == [
        "penicillin",
        "No allergies.",
    ]
    for field in output.fields:
        for span in field.evidence:
            assert MIXED_STATE_TRANSCRIPT[span.start : span.end] == span.text
            assert span.speaker == "patient"

    assert diagnostics["protocol_version"] == STATE_SPAN_DIAGNOSTICS_VERSION
    assert diagnostics["raw_summary"] == MIXED_STATE_SUMMARY
    traces = diagnostics["fields"]
    assert isinstance(traces, list)
    assert [trace["raw_state_code"] for trace in traces] == list("SMUAC")
    assert all(trace["verified"] is True for trace in traces)
    assert [trace["field"] for trace in traces] == [
        field.value for field in FIELD_ORDER
    ]


def test_wrong_states_become_uncertain_without_preserving_binder_state():
    summary = (
        "CC:S[chest pain]|DUR:U[]|SEV:U[I cannot tell.]|"
        "MED:A[No, nothing yet.]|ALG:S[penicillin]"
    )
    request = NanoInput(item_id="wrong-states", transcript=MIXED_STATE_TRANSCRIPT)
    solver = StateSpanSolver(lambda transcript: summary)

    output, diagnostics = solver.infer_with_state_diagnostics(request)
    fields = _by_field(output)

    assert fields[FieldName.DURATION].state is FieldState.UNCERTAIN
    assert fields[FieldName.DURATION].evidence == ()
    assert fields[FieldName.ALLERGY].state is FieldState.UNCERTAIN
    assert [span.text for span in fields[FieldName.ALLERGY].evidence] == ["penicillin"]
    traces = diagnostics["fields"]
    assert isinstance(traces, list)
    duration_trace = traces[1]
    allergy_trace = traces[4]
    assert (duration_trace["raw_state"], duration_trace["verifier_state"]) == (
        "uncertain",
        "missing",
    )
    assert (allergy_trace["raw_state"], allergy_trace["verifier_state"]) == (
        "supported",
        "conflicting",
    )
    assert duration_trace["decision"] == "rejected_to_uncertain"
    assert allergy_trace["decision"] == "rejected_to_uncertain"


def test_wrong_supported_span_is_blocked_even_when_it_is_patient_authored():
    summary = (
        "CC:S[penicillin]|DUR:M|SEV:U[I cannot tell.]|MED:A[No, nothing yet.]|ALG:M"
    )
    request = NanoInput(item_id="wrong-value", transcript=MIXED_STATE_TRANSCRIPT)
    solver = StateSpanSolver(lambda transcript: summary)

    output, diagnostics = solver.infer_with_state_diagnostics(request)

    complaint = _by_field(output)[FieldName.CHIEF_COMPLAINT]
    assert complaint.state is FieldState.UNCERTAIN
    assert complaint.value is None
    assert complaint.evidence[0].text == "penicillin"
    complaint_trace = diagnostics["fields"][0]
    assert complaint_trace["verified"] is False
    assert complaint_trace["reason"] == "normalized_value_mismatch"


def test_empty_uncertainty_span_is_verified_only_without_binder_evidence():
    no_reply = "Doctor: How bad would you say it is?\nPatient:"
    request = NanoInput(item_id="empty-reply", transcript=no_reply)
    summary = "CC:M|DUR:M|SEV:U[]|MED:M|ALG:M"

    output, diagnostics = StateSpanSolver(
        lambda transcript: summary
    ).infer_with_state_diagnostics(request)

    severity = _by_field(output)[FieldName.SEVERITY]
    assert severity.state is FieldState.UNCERTAIN
    assert severity.evidence == ()
    assert diagnostics["fields"][2]["verified"] is True
    assert diagnostics["fields"][2]["reason"] == ("verified_evidenceless_uncertainty")

    evidence_request = NanoInput(
        item_id="uncertain-with-evidence",
        transcript=MIXED_STATE_TRANSCRIPT,
    )
    omitted = (
        "CC:S[chest pain]|DUR:M|SEV:U[]|"
        "MED:A[No, nothing yet.]|ALG:C[penicillin;No allergies.]"
    )
    _, omitted_diagnostics = StateSpanSolver(
        lambda transcript: omitted
    ).infer_with_state_diagnostics(evidence_request)
    assert omitted_diagnostics["fields"][2]["verified"] is False
    assert omitted_diagnostics["fields"][2]["reason"] == (
        "uncertainty_evidence_omitted"
    )


@pytest.mark.parametrize(
    "summary",
    [
        "DUR:M|CC:S[chest pain]|SEV:U[I cannot tell.]|MED:M|ALG:M",
        "CC:s[chest pain]|DUR:M|SEV:U[I cannot tell.]|MED:M|ALG:M",
        "CC:S[chest pain]|DUR:M[]|SEV:U[I cannot tell.]|MED:M|ALG:M",
        "CC:S[chest pain] |DUR:M|SEV:U[I cannot tell.]|MED:M|ALG:M",
        "CC:S[chest pain]|DUR:M|SEV:U|MED:M|ALG:M",
        "CC:S[chest pain]|DUR:M|SEV:U[I cannot tell.]|MED:M|ALG:C[penicillin]",
        "CC:S[chest pain]|DUR:M|SEV:U[I cannot tell.]|MED:M|ALG:C[penicillin;No allergies.;third]",
    ],
)
def test_malformed_grammar_fails_closed(summary):
    with pytest.raises(StateSpanFormatError):
        parse_state_span_summary(summary, MIXED_STATE_TRANSCRIPT)


@pytest.mark.parametrize(
    "summary, message",
    [
        (
            "CC:S[hallucination]|DUR:M|SEV:U[I cannot tell.]|MED:M|ALG:M",
            "not exact text",
        ),
        (
            "CC:S[Good morning]|DUR:M|SEV:U[I cannot tell.]|MED:M|ALG:M",
            "not exact text",
        ),
        (
            ("CC:S[chest pain]|DUR:M|SEV:U[I cannot tell.]|MED:S[chest pain]|ALG:M"),
            "duplicated across field",
        ),
    ],
)
def test_unlocatable_doctor_only_and_duplicate_spans_fail_closed(summary, message):
    with pytest.raises(StateSpanFormatError, match=message):
        parse_state_span_summary(summary, MIXED_STATE_TRANSCRIPT)


def test_repeated_patient_text_is_ambiguous_even_if_one_use_would_be_grounded():
    transcript = """Doctor: Good morning, what brings you in today?
Patient: pain
Patient: pain"""
    summary = "CC:S[pain]|DUR:M|SEV:M|MED:M|ALG:M"

    with pytest.raises(StateSpanFormatError, match="ambiguous"):
        parse_state_span_summary(summary, transcript)


@pytest.mark.parametrize("prediction", [None, "free-form output"])
def test_malformed_native_output_is_a_solver_exception(prediction):
    request = NanoInput(item_id="bad-output", transcript=MIXED_STATE_TRANSCRIPT)
    solver = StateSpanSolver(lambda transcript: prediction)

    result = run_inference(solver, request)

    assert not result.ok
    assert result.output is None
    assert result.diagnostics is None
    assert result.failure is not None
    assert result.failure.category is InferenceFailureCategory.SOLVER_EXCEPTION
    assert result.failure.exception_type == "StateSpanFormatError"


def test_standard_inference_does_not_claim_legacy_diagnostic_support():
    request = NanoInput(item_id="plain-runner", transcript=MIXED_STATE_TRANSCRIPT)
    solver = StateSpanSolver(lambda transcript: MIXED_STATE_SUMMARY)

    result = run_inference(solver, request)

    assert result.ok
    assert result.diagnostics is None
    assert not hasattr(solver, "infer_with_diagnostics")
