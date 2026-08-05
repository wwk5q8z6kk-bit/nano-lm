from __future__ import annotations

import pytest

from nano_ai.adapters import (
    LEGACY_DIAGNOSTICS_VERSION,
    DeterministicV0Solver,
    LegacySummaryFormatError,
    LegacySummarySolver,
)
from nano_ai.contract import FieldName, FieldState, NanoInput
from nano_ai.fixtures import load_contract_smoke
from nano_ai.solver import InferenceFailureCategory, run_inference

TRANSCRIPT = """Doctor: Good morning, what brings you in today?
Patient: I've been having a cough.
Doctor: How long has this been going on?
Patient: For about 3 days now.
Doctor: Is it mild, moderate, or severe?
Patient: Pretty mild.
Doctor: Have you taken anything for it?
Patient: No, nothing yet.
Doctor: Any allergies I should know about?
Patient: I'm allergic to penicillin."""


def _by_field(output):
    return {field.field: field for field in output.fields}


def test_deterministic_v0_binds_values_and_explicit_absence_to_patient_spans():
    item = NanoInput(item_id="case-1", transcript=TRANSCRIPT)

    output = DeterministicV0Solver().infer(item)
    fields = _by_field(output)

    assert fields[FieldName.CHIEF_COMPLAINT].state is FieldState.SUPPORTED
    assert fields[FieldName.CHIEF_COMPLAINT].value == "cough"
    assert fields[FieldName.DURATION].value == "3 days"
    assert fields[FieldName.SEVERITY].value == "mild"
    assert fields[FieldName.MEDICATION].state is FieldState.ABSENT
    assert fields[FieldName.ALLERGY].value == "penicillin"

    for field in fields.values():
        for span in field.evidence:
            assert span.speaker == "patient"
            assert TRANSCRIPT[span.start : span.end] == span.text

    denial = fields[FieldName.MEDICATION].evidence[0]
    assert denial.text == "No, nothing yet."


def test_deterministic_v0_distinguishes_missing_uncertain_and_conflicting():
    transcript = """Doctor: Good morning, what brings you in today?
Patient: I've been having chest pain.
Doctor: How long has this been going on?
Patient: I cannot remember.
Doctor: How bad would you say it is?
Patient: I'd call it moderate.
Doctor: Any allergies I should know about?
Patient: I'm allergic to penicillin.
Doctor: Are you allergic to anything?
Patient: No allergies."""
    output = DeterministicV0Solver().infer(
        NanoInput(item_id="case-2", transcript=transcript)
    )
    fields = _by_field(output)

    assert fields[FieldName.DURATION].state is FieldState.UNCERTAIN
    assert fields[FieldName.DURATION].evidence[0].text == "I cannot remember."
    assert fields[FieldName.MEDICATION].state is FieldState.MISSING
    assert fields[FieldName.MEDICATION].evidence == ()
    assert fields[FieldName.ALLERGY].state is FieldState.CONFLICTING
    assert [span.text for span in fields[FieldName.ALLERGY].evidence] == [
        "penicillin",
        "No allergies.",
    ]


def test_deterministic_v0_handles_an_empty_patient_reply_without_inventing_evidence():
    transcript = "Doctor: How long has this been going on?\nPatient:"
    output = DeterministicV0Solver().infer(
        NanoInput(item_id="case-empty", transcript=transcript)
    )

    duration = _by_field(output)[FieldName.DURATION]
    assert duration.state is FieldState.UNCERTAIN
    assert duration.evidence == ()


def test_legacy_adapter_receives_only_transcript_and_preserves_grounded_fields():
    received: list[str] = []

    def predict(transcript: str) -> str:
        received.append(transcript)
        return "CC: cough | DUR: 3 days | SEV: mild | MED: none | ALG: penicillin"

    item = NanoInput(item_id="case-3", transcript=TRANSCRIPT)
    output, diagnostics = LegacySummarySolver(predict).infer_with_diagnostics(item)

    assert received == [TRANSCRIPT]
    assert all(
        field.state in {FieldState.SUPPORTED, FieldState.ABSENT}
        for field in output.fields
    )
    assert _by_field(output)[FieldName.MEDICATION].state is FieldState.ABSENT
    assert diagnostics["protocol_version"] == LEGACY_DIAGNOSTICS_VERSION
    assert diagnostics["raw_summary"] == (
        "CC: cough | DUR: 3 days | SEV: mild | MED: none | ALG: penicillin"
    )
    traces = diagnostics["fields"]
    assert isinstance(traces, list)
    assert [trace["field"] for trace in traces] == [field.value for field in FieldName]
    assert [trace["decision"] for trace in traces] == [
        "accepted_supported",
        "accepted_supported",
        "accepted_supported",
        "accepted_absent",
        "accepted_supported",
    ]


def test_legacy_adapter_abstains_from_ungrounded_values_and_unproven_none():
    summary = "CC: rash | DUR: 9 weeks | SEV: severe | MED: aspirin | ALG: none"
    output, diagnostics = LegacySummarySolver(
        lambda transcript: summary
    ).infer_with_diagnostics(NanoInput(item_id="case-4", transcript=TRANSCRIPT))

    fields = _by_field(output)
    assert all(field.state is FieldState.UNCERTAIN for field in fields.values())
    assert fields[FieldName.ALLERGY].evidence[0].text == "penicillin"
    traces = diagnostics["fields"]
    assert isinstance(traces, list)
    assert [trace["decision"] for trace in traces] == [
        "rejected_ungrounded",
        "rejected_ungrounded",
        "rejected_ungrounded",
        "rejected_ungrounded",
        "rejected_unproven_absence",
    ]
    assert traces[-1] == {
        "field": "allergy",
        "raw_proposal": "none",
        "proposal_kind": "absence",
        "decision": "rejected_unproven_absence",
        "reason": "explicit_denial_not_verified",
    }


@pytest.mark.parametrize("prediction", ["free-form answer", None])
def test_legacy_adapter_fails_closed_on_invalid_model_output(prediction: object):
    solver = LegacySummarySolver(lambda transcript: prediction)
    request = NanoInput(item_id="case-5", transcript=TRANSCRIPT)

    with pytest.raises(LegacySummaryFormatError):
        solver.infer(request)

    result = run_inference(solver, request)
    assert result.output is None
    assert result.diagnostics is None
    assert result.failure is not None
    assert result.failure.category is InferenceFailureCategory.SOLVER_EXCEPTION
    assert result.failure.exception_type == "LegacySummaryFormatError"


def test_legacy_diagnostics_distinguish_native_abstention_and_preserved_conflict():
    transcript = """Doctor: What brings you in today?
Patient: headache
Doctor: Any allergies I should know about?
Patient: I'm allergic to penicillin.
Doctor: Are you allergic to anything?
Patient: No allergies."""
    summary = "CC: | DUR: | SEV: | MED: | ALG: penicillin"
    output, diagnostics = LegacySummarySolver(
        lambda source: summary
    ).infer_with_diagnostics(NanoInput(item_id="case-trace", transcript=transcript))

    traces = diagnostics["fields"]
    assert isinstance(traces, list)
    assert traces[0] == {
        "field": "chief_complaint",
        "raw_proposal": None,
        "proposal_kind": "missing",
        "decision": "native_abstention",
        "reason": "no_proposal",
    }
    assert traces[-1]["decision"] == "preserved_conflict"
    assert traces[-1]["reason"] == "conflicting_transcript_evidence"
    assert _by_field(output)[FieldName.ALLERGY].state is FieldState.CONFLICTING


def test_deterministic_reference_matches_frozen_supported_and_absence_smoke_cases():
    solver = DeterministicV0Solver()

    for case in load_contract_smoke()[:2]:
        output = solver.infer(case.request)
        assert [
            (field.state, field.value, field.evidence) for field in output.fields
        ] == [(field.state, field.value, field.evidence) for field in case.gold.fields]
