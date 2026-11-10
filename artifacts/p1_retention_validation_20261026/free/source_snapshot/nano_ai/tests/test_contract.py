from __future__ import annotations

import json

import pytest

from nano_ai.contract import (
    CONTRACT_VERSION,
    FIELD_ORDER,
    ContractValidationError,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)

TRANSCRIPT = """Doctor: What brings you in?
Patient: headache
Doctor: How long?
Patient: 2 days
Doctor: How severe?
Patient: mild
Doctor: Any medication?
Patient: aspirin
Doctor: Any allergies?
Patient: No allergies."""


def _span(text: str) -> EvidenceSpan:
    start = TRANSCRIPT.index(text)
    return EvidenceSpan(start=start, end=start + len(text), text=text)


def _fields() -> tuple[FieldOutput, ...]:
    values = {
        FieldName.CHIEF_COMPLAINT: "headache",
        FieldName.DURATION: "2 days",
        FieldName.SEVERITY: "mild",
        FieldName.MEDICATION: "aspirin",
    }
    fields = [
        FieldOutput(
            field=name,
            state=FieldState.SUPPORTED,
            value=value,
            evidence=(_span(value),),
        )
        for name, value in values.items()
    ]
    fields.append(
        FieldOutput(
            field=FieldName.ALLERGY,
            state=FieldState.ABSENT,
            evidence=(_span("No allergies."),),
        )
    )
    return tuple(fields)


def test_complete_output_round_trips_canonical_json_and_validates() -> None:
    request = NanoInput(item_id="case-001", transcript=TRANSCRIPT)
    output = NanoOutput(item_id=request.item_id, solver_id="test", fields=_fields())

    output.validate_against(request)
    encoded = output.to_json()

    assert NanoOutput.from_json(encoded) == output
    assert encoded == json.dumps(
        output.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert request.schema_version == CONTRACT_VERSION
    assert NanoInput.from_json(request.to_json()) == request


def test_deserialization_rejects_extra_keys_and_duplicate_json_keys() -> None:
    payload = NanoInput(item_id="case-001", transcript=TRANSCRIPT).to_dict()
    payload["gold"] = "must never reach a solver"

    with pytest.raises(ContractValidationError, match="unexpected keys"):
        NanoInput.from_dict(payload)

    with pytest.raises(ContractValidationError, match="duplicate JSON key"):
        NanoInput.from_json(
            '{"schema_version":"nano.scribe.v0","item_id":"a","item_id":"b","transcript":"x"}'
        )


def test_evidence_must_match_offsets_inside_a_patient_turn() -> None:
    request = NanoInput(item_id="case-001", transcript=TRANSCRIPT)
    mismatch = EvidenceSpan(start=0, end=8, text="headache")
    doctor_text = "What brings you in?"
    doctor_span = EvidenceSpan(
        start=TRANSCRIPT.index(doctor_text),
        end=TRANSCRIPT.index(doctor_text) + len(doctor_text),
        text=doctor_text,
    )

    with pytest.raises(ContractValidationError) as mismatch_error:
        mismatch.validate_against(request.transcript)
    assert mismatch_error.value.code == "evidence_text_mismatch"

    with pytest.raises(ContractValidationError) as speaker_error:
        doctor_span.validate_against(request.transcript)
    assert speaker_error.value.code == "evidence_not_patient"


def test_field_states_enforce_evidence_and_abstention_rules() -> None:
    with pytest.raises(ContractValidationError, match="normalize exactly"):
        FieldOutput(
            field=FieldName.CHIEF_COMPLAINT,
            state=FieldState.SUPPORTED,
            value="migraine",
            evidence=(_span("headache"),),
        )

    with pytest.raises(ContractValidationError, match="explicit denial"):
        FieldOutput(
            field=FieldName.ALLERGY,
            state=FieldState.ABSENT,
            evidence=(_span("headache"),),
        )

    uncertain_text = "Patient: I'm not sure whether I have allergies."
    uncertain_span = EvidenceSpan(
        start=len("Patient: "),
        end=len(uncertain_text),
        text="I'm not sure whether I have allergies.",
    )
    with pytest.raises(ContractValidationError, match="explicit denial"):
        FieldOutput(
            field=FieldName.ALLERGY,
            state=FieldState.ABSENT,
            evidence=(uncertain_span,),
        )

    with pytest.raises(ContractValidationError, match="cannot carry evidence"):
        FieldOutput(
            field=FieldName.MEDICATION,
            state=FieldState.MISSING,
            evidence=(_span("aspirin"),),
        )

    uncertain = FieldOutput(field=FieldName.DURATION, state=FieldState.UNCERTAIN)
    assert uncertain.abstained

    absent = FieldOutput(
        field=FieldName.ALLERGY,
        state=FieldState.ABSENT,
        evidence=(_span("No allergies."),),
    )
    assert absent.presented
    assert not absent.abstained

    with pytest.raises(ContractValidationError, match="two distinct"):
        FieldOutput(
            field=FieldName.ALLERGY,
            state=FieldState.CONFLICTING,
            evidence=(_span("No allergies."), _span("No allergies.")),
        )


def test_absence_denials_are_field_specific_and_not_ambiguous() -> None:
    transcript = """Doctor: Have you taken anything for it?
Patient: No, nothing yet.
Doctor: Any allergies I should know about?
Patient: No allergies.
Patient: No."""

    def span(text: str) -> EvidenceSpan:
        start = transcript.index(text)
        return EvidenceSpan(start=start, end=start + len(text), text=text)

    medication_denial = span("No, nothing yet.")
    allergy_denial = span("No allergies.")

    FieldOutput(
        field=FieldName.MEDICATION,
        state=FieldState.ABSENT,
        evidence=(medication_denial,),
    )
    FieldOutput(
        field=FieldName.ALLERGY,
        state=FieldState.ABSENT,
        evidence=(allergy_denial,),
    )

    with pytest.raises(ContractValidationError) as medication_error:
        FieldOutput(
            field=FieldName.MEDICATION,
            state=FieldState.ABSENT,
            evidence=(allergy_denial,),
        )
    assert medication_error.value.code == "absence_without_denial"

    with pytest.raises(ContractValidationError) as allergy_error:
        FieldOutput(
            field=FieldName.ALLERGY,
            state=FieldState.ABSENT,
            evidence=(medication_denial,),
        )
    assert allergy_error.value.code == "absence_without_denial"

    with pytest.raises(ContractValidationError) as ambiguous_error:
        FieldOutput(
            field=FieldName.ALLERGY,
            state=FieldState.ABSENT,
            evidence=(span("No."),),
        )
    assert ambiguous_error.value.code == "absence_without_denial"


def test_conflict_requires_textually_distinct_observations() -> None:
    transcript = "Patient: penicillin\nPatient: penicillin"
    first = transcript.index("penicillin")
    second = transcript.rindex("penicillin")

    with pytest.raises(ContractValidationError) as error:
        FieldOutput(
            field=FieldName.ALLERGY,
            state=FieldState.CONFLICTING,
            evidence=(
                EvidenceSpan(first, first + len("penicillin"), "penicillin"),
                EvidenceSpan(second, second + len("penicillin"), "penicillin"),
            ),
        )

    assert error.value.code == "duplicate_conflict_evidence"


def test_output_requires_exactly_five_canonical_fields_in_order() -> None:
    assert tuple(field.field for field in _fields()) == FIELD_ORDER

    with pytest.raises(ContractValidationError) as error:
        NanoOutput(
            item_id="case-001", solver_id="test", fields=tuple(reversed(_fields()))
        )

    assert error.value.code == "field_order"
