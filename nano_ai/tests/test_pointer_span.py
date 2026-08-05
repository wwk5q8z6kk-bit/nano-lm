from __future__ import annotations

from dataclasses import replace

import pytest

from nano_ai.adapters import pointer_span
from nano_ai.adapters.deterministic_v0 import _extract_fields
from nano_ai.adapters.pointer_span import (
    POINTER_SPAN_DIAGNOSTICS_VERSION,
    PointerSpanSolver,
    validate_pointer_proposals,
)
from nano_ai.adapters.state_span import (
    StateSpanFormatError,
    StateSpanProposal,
    parse_state_span_summary,
)
from nano_ai.contract import (
    EvidenceSpan,
    FieldOutput,
    FieldState,
    NanoInput,
    normalize_value,
)
from nano_ai.solver import run_inference
from nano_ai.training.pointer_model import NANO_POINTER_PARAMETER_COUNT
from nano_ai.training.state_span_data import generate_split


def _example_and_proposals():
    example = generate_split("dev", worlds=5)[0]
    return example, parse_state_span_summary(example.target, example.transcript)


def test_pointer_solver_accepts_exact_raw_proposals() -> None:
    example, proposals = _example_and_proposals()
    request = NanoInput(item_id=example.example_id, transcript=example.transcript)
    solver = PointerSpanSolver(lambda _transcript: proposals)

    result = run_inference(solver, request)

    assert result.ok
    assert result.output is not None
    assert result.output.fields == _extract_fields(request)
    assert result.diagnostics is not None
    assert result.diagnostics["protocol_version"] == POINTER_SPAN_DIAGNOSTICS_VERSION
    assert all(row["verified"] for row in result.diagnostics["fields"])
    assert solver.descriptor.parameter_count == NANO_POINTER_PARAMETER_COUNT


def test_unverified_pointer_is_rejected_to_uncertain() -> None:
    example, proposals = _example_and_proposals()
    request = NanoInput(item_id=example.example_id, transcript=example.transcript)
    original = proposals[0]
    patient_prefix = "Patient: "
    line_start = example.transcript.index(patient_prefix) + len(patient_prefix)
    line_end = example.transcript.index("\n", line_start)
    wrong = EvidenceSpan(
        start=line_start,
        end=line_end,
        text=example.transcript[line_start:line_end],
        speaker="patient",
    )
    changed = (
        replace(original, spans=(wrong,)),
        *proposals[1:],
    )

    output, diagnostics = PointerSpanSolver(
        lambda _transcript: changed
    ).infer_with_diagnostics(request)

    assert output.fields[0].state is FieldState.UNCERTAIN
    assert diagnostics["fields"][0]["verified"] is False
    assert diagnostics["fields"][0]["decision"] == "rejected_to_uncertain"


def test_pointer_boundary_rejects_wrong_cardinality_and_order() -> None:
    example, proposals = _example_and_proposals()
    span = proposals[0].spans[0]
    invalid_missing = replace(
        proposals[0],
        state=FieldState.MISSING,
        state_code="M",
        spans=(span,),
    )
    with pytest.raises(StateSpanFormatError, match="missing requires 0"):
        validate_pointer_proposals(
            (invalid_missing, *proposals[1:]), example.transcript
        )

    with pytest.raises(StateSpanFormatError, match="canonically ordered"):
        validate_pointer_proposals(
            (proposals[1], proposals[0], *proposals[2:]), example.transcript
        )


def test_pointer_boundary_allows_cross_field_span_reuse_for_verifier_to_reject() -> (
    None
):
    example, proposals = _example_and_proposals()
    duplicated = StateSpanProposal(
        field=proposals[1].field,
        state_code=proposals[1].state_code,
        state=proposals[1].state,
        spans=proposals[0].spans,
    )
    validated = validate_pointer_proposals(
        (proposals[0], duplicated, *proposals[2:]), example.transcript
    )

    assert validated[1].spans == validated[0].spans


def test_supported_value_is_owned_by_model_span_not_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example, proposals = _example_and_proposals()
    request = NanoInput(item_id=example.example_id, transcript=example.transcript)
    source = list(_extract_fields(request))
    first = source[0]
    assert first.state is FieldState.SUPPORTED
    assert first.value is not None
    verifier_surface = f"the {first.value}"
    source[0] = FieldOutput(
        field=first.field,
        state=first.state,
        value=verifier_surface,
        evidence=first.evidence,
    )
    monkeypatch.setattr(pointer_span, "_extract_fields", lambda _item: tuple(source))

    output = PointerSpanSolver(lambda _transcript: proposals).infer(request)

    assert output.fields[0].value == normalize_value(proposals[0].spans[0].text)
    assert output.fields[0].value != verifier_surface
