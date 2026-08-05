from __future__ import annotations

from dataclasses import replace

import pytest

from nano_ai.adapters import pointer_span
from nano_ai.adapters.deterministic_v0 import _extract_fields
from nano_ai.adapters.evidence_query_pointer import (
    EVIDENCE_QUERY_POINTER_PIPELINE_VERSION,
    EvidenceQueryPointerSolver,
)
from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import (
    EvidenceSpan,
    FieldOutput,
    FieldState,
    NanoInput,
    normalize_value,
)
from nano_ai.training.evidence_query_model import (
    ARCHITECTURE_VERSION,
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
)
from nano_ai.training.state_span_data import generate_split


def _example_and_proposals():
    example = generate_split("dev", worlds=5)[0]
    proposals = parse_state_span_summary(example.target, example.transcript)
    return example, proposals


def test_h3_solver_reports_evidence_query_identity_and_parameter_count() -> None:
    example, proposals = _example_and_proposals()
    solver = EvidenceQueryPointerSolver(lambda _transcript: proposals)

    output = solver.infer(
        NanoInput(item_id=example.example_id, transcript=example.transcript)
    )

    assert output.fields == _extract_fields(
        NanoInput(item_id=example.example_id, transcript=example.transcript)
    )
    assert solver.descriptor.version == ARCHITECTURE_VERSION
    assert EVIDENCE_QUERY_POINTER_PIPELINE_VERSION == ARCHITECTURE_VERSION
    assert solver.descriptor.parameter_count == NANO_EVIDENCE_QUERY_PARAMETER_COUNT
    assert solver.descriptor.parameter_count == 3_286_469


def test_h3_verifier_rejects_but_never_repairs_model_span() -> None:
    example, proposals = _example_and_proposals()
    request = NanoInput(item_id=example.example_id, transcript=example.transcript)
    patient_prefix = "Patient: "
    start = example.transcript.index(patient_prefix) + len(patient_prefix)
    end = example.transcript.index("\n", start)
    wrong = EvidenceSpan(
        start=start,
        end=end,
        text=example.transcript[start:end],
        speaker="patient",
    )
    changed = (replace(proposals[0], spans=(wrong,)), *proposals[1:])

    output, diagnostics = EvidenceQueryPointerSolver(
        lambda _transcript: changed
    ).infer_with_diagnostics(request)

    assert output.fields[0].state is FieldState.UNCERTAIN
    assert output.fields[0].value is None
    assert output.fields[0].evidence == (wrong,)
    assert diagnostics["fields"][0]["decision"] == "rejected_to_uncertain"


def test_h3_verifier_cannot_substitute_its_supported_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example, proposals = _example_and_proposals()
    request = NanoInput(item_id=example.example_id, transcript=example.transcript)
    source = list(_extract_fields(request))
    first = source[0]
    assert first.state is FieldState.SUPPORTED
    source[0] = FieldOutput(
        field=first.field,
        state=first.state,
        value=f"the {first.value}",
        evidence=first.evidence,
    )
    monkeypatch.setattr(pointer_span, "_extract_fields", lambda _item: tuple(source))

    output = EvidenceQueryPointerSolver(lambda _transcript: proposals).infer(request)

    assert output.fields[0].value != source[0].value
    assert output.fields[0].value is not None
    assert output.fields[0].value == normalize_value(proposals[0].spans[0].text)
