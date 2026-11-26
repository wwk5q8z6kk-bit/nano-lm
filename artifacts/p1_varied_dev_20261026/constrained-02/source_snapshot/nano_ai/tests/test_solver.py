from __future__ import annotations

from dataclasses import dataclass, field

from nano_ai.contract import (
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)
from nano_ai.solver import (
    InferenceFailureCategory,
    SolverDescriptor,
    SolverKind,
    run_inference,
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
REQUEST = NanoInput(item_id="case-001", transcript=TRANSCRIPT)


def _span(text: str) -> EvidenceSpan:
    start = TRANSCRIPT.index(text)
    return EvidenceSpan(start=start, end=start + len(text), text=text)


def _valid_output(solver_id: str = "good") -> NanoOutput:
    return NanoOutput(
        item_id=REQUEST.item_id,
        solver_id=solver_id,
        fields=(
            FieldOutput(
                FieldName.CHIEF_COMPLAINT,
                FieldState.SUPPORTED,
                "headache",
                (_span("headache"),),
            ),
            FieldOutput(
                FieldName.DURATION,
                FieldState.SUPPORTED,
                "2 days",
                (_span("2 days"),),
            ),
            FieldOutput(
                FieldName.SEVERITY,
                FieldState.SUPPORTED,
                "mild",
                (_span("mild"),),
            ),
            FieldOutput(
                FieldName.MEDICATION,
                FieldState.SUPPORTED,
                "aspirin",
                (_span("aspirin"),),
            ),
            FieldOutput(
                FieldName.ALLERGY,
                FieldState.ABSENT,
                evidence=(_span("No allergies."),),
            ),
        ),
    )


@dataclass
class GoodSolver:
    descriptor: SolverDescriptor = field(
        default_factory=lambda: SolverDescriptor("good", SolverKind.REFERENCE)
    )

    def infer(self, request: NanoInput) -> NanoOutput:
        assert request is REQUEST
        return _valid_output()


def test_run_inference_returns_only_a_validated_output_on_success() -> None:
    result = run_inference(GoodSolver(), REQUEST)

    assert result.ok
    assert result.output == _valid_output()
    assert result.failure is None
    assert result.diagnostics is None


def test_optional_diagnostic_inference_is_validated_copied_and_returned() -> None:
    raw_diagnostics = {"protocol_version": "test-v0", "events": ["accepted"]}

    class DiagnosticSolver(GoodSolver):
        def infer(self, request: NanoInput) -> NanoOutput:
            raise AssertionError(
                "diagnostic method should be the single inference call"
            )

        def infer_with_diagnostics(
            self, request: NanoInput
        ) -> tuple[NanoOutput, dict[str, object]]:
            assert request is REQUEST
            return _valid_output(), raw_diagnostics

    result = run_inference(DiagnosticSolver(), REQUEST)
    raw_diagnostics["events"].append("mutated")

    assert result.ok
    assert result.output == _valid_output()
    assert result.failure is None
    assert result.diagnostics == {
        "protocol_version": "test-v0",
        "events": ["accepted"],
    }


def test_invalid_diagnostic_protocol_fails_without_synthesizing_output() -> None:
    class WrongShapeSolver(GoodSolver):
        def infer_with_diagnostics(self, request: NanoInput) -> object:
            return [_valid_output(), {}]

    wrong_shape = run_inference(WrongShapeSolver(), REQUEST)
    assert wrong_shape.output is None
    assert wrong_shape.diagnostics is None
    assert wrong_shape.failure is not None
    assert wrong_shape.failure.category is InferenceFailureCategory.INVALID_DIAGNOSTICS

    class NonMappingDiagnosticsSolver(GoodSolver):
        def infer_with_diagnostics(
            self, request: NanoInput
        ) -> tuple[NanoOutput, object]:
            return _valid_output(), ["not", "a", "mapping"]

    non_mapping = run_inference(NonMappingDiagnosticsSolver(), REQUEST)
    assert non_mapping.output is None
    assert non_mapping.diagnostics is None
    assert non_mapping.failure is not None
    assert non_mapping.failure.category is InferenceFailureCategory.INVALID_DIAGNOSTICS


def test_non_json_diagnostics_fail_closed() -> None:
    class NonJsonDiagnosticsSolver(GoodSolver):
        def infer_with_diagnostics(
            self, request: NanoInput
        ) -> tuple[NanoOutput, dict[str, object]]:
            return _valid_output(), {
                "non_string_key": {1: "not valid JSON object semantics"},
                "non_finite": float("nan"),
            }

    result = run_inference(NonJsonDiagnosticsSolver(), REQUEST)

    assert result.output is None
    assert result.diagnostics is None
    assert result.failure is not None
    assert result.failure.category is InferenceFailureCategory.INVALID_DIAGNOSTICS


def test_descriptor_is_pinned_before_and_during_inference() -> None:
    original = SolverDescriptor("good", SolverKind.REFERENCE)
    changed = SolverDescriptor("changed", SolverKind.REFERENCE)

    class DriftSolver(GoodSolver):
        def __init__(self) -> None:
            self.read_count = 0

        @property
        def descriptor(self) -> SolverDescriptor:
            self.read_count += 1
            return original if self.read_count == 1 else changed

    solver = DriftSolver()
    result = run_inference(solver, REQUEST, expected_descriptor=original)

    assert result.output is None
    assert result.diagnostics is None
    assert result.failure is not None
    assert result.failure.category is InferenceFailureCategory.INVALID_SOLVER
    assert result.failure.message == "solver descriptor changed during inference"


def test_solver_exception_is_a_failure_not_a_synthetic_abstention() -> None:
    class BrokenSolver(GoodSolver):
        def infer(self, request: NanoInput) -> NanoOutput:
            raise RuntimeError("checkpoint unavailable")

    result = run_inference(BrokenSolver(), REQUEST)

    assert not result.ok
    assert result.output is None
    assert result.failure is not None
    assert result.failure.category is InferenceFailureCategory.SOLVER_EXCEPTION
    assert result.failure.exception_type == "RuntimeError"


def test_wrong_output_type_and_solver_identity_fail_closed() -> None:
    class WrongTypeSolver(GoodSolver):
        def infer(self, request: NanoInput) -> NanoOutput:
            return {"fields": []}  # type: ignore[return-value]

    wrong_type = run_inference(WrongTypeSolver(), REQUEST)
    assert wrong_type.failure is not None
    assert wrong_type.failure.category is InferenceFailureCategory.INVALID_OUTPUT_TYPE
    assert wrong_type.output is None

    class WrongIdentitySolver(GoodSolver):
        def infer(self, request: NanoInput) -> NanoOutput:
            return _valid_output("someone-else")

    wrong_identity = run_inference(WrongIdentitySolver(), REQUEST)
    assert wrong_identity.failure is not None
    assert (
        wrong_identity.failure.category is InferenceFailureCategory.SOLVER_ID_MISMATCH
    )
    assert wrong_identity.output is None


def test_broken_solver_boundary_is_structured() -> None:
    class BrokenBoundary:
        descriptor = SolverDescriptor("broken", SolverKind.REFERENCE)

        @property
        def infer(self) -> object:
            raise LookupError("method unavailable")

    result = run_inference(BrokenBoundary(), REQUEST)  # type: ignore[arg-type]

    assert result.output is None
    assert result.failure is not None
    assert result.failure.category is InferenceFailureCategory.INVALID_SOLVER
    assert result.failure.exception_type == "LookupError"


def test_transcript_grounding_violation_is_structured() -> None:
    bad_span = EvidenceSpan(start=0, end=len("headache"), text="headache")

    class UngroundedSolver(GoodSolver):
        def infer(self, request: NanoInput) -> NanoOutput:
            fields = list(_valid_output().fields)
            fields[0] = FieldOutput(
                FieldName.CHIEF_COMPLAINT,
                FieldState.SUPPORTED,
                "headache",
                (bad_span,),
            )
            return NanoOutput(request.item_id, "good", tuple(fields))

    result = run_inference(UngroundedSolver(), REQUEST)

    assert result.output is None
    assert result.failure is not None
    assert result.failure.category is InferenceFailureCategory.CONTRACT_VIOLATION
    assert result.failure.validation_code == "evidence_text_mismatch"


def test_contract_error_while_building_output_is_not_a_solver_exception() -> None:
    class MalformedSolver(GoodSolver):
        def infer(self, request: NanoInput) -> NanoOutput:
            return NanoOutput(
                request.item_id, "good", tuple(reversed(_valid_output().fields))
            )

    result = run_inference(MalformedSolver(), REQUEST)

    assert result.output is None
    assert result.failure is not None
    assert result.failure.category is InferenceFailureCategory.CONTRACT_VIOLATION
    assert result.failure.validation_code == "field_order"
