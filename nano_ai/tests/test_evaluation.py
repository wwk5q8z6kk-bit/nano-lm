from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from nano_ai.adapters import LegacySummarySolver
from nano_ai.contract import (
    FIELD_ORDER,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)
from nano_ai.evaluation import evaluate_solver
from nano_ai.fixtures import FixtureCase, load_contract_smoke
from nano_ai.solver import SolverDescriptor, SolverKind


@dataclass
class GoldReplaySolver:
    outputs: dict[str, NanoOutput]
    descriptor: SolverDescriptor = field(
        default_factory=lambda: SolverDescriptor(
            "test/gold-replay", SolverKind.REFERENCE
        )
    )

    def infer(self, request: NanoInput) -> NanoOutput:
        gold = self.outputs[request.item_id]
        return NanoOutput(
            item_id=request.item_id,
            solver_id=self.descriptor.solver_id,
            fields=gold.fields,
        )


def _cases() -> tuple[FixtureCase, ...]:
    return load_contract_smoke()


def test_perfect_report_separates_quality_and_operational_metrics() -> None:
    cases = _cases()
    solver = GoldReplaySolver({case.case_id: case.gold for case in cases})

    report = evaluate_solver(solver, cases)

    assert report.quality["exact_field_accuracy"] == 1.0
    assert report.quality["state_accuracy"] == 1.0
    assert report.quality["coverage"] == 12 / 15
    assert report.quality["intentional_abstention_rate"] == 3 / 15
    assert report.quality["assertion_grounded_rate"] == 1.0
    assert report.failures["count"] == 0
    assert report.operational["latency_measured"] is False
    assert report.operational["inference_latency_ms_p50"] is None
    assert report.quality["held_seen"] is None
    assert report.quality["paired_robustness"] is None


def test_evaluation_passes_only_nano_input_to_solver() -> None:
    cases = _cases()
    seen: list[set[str]] = []

    class InspectingSolver(GoldReplaySolver):
        def infer(self, request: NanoInput) -> NanoOutput:
            seen.append(set(request.to_dict()))
            return super().infer(request)

    solver = InspectingSolver({case.case_id: case.gold for case in cases})
    evaluate_solver(solver, cases)

    assert seen == [
        {"schema_version", "item_id", "transcript"},
        {"schema_version", "item_id", "transcript"},
        {"schema_version", "item_id", "transcript"},
    ]


def test_solver_failure_is_incorrect_but_not_a_fake_abstention() -> None:
    cases = _cases()[:1]

    class BrokenSolver:
        descriptor = SolverDescriptor("test/broken", SolverKind.TRAINED)

        def infer(self, request: NanoInput) -> NanoOutput:
            raise RuntimeError("checkpoint unavailable")

    report = evaluate_solver(BrokenSolver(), cases)

    assert report.quality["exact_field_count"] == 0
    assert report.quality["failed_inference_field_count"] == 5
    assert report.quality["intentional_abstention_count"] == 0
    assert report.failures["count"] == 1
    assert report.failures["by_category"]["solver_exception"] == 1
    assert report.failures["synthetic_abstentions_created"] == 0


def test_report_json_is_deterministic_when_latency_is_disabled() -> None:
    cases = _cases()
    solver = GoldReplaySolver({case.case_id: case.gold for case in cases})

    assert (
        evaluate_solver(solver, cases).to_json()
        == evaluate_solver(solver, cases).to_json()
    )


def test_exact_and_normalized_value_accuracy_are_not_conflated() -> None:
    case = _cases()[0]
    fields = list(case.gold.fields)
    complaint = fields[0]
    assert complaint.value is not None
    fields[0] = FieldOutput(
        field=complaint.field,
        state=complaint.state,
        value=f"the {complaint.value}",
        evidence=complaint.evidence,
    )
    prediction = NanoOutput(
        item_id=case.case_id,
        solver_id="test/gold-replay",
        fields=tuple(fields),
    )

    report = evaluate_solver(
        GoldReplaySolver({case.case_id: prediction}),
        (case,),
    )

    assert report.quality["exact_field_accuracy"] == 4 / 5
    assert report.quality["normalized_field_accuracy"] == 1.0
    complaint_result = report.items[0]["field_results"][0]
    assert complaint_result["exact"] is False
    assert complaint_result["normalized"] is True


def test_grounding_requires_the_evaluator_owned_evidence_annotation() -> None:
    transcript = "Patient: headache\nPatient: headache"
    request = NanoInput(item_id="duplicate-span", transcript=transcript)
    first_start = transcript.index("headache")
    second_start = transcript.rindex("headache")
    first = EvidenceSpan(first_start, first_start + 8, "headache")
    second = EvidenceSpan(second_start, second_start + 8, "headache")

    def output(span: EvidenceSpan, solver_id: str) -> NanoOutput:
        fields = (
            FieldOutput(
                field=FieldName.CHIEF_COMPLAINT,
                state=FieldState.SUPPORTED,
                value="headache",
                evidence=(span,),
            ),
            *(
                FieldOutput(field=field, state=FieldState.MISSING)
                for field in FIELD_ORDER[1:]
            ),
        )
        return NanoOutput(
            item_id=request.item_id,
            solver_id=solver_id,
            fields=fields,
        )

    case = FixtureCase(
        case_id=request.item_id,
        partition="test",
        request=request,
        gold=output(first, "fixture-gold-v0"),
        provenance={},
    )
    prediction = output(second, "test/gold-replay")

    report = evaluate_solver(
        GoldReplaySolver({case.case_id: prediction}),
        (case,),
    )

    assert report.quality["state_accuracy"] == 1.0
    assert report.quality["exact_field_accuracy"] == 1.0
    assert report.quality["grounded_exact_field_accuracy"] == 4 / 5
    assert report.quality["assertion_grounded_rate"] == 0.0
    assert report.quality["evidence_alignment_rate"] == 0.0
    assert report.items[0]["field_results"][0]["evidence_aligned"] is False


def test_raw_model_proposals_and_verified_outputs_are_reported_separately() -> None:
    case = _cases()[0]
    summary = (
        "CC: shortness of breath | DUR: 9 weeks | SEV: moderate | "
        "MED: none | ALG: penicillin"
    )

    report = evaluate_solver(
        LegacySummarySolver(lambda transcript: summary),
        (case,),
    )

    raw = report.pipeline["raw_proposal"]
    verification = report.pipeline["verification"]
    assert raw["available"] is True
    assert raw["fields"] == 5
    assert raw["presented_field_count"] == 5
    assert raw["strict_content_correct_count"] == 3
    assert raw["strict_false_presented_count"] == 2
    assert verification["accepted_count"] == 3
    assert verification["rejected_count"] == 2
    assert verification["decision_counts"]["rejected_ungrounded"] == 1
    assert verification["decision_counts"]["rejected_unproven_absence"] == 1
    assert report.quality["coverage"] == 3 / 5
    assert report.quality["selective_accuracy"] == 1.0
    assert report.items[0]["diagnostics"]["raw_summary"] == summary


def _solver_with_diagnostics(
    output: NanoOutput, diagnostics: dict[str, object]
) -> object:
    descriptor = SolverDescriptor(output.solver_id, SolverKind.LEGACY_ADAPTER)

    class DiagnosticSolver:
        def __init__(self) -> None:
            self.descriptor = descriptor

        def infer(self, request: NanoInput) -> NanoOutput:
            return output

        def infer_with_diagnostics(
            self, request: NanoInput
        ) -> tuple[NanoOutput, dict[str, object]]:
            return output, diagnostics

    return DiagnosticSolver()


def test_forged_raw_proposal_is_a_structured_diagnostic_failure() -> None:
    case = _cases()[0]
    summary = (
        "CC: shortness of breath | DUR: 3 weeks | SEV: moderate | "
        "MED: zinc tablets | ALG: penicillin"
    )
    output, original = LegacySummarySolver(
        lambda transcript: summary,
        solver_id="test/forged-proposal",
    ).infer_with_diagnostics(case.request)
    diagnostics = deepcopy(original)
    fields = diagnostics["fields"]
    assert isinstance(fields, list)
    fields[0]["raw_proposal"] = "fabricated complaint"

    report = evaluate_solver(_solver_with_diagnostics(output, diagnostics), (case,))

    assert report.quality["inference_success_count"] == 0
    assert report.failures["by_category"]["invalid_diagnostics"] == 1
    assert report.items[0]["failure"]["validation_code"] == (
        "legacy_diagnostics_protocol"
    )
    assert report.items[0]["output"] is None


def test_diagnostic_decision_that_contradicts_output_fails_closed() -> None:
    case = _cases()[0]
    summary = (
        "CC: shortness of breath | DUR: 3 weeks | SEV: moderate | "
        "MED: zinc tablets | ALG: penicillin"
    )
    output, original = LegacySummarySolver(
        lambda transcript: summary,
        solver_id="test/forged-decision",
    ).infer_with_diagnostics(case.request)
    diagnostics = deepcopy(original)
    fields = diagnostics["fields"]
    assert isinstance(fields, list)
    fields[0]["decision"] = "rejected_ungrounded"
    fields[0]["reason"] = "proposal_value_not_grounded"

    report = evaluate_solver(_solver_with_diagnostics(output, diagnostics), (case,))

    assert report.quality["inference_success_count"] == 0
    assert report.failures["by_category"]["invalid_diagnostics"] == 1
    assert report.items[0]["failure"]["message"] == (
        "legacy diagnostic decision contradicts final output"
    )
