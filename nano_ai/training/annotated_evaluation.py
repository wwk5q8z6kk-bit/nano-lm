"""Annotation-owned evaluation for future state/span instruments.

This is deliberately separate from consumed H6/H7 evaluators. It validates
the entire corpus and explicit field/state quotas before running a solver.
Annotation syntax and Patient evidence are checked; semantic annotation
correctness still requires independent review. Never use this as inference.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import (
    FIELD_ORDER,
    ContractValidationError,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
    normalize_value,
)
from nano_ai.evaluation import EvaluationReport, evaluate_solver
from nano_ai.fixtures import FixtureCase
from nano_ai.solver import NanoSolver, SolverDescriptor, SolverKind

if TYPE_CHECKING:
    from nano_ai.training.state_span_data import StateSpanExample

FieldStateCounts = Mapping[FieldName, Mapping[FieldState, int]]


@dataclass(frozen=True)
class EvaluatorIntegrityReport:
    """Results of the six mandatory pre-inference evaluator negative controls."""

    passed: bool
    controls: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "controls": self.controls}


class _ControlledSolver:
    """Lightweight deterministic solver wrapper for negative-control fixtures."""

    def __init__(self, solver_id: str, fn: Callable[[NanoInput], tuple[FieldOutput, ...]]) -> None:
        self.descriptor = SolverDescriptor(solver_id, SolverKind.REFERENCE)
        self._fn = fn

    def infer(self, request: NanoInput) -> NanoOutput:
        return NanoOutput(request.item_id, self.descriptor.solver_id, self._fn(request))


def annotated_cases(
    examples: Iterable[StateSpanExample], *, expected_counts: FieldStateCounts
) -> tuple[FixtureCase, ...]:
    """Bind supplied targets to source spans without interpreting questions.

    Quotas must include every field/state cell, including explicit zeros, and
    come from the instrument specification independently of this conversion.
    """
    if set(expected_counts) != set(FIELD_ORDER):
        raise ValueError("expected counts must specify all five fields")
    for counts in expected_counts.values():
        if set(counts) != set(FieldState) or any(
            type(count) is not int or count < 0 for count in counts.values()
        ):
            raise ValueError("expected counts require all states and nonnegative integers")
    observed = {field: {state: 0 for state in FieldState} for field in FIELD_ORDER}
    cases: list[FixtureCase] = []
    seen: set[str] = set()
    for example in examples:
        if example.example_id in seen:
            raise ValueError("duplicate annotation case ID")
        seen.add(example.example_id)
        request = NanoInput(example.example_id, example.transcript)
        proposals = parse_state_span_summary(example.target, example.transcript)
        if example.target_state is not None and not any(
            proposal.field is example.target_field
            and proposal.state is example.target_state
            for proposal in proposals
        ):
            raise ValueError("annotation target state disagrees with record metadata")
        fields = tuple(
            FieldOutput(
                field=proposal.field,
                state=proposal.state,
                value=(
                    normalize_value(proposal.spans[0].text)
                    if proposal.state is FieldState.SUPPORTED else None
                ),
                evidence=proposal.spans,
            )
            for proposal in proposals
        )
        gold = NanoOutput(example.example_id, "annotated-state-span-gold-v1", fields)
        gold.validate_against(request)
        for field in fields:
            observed[field.field][field.state] += 1
        cases.append(
            FixtureCase(
                case_id=example.example_id,
                partition=example.split,
                request=request,
                gold=gold,
                provenance={
                    "role": "source_annotation_transport",
                    "world_id": example.world_id,
                    "variant": example.variant,
                    "annotation_target": example.target,
                },
            )
        )
    if not cases:
        raise ValueError("annotation corpus must not be empty")
    if observed != expected_counts:
        raise ValueError(f"annotation field/state counts disagree: observed={observed!r}")
    return tuple(cases)


def evaluate_annotated(
    solver: NanoSolver,
    examples: Iterable[StateSpanExample],
    *,
    expected_counts: FieldStateCounts,
    measure_latency: bool = False,
) -> EvaluationReport:
    """Validate all gold first; the standard evaluator sends only inputs to solver."""
    cases = annotated_cases(examples, expected_counts=expected_counts)
    return evaluate_solver(solver, cases, measure_latency=measure_latency)


def _first_speaker_span(transcript: str, speaker_prefix: str) -> EvidenceSpan | None:
    offset = 0
    for line in transcript.splitlines(keepends=True):
        visible = line.rstrip("\r\n")
        if visible.lstrip().casefold().startswith(speaker_prefix):
            stripped = visible.strip()
            if stripped:
                rel = visible.find(stripped)
                return EvidenceSpan(offset + rel, offset + rel + len(stripped), stripped)
        offset += len(line)
    return None


def _raises(exc_type: type[Exception], fn: Callable[[], Any]) -> bool:
    try:
        fn()
    except exc_type:
        return True
    return False


def verify_evaluator_negative_controls(
    examples: Iterable[StateSpanExample],
    *,
    expected_counts: FieldStateCounts,
) -> EvaluatorIntegrityReport:
    """Execute the six mandatory evaluator negative controls before inference."""
    frozen_examples = tuple(examples)
    cases = annotated_cases(frozen_examples, expected_counts=expected_counts)
    gold_by_id = {case.case_id: case.gold for case in cases}

    # 1. Gold oracle must score 1.000 exactly with zero failures
    oracle_report = evaluate_solver(
        _ControlledSolver("control-oracle", lambda req: gold_by_id[req.item_id].fields), cases
    )
    oracle_ok = (
        oracle_report.failures["count"] == 0
        and oracle_report.quality["state_accuracy"] == 1.0
        and oracle_report.quality["grounded_exact_field_accuracy"] == 1.0
    )

    # 2. All-missing negative + quota mismatch guard (prevents H7 defect)
    all_missing_report = evaluate_solver(
        _ControlledSolver(
            "control-all-missing",
            lambda req: tuple(FieldOutput(f, FieldState.MISSING) for f in FIELD_ORDER),
        ),
        cases,
    )
    has_non_missing = any(
        count > 0
        for smap in expected_counts.values()
        for state, count in smap.items()
        if state is not FieldState.MISSING
    )
    mutated = {f: dict(smap) for f, smap in expected_counts.items()}
    first_field = FIELD_ORDER[0]
    nz = next(s for s, c in mutated[first_field].items() if c > 0)
    alt = next(s for s in FieldState if s is not nz)
    mutated[first_field][nz] -= 1
    mutated[first_field][alt] += 1
    quota_failed_closed = _raises(
        ValueError, lambda: annotated_cases(frozen_examples, expected_counts=mutated)
    )
    all_missing_ok = quota_failed_closed and (
        not has_non_missing
        or (
            all_missing_report.quality["grounded_exact_field_accuracy"] < 1.0
            and all_missing_report.quality["state_accuracy"] < 1.0
        )
    )

    # 3. Shuffled-labels negative (state-shift + cross-case rotation when >=2 cases)
    shuffled_report = evaluate_solver(
        _ControlledSolver(
            "control-shuffled",
            lambda req: tuple(
                FieldOutput(
                    f.field,
                    FieldState.UNCERTAIN if f.state is FieldState.MISSING else FieldState.MISSING,
                )
                for f in gold_by_id[req.item_id].fields
            ),
        ),
        cases,
    )
    rotated = {
        case.case_id: cases[(i + 1) % len(cases)].gold.fields
        for i, case in enumerate(cases)
    }
    cross_case_ok = None
    def label_key(fields: tuple[FieldOutput, ...]) -> tuple:
        return tuple((f.field, f.state, f.value, frozenset(f.evidence)) for f in fields)

    if any(label_key(rotated[c.case_id]) != label_key(c.gold.fields) for c in cases):
        cross_report = evaluate_solver(
            _ControlledSolver("control-cross-case-shuffle", lambda req: rotated[req.item_id]),
            cases,
        )
        cross_case_ok = (
            cross_report.failures["count"] > 0
            or (cross_report.quality["grounded_exact_field_accuracy"] or 0.0) < 1.0
        )
    shuffled_ok = (
        shuffled_report.quality["state_accuracy"] == 0.0
        and shuffled_report.quality["grounded_exact_field_accuracy"] == 0.0
        and cross_case_ok is not False
    )

    # 4. Wrong-patient negative (evidence inside Doctor: turn must fail)
    doctor_cases = tuple(c for c in cases if _first_speaker_span(c.request.transcript, "doctor:") is not None)
    if doctor_cases:
        def _doc_out(req: NanoInput) -> tuple[FieldOutput, ...]:
            sp = _first_speaker_span(req.transcript, "doctor:")
            assert sp is not None
            return (
                FieldOutput(FieldName.CHIEF_COMPLAINT, FieldState.SUPPORTED, value=sp.text, evidence=(sp,)),
                *(FieldOutput(f, FieldState.MISSING) for f in FIELD_ORDER[1:]),
            )
        wp_report = evaluate_solver(_ControlledSolver("control-wrong-patient", _doc_out), doctor_cases)
        wrong_patient_ok = (
            wp_report.failures["count"] == len(doctor_cases)
            and wp_report.quality["grounded_exact_field_accuracy"] == 0.0
        )
    else:
        probe_req = NanoInput("wp-probe", "Doctor: headache\nPatient: ok")
        doc_sp = _first_speaker_span(probe_req.transcript, "doctor:")
        assert doc_sp is not None
        wrong_patient_ok = _raises(ContractValidationError, lambda: doc_sp.validate_against(probe_req.transcript))

    # 5. Corrupted-offset negative (shifted span offset must fail contract validation)
    evidence_cases = tuple(c for c in cases if any(f.evidence for f in c.gold.fields))
    if evidence_cases:
        def _corrupt_out(req: NanoInput) -> tuple[FieldOutput, ...]:
            new_fields, done = [], False
            for f in gold_by_id[req.item_id].fields:
                if f.evidence and not done:
                    sp = f.evidence[0]
                    new_fields.append(FieldOutput(
                        f.field, f.state, value=f.value,
                        evidence=(EvidenceSpan(sp.start + 1, sp.end + 1, sp.text), *f.evidence[1:]),
                    ))
                    done = True
                else:
                    new_fields.append(f)
            return tuple(new_fields)
        corrupt_report = evaluate_solver(
            _ControlledSolver("control-corrupted-offset", _corrupt_out), evidence_cases
        )
        corrupted_ok = (
            corrupt_report.failures["count"] == len(evidence_cases)
            and corrupt_report.quality["grounded_exact_field_accuracy"] == 0.0
        )
    else:
        corrupted_ok = True

    # 6. Denial != missing/uncertain, conflict collapse rejected, and no-data abstain passes
    denial_contract_ok = _raises(
        ContractValidationError, lambda: FieldOutput(FieldName.MEDICATION, FieldState.ABSENT, evidence=())
    )
    conflict_collapse_ok = _raises(
        ContractValidationError,
        lambda: FieldOutput(FieldName.ALLERGY, FieldState.CONFLICTING, evidence=(EvidenceSpan(0, 1, "x"),)),
    )
    abstain_req = NanoInput("abstain-probe", "Doctor: Good morning.\nPatient: Let me hang up my coat.")
    abstain_fields = tuple(FieldOutput(f, FieldState.MISSING) for f in FIELD_ORDER)
    abstain_case = FixtureCase(
        "abstain-probe", "control", abstain_req,
        NanoOutput("abstain-probe", "annotated-state-span-gold-v1", abstain_fields),
        provenance={"role": "negative_control_abstain"},
    )
    abstain_report = evaluate_solver(
        _ControlledSolver("control-abstain", lambda req: abstain_fields),
        (abstain_case,),
    )
    abstain_ok = (
        abstain_report.failures["count"] == 0
        and abstain_report.quality["grounded_exact_field_accuracy"] == 1.0
        and abstain_report.quality["presented_field_count"] == 0
    )

    controls = {
        "gold_oracle": {
            "passed": oracle_ok,
            "state_accuracy": oracle_report.quality["state_accuracy"],
            "grounded_exact_field_accuracy": oracle_report.quality["grounded_exact_field_accuracy"],
        },
        "all_missing_negative": {
            "passed": all_missing_ok,
            "quota_mismatch_failed_closed": quota_failed_closed,
            "all_missing_grounded_exact": all_missing_report.quality["grounded_exact_field_accuracy"],
        },
        "shuffled_labels_negative": {
            "passed": shuffled_ok,
            "shuffled_grounded_exact": shuffled_report.quality["grounded_exact_field_accuracy"],
            "cross_case_ok": cross_case_ok,
        },
        "wrong_patient_negative": {
            "passed": wrong_patient_ok,
            "doctor_cases_checked": len(doctor_cases),
        },
        "corrupted_offset_negative": {
            "passed": corrupted_ok,
            "evidence_cases_checked": len(evidence_cases),
        },
        "denial_conflict_and_abstain_controls": {
            "passed": denial_contract_ok and conflict_collapse_ok and abstain_ok,
            "denial_requires_evidence": denial_contract_ok,
            "conflict_requires_two_spans": conflict_collapse_ok,
            "abstain_control_grounded_exact": abstain_report.quality["grounded_exact_field_accuracy"],
            "abstain_control_presented": abstain_report.quality["presented_field_count"],
        },
    }
    return EvaluatorIntegrityReport(
        passed=all(c["passed"] for c in controls.values()), controls=controls
    )


__all__ = [
    "EvaluatorIntegrityReport",
    "FieldStateCounts",
    "annotated_cases",
    "evaluate_annotated",
    "verify_evaluator_negative_controls",
]
