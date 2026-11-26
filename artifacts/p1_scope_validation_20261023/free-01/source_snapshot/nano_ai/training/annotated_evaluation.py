"""Annotation-owned evaluation for future state/span instruments.

This is deliberately separate from consumed H6/H7 evaluators. It validates
the entire corpus and explicit field/state quotas before running a solver.
Annotation syntax and Patient evidence are checked; semantic annotation
correctness still requires independent review. Never use this as inference.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import (
    FIELD_ORDER,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
    normalize_value,
)
from nano_ai.evaluation import EvaluationReport, evaluate_solver
from nano_ai.fixtures import FixtureCase
from nano_ai.solver import NanoSolver

if TYPE_CHECKING:
    from nano_ai.training.state_span_data import StateSpanExample

FieldStateCounts = Mapping[FieldName, Mapping[FieldState, int]]


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
