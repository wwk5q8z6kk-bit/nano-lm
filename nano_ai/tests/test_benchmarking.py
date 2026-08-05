from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import Any

import pytest

from nano_ai.benchmarking import (
    BENCHMARK_CASE_SCHEMA_VERSION,
    BenchmarkIntegrityError,
    aggregate_benchmark_report,
    attach_execution_provenance,
)
from nano_ai.contract import (
    FIELD_ORDER,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    NanoInput,
    NanoOutput,
)
from nano_ai.evaluation import EvaluationReport, evaluate_solver
from nano_ai.fixtures import FixtureCase
from nano_ai.solver import SolverDescriptor, SolverKind

_TARGET_STATES = (
    FieldState.MISSING,
    FieldState.UNCERTAIN,
    FieldState.CONFLICTING,
)


def _values(seed: str) -> dict[FieldName, str]:
    return {field: f"{field.value}-{seed}" for field in FIELD_ORDER}


def _case(
    case_id: str,
    *,
    family: str,
    value_band: str,
    values: dict[FieldName, str],
    pair_id: str | None = None,
    target_state: FieldState | None = None,
    target_field: FieldName | None = None,
) -> FixtureCase:
    lines: list[str] = []
    field_texts: dict[FieldName, tuple[str, ...]] = {}
    for field in FIELD_ORDER:
        if family == "state_challenge" and field is target_field:
            if target_state in {FieldState.MISSING, FieldState.UNCERTAIN}:
                field_texts[field] = ()
                continue
            assert target_state is FieldState.CONFLICTING
            field_texts[field] = (
                f"{field.value}-old-{pair_id}",
                f"{field.value}-new-{pair_id}",
            )
        else:
            field_texts[field] = (values[field],)
        lines.extend(f"Patient: {text}" for text in field_texts[field])
    transcript = "\n".join(lines)
    request = NanoInput(item_id=case_id, transcript=transcript)

    outputs: list[FieldOutput] = []
    for field in FIELD_ORDER:
        texts = field_texts[field]
        evidence = tuple(
            EvidenceSpan(
                start=transcript.index(text),
                end=transcript.index(text) + len(text),
                text=text,
            )
            for text in texts
        )
        if family == "state_challenge" and field is target_field:
            assert target_state is not None
            outputs.append(
                FieldOutput(field=field, state=target_state, evidence=evidence)
            )
        else:
            outputs.append(
                FieldOutput(
                    field=field,
                    state=FieldState.SUPPORTED,
                    value=values[field],
                    evidence=evidence,
                )
            )
    gold = NanoOutput(
        item_id=case_id,
        solver_id="fixture-gold-v0",
        fields=tuple(outputs),
    )
    benchmark_provenance = {
        "schema_version": BENCHMARK_CASE_SCHEMA_VERSION,
        "family": family,
        "variant": "normal" if family == "normal" else "challenge",
        "value_band": value_band,
        "target_state": target_state.value if target_state else None,
        "target_field": target_field.value if target_field else None,
        "pair_id": pair_id,
    }
    return FixtureCase(
        case_id=case_id,
        partition="synthetic-fresh-v0",
        request=request,
        gold=gold,
        provenance={"benchmark": benchmark_provenance},
    )


def _frozen_cases() -> tuple[FixtureCase, ...]:
    paired_normals: list[FixtureCase] = []
    challenges: list[FixtureCase] = []
    pair_index = 0
    for state in _TARGET_STATES:
        for target_field in FIELD_ORDER:
            for value_band in ("seen", "held"):
                for repetition in range(2):
                    pair_id = f"pair-{pair_index:03d}"
                    values = _values(pair_id)
                    paired_normals.append(
                        _case(
                            f"normal-{pair_id}",
                            family="normal",
                            value_band=value_band,
                            values=values,
                            pair_id=pair_id,
                            target_state=state,
                            target_field=target_field,
                        )
                    )
                    challenges.append(
                        _case(
                            f"challenge-{pair_id}",
                            family="state_challenge",
                            value_band=value_band,
                            values=values,
                            pair_id=pair_id,
                            target_state=state,
                            target_field=target_field,
                        )
                    )
                    pair_index += 1

    unpaired = [
        _case(
            f"normal-unpaired-{band}-{index:03d}",
            family="normal",
            value_band=band,
            values=_values(f"unpaired-{band}-{index:03d}"),
        )
        for band in ("seen", "held")
        for index in range(50)
    ]
    cases = tuple(paired_normals + unpaired + challenges)
    assert len(cases) == 220
    return cases


@pytest.fixture(scope="module")
def cases() -> tuple[FixtureCase, ...]:
    return _frozen_cases()


@dataclass
class _GoldReplay:
    cases: dict[str, FixtureCase]
    descriptor: SolverDescriptor = dataclass_field(
        default_factory=lambda: SolverDescriptor(
            "test/benchmark-gold", SolverKind.REFERENCE
        )
    )

    def infer(self, request: NanoInput) -> NanoOutput:
        gold = self.cases[request.item_id].gold
        return NanoOutput(
            item_id=request.item_id,
            solver_id=self.descriptor.solver_id,
            fields=gold.fields,
        )


@dataclass
class _DiagnosticReplay:
    cases: dict[str, FixtureCase]
    fail_case_id: str | None = None
    descriptor: SolverDescriptor = dataclass_field(
        default_factory=lambda: SolverDescriptor(
            "test/benchmark-diagnostic", SolverKind.LEGACY_ADAPTER
        )
    )

    def infer(self, request: NanoInput) -> NanoOutput:
        output, _ = self.infer_with_diagnostics(request)
        return output

    def infer_with_diagnostics(
        self, request: NanoInput
    ) -> tuple[NanoOutput, dict[str, Any]]:
        if request.item_id == self.fail_case_id:
            raise RuntimeError("synthetic parse failure")
        gold = self.cases[request.item_id].gold
        fields: list[FieldOutput] = []
        diagnostic_fields: list[dict[str, Any]] = []
        summary_values: list[str] = []
        for gold_field in gold.fields:
            if gold_field.state is FieldState.SUPPORTED:
                proposal = gold_field.value
                proposal_kind = "value"
                decision = "accepted_supported"
                reason = "verified_value_match"
                output_field = gold_field
            elif gold_field.state in {FieldState.MISSING, FieldState.UNCERTAIN}:
                proposal = None
                proposal_kind = "missing"
                decision = "native_abstention"
                reason = "no_proposal"
                output_field = FieldOutput(
                    field=gold_field.field,
                    state=FieldState.UNCERTAIN,
                )
            else:
                assert gold_field.state is FieldState.CONFLICTING
                proposal = gold_field.evidence[0].text
                proposal_kind = "value"
                decision = "preserved_conflict"
                reason = "conflicting_transcript_evidence"
                output_field = gold_field
            summary_values.append(proposal or "")
            diagnostic_fields.append(
                {
                    "field": gold_field.field.value,
                    "raw_proposal": proposal,
                    "proposal_kind": proposal_kind,
                    "decision": decision,
                    "reason": reason,
                }
            )
            fields.append(output_field)
        raw_summary = (
            f"CC: {summary_values[0]} | DUR: {summary_values[1]} | "
            f"SEV: {summary_values[2]} | MED: {summary_values[3]} | "
            f"ALG: {summary_values[4]}"
        )
        return (
            NanoOutput(
                item_id=request.item_id,
                solver_id=self.descriptor.solver_id,
                fields=tuple(fields),
            ),
            {
                "protocol_version": "legacy-summary-diagnostics-v0",
                "raw_summary": raw_summary,
                "fields": diagnostic_fields,
            },
        )


def _evaluate(cases: tuple[FixtureCase, ...]) -> EvaluationReport:
    return evaluate_solver(_GoldReplay({case.case_id: case for case in cases}), cases)


def test_aggregate_freezes_primary_comparisons_strata_and_canonical_json(
    cases: tuple[FixtureCase, ...],
) -> None:
    evaluation = _evaluate(cases)
    first = aggregate_benchmark_report(cases, evaluation)
    second = aggregate_benchmark_report(cases, evaluation)

    assert first.primary == {
        "metric": "grounded_exact_field_accuracy",
        "grounded_exact_field_count": 1100,
        "field_count": 1100,
        "grounded_exact_field_accuracy": 1.0,
    }
    assert first.benchmark["composition"]["normal_cases"] == 160
    assert first.benchmark["composition"]["state_challenge_cases"] == 60
    assert (
        first.comparisons["held_vs_seen"]["held_minus_seen_grounded_exact_accuracy"]
        == 0.0
    )
    paired = first.comparisons["paired_state_challenge"]
    assert paired["pair_count"] == 60
    assert paired["all_fields"]["challenge_minus_normal_grounded_exact_accuracy"] == 0.0
    assert first.strata["family"]["normal"]["case_count"] == 160
    assert first.strata["target_state"]["missing"]["case_count"] == 40
    assert first.strata["target_state"]["null"]["case_count"] == 100
    assert first.raw_proposal["available"] is False
    assert first.execution_provenance is None
    assert "utility" not in first.to_json()
    assert first.to_json() == second.to_json()


def test_raw_and_verification_metrics_retain_failures_in_full_denominator(
    cases: tuple[FixtureCase, ...],
) -> None:
    fail_case = next(
        case
        for case in cases
        if case.provenance["benchmark"]["pair_id"] is None
        and case.provenance["benchmark"]["value_band"] == "seen"
    )
    solver = _DiagnosticReplay(
        {case.case_id: case for case in cases}, fail_case_id=fail_case.case_id
    )
    aggregate = aggregate_benchmark_report(cases, evaluate_solver(solver, cases))

    assert aggregate.primary["grounded_exact_field_count"] == 1075
    assert aggregate.raw_proposal == {
        "available": True,
        "benchmark_field_count": 1100,
        "strict_correct_count": 1035,
        "strict_accuracy_over_all_benchmark_fields": 1035 / 1100,
        "presented_field_count": 1055,
        "coverage_over_all_benchmark_fields": 1055 / 1100,
        "native_abstention_count": 40,
        "failed_evaluation_field_count": 5,
        "errors_retained_in_denominator": True,
    }
    assert aggregate.verification["decision_count"] == 1095
    assert aggregate.verification["accepted_count"] == 1035
    assert aggregate.verification["preserved_conflict_count"] == 20
    assert aggregate.verification["native_abstention_count"] == 40
    assert aggregate.verification["failed_evaluation_field_count"] == 5
    paired = aggregate.comparisons["paired_state_challenge"]
    assert paired["all_fields"][
        "challenge_minus_normal_grounded_exact_accuracy"
    ] == pytest.approx(-1 / 15)
    assert paired["target_field"][
        "challenge_minus_normal_grounded_exact_accuracy"
    ] == pytest.approx(-1 / 3)
    assert aggregate.failures == {
        "case_count": 1,
        "field_count": 5,
        "by_category": {"solver_exception": 1},
    }


def test_composition_and_evaluation_mismatches_fail_closed(
    cases: tuple[FixtureCase, ...],
) -> None:
    evaluation = _evaluate(cases)
    challenge_index = next(
        index
        for index, case in enumerate(cases)
        if case.provenance["benchmark"]["family"] == "state_challenge"
    )
    challenge = cases[challenge_index]
    benchmark = dict(challenge.provenance["benchmark"])
    benchmark["pair_id"] = "unmatched-pair"
    bad_case = replace(challenge, provenance={"benchmark": benchmark})
    bad_cases = list(cases)
    bad_cases[challenge_index] = bad_case

    with pytest.raises(BenchmarkIntegrityError, match="60 pairs|must contain two"):
        aggregate_benchmark_report(bad_cases, evaluation)

    items = list(evaluation.items)
    first_item = dict(items[0])
    rows = [dict(row) for row in first_item["field_results"]]
    rows[0]["grounded_exact"] = False
    first_item["field_results"] = rows
    items[0] = first_item
    forged = replace(evaluation, items=tuple(items))
    with pytest.raises(BenchmarkIntegrityError, match="grounded count"):
        aggregate_benchmark_report(cases, forged)


def test_execution_provenance_is_attached_without_rescoring(
    cases: tuple[FixtureCase, ...],
) -> None:
    aggregate = aggregate_benchmark_report(cases, _evaluate(cases))
    enriched = attach_execution_provenance(
        aggregate,
        resources={"checkpoint_sha256": "a" * 64, "parameter_count": 3_148_608},
        environment={"device": "cpu", "latency_ms": 12.5},
    )

    assert aggregate.execution_provenance is None
    assert enriched.primary == aggregate.primary
    assert enriched.execution_provenance == {
        "resources": {
            "checkpoint_sha256": "a" * 64,
            "parameter_count": 3_148_608,
        },
        "environment": {"device": "cpu", "latency_ms": 12.5},
    }
    with pytest.raises(BenchmarkIntegrityError, match="non-finite"):
        attach_execution_provenance(
            aggregate,
            resources={},
            environment={"peak_memory_bytes": float("nan")},
        )
