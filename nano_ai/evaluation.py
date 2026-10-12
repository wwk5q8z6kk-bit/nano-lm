"""Solver-neutral evaluation for the Nano v0 AI contract.

The evaluator owns fixture truth and passes only :class:`NanoInput` to a
solver.  Quality, failures, and operational observations remain separate; no
single utility number hides their trade-offs.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

from .contract import (
    FIELD_ORDER,
    FieldName,
    FieldOutput,
    FieldState,
    NanoOutput,
    normalize_value,
)
from .fixtures import FixtureCase
from .solver import (
    InferenceFailure,
    InferenceFailureCategory,
    NanoSolver,
    SolverDescriptor,
    run_inference,
)

EVALUATION_SCHEMA_VERSION = "nano.evaluation.v0"
LEGACY_DIAGNOSTICS_VERSION = "legacy-summary-diagnostics-v0"
_PRESENTED_STATES = frozenset({FieldState.SUPPORTED, FieldState.ABSENT})
_ABSTENTION_STATES = frozenset(
    {FieldState.MISSING, FieldState.UNCERTAIN, FieldState.CONFLICTING}
)
_EVIDENCE_REQUIRED_STATES = frozenset(
    {FieldState.SUPPORTED, FieldState.ABSENT, FieldState.CONFLICTING}
)
_LEGACY_PROPOSAL_KINDS = frozenset({"value", "absence", "missing"})
_LEGACY_DECISIONS = (
    "accepted_supported",
    "accepted_absent",
    "rejected_ungrounded",
    "rejected_unproven_absence",
    "preserved_conflict",
    "native_abstention",
)
_LEGACY_DECISIONS_BY_PROPOSAL = {
    "value": frozenset(
        {"accepted_supported", "rejected_ungrounded", "preserved_conflict"}
    ),
    "absence": frozenset(
        {"accepted_absent", "rejected_unproven_absence", "preserved_conflict"}
    ),
    "missing": frozenset({"native_abstention", "preserved_conflict"}),
}
_LEGACY_KEYS = {
    FieldName.CHIEF_COMPLAINT: "cc",
    FieldName.DURATION: "dur",
    FieldName.SEVERITY: "sev",
    FieldName.MEDICATION: "med",
    FieldName.ALLERGY: "alg",
}
_LEGACY_SUMMARY_RE = re.compile(
    r"^\s*CC:\s*(?P<cc>.*?)\s*\|\s*"
    r"DUR:\s*(?P<dur>.*?)\s*\|\s*"
    r"SEV:\s*(?P<sev>.*?)\s*\|\s*"
    r"MED:\s*(?P<med>.*?)\s*\|\s*"
    r"ALG:\s*(?P<alg>.*?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_LEGACY_REASON_BY_DECISION = {
    "accepted_supported": "verified_value_match",
    "accepted_absent": "verified_explicit_denial",
    "rejected_ungrounded": "proposal_value_not_grounded",
    "rejected_unproven_absence": "explicit_denial_not_verified",
    "preserved_conflict": "conflicting_transcript_evidence",
    "native_abstention": "no_proposal",
}
_LEGACY_OUTPUT_STATE_BY_DECISION = {
    "accepted_supported": FieldState.SUPPORTED,
    "accepted_absent": FieldState.ABSENT,
    "rejected_ungrounded": FieldState.UNCERTAIN,
    "rejected_unproven_absence": FieldState.UNCERTAIN,
    "preserved_conflict": FieldState.CONFLICTING,
    "native_abstention": FieldState.UNCERTAIN,
}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _span_key(span: Any) -> tuple[int, int, str, str]:
    return (span.start, span.end, span.text, span.speaker)


def _evidence_aligned(predicted: FieldOutput, gold: FieldOutput) -> bool:
    """Check field evidence against evaluator-owned annotations, not just offsets."""

    if predicted.state is not gold.state:
        return False
    if predicted.state not in _EVIDENCE_REQUIRED_STATES:
        return True
    predicted_spans = {_span_key(span) for span in predicted.evidence}
    gold_spans = {_span_key(span) for span in gold.evidence}
    minimum_matches = 2 if predicted.state is FieldState.CONFLICTING else 1
    return len(predicted_spans & gold_spans) >= minimum_matches


def _content_correct(
    predicted: FieldOutput,
    gold: FieldOutput,
    *,
    normalized: bool,
) -> bool:
    if predicted.state is not gold.state:
        return False
    if predicted.state is FieldState.SUPPORTED:
        assert predicted.value is not None and gold.value is not None
        if normalized:
            return normalize_value(predicted.value) == normalize_value(gold.value)
        return predicted.value == gold.value
    return True


def _grounded_field_correct(
    predicted: FieldOutput,
    gold: FieldOutput,
    *,
    normalized: bool,
) -> bool:
    return _content_correct(
        predicted, gold, normalized=normalized
    ) and _evidence_aligned(predicted, gold)


def _validate_fixture_case(case: FixtureCase) -> None:
    if not isinstance(case, FixtureCase):
        raise TypeError("cases must contain FixtureCase values")
    if not isinstance(case.case_id, str) or not case.case_id:
        raise ValueError("fixture case_id must be a non-empty string")
    if case.request.item_id != case.case_id or case.gold.item_id != case.case_id:
        raise ValueError("fixture case, request, and gold item IDs must match")
    case.gold.validate_against(case.request)


def _legacy_diagnostic_rows(
    diagnostics: Mapping[str, Any],
    output: NanoOutput,
) -> tuple[Mapping[str, Any], ...] | None:
    """Validate a known trace against both its raw summary and final output."""

    if diagnostics.get("protocol_version") != LEGACY_DIAGNOSTICS_VERSION:
        return None
    if set(diagnostics) != {"protocol_version", "raw_summary", "fields"}:
        raise ValueError("legacy diagnostics have unexpected top-level keys")
    raw_summary = diagnostics["raw_summary"]
    if not isinstance(raw_summary, str):
        raise TypeError("legacy diagnostics raw_summary must be text")
    summary_match = _LEGACY_SUMMARY_RE.fullmatch(raw_summary)
    if summary_match is None:
        raise ValueError("legacy diagnostics raw_summary violates the protocol")
    rows = diagnostics["fields"]
    if not isinstance(rows, list) or len(rows) != len(FIELD_ORDER):
        raise ValueError("legacy diagnostics must contain five ordered fields")
    expected_keys = {"field", "raw_proposal", "proposal_kind", "decision", "reason"}
    validated: list[Mapping[str, Any]] = []
    for expected_field, row in zip(FIELD_ORDER, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise ValueError("legacy diagnostic field keys do not match the protocol")
        if row["field"] != expected_field.value:
            raise ValueError("legacy diagnostic fields are not in canonical order")
        proposal_kind = row["proposal_kind"]
        decision = row["decision"]
        proposal = row["raw_proposal"]
        parsed_proposal = summary_match.group(_LEGACY_KEYS[expected_field]).strip()
        expected_proposal = parsed_proposal or None
        expected_proposal_kind = (
            "missing"
            if not parsed_proposal
            else "absence"
            if parsed_proposal.casefold() == "none"
            else "value"
        )
        if proposal != expected_proposal or proposal_kind != expected_proposal_kind:
            raise ValueError(
                "legacy diagnostic proposal does not match the raw summary"
            )
        if proposal_kind not in _LEGACY_PROPOSAL_KINDS:
            raise ValueError("legacy diagnostic proposal_kind is invalid")
        if decision not in _LEGACY_DECISIONS_BY_PROPOSAL[proposal_kind]:
            raise ValueError("legacy diagnostic decision is inconsistent")
        if proposal_kind == "missing":
            if proposal is not None:
                raise ValueError("missing legacy proposals must be null")
        elif not isinstance(proposal, str) or not proposal:
            raise ValueError("presented legacy proposals must be non-empty text")
        if row["reason"] != _LEGACY_REASON_BY_DECISION[decision]:
            raise ValueError("legacy diagnostic reason is inconsistent")
        final = output.field(expected_field)
        if final.state is not _LEGACY_OUTPUT_STATE_BY_DECISION[decision]:
            raise ValueError("legacy diagnostic decision contradicts final output")
        if decision == "accepted_supported":
            assert isinstance(proposal, str) and final.value is not None
            if _legacy_canonical_value(expected_field, proposal) != final.value:
                raise ValueError("accepted legacy proposal contradicts the final value")
        validated.append(row)
    return tuple(validated)


def _legacy_canonical_value(field: FieldName, value: str) -> str:
    canonical = " ".join(value.strip().split()).casefold()
    if field is FieldName.CHIEF_COMPLAINT:
        canonical = re.sub(r"^(?:a|an|the)\s+", "", canonical)
    return canonical


def _raw_proposal_correct(
    row: Mapping[str, Any],
    gold: FieldOutput,
    *,
    normalized: bool,
) -> bool:
    proposal_kind = row["proposal_kind"]
    if proposal_kind == "absence":
        return gold.state is FieldState.ABSENT
    if proposal_kind != "value" or gold.state is not FieldState.SUPPORTED:
        return False
    proposal = row["raw_proposal"]
    assert isinstance(proposal, str) and gold.value is not None
    if normalized:
        return normalize_value(proposal) == normalize_value(gold.value)
    return proposal == gold.value


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _empty_confusion() -> dict[str, dict[str, int]]:
    return {
        gold.value: {predicted.value: 0 for predicted in FieldState}
        for gold in FieldState
    }


@dataclass(frozen=True)
class EvaluationReport:
    """Deterministic report when timing measurement is disabled."""

    solver: SolverDescriptor
    quality: dict[str, Any]
    failures: dict[str, Any]
    operational: dict[str, Any]
    pipeline: dict[str, Any]
    items: tuple[dict[str, Any], ...]
    schema_version: str = EVALUATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "solver": self.solver.to_dict(),
            "quality": self.quality,
            "failures": self.failures,
            "operational": self.operational,
            "pipeline": self.pipeline,
            "items": list(self.items),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def evaluate_solver(
    solver: NanoSolver,
    cases: Iterable[FixtureCase],
    *,
    measure_latency: bool = False,
) -> EvaluationReport:
    """Evaluate a solver without ever placing fixture truth in its request.

    Failed inference counts as incorrect for all five fields, but remains a
    structured runner failure rather than a fabricated model abstention.
    Latency is opt-in because wall-clock values make otherwise canonical reports
    non-reproducible.
    """

    descriptor = solver.descriptor
    if not isinstance(descriptor, SolverDescriptor):
        raise TypeError("solver.descriptor must be a SolverDescriptor")
    frozen_cases = tuple(cases)
    if not frozen_cases:
        raise ValueError("evaluation requires at least one fixture case")
    for case in frozen_cases:
        _validate_fixture_case(case)
    if len({case.case_id for case in frozen_cases}) != len(frozen_cases):
        raise ValueError("fixture case IDs must be unique")

    total_fields = len(frozen_cases) * len(FIELD_ORDER)
    success_count = 0
    exact_count = 0
    normalized_count = 0
    grounded_exact_count = 0
    grounded_normalized_count = 0
    state_correct_count = 0
    presented_count = 0
    presented_content_correct_count = 0
    presented_correct_count = 0
    false_presented_count = 0
    normalized_presented_content_correct_count = 0
    normalized_presented_correct_count = 0
    intentional_abstention_count = 0
    assertion_grounded_count = 0
    evidence_required_count = 0
    evidence_aligned_count = 0
    evidence_span_count = 0
    evidence_span_valid_count = 0
    diagnostic_protocol_counts: dict[str, int] = {}
    diagnostic_item_count = 0
    raw_proposal_field_count = 0
    raw_presented_count = 0
    raw_exact_presented_correct_count = 0
    raw_normalized_presented_correct_count = 0
    raw_native_abstention_count = 0
    verification_decision_counts = {decision: 0 for decision in _LEGACY_DECISIONS}
    raw_per_field = {
        field.value: {
            "presented": 0,
            "exact_presented_correct": 0,
            "normalized_presented_correct": 0,
            "native_abstention": 0,
            "total": 0,
        }
        for field in FIELD_ORDER
    }
    per_field = {
        field.value: {
            "exact": 0,
            "normalized": 0,
            "grounded_exact": 0,
            "grounded_normalized": 0,
            "state_correct": 0,
            "evidence_aligned": 0,
            "evidence_required": 0,
            "total": len(frozen_cases),
        }
        for field in FIELD_ORDER
    }
    confusion = _empty_confusion()
    failure_counts = {category.value: 0 for category in InferenceFailureCategory}
    item_rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    for case in frozen_cases:
        started = perf_counter_ns() if measure_latency else 0
        result = run_inference(
            solver,
            case.request,
            expected_descriptor=descriptor,
        )
        if measure_latency:
            latencies_ms.append((perf_counter_ns() - started) / 1_000_000)

        if not result.ok:
            assert result.failure is not None
            failure_counts[result.failure.category.value] += 1
            item_rows.append(
                {
                    "case_id": case.case_id,
                    "partition": case.partition,
                    "status": "failure",
                    "failure": result.failure.to_dict(),
                    "output": None,
                    "diagnostics": None,
                    "field_results": None,
                }
            )
            continue

        assert result.output is not None
        output = result.output
        diagnostics = result.diagnostics
        diagnostic_rows = None
        if diagnostics is not None:
            protocol = diagnostics.get("protocol_version")
            protocol_label = protocol if isinstance(protocol, str) else "<unversioned>"
            diagnostic_protocol_counts[protocol_label] = (
                diagnostic_protocol_counts.get(protocol_label, 0) + 1
            )
            try:
                diagnostic_rows = _legacy_diagnostic_rows(diagnostics, output)
            except (TypeError, ValueError) as exc:
                failure = InferenceFailure(
                    category=InferenceFailureCategory.INVALID_DIAGNOSTICS,
                    message=str(exc),
                    exception_type=type(exc).__name__,
                    validation_code="legacy_diagnostics_protocol",
                )
                failure_counts[failure.category.value] += 1
                item_rows.append(
                    {
                        "case_id": case.case_id,
                        "partition": case.partition,
                        "status": "failure",
                        "failure": failure.to_dict(),
                        "output": None,
                        "diagnostics": None,
                        "field_results": None,
                    }
                )
                continue
        success_count += 1
        if diagnostic_rows is not None:
            diagnostic_item_count += 1
            for field_name, row in zip(FIELD_ORDER, diagnostic_rows, strict=True):
                gold = case.gold.field(field_name)
                per_field_raw = raw_per_field[field_name.value]
                per_field_raw["total"] += 1
                raw_proposal_field_count += 1
                decision = row["decision"]
                verification_decision_counts[decision] += 1
                if row["proposal_kind"] == "missing":
                    raw_native_abstention_count += 1
                    per_field_raw["native_abstention"] += 1
                    continue
                raw_presented_count += 1
                per_field_raw["presented"] += 1
                if _raw_proposal_correct(row, gold, normalized=False):
                    raw_exact_presented_correct_count += 1
                    per_field_raw["exact_presented_correct"] += 1
                if _raw_proposal_correct(row, gold, normalized=True):
                    raw_normalized_presented_correct_count += 1
                    per_field_raw["normalized_presented_correct"] += 1
        field_rows: list[dict[str, Any]] = []
        for field_name in FIELD_ORDER:
            predicted = output.field(field_name)
            gold = case.gold.field(field_name)
            exact = _content_correct(predicted, gold, normalized=False)
            normalized = _content_correct(predicted, gold, normalized=True)
            grounded_exact = _grounded_field_correct(predicted, gold, normalized=False)
            grounded_normalized = _grounded_field_correct(
                predicted, gold, normalized=True
            )
            state_correct = predicted.state is gold.state
            evidence_aligned = _evidence_aligned(predicted, gold)
            if exact:
                exact_count += 1
                per_field[field_name.value]["exact"] += 1
            if normalized:
                normalized_count += 1
                per_field[field_name.value]["normalized"] += 1
            if grounded_exact:
                grounded_exact_count += 1
                per_field[field_name.value]["grounded_exact"] += 1
            if grounded_normalized:
                grounded_normalized_count += 1
                per_field[field_name.value]["grounded_normalized"] += 1
            if state_correct:
                state_correct_count += 1
                per_field[field_name.value]["state_correct"] += 1
            confusion[gold.state.value][predicted.state.value] += 1

            is_presented = predicted.state in _PRESENTED_STATES
            if is_presented:
                presented_count += 1
                if evidence_aligned:
                    assertion_grounded_count += 1
                if exact:
                    presented_content_correct_count += 1
                if grounded_exact:
                    presented_correct_count += 1
                else:
                    false_presented_count += 1
                if normalized:
                    normalized_presented_content_correct_count += 1
                if grounded_normalized:
                    normalized_presented_correct_count += 1
            elif predicted.state in _ABSTENTION_STATES:
                intentional_abstention_count += 1

            if predicted.state in _EVIDENCE_REQUIRED_STATES:
                evidence_required_count += 1
                per_field[field_name.value]["evidence_required"] += 1
                if evidence_aligned:
                    evidence_aligned_count += 1
                    per_field[field_name.value]["evidence_aligned"] += 1

            for span in predicted.evidence:
                evidence_span_count += 1
                # run_inference has already validated every span.  Repeat the
                # independent check so this metric remains explicit.
                span.validate_against(case.request.transcript)
                evidence_span_valid_count += 1

            field_rows.append(
                {
                    "field": field_name.value,
                    "gold_state": gold.state.value,
                    "gold_value": gold.value,
                    "predicted_state": predicted.state.value,
                    "predicted_value": predicted.value,
                    "exact": exact,
                    "normalized": normalized,
                    "grounded_exact": grounded_exact,
                    "grounded_normalized": grounded_normalized,
                    "state_correct": state_correct,
                    "evidence_aligned": (
                        evidence_aligned
                        if predicted.state in _EVIDENCE_REQUIRED_STATES
                        else None
                    ),
                }
            )
        item_rows.append(
            {
                "case_id": case.case_id,
                "partition": case.partition,
                "status": "ok",
                "failure": None,
                "output": output.to_dict(),
                "diagnostics": diagnostics,
                "field_results": field_rows,
            }
        )

    for counts in per_field.values():
        counts["exact_accuracy"] = _rate(counts["exact"], counts["total"])
        counts["normalized_accuracy"] = _rate(counts["normalized"], counts["total"])
        counts["grounded_exact_accuracy"] = _rate(
            counts["grounded_exact"], counts["total"]
        )
        counts["grounded_normalized_accuracy"] = _rate(
            counts["grounded_normalized"], counts["total"]
        )
        counts["state_accuracy"] = _rate(counts["state_correct"], counts["total"])
        counts["evidence_alignment_rate"] = _rate(
            counts["evidence_aligned"], counts["evidence_required"]
        )
    for counts in raw_per_field.values():
        counts["coverage"] = _rate(counts["presented"], counts["total"])
        counts["strict_content_selective_accuracy"] = _rate(
            counts["exact_presented_correct"], counts["presented"]
        )
        counts["normalized_content_selective_accuracy"] = _rate(
            counts["normalized_presented_correct"], counts["presented"]
        )

    failed_items = len(frozen_cases) - success_count
    quality: dict[str, Any] = {
        "items": len(frozen_cases),
        "fields": total_fields,
        "inference_success_count": success_count,
        "inference_success_rate": _rate(success_count, len(frozen_cases)),
        "contract_valid_output_count": success_count,
        "contract_valid_output_rate": _rate(success_count, len(frozen_cases)),
        "exact_field_count": exact_count,
        "exact_field_accuracy": _rate(exact_count, total_fields),
        "normalized_field_count": normalized_count,
        "normalized_field_accuracy": _rate(normalized_count, total_fields),
        "grounded_exact_field_count": grounded_exact_count,
        "grounded_exact_field_accuracy": _rate(grounded_exact_count, total_fields),
        "grounded_normalized_field_count": grounded_normalized_count,
        "grounded_normalized_field_accuracy": _rate(
            grounded_normalized_count, total_fields
        ),
        "state_correct_count": state_correct_count,
        "state_accuracy": _rate(state_correct_count, total_fields),
        "presented_field_count": presented_count,
        "coverage": _rate(presented_count, total_fields),
        "presented_content_correct_count": presented_content_correct_count,
        "content_selective_accuracy": _rate(
            presented_content_correct_count, presented_count
        ),
        "presented_correct_count": presented_correct_count,
        "selective_accuracy": _rate(presented_correct_count, presented_count),
        "normalized_presented_content_correct_count": (
            normalized_presented_content_correct_count
        ),
        "normalized_content_selective_accuracy": _rate(
            normalized_presented_content_correct_count, presented_count
        ),
        "normalized_presented_correct_count": normalized_presented_correct_count,
        "normalized_selective_accuracy": _rate(
            normalized_presented_correct_count, presented_count
        ),
        "false_presented_count": false_presented_count,
        "false_presented_rate": _rate(false_presented_count, presented_count),
        "intentional_abstention_count": intentional_abstention_count,
        "intentional_abstention_rate": _rate(
            intentional_abstention_count, total_fields
        ),
        "failed_inference_field_count": failed_items * len(FIELD_ORDER),
        "assertion_grounded_count": assertion_grounded_count,
        "assertion_grounded_rate": _rate(assertion_grounded_count, presented_count),
        "evidence_required_field_count": evidence_required_count,
        "evidence_aligned_field_count": evidence_aligned_count,
        "evidence_alignment_rate": _rate(
            evidence_aligned_count, evidence_required_count
        ),
        "contract_evidence_span_count": evidence_span_count,
        "contract_evidence_span_valid_count": evidence_span_valid_count,
        "contract_evidence_span_valid_rate": _rate(
            evidence_span_valid_count, evidence_span_count
        ),
        "by_field": per_field,
        "state_confusion": confusion,
        "held_seen": None,
        "paired_robustness": None,
    }
    failures: dict[str, Any] = {
        "count": failed_items,
        "rate": _rate(failed_items, len(frozen_cases)),
        "by_category": failure_counts,
        "synthetic_abstentions_created": 0,
    }
    operational: dict[str, Any] = {
        "latency_measured": measure_latency,
        "inference_latency_ms_p50": _percentile(latencies_ms, 0.50),
        "inference_latency_ms_p95": _percentile(latencies_ms, 0.95),
        "cold_load_ms": None,
        "parameter_count": descriptor.parameter_count,
        "artifact_bytes": descriptor.artifact_bytes,
        "peak_memory_bytes": None,
    }
    raw_false_presented_count = raw_presented_count - raw_exact_presented_correct_count
    raw_normalized_false_presented_count = (
        raw_presented_count - raw_normalized_presented_correct_count
    )
    verification_accepted_count = (
        verification_decision_counts["accepted_supported"]
        + verification_decision_counts["accepted_absent"]
    )
    verification_rejected_count = (
        verification_decision_counts["rejected_ungrounded"]
        + verification_decision_counts["rejected_unproven_absence"]
    )
    pipeline: dict[str, Any] = {
        "diagnostic_protocol_counts": dict(sorted(diagnostic_protocol_counts.items())),
        "raw_proposal": {
            "available": diagnostic_item_count > 0,
            "protocol_version": (
                LEGACY_DIAGNOSTICS_VERSION if diagnostic_item_count else None
            ),
            "items": diagnostic_item_count,
            "fields": raw_proposal_field_count,
            "presented_field_count": raw_presented_count,
            "coverage": _rate(raw_presented_count, raw_proposal_field_count),
            "strict_content_correct_count": raw_exact_presented_correct_count,
            "strict_content_selective_accuracy": _rate(
                raw_exact_presented_correct_count, raw_presented_count
            ),
            "strict_false_presented_count": raw_false_presented_count,
            "strict_false_presented_rate": _rate(
                raw_false_presented_count, raw_presented_count
            ),
            "normalized_content_correct_count": (
                raw_normalized_presented_correct_count
            ),
            "normalized_content_selective_accuracy": _rate(
                raw_normalized_presented_correct_count, raw_presented_count
            ),
            "normalized_false_presented_count": raw_normalized_false_presented_count,
            "normalized_false_presented_rate": _rate(
                raw_normalized_false_presented_count, raw_presented_count
            ),
            "native_abstention_count": raw_native_abstention_count,
            "by_field": raw_per_field,
        },
        "verification": {
            "decision_counts": verification_decision_counts,
            "accepted_count": verification_accepted_count,
            "rejected_count": verification_rejected_count,
            "preserved_conflict_count": verification_decision_counts[
                "preserved_conflict"
            ],
            "native_abstention_count": verification_decision_counts[
                "native_abstention"
            ],
            "acceptance_rate_over_decisions": _rate(
                verification_accepted_count,
                raw_proposal_field_count,
            ),
        },
    }
    return EvaluationReport(
        solver=descriptor,
        quality=quality,
        failures=failures,
        operational=operational,
        pipeline=pipeline,
        items=tuple(item_rows),
    )


__all__ = ["EVALUATION_SCHEMA_VERSION", "EvaluationReport", "evaluate_solver"]
