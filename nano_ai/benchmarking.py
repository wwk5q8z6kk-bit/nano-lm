"""Frozen aggregation for the fresh Nano v0 benchmark.

This module consumes evaluator output; it does not load data or run a solver.
The primary result is strict, evidence-grounded field accuracy.  Comparisons,
strata, raw proposals, verification decisions, failures, and execution facts
remain separate so no composite score can hide a failure mode.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .contract import FIELD_ORDER, FieldName, FieldState, NanoOutput
from .evaluation import (
    EVALUATION_SCHEMA_VERSION,
    LEGACY_DIAGNOSTICS_VERSION,
    EvaluationReport,
)
from .fixtures import FixtureCase

BENCHMARK_SCHEMA_VERSION = "nano.benchmark-report.v0"
BENCHMARK_CASE_SCHEMA_VERSION = "nano.benchmark-case.v0"

_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "family",
        "variant",
        "value_band",
        "target_state",
        "target_field",
        "pair_id",
    }
)
_TARGET_STATES = (
    FieldState.MISSING,
    FieldState.UNCERTAIN,
    FieldState.CONFLICTING,
)
_DIAGNOSTIC_KEYS = frozenset({"protocol_version", "raw_summary", "fields"})
_DIAGNOSTIC_FIELD_KEYS = frozenset(
    {"field", "raw_proposal", "proposal_kind", "decision", "reason"}
)
_PROPOSAL_KINDS = frozenset({"value", "absence", "missing"})
_DECISIONS = (
    "accepted_supported",
    "accepted_absent",
    "rejected_ungrounded",
    "rejected_unproven_absence",
    "preserved_conflict",
    "native_abstention",
)


class BenchmarkIntegrityError(ValueError):
    """Raised when cases, composition, or evaluator output is incoherent."""


@dataclass(frozen=True, slots=True)
class BenchmarkComposition:
    """The preregistered Nano v0 test composition."""

    normal_cases: int = 160
    state_challenge_cases: int = 60
    pairs: int = 60
    normal_per_value_band: int = 80
    challenge_per_target_state: int = 20
    challenge_per_state_field: int = 4
    challenge_per_state_field_value_band: int = 2

    @property
    def cases(self) -> int:
        return self.normal_cases + self.state_challenge_cases

    @property
    def fields(self) -> int:
        return self.cases * len(FIELD_ORDER)

    def to_dict(self) -> dict[str, int]:
        return {
            "cases": self.cases,
            "fields": self.fields,
            "normal_cases": self.normal_cases,
            "state_challenge_cases": self.state_challenge_cases,
            "pairs": self.pairs,
            "normal_per_value_band": self.normal_per_value_band,
            "challenge_per_target_state": self.challenge_per_target_state,
            "challenge_per_state_field": self.challenge_per_state_field,
            "challenge_per_state_field_value_band": (
                self.challenge_per_state_field_value_band
            ),
        }


FROZEN_BENCHMARK_COMPOSITION = BenchmarkComposition()


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """A canonical, solver-specific view of the frozen benchmark."""

    solver: dict[str, Any]
    benchmark: dict[str, Any]
    primary: dict[str, Any]
    comparisons: dict[str, Any]
    strata: dict[str, Any]
    raw_proposal: dict[str, Any]
    verification: dict[str, Any]
    failures: dict[str, Any]
    execution_provenance: dict[str, Any] | None = None
    schema_version: str = BENCHMARK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "solver": self.solver,
            "benchmark": self.benchmark,
            "primary": self.primary,
            "comparisons": self.comparisons,
            "strata": self.strata,
            "raw_proposal": self.raw_proposal,
            "verification": self.verification,
            "failures": self.failures,
            "execution_provenance": self.execution_provenance,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class _CaseMeta:
    family: str
    variant: str
    value_band: str
    target_state: FieldState | None
    target_field: FieldName | None
    pair_id: str | None


@dataclass(frozen=True, slots=True)
class _CaseResult:
    case: FixtureCase
    meta: _CaseMeta
    grounded: dict[FieldName, bool]
    failed: bool
    failure_category: str | None
    diagnostic_rows: tuple[Mapping[str, Any], ...] | None


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkIntegrityError(f"{label} must be an integer")
    return value


def _require_rate(value: object, expected: float | None, label: str) -> None:
    if expected is None:
        if value is not None:
            raise BenchmarkIntegrityError(f"{label} does not match evaluator items")
        return
    if type(value) is not float or not math.isclose(value, expected, abs_tol=1e-15):
        raise BenchmarkIntegrityError(f"{label} does not match evaluator items")


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise BenchmarkIntegrityError(
            f"{label} must be a non-empty, edge-trimmed string"
        )
    return value


def _json_copy(value: object, *, path: str = "$", depth: int = 0) -> Any:
    if depth > 64:
        raise BenchmarkIntegrityError(f"{path} is nested too deeply")
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise BenchmarkIntegrityError(f"{path} contains a non-finite float")
        return value
    if type(value) in {list, tuple}:
        return [
            _json_copy(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise BenchmarkIntegrityError(f"{path} contains a non-string key")
            copied[key] = _json_copy(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
        return copied
    raise BenchmarkIntegrityError(f"{path} is not strict JSON data")


def _parse_meta(case: FixtureCase) -> _CaseMeta:
    raw = case.provenance.get("benchmark")
    if not isinstance(raw, Mapping) or set(raw) != _PROVENANCE_KEYS:
        raise BenchmarkIntegrityError(
            f"{case.case_id}: provenance.benchmark has an invalid schema"
        )
    if raw["schema_version"] != BENCHMARK_CASE_SCHEMA_VERSION:
        raise BenchmarkIntegrityError(
            f"{case.case_id}: benchmark case schema version is not frozen v0"
        )
    family = raw["family"]
    variant = raw["variant"]
    value_band = raw["value_band"]
    if family not in {"normal", "state_challenge"}:
        raise BenchmarkIntegrityError(f"{case.case_id}: invalid benchmark family")
    expected_variant = "normal" if family == "normal" else "challenge"
    if variant != expected_variant:
        raise BenchmarkIntegrityError(
            f"{case.case_id}: family and variant are inconsistent"
        )
    if value_band not in {"seen", "held"}:
        raise BenchmarkIntegrityError(f"{case.case_id}: invalid value_band")

    raw_state = raw["target_state"]
    raw_field = raw["target_field"]
    pair_id = raw["pair_id"]
    null_count = sum(value is None for value in (raw_state, raw_field, pair_id))
    if null_count not in {0, 3}:
        raise BenchmarkIntegrityError(
            f"{case.case_id}: pair and target metadata must be all null or all set"
        )
    target_state: FieldState | None = None
    target_field: FieldName | None = None
    if null_count == 0:
        try:
            target_state = FieldState(raw_state)
            target_field = FieldName(raw_field)
        except (TypeError, ValueError) as exc:
            raise BenchmarkIntegrityError(
                f"{case.case_id}: invalid target state or field"
            ) from exc
        if target_state not in _TARGET_STATES:
            raise BenchmarkIntegrityError(
                f"{case.case_id}: target_state is not a frozen challenge state"
            )
        pair_id = _require_string(pair_id, f"{case.case_id} pair_id")

    if family == "state_challenge" and pair_id is None:
        raise BenchmarkIntegrityError(
            f"{case.case_id}: every state challenge must be paired"
        )
    if family == "state_challenge":
        assert target_field is not None and target_state is not None
        if case.gold.field(target_field).state is not target_state:
            raise BenchmarkIntegrityError(
                f"{case.case_id}: challenge gold does not match target_state"
            )
    return _CaseMeta(
        family=family,
        variant=variant,
        value_band=value_band,
        target_state=target_state,
        target_field=target_field,
        pair_id=pair_id,
    )


def _validate_composition(
    cases: tuple[FixtureCase, ...],
) -> tuple[dict[str, _CaseMeta], dict[str, tuple[str, str]]]:
    frozen = FROZEN_BENCHMARK_COMPOSITION
    if len(cases) != frozen.cases:
        raise BenchmarkIntegrityError(
            f"benchmark requires exactly {frozen.cases} cases"
        )
    if any(not isinstance(case, FixtureCase) for case in cases):
        raise BenchmarkIntegrityError("benchmark cases must be FixtureCase values")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise BenchmarkIntegrityError("benchmark case IDs must be unique")
    partitions = {case.partition for case in cases}
    if len(partitions) != 1:
        raise BenchmarkIntegrityError("benchmark cases must share one partition")

    metas = {case.case_id: _parse_meta(case) for case in cases}
    family_counts = Counter(meta.family for meta in metas.values())
    if family_counts != Counter(
        normal=frozen.normal_cases,
        state_challenge=frozen.state_challenge_cases,
    ):
        raise BenchmarkIntegrityError("benchmark family composition is not frozen v0")
    normal_band_counts = Counter(
        meta.value_band for meta in metas.values() if meta.family == "normal"
    )
    if normal_band_counts != Counter(
        seen=frozen.normal_per_value_band,
        held=frozen.normal_per_value_band,
    ):
        raise BenchmarkIntegrityError("normal seen/held composition is not balanced")

    pair_members: dict[str, list[FixtureCase]] = defaultdict(list)
    by_id = {case.case_id: case for case in cases}
    for case in cases:
        pair_id = metas[case.case_id].pair_id
        if pair_id is not None:
            pair_members[pair_id].append(case)
    if len(pair_members) != frozen.pairs:
        raise BenchmarkIntegrityError("benchmark must contain exactly 60 pairs")

    pairs: dict[str, tuple[str, str]] = {}
    for pair_id, members in pair_members.items():
        if len(members) != 2:
            raise BenchmarkIntegrityError(f"pair {pair_id} must contain two cases")
        normal = next(
            (case for case in members if metas[case.case_id].family == "normal"),
            None,
        )
        challenge = next(
            (
                case
                for case in members
                if metas[case.case_id].family == "state_challenge"
            ),
            None,
        )
        if normal is None or challenge is None:
            raise BenchmarkIntegrityError(
                f"pair {pair_id} must map normal to state_challenge"
            )
        normal_meta = metas[normal.case_id]
        challenge_meta = metas[challenge.case_id]
        if (
            normal_meta.value_band != challenge_meta.value_band
            or normal_meta.target_state is not challenge_meta.target_state
            or normal_meta.target_field is not challenge_meta.target_field
        ):
            raise BenchmarkIntegrityError(f"pair {pair_id} metadata does not match")
        assert normal_meta.target_field is not None
        if (
            normal.gold.field(normal_meta.target_field).state
            is normal_meta.target_state
        ):
            raise BenchmarkIntegrityError(
                f"pair {pair_id} normal case already has the challenge state"
            )
        for field in FIELD_ORDER:
            if field is normal_meta.target_field:
                continue
            normal_gold = normal.gold.field(field)
            challenge_gold = challenge.gold.field(field)
            if (normal_gold.state, normal_gold.value) != (
                challenge_gold.state,
                challenge_gold.value,
            ):
                raise BenchmarkIntegrityError(
                    f"pair {pair_id} changes a non-target gold field"
                )
        pairs[pair_id] = (normal.case_id, challenge.case_id)

    challenges = [meta for meta in metas.values() if meta.family == "state_challenge"]
    state_counts = Counter(meta.target_state for meta in challenges)
    if state_counts != Counter(
        {state: frozen.challenge_per_target_state for state in _TARGET_STATES}
    ):
        raise BenchmarkIntegrityError("challenge target-state balance is invalid")
    state_field_counts = Counter(
        (meta.target_state, meta.target_field) for meta in challenges
    )
    expected_state_field = Counter(
        {
            (state, field): frozen.challenge_per_state_field
            for state in _TARGET_STATES
            for field in FIELD_ORDER
        }
    )
    if state_field_counts != expected_state_field:
        raise BenchmarkIntegrityError("challenge state-by-field balance is invalid")
    state_field_band_counts = Counter(
        (meta.target_state, meta.target_field, meta.value_band) for meta in challenges
    )
    expected_state_field_band = Counter(
        {
            (state, field, band): frozen.challenge_per_state_field_value_band
            for state in _TARGET_STATES
            for field in FIELD_ORDER
            for band in ("seen", "held")
        }
    )
    if state_field_band_counts != expected_state_field_band:
        raise BenchmarkIntegrityError(
            "challenge state-by-field seen/held balance is invalid"
        )
    if len(by_id) != frozen.cases:  # pragma: no cover - guarded by uniqueness.
        raise BenchmarkIntegrityError("benchmark case map is incomplete")
    return metas, dict(sorted(pairs.items()))


def _diagnostic_rows(
    diagnostics: object,
    *,
    case_id: str,
) -> tuple[Mapping[str, Any], ...] | None:
    if diagnostics is None:
        return None
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != _DIAGNOSTIC_KEYS:
        raise BenchmarkIntegrityError(f"{case_id}: malformed diagnostics")
    if diagnostics["protocol_version"] != LEGACY_DIAGNOSTICS_VERSION:
        raise BenchmarkIntegrityError(f"{case_id}: unsupported diagnostic protocol")
    rows = diagnostics["fields"]
    if not isinstance(rows, list) or len(rows) != len(FIELD_ORDER):
        raise BenchmarkIntegrityError(f"{case_id}: diagnostics need five fields")
    validated: list[Mapping[str, Any]] = []
    for expected_field, row in zip(FIELD_ORDER, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != _DIAGNOSTIC_FIELD_KEYS:
            raise BenchmarkIntegrityError(f"{case_id}: malformed diagnostic field")
        if row["field"] != expected_field.value:
            raise BenchmarkIntegrityError(f"{case_id}: diagnostic field order changed")
        if row["proposal_kind"] not in _PROPOSAL_KINDS:
            raise BenchmarkIntegrityError(f"{case_id}: invalid proposal kind")
        if row["decision"] not in _DECISIONS:
            raise BenchmarkIntegrityError(f"{case_id}: invalid verification decision")
        validated.append(row)
    return tuple(validated)


def _parse_evaluation(
    cases: tuple[FixtureCase, ...],
    metas: dict[str, _CaseMeta],
    report: EvaluationReport,
) -> tuple[_CaseResult, ...]:
    if not isinstance(report, EvaluationReport):
        raise TypeError("evaluation must be an EvaluationReport")
    if report.schema_version != EVALUATION_SCHEMA_VERSION:
        raise BenchmarkIntegrityError("evaluation schema version is unsupported")
    if len(report.items) != len(cases):
        raise BenchmarkIntegrityError("evaluation item count does not match cases")

    parsed: list[_CaseResult] = []
    failure_categories: Counter[str] = Counter()
    for case, item in zip(cases, report.items, strict=True):
        if not isinstance(item, Mapping):
            raise BenchmarkIntegrityError("evaluation items must be objects")
        if (
            item.get("case_id") != case.case_id
            or item.get("partition") != case.partition
        ):
            raise BenchmarkIntegrityError(
                f"{case.case_id}: evaluation order or identity does not match"
            )
        status = item.get("status")
        if status == "failure":
            if (
                item.get("field_results") is not None
                or item.get("diagnostics") is not None
            ):
                raise BenchmarkIntegrityError(
                    f"{case.case_id}: failed item contains scored output"
                )
            failure = item.get("failure")
            if not isinstance(failure, Mapping):
                raise BenchmarkIntegrityError(
                    f"{case.case_id}: failed item has no structured failure"
                )
            category = _require_string(
                failure.get("category"), f"{case.case_id} failure category"
            )
            failure_categories[category] += 1
            parsed.append(
                _CaseResult(
                    case=case,
                    meta=metas[case.case_id],
                    grounded={field: False for field in FIELD_ORDER},
                    failed=True,
                    failure_category=category,
                    diagnostic_rows=None,
                )
            )
            continue
        if status != "ok" or item.get("failure") is not None:
            raise BenchmarkIntegrityError(f"{case.case_id}: invalid evaluation status")

        try:
            output = NanoOutput.from_dict(item.get("output"))
            output.validate_against(case.request)
        except (TypeError, ValueError) as exc:
            raise BenchmarkIntegrityError(
                f"{case.case_id}: evaluation output is not contract-valid"
            ) from exc
        if output.solver_id != report.solver.solver_id:
            raise BenchmarkIntegrityError(
                f"{case.case_id}: evaluation output solver identity changed"
            )
        field_rows = item.get("field_results")
        if not isinstance(field_rows, list) or len(field_rows) != len(FIELD_ORDER):
            raise BenchmarkIntegrityError(
                f"{case.case_id}: evaluation needs five field results"
            )
        grounded: dict[FieldName, bool] = {}
        for field, row in zip(FIELD_ORDER, field_rows, strict=True):
            if not isinstance(row, Mapping) or row.get("field") != field.value:
                raise BenchmarkIntegrityError(
                    f"{case.case_id}: evaluation field order changed"
                )
            gold = case.gold.field(field)
            predicted = output.field(field)
            if (
                row.get("gold_state") != gold.state.value
                or row.get("gold_value") != gold.value
            ):
                raise BenchmarkIntegrityError(
                    f"{case.case_id}: evaluator-owned truth changed"
                )
            if (
                row.get("predicted_state") != predicted.state.value
                or row.get("predicted_value") != predicted.value
                or type(row.get("grounded_exact")) is not bool
            ):
                raise BenchmarkIntegrityError(
                    f"{case.case_id}: field result contradicts evaluated output"
                )
            grounded[field] = row["grounded_exact"]
        parsed.append(
            _CaseResult(
                case=case,
                meta=metas[case.case_id],
                grounded=grounded,
                failed=False,
                failure_category=None,
                diagnostic_rows=_diagnostic_rows(
                    item.get("diagnostics"), case_id=case.case_id
                ),
            )
        )

    correct = sum(sum(result.grounded.values()) for result in parsed)
    failed = sum(result.failed for result in parsed)
    quality = report.quality
    if _require_int(quality.get("items"), "evaluation quality.items") != len(cases):
        raise BenchmarkIntegrityError("evaluation quality item count is incoherent")
    if _require_int(quality.get("fields"), "evaluation quality.fields") != len(
        cases
    ) * len(FIELD_ORDER):
        raise BenchmarkIntegrityError("evaluation quality field count is incoherent")
    if (
        _require_int(
            quality.get("grounded_exact_field_count"),
            "evaluation quality.grounded_exact_field_count",
        )
        != correct
    ):
        raise BenchmarkIntegrityError("evaluation grounded count is incoherent")
    _require_rate(
        quality.get("grounded_exact_field_accuracy"),
        _rate(correct, len(cases) * len(FIELD_ORDER)),
        "evaluation grounded accuracy",
    )
    if _require_int(report.failures.get("count"), "evaluation failure count") != failed:
        raise BenchmarkIntegrityError("evaluation failure count is incoherent")
    reported_by_category = report.failures.get("by_category")
    if not isinstance(reported_by_category, Mapping):
        raise BenchmarkIntegrityError("evaluation failure categories are missing")
    for category, count in failure_categories.items():
        if reported_by_category.get(category) != count:
            raise BenchmarkIntegrityError(
                "evaluation failure categories are incoherent"
            )
    return tuple(parsed)


def _field_metric(
    results: Iterable[_CaseResult],
    *,
    target_only: bool = False,
) -> dict[str, Any]:
    frozen = tuple(results)
    correct = 0
    fields = 0
    for result in frozen:
        selected = (
            (result.meta.target_field,)
            if target_only and result.meta.target_field is not None
            else ()
            if target_only
            else FIELD_ORDER
        )
        for field in selected:
            assert field is not None
            fields += 1
            correct += int(result.grounded[field])
    return {
        "grounded_exact_field_count": correct,
        "field_count": fields,
        "grounded_exact_field_accuracy": _rate(correct, fields),
    }


def _stratum(results: Iterable[_CaseResult]) -> dict[str, Any]:
    frozen = tuple(results)
    target_metric = _field_metric(frozen, target_only=True)
    return {
        "case_count": len(frozen),
        "failed_case_count": sum(result.failed for result in frozen),
        "all_fields": _field_metric(frozen),
        "target_field": target_metric if target_metric["field_count"] else None,
    }


def _strata(results: tuple[_CaseResult, ...]) -> dict[str, Any]:
    dimensions = {
        "family": lambda result: result.meta.family,
        "variant": lambda result: result.meta.variant,
        "value_band": lambda result: result.meta.value_band,
        "target_state": lambda result: (
            result.meta.target_state.value if result.meta.target_state else "null"
        ),
        "target_field": lambda result: (
            result.meta.target_field.value if result.meta.target_field else "null"
        ),
    }
    output: dict[str, Any] = {}
    for dimension, getter in dimensions.items():
        grouped: dict[str, list[_CaseResult]] = defaultdict(list)
        for result in results:
            grouped[getter(result)].append(result)
        output[dimension] = {key: _stratum(grouped[key]) for key in sorted(grouped)}
    return output


def _raw_correct(row: Mapping[str, Any], case: FixtureCase, field: FieldName) -> bool:
    gold = case.gold.field(field)
    kind = row["proposal_kind"]
    if kind == "absence":
        return gold.state is FieldState.ABSENT
    if kind != "value" or gold.state is not FieldState.SUPPORTED:
        return False
    return row["raw_proposal"] == gold.value


def _pipeline_metrics(
    results: tuple[_CaseResult, ...], report: EvaluationReport
) -> tuple[dict[str, Any], dict[str, Any]]:
    successful = tuple(result for result in results if not result.failed)
    diagnostic_count = sum(result.diagnostic_rows is not None for result in successful)
    if diagnostic_count not in {0, len(successful)}:
        raise BenchmarkIntegrityError(
            "successful evaluator items mix present and absent raw diagnostics"
        )
    total_fields = len(results) * len(FIELD_ORDER)
    failed_fields = sum(result.failed for result in results) * len(FIELD_ORDER)
    available = diagnostic_count > 0
    decision_counts = Counter({decision: 0 for decision in _DECISIONS})
    presented = 0
    strict_correct = 0
    native_abstentions = 0
    if available:
        for result in successful:
            assert result.diagnostic_rows is not None
            for field, row in zip(FIELD_ORDER, result.diagnostic_rows, strict=True):
                decision_counts[row["decision"]] += 1
                if row["proposal_kind"] == "missing":
                    native_abstentions += 1
                else:
                    presented += 1
                    strict_correct += int(_raw_correct(row, result.case, field))

    pipeline = report.pipeline
    raw_source = pipeline.get("raw_proposal")
    verification_source = pipeline.get("verification")
    if not isinstance(raw_source, Mapping) or not isinstance(
        verification_source, Mapping
    ):
        raise BenchmarkIntegrityError("evaluation pipeline summaries are missing")
    successful_diagnostic_fields = diagnostic_count * len(FIELD_ORDER)
    if (
        raw_source.get("available") is not available
        or raw_source.get("fields") != successful_diagnostic_fields
        or raw_source.get("presented_field_count") != presented
        or raw_source.get("strict_content_correct_count") != strict_correct
    ):
        raise BenchmarkIntegrityError("evaluation raw-proposal summary is incoherent")
    source_decisions = verification_source.get("decision_counts")
    if not isinstance(source_decisions, Mapping) or any(
        source_decisions.get(decision) != decision_counts[decision]
        for decision in _DECISIONS
    ):
        raise BenchmarkIntegrityError("evaluation verification summary is incoherent")

    if available:
        raw = {
            "available": True,
            "benchmark_field_count": total_fields,
            "strict_correct_count": strict_correct,
            "strict_accuracy_over_all_benchmark_fields": _rate(
                strict_correct, total_fields
            ),
            "presented_field_count": presented,
            "coverage_over_all_benchmark_fields": _rate(presented, total_fields),
            "native_abstention_count": native_abstentions,
            "failed_evaluation_field_count": failed_fields,
            "errors_retained_in_denominator": True,
        }
        verification = {
            "available": True,
            "benchmark_field_count": total_fields,
            "decision_count": sum(decision_counts.values()),
            "decision_counts": dict(sorted(decision_counts.items())),
            "accepted_count": (
                decision_counts["accepted_supported"]
                + decision_counts["accepted_absent"]
            ),
            "rejected_count": (
                decision_counts["rejected_ungrounded"]
                + decision_counts["rejected_unproven_absence"]
            ),
            "preserved_conflict_count": decision_counts["preserved_conflict"],
            "native_abstention_count": decision_counts["native_abstention"],
            "failed_evaluation_field_count": failed_fields,
        }
    else:
        raw = {
            "available": False,
            "benchmark_field_count": total_fields,
            "strict_correct_count": None,
            "strict_accuracy_over_all_benchmark_fields": None,
            "presented_field_count": None,
            "coverage_over_all_benchmark_fields": None,
            "native_abstention_count": None,
            "failed_evaluation_field_count": failed_fields,
            "errors_retained_in_denominator": True,
        }
        verification = {
            "available": False,
            "benchmark_field_count": total_fields,
            "decision_count": None,
            "decision_counts": None,
            "accepted_count": None,
            "rejected_count": None,
            "preserved_conflict_count": None,
            "native_abstention_count": None,
            "failed_evaluation_field_count": failed_fields,
        }
    return raw, verification


def _case_set_sha256(cases: tuple[FixtureCase, ...]) -> str:
    records = [
        {
            "case_id": case.case_id,
            "partition": case.partition,
            "request": case.request.to_dict(),
            "gold": case.gold.to_dict(),
            "provenance": _json_copy(case.provenance, path="$.provenance"),
        }
        for case in cases
    ]
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def aggregate_benchmark_report(
    cases: Iterable[FixtureCase],
    evaluation: EvaluationReport,
) -> BenchmarkReport:
    """Validate and aggregate one solver run over the sealed Nano v0 cases."""

    frozen_cases = tuple(cases)
    metas, pairs = _validate_composition(frozen_cases)
    results = _parse_evaluation(frozen_cases, metas, evaluation)
    by_id = {result.case.case_id: result for result in results}

    seen = tuple(result for result in results if result.meta.value_band == "seen")
    held = tuple(result for result in results if result.meta.value_band == "held")
    seen_metric = _field_metric(seen)
    held_metric = _field_metric(held)
    paired_normal = tuple(by_id[normal_id] for normal_id, _ in pairs.values())
    paired_challenge = tuple(by_id[challenge_id] for _, challenge_id in pairs.values())
    normal_metric = _field_metric(paired_normal)
    challenge_metric = _field_metric(paired_challenge)
    normal_target_metric = _field_metric(paired_normal, target_only=True)
    challenge_target_metric = _field_metric(paired_challenge, target_only=True)

    raw, verification = _pipeline_metrics(results, evaluation)
    failure_categories = Counter(
        result.failure_category for result in results if result.failure_category
    )
    partition = frozen_cases[0].partition
    return BenchmarkReport(
        solver=evaluation.solver.to_dict(),
        benchmark={
            "case_schema_version": BENCHMARK_CASE_SCHEMA_VERSION,
            "partition": partition,
            "case_set_sha256": _case_set_sha256(frozen_cases),
            "composition": FROZEN_BENCHMARK_COMPOSITION.to_dict(),
        },
        primary={
            "metric": "grounded_exact_field_accuracy",
            **_field_metric(results),
        },
        comparisons={
            "held_vs_seen": {
                "seen": seen_metric,
                "held": held_metric,
                "held_minus_seen_grounded_exact_accuracy": (
                    held_metric["grounded_exact_field_accuracy"]
                    - seen_metric["grounded_exact_field_accuracy"]
                ),
            },
            "paired_state_challenge": {
                "pair_count": len(pairs),
                "all_fields": {
                    "normal": normal_metric,
                    "state_challenge": challenge_metric,
                    "challenge_minus_normal_grounded_exact_accuracy": (
                        challenge_metric["grounded_exact_field_accuracy"]
                        - normal_metric["grounded_exact_field_accuracy"]
                    ),
                },
                "target_field": {
                    "normal": normal_target_metric,
                    "state_challenge": challenge_target_metric,
                    "challenge_minus_normal_grounded_exact_accuracy": (
                        challenge_target_metric["grounded_exact_field_accuracy"]
                        - normal_target_metric["grounded_exact_field_accuracy"]
                    ),
                },
            },
        },
        strata=_strata(results),
        raw_proposal=raw,
        verification=verification,
        failures={
            "case_count": sum(result.failed for result in results),
            "field_count": sum(result.failed for result in results) * len(FIELD_ORDER),
            "by_category": dict(sorted(failure_categories.items())),
        },
    )


def attach_execution_provenance(
    report: BenchmarkReport,
    *,
    resources: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> BenchmarkReport:
    """Attach runner-owned resource and environment facts without rescoring."""

    if not isinstance(report, BenchmarkReport):
        raise TypeError("report must be a BenchmarkReport")
    if not isinstance(resources, Mapping) or not isinstance(environment, Mapping):
        raise TypeError("resources and environment must be mappings")
    provenance = {
        "resources": _json_copy(resources, path="$.resources"),
        "environment": _json_copy(environment, path="$.environment"),
    }
    return replace(report, execution_provenance=provenance)


__all__ = [
    "BENCHMARK_CASE_SCHEMA_VERSION",
    "BENCHMARK_SCHEMA_VERSION",
    "FROZEN_BENCHMARK_COMPOSITION",
    "BenchmarkComposition",
    "BenchmarkIntegrityError",
    "BenchmarkReport",
    "aggregate_benchmark_report",
    "attach_execution_provenance",
]
