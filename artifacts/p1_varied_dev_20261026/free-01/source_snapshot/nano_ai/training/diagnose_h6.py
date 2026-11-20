"""Deterministic row-level diagnosis for Nano's terminal H6 evaluation.

This module reads the already-consumed H6 development result.  It never loads a
model, opens another dataset, or reruns evaluation.  The terminal decision hash
binds the input before the row-level state and evidence-binding errors are
decomposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nano_ai.contract import FIELD_ORDER, FieldState

DIAGNOSIS_SCHEMA_VERSION = "nano.h6-mechanism-diagnosis.v2"
H6_EVALUATION_SCHEMA_VERSION = "nano.evidence-query-h6-development-evaluation.v1"
H6_TERMINAL_SCHEMA_VERSION = "nano.h6.terminal-decision.v1"
CANONICAL_H6_EVALUATION_SHA256 = (
    "cb612410bd9530062147c1ed36bdcd606f9292a77b750376baf7a9860639e11a"
)
CANONICAL_H6_TERMINAL_DECISION_SHA256 = (
    "297a7b7e914bd9be938479a726a8e690c56f1946a62bd02e6806ed243d3471e7"
)

ERROR_AXES = (
    "state_only_error",
    "binding_only_error",
    "combined_error",
)
SPAN_RELATIONS = (
    "raw_contains_gold",
    "gold_contains_raw",
    "partial_overlap",
    "disjoint",
    "speaker_mismatch",
    "count_mismatch",
)
_VALID_STATES = frozenset(state.value for state in FieldState)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_PATH = (
    _REPOSITORY_ROOT
    / "artifacts/nano_h6/kaggle/eval-20260805/results/development_evaluation.json"
)
DEFAULT_TERMINAL_DECISION_PATH = (
    _REPOSITORY_ROOT / "artifacts/nano_h6/H6_TERMINAL_DECISION.json"
)


class H6DiagnosisError(ValueError):
    """Raised when terminal H6 evidence is missing, mutated, or incoherent."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, role: str) -> tuple[Mapping[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise H6DiagnosisError(f"cannot read {role}: {path}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise H6DiagnosisError(f"{role} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise H6DiagnosisError(f"{role} must be a JSON object")
    return value, _sha256(payload)


def _require_mapping(value: object, *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise H6DiagnosisError(f"{role} must be an object")
    return value


def _require_bool(value: object, *, role: str) -> bool:
    if not isinstance(value, bool):
        raise H6DiagnosisError(f"{role} must be a boolean")
    return value


def _classify_field(field: Mapping[str, Any], *, role: str) -> str:
    state_correct = _require_bool(field.get("state_correct"), role=f"{role}.state_correct")
    span_exact = _require_bool(field.get("span_exact"), role=f"{role}.span_exact")
    joint_exact = _require_bool(field.get("joint_exact"), role=f"{role}.joint_exact")
    if joint_exact != (state_correct and span_exact):
        raise H6DiagnosisError(f"{role} has incoherent joint_exact")
    if joint_exact:
        return "joint_correct"
    if not state_correct and span_exact:
        return "state_only_error"
    if state_correct and not span_exact:
        return "binding_only_error"
    return "combined_error"


def _validated_spans(value: object, *, role: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise H6DiagnosisError(f"{role} must be an array")
    validated: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        span = _require_mapping(item, role=f"{role}[{index}]")
        start = span.get("start")
        end = span.get("end")
        text = span.get("text")
        speaker = span.get("speaker")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or not isinstance(text, str)
            or not text
            or not isinstance(speaker, str)
            or not speaker
        ):
            raise H6DiagnosisError(f"{role}[{index}] is not a valid evidence span")
        validated.append(span)
    return tuple(validated)


def _paired_span_relation(raw: Mapping[str, Any], gold: Mapping[str, Any]) -> str:
    if raw == gold:
        return "exact"
    if raw["speaker"] != gold["speaker"]:
        return "speaker_mismatch"
    if raw["start"] <= gold["start"] and raw["end"] >= gold["end"]:
        return "raw_contains_gold"
    if gold["start"] <= raw["start"] and gold["end"] >= raw["end"]:
        return "gold_contains_raw"
    if max(raw["start"], gold["start"]) < min(raw["end"], gold["end"]):
        return "partial_overlap"
    return "disjoint"


def _span_identity(span: Mapping[str, Any]) -> tuple[int, int, str, str]:
    return (span["start"], span["end"], span["text"], span["speaker"])


def _span_relation(field: Mapping[str, Any], *, role: str) -> str | None:
    """Classify one non-exact field without repairing or rescoring it.

    Multiple gold spans are ordered by the frozen evaluator.  When cardinality
    matches, the same positional pairing therefore preserves its semantics.
    Mixed signatures stay explicit instead of being collapsed into a favorable
    bucket.
    """

    raw = _validated_spans(field.get("raw_spans"), role=f"{role}.raw_spans")
    gold = _validated_spans(field.get("gold_spans"), role=f"{role}.gold_spans")
    raw = tuple(sorted(raw, key=_span_identity))
    gold = tuple(sorted(gold, key=_span_identity))
    if tuple(map(_span_identity, raw)) == tuple(map(_span_identity, gold)):
        return None
    if len(raw) != len(gold):
        return "count_mismatch"
    relations = tuple(
        _paired_span_relation(raw_span, gold_span)
        for raw_span, gold_span in zip(raw, gold, strict=True)
    )
    if len(set(relations)) == 1:
        return relations[0]
    return "mixed:" + "+".join(sorted(set(relations)))


def _summary(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    if total < 1:
        raise H6DiagnosisError("diagnostic bucket cannot be empty")
    errors = {axis: counter[axis] for axis in ERROR_AXES}
    error_total = sum(errors.values())
    dominant = None
    if error_total:
        dominant = max(ERROR_AXES, key=lambda axis: (errors[axis], -ERROR_AXES.index(axis)))
    return {
        "total": total,
        "joint_correct": counter["joint_correct"],
        "joint_accuracy": counter["joint_correct"] / total,
        "error_total": error_total,
        "errors": errors,
        "dominant_error_axis": dominant,
        "dominant_error_share": (
            errors[dominant] / error_total if dominant is not None else None
        ),
    }


def diagnose_h6(
    evaluation: Mapping[str, Any],
    *,
    evaluation_sha256: str,
    terminal_decision: Mapping[str, Any],
    terminal_decision_sha256: str,
) -> dict[str, Any]:
    """Return a deterministic mechanism decomposition of authenticated H6 rows."""

    if evaluation.get("schema_version") != H6_EVALUATION_SCHEMA_VERSION:
        raise H6DiagnosisError("unsupported H6 evaluation schema")
    if evaluation.get("status") != "complete":
        raise H6DiagnosisError("H6 evaluation is not complete")
    if terminal_decision.get("schema") != H6_TERMINAL_SCHEMA_VERSION:
        raise H6DiagnosisError("unsupported H6 terminal-decision schema")
    if terminal_decision.get("decision") != "REJECTED":
        raise H6DiagnosisError("H6 terminal decision is not REJECTED")
    if terminal_decision.get("one_shot_consumed") is not True:
        raise H6DiagnosisError("H6 one-shot boundary is not marked consumed")

    terminal_evaluation = _require_mapping(
        terminal_decision.get("evaluation"), role="terminal evaluation"
    )
    if terminal_evaluation.get("result_sha256") != evaluation_sha256:
        raise H6DiagnosisError("H6 evaluation hash does not match terminal decision")

    raw = _require_mapping(evaluation.get("uncalibrated_raw"), role="uncalibrated_raw")
    diagnostics = raw.get("item_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise H6DiagnosisError("H6 item diagnostics must be a non-empty array")
    if raw.get("decode_failure_items") != 0:
        raise H6DiagnosisError("canonical H6 diagnosis requires zero decode failures")
    if raw.get("items") != len(diagnostics):
        raise H6DiagnosisError("H6 item count does not match row diagnostics")

    overall: Counter[str] = Counter()
    by_gold_state: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_field: defaultdict[str, Counter[str]] = defaultdict(Counter)
    raw_state_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    span_relations: Counter[str] = Counter()
    span_relations_by_field: defaultdict[str, Counter[str]] = defaultdict(Counter)
    span_relations_by_gold_state: defaultdict[str, Counter[str]] = defaultdict(Counter)

    expected_fields = set(FIELD_ORDER)
    for row_index, row_value in enumerate(diagnostics):
        row = _require_mapping(row_value, role=f"row[{row_index}]")
        if row.get("decode_status") != "decoded" or row.get("decode_error") is not None:
            raise H6DiagnosisError(f"row[{row_index}] is not a successful decode")
        fields = row.get("fields")
        if not isinstance(fields, list) or len(fields) != len(FIELD_ORDER):
            raise H6DiagnosisError(f"row[{row_index}] does not contain five fields")
        names = {field.get("field") for field in fields if isinstance(field, Mapping)}
        if names != expected_fields:
            raise H6DiagnosisError(f"row[{row_index}] field identity changed")

        for field_index, field_value in enumerate(fields):
            role = f"row[{row_index}].fields[{field_index}]"
            field = _require_mapping(field_value, role=role)
            field_name = field.get("field")
            gold_state = field.get("gold_state")
            raw_state = field.get("raw_state")
            if field_name not in expected_fields:
                raise H6DiagnosisError(f"{role}.field is invalid")
            if gold_state not in _VALID_STATES or raw_state not in _VALID_STATES:
                raise H6DiagnosisError(f"{role} has an invalid state")
            axis = _classify_field(field, role=role)
            overall[axis] += 1
            by_gold_state[str(gold_state)][axis] += 1
            by_field[str(field_name)][axis] += 1
            raw_state_counts[str(gold_state)][str(raw_state)] += 1
            relation = _span_relation(field, role=role)
            if relation is not None:
                if axis not in {"binding_only_error", "combined_error"}:
                    raise H6DiagnosisError(
                        f"{role} has a span mismatch outside a binding error axis"
                    )
                span_relations[relation] += 1
                span_relations_by_field[str(field_name)][relation] += 1
                span_relations_by_gold_state[str(gold_state)][relation] += 1

    expected_field_total = len(diagnostics) * len(FIELD_ORDER)
    if sum(overall.values()) != expected_field_total:
        raise H6DiagnosisError("H6 field population is incomplete")

    gold_state_report: dict[str, Any] = {}
    for state in sorted(_VALID_STATES):
        state_summary = _summary(by_gold_state[state])
        state_summary["raw_state_counts"] = dict(sorted(raw_state_counts[state].items()))
        gold_state_report[state] = state_summary

    decision = _require_mapping(evaluation.get("decision"), role="evaluation decision")
    stages = {
        "uncalibrated_raw": {"assessed": True},
        "threshold_calibration": {
            "assessed": evaluation.get("calibrated_raw") is not None
        },
        "verifier_final": {"assessed": evaluation.get("verifier_final") is not None},
        "latency": {"assessed": decision.get("latency_assessed") is True},
        "sealed_confirmation": {
            "assessed": decision.get("sealed_confirmation_assessed") is True
        },
    }

    return {
        "schema": DIAGNOSIS_SCHEMA_VERSION,
        "experiment_id": terminal_decision.get("experiment_id"),
        "terminal_decision": terminal_decision.get("decision"),
        "one_shot_consumed": True,
        "source": {
            "evaluation_sha256": evaluation_sha256,
            "terminal_decision_sha256": terminal_decision_sha256,
            "rows": len(diagnostics),
            "fields": expected_field_total,
        },
        "stages": stages,
        "overall": _summary(overall),
        "by_gold_state": gold_state_report,
        "by_field": {
            field_name: _summary(by_field[field_name]) for field_name in FIELD_ORDER
        },
        "binding_shape": {
            "span_error_fields": sum(span_relations.values()),
            "relations": dict(sorted(span_relations.items())),
            "by_field": {
                field_name: dict(sorted(span_relations_by_field[field_name].items()))
                for field_name in FIELD_ORDER
            },
            "by_gold_state": {
                state: dict(sorted(span_relations_by_gold_state[state].items()))
                for state in sorted(_VALID_STATES)
            },
            "interpretation": (
                "descriptive_post_hoc_only; exact H6 evaluation remains terminal"
            ),
        },
        "scientific_boundary": {
            "raw_error_attribution_available": True,
            "threshold_effect": (
                "assessed" if stages["threshold_calibration"]["assessed"] else "not_assessed"
            ),
            "verifier_effect": (
                "assessed" if stages["verifier_final"]["assessed"] else "not_assessed"
            ),
            "requires_new_model_run": False,
        },
    }


def load_h6_diagnosis(
    evaluation_path: str | Path = DEFAULT_EVALUATION_PATH,
    terminal_decision_path: str | Path = DEFAULT_TERMINAL_DECISION_PATH,
) -> dict[str, Any]:
    evaluation, evaluation_sha256 = _read_json(
        Path(evaluation_path), role="H6 evaluation"
    )
    terminal_decision, terminal_decision_sha256 = _read_json(
        Path(terminal_decision_path), role="H6 terminal decision"
    )
    if evaluation_sha256 != CANONICAL_H6_EVALUATION_SHA256:
        raise H6DiagnosisError("H6 evaluation does not match the canonical hash")
    if terminal_decision_sha256 != CANONICAL_H6_TERMINAL_DECISION_SHA256:
        raise H6DiagnosisError("H6 terminal decision does not match the canonical hash")
    return diagnose_h6(
        evaluation,
        evaluation_sha256=evaluation_sha256,
        terminal_decision=terminal_decision,
        terminal_decision_sha256=terminal_decision_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose authenticated row-level H6 errors without rerunning H6."
    )
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION_PATH)
    parser.add_argument(
        "--terminal-decision", type=Path, default=DEFAULT_TERMINAL_DECISION_PATH
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = load_h6_diagnosis(args.evaluation, args.terminal_decision)
    except H6DiagnosisError as exc:
        print(f"H6_DIAGNOSIS_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
