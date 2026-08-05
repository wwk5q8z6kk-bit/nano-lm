"""Known-development evaluation for Nano's frozen H5 replay intervention.

Both fixed-seed training reports, their selected checkpoints, and the complete
replay-mixture training bundle are authenticated before known development is
opened.  Development decides only the preregistered quality gate; it cannot
select a seed, epoch, threshold, or replay family.

H5 deliberately reuses an H4 calibration partition whose surfaces/templates
are unseen by the fit mixture while some legacy values are familiar.  This
module records that limitation explicitly and never describes calibration as
open-value disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.contract import FIELD_ORDER, FieldState
from nano_ai.training import evaluate_evidence_query as h3_evaluation
from nano_ai.training import (
    evidence_query_inference,
    evidence_query_model,
    pointer_data,
    replay_mixture_data,
    state_span_data,
)
from nano_ai.training.evaluate_state_span import (
    DEVELOPMENT_PARTITION_ID,
    _fixture_cases,
    acceptance_diagnostics,
    final_state_diagnostics,
    load_development_bundle,
)
from nano_ai.training.evidence_query_inference import (
    CALIBRATION_THRESHOLD_POLICY,
    apply_global_threshold,
    batched_evidence_query_inference,
    build_pointer_inference_inputs,
    raw_pointer_diagnostics,
)
from nano_ai.training.evidence_query_model import (
    ARCHITECTURE_VERSION,
    EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
)
from nano_ai.training.pointer_data import (
    POINTER_SUPERVISION_VERSION,
    STATE_ORDER,
    encode_pointer_partition,
    load_pointer_tokenizer,
)
from nano_ai.training.pointer_model import NANO_TRUNK_PARAMETER_COUNT
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    TARGET_GRAMMAR_VERSION,
)
from nano_ai.training.train_evidence_query_h5 import (
    _TRAINING_SELECTION_NOTE,
    CALIBRATION_RECORD_COUNT,
    CALIBRATION_STATE_CLASS_COUNTS,
    CALIBRATION_WORLD_COUNT,
    FIT_RECORD_COUNT,
    FIT_STATE_CLASS_COUNTS,
    FIT_WORLD_COUNT,
    H5_TRAINING_RECIPE_VERSION,
    H5_TRAINING_REPORT_SCHEMA_VERSION,
    PRESERVED_H3_SOURCE_SHA256,
    REPORT_DATASET_KEYS,
    SOURCE_WORLD_COUNT,
    TRAINING_SEEDS,
    changed_source_paths,
    preserved_source_paths,
)

H5_DEVELOPMENT_EVALUATION_SCHEMA_VERSION = (
    "nano.evidence-query-h5-development-evaluation.v1"
)
DEFAULT_BATCH_SIZE = 32
DEVELOPMENT_RECORDS = 1_000
MODAL_STATE_MAXIMUM = 949

_REPORT_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "recipe",
        "status",
        "seed",
        "device",
        "architecture_version",
        "parameter_count",
        "trunk_parameter_count",
        "evidence_query_head_parameter_count",
        "architecture_identity",
        "base_checkpoint_sha256",
        "tokenizer_sha256",
        "dataset_manifest_sha256",
        "dataset",
        "hyperparameters",
        "epochs",
        "candidate",
        "calibration",
        "dev_used_for_selection",
        "legacy_record_artifact_accessed",
        "fresh_v1_accessed",
        "preserved_source_sha256",
        "changed_source_sha256",
        "runtime",
        "selection_note",
    }
)
_CHECKPOINT_KEYS = frozenset({"filename", "sha256", "bytes"})
_CALIBRATION_KEYS = frozenset(
    {"uncalibrated", "global_threshold", "calibrated", "threshold_policy"}
)
_SEMANTIC_GATES = {
    "overall": (3_041, 5_000),
    "held_value": (2_167, 2_987),
    "missing_target": (219, 250),
    "absence": (383, 413),
    "conflict_target": (236, 250),
    "uncertain_target": (228, 250),
}

# H5 keeps the accepted H3 inference and verification surfaces, but owns its
# stricter held-value floor and field-collapse admission gate below.
EvidenceQueryEvaluationError = h3_evaluation.EvidenceQueryEvaluationError
_final_metrics = h3_evaluation._final_metrics
_epoch_calibration = h3_evaluation._epoch_calibration
_seed_evaluation = h3_evaluation._seed_evaluation
_load_evidence_query_model = h3_evaluation._load_evidence_query_model
_evaluate_predictions = h3_evaluation._evaluate_predictions


@dataclass(frozen=True, slots=True)
class EvidenceQueryH5Candidate:
    """One report-selected H5 checkpoint with training-only ranking facts."""

    seed: int
    epoch: int
    path: Path
    sha256: str
    artifact_bytes: int
    report_sha256: str
    global_threshold: float
    macro_joint: float
    overall_joint: float
    report: Mapping[str, Any] = field(repr=False)

    @property
    def label(self) -> str:
        return f"seed-{self.seed}-epoch-{self.epoch}"

    @property
    def ranking_key(self) -> tuple[float, float, int, int]:
        return (
            self.macro_joint,
            self.overall_joint,
            -self.epoch,
            int(self.seed == TRAINING_SEEDS[0]),
        )


EvidenceQueryCandidate = EvidenceQueryH5Candidate


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceQueryEvaluationError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _read_regular_file(path: Path, *, role: str) -> bytes:
    return h3_evaluation._read_regular_file(Path(path), role=role)


def _read_verified_file(path: Path, expected_sha256: str, *, role: str) -> bytes:
    return h3_evaluation._read_verified_file(Path(path), expected_sha256, role=role)


def _source_hashes(paths: Mapping[str, Path], *, role: str) -> dict[str, str]:
    return {
        name: _sha256(_read_regular_file(Path(path), role=f"{role} {name} source"))
        for name, path in sorted(paths.items())
    }


def _preserved_source_hashes() -> dict[str, str]:
    return _source_hashes(preserved_source_paths(), role="preserved H3")


def _changed_source_hashes() -> dict[str, str]:
    return _source_hashes(changed_source_paths(), role="changed H5")


def _checkpoint_record(value: object) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_KEYS:
        raise EvidenceQueryEvaluationError("H5 candidate checkpoint record is invalid")
    filename = value["filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise EvidenceQueryEvaluationError("H5 checkpoint filename is unsafe")
    _require_sha256(value["sha256"], "H5 checkpoint")
    byte_count = value["bytes"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise EvidenceQueryEvaluationError("H5 checkpoint byte count is invalid")
    return value


def _state_class_counts(values: Sequence[int]) -> dict[str, int]:
    return {
        state.value: count for state, count in zip(STATE_ORDER, values, strict=True)
    }


def _source_record(
    value: object,
    *,
    expected: Mapping[str, Any],
    role: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise EvidenceQueryEvaluationError(f"H5 {role} identity is invalid")
    byte_count = value.get("bytes")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise EvidenceQueryEvaluationError(f"H5 {role} byte count is invalid")
    _require_sha256(value.get("sha256"), f"H5 {role}")
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise EvidenceQueryEvaluationError(f"H5 {role} identity changed")
    return value


def _validate_report_dataset_metadata(
    value: object,
    *,
    training_data_dir: str | Path,
    training_bundle: Any,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != REPORT_DATASET_KEYS:
        raise EvidenceQueryEvaluationError("H5 training dataset record is invalid")
    if (
        value["schema_version"] != DATASET_SCHEMA_VERSION
        or value["generator"] != replay_mixture_data.GENERATOR_VERSION
        or value["selection_policy"] != replay_mixture_data.SELECTION_POLICY_VERSION
        or value["target_grammar"] != TARGET_GRAMMAR_VERSION
    ):
        raise EvidenceQueryEvaluationError("H5 dataset protocol identity is invalid")
    for name in ("generator_sha256", "normalization", "training_identity"):
        if value[name] != training_bundle.manifest[name]:
            raise EvidenceQueryEvaluationError(f"H5 {name} identity changed")

    root = Path(training_data_dir)
    expected_sources = {
        "source_manifest": {
            "filename": "manifest.json",
            "bytes": len(
                _read_regular_file(root / "manifest.json", role="H5 manifest")
            ),
            "sha256": expected_manifest_sha256,
        },
        "source_fit": {
            "filename": "fit.jsonl",
            "bytes": len(_read_regular_file(root / "fit.jsonl", role="H5 fit")),
            "sha256": training_bundle.input_sha256["fit"],
            "records": FIT_RECORD_COUNT,
            "worlds": FIT_WORLD_COUNT,
            "legacy_worlds": SOURCE_WORLD_COUNT,
            "surface_worlds": SOURCE_WORLD_COUNT,
            "gradient_bearing": True,
        },
        "source_calibration": {
            "filename": "calibration.jsonl",
            "bytes": len(
                _read_regular_file(root / "calibration.jsonl", role="H5 calibration")
            ),
            "sha256": training_bundle.input_sha256["calibration"],
            "records": CALIBRATION_RECORD_COUNT,
            "worlds": CALIBRATION_WORLD_COUNT,
            "namespace": "train-calibration",
            "gradient_bearing": False,
            "reused_unchanged_from_h4": True,
        },
    }
    for name, expected in expected_sources.items():
        _source_record(value[name], expected=expected, role=name)

    expected_fit = {
        **dict(training_bundle.manifest["partitions"]["fit"]),
        "state_class_counts": _state_class_counts(FIT_STATE_CLASS_COUNTS),
    }
    expected_calibration = {
        **dict(training_bundle.manifest["partitions"]["calibration"]),
        "state_class_counts": _state_class_counts(CALIBRATION_STATE_CLASS_COUNTS),
    }
    if value["fit"] != expected_fit:
        raise EvidenceQueryEvaluationError("H5 fit partition identity changed")
    if value["calibration"] != expected_calibration:
        raise EvidenceQueryEvaluationError("H5 calibration partition identity changed")
    for name in ("sources", "overlap_audit", "restrictions"):
        expected = training_bundle.manifest[name]
        if value[name] != expected:
            raise EvidenceQueryEvaluationError(f"H5 {name} identity changed")

    overlap = value["overlap_audit"]
    restrictions = value["restrictions"]
    if (
        not isinstance(overlap, Mapping)
        or overlap.get(
            "calibration_open_value_literal_substring_occurrence_is_eligibility_rule"
        )
        is not False
        or overlap.get("all_hard_intersections_zero") is not True
        or overlap.get("calibration_records_modified") is not False
        or not isinstance(restrictions, Mapping)
        or restrictions.get("legacy_record_artifact_read") is not False
        or restrictions.get("development_read") is not False
        or restrictions.get("benchmark_read") is not False
        or restrictions.get("sealed_confirmation_read") is not False
    ):
        raise EvidenceQueryEvaluationError("H5 replay isolation claims are invalid")
    value_audit = overlap.get("calibration_open_value_literal_substring_occurrence")
    if (
        not isinstance(value_audit, Mapping)
        or value_audit.get("policy") != "expected_recorded_nonblocking"
        or value_audit.get("candidate_worlds") != 2_800
        or value_audit.get("literal_substring_disjoint_worlds") != 1_053
        or value_audit.get("worlds_with_literal_substring_occurrence") != 1_747
        or value_audit.get("exact_value_identity_not_claimed") is not True
    ):
        raise EvidenceQueryEvaluationError("H5 calibration-value limitation drifted")
    return value


def _validate_report_source_hashes(report: Mapping[str, Any]) -> None:
    preserved = report["preserved_source_sha256"]
    if not isinstance(preserved, dict) or set(preserved) != set(
        PRESERVED_H3_SOURCE_SHA256
    ):
        raise EvidenceQueryEvaluationError("H5 preserved source hashes are incomplete")
    for name, digest in preserved.items():
        _require_sha256(digest, f"H5 preserved source {name}")
    if preserved != dict(PRESERVED_H3_SOURCE_SHA256):
        raise EvidenceQueryEvaluationError("H5 preserved source pins changed")
    if preserved != _preserved_source_hashes():
        raise EvidenceQueryEvaluationError(
            "H5 preserved source hashes do not match the executable recipe"
        )

    changed = report["changed_source_sha256"]
    expected_changed = _changed_source_hashes()
    if not isinstance(changed, dict) or set(changed) != set(expected_changed):
        raise EvidenceQueryEvaluationError("H5 changed source hashes are incomplete")
    for name, digest in changed.items():
        _require_sha256(digest, f"H5 changed source {name}")
    if changed != expected_changed:
        raise EvidenceQueryEvaluationError(
            "H5 changed source hashes do not match the executable recipe"
        )


def _candidate_checkpoint_record(candidate: object) -> tuple[int, Mapping[str, Any]]:
    if not isinstance(candidate, dict) or set(candidate) != {
        "epoch",
        *_CHECKPOINT_KEYS,
    }:
        raise EvidenceQueryEvaluationError("H5 selected candidate record is invalid")
    epoch = candidate["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise EvidenceQueryEvaluationError("H5 selected candidate epoch is invalid")
    return epoch, _checkpoint_record({key: candidate[key] for key in _CHECKPOINT_KEYS})


def load_candidate_from_training_report(
    report_path: str | Path,
    *,
    expected_report_sha256: str,
    expected_manifest_sha256: str,
    training_data_dir: str | Path,
    training_bundle: Any,
) -> EvidenceQueryH5Candidate:
    """Authenticate one fixed-seed H5 report and selected checkpoint."""

    path = Path(report_path)
    snapshot = _read_verified_file(
        path, expected_report_sha256, role="H5 training report"
    )
    report = h3_evaluation._parse_json(snapshot, role="H5 training report")
    if not isinstance(report, dict) or set(report) != _REPORT_REQUIRED_KEYS:
        raise EvidenceQueryEvaluationError("H5 training report schema is invalid")
    if (
        report["schema_version"] != H5_TRAINING_REPORT_SCHEMA_VERSION
        or report["recipe"] != H5_TRAINING_RECIPE_VERSION
        or report["status"] != "complete"
    ):
        raise EvidenceQueryEvaluationError("H5 training report is not complete v1")
    seed = report["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in TRAINING_SEEDS
    ):
        raise EvidenceQueryEvaluationError("H5 training seed is not frozen")
    if (
        report["architecture_version"] != ARCHITECTURE_VERSION
        or report["architecture_identity"] != FROZEN_NANO_V01.architecture_identity
        or report["parameter_count"] != NANO_EVIDENCE_QUERY_PARAMETER_COUNT
        or report["trunk_parameter_count"] != NANO_TRUNK_PARAMETER_COUNT
        or report["evidence_query_head_parameter_count"]
        != EVIDENCE_QUERY_HEAD_PARAMETER_COUNT
        or report["base_checkpoint_sha256"] != FROZEN_NANO_V01.checkpoint_sha256
        or report["tokenizer_sha256"] != FROZEN_NANO_V01.tokenizer_sha256
    ):
        raise EvidenceQueryEvaluationError("H5 training model identity is invalid")
    expected_manifest = _require_sha256(
        expected_manifest_sha256, "expected H5 training manifest"
    )
    if report["dataset_manifest_sha256"] != expected_manifest:
        raise EvidenceQueryEvaluationError("H5 training used another dataset manifest")
    _validate_report_dataset_metadata(
        report["dataset"],
        training_data_dir=training_data_dir,
        training_bundle=training_bundle,
        expected_manifest_sha256=expected_manifest,
    )
    if report["dev_used_for_selection"] is not False:
        raise EvidenceQueryEvaluationError("development influenced H5 selection")
    if report["legacy_record_artifact_accessed"] is not False:
        raise EvidenceQueryEvaluationError(
            "H5 training read the legacy record artifact"
        )
    if report["fresh_v1_accessed"] is not False:
        raise EvidenceQueryEvaluationError("H5 training accessed sealed confirmation")
    if report["selection_note"] != _TRAINING_SELECTION_NOTE:
        raise EvidenceQueryEvaluationError("H5 training selection boundary changed")
    h3_evaluation._validate_hyperparameters(report["hyperparameters"])
    h3_evaluation._validate_runtime(report["runtime"], device=report["device"])
    _validate_report_source_hashes(report)

    epochs = report["epochs"]
    if not isinstance(epochs, list) or len(epochs) != 3:
        raise EvidenceQueryEvaluationError("H5 report must contain three epochs")
    ranking_rows: list[tuple[float, float, int, Mapping[str, Any]]] = []
    for expected_epoch, row in enumerate(epochs, 1):
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "epoch",
                "train_loss",
                "state_loss",
                "pointer_loss",
                "seconds",
                "checkpoint",
                "calibration",
            }
            or row.get("epoch") != expected_epoch
        ):
            raise EvidenceQueryEvaluationError("H5 training epochs are not ordered")
        for metric in ("train_loss", "state_loss", "pointer_loss", "seconds"):
            value = row[metric]
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise EvidenceQueryEvaluationError(f"H5 epoch {metric} is invalid")
        _checkpoint_record(row["checkpoint"])
        macro, overall, _threshold = _epoch_calibration(row)
        ranking_rows.append((macro, overall, -expected_epoch, row))

    selected_row = max(ranking_rows, key=lambda item: item[:3])[3]
    selected_epoch = int(selected_row["epoch"])
    candidate_epoch, checkpoint = _candidate_checkpoint_record(report["candidate"])
    if candidate_epoch != selected_epoch or checkpoint != selected_row["checkpoint"]:
        raise EvidenceQueryEvaluationError(
            "H5 candidate was not selected by uncalibrated calibration ranking"
        )
    calibration = report["calibration"]
    if not isinstance(calibration, Mapping) or set(calibration) != {
        "selected_epoch",
        *_CALIBRATION_KEYS,
    }:
        raise EvidenceQueryEvaluationError("H5 selected calibration schema is invalid")
    if calibration["selected_epoch"] != selected_epoch:
        raise EvidenceQueryEvaluationError("H5 selected calibration epoch disagrees")
    if {key: calibration[key] for key in _CALIBRATION_KEYS} != selected_row[
        "calibration"
    ]:
        raise EvidenceQueryEvaluationError("H5 selected calibration record disagrees")
    macro, overall, threshold = _epoch_calibration(selected_row)

    checkpoint_snapshot = _read_verified_file(
        path.parent / checkpoint["filename"],
        checkpoint["sha256"],
        role=f"H5 checkpoint seed {seed}",
    )
    if len(checkpoint_snapshot) != checkpoint["bytes"]:
        raise EvidenceQueryEvaluationError("H5 checkpoint byte count changed")
    return EvidenceQueryH5Candidate(
        seed=seed,
        epoch=selected_epoch,
        path=path.parent / checkpoint["filename"],
        sha256=checkpoint["sha256"],
        artifact_bytes=checkpoint["bytes"],
        report_sha256=_sha256(snapshot),
        global_threshold=threshold,
        macro_joint=macro,
        overall_joint=overall,
        report=report,
    )


def authenticate_and_select_primary(
    training_reports: Sequence[tuple[str | Path, str]],
    *,
    expected_manifest_sha256: str,
    training_data_dir: str | Path,
    training_bundle: Any,
) -> tuple[EvidenceQueryH5Candidate, tuple[EvidenceQueryH5Candidate, ...]]:
    """Select H5 solely from authenticated training-calibration evidence."""

    if len(training_reports) != len(TRAINING_SEEDS):
        raise EvidenceQueryEvaluationError(
            "exactly two fixed-seed H5 training reports are required"
        )
    candidates = tuple(
        load_candidate_from_training_report(
            path,
            expected_report_sha256=digest,
            expected_manifest_sha256=expected_manifest_sha256,
            training_data_dir=training_data_dir,
            training_bundle=training_bundle,
        )
        for path, digest in training_reports
    )
    if {candidate.seed for candidate in candidates} != set(TRAINING_SEEDS):
        raise EvidenceQueryEvaluationError("H5 reports do not cover both fixed seeds")
    if len({candidate.report_sha256 for candidate in candidates}) != 2:
        raise EvidenceQueryEvaluationError("H5 report digests must be unique")
    if len({candidate.sha256 for candidate in candidates}) != 2:
        raise EvidenceQueryEvaluationError("H5 checkpoint digests must be unique")
    ordered = tuple(sorted(candidates, key=lambda item: item.seed))
    return max(ordered, key=lambda item: item.ranking_key), ordered


def _require_metric(
    metrics: Mapping[str, Any],
    name: str,
    *,
    expected_denominator: int | None,
) -> Mapping[str, Any]:
    value = metrics.get(name)
    if not isinstance(value, Mapping):
        raise EvidenceQueryEvaluationError(f"H5 metric {name} is missing")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    rate = value.get("rate")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator < 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 0
        or numerator > denominator
        or (expected_denominator is not None and denominator != expected_denominator)
        or (denominator == 0 and (numerator != 0 or rate is not None))
        or (
            denominator > 0
            and (
                type(rate) not in {int, float}
                or not math.isfinite(rate)
                or not math.isclose(
                    float(rate),
                    numerator / denominator,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            )
        )
    ):
        raise EvidenceQueryEvaluationError(f"H5 metric {name} is inconsistent")
    return value


def _quality_gates(
    metrics: Mapping[str, Any],
    *,
    require_zero_wrong_presented: bool,
) -> dict[str, Any]:
    """Apply H5's preregistered semantic, retention, and failure gates."""

    evidence: dict[str, Any] = {}
    for name, (minimum, denominator) in _SEMANTIC_GATES.items():
        metric = _require_metric(metrics, name, expected_denominator=denominator)
        evidence[name] = {
            "metric": dict(metric),
            "minimum_numerator": minimum,
            "passed": metric["numerator"] >= minimum,
        }
    failures = _require_metric(metrics, "failures", expected_denominator=1_000)
    evidence["failures"] = {
        "metric": dict(failures),
        "maximum_numerator": 10,
        "passed": failures["numerator"] <= 10,
    }
    semantic_pass = all(row["passed"] for row in evidence.values())
    result: dict[str, Any] = {
        "gate_evidence": evidence,
        "semantic_and_retention_passed": semantic_pass,
        "zero_wrong_presented_required": require_zero_wrong_presented,
    }
    if require_zero_wrong_presented:
        wrong = _require_metric(metrics, "false_presented", expected_denominator=None)
        zero_wrong = wrong["numerator"] == 0
        result["gate_evidence"]["zero_wrong_presented"] = {
            "metric": dict(wrong),
            "maximum_numerator": 0,
            "passed": zero_wrong,
        }
        result["all_quality_gates_passed"] = semantic_pass and zero_wrong
    else:
        result["all_quality_gates_passed"] = semantic_pass
    return result


def _uncalibrated_state_balance(predictions: Sequence[Any]) -> dict[str, Any]:
    """Gate literal per-field raw-state collapse before calibration."""

    if len(predictions) != DEVELOPMENT_RECORDS:
        raise EvidenceQueryEvaluationError(
            "H5 state-balance gate requires exactly 1,000 development rows"
        )
    counts = {
        field_name.value: Counter({state.value: 0 for state in FieldState})
        for field_name in FIELD_ORDER
    }
    failed_items = 0
    for prediction in predictions:
        error = getattr(prediction, "error", None)
        proposals = getattr(prediction, "proposals", None)
        if error is not None:
            if proposals is not None:
                raise EvidenceQueryEvaluationError(
                    "H5 failed prediction unexpectedly contains proposals"
                )
            failed_items += 1
            continue
        if proposals is None or len(proposals) != len(FIELD_ORDER):
            raise EvidenceQueryEvaluationError("H5 decoded prediction shape is invalid")
        for expected_field, proposal in zip(FIELD_ORDER, proposals, strict=True):
            if getattr(proposal, "field", None) is not expected_field or not isinstance(
                getattr(proposal, "state", None), FieldState
            ):
                raise EvidenceQueryEvaluationError(
                    "H5 decoded field/state identity is invalid"
                )
            counts[expected_field.value][proposal.state.value] += 1

    fields: dict[str, Any] = {}
    for field_name in FIELD_ORDER:
        state_counts = dict(counts[field_name.value])
        assigned = sum(state_counts.values())
        if assigned != DEVELOPMENT_RECORDS - failed_items:
            raise EvidenceQueryEvaluationError(
                "H5 state-balance counts are inconsistent"
            )
        modal_state, modal_count = min(
            state_counts.items(), key=lambda item: (-item[1], item[0])
        )
        fields[field_name.value] = {
            "state_counts": state_counts,
            "assigned_rows": assigned,
            "development_rows": DEVELOPMENT_RECORDS,
            "modal_state": modal_state,
            "modal_count": modal_count,
            "maximum_modal_count": MODAL_STATE_MAXIMUM,
            "passed": modal_count <= MODAL_STATE_MAXIMUM,
        }
    passed = all(row["passed"] for row in fields.values())
    return {
        "definition": (
            "No field may assign one uncalibrated raw state to 950 or more of "
            "the 1,000 development rows; decode failures assign no state."
        ),
        "development_rows": DEVELOPMENT_RECORDS,
        "decode_failure_items": failed_items,
        "maximum_modal_count": MODAL_STATE_MAXIMUM,
        "fields": fields,
        "passed": passed,
    }


def _uncalibrated_admission(
    metrics: Mapping[str, Any], predictions: Sequence[Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = _quality_gates(metrics, require_zero_wrong_presented=False)
    state_balance = _uncalibrated_state_balance(predictions)
    gate["uncalibrated_state_balance"] = state_balance
    gate["uncalibrated_state_balance_passed"] = state_balance["passed"]
    gate["all_quality_gates_passed"] = bool(
        gate["all_quality_gates_passed"] and state_balance["passed"]
    )
    return gate, state_balance


def _canonical_evaluation_runtime(
    primary: EvidenceQueryH5Candidate,
    candidates: Sequence[EvidenceQueryH5Candidate],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    return h3_evaluation._canonical_evaluation_runtime(
        primary, candidates, device=device, batch_size=batch_size
    )


def _resolve_device(device: str) -> str:
    return h3_evaluation._resolve_device(device)


def _evaluation_source_paths() -> dict[str, Path]:
    package_dir = Path(__file__).parent.parent
    return {
        "adapter": package_dir / "adapters" / "evidence_query_pointer.py",
        "base_evaluator": Path(h3_evaluation.__file__),
        "data_generator": Path(state_span_data.__file__),
        "evaluator": Path(__file__),
        "inference": Path(evidence_query_inference.__file__),
        "model": Path(evidence_query_model.__file__),
        "pointer_data": Path(pointer_data.__file__),
        "replay_data": Path(replay_mixture_data.__file__),
    }


def _evaluation_source_hashes() -> dict[str, str]:
    return _source_hashes(_evaluation_source_paths(), role="H5 evaluation")


def _require_training_data_unchanged(
    training_data_dir: str | Path, input_sha256: Mapping[str, str]
) -> None:
    root = Path(training_data_dir)
    filenames = {
        "manifest": "manifest.json",
        "fit": "fit.jsonl",
        "calibration": "calibration.jsonl",
    }
    observed = {
        name: _sha256(
            _read_regular_file(root / filename, role=f"H5 training {filename}")
        )
        for name, filename in filenames.items()
    }
    if observed != dict(input_sha256):
        raise EvidenceQueryEvaluationError("H5 training data changed during evaluation")


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    h3_evaluation._write_json_no_clobber(path, value)


def evaluate_development(
    *,
    training_data_dir: str | Path,
    training_manifest_sha256: str,
    development_data_dir: str | Path,
    development_manifest_sha256: str,
    tokenizer_path: str | Path,
    training_reports: Sequence[tuple[str | Path, str]],
    output_path: str | Path,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Mapping[str, Any]:
    """Select H5 without development, then execute only admitted stages."""

    output = Path(output_path)
    if output.exists():
        raise EvidenceQueryEvaluationError("H5 evaluation output already exists")
    if not output.parent.is_dir():
        raise EvidenceQueryEvaluationError("H5 evaluation output parent must exist")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise EvidenceQueryEvaluationError("batch_size must be a positive integer")
    expected_training_manifest = _require_sha256(
        training_manifest_sha256, "H5 training manifest"
    )
    expected_development_manifest = _require_sha256(
        development_manifest_sha256, "H2 development manifest"
    )
    resolved_device = _resolve_device(device)
    source_hashes = _evaluation_source_hashes()

    # This ordering is the central H5 boundary.  Only replay inputs, reports,
    # and checkpoints are open until both fixed seeds and the primary are frozen.
    try:
        training_bundle = replay_mixture_data.load_replay_mixture_dataset(
            Path(training_data_dir)
        )
    except Exception as exc:
        raise EvidenceQueryEvaluationError(
            "H5 replay-mixture training inputs could not be verified"
        ) from exc
    if not hmac.compare_digest(
        training_bundle.manifest_sha256, expected_training_manifest
    ):
        raise EvidenceQueryEvaluationError("H5 training manifest SHA-256 mismatch")
    primary, candidates = authenticate_and_select_primary(
        training_reports,
        expected_manifest_sha256=expected_training_manifest,
        training_data_dir=training_data_dir,
        training_bundle=training_bundle,
    )
    evaluation_runtime = _canonical_evaluation_runtime(
        primary,
        candidates,
        device=resolved_device,
        batch_size=batch_size,
    )
    _require_training_data_unchanged(training_data_dir, training_bundle.input_sha256)

    try:
        development_bundle = load_development_bundle(
            development_data_dir,
            expected_manifest_sha256=expected_development_manifest,
        )
        tokenizer = load_pointer_tokenizer(Path(tokenizer_path))
        encoded = encode_pointer_partition(
            tokenizer, development_bundle.examples, expected_split="dev"
        )
        inputs = build_pointer_inference_inputs(development_bundle.examples, encoded)
        cases = _fixture_cases(development_bundle.examples)
    except EvidenceQueryEvaluationError:
        raise
    except Exception as exc:
        raise EvidenceQueryEvaluationError(
            "known H2 development inputs could not be verified"
        ) from exc

    _seed_evaluation()
    model = _load_evidence_query_model(primary, device=resolved_device)
    inference = batched_evidence_query_inference(
        model, inputs, device=resolved_device, batch_size=batch_size
    )
    del model
    if tuple(inference.example_ids) != tuple(item.example_id for item in inputs):
        raise EvidenceQueryEvaluationError("H5 inference changed development order")

    uncalibrated_raw = raw_pointer_diagnostics(
        development_bundle.examples, cases, inference.predictions
    )
    uncalibrated_gate, state_balance = _uncalibrated_admission(
        uncalibrated_raw["acceptance"]["metrics"], inference.predictions
    )
    admission_passed = bool(uncalibrated_gate["all_quality_gates_passed"])
    calibrated_raw: Mapping[str, Any] | None = None
    calibrated_gate: Mapping[str, Any] | None = None
    verifier_final_section: Mapping[str, Any] | None = None
    verifier_gate: Mapping[str, Any] | None = None

    if admission_passed:
        calibrated_predictions = apply_global_threshold(
            inference, primary.global_threshold
        )
        calibrated_raw = raw_pointer_diagnostics(
            development_bundle.examples, cases, calibrated_predictions
        )
        calibrated_gate = _quality_gates(
            calibrated_raw["acceptance"]["metrics"],
            require_zero_wrong_presented=True,
        )
        if calibrated_gate["all_quality_gates_passed"]:
            verifier_report = _evaluate_predictions(
                inputs=inputs,
                predictions=calibrated_predictions,
                cases=cases,
                candidate=primary,
            )
            verifier_final = final_state_diagnostics(
                verifier_report, development_bundle.examples
            )
            verifier_acceptance = acceptance_diagnostics(
                verifier_report, development_bundle.examples
            )
            verifier_final_section = {
                "evaluation": verifier_report.to_dict(),
                "state": verifier_final,
                "acceptance": verifier_acceptance,
            }
            verifier_gate = _quality_gates(
                _final_metrics(verifier_acceptance, verifier_final),
                require_zero_wrong_presented=True,
            )

    quality_passed = bool(
        admission_passed
        and calibrated_gate is not None
        and calibrated_gate["all_quality_gates_passed"]
        and verifier_gate is not None
        and verifier_gate["all_quality_gates_passed"]
    )
    _require_training_data_unchanged(training_data_dir, training_bundle.input_sha256)
    if _evaluation_source_hashes() != source_hashes:
        raise EvidenceQueryEvaluationError("H5 evaluation source changed during run")

    candidate_rows = [
        {
            "label": candidate.label,
            "seed": candidate.seed,
            "epoch": candidate.epoch,
            "checkpoint_sha256": candidate.sha256,
            "checkpoint_bytes": candidate.artifact_bytes,
            "training_report_sha256": candidate.report_sha256,
            "uncalibrated_calibration_ranking": {
                "macro_joint": candidate.macro_joint,
                "overall_joint": candidate.overall_joint,
            },
            "global_threshold": candidate.global_threshold,
            "primary": candidate == primary,
        }
        for candidate in candidates
    ]
    calibration_scope = {
        "partition_reused_unchanged_from_h4": True,
        "familiar_legacy_values_present": True,
        "unseen_h4_surfaces_and_templates": True,
        "open_value_disjoint": False,
        "selection_role": "surface_and_template_transfer_ranking_only",
        "quality_decision_partition": "known_development",
    }
    result: dict[str, Any] = {
        "schema_version": H5_DEVELOPMENT_EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "training_data": {
            "schema_version": replay_mixture_data.MANIFEST_SCHEMA_VERSION,
            "generator": replay_mixture_data.GENERATOR_VERSION,
            "recipe": H5_TRAINING_RECIPE_VERSION,
            "selection_policy": replay_mixture_data.SELECTION_POLICY_VERSION,
            "manifest_sha256": training_bundle.manifest_sha256,
            "fit_sha256": training_bundle.input_sha256["fit"],
            "calibration_sha256": training_bundle.input_sha256["calibration"],
            "fit_records": len(training_bundle.fit),
            "fit_worlds": FIT_WORLD_COUNT,
            "fit_legacy_worlds": SOURCE_WORLD_COUNT,
            "fit_surface_worlds": SOURCE_WORLD_COUNT,
            "calibration_records": len(training_bundle.calibration),
            "development_records_used": 0,
            "calibration_scope": calibration_scope,
        },
        "partition": {
            "partition_id": DEVELOPMENT_PARTITION_ID,
            "role": "known_adaptive_development_quality_only",
            "manifest_sha256": development_bundle.manifest_sha256,
            "development_sha256": development_bundle.dev_sha256,
            "records": len(development_bundle.examples),
            "worlds": len(
                {example.world_id for example in development_bundle.examples}
            ),
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "used_for_epoch_selection": False,
            "used_for_seed_selection": False,
            "used_for_threshold_selection": False,
            "used_for_replay_selection": False,
            "historical_benchmark_read": False,
            "sealed_confirmation_read": False,
        },
        "artifacts": {
            "architecture_version": ARCHITECTURE_VERSION,
            "architecture_identity": FROZEN_NANO_V01.architecture_identity,
            "base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
            "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
            "trunk_parameter_count": NANO_TRUNK_PARAMETER_COUNT,
            "evidence_query_head_parameter_count": EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
            "parameter_count": NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
            "candidates": candidate_rows,
            "primary_label": primary.label,
        },
        "protocol": {
            "intervention": "fixed_50_50_fit_replay_mixture_only",
            "supervision_version": POINTER_SUPERVISION_VERSION,
            "state_class_order": [state.value for state in STATE_ORDER],
            "primary_selection_order": [
                "uncalibrated_calibration_macro_joint_desc",
                "uncalibrated_calibration_overall_joint_desc",
                "earlier_epoch",
                "seed_20260805",
            ],
            "global_threshold": primary.global_threshold,
            "threshold_source": "unchanged_h4_training_only_calibration",
            "threshold_policy": CALIBRATION_THRESHOLD_POLICY,
            "threshold_applied": calibrated_gate is not None,
            "calibration_scope": calibration_scope,
            "uncalibrated_modal_state_maximum_per_field": MODAL_STATE_MAXIMUM,
            "verifier_policy": "accept_or_reject_model_owned_proposals_only",
            "verifier_supplies_or_corrects_values": False,
            "batch_size": batch_size,
            "latency_measured": False,
            "sealed_confirmation_assessed": False,
        },
        "runtime": {
            "device": resolved_device,
            "deterministic_algorithms": True,
            **evaluation_runtime,
        },
        "source_sha256": source_hashes,
        "uncalibrated_raw": uncalibrated_raw,
        "uncalibrated_state_balance": state_balance,
        "uncalibrated_admission": uncalibrated_gate,
        "calibrated_raw": calibrated_raw,
        "calibrated_quality": calibrated_gate,
        "verifier_final": verifier_final_section,
        "verifier_final_quality": verifier_gate,
        "decision": {
            "uncalibrated_semantic_and_retention_passed": bool(
                uncalibrated_gate["semantic_and_retention_passed"]
            ),
            "uncalibrated_state_balance_passed": bool(state_balance["passed"]),
            "uncalibrated_admission_passed": admission_passed,
            "calibrated_raw_quality_passed": (
                None
                if calibrated_gate is None
                else bool(calibrated_gate["all_quality_gates_passed"])
            ),
            "verifier_final_quality_passed": (
                None
                if verifier_gate is None
                else bool(verifier_gate["all_quality_gates_passed"])
            ),
            "quality_gate_passed": quality_passed,
            "latency_assessed": False,
            "sealed_confirmation_assessed": False,
            "next_step": (
                "measure matched latency before sealed confirmation"
                if quality_passed
                else (
                    "retain H4 and H5 as negative results; stop before latency and "
                    "sealed confirmation; next change at most one representation mechanism"
                )
            ),
        },
        "selection_boundary": (
            "The replay bundle and both fixed-seed H5 reports, checkpoints, sources, "
            "runtime, primary epoch, seed, and threshold were authenticated and frozen "
            "before the separate known-development manifest was opened. The unchanged "
            "H4 calibration contains familiar legacy values and ranks transfer only to "
            "unseen H4 surfaces/templates. Uncalibrated semantic or field-balance "
            "failure stops before threshold and verifier; calibrated quality failure "
            "stops before verifier; verifier quality failure stops before latency and "
            "sealed confirmation."
        ),
    }
    _write_json_no_clobber(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate Nano's frozen H5 replay-mixture candidate"
    )
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--training-manifest-sha256", required=True)
    parser.add_argument("--development-data-dir", type=Path, required=True)
    parser.add_argument("--development-manifest-sha256", required=True)
    parser.add_argument(
        "--tokenizer", type=Path, default=root / "sft" / "tokenizer.json"
    )
    parser.add_argument(
        "--training-report",
        nargs=2,
        action="append",
        metavar=("PATH", "SHA256"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evaluate_development(
        training_data_dir=args.training_data_dir,
        training_manifest_sha256=args.training_manifest_sha256,
        development_data_dir=args.development_data_dir,
        development_manifest_sha256=args.development_manifest_sha256,
        tokenizer_path=args.tokenizer,
        training_reports=tuple(
            (Path(path), digest) for path, digest in args.training_report
        ),
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEVELOPMENT_RECORDS",
    "H5_DEVELOPMENT_EVALUATION_SCHEMA_VERSION",
    "MODAL_STATE_MAXIMUM",
    "EvidenceQueryCandidate",
    "EvidenceQueryEvaluationError",
    "EvidenceQueryH5Candidate",
    "authenticate_and_select_primary",
    "evaluate_development",
    "load_candidate_from_training_report",
]
