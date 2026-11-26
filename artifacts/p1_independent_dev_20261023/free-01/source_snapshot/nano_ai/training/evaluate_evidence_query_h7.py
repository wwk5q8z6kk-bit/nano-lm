"""One-pass scoring for the preregistered H7 boundary-margin screen.

The evaluator authenticates the frozen control, both treatment reports and
their report-selected checkpoints before opening the H7 screen.  It then scores
the control exactly once and each treatment seed exactly once with the shared,
uncalibrated H6 inference path.  H6 development is never read or rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import math
import os
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.contract import FIELD_ORDER
from nano_ai.training import h7_boundary_screen_data, train_evidence_query_h7
from nano_ai.training.diagnose_h6 import (
    H6DiagnosisError,
    _classify_field,
    _span_relation,
)
from nano_ai.training.evaluate_state_span import _fixture_cases
from nano_ai.training.evidence_query_inference import (
    batched_evidence_query_inference,
    build_pointer_inference_inputs,
    raw_pointer_diagnostics,
)
from nano_ai.training.pointer_data import (
    encode_pointer_partition,
    load_pointer_tokenizer,
)
from nano_ai.training.pointer_model import NANO_TRUNK_PARAMETER_COUNT
from nano_ai.training.state_conditioned_evidence_query_model import (
    ARCHITECTURE_VERSION,
    EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6,
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6,
    NanoStateConditionedEvidenceQueryPointerModel,
)
from nano_ai.training.state_span_data import StateSpanExample
from nano_ai.training.train_evidence_query import select_epoch_report

H7_SCREEN_EVALUATION_SCHEMA_VERSION = "nano.h7-boundary-screen-evaluation.v1"
H7_SCREEN_MANIFEST_SHA256 = (
    "a766a41ed99bcffcd8e078c652c8a85fe6d796fe44268e03e0e309d9070cd6a6"
)
H7_SCREEN_SHA256 = (
    "9b18c5eef105d0fa2ff0a1e575e764c0014a4ff62e40e8b8205dc12674c436f2"
)
CONTROL_CHECKPOINT_SHA256 = (
    "07547ec5c01b46fa87e8d53a40497a7c4b4a6bc92b596dddd67f4be4d18c89e0"
)
CONTROL_CHECKPOINT_BYTES = 13_168_916
DEFAULT_BATCH_SIZE = 32
FIELD_COLLAPSE_MODAL_MAXIMUM = 949

_REPORT_KEYS = frozenset(
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
_REQUIRED_HYPERPARAMETERS = {
    "epochs": 3,
    "batch_size": 32,
    "paired_variants_per_world": 4,
    "steps_per_epoch": 350,
    "total_steps": 1_050,
    "boundary_margin": 1.0,
    "boundary_radius": 2,
    "boundary_loss_weight": 0.25,
    "soft_state_conditioned_boundaries": True,
    "state_posterior_detached": True,
    "gold_state_teacher_forced": False,
    "patient_token_masked": True,
    "deterministic_algorithms": True,
    "world_grouped_batches": True,
}


class H7EvaluationError(ValueError):
    """A frozen H7 input, report, checkpoint, or result invariant failed."""


@dataclass(frozen=True, slots=True)
class H7Candidate:
    label: str
    path: Path
    sha256: str
    artifact_bytes: int
    seed: int | None
    epoch: int | None
    report_sha256: str | None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise H7EvaluationError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _read_regular_file(path: Path, *, role: str) -> bytes:
    try:
        with Path(path).open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise H7EvaluationError(f"{role} is not a regular file")
            return handle.read()
    except H7EvaluationError:
        raise
    except OSError as exc:
        raise H7EvaluationError(f"{role} is unavailable") from exc


def _read_verified_file(path: Path, expected: str, *, role: str) -> bytes:
    expected_digest = _require_sha256(expected, role)
    payload = _read_regular_file(path, role=role)
    observed = _sha256(payload)
    if not hmac.compare_digest(observed, expected_digest):
        raise H7EvaluationError(
            f"{role} SHA-256 mismatch: expected {expected_digest}, observed {observed}"
        )
    return payload


def _reject_json_constant(value: str) -> None:
    raise H7EvaluationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise H7EvaluationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, *, role: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except H7EvaluationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise H7EvaluationError(f"{role} is invalid JSON") from exc


def _checkpoint_record(value: object, *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CHECKPOINT_KEYS:
        raise H7EvaluationError(f"{role} checkpoint record is invalid")
    filename = value["filename"]
    byte_count = value["bytes"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise H7EvaluationError(f"{role} checkpoint filename is unsafe")
    _require_sha256(value["sha256"], f"{role} checkpoint")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
        raise H7EvaluationError(f"{role} checkpoint byte count is invalid")
    return value


def _validate_treatment_report(
    report_path: str | Path,
    *,
    expected_report_sha256: str,
) -> H7Candidate:
    path = Path(report_path)
    snapshot = _read_verified_file(
        path, expected_report_sha256, role="H7 treatment training report"
    )
    report = _parse_json(snapshot, role="H7 treatment training report")
    if not isinstance(report, Mapping) or set(report) != _REPORT_KEYS:
        raise H7EvaluationError("H7 treatment report schema is invalid")
    if (
        report["schema_version"]
        != train_evidence_query_h7.H7_TRAINING_REPORT_SCHEMA_VERSION
        or report["recipe"] != train_evidence_query_h7.H7_TRAINING_RECIPE_VERSION
        or report["status"] != "complete"
    ):
        raise H7EvaluationError("H7 treatment report is not a complete frozen run")
    seed = report["seed"]
    if isinstance(seed, bool) or seed not in train_evidence_query_h7.TRAINING_SEEDS:
        raise H7EvaluationError("H7 treatment seed is not frozen")
    if (
        report["architecture_version"] != ARCHITECTURE_VERSION
        or report["parameter_count"] != NANO_EVIDENCE_QUERY_PARAMETER_COUNT_H6
        or report["trunk_parameter_count"] != NANO_TRUNK_PARAMETER_COUNT
        or report["evidence_query_head_parameter_count"]
        != EVIDENCE_QUERY_HEAD_PARAMETER_COUNT_H6
        or report["architecture_identity"] != FROZEN_NANO_V01.architecture_identity
        or report["base_checkpoint_sha256"] != FROZEN_NANO_V01.checkpoint_sha256
        or report["tokenizer_sha256"] != FROZEN_NANO_V01.tokenizer_sha256
        or report["dataset_manifest_sha256"]
        != train_evidence_query_h7.H5_DATASET_FILE_IDENTITY["manifest"]["sha256"]
    ):
        raise H7EvaluationError("H7 treatment model or dataset identity changed")
    if (
        report["dev_used_for_selection"] is not False
        or report["legacy_record_artifact_accessed"] is not False
        or report["fresh_v1_accessed"] is not False
    ):
        raise H7EvaluationError("H7 treatment crossed a forbidden data boundary")

    dataset = report["dataset"]
    if not isinstance(dataset, Mapping):
        raise H7EvaluationError("H7 treatment dataset record is invalid")
    for role in ("manifest", "fit", "calibration"):
        record = dataset.get(f"source_{role}")
        expected = train_evidence_query_h7.H5_DATASET_FILE_IDENTITY[role]
        if not isinstance(record, Mapping) or any(
            record.get(key) != value for key, value in expected.items()
        ):
            raise H7EvaluationError(f"H7 treatment {role} identity changed")

    hyperparameters = report["hyperparameters"]
    if not isinstance(hyperparameters, Mapping) or any(
        hyperparameters.get(key) != value
        for key, value in _REQUIRED_HYPERPARAMETERS.items()
    ):
        raise H7EvaluationError("H7 treatment hyperparameters changed")
    if report["preserved_source_sha256"] != dict(
        train_evidence_query_h7._require_preserved_h3_sources()
    ):
        raise H7EvaluationError("H7 preserved source identity changed")
    if report["changed_source_sha256"] != dict(
        train_evidence_query_h7._changed_source_hashes()
    ):
        raise H7EvaluationError("H7 changed source identity changed")
    if report["selection_note"] != train_evidence_query_h7._TRAINING_SELECTION_NOTE:
        raise H7EvaluationError("H7 treatment selection policy changed")

    epochs = report["epochs"]
    if not isinstance(epochs, list) or len(epochs) != 3:
        raise H7EvaluationError("H7 treatment report must contain three epochs")
    for expected_epoch, row in enumerate(epochs, 1):
        if not isinstance(row, Mapping) or row.get("epoch") != expected_epoch:
            raise H7EvaluationError("H7 treatment epochs are not ordered")
        _checkpoint_record(row.get("checkpoint"), role=f"H7 epoch {expected_epoch}")
        for key in (
            "train_loss",
            "state_loss",
            "pointer_loss",
            "base_loss",
            "boundary_loss",
            "boundary_start_loss",
            "boundary_end_loss",
            "seconds",
        ):
            metric = row.get(key)
            if type(metric) not in {int, float} or not math.isfinite(metric) or metric < 0:
                raise H7EvaluationError(f"H7 epoch {key} is invalid")
    selected = select_epoch_report(epochs)
    candidate = report["candidate"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "epoch",
        *_CHECKPOINT_KEYS,
    }:
        raise H7EvaluationError("H7 selected candidate record is invalid")
    expected_candidate = {"epoch": selected["epoch"], **selected["checkpoint"]}
    if dict(candidate) != expected_candidate:
        raise H7EvaluationError("H7 treatment was not selected by frozen calibration")
    calibration = report["calibration"]
    if not isinstance(calibration, Mapping):
        raise H7EvaluationError("H7 treatment calibration record is invalid")
    if calibration.get("selected_epoch") != selected["epoch"]:
        raise H7EvaluationError("H7 treatment calibration disagrees with selection")
    checkpoint = _checkpoint_record(
        {key: candidate[key] for key in _CHECKPOINT_KEYS}, role="H7 selected"
    )
    checkpoint_snapshot = _read_verified_file(
        path.parent / checkpoint["filename"],
        checkpoint["sha256"],
        role=f"H7 treatment seed {seed} checkpoint",
    )
    if len(checkpoint_snapshot) != checkpoint["bytes"]:
        raise H7EvaluationError("H7 treatment checkpoint byte count changed")
    return H7Candidate(
        label=f"treatment-seed-{seed}",
        path=path.parent / checkpoint["filename"],
        sha256=checkpoint["sha256"],
        artifact_bytes=checkpoint["bytes"],
        seed=int(seed),
        epoch=int(candidate["epoch"]),
        report_sha256=_sha256(snapshot),
    )


def _control_candidate(path: str | Path) -> H7Candidate:
    snapshot = _read_verified_file(
        Path(path), CONTROL_CHECKPOINT_SHA256, role="H7 frozen H6 control checkpoint"
    )
    if len(snapshot) != CONTROL_CHECKPOINT_BYTES:
        raise H7EvaluationError("H7 control checkpoint byte count changed")
    return H7Candidate(
        label="control-h6-seed-20260805-epoch-2",
        path=Path(path),
        sha256=CONTROL_CHECKPOINT_SHA256,
        artifact_bytes=CONTROL_CHECKPOINT_BYTES,
        seed=None,
        epoch=2,
        report_sha256=None,
    )


def _load_model(candidate: H7Candidate) -> NanoStateConditionedEvidenceQueryPointerModel:
    snapshot = _read_verified_file(
        candidate.path, candidate.sha256, role=f"H7 {candidate.label} checkpoint"
    )
    if len(snapshot) != candidate.artifact_bytes:
        raise H7EvaluationError(f"H7 {candidate.label} byte count changed")
    try:
        state_dict = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise H7EvaluationError(f"H7 {candidate.label} could not be loaded safely") from exc
    if not isinstance(state_dict, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in state_dict.items()
    ):
        raise H7EvaluationError(f"H7 {candidate.label} is not a tensor state mapping")
    model = NanoStateConditionedEvidenceQueryPointerModel()
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise H7EvaluationError(
            f"H7 {candidate.label} does not match {ARCHITECTURE_VERSION}"
        ) from exc
    return model.eval()


def _load_screen(screen_dir: str | Path) -> tuple[Mapping[str, Any], tuple[StateSpanExample, ...]]:
    root = Path(screen_dir)
    manifest_snapshot = _read_verified_file(
        root / "manifest.json", H7_SCREEN_MANIFEST_SHA256, role="H7 screen manifest"
    )
    screen_snapshot = _read_verified_file(
        root / "screen.jsonl", H7_SCREEN_SHA256, role="H7 screen records"
    )
    manifest = _parse_json(manifest_snapshot, role="H7 screen manifest")
    if not isinstance(manifest, Mapping):
        raise H7EvaluationError("H7 screen manifest is not an object")
    if (
        manifest.get("schema_version") != h7_boundary_screen_data.MANIFEST_SCHEMA_VERSION
        or manifest.get("generator") != h7_boundary_screen_data.GENERATOR_VERSION
        or manifest.get("generator_sha256")
        != _sha256(Path(h7_boundary_screen_data.__file__).read_bytes())
        or manifest.get("control_checkpoint_sha256") != CONTROL_CHECKPOINT_SHA256
        or manifest.get("screen", {}).get("sha256") != H7_SCREEN_SHA256
        or manifest.get("isolation", {}).get("evaluation_only") is not True
        or manifest.get("isolation", {}).get("private_or_phi_records") != 0
    ):
        raise H7EvaluationError("H7 screen manifest contract changed")
    records: list[StateSpanExample] = []
    for line_number, line in enumerate(screen_snapshot.splitlines(), 1):
        try:
            value = _parse_json(line, role=f"H7 screen line {line_number}")
            records.append(StateSpanExample.from_dict(value))
        except (TypeError, ValueError) as exc:
            raise H7EvaluationError(
                f"H7 screen has an invalid record at line {line_number}"
            ) from exc
    frozen = tuple(records)
    if frozen != h7_boundary_screen_data.generate_screen():
        raise H7EvaluationError("H7 screen records do not equal frozen generation")
    if len(frozen) != h7_boundary_screen_data.SCREEN_RECORDS:
        raise H7EvaluationError("H7 screen record count changed")
    return manifest, frozen


def _resolve_device(device: str) -> str:
    if device not in {"cpu", "mps", "cuda"}:
        raise H7EvaluationError("device must be cpu, mps, or cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise H7EvaluationError("MPS was requested but is unavailable")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise H7EvaluationError("CUDA was requested but is unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise H7EvaluationError("deterministic CUDA cuBLAS configuration is missing")
    return device


def _mechanism_summary(raw: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = raw.get("item_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        raise H7EvaluationError("H7 raw diagnostics are missing item rows")
    axes: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    raw_states: defaultdict[str, Counter[str]] = defaultdict(Counter)
    try:
        for row_index, row in enumerate(diagnostics):
            if not isinstance(row, Mapping):
                raise H7EvaluationError(f"H7 item row {row_index} is invalid")
            fields = row.get("fields")
            if fields is None:
                continue
            if not isinstance(fields, list) or len(fields) != len(FIELD_ORDER):
                raise H7EvaluationError(f"H7 item row {row_index} fields are invalid")
            for field_index, field in enumerate(fields):
                if not isinstance(field, Mapping):
                    raise H7EvaluationError("H7 field diagnostic is invalid")
                role = f"row[{row_index}].fields[{field_index}]"
                axis = _classify_field(field, role=role)
                axes[axis] += 1
                raw_states[str(field["field"])][str(field["raw_state"])] += 1
                relation = _span_relation(field, role=role)
                if relation is not None:
                    relations[relation] += 1
    except H6DiagnosisError as exc:
        raise H7EvaluationError("H7 diagnostic taxonomy could not be applied") from exc
    field_collapse = {
        field: max(states.values(), default=0)
        for field, states in sorted(raw_states.items())
        if max(states.values(), default=0) > FIELD_COLLAPSE_MODAL_MAXIMUM
    }
    disjoint_or_speaker = sum(
        count
        for relation, count in relations.items()
        if "disjoint" in relation or "speaker_mismatch" in relation
    )
    return {
        "field_total": sum(axes.values()),
        "joint_correct_fields": axes["joint_correct"],
        "state_only_error_fields": axes["state_only_error"],
        "binding_only_error_fields": axes["binding_only_error"],
        "combined_error_fields": axes["combined_error"],
        "binding_error_fields": sum(relations.values()),
        "span_relations": dict(sorted(relations.items())),
        "disjoint_or_speaker_mismatch_fields": disjoint_or_speaker,
        "field_collapse_count": len(field_collapse),
        "field_collapse_modal_counts": field_collapse,
        "raw_state_counts_by_field": {
            field: dict(sorted(states.items()))
            for field, states in sorted(raw_states.items())
        },
    }


def _score_candidate(
    candidate: H7Candidate,
    model: NanoStateConditionedEvidenceQueryPointerModel,
    *,
    examples: Sequence[StateSpanExample],
    encoded: Sequence[Any],
    cases: Sequence[Any],
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    inputs = build_pointer_inference_inputs(examples, encoded)
    model = model.to(device).eval()
    inference = batched_evidence_query_inference(
        model, inputs, device=device, batch_size=batch_size
    )
    model.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()
    raw = raw_pointer_diagnostics(examples, cases, inference.predictions)
    return {
        "candidate": {
            "label": candidate.label,
            "seed": candidate.seed,
            "epoch": candidate.epoch,
            "checkpoint_sha256": candidate.sha256,
            "checkpoint_bytes": candidate.artifact_bytes,
            "training_report_sha256": candidate.report_sha256,
        },
        "metrics": {
            "joint_exact_accuracy": raw["fields"]["joint_exact_accuracy"],
            "state_accuracy": raw["fields"]["state_accuracy"],
            "decode_failure_items": raw["decode_failure_items"],
        },
        "mechanism": _mechanism_summary(raw),
        "raw": raw,
    }


def _comparison(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    control_rows = control["raw"]["item_diagnostics"]
    treatment_rows = treatment["raw"]["item_diagnostics"]
    if len(control_rows) != len(treatment_rows):
        raise H7EvaluationError("H7 arm row counts do not match")
    shared = 0
    control_span_exact = 0
    treatment_span_exact = 0
    for control_row, treatment_row in zip(control_rows, treatment_rows, strict=True):
        if control_row["example_id"] != treatment_row["example_id"]:
            raise H7EvaluationError("H7 arm row identities do not match")
        control_fields = control_row["fields"]
        treatment_fields = treatment_row["fields"]
        if control_fields is None or treatment_fields is None:
            continue
        for control_field, treatment_field in zip(
            control_fields, treatment_fields, strict=True
        ):
            if control_field["field"] != treatment_field["field"]:
                raise H7EvaluationError("H7 arm field identities do not match")
            if control_field["state_correct"] and treatment_field["state_correct"]:
                shared += 1
                control_span_exact += int(control_field["span_exact"])
                treatment_span_exact += int(treatment_field["span_exact"])
    if shared < 1:
        raise H7EvaluationError("H7 shared-state-correct cohort is empty")
    control_binding = control["mechanism"]["binding_error_fields"]
    treatment_binding = treatment["mechanism"]["binding_error_fields"]
    relative_reduction = (
        (control_binding - treatment_binding) / control_binding
        if control_binding
        else (0.0 if treatment_binding == 0 else -math.inf)
    )
    control_shared_rate = control_span_exact / shared
    treatment_shared_rate = treatment_span_exact / shared
    state_delta = (
        treatment["metrics"]["state_accuracy"] - control["metrics"]["state_accuracy"]
    )
    safety_delta = (
        treatment["mechanism"]["disjoint_or_speaker_mismatch_fields"]
        - control["mechanism"]["disjoint_or_speaker_mismatch_fields"]
    )
    no_failures = treatment["metrics"]["decode_failure_items"] == 0
    no_collapse = treatment["mechanism"]["field_collapse_count"] == 0
    state_noninferior = state_delta >= -0.01 or math.isclose(
        state_delta, -0.01, rel_tol=0.0, abs_tol=1e-12
    )
    bars = {
        "binding_relative_reduction_at_least_0_20": relative_reduction >= 0.20,
        "shared_state_span_gain_at_least_0_05": (
            treatment_shared_rate - control_shared_rate >= 0.05
        ),
        "state_noninferiority_minus_0_01": state_noninferior,
        "no_disjoint_or_speaker_mismatch_increase": safety_delta <= 0,
        "zero_decode_failures": no_failures,
        "no_field_collapse": no_collapse,
    }
    return {
        "treatment_seed": treatment["candidate"]["seed"],
        "binding_error_fields": {
            "control": control_binding,
            "treatment": treatment_binding,
            "relative_reduction": relative_reduction,
        },
        "span_exact_on_shared_state_correct": {
            "denominator": shared,
            "control_numerator": control_span_exact,
            "control_rate": control_shared_rate,
            "treatment_numerator": treatment_span_exact,
            "treatment_rate": treatment_shared_rate,
            "absolute_gain": treatment_shared_rate - control_shared_rate,
        },
        "state_accuracy_delta": state_delta,
        "disjoint_or_speaker_mismatch_delta": safety_delta,
        "bars": bars,
        "all_pass_bars": all(bars.values()),
        "safety_or_state_violation": not all(
            bars[name]
            for name in (
                "state_noninferiority_minus_0_01",
                "no_disjoint_or_speaker_mismatch_increase",
                "zero_decode_failures",
                "no_field_collapse",
            )
        ),
        "boundary_improvement_bars_met": all(
            bars[name]
            for name in (
                "binding_relative_reduction_at_least_0_20",
                "shared_state_span_gain_at_least_0_05",
            )
        ),
    }


def _decision(comparisons: Sequence[Mapping[str, Any]]) -> str:
    if len(comparisons) != 2:
        raise H7EvaluationError("H7 decision requires exactly two treatment seeds")
    if all(item["all_pass_bars"] for item in comparisons):
        return "PASS"
    if any(item["safety_or_state_violation"] for item in comparisons):
        return "REJECT"
    if not any(item["boundary_improvement_bars_met"] for item in comparisons):
        return "REJECT"
    return "GRAY"


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise H7EvaluationError("H7 evaluation output could not be written") from exc


def evaluate_h7_screen(
    *,
    screen_dir: str | Path,
    control_checkpoint: str | Path,
    tokenizer_path: str | Path,
    treatment_reports: Sequence[tuple[str | Path, str]],
    output_path: str | Path,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Mapping[str, Any]:
    """Authenticate all arms, then open and score the frozen screen once."""

    output = Path(output_path)
    if output.exists() or not output.parent.is_dir():
        raise H7EvaluationError("H7 output must be a new file in an existing directory")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise H7EvaluationError("batch_size must be a positive integer")
    if len(treatment_reports) != 2:
        raise H7EvaluationError("exactly two H7 treatment reports are required")
    resolved_device = _resolve_device(device)

    # Selection and checkpoint authentication must finish before screen access.
    control = _control_candidate(control_checkpoint)
    treatments = tuple(
        _validate_treatment_report(path, expected_report_sha256=digest)
        for path, digest in treatment_reports
    )
    if {candidate.seed for candidate in treatments} != set(
        train_evidence_query_h7.TRAINING_SEEDS
    ):
        raise H7EvaluationError("H7 treatment reports do not cover both fixed seeds")
    if len({candidate.sha256 for candidate in treatments}) != 2:
        raise H7EvaluationError("H7 treatment checkpoints must be distinct")
    candidates = (control, *sorted(treatments, key=lambda item: int(item.seed or 0)))
    models = {candidate.label: _load_model(candidate) for candidate in candidates}
    try:
        tokenizer = load_pointer_tokenizer(Path(tokenizer_path))
    except Exception as exc:
        raise H7EvaluationError("H7 frozen tokenizer could not be authenticated") from exc

    manifest, examples = _load_screen(screen_dir)
    try:
        encoded = encode_pointer_partition(tokenizer, examples, expected_split="dev")
        cases = _fixture_cases(examples)
    except Exception as exc:
        raise H7EvaluationError("H7 screen could not be encoded") from exc

    scored = [
        _score_candidate(
            candidate,
            models[candidate.label],
            examples=examples,
            encoded=encoded,
            cases=cases,
            device=resolved_device,
            batch_size=batch_size,
        )
        for candidate in candidates
    ]
    control_score = scored[0]
    comparisons = [_comparison(control_score, item) for item in scored[1:]]
    report = {
        "schema_version": H7_SCREEN_EVALUATION_SCHEMA_VERSION,
        "experiment_id": "nano-p1-h7-boundary-margin-screen-v1",
        "status": "complete",
        "screen": {
            "manifest_sha256": H7_SCREEN_MANIFEST_SHA256,
            "records_sha256": H7_SCREEN_SHA256,
            "records": len(examples),
            "manifest": manifest,
        },
        "runtime": {
            "device": resolved_device,
            "batch_size": batch_size,
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if resolved_device == "cuda"
                else None
            ),
        },
        "arms": scored,
        "comparisons": comparisons,
        "decision": _decision(comparisons),
        "termination": {
            "control_scoring_passes": 1,
            "treatment_scoring_passes": 2,
            "third_seed_allowed": False,
            "threshold_sweep_performed": False,
            "h6_development_opened": False,
        },
    }
    _write_json_no_clobber(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Nano's frozen H7 screen once")
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--treatment-report",
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
    evaluate_h7_screen(
        screen_dir=args.screen_dir,
        control_checkpoint=args.control_checkpoint,
        tokenizer_path=args.tokenizer,
        treatment_reports=tuple(
            (Path(path), digest) for path, digest in args.treatment_report
        ),
        output_path=args.output,
        device=args.device,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONTROL_CHECKPOINT_SHA256",
    "H7_SCREEN_EVALUATION_SCHEMA_VERSION",
    "H7_SCREEN_MANIFEST_SHA256",
    "H7_SCREEN_SHA256",
    "H7EvaluationError",
    "evaluate_h7_screen",
]
