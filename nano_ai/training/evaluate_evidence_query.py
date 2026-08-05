"""Known-development evaluation for Nano's preregistered H3 intervention.

The two fixed-seed checkpoints are authenticated and the primary checkpoint is
selected from training-only, uncalibrated calibration results *before* this
module loads the known adaptive development partition.  Development can only
decide the frozen quality gates.  It cannot select an epoch, seed, confidence
threshold, loss weight, or any fresh confirmation input.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import math
import os
import platform
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.adapters.evidence_query_pointer import EvidenceQueryPointerSolver
from nano_ai.adapters.state_span import StateSpanProposal
from nano_ai.evaluation import EvaluationReport, evaluate_solver
from nano_ai.training import (
    evidence_query_inference,
    evidence_query_model,
    pointer_data,
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
    PointerDecodeError,
    PointerInferenceInput,
    PointerPrediction,
    apply_global_threshold,
    batched_evidence_query_inference,
    build_pointer_inference_inputs,
    raw_pointer_diagnostics,
)
from nano_ai.training.evidence_query_model import (
    ARCHITECTURE_VERSION,
    EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
    NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
    NanoEvidenceQueryPointerModel,
)
from nano_ai.training.pointer_data import (
    POINTER_PROMPT_TEMPLATE_ID,
    POINTER_SUPERVISION_VERSION,
    STATE_ORDER,
    encode_pointer_partition,
    load_pointer_tokenizer,
)
from nano_ai.training.pointer_model import NANO_TRUNK_PARAMETER_COUNT
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    DEV_WORLDS,
    TARGET_GRAMMAR_VERSION,
    TRAIN_WORLDS,
    canonical_json_bytes,
)
from nano_ai.training.train_evidence_query import (
    CALIBRATION_THRESHOLD_POLICY,
    expected_training_source_paths,
)

EVIDENCE_QUERY_DEVELOPMENT_EVALUATION_SCHEMA_VERSION = (
    "nano.evidence-query-development-evaluation.v0"
)
EVIDENCE_QUERY_TRAINING_REPORT_SCHEMA_VERSION = "nano.evidence-query-training-report.v0"
EVIDENCE_QUERY_TRAINING_RECIPE_VERSION = "nano-evidence-query-architecture-only-v0"
TRAINING_SEEDS = (20260805, 20260806)
DEFAULT_BATCH_SIZE = 32
_TRAINING_SELECTION_NOTE = (
    "The selected H3 epoch was chosen using calibration worlds 2800-2999. "
    "Gradients used only disjoint fit worlds 0000-2799. The inspected "
    "development partition was authenticated as part of the frozen H2 data "
    "bundle but received no inference, loss, threshold, or selection access. "
    "Historical fresh-v0 and sealed fresh-v1 were not read."
)

_REPORT_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "recipe",
        "status",
        "seed",
        "device",
        "architecture_version",
        "architecture_identity",
        "parameter_count",
        "trunk_parameter_count",
        "evidence_query_head_parameter_count",
        "base_checkpoint_sha256",
        "tokenizer_sha256",
        "dataset_manifest_sha256",
        "dataset",
        "hyperparameters",
        "epochs",
        "candidate",
        "calibration",
        "source_sha256",
        "runtime",
        "selection_note",
        "dev_used_for_selection",
        "fresh_v1_accessed",
    }
)
_CHECKPOINT_KEYS = frozenset({"filename", "sha256", "bytes"})
_RANKING_KEYS = frozenset({"macro_joint", "overall_joint"})
_CALIBRATION_KEYS = frozenset(
    {"uncalibrated", "global_threshold", "calibrated", "threshold_policy"}
)
_CALIBRATION_PHASE_KEYS = frozenset({"slices", "selection", "wrong_presented"})
_CALIBRATION_SLICE_KEYS = frozenset(
    {
        "overall",
        "absence",
        "missing_target",
        "uncertain_target",
        "conflicting_target",
    }
)
_CALIBRATION_SLICE_DENOMINATORS = {
    "overall": 4_000,
    "absence": 330,
    "missing_target": 200,
    "uncertain_target": 200,
    "conflicting_target": 200,
}
_SEMANTIC_GATES = {
    "overall": (3_041, 5_000),
    "held_value": (1_905, 2_987),
    "missing_target": (219, 250),
    "absence": (383, 413),
    "conflict_target": (236, 250),
    "uncertain_target": (228, 250),
}


class EvidenceQueryEvaluationError(ValueError):
    """An H3 report, artifact, partition, or gate invariant failed."""


@dataclass(frozen=True, slots=True)
class EvidenceQueryCandidate:
    """One report-selected H3 checkpoint with training-only calibration facts."""

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
        """Frozen primary ordering; development is intentionally absent."""

        return (
            self.macro_joint,
            self.overall_joint,
            -self.epoch,
            int(self.seed == TRAINING_SEEDS[0]),
        )


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
    try:
        with Path(path).open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise EvidenceQueryEvaluationError(f"{role} is not a regular file")
            return handle.read()
    except EvidenceQueryEvaluationError:
        raise
    except OSError as exc:
        raise EvidenceQueryEvaluationError(f"{role} is unavailable") from exc


def _read_verified_file(path: Path, expected_sha256: str, *, role: str) -> bytes:
    expected = _require_sha256(expected_sha256, role)
    snapshot = _read_regular_file(path, role=role)
    observed = _sha256(snapshot)
    if not hmac.compare_digest(observed, expected):
        raise EvidenceQueryEvaluationError(
            f"{role} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return snapshot


def _reject_json_constant(value: str) -> None:
    raise EvidenceQueryEvaluationError(
        f"non-finite JSON constant is forbidden: {value}"
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceQueryEvaluationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(snapshot: bytes, *, role: str) -> Any:
    try:
        return json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except EvidenceQueryEvaluationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise EvidenceQueryEvaluationError(f"{role} is invalid JSON") from exc


def _finite_rate(value: object, role: str) -> float:
    if (
        type(value) not in {int, float}
        or not math.isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise EvidenceQueryEvaluationError(f"{role} must be a finite rate in [0, 1]")
    return float(value)


def _training_source_paths() -> dict[str, Path]:
    """Use the trainer's exact content-addressed source boundary."""

    return {name: Path(path) for name, path in expected_training_source_paths().items()}


def _training_source_hashes() -> dict[str, str]:
    return {
        name: _sha256(_read_regular_file(path, role=f"training {name} source"))
        for name, path in sorted(_training_source_paths().items())
    }


def _checkpoint_record(value: object) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_KEYS:
        raise EvidenceQueryEvaluationError("H3 candidate checkpoint record is invalid")
    filename = value["filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise EvidenceQueryEvaluationError("H3 checkpoint filename is unsafe")
    _require_sha256(value["sha256"], "H3 checkpoint")
    byte_count = value["bytes"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise EvidenceQueryEvaluationError("H3 checkpoint byte count is invalid")
    return value


def _validate_hyperparameters(value: object) -> None:
    if not isinstance(value, dict):
        raise EvidenceQueryEvaluationError("H3 hyperparameters are invalid")
    expected = {
        "epochs": 3,
        "batch_size": 32,
        "paired_variants_per_world": 4,
        "peak_learning_rate": 1.5e-4,
        "warmup_fraction": 0.03,
        "cosine_floor": 0.1,
        "weight_decay": 0.1,
        "gradient_clip": 1.0,
        "adam_betas": [0.9, 0.95],
        "adam_epsilon": 1e-8,
        "steps_per_epoch": 350,
        "total_steps": 1_050,
        "state_class_order": [state.value for state in STATE_ORDER],
        "state_class_weight_source_counts": {
            "supported": 46_050,
            "absent": 4_950,
            "missing": 3_000,
            "uncertain": 3_000,
            "conflicting": 3_000,
        },
        "state_class_weights": [
            0.26058631921824105,
            2.4242424242424243,
            4.0,
            4.0,
            4.0,
        ],
        "state_loss_weight": 1.0,
        "state_loss_definition": (
            "train_inverse_frequency_weighted_cross_entropy_mean_by_weight_mass"
        ),
        "pointer_loss_weight": 1.0,
        "pointer_loss_definition": ("mean_of_start_and_end_cross_entropy_active_slots"),
        "patient_token_masked": True,
        "prompt_template_id": POINTER_PROMPT_TEMPLATE_ID,
        "supervision_version": POINTER_SUPERVISION_VERSION,
        "uncertain_pointer_count": 1,
        "full_context_evidence_queries": True,
        "shared_state_classifier": True,
        "deterministic_algorithms": True,
        "full_trunk_trainable": True,
        "world_grouped_batches": True,
    }
    if value != expected:
        raise EvidenceQueryEvaluationError(
            "H3 hyperparameters do not match the architecture-only recipe"
        )


def _validate_runtime(value: object, *, device: object) -> None:
    keys = {
        "python",
        "torch",
        "tokenizers",
        "cuda",
        "gpu",
        "cublas_workspace_config",
        "platform",
        "seconds",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise EvidenceQueryEvaluationError("H3 training runtime record is invalid")
    if device not in {"cpu", "mps", "cuda"}:
        raise EvidenceQueryEvaluationError("H3 training device is invalid")
    for name in ("python", "torch", "tokenizers", "platform"):
        if not isinstance(value[name], str) or not value[name]:
            raise EvidenceQueryEvaluationError(f"H3 training runtime {name} is invalid")
    for name in ("cuda", "gpu"):
        if value[name] is not None and (
            not isinstance(value[name], str) or not value[name]
        ):
            raise EvidenceQueryEvaluationError(f"H3 training runtime {name} is invalid")
    if device == "cuda" and (
        value["cuda"] is None
        or value["gpu"] is None
        or value["cublas_workspace_config"] not in {":4096:8", ":16:8"}
    ):
        raise EvidenceQueryEvaluationError("H3 CUDA runtime identity is incomplete")
    if device != "cuda" and (
        value["gpu"] is not None or value["cublas_workspace_config"] is not None
    ):
        raise EvidenceQueryEvaluationError(
            "non-CUDA H3 training reported CUDA-only runtime state"
        )
    seconds = value["seconds"]
    if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0:
        raise EvidenceQueryEvaluationError("H3 runtime seconds is invalid")


def _calibration_metric(value: object, role: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "numerator",
        "denominator",
        "rate",
    }:
        raise EvidenceQueryEvaluationError(f"{role} calibration metric is invalid")
    numerator = value["numerator"]
    denominator = value["denominator"]
    rate = value["rate"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator < 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 1
        or numerator > denominator
        or type(rate) not in {int, float}
        or not math.isfinite(rate)
        or not math.isclose(
            float(rate), numerator / denominator, rel_tol=0.0, abs_tol=1e-15
        )
    ):
        raise EvidenceQueryEvaluationError(f"{role} calibration metric is inconsistent")
    return value


def _calibration_phase(value: object, role: str) -> tuple[float, float, int]:
    if not isinstance(value, dict) or set(value) != _CALIBRATION_PHASE_KEYS:
        raise EvidenceQueryEvaluationError(f"{role} calibration phase is invalid")
    slices = value["slices"]
    if not isinstance(slices, dict) or set(slices) != _CALIBRATION_SLICE_KEYS:
        raise EvidenceQueryEvaluationError(f"{role} calibration slices are invalid")
    checked = {
        name: _calibration_metric(metric, f"{role} {name}")
        for name, metric in slices.items()
    }
    if any(
        checked[name]["denominator"] != expected
        for name, expected in _CALIBRATION_SLICE_DENOMINATORS.items()
    ):
        raise EvidenceQueryEvaluationError(
            f"{role} calibration slice denominators changed"
        )
    selection = value["selection"]
    if not isinstance(selection, dict) or set(selection) != _RANKING_KEYS:
        raise EvidenceQueryEvaluationError(f"{role} calibration ranking is invalid")
    macro = _finite_rate(selection["macro_joint"], f"{role} macro_joint")
    overall = _finite_rate(selection["overall_joint"], f"{role} overall_joint")
    expected_macro = (
        sum(
            float(checked[name]["rate"])
            for name in (
                "absence",
                "missing_target",
                "uncertain_target",
                "conflicting_target",
            )
        )
        / 4.0
    )
    if not math.isclose(macro, expected_macro, rel_tol=0.0, abs_tol=1e-15):
        raise EvidenceQueryEvaluationError(f"{role} calibration macro disagrees")
    if not math.isclose(
        overall,
        float(checked["overall"]["rate"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise EvidenceQueryEvaluationError(f"{role} calibration overall disagrees")
    wrong = value["wrong_presented"]
    if not isinstance(wrong, dict) or set(wrong) != {
        "numerator",
        "denominator",
        "rate",
    }:
        raise EvidenceQueryEvaluationError(
            f"{role} calibration wrong_presented is invalid"
        )
    numerator = wrong["numerator"]
    denominator = wrong["denominator"]
    rate = wrong["rate"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or numerator < 0
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator < 0
        or numerator > denominator
        or type(rate) not in {int, float}
        or not math.isfinite(rate)
        or not math.isclose(
            float(rate),
            numerator / denominator if denominator else 0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise EvidenceQueryEvaluationError(
            f"{role} calibration wrong_presented is inconsistent"
        )
    if denominator > checked["overall"]["denominator"]:
        raise EvidenceQueryEvaluationError(
            f"{role} calibration presented denominator is impossible"
        )
    return macro, overall, numerator


def _validate_report_dataset_metadata(
    value: object,
    *,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "target_grammar",
        "source_manifest",
        "source_train",
        "source_dev",
        "fit",
        "calibration",
    }:
        raise EvidenceQueryEvaluationError("H3 training dataset record is invalid")
    if (
        value["schema_version"] != DATASET_SCHEMA_VERSION
        or value["target_grammar"] != TARGET_GRAMMAR_VERSION
    ):
        raise EvidenceQueryEvaluationError("H3 dataset protocol identity is invalid")

    expected_sources = {
        "source_manifest": {
            "keys": {"filename", "bytes", "sha256"},
            "filename": "manifest.json",
            "sha256": expected_manifest_sha256,
        },
        "source_train": {
            "keys": {"filename", "bytes", "sha256", "records", "worlds"},
            "filename": "train.jsonl",
            "records": TRAIN_WORLDS * 4,
            "worlds": TRAIN_WORLDS,
        },
        "source_dev": {
            "keys": {
                "filename",
                "bytes",
                "sha256",
                "records",
                "worlds",
                "usage",
            },
            "filename": "dev.jsonl",
            "records": DEV_WORLDS * 4,
            "worlds": DEV_WORLDS,
            "usage": "source_authentication_only",
        },
    }
    for name, expected in expected_sources.items():
        source = value[name]
        keys = expected["keys"]
        if not isinstance(source, dict) or set(source) != keys:
            raise EvidenceQueryEvaluationError(f"H3 {name} identity is invalid")
        byte_count = source["bytes"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
        ):
            raise EvidenceQueryEvaluationError(f"H3 {name} byte count is invalid")
        _require_sha256(source["sha256"], f"H3 {name}")
        for key, expected_value in expected.items():
            if key != "keys" and source.get(key) != expected_value:
                raise EvidenceQueryEvaluationError(f"H3 {name} identity changed")

    expected_partitions = {
        "fit": {
            "role": "fit",
            "records": 11_200,
            "worlds": 2_800,
            "first_world_id": "train-world-0000",
            "final_world_id": "train-world-2799",
            "gradient_bearing": True,
            "state_class_counts": {
                "supported": 42_980,
                "absent": 4_620,
                "missing": 2_800,
                "uncertain": 2_800,
                "conflicting": 2_800,
            },
        },
        "calibration": {
            "role": "calibration",
            "records": 800,
            "worlds": 200,
            "first_world_id": "train-world-2800",
            "final_world_id": "train-world-2999",
            "gradient_bearing": False,
            "state_class_counts": {
                "supported": 3_070,
                "absent": 330,
                "missing": 200,
                "uncertain": 200,
                "conflicting": 200,
            },
        },
    }
    partition_keys = {
        "role",
        "records",
        "worlds",
        "first_world_id",
        "final_world_id",
        "records_sha256",
        "transcript_multiset_sha256",
        "gradient_bearing",
        "state_class_counts",
    }
    for name, expected in expected_partitions.items():
        partition = value[name]
        if not isinstance(partition, dict) or set(partition) != partition_keys:
            raise EvidenceQueryEvaluationError(
                f"H3 {name} partition identity is invalid"
            )
        if any(
            partition[key] != expected_value for key, expected_value in expected.items()
        ):
            raise EvidenceQueryEvaluationError(f"H3 {name} partition identity changed")
        _require_sha256(partition["records_sha256"], f"H3 {name} records")
        _require_sha256(
            partition["transcript_multiset_sha256"],
            f"H3 {name} transcript multiset",
        )
    return value


def _epoch_calibration(row: Mapping[str, Any]) -> tuple[float, float, float]:
    calibration = row.get("calibration")
    if not isinstance(calibration, dict) or set(calibration) != _CALIBRATION_KEYS:
        raise EvidenceQueryEvaluationError("H3 epoch calibration record is invalid")
    if calibration["threshold_policy"] != CALIBRATION_THRESHOLD_POLICY:
        raise EvidenceQueryEvaluationError("H3 threshold policy is invalid")
    macro, overall, _uncalibrated_wrong = _calibration_phase(
        calibration["uncalibrated"], "uncalibrated"
    )
    _calibrated_macro, _calibrated_overall, calibrated_wrong = _calibration_phase(
        calibration["calibrated"], "calibrated"
    )
    if calibrated_wrong != 0:
        raise EvidenceQueryEvaluationError(
            "H3 training-only threshold did not eliminate wrong-presented fields"
        )
    threshold = _finite_rate(
        calibration["global_threshold"], "calibration global_threshold"
    )
    return macro, overall, threshold


def _candidate_checkpoint_record(
    candidate: object,
) -> tuple[int, Mapping[str, Any]]:
    if not isinstance(candidate, dict) or set(candidate) != {
        "epoch",
        *_CHECKPOINT_KEYS,
    }:
        raise EvidenceQueryEvaluationError("H3 selected candidate record is invalid")
    checkpoint = _checkpoint_record({key: candidate[key] for key in _CHECKPOINT_KEYS})
    epoch = candidate["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise EvidenceQueryEvaluationError("H3 selected candidate epoch is invalid")
    return epoch, checkpoint


def load_candidate_from_training_report(
    report_path: str | Path,
    *,
    expected_report_sha256: str,
    expected_manifest_sha256: str,
) -> EvidenceQueryCandidate:
    """Authenticate one fixed-seed report and its training-selected checkpoint."""

    path = Path(report_path)
    snapshot = _read_verified_file(
        path,
        expected_report_sha256,
        role="H3 training report",
    )
    report = _parse_json(snapshot, role="H3 training report")
    if not isinstance(report, dict) or set(report) != _REPORT_REQUIRED_KEYS:
        raise EvidenceQueryEvaluationError("H3 training report schema is invalid")
    if (
        report["schema_version"] != EVIDENCE_QUERY_TRAINING_REPORT_SCHEMA_VERSION
        or report["recipe"] != EVIDENCE_QUERY_TRAINING_RECIPE_VERSION
        or report["status"] != "complete"
    ):
        raise EvidenceQueryEvaluationError("H3 training report is not complete v0")
    seed = report["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in TRAINING_SEEDS
    ):
        raise EvidenceQueryEvaluationError("H3 training seed is not frozen")
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
        raise EvidenceQueryEvaluationError("H3 training model identity is invalid")
    if report["dataset_manifest_sha256"] != _require_sha256(
        expected_manifest_sha256, "expected H3 manifest"
    ):
        raise EvidenceQueryEvaluationError("H3 training used another dataset manifest")
    _validate_report_dataset_metadata(
        report["dataset"],
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if report["dev_used_for_selection"] is not False:
        raise EvidenceQueryEvaluationError("development influenced H3 selection")
    if report["fresh_v1_accessed"] is not False:
        raise EvidenceQueryEvaluationError("H3 training accessed fresh-v1")
    if report["selection_note"] != _TRAINING_SELECTION_NOTE:
        raise EvidenceQueryEvaluationError("H3 training selection boundary changed")
    _validate_hyperparameters(report["hyperparameters"])
    _validate_runtime(report["runtime"], device=report["device"])

    source_hashes = report["source_sha256"]
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(
        _training_source_paths()
    ):
        raise EvidenceQueryEvaluationError("H3 training source hashes are incomplete")
    for name, digest in source_hashes.items():
        _require_sha256(digest, f"H3 training source {name}")
    if source_hashes != _training_source_hashes():
        raise EvidenceQueryEvaluationError(
            "H3 training source hashes do not match the executable recipe"
        )

    epochs = report["epochs"]
    if not isinstance(epochs, list) or len(epochs) != 3:
        raise EvidenceQueryEvaluationError("H3 report must contain three epochs")
    epoch_rows: dict[int, Mapping[str, Any]] = {}
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
            raise EvidenceQueryEvaluationError("H3 training epochs are not ordered")
        for metric in ("train_loss", "state_loss", "pointer_loss", "seconds"):
            value = row[metric]
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise EvidenceQueryEvaluationError(f"H3 epoch {metric} is invalid")
        _checkpoint_record(row.get("checkpoint"))
        macro, overall, _threshold = _epoch_calibration(row)
        epoch_rows[expected_epoch] = row
        ranking_rows.append((macro, overall, -expected_epoch, row))
    selected_row = max(ranking_rows, key=lambda item: item[:3])[3]
    selected_epoch = int(selected_row["epoch"])

    candidate_epoch, checkpoint = _candidate_checkpoint_record(report["candidate"])
    if candidate_epoch != selected_epoch or checkpoint != selected_row["checkpoint"]:
        raise EvidenceQueryEvaluationError(
            "H3 candidate was not selected by uncalibrated calibration ranking"
        )
    calibration = report["calibration"]
    if not isinstance(calibration, Mapping):
        raise EvidenceQueryEvaluationError("H3 selected calibration is invalid")
    macro, overall, threshold = _epoch_calibration(selected_row)
    if set(calibration) != {"selected_epoch", *_CALIBRATION_KEYS}:
        raise EvidenceQueryEvaluationError("H3 selected calibration schema is invalid")
    if calibration["selected_epoch"] != selected_epoch:
        raise EvidenceQueryEvaluationError("H3 selected calibration epoch disagrees")
    selected_calibration = {key: calibration[key] for key in _CALIBRATION_KEYS}
    if selected_calibration != selected_row["calibration"]:
        raise EvidenceQueryEvaluationError("H3 selected calibration record disagrees")
    top_threshold = threshold

    checkpoint_snapshot = _read_verified_file(
        path.parent / checkpoint["filename"],
        checkpoint["sha256"],
        role=f"H3 checkpoint seed {seed}",
    )
    if len(checkpoint_snapshot) != checkpoint["bytes"]:
        raise EvidenceQueryEvaluationError("H3 checkpoint byte count changed")
    return EvidenceQueryCandidate(
        seed=seed,
        epoch=selected_epoch,
        path=path.parent / checkpoint["filename"],
        sha256=checkpoint["sha256"],
        artifact_bytes=checkpoint["bytes"],
        report_sha256=_sha256(snapshot),
        global_threshold=top_threshold,
        macro_joint=macro,
        overall_joint=overall,
        report=report,
    )


def authenticate_and_select_primary(
    training_reports: Sequence[tuple[str | Path, str]],
    *,
    expected_manifest_sha256: str,
) -> tuple[EvidenceQueryCandidate, tuple[EvidenceQueryCandidate, ...]]:
    """Select solely from authenticated training-only calibration evidence."""

    if len(training_reports) != len(TRAINING_SEEDS):
        raise EvidenceQueryEvaluationError(
            "exactly two fixed-seed H3 training reports are required"
        )
    candidates = tuple(
        load_candidate_from_training_report(
            path,
            expected_report_sha256=digest,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        for path, digest in training_reports
    )
    if {candidate.seed for candidate in candidates} != set(TRAINING_SEEDS):
        raise EvidenceQueryEvaluationError("H3 reports do not cover both fixed seeds")
    if len({candidate.report_sha256 for candidate in candidates}) != 2:
        raise EvidenceQueryEvaluationError("H3 report digests must be unique")
    if len({candidate.sha256 for candidate in candidates}) != 2:
        raise EvidenceQueryEvaluationError("H3 checkpoint digests must be unique")
    ordered = tuple(sorted(candidates, key=lambda item: item.seed))
    return max(ordered, key=lambda item: item.ranking_key), ordered


def _validate_dataset_identity(
    candidate: EvidenceQueryCandidate,
    *,
    data_dir: str | Path,
    manifest_sha256: str,
    train_sha256: str,
    dev_sha256: str,
) -> None:
    """Cross-check report data only after the primary seed is already frozen."""

    report = candidate.report
    if report["dataset_manifest_sha256"] != manifest_sha256:
        raise EvidenceQueryEvaluationError("H3 report manifest identity changed")
    dataset = _validate_report_dataset_metadata(
        report["dataset"],
        expected_manifest_sha256=manifest_sha256,
    )

    root = Path(data_dir)

    def source_file(
        value: object,
        *,
        filename: str,
        digest: str,
        records: int | None = None,
        worlds: int | None = None,
        usage: str | None = None,
    ) -> None:
        keys = {"filename", "bytes", "sha256"}
        if records is not None:
            keys.update({"records", "worlds"})
        if usage is not None:
            keys.add("usage")
        if not isinstance(value, dict) or set(value) != keys:
            raise EvidenceQueryEvaluationError(
                f"H3 dataset source {filename} record is invalid"
            )
        byte_count = value["bytes"]
        if (
            value["filename"] != filename
            or value["sha256"] != digest
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 1
            or (records is not None and value["records"] != records)
            or (worlds is not None and value["worlds"] != worlds)
            or (usage is not None and value["usage"] != usage)
        ):
            raise EvidenceQueryEvaluationError(
                f"H3 dataset source {filename} identity is inconsistent"
            )
        snapshot = _read_verified_file(
            root / filename,
            digest,
            role=f"H3 dataset {filename}",
        )
        if len(snapshot) != byte_count:
            raise EvidenceQueryEvaluationError(
                f"H3 dataset source {filename} byte count changed"
            )

    source_file(
        dataset["source_manifest"],
        filename="manifest.json",
        digest=manifest_sha256,
    )
    source_file(
        dataset["source_train"],
        filename="train.jsonl",
        digest=train_sha256,
        records=TRAIN_WORLDS * 4,
        worlds=TRAIN_WORLDS,
    )
    source_file(
        dataset["source_dev"],
        filename="dev.jsonl",
        digest=dev_sha256,
        records=DEV_WORLDS * 4,
        worlds=DEV_WORLDS,
        usage="source_authentication_only",
    )


def _resolve_device(device: str) -> str:
    if device not in {"cpu", "mps", "cuda"}:
        raise EvidenceQueryEvaluationError("device must be cpu, mps, or cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise EvidenceQueryEvaluationError("MPS was requested but is unavailable")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise EvidenceQueryEvaluationError("CUDA was requested but is unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise EvidenceQueryEvaluationError(
                "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
    return device


def _canonical_evaluation_runtime(
    primary: EvidenceQueryCandidate,
    candidates: Sequence[EvidenceQueryCandidate],
    *,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """Bind the gate decision to the training confidence implementation."""

    if batch_size != DEFAULT_BATCH_SIZE:
        raise EvidenceQueryEvaluationError(
            f"canonical H3 evaluation requires batch_size={DEFAULT_BATCH_SIZE}"
        )
    runtime_keys = (
        "python",
        "torch",
        "tokenizers",
        "cuda",
        "gpu",
        "cublas_workspace_config",
        "platform",
    )
    training_identities = []
    for candidate in candidates:
        report = candidate.report
        runtime = report.get("runtime") if isinstance(report, Mapping) else None
        if not isinstance(runtime, Mapping) or any(
            key not in runtime for key in runtime_keys
        ):
            raise EvidenceQueryEvaluationError(
                "H3 candidate runtime identity is incomplete"
            )
        training_identities.append(
            (
                report.get("device"),
                *(runtime[key] for key in runtime_keys),
            )
        )
    if len(set(training_identities)) != 1:
        raise EvidenceQueryEvaluationError(
            "H3 seeds were not trained on one canonical runtime"
        )
    if device != primary.report["device"]:
        raise EvidenceQueryEvaluationError(
            "canonical H3 evaluation device must match training"
        )

    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tokenizers": getattr(__import__("tokenizers"), "__version__", None),
        "cuda": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if device == "cuda"
            else None
        ),
        "cublas_workspace_config": (
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device == "cuda" else None
        ),
        "platform": platform.platform(),
    }
    expected = primary.report["runtime"]
    if any(observed[key] != expected[key] for key in runtime_keys):
        raise EvidenceQueryEvaluationError(
            "canonical H3 evaluation runtime must match training"
        )
    return observed


def _seed_evaluation() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def _load_evidence_query_model(
    candidate: EvidenceQueryCandidate,
    *,
    device: str,
) -> NanoEvidenceQueryPointerModel:
    snapshot = _read_verified_file(
        candidate.path,
        candidate.sha256,
        role=f"H3 checkpoint {candidate.label}",
    )
    if len(snapshot) != candidate.artifact_bytes:
        raise EvidenceQueryEvaluationError(
            f"H3 checkpoint {candidate.label} byte count changed"
        )
    try:
        state_dict = torch.load(
            io.BytesIO(snapshot),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise EvidenceQueryEvaluationError(
            f"H3 checkpoint {candidate.label} could not be loaded safely"
        ) from exc
    if not isinstance(state_dict, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in state_dict.items()
    ):
        raise EvidenceQueryEvaluationError(
            f"H3 checkpoint {candidate.label} is not a tensor state mapping"
        )
    model = NanoEvidenceQueryPointerModel()
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise EvidenceQueryEvaluationError(
            f"H3 checkpoint {candidate.label} does not match {ARCHITECTURE_VERSION}"
        ) from exc
    return model.to(device).eval()


def _prediction_map(
    inputs: Sequence[PointerInferenceInput],
    predictions: Sequence[PointerPrediction],
) -> dict[str, PointerPrediction]:
    if len(inputs) != len(predictions):
        raise EvidenceQueryEvaluationError("H3 prediction row count is invalid")
    result: dict[str, PointerPrediction] = {}
    for item, prediction in zip(inputs, predictions, strict=True):
        previous = result.setdefault(item.transcript, prediction)
        if previous != prediction:
            raise EvidenceQueryEvaluationError(
                "identical transcripts received inconsistent H3 predictions"
            )
    return result


def _evaluate_predictions(
    *,
    inputs: Sequence[PointerInferenceInput],
    predictions: Sequence[PointerPrediction],
    cases: Sequence[Any],
    candidate: EvidenceQueryCandidate,
) -> EvaluationReport:
    prediction_by_transcript = _prediction_map(inputs, predictions)

    def predict(transcript: str) -> tuple[StateSpanProposal, ...]:
        prediction = prediction_by_transcript[transcript]
        if prediction.error is not None:
            raise PointerDecodeError(prediction.error)
        assert prediction.proposals is not None
        return prediction.proposals

    solver = EvidenceQueryPointerSolver(
        predict,
        solver_id=(
            f"development/evidence-query/{candidate.label}/sha-{candidate.sha256}"
        ),
        version=ARCHITECTURE_VERSION,
        artifact_bytes=candidate.artifact_bytes,
    )
    return evaluate_solver(solver, cases, measure_latency=False)


def _require_metric(
    metrics: Mapping[str, Any],
    name: str,
    *,
    expected_denominator: int | None,
) -> Mapping[str, Any]:
    value = metrics.get(name)
    if not isinstance(value, Mapping):
        raise EvidenceQueryEvaluationError(f"H3 metric {name} is missing")
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
        raise EvidenceQueryEvaluationError(f"H3 metric {name} is inconsistent")
    return value


def _quality_gates(
    metrics: Mapping[str, Any],
    *,
    require_zero_wrong_presented: bool,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name, (minimum, denominator) in _SEMANTIC_GATES.items():
        metric = _require_metric(
            metrics,
            name,
            expected_denominator=denominator,
        )
        evidence[name] = {
            "metric": dict(metric),
            "minimum_numerator": minimum,
            "passed": metric["numerator"] >= minimum,
        }
    failures = _require_metric(
        metrics,
        "failures",
        expected_denominator=1_000,
    )
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
        wrong = _require_metric(
            metrics,
            "false_presented",
            expected_denominator=None,
        )
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


def _final_metrics(
    acceptance: Mapping[str, Any],
    final_state: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = acceptance.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvidenceQueryEvaluationError("H3 final acceptance metrics are invalid")
    result = dict(metrics)
    challenge = final_state.get("target_challenge")
    if not isinstance(challenge, Mapping):
        raise EvidenceQueryEvaluationError("H3 final challenge metrics are invalid")
    uncertain = challenge.get("uncertain")
    if not isinstance(uncertain, Mapping):
        raise EvidenceQueryEvaluationError("H3 final uncertain metric is invalid")
    result["uncertain_target"] = {
        "numerator": uncertain.get("grounded_exact"),
        "denominator": uncertain.get("total"),
        "rate": uncertain.get("grounded_exact_accuracy"),
    }
    return result


def _evaluation_source_paths() -> dict[str, Path]:
    package_dir = Path(__file__).parent.parent
    return {
        "adapter": package_dir / "adapters" / "evidence_query_pointer.py",
        "data_generator": Path(state_span_data.__file__),
        "evaluator": Path(__file__),
        "inference": Path(evidence_query_inference.__file__),
        "model": Path(evidence_query_model.__file__),
        "pointer_data": Path(pointer_data.__file__),
    }


def _evaluation_source_hashes() -> dict[str, str]:
    return {
        name: _sha256(_read_regular_file(path, role=f"evaluation {name} source"))
        for name, path in sorted(_evaluation_source_paths().items())
    }


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise EvidenceQueryEvaluationError(
            "H3 evaluation output could not be created"
        ) from exc


def evaluate_development(
    *,
    data_dir: str | Path,
    manifest_sha256: str,
    tokenizer_path: str | Path,
    training_reports: Sequence[tuple[str | Path, str]],
    output_path: str | Path,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Mapping[str, Any]:
    """Select H3 without development, then stop at the frozen quality gate."""

    output = Path(output_path)
    if output.exists():
        raise EvidenceQueryEvaluationError("H3 evaluation output already exists")
    if not output.parent.is_dir():
        raise EvidenceQueryEvaluationError("H3 evaluation output parent must exist")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise EvidenceQueryEvaluationError("batch_size must be an integer")
    expected_manifest = _require_sha256(manifest_sha256, "H3 manifest")
    resolved_device = _resolve_device(device)
    source_hashes = _evaluation_source_hashes()

    # This ordering is the central H3 selection boundary.  Report/checkpoint
    # authentication and primary selection happen before development is opened.
    primary, candidates = authenticate_and_select_primary(
        training_reports,
        expected_manifest_sha256=expected_manifest,
    )
    evaluation_runtime = _canonical_evaluation_runtime(
        primary,
        candidates,
        device=resolved_device,
        batch_size=batch_size,
    )

    try:
        bundle = load_development_bundle(
            data_dir,
            expected_manifest_sha256=expected_manifest,
        )
        for candidate in candidates:
            _validate_dataset_identity(
                candidate,
                data_dir=data_dir,
                manifest_sha256=bundle.manifest_sha256,
                train_sha256=bundle.manifest["train"]["sha256"],
                dev_sha256=bundle.dev_sha256,
            )
        tokenizer = load_pointer_tokenizer(Path(tokenizer_path))
        encoded = encode_pointer_partition(
            tokenizer,
            bundle.examples,
            expected_split="dev",
        )
        inputs = build_pointer_inference_inputs(bundle.examples, encoded)
        cases = _fixture_cases(bundle.examples)
    except EvidenceQueryEvaluationError:
        raise
    except Exception as exc:
        raise EvidenceQueryEvaluationError(
            "known H3 development inputs could not be verified"
        ) from exc

    _seed_evaluation()
    model = _load_evidence_query_model(primary, device=resolved_device)
    inference = batched_evidence_query_inference(
        model,
        inputs,
        device=resolved_device,
        batch_size=batch_size,
    )
    del model
    if tuple(inference.example_ids) != tuple(item.example_id for item in inputs):
        raise EvidenceQueryEvaluationError("H3 inference changed development order")

    uncalibrated_raw = raw_pointer_diagnostics(
        bundle.examples,
        cases,
        inference.predictions,
    )
    uncalibrated_metrics = uncalibrated_raw["acceptance"]["metrics"]
    uncalibrated_gate = _quality_gates(
        uncalibrated_metrics,
        require_zero_wrong_presented=False,
    )
    admission_passed = bool(uncalibrated_gate["all_quality_gates_passed"])
    calibrated_raw: Mapping[str, Any] | None = None
    calibrated_gate: Mapping[str, Any] | None = None
    verifier_final_section: Mapping[str, Any] | None = None
    verifier_gate: Mapping[str, Any] | None = None

    # The uncalibrated result is a frozen admission boundary.  Applying the
    # training-only threshold or invoking the verifier after a failed
    # admission would execute stages that the protocol explicitly forbids.
    if admission_passed:
        calibrated_predictions = apply_global_threshold(
            inference,
            primary.global_threshold,
        )
        calibrated_raw = raw_pointer_diagnostics(
            bundle.examples,
            cases,
            calibrated_predictions,
        )
        calibrated_gate = _quality_gates(
            calibrated_raw["acceptance"]["metrics"],
            require_zero_wrong_presented=True,
        )
        verifier_report = _evaluate_predictions(
            inputs=inputs,
            predictions=calibrated_predictions,
            cases=cases,
            candidate=primary,
        )
        verifier_final = final_state_diagnostics(verifier_report, bundle.examples)
        verifier_acceptance = acceptance_diagnostics(
            verifier_report,
            bundle.examples,
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

    calibrated_assessed = calibrated_gate is not None
    quality_passed = bool(
        admission_passed
        and calibrated_gate is not None
        and calibrated_gate["all_quality_gates_passed"]
        and verifier_gate is not None
        and verifier_gate["all_quality_gates_passed"]
    )
    if _evaluation_source_hashes() != source_hashes:
        raise EvidenceQueryEvaluationError("H3 evaluation source changed during run")

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
    result: dict[str, Any] = {
        "schema_version": EVIDENCE_QUERY_DEVELOPMENT_EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "partition": {
            "partition_id": DEVELOPMENT_PARTITION_ID,
            "role": "known_adaptive_development_quality_only",
            "manifest_sha256": bundle.manifest_sha256,
            "development_sha256": bundle.dev_sha256,
            "records": len(bundle.examples),
            "worlds": len({example.world_id for example in bundle.examples}),
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "used_for_epoch_selection": False,
            "used_for_seed_selection": False,
            "used_for_threshold_selection": False,
            "historical_benchmark_read": False,
            "fresh_v1_read": False,
        },
        "artifacts": {
            "architecture_version": ARCHITECTURE_VERSION,
            "architecture_identity": FROZEN_NANO_V01.architecture_identity,
            "base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
            "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
            "trunk_parameter_count": NANO_TRUNK_PARAMETER_COUNT,
            "evidence_query_head_parameter_count": (
                EVIDENCE_QUERY_HEAD_PARAMETER_COUNT
            ),
            "parameter_count": NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
            "candidates": candidate_rows,
            "primary_label": primary.label,
        },
        "protocol": {
            "supervision_version": POINTER_SUPERVISION_VERSION,
            "state_class_order": [state.value for state in STATE_ORDER],
            "primary_selection_order": [
                "uncalibrated_calibration_macro_joint_desc",
                "uncalibrated_calibration_overall_joint_desc",
                "earlier_epoch",
                "seed_20260805",
            ],
            "global_threshold": primary.global_threshold,
            "threshold_source": "training_only_calibration",
            "threshold_policy": CALIBRATION_THRESHOLD_POLICY,
            "threshold_applied": calibrated_assessed,
            "verifier_policy": "accept_or_reject_model_owned_proposals_only",
            "verifier_supplies_or_corrects_values": False,
            "batch_size": batch_size,
            "latency_measured": False,
            "fresh_v1_confirmation_assessed": False,
        },
        "runtime": {
            "device": resolved_device,
            "deterministic_algorithms": True,
            **evaluation_runtime,
        },
        "source_sha256": source_hashes,
        "uncalibrated_raw": uncalibrated_raw,
        "uncalibrated_admission": uncalibrated_gate,
        "calibrated_raw": calibrated_raw,
        "calibrated_quality": calibrated_gate,
        "verifier_final": verifier_final_section,
        "verifier_final_quality": verifier_gate,
        "decision": {
            "uncalibrated_semantic_admission_passed": admission_passed,
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
            "fresh_v1_assessed": False,
            "next_step": (
                "measure matched latency before any fresh-v1 confirmation"
                if quality_passed
                else "reject H3 quality candidate; do not measure latency or open fresh-v1"
            ),
        },
        "selection_boundary": (
            "Primary epoch, seed, and threshold were fixed from authenticated "
            "training-only calibration before known development was loaded. "
            "Uncalibrated admission failure stops before threshold application "
            "and verifier evaluation. Quality failure stops before latency and "
            "fresh-v1."
        ),
    }
    _write_json_no_clobber(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate Nano H3 on known adaptive development"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--tokenizer", type=Path, default=root / "sft" / "tokenizer.json"
    )
    parser.add_argument(
        "--training-report",
        nargs=2,
        action="append",
        required=True,
        metavar=("REPORT", "SHA256"),
        help="repeat exactly twice, once for each digest-pinned H3 seed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evaluate_development(
        data_dir=args.data_dir,
        manifest_sha256=args.manifest_sha256,
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
    "EVIDENCE_QUERY_DEVELOPMENT_EVALUATION_SCHEMA_VERSION",
    "EVIDENCE_QUERY_TRAINING_RECIPE_VERSION",
    "EVIDENCE_QUERY_TRAINING_REPORT_SCHEMA_VERSION",
    "EvidenceQueryCandidate",
    "EvidenceQueryEvaluationError",
    "authenticate_and_select_primary",
    "evaluate_development",
    "load_candidate_from_training_report",
]
