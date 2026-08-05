"""Authoritative sealed-development evaluation for Nano's pointer intervention.

Only the already sealed native state/span development partition is encoded.
The frozen generative reference is not rerun: callers must pin the complete H1
development report by SHA-256, and this module authenticates its partition and
recomputes its recorded aggregate metrics before comparison.  Candidate models
receive transcript-derived tokens, offsets, and Patient masks only; evaluator
truth remains behind the standard evaluation boundary.
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
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.adapters.pointer_span import PointerSpanSolver
from nano_ai.adapters.state_span import StateSpanProposal
from nano_ai.contract import (
    FIELD_ORDER,
    EvidenceSpan,
    FieldName,
    FieldOutput,
    FieldState,
    normalize_value,
)
from nano_ai.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationReport,
    evaluate_solver,
)
from nano_ai.training import pointer_data, state_span_data
from nano_ai.training.evaluate_state_span import (
    DEVELOPMENT_EVALUATION_SCHEMA_VERSION as H1_EVALUATION_SCHEMA_VERSION,
)
from nano_ai.training.evaluate_state_span import (
    DEVELOPMENT_PARTITION_ID,
    _fixture_cases,
    acceptance_diagnostics,
    final_state_diagnostics,
    load_development_bundle,
)
from nano_ai.training.model import NANO_MODEL_CONFIG
from nano_ai.training.pointer_data import (
    POINTER_PROMPT_TEMPLATE_ID,
    POINTER_SUPERVISION_VERSION,
    STATE_ORDER,
    PointerSupervision,
    TokenOffset,
    encode_pointer_partition,
    load_pointer_tokenizer,
    token_span_to_evidence,
)
from nano_ai.training.pointer_model import (
    NANO_POINTER_PARAMETER_COUNT,
    NANO_TRUNK_PARAMETER_COUNT,
    POINTER_HEAD_PARAMETER_COUNT,
    NanoPointerModel,
)
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    DEV_WORLDS,
    TARGET_GRAMMAR_VERSION,
    TRAIN_WORLDS,
    StateSpanExample,
)

POINTER_DEVELOPMENT_EVALUATION_SCHEMA_VERSION = (
    "nano.pointer-span-development-evaluation.v0"
)
POINTER_TRAINING_REPORT_SCHEMA_VERSION = "nano.pointer-span-training-report.v0"
POINTER_TRAINING_RECIPE_VERSION = "nano-pointer-span-supervision-v0"
DEFAULT_BATCH_SIZE = 32
TRAINING_SEEDS = (20260805, 20260806)

_LABEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_PATIENT_PREFIX_RE = re.compile(r"^\s*patient\s*:\s*", flags=re.IGNORECASE)
_PRESENTED_STATES = frozenset({FieldState.SUPPORTED, FieldState.ABSENT})
_STATE_CODE = {
    FieldState.SUPPORTED: "S",
    FieldState.ABSENT: "A",
    FieldState.MISSING: "M",
    FieldState.UNCERTAIN: "U",
    FieldState.CONFLICTING: "C",
}
_POINTER_COUNT = {
    FieldState.SUPPORTED: 1,
    FieldState.ABSENT: 1,
    FieldState.MISSING: 0,
    FieldState.UNCERTAIN: 1,
    FieldState.CONFLICTING: 2,
}
_TRAINING_REPORT_KEYS = {
    "schema_version",
    "recipe",
    "status",
    "seed",
    "device",
    "parameter_count",
    "trunk_parameter_count",
    "pointer_head_parameter_count",
    "architecture_identity",
    "base_checkpoint_sha256",
    "tokenizer_sha256",
    "dataset_manifest_sha256",
    "dataset",
    "hyperparameters",
    "epochs",
    "candidate",
    "source_sha256",
    "runtime",
    "selection_note",
}
_DATASET_KEYS = {
    "schema_version",
    "target_grammar",
    "train_sha256",
    "dev_sha256",
    "train_records",
    "dev_records",
}
_CHECKPOINT_KEYS = {"filename", "sha256", "bytes"}
_EPOCH_KEYS = {
    "epoch",
    "train_loss",
    "state_loss",
    "pointer_loss",
    "dev_loss",
    "dev_state_loss",
    "dev_pointer_loss",
    "seconds",
    "checkpoint",
}
_TRAINING_SOURCE_KEYS = {
    "data_generator",
    "base_model",
    "pointer_data",
    "pointer_model",
    "state_span_adapter",
    "h1_training_loader",
    "training",
}
_TRAINING_RUNTIME_KEYS = {
    "python",
    "torch",
    "tokenizers",
    "cuda",
    "gpu",
    "cublas_workspace_config",
    "platform",
    "seconds",
}
_DETERMINISTIC_CUBLAS_CONFIGS = frozenset({":4096:8", ":16:8"})
_STATE_CLASS_COUNTS = (46_050, 4_950, 3_000, 3_000, 3_000)
_STATE_CLASS_WEIGHTS = (
    0.26058631921824105,
    2.4242424242424243,
    4.0,
    4.0,
    4.0,
)
_TRAINING_SELECTION_NOTE = (
    "This is an unselected H2 development candidate. Historical fresh-v0 "
    "and the sealed fresh-v1 confirmation partition were not read. Causal "
    "pointer logits at earlier transcript tokens can use prefix context only."
)


class PointerEvaluationError(ValueError):
    """A sealed input, artifact, decoding, or evaluation invariant failed."""


class PointerDecodeError(ValueError):
    """Pointer logits cannot produce the complete constrained proposal grammar."""


@dataclass(frozen=True, slots=True)
class PointerCandidateCheckpoint:
    """One digest-pinned pointer checkpoint authenticated by its training report."""

    label: str
    path: Path
    sha256: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or _LABEL_RE.fullmatch(self.label) is None:
            raise PointerEvaluationError(
                "candidate label must match [a-z0-9][a-z0-9._-]{0,127}"
            )
        object.__setattr__(self, "path", Path(self.path))
        _require_sha256(self.sha256, "candidate checkpoint")
        if not isinstance(self.provenance, Mapping) or any(
            not isinstance(key, str) for key in self.provenance
        ):
            raise PointerEvaluationError(
                "candidate provenance must be a string-keyed mapping"
            )


@dataclass(frozen=True, slots=True)
class PointerInferenceInput:
    """Transcript-derived model inputs with all evaluator labels removed."""

    example_id: str
    transcript: str
    token_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    pointer_mask: tuple[bool, ...]
    token_offsets: tuple[TokenOffset, ...]

    def __post_init__(self) -> None:
        length = len(self.token_ids)
        if not self.example_id or not self.transcript or length < 1:
            raise PointerEvaluationError("pointer inference input is incomplete")
        if not (
            len(self.attention_mask)
            == len(self.pointer_mask)
            == len(self.token_offsets)
            == length
        ):
            raise PointerEvaluationError("pointer inference tensors do not align")
        if not all(self.attention_mask):
            raise PointerEvaluationError("unpadded inference input mask must be true")
        if any(
            pointer and offset is None
            for pointer, offset in zip(
                self.pointer_mask, self.token_offsets, strict=True
            )
        ):
            raise PointerEvaluationError("pointer mask escaped transcript offsets")


@dataclass(frozen=True, slots=True)
class PointerPrediction:
    """One decoded prediction, or an explicit item-level decoding failure."""

    proposals: tuple[StateSpanProposal, ...] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if (self.proposals is None) == (self.error is None):
            raise ValueError("exactly one of proposals or error is required")
        if self.proposals is not None:
            object.__setattr__(self, "proposals", tuple(self.proposals))
        if self.error is not None and (
            not self.error or self.error.strip() != self.error
        ):
            raise ValueError("prediction error must be non-empty edge-trimmed text")


@dataclass(frozen=True, slots=True)
class AuthenticatedFrozenBase:
    """Metric-only frozen reference recovered from one digest-pinned H1 report."""

    report_sha256: str
    checkpoint_sha256: str
    evaluation: Mapping[str, Any]
    final_state: Mapping[str, Any]
    acceptance: Mapping[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PointerEvaluationError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _read_regular_file(path: Path, *, role: str) -> bytes:
    try:
        with Path(path).open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise PointerEvaluationError(f"{role} is not a regular file")
            return handle.read()
    except PointerEvaluationError:
        raise
    except OSError as exc:
        raise PointerEvaluationError(f"{role} is unavailable") from exc


def _read_verified_file(path: Path, expected_sha256: str, *, role: str) -> bytes:
    expected = _require_sha256(expected_sha256, role)
    snapshot = _read_regular_file(path, role=role)
    observed = _sha256(snapshot)
    if not hmac.compare_digest(observed, expected):
        raise PointerEvaluationError(
            f"{role} SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return snapshot


def _reject_json_constant(value: str) -> None:
    raise PointerEvaluationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PointerEvaluationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json(snapshot: bytes, *, role: str) -> Any:
    try:
        return json.loads(
            snapshot.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except PointerEvaluationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise PointerEvaluationError(f"{role} is invalid JSON") from exc


def _training_source_paths() -> dict[str, Path]:
    training_dir = Path(__file__).parent
    package_dir = training_dir.parent
    return {
        "data_generator": Path(state_span_data.__file__),
        "base_model": training_dir / "model.py",
        "pointer_data": Path(pointer_data.__file__),
        "pointer_model": training_dir / "pointer_model.py",
        "state_span_adapter": package_dir / "adapters" / "state_span.py",
        "h1_training_loader": training_dir / "train_state_span.py",
        "training": training_dir / "train_pointer.py",
    }


def _training_source_hashes() -> dict[str, str]:
    paths = _training_source_paths()
    if set(paths) != _TRAINING_SOURCE_KEYS:
        raise RuntimeError("pointer training source registry is incomplete")
    return {
        name: _sha256(_read_regular_file(path, role=f"training {name} source"))
        for name, path in sorted(paths.items())
    }


def _checkpoint_record(
    report_path: Path,
    value: object,
    *,
    label: str,
    provenance: Mapping[str, Any],
) -> PointerCandidateCheckpoint:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_KEYS:
        raise PointerEvaluationError("training checkpoint record is invalid")
    filename = value["filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise PointerEvaluationError("training checkpoint filename is unsafe")
    digest = _require_sha256(value["sha256"], "training checkpoint")
    byte_count = value["bytes"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise PointerEvaluationError("training checkpoint byte count is invalid")
    return PointerCandidateCheckpoint(
        label=label,
        path=report_path.parent / filename,
        sha256=digest,
        provenance={**provenance, "artifact_bytes": byte_count},
    )


def _validate_hyperparameters(value: object) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PointerEvaluationError("training hyperparameters must be an object")
    required = {
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
        "state_class_order": [state.value for state in STATE_ORDER],
        "state_class_counts": {
            state.value: count
            for state, count in zip(STATE_ORDER, _STATE_CLASS_COUNTS, strict=True)
        },
        "state_class_weights": list(_STATE_CLASS_WEIGHTS),
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
        "causal_pointer_heads": True,
        "deterministic_algorithms": True,
        "full_trunk_trainable": True,
        "world_grouped_batches": True,
    }
    if set(value) != {*required, "steps_per_epoch", "total_steps"}:
        raise PointerEvaluationError("training hyperparameter keys are not frozen")
    for key, expected in required.items():
        if value.get(key) != expected:
            raise PointerEvaluationError(f"training hyperparameter {key} is not frozen")
    for key in ("steps_per_epoch", "total_steps"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise PointerEvaluationError(f"training hyperparameter {key} is invalid")
    if value["total_steps"] != value["steps_per_epoch"] * 3:
        raise PointerEvaluationError("training step schedule is inconsistent")
    expected_steps_per_epoch = (TRAIN_WORLDS * 4 + 32 - 1) // 32
    if value["steps_per_epoch"] != expected_steps_per_epoch:
        raise PointerEvaluationError("training steps per epoch are not frozen")
    return value


def load_candidates_from_training_report(
    report_path: str | Path,
    *,
    expected_report_sha256: str,
    expected_manifest_sha256: str,
    expected_dev_sha256: str,
    expected_train_sha256: str,
) -> tuple[PointerCandidateCheckpoint, ...]:
    """Authenticate one H2 report and return its three immutable epochs."""

    path = Path(report_path)
    snapshot = _read_verified_file(
        path,
        expected_report_sha256,
        role="pointer training report",
    )
    report = _parse_json(snapshot, role="pointer training report")
    if not isinstance(report, dict) or set(report) != _TRAINING_REPORT_KEYS:
        raise PointerEvaluationError("pointer training report schema is invalid")
    if (
        report["schema_version"] != POINTER_TRAINING_REPORT_SCHEMA_VERSION
        or report["recipe"] != POINTER_TRAINING_RECIPE_VERSION
        or report["status"] != "complete"
    ):
        raise PointerEvaluationError("pointer training report is not a complete v0 run")
    seed = report["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in TRAINING_SEEDS
    ):
        raise PointerEvaluationError("pointer training seed is not frozen")
    if (
        report["parameter_count"] != NANO_POINTER_PARAMETER_COUNT
        or report["trunk_parameter_count"] != NANO_TRUNK_PARAMETER_COUNT
        or report["pointer_head_parameter_count"] != POINTER_HEAD_PARAMETER_COUNT
        or report["architecture_identity"] != FROZEN_NANO_V01.architecture_identity
        or report["base_checkpoint_sha256"] != FROZEN_NANO_V01.checkpoint_sha256
        or report["tokenizer_sha256"] != FROZEN_NANO_V01.tokenizer_sha256
    ):
        raise PointerEvaluationError("pointer training model identity is invalid")
    if report["dataset_manifest_sha256"] != _require_sha256(
        expected_manifest_sha256, "expected manifest"
    ):
        raise PointerEvaluationError("pointer training used another dataset manifest")

    dataset = report["dataset"]
    if not isinstance(dataset, dict) or set(dataset) != _DATASET_KEYS:
        raise PointerEvaluationError("pointer training dataset record is invalid")
    if (
        dataset["schema_version"] != DATASET_SCHEMA_VERSION
        or dataset["target_grammar"] != TARGET_GRAMMAR_VERSION
        or dataset["train_sha256"]
        != _require_sha256(expected_train_sha256, "expected training data")
        or dataset["dev_sha256"]
        != _require_sha256(expected_dev_sha256, "expected development data")
        or dataset["train_records"] != TRAIN_WORLDS * 4
        or dataset["dev_records"] != DEV_WORLDS * 4
    ):
        raise PointerEvaluationError("pointer training dataset identity is invalid")
    _validate_hyperparameters(report["hyperparameters"])

    source_hashes = report["source_sha256"]
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != _TRAINING_SOURCE_KEYS
    ):
        raise PointerEvaluationError("pointer training source hashes are incomplete")
    for name, digest in source_hashes.items():
        _require_sha256(digest, f"pointer training source {name}")
    if source_hashes != _training_source_hashes():
        raise PointerEvaluationError(
            "pointer training source hashes do not match the executable recipe"
        )

    if report["device"] not in {"cpu", "mps", "cuda"}:
        raise PointerEvaluationError("pointer training device is invalid")
    runtime = report["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != _TRAINING_RUNTIME_KEYS:
        raise PointerEvaluationError("pointer training runtime record is invalid")
    for name in ("python", "torch", "tokenizers", "platform"):
        if not isinstance(runtime[name], str) or not runtime[name]:
            raise PointerEvaluationError(f"pointer training runtime {name} is invalid")
    for name in ("cuda", "gpu"):
        if runtime[name] is not None and (
            not isinstance(runtime[name], str) or not runtime[name]
        ):
            raise PointerEvaluationError(f"pointer training runtime {name} is invalid")
    if report["device"] == "cuda" and (
        runtime["cuda"] is None or runtime["gpu"] is None
    ):
        raise PointerEvaluationError("CUDA training runtime identity is incomplete")
    if report["device"] != "cuda" and runtime["gpu"] is not None:
        raise PointerEvaluationError("non-CUDA training reported a CUDA GPU")
    if report["device"] == "cuda" and (
        runtime["cublas_workspace_config"] not in _DETERMINISTIC_CUBLAS_CONFIGS
    ):
        raise PointerEvaluationError(
            "CUDA training runtime lacks deterministic cuBLAS configuration"
        )
    if report["device"] != "cuda" and runtime["cublas_workspace_config"] is not None:
        raise PointerEvaluationError(
            "non-CUDA training reported a cuBLAS workspace configuration"
        )
    if (
        type(runtime["seconds"]) not in {int, float}
        or not math.isfinite(runtime["seconds"])
        or runtime["seconds"] < 0
    ):
        raise PointerEvaluationError("pointer training runtime seconds is invalid")
    if report["selection_note"] != _TRAINING_SELECTION_NOTE:
        raise PointerEvaluationError("pointer training selection note is invalid")

    epochs = report["epochs"]
    if not isinstance(epochs, list) or len(epochs) != 3:
        raise PointerEvaluationError(
            "pointer training report must contain three epochs"
        )
    report_identity = _sha256(snapshot)
    candidates: list[PointerCandidateCheckpoint] = []
    for expected_epoch, row in enumerate(epochs, 1):
        if not isinstance(row, dict) or set(row) != _EPOCH_KEYS:
            raise PointerEvaluationError("pointer training epoch record is invalid")
        if row["epoch"] != expected_epoch:
            raise PointerEvaluationError("pointer training epochs are not ordered")
        for metric in _EPOCH_KEYS - {"epoch", "checkpoint"}:
            metric_value = row[metric]
            if (
                type(metric_value) not in {int, float}
                or not math.isfinite(metric_value)
                or metric_value < 0
            ):
                raise PointerEvaluationError(
                    f"pointer training epoch {metric} is invalid"
                )
        provenance = {
            "kind": "verified_pointer_training_report",
            "training_report_sha256": report_identity,
            "training_recipe": report["recipe"],
            "training_seed": seed,
            "epoch": expected_epoch,
            "train_loss": row["train_loss"],
            "dev_loss": row["dev_loss"],
            "training_source_sha256": dict(sorted(source_hashes.items())),
        }
        candidates.append(
            _checkpoint_record(
                path,
                row["checkpoint"],
                label=f"seed-{seed}-epoch-{expected_epoch}",
                provenance=provenance,
            )
        )
    if report["candidate"] != epochs[-1]["checkpoint"]:
        raise PointerEvaluationError(
            "pointer training candidate is not the final epoch checkpoint"
        )
    return tuple(candidates)


def _resolve_device(device: str) -> str:
    if device not in {"cpu", "mps", "cuda"}:
        raise PointerEvaluationError("device must be cpu, mps, or cuda")
    if device == "mps" and not torch.backends.mps.is_available():
        raise PointerEvaluationError("MPS was requested but is unavailable")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise PointerEvaluationError("CUDA was requested but is unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            raise PointerEvaluationError(
                "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
            )
    return device


def _seed_evaluation() -> None:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def _load_pointer_model(
    candidate: PointerCandidateCheckpoint,
    *,
    device: str,
) -> NanoPointerModel:
    snapshot = _read_verified_file(
        candidate.path,
        candidate.sha256,
        role=f"pointer checkpoint {candidate.label}",
    )
    expected_bytes = candidate.provenance.get("artifact_bytes")
    if expected_bytes is not None and len(snapshot) != expected_bytes:
        raise PointerEvaluationError(
            f"pointer checkpoint {candidate.label} byte count changed"
        )
    try:
        state_dict = torch.load(
            io.BytesIO(snapshot),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise PointerEvaluationError(
            f"pointer checkpoint {candidate.label} could not be loaded safely"
        ) from exc
    if not isinstance(state_dict, Mapping) or any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in state_dict.items()
    ):
        raise PointerEvaluationError(
            f"pointer checkpoint {candidate.label} is not a tensor state mapping"
        )
    model = NanoPointerModel()
    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PointerEvaluationError(
            f"pointer checkpoint {candidate.label} does not match the H2 architecture"
        ) from exc
    return model.to(device).eval()


def build_pointer_inference_inputs(
    examples: Sequence[StateSpanExample],
    encoded: Sequence[PointerSupervision],
) -> tuple[PointerInferenceInput, ...]:
    """Strip state/span labels before any candidate model or decoder sees rows."""

    if len(examples) != len(encoded):
        raise PointerEvaluationError("pointer encoding returned the wrong row count")
    inputs: list[PointerInferenceInput] = []
    for example, record in zip(examples, encoded, strict=True):
        if example.example_id != record.example_id or example.transcript == "":
            raise PointerEvaluationError("pointer encoding changed source row identity")
        inputs.append(
            PointerInferenceInput(
                example_id=example.example_id,
                transcript=example.transcript,
                token_ids=record.token_ids,
                attention_mask=record.attention_mask,
                pointer_mask=record.pointer_mask,
                token_offsets=record.token_offsets,
            )
        )
    return tuple(inputs)


def _patient_content_ranges(transcript: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in transcript.splitlines(keepends=True):
        visible = line.rstrip("\r\n")
        match = _PATIENT_PREFIX_RE.match(visible)
        if match is not None and match.end() < len(visible):
            ranges.append((offset + match.end(), offset + len(visible)))
        offset += len(line)
    return tuple(ranges)


def _trimmed_token_envelope(
    transcript: str,
    offset: tuple[int, int],
) -> tuple[int, int]:
    """Match pointer-supervision masking for whitespace-absorbing BPE tokens."""

    raw_start, raw_end = offset
    start, end = raw_start, raw_end
    while start < end and transcript[start].isspace():
        start += 1
    while end > start and transcript[end - 1].isspace():
        end -= 1
    return (raw_start, raw_end) if start == end else (start, end)


def _candidate_spans(
    item: PointerInferenceInput,
    start_scores: Tensor,
    end_scores: Tensor,
) -> tuple[tuple[float, EvidenceSpan], ...]:
    """Enumerate scored spans constrained to one Patient turn."""

    if start_scores.ndim != 1 or end_scores.ndim != 1:
        raise PointerDecodeError("pointer score vectors must be one-dimensional")
    if len(start_scores) != len(item.token_ids) or len(end_scores) != len(
        item.token_ids
    ):
        raise PointerDecodeError("pointer score vectors have the wrong length")
    if not bool(torch.isfinite(start_scores).all()) or not bool(
        torch.isfinite(end_scores).all()
    ):
        raise PointerDecodeError("pointer scores must be finite")

    best_by_offsets: dict[tuple[int, int], tuple[float, EvidenceSpan]] = {}
    for patient_start, patient_end in _patient_content_ranges(item.transcript):
        turn_indices = [
            index
            for index, (allowed, offset) in enumerate(
                zip(item.pointer_mask, item.token_offsets, strict=True)
            )
            if allowed
            and offset is not None
            and patient_start <= _trimmed_token_envelope(item.transcript, offset)[0]
            and _trimmed_token_envelope(item.transcript, offset)[1] <= patient_end
        ]
        for start_position, start_index in enumerate(turn_indices):
            for end_index in turn_indices[start_position:]:
                if not all(item.pointer_mask[start_index : end_index + 1]):
                    continue
                try:
                    evidence = token_span_to_evidence(
                        item.transcript,
                        item.token_offsets,
                        start_index,
                        end_index,
                    )
                except (TypeError, ValueError):
                    continue
                if not (
                    patient_start <= evidence.start and evidence.end <= patient_end
                ):
                    continue
                score = float(start_scores[start_index]) + float(end_scores[end_index])
                key = (evidence.start, evidence.end)
                previous = best_by_offsets.get(key)
                if previous is None or score > previous[0]:
                    best_by_offsets[key] = (score, evidence)
    return tuple(
        sorted(
            best_by_offsets.values(),
            key=lambda value: (-value[0], value[1].start, value[1].end),
        )
    )


def decode_pointer_logits(
    item: PointerInferenceInput,
    state_logits: Tensor,
    start_logits: Tensor,
    end_logits: Tensor,
) -> tuple[StateSpanProposal, ...]:
    """Decode one row under the complete state/pointer structural grammar."""

    token_count = len(item.token_ids)
    if tuple(state_logits.shape) != (len(FIELD_ORDER), len(STATE_ORDER)):
        raise PointerDecodeError("state logits must have shape [5, 5]")
    expected_pointer_shape = (token_count, len(FIELD_ORDER), 2)
    if (
        tuple(start_logits.shape) != expected_pointer_shape
        or tuple(end_logits.shape) != expected_pointer_shape
    ):
        raise PointerDecodeError("pointer logits must have shape [tokens, 5, 2]")
    if not bool(torch.isfinite(state_logits).all()):
        raise PointerDecodeError("state logits must be finite")

    predicted_state_indices = state_logits.argmax(dim=-1).tolist()
    proposals: list[StateSpanProposal] = []
    for field_index, field_name in enumerate(FIELD_ORDER):
        state = STATE_ORDER[predicted_state_indices[field_index]]
        required = _POINTER_COUNT[state]
        selected: list[EvidenceSpan] = []
        selected_offsets: set[tuple[int, int]] = set()
        selected_normalized_text: set[str] = set()
        for slot in range(required):
            ranked = _candidate_spans(
                item,
                start_logits[:, field_index, slot],
                end_logits[:, field_index, slot],
            )
            chosen = next(
                (
                    evidence
                    for _score, evidence in ranked
                    if (evidence.start, evidence.end) not in selected_offsets
                    and not (
                        state is FieldState.CONFLICTING
                        and normalize_value(evidence.text) in selected_normalized_text
                    )
                ),
                None,
            )
            if chosen is None:
                raise PointerDecodeError(
                    f"{field_name.value} slot {slot} has no valid field-local Patient span"
                )
            selected.append(chosen)
            selected_offsets.add((chosen.start, chosen.end))
            selected_normalized_text.add(normalize_value(chosen.text))
        proposals.append(
            StateSpanProposal(
                field=field_name,
                state_code=_STATE_CODE[state],
                state=state,
                spans=tuple(selected),
            )
        )
    return tuple(proposals)


def batched_pointer_inference(
    model: NanoPointerModel,
    inputs: Sequence[PointerInferenceInput],
    *,
    device: str,
    batch_size: int,
) -> tuple[PointerPrediction, ...]:
    """Run direct state/pointer inference in right-padded batches."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise PointerEvaluationError("batch_size must be a positive integer")
    frozen = tuple(inputs)
    if not frozen:
        raise PointerEvaluationError("pointer inference requires at least one input")
    predictions: list[PointerPrediction] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frozen), batch_size):
            batch = frozen[start : start + batch_size]
            maximum = max(len(item.token_ids) for item in batch)
            token_rows = [
                [*item.token_ids, *([0] * (maximum - len(item.token_ids)))]
                for item in batch
            ]
            mask_rows = [
                [*item.attention_mask, *([False] * (maximum - len(item.token_ids)))]
                for item in batch
            ]
            token_ids = torch.tensor(token_rows, dtype=torch.long, device=device)
            attention_mask = torch.tensor(mask_rows, dtype=torch.bool, device=device)
            outputs = model(token_ids, attention_mask=attention_mask)
            if (
                tuple(outputs.state_logits.shape)
                != (len(batch), len(FIELD_ORDER), len(STATE_ORDER))
                or tuple(outputs.start_logits.shape)
                != (len(batch), maximum, len(FIELD_ORDER), 2)
                or tuple(outputs.end_logits.shape)
                != (len(batch), maximum, len(FIELD_ORDER), 2)
            ):
                raise PointerEvaluationError(
                    "pointer model returned invalid logit shapes"
                )
            state_rows = outputs.state_logits.detach().cpu()
            start_rows = outputs.start_logits.detach().cpu()
            end_rows = outputs.end_logits.detach().cpu()
            for row, item in enumerate(batch):
                length = len(item.token_ids)
                try:
                    proposals = decode_pointer_logits(
                        item,
                        state_rows[row],
                        start_rows[row, :length],
                        end_rows[row, :length],
                    )
                except PointerDecodeError as exc:
                    predictions.append(PointerPrediction(error=str(exc)))
                else:
                    predictions.append(PointerPrediction(proposals=proposals))
    if len(predictions) != len(frozen):
        raise RuntimeError("pointer inference returned the wrong row count")
    return tuple(predictions)


def _span_key(span: EvidenceSpan) -> tuple[int, int, str, str]:
    return (span.start, span.end, span.text, span.speaker)


def _proposal_exact(proposal: StateSpanProposal, gold: FieldOutput) -> bool:
    if proposal.state is not gold.state or {
        _span_key(span) for span in proposal.spans
    } != {_span_key(span) for span in gold.evidence}:
        return False
    if proposal.state is FieldState.SUPPORTED:
        return (
            gold.value is not None
            and len(proposal.spans) == 1
            and normalize_value(proposal.spans[0].text) == gold.value
        )
    return True


def _span_exact(proposal: StateSpanProposal, gold: FieldOutput) -> bool:
    return {_span_key(span) for span in proposal.spans} == {
        _span_key(span) for span in gold.evidence
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _empty_raw_bucket() -> dict[str, int]:
    return {
        "total": 0,
        "decoded": 0,
        "state_correct": 0,
        "span_exact": 0,
        "joint_exact": 0,
        "presented": 0,
        "wrong_presented": 0,
    }


def _finish_raw_bucket(bucket: Mapping[str, int]) -> dict[str, int | float | None]:
    return {
        **bucket,
        "decode_rate": _rate(bucket["decoded"], bucket["total"]),
        "state_accuracy": _rate(bucket["state_correct"], bucket["total"]),
        "span_exact_accuracy": _rate(bucket["span_exact"], bucket["total"]),
        "joint_exact_accuracy": _rate(bucket["joint_exact"], bucket["total"]),
        "wrong_presented_rate": _rate(bucket["wrong_presented"], bucket["presented"]),
    }


def raw_pointer_diagnostics(
    examples: Sequence[StateSpanExample],
    cases: Sequence[Any],
    predictions: Sequence[PointerPrediction],
) -> dict[str, Any]:
    """Score model states and reconstructed spans before verifier gating."""

    if not (len(examples) == len(cases) == len(predictions)):
        raise PointerEvaluationError("raw pointer diagnostic row counts do not match")
    aggregate = _empty_raw_bucket()
    by_field = {field.value: _empty_raw_bucket() for field in FIELD_ORDER}
    by_gold_state = {state.value: _empty_raw_bucket() for state in FieldState}
    target_challenge = {
        state.value: _empty_raw_bucket()
        for state in (
            FieldState.MISSING,
            FieldState.UNCERTAIN,
            FieldState.CONFLICTING,
        )
    }
    held_value_fields = frozenset(
        {
            FieldName.CHIEF_COMPLAINT,
            FieldName.DURATION,
            FieldName.MEDICATION,
            FieldName.ALLERGY,
        }
    )
    acceptance_counts = {
        "overall": [0, 0],
        "held_value": [0, 0],
        "missing_target": [0, 0],
        "uncertain_target": [0, 0],
        "conflict_target": [0, 0],
        "absence": [0, 0],
    }
    failure_items = 0
    item_rows: list[dict[str, Any]] = []

    for example, case, prediction in zip(examples, cases, predictions, strict=True):
        for field_name in FIELD_ORDER:
            gold = case.gold.field(field_name)
            aggregate["total"] += 1
            by_field[field_name.value]["total"] += 1
            by_gold_state[gold.state.value]["total"] += 1
            acceptance_counts["overall"][1] += 1
            if gold.state is FieldState.SUPPORTED and field_name in held_value_fields:
                acceptance_counts["held_value"][1] += 1
            if gold.state is FieldState.ABSENT:
                acceptance_counts["absence"][1] += 1
        if example.target_state is not None:
            target_challenge[example.target_state.value]["total"] += 1
        target_metric = {
            "missing": "missing_target",
            "uncertain": "uncertain_target",
            "conflicting": "conflict_target",
        }.get(example.variant)
        if target_metric is not None:
            acceptance_counts[target_metric][1] += 1
        if prediction.error is not None:
            failure_items += 1
            item_rows.append(
                {
                    "example_id": example.example_id,
                    "variant": example.variant,
                    "target_field": example.target_field.value,
                    "target_state": (
                        None
                        if example.target_state is None
                        else example.target_state.value
                    ),
                    "decode_status": "failure",
                    "decode_error": prediction.error,
                    "fields": None,
                }
            )
            continue
        assert prediction.proposals is not None
        field_rows: list[dict[str, Any]] = []
        for expected_field, proposal in zip(
            FIELD_ORDER, prediction.proposals, strict=True
        ):
            if proposal.field is not expected_field:
                raise PointerEvaluationError(
                    "decoded pointer fields are not canonically ordered"
                )
            gold = case.gold.field(proposal.field)
            state_correct = proposal.state is gold.state
            span_exact = _span_exact(proposal, gold)
            joint_exact = _proposal_exact(proposal, gold)
            presented = proposal.state in _PRESENTED_STATES
            wrong_presented = presented and not joint_exact
            buckets = (
                aggregate,
                by_field[proposal.field.value],
                by_gold_state[gold.state.value],
            )
            for bucket in buckets:
                bucket["decoded"] += 1
                bucket["state_correct"] += int(state_correct)
                bucket["span_exact"] += int(span_exact)
                bucket["joint_exact"] += int(joint_exact)
                bucket["presented"] += int(presented)
                bucket["wrong_presented"] += int(wrong_presented)
            acceptance_counts["overall"][0] += int(joint_exact)
            if (
                gold.state is FieldState.SUPPORTED
                and proposal.field in held_value_fields
            ):
                acceptance_counts["held_value"][0] += int(joint_exact)
            if gold.state is FieldState.ABSENT:
                acceptance_counts["absence"][0] += int(joint_exact)
            if target_metric is not None and proposal.field is example.target_field:
                acceptance_counts[target_metric][0] += int(joint_exact)
            if (
                example.target_state is not None
                and proposal.field is example.target_field
            ):
                target = target_challenge[example.target_state.value]
                target["decoded"] += 1
                target["state_correct"] += int(state_correct)
                target["span_exact"] += int(span_exact)
                target["joint_exact"] += int(joint_exact)
                target["presented"] += int(presented)
                target["wrong_presented"] += int(wrong_presented)
            field_rows.append(
                {
                    "field": proposal.field.value,
                    "raw_state": proposal.state.value,
                    "gold_state": gold.state.value,
                    "state_correct": state_correct,
                    "span_exact": span_exact,
                    "joint_exact": joint_exact,
                    "presented": presented,
                    "wrong_presented": wrong_presented,
                    "raw_spans": [span.to_dict() for span in proposal.spans],
                    "gold_spans": [span.to_dict() for span in gold.evidence],
                }
            )
        item_rows.append(
            {
                "example_id": example.example_id,
                "variant": example.variant,
                "target_field": example.target_field.value,
                "target_state": (
                    None if example.target_state is None else example.target_state.value
                ),
                "decode_status": "decoded",
                "decode_error": None,
                "fields": field_rows,
            }
        )
    acceptance_metrics = {
        name: {
            "numerator": numerator,
            "denominator": denominator,
            "rate": _rate(numerator, denominator),
        }
        for name, (numerator, denominator) in acceptance_counts.items()
    }
    acceptance_metrics["failures"] = {
        "numerator": failure_items,
        "denominator": len(examples),
        "rate": _rate(failure_items, len(examples)),
    }
    acceptance_metrics["false_presented"] = {
        "numerator": aggregate["wrong_presented"],
        "denominator": aggregate["presented"],
        "rate": _rate(aggregate["wrong_presented"], aggregate["presented"]),
    }
    if acceptance_metrics["overall"]["numerator"] != aggregate["joint_exact"]:
        raise RuntimeError("raw acceptance overall metric disagrees with diagnostics")
    if acceptance_metrics["failures"]["rate"] != _rate(failure_items, len(examples)):
        raise RuntimeError("raw failure metric disagrees with diagnostics")
    return {
        "items": len(examples),
        "decoded_items": len(examples) - failure_items,
        "decode_failure_items": failure_items,
        "decode_failure_rate": _rate(failure_items, len(examples)),
        "fields": _finish_raw_bucket(aggregate),
        "wrong_presented_field_count": aggregate["wrong_presented"],
        "by_field": {
            name: _finish_raw_bucket(bucket) for name, bucket in by_field.items()
        },
        "by_gold_state": {
            name: _finish_raw_bucket(bucket) for name, bucket in by_gold_state.items()
        },
        "target_challenge": {
            name: _finish_raw_bucket(bucket)
            for name, bucket in target_challenge.items()
        },
        "acceptance": {
            "metrics": acceptance_metrics,
            "definitions": {
                "overall": (
                    "raw joint state, model-owned normalized value, and exact-span "
                    "correctness over every development field"
                ),
                "held_value": (
                    "raw joint state, model-owned normalized value, and exact-span "
                    "correctness on gold-supported chief complaint, duration, "
                    "medication, and allergy fields"
                ),
                "missing_target": (
                    "raw joint correctness on the designated missing-variant field"
                ),
                "uncertain_target": (
                    "raw joint correctness on the designated uncertain-variant field"
                ),
                "conflict_target": (
                    "raw joint correctness on the designated conflicting-variant field"
                ),
                "absence": "raw joint correctness over every gold-absent field",
                "failures": "raw item decoding failures over every development item",
                "false_presented": (
                    "raw supported/absent proposals lacking joint state, model-owned "
                    "normalized value, and exact-span correctness"
                ),
            },
            "failure_denominator_note": (
                "Decode failures remain incorrect in every applicable metric denominator."
            ),
        },
        "item_diagnostics": item_rows,
    }


def _prediction_map(
    inputs: Sequence[PointerInferenceInput],
    predictions: Sequence[PointerPrediction],
) -> dict[str, PointerPrediction]:
    if len(inputs) != len(predictions):
        raise PointerEvaluationError("pointer prediction row count is invalid")
    result: dict[str, PointerPrediction] = {}
    for item, prediction in zip(inputs, predictions, strict=True):
        previous = result.setdefault(item.transcript, prediction)
        if previous != prediction:
            raise PointerEvaluationError(
                "identical transcripts received inconsistent pointer predictions"
            )
    return result


def _evaluate_predictions(
    *,
    inputs: Sequence[PointerInferenceInput],
    predictions: Sequence[PointerPrediction],
    cases: Sequence[Any],
    candidate: PointerCandidateCheckpoint,
) -> EvaluationReport:
    prediction_by_transcript = _prediction_map(inputs, predictions)

    def predict(transcript: str) -> tuple[StateSpanProposal, ...]:
        prediction = prediction_by_transcript[transcript]
        if prediction.error is not None:
            raise PointerDecodeError(prediction.error)
        assert prediction.proposals is not None
        return prediction.proposals

    solver = PointerSpanSolver(
        predict,
        solver_id=(
            f"development/native-pointer/{candidate.label}/sha-{candidate.sha256}"
        ),
        version="sealed-dev-v0",
        artifact_bytes=candidate.provenance.get("artifact_bytes"),
    )
    return evaluate_solver(solver, cases, measure_latency=False)


def load_authenticated_frozen_base(
    report_path: str | Path,
    *,
    expected_report_sha256: str,
    manifest_sha256: str,
    development_sha256: str,
    examples: Sequence[StateSpanExample],
) -> AuthenticatedFrozenBase:
    """Authenticate and independently recompute the frozen H1 base aggregates."""

    snapshot = _read_verified_file(
        Path(report_path),
        expected_report_sha256,
        role="frozen-base H1 development report",
    )
    report = _parse_json(snapshot, role="frozen-base H1 development report")
    top_keys = {
        "schema_version",
        "status",
        "partition",
        "artifacts",
        "protocol",
        "runtime",
        "source_sha256",
        "frozen_base",
        "candidates",
        "selection_boundary",
    }
    if not isinstance(report, dict) or set(report) != top_keys:
        raise PointerEvaluationError("frozen-base H1 report schema is invalid")
    if (
        report["schema_version"] != H1_EVALUATION_SCHEMA_VERSION
        or report["status"] != "complete"
    ):
        raise PointerEvaluationError("frozen-base H1 report is not complete v0")

    partition = report["partition"]
    if not isinstance(partition, dict) or (
        partition.get("partition_id") != DEVELOPMENT_PARTITION_ID
        or partition.get("manifest_sha256") != manifest_sha256
        or partition.get("development_sha256") != development_sha256
        or partition.get("records") != len(examples)
        or partition.get("worlds") != len({example.world_id for example in examples})
        or partition.get("target_grammar") != TARGET_GRAMMAR_VERSION
        or partition.get("historical_benchmark_read") is not False
    ):
        raise PointerEvaluationError(
            "frozen-base H1 report used another development partition"
        )
    artifacts = report["artifacts"]
    if not isinstance(artifacts, dict) or (
        artifacts.get("frozen_base_checkpoint_sha256")
        != FROZEN_NANO_V01.checkpoint_sha256
        or artifacts.get("tokenizer_sha256") != FROZEN_NANO_V01.tokenizer_sha256
        or artifacts.get("architecture_identity")
        != FROZEN_NANO_V01.architecture_identity
        or artifacts.get("parameter_count") != NANO_MODEL_CONFIG.parameter_count
    ):
        raise PointerEvaluationError("frozen-base H1 model identity is invalid")

    frozen = report["frozen_base"]
    if not isinstance(frozen, dict) or set(frozen) != {
        "checkpoint_sha256",
        "evaluation",
        "final_state",
        "acceptance",
    }:
        raise PointerEvaluationError("frozen-base H1 metric record is invalid")
    if frozen["checkpoint_sha256"] != FROZEN_NANO_V01.checkpoint_sha256:
        raise PointerEvaluationError("frozen-base H1 checkpoint identity changed")
    evaluation = frozen["evaluation"]
    if not isinstance(evaluation, dict) or (
        evaluation.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or not isinstance(evaluation.get("items"), list)
        or len(evaluation["items"]) != len(examples)
        or [row.get("case_id") for row in evaluation["items"]]
        != [example.example_id for example in examples]
        or not isinstance(evaluation.get("quality"), dict)
        or not isinstance(evaluation.get("failures"), dict)
    ):
        raise PointerEvaluationError("frozen-base standard evaluation is invalid")
    solver = evaluation.get("solver")
    operational = evaluation.get("operational")
    if not isinstance(solver, dict) or solver.get("parameter_count") != (
        NANO_MODEL_CONFIG.parameter_count
    ):
        raise PointerEvaluationError("frozen-base solver identity is invalid")
    if (
        not isinstance(operational, dict)
        or operational.get("latency_measured") is not False
    ):
        raise PointerEvaluationError("frozen-base comparison must be untimed")

    proxy = SimpleNamespace(
        items=tuple(evaluation["items"]),
        quality=evaluation["quality"],
        failures=evaluation["failures"],
    )
    try:
        recomputed_final = final_state_diagnostics(proxy, examples)
        recomputed_acceptance = acceptance_diagnostics(proxy, examples)
    except Exception as exc:
        raise PointerEvaluationError(
            "frozen-base H1 metrics could not be recomputed"
        ) from exc
    if frozen["final_state"] != recomputed_final:
        raise PointerEvaluationError("frozen-base final-state metrics are inconsistent")
    if frozen["acceptance"] != recomputed_acceptance:
        raise PointerEvaluationError("frozen-base acceptance metrics are inconsistent")
    return AuthenticatedFrozenBase(
        report_sha256=_sha256(snapshot),
        checkpoint_sha256=FROZEN_NANO_V01.checkpoint_sha256,
        evaluation={
            key: evaluation[key]
            for key in (
                "schema_version",
                "solver",
                "quality",
                "failures",
                "operational",
                "pipeline",
            )
        },
        final_state=recomputed_final,
        acceptance=recomputed_acceptance,
    )


def _comparison_to_base(
    candidate_report: EvaluationReport,
    candidate_raw: Mapping[str, Any],
    candidate_final: Mapping[str, Any],
    candidate_acceptance: Mapping[str, Any],
    frozen_base: AuthenticatedFrozenBase,
) -> dict[str, Any]:
    candidate_quality = candidate_report.quality
    base_quality = frozen_base.evaluation["quality"]

    def delta(key: str) -> float:
        return float(candidate_quality[key]) - float(base_quality[key])

    challenge_delta: dict[str, float] = {}
    for state in ("missing", "uncertain", "conflicting"):
        candidate_rate = candidate_final["target_challenge"][state][
            "grounded_exact_accuracy"
        ]
        base_rate = frozen_base.final_state["target_challenge"][state][
            "grounded_exact_accuracy"
        ]
        if candidate_rate is None or base_rate is None:
            raise PointerEvaluationError("challenge comparison has no denominator")
        challenge_delta[state] = candidate_rate - base_rate

    acceptance_comparison: dict[str, Any] = {}
    for name in (
        "overall",
        "held_value",
        "missing_target",
        "conflict_target",
        "absence",
        "failures",
        "false_presented",
    ):
        candidate_metric = candidate_acceptance["metrics"][name]
        base_metric = frozen_base.acceptance["metrics"][name]
        candidate_rate = candidate_metric["rate"]
        base_rate = base_metric["rate"]
        rate_delta = (
            None
            if candidate_rate is None or base_rate is None
            else candidate_rate - base_rate
        )
        acceptance_comparison[name] = {
            "candidate": candidate_metric,
            "frozen_base": base_metric,
            "candidate_minus_base_rate": rate_delta,
        }

    def measured_delta(name: str) -> float:
        value = acceptance_comparison[name]["candidate_minus_base_rate"]
        if value is None:
            raise PointerEvaluationError(f"quality gate {name} has no denominator")
        return float(value)

    final_gates = {
        "overall_gain_at_least_5pp": measured_delta("overall") >= 0.05,
        "held_value_gain_at_least_10pp": measured_delta("held_value") >= 0.10,
        "missing_target_gain_at_least_50pp": (measured_delta("missing_target") >= 0.50),
        "failure_rate_at_most_1pct": (
            candidate_acceptance["metrics"]["failures"]["rate"] <= 0.01
        ),
        "zero_false_presented": (
            candidate_acceptance["metrics"]["false_presented"]["numerator"] == 0
        ),
        "absence_regression_no_more_than_1pp": measured_delta("absence") >= -0.01,
        "conflict_target_regression_no_more_than_1pp": (
            measured_delta("conflict_target") >= -0.01
        ),
    }

    raw_acceptance = candidate_raw.get("acceptance")
    if not isinstance(raw_acceptance, Mapping) or not isinstance(
        raw_acceptance.get("metrics"), Mapping
    ):
        raise PointerEvaluationError("raw acceptance metrics are unavailable")
    raw_metrics = raw_acceptance["metrics"]

    def require_metric(
        collection: Mapping[str, Any],
        name: str,
        role: str,
        *,
        allow_zero_denominator: bool = False,
    ) -> Mapping[str, Any]:
        metric = collection.get(name)
        if not isinstance(metric, Mapping) or set(metric) != {
            "numerator",
            "denominator",
            "rate",
        }:
            raise PointerEvaluationError(f"{role} {name} metric is invalid")
        numerator = metric["numerator"]
        denominator = metric["denominator"]
        rate = metric["rate"]
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or numerator < 0
            or isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator < (0 if allow_zero_denominator else 1)
            or numerator > denominator
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
            raise PointerEvaluationError(f"{role} {name} metric is inconsistent")
        return metric

    base_acceptance_metrics = frozen_base.acceptance["metrics"]
    if not isinstance(base_acceptance_metrics, Mapping):
        raise PointerEvaluationError("frozen-base acceptance metrics are invalid")

    def delta_gate(
        *,
        candidate_name: str,
        base_name: str,
        required_delta: float,
    ) -> dict[str, Any]:
        candidate_metric = require_metric(raw_metrics, candidate_name, "raw")
        base_metric = require_metric(base_acceptance_metrics, base_name, "frozen-base")
        candidate_rate = float(candidate_metric["rate"])
        base_rate = float(base_metric["rate"])
        threshold_rate = base_rate + required_delta
        return {
            "candidate": dict(candidate_metric),
            "frozen_base": dict(base_metric),
            "required_delta": required_delta,
            "threshold_rate": threshold_rate,
            "candidate_minus_base_rate": candidate_rate - base_rate,
            "candidate_minus_threshold_rate": candidate_rate - threshold_rate,
            "passed": candidate_rate >= threshold_rate,
        }

    raw_gate_evidence = {
        "raw_overall_gain_at_least_5pp": delta_gate(
            candidate_name="overall", base_name="overall", required_delta=0.05
        ),
        "raw_held_value_gain_at_least_10pp": delta_gate(
            candidate_name="held_value",
            base_name="held_value",
            required_delta=0.10,
        ),
        "raw_missing_target_gain_at_least_50pp": delta_gate(
            candidate_name="missing_target",
            base_name="missing_target",
            required_delta=0.50,
        ),
        "raw_absence_regression_no_more_than_1pp": delta_gate(
            candidate_name="absence", base_name="absence", required_delta=-0.01
        ),
        "raw_conflict_target_regression_no_more_than_1pp": delta_gate(
            candidate_name="conflict_target",
            base_name="conflict_target",
            required_delta=-0.01,
        ),
    }

    base_uncertain = frozen_base.final_state["target_challenge"]["uncertain"]
    if not isinstance(base_uncertain, Mapping):
        raise PointerEvaluationError("frozen-base uncertain metric is invalid")
    base_uncertain_metric = {
        "numerator": base_uncertain.get("grounded_exact"),
        "denominator": base_uncertain.get("total"),
        "rate": base_uncertain.get("grounded_exact_accuracy"),
    }
    # The H1 acceptance table omitted uncertain-target, so recover the exact
    # matched numerator and denominator from its authenticated final-state table.
    uncertain_candidate = require_metric(raw_metrics, "uncertain_target", "raw")
    uncertain_base = require_metric(
        {"uncertain_target": base_uncertain_metric},
        "uncertain_target",
        "frozen-base",
    )
    uncertain_threshold = float(uncertain_base["rate"]) - 0.01
    raw_gate_evidence["raw_uncertain_target_regression_no_more_than_1pp"] = {
        "candidate": dict(uncertain_candidate),
        "frozen_base": dict(uncertain_base),
        "required_delta": -0.01,
        "threshold_rate": uncertain_threshold,
        "candidate_minus_base_rate": (
            float(uncertain_candidate["rate"]) - float(uncertain_base["rate"])
        ),
        "candidate_minus_threshold_rate": (
            float(uncertain_candidate["rate"]) - uncertain_threshold
        ),
        "passed": float(uncertain_candidate["rate"]) >= uncertain_threshold,
    }

    raw_failures = require_metric(raw_metrics, "failures", "raw")
    base_failures = require_metric(base_acceptance_metrics, "failures", "frozen-base")
    raw_gate_evidence["raw_decode_failure_rate_at_most_1pct"] = {
        "candidate": dict(raw_failures),
        "frozen_base": dict(base_failures),
        "threshold_rate": 0.01,
        "candidate_minus_threshold_rate": float(raw_failures["rate"]) - 0.01,
        "passed": float(raw_failures["rate"]) <= 0.01,
    }
    raw_false_presented = require_metric(
        raw_metrics,
        "false_presented",
        "raw",
        allow_zero_denominator=True,
    )
    base_false_presented = require_metric(
        base_acceptance_metrics, "false_presented", "frozen-base"
    )
    raw_gate_evidence["zero_raw_wrong_presented"] = {
        "candidate": dict(raw_false_presented),
        "frozen_base": dict(base_false_presented),
        "threshold_numerator": 0,
        "candidate_minus_threshold_numerator": raw_false_presented["numerator"],
        "passed": raw_false_presented["numerator"] == 0,
    }
    raw_gate_results = {
        name: bool(evidence["passed"]) for name, evidence in raw_gate_evidence.items()
    }
    raw_all = all(raw_gate_results.values())
    final_all = all(final_gates.values())
    combined_all = raw_all and final_all
    return {
        "grounded_exact_field_accuracy_delta": delta("grounded_exact_field_accuracy"),
        "state_accuracy_delta": delta("state_accuracy"),
        "inference_failure_rate_delta": (
            float(candidate_report.failures["rate"])
            - float(frozen_base.evaluation["failures"]["rate"])
        ),
        "target_challenge_grounded_exact_accuracy_delta": challenge_delta,
        "wrong_presented_field_count_delta": (
            candidate_report.quality["false_presented_count"]
            - frozen_base.evaluation["quality"]["false_presented_count"]
        ),
        "acceptance_metrics": acceptance_comparison,
        "acceptance_gates": {
            **final_gates,
            "all_measured_quality_gates_pass": final_all,
            "latency_gate_assessed": False,
            "fresh_v1_confirmation_assessed": False,
        },
        "raw_eligibility": {
            "gate_evidence": raw_gate_evidence,
            "gates": {
                **raw_gate_results,
                "all_raw_eligibility_gates_pass": raw_all,
            },
            "scope_note": (
                "Raw gates score model-owned state and exact span proposals before "
                "PointerSpanSolver verification; decode failures remain incorrect "
                "in every applicable denominator."
            ),
        },
        "quality_eligibility": {
            "raw_all": raw_all,
            "final_all": final_all,
            "all_quality_gates_pass": combined_all,
            "latency_gate_assessed": False,
            "fresh_v1_confirmation_assessed": False,
        },
    }


def _source_paths() -> dict[str, Path]:
    package_dir = Path(__file__).parent.parent
    return {
        "development_evaluator": Path(__file__),
        "sealed_development_loader": Path(__file__).with_name("evaluate_state_span.py"),
        "state_span_data": Path(state_span_data.__file__),
        "evaluation": package_dir / "evaluation.py",
        "deterministic_binder": package_dir / "adapters" / "deterministic_v0.py",
        "state_span_verifier": package_dir / "adapters" / "state_span.py",
        "pointer_adapter": package_dir / "adapters" / "pointer_span.py",
        "pointer_data": Path(pointer_data.__file__),
        "pointer_model": Path(__file__).with_name("pointer_model.py"),
    }


def _source_hashes() -> dict[str, str]:
    return {
        name: _sha256(_read_regular_file(path, role=f"{name} source"))
        for name, path in sorted(_source_paths().items())
    }


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        with Path(path).open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise PointerEvaluationError("evaluation output already exists") from exc


def evaluate_development(
    *,
    data_dir: str | Path,
    manifest_sha256: str,
    tokenizer_path: str | Path,
    frozen_base_report: str | Path,
    frozen_base_report_sha256: str,
    training_reports: Sequence[tuple[str | Path, str]],
    output_path: str | Path,
    device: str = "cpu",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Mapping[str, Any]:
    """Evaluate both fixed H2 seeds and stop at quality gates."""

    output = Path(output_path)
    if output.exists():
        raise PointerEvaluationError("evaluation output already exists")
    if not output.parent.is_dir():
        raise PointerEvaluationError("evaluation output parent must exist")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise PointerEvaluationError("batch_size must be a positive integer")
    resolved_device = _resolve_device(device)
    source_hashes = _source_hashes()

    try:
        bundle = load_development_bundle(
            data_dir,
            expected_manifest_sha256=manifest_sha256,
        )
        tokenizer = load_pointer_tokenizer(Path(tokenizer_path))
        encoded = encode_pointer_partition(
            tokenizer,
            bundle.examples,
            expected_split="dev",
        )
    except PointerEvaluationError:
        raise
    except Exception as exc:
        raise PointerEvaluationError(
            "sealed pointer development inputs could not be verified"
        ) from exc
    inputs = build_pointer_inference_inputs(bundle.examples, encoded)
    # The H1 helper is executable evidence and is source-hashed above.
    cases = _fixture_cases(bundle.examples)
    frozen_base = load_authenticated_frozen_base(
        frozen_base_report,
        expected_report_sha256=frozen_base_report_sha256,
        manifest_sha256=bundle.manifest_sha256,
        development_sha256=bundle.dev_sha256,
        examples=bundle.examples,
    )

    if len(training_reports) != len(TRAINING_SEEDS):
        raise PointerEvaluationError(
            "exactly two fixed-seed training reports are required"
        )
    candidates: list[PointerCandidateCheckpoint] = []
    for path, digest in training_reports:
        candidates.extend(
            load_candidates_from_training_report(
                path,
                expected_report_sha256=digest,
                expected_manifest_sha256=bundle.manifest_sha256,
                expected_dev_sha256=bundle.dev_sha256,
                expected_train_sha256=bundle.manifest["train"]["sha256"],
            )
        )
    seeds = {candidate.provenance["training_seed"] for candidate in candidates}
    if seeds != set(TRAINING_SEEDS) or len(candidates) != 6:
        raise PointerEvaluationError(
            "training reports do not cover both fixed H2 seeds"
        )
    candidates.sort(
        key=lambda candidate: (
            candidate.provenance["training_seed"],
            candidate.provenance["epoch"],
        )
    )
    labels = [candidate.label for candidate in candidates]
    digests = [candidate.sha256 for candidate in candidates]
    if len(set(labels)) != len(labels) or len(set(digests)) != len(digests):
        raise PointerEvaluationError(
            "candidate labels and checkpoint digests must be unique"
        )

    _seed_evaluation()
    candidate_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        model = _load_pointer_model(candidate, device=resolved_device)
        predictions = batched_pointer_inference(
            model,
            inputs,
            device=resolved_device,
            batch_size=batch_size,
        )
        del model
        evaluation = _evaluate_predictions(
            inputs=inputs,
            predictions=predictions,
            cases=cases,
            candidate=candidate,
        )
        raw = raw_pointer_diagnostics(
            bundle.examples,
            cases,
            predictions,
        )
        final = final_state_diagnostics(evaluation, bundle.examples)
        acceptance = acceptance_diagnostics(evaluation, bundle.examples)
        comparison = _comparison_to_base(
            evaluation,
            raw,
            final,
            acceptance,
            frozen_base,
        )
        candidate_rows.append(
            {
                "label": candidate.label,
                "checkpoint_sha256": candidate.sha256,
                "provenance": dict(candidate.provenance),
                "evaluation": evaluation.to_dict(),
                "raw_state_pointer": raw,
                "final_state": final,
                "acceptance": acceptance,
                "comparison_to_frozen_base": comparison,
            }
        )

    passing = [
        row["label"]
        for row in candidate_rows
        if row["comparison_to_frozen_base"]["quality_eligibility"][
            "all_quality_gates_pass"
        ]
    ]
    if _source_hashes() != source_hashes:
        raise PointerEvaluationError("evaluation source changed during the run")
    result = {
        "schema_version": POINTER_DEVELOPMENT_EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "partition": {
            "partition_id": DEVELOPMENT_PARTITION_ID,
            "role": "sealed_development_model_selection_only",
            "manifest_sha256": bundle.manifest_sha256,
            "development_sha256": bundle.dev_sha256,
            "records": len(bundle.examples),
            "worlds": len({example.world_id for example in bundle.examples}),
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "historical_benchmark_read": False,
            "fresh_v1_read": False,
        },
        "artifacts": {
            "frozen_base_checkpoint_sha256": FROZEN_NANO_V01.checkpoint_sha256,
            "tokenizer_sha256": FROZEN_NANO_V01.tokenizer_sha256,
            "architecture_identity": FROZEN_NANO_V01.architecture_identity,
            "trunk_parameter_count": NANO_TRUNK_PARAMETER_COUNT,
            "pointer_head_parameter_count": POINTER_HEAD_PARAMETER_COUNT,
            "parameter_count": NANO_POINTER_PARAMETER_COUNT,
        },
        "protocol": {
            "supervision_version": POINTER_SUPERVISION_VERSION,
            "state_class_order": [state.value for state in STATE_ORDER],
            "state_to_pointer_count": {
                state.value: count for state, count in _POINTER_COUNT.items()
            },
            "patient_only_pointer_mask": True,
            "pointer_constraint": "start<=end_and_same_patient_turn",
            "cross_field_span_reuse_allowed": True,
            "conflict_spans_field_local_normalized_distinct": True,
            "uncertain_pointer_count": 1,
            "causal_pointer_heads": True,
            "edge_whitespace_trim_only": True,
            "batch_size": batch_size,
            "latency_measured": False,
            "fresh_v1_confirmation_assessed": False,
        },
        "runtime": {
            "device": resolved_device,
            "deterministic_algorithms": True,
            "cublas_workspace_config": (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if resolved_device == "cuda"
                else None
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tokenizers": getattr(__import__("tokenizers"), "__version__", None),
        },
        "source_sha256": source_hashes,
        "frozen_base": {
            "h1_development_report_sha256": frozen_base.report_sha256,
            "checkpoint_sha256": frozen_base.checkpoint_sha256,
            "evaluation": dict(frozen_base.evaluation),
            "final_state": dict(frozen_base.final_state),
            "acceptance": dict(frozen_base.acceptance),
        },
        "candidates": candidate_rows,
        "decision": {
            "quality_gate_passed_candidates": passing,
            "quality_gate_passed": bool(passing),
            "latency_assessed": False,
            "fresh_v1_assessed": False,
            "next_step": (
                "measure matched latency before any fresh-v1 confirmation"
                if passing
                else "reject H2; do not measure latency or open fresh-v1"
            ),
        },
        "selection_boundary": (
            "This report is sealed-development evidence only. No checkpoint is "
            "promoted here. Latency and fresh-v1 remain inaccessible unless a "
            "candidate passes every measured quality gate."
        ),
    }
    _write_json_no_clobber(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate Nano pointer candidates on sealed development"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument(
        "--tokenizer", type=Path, default=root / "sft" / "tokenizer.json"
    )
    parser.add_argument("--frozen-base-report", type=Path, required=True)
    parser.add_argument("--frozen-base-report-sha256", required=True)
    parser.add_argument(
        "--training-report",
        nargs=2,
        action="append",
        required=True,
        metavar=("REPORT", "SHA256"),
        help="repeat exactly twice, once for each digest-pinned H2 seed",
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
        frozen_base_report=args.frozen_base_report,
        frozen_base_report_sha256=args.frozen_base_report_sha256,
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
    "POINTER_DEVELOPMENT_EVALUATION_SCHEMA_VERSION",
    "AuthenticatedFrozenBase",
    "PointerCandidateCheckpoint",
    "PointerDecodeError",
    "PointerEvaluationError",
    "PointerInferenceInput",
    "PointerPrediction",
    "batched_pointer_inference",
    "build_pointer_inference_inputs",
    "decode_pointer_logits",
    "evaluate_development",
    "load_authenticated_frozen_base",
    "load_candidates_from_training_report",
    "raw_pointer_diagnostics",
]
