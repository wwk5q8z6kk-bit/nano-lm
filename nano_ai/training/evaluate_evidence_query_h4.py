"""Known-development evaluation for Nano's frozen H4 data-only intervention.

H4's two fixed-seed reports, checkpoints, training data, source pins, and
runtime are authenticated and the primary candidate is selected before the
known H2 development partition is opened.  Development can decide only the
frozen quality gate; it cannot select a seed, epoch, threshold, or data family.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.training import evaluate_evidence_query as h3_evaluation
from nano_ai.training import (
    evidence_query_inference,
    evidence_query_model,
    pointer_data,
    state_span_data,
    surface_transfer_data,
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
from nano_ai.training.train_evidence_query_h4 import (
    CALIBRATION_STATE_CLASS_COUNTS,
    FIT_STATE_CLASS_COUNTS,
    H4_TRAINING_RECIPE_VERSION,
    H4_TRAINING_REPORT_SCHEMA_VERSION,
    PRESERVED_H3_SOURCE_SHA256,
    changed_source_paths,
    load_h4_training_bundle,
    preserved_source_paths,
)

H4_DEVELOPMENT_EVALUATION_SCHEMA_VERSION = (
    "nano.evidence-query-h4-development-evaluation.v1"
)
TRAINING_SEEDS = (20260805, 20260806)
DEFAULT_BATCH_SIZE = 32
_TRAINING_SELECTION_NOTE = (
    "The selected H4 epoch used only the disjoint 200-world "
    "training-calibration partition. Gradients used only the 2,800-world "
    "fit partition. The training command read no development, benchmark, "
    "historical-fresh, or sealed-confirmation records. H3 architecture, "
    "objective, optimizer, exposure, seeds, inference, calibration, and "
    "selection rules were preserved by literal source pins."
)

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

# H4 intentionally reuses the accepted H3 evaluator, inference, and exact
# semantic floors.  The alias also keeps malformed H4 artifacts on the same
# fail-closed exception surface as H3.
EvidenceQueryEvaluationError = h3_evaluation.EvidenceQueryEvaluationError
_quality_gates = h3_evaluation._quality_gates
_final_metrics = h3_evaluation._final_metrics
_epoch_calibration = h3_evaluation._epoch_calibration
_seed_evaluation = h3_evaluation._seed_evaluation
_load_evidence_query_model = h3_evaluation._load_evidence_query_model
_evaluate_predictions = h3_evaluation._evaluate_predictions


@dataclass(frozen=True, slots=True)
class EvidenceQueryH4Candidate:
    """One report-selected H4 checkpoint with training-only ranking facts."""

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


EvidenceQueryCandidate = EvidenceQueryH4Candidate


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
    return _source_hashes(changed_source_paths(), role="changed H4")


def _checkpoint_record(value: object) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_KEYS:
        raise EvidenceQueryEvaluationError("H4 candidate checkpoint record is invalid")
    filename = value["filename"]
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise EvidenceQueryEvaluationError("H4 checkpoint filename is unsafe")
    _require_sha256(value["sha256"], "H4 checkpoint")
    byte_count = value["bytes"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise EvidenceQueryEvaluationError("H4 checkpoint byte count is invalid")
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
        raise EvidenceQueryEvaluationError(f"H4 {role} identity is invalid")
    byte_count = value.get("bytes")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise EvidenceQueryEvaluationError(f"H4 {role} byte count is invalid")
    _require_sha256(value.get("sha256"), f"H4 {role}")
    if any(value[key] != expected_value for key, expected_value in expected.items()):
        raise EvidenceQueryEvaluationError(f"H4 {role} identity changed")
    return value


def _validate_report_dataset_metadata(
    value: object,
    *,
    training_data_dir: str | Path,
    training_bundle: Any,
    expected_manifest_sha256: str,
) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "generator",
        "target_grammar",
        "source_manifest",
        "source_fit",
        "source_calibration",
        "fit",
        "calibration",
        "isolation",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise EvidenceQueryEvaluationError("H4 training dataset record is invalid")
    if (
        value["schema_version"] != DATASET_SCHEMA_VERSION
        or value["generator"] != surface_transfer_data.GENERATOR_VERSION
        or value["target_grammar"] != TARGET_GRAMMAR_VERSION
    ):
        raise EvidenceQueryEvaluationError("H4 dataset protocol identity is invalid")

    root = Path(training_data_dir)
    sources = {
        "source_manifest": (
            "manifest.json",
            {
                "filename": "manifest.json",
                "bytes": len(
                    _read_regular_file(root / "manifest.json", role="H4 manifest")
                ),
                "sha256": expected_manifest_sha256,
            },
        ),
        "source_fit": (
            "fit.jsonl",
            {
                "filename": "fit.jsonl",
                "bytes": len(_read_regular_file(root / "fit.jsonl", role="H4 fit")),
                "sha256": training_bundle.input_sha256["fit"],
                "records": surface_transfer_data.FIT_RECORDS,
                "worlds": surface_transfer_data.FIT_WORLDS,
                "namespace": "train-fit",
                "gradient_bearing": True,
            },
        ),
        "source_calibration": (
            "calibration.jsonl",
            {
                "filename": "calibration.jsonl",
                "bytes": len(
                    _read_regular_file(
                        root / "calibration.jsonl", role="H4 calibration"
                    )
                ),
                "sha256": training_bundle.input_sha256["calibration"],
                "records": surface_transfer_data.CALIBRATION_RECORDS,
                "worlds": surface_transfer_data.CALIBRATION_WORLDS,
                "namespace": "train-calibration",
                "gradient_bearing": False,
            },
        ),
    }
    for name, (_filename, expected) in sources.items():
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
        raise EvidenceQueryEvaluationError("H4 fit partition identity changed")
    if value["calibration"] != expected_calibration:
        raise EvidenceQueryEvaluationError("H4 calibration partition identity changed")
    if value["isolation"] != training_bundle.manifest["isolation"]:
        raise EvidenceQueryEvaluationError("H4 isolation identity changed")
    return value


def _validate_report_source_hashes(report: Mapping[str, Any]) -> None:
    preserved = report["preserved_source_sha256"]
    if not isinstance(preserved, dict) or set(preserved) != set(
        PRESERVED_H3_SOURCE_SHA256
    ):
        raise EvidenceQueryEvaluationError("H4 preserved source hashes are incomplete")
    for name, digest in preserved.items():
        _require_sha256(digest, f"H4 preserved source {name}")
    if preserved != dict(PRESERVED_H3_SOURCE_SHA256):
        raise EvidenceQueryEvaluationError("H4 preserved source pins changed")
    if preserved != _preserved_source_hashes():
        raise EvidenceQueryEvaluationError(
            "H4 preserved source hashes do not match the executable recipe"
        )

    changed = report["changed_source_sha256"]
    expected_changed = _changed_source_hashes()
    if not isinstance(changed, dict) or set(changed) != set(expected_changed):
        raise EvidenceQueryEvaluationError("H4 changed source hashes are incomplete")
    for name, digest in changed.items():
        _require_sha256(digest, f"H4 changed source {name}")
    if changed != expected_changed:
        raise EvidenceQueryEvaluationError(
            "H4 changed source hashes do not match the executable recipe"
        )


def _candidate_checkpoint_record(candidate: object) -> tuple[int, Mapping[str, Any]]:
    if not isinstance(candidate, dict) or set(candidate) != {
        "epoch",
        *_CHECKPOINT_KEYS,
    }:
        raise EvidenceQueryEvaluationError("H4 selected candidate record is invalid")
    epoch = candidate["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise EvidenceQueryEvaluationError("H4 selected candidate epoch is invalid")
    return epoch, _checkpoint_record({key: candidate[key] for key in _CHECKPOINT_KEYS})


def load_candidate_from_training_report(
    report_path: str | Path,
    *,
    expected_report_sha256: str,
    expected_manifest_sha256: str,
    training_data_dir: str | Path,
    training_bundle: Any,
) -> EvidenceQueryH4Candidate:
    """Authenticate one fixed-seed H4 report and its selected checkpoint."""

    path = Path(report_path)
    snapshot = _read_verified_file(
        path, expected_report_sha256, role="H4 training report"
    )
    report = h3_evaluation._parse_json(snapshot, role="H4 training report")
    if not isinstance(report, dict) or set(report) != _REPORT_REQUIRED_KEYS:
        raise EvidenceQueryEvaluationError("H4 training report schema is invalid")
    if (
        report["schema_version"] != H4_TRAINING_REPORT_SCHEMA_VERSION
        or report["recipe"] != H4_TRAINING_RECIPE_VERSION
        or report["status"] != "complete"
    ):
        raise EvidenceQueryEvaluationError("H4 training report is not complete v1")
    seed = report["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in TRAINING_SEEDS
    ):
        raise EvidenceQueryEvaluationError("H4 training seed is not frozen")
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
        raise EvidenceQueryEvaluationError("H4 training model identity is invalid")
    expected_manifest = _require_sha256(
        expected_manifest_sha256, "expected H4 training manifest"
    )
    if report["dataset_manifest_sha256"] != expected_manifest:
        raise EvidenceQueryEvaluationError("H4 training used another dataset manifest")
    _validate_report_dataset_metadata(
        report["dataset"],
        training_data_dir=training_data_dir,
        training_bundle=training_bundle,
        expected_manifest_sha256=expected_manifest,
    )
    if report["dev_used_for_selection"] is not False:
        raise EvidenceQueryEvaluationError("development influenced H4 selection")
    if report["fresh_v1_accessed"] is not False:
        raise EvidenceQueryEvaluationError("H4 training accessed sealed confirmation")
    if report["selection_note"] != _TRAINING_SELECTION_NOTE:
        raise EvidenceQueryEvaluationError("H4 training selection boundary changed")
    h3_evaluation._validate_hyperparameters(report["hyperparameters"])
    h3_evaluation._validate_runtime(report["runtime"], device=report["device"])
    _validate_report_source_hashes(report)

    epochs = report["epochs"]
    if not isinstance(epochs, list) or len(epochs) != 3:
        raise EvidenceQueryEvaluationError("H4 report must contain three epochs")
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
            raise EvidenceQueryEvaluationError("H4 training epochs are not ordered")
        for metric in ("train_loss", "state_loss", "pointer_loss", "seconds"):
            value = row[metric]
            if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                raise EvidenceQueryEvaluationError(f"H4 epoch {metric} is invalid")
        _checkpoint_record(row["checkpoint"])
        macro, overall, _threshold = _epoch_calibration(row)
        ranking_rows.append((macro, overall, -expected_epoch, row))

    selected_row = max(ranking_rows, key=lambda item: item[:3])[3]
    selected_epoch = int(selected_row["epoch"])
    candidate_epoch, checkpoint = _candidate_checkpoint_record(report["candidate"])
    if candidate_epoch != selected_epoch or checkpoint != selected_row["checkpoint"]:
        raise EvidenceQueryEvaluationError(
            "H4 candidate was not selected by uncalibrated calibration ranking"
        )
    calibration = report["calibration"]
    if not isinstance(calibration, Mapping) or set(calibration) != {
        "selected_epoch",
        *_CALIBRATION_KEYS,
    }:
        raise EvidenceQueryEvaluationError("H4 selected calibration schema is invalid")
    if calibration["selected_epoch"] != selected_epoch:
        raise EvidenceQueryEvaluationError("H4 selected calibration epoch disagrees")
    if {key: calibration[key] for key in _CALIBRATION_KEYS} != selected_row[
        "calibration"
    ]:
        raise EvidenceQueryEvaluationError("H4 selected calibration record disagrees")
    macro, overall, threshold = _epoch_calibration(selected_row)

    checkpoint_snapshot = _read_verified_file(
        path.parent / checkpoint["filename"],
        checkpoint["sha256"],
        role=f"H4 checkpoint seed {seed}",
    )
    if len(checkpoint_snapshot) != checkpoint["bytes"]:
        raise EvidenceQueryEvaluationError("H4 checkpoint byte count changed")
    return EvidenceQueryH4Candidate(
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
) -> tuple[EvidenceQueryH4Candidate, tuple[EvidenceQueryH4Candidate, ...]]:
    """Select H4 solely from authenticated training-calibration evidence."""

    if len(training_reports) != len(TRAINING_SEEDS):
        raise EvidenceQueryEvaluationError(
            "exactly two fixed-seed H4 training reports are required"
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
        raise EvidenceQueryEvaluationError("H4 reports do not cover both fixed seeds")
    if len({candidate.report_sha256 for candidate in candidates}) != 2:
        raise EvidenceQueryEvaluationError("H4 report digests must be unique")
    if len({candidate.sha256 for candidate in candidates}) != 2:
        raise EvidenceQueryEvaluationError("H4 checkpoint digests must be unique")
    ordered = tuple(sorted(candidates, key=lambda item: item.seed))
    return max(ordered, key=lambda item: item.ranking_key), ordered


def _canonical_evaluation_runtime(
    primary: EvidenceQueryH4Candidate,
    candidates: Sequence[EvidenceQueryH4Candidate],
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
    }


def _evaluation_source_hashes() -> dict[str, str]:
    return _source_hashes(_evaluation_source_paths(), role="H4 evaluation")


def _require_training_data_unchanged(
    training_data_dir: str | Path,
    input_sha256: Mapping[str, str],
) -> None:
    root = Path(training_data_dir)
    filenames = {
        "manifest": "manifest.json",
        "fit": "fit.jsonl",
        "calibration": "calibration.jsonl",
    }
    observed = {
        name: _sha256(
            _read_regular_file(root / filename, role=f"H4 training {filename}")
        )
        for name, filename in filenames.items()
    }
    if observed != dict(input_sha256):
        raise EvidenceQueryEvaluationError("H4 training data changed during evaluation")


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
    """Select H4 without development, then stop at the frozen quality gate."""

    output = Path(output_path)
    if output.exists():
        raise EvidenceQueryEvaluationError("H4 evaluation output already exists")
    if not output.parent.is_dir():
        raise EvidenceQueryEvaluationError("H4 evaluation output parent must exist")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise EvidenceQueryEvaluationError("batch_size must be an integer")
    expected_training_manifest = _require_sha256(
        training_manifest_sha256, "H4 training manifest"
    )
    expected_development_manifest = _require_sha256(
        development_manifest_sha256, "H2 development manifest"
    )
    resolved_device = _resolve_device(device)
    source_hashes = _evaluation_source_hashes()

    # Only H4 training inputs are opened here.  The known-development loader
    # remains unreachable until both fixed seeds and the primary are frozen.
    try:
        training_bundle = load_h4_training_bundle(Path(training_data_dir))
    except Exception as exc:
        raise EvidenceQueryEvaluationError(
            "H4 training inputs could not be verified"
        ) from exc
    if not hmac.compare_digest(
        training_bundle.manifest_sha256, expected_training_manifest
    ):
        raise EvidenceQueryEvaluationError("H4 training manifest SHA-256 mismatch")
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
            tokenizer,
            development_bundle.examples,
            expected_split="dev",
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
        model,
        inputs,
        device=resolved_device,
        batch_size=batch_size,
    )
    del model
    if tuple(inference.example_ids) != tuple(item.example_id for item in inputs):
        raise EvidenceQueryEvaluationError("H4 inference changed development order")

    uncalibrated_raw = raw_pointer_diagnostics(
        development_bundle.examples, cases, inference.predictions
    )
    uncalibrated_gate = _quality_gates(
        uncalibrated_raw["acceptance"]["metrics"],
        require_zero_wrong_presented=False,
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
        raise EvidenceQueryEvaluationError("H4 evaluation source changed during run")

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
        "schema_version": H4_DEVELOPMENT_EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "training_data": {
            "schema_version": surface_transfer_data.MANIFEST_SCHEMA_VERSION,
            "generator": surface_transfer_data.GENERATOR_VERSION,
            "recipe": H4_TRAINING_RECIPE_VERSION,
            "manifest_sha256": training_bundle.manifest_sha256,
            "fit_sha256": training_bundle.input_sha256["fit"],
            "calibration_sha256": training_bundle.input_sha256["calibration"],
            "fit_records": len(training_bundle.fit),
            "calibration_records": len(training_bundle.calibration),
            "development_records_used": 0,
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
            "intervention": "training_data_only",
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
            "threshold_applied": calibrated_gate is not None,
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
            "sealed_confirmation_assessed": False,
            "next_step": (
                "measure matched latency before sealed confirmation"
                if quality_passed
                else "retain H4 as a negative result; stop before latency and sealed confirmation"
            ),
        },
        "selection_boundary": (
            "Both fixed-seed H4 reports, checkpoints, training data, sources, "
            "runtime, primary epoch, seed, and threshold were authenticated and "
            "frozen before the separate H2 known-development manifest was opened. "
            "Uncalibrated admission failure stops before threshold and verifier; "
            "quality failure stops before latency and sealed confirmation."
        ),
    }
    _write_json_no_clobber(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate Nano's frozen H4 data-only candidate"
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
        required=True,
        metavar=("REPORT", "SHA256"),
        help="repeat exactly twice, once for each digest-pinned H4 seed",
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
    "H4_DEVELOPMENT_EVALUATION_SCHEMA_VERSION",
    "EvidenceQueryCandidate",
    "EvidenceQueryEvaluationError",
    "EvidenceQueryH4Candidate",
    "authenticate_and_select_primary",
    "evaluate_development",
    "load_candidate_from_training_report",
]
