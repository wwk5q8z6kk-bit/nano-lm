"""Frozen data-only H4 trainer for Nano's evidence-query scribe.

H4 changes the fit and training-only calibration surfaces, and nothing else.
The H3 architecture, tokenizer, prompt, objective, optimizer, schedule, exposure,
seeds, calibration rule, and epoch-selection rule remain frozen.  This module
never imports or opens a development, benchmark, or sealed-confirmation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tokenizers
import torch

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.contract import FIELD_ORDER, FieldName, FieldState
from nano_ai.training import surface_transfer_data
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
    STATE_POINTER_COUNTS,
    PointerSupervision,
    encode_pointer_partition,
    load_pointer_tokenizer,
)
from nano_ai.training.pointer_model import NANO_TRUNK_PARAMETER_COUNT
from nano_ai.training.state_span_data import (
    DATASET_SCHEMA_VERSION,
    STATE_VARIANTS,
    TARGET_GRAMMAR_VERSION,
    VARIANTS,
    StateSpanExample,
    canonical_json_bytes,
)
from nano_ai.training.train_evidence_query import (
    _accumulate_loss,
    _calibrate_model,
    _finish_loss,
    _loss_aggregates,
    _optimizer,
    _resolve_evidence_query_device,
    _save_checkpoint,
    _write_json_no_clobber,
    select_epoch_report,
)
from nano_ai.training.train_pointer import (
    POINTER_LOSS_WEIGHT,
    STATE_CLASS_COUNTS,
    STATE_CLASS_WEIGHTS,
    STATE_LOSS_DEFINITION,
    STATE_LOSS_WEIGHT,
    collate_pointer_batch,
    pointer_objective,
)
from nano_ai.training.train_state_span import (
    ADAM_BETAS,
    ADAM_EPSILON,
    COSINE_FLOOR,
    GRADIENT_CLIP,
    LOG_EVERY_STEPS,
    PEAK_LEARNING_RATE,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    TrainingInputError,
    _seed_training,
    grouped_batch_indices,
    learning_rate_at,
)

H4_TRAINING_REPORT_SCHEMA_VERSION = "nano.evidence-query-h4-training-report.v1"
H4_TRAINING_RECIPE_VERSION = "nano-evidence-query-data-only-v1"

FIT_PARTITION = surface_transfer_data.FIT_PARTITION
CALIBRATION_PARTITION = surface_transfer_data.CALIBRATION_PARTITION
FIT_RECORD_COUNT = surface_transfer_data.FIT_RECORDS
CALIBRATION_RECORD_COUNT = surface_transfer_data.CALIBRATION_RECORDS
FIT_WORLD_COUNT = surface_transfer_data.FIT_WORLDS
CALIBRATION_WORLD_COUNT = surface_transfer_data.CALIBRATION_WORLDS
VARIANTS_PER_WORLD = surface_transfer_data.VARIANTS_PER_WORLD
TRAINING_SEEDS = surface_transfer_data.TRAINING_SEEDS
BATCH_SIZE = surface_transfer_data.BATCH_SIZE
EPOCHS = surface_transfer_data.EPOCHS
STEPS_PER_EPOCH = surface_transfer_data.STEPS_PER_EPOCH
STEPS_PER_SEED = surface_transfer_data.STEPS_PER_SEED

FIT_STATE_CLASS_COUNTS: tuple[int, ...] = (42_980, 4_620, 2_800, 2_800, 2_800)
CALIBRATION_STATE_CLASS_COUNTS: tuple[int, ...] = (3_070, 330, 200, 200, 200)

FIT_STATE_FIELD_QUOTA: Mapping[str, int] = {
    **{
        f"{field.value}:{FieldState.SUPPORTED.value}": 9_520
        for field in (
            FieldName.CHIEF_COMPLAINT,
            FieldName.DURATION,
            FieldName.SEVERITY,
        )
    },
    f"{FieldName.MEDICATION.value}:{FieldState.SUPPORTED.value}": 7_140,
    f"{FieldName.MEDICATION.value}:{FieldState.ABSENT.value}": 2_380,
    f"{FieldName.ALLERGY.value}:{FieldState.SUPPORTED.value}": 7_280,
    f"{FieldName.ALLERGY.value}:{FieldState.ABSENT.value}": 2_240,
    **{
        f"{field.value}:{state.value}": 560
        for field in FIELD_ORDER
        for state in (
            FieldState.MISSING,
            FieldState.UNCERTAIN,
            FieldState.CONFLICTING,
        )
    },
}
CALIBRATION_STATE_FIELD_QUOTA: Mapping[str, int] = {
    **{
        f"{field.value}:{FieldState.SUPPORTED.value}": 680
        for field in (
            FieldName.CHIEF_COMPLAINT,
            FieldName.DURATION,
            FieldName.SEVERITY,
        )
    },
    f"{FieldName.MEDICATION.value}:{FieldState.SUPPORTED.value}": 510,
    f"{FieldName.MEDICATION.value}:{FieldState.ABSENT.value}": 170,
    f"{FieldName.ALLERGY.value}:{FieldState.SUPPORTED.value}": 520,
    f"{FieldName.ALLERGY.value}:{FieldState.ABSENT.value}": 160,
    **{
        f"{field.value}:{state.value}": 40
        for field in FIELD_ORDER
        for state in (
            FieldState.MISSING,
            FieldState.UNCERTAIN,
            FieldState.CONFLICTING,
        )
    },
}

# These are literal pins to the accepted H3 implementation.  A data-only H4
# run must fail closed if any architecture, objective, tokenizer-facing,
# inference, or optimization authority changes.
PRESERVED_H3_SOURCE_SHA256: Mapping[str, str] = {
    "base_model": "3089b7e0e7b527ef08d7251d0b84cd064f0c6f2d6330050936e8d55a55e5702c",
    "data_generator": "2d3fd33d694893aea5aa80b514e34256e9f6046f71b13f6410dbabec1278d707",
    "evidence_query_inference": "58a141a5f091f5b4733d082b9d3d65002afbb4ec6b1d8942f34c685e4544875d",
    "evidence_query_model": "6e7e666e48b06f6ed53ead4996982ac052bf0dd446cc6945d2eb506c52aaefbc",
    "h1_training_loader": "11056fe0773e5a64fa4dc143a01725d3d472b97c0945a3419adf45e153964f57",
    "h2_objective": "6e12d8b5f9650c6e4b8af4fa93303eae111adf8fc94d5c756c392bb51e6b6d72",
    "pointer_data": "9719b435b8b58b7e516539153fb75eac0c172e293af652b3fcccd06849579b1e",
    "pointer_decoder": "e226bab817c93ca54906deb825b1a6ee8fb9ae2e95f37044f893de48783a6a24",
    "state_span_adapter": "b0ffd2a6cf9909e919a12311077d47b8426567b239e8dde52794dc5aef0a86f5",
    "training": "2674d52f51dac5b9c8ef09e4bf2be6bbdd497418b3cfdf479886c1e0e85c0cfb",
}

_WORLD_RE = re.compile(r"train-world-(fit|calibration)-(\d{4})\Z")


@dataclass(frozen=True, slots=True)
class H4TrainingBundle:
    manifest: Mapping[str, Any]
    manifest_sha256: str
    fit: tuple[StateSpanExample, ...]
    calibration: tuple[StateSpanExample, ...]
    input_sha256: Mapping[str, str]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def preserved_source_paths() -> Mapping[str, Path]:
    """Return H3 files that H4 is forbidden to change."""

    from nano_ai.training.train_evidence_query import training_source_paths

    return training_source_paths()


def changed_source_paths() -> Mapping[str, Path]:
    """Return the two H4-specific implementation files."""

    return {
        "data_generator": Path(surface_transfer_data.__file__).resolve(),
        "training": Path(__file__).resolve(),
    }


def _hash_paths(paths: Mapping[str, Path], *, label: str) -> dict[str, str]:
    missing = sorted(name for name, path in paths.items() if not path.is_file())
    if missing:
        raise TrainingInputError(f"{label} source is incomplete: " + ", ".join(missing))
    return {
        name: _sha256(path.read_bytes())
        for name, path in sorted(paths.items())
    }


def _require_preserved_h3_sources() -> dict[str, str]:
    observed = _hash_paths(preserved_source_paths(), label="preserved H3")
    if observed != dict(PRESERVED_H3_SOURCE_SHA256):
        changed = sorted(
            name
            for name in set(observed) | set(PRESERVED_H3_SOURCE_SHA256)
            if observed.get(name) != PRESERVED_H3_SOURCE_SHA256.get(name)
        )
        raise TrainingInputError(
            "H4 preserved H3 source hash mismatch: " + ", ".join(changed)
        )
    return observed


def _changed_source_hashes() -> dict[str, str]:
    return _hash_paths(changed_source_paths(), label="changed H4")


def _require_unchanged_sources(changed_expected: Mapping[str, str]) -> None:
    _require_preserved_h3_sources()
    if _changed_source_hashes() != dict(changed_expected):
        raise TrainingInputError("H4 changed source changed during execution")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _parse_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingInputError(f"invalid H4 JSON in {label}") from exc


def _read_file(path: Path, *, label: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise TrainingInputError(f"H4 {label} is unavailable") from exc


def _load_partition(payload: bytes, *, filename: str) -> tuple[StateSpanExample, ...]:
    records: list[StateSpanExample] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line:
            raise TrainingInputError(f"blank H4 record at {filename}:{line_number}")
        value = _parse_json(line, label=f"{filename}:{line_number}")
        try:
            records.append(StateSpanExample.from_dict(value))
        except (TypeError, ValueError) as exc:
            raise TrainingInputError(
                f"invalid H4 record at {filename}:{line_number}"
            ) from exc
    if not records or surface_transfer_data.records_bytes(records) != payload:
        raise TrainingInputError(f"H4 {filename} is not canonical JSONL")
    return tuple(records)


def _file_sha256(path: Path, *, label: str) -> str:
    return _sha256(_read_file(path, label=label))


def _require_digest(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TrainingInputError(f"H4 {label} digest is invalid")
    return value


def _state_field_quota(examples: Sequence[StateSpanExample]) -> dict[str, int]:
    from nano_ai.adapters.state_span import parse_state_span_summary

    counts: Counter[str] = Counter()
    for example in examples:
        for proposal in parse_state_span_summary(example.target, example.transcript):
            counts[f"{proposal.field.value}:{proposal.state.value}"] += 1
    return dict(sorted(counts.items()))


def _string_multiset_sha256(values: Sequence[str]) -> str:
    return _sha256(canonical_json_bytes(sorted(values)))


def _validate_world_families(
    records: Sequence[StateSpanExample],
    *,
    partition: str,
    expected_worlds: int,
    expected_records: int,
) -> None:
    if len(records) != expected_records:
        raise TrainingInputError(f"H4 {partition} record count drifted")
    grouped: dict[str, list[StateSpanExample]] = {}
    expected_indices = set(range(expected_worlds))
    observed_indices: set[int] = set()
    record_ids: set[str] = set()
    for record in records:
        if record.split != "train":
            raise TrainingInputError(f"H4 {partition} contains a non-training record")
        match = _WORLD_RE.fullmatch(record.world_id)
        if match is None or match.group(1) != partition:
            raise TrainingInputError(f"H4 {partition} world namespace drifted")
        index = int(match.group(2))
        observed_indices.add(index)
        expected_id = f"train-{partition}-{index:04d}-{record.variant}"
        if record.example_id != expected_id or record.example_id in record_ids:
            raise TrainingInputError(f"H4 {partition} record namespace drifted")
        record_ids.add(record.example_id)
        grouped.setdefault(record.world_id, []).append(record)
    if observed_indices != expected_indices or len(grouped) != expected_worlds:
        raise TrainingInputError(f"H4 {partition} world identities are incomplete")
    expected_variants = set(VARIANTS)
    for world_id, family in grouped.items():
        if (
            len(family) != VARIANTS_PER_WORLD
            or {record.variant for record in family} != expected_variants
            or len({record.target_field for record in family}) != 1
        ):
            raise TrainingInputError(f"H4 world {world_id} is not one paired family")
        for record in family:
            expected_state = (
                None if record.variant == "normal" else STATE_VARIANTS[record.variant]
            )
            if record.target_state != expected_state:
                raise TrainingInputError(f"H4 world {world_id} mutation state drifted")


def _validate_manifest_partition(
    identity: object,
    records: Sequence[StateSpanExample],
    *,
    expected_identity: Mapping[str, Any],
    partition: str,
    seed: int,
    expected_worlds: int,
    expected_quota: Mapping[str, int],
    file_sha256: str,
) -> None:
    required = {
        "namespace",
        "seed",
        "records",
        "worlds",
        "ordered_records_sha256",
        "transcript_multiset_sha256",
        "world_id_multiset_sha256",
        "record_id_multiset_sha256",
        "open_value_set_sha256",
        "open_value_set_by_field",
        "component_line_skeleton_multiset_sha256",
        "component_line_skeleton_unique_count",
        "state_field_quota",
        "coverage",
        "quality",
        "templates",
    }
    if not isinstance(identity, dict) or set(identity) != required:
        raise TrainingInputError(f"H4 {partition} manifest identity is malformed")
    if identity != dict(expected_identity):
        raise TrainingInputError(
            f"H4 {partition} manifest does not reproduce from source"
        )
    if (
        identity["namespace"] != f"train-{partition}"
        or identity["seed"] != seed
        or identity["records"] != len(records)
        or identity["worlds"] != expected_worlds
        or identity["ordered_records_sha256"] != file_sha256
        or identity["transcript_multiset_sha256"]
        != surface_transfer_data.transcript_multiset_sha256(records)
        or identity["world_id_multiset_sha256"]
        != _string_multiset_sha256([record.world_id for record in records])
        or identity["record_id_multiset_sha256"]
        != _string_multiset_sha256([record.example_id for record in records])
        or identity["state_field_quota"] != dict(expected_quota)
        or _state_field_quota(records) != dict(expected_quota)
    ):
        raise TrainingInputError(f"H4 {partition} manifest identity drifted")
    for key in (
        "open_value_set_sha256",
        "component_line_skeleton_multiset_sha256",
    ):
        _require_digest(identity[key], label=f"{partition}.{key}")
    open_fields = {field.value for field in FIELD_ORDER if field is not FieldName.SEVERITY}
    values = identity["open_value_set_by_field"]
    if not isinstance(values, dict) or set(values) != open_fields:
        raise TrainingInputError(f"H4 {partition} open-value identity is malformed")
    for field, field_identity in values.items():
        if (
            not isinstance(field_identity, dict)
            or set(field_identity) != {"count", "sha256"}
            or isinstance(field_identity["count"], bool)
            or not isinstance(field_identity["count"], int)
            or field_identity["count"] <= 0
        ):
            raise TrainingInputError(
                f"H4 {partition} open-value identity drifted for {field}"
            )
        _require_digest(field_identity["sha256"], label=f"{partition}.{field}")
    if (
        not isinstance(identity["coverage"], dict)
        or not isinstance(identity["quality"], dict)
        or not isinstance(identity["templates"], dict)
    ):
        raise TrainingInputError(f"H4 {partition} coverage identity is malformed")


def _validate_manifest(
    manifest: object,
    *,
    fit: Sequence[StateSpanExample],
    calibration: Sequence[StateSpanExample],
    fit_sha256: str,
    calibration_sha256: str,
    generator_sha256: str,
    expected_fit_identity: Mapping[str, Any],
    expected_calibration_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "generator",
        "recipe",
        "target_grammar",
        "generator_sha256",
        "tokenizer_sha256",
        "base_checkpoint_sha256",
        "training_identity",
        "partitions",
        "isolation",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise TrainingInputError("H4 manifest has unexpected keys")
    if (
        manifest["schema_version"] != surface_transfer_data.MANIFEST_SCHEMA_VERSION
        or manifest["generator"] != surface_transfer_data.GENERATOR_VERSION
        or manifest["recipe"] != H4_TRAINING_RECIPE_VERSION
        or manifest["target_grammar"] != TARGET_GRAMMAR_VERSION
        or manifest["generator_sha256"] != generator_sha256
        or manifest["tokenizer_sha256"] != FROZEN_NANO_V01.tokenizer_sha256
        or manifest["base_checkpoint_sha256"] != FROZEN_NANO_V01.checkpoint_sha256
    ):
        raise TrainingInputError("H4 manifest frozen identity drifted")
    expected_training_identity = {
        "training_seeds": list(TRAINING_SEEDS),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "steps_per_seed": STEPS_PER_SEED,
        "gradient_records": FIT_RECORD_COUNT,
        "calibration_records": CALIBRATION_RECORD_COUNT,
        "calibration_gradient_bearing": False,
    }
    if manifest["training_identity"] != expected_training_identity:
        raise TrainingInputError("H4 manifest training identity drifted")
    partitions = manifest["partitions"]
    if not isinstance(partitions, dict) or set(partitions) != {
        FIT_PARTITION,
        CALIBRATION_PARTITION,
    }:
        raise TrainingInputError("H4 manifest partitions are malformed")
    _validate_manifest_partition(
        partitions[FIT_PARTITION],
        fit,
        expected_identity=expected_fit_identity,
        partition=FIT_PARTITION,
        seed=surface_transfer_data.FIT_GENERATOR_SEED,
        expected_worlds=FIT_WORLD_COUNT,
        expected_quota=FIT_STATE_FIELD_QUOTA,
        file_sha256=fit_sha256,
    )
    _validate_manifest_partition(
        partitions[CALIBRATION_PARTITION],
        calibration,
        expected_identity=expected_calibration_identity,
        partition=CALIBRATION_PARTITION,
        seed=surface_transfer_data.CALIBRATION_GENERATOR_SEED,
        expected_worlds=CALIBRATION_WORLD_COUNT,
        expected_quota=CALIBRATION_STATE_FIELD_QUOTA,
        file_sha256=calibration_sha256,
    )
    isolation = manifest["isolation"]
    required_true = {
        "world_ids_disjoint",
        "record_ids_disjoint",
        "exact_transcripts_disjoint",
        "open_values_disjoint_except_severity",
        "exact_templates_disjoint",
        "normalized_component_skeletons_disjoint",
        "open_values_disjoint",
        "fit_component_skeletons_disjoint",
        "calibration_component_skeletons_disjoint",
        "gold_built_from_annotated_turn_offsets",
    }
    required_false = {
        "development_used_for_template_or_value_selection",
        "deterministic_baseline_used_for_labels",
        "historical_sentinels_present",
        "historical_fresh_v0_read",
        "sealed_confirmation_read",
    }
    if (
        not isinstance(isolation, dict)
        or any(isolation.get(key) is not True for key in required_true)
        or any(isolation.get(key) is not False for key in required_false)
        or isolation.get("development_records_in_training") != 0
        or isolation.get("known_development_role")
        != "leakage_rejection_only_after_design_freeze"
    ):
        raise TrainingInputError("H4 manifest isolation evidence drifted")
    return manifest


def load_h4_training_bundle(data_dir: Path) -> H4TrainingBundle:
    """Load and fully authenticate only H4 manifest, fit, and calibration files."""

    root = Path(data_dir)
    paths = {
        "manifest": root / "manifest.json",
        FIT_PARTITION: root / "fit.jsonl",
        CALIBRATION_PARTITION: root / "calibration.jsonl",
    }
    snapshots = {
        name: _read_file(path, label=name) for name, path in paths.items()
    }
    digests = {name: _sha256(payload) for name, payload in snapshots.items()}
    fit = _load_partition(snapshots[FIT_PARTITION], filename="fit.jsonl")
    calibration = _load_partition(
        snapshots[CALIBRATION_PARTITION], filename="calibration.jsonl"
    )
    _validate_world_families(
        fit,
        partition=FIT_PARTITION,
        expected_worlds=FIT_WORLD_COUNT,
        expected_records=FIT_RECORD_COUNT,
    )
    _validate_world_families(
        calibration,
        partition=CALIBRATION_PARTITION,
        expected_worlds=CALIBRATION_WORLD_COUNT,
        expected_records=CALIBRATION_RECORD_COUNT,
    )
    if {record.world_id for record in fit} & {
        record.world_id for record in calibration
    } or {record.example_id for record in fit} & {
        record.example_id for record in calibration
    }:
        raise TrainingInputError("H4 fit and calibration namespaces overlap")
    if {record.transcript for record in fit} & {
        record.transcript for record in calibration
    }:
        raise TrainingInputError("H4 fit and calibration transcripts overlap")

    # Reproduction from the pinned generator proves every record and manifest
    # partition identity (including values, surfaces, coverage, and quotas)
    # without consulting any development partition.
    expected_fit, fit_traces = surface_transfer_data._generate_with_trace(
        FIT_PARTITION
    )
    expected_calibration, calibration_traces = (
        surface_transfer_data._generate_with_trace(CALIBRATION_PARTITION)
    )
    if fit != expected_fit:
        raise TrainingInputError("H4 fit records do not reproduce from source")
    if calibration != expected_calibration:
        raise TrainingInputError("H4 calibration records do not reproduce from source")
    expected_fit_identity = surface_transfer_data._partition_identity(
        FIT_PARTITION, expected_fit, fit_traces
    )
    expected_calibration_identity = surface_transfer_data._partition_identity(
        CALIBRATION_PARTITION, expected_calibration, calibration_traces
    )

    generator_sha256 = _file_sha256(
        Path(surface_transfer_data.__file__), label="generator source"
    )
    manifest = _validate_manifest(
        _parse_json(snapshots["manifest"], label="manifest.json"),
        fit=fit,
        calibration=calibration,
        fit_sha256=digests[FIT_PARTITION],
        calibration_sha256=digests[CALIBRATION_PARTITION],
        generator_sha256=generator_sha256,
        expected_fit_identity=expected_fit_identity,
        expected_calibration_identity=expected_calibration_identity,
    )
    return H4TrainingBundle(
        manifest=manifest,
        manifest_sha256=digests["manifest"],
        fit=fit,
        calibration=calibration,
        input_sha256=digests,
    )


def _state_class_counts(records: Sequence[PointerSupervision]) -> tuple[int, ...]:
    counts = [0] * len(STATE_ORDER)
    for record in records:
        for label in record.state_labels:
            counts[label] += 1
    return tuple(counts)


def _require_partition_state_distributions(
    fit_records: Sequence[PointerSupervision],
    calibration_records: Sequence[PointerSupervision],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    fit_counts = _state_class_counts(fit_records)
    calibration_counts = _state_class_counts(calibration_records)
    if fit_counts != FIT_STATE_CLASS_COUNTS:
        raise TrainingInputError("H4 fit state-class distribution drifted")
    if calibration_counts != CALIBRATION_STATE_CLASS_COUNTS:
        raise TrainingInputError("H4 calibration state-class distribution drifted")
    if tuple(
        fit + calibration
        for fit, calibration in zip(fit_counts, calibration_counts, strict=True)
    ) != STATE_CLASS_COUNTS:
        raise RuntimeError("H4 partitions do not reconstruct frozen class counts")
    return fit_counts, calibration_counts


def _require_exact_training_steps(
    examples: Sequence[StateSpanExample], *, seed: int
) -> Mapping[int, tuple[tuple[int, ...], ...]]:
    batches_by_epoch: dict[int, tuple[tuple[int, ...], ...]] = {}
    for epoch in range(1, EPOCHS + 1):
        batches = grouped_batch_indices(
            examples,
            batch_size=BATCH_SIZE,
            seed=seed,
            epoch=epoch,
        )
        if len(batches) != STEPS_PER_EPOCH or any(
            len(batch) != BATCH_SIZE for batch in batches
        ):
            raise TrainingInputError("H4 exposure is not exactly 350 full batches")
        batches_by_epoch[epoch] = batches
    if sum(len(value) for value in batches_by_epoch.values()) != STEPS_PER_SEED:
        raise TrainingInputError("H4 exposure is not exactly 1,050 steps per seed")
    return batches_by_epoch


def _require_input_hashes(
    *,
    data_dir: Path,
    base_checkpoint: Path,
    tokenizer_path: Path,
    expected: Mapping[str, str],
) -> None:
    paths = {
        "manifest": Path(data_dir) / "manifest.json",
        FIT_PARTITION: Path(data_dir) / "fit.jsonl",
        CALIBRATION_PARTITION: Path(data_dir) / "calibration.jsonl",
        "base_checkpoint": Path(base_checkpoint),
        "tokenizer": Path(tokenizer_path),
    }
    observed = {
        name: _file_sha256(path, label=name) for name, path in paths.items()
    }
    if observed != dict(expected):
        raise TrainingInputError("H4 authenticated input changed during execution")


def _dataset_file_identity(path: Path) -> dict[str, Any]:
    snapshot = _read_file(path, label=path.name)
    return {
        "filename": path.name,
        "bytes": len(snapshot),
        "sha256": _sha256(snapshot),
    }


def train_evidence_query_h4_candidate(
    *,
    data_dir: Path,
    base_checkpoint: Path,
    tokenizer_path: Path,
    output_dir: Path,
    seed: int,
    device: str,
) -> Mapping[str, Any]:
    """Train one H4 seed after complete fail-closed, no-output preflight."""

    if seed not in TRAINING_SEEDS:
        raise TrainingInputError(f"seed must be one of the frozen seeds {TRAINING_SEEDS}")
    output = Path(output_dir)
    if output.exists():
        raise TrainingInputError("candidate output directory must not exist")

    started = time.monotonic()
    resolved_device = _resolve_evidence_query_device(device)
    preserved_hashes = _require_preserved_h3_sources()
    changed_hashes = _changed_source_hashes()

    base_sha256 = _file_sha256(Path(base_checkpoint), label="base checkpoint")
    tokenizer_sha256 = _file_sha256(Path(tokenizer_path), label="tokenizer")
    if base_sha256 != FROZEN_NANO_V01.checkpoint_sha256:
        raise TrainingInputError("H4 base checkpoint identity drifted")
    if tokenizer_sha256 != FROZEN_NANO_V01.tokenizer_sha256:
        raise TrainingInputError("H4 tokenizer identity drifted")

    bundle = load_h4_training_bundle(Path(data_dir))
    tokenizer = load_pointer_tokenizer(Path(tokenizer_path))
    fit_records = encode_pointer_partition(
        tokenizer,
        bundle.fit,
        expected_split="train",
    )
    calibration_records = encode_pointer_partition(
        tokenizer,
        bundle.calibration,
        expected_split="train",
    )
    fit_counts, calibration_counts = _require_partition_state_distributions(
        fit_records, calibration_records
    )
    batches_by_epoch = _require_exact_training_steps(bundle.fit, seed=seed)

    authenticated_inputs = {
        **bundle.input_sha256,
        "base_checkpoint": base_sha256,
        "tokenizer": tokenizer_sha256,
    }
    _require_unchanged_sources(changed_hashes)
    _require_input_hashes(
        data_dir=Path(data_dir),
        base_checkpoint=Path(base_checkpoint),
        tokenizer_path=Path(tokenizer_path),
        expected=authenticated_inputs,
    )

    _seed_training(seed)
    model = NanoEvidenceQueryPointerModel.from_frozen_base(Path(base_checkpoint))
    if (
        model.architecture_version != ARCHITECTURE_VERSION
        or model.parameter_count != NANO_EVIDENCE_QUERY_PARAMETER_COUNT
        or model.head_parameter_count != EVIDENCE_QUERY_HEAD_PARAMETER_COUNT
        or model.trunk.config.parameter_count != NANO_TRUNK_PARAMETER_COUNT
    ):
        raise RuntimeError("H4 model identity drifted before training")
    model.to(resolved_device).train()
    optimizer = _optimizer(model)
    _require_input_hashes(
        data_dir=Path(data_dir),
        base_checkpoint=Path(base_checkpoint),
        tokenizer_path=Path(tokenizer_path),
        expected=authenticated_inputs,
    )
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise TrainingInputError("candidate output directory must not exist") from exc

    global_step = 0
    epoch_reports: list[dict[str, Any]] = []
    print(
        json.dumps(
            {
                "event": "training_start",
                "recipe": H4_TRAINING_RECIPE_VERSION,
                "seed": seed,
                "device": resolved_device,
                "fit_records": len(fit_records),
                "calibration_records": len(calibration_records),
                "development_records_used": 0,
                "steps": STEPS_PER_SEED,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for epoch in range(1, EPOCHS + 1):
        totals = _loss_aggregates()
        epoch_started = time.monotonic()
        for indices in batches_by_epoch[epoch]:
            global_step += 1
            learning_rate = learning_rate_at(
                global_step, total_steps=STEPS_PER_SEED
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch = collate_pointer_batch(
                [fit_records[index] for index in indices],
                device=resolved_device,
            )
            loss = pointer_objective(
                model(batch.token_ids, attention_mask=batch.attention_mask), batch
            )
            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP
            )
            optimizer.step()
            _accumulate_loss(totals, loss)
            if global_step == 1 or global_step % LOG_EVERY_STEPS == 0:
                print(
                    json.dumps(
                        {
                            "event": "training_step",
                            "epoch": epoch,
                            "step": global_step,
                            "steps": STEPS_PER_SEED,
                            "loss": round(float(loss.total.item()), 6),
                            "state_loss": round(float(loss.state.item()), 6),
                            "pointer_loss": round(float(loss.pointer.item()), 6),
                            "gradient_norm": round(float(gradient_norm.item()), 6),
                            "learning_rate": learning_rate,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        train_loss = _finish_loss(totals)
        _require_unchanged_sources(changed_hashes)
        _require_input_hashes(
            data_dir=Path(data_dir),
            base_checkpoint=Path(base_checkpoint),
            tokenizer_path=Path(tokenizer_path),
            expected=authenticated_inputs,
        )
        checkpoint = _save_checkpoint(output / f"epoch-{epoch}.pt", model)
        calibration = _calibrate_model(
            model,
            bundle.calibration,
            calibration_records,
            device=resolved_device,
        )
        model.train()
        _require_unchanged_sources(changed_hashes)
        epoch_report = {
            "epoch": epoch,
            "train_loss": train_loss["loss"],
            "state_loss": train_loss["state_loss"],
            "pointer_loss": train_loss["pointer_loss"],
            "seconds": time.monotonic() - epoch_started,
            "checkpoint": checkpoint,
            "calibration": calibration,
        }
        epoch_reports.append(epoch_report)
        print(
            json.dumps({"event": "epoch_complete", **epoch_report}, sort_keys=True),
            flush=True,
        )

    if global_step != STEPS_PER_SEED:
        raise RuntimeError("H4 completed with an unexpected optimizer-step count")
    selected = select_epoch_report(epoch_reports)
    selected_checkpoint = dict(selected["checkpoint"])
    candidate = {"epoch": selected["epoch"], **selected_checkpoint}

    report = {
        "schema_version": H4_TRAINING_REPORT_SCHEMA_VERSION,
        "recipe": H4_TRAINING_RECIPE_VERSION,
        "status": "complete",
        "seed": seed,
        "device": resolved_device,
        "architecture_version": ARCHITECTURE_VERSION,
        "parameter_count": NANO_EVIDENCE_QUERY_PARAMETER_COUNT,
        "trunk_parameter_count": NANO_TRUNK_PARAMETER_COUNT,
        "evidence_query_head_parameter_count": EVIDENCE_QUERY_HEAD_PARAMETER_COUNT,
        "architecture_identity": FROZEN_NANO_V01.architecture_identity,
        "base_checkpoint_sha256": base_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "dataset_manifest_sha256": bundle.manifest_sha256,
        "dataset": {
            "schema_version": DATASET_SCHEMA_VERSION,
            "generator": surface_transfer_data.GENERATOR_VERSION,
            "target_grammar": TARGET_GRAMMAR_VERSION,
            "source_manifest": _dataset_file_identity(
                Path(data_dir) / "manifest.json"
            ),
            "source_fit": {
                **_dataset_file_identity(Path(data_dir) / "fit.jsonl"),
                "records": len(bundle.fit),
                "worlds": FIT_WORLD_COUNT,
                "namespace": "train-fit",
                "gradient_bearing": True,
            },
            "source_calibration": {
                **_dataset_file_identity(Path(data_dir) / "calibration.jsonl"),
                "records": len(bundle.calibration),
                "worlds": CALIBRATION_WORLD_COUNT,
                "namespace": "train-calibration",
                "gradient_bearing": False,
            },
            "fit": {
                **dict(bundle.manifest["partitions"][FIT_PARTITION]),
                "state_class_counts": {
                    state.value: count
                    for state, count in zip(STATE_ORDER, fit_counts, strict=True)
                },
            },
            "calibration": {
                **dict(bundle.manifest["partitions"][CALIBRATION_PARTITION]),
                "state_class_counts": {
                    state.value: count
                    for state, count in zip(
                        STATE_ORDER, calibration_counts, strict=True
                    )
                },
            },
            "isolation": dict(bundle.manifest["isolation"]),
        },
        "hyperparameters": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "paired_variants_per_world": VARIANTS_PER_WORLD,
            "peak_learning_rate": PEAK_LEARNING_RATE,
            "warmup_fraction": WARMUP_FRACTION,
            "cosine_floor": COSINE_FLOOR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "adam_betas": list(ADAM_BETAS),
            "adam_epsilon": ADAM_EPSILON,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "total_steps": STEPS_PER_SEED,
            "state_class_order": [state.value for state in STATE_ORDER],
            "state_class_weight_source_counts": {
                state.value: count
                for state, count in zip(STATE_ORDER, STATE_CLASS_COUNTS, strict=True)
            },
            "state_class_weights": list(STATE_CLASS_WEIGHTS),
            "state_loss_weight": STATE_LOSS_WEIGHT,
            "state_loss_definition": STATE_LOSS_DEFINITION,
            "pointer_loss_weight": POINTER_LOSS_WEIGHT,
            "pointer_loss_definition": (
                "mean_of_start_and_end_cross_entropy_active_slots"
            ),
            "patient_token_masked": True,
            "prompt_template_id": POINTER_PROMPT_TEMPLATE_ID,
            "supervision_version": POINTER_SUPERVISION_VERSION,
            "uncertain_pointer_count": STATE_POINTER_COUNTS[
                STATE_ORDER.index(FieldState.UNCERTAIN)
            ],
            "full_context_evidence_queries": True,
            "shared_state_classifier": True,
            "deterministic_algorithms": True,
            "full_trunk_trainable": True,
            "world_grouped_batches": True,
        },
        "epochs": epoch_reports,
        "candidate": candidate,
        "calibration": {
            "selected_epoch": selected["epoch"],
            **selected["calibration"],
        },
        "dev_used_for_selection": False,
        "fresh_v1_accessed": False,
        "preserved_source_sha256": preserved_hashes,
        "changed_source_sha256": changed_hashes,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "tokenizers": tokenizers.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(torch.cuda.current_device())
                if resolved_device == "cuda"
                else None
            ),
            "cublas_workspace_config": (
                os.environ.get("CUBLAS_WORKSPACE_CONFIG")
                if resolved_device == "cuda"
                else None
            ),
            "platform": platform.platform(),
            "seconds": time.monotonic() - started,
        },
        "selection_note": (
            "The selected H4 epoch used only the disjoint 200-world "
            "training-calibration partition. Gradients used only the 2,800-world "
            "fit partition. The training command read no development, benchmark, "
            "historical-fresh, or sealed-confirmation records. H3 architecture, "
            "objective, optimizer, exposure, seeds, inference, calibration, and "
            "selection rules were preserved by literal source pins."
        ),
    }
    _require_unchanged_sources(changed_hashes)
    _require_input_hashes(
        data_dir=Path(data_dir),
        base_checkpoint=Path(base_checkpoint),
        tokenizer_path=Path(tokenizer_path),
        expected=authenticated_inputs,
    )
    _write_json_no_clobber(output / "training_report.json", report)
    print(
        json.dumps(
            {
                "event": "training_complete",
                "seed": seed,
                "candidate": candidate,
                "report": "training_report.json",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Train Nano's frozen H4 data-only candidate"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=root / "checkpoints" / "anchors" / "nano_v01_scribe.pt",
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=root / "sft" / "tokenizer.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_evidence_query_h4_candidate(
        data_dir=args.data_dir,
        base_checkpoint=args.base_checkpoint,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CALIBRATION_STATE_CLASS_COUNTS",
    "CALIBRATION_STATE_FIELD_QUOTA",
    "FIT_STATE_CLASS_COUNTS",
    "FIT_STATE_FIELD_QUOTA",
    "H4_TRAINING_RECIPE_VERSION",
    "H4_TRAINING_REPORT_SCHEMA_VERSION",
    "PRESERVED_H3_SOURCE_SHA256",
    "changed_source_paths",
    "load_h4_training_bundle",
    "preserved_source_paths",
    "train_evidence_query_h4_candidate",
]
