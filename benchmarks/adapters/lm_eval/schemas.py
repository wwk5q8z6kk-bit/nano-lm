"""Typed manifest schemas and content-addressed run identity for Program 0."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "benchmark_lab.manifest.v1"


class RunStatus(str, Enum):
    PLANNED = "PLANNED"
    VALIDATED = "VALIDATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    VOID = "VOID"
    CANCELLED = "CANCELLED"
    PARTIAL = "PARTIAL"


class DecisionStatus(str, Enum):
    INFRA_SMOKE_PASS = "INFRA_SMOKE_PASS"
    INFRA_SMOKE_FAIL = "INFRA_SMOKE_FAIL"
    LEADERBOARD_ONLY = "LEADERBOARD_ONLY"
    CANDIDATE_FOR_REPLICATION = "CANDIDATE_FOR_REPLICATION"
    CANDIDATE_FOR_PROMOTION = "CANDIDATE_FOR_PROMOTION"
    REJECTED = "REJECTED"


def canonical_json(obj: Any) -> str:
    """Stable JSON for hashing (sorted keys, no whitespace drift)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_run_id(
    *,
    benchmark_manifest_sha256: str,
    model_or_solver_manifest_sha256: str,
    config_sha256: str,
    code_git_commit: str,
) -> str:
    payload = {
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "model_or_solver_manifest_sha256": model_or_solver_manifest_sha256,
        "config_sha256": config_sha256,
        "code_git_commit": code_git_commit,
    }
    return sha256_hex(canonical_json(payload))


@dataclass
class BenchmarkManifest:
    schema_version: str
    suite_id: str
    task_id: str
    task_version: str
    task_yaml_sha256: str
    source_instrument_paths: list[str]
    source_instrument_git_commit: str
    source_artifact_sha256: str
    record_count: int
    prompt_template_hash: str
    scorer_hash: str
    filter_pipeline_hash: str
    metric_definitions: list[dict[str, Any]]
    protected_metrics: list[str]
    benchmark_license: str
    contamination_status: str
    schema_id: str = "held_value_scribe_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))


@dataclass
class ModelManifest:
    schema_version: str
    model_id: str
    model_family: str
    architecture: str
    parameter_count: int | None
    checkpoint_sha256: str | None
    tokenizer_sha256: str | None
    training_token_metadata: dict[str, Any]
    adapter_or_finetune_state: str
    quantization: str
    backend: str
    decoding_configuration: dict[str, Any]
    resource_class_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))


@dataclass
class SolverManifest:
    """Deterministic classical / template solver (not a neural model)."""

    schema_version: str
    solver_id: str
    solver_family: str
    method: str
    implementation_hash: str
    resource_class_ids: list[str]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))


@dataclass
class RunManifest:
    schema_version: str
    run_id: str
    benchmark_manifest_hash: str
    model_or_solver_manifest_hash: str
    config_hash: str
    environment_hash: str
    code_git_commit: str
    start_utc: str
    end_utc: str
    run_status: str
    decision_status: str
    artifact_paths_and_hashes: dict[str, str]
    cost_reference: dict[str, Any]
    failure_information: dict[str, Any] = field(default_factory=dict)
    promote: bool = False
    leaderboard_eligible: bool = False
    evidence_ledger_eligible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))
