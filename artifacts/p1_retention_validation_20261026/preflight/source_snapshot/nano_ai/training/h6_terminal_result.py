"""One-shot, result-only terminal evidence wrapper for frozen Nano H6.

This wrapper is the last conservative boundary before the frozen evaluator can
open development data.  It authenticates the materialized package, recovered
training artifacts, admission receipt, and current RunPod ledger; consumes a
fixed package-root invocation marker; runs the exact launcher in a controlled
environment; and preserves a result-only success or failure bundle.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = "nano.h6-terminal-evaluation-evidence.v2"
WRAPPER_ID = "nano.h6-terminal-result-wrapper.v2"
PACKAGE_ID = "nano-h6-runpod-input-v1"
EVALUATION_AUTHORITY_ID = "nano-h6-dev-one-shot-v1"
EVALUATION_SCRIPT_NAME = "RUN_H6_EVALUATE.sh"
EVALUATOR_SOURCE_RELATIVE = "nano_ai/training/evaluate_evidence_query_h6.py"
ARCHIVE_ROOT = "h6-terminal-evaluation-evidence"
INVOCATION_MARKER_NAME = ".h6-terminal-evaluation-invoked.json"
WRAPPER_FAILURE_EXIT_CODE = 70

_LEDGER_SCHEMA = "nano.runpod.ledger.v2"
_ADMISSION_SCHEMA = "nano.h6-destination-admission.v1"
_GUARD_ADMISSION_SCHEMA = "nano.runpod.destination-admission.v1"
_ADMISSION_SYNC_SCHEMA = "nano.runpod.destination-admission-sync.v1"
_ADMISSION_SYNC_PREPARED_EVENT = "DESTINATION_ADMISSION_SYNC_PREPARED"
_ADMISSION_SYNC_PROTOCOL = "atomic_content_addressed_authority_mirror_v1"
_ADMISSION_RECEIPT_BINDING_SCHEMA = (
    "nano.runpod.destination-admission-receipt-binding.v1"
)
_ADMISSION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "generated_at_utc",
        "run",
        "authority_hashes",
        "authority_identities",
        "authenticated_sources",
        "continuation",
        "destination_attestation",
        "admission_sync",
        "upload_boundary",
        "artifacts",
        "selection",
        "runtime_observation",
        "strict_checkpoint_loads",
        "prohibited_data",
        "admission_programs",
        "provider_state_mutated",
        "guard_state_mutated",
    }
)
_ADMISSION_SYNC_PATH_ROLES = frozenset(
    {
        "run_spec",
        "run_events",
        "destination_attestation",
        "admission_receipt",
    }
)
_ADMISSION_SYNC_PREPARED_KEYS = frozenset(
    {
        "schema",
        "transaction_id",
        "run_id",
        "phase",
        "ledger_tail_before",
        "final_network_volume_snapshot_event_index",
        "final_network_volume_snapshot_event_sha256",
        "final_destination_snapshot_event_index",
        "final_destination_snapshot_event_sha256",
        "destination_resource_id",
        "network_volume_resource_id",
        "authorization_envelope_sha256",
        "upload_allowlist_sha256",
        "runtime_authority_sha256",
        "remote_root",
        "relative_paths",
        "transfer_protocol",
        "transfer_provider_state_mutated",
        "transfer_guard_state_mutated",
    }
)
_ADMISSION_RECEIPT_BINDING_KEYS = frozenset(
    {
        "schema",
        "receipt",
        "run",
        "authority_hashes",
        "authority_identities",
        "authenticated_sources",
        "continuation",
        "upload_boundary",
        "destination_attestation",
        "admission_programs",
        "provider_state_mutated",
        "guard_state_mutated",
    }
)
_ADMISSION_RECEIPT_IDENTITY_KEYS = frozenset(
    {
        "sha256",
        "bytes",
        "relative_path",
        "schema_version",
        "status",
        "generated_at_utc",
    }
)
_ADMISSION_RECEIPT_RUN_KEYS = frozenset(
    {
        "run_id",
        "destination_id",
        "observation_operation_id",
        "creation_operation_id",
        "rehydration_operation_id",
        "ledger",
    }
)
_ADMISSION_RECEIPT_LEDGER_KEYS = frozenset(
    {"event_count", "tail_sha256", "file_sha256", "bytes"}
)
_ADMISSION_RECEIPT_BOUND_SLICES = (
    "authority_hashes",
    "authority_identities",
    "authenticated_sources",
    "continuation",
    "upload_boundary",
    "destination_attestation",
)
_ADMISSION_RECEIPT_PROGRAM_BINDING_KEYS = frozenset({"authority", "receipt"})
_ADMISSION_RECEIPT_PROGRAM_AUTHORITY_KEYS = frozenset(
    {
        "continuation_authorization_sha256",
        "admission_program_sha256",
        "terminal_wrapper_sha256",
    }
)
_ADMISSION_RECEIPT_PROGRAM_KEYS = frozenset(
    {"destination_admission", "terminal_wrapper"}
)
_RECEIPT_IDENTITY_KEYS = frozenset({"bytes", "sha256"})
_RECEIPT_SOURCE_IDENTITY_KEYS = frozenset({"path", "bytes", "sha256"})
_RECEIPT_AUTHORITY_IDENTITY_KEYS = frozenset(
    {
        "run_spec",
        "run_events",
        "preregistration",
        "training_report_freeze",
        "readiness",
        "package",
    }
)
_RECEIPT_CONTINUATION_KEYS = frozenset(
    {
        "authorization",
        "predecessor",
        "runtime_authority",
        "destination_selection",
        "required_artifacts_manifest",
    }
)
_RECEIPT_PREDECESSOR_KEYS = frozenset(
    {
        "run_id",
        "phase",
        "ledger",
        "source_pod_id",
        "source_machine_id",
        "source_required_state",
        "source_access_policy",
    }
)
_RECEIPT_UPLOAD_BOUNDARY_KEYS = frozenset(
    {
        "policy_id",
        "identity",
        "static_file_count",
        "dynamic_file_count",
        "derived_materialized_file_count",
        "pre_admission_file_count",
        "derived_environment_exception",
        "no_extra_files",
        "authorization_hash_embedded",
        "receipt_relative_path",
        "receipt_was_absent_before_admission",
    }
)
_RECEIPT_ATTESTATION_KEYS = frozenset(
    {
        "attestation_id",
        "identity",
        "provider_snapshot_event_index",
        "provider_snapshot_event_sha256",
        "destination_observation_sha256",
        "network_volume_provider_snapshot_event_index",
        "network_volume_provider_snapshot_event_sha256",
        "network_volume_observation_sha256",
        "network_volume_observation_evidence_sha256",
        "operation_lineage",
        "destination",
    }
)
_RECEIPT_OPERATION_LINEAGE_KEYS = frozenset(
    {"create_volume", "create_pod", "rehydrate_destination"}
)
_RECEIPT_OPERATION_KEYS = frozenset(
    {
        "operation_id",
        "kind",
        "start_event_index",
        "start_event_sha256",
        "finish_event_index",
        "finish_event_sha256",
        "outcome",
    }
)
_RECEIPT_OPERATION_PROVIDER_KEYS = _RECEIPT_OPERATION_KEYS | {
    "provider_snapshot_event_index",
    "provider_snapshot_event_sha256",
}
_RECEIPT_DESTINATION_KEYS = frozenset(
    {
        "pod_id",
        "machine_id",
        "network_volume_id",
        "data_center_id",
        "cloud_type",
        "secure_cloud",
        "lifecycle_state",
        "image_id",
        "observed_image_digest",
        "configured_gpu_id",
        "provider_gpu_model",
        "configured_gpu_count",
        "runtime_gpu_name",
        "runtime_gpu_count",
        "workspace_mount_path",
    }
)
_DESTINATION_ADMISSION_PROGRAM_IDENTITY = {
    "bytes": 142579,
    "sha256": "4abe19110692cb86ac6d5e3f3e2527c16b0d1c72e27502cde9a973eace759d49",
}
_ISOLATED_EXECUTION_SCHEMA = "nano.h6.isolated-execution-contract.v1"
_ISOLATED_RUNTIME_SCHEMA = "nano.h6-isolated-runtime-evidence.v1"
_ISOLATED_INTERPRETER_RELATIVE = ".h6-venv/bin/python"
_TERMINAL_WRAPPER_RELATIVE = "nano_ai/training/h6_terminal_result.py"
_ADMISSION_PROGRAM_RELATIVE = "nano_ai/training/admit_evidence_query_h6.py"
_EXPECTED_EVALUATOR_MODULE = "nano_ai.training.evaluate_evidence_query_h6"
_ISOLATED_BRIDGE_ID = "module_only_v1"
_ISOLATED_SYSCONFIG_PROBE = r"""
import json
import sys
import sysconfig

print(json.dumps({
    "schema": "nano.h6-isolated-sysconfig-probe.v1",
    "executable": sys.executable,
    "flags": {"isolated": sys.flags.isolated, "no_site": sys.flags.no_site},
    "version": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
    "prefixes": {
        "prefix": sys.prefix,
        "exec_prefix": sys.exec_prefix,
        "base_prefix": sys.base_prefix,
        "base_exec_prefix": sys.base_exec_prefix,
    },
    "paths": {name: sysconfig.get_path(name) for name in ("purelib", "platlib", "data")},
    "sys_path": sys.path,
}, sort_keys=True, separators=(",", ":")))
""".strip()
_ISOLATED_MODULE_BRIDGE = r"""
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

EXPECTED_MODULE = "nano_ai.training.evaluate_evidence_query_h6"

def refuse(reason):
    raise SystemExit("H6 isolated evaluator bridge refused: " + reason)

if len(sys.argv) < 5 or sys.argv[1] not in {"audit", "run"}:
    refuse("invalid bridge mode")
mode, source_text, dependencies_text, expected_executable = sys.argv[1:5]
if sys.flags.isolated != 1 or sys.flags.no_site != 1:
    refuse("Python startup is not -I -S")
if os.path.abspath(sys.executable) != expected_executable:
    refuse("interpreter identity changed")
try:
    source = Path(source_text).resolve(strict=True)
    dependencies = tuple(Path(value).resolve(strict=True) for value in json.loads(dependencies_text))
except (OSError, TypeError, ValueError, json.JSONDecodeError):
    refuse("runtime roots are invalid")
if not source.is_dir() or not dependencies or any(not path.is_dir() for path in dependencies):
    refuse("runtime roots are unavailable")
if any(name == "nano_ai" or name.startswith("nano_ai.") for name in sys.modules):
    refuse("Nano was imported before source authentication")
if "torch" in sys.modules or "tokenizers" in sys.modules:
    refuse("dependencies were imported before path authentication")
dependency_strings = [os.fspath(path) for path in dependencies]
if os.fspath(source) in sys.path or any(value in sys.path for value in dependency_strings):
    refuse("runtime roots were present before bootstrap")
sys.path.extend(dependency_strings)
origins = {}
for name in ("torch", "tokenizers"):
    module = importlib.import_module(name)
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str) or not raw_origin:
        refuse(name + " has no file-backed origin")
    try:
        origin = Path(raw_origin).resolve(strict=True)
    except OSError:
        refuse(name + " origin is unavailable")
    accepted_root = next(
        (root for root in dependencies if origin == root or root in origin.parents),
        None,
    )
    if accepted_root is None:
        refuse(name + " escaped audited dependency roots")
    origins[name] = {
        "origin": os.fspath(origin),
        "dependency_root": os.fspath(accepted_root),
    }
if any(name == "nano_ai" or name.startswith("nano_ai.") for name in sys.modules):
    refuse("a dependency imported Nano before source authentication")
if any(name in sys.modules for name in ("site", "sitecustomize", "usercustomize")):
    refuse("a startup customization module executed")
sys.path.insert(0, os.fspath(source))
if mode == "audit":
    if len(sys.argv) != 5:
        refuse("audit received evaluator arguments")
    print(json.dumps({
        "schema": "nano.h6-isolated-module-audit.v1",
        "flags": {"isolated": sys.flags.isolated, "no_site": sys.flags.no_site},
        "imports": origins,
        "source_precedence": sys.path[0] == os.fspath(source),
        "startup_hooks_loaded": False,
    }, sort_keys=True, separators=(",", ":")))
else:
    if len(sys.argv) < 7 or sys.argv[5] != "-m" or sys.argv[6] != EXPECTED_MODULE:
        refuse("only the frozen evaluator module is permitted")
    sys.argv = [EXPECTED_MODULE, *sys.argv[7:]]
    runpy.run_module(EXPECTED_MODULE, run_name="__main__", alter_sys=True)
""".strip()
_ISOLATED_LAUNCHER_BRIDGE = r"""
set -euo pipefail
readonly nano_interpreter="$1"
readonly nano_source="$2"
readonly nano_dependencies="$3"
readonly nano_bridge="$4"
readonly nano_module="nano_ai.training.evaluate_evidence_query_h6"
python() {
    if (( $# < 2 )) || [[ "$1" != "-m" || "$2" != "$nano_module" ]]; then
        printf '%s\n' 'H6 isolated evaluator bridge refused non-module Python invocation' >&2
        return 70
    fi
    shift 2
    "$nano_interpreter" -I -S -c "$nano_bridge" run \
        "$nano_source" "$nano_dependencies" "$nano_interpreter" \
        -m "$nano_module" "$@"
}
source "$0"
""".strip()
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\-]{0,255}")


@dataclass(frozen=True)
class ExpectedFile:
    sha256: str
    bytes: int | None = None


@dataclass(frozen=True)
class FrozenIdentity:
    package_id: str
    package_sha256: str
    archive_root: str
    bundle_manifest_sha256: str
    evaluation_authority_id: str
    evaluation_script_sha256: str
    evaluator_sha256: str
    development_sha256: str
    development_manifest_sha256: str
    recovered_files: Mapping[str, ExpectedFile]
    development_files: Mapping[str, ExpectedFile]


_PRODUCTION_IDENTITY = FrozenIdentity(
    package_id=PACKAGE_ID,
    package_sha256="b1eff7c9ccb06f05e46f0639c88a7e74daeb0880dc8ecbb0641359d74b5ea505",
    archive_root="nano-h6-runpod",
    bundle_manifest_sha256=(
        "38ec1d3efd8476a7125839c937276f9fcf9c9c14d3807f73fea546ac6319bdae"
    ),
    evaluation_authority_id=EVALUATION_AUTHORITY_ID,
    evaluation_script_sha256=(
        "234b9e1e1edb486dd758bd4904c75ef3513c7052a2c709f41a13810b7bfad8ab"
    ),
    evaluator_sha256=(
        "c796c2e150aaae50b4f80e50296bf89d4cfdc3ebf105d02e447e2580d37cfa39"
    ),
    development_sha256=(
        "9c893d8e64110287b433d567e0e9abb42c611ecba33b40de192741324d37e290"
    ),
    development_manifest_sha256=(
        "47ee157ac037c0771100b8546c90da91dbd2006198700bb642f1561d2124c1a3"
    ),
    recovered_files={
        "results/seed-20260805/training_report.json": ExpectedFile(
            "9b7bfe0be903d7f43b36d22dc61ecd2ae4c9cf3deb5e9f1a993733bf18d09932",
            41461,
        ),
        "results/seed-20260805/epoch-3.pt": ExpectedFile(
            "4ca7b2513ae8b0cf8b9c525a853144d1b75b422b560a71818caa70d981c30472",
            13168916,
        ),
        "results/seed-20260806/training_report.json": ExpectedFile(
            "2d5bb2e95913f46c181f9273677e9ec3ad482dbf31d0111b2bc9433e15379d8f",
            41467,
        ),
        "results/seed-20260806/epoch-2.pt": ExpectedFile(
            "40611c2733e731bdd0eeb827f852fb2f14312445c05cd293a5533296e2ce982f",
            13168916,
        ),
        "results/TRAINING_REPORT_SHA256SUMS": ExpectedFile(
            "e1509af3f11b03ad68131876aa591fe3998a8a7e8e2a4f9c4d271e738302b464",
            218,
        ),
        "sft/tokenizer.json": ExpectedFile(
            "bae49648bfcc4904c50e2f006ee184bd26e74454ee170663e30a8e71640ce3c9",
            262846,
        ),
    },
    development_files={
        "h2_development/manifest.json": ExpectedFile(
            "47ee157ac037c0771100b8546c90da91dbd2006198700bb642f1561d2124c1a3"
        ),
        "h2_development/dev.jsonl": ExpectedFile(
            "9c893d8e64110287b433d567e0e9abb42c611ecba33b40de192741324d37e290"
        ),
    },
)


# The frozen evaluator's archive contains training reports and is intentionally
# excluded.  Only the terminal result/checksum files below may be recovered.
_GENERATED_RESULT_ALLOWLIST: tuple[tuple[str, str, int], ...] = (
    (
        "results/development_evaluation.json",
        "generated-results/development_evaluation.json",
        64 * 1024 * 1024,
    ),
    (
        "results/development_evaluation.log",
        "generated-results/development_evaluation.log",
        64 * 1024 * 1024,
    ),
    ("results/SHA256SUMS", "generated-results/SHA256SUMS", 1024 * 1024),
    (
        "nano-h6-results.tar.gz.sha256",
        "generated-results/nano-h6-results.tar.gz.sha256",
        1024 * 1024,
    ),
)
_EVALUATOR_TEMPORARY_PATHS = (
    "SHA256SUMS.tmp",
    "nano-h6-results.tar.gz.tmp",
    "nano-h6-results.tar.gz.sha256.tmp",
)
_EVALUATOR_EXCLUDED_FINAL_PATHS = ("nano-h6-results.tar.gz",)
_PROHIBITED_PACKAGE_PATHS = (
    "fresh",
    "fresh-v1",
    "private",
    "private-corpus",
)
_FORBIDDEN_MEMBER_MARKERS = (
    "h2_development",
    "fresh",
    "private",
    "cache",
    "__pycache__",
    "checkpoint",
    "h6_data",
    "fit.jsonl",
    "calibration.jsonl",
    "training_report",
    ".pt",
)


class H6TerminalResultError(RuntimeError):
    """Raised when a one-shot terminal evidence invariant is not satisfied."""


@dataclass(frozen=True)
class AuthorityBindings:
    wrapper_sha256: str
    admission_receipt: Path
    admission_receipt_sha256: str
    admission_event_sha256: str
    run_spec: Path
    run_spec_sha256: str
    run_events: Path
    run_events_sha256: str
    run_events_tail_sha256: str


@dataclass(frozen=True)
class TerminalRunOutcome:
    evaluator_returncode: int | None
    exit_code: int
    archive_sha256: str
    terminal_dir: Path
    terminal_archive: Path
    status: str


@dataclass(frozen=True)
class LedgerAuthority:
    spec: Mapping[str, object]
    events: tuple[Mapping[str, object], ...]
    file_payload: bytes
    file_sha256: str
    tail_sha256: str
    admission_event: Mapping[str, object]
    release_event: Mapping[str, object]
    open_event: Mapping[str, object]
    receipt: Mapping[str, object]
    receipt_sha256: str


@dataclass(frozen=True)
class AdmissionSyncBoundary:
    transaction_id: str
    prepared_event_index: int
    prepared_event_sha256: str
    receipt_relative_path: str
    receipt_echo: Mapping[str, object]
    final_network_volume_snapshot_event_index: int
    final_network_volume_snapshot_event_sha256: str
    final_destination_snapshot_event_index: int
    final_destination_snapshot_event_sha256: str


@dataclass(frozen=True)
class IsolatedRuntime:
    interpreter: Path
    dependency_roots: tuple[Path, ...]
    dependency_roots_json: str
    evidence: Mapping[str, object]


def isolated_execution_contract() -> dict[str, object]:
    """Return the literal admission/guard-bindable execution contract."""

    return {
        "schema": _ISOLATED_EXECUTION_SCHEMA,
        "interpreter_relative_path": _ISOLATED_INTERPRETER_RELATIVE,
        "python_flags": ["-I", "-S"],
        "admission_program_relative_path": _ADMISSION_PROGRAM_RELATIVE,
        "terminal_wrapper_relative_path": _TERMINAL_WRAPPER_RELATIVE,
        "evaluator_launcher_relative_path": EVALUATION_SCRIPT_NAME,
        "evaluator_launcher_shell": "/bin/bash",
        "evaluator_launcher_flags": ["--noprofile", "--norc"],
        "terminal_wrapper_bridge": _ISOLATED_BRIDGE_ID,
    }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise H6TerminalResultError("value is not canonical JSON") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _parse_object(payload: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise H6TerminalResultError(f"{label} is not valid strict JSON") from exc
    if not isinstance(value, dict):
        raise H6TerminalResultError(f"{label} must be a JSON object")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise H6TerminalResultError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise H6TerminalResultError(f"{label} is not a safe stable identifier")
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise H6TerminalResultError(f"{label} must be an object")
    return value


def _receipt_identity(value: object, label: str) -> Mapping[str, object]:
    identity = _mapping(value, label)
    size = identity.get("bytes")
    if (
        set(identity) != _RECEIPT_IDENTITY_KEYS
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise H6TerminalResultError(f"{label} has an invalid identity shape")
    _require_sha256(identity.get("sha256"), f"{label} SHA-256")
    return identity


def _verify_receipt_slice_shapes(receipt: Mapping[str, object]) -> None:
    """Mirror the guard's exact authenticated receipt sub-schemas.

    The persisted guard binding is an exact copy of these slices.  Exact-copy
    equality alone would still accept a jointly forged receipt and binding with
    ignored extra fields, so the terminal boundary independently authenticates
    every nested key set that the guard recognizes.
    """

    identities = _mapping(
        receipt.get("authority_identities"), "H6 admission authority identities"
    )
    if set(identities) != _RECEIPT_AUTHORITY_IDENTITY_KEYS:
        raise H6TerminalResultError(
            "H6 admission authority identity inventory is invalid"
        )
    for name in _RECEIPT_AUTHORITY_IDENTITY_KEYS:
        _receipt_identity(identities.get(name), f"H6 admission {name} identity")

    sources = _mapping(
        receipt.get("authenticated_sources"), "H6 admission authenticated sources"
    )
    if not sources:
        raise H6TerminalResultError("H6 admission authenticated sources are empty")
    observed_paths: set[str] = set()
    for name, raw_source in sources.items():
        if not isinstance(name, str) or not name:
            raise H6TerminalResultError(
                "H6 admission authenticated source name is invalid"
            )
        source = _mapping(raw_source, f"H6 admission authenticated source {name}")
        size = source.get("bytes")
        if (
            set(source) != _RECEIPT_SOURCE_IDENTITY_KEYS
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise H6TerminalResultError(
                f"H6 admission authenticated source {name} is invalid"
            )
        path = _safe_relative(
            source.get("path"), f"H6 admission authenticated source {name} path"
        )
        if path in observed_paths:
            raise H6TerminalResultError(
                "H6 admission authenticated source paths are duplicated"
            )
        observed_paths.add(path)
        _require_sha256(
            source.get("sha256"), f"H6 admission authenticated source {name}"
        )

    continuation = _mapping(receipt.get("continuation"), "H6 admission continuation")
    if set(continuation) != _RECEIPT_CONTINUATION_KEYS:
        raise H6TerminalResultError("H6 admission continuation shape is invalid")
    for name in (
        "authorization",
        "runtime_authority",
        "destination_selection",
        "required_artifacts_manifest",
    ):
        _receipt_identity(continuation.get(name), f"H6 admission continuation {name}")
    predecessor = _mapping(continuation.get("predecessor"), "H6 admission predecessor")
    predecessor_ledger = _mapping(
        predecessor.get("ledger"), "H6 admission predecessor ledger"
    )
    if (
        set(predecessor) != _RECEIPT_PREDECESSOR_KEYS
        or set(predecessor_ledger) != _ADMISSION_RECEIPT_LEDGER_KEYS
        or predecessor.get("phase") != "REPORTS_FROZEN"
        or predecessor.get("source_required_state") != "stopped"
        or predecessor.get("source_access_policy") != "lineage_only_no_restart"
    ):
        raise H6TerminalResultError("H6 admission predecessor shape is invalid")
    for name in ("run_id", "source_pod_id", "source_machine_id"):
        _require_identifier(predecessor.get(name), f"H6 admission predecessor {name}")
    for name in ("event_count", "bytes"):
        observed = predecessor_ledger.get(name)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed <= 0:
            raise H6TerminalResultError(
                "H6 admission predecessor ledger identity is invalid"
            )
    for name in ("tail_sha256", "file_sha256"):
        _require_sha256(
            predecessor_ledger.get(name), f"H6 admission predecessor ledger {name}"
        )

    upload = _mapping(receipt.get("upload_boundary"), "H6 admission upload boundary")
    if set(upload) != _RECEIPT_UPLOAD_BOUNDARY_KEYS:
        raise H6TerminalResultError("H6 admission upload boundary shape is invalid")
    _receipt_identity(upload.get("identity"), "H6 admission upload identity")
    if not isinstance(upload.get("policy_id"), str) or not upload.get("policy_id"):
        raise H6TerminalResultError("H6 admission upload policy ID is invalid")
    for name in (
        "static_file_count",
        "dynamic_file_count",
        "derived_materialized_file_count",
        "pre_admission_file_count",
    ):
        observed = upload.get(name)
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise H6TerminalResultError("H6 admission upload counts are invalid")
    _safe_relative(
        upload.get("derived_environment_exception"),
        "H6 admission derived environment exception",
    )
    _safe_relative(
        upload.get("receipt_relative_path"),
        "H6 admission receipt relative path",
    )
    if (
        upload.get("no_extra_files") is not True
        or upload.get("authorization_hash_embedded") is not False
        or upload.get("receipt_was_absent_before_admission") is not True
    ):
        raise H6TerminalResultError("H6 admission upload boundary is invalid")

    attestation = _mapping(
        receipt.get("destination_attestation"), "H6 admission attestation"
    )
    if set(attestation) != _RECEIPT_ATTESTATION_KEYS:
        raise H6TerminalResultError("H6 admission attestation shape is invalid")
    _require_identifier(
        attestation.get("attestation_id"), "H6 admission attestation ID"
    )
    _receipt_identity(attestation.get("identity"), "H6 admission attestation identity")
    lineage = _mapping(
        attestation.get("operation_lineage"), "H6 admission operation lineage"
    )
    if set(lineage) != _RECEIPT_OPERATION_LINEAGE_KEYS:
        raise H6TerminalResultError("H6 admission operation lineage shape is invalid")
    for role in _RECEIPT_OPERATION_LINEAGE_KEYS:
        operation = _mapping(lineage.get(role), f"H6 admission {role} operation")
        expected_keys = (
            _RECEIPT_OPERATION_KEYS
            if role == "rehydrate_destination"
            else _RECEIPT_OPERATION_PROVIDER_KEYS
        )
        if (
            set(operation) != expected_keys
            or operation.get("kind") != role
            or operation.get("outcome") not in {"succeeded", "reconciled_succeeded"}
        ):
            raise H6TerminalResultError(
                f"H6 admission {role} operation shape is invalid"
            )
        _require_identifier(
            operation.get("operation_id"), f"H6 admission {role} operation ID"
        )
        index_names = ["start_event_index", "finish_event_index"]
        hash_names = ["start_event_sha256", "finish_event_sha256"]
        if role != "rehydrate_destination":
            index_names.append("provider_snapshot_event_index")
            hash_names.append("provider_snapshot_event_sha256")
        for name in index_names:
            observed = operation.get(name)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, int)
                or observed <= 0
            ):
                raise H6TerminalResultError(
                    f"H6 admission {role} operation index is invalid"
                )
        for name in hash_names:
            _require_sha256(
                operation.get(name), f"H6 admission {role} operation {name}"
            )

    destination = _mapping(
        attestation.get("destination"), "H6 admission attested destination"
    )
    if set(destination) != _RECEIPT_DESTINATION_KEYS:
        raise H6TerminalResultError(
            "H6 admission attested destination shape is invalid"
        )


def _lexists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_relative(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise H6TerminalResultError(f"{label} must be relative text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise H6TerminalResultError(f"{label} is unsafe")
    return value


def _resolve_new_path(path: Path, label: str) -> Path:
    if not path.name or path.name in {".", ".."}:
        raise H6TerminalResultError(f"{label} must name a new path")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise H6TerminalResultError(f"{label} parent does not exist") from exc
    if not parent.is_dir():
        raise H6TerminalResultError(f"{label} parent is not a directory")
    resolved = parent / path.name
    if _lexists(resolved):
        raise H6TerminalResultError(f"{label} already exists: {resolved}")
    return resolved


def _require_real_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise H6TerminalResultError(f"{label} must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise H6TerminalResultError(f"{label} does not exist") from exc
    if not resolved.is_dir():
        raise H6TerminalResultError(f"{label} is not a directory")
    return resolved


def _require_real_regular_file(path: Path, label: str) -> os.stat_result:
    if path.is_symlink():
        raise H6TerminalResultError(f"{label} must not be a symbolic link")
    try:
        metadata = path.stat()
    except OSError as exc:
        raise H6TerminalResultError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise H6TerminalResultError(f"{label} must be a regular file")
    if metadata.st_nlink != 1:
        raise H6TerminalResultError(f"{label} must not be hard-linked")
    return metadata


def _identity_tuple(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _read_stable_file(
    path: Path,
    label: str,
    *,
    root: Path | None = None,
    maximum_bytes: int | None = None,
) -> tuple[bytes, dict[str, object]]:
    before = _require_real_regular_file(path, label)
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        raise H6TerminalResultError(f"{label} is unexpectedly large")
    resolved = path.resolve(strict=True)
    if root is not None and not _is_relative_to(resolved, root):
        raise H6TerminalResultError(f"{label} escapes its authenticated root")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise H6TerminalResultError(f"{label} could not be read") from exc
    after = _require_real_regular_file(path, label)
    if (
        _identity_tuple(before) != _identity_tuple(after)
        or len(payload) != after.st_size
    ):
        raise H6TerminalResultError(f"{label} changed while read")
    return payload, {"bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _verify_expected_file(
    root: Path, relative: str, expected: ExpectedFile, label: str
) -> dict[str, object]:
    relative = _safe_relative(relative, label)
    _payload, identity = _read_stable_file(root / relative, label, root=root)
    if identity["sha256"] != _require_sha256(expected.sha256, f"{label} SHA-256"):
        raise H6TerminalResultError(f"{label} SHA-256 mismatch")
    if expected.bytes is not None and identity["bytes"] != expected.bytes:
        raise H6TerminalResultError(f"{label} byte count mismatch")
    return {"path": relative, **identity}


def _snapshot_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    normalized = sorted((dict(row) for row in rows), key=lambda row: str(row["path"]))
    return {
        "files": normalized,
        "snapshot_sha256": _sha256_bytes(_canonical_bytes(normalized)),
    }


def _verify_package_tree(package: Path, identity: FrozenIdentity) -> dict[str, object]:
    manifest_payload, manifest_identity = _read_stable_file(
        package / "BUNDLE_MANIFEST.json",
        "frozen H6 bundle manifest",
        root=package,
        maximum_bytes=16 * 1024 * 1024,
    )
    if manifest_identity["sha256"] != identity.bundle_manifest_sha256:
        raise H6TerminalResultError("frozen H6 bundle manifest SHA-256 mismatch")
    manifest = _parse_object(manifest_payload, "frozen H6 bundle manifest")
    if manifest.get("archive_root") != identity.archive_root:
        raise H6TerminalResultError("frozen H6 bundle archive root mismatch")
    members = _mapping(manifest.get("members"), "frozen H6 bundle members")
    if not members:
        raise H6TerminalResultError("frozen H6 bundle member inventory is empty")
    rows: list[dict[str, object]] = []
    for member_name, raw_expected in sorted(members.items()):
        relative = _safe_relative(member_name, "frozen H6 package member")
        expected = _mapping(raw_expected, f"frozen H6 member {relative} identity")
        if set(expected) != {"bytes", "sha256"}:
            raise H6TerminalResultError(
                f"frozen H6 member {relative} identity is not exact"
            )
        byte_count = expected.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise H6TerminalResultError(
                f"frozen H6 member {relative} byte count is invalid"
            )
        rows.append(
            _verify_expected_file(
                package,
                relative,
                ExpectedFile(
                    _require_sha256(
                        expected.get("sha256"), f"frozen H6 member {relative} SHA-256"
                    ),
                    byte_count,
                ),
                f"frozen H6 package member {relative}",
            )
        )
    snapshot = _snapshot_rows(rows)
    return {
        "archive_package_sha256": identity.package_sha256,
        "manifest": manifest_identity,
        **snapshot,
    }


def _verify_file_set(
    package: Path, expected: Mapping[str, ExpectedFile], role: str
) -> dict[str, object]:
    rows = [
        _verify_expected_file(package, relative, value, f"{role} {relative}")
        for relative, value in sorted(expected.items())
    ]
    return _snapshot_rows(rows)


def _assert_prohibited_paths_absent(package: Path) -> None:
    for relative in _PROHIBITED_PACKAGE_PATHS:
        if _lexists(package / relative):
            raise H6TerminalResultError(
                f"prohibited data path is present in H6 package: {relative}"
            )


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except OSError as exc:
        raise H6TerminalResultError(f"refusing to clobber {path}") from exc
    # If writing fails, retain the partial exclusive file as evidence that this
    # one-shot path was consumed; a retry must never overwrite it.
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_row(path: str, payload: bytes, role: str) -> dict[str, object]:
    return {
        "bytes": len(payload),
        "path": path,
        "role": role,
        "sha256": _sha256_bytes(payload),
    }


def _verify_ledger_records(
    payload: bytes, expected_file_sha256: str, expected_tail_sha256: str
) -> tuple[tuple[Mapping[str, object], ...], list[int]]:
    if not payload or not payload.endswith(b"\n"):
        raise H6TerminalResultError("RunPod ledger is empty or has a partial record")
    expected_keys = {
        "schema",
        "index",
        "timestamp_utc",
        "event",
        "payload",
        "previous_sha256",
        "event_sha256",
    }
    previous: str | None = None
    events: list[Mapping[str, object]] = []
    cumulative_bytes: list[int] = []
    offset = 0
    for index, raw_line in enumerate(payload.splitlines(keepends=True), start=1):
        if not raw_line.endswith(b"\n"):
            raise H6TerminalResultError("RunPod ledger has a partial record")
        event = _parse_object(raw_line[:-1], f"RunPod ledger record {index}")
        if set(event) != expected_keys:
            raise H6TerminalResultError(
                f"RunPod ledger record {index} shape is invalid"
            )
        if event.get("schema") != _LEDGER_SCHEMA or event.get("index") != index:
            raise H6TerminalResultError(
                f"RunPod ledger record {index} schema or sequence is invalid"
            )
        if event.get("previous_sha256") != previous:
            raise H6TerminalResultError(f"RunPod ledger chain breaks at record {index}")
        base = {key: event[key] for key in expected_keys - {"event_sha256"}}
        observed = _sha256_bytes(_canonical_bytes(base))
        if event.get("event_sha256") != observed:
            raise H6TerminalResultError(
                f"RunPod ledger hash mismatch at record {index}"
            )
        if raw_line != _canonical_json_bytes(event):
            raise H6TerminalResultError(
                f"RunPod ledger record {index} is not canonical"
            )
        previous = observed
        events.append(event)
        offset += len(raw_line)
        cumulative_bytes.append(offset)
    if _sha256_bytes(payload) != expected_file_sha256:
        raise H6TerminalResultError("RunPod ledger file SHA-256 mismatch")
    if previous != expected_tail_sha256:
        raise H6TerminalResultError("RunPod ledger tail changed or was misbound")
    return tuple(events), cumulative_bytes


def _unique_event(
    events: Sequence[Mapping[str, object]], name: str
) -> Mapping[str, object]:
    matches = [event for event in events if event.get("event") == name]
    if len(matches) != 1:
        raise H6TerminalResultError(f"RunPod ledger must contain one {name} event")
    return matches[0]


def _verify_admission_sync_boundary(
    *,
    events: Sequence[Mapping[str, object]],
    receipt: Mapping[str, object],
    spec: Mapping[str, object],
    run_id: str,
    admission: Mapping[str, object],
    admission_payload: Mapping[str, object],
) -> AdmissionSyncBoundary:
    """Require the exact PREPARED mirror boundary consumed by admission."""

    prepared = _unique_event(events, _ADMISSION_SYNC_PREPARED_EVENT)
    prepared_index = prepared.get("index")
    admission_index = admission.get("index")
    if (
        isinstance(prepared_index, bool)
        or not isinstance(prepared_index, int)
        or isinstance(admission_index, bool)
        or not isinstance(admission_index, int)
        or prepared_index < 4
        or admission_index != prepared_index + 1
    ):
        raise H6TerminalResultError(
            "guard destination admission does not immediately follow PREPARED"
        )

    prepared_sha256 = _require_sha256(
        prepared.get("event_sha256"), "admission-sync PREPARED event SHA-256"
    )
    if admission.get("previous_sha256") != prepared_sha256:
        raise H6TerminalResultError(
            "guard destination admission is not chained from PREPARED"
        )

    payload = _mapping(prepared.get("payload"), "admission-sync PREPARED payload")
    if set(payload) != _ADMISSION_SYNC_PREPARED_KEYS:
        raise H6TerminalResultError("admission-sync PREPARED payload shape is invalid")

    transaction_id = _require_identifier(
        payload.get("transaction_id"), "admission-sync transaction ID"
    )
    ledger_tail_before = _require_sha256(
        payload.get("ledger_tail_before"), "admission-sync prior ledger tail"
    )
    volume_index = payload.get("final_network_volume_snapshot_event_index")
    destination_index = payload.get("final_destination_snapshot_event_index")
    if (
        isinstance(volume_index, bool)
        or not isinstance(volume_index, int)
        or isinstance(destination_index, bool)
        or not isinstance(destination_index, int)
        or volume_index != prepared_index - 2
        or destination_index != prepared_index - 1
    ):
        raise H6TerminalResultError(
            "admission-sync PREPARED snapshot positions are invalid"
        )
    volume_event = events[volume_index - 1]
    destination_event = events[destination_index - 1]
    volume_payload = _mapping(
        volume_event.get("payload"), "final network-volume snapshot payload"
    )
    destination_payload = _mapping(
        destination_event.get("payload"), "final destination snapshot payload"
    )
    volume_sha256 = _require_sha256(
        volume_event.get("event_sha256"), "final network-volume snapshot event"
    )
    destination_sha256 = _require_sha256(
        destination_event.get("event_sha256"), "final destination snapshot event"
    )
    if (
        volume_event.get("event") != "PROVIDER_SNAPSHOT"
        or destination_event.get("event") != "PROVIDER_SNAPSHOT"
        or volume_payload.get("resource_role") != "network_volume"
        or destination_payload.get("resource_role") != "destination"
        or volume_payload.get("observation_for_operation_id") is not None
        or destination_payload.get("observation_for_operation_id") is not None
        or payload.get("final_network_volume_snapshot_event_sha256") != volume_sha256
        or payload.get("final_destination_snapshot_event_sha256") != destination_sha256
        or ledger_tail_before != destination_sha256
        or prepared.get("previous_sha256") != destination_sha256
        or payload.get("network_volume_resource_id")
        != volume_payload.get("resource_id")
        or payload.get("destination_resource_id")
        != destination_payload.get("resource_id")
        or payload.get("destination_resource_id")
        != admission_payload.get("destination_resource_id")
    ):
        raise H6TerminalResultError(
            "admission-sync PREPARED final provider boundary is invalid"
        )

    relative_paths = _mapping(
        payload.get("relative_paths"), "admission-sync relative paths"
    )
    if set(relative_paths) != _ADMISSION_SYNC_PATH_ROLES:
        raise H6TerminalResultError("admission-sync relative-path inventory is invalid")
    normalized_paths = {
        role: _safe_relative(relative_paths[role], f"admission-sync {role} path")
        for role in _ADMISSION_SYNC_PATH_ROLES
    }
    if len(set(normalized_paths.values())) != len(normalized_paths):
        raise H6TerminalResultError("admission-sync relative paths are not unique")

    remote_root = payload.get("remote_root")
    if not isinstance(remote_root, str):
        raise H6TerminalResultError("admission-sync remote root is invalid")
    remote_path = PurePosixPath(remote_root)
    if (
        not remote_path.is_absolute()
        or remote_root in {"", "/"}
        or remote_root.startswith("//")
        or remote_path.as_posix() != remote_root
        or any(part in {"", ".", ".."} for part in remote_path.parts[1:])
    ):
        raise H6TerminalResultError("admission-sync remote root is invalid")

    authority_bindings = {
        "authorization_envelope_sha256": _require_sha256(
            spec.get("authorization_envelope_sha256"),
            "RunPod authorization envelope SHA-256",
        ),
        "upload_allowlist_sha256": _require_sha256(
            spec.get("upload_allowlist_sha256"),
            "RunPod upload allowlist SHA-256",
        ),
        "runtime_authority_sha256": _require_sha256(
            spec.get("runtime_authority_sha256"),
            "RunPod runtime authority SHA-256",
        ),
    }
    if (
        payload.get("schema") != _ADMISSION_SYNC_SCHEMA
        or payload.get("run_id") != run_id
        or payload.get("phase") != "REPORTS_FROZEN"
        or any(payload.get(key) != value for key, value in authority_bindings.items())
        or payload.get("transfer_protocol") != _ADMISSION_SYNC_PROTOCOL
        or payload.get("transfer_provider_state_mutated") is not False
        or payload.get("transfer_guard_state_mutated") is not False
    ):
        raise H6TerminalResultError(
            "admission-sync PREPARED authority binding is invalid"
        )

    expected_receipt_sync = {
        "schema": _ADMISSION_SYNC_SCHEMA,
        "transaction_id": transaction_id,
        "prepared_event_index": prepared_index,
        "prepared_event_sha256": prepared_sha256,
        "ledger_tail_before": ledger_tail_before,
        "remote_root": remote_root,
        "relative_paths": normalized_paths,
        "transfer_protocol": _ADMISSION_SYNC_PROTOCOL,
    }
    receipt_sync = _mapping(
        receipt.get("admission_sync"), "H6 admission receipt admission sync"
    )
    if dict(receipt_sync) != expected_receipt_sync:
        raise H6TerminalResultError(
            "H6 admission receipt admission sync differs from PREPARED"
        )

    return AdmissionSyncBoundary(
        transaction_id=transaction_id,
        prepared_event_index=prepared_index,
        prepared_event_sha256=prepared_sha256,
        receipt_relative_path=normalized_paths["admission_receipt"],
        receipt_echo=expected_receipt_sync,
        final_network_volume_snapshot_event_index=volume_index,
        final_network_volume_snapshot_event_sha256=volume_sha256,
        final_destination_snapshot_event_index=destination_index,
        final_destination_snapshot_event_sha256=destination_sha256,
    )


def _verify_admission_receipt_binding(
    *,
    admission_payload: Mapping[str, object],
    receipt: Mapping[str, object],
    receipt_file_identity: Mapping[str, object],
    spec_file_identity: Mapping[str, object],
    ledger_prefix_identity: Mapping[str, object],
    authorization_envelope_sha256: str,
    sync: AdmissionSyncBoundary,
) -> None:
    """Require A to durably preserve the exact authenticated receipt boundary."""

    binding = _mapping(
        admission_payload.get("admission_receipt_binding"),
        "guard destination admission receipt binding",
    )
    if (
        set(binding) != _ADMISSION_RECEIPT_BINDING_KEYS
        or binding.get("schema") != _ADMISSION_RECEIPT_BINDING_SCHEMA
        or binding.get("provider_state_mutated") is not False
        or binding.get("guard_state_mutated") is not False
    ):
        raise H6TerminalResultError(
            "guard destination admission receipt binding shape is invalid"
        )

    generated_at_utc = receipt.get("generated_at_utc")
    if not isinstance(generated_at_utc, str) or not generated_at_utc:
        raise H6TerminalResultError("H6 admission receipt generation time is invalid")
    receipt_binding = _mapping(
        binding.get("receipt"), "guard admission raw receipt identity"
    )
    expected_receipt_binding = {
        "sha256": receipt_file_identity["sha256"],
        "bytes": receipt_file_identity["bytes"],
        "relative_path": sync.receipt_relative_path,
        "schema_version": receipt.get("schema_version"),
        "status": receipt.get("status"),
        "generated_at_utc": generated_at_utc,
    }
    if (
        set(receipt_binding) != _ADMISSION_RECEIPT_IDENTITY_KEYS
        or dict(receipt_binding) != expected_receipt_binding
    ):
        raise H6TerminalResultError(
            "guard destination admission raw receipt identity is invalid"
        )

    receipt_run = _mapping(receipt.get("run"), "H6 admission receipt run")
    binding_run = _mapping(binding.get("run"), "guard admission bound run")
    receipt_ledger = _mapping(receipt_run.get("ledger"), "H6 admission receipt ledger")
    if (
        set(receipt_run) != _ADMISSION_RECEIPT_RUN_KEYS
        or set(binding_run) != _ADMISSION_RECEIPT_RUN_KEYS
        or dict(binding_run) != dict(receipt_run)
        or set(receipt_ledger) != _ADMISSION_RECEIPT_LEDGER_KEYS
        or dict(receipt_ledger) != dict(ledger_prefix_identity)
        or receipt_ledger.get("tail_sha256") != sync.prepared_event_sha256
        or receipt_ledger.get("event_count") != sync.prepared_event_index
        or receipt_run.get("destination_id")
        != admission_payload.get("destination_resource_id")
    ):
        raise H6TerminalResultError("guard destination admission bound run is invalid")

    for name in _ADMISSION_RECEIPT_BOUND_SLICES:
        receipt_slice = _mapping(receipt.get(name), f"H6 admission receipt {name}")
        bound_slice = _mapping(binding.get(name), f"guard admission bound {name}")
        if dict(bound_slice) != dict(receipt_slice):
            raise H6TerminalResultError(
                f"guard destination admission {name} differs from receipt"
            )

    receipt_programs = _mapping(
        receipt.get("admission_programs"), "H6 admission receipt programs"
    )
    bound_programs = _mapping(
        binding.get("admission_programs"), "guard admission bound programs"
    )
    bound_program_authority = _mapping(
        bound_programs.get("authority"), "guard admission program authority"
    )
    bound_receipt_programs = _mapping(
        bound_programs.get("receipt"), "guard admission receipt programs"
    )
    receipt_admission_program = _mapping(
        receipt_programs.get("destination_admission"),
        "H6 destination admission program identity",
    )
    receipt_terminal_wrapper = _mapping(
        receipt_programs.get("terminal_wrapper"),
        "H6 terminal wrapper identity",
    )
    expected_program_authority = {
        "continuation_authorization_sha256": authorization_envelope_sha256,
        "admission_program_sha256": receipt_admission_program.get("sha256"),
        "terminal_wrapper_sha256": receipt_terminal_wrapper.get("sha256"),
    }
    if (
        set(receipt_programs) != _ADMISSION_RECEIPT_PROGRAM_KEYS
        or set(bound_programs) != _ADMISSION_RECEIPT_PROGRAM_BINDING_KEYS
        or set(bound_program_authority) != _ADMISSION_RECEIPT_PROGRAM_AUTHORITY_KEYS
        or dict(bound_program_authority) != expected_program_authority
        or set(bound_receipt_programs) != _ADMISSION_RECEIPT_PROGRAM_KEYS
        or dict(bound_receipt_programs) != dict(receipt_programs)
    ):
        raise H6TerminalResultError(
            "guard destination admission program authority is invalid"
        )

    authority_identities = _mapping(
        receipt.get("authority_identities"),
        "H6 admission receipt authority identities",
    )
    run_spec_identity = _mapping(
        authority_identities.get("run_spec"), "H6 admission run spec identity"
    )
    run_events_identity = _mapping(
        authority_identities.get("run_events"), "H6 admission run events identity"
    )
    expected_run_events_identity = {
        "bytes": ledger_prefix_identity["bytes"],
        "sha256": ledger_prefix_identity["file_sha256"],
    }
    if (
        dict(run_spec_identity) != dict(spec_file_identity)
        or dict(run_events_identity) != expected_run_events_identity
    ):
        raise H6TerminalResultError(
            "H6 admission receipt authority file identities are invalid"
        )

    attestation = _mapping(
        receipt.get("destination_attestation"),
        "H6 admission receipt destination attestation",
    )
    attestation_identity = _mapping(
        attestation.get("identity"), "H6 destination attestation identity"
    )
    attestation_bytes = attestation_identity.get("bytes")
    destination_observation_sha256 = _require_sha256(
        admission_payload.get("destination_observation_sha256"),
        "guard destination observation SHA-256",
    )
    destination_observation_evidence_sha256 = _require_sha256(
        admission_payload.get("destination_observation_evidence_sha256"),
        "guard destination observation evidence SHA-256",
    )
    network_volume_observation_sha256 = _require_sha256(
        admission_payload.get("network_volume_observation_sha256"),
        "guard network-volume observation SHA-256",
    )
    network_volume_observation_evidence_sha256 = _require_sha256(
        admission_payload.get("network_volume_observation_evidence_sha256"),
        "guard network-volume observation evidence SHA-256",
    )
    expected_attestation_bindings = {
        "provider_snapshot_event_index": (sync.final_destination_snapshot_event_index),
        "provider_snapshot_event_sha256": (
            sync.final_destination_snapshot_event_sha256
        ),
        "destination_observation_sha256": destination_observation_sha256,
        "network_volume_provider_snapshot_event_index": (
            sync.final_network_volume_snapshot_event_index
        ),
        "network_volume_provider_snapshot_event_sha256": (
            sync.final_network_volume_snapshot_event_sha256
        ),
        "network_volume_observation_sha256": network_volume_observation_sha256,
        "network_volume_observation_evidence_sha256": (
            network_volume_observation_evidence_sha256
        ),
    }
    attested_destination = _mapping(
        attestation.get("destination"), "H6 attested destination identity"
    )
    if (
        set(attestation_identity) != {"bytes", "sha256"}
        or isinstance(attestation_bytes, bool)
        or not isinstance(attestation_bytes, int)
        or attestation_bytes <= 0
        or attestation_identity.get("sha256") != destination_observation_evidence_sha256
        or any(
            attestation.get(key) != value
            for key, value in expected_attestation_bindings.items()
        )
        or attested_destination.get("pod_id")
        != admission_payload.get("destination_resource_id")
        or attested_destination.get("network_volume_id")
        != admission_payload.get("network_volume_resource_id")
    ):
        raise H6TerminalResultError(
            "H6 admission receipt destination identity is invalid"
        )


def _verify_authority(
    bindings: AuthorityBindings,
    identity: FrozenIdentity,
    wrapper_identity: Mapping[str, object],
) -> LedgerAuthority:
    spec_payload, spec_file_identity = _read_stable_file(
        Path(bindings.run_spec), "current RunPod run spec"
    )
    if spec_file_identity["sha256"] != _require_sha256(
        bindings.run_spec_sha256, "current RunPod run spec SHA-256"
    ):
        raise H6TerminalResultError("current RunPod run spec SHA-256 mismatch")
    spec = _parse_object(spec_payload, "current RunPod run spec")
    run_id = _require_identifier(spec.get("run_id"), "RunPod run ID")
    if spec.get("schema") != "nano.runpod.spec.v3" or spec.get("project") != "nano-lm":
        raise H6TerminalResultError("current RunPod run spec identity is invalid")
    expected_spec = {
        "package_sha256": identity.package_sha256,
        "evaluator_sha256": identity.evaluator_sha256,
        "development_sha256": identity.development_sha256,
        "evaluation_authority_id": identity.evaluation_authority_id,
    }
    for field, expected in expected_spec.items():
        if spec.get(field) != expected:
            raise H6TerminalResultError(f"current RunPod run spec {field} mismatch")
    runtime_sha256 = _require_sha256(
        spec.get("runtime_authority_sha256"), "RunPod runtime authority SHA-256"
    )
    required_manifest_sha256 = _require_sha256(
        spec.get("required_artifacts_manifest_sha256"),
        "RunPod required-artifacts manifest SHA-256",
    )

    events_payload, events_file_identity = _read_stable_file(
        Path(bindings.run_events), "current RunPod run ledger"
    )
    expected_events_file_sha = _require_sha256(
        bindings.run_events_sha256, "current RunPod run ledger SHA-256"
    )
    expected_tail = _require_sha256(
        bindings.run_events_tail_sha256, "current RunPod ledger tail SHA-256"
    )
    events, cumulative_bytes = _verify_ledger_records(
        events_payload, expected_events_file_sha, expected_tail
    )
    first_payload = _mapping(events[0].get("payload"), "RunPod ledger genesis payload")
    if events[0].get("event") != "RUN_INITIALIZED" or first_payload.get("spec") != spec:
        raise H6TerminalResultError("run spec does not match RunPod ledger genesis")

    receipt_payload, receipt_file_identity = _read_stable_file(
        Path(bindings.admission_receipt), "H6 destination admission receipt"
    )
    expected_receipt_sha = _require_sha256(
        bindings.admission_receipt_sha256, "H6 admission receipt SHA-256"
    )
    if receipt_file_identity["sha256"] != expected_receipt_sha:
        raise H6TerminalResultError("H6 admission receipt SHA-256 mismatch")
    receipt = _parse_object(receipt_payload, "H6 destination admission receipt")
    if (
        set(receipt) != _ADMISSION_RECEIPT_KEYS
        or receipt.get("schema_version") != _ADMISSION_SCHEMA
        or receipt.get("status") != "ADMITTED_DEVELOPMENT_FREE"
        or receipt.get("provider_state_mutated") is not False
        or receipt.get("guard_state_mutated") is not False
    ):
        raise H6TerminalResultError("H6 destination admission receipt is invalid")
    _verify_receipt_slice_shapes(receipt)
    receipt_run = _mapping(receipt.get("run"), "H6 admission receipt run")
    receipt_ledger = _mapping(receipt_run.get("ledger"), "H6 admission receipt ledger")
    if receipt_run.get("run_id") != run_id:
        raise H6TerminalResultError("H6 admission receipt run ID mismatch")
    authority_hashes = _mapping(
        receipt.get("authority_hashes"), "H6 admission receipt authority hashes"
    )
    expected_authority_hashes = {
        key: value
        for key, value in sorted(spec.items())
        if isinstance(key, str) and key.endswith("_sha256")
    }
    if dict(authority_hashes) != expected_authority_hashes:
        raise H6TerminalResultError("H6 admission receipt authority hashes mismatch")
    programs = _mapping(
        receipt.get("admission_programs"), "H6 admission receipt programs"
    )
    receipt_wrapper = _mapping(
        programs.get("terminal_wrapper"), "H6 admission terminal wrapper identity"
    )
    receipt_admission = _mapping(
        programs.get("destination_admission"),
        "H6 destination admission program identity",
    )
    if dict(receipt_wrapper) != dict(wrapper_identity):
        raise H6TerminalResultError("H6 admission receipt wrapper identity mismatch")
    if dict(receipt_admission) != _DESTINATION_ADMISSION_PROGRAM_IDENTITY:
        raise H6TerminalResultError(
            "H6 destination admission program identity mismatch"
        )

    admission = _unique_event(events, "DESTINATION_ADMITTED")
    admission_sha = _require_sha256(
        bindings.admission_event_sha256, "guard admission event SHA-256"
    )
    if admission.get("event_sha256") != admission_sha:
        raise H6TerminalResultError("guard admission event SHA-256 mismatch")
    admission_payload = _mapping(
        admission.get("payload"), "guard destination admission payload"
    )
    sync = _verify_admission_sync_boundary(
        events=events,
        receipt=receipt,
        spec=spec,
        run_id=run_id,
        admission=admission,
        admission_payload=admission_payload,
    )
    if (
        admission_payload.get("schema") != _GUARD_ADMISSION_SCHEMA
        or admission_payload.get("run_id") != run_id
        or admission_payload.get("admission_evidence_sha256") != expected_receipt_sha
        or admission_payload.get("evaluation_authority_id")
        != identity.evaluation_authority_id
        or admission_payload.get("package_sha256") != identity.package_sha256
        or admission_payload.get("evaluator_sha256") != identity.evaluator_sha256
        or admission_payload.get("runtime_authority_sha256") != runtime_sha256
        or admission_payload.get("required_artifacts_manifest_sha256")
        != required_manifest_sha256
        or admission_payload.get("admission_sync_transaction_id") != sync.transaction_id
        or admission_payload.get("admission_sync_prepared_event_index")
        != sync.prepared_event_index
        or admission_payload.get("admission_sync_prepared_event_sha256")
        != sync.prepared_event_sha256
    ):
        raise H6TerminalResultError("guard destination admission binding is invalid")
    admission_index = admission.get("index")
    if isinstance(admission_index, bool) or not isinstance(admission_index, int):
        raise H6TerminalResultError("guard destination admission index is invalid")
    if admission_index < 2:
        raise H6TerminalResultError("guard destination admission has no predecessor")
    prefix_bytes = cumulative_bytes[admission_index - 2]
    ledger_prefix_identity = {
        "event_count": admission_index - 1,
        "tail_sha256": sync.prepared_event_sha256,
        "file_sha256": _sha256_bytes(events_payload[:prefix_bytes]),
        "bytes": prefix_bytes,
    }
    receipt_event_count = receipt_ledger.get("event_count")
    if (
        receipt_event_count != admission_index - 1
        or receipt_ledger.get("bytes") != prefix_bytes
        or receipt_ledger.get("tail_sha256") != admission.get("previous_sha256")
        or receipt_ledger.get("tail_sha256")
        != admission_payload.get("ledger_tail_before")
        or receipt_ledger.get("file_sha256")
        != _sha256_bytes(events_payload[:prefix_bytes])
    ):
        raise H6TerminalResultError("H6 admission receipt ledger prefix is invalid")

    _verify_admission_receipt_binding(
        admission_payload=admission_payload,
        receipt=receipt,
        receipt_file_identity=receipt_file_identity,
        spec_file_identity=spec_file_identity,
        ledger_prefix_identity=ledger_prefix_identity,
        authorization_envelope_sha256=_require_sha256(
            spec.get("authorization_envelope_sha256"),
            "RunPod authorization envelope SHA-256",
        ),
        sync=sync,
    )

    release = _unique_event(events, "DEV_RELEASE_MARKED")
    release_payload = _mapping(release.get("payload"), "development release payload")
    destination_id = admission_payload.get("destination_resource_id")
    if (
        not isinstance(destination_id, str)
        or receipt_run.get("destination_id") != destination_id
        or release.get("index", 0) <= admission_index
        or release_payload.get("admission_sha256") != admission_sha
        or release_payload.get("dev_sha256") != identity.development_sha256
        or release_payload.get("package_sha256") != identity.package_sha256
        or release_payload.get("evaluator_sha256") != identity.evaluator_sha256
        or release_payload.get("runtime_sha256") != runtime_sha256
        or release_payload.get("destination_resource_id") != destination_id
    ):
        raise H6TerminalResultError("development release binding is invalid")

    open_event = _unique_event(events, "DEV_OPEN_MARKED")
    open_payload = _mapping(open_event.get("payload"), "development open payload")
    operation_id = _require_identifier(
        open_payload.get("evaluation_operation_id"), "evaluation operation ID"
    )
    launch_events = [
        event
        for event in events
        if event.get("event") == "OPERATION_STARTED"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("kind") == "launch_evaluation"
        and event["payload"].get("operation_id") == operation_id
    ]
    if len(launch_events) != 1:
        raise H6TerminalResultError(
            "development open must bind one launch-evaluation operation"
        )
    launch_event = launch_events[0]
    launch_payload = _mapping(
        launch_event.get("payload"), "launch-evaluation operation payload"
    )
    launch_index = launch_event.get("index", 0)
    unfinished = not any(
        event.get("event") == "OPERATION_FINISHED"
        and isinstance(event.get("payload"), dict)
        and event["payload"].get("operation_id") == operation_id
        for event in events
    )
    if (
        events[-1] is not open_event
        or open_event.get("event_sha256") != expected_tail
        or not isinstance(launch_index, int)
        or isinstance(launch_index, bool)
        or launch_index <= release.get("index", 0)
        or open_event.get("index", 0) <= launch_index
        or launch_payload.get("resource_role") != "destination"
        or launch_payload.get("target_id") != destination_id
        or not unfinished
        or open_payload.get("runtime_sha256") != runtime_sha256
        or open_payload.get("evaluator_sha256") != identity.evaluator_sha256
        or open_payload.get("package_sha256") != identity.package_sha256
        or open_payload.get("destination_resource_id") != destination_id
    ):
        raise H6TerminalResultError(
            "current RunPod ledger must end at the bound development-open event"
        )
    return LedgerAuthority(
        spec=spec,
        events=events,
        file_payload=events_payload,
        file_sha256=events_file_identity["sha256"],
        tail_sha256=expected_tail,
        admission_event=admission,
        release_event=release,
        open_event=open_event,
        receipt=receipt,
        receipt_sha256=expected_receipt_sha,
    )


def _read_allowlisted_file(
    package_root: Path, source_relative: str, *, maximum_bytes: int
) -> bytes | None:
    source = package_root / source_relative
    if not _lexists(source):
        return None
    payload, _identity = _read_stable_file(
        source,
        f"allowlisted evaluator output {source_relative}",
        root=package_root,
        maximum_bytes=maximum_bytes,
    )
    return payload


def _assert_safe_inventory(terminal_dir: Path, expected: set[str]) -> None:
    observed: set[str] = set()
    for path in terminal_dir.rglob("*"):
        relative = path.relative_to(terminal_dir).as_posix()
        if path.is_symlink():
            raise H6TerminalResultError(
                f"terminal evidence contains a symbolic link: {relative}"
            )
        if path.is_dir():
            continue
        _require_real_regular_file(path, f"terminal evidence member {relative}")
        observed.add(relative)
    if observed != expected:
        raise H6TerminalResultError(
            "terminal evidence inventory is not the exact result allowlist"
        )
    for relative in observed:
        folded = relative.casefold()
        if any(marker in folded for marker in _FORBIDDEN_MEMBER_MARKERS):
            raise H6TerminalResultError(
                f"forbidden path would enter terminal bundle: {relative}"
            )


def _normalize_evidence_metadata(terminal_dir: Path) -> None:
    paths = sorted(
        terminal_dir.rglob("*"),
        key=lambda path: len(path.relative_to(terminal_dir).parts),
        reverse=True,
    )
    for path in paths:
        if path.is_symlink():
            raise H6TerminalResultError(
                "terminal evidence metadata normalization encountered a symlink"
            )
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
        os.utime(path, ns=(0, 0), follow_symlinks=False)
    os.chmod(terminal_dir, 0o755)
    os.utime(terminal_dir, ns=(0, 0), follow_symlinks=False)


class _BytesReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _write_deterministic_archive(
    terminal_dir: Path, terminal_archive: Path, expected: set[str]
) -> str:
    temporary = terminal_archive.with_name(f".{terminal_archive.name}.tmp")
    if _lexists(temporary):
        raise H6TerminalResultError(
            f"terminal archive temporary path already exists: {temporary}"
        )
    _assert_safe_inventory(terminal_dir, expected)
    try:
        with temporary.open("xb") as raw:
            with (
                gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0
                ) as compressed,
                tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
                ) as archive,
            ):
                for relative in sorted(expected):
                    payload = (terminal_dir / relative).read_bytes()
                    info = tarfile.TarInfo(str(PurePosixPath(ARCHIVE_ROOT) / relative))
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    archive.addfile(info, fileobj=_BytesReader(payload))
            raw.flush()
            os.fsync(raw.fileno())
        os.chmod(temporary, 0o644)
        try:
            os.link(temporary, terminal_archive)
        except OSError as exc:
            raise H6TerminalResultError(
                f"refusing to clobber terminal archive {terminal_archive}"
            ) from exc
        os.utime(terminal_archive, ns=(0, 0), follow_symlinks=False)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return _sha256_file(terminal_archive)


def _normalized_exit_code(returncode: int) -> int:
    return 128 + abs(returncode) if returncode < 0 else returncode


def _controlled_environment() -> dict[str, str]:
    return {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LD_LIBRARY_PATH": (
            "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
        ),
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
    }


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _real_runtime_directory(path: Path, label: str) -> Path:
    absolute = _absolute_path(path)
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise H6TerminalResultError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise H6TerminalResultError(f"{label} must be a real directory")
    if resolved != absolute:
        raise H6TerminalResultError(f"{label} has a symlinked ancestor")
    return resolved


def _isolated_interpreter(package: Path) -> tuple[Path, Path]:
    environment = _real_runtime_directory(
        package / ".h6-venv", "H6 isolated environment"
    )
    _real_runtime_directory(environment / "bin", "H6 isolated executable directory")
    interpreter = package / _ISOLATED_INTERPRETER_RELATIVE
    try:
        lexical_metadata = interpreter.lstat()
        target = interpreter.resolve(strict=True)
        target_metadata = target.stat()
    except OSError as exc:
        raise H6TerminalResultError(
            "H6 isolated Python interpreter is unavailable"
        ) from exc
    if (
        not (
            stat.S_ISREG(lexical_metadata.st_mode)
            or stat.S_ISLNK(lexical_metadata.st_mode)
        )
        or not stat.S_ISREG(target_metadata.st_mode)
        or not os.access(interpreter, os.X_OK)
    ):
        raise H6TerminalResultError(
            "H6 isolated Python interpreter is not executable file material"
        )
    return interpreter, target


def _run_isolated_capture(
    argv: Sequence[str], *, cwd: Path, label: str
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            tuple(argv),
            cwd=cwd,
            env=_controlled_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise H6TerminalResultError(f"{label} could not run") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="backslashreplace").strip()
        suffix = f": {detail}" if detail else ""
        raise H6TerminalResultError(f"{label} failed{suffix}")
    if completed.stderr:
        raise H6TerminalResultError(f"{label} emitted unexpected diagnostics")
    return completed


def _probe_isolated_sysconfig(interpreter: Path, package: Path) -> Mapping[str, object]:
    completed = _run_isolated_capture(
        (
            os.fspath(interpreter),
            "-I",
            "-S",
            "-c",
            _ISOLATED_SYSCONFIG_PROBE,
        ),
        cwd=package,
        label="H6 isolated sysconfig probe",
    )
    probe = _parse_object(completed.stdout, "H6 isolated sysconfig probe")
    if (
        set(probe)
        != {
            "schema",
            "executable",
            "flags",
            "version",
            "prefixes",
            "paths",
            "sys_path",
        }
        or probe.get("schema") != "nano.h6-isolated-sysconfig-probe.v1"
    ):
        raise H6TerminalResultError("H6 isolated sysconfig probe shape is invalid")
    flags = _mapping(probe.get("flags"), "H6 isolated sysconfig flags")
    prefixes = _mapping(probe.get("prefixes"), "H6 isolated sysconfig prefixes")
    paths = _mapping(probe.get("paths"), "H6 isolated sysconfig paths")
    version = probe.get("version")
    sys_path = probe.get("sys_path")
    if (
        set(flags) != {"isolated", "no_site"}
        or flags.get("isolated") != 1
        or flags.get("no_site") != 1
        or probe.get("executable") != os.fspath(interpreter)
        or set(prefixes) != {"prefix", "exec_prefix", "base_prefix", "base_exec_prefix"}
        or set(paths) != {"purelib", "platlib", "data"}
        or not isinstance(version, list)
        or len(version) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in version
        )
        or not isinstance(sys_path, list)
        or not sys_path
        or any(not isinstance(value, str) or not value for value in sys_path)
    ):
        raise H6TerminalResultError("H6 isolated sysconfig probe is invalid")
    for name, value in prefixes.items():
        if not isinstance(value, str) or not value:
            raise H6TerminalResultError(
                f"H6 isolated sysconfig {name} prefix is invalid"
            )
    for name, value in paths.items():
        if not isinstance(value, str) or not value:
            raise H6TerminalResultError(f"H6 isolated sysconfig {name} path is invalid")
    return probe


def _reject_dependency_nano_shadow(dependency_root: Path) -> None:
    try:
        names = tuple(path.name.casefold() for path in dependency_root.iterdir())
    except OSError as exc:
        raise H6TerminalResultError(
            f"H6 dependency root is unreadable: {dependency_root}"
        ) from exc
    if any(name == "nano_ai" or name.startswith("nano_ai.") for name in names):
        raise H6TerminalResultError(
            f"H6 dependency root contains a top-level Nano shadow: {dependency_root}"
        )


def _runtime_path_label(path: Path, package: Path) -> str:
    if _is_relative_to(path, package):
        return path.relative_to(package).as_posix()
    return os.fspath(path)


def _prepare_isolated_runtime(package: Path) -> IsolatedRuntime:
    """Audit the exact no-site runtime without importing Nano in the child."""

    interpreter, interpreter_target = _isolated_interpreter(package)
    probe = _probe_isolated_sysconfig(interpreter, package)
    version = probe["version"]
    if not isinstance(version, list):  # narrowed by the probe validator
        raise H6TerminalResultError("H6 isolated Python version is invalid")
    environment_packages = _real_runtime_directory(
        package
        / ".h6-venv"
        / "lib"
        / f"python{version[0]}.{version[1]}"
        / "site-packages",
        "H6 isolated site-packages",
    )
    if not _is_relative_to(environment_packages, package / ".h6-venv"):
        raise H6TerminalResultError("H6 isolated site-packages escapes its environment")

    prefixes = _mapping(probe["prefixes"], "H6 isolated sysconfig prefixes")
    paths = _mapping(probe["paths"], "H6 isolated sysconfig paths")
    base_prefixes: list[Path] = []
    for name in ("base_prefix", "base_exec_prefix"):
        prefix = _real_runtime_directory(Path(str(prefixes[name])), f"H6 Python {name}")
        if prefix not in base_prefixes:
            base_prefixes.append(prefix)
    data_prefix = _real_runtime_directory(
        Path(str(paths["data"])), "H6 Python sysconfig data prefix"
    )
    if data_prefix not in base_prefixes:
        base_prefixes.append(data_prefix)

    dependency_roots = [environment_packages]
    for name in ("purelib", "platlib"):
        candidate = _real_runtime_directory(
            Path(str(paths[name])), f"H6 Python {name} dependency root"
        )
        if not any(_is_relative_to(candidate, prefix) for prefix in base_prefixes):
            raise H6TerminalResultError(
                f"H6 Python {name} dependency root escapes probed base prefixes"
            )
        if candidate not in dependency_roots:
            dependency_roots.append(candidate)
    for dependency_root in dependency_roots:
        _reject_dependency_nano_shadow(dependency_root)

    roots_tuple = tuple(dependency_roots)
    roots_json = _canonical_bytes([os.fspath(path) for path in roots_tuple]).decode(
        "utf-8"
    )
    audit = _run_isolated_capture(
        (
            os.fspath(interpreter),
            "-I",
            "-S",
            "-c",
            _ISOLATED_MODULE_BRIDGE,
            "audit",
            os.fspath(package),
            roots_json,
            os.fspath(interpreter),
        ),
        cwd=package,
        label="H6 isolated dependency audit",
    )
    audit_result = _parse_object(audit.stdout, "H6 isolated dependency audit")
    audit_flags = _mapping(
        audit_result.get("flags"), "H6 isolated dependency audit flags"
    )
    imports = _mapping(audit_result.get("imports"), "H6 isolated dependency imports")
    if (
        set(audit_result)
        != {
            "schema",
            "flags",
            "imports",
            "source_precedence",
            "startup_hooks_loaded",
        }
        or audit_result.get("schema") != "nano.h6-isolated-module-audit.v1"
        or dict(audit_flags) != {"isolated": 1, "no_site": 1}
        or set(imports) != {"torch", "tokenizers"}
        or audit_result.get("source_precedence") is not True
        or audit_result.get("startup_hooks_loaded") is not False
    ):
        raise H6TerminalResultError("H6 isolated dependency audit is invalid")

    normalized_imports: dict[str, object] = {}
    for name in ("torch", "tokenizers"):
        record = _mapping(imports.get(name), f"H6 isolated {name} import")
        if set(record) != {"origin", "dependency_root"}:
            raise H6TerminalResultError(
                f"H6 isolated {name} import identity is invalid"
            )
        try:
            origin = Path(str(record["origin"])).resolve(strict=True)
            dependency_root = Path(str(record["dependency_root"])).resolve(strict=True)
        except OSError as exc:
            raise H6TerminalResultError(
                f"H6 isolated {name} import origin is unavailable"
            ) from exc
        if dependency_root not in roots_tuple or not _is_relative_to(
            origin, dependency_root
        ):
            raise H6TerminalResultError(
                f"H6 isolated {name} import escaped accepted dependency roots"
            )
        normalized_imports[name] = {
            "origin": _runtime_path_label(origin, package),
            "dependency_root": _runtime_path_label(dependency_root, package),
        }

    evidence = {
        "schema": _ISOLATED_RUNTIME_SCHEMA,
        "contract": isolated_execution_contract(),
        "interpreter": {
            "relative_path": _ISOLATED_INTERPRETER_RELATIVE,
            "resolved_path": os.fspath(interpreter_target),
            "version": list(version),
        },
        "accepted_dependency_roots": [
            {
                "kind": "environment" if path == environment_packages else "base",
                "path": _runtime_path_label(path, package),
            }
            for path in roots_tuple
        ],
        "authenticated_import_origins": normalized_imports,
        "bridge_sha256": _sha256_bytes(_ISOLATED_MODULE_BRIDGE.encode("utf-8")),
        "launcher_bridge_sha256": _sha256_bytes(
            _ISOLATED_LAUNCHER_BRIDGE.encode("utf-8")
        ),
        "sysconfig_probe_sha256": _sha256_bytes(
            _ISOLATED_SYSCONFIG_PROBE.encode("utf-8")
        ),
    }
    return IsolatedRuntime(
        interpreter=interpreter,
        dependency_roots=roots_tuple,
        dependency_roots_json=roots_json,
        evidence=evidence,
    )


def _authority_status(
    authority: LedgerAuthority,
    wrapper_identity: Mapping[str, object],
    package_snapshot: Mapping[str, object],
    recovered_snapshot: Mapping[str, object],
    development_snapshot: Mapping[str, object],
    identity: FrozenIdentity,
    isolated_runtime: Mapping[str, object],
) -> dict[str, object]:
    return {
        "admission": {
            "event_sha256": authority.admission_event["event_sha256"],
            "receipt_sha256": authority.receipt_sha256,
        },
        "development_open_event_sha256": authority.open_event["event_sha256"],
        "development_release_event_sha256": authority.release_event["event_sha256"],
        "package": {
            "id": identity.package_id,
            "archive_sha256": identity.package_sha256,
            "materialized_snapshot_sha256": package_snapshot["snapshot_sha256"],
            "manifest_sha256": identity.bundle_manifest_sha256,
        },
        "recovered_artifacts_snapshot_sha256": recovered_snapshot["snapshot_sha256"],
        "development_inputs_snapshot_sha256": development_snapshot["snapshot_sha256"],
        "run": {
            "id": authority.spec["run_id"],
            "events_sha256": authority.file_sha256,
            "ledger_tail_sha256": authority.tail_sha256,
        },
        "isolated_runtime": dict(isolated_runtime),
        "wrapper": {"id": WRAPPER_ID, **dict(wrapper_identity)},
    }


def _bundle_policy() -> dict[str, object]:
    return {
        "allowlisted_generated_result_paths": [
            row[1] for row in _GENERATED_RESULT_ALLOWLIST
        ],
        "development_inputs_included": False,
        "fresh_or_private_data_included": False,
        "training_inputs_included": False,
    }


def _append_failure_log(path: Path, reason: str) -> None:
    metadata = _require_real_regular_file(path, "captured evaluator log")
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_dev != metadata.st_dev
            or observed.st_ino != metadata.st_ino
        ):
            raise H6TerminalResultError("captured evaluator log identity changed")
        payload = f"\nH6 terminal wrapper failure after invocation: {reason}\n".encode(
            "utf-8", errors="backslashreplace"
        )
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _failure_bundle(
    *,
    final_dir: Path,
    final_archive: Path,
    captured_log: Path,
    reason: str,
    evaluator_returncode: int | None,
    authority_bindings: Mapping[str, object],
    created_results: Sequence[Path],
) -> TerminalRunOutcome:
    # Remove only files this wrapper created from already-authenticated result
    # bytes.  A post-marker failure bundle deliberately contains status/log only.
    for path in reversed(tuple(created_results)):
        if path.exists() and not path.is_symlink():
            path.unlink()
    generated_dir = final_dir / "generated-results"
    if generated_dir.exists() and not generated_dir.is_symlink():
        try:
            generated_dir.rmdir()
        except OSError:
            pass
    _append_failure_log(captured_log, reason)
    captured_payload, _identity = _read_stable_file(
        captured_log, "captured evaluator log", root=final_dir
    )
    status = {
        "schema": SCHEMA_VERSION,
        "status": "wrapper_failed_after_invocation",
        "bindings": dict(authority_bindings),
        "evaluator_outcome": {
            "exit_code": WRAPPER_FAILURE_EXIT_CODE,
            "invocation_count": 1,
            "returncode": evaluator_returncode,
            "stdout_stderr_path": "evaluator.log",
        },
        "wrapper_failure": {
            "exit_code": WRAPPER_FAILURE_EXIT_CODE,
            "reason": reason,
            "retry_permitted": False,
        },
        "artifacts": [
            _artifact_row("evaluator.log", captured_payload, "captured_stdout_stderr")
        ],
        "bundle_policy": _bundle_policy(),
    }
    _write_exclusive(final_dir / "terminal-status.json", _canonical_json_bytes(status))
    expected = {"evaluator.log", "terminal-status.json"}
    _normalize_evidence_metadata(final_dir)
    archive_sha256 = _write_deterministic_archive(final_dir, final_archive, expected)
    return TerminalRunOutcome(
        evaluator_returncode=evaluator_returncode,
        exit_code=WRAPPER_FAILURE_EXIT_CODE,
        archive_sha256=archive_sha256,
        terminal_dir=final_dir,
        terminal_archive=final_archive,
        status="wrapper_failed_after_invocation",
    )


def run_terminal_evaluation(
    *,
    package_root: str | Path,
    terminal_dir: str | Path,
    terminal_archive: str | Path,
    bindings: AuthorityBindings,
    _identity: FrozenIdentity = _PRODUCTION_IDENTITY,
) -> TerminalRunOutcome:
    """Run the exact frozen evaluator once and build result-only evidence."""

    identity = _identity
    package = _require_real_directory(Path(package_root), "H6 package root")
    results_dir = _require_real_directory(package / "results", "H6 results directory")
    if results_dir.parent != package:
        raise H6TerminalResultError("H6 results directory escapes package root")

    wrapper_path = Path(__file__).resolve()
    wrapper_payload, wrapper_file_identity = _read_stable_file(
        wrapper_path, "H6 terminal wrapper"
    )
    wrapper_sha256 = _require_sha256(bindings.wrapper_sha256, "wrapper SHA-256")
    if wrapper_file_identity["sha256"] != wrapper_sha256:
        raise H6TerminalResultError("wrapper SHA-256 does not match executing code")
    wrapper_identity = {
        "bytes": len(wrapper_payload),
        "sha256": wrapper_sha256,
    }

    evaluation_script = package / EVALUATION_SCRIPT_NAME
    _require_real_regular_file(evaluation_script, "frozen H6 evaluation script")
    if not os.access(evaluation_script, os.X_OK):
        raise H6TerminalResultError("frozen H6 evaluation script is not executable")
    evaluator_source = package / EVALUATOR_SOURCE_RELATIVE
    _require_real_regular_file(evaluator_source, "frozen H6 evaluator source")

    package_snapshot = _verify_package_tree(package, identity)
    recovered_snapshot = _verify_file_set(
        package, identity.recovered_files, "recovered H6 artifact"
    )
    development_snapshot = _verify_file_set(
        package, identity.development_files, "released development input"
    )
    _assert_prohibited_paths_absent(package)
    if _sha256_file(evaluation_script) != identity.evaluation_script_sha256:
        raise H6TerminalResultError("evaluation script SHA-256 mismatch")
    if _sha256_file(evaluator_source) != identity.evaluator_sha256:
        raise H6TerminalResultError("evaluator source SHA-256 mismatch")
    authority = _verify_authority(bindings, identity, wrapper_identity)
    isolated_runtime = _prepare_isolated_runtime(package)
    authority_bindings = _authority_status(
        authority,
        wrapper_identity,
        package_snapshot,
        recovered_snapshot,
        development_snapshot,
        identity,
        isolated_runtime.evidence,
    )

    final_dir = _resolve_new_path(Path(terminal_dir), "terminal evidence directory")
    final_archive = _resolve_new_path(
        Path(terminal_archive), "terminal evidence archive"
    )
    if final_archive.suffixes[-2:] != [".tar", ".gz"]:
        raise H6TerminalResultError("terminal evidence archive must end in .tar.gz")
    if final_dir == final_archive:
        raise H6TerminalResultError("terminal evidence paths must be distinct")
    if _is_relative_to(final_dir, package) or _is_relative_to(final_archive, package):
        raise H6TerminalResultError(
            "terminal evidence outputs must be outside the H6 package root"
        )
    if _is_relative_to(final_archive, final_dir):
        raise H6TerminalResultError(
            "terminal evidence archive must be outside its evidence directory"
        )
    archive_temporary = final_archive.with_name(f".{final_archive.name}.tmp")
    if _lexists(archive_temporary):
        raise H6TerminalResultError(
            f"terminal archive temporary path already exists: {archive_temporary}"
        )

    invocation_marker = package / INVOCATION_MARKER_NAME
    if _lexists(invocation_marker):
        raise H6TerminalResultError(
            "fixed H6 invocation marker already exists; development evaluation cannot rerun"
        )
    for (
        source_relative,
        _bundle_relative,
        _maximum_bytes,
    ) in _GENERATED_RESULT_ALLOWLIST:
        if _lexists(package / source_relative):
            raise H6TerminalResultError(
                f"refusing to clobber evaluator output {source_relative}"
            )
    for relative in _EVALUATOR_EXCLUDED_FINAL_PATHS:
        if _lexists(package / relative):
            raise H6TerminalResultError(
                f"refusing to clobber evaluator output {relative}"
            )
    for relative in _EVALUATOR_TEMPORARY_PATHS:
        if _lexists(package / relative):
            raise H6TerminalResultError(
                f"refusing to clobber evaluator temporary output {relative}"
            )

    try:
        final_dir.mkdir(mode=0o700)
    except OSError as exc:
        raise H6TerminalResultError(
            f"could not reserve terminal evidence directory {final_dir}"
        ) from exc
    captured_log = final_dir / "evaluator.log"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        log_descriptor = os.open(captured_log, flags, 0o600)
    except OSError as exc:
        raise H6TerminalResultError("could not reserve captured evaluator log") from exc

    marker = {
        "schema": "nano.h6-terminal-invocation-marker.v1",
        "wrapper": {"id": WRAPPER_ID, **wrapper_identity},
        "package": {
            "id": identity.package_id,
            "archive_sha256": identity.package_sha256,
            "manifest_sha256": identity.bundle_manifest_sha256,
            "materialized_snapshot_sha256": package_snapshot["snapshot_sha256"],
        },
        "recovered_artifacts_snapshot_sha256": recovered_snapshot["snapshot_sha256"],
        "admission_receipt_sha256": authority.receipt_sha256,
        "admission_event_sha256": authority.admission_event["event_sha256"],
        "run_id": authority.spec["run_id"],
        "run_events_sha256": authority.file_sha256,
        "run_events_tail_sha256": authority.tail_sha256,
        "development_open_event_sha256": authority.open_event["event_sha256"],
        "isolated_runtime": dict(isolated_runtime.evidence),
        "invocation_count": 1,
        "retry_permitted": False,
    }
    try:
        _write_exclusive(invocation_marker, _canonical_json_bytes(marker), mode=0o400)
        _fsync_directory(package)
    except BaseException:
        os.close(log_descriptor)
        raise

    evaluator_returncode: int | None = None
    created_results: list[Path] = []
    try:
        with os.fdopen(log_descriptor, "wb") as combined_output:
            completed = subprocess.run(
                (
                    "/bin/bash",
                    "--noprofile",
                    "--norc",
                    "-c",
                    _ISOLATED_LAUNCHER_BRIDGE,
                    str(evaluation_script),
                    os.fspath(isolated_runtime.interpreter),
                    os.fspath(package),
                    isolated_runtime.dependency_roots_json,
                    _ISOLATED_MODULE_BRIDGE,
                ),
                cwd=package,
                env=_controlled_environment(),
                stdin=subprocess.DEVNULL,
                stdout=combined_output,
                stderr=subprocess.STDOUT,
                check=False,
            )
            combined_output.flush()
            os.fsync(combined_output.fileno())
        evaluator_returncode = completed.returncode

        # Everything capable of changing the scientific identity is rehashed
        # immediately after the launcher returns and before any result is copied.
        if _verify_package_tree(package, identity) != package_snapshot:
            raise H6TerminalResultError("frozen package changed during evaluation")
        if (
            _verify_file_set(package, identity.recovered_files, "recovered H6 artifact")
            != recovered_snapshot
        ):
            raise H6TerminalResultError(
                "recovered H6 artifacts changed during evaluation"
            )
        if (
            _verify_file_set(
                package, identity.development_files, "released development input"
            )
            != development_snapshot
        ):
            raise H6TerminalResultError("development inputs changed during evaluation")
        _assert_prohibited_paths_absent(package)
        post_authority = _verify_authority(bindings, identity, wrapper_identity)
        if (
            post_authority.file_sha256 != authority.file_sha256
            or post_authority.tail_sha256 != authority.tail_sha256
            or post_authority.receipt_sha256 != authority.receipt_sha256
        ):
            raise H6TerminalResultError(
                "H6 control authority changed during evaluation"
            )
        post_runtime = _prepare_isolated_runtime(package)
        if (
            dict(post_runtime.evidence) != dict(isolated_runtime.evidence)
            or post_runtime.interpreter != isolated_runtime.interpreter
            or post_runtime.dependency_roots != isolated_runtime.dependency_roots
        ):
            raise H6TerminalResultError("H6 isolated runtime changed during evaluation")

        copied_results: list[tuple[str, str, bytes]] = []
        for (
            source_relative,
            bundle_relative,
            maximum_bytes,
        ) in _GENERATED_RESULT_ALLOWLIST:
            payload = _read_allowlisted_file(
                package, source_relative, maximum_bytes=maximum_bytes
            )
            if payload is not None:
                copied_results.append((source_relative, bundle_relative, payload))

        captured_payload, _captured_identity = _read_stable_file(
            captured_log, "captured evaluator log", root=final_dir
        )
        artifact_rows: list[dict[str, object]] = [
            _artifact_row("evaluator.log", captured_payload, "captured_stdout_stderr")
        ]
        expected_inventory = {"evaluator.log", "terminal-status.json"}
        if copied_results:
            generated_dir = final_dir / "generated-results"
            generated_dir.mkdir(mode=0o700)
            for source_relative, bundle_relative, payload in copied_results:
                destination = final_dir / bundle_relative
                created_results.append(destination)
                _write_exclusive(destination, payload)
                expected_inventory.add(bundle_relative)
                row = _artifact_row(
                    bundle_relative, payload, "allowlisted_generated_result"
                )
                row["source_path"] = source_relative
                artifact_rows.append(row)

        exit_code = _normalized_exit_code(evaluator_returncode)
        status_name = (
            "evaluator_succeeded" if evaluator_returncode == 0 else "evaluator_failed"
        )
        status = {
            "schema": SCHEMA_VERSION,
            "status": status_name,
            "bindings": authority_bindings,
            "evaluator_outcome": {
                "exit_code": exit_code,
                "invocation_count": 1,
                "returncode": evaluator_returncode,
                "stdout_stderr_path": "evaluator.log",
            },
            "artifacts": sorted(artifact_rows, key=lambda row: str(row["path"])),
            "bundle_policy": _bundle_policy(),
        }
        _write_exclusive(
            final_dir / "terminal-status.json", _canonical_json_bytes(status)
        )
        _normalize_evidence_metadata(final_dir)
        archive_sha256 = _write_deterministic_archive(
            final_dir, final_archive, expected_inventory
        )
        return TerminalRunOutcome(
            evaluator_returncode=evaluator_returncode,
            exit_code=exit_code,
            archive_sha256=archive_sha256,
            terminal_dir=final_dir,
            terminal_archive=final_archive,
            status=status_name,
        )
    except BaseException as exc:  # noqa: BLE001 - marker makes every exit terminal
        reason = str(exc) or exc.__class__.__name__
        try:
            return _failure_bundle(
                final_dir=final_dir,
                final_archive=final_archive,
                captured_log=captured_log,
                reason=reason,
                evaluator_returncode=evaluator_returncode,
                authority_bindings=authority_bindings,
                created_results=created_results,
            )
        except BaseException as bundle_exc:
            raise H6TerminalResultError(
                "H6 invocation marker was consumed and terminal failure bundling failed"
            ) from bundle_exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen H6 evaluator once and preserve terminal evidence"
    )
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--terminal-dir", type=Path, required=True)
    parser.add_argument("--terminal-archive", type=Path, required=True)
    parser.add_argument("--wrapper-sha256", required=True)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--admission-receipt-sha256", required=True)
    parser.add_argument("--admission-event-sha256", required=True)
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--run-spec-sha256", required=True)
    parser.add_argument("--run-events", type=Path, required=True)
    parser.add_argument("--run-events-sha256", required=True)
    parser.add_argument("--run-events-tail-sha256", required=True)
    return parser


def _require_isolated_cli_startup() -> None:
    """Refuse before argument handling unless the literal CLI contract is active."""

    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        raise H6TerminalResultError("H6 terminal wrapper must run with Python -I -S")
    wrapper = Path(__file__).resolve()
    invoked = Path(sys.argv[0])
    if (
        __spec__ is not None
        or not invoked.is_absolute()
        or _absolute_path(invoked) != wrapper
    ):
        raise H6TerminalResultError(
            "H6 terminal wrapper must be invoked as its absolute script path"
        )
    source_root = wrapper.parents[2]
    expected_interpreter = source_root / _ISOLATED_INTERPRETER_RELATIVE
    if _absolute_path(Path(sys.executable)) != expected_interpreter:
        raise H6TerminalResultError(
            "H6 terminal wrapper is not running through .h6-venv/bin/python"
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        _require_isolated_cli_startup()
    except H6TerminalResultError as exc:
        print(f"H6 terminal wrapper refused: {exc}", file=sys.stderr)
        return WRAPPER_FAILURE_EXIT_CODE
    args = _parser().parse_args(argv)
    try:
        outcome = run_terminal_evaluation(
            package_root=args.package_root,
            terminal_dir=args.terminal_dir,
            terminal_archive=args.terminal_archive,
            bindings=AuthorityBindings(
                wrapper_sha256=args.wrapper_sha256,
                admission_receipt=args.admission_receipt,
                admission_receipt_sha256=args.admission_receipt_sha256,
                admission_event_sha256=args.admission_event_sha256,
                run_spec=args.run_spec,
                run_spec_sha256=args.run_spec_sha256,
                run_events=args.run_events,
                run_events_sha256=args.run_events_sha256,
                run_events_tail_sha256=args.run_events_tail_sha256,
            ),
        )
    except H6TerminalResultError as exc:
        print(f"H6 terminal wrapper refused: {exc}", file=sys.stderr)
        return WRAPPER_FAILURE_EXIT_CODE
    print(
        json.dumps(
            {
                "archive_sha256": outcome.archive_sha256,
                "evaluator_returncode": outcome.evaluator_returncode,
                "exit_code": outcome.exit_code,
                "schema": SCHEMA_VERSION,
                "status": outcome.status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
