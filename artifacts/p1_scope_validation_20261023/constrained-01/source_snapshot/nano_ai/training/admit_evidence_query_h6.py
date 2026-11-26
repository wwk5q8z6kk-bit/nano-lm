"""Development-free destination admission for Nano H6.

This command authenticates the frozen H6 training package and recovered
training artifacts on a replacement destination.  It intentionally has no
development-data input: caller-named development, fresh, or private paths must
be absent, and the only output is a no-clobber canonical admission receipt.
Provider resources and the RunPod guard ledger are read-only inputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import re
import stat
import sys
import sysconfig
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

sys.dont_write_bytecode = True

H6_DESTINATION_ADMISSION_SCHEMA_VERSION = "nano.h6-destination-admission.v1"
_UPLOAD_POLICY_SCHEMA = "nano.h6.evaluation-upload-policy.v1"
_LEDGER_SCHEMA = "nano.runpod.ledger.v2"
_ADMISSION_SYNC_SCHEMA = "nano.runpod.destination-admission-sync.v1"
_ADMISSION_SYNC_EVENT = "DESTINATION_ADMISSION_SYNC_PREPARED"
_ADMISSION_SYNC_PROTOCOL = "atomic_content_addressed_authority_mirror_v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DESTINATION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_SELECTION_ORDER = (
    "uncalibrated_training_calibration_macro_joint_desc",
    "uncalibrated_training_calibration_overall_joint_desc",
    "earlier_epoch",
    "seed_20260805",
)
_TRAINING_SEEDS = (20260805, 20260806)
_AUTHORIZED_DATA_CENTERS = ("EU-RO-1",)
_EXPECTED_DESTINATION_SELECTION_SHA256 = (
    "c6d77974c24f99c8255651028537320b02ca635ee78d9336e999329fa44996d6"
)
_EXPECTED_PREDECESSOR_BINDING_SHA256 = (
    "11db01a6f5b95f0ca331f65e1a542eca61bb97671b993a19fab6f2311175ac3c"
)
_EXPECTED_RUNTIME_AUTHORITY_SHA256 = (
    "06967da8e87193e14a8e3c079dac975634dac4b53da80cac24e89ac5fba680b2"
)
_EXPECTED_REQUIRED_ARTIFACTS_SHA256 = (
    "430f33bff844905d1e20e19d16c20305cdeeb8b17e7734f460dad6ec122a85c9"
)
_EXPECTED_DEVELOPMENT_MANIFEST_SHA256 = (
    "47ee157ac037c0771100b8546c90da91dbd2006198700bb642f1561d2124c1a3"
)
_EXPECTED_IMAGE_ID = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
_EXPECTED_CONFIGURED_GPU_ID = "NVIDIA GeForce RTX 5090"
_EXPECTED_SPEC_GPU_MODEL = "RTX_5090"
_ISOLATED_EXECUTION_CONTRACT = {
    "schema": "nano.h6.isolated-execution-contract.v1",
    "interpreter_relative_path": ".h6-venv/bin/python",
    "python_flags": ["-I", "-S"],
    "admission_program_relative_path": ("nano_ai/training/admit_evidence_query_h6.py"),
    "terminal_wrapper_relative_path": "nano_ai/training/h6_terminal_result.py",
    "evaluator_launcher_relative_path": "RUN_H6_EVALUATE.sh",
    "evaluator_launcher_shell": "/bin/bash",
    "evaluator_launcher_flags": ["--noprofile", "--norc"],
    "terminal_wrapper_bridge": "module_only_v1",
}
_CONTINUATION_AUTHORIZATION_KEYS = frozenset(
    {
        "schema",
        "authority",
        "authorized_actions",
        "authorized_at_utc",
        "objective",
        "predecessor",
        "destination_policy",
        "isolated_execution_contract",
        "max_predevelopment_destination_attempts",
        "max_total_runpod_usd",
        "failed_admission_rule",
        "termination_gate",
        "prohibited_actions",
        "frozen_scientific_identity",
    }
)
_CONTINUATION_FROZEN_IDENTITY_KEYS = frozenset(
    {
        "evaluation_authority_id",
        "preregistration_sha256",
        "training_report_freeze_sha256",
        "package_sha256",
        "evaluator_sha256",
        "development_sha256",
        "required_artifacts_manifest_sha256",
        "upload_policy_sha256",
        "admission_program_sha256",
        "terminal_wrapper_sha256",
    }
)
_CONTINUATION_PREDECESSOR_KEYS = frozenset(
    {
        "run_id",
        "phase",
        "ledger_tail_sha256",
        "recovery_binding_sha256",
        "source_pod_id",
        "source_machine_id",
        "source_required_state",
        "source_access_policy",
    }
)
_CONTINUATION_DESTINATION_POLICY_KEYS = frozenset(
    {
        "cloud_tier",
        "allowed_data_center_ids",
        "source_data_center_required",
        "selection_evidence",
        "reason",
    }
)
_SUCCESSFUL_OPERATION_OUTCOMES = {"succeeded", "reconciled_succeeded"}
_USABLE_NETWORK_VOLUME_STATES = {"available", "attached", "detached", "retained"}
_EXPECTED_FIXED_ARTIFACT_ROLES = {
    "selected_checkpoint_seed_20260805",
    "selected_checkpoint_seed_20260806",
    "training_report_checksum_manifest",
    "training_report_seed_20260805",
    "training_report_seed_20260806",
    "vocabulary",
}
_EXPECTED_PENDING_ARTIFACT_ROLES = {
    "one_shot_terminal_evaluation_evidence",
}
_PROHIBITED_STAGING_COMPONENTS = {
    "dev",
    "development",
    "fresh",
    "fresh-v1",
    "private",
    "private-corpus",
}
_DYNAMIC_UPLOAD_SOURCES = {
    "run_spec": {
        "sha256_source": "ledger_first_event.run_spec_sha256",
    },
    "run_events": {
        "tail_sha256_source": "admission_sync_prepared_ledger_tail",
    },
    "authorization_envelope": {
        "sha256_source": "run_spec.authorization_envelope_sha256",
    },
    "upload_policy": {
        "sha256_source": "run_spec.upload_allowlist_sha256",
    },
    "destination_attestation": {
        "sha256_source": ("ledger_final_provider_snapshot.observation_evidence_sha256"),
    },
    "predecessor_run_spec": {
        "sha256_source": ("predecessor_binding.predecessor.run_spec_sha256"),
    },
    "predecessor_run_events": {
        "sha256_source": ("predecessor_binding.predecessor.run_events_sha256"),
        "tail_sha256_source": ("predecessor_binding.predecessor.ledger_tail_sha256"),
    },
}
_FROZEN_SOURCE_PATHS = {
    "state_conditioned_evidence_query_model": (
        "nano_ai/training/state_conditioned_evidence_query_model.py"
    ),
    "train_evidence_query_h6": "nano_ai/training/train_evidence_query_h6.py",
    "evaluate_evidence_query_h6": ("nano_ai/training/evaluate_evidence_query_h6.py"),
    "package_evidence_query_h6": "nano_ai/training/package_evidence_query_h6.py",
    "replay_mixture_data": "nano_ai/training/replay_mixture_data.py",
    "runpod_requirements": "requirements-h4-runpod.txt",
}
_FROZEN_TEST_PATHS = {
    "state_conditioned_model": (
        "nano_ai/tests/test_state_conditioned_evidence_query_model.py"
    ),
    "training": "nano_ai/tests/test_train_evidence_query_h6.py",
    "evaluation": "nano_ai/tests/test_evaluate_evidence_query_h6.py",
    "package": "nano_ai/tests/test_package_evidence_query_h6.py",
}
_CLI_DEPENDENCY_ROOTS: tuple[Path, ...] = ()
_AUTHORITY_HASH_FIELDS = (
    "authorization_envelope_sha256",
    "preregistration_sha256",
    "freeze_sha256",
    "package_sha256",
    "evaluator_sha256",
    "upload_allowlist_sha256",
    "runtime_authority_sha256",
    "development_sha256",
    "required_artifacts_manifest_sha256",
)


class H6DestinationAdmissionError(RuntimeError):
    """Raised when a destination cannot be admitted without development."""


def _require_isolated_no_site_startup(flags: Any | None = None) -> None:
    """Fail before argument or file handling unless Python startup is inert."""

    observed = sys.flags if flags is None else flags
    if (
        getattr(observed, "isolated", None) != 1
        or getattr(observed, "no_site", None) != 1
    ):
        raise H6DestinationAdmissionError(
            "H6 destination admission must run with Python -I -S"
        )


def _real_dependency_directory(path: Path, *, role: str) -> Path:
    """Return one existing real directory without accepting a symlink entry."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        path_stat = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise H6DestinationAdmissionError(f"{role} is unavailable") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise H6DestinationAdmissionError(f"{role} is not a real directory")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _bootstrap_isolated_import_paths(source_root: Path) -> tuple[Path, ...]:
    """Add only the authenticated-source slot and audited dependency roots.

    ``-I -S`` deliberately leaves only the standard-library paths.  Directly
    adding these directories does not execute ``.pth`` files or import Python's
    ``sitecustomize``/``usercustomize`` startup hooks.
    """

    global _CLI_DEPENDENCY_ROOTS

    source = _real_dependency_directory(source_root, role="H6 source root")
    if source != Path(os.path.abspath(os.fspath(source_root))):
        raise H6DestinationAdmissionError("H6 source root has a symlinked ancestor")

    environment = _real_dependency_directory(
        source / ".h6-venv", role="H6 isolated environment"
    )
    if environment != source / ".h6-venv":
        raise H6DestinationAdmissionError(
            "H6 isolated environment has a symlinked ancestor"
        )

    interpreter = source / ".h6-venv/bin/python"
    try:
        interpreter_stat = interpreter.lstat()
        interpreter_target = interpreter.resolve(strict=True)
        interpreter_target_stat = interpreter_target.stat()
    except OSError as exc:
        raise H6DestinationAdmissionError(
            "H6 isolated Python interpreter is unavailable"
        ) from exc
    if not (
        stat.S_ISREG(interpreter_stat.st_mode) or stat.S_ISLNK(interpreter_stat.st_mode)
    ) or not stat.S_ISREG(interpreter_target_stat.st_mode):
        raise H6DestinationAdmissionError(
            "H6 isolated Python interpreter is not executable file material"
        )
    if Path(os.path.abspath(sys.executable)) != interpreter:
        raise H6DestinationAdmissionError(
            "H6 admission is not running through .h6-venv/bin/python"
        )

    if any(name == "nano_ai" or name.startswith("nano_ai.") for name in sys.modules):
        raise H6DestinationAdmissionError(
            "Nano modules were imported before source authentication"
        )
    if "torch" in sys.modules or "tokenizers" in sys.modules:
        raise H6DestinationAdmissionError(
            "H6 dependencies were imported before isolated path bootstrap"
        )

    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    environment_packages = _real_dependency_directory(
        environment / "lib" / version / "site-packages",
        role="H6 isolated site-packages",
    )
    if not _is_within(environment_packages, environment):
        raise H6DestinationAdmissionError(
            "H6 isolated site-packages escapes its environment"
        )

    base_prefixes: list[Path] = []
    for label, raw_prefix in (
        ("base prefix", sys.base_prefix),
        ("base executable prefix", sys.base_exec_prefix),
    ):
        prefix = _real_dependency_directory(Path(raw_prefix), role=f"Python {label}")
        if prefix not in base_prefixes:
            base_prefixes.append(prefix)

    dependency_roots = [environment_packages]
    configured_paths = sysconfig.get_paths()
    for name in ("purelib", "platlib"):
        raw_path = configured_paths.get(name)
        if not isinstance(raw_path, str) or not raw_path:
            raise H6DestinationAdmissionError(
                f"isolated sysconfig omitted the Python {name} dependency root"
            )
        candidate = _real_dependency_directory(
            Path(raw_path), role=f"Python {name} dependency root"
        )
        if not any(_is_within(candidate, prefix) for prefix in base_prefixes):
            raise H6DestinationAdmissionError(
                f"Python {name} dependency root escapes the base runtime"
            )
        if candidate not in dependency_roots:
            dependency_roots.append(candidate)

    for dependency_root in dependency_roots:
        if os.path.lexists(dependency_root / "nano_ai"):
            raise H6DestinationAdmissionError(
                f"dependency root contains a top-level Nano shadow: {dependency_root}"
            )

    source_text = os.fspath(source)
    dependency_text = [os.fspath(path) for path in dependency_roots]
    if source_text in sys.path or any(path in sys.path for path in dependency_text):
        raise H6DestinationAdmissionError(
            "H6 import roots were present before isolated path bootstrap"
        )
    sys.path.insert(0, source_text)
    sys.path.extend(dependency_text)
    _CLI_DEPENDENCY_ROOTS = tuple(dependency_roots)
    return _CLI_DEPENDENCY_ROOTS


def _require_dependency_module_origin(module: Any, *, role: str) -> None:
    """Bind an imported third-party package to one bootstrapped root."""

    if not _CLI_DEPENDENCY_ROOTS:
        return
    raw_origin = getattr(module, "__file__", None)
    if not isinstance(raw_origin, str) or not raw_origin:
        raise H6DestinationAdmissionError(f"{role} has no file-backed origin")
    try:
        origin = Path(raw_origin).resolve(strict=True)
    except OSError as exc:
        raise H6DestinationAdmissionError(f"{role} origin is unavailable") from exc
    if not any(_is_within(origin, root) for root in _CLI_DEPENDENCY_ROOTS):
        raise H6DestinationAdmissionError(
            f"{role} was imported outside the audited dependency roots"
        )


def _require_isolated_execution_contract(value: Any) -> dict[str, Any]:
    """Require the literal no-site/no-profile H6 launch contract."""

    if not isinstance(value, Mapping) or dict(value) != _ISOLATED_EXECUTION_CONTRACT:
        raise H6DestinationAdmissionError("H6 isolated execution contract is not exact")
    return dict(_ISOLATED_EXECUTION_CONTRACT)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise H6DestinationAdmissionError(
            "admission receipt is not canonical JSON"
        ) from exc


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise H6DestinationAdmissionError(f"{role} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, role: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise H6DestinationAdmissionError(f"{role} must be non-empty edge-trimmed text")
    return value


def _require_positive_int(value: object, role: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise H6DestinationAdmissionError(f"{role} must be a positive integer")
    return value


def _read_regular_file(path: Path, *, role: str) -> tuple[bytes, dict[str, Any]]:
    """Read one non-symlink regular file and detect changes during hashing."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise H6DestinationAdmissionError(
            f"{role} is unavailable or cannot be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise H6DestinationAdmissionError(f"{role} is not a regular file")
        chunks: list[bytes] = []
        hasher = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            hasher.update(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if before_identity != after_identity:
            raise H6DestinationAdmissionError(f"{role} changed while authenticated")
        payload = b"".join(chunks)
        return payload, {"bytes": len(payload), "sha256": hasher.hexdigest()}
    finally:
        os.close(descriptor)


def _read_verified_file(
    path: Path,
    expected_sha256: object,
    *,
    role: str,
    expected_bytes: object | None = None,
) -> tuple[bytes, dict[str, Any]]:
    payload, identity = _read_regular_file(path, role=role)
    if identity["sha256"] != _require_sha256(expected_sha256, role):
        raise H6DestinationAdmissionError(f"{role} SHA-256 mismatch")
    if expected_bytes is not None:
        byte_count = _require_positive_int(expected_bytes, f"{role} byte count")
        if identity["bytes"] != byte_count:
            raise H6DestinationAdmissionError(f"{role} byte count mismatch")
    return payload, identity


def _parse_json(payload: bytes, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {token}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise H6DestinationAdmissionError(f"{role} is not valid strict JSON") from exc
    if not isinstance(value, dict):
        raise H6DestinationAdmissionError(f"{role} must be a JSON object")
    return value


def _safe_relative(value: object, *, role: str) -> str:
    if not isinstance(value, str):
        raise H6DestinationAdmissionError(f"{role} must be relative text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise H6DestinationAdmissionError(f"{role} contains an unsafe path")
    return value


def _relative_to_root(path: Path, root: Path, *, role: str) -> str:
    """Return a canonical relative path without following a caller symlink."""

    absolute_path = Path(os.path.abspath(os.fspath(path)))
    absolute_root = Path(os.path.abspath(os.fspath(root)))
    try:
        relative = absolute_path.relative_to(absolute_root).as_posix()
    except ValueError as exc:
        raise H6DestinationAdmissionError(
            f"{role} is outside the authenticated upload root"
        ) from exc
    return _safe_relative(relative, role=role)


def _merge_tracked(
    destination: dict[Path, dict[str, Any]],
    addition: Mapping[Path, Mapping[str, Any]],
) -> None:
    """Merge authenticated identities while rejecting conflicting aliases."""

    for path, identity in addition.items():
        normalized = Path(os.path.abspath(os.fspath(path)))
        observed = dict(identity)
        if normalized in destination and destination[normalized] != observed:
            raise H6DestinationAdmissionError(
                f"authenticated input has conflicting identities: {normalized}"
            )
        destination[normalized] = observed


def _identity_record(value: object, *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != {"bytes", "sha256"}:
        raise H6DestinationAdmissionError(f"{role} identity is invalid")
    byte_count = value["bytes"]
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
    ):
        raise H6DestinationAdmissionError(f"{role} byte count is invalid")
    _require_sha256(value["sha256"], role)
    return value


def _require_exact_static_policy_rows(
    entries: object,
    *,
    expected_by_role: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind every static upload identity to its exact semantic role and path."""

    if not isinstance(entries, list) or len(entries) != len(expected_by_role):
        raise H6DestinationAdmissionError(
            "H6 upload policy static inventory is not exact"
        )
    observed_by_role: dict[str, dict[str, Any]] = {}
    observed_by_path: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != {
            "role",
            "relative_path",
            "sha256",
            "bytes",
        }:
            raise H6DestinationAdmissionError(
                f"H6 upload policy static entry {index} is invalid"
            )
        semantic_role = _require_text(entry.get("role"), f"static entry {index} role")
        relative = _safe_relative(
            entry.get("relative_path"), role=f"static entry {index} path"
        )
        identity = dict(
            _identity_record(
                {"sha256": entry.get("sha256"), "bytes": entry.get("bytes")},
                role=f"static entry {index}",
            )
        )
        if semantic_role in observed_by_role:
            raise H6DestinationAdmissionError(
                "H6 upload policy static roles must be unique"
            )
        if relative in observed_by_path:
            raise H6DestinationAdmissionError(
                "H6 upload policy static paths must be unique"
            )
        observed_by_role[semantic_role] = {
            "role": semantic_role,
            "relative_path": relative,
            **identity,
        }
        observed_by_path[relative] = identity
    if observed_by_role != {role: dict(row) for role, row in expected_by_role.items()}:
        raise H6DestinationAdmissionError(
            "H6 upload policy does not bind exact static identities to roles"
        )
    return observed_by_path


def _verify_ledger(
    path: Path, *, expected_tail_sha256: str
) -> tuple[
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
]:
    payload, file_identity = _read_regular_file(path, role="RunPod run ledger")
    if not payload or not payload.endswith(b"\n"):
        raise H6DestinationAdmissionError(
            "RunPod run ledger is empty or has a partial trailing record"
        )
    expected_tail = _require_sha256(expected_tail_sha256, "expected ledger tail")
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
    first: Mapping[str, Any] | None = None
    events: list[Mapping[str, Any]] = []
    count = 0
    for count, line in enumerate(payload.splitlines(), start=1):
        event = _parse_json(line, role=f"RunPod ledger record {count}")
        if line + b"\n" != _canonical_json_bytes(event):
            raise H6DestinationAdmissionError(
                f"RunPod ledger record {count} is not canonical"
            )
        if set(event) != expected_keys:
            raise H6DestinationAdmissionError(
                f"RunPod ledger record {count} has an invalid shape"
            )
        if event["schema"] != _LEDGER_SCHEMA or event["index"] != count:
            raise H6DestinationAdmissionError(
                f"RunPod ledger record {count} has an invalid schema or sequence"
            )
        if event["previous_sha256"] != previous:
            raise H6DestinationAdmissionError(
                f"RunPod ledger chain breaks at record {count}"
            )
        base = {key: event[key] for key in expected_keys - {"event_sha256"}}
        observed = _digest(_canonical_json_bytes(base).removesuffix(b"\n"))
        if event["event_sha256"] != observed:
            raise H6DestinationAdmissionError(
                f"RunPod ledger hash mismatch at record {count}"
            )
        previous = observed
        events.append(event)
        if first is None:
            first = event
    if previous != expected_tail:
        raise H6DestinationAdmissionError("RunPod ledger tail changed or was misbound")
    assert first is not None
    return (
        first,
        tuple(events),
        {
            "event_count": count,
            "tail_sha256": previous,
            "file_sha256": file_identity["sha256"],
            "bytes": file_identity["bytes"],
        },
        file_identity,
    )


def _assert_paths_absent(paths: Sequence[Path]) -> tuple[str, ...]:
    if len(paths) < 2:
        raise H6DestinationAdmissionError(
            "development data and manifest paths must both be specified"
        )
    normalized: list[str] = []
    seen: set[str] = set()
    for path in paths:
        absolute = os.path.abspath(os.fspath(path))
        if absolute in seen:
            raise H6DestinationAdmissionError("prohibited-data paths must be unique")
        seen.add(absolute)
        if os.path.lexists(absolute):
            raise H6DestinationAdmissionError(
                f"prohibited data path is present: {absolute}"
            )
        normalized.append(absolute)
    return tuple(normalized)


def _package_manifest(
    package_payload: bytes,
    *,
    archive_root: str,
) -> tuple[
    Mapping[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    manifest_name = f"{archive_root}/BUNDLE_MANIFEST.json"
    try:
        with tarfile.open(fileobj=io.BytesIO(package_payload), mode="r:gz") as archive:
            entries = archive.getmembers()
            names = [entry.name for entry in entries]
            if len(names) != len(set(names)) or manifest_name not in names:
                raise H6DestinationAdmissionError(
                    "H6 package has a missing or duplicate manifest"
                )
            manifest_entry = archive.getmember(manifest_name)
            if not manifest_entry.isfile():
                raise H6DestinationAdmissionError("H6 package manifest is not regular")
            manifest_file = archive.extractfile(manifest_entry)
            if manifest_file is None:
                raise H6DestinationAdmissionError("H6 package manifest is unreadable")
            manifest_payload = manifest_file.read()
            manifest = _parse_json(manifest_payload, role="H6 package bundle manifest")
            manifest_identity = {
                "bytes": len(manifest_payload),
                "sha256": _digest(manifest_payload),
            }
            members = manifest.get("members")
            if not isinstance(members, dict):
                raise H6DestinationAdmissionError(
                    "H6 package member authority is invalid"
                )
            expected_names = {manifest_name} | {
                f"{archive_root}/{_safe_relative(name, role='H6 package member')}"
                for name in members
            }
            if set(names) != expected_names:
                raise H6DestinationAdmissionError(
                    "H6 package member inventory is not exact"
                )
            authenticated: dict[str, dict[str, Any]] = {}
            for relative, expected_value in sorted(members.items()):
                expected = _identity_record(
                    expected_value, role=f"H6 package member {relative}"
                )
                entry = archive.getmember(f"{archive_root}/{relative}")
                if not entry.isfile() or entry.issym() or entry.islnk():
                    raise H6DestinationAdmissionError(
                        f"H6 package member is not regular: {relative}"
                    )
                member_file = archive.extractfile(entry)
                if member_file is None:
                    raise H6DestinationAdmissionError(
                        f"H6 package member is unreadable: {relative}"
                    )
                payload = member_file.read()
                identity = {"bytes": len(payload), "sha256": _digest(payload)}
                if identity != expected:
                    raise H6DestinationAdmissionError(
                        f"H6 package member identity mismatch: {relative}"
                    )
                authenticated[relative] = identity
    except H6DestinationAdmissionError:
        raise
    except (OSError, tarfile.TarError, ValueError) as exc:
        raise H6DestinationAdmissionError(
            "H6 package could not be authenticated"
        ) from exc
    return manifest, manifest_identity, authenticated


def _authenticate_static_authorities(
    *,
    run_spec_path: Path,
    run_events_path: Path,
    expected_ledger_tail_sha256: str,
    preregistration_path: Path,
    freeze_path: Path,
    readiness_path: Path,
    package_path: Path,
    source_root: Path,
    run_id: str,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    first_event, ledger_events, ledger, ledger_file_identity = _verify_ledger(
        run_events_path,
        expected_tail_sha256=expected_ledger_tail_sha256,
    )
    tracked: dict[Path, dict[str, Any]] = {run_events_path: ledger_file_identity}
    spec_payload, spec_identity = _read_regular_file(run_spec_path, role="run spec")
    tracked[run_spec_path] = spec_identity
    spec = _parse_json(spec_payload, role="run spec")
    genesis_payload = first_event.get("payload")
    if (
        first_event.get("event") != "RUN_INITIALIZED"
        or not isinstance(genesis_payload, Mapping)
        or genesis_payload.get("spec") != spec
    ):
        raise H6DestinationAdmissionError(
            "run spec does not match the authenticated ledger genesis"
        )
    if (
        spec.get("schema") != "nano.runpod.spec.v3"
        or spec.get("project") != "nano-lm"
        or spec.get("run_id") != run_id
    ):
        raise H6DestinationAdmissionError("run spec identity is invalid")

    observed_authority_fields = {
        name for name in spec if isinstance(name, str) and name.endswith("_sha256")
    }
    if not set(_AUTHORITY_HASH_FIELDS).issubset(observed_authority_fields):
        raise H6DestinationAdmissionError(
            "run spec authority-hash inventory is incomplete"
        )
    authority_hashes: dict[str, str] = {}
    for name in sorted(observed_authority_fields):
        authority_hashes[name] = _require_sha256(spec[name], f"run spec {name}")

    prereg_payload, prereg_identity = _read_verified_file(
        preregistration_path,
        authority_hashes["preregistration_sha256"],
        role="H6 preregistration",
    )
    tracked[preregistration_path] = prereg_identity
    prereg = _parse_json(prereg_payload, role="H6 preregistration")
    if (
        prereg.get("schema_version") != "nano.h6-preregistration.v1"
        or prereg.get("status") != "FROZEN_BEFORE_GPU_TRAINING"
        or prereg.get("selection", {}).get("development_used") is not False
        or tuple(prereg.get("selection", {}).get("order", ())) != _SELECTION_ORDER
    ):
        raise H6DestinationAdmissionError("H6 preregistration boundary is invalid")
    prohibited = prereg.get("prohibited_access")
    if not isinstance(prohibited, Mapping) or any(
        value is not False for value in prohibited.values()
    ):
        raise H6DestinationAdmissionError(
            "H6 preregistration prohibited-access boundary is invalid"
        )

    freeze_payload, freeze_identity = _read_verified_file(
        freeze_path,
        authority_hashes["freeze_sha256"],
        role="H6 training-report freeze",
    )
    tracked[freeze_path] = freeze_identity
    freeze = _parse_json(freeze_payload, role="H6 training-report freeze")
    if (
        freeze.get("schema_version") != "nano.h6.training-report-freeze.v1"
        or freeze.get("experiment_id") != prereg.get("experiment_id")
        or freeze.get("frozen_before_development_transfer") is not True
        or freeze.get("development_present_at_freeze") is not False
        or freeze.get("development_records_used") != 0
        or freeze.get("preregistration_sha256") != prereg_identity["sha256"]
        or freeze.get("input_bundle_sha256") != authority_hashes["package_sha256"]
        or tuple(freeze.get("selection_order", ())) != _SELECTION_ORDER
    ):
        raise H6DestinationAdmissionError("H6 training-report freeze is invalid")

    readiness_sha256 = _require_sha256(freeze.get("readiness_sha256"), "H6 readiness")
    readiness_payload, readiness_identity = _read_verified_file(
        readiness_path,
        readiness_sha256,
        role="H6 readiness",
    )
    tracked[readiness_path] = readiness_identity
    readiness = _parse_json(readiness_payload, role="H6 readiness")
    archive = readiness.get("archive")
    readiness_inputs = readiness.get("inputs")
    readiness_development = (
        readiness_inputs.get("development")
        if isinstance(readiness_inputs, Mapping)
        else None
    )
    readiness_development_manifest = (
        readiness_inputs.get("development_manifest")
        if isinstance(readiness_inputs, Mapping)
        else None
    )
    if (
        readiness.get("schema_version") != "nano.h6-readiness.v1"
        or readiness.get("status") != "READY_FOR_TWO_SEED_RUN"
        or readiness.get("known_development_included") is not False
        or readiness.get("fresh_suites_included") is not False
        or readiness.get("private_corpus_included") is not False
        or readiness.get("benchmarks_included") is not False
        or not isinstance(archive, Mapping)
        or archive.get("sha256") != authority_hashes["package_sha256"]
        or not isinstance(readiness_development, Mapping)
        or readiness_development.get("included_in_training_bundle") is not False
        or readiness_development.get("sha256") != authority_hashes["development_sha256"]
        or not isinstance(readiness_development_manifest, Mapping)
        or readiness_development_manifest.get("included_in_training_bundle")
        is not False
        or readiness_development_manifest.get("sha256")
        != _EXPECTED_DEVELOPMENT_MANIFEST_SHA256
    ):
        raise H6DestinationAdmissionError("H6 readiness boundary is invalid")
    archive_bytes = archive.get("bytes")
    archive_root = archive.get("root")
    if (
        isinstance(archive_bytes, bool)
        or not isinstance(archive_bytes, int)
        or archive_bytes < 1
        or not isinstance(archive_root, str)
        or not archive_root
    ):
        raise H6DestinationAdmissionError("H6 readiness archive identity is invalid")
    package_payload, package_identity = _read_verified_file(
        package_path,
        authority_hashes["package_sha256"],
        role="H6 frozen package",
        expected_bytes=archive_bytes,
    )
    tracked[package_path] = package_identity
    bundle_manifest, bundle_manifest_identity, package_members = _package_manifest(
        package_payload,
        archive_root=archive_root,
    )
    if (
        bundle_manifest.get("schema_version") != "nano.h6-runpod-bundle.v1"
        or bundle_manifest.get("recipe") != readiness.get("recipe")
        or bundle_manifest.get("archive_root") != archive_root
        or tuple(bundle_manifest.get("training_seeds", ())) != _TRAINING_SEEDS
        or bundle_manifest.get("development_manifest_sha256")
        != _EXPECTED_DEVELOPMENT_MANIFEST_SHA256
    ):
        raise H6DestinationAdmissionError("H6 package manifest boundary is invalid")

    frozen_sources = prereg.get("frozen_source_sha256")
    if not isinstance(frozen_sources, Mapping):
        raise H6DestinationAdmissionError("H6 frozen source authority is invalid")
    top_level = set(frozen_sources) - {"tests"}
    if top_level != set(_FROZEN_SOURCE_PATHS):
        raise H6DestinationAdmissionError("H6 frozen source inventory is incomplete")
    frozen_tests = frozen_sources.get("tests")
    if not isinstance(frozen_tests, Mapping) or set(frozen_tests) != set(
        _FROZEN_TEST_PATHS
    ):
        raise H6DestinationAdmissionError("H6 frozen test inventory is incomplete")

    source_identities: dict[str, dict[str, Any]] = {}
    for name, relative in {**_FROZEN_SOURCE_PATHS, **_FROZEN_TEST_PATHS}.items():
        expected = (
            frozen_tests[name] if name in _FROZEN_TEST_PATHS else frozen_sources[name]
        )
        expected_digest = _require_sha256(expected, f"H6 source {name}")
        path = source_root / _safe_relative(relative, role=f"H6 source {name}")
        _payload, identity = _read_verified_file(
            path, expected_digest, role=f"H6 source {name}"
        )
        if package_members.get(relative) != identity:
            raise H6DestinationAdmissionError(
                f"extracted H6 source differs from the frozen package: {name}"
            )
        tracked[path] = identity
        source_identities[name] = {"path": relative, **identity}
    if (
        authority_hashes.get("evaluator_sha256")
        != frozen_sources["evaluate_evidence_query_h6"]
    ):
        raise H6DestinationAdmissionError(
            "run evaluator authority disagrees with the preregistration"
        )

    return {
        "spec": spec,
        "preregistration": prereg,
        "freeze": freeze,
        "readiness": readiness,
        "bundle_manifest": bundle_manifest,
        "bundle_manifest_identity": bundle_manifest_identity,
        "package_members": package_members,
        "ledger": ledger,
        "ledger_events": ledger_events,
        "authority_hashes": authority_hashes,
        "authority_identities": {
            "run_spec": spec_identity,
            "run_events": ledger_file_identity,
            "preregistration": prereg_identity,
            "training_report_freeze": freeze_identity,
            "readiness": readiness_identity,
            "package": package_identity,
        },
        "source_identities": source_identities,
    }, tracked


def _require_guard_destination_spec(
    spec: Mapping[str, Any], authority_hashes: Mapping[str, str]
) -> None:
    """Require the destination fields emitted by the current RunPod guard."""

    if (
        authority_hashes["runtime_authority_sha256"]
        != _EXPECTED_RUNTIME_AUTHORITY_SHA256
        or authority_hashes["required_artifacts_manifest_sha256"]
        != _EXPECTED_REQUIRED_ARTIFACTS_SHA256
        or tuple(spec.get("allowed_data_center_ids", ())) != _AUTHORIZED_DATA_CENTERS
        or tuple(spec.get("allowed_image_ids", ())) != (_EXPECTED_IMAGE_ID,)
        or tuple(spec.get("allowed_gpu_models", ())) != (_EXPECTED_SPEC_GPU_MODEL,)
        or spec.get("storage_class") != "network"
    ):
        raise H6DestinationAdmissionError(
            "current run spec does not encode the exact destination continuation"
        )


def _authenticate_continuation_controls(
    *,
    authority: Mapping[str, Any],
    continuation_authorization_path: Path,
    expected_continuation_authorization_sha256: str,
    predecessor_binding_path: Path,
    predecessor_run_spec_path: Path,
    predecessor_run_events_path: Path,
    runtime_authority_path: Path,
    required_artifacts_path: Path,
    destination_selection_path: Path,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    """Authenticate the immutable authority chain for this continuation."""

    spec = authority["spec"]
    authority_hashes = authority["authority_hashes"]
    expected_authorization = _require_sha256(
        expected_continuation_authorization_sha256,
        "expected continuation authorization",
    )
    if authority_hashes["authorization_envelope_sha256"] != expected_authorization:
        raise H6DestinationAdmissionError(
            "current run spec is not bound to the expected continuation authorization"
        )

    tracked: dict[Path, dict[str, Any]] = {}
    authorization_payload, authorization_identity = _read_verified_file(
        continuation_authorization_path,
        expected_authorization,
        role="H6 continuation authorization",
    )
    tracked[continuation_authorization_path] = authorization_identity
    authorization = _parse_json(
        authorization_payload, role="H6 continuation authorization"
    )
    predecessor_authority = authorization.get("predecessor")
    frozen_identity = authorization.get("frozen_scientific_identity")
    destination_policy = authorization.get("destination_policy")
    isolated_execution_contract = authorization.get("isolated_execution_contract")
    if (
        set(authorization) != _CONTINUATION_AUTHORIZATION_KEYS
        or authorization.get("schema")
        != "nano.h6.evaluation-continuation-authorization.v1"
        or not isinstance(predecessor_authority, Mapping)
        or not isinstance(frozen_identity, Mapping)
        or not isinstance(destination_policy, Mapping)
        or set(predecessor_authority) != _CONTINUATION_PREDECESSOR_KEYS
        or set(frozen_identity) != _CONTINUATION_FROZEN_IDENTITY_KEYS
        or set(destination_policy) != _CONTINUATION_DESTINATION_POLICY_KEYS
        or predecessor_authority.get("source_required_state") != "stopped"
        or predecessor_authority.get("source_access_policy")
        != "lineage_only_no_restart"
        or destination_policy.get("cloud_tier") != "secure"
        or tuple(destination_policy.get("allowed_data_center_ids", ()))
        != _AUTHORIZED_DATA_CENTERS
        or destination_policy.get("source_data_center_required") is not False
    ):
        raise H6DestinationAdmissionError(
            "H6 continuation authorization boundary is invalid"
        )
    _require_isolated_execution_contract(isolated_execution_contract)
    if frozen_identity.get("evaluation_authority_id") != "nano-h6-dev-one-shot-v1":
        raise H6DestinationAdmissionError(
            "H6 continuation evaluation authority is invalid"
        )

    frozen_to_spec = {
        "preregistration_sha256": "preregistration_sha256",
        "training_report_freeze_sha256": "freeze_sha256",
        "package_sha256": "package_sha256",
        "evaluator_sha256": "evaluator_sha256",
        "development_sha256": "development_sha256",
        "required_artifacts_manifest_sha256": ("required_artifacts_manifest_sha256"),
        "upload_policy_sha256": "upload_allowlist_sha256",
    }
    for frozen_name, spec_name in frozen_to_spec.items():
        if (
            _require_sha256(
                frozen_identity.get(frozen_name),
                f"continuation {frozen_name}",
            )
            != authority_hashes[spec_name]
        ):
            raise H6DestinationAdmissionError(
                f"continuation authority disagrees with run spec: {frozen_name}"
            )

    for program_name in ("admission_program_sha256", "terminal_wrapper_sha256"):
        _require_sha256(
            frozen_identity.get(program_name),
            f"continuation {program_name}",
        )

    _require_guard_destination_spec(spec, authority_hashes)

    predecessor_binding_payload, predecessor_binding_identity = _read_verified_file(
        predecessor_binding_path,
        _EXPECTED_PREDECESSOR_BINDING_SHA256,
        role="H6 predecessor recovery binding",
    )
    tracked[predecessor_binding_path] = predecessor_binding_identity
    predecessor_binding = _parse_json(
        predecessor_binding_payload, role="H6 predecessor recovery binding"
    )
    bound_predecessor = predecessor_binding.get("predecessor")
    bound_source = predecessor_binding.get("source")
    if (
        predecessor_authority.get("recovery_binding_sha256")
        != predecessor_binding_identity["sha256"]
        or predecessor_binding.get("schema")
        != "nano.h6.predecessor-recovery-binding.v1"
        or not isinstance(bound_predecessor, Mapping)
        or not isinstance(bound_source, Mapping)
        or bound_source.get("required_state") != "stopped"
        or bound_source.get("access_policy_for_continuation")
        != "lineage_only_no_restart"
        or predecessor_authority.get("run_id") != bound_predecessor.get("run_id")
        or predecessor_authority.get("phase") != bound_predecessor.get("phase")
        or predecessor_authority.get("source_pod_id") != bound_source.get("pod_id")
        or predecessor_authority.get("source_machine_id")
        != bound_source.get("machine_id")
        or predecessor_authority.get("ledger_tail_sha256")
        != bound_predecessor.get("ledger_tail_sha256")
    ):
        raise H6DestinationAdmissionError(
            "predecessor recovery authority chain is inconsistent"
        )

    predecessor_tail = _require_sha256(
        bound_predecessor.get("ledger_tail_sha256"), "predecessor ledger tail"
    )
    (
        predecessor_first,
        _predecessor_events,
        predecessor_ledger,
        predecessor_events_identity,
    ) = _verify_ledger(
        predecessor_run_events_path,
        expected_tail_sha256=predecessor_tail,
    )
    tracked[predecessor_run_events_path] = predecessor_events_identity
    predecessor_spec_payload, predecessor_spec_identity = _read_verified_file(
        predecessor_run_spec_path,
        bound_predecessor.get("run_spec_sha256"),
        role="predecessor run spec",
    )
    tracked[predecessor_run_spec_path] = predecessor_spec_identity
    predecessor_spec = _parse_json(
        predecessor_spec_payload, role="predecessor run spec"
    )
    predecessor_genesis = predecessor_first.get("payload")
    if (
        predecessor_events_identity["sha256"]
        != _require_sha256(
            bound_predecessor.get("run_events_sha256"),
            "predecessor run-events file",
        )
        or predecessor_first.get("event") != "RUN_INITIALIZED"
        or not isinstance(predecessor_genesis, Mapping)
        or predecessor_genesis.get("spec") != predecessor_spec
        or predecessor_spec.get("run_id") != bound_predecessor.get("run_id")
    ):
        raise H6DestinationAdmissionError(
            "predecessor run files do not match the recovery binding"
        )

    runtime_payload, runtime_identity = _read_verified_file(
        runtime_authority_path,
        _EXPECTED_RUNTIME_AUTHORITY_SHA256,
        role="H6 evaluation runtime authority",
    )
    tracked[runtime_authority_path] = runtime_identity
    runtime_authority = _parse_json(
        runtime_payload, role="H6 evaluation runtime authority"
    )
    required_destination = runtime_authority.get("required_destination")
    source_policy = runtime_authority.get("source_policy")
    expected_runtime_fields = {
        "configured_gpu_id",
        "runtime_gpu_name",
        "runtime_gpu_count",
        "image_id",
        "workspace_mount_path",
        "python",
        "torch",
        "cuda",
        "tokenizers",
        "platform",
        "cublas_workspace_config",
    }
    if (
        runtime_authority.get("schema") != "nano.h6.evaluation-runtime-authority.v1"
        or runtime_authority.get("scope") != "one_shot_known_development_evaluation"
        or runtime_authority.get("provider") != "runpod"
        or runtime_authority.get("cloud_tier") != "secure"
        or runtime_authority.get("storage_class") != "runpod_network_volume"
        or tuple(runtime_authority.get("allowed_data_center_ids", ()))
        != _AUTHORIZED_DATA_CENTERS
        or runtime_authority.get("fresh_or_private_data_authorized") is not False
        or not isinstance(required_destination, Mapping)
        or set(required_destination) != expected_runtime_fields
        or required_destination.get("configured_gpu_id") != _EXPECTED_CONFIGURED_GPU_ID
        or required_destination.get("runtime_gpu_name") != _EXPECTED_CONFIGURED_GPU_ID
        or required_destination.get("runtime_gpu_count") != 1
        or required_destination.get("image_id") != _EXPECTED_IMAGE_ID
        or required_destination.get("workspace_mount_path") != "/workspace"
        or not isinstance(source_policy, Mapping)
        or source_policy.get("pod_id") != bound_source.get("pod_id")
        or source_policy.get("required_state") != "stopped"
        or source_policy.get("restart_authorized") is not False
        or source_policy.get("termination_authorized") is not False
    ):
        raise H6DestinationAdmissionError("H6 exact runtime authority is invalid")

    required_payload, required_identity = _read_verified_file(
        required_artifacts_path,
        _EXPECTED_REQUIRED_ARTIFACTS_SHA256,
        role="H6 required-artifacts manifest",
    )
    tracked[required_artifacts_path] = required_identity
    required_artifacts = _parse_json(
        required_payload, role="H6 required-artifacts manifest"
    )
    artifacts = required_artifacts.get("artifacts")
    if (
        frozen_identity.get("required_artifacts_manifest_sha256")
        != required_identity["sha256"]
        or required_artifacts.get("schema") != "nano.required-artifacts.v1"
        or not isinstance(artifacts, list)
        or spec.get("required_artifacts") != artifacts
    ):
        raise H6DestinationAdmissionError(
            "H6 required-artifacts authority chain is inconsistent"
        )
    artifact_by_role: dict[str, Mapping[str, Any]] = {}
    artifact_ids: set[str] = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "artifact_id",
            "class",
            "role",
            "expected_sha256",
        }:
            raise H6DestinationAdmissionError(
                f"required artifact {index} has an invalid shape"
            )
        role = _require_text(artifact.get("role"), f"required artifact {index} role")
        artifact_id = _require_text(
            artifact.get("artifact_id"), f"required artifact {index} ID"
        )
        if role in artifact_by_role or artifact_id in artifact_ids:
            raise H6DestinationAdmissionError(
                "required artifact roles and IDs must be unique"
            )
        artifact_by_role[role] = artifact
        artifact_ids.add(artifact_id)
    if set(artifact_by_role) != (
        _EXPECTED_FIXED_ARTIFACT_ROLES | _EXPECTED_PENDING_ARTIFACT_ROLES
    ):
        raise H6DestinationAdmissionError(
            "required-artifact role inventory is not exact"
        )
    for role in _EXPECTED_FIXED_ARTIFACT_ROLES:
        artifact = artifact_by_role[role]
        _require_sha256(artifact.get("expected_sha256"), f"artifact {role}")
        if artifact.get("class") != "training":
            raise H6DestinationAdmissionError(
                f"fixed artifact has an invalid class: {role}"
            )
    for role in _EXPECTED_PENDING_ARTIFACT_ROLES:
        artifact = artifact_by_role[role]
        if (
            artifact.get("class") != "result"
            or artifact.get("expected_sha256") is not None
        ):
            raise H6DestinationAdmissionError(
                f"pending artifact has an invalid boundary: {role}"
            )

    selection_payload, selection_identity = _read_verified_file(
        destination_selection_path,
        _EXPECTED_DESTINATION_SELECTION_SHA256,
        role="H6 destination-selection evidence",
    )
    tracked[destination_selection_path] = selection_identity
    destination_selection = _parse_json(
        selection_payload, role="H6 destination-selection evidence"
    )
    decision = destination_selection.get("decision")
    if (
        destination_policy.get("selection_evidence") != destination_selection_path.name
        or destination_selection.get("schema")
        != "nano.h6.destination-selection-evidence.v1"
        or destination_selection.get("provider") != "runpod"
        or not isinstance(decision, Mapping)
        or decision.get("selected_first") != _AUTHORIZED_DATA_CENTERS[0]
        or tuple(decision.get("authorized_for_this_continuation", ()))
        != _AUTHORIZED_DATA_CENTERS
    ):
        raise H6DestinationAdmissionError(
            "H6 destination-selection evidence is invalid"
        )

    return {
        "authorization": authorization,
        "predecessor_binding": predecessor_binding,
        "predecessor": {
            "run_id": bound_predecessor["run_id"],
            "phase": bound_predecessor["phase"],
            "ledger": predecessor_ledger,
            "source_pod_id": bound_source["pod_id"],
            "source_machine_id": bound_source["machine_id"],
            "source_required_state": "stopped",
            "source_access_policy": "lineage_only_no_restart",
        },
        "runtime_authority": runtime_authority,
        "required_artifacts": required_artifacts,
        "required_artifact_by_role": artifact_by_role,
        "destination_selection": destination_selection,
        "identities": {
            "continuation_authorization": authorization_identity,
            "predecessor_binding": predecessor_binding_identity,
            "predecessor_run_spec": predecessor_spec_identity,
            "predecessor_run_events": predecessor_events_identity,
            "runtime_authority": runtime_identity,
            "required_artifacts": required_identity,
            "destination_selection": selection_identity,
        },
    }, tracked


def _require_guard_network_volume_identity(
    storage_identity: object,
    *,
    network_volume_id: object,
) -> None:
    """Require the native network-volume identity emitted by the guard."""

    descriptor = (
        storage_identity.get("descriptor")
        if isinstance(storage_identity, Mapping)
        else None
    )
    if (
        not isinstance(storage_identity, Mapping)
        or not isinstance(descriptor, Mapping)
        or set(storage_identity)
        != {
            "kind",
            "identity_id",
            "provider_resource_id",
            "descriptor",
        }
        or storage_identity.get("kind") != "runpod_network_volume_v1"
        or storage_identity.get("identity_id") != network_volume_id
        or storage_identity.get("provider_resource_id") != network_volume_id
        or dict(descriptor)
        != {
            "provider": "runpod",
            "resource_type": "network_volume",
            "resource_id": network_volume_id,
        }
    ):
        raise H6DestinationAdmissionError(
            "provider snapshot network-volume identity is invalid"
        )


def _guard_sanitized_observation_sha256(snapshot: Mapping[str, Any]) -> str:
    """Recompute the guard's self-digest for one provider observation."""

    claimed = _require_sha256(
        snapshot.get("sanitized_observation_sha256"),
        "provider sanitized observation",
    )
    unsigned = {
        key: value
        for key, value in snapshot.items()
        if key != "sanitized_observation_sha256"
    }
    if _digest(_canonical_json_bytes(unsigned).removesuffix(b"\n")) != claimed:
        raise H6DestinationAdmissionError(
            "provider snapshot sanitized observation hash is invalid"
        )
    return claimed


def _guard_operation_history(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pair guard operation starts/finishes and their bound observations."""

    active_operation_id: str | None = None
    operations: dict[str, dict[str, Any]] = {}
    for event in events:
        event_name = event.get("event")
        if event_name not in {
            "OPERATION_STARTED",
            "OPERATION_FINISHED",
            "PROVIDER_SNAPSHOT",
        }:
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            raise H6DestinationAdmissionError(f"guard {event_name} payload is invalid")
        if event_name == "OPERATION_STARTED":
            operation_id = _require_text(
                payload.get("operation_id"), "started operation ID"
            )
            if active_operation_id is not None or operation_id in operations:
                raise H6DestinationAdmissionError(
                    "guard operation history has overlapping or duplicate starts"
                )
            operations[operation_id] = {
                "start": payload,
                "start_event": event,
                "snapshots": [],
                "finish": None,
                "finish_event": None,
            }
            active_operation_id = operation_id
        elif event_name == "PROVIDER_SNAPSHOT":
            observation_operation_id = payload.get("observation_for_operation_id")
            if observation_operation_id is None:
                continue
            operation_id = _require_text(
                observation_operation_id,
                "provider-observation operation ID",
            )
            if active_operation_id != operation_id:
                raise H6DestinationAdmissionError(
                    "operation-bound provider snapshot is outside its active operation"
                )
            operations[operation_id]["snapshots"].append(event)
        else:
            operation_id = _require_text(
                payload.get("operation_id"), "finished operation ID"
            )
            if active_operation_id != operation_id:
                raise H6DestinationAdmissionError(
                    "guard operation finish does not match the active operation"
                )
            operation = operations[operation_id]
            if operation["finish"] is not None:
                raise H6DestinationAdmissionError(
                    "guard operation has duplicate finishes"
                )
            operation["finish"] = payload
            operation["finish_event"] = event
            active_operation_id = None
    if active_operation_id is not None:
        raise H6DestinationAdmissionError(
            "destination admission is blocked by an unresolved operation"
        )
    return operations


def _successful_operation(
    operations: Mapping[str, Mapping[str, Any]],
    operation_id: str,
    *,
    kind: str,
    resource_role: str,
    target_id: str | None = None,
    evidence_sha256: str | None = None,
) -> Mapping[str, Any]:
    operation = operations.get(operation_id)
    if not isinstance(operation, Mapping):
        raise H6DestinationAdmissionError(
            f"required successful {kind} operation is absent"
        )
    start = operation.get("start")
    finish = operation.get("finish")
    if (
        not isinstance(start, Mapping)
        or not isinstance(finish, Mapping)
        or start.get("kind") != kind
        or start.get("resource_role") != resource_role
        or finish.get("outcome") not in _SUCCESSFUL_OPERATION_OUTCOMES
        or (target_id is not None and start.get("target_id") != target_id)
        or (
            evidence_sha256 is not None
            and finish.get("evidence_sha256") != evidence_sha256
        )
    ):
        raise H6DestinationAdmissionError(
            f"required {kind} operation is not a matching successful operation"
        )
    return operation


def _require_destination_operation_lineage(
    *,
    events: Sequence[Mapping[str, Any]],
    destination_id: str,
    network_volume_id: str,
    creation_operation_id: str,
    rehydration_operation_id: str,
    attestation_sha256: str,
    final_volume_event: Mapping[str, Any],
    final_destination_event: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Bind create-volume, create-pod, and rehydration provenance."""

    operations = _guard_operation_history(events)
    creation = _successful_operation(
        operations,
        creation_operation_id,
        kind="create_pod",
        resource_role="destination",
    )
    rehydration = _successful_operation(
        operations,
        rehydration_operation_id,
        kind="rehydrate_destination",
        resource_role="destination",
        target_id=destination_id,
        evidence_sha256=attestation_sha256,
    )

    creation_snapshots = [
        snapshot
        for snapshot in creation["snapshots"]
        if isinstance(snapshot.get("payload"), Mapping)
        and snapshot["payload"].get("resource_role") == "destination"
        and snapshot["payload"].get("resource_type") == "pod"
        and snapshot["payload"].get("resource_id") == destination_id
        and snapshot["payload"].get("storage_resource_id") == network_volume_id
    ]
    if len(creation_snapshots) != 1:
        raise H6DestinationAdmissionError(
            "successful create_pod operation lacks one exact bound snapshot"
        )
    creation_snapshot = creation_snapshots[0]
    creation_payload = creation_snapshot["payload"]
    if creation["start"].get("target_id") != creation_payload.get("resource_name"):
        raise H6DestinationAdmissionError(
            "create_pod target does not match the created destination name"
        )
    _require_guard_network_volume_identity(
        creation_payload.get("storage_identity"),
        network_volume_id=network_volume_id,
    )

    volume_candidates: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    for operation_id, operation in operations.items():
        start = operation.get("start")
        finish = operation.get("finish")
        if (
            not isinstance(start, Mapping)
            or not isinstance(finish, Mapping)
            or start.get("kind") != "create_volume"
            or start.get("resource_role") != "network_volume"
            or finish.get("outcome") not in _SUCCESSFUL_OPERATION_OUTCOMES
        ):
            continue
        matching_snapshots = [
            snapshot
            for snapshot in operation["snapshots"]
            if isinstance(snapshot.get("payload"), Mapping)
            and snapshot["payload"].get("resource_role") == "network_volume"
            and snapshot["payload"].get("resource_type") == "network_volume"
            and snapshot["payload"].get("resource_id") == network_volume_id
        ]
        if len(matching_snapshots) == 1:
            volume_candidates.append((operation_id, operation, matching_snapshots[0]))
    if len(volume_candidates) != 1:
        raise H6DestinationAdmissionError(
            "destination volume lacks one exact successful create operation"
        )
    volume_operation_id, volume_creation, volume_creation_snapshot = volume_candidates[
        0
    ]
    if volume_creation["start"].get("target_id") != volume_creation_snapshot[
        "payload"
    ].get("resource_name"):
        raise H6DestinationAdmissionError(
            "create_volume target does not match the created volume name"
        )

    ordered_indexes = (
        volume_creation["start_event"]["index"],
        volume_creation_snapshot["index"],
        volume_creation["finish_event"]["index"],
        creation["start_event"]["index"],
        creation_snapshot["index"],
        creation["finish_event"]["index"],
        rehydration["start_event"]["index"],
        rehydration["finish_event"]["index"],
        final_volume_event["index"],
        final_destination_event["index"],
    )
    if tuple(sorted(ordered_indexes)) != ordered_indexes:
        raise H6DestinationAdmissionError(
            "destination creation, rehydration, and final observations are out of order"
        )

    def operation_receipt(
        operation_id: str,
        operation: Mapping[str, Any],
        snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "operation_id": operation_id,
            "kind": operation["start"]["kind"],
            "start_event_index": operation["start_event"]["index"],
            "start_event_sha256": operation["start_event"]["event_sha256"],
            "finish_event_index": operation["finish_event"]["index"],
            "finish_event_sha256": operation["finish_event"]["event_sha256"],
            "outcome": operation["finish"]["outcome"],
        }
        if snapshot is not None:
            result.update(
                {
                    "provider_snapshot_event_index": snapshot["index"],
                    "provider_snapshot_event_sha256": snapshot["event_sha256"],
                }
            )
        return result

    return {
        "create_volume": operation_receipt(
            volume_operation_id,
            volume_creation,
            volume_creation_snapshot,
        ),
        "create_pod": operation_receipt(
            creation_operation_id,
            creation,
            creation_snapshot,
        ),
        "rehydrate_destination": operation_receipt(
            rehydration_operation_id,
            rehydration,
        ),
    }


def _authenticate_admission_sync_preparation(
    *,
    authority: Mapping[str, Any],
    isolated_execution_contract: Mapping[str, Any],
    final_volume_event: Mapping[str, Any],
    final_destination_event: Mapping[str, Any],
    prepared_event: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the guard-native, immutable admission mirror boundary."""

    payload = prepared_event.get("payload")
    volume = final_volume_event.get("payload")
    destination = final_destination_event.get("payload")
    expected_keys = {
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
        "isolated_execution_contract",
        "remote_root",
        "relative_paths",
        "transfer_protocol",
        "transfer_provider_state_mutated",
        "transfer_guard_state_mutated",
    }
    if (
        prepared_event.get("event") != _ADMISSION_SYNC_EVENT
        or not isinstance(payload, Mapping)
        or set(payload) != expected_keys
        or not isinstance(volume, Mapping)
        or not isinstance(destination, Mapping)
    ):
        raise H6DestinationAdmissionError(
            "ledger admission-sync preparation is invalid"
        )

    spec = authority["spec"]
    final_destination_sha256 = _require_sha256(
        final_destination_event.get("event_sha256"),
        "final destination snapshot event",
    )
    prepared_sha256 = _require_sha256(
        prepared_event.get("event_sha256"), "admission-sync prepared event"
    )
    transaction_id = _require_text(
        payload.get("transaction_id"), "admission-sync transaction ID"
    )
    if _DESTINATION_ID_RE.fullmatch(transaction_id) is None:
        raise H6DestinationAdmissionError("admission-sync transaction ID is invalid")

    relative_paths = payload.get("relative_paths")
    expected_path_roles = {
        "run_spec",
        "run_events",
        "destination_attestation",
        "admission_receipt",
    }
    if not isinstance(relative_paths, Mapping) or set(relative_paths) != (
        expected_path_roles
    ):
        raise H6DestinationAdmissionError(
            "admission-sync relative-path inventory is invalid"
        )
    normalized_paths = {
        role: _safe_relative(value, role=f"admission-sync {role}")
        for role, value in relative_paths.items()
    }
    if len(set(normalized_paths.values())) != len(normalized_paths):
        raise H6DestinationAdmissionError(
            "admission-sync relative paths must be unique"
        )

    remote_root = _require_text(payload.get("remote_root"), "admission-sync root")
    remote_root_path = PurePosixPath(remote_root)
    if (
        not remote_root_path.is_absolute()
        or remote_root_path.as_posix() != remote_root
        or any(part in {"", ".", ".."} for part in remote_root_path.parts)
    ):
        raise H6DestinationAdmissionError(
            "admission-sync root must be a canonical absolute POSIX path"
        )

    expected_values = {
        "schema": _ADMISSION_SYNC_SCHEMA,
        "run_id": spec["run_id"],
        "phase": "REPORTS_FROZEN",
        "ledger_tail_before": final_destination_sha256,
        "final_network_volume_snapshot_event_index": final_volume_event["index"],
        "final_network_volume_snapshot_event_sha256": final_volume_event[
            "event_sha256"
        ],
        "final_destination_snapshot_event_index": final_destination_event["index"],
        "final_destination_snapshot_event_sha256": final_destination_sha256,
        "destination_resource_id": destination.get("resource_id"),
        "network_volume_resource_id": volume.get("resource_id"),
        "authorization_envelope_sha256": spec.get("authorization_envelope_sha256"),
        "upload_allowlist_sha256": spec.get("upload_allowlist_sha256"),
        "runtime_authority_sha256": spec.get("runtime_authority_sha256"),
        "isolated_execution_contract": dict(isolated_execution_contract),
        "transfer_protocol": _ADMISSION_SYNC_PROTOCOL,
        "transfer_provider_state_mutated": False,
        "transfer_guard_state_mutated": False,
    }
    if (
        any(payload.get(key) != value for key, value in expected_values.items())
        or prepared_event.get("previous_sha256") != final_destination_sha256
        or prepared_sha256 != authority["ledger"]["tail_sha256"]
    ):
        raise H6DestinationAdmissionError(
            "admission-sync preparation does not bind the final destination state"
        )

    return {
        "schema": _ADMISSION_SYNC_SCHEMA,
        "transaction_id": transaction_id,
        "prepared_event_index": prepared_event["index"],
        "prepared_event_sha256": prepared_sha256,
        "ledger_tail_before": final_destination_sha256,
        "remote_root": remote_root,
        "relative_paths": normalized_paths,
        "isolated_execution_contract": dict(isolated_execution_contract),
        "transfer_protocol": _ADMISSION_SYNC_PROTOCOL,
    }


def _require_admission_sync_paths(
    sync: Mapping[str, Any],
    *,
    upload_root: Path,
    run_spec_path: Path,
    run_events_path: Path,
    destination_attestation_path: Path,
    output_path: Path,
) -> None:
    """Bind prepared remote names to the files this admission actually reads."""

    root = Path(os.path.abspath(os.fspath(upload_root)))
    expected_paths = {
        "run_spec": _relative_to_root(run_spec_path, root, role="run spec"),
        "run_events": _relative_to_root(run_events_path, root, role="run events"),
        "destination_attestation": _relative_to_root(
            destination_attestation_path,
            root,
            role="destination attestation",
        ),
        "admission_receipt": _relative_to_root(
            output_path,
            root,
            role="destination admission receipt",
        ),
    }
    if (
        sync.get("remote_root") != root.as_posix()
        or sync.get("relative_paths") != expected_paths
    ):
        raise H6DestinationAdmissionError(
            "admission-sync preparation paths differ from the authenticated invocation"
        )


def _authenticate_destination_attestation(
    *,
    authority: Mapping[str, Any],
    controls: Mapping[str, Any],
    attestation_path: Path,
    destination_id: str,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    """Bind exact provider and measured-runtime evidence to the current tail."""

    payload, identity = _read_regular_file(
        attestation_path, role="H6 destination attestation"
    )
    attestation = _parse_json(payload, role="H6 destination attestation")
    expected_top_level = {
        "schema",
        "attestation_id",
        "run_id",
        "observation_operation_id",
        "creation_operation_id",
        "rehydration_operation_id",
        "observed_at_utc",
        "provider",
        "destination",
        "runtime",
    }
    expected_destination_fields = {
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
    expected_runtime_fields = {
        "python",
        "torch",
        "tokenizers",
        "cuda",
        "gpu",
        "gpu_count",
        "cublas_workspace_config",
        "platform",
    }
    destination = attestation.get("destination")
    runtime = attestation.get("runtime")
    required_destination = controls["runtime_authority"]["required_destination"]
    if (
        set(attestation) != expected_top_level
        or attestation.get("schema") != "nano.h6.destination-attestation.v1"
        or attestation.get("run_id") != authority["spec"].get("run_id")
        or attestation.get("provider") != "runpod"
        or attestation.get("observation_operation_id") is not None
        or not isinstance(destination, Mapping)
        or set(destination) != expected_destination_fields
        or not isinstance(runtime, Mapping)
        or set(runtime) != expected_runtime_fields
        or destination.get("pod_id") != destination_id
        or destination.get("data_center_id") != _AUTHORIZED_DATA_CENTERS[0]
        or destination.get("cloud_type") != "SECURE"
        or destination.get("secure_cloud") is not True
        or destination.get("lifecycle_state") != "running"
        or destination.get("image_id") != required_destination["image_id"]
        or destination.get("configured_gpu_id")
        != required_destination["configured_gpu_id"]
        or destination.get("provider_gpu_model") != _EXPECTED_SPEC_GPU_MODEL
        or destination.get("configured_gpu_count") != 1
        or destination.get("runtime_gpu_name")
        != required_destination["runtime_gpu_name"]
        or destination.get("runtime_gpu_count")
        != required_destination["runtime_gpu_count"]
        or destination.get("workspace_mount_path")
        != required_destination["workspace_mount_path"]
        or runtime.get("gpu") != destination.get("runtime_gpu_name")
        or runtime.get("gpu_count") != destination.get("runtime_gpu_count")
    ):
        raise H6DestinationAdmissionError(
            "H6 destination attestation has an invalid exact boundary"
        )
    _require_text(attestation.get("attestation_id"), "destination attestation ID")
    creation_operation_id = _require_text(
        attestation.get("creation_operation_id"),
        "destination creation operation ID",
    )
    rehydration_operation_id = _require_text(
        attestation.get("rehydration_operation_id"),
        "destination rehydration operation ID",
    )
    _require_text(attestation.get("observed_at_utc"), "destination observation time")
    _require_text(destination.get("machine_id"), "destination machine ID")
    _require_text(destination.get("network_volume_id"), "destination network-volume ID")
    observed_image_digest = destination.get("observed_image_digest")
    if observed_image_digest is not None:
        if not isinstance(
            observed_image_digest, str
        ) or not observed_image_digest.startswith("sha256:"):
            raise H6DestinationAdmissionError(
                "observed destination image digest is invalid"
            )
        _require_sha256(
            observed_image_digest.removeprefix("sha256:"),
            "observed destination image digest",
        )

    runtime_mapping = {
        "python": "python",
        "torch": "torch",
        "tokenizers": "tokenizers",
        "cuda": "cuda",
        "gpu": "runtime_gpu_name",
        "gpu_count": "runtime_gpu_count",
        "cublas_workspace_config": "cublas_workspace_config",
        "platform": "platform",
    }
    if any(
        runtime.get(observed_name) != required_destination[required_name]
        for observed_name, required_name in runtime_mapping.items()
    ):
        raise H6DestinationAdmissionError(
            "destination attestation does not match the runtime authority"
        )

    events = authority["ledger_events"]
    if (
        len(events) < 3
        or events[-3].get("event") != "PROVIDER_SNAPSHOT"
        or events[-2].get("event") != "PROVIDER_SNAPSHOT"
        or events[-1].get("event") != _ADMISSION_SYNC_EVENT
        or not isinstance(events[-3].get("payload"), Mapping)
        or not isinstance(events[-2].get("payload"), Mapping)
    ):
        raise H6DestinationAdmissionError(
            "ledger does not end in final volume and destination snapshots followed "
            "by an admission-sync preparation"
        )
    final_volume_event = events[-3]
    final_destination_event = events[-2]
    prepared_event = events[-1]
    volume_snapshot = final_volume_event["payload"]
    snapshot = final_destination_event["payload"]
    matching = [
        event
        for event in events
        if event.get("event") == "PROVIDER_SNAPSHOT"
        and isinstance(event.get("payload"), Mapping)
        and event["payload"].get("observation_evidence_sha256") == identity["sha256"]
    ]
    if len(matching) != 1 or matching[0] is not final_destination_event:
        raise H6DestinationAdmissionError(
            "destination attestation is not the unique current provider snapshot"
        )
    snapshot_event = final_destination_event
    event_bindings = {
        "provider": "runpod",
        "resource_type": "pod",
        "resource_role": "destination",
        "resource_id": destination["pod_id"],
        "machine_id": destination["machine_id"],
        "storage_resource_id": destination["network_volume_id"],
        "data_center_id": destination["data_center_id"],
        "image_id": destination["image_id"],
        "configured_gpu_count": destination["configured_gpu_count"],
        "gpu_model": destination["provider_gpu_model"],
        "gpu_count": destination["runtime_gpu_count"],
        "lifecycle_state": "running",
        "secure_cloud": True,
        "workspace_state": "accessible",
        "observation_for_operation_id": None,
        "observed_at_utc": attestation["observed_at_utc"],
        "runtime_sha256": controls["identities"]["runtime_authority"]["sha256"],
    }
    if any(snapshot.get(key) != value for key, value in event_bindings.items()):
        raise H6DestinationAdmissionError(
            "provider snapshot does not match the destination attestation"
        )
    _require_guard_network_volume_identity(
        snapshot.get("storage_identity"),
        network_volume_id=destination["network_volume_id"],
    )
    destination_observation_sha256 = _guard_sanitized_observation_sha256(snapshot)

    if (
        volume_snapshot.get("provider") != "runpod"
        or volume_snapshot.get("resource_role") != "network_volume"
        or volume_snapshot.get("resource_type") != "network_volume"
        or volume_snapshot.get("resource_id") != destination["network_volume_id"]
        or volume_snapshot.get("storage_resource_id")
        != destination["network_volume_id"]
        or volume_snapshot.get("data_center_id") != destination["data_center_id"]
        or volume_snapshot.get("lifecycle_state") not in _USABLE_NETWORK_VOLUME_STATES
        or volume_snapshot.get("observation_for_operation_id") is not None
        or volume_snapshot.get("secure_cloud") not in {None, True}
    ):
        raise H6DestinationAdmissionError(
            "final provider network-volume snapshot is invalid"
        )
    _require_sha256(
        volume_snapshot.get("observation_evidence_sha256"),
        "network-volume observation evidence",
    )
    _require_text(
        volume_snapshot.get("observed_at_utc"),
        "network-volume observation time",
    )
    _require_guard_network_volume_identity(
        volume_snapshot.get("storage_identity"),
        network_volume_id=destination["network_volume_id"],
    )
    volume_observation_sha256 = _guard_sanitized_observation_sha256(volume_snapshot)
    admission_sync = _authenticate_admission_sync_preparation(
        authority=authority,
        isolated_execution_contract=controls["authorization"][
            "isolated_execution_contract"
        ],
        final_volume_event=final_volume_event,
        final_destination_event=final_destination_event,
        prepared_event=prepared_event,
    )

    operation_lineage = _require_destination_operation_lineage(
        events=events,
        destination_id=destination_id,
        network_volume_id=destination["network_volume_id"],
        creation_operation_id=creation_operation_id,
        rehydration_operation_id=rehydration_operation_id,
        attestation_sha256=identity["sha256"],
        final_volume_event=final_volume_event,
        final_destination_event=final_destination_event,
    )

    return {
        "attestation_id": attestation["attestation_id"],
        "observation_operation_id": None,
        "creation_operation_id": creation_operation_id,
        "rehydration_operation_id": rehydration_operation_id,
        "identity": identity,
        "operation_lineage": operation_lineage,
        "network_volume_provider_snapshot_event_index": final_volume_event["index"],
        "network_volume_provider_snapshot_event_sha256": final_volume_event[
            "event_sha256"
        ],
        "network_volume_observation_sha256": volume_observation_sha256,
        "network_volume_observation_evidence_sha256": volume_snapshot[
            "observation_evidence_sha256"
        ],
        "provider_snapshot_event_index": snapshot_event["index"],
        "provider_snapshot_event_sha256": snapshot_event["event_sha256"],
        "destination_observation_sha256": destination_observation_sha256,
        "destination": dict(destination),
        "runtime": dict(runtime),
        "admission_sync": admission_sync,
    }, {attestation_path: identity}


def _assert_isolated_environment_has_no_staging(environment_root: Path) -> None:
    """Reject development/fresh/private staging names inside the derived venv."""

    def prohibited(parts: Sequence[str]) -> bool:
        for raw_part in parts:
            part = raw_part.casefold().lstrip(".")
            stem = PurePosixPath(part).stem
            for candidate in (part, stem):
                if candidate in _PROHIBITED_STAGING_COMPONENTS or any(
                    candidate.startswith(f"{prefix}{separator}")
                    for prefix in _PROHIBITED_STAGING_COMPONENTS
                    for separator in ("-", "_")
                ):
                    return True
        return False

    try:
        for current, directory_names, file_names in os.walk(
            environment_root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            relative_current = current_path.relative_to(environment_root)
            for name in tuple(directory_names) + tuple(file_names):
                relative = relative_current / name
                if prohibited(relative.parts):
                    raise H6DestinationAdmissionError(
                        "isolated H6 environment contains prohibited staging: "
                        f"{relative.as_posix()}"
                    )
    except H6DestinationAdmissionError:
        raise
    except OSError as exc:
        raise H6DestinationAdmissionError(
            "isolated H6 environment could not be inspected safely"
        ) from exc


def _authenticate_materialized_tree(
    *,
    authority: Mapping[str, Any],
    source_root: Path,
    report_root: Path,
    admission_source: Path,
    admission_source_identity: Mapping[str, Any],
    terminal_wrapper_path: Path,
    terminal_wrapper_identity: Mapping[str, Any],
) -> dict[Path, dict[str, Any]]:
    """Reject any scientific/data file outside the authenticated rehydration set."""

    expected_source_root = (
        Path("/workspace") / authority["readiness"]["archive"]["root"]
    )
    if (
        source_root.absolute() != expected_source_root
        or source_root.resolve() != expected_source_root
        or report_root.absolute() != expected_source_root
        or report_root.resolve() != expected_source_root
        or admission_source.resolve()
        != (source_root / "nano_ai/training/admit_evidence_query_h6.py").resolve()
        or terminal_wrapper_path.resolve()
        != (source_root / "nano_ai/training/h6_terminal_result.py").resolve()
    ):
        raise H6DestinationAdmissionError(
            "H6 materialized tree is not the exact /workspace archive root"
        )
    try:
        root_stat = source_root.lstat()
        environment_stat = (source_root / ".h6-venv").lstat()
    except OSError as exc:
        raise H6DestinationAdmissionError(
            "H6 source root or isolated environment is unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
        or not stat.S_ISDIR(environment_stat.st_mode)
        or stat.S_ISLNK(environment_stat.st_mode)
    ):
        raise H6DestinationAdmissionError(
            "H6 source root or isolated environment is not a real directory"
        )
    _assert_isolated_environment_has_no_staging(source_root / ".h6-venv")

    expected: dict[str, Mapping[str, Any]] = {
        relative: identity
        for relative, identity in authority["package_members"].items()
    }
    expected["BUNDLE_MANIFEST.json"] = authority["bundle_manifest_identity"]
    expected["nano_ai/training/admit_evidence_query_h6.py"] = admission_source_identity
    expected["nano_ai/training/h6_terminal_result.py"] = terminal_wrapper_identity
    freeze = authority["freeze"]
    reports = freeze.get("reports")
    checksum = freeze.get("training_report_checksum_file")
    if not isinstance(reports, Mapping) or not isinstance(checksum, Mapping):
        raise H6DestinationAdmissionError(
            "H6 freeze cannot define an exact materialized tree"
        )
    for report_relative, report in reports.items():
        relative = _safe_relative(report_relative, role="materialized H6 report")
        if not isinstance(report, Mapping):
            raise H6DestinationAdmissionError(
                "materialized H6 report authority is invalid"
            )
        expected[relative] = _identity_record(
            {"bytes": report.get("bytes"), "sha256": report.get("sha256")},
            role=f"materialized H6 report {relative}",
        )
        candidate = report.get("candidate")
        if not isinstance(candidate, Mapping):
            raise H6DestinationAdmissionError(
                "materialized H6 checkpoint authority is invalid"
            )
        candidate_relative = (
            PurePosixPath(relative).parent
            / _safe_relative(candidate.get("filename"), role="H6 checkpoint name")
        ).as_posix()
        if len(PurePosixPath(candidate_relative).parts) != 3:
            raise H6DestinationAdmissionError(
                "materialized H6 checkpoint must be directly beside its report"
            )
        expected[candidate_relative] = _identity_record(
            {
                "bytes": candidate.get("bytes"),
                "sha256": candidate.get("sha256"),
            },
            role=f"materialized H6 checkpoint {candidate_relative}",
        )
    checksum_relative = _safe_relative(
        checksum.get("path"), role="materialized H6 checksum file"
    )
    expected[checksum_relative] = _identity_record(
        {"bytes": checksum.get("bytes"), "sha256": checksum.get("sha256")},
        role="materialized H6 checksum file",
    )

    observed: set[str] = set()
    try:
        for current, directory_names, file_names in os.walk(
            source_root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            relative_current = current_path.relative_to(source_root)
            if relative_current == Path(".") and ".h6-venv" in directory_names:
                directory_names.remove(".h6-venv")
            for directory_name in directory_names:
                directory_path = current_path / directory_name
                directory_stat = directory_path.lstat()
                if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
                    directory_stat.st_mode
                ):
                    raise H6DestinationAdmissionError(
                        f"materialized H6 tree has an unsafe directory: {directory_path}"
                    )
            for file_name in file_names:
                file_path = current_path / file_name
                file_stat = file_path.lstat()
                if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(
                    file_stat.st_mode
                ):
                    raise H6DestinationAdmissionError(
                        f"materialized H6 tree has an unsafe file: {file_path}"
                    )
                observed.add(file_path.relative_to(source_root).as_posix())
    except H6DestinationAdmissionError:
        raise
    except OSError as exc:
        raise H6DestinationAdmissionError(
            "materialized H6 tree could not be inspected safely"
        ) from exc
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        extra = sorted(observed - set(expected))
        raise H6DestinationAdmissionError(
            f"materialized H6 tree is not exact; missing={missing}, extra={extra}"
        )

    tracked: dict[Path, dict[str, Any]] = {}
    for relative, expected_identity in sorted(expected.items()):
        path = source_root / relative
        _payload, identity = _read_verified_file(
            path,
            expected_identity.get("sha256"),
            role=f"materialized H6 file {relative}",
            expected_bytes=expected_identity.get("bytes"),
        )
        tracked[path] = identity
    return tracked


def _static_upload_files(
    *,
    authority: Mapping[str, Any],
    controls: Mapping[str, Any],
    package_path: Path,
    preregistration_path: Path,
    freeze_path: Path,
    readiness_path: Path,
    runtime_authority_path: Path,
    destination_selection_path: Path,
    predecessor_binding_path: Path,
    required_artifacts_path: Path,
    admission_source: Path,
    admission_source_identity: Mapping[str, Any],
    terminal_wrapper_path: Path,
    terminal_wrapper_identity: Mapping[str, Any],
    source_root: Path,
    materialized_files: Mapping[Path, Mapping[str, Any]],
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    """Build the exact 16-file static upload authority from frozen records."""

    normalized_materialized = {
        Path(os.path.abspath(os.fspath(path))): identity
        for path, identity in materialized_files.items()
    }

    def materialized(relative: str, role: str) -> tuple[Path, Mapping[str, Any]]:
        path = Path(
            os.path.abspath(
                os.fspath(source_root / _safe_relative(relative, role=role))
            )
        )
        try:
            return path, normalized_materialized[path]
        except KeyError as exc:
            raise H6DestinationAdmissionError(
                f"static upload lacks materialized identity: {role}"
            ) from exc

    static: dict[str, tuple[Path, Mapping[str, Any]]] = {
        "package": (
            package_path,
            authority["authority_identities"]["package"],
        ),
        "preregistration": (
            preregistration_path,
            authority["authority_identities"]["preregistration"],
        ),
        "training_report_freeze": (
            freeze_path,
            authority["authority_identities"]["training_report_freeze"],
        ),
        "readiness": (
            readiness_path,
            authority["authority_identities"]["readiness"],
        ),
        "runtime_authority": (
            runtime_authority_path,
            controls["identities"]["runtime_authority"],
        ),
        "destination_selection": (
            destination_selection_path,
            controls["identities"]["destination_selection"],
        ),
        "predecessor_binding": (
            predecessor_binding_path,
            controls["identities"]["predecessor_binding"],
        ),
        "required_artifacts": (
            required_artifacts_path,
            controls["identities"]["required_artifacts"],
        ),
        "admission_program": (
            admission_source,
            admission_source_identity,
        ),
        "terminal_wrapper": (
            terminal_wrapper_path,
            terminal_wrapper_identity,
        ),
    }
    reports = authority["freeze"].get("reports")
    checksum = authority["freeze"].get("training_report_checksum_file")
    if not isinstance(reports, Mapping) or not isinstance(checksum, Mapping):
        raise H6DestinationAdmissionError(
            "frozen H6 recovered artifact inventory is invalid"
        )
    for report_relative, report in reports.items():
        if not isinstance(report, Mapping):
            raise H6DestinationAdmissionError(
                "frozen H6 report upload record is invalid"
            )
        seed = report.get("seed")
        if seed not in _TRAINING_SEEDS:
            raise H6DestinationAdmissionError("frozen H6 report upload seed is invalid")
        static[f"training_report_seed_{seed}"] = materialized(
            report_relative, f"static report seed {seed}"
        )
        candidate = report.get("candidate")
        if not isinstance(candidate, Mapping):
            raise H6DestinationAdmissionError(
                "frozen H6 checkpoint upload record is invalid"
            )
        filename = _safe_relative(
            candidate.get("filename"), role=f"static checkpoint seed {seed}"
        )
        if len(PurePosixPath(filename).parts) != 1:
            raise H6DestinationAdmissionError(
                "static checkpoint filename must have one path component"
            )
        candidate_relative = (
            PurePosixPath(_safe_relative(report_relative, role="static report"))
            .parent.joinpath(filename)
            .as_posix()
        )
        static[f"selected_checkpoint_seed_{seed}"] = materialized(
            candidate_relative, f"static checkpoint seed {seed}"
        )
    checksum_relative = _safe_relative(
        checksum.get("path"), role="static report checksum"
    )
    static["training_report_checksum_manifest"] = materialized(
        checksum_relative, "static report checksum"
    )
    static["vocabulary"] = materialized("sft/tokenizer.json", "static vocabulary")
    if set(static) != {
        "package",
        "preregistration",
        "training_report_freeze",
        "readiness",
        "runtime_authority",
        "destination_selection",
        "predecessor_binding",
        "required_artifacts",
        "admission_program",
        "terminal_wrapper",
        "training_report_seed_20260805",
        "training_report_seed_20260806",
        "selected_checkpoint_seed_20260805",
        "selected_checkpoint_seed_20260806",
        "training_report_checksum_manifest",
        "vocabulary",
    }:
        raise H6DestinationAdmissionError(
            "static H6 upload semantic inventory is not exact"
        )
    return static


def _require_upload_policy_header(
    policy: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> str:
    """Authenticate the policy boundary and its one-way authority hash DAG."""

    expected_policy_keys = {
        "schema",
        "policy_id",
        "scope",
        "authorities",
        "exact_static_files",
        "dynamic_authenticated_files",
        "prohibited_before_dev_release",
        "no_extra_files",
    }
    scope = policy.get("scope")
    exact_authorities = {
        "runtime_authority_sha256": controls["identities"]["runtime_authority"][
            "sha256"
        ],
        "destination_selection_sha256": controls["identities"]["destination_selection"][
            "sha256"
        ],
        "predecessor_binding_sha256": controls["identities"]["predecessor_binding"][
            "sha256"
        ],
        "required_artifacts_manifest_sha256": controls["identities"][
            "required_artifacts"
        ]["sha256"],
    }
    if (
        set(policy) != expected_policy_keys
        or policy.get("schema") != _UPLOAD_POLICY_SCHEMA
        or not isinstance(scope, Mapping)
        or set(scope) != {"experiment_id", "continuation_id"}
        or scope.get("experiment_id")
        != authority["preregistration"].get("experiment_id")
        or scope.get("continuation_id") != authority["spec"].get("run_id")
        or policy.get("authorities") != exact_authorities
        or tuple(policy.get("prohibited_before_dev_release", ()))
        != ("dev", "fresh", "private")
        or policy.get("no_extra_files") is not True
    ):
        raise H6DestinationAdmissionError(
            "H6 upload policy boundary or authority DAG is invalid"
        )
    return _require_text(policy.get("policy_id"), "H6 upload policy ID")


def _authenticate_upload_policy(
    *,
    authority: Mapping[str, Any],
    controls: Mapping[str, Any],
    upload_policy_path: Path,
    upload_root: Path,
    source_root: Path,
    output_path: Path,
    static_files: Mapping[str, tuple[Path, Mapping[str, Any]]],
    dynamic_files: Mapping[str, tuple[Path, Mapping[str, Any]]],
    materialized_files: Mapping[Path, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    """Authenticate the upload DAG and reject every unlisted workspace file."""

    expected_root = Path("/workspace")
    try:
        root_stat = upload_root.lstat()
    except OSError as exc:
        raise H6DestinationAdmissionError("H6 upload root is unavailable") from exc
    if (
        upload_root.absolute() != expected_root
        or upload_root.resolve() != expected_root
        or not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
    ):
        raise H6DestinationAdmissionError(
            "H6 upload root must be the real /workspace directory"
        )
    _relative_to_root(source_root, upload_root, role="H6 source root")
    output_relative = _relative_to_root(
        output_path, upload_root, role="H6 admission receipt"
    )
    if os.path.lexists(output_path):
        raise H6DestinationAdmissionError(
            "admission receipt already exists; refusing to overwrite"
        )

    policy_payload, policy_identity = _read_verified_file(
        upload_policy_path,
        authority["authority_hashes"]["upload_allowlist_sha256"],
        role="H6 evaluation upload policy",
    )
    policy = _parse_json(policy_payload, role="H6 evaluation upload policy")
    policy_id = _require_upload_policy_header(
        policy,
        authority=authority,
        controls=controls,
    )

    if len(static_files) != 16:
        raise H6DestinationAdmissionError(
            "H6 static upload inventory must contain exactly 16 files"
        )
    expected_static_by_role: dict[str, dict[str, Any]] = {}
    expected_static_paths: set[str] = set()
    for semantic_role, (path, identity_value) in static_files.items():
        _require_text(semantic_role, "static upload semantic role")
        identity = _identity_record(
            dict(identity_value), role=f"static upload {semantic_role}"
        )
        relative = _relative_to_root(
            path, upload_root, role=f"static upload {semantic_role}"
        )
        if relative in expected_static_paths:
            raise H6DestinationAdmissionError("static upload paths must be unique")
        expected_static_paths.add(relative)
        expected_static_by_role[semantic_role] = {
            "role": semantic_role,
            "relative_path": relative,
            **dict(identity),
        }

    observed_static = _require_exact_static_policy_rows(
        policy.get("exact_static_files"),
        expected_by_role=expected_static_by_role,
    )

    expected_dynamic_names = set(_DYNAMIC_UPLOAD_SOURCES) - {"upload_policy"}
    if set(dynamic_files) != expected_dynamic_names:
        raise H6DestinationAdmissionError("H6 dynamic upload inputs are incomplete")
    actual_dynamic: dict[str, tuple[Path, dict[str, Any]]] = {
        name: (path, dict(_identity_record(dict(identity), role=name)))
        for name, (path, identity) in dynamic_files.items()
    }
    actual_dynamic["upload_policy"] = (
        upload_policy_path,
        policy_identity,
    )
    dynamic_entries = policy.get("dynamic_authenticated_files")
    if not isinstance(dynamic_entries, Mapping) or set(dynamic_entries) != set(
        _DYNAMIC_UPLOAD_SOURCES
    ):
        raise H6DestinationAdmissionError(
            "H6 upload policy dynamic inventory is not exact"
        )
    dynamic_paths: set[str] = set()
    for name, source_fields in _DYNAMIC_UPLOAD_SOURCES.items():
        entry = dynamic_entries[name]
        expected_keys = {"relative_path", *source_fields}
        if (
            not isinstance(entry, Mapping)
            or set(entry) != expected_keys
            or any(entry.get(key) != value for key, value in source_fields.items())
        ):
            raise H6DestinationAdmissionError(
                f"H6 upload policy dynamic entry is invalid: {name}"
            )
        relative = _safe_relative(
            entry.get("relative_path"), role=f"dynamic entry {name} path"
        )
        actual_relative = _relative_to_root(
            actual_dynamic[name][0], upload_root, role=f"dynamic upload {name}"
        )
        if relative != actual_relative or relative in dynamic_paths:
            raise H6DestinationAdmissionError(
                "H6 upload policy dynamic paths are inconsistent"
            )
        dynamic_paths.add(relative)
    if dynamic_paths.intersection(observed_static):
        raise H6DestinationAdmissionError(
            "H6 upload policy static and dynamic paths overlap"
        )
    if output_relative in observed_static or output_relative in dynamic_paths:
        raise H6DestinationAdmissionError(
            "admission receipt must be the sole post-admission generated file"
        )

    tracked: dict[Path, dict[str, Any]] = {}
    for path, identity in tuple(static_files.values()) + tuple(actual_dynamic.values()):
        _merge_tracked(tracked, {path: identity})
    _merge_tracked(tracked, materialized_files)
    allowed_by_relative: dict[str, dict[str, Any]] = {}
    for path, identity in tracked.items():
        relative = _relative_to_root(
            path, upload_root, role=f"authenticated workspace file {path.name}"
        )
        if (
            relative in allowed_by_relative
            and allowed_by_relative[relative] != identity
        ):
            raise H6DestinationAdmissionError(
                "authenticated workspace path has conflicting identities"
            )
        allowed_by_relative[relative] = identity

    observed_files: set[str] = set()
    try:
        for current, directory_names, file_names in os.walk(
            upload_root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            if current_path == source_root and ".h6-venv" in directory_names:
                directory_names.remove(".h6-venv")
            for directory_name in directory_names:
                directory_path = current_path / directory_name
                directory_stat = directory_path.lstat()
                if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(
                    directory_stat.st_mode
                ):
                    raise H6DestinationAdmissionError(
                        f"H6 upload tree has an unsafe directory: {directory_path}"
                    )
            for file_name in file_names:
                file_path = current_path / file_name
                file_stat = file_path.lstat()
                if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(
                    file_stat.st_mode
                ):
                    raise H6DestinationAdmissionError(
                        f"H6 upload tree has an unsafe file: {file_path}"
                    )
                observed_files.add(file_path.relative_to(upload_root).as_posix())
    except H6DestinationAdmissionError:
        raise
    except OSError as exc:
        raise H6DestinationAdmissionError(
            "H6 upload tree could not be inspected safely"
        ) from exc
    if observed_files != set(allowed_by_relative):
        missing = sorted(set(allowed_by_relative) - observed_files)
        extra = sorted(observed_files - set(allowed_by_relative))
        raise H6DestinationAdmissionError(
            f"H6 upload tree is not exact; missing={missing}, extra={extra}"
        )
    return {
        "policy_id": policy_id,
        "identity": policy_identity,
        "static_file_count": len(observed_static),
        "dynamic_file_count": len(dynamic_paths),
        "derived_materialized_file_count": len(materialized_files),
        "pre_admission_file_count": len(observed_files),
        "derived_environment_exception": (
            source_root.joinpath(".h6-venv").relative_to(upload_root).as_posix()
        ),
        "no_extra_files": True,
        "authorization_hash_embedded": False,
    }, tracked


def _authenticate_training_and_runtime(
    *,
    authority: Mapping[str, Any],
    source_root: Path,
    report_root: Path,
    device: str,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    if device != "cuda":
        raise H6DestinationAdmissionError(
            "H6 destination admission requires the exact CUDA runtime"
        )
    # Imports occur only after prohibited paths have been proven absent.
    from nano_ai.training import evaluate_evidence_query_h6 as evaluator
    from nano_ai.training import pointer_data, replay_mixture_data

    load_pointer_tokenizer = pointer_data.load_pointer_tokenizer

    expected_module = source_root / "nano_ai/training/evaluate_evidence_query_h6.py"
    if Path(evaluator.__file__).resolve() != expected_module.resolve():
        raise H6DestinationAdmissionError(
            "loaded H6 evaluator does not come from the authenticated source root"
        )
    expected_nano_modules = {
        replay_mixture_data: source_root / "nano_ai/training/replay_mixture_data.py",
        pointer_data: source_root / "nano_ai/training/pointer_data.py",
    }
    if any(
        Path(module.__file__).resolve() != expected.resolve()
        for module, expected in expected_nano_modules.items()
    ):
        raise H6DestinationAdmissionError(
            "loaded H6 support module does not come from the authenticated source root"
        )
    tokenizers_module = sys.modules.get("tokenizers")
    if tokenizers_module is None:
        raise H6DestinationAdmissionError("H6 tokenizers dependency was not imported")
    _require_dependency_module_origin(evaluator.torch, role="H6 torch dependency")
    _require_dependency_module_origin(
        tokenizers_module, role="H6 tokenizers dependency"
    )

    prereg = authority["preregistration"]
    freeze = authority["freeze"]
    readiness = authority["readiness"]
    package_members = authority["package_members"]
    frozen_inputs = prereg.get("frozen_inputs")
    if not isinstance(frozen_inputs, Mapping):
        raise H6DestinationAdmissionError("H6 frozen input authority is invalid")

    training_data_dir = source_root / "h6_data"
    training_input_paths = {
        "manifest": training_data_dir / "manifest.json",
        "fit": training_data_dir / "fit.jsonl",
        "calibration": training_data_dir / "calibration.jsonl",
    }
    input_roles = {
        "manifest": "training_manifest",
        "fit": "fit",
        "calibration": "calibration",
    }
    tracked: dict[Path, dict[str, Any]] = {}
    training_inputs: dict[str, dict[str, Any]] = {}
    for name, path in training_input_paths.items():
        expected = frozen_inputs[input_roles[name]]
        if not isinstance(expected, Mapping):
            raise H6DestinationAdmissionError(f"H6 {name} authority is invalid")
        payload, identity = _read_verified_file(
            path,
            expected.get("sha256"),
            role=f"H6 training {name}",
            expected_bytes=expected.get("bytes"),
        )
        del payload
        member_name = f"h6_data/{path.name}"
        if package_members.get(member_name) != identity:
            raise H6DestinationAdmissionError(
                f"extracted H6 training {name} differs from the frozen package"
            )
        tracked[path] = identity
        training_inputs[name] = identity

    try:
        training_bundle = replay_mixture_data.load_replay_mixture_dataset(
            training_data_dir
        )
        evaluator._require_frozen_h5_training_bundle(
            training_inputs["manifest"]["sha256"], training_bundle
        )
    except Exception as exc:
        raise H6DestinationAdmissionError(
            "H6 replay-mixture training inputs could not be authenticated"
        ) from exc

    reports = freeze.get("reports")
    report_checksums = freeze.get("report_checksums")
    if not isinstance(reports, Mapping) or not isinstance(report_checksums, Mapping):
        raise H6DestinationAdmissionError("H6 frozen report inventory is invalid")
    expected_report_paths = {
        f"results/seed-{seed}/training_report.json" for seed in _TRAINING_SEEDS
    }
    if (
        set(reports) != expected_report_paths
        or set(report_checksums) != expected_report_paths
    ):
        raise H6DestinationAdmissionError("H6 frozen report inventory is not exact")

    checksum_record = freeze.get("training_report_checksum_file")
    if not isinstance(checksum_record, Mapping):
        raise H6DestinationAdmissionError("H6 report checksum authority is invalid")
    checksum_relative = _safe_relative(
        checksum_record.get("path"), role="H6 report checksum file"
    )
    checksum_path = report_root / checksum_relative
    checksum_payload, checksum_identity = _read_verified_file(
        checksum_path,
        checksum_record.get("sha256"),
        role="H6 report checksum file",
        expected_bytes=checksum_record.get("bytes"),
    )
    tracked[checksum_path] = checksum_identity
    try:
        checksum_lines = checksum_payload.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise H6DestinationAdmissionError(
            "H6 report checksum file is malformed"
        ) from exc
    parsed_checksums: dict[str, str] = {}
    for index, line in enumerate(checksum_lines):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise H6DestinationAdmissionError(
                f"H6 report checksum line {index + 1} is malformed"
            )
        digest, raw_relative = match.groups()
        relative = _safe_relative(
            raw_relative, role=f"H6 report checksum line {index + 1} path"
        )
        if relative in parsed_checksums:
            raise H6DestinationAdmissionError(
                "H6 report checksum file contains a duplicate path"
            )
        parsed_checksums[relative] = digest
    if parsed_checksums != dict(report_checksums):
        raise H6DestinationAdmissionError(
            "H6 report checksum file disagrees with the freeze"
        )

    report_arguments: list[tuple[Path, str]] = []
    report_identities: list[dict[str, Any]] = []
    freeze_by_seed: dict[int, Mapping[str, Any]] = {}
    for relative, value in sorted(reports.items()):
        if not isinstance(value, Mapping):
            raise H6DestinationAdmissionError("H6 frozen report record is invalid")
        seed = value.get("seed")
        if seed not in _TRAINING_SEEDS or seed in freeze_by_seed:
            raise H6DestinationAdmissionError("H6 frozen report seed is invalid")
        path = report_root / _safe_relative(relative, role="H6 training report")
        _payload, identity = _read_verified_file(
            path,
            value.get("sha256"),
            role=f"H6 training report seed {seed}",
            expected_bytes=value.get("bytes"),
        )
        if report_checksums.get(relative) != identity["sha256"]:
            raise H6DestinationAdmissionError(
                f"H6 report checksum disagreement for seed {seed}"
            )
        tracked[path] = identity
        report_arguments.append((path, identity["sha256"]))
        report_identities.append({"seed": seed, "path": relative, **identity})
        freeze_by_seed[seed] = value

    try:
        primary, candidates = evaluator.authenticate_and_select_primary(
            report_arguments,
            expected_manifest_sha256=training_inputs["manifest"]["sha256"],
            training_data_dir=training_data_dir,
            training_bundle=training_bundle,
        )
    except Exception as exc:
        raise H6DestinationAdmissionError(
            "H6 reports or training-only primary selection failed authentication"
        ) from exc
    if (
        len(candidates) != len(_TRAINING_SEEDS)
        or {candidate.seed for candidate in candidates} != set(_TRAINING_SEEDS)
        or primary not in candidates
    ):
        raise H6DestinationAdmissionError(
            "H6 training-only selection did not return the exact two candidates"
        )

    checkpoint_identities: list[dict[str, Any]] = []
    strict_loads: list[dict[str, Any]] = []
    for candidate in candidates:
        frozen_candidate = freeze_by_seed[candidate.seed].get("candidate")
        expected_candidate = {
            "epoch": candidate.epoch,
            "filename": candidate.path.name,
            "bytes": candidate.artifact_bytes,
            "sha256": candidate.sha256,
        }
        if frozen_candidate != expected_candidate:
            raise H6DestinationAdmissionError(
                f"H6 candidate for seed {candidate.seed} disagrees with the freeze"
            )
        try:
            candidate_relative = candidate.path.relative_to(report_root).as_posix()
        except ValueError as exc:
            raise H6DestinationAdmissionError(
                "H6 selected checkpoint is outside the authenticated report root"
            ) from exc
        expected_relative = (
            PurePosixPath(f"results/seed-{candidate.seed}") / candidate.path.name
        ).as_posix()
        if candidate_relative != expected_relative:
            raise H6DestinationAdmissionError(
                "H6 selected checkpoint is not in its frozen seed directory"
            )
        _payload, checkpoint_identity = _read_verified_file(
            candidate.path,
            candidate.sha256,
            role=f"H6 selected checkpoint seed {candidate.seed}",
            expected_bytes=candidate.artifact_bytes,
        )
        tracked[candidate.path] = checkpoint_identity
        checkpoint_identities.append(
            {
                "seed": candidate.seed,
                "epoch": candidate.epoch,
                "path": candidate_relative,
                **checkpoint_identity,
            }
        )

    try:
        resolved_device = evaluator._resolve_device(device)
        runtime = evaluator._canonical_evaluation_runtime(
            primary,
            candidates,
            device=resolved_device,
            batch_size=evaluator.DEFAULT_BATCH_SIZE,
        )
        runtime["gpu_count"] = evaluator.torch.cuda.device_count()
        if runtime["gpu_count"] != 1:
            raise H6DestinationAdmissionError(
                "H6 exact runtime requires one visible CUDA GPU"
            )
        evaluator._seed_evaluation()
        for candidate in candidates:
            model = evaluator._load_evidence_query_model(
                candidate, device=resolved_device
            )
            strict_loads.append(
                {
                    "seed": candidate.seed,
                    "epoch": candidate.epoch,
                    "architecture_version": evaluator.ARCHITECTURE_VERSION,
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "strict": True,
                }
            )
            del model
    except Exception as exc:
        raise H6DestinationAdmissionError(
            "H6 exact runtime or strict checkpoint loading failed"
        ) from exc

    tokenizer_path = source_root / "sft/tokenizer.json"
    tokenizer_expected = frozen_inputs.get("tokenizer")
    readiness_tokenizer = readiness.get("inputs", {}).get("tokenizer")
    if (
        not isinstance(tokenizer_expected, Mapping)
        or readiness_tokenizer != tokenizer_expected
    ):
        raise H6DestinationAdmissionError("H6 tokenizer authorities disagree")
    _payload, tokenizer_identity = _read_verified_file(
        tokenizer_path,
        tokenizer_expected.get("sha256"),
        role="H6 tokenizer",
        expected_bytes=tokenizer_expected.get("bytes"),
    )
    if package_members.get("sft/tokenizer.json") != tokenizer_identity:
        raise H6DestinationAdmissionError(
            "extracted H6 tokenizer differs from the frozen package"
        )
    tracked[tokenizer_path] = tokenizer_identity
    try:
        tokenizer = load_pointer_tokenizer(tokenizer_path)
        vocabulary_size = tokenizer.get_vocab_size(with_added_tokens=True)
    except Exception as exc:
        raise H6DestinationAdmissionError(
            "H6 tokenizer could not be structurally authenticated"
        ) from exc

    try:
        evaluator._require_training_data_unchanged(
            training_data_dir, training_bundle.input_sha256
        )
    except Exception as exc:
        raise H6DestinationAdmissionError(
            "H6 training data changed during admission"
        ) from exc
    selection = {
        "order": list(_SELECTION_ORDER),
        "development_used": False,
        "candidates": [
            {
                "seed": candidate.seed,
                "epoch": candidate.epoch,
                "checkpoint_sha256": candidate.sha256,
                "report_sha256": candidate.report_sha256,
                "uncalibrated_training_calibration_macro_joint": candidate.macro_joint,
                "uncalibrated_training_calibration_overall_joint": candidate.overall_joint,
                "global_threshold": candidate.global_threshold,
            }
            for candidate in candidates
        ],
        "primary": {
            "seed": primary.seed,
            "epoch": primary.epoch,
            "checkpoint_sha256": primary.sha256,
            "report_sha256": primary.report_sha256,
        },
    }
    return {
        "training_inputs": training_inputs,
        "reports": report_identities,
        "report_checksum_file": {
            "path": checksum_relative,
            **checksum_identity,
        },
        "selected_checkpoints": checkpoint_identities,
        "tokenizer": {
            "path": "sft/tokenizer.json",
            **tokenizer_identity,
            "vocabulary_size": vocabulary_size,
        },
        "runtime_observation": runtime,
        "strict_checkpoint_loads": strict_loads,
        "selection": selection,
    }, tracked


def _require_training_matches_controls(
    *,
    authority: Mapping[str, Any],
    controls: Mapping[str, Any],
    attestation: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Bind the six recovered artifacts and measured runtime to authority."""

    runtime = training.get("runtime_observation")
    if not isinstance(runtime, Mapping) or dict(runtime) != attestation.get("runtime"):
        raise H6DestinationAdmissionError(
            "strict checkpoint runtime differs from destination attestation"
        )
    required_destination = controls["runtime_authority"]["required_destination"]
    runtime_mapping = {
        "python": "python",
        "torch": "torch",
        "tokenizers": "tokenizers",
        "cuda": "cuda",
        "gpu": "runtime_gpu_name",
        "gpu_count": "runtime_gpu_count",
        "cublas_workspace_config": "cublas_workspace_config",
        "platform": "platform",
    }
    if any(
        runtime.get(observed_name) != required_destination[required_name]
        for observed_name, required_name in runtime_mapping.items()
    ):
        raise H6DestinationAdmissionError(
            "strict checkpoint runtime differs from exact runtime authority"
        )

    prereg_runtime = authority["preregistration"].get("runtime")
    bundle_runtime = authority["bundle_manifest"].get("expected_runtime")
    if (
        not isinstance(prereg_runtime, Mapping)
        or set(prereg_runtime)
        != {
            "provider",
            "cloud_tier",
            "gpu",
            "container_image",
            "python",
            "torch",
            "tokenizers",
            "cuda",
            "cublas_workspace_config",
            "deterministic_algorithms",
            "larger_gpu_or_model_scaling_authorized",
        }
        or prereg_runtime.get("provider") != "runpod"
        or prereg_runtime.get("cloud_tier") != "secure"
        or prereg_runtime.get("gpu") != runtime["gpu"]
        or prereg_runtime.get("container_image") != _EXPECTED_IMAGE_ID
        or prereg_runtime.get("python") != "3.12.x"
        or prereg_runtime.get("torch") != runtime["torch"]
        or prereg_runtime.get("tokenizers") != runtime["tokenizers"]
        or prereg_runtime.get("cuda") != runtime["cuda"]
        or prereg_runtime.get("cublas_workspace_config")
        != runtime["cublas_workspace_config"]
        or prereg_runtime.get("deterministic_algorithms") is not True
        or prereg_runtime.get("larger_gpu_or_model_scaling_authorized") is not False
        or not isinstance(bundle_runtime, Mapping)
        or set(bundle_runtime)
        != {
            "python",
            "torch",
            "tokenizers",
            "cublas_workspace_config",
            "gpu_name",
            "minimum_gpu_memory_gib",
            "dependency_install",
            "isolated_environment",
        }
        or bundle_runtime.get("python") != "3.12.x"
        or bundle_runtime.get("torch") != runtime["torch"]
        or bundle_runtime.get("tokenizers") != runtime["tokenizers"]
        or bundle_runtime.get("cublas_workspace_config")
        != runtime["cublas_workspace_config"]
        or bundle_runtime.get("gpu_name") != runtime["gpu"]
        or bundle_runtime.get("minimum_gpu_memory_gib") != 20
        or bundle_runtime.get("dependency_install") != "requirements-h4-runpod.txt"
        or bundle_runtime.get("isolated_environment")
        != ".h6-venv with system-site-packages"
    ):
        raise H6DestinationAdmissionError(
            "frozen H6 runtime declarations are inconsistent"
        )

    reports = training.get("reports")
    checkpoints = training.get("selected_checkpoints")
    checksum = training.get("report_checksum_file")
    tokenizer = training.get("tokenizer")
    if (
        not isinstance(reports, list)
        or not isinstance(checkpoints, list)
        or not isinstance(checksum, Mapping)
        or not isinstance(tokenizer, Mapping)
    ):
        raise H6DestinationAdmissionError(
            "authenticated H6 training artifact inventory is invalid"
        )
    report_by_seed = {
        report.get("seed"): report for report in reports if isinstance(report, Mapping)
    }
    checkpoint_by_seed = {
        checkpoint.get("seed"): checkpoint
        for checkpoint in checkpoints
        if isinstance(checkpoint, Mapping)
    }
    if (
        set(report_by_seed) != set(_TRAINING_SEEDS)
        or set(checkpoint_by_seed) != set(_TRAINING_SEEDS)
        or len(report_by_seed) != len(reports)
        or len(checkpoint_by_seed) != len(checkpoints)
    ):
        raise H6DestinationAdmissionError(
            "authenticated H6 seed artifact inventory is not exact"
        )
    observed_by_role: dict[str, Mapping[str, Any]] = {
        "selected_checkpoint_seed_20260805": checkpoint_by_seed[20260805],
        "selected_checkpoint_seed_20260806": checkpoint_by_seed[20260806],
        "training_report_checksum_manifest": checksum,
        "training_report_seed_20260805": report_by_seed[20260805],
        "training_report_seed_20260806": report_by_seed[20260806],
        "vocabulary": tokenizer,
    }
    if set(observed_by_role) != _EXPECTED_FIXED_ARTIFACT_ROLES:
        raise H6DestinationAdmissionError(
            "authenticated H6 required-artifact role map is incomplete"
        )
    authenticated: dict[str, dict[str, Any]] = {}
    for role, observed in observed_by_role.items():
        artifact = controls["required_artifact_by_role"][role]
        observed_sha256 = _require_sha256(
            observed.get("sha256"), f"observed required artifact {role}"
        )
        if (
            artifact.get("class") != "training"
            or artifact.get("expected_sha256") != observed_sha256
        ):
            raise H6DestinationAdmissionError(
                f"required-artifact hash does not match materialized {role}"
            )
        authenticated[role] = {
            "artifact_id": artifact["artifact_id"],
            "class": "training",
            "sha256": observed_sha256,
        }
    return authenticated


def _require_tracked_files_unchanged(
    tracked: Mapping[Path, Mapping[str, Any]],
) -> None:
    for path, expected in tracked.items():
        _payload, observed = _read_regular_file(path, role=f"tracked input {path.name}")
        if observed != expected:
            raise H6DestinationAdmissionError(
                f"authenticated input changed before receipt: {path}"
            )


def _write_no_clobber(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if not path.parent.is_dir():
        raise H6DestinationAdmissionError(
            "admission receipt parent directory must already exist"
        )
    payload = _canonical_json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise H6DestinationAdmissionError(
            "admission receipt already exists; refusing to overwrite"
        ) from exc
    except OSError as exc:
        raise H6DestinationAdmissionError(
            "admission receipt could not be written safely"
        ) from exc
    return {"bytes": len(payload), "sha256": _digest(payload)}


def admit_h6_destination(
    *,
    run_spec_path: str | Path,
    run_events_path: str | Path,
    expected_ledger_tail_sha256: str,
    continuation_authorization_path: str | Path,
    expected_continuation_authorization_sha256: str,
    predecessor_binding_path: str | Path,
    predecessor_run_spec_path: str | Path,
    predecessor_run_events_path: str | Path,
    runtime_authority_path: str | Path,
    required_artifacts_path: str | Path,
    destination_selection_path: str | Path,
    upload_policy_path: str | Path,
    destination_attestation_path: str | Path,
    preregistration_path: str | Path,
    freeze_path: str | Path,
    readiness_path: str | Path,
    package_path: str | Path,
    source_root: str | Path,
    report_root: str | Path,
    terminal_wrapper_path: str | Path,
    upload_root: str | Path,
    output_path: str | Path,
    run_id: str,
    destination_id: str,
    prohibited_data_paths: Sequence[str | Path],
    device: str = "cuda",
) -> Mapping[str, Any]:
    """Admit one exact H6 evaluation destination without opening development."""

    if not isinstance(run_id, str) or not run_id or run_id.strip() != run_id:
        raise H6DestinationAdmissionError("run ID must be non-empty edge-trimmed text")
    if (
        not isinstance(destination_id, str)
        or _DESTINATION_ID_RE.fullmatch(destination_id) is None
    ):
        raise H6DestinationAdmissionError("destination ID is invalid")
    run_spec = Path(run_spec_path)
    run_events = Path(run_events_path)
    authorization_path = Path(continuation_authorization_path)
    predecessor_binding = Path(predecessor_binding_path)
    predecessor_spec = Path(predecessor_run_spec_path)
    predecessor_events = Path(predecessor_run_events_path)
    runtime_authority = Path(runtime_authority_path)
    required_artifacts = Path(required_artifacts_path)
    destination_selection = Path(destination_selection_path)
    upload_policy = Path(upload_policy_path)
    destination_attestation = Path(destination_attestation_path)
    preregistration = Path(preregistration_path)
    freeze = Path(freeze_path)
    readiness = Path(readiness_path)
    package = Path(package_path)
    source = Path(source_root)
    report = Path(report_root)
    terminal_wrapper = Path(terminal_wrapper_path)
    workspace = Path(upload_root)
    output = Path(output_path)
    if os.path.lexists(output):
        raise H6DestinationAdmissionError(
            "admission receipt already exists; refusing to overwrite"
        )
    absent_paths = _assert_paths_absent(
        tuple(Path(path) for path in prohibited_data_paths)
    )

    authority, static_tracked = _authenticate_static_authorities(
        run_spec_path=run_spec,
        run_events_path=run_events,
        expected_ledger_tail_sha256=expected_ledger_tail_sha256,
        preregistration_path=preregistration,
        freeze_path=freeze,
        readiness_path=readiness,
        package_path=package,
        source_root=source,
        run_id=run_id,
    )
    controls, control_tracked = _authenticate_continuation_controls(
        authority=authority,
        continuation_authorization_path=authorization_path,
        expected_continuation_authorization_sha256=(
            expected_continuation_authorization_sha256
        ),
        predecessor_binding_path=predecessor_binding,
        predecessor_run_spec_path=predecessor_spec,
        predecessor_run_events_path=predecessor_events,
        runtime_authority_path=runtime_authority,
        required_artifacts_path=required_artifacts,
        destination_selection_path=destination_selection,
    )
    frozen_identity = controls["authorization"]["frozen_scientific_identity"]
    admission_source = Path(os.path.abspath(os.fspath(Path(__file__))))
    _source_payload, admission_source_identity = _read_verified_file(
        admission_source,
        frozen_identity.get("admission_program_sha256"),
        role="H6 destination admission implementation",
    )
    _wrapper_payload, terminal_wrapper_identity = _read_verified_file(
        terminal_wrapper,
        frozen_identity.get("terminal_wrapper_sha256"),
        role="H6 terminal evaluation wrapper",
    )
    attestation, attestation_tracked = _authenticate_destination_attestation(
        authority=authority,
        controls=controls,
        attestation_path=destination_attestation,
        destination_id=destination_id,
    )
    _require_admission_sync_paths(
        attestation["admission_sync"],
        upload_root=workspace,
        run_spec_path=run_spec,
        run_events_path=run_events,
        destination_attestation_path=destination_attestation,
        output_path=output,
    )
    materialized_tracked = _authenticate_materialized_tree(
        authority=authority,
        source_root=source,
        report_root=report,
        admission_source=admission_source,
        admission_source_identity=admission_source_identity,
        terminal_wrapper_path=terminal_wrapper,
        terminal_wrapper_identity=terminal_wrapper_identity,
    )
    static_uploads = _static_upload_files(
        authority=authority,
        controls=controls,
        package_path=package,
        preregistration_path=preregistration,
        freeze_path=freeze,
        readiness_path=readiness,
        runtime_authority_path=runtime_authority,
        destination_selection_path=destination_selection,
        predecessor_binding_path=predecessor_binding,
        required_artifacts_path=required_artifacts,
        admission_source=admission_source,
        admission_source_identity=admission_source_identity,
        terminal_wrapper_path=terminal_wrapper,
        terminal_wrapper_identity=terminal_wrapper_identity,
        source_root=source,
        materialized_files=materialized_tracked,
    )
    dynamic_uploads = {
        "run_spec": (
            run_spec,
            authority["authority_identities"]["run_spec"],
        ),
        "run_events": (
            run_events,
            authority["authority_identities"]["run_events"],
        ),
        "authorization_envelope": (
            authorization_path,
            controls["identities"]["continuation_authorization"],
        ),
        "destination_attestation": (
            destination_attestation,
            attestation["identity"],
        ),
        "predecessor_run_spec": (
            predecessor_spec,
            controls["identities"]["predecessor_run_spec"],
        ),
        "predecessor_run_events": (
            predecessor_events,
            controls["identities"]["predecessor_run_events"],
        ),
    }
    upload_boundary, upload_tracked = _authenticate_upload_policy(
        authority=authority,
        controls=controls,
        upload_policy_path=upload_policy,
        upload_root=workspace,
        source_root=source,
        output_path=output,
        static_files=static_uploads,
        dynamic_files=dynamic_uploads,
        materialized_files=materialized_tracked,
    )
    tracked: dict[Path, dict[str, Any]] = {}
    _merge_tracked(tracked, static_tracked)
    _merge_tracked(tracked, control_tracked)
    _merge_tracked(tracked, attestation_tracked)
    _merge_tracked(tracked, upload_tracked)

    _assert_paths_absent(tuple(Path(path) for path in absent_paths))
    training, training_tracked = _authenticate_training_and_runtime(
        authority=authority,
        source_root=source,
        report_root=report,
        device=device,
    )
    required_artifact_bindings = _require_training_matches_controls(
        authority=authority,
        controls=controls,
        attestation=attestation,
        training=training,
    )
    _merge_tracked(tracked, training_tracked)
    _assert_paths_absent(tuple(Path(path) for path in absent_paths))
    _require_tracked_files_unchanged(tracked)

    generated_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    receipt = {
        "schema_version": H6_DESTINATION_ADMISSION_SCHEMA_VERSION,
        "status": "ADMITTED_DEVELOPMENT_FREE",
        "generated_at_utc": generated_at,
        "run": {
            "run_id": run_id,
            "destination_id": destination_id,
            "observation_operation_id": attestation["observation_operation_id"],
            "creation_operation_id": attestation["creation_operation_id"],
            "rehydration_operation_id": attestation["rehydration_operation_id"],
            "ledger": authority["ledger"],
        },
        "authority_hashes": authority["authority_hashes"],
        "authority_identities": authority["authority_identities"],
        "authenticated_sources": authority["source_identities"],
        "continuation": {
            "authorization": controls["identities"]["continuation_authorization"],
            "predecessor": controls["predecessor"],
            "runtime_authority": controls["identities"]["runtime_authority"],
            "destination_selection": controls["identities"]["destination_selection"],
            "required_artifacts_manifest": controls["identities"]["required_artifacts"],
        },
        "destination_attestation": {
            "attestation_id": attestation["attestation_id"],
            "identity": attestation["identity"],
            "provider_snapshot_event_index": attestation[
                "provider_snapshot_event_index"
            ],
            "provider_snapshot_event_sha256": attestation[
                "provider_snapshot_event_sha256"
            ],
            "destination_observation_sha256": attestation[
                "destination_observation_sha256"
            ],
            "network_volume_provider_snapshot_event_index": attestation[
                "network_volume_provider_snapshot_event_index"
            ],
            "network_volume_provider_snapshot_event_sha256": attestation[
                "network_volume_provider_snapshot_event_sha256"
            ],
            "network_volume_observation_sha256": attestation[
                "network_volume_observation_sha256"
            ],
            "network_volume_observation_evidence_sha256": attestation[
                "network_volume_observation_evidence_sha256"
            ],
            "operation_lineage": attestation["operation_lineage"],
            "destination": attestation["destination"],
        },
        "admission_sync": attestation["admission_sync"],
        "upload_boundary": {
            **upload_boundary,
            "receipt_relative_path": _relative_to_root(
                output, workspace, role="H6 admission receipt"
            ),
            "receipt_was_absent_before_admission": True,
        },
        "artifacts": {
            "training_inputs": training["training_inputs"],
            "reports": training["reports"],
            "report_checksum_file": training["report_checksum_file"],
            "selected_checkpoints": training["selected_checkpoints"],
            "tokenizer": training["tokenizer"],
            "required_artifact_bindings": required_artifact_bindings,
        },
        "selection": training["selection"],
        "runtime_observation": training["runtime_observation"],
        "strict_checkpoint_loads": training["strict_checkpoint_loads"],
        "prohibited_data": {
            "paths_asserted_absent": list(absent_paths),
            "development_records_read": 0,
            "fresh_records_read": 0,
            "private_records_read": 0,
            "development_sha256_authenticated_not_opened": authority[
                "authority_hashes"
            ]["development_sha256"],
            "development_manifest_sha256_authenticated_not_opened": (
                _EXPECTED_DEVELOPMENT_MANIFEST_SHA256
            ),
        },
        "admission_programs": {
            "destination_admission": admission_source_identity,
            "terminal_wrapper": terminal_wrapper_identity,
        },
        "provider_state_mutated": False,
        "guard_state_mutated": False,
    }
    _write_no_clobber(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authenticate an exact, development-free Nano H6 destination"
    )
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--run-events", type=Path, required=True)
    parser.add_argument("--expected-ledger-tail-sha256", required=True)
    parser.add_argument("--continuation-authorization", type=Path, required=True)
    parser.add_argument("--expected-continuation-authorization-sha256", required=True)
    parser.add_argument("--predecessor-binding", type=Path, required=True)
    parser.add_argument("--predecessor-run-spec", type=Path, required=True)
    parser.add_argument("--predecessor-run-events", type=Path, required=True)
    parser.add_argument("--runtime-authority", type=Path, required=True)
    parser.add_argument("--required-artifacts", type=Path, required=True)
    parser.add_argument("--destination-selection", type=Path, required=True)
    parser.add_argument("--upload-policy", type=Path, required=True)
    parser.add_argument("--destination-attestation", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--training-report-freeze", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--terminal-wrapper", type=Path, required=True)
    parser.add_argument("--upload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--destination-id", required=True)
    parser.add_argument(
        "--prohibited-data-path",
        type=Path,
        action="append",
        required=True,
        help=(
            "path that must be absent (repeat for development, fresh, and private "
            "locations)"
        ),
    )
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _require_isolated_no_site_startup()
    args = _parser().parse_args(argv)
    _bootstrap_isolated_import_paths(args.source_root)
    admit_h6_destination(
        run_spec_path=args.run_spec,
        run_events_path=args.run_events,
        expected_ledger_tail_sha256=args.expected_ledger_tail_sha256,
        continuation_authorization_path=args.continuation_authorization,
        expected_continuation_authorization_sha256=(
            args.expected_continuation_authorization_sha256
        ),
        predecessor_binding_path=args.predecessor_binding,
        predecessor_run_spec_path=args.predecessor_run_spec,
        predecessor_run_events_path=args.predecessor_run_events,
        runtime_authority_path=args.runtime_authority,
        required_artifacts_path=args.required_artifacts,
        destination_selection_path=args.destination_selection,
        upload_policy_path=args.upload_policy,
        destination_attestation_path=args.destination_attestation,
        preregistration_path=args.preregistration,
        freeze_path=args.training_report_freeze,
        readiness_path=args.readiness,
        package_path=args.package,
        source_root=args.source_root,
        report_root=args.report_root,
        terminal_wrapper_path=args.terminal_wrapper,
        upload_root=args.upload_root,
        output_path=args.output,
        run_id=args.run_id,
        destination_id=args.destination_id,
        prohibited_data_paths=args.prohibited_data_path,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "H6_DESTINATION_ADMISSION_SCHEMA_VERSION",
    "H6DestinationAdmissionError",
    "admit_h6_destination",
]
