from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from nano_ai.training import h6_terminal_result as terminal


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _event(
    index: int, name: str, payload: dict[str, object], previous: str | None
) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": "nano.runpod.ledger.v2",
        "index": index,
        "timestamp_utc": f"2026-08-03T00:00:0{index}Z",
        "event": name,
        "payload": payload,
        "previous_sha256": previous,
    }
    return {**base, "event_sha256": _digest(_canonical(base)[:-1])}


def _write(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    if executable:
        path.chmod(0o755)


@dataclass(frozen=True)
class Harness:
    root: Path
    package: Path
    identity: terminal.FrozenIdentity
    bindings: terminal.AuthorityBindings


def _harness(
    tmp_path: Path,
    script_body: str,
    *,
    authority_case: str = "valid",
    evaluator_payload: bytes = b"# frozen evaluator fixture\n",
) -> Harness:
    package = tmp_path / "nano-h6-runpod"
    (package / "results").mkdir(parents=True)

    environment_packages = (
        package
        / ".h6-venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    for dependency in ("torch", "tokenizers"):
        _write(
            environment_packages / dependency / "__init__.py",
            f'__version__ = "h6-test-{dependency}"\n'.encode(),
        )
    interpreter = package / ".h6-venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys.executable).resolve())

    script_payload = ("#!/bin/bash\nset -eu\n" + script_body).encode("utf-8")
    script_path = package / terminal.EVALUATION_SCRIPT_NAME
    _write(script_path, script_payload, executable=True)
    evaluator_path = package / terminal.EVALUATOR_SOURCE_RELATIVE
    _write(evaluator_path, evaluator_payload)
    base_payload = b"frozen package member\n"
    _write(package / "base.txt", base_payload)

    recovered_payload = b"frozen recovered result\n"
    recovered_relative = "results/seed-a/frozen-report.json"
    _write(package / recovered_relative, recovered_payload)
    development_manifest = b'{"partition":"development"}\n'
    development_payload = b'{"record_id":"dev-1"}\n'
    _write(package / "h2_development/manifest.json", development_manifest)
    _write(package / "h2_development/dev.jsonl", development_payload)

    members = {
        terminal.EVALUATION_SCRIPT_NAME: {
            "bytes": len(script_payload),
            "sha256": _digest(script_payload),
        },
        terminal.EVALUATOR_SOURCE_RELATIVE: {
            "bytes": len(evaluator_payload),
            "sha256": _digest(evaluator_payload),
        },
        "base.txt": {"bytes": len(base_payload), "sha256": _digest(base_payload)},
    }
    bundle_manifest = _canonical({"archive_root": package.name, "members": members})
    _write(package / "BUNDLE_MANIFEST.json", bundle_manifest)

    identity = terminal.FrozenIdentity(
        package_id="nano-h6-test-package-v1",
        package_sha256="a" * 64,
        archive_root=package.name,
        bundle_manifest_sha256=_digest(bundle_manifest),
        evaluation_authority_id="nano-h6-dev-one-shot-v1",
        evaluation_script_sha256=_digest(script_payload),
        evaluator_sha256=_digest(evaluator_payload),
        development_sha256=_digest(development_payload),
        development_manifest_sha256=_digest(development_manifest),
        recovered_files={
            recovered_relative: terminal.ExpectedFile(
                _digest(recovered_payload), len(recovered_payload)
            )
        },
        development_files={
            "h2_development/manifest.json": terminal.ExpectedFile(
                _digest(development_manifest), len(development_manifest)
            ),
            "h2_development/dev.jsonl": terminal.ExpectedFile(
                _digest(development_payload), len(development_payload)
            ),
        },
    )

    runtime_sha256 = "b" * 64
    required_sha256 = "c" * 64
    authorization_sha256 = "d" * 64
    upload_allowlist_sha256 = "e" * 64
    run_id = "h6-test-run"
    destination_id = "pod-test-1"
    network_volume_id = "volume-test-1"
    operation_id = "evaluation-operation-1"
    destination_observation_sha256 = "1" * 64
    destination_observation_evidence_sha256 = "2" * 64
    network_volume_observation_sha256 = "3" * 64
    network_volume_observation_evidence_sha256 = "4" * 64
    spec = {
        "schema": "nano.runpod.spec.v3",
        "project": "nano-lm",
        "run_id": run_id,
        "package_sha256": identity.package_sha256,
        "evaluator_sha256": identity.evaluator_sha256,
        "development_sha256": identity.development_sha256,
        "evaluation_authority_id": identity.evaluation_authority_id,
        "runtime_authority_sha256": runtime_sha256,
        "required_artifacts_manifest_sha256": required_sha256,
        "authorization_envelope_sha256": authorization_sha256,
        "upload_allowlist_sha256": upload_allowlist_sha256,
    }
    run_spec = tmp_path / "RUN_SPEC.json"
    spec_payload = _canonical(spec)
    _write(run_spec, spec_payload)

    genesis = _event(1, "RUN_INITIALIZED", {"spec": spec}, None)
    volume_snapshot = _event(
        2,
        "PROVIDER_SNAPSHOT",
        {
            "resource_role": "network_volume",
            "resource_id": network_volume_id,
            "observation_for_operation_id": None,
        },
        str(genesis["event_sha256"]),
    )
    destination_snapshot = _event(
        3,
        "PROVIDER_SNAPSHOT",
        {
            "resource_role": "destination",
            "resource_id": destination_id,
            "observation_for_operation_id": None,
        },
        str(volume_snapshot["event_sha256"]),
    )
    prepared_payload: dict[str, object] = {
        "schema": "nano.runpod.destination-admission-sync.v1",
        "transaction_id": "h6-test-sync-1",
        "run_id": run_id,
        "phase": "REPORTS_FROZEN",
        "ledger_tail_before": destination_snapshot["event_sha256"],
        "final_network_volume_snapshot_event_index": volume_snapshot["index"],
        "final_network_volume_snapshot_event_sha256": volume_snapshot["event_sha256"],
        "final_destination_snapshot_event_index": destination_snapshot["index"],
        "final_destination_snapshot_event_sha256": destination_snapshot["event_sha256"],
        "destination_resource_id": destination_id,
        "network_volume_resource_id": network_volume_id,
        "authorization_envelope_sha256": authorization_sha256,
        "upload_allowlist_sha256": upload_allowlist_sha256,
        "runtime_authority_sha256": runtime_sha256,
        "remote_root": "/workspace",
        "relative_paths": {
            "run_spec": "authority/RUN_SPEC.json",
            "run_events": "authority/RUN_EVENTS.jsonl",
            "destination_attestation": "authority/destination-attestation.json",
            "admission_receipt": "receipts/DESTINATION_ADMISSION.json",
        },
        "transfer_protocol": "atomic_content_addressed_authority_mirror_v1",
        "transfer_provider_state_mutated": False,
        "transfer_guard_state_mutated": False,
    }
    if authority_case == "prepared_provider_mutated":
        prepared_payload["transfer_provider_state_mutated"] = True
    if authority_case == "prepared_guard_mutated":
        prepared_payload["transfer_guard_state_mutated"] = True
    prepared = _event(
        4,
        "DESTINATION_ADMISSION_SYNC_PREPARED",
        prepared_payload,
        str(destination_snapshot["event_sha256"]),
    )
    pre_admission_events = [genesis, volume_snapshot, destination_snapshot]
    if authority_case != "missing_prepared":
        pre_admission_events.append(prepared)
    if authority_case == "duplicate_prepared":
        pre_admission_events.append(
            _event(
                5,
                "DESTINATION_ADMISSION_SYNC_PREPARED",
                dict(prepared_payload),
                str(prepared["event_sha256"]),
            )
        )
    if authority_case == "suffix_after_prepared":
        pre_admission_events.append(
            _event(
                5,
                "COST_SNAPSHOT",
                {"compute_usd": "0", "storage_usd": "0"},
                str(prepared["event_sha256"]),
            )
        )
    prefix = b"".join(_canonical(event) for event in pre_admission_events)
    wrapper_payload = Path(terminal.__file__).resolve().read_bytes()
    wrapper_identity = {
        "bytes": len(wrapper_payload),
        "sha256": _digest(wrapper_payload),
    }
    admission_sync = {
        "schema": "nano.runpod.destination-admission-sync.v1",
        "transaction_id": prepared_payload["transaction_id"],
        "prepared_event_index": prepared["index"],
        "prepared_event_sha256": prepared["event_sha256"],
        "ledger_tail_before": prepared_payload["ledger_tail_before"],
        "remote_root": prepared_payload["remote_root"],
        "relative_paths": prepared_payload["relative_paths"],
        "transfer_protocol": prepared_payload["transfer_protocol"],
    }
    if authority_case == "receipt_transaction_mismatch":
        admission_sync["transaction_id"] = "h6-test-sync-wrong"
    admission_program_identity = dict(terminal._DESTINATION_ADMISSION_PROGRAM_IDENTITY)
    if authority_case == "admission_program_mismatch":
        admission_program_identity["sha256"] = "f" * 64
    ledger_identity = {
        "event_count": len(pre_admission_events),
        "bytes": len(prefix),
        "tail_sha256": pre_admission_events[-1]["event_sha256"],
        "file_sha256": _digest(prefix),
    }
    authority_identities = {
        "run_spec": {"bytes": len(spec_payload), "sha256": _digest(spec_payload)},
        "run_events": {"bytes": len(prefix), "sha256": _digest(prefix)},
        "preregistration": {"bytes": 1, "sha256": "5" * 64},
        "training_report_freeze": {"bytes": 1, "sha256": "6" * 64},
        "readiness": {"bytes": 1, "sha256": "7" * 64},
        "package": {"bytes": 1, "sha256": identity.package_sha256},
    }
    authenticated_sources = {
        "evaluate_evidence_query_h6": {
            "path": terminal.EVALUATOR_SOURCE_RELATIVE,
            "bytes": len(evaluator_payload),
            "sha256": identity.evaluator_sha256,
        }
    }
    continuation = {
        "authorization": {"bytes": 1, "sha256": authorization_sha256},
        "predecessor": {
            "run_id": "h6-test-predecessor",
            "phase": "REPORTS_FROZEN",
            "ledger": {
                "event_count": 1,
                "tail_sha256": "9" * 64,
                "file_sha256": "a" * 64,
                "bytes": 1,
            },
            "source_pod_id": "pod-source-1",
            "source_machine_id": "machine-source-1",
            "source_required_state": "stopped",
            "source_access_policy": "lineage_only_no_restart",
        },
        "runtime_authority": {"bytes": 1, "sha256": runtime_sha256},
        "destination_selection": {"bytes": 1, "sha256": "8" * 64},
        "required_artifacts_manifest": {"bytes": 1, "sha256": required_sha256},
    }
    upload_boundary = {
        "identity": {"bytes": 1, "sha256": upload_allowlist_sha256},
        "policy_id": "h6-test-upload-policy",
        "static_file_count": 1,
        "dynamic_file_count": 1,
        "derived_materialized_file_count": 0,
        "pre_admission_file_count": 2,
        "derived_environment_exception": ".h6-venv",
        "no_extra_files": True,
        "authorization_hash_embedded": False,
        "receipt_relative_path": prepared_payload["relative_paths"][
            "admission_receipt"
        ],
        "receipt_was_absent_before_admission": True,
    }
    destination_attestation = {
        "attestation_id": "h6-test-destination-attestation",
        "identity": {
            "bytes": 1,
            "sha256": destination_observation_evidence_sha256,
        },
        "provider_snapshot_event_index": destination_snapshot["index"],
        "provider_snapshot_event_sha256": destination_snapshot["event_sha256"],
        "destination_observation_sha256": destination_observation_sha256,
        "network_volume_provider_snapshot_event_index": volume_snapshot["index"],
        "network_volume_provider_snapshot_event_sha256": volume_snapshot[
            "event_sha256"
        ],
        "network_volume_observation_sha256": network_volume_observation_sha256,
        "network_volume_observation_evidence_sha256": (
            network_volume_observation_evidence_sha256
        ),
        "operation_lineage": {
            "create_volume": {
                "operation_id": "create-volume-test-1",
                "kind": "create_volume",
                "start_event_index": 1,
                "start_event_sha256": genesis["event_sha256"],
                "finish_event_index": 2,
                "finish_event_sha256": volume_snapshot["event_sha256"],
                "provider_snapshot_event_index": 2,
                "provider_snapshot_event_sha256": volume_snapshot["event_sha256"],
                "outcome": "succeeded",
            },
            "create_pod": {
                "operation_id": "create-pod-test-1",
                "kind": "create_pod",
                "start_event_index": 1,
                "start_event_sha256": genesis["event_sha256"],
                "finish_event_index": 3,
                "finish_event_sha256": destination_snapshot["event_sha256"],
                "provider_snapshot_event_index": 3,
                "provider_snapshot_event_sha256": destination_snapshot["event_sha256"],
                "outcome": "succeeded",
            },
            "rehydrate_destination": {
                "operation_id": "rehydrate-test-1",
                "kind": "rehydrate_destination",
                "start_event_index": 2,
                "start_event_sha256": volume_snapshot["event_sha256"],
                "finish_event_index": 3,
                "finish_event_sha256": destination_snapshot["event_sha256"],
                "outcome": "succeeded",
            },
        },
        "destination": {
            "pod_id": destination_id,
            "machine_id": "machine-test-1",
            "network_volume_id": network_volume_id,
            "data_center_id": "EU-RO-1",
            "cloud_type": "COMMUNITY",
            "secure_cloud": False,
            "lifecycle_state": "RUNNING",
            "image_id": "runpod/pytorch:test",
            "observed_image_digest": None,
            "configured_gpu_id": "NVIDIA GeForce RTX 5090",
            "provider_gpu_model": "RTX_5090",
            "configured_gpu_count": 1,
            "runtime_gpu_name": "NVIDIA GeForce RTX 5090",
            "runtime_gpu_count": 1,
            "workspace_mount_path": "/workspace",
        },
    }
    admission_programs = {
        "destination_admission": admission_program_identity,
        "terminal_wrapper": wrapper_identity,
    }
    receipt = {
        "schema_version": "nano.h6-destination-admission.v1",
        "status": "ADMITTED_DEVELOPMENT_FREE",
        "generated_at_utc": "2026-08-03T00:00:04Z",
        "run": {
            "run_id": run_id,
            "destination_id": destination_id,
            "observation_operation_id": None,
            "creation_operation_id": "create-pod-test-1",
            "rehydration_operation_id": "rehydrate-test-1",
            "ledger": ledger_identity,
        },
        "authority_hashes": {
            key: value for key, value in spec.items() if key.endswith("_sha256")
        },
        "authority_identities": authority_identities,
        "authenticated_sources": authenticated_sources,
        "continuation": continuation,
        "upload_boundary": upload_boundary,
        "destination_attestation": destination_attestation,
        "admission_sync": admission_sync,
        "artifacts": {},
        "selection": {},
        "runtime_observation": {},
        "strict_checkpoint_loads": [],
        "prohibited_data": {
            "paths_asserted_absent": ["development", "private"],
            "development_records_read": 0,
            "fresh_records_read": 0,
            "private_records_read": 0,
            "development_sha256_authenticated_not_opened": (
                identity.development_sha256
            ),
            "development_manifest_sha256_authenticated_not_opened": (
                identity.development_manifest_sha256
            ),
        },
        "admission_programs": admission_programs,
        "provider_state_mutated": authority_case == "receipt_provider_mutated",
        "guard_state_mutated": False,
    }
    if authority_case == "receipt_guard_mutated":
        receipt["guard_state_mutated"] = True
    if authority_case == "receipt_extra_key":
        receipt["unexpected"] = True
    if authority_case == "receipt_missing_key":
        del receipt["artifacts"]
    if authority_case == "receipt_authority_identities_extra_key":
        receipt["authority_identities"]["unexpected"] = {  # type: ignore[index]
            "bytes": 1,
            "sha256": "f" * 64,
        }
    if authority_case == "receipt_authority_identity_extra_key":
        receipt["authority_identities"]["run_spec"]["unexpected"] = True  # type: ignore[index]
    if authority_case == "receipt_source_extra_key":
        receipt["authenticated_sources"]["evaluate_evidence_query_h6"][  # type: ignore[index]
            "unexpected"
        ] = True
    if authority_case == "receipt_continuation_extra_key":
        receipt["continuation"]["unexpected"] = True  # type: ignore[index]
    if authority_case == "receipt_predecessor_extra_key":
        receipt["continuation"]["predecessor"]["unexpected"] = True  # type: ignore[index]
    if authority_case == "receipt_predecessor_ledger_extra_key":
        receipt["continuation"]["predecessor"]["ledger"][  # type: ignore[index]
            "unexpected"
        ] = True
    if authority_case == "receipt_upload_extra_key":
        receipt["upload_boundary"]["unexpected"] = True  # type: ignore[index]
    if authority_case == "receipt_attestation_extra_key":
        receipt["destination_attestation"]["unexpected"] = True  # type: ignore[index]
    if authority_case == "receipt_attestation_identity_extra_key":
        receipt["destination_attestation"]["identity"]["unexpected"] = True  # type: ignore[index]
    if authority_case == "receipt_lineage_extra_key":
        receipt["destination_attestation"]["operation_lineage"][  # type: ignore[index]
            "unexpected"
        ] = {}
    if authority_case == "receipt_operation_extra_key":
        receipt["destination_attestation"]["operation_lineage"][  # type: ignore[index]
            "create_pod"
        ]["unexpected"] = True
    if authority_case == "receipt_destination_extra_key":
        receipt["destination_attestation"]["destination"]["unexpected"] = True  # type: ignore[index]
    receipt_path = tmp_path / "DESTINATION_ADMISSION.json"
    receipt_payload = _canonical(receipt)
    _write(receipt_path, receipt_payload)
    receipt_sha256 = _digest(receipt_payload)

    admission_receipt_binding = {
        "schema": "nano.runpod.destination-admission-receipt-binding.v1",
        "receipt": {
            "sha256": receipt_sha256,
            "bytes": len(receipt_payload),
            "relative_path": prepared_payload["relative_paths"]["admission_receipt"],
            "schema_version": receipt["schema_version"],
            "status": receipt["status"],
            "generated_at_utc": receipt["generated_at_utc"],
        },
        "run": copy.deepcopy(receipt["run"]),
        "authority_hashes": copy.deepcopy(receipt["authority_hashes"]),
        "authority_identities": copy.deepcopy(receipt["authority_identities"]),
        "authenticated_sources": copy.deepcopy(receipt["authenticated_sources"]),
        "continuation": copy.deepcopy(receipt["continuation"]),
        "upload_boundary": copy.deepcopy(receipt["upload_boundary"]),
        "destination_attestation": copy.deepcopy(receipt["destination_attestation"]),
        "admission_programs": {
            "authority": {
                "continuation_authorization_sha256": authorization_sha256,
                "admission_program_sha256": admission_program_identity["sha256"],
                "terminal_wrapper_sha256": wrapper_identity["sha256"],
            },
            "receipt": copy.deepcopy(receipt["admission_programs"]),
        },
        "provider_state_mutated": False,
        "guard_state_mutated": False,
    }
    if authority_case == "binding_receipt_sha_mismatch":
        admission_receipt_binding["receipt"]["sha256"] = "f" * 64
    if authority_case == "binding_receipt_path_mismatch":
        admission_receipt_binding["receipt"]["relative_path"] = "receipts/WRONG.json"
    if authority_case == "binding_ledger_mismatch":
        admission_receipt_binding["run"]["ledger"]["tail_sha256"] = "f" * 64
    if authority_case == "binding_authority_identity_mismatch":
        admission_receipt_binding["authority_identities"]["run_spec"]["sha256"] = (
            "f" * 64
        )
    if authority_case == "binding_destination_identity_mismatch":
        admission_receipt_binding["destination_attestation"]["destination"][
            "pod_id"
        ] = "pod-wrong"
    if authority_case == "binding_program_authority_mismatch":
        admission_receipt_binding["admission_programs"]["authority"][
            "terminal_wrapper_sha256"
        ] = "f" * 64
    if authority_case == "binding_program_receipt_mismatch":
        admission_receipt_binding["admission_programs"]["receipt"]["terminal_wrapper"][
            "sha256"
        ] = "f" * 64
    if authority_case == "binding_provider_mutated":
        admission_receipt_binding["provider_state_mutated"] = True
    if authority_case == "binding_guard_mutated":
        admission_receipt_binding["guard_state_mutated"] = True
    if authority_case == "binding_extra_key":
        admission_receipt_binding["unexpected"] = True
    if authority_case == "binding_receipt_extra_key":
        admission_receipt_binding["receipt"]["unexpected"] = True
    if authority_case == "binding_run_extra_key":
        admission_receipt_binding["run"]["unexpected"] = True
    if authority_case == "binding_program_authority_extra_key":
        admission_receipt_binding["admission_programs"]["authority"]["unexpected"] = (
            True
        )

    admission_payload = {
        "schema": "nano.runpod.destination-admission.v1",
        "run_id": run_id,
        "ledger_tail_before": pre_admission_events[-1]["event_sha256"],
        "destination_resource_id": destination_id,
        "destination_observation_sha256": destination_observation_sha256,
        "destination_observation_evidence_sha256": (
            destination_observation_evidence_sha256
        ),
        "network_volume_resource_id": network_volume_id,
        "network_volume_observation_sha256": network_volume_observation_sha256,
        "network_volume_observation_evidence_sha256": (
            network_volume_observation_evidence_sha256
        ),
        "evaluation_authority_id": identity.evaluation_authority_id,
        "package_sha256": identity.package_sha256,
        "evaluator_sha256": identity.evaluator_sha256,
        "runtime_authority_sha256": runtime_sha256,
        "required_artifacts_manifest_sha256": required_sha256,
        "admission_evidence_sha256": receipt_sha256,
        "admission_sync_transaction_id": prepared_payload["transaction_id"],
        "admission_sync_prepared_event_index": prepared["index"],
        "admission_sync_prepared_event_sha256": prepared["event_sha256"],
        "admission_receipt_binding": admission_receipt_binding,
    }
    if authority_case == "flat_sync_mismatch":
        admission_payload["admission_sync_transaction_id"] = "h6-test-sync-wrong"
    if authority_case == "missing_binding":
        del admission_payload["admission_receipt_binding"]

    admission = _event(
        len(pre_admission_events) + 1,
        "DESTINATION_ADMITTED",
        admission_payload,
        str(pre_admission_events[-1]["event_sha256"]),
    )
    release = _event(
        int(admission["index"]) + 1,
        "DEV_RELEASE_MARKED",
        {
            "dev_sha256": identity.development_sha256,
            "admission_sha256": admission["event_sha256"],
            "package_sha256": identity.package_sha256,
            "evaluator_sha256": identity.evaluator_sha256,
            "runtime_sha256": runtime_sha256,
            "destination_resource_id": destination_id,
        },
        str(admission["event_sha256"]),
    )
    launch = _event(
        int(release["index"]) + 1,
        "OPERATION_STARTED",
        {
            "operation_id": operation_id,
            "kind": "launch_evaluation",
            "resource_role": "destination",
            "target_id": destination_id,
        },
        str(release["event_sha256"]),
    )
    opened = _event(
        int(launch["index"]) + 1,
        "DEV_OPEN_MARKED",
        {
            "evaluation_operation_id": operation_id,
            "runtime_sha256": runtime_sha256,
            "evaluator_sha256": identity.evaluator_sha256,
            "package_sha256": identity.package_sha256,
            "destination_resource_id": destination_id,
        },
        str(launch["event_sha256"]),
    )
    ledger_payload = b"".join(
        _canonical(row)
        for row in (*pre_admission_events, admission, release, launch, opened)
    )
    run_events = tmp_path / "RUN_EVENTS.jsonl"
    _write(run_events, ledger_payload)
    bindings = terminal.AuthorityBindings(
        wrapper_sha256=wrapper_identity["sha256"],
        admission_receipt=receipt_path,
        admission_receipt_sha256=receipt_sha256,
        admission_event_sha256=str(admission["event_sha256"]),
        run_spec=run_spec,
        run_spec_sha256=_digest(spec_payload),
        run_events=run_events,
        run_events_sha256=_digest(ledger_payload),
        run_events_tail_sha256=str(opened["event_sha256"]),
    )
    return Harness(tmp_path, package, identity, bindings)


def _invoke(
    harness: Harness, name: str = "terminal-evidence"
) -> terminal.TerminalRunOutcome:
    return terminal.run_terminal_evaluation(
        package_root=harness.package,
        terminal_dir=harness.root / name,
        terminal_archive=harness.root / f"{name}.tar.gz",
        bindings=harness.bindings,
        _identity=harness.identity,
    )


def _archive_members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, mode="r:gz") as archive:
        return {
            member.name: archive.extractfile(member).read()  # type: ignore[union-attr]
            for member in archive.getmembers()
            if member.isfile()
        }


SUCCESS_SCRIPT = """
printf 'called\\n' >> invocation-count
printf 'combined stdout\\n'
printf 'combined stderr\\n' >&2
printf '{"status":"complete"}\\n' > results/development_evaluation.json
printf 'frozen evaluator log\\n' > results/development_evaluation.log
printf 'result digest\\n' > results/SHA256SUMS
printf 'archive digest\\n' > nano-h6-results.tar.gz.sha256
mkdir -p checkpoints h6_data
printf 'excluded checkpoint\\n' > checkpoints/other.pt
printf 'excluded training\\n' > h6_data/extra.jsonl
"""


ISOLATED_EVALUATOR = b"""\
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

source_root = Path(__file__).resolve().parents[2]
result = {
    "flags": {"isolated": sys.flags.isolated, "no_site": sys.flags.no_site},
    "nano_origin": Path(__file__).resolve().relative_to(source_root).as_posix(),
    "pythonpath": os.environ.get("PYTHONPATH"),
    "source_precedence": Path(sys.path[0]).resolve() == source_root,
    "startup_hooks_loaded": any(
        name in sys.modules for name in ("site", "sitecustomize", "usercustomize")
    ),
}
Path("results/development_evaluation.json").write_text(
    json.dumps(result, sort_keys=True, separators=(",", ":")) + "\\n",
    encoding="utf-8",
)
"""


def _environment_packages(harness: Harness) -> Path:
    return (
        harness.package
        / ".h6-venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )


def test_success_binds_real_authority_and_bundles_only_results(tmp_path: Path) -> None:
    harness = _harness(tmp_path, SUCCESS_SCRIPT)

    outcome = _invoke(harness)

    assert outcome.exit_code == 0
    assert outcome.status == "evaluator_succeeded"
    assert (harness.package / "invocation-count").read_text() == "called\n"
    marker = json.loads((harness.package / terminal.INVOCATION_MARKER_NAME).read_text())
    assert marker["retry_permitted"] is False
    assert marker["admission_event_sha256"] == harness.bindings.admission_event_sha256
    status = json.loads((outcome.terminal_dir / "terminal-status.json").read_text())
    assert status["bindings"]["run"] == {
        "id": "h6-test-run",
        "events_sha256": harness.bindings.run_events_sha256,
        "ledger_tail_sha256": harness.bindings.run_events_tail_sha256,
    }
    assert status["bindings"]["admission"]["receipt_sha256"] == (
        harness.bindings.admission_receipt_sha256
    )
    assert status["bindings"]["wrapper"]["sha256"] == harness.bindings.wrapper_sha256
    members = _archive_members(outcome.terminal_archive)
    prefix = terminal.ARCHIVE_ROOT + "/"
    assert set(members) == {
        prefix + "evaluator.log",
        prefix + "terminal-status.json",
        prefix + "generated-results/development_evaluation.json",
        prefix + "generated-results/development_evaluation.log",
        prefix + "generated-results/SHA256SUMS",
        prefix + "generated-results/nano-h6-results.tar.gz.sha256",
    }
    assert not any("checkpoint" in name or ".pt" in name for name in members)


def test_isolated_module_bridge_ignores_startup_hooks_and_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(
        tmp_path,
        "python -m nano_ai.training.evaluate_evidence_query_h6\n",
        evaluator_payload=ISOLATED_EVALUATOR,
    )
    hook_marker = tmp_path / "startup-hook-ran"
    environment_packages = _environment_packages(harness)
    _write(
        environment_packages / "hostile-startup.pth",
        f"import pathlib; pathlib.Path({str(hook_marker)!r}).touch()\n".encode(),
    )
    for name in ("sitecustomize.py", "usercustomize.py"):
        _write(
            environment_packages / name,
            f"from pathlib import Path\nPath({str(hook_marker)!r}).touch()\n".encode(),
        )
    hostile_pythonpath = tmp_path / "hostile-pythonpath"
    for name in ("sitecustomize.py", "usercustomize.py", "nano_ai.py"):
        _write(
            hostile_pythonpath / name,
            f"from pathlib import Path\nPath({str(hook_marker)!r}).touch()\n".encode(),
        )
    monkeypatch.setenv("PYTHONPATH", str(hostile_pythonpath))

    outcome = _invoke(harness)

    assert outcome.exit_code == 0
    result = json.loads(
        (
            outcome.terminal_dir / "generated-results/development_evaluation.json"
        ).read_text()
    )
    assert result == {
        "flags": {"isolated": 1, "no_site": 1},
        "nano_origin": terminal.EVALUATOR_SOURCE_RELATIVE,
        "pythonpath": None,
        "source_precedence": True,
        "startup_hooks_loaded": False,
    }
    assert not hook_marker.exists()
    marker = json.loads((harness.package / terminal.INVOCATION_MARKER_NAME).read_text())
    runtime = marker["isolated_runtime"]
    assert runtime["contract"] == terminal.isolated_execution_contract()
    assert runtime["interpreter"]["relative_path"] == ".h6-venv/bin/python"
    assert runtime["accepted_dependency_roots"][0] == {
        "kind": "environment",
        "path": (
            ".h6-venv/lib/"
            f"python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        ),
    }
    assert set(runtime["authenticated_import_origins"]) == {"tokenizers", "torch"}
    status = json.loads((outcome.terminal_dir / "terminal-status.json").read_text())
    assert status["bindings"]["isolated_runtime"] == runtime


@pytest.mark.parametrize(
    "invocation",
    (
        "python -c 'raise SystemExit(0)'",
        "python -m nano_ai.training.not_the_frozen_evaluator",
    ),
)
def test_launcher_bridge_refuses_non_evaluator_python_invocation(
    tmp_path: Path, invocation: str
) -> None:
    harness = _harness(tmp_path, invocation + "\n")

    outcome = _invoke(harness)

    assert outcome.evaluator_returncode == terminal.WRAPPER_FAILURE_EXIT_CODE
    assert outcome.exit_code == terminal.WRAPPER_FAILURE_EXIT_CODE
    assert outcome.status == "evaluator_failed"
    assert (
        "refused non-module Python invocation"
        in (outcome.terminal_dir / "evaluator.log").read_text()
    )


@pytest.mark.parametrize("shadow_kind", ("directory", "file", "symlink"))
def test_dependency_nano_shadow_refuses_before_marker(
    tmp_path: Path, shadow_kind: str
) -> None:
    harness = _harness(
        tmp_path / shadow_kind,
        "printf 'called\\n' >> invocation-count\n",
    )
    shadow = _environment_packages(harness) / (
        "nano_ai" if shadow_kind != "file" else "nano_ai.py"
    )
    if shadow_kind == "directory":
        shadow.mkdir()
    elif shadow_kind == "file":
        _write(shadow, b"raise RuntimeError('shadow imported')\n")
    else:
        shadow.symlink_to(harness.package / "nano_ai", target_is_directory=True)

    with pytest.raises(terminal.H6TerminalResultError, match="Nano shadow"):
        _invoke(harness)

    assert not (harness.package / terminal.INVOCATION_MARKER_NAME).exists()
    assert not (harness.package / "invocation-count").exists()


def test_isolated_execution_contract_is_literal() -> None:
    assert terminal.isolated_execution_contract() == {
        "schema": "nano.h6.isolated-execution-contract.v1",
        "interpreter_relative_path": ".h6-venv/bin/python",
        "python_flags": ["-I", "-S"],
        "admission_program_relative_path": (
            "nano_ai/training/admit_evidence_query_h6.py"
        ),
        "terminal_wrapper_relative_path": "nano_ai/training/h6_terminal_result.py",
        "evaluator_launcher_relative_path": "RUN_H6_EVALUATE.sh",
        "evaluator_launcher_shell": "/bin/bash",
        "evaluator_launcher_flags": ["--noprofile", "--norc"],
        "terminal_wrapper_bridge": "module_only_v1",
    }


def test_cli_requires_exact_isolated_direct_script_startup(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    wrapper = source_root / terminal._TERMINAL_WRAPPER_RELATIVE
    _write(wrapper, Path(terminal.__file__).read_bytes())
    interpreter = source_root / terminal._ISOLATED_INTERPRETER_RELATIVE
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(sys.executable).resolve())
    clean_environment = {
        "HOME": str(tmp_path),
        "LANG": "C.UTF-8",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    }

    accepted = subprocess.run(
        (str(interpreter), "-I", "-S", str(wrapper), "--help"),
        cwd=tmp_path,
        env=clean_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr

    missing_flags = subprocess.run(
        (str(interpreter), str(wrapper), "--help"),
        cwd=tmp_path,
        env=clean_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing_flags.returncode == terminal.WRAPPER_FAILURE_EXIT_CODE
    assert "must run with Python -I -S" in missing_flags.stderr

    relative_script = subprocess.run(
        (
            str(interpreter),
            "-I",
            "-S",
            terminal._TERMINAL_WRAPPER_RELATIVE,
            "--help",
        ),
        cwd=source_root,
        env=clean_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert relative_script.returncode == terminal.WRAPPER_FAILURE_EXIT_CODE
    assert "absolute script path" in relative_script.stderr

    wrong_interpreter = subprocess.run(
        (sys.executable, "-I", "-S", str(wrapper), "--help"),
        cwd=tmp_path,
        env=clean_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_interpreter.returncode == terminal.WRAPPER_FAILURE_EXIT_CODE
    assert ".h6-venv/bin/python" in wrong_interpreter.stderr


def test_evaluator_failure_is_terminal_evidence_and_preserves_exit_code(
    tmp_path: Path,
) -> None:
    harness = _harness(
        tmp_path,
        "printf 'called\\n' >> invocation-count\n"
        "printf 'ordinary failure\\n' >&2\n"
        "exit 23\n",
    )

    outcome = _invoke(harness)

    assert outcome.exit_code == 23
    assert outcome.evaluator_returncode == 23
    assert outcome.status == "evaluator_failed"
    status = json.loads((outcome.terminal_dir / "terminal-status.json").read_text())
    assert status["evaluator_outcome"]["returncode"] == 23
    assert set(_archive_members(outcome.terminal_archive)) == {
        terminal.ARCHIVE_ROOT + "/evaluator.log",
        terminal.ARCHIVE_ROOT + "/terminal-status.json",
    }


def test_fixed_marker_blocks_retry_with_different_output_paths(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        "printf 'called\\n' >> invocation-count\nprintf 'failed\\n' >&2\nexit 41\n",
    )
    first = _invoke(harness, "first")
    assert first.exit_code == 41

    with pytest.raises(terminal.H6TerminalResultError, match="cannot rerun"):
        _invoke(harness, "second")

    assert (harness.package / "invocation-count").read_text() == "called\n"
    assert not (tmp_path / "second").exists()


def test_package_or_authority_tamper_refuses_before_marker(tmp_path: Path) -> None:
    package_case = _harness(tmp_path / "package-case", "printf 'called\\n'\n")
    (package_case.package / "base.txt").write_text("tampered\n")
    with pytest.raises(terminal.H6TerminalResultError, match="SHA-256 mismatch"):
        _invoke(package_case)
    assert not (package_case.package / terminal.INVOCATION_MARKER_NAME).exists()

    ledger_case = _harness(tmp_path / "ledger-case", "printf 'called\\n'\n")
    bad_bindings = dataclasses.replace(
        ledger_case.bindings, run_events_tail_sha256="f" * 64
    )
    ledger_case = dataclasses.replace(ledger_case, bindings=bad_bindings)
    with pytest.raises(terminal.H6TerminalResultError, match="tail"):
        _invoke(ledger_case)
    assert not (ledger_case.package / terminal.INVOCATION_MARKER_NAME).exists()


@pytest.mark.parametrize(
    "authority_case",
    ("missing_prepared", "duplicate_prepared", "suffix_after_prepared"),
)
def test_prepared_event_is_unique_and_immediately_precedes_admission(
    tmp_path: Path, authority_case: str
) -> None:
    harness = _harness(
        tmp_path / authority_case,
        "printf 'called\\n' >> invocation-count\n",
        authority_case=authority_case,
    )

    with pytest.raises(terminal.H6TerminalResultError):
        _invoke(harness)

    assert not (harness.package / terminal.INVOCATION_MARKER_NAME).exists()
    assert not (harness.package / "invocation-count").exists()


@pytest.mark.parametrize(
    "authority_case",
    ("receipt_transaction_mismatch", "flat_sync_mismatch"),
)
def test_receipt_and_admission_sync_exactly_bind_prepared_event(
    tmp_path: Path, authority_case: str
) -> None:
    harness = _harness(
        tmp_path / authority_case,
        "printf 'called\\n' >> invocation-count\n",
        authority_case=authority_case,
    )

    with pytest.raises(terminal.H6TerminalResultError):
        _invoke(harness)

    assert not (harness.package / terminal.INVOCATION_MARKER_NAME).exists()
    assert not (harness.package / "invocation-count").exists()


@pytest.mark.parametrize(
    "authority_case",
    (
        "prepared_provider_mutated",
        "prepared_guard_mutated",
        "receipt_provider_mutated",
        "receipt_guard_mutated",
        "binding_provider_mutated",
        "binding_guard_mutated",
    ),
)
def test_transfer_receipt_and_binding_mutation_flags_are_false(
    tmp_path: Path, authority_case: str
) -> None:
    harness = _harness(
        tmp_path / authority_case,
        "printf 'called\\n' >> invocation-count\n",
        authority_case=authority_case,
    )

    with pytest.raises(terminal.H6TerminalResultError):
        _invoke(harness)

    assert not (harness.package / terminal.INVOCATION_MARKER_NAME).exists()
    assert not (harness.package / "invocation-count").exists()


@pytest.mark.parametrize(
    "authority_case",
    (
        "admission_program_mismatch",
        "binding_program_authority_mismatch",
        "binding_program_receipt_mismatch",
    ),
)
def test_destination_admission_program_identity_is_frozen(
    tmp_path: Path, authority_case: str
) -> None:
    harness = _harness(
        tmp_path / authority_case,
        "printf 'called\\n' >> invocation-count\n",
        authority_case=authority_case,
    )

    with pytest.raises(terminal.H6TerminalResultError):
        _invoke(harness)

    assert not (harness.package / terminal.INVOCATION_MARKER_NAME).exists()
    assert not (harness.package / "invocation-count").exists()


@pytest.mark.parametrize(
    "authority_case",
    (
        "receipt_extra_key",
        "receipt_missing_key",
        "receipt_authority_identities_extra_key",
        "receipt_authority_identity_extra_key",
        "receipt_source_extra_key",
        "receipt_continuation_extra_key",
        "receipt_predecessor_extra_key",
        "receipt_predecessor_ledger_extra_key",
        "receipt_upload_extra_key",
        "receipt_attestation_extra_key",
        "receipt_attestation_identity_extra_key",
        "receipt_lineage_extra_key",
        "receipt_operation_extra_key",
        "receipt_destination_extra_key",
        "missing_binding",
        "binding_extra_key",
        "binding_receipt_extra_key",
        "binding_run_extra_key",
        "binding_program_authority_extra_key",
        "binding_receipt_sha_mismatch",
        "binding_receipt_path_mismatch",
        "binding_ledger_mismatch",
        "binding_authority_identity_mismatch",
        "binding_destination_identity_mismatch",
    ),
)
def test_guard_admission_durably_binds_receipt_and_authority(
    tmp_path: Path, authority_case: str
) -> None:
    harness = _harness(
        tmp_path / authority_case,
        "printf 'called\\n' >> invocation-count\n",
        authority_case=authority_case,
    )

    with pytest.raises(terminal.H6TerminalResultError):
        _invoke(harness)

    assert not (harness.package / terminal.INVOCATION_MARKER_NAME).exists()
    assert not (harness.package / "invocation-count").exists()


def test_post_invocation_artifact_tamper_yields_status_log_only_bundle(
    tmp_path: Path,
) -> None:
    harness = _harness(
        tmp_path,
        "printf 'called\\n' >> invocation-count\n"
        "printf 'tampered\\n' > results/seed-a/frozen-report.json\n",
    )

    outcome = _invoke(harness)

    assert outcome.exit_code == terminal.WRAPPER_FAILURE_EXIT_CODE
    assert outcome.status == "wrapper_failed_after_invocation"
    assert (harness.package / terminal.INVOCATION_MARKER_NAME).exists()
    members = _archive_members(outcome.terminal_archive)
    assert set(members) == {
        terminal.ARCHIVE_ROOT + "/evaluator.log",
        terminal.ARCHIVE_ROOT + "/terminal-status.json",
    }


def test_symlink_result_is_never_followed_or_bundled(tmp_path: Path) -> None:
    harness = _harness(
        tmp_path,
        "ln -s /etc/passwd results/development_evaluation.json\n",
    )

    outcome = _invoke(harness)

    assert outcome.exit_code == terminal.WRAPPER_FAILURE_EXIT_CODE
    assert set(_archive_members(outcome.terminal_archive)) == {
        terminal.ARCHIVE_ROOT + "/evaluator.log",
        terminal.ARCHIVE_ROOT + "/terminal-status.json",
    }


def test_hostile_parent_environment_is_not_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    poison = tmp_path / "poison.sh"
    poison.write_text("touch poisoned-by-bash-env\n")
    monkeypatch.setenv("BASH_ENV", str(poison))
    monkeypatch.setenv("PYTHONPATH", "/hostile/python")
    monkeypatch.setenv("LD_PRELOAD", "/hostile/library.so")
    monkeypatch.setenv("PATH", "/hostile/bin")
    harness = _harness(
        tmp_path,
        "printf '%s|%s|%s|%s|%s\\n' \"${BASH_ENV-unset}\" "
        '"${PYTHONPATH-unset}" "${LD_PRELOAD-unset}" "$PATH" '
        '"${PYTHONNOUSERSITE-unset}" > results/development_evaluation.log\n',
    )

    outcome = _invoke(harness)

    observed = (
        (outcome.terminal_dir / "generated-results/development_evaluation.log")
        .read_text()
        .strip()
    )
    assert observed == (
        "unset|unset|unset|"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin|1"
    )
    assert not (harness.package / "poisoned-by-bash-env").exists()


def test_injected_private_or_fresh_data_refuses_before_invocation(
    tmp_path: Path,
) -> None:
    for name in terminal._PROHIBITED_PACKAGE_PATHS:
        harness = _harness(tmp_path / name, "printf 'called\\n'\n")
        (harness.package / name).mkdir()
        with pytest.raises(terminal.H6TerminalResultError, match="prohibited data"):
            _invoke(harness)
        assert not (harness.package / terminal.INVOCATION_MARKER_NAME).exists()


@pytest.mark.parametrize("name", terminal._PROHIBITED_PACKAGE_PATHS)
def test_private_or_fresh_data_created_during_evaluation_forces_failure_bundle(
    tmp_path: Path, name: str
) -> None:
    harness = _harness(
        tmp_path,
        f"mkdir {name}\nprintf 'prohibited\\n' > {name}/x\n",
    )

    outcome = _invoke(harness)

    assert outcome.exit_code == terminal.WRAPPER_FAILURE_EXIT_CODE
    assert set(_archive_members(outcome.terminal_archive)) == {
        terminal.ARCHIVE_ROOT + "/evaluator.log",
        terminal.ARCHIVE_ROOT + "/terminal-status.json",
    }


def test_wrapper_identity_and_output_alias_refuse_before_marker(tmp_path: Path) -> None:
    identity_case = _harness(tmp_path / "identity", "printf 'called\\n'\n")
    identity_case = dataclasses.replace(
        identity_case,
        bindings=dataclasses.replace(identity_case.bindings, wrapper_sha256="f" * 64),
    )
    with pytest.raises(terminal.H6TerminalResultError, match="executing code"):
        _invoke(identity_case)
    assert not (identity_case.package / terminal.INVOCATION_MARKER_NAME).exists()

    alias_case = _harness(tmp_path / "alias", "printf 'called\\n'\n")
    alias = alias_case.root / "same.tar.gz"
    with pytest.raises(terminal.H6TerminalResultError, match="distinct"):
        terminal.run_terminal_evaluation(
            package_root=alias_case.package,
            terminal_dir=alias,
            terminal_archive=alias,
            bindings=alias_case.bindings,
            _identity=alias_case.identity,
        )
    assert not (alias_case.package / terminal.INVOCATION_MARKER_NAME).exists()


def test_terminal_archive_is_byte_deterministic(tmp_path: Path) -> None:
    archives: list[bytes] = []
    for name in ("one", "two"):
        harness = _harness(tmp_path / name, SUCCESS_SCRIPT)
        archives.append(_invoke(harness).terminal_archive.read_bytes())
    assert archives[0] == archives[1]
