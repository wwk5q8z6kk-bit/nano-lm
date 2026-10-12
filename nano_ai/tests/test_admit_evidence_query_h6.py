from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from nano_ai.training import admit_evidence_query_h6 as admission


def _isolated_bootstrap_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    source_root = tmp_path / "nano-h6-runpod"
    environment_packages = (
        source_root
        / ".h6-venv"
        / "lib"
        / f"python{admission.sys.version_info.major}.{admission.sys.version_info.minor}"
        / "site-packages"
    )
    environment_packages.mkdir(parents=True)
    interpreter = source_root / ".h6-venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(Path(admission.sys.executable).resolve())

    base_prefix = tmp_path / "base-python"
    base_packages = base_prefix / "lib/python/site-packages"
    base_packages.mkdir(parents=True)
    monkeypatch.setattr(admission.sys, "executable", os.fspath(interpreter))
    monkeypatch.setattr(admission.sys, "base_prefix", os.fspath(base_prefix))
    monkeypatch.setattr(admission.sys, "base_exec_prefix", os.fspath(base_prefix))
    monkeypatch.setattr(
        admission.sysconfig,
        "get_paths",
        lambda: {
            "purelib": os.fspath(base_packages),
            "platlib": os.fspath(base_packages),
        },
    )
    monkeypatch.setattr(admission.sys, "path", ["/isolated-stdlib"])
    monkeypatch.setattr(admission, "_CLI_DEPENDENCY_ROOTS", ())
    for name in tuple(admission.sys.modules):
        if (
            name == "nano_ai"
            or name.startswith("nano_ai.")
            or name in {"torch", "tokenizers"}
        ):
            monkeypatch.delitem(admission.sys.modules, name)
    return source_root, environment_packages, base_packages


@pytest.mark.parametrize(
    ("isolated", "no_site"),
    [(0, 1), (1, 0), (0, 0), (None, 1), (1, None)],
)
def test_cli_startup_requires_both_isolated_and_no_site(
    isolated: int | None, no_site: int | None
) -> None:
    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="must run with Python -I -S",
    ):
        admission._require_isolated_no_site_startup(
            SimpleNamespace(isolated=isolated, no_site=no_site)
        )

    admission._require_isolated_no_site_startup(SimpleNamespace(isolated=1, no_site=1))


def test_isolated_bootstrap_does_not_process_startup_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, environment_packages, base_packages = _isolated_bootstrap_layout(
        tmp_path, monkeypatch
    )
    marker = tmp_path / "startup-hook-ran"
    startup_payload = (
        "from pathlib import Path\n"
        f"Path({os.fspath(marker)!r}).write_text('unsafe', encoding='utf-8')\n"
    )
    (environment_packages / "sitecustomize.py").write_text(
        startup_payload, encoding="utf-8"
    )
    (environment_packages / "usercustomize.py").write_text(
        startup_payload, encoding="utf-8"
    )
    (environment_packages / "hostile.pth").write_text(
        f"import pathlib; pathlib.Path({os.fspath(marker)!r}).write_text('unsafe')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", os.fspath(tmp_path / "hostile-pythonpath"))

    roots = admission._bootstrap_isolated_import_paths(source_root)

    assert roots == (environment_packages, base_packages)
    assert admission.sys.path == [
        os.fspath(source_root),
        "/isolated-stdlib",
        os.fspath(environment_packages),
        os.fspath(base_packages),
    ]
    assert not marker.exists()


def test_isolated_bootstrap_rejects_dependency_nano_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, environment_packages, _base_packages = _isolated_bootstrap_layout(
        tmp_path, monkeypatch
    )
    (environment_packages / "nano_ai").mkdir()

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="top-level Nano shadow",
    ):
        admission._bootstrap_isolated_import_paths(source_root)


def test_isolated_execution_contract_is_literal_and_exact() -> None:
    expected = copy.deepcopy(admission._ISOLATED_EXECUTION_CONTRACT)

    assert admission._require_isolated_execution_contract(expected) == expected

    for mutation in ("extra", "flags", "launcher", "bridge"):
        tampered = copy.deepcopy(expected)
        if mutation == "extra":
            tampered["comment"] = "ignored"
        elif mutation == "flags":
            tampered["python_flags"] = ["-S", "-I"]
        elif mutation == "launcher":
            tampered["evaluator_launcher_relative_path"] = "other.sh"
        else:
            tampered["terminal_wrapper_bridge"] = "unrestricted"
        with pytest.raises(
            admission.H6DestinationAdmissionError,
            match="execution contract is not exact",
        ):
            admission._require_isolated_execution_contract(tampered)


def _identity(number: int) -> dict[str, object]:
    return {"bytes": number, "sha256": f"{number:064x}"}


def _policy_authorities() -> dict[str, str]:
    return {
        "runtime_authority_sha256": "a" * 64,
        "destination_selection_sha256": "b" * 64,
        "predecessor_binding_sha256": "c" * 64,
        "required_artifacts_manifest_sha256": "d" * 64,
    }


def _policy() -> dict[str, object]:
    return {
        "schema": admission._UPLOAD_POLICY_SCHEMA,
        "policy_id": "nano-h6-development-free-upload-v1",
        "scope": {"experiment_id": "h6", "continuation_id": "h6-eval"},
        "authorities": _policy_authorities(),
        "exact_static_files": [],
        "dynamic_authenticated_files": {},
        "prohibited_before_dev_release": ["dev", "fresh", "private"],
        "no_extra_files": True,
    }


def _policy_context() -> tuple[dict[str, object], dict[str, object]]:
    authorities = _policy_authorities()
    authority = {
        "preregistration": {"experiment_id": "h6"},
        "spec": {"run_id": "h6-eval"},
    }
    controls = {
        "identities": {
            "runtime_authority": {"sha256": authorities["runtime_authority_sha256"]},
            "destination_selection": {
                "sha256": authorities["destination_selection_sha256"]
            },
            "predecessor_binding": {
                "sha256": authorities["predecessor_binding_sha256"]
            },
            "required_artifacts": {
                "sha256": authorities["required_artifacts_manifest_sha256"]
            },
        }
    }
    return authority, controls


def _signed_event(
    index: int,
    previous_sha256: str | None,
    *,
    event: str,
    payload: dict[str, object],
) -> dict[str, object]:
    base: dict[str, object] = {
        "schema": admission._LEDGER_SCHEMA,
        "index": index,
        "timestamp_utc": "2026-08-03T00:00:00Z",
        "event": event,
        "payload": payload,
        "previous_sha256": previous_sha256,
    }
    event_sha256 = hashlib.sha256(
        admission._canonical_json_bytes(base).removesuffix(b"\n")
    ).hexdigest()
    return {**base, "event_sha256": event_sha256}


def _guard_snapshot(payload: dict[str, object]) -> dict[str, object]:
    snapshot = copy.deepcopy(payload)
    snapshot["sanitized_observation_sha256"] = hashlib.sha256(
        admission._canonical_json_bytes(snapshot).removesuffix(b"\n")
    ).hexdigest()
    return snapshot


def _destination_attestation_context(
    tmp_path: Path,
    *,
    snapshot_runtime_sha256: str | None = None,
) -> tuple[dict[str, object], dict[str, object], Path]:
    run_id = "h6-eval"
    spec = {
        "run_id": run_id,
        "authorization_envelope_sha256": "1" * 64,
        "upload_allowlist_sha256": "2" * 64,
        "runtime_authority_sha256": admission._EXPECTED_RUNTIME_AUTHORITY_SHA256,
    }
    pod_id = "pod-123"
    machine_id = "machine-123"
    network_volume_id = "volume-123"
    creation_operation_id = "create-pod-op"
    rehydration_operation_id = "rehydrate-op"
    required_destination = {
        "configured_gpu_id": admission._EXPECTED_CONFIGURED_GPU_ID,
        "runtime_gpu_name": admission._EXPECTED_CONFIGURED_GPU_ID,
        "runtime_gpu_count": 1,
        "image_id": admission._EXPECTED_IMAGE_ID,
        "workspace_mount_path": "/workspace",
        "python": "3.12.3",
        "torch": "2.8.0+cu128",
        "cuda": "12.8",
        "tokenizers": "0.22.2",
        "platform": "Linux-6.8.0-90-generic-x86_64-with-glibc2.39",
        "cublas_workspace_config": ":4096:8",
    }
    runtime = {
        "python": required_destination["python"],
        "torch": required_destination["torch"],
        "tokenizers": required_destination["tokenizers"],
        "cuda": required_destination["cuda"],
        "gpu": required_destination["runtime_gpu_name"],
        "gpu_count": 1,
        "cublas_workspace_config": required_destination["cublas_workspace_config"],
        "platform": required_destination["platform"],
    }
    attestation = {
        "schema": "nano.h6.destination-attestation.v1",
        "attestation_id": "destination-attestation-1",
        "run_id": run_id,
        "observation_operation_id": None,
        "creation_operation_id": creation_operation_id,
        "rehydration_operation_id": rehydration_operation_id,
        "observed_at_utc": "2026-08-03T00:10:00Z",
        "provider": "runpod",
        "destination": {
            "pod_id": pod_id,
            "machine_id": machine_id,
            "network_volume_id": network_volume_id,
            "data_center_id": "EU-RO-1",
            "cloud_type": "SECURE",
            "secure_cloud": True,
            "lifecycle_state": "running",
            "image_id": admission._EXPECTED_IMAGE_ID,
            "observed_image_digest": None,
            "configured_gpu_id": admission._EXPECTED_CONFIGURED_GPU_ID,
            "provider_gpu_model": admission._EXPECTED_SPEC_GPU_MODEL,
            "configured_gpu_count": 1,
            "runtime_gpu_name": admission._EXPECTED_CONFIGURED_GPU_ID,
            "runtime_gpu_count": 1,
            "workspace_mount_path": "/workspace",
        },
        "runtime": runtime,
    }
    attestation_path = tmp_path / "destination-attestation.json"
    attestation_path.write_bytes(admission._canonical_json_bytes(attestation))
    attestation_sha256 = hashlib.sha256(attestation_path.read_bytes()).hexdigest()
    storage_identity = {
        "kind": "runpod_network_volume_v1",
        "identity_id": network_volume_id,
        "provider_resource_id": network_volume_id,
        "descriptor": {
            "provider": "runpod",
            "resource_type": "network_volume",
            "resource_id": network_volume_id,
        },
    }
    volume_snapshot_base = {
        "provider": "runpod",
        "resource_role": "network_volume",
        "resource_type": "network_volume",
        "resource_id": network_volume_id,
        "resource_name": "nano-h6-volume",
        "storage_resource_id": network_volume_id,
        "storage_identity": storage_identity,
        "data_center_id": "EU-RO-1",
        "secure_cloud": None,
        "lifecycle_state": "available",
        "observed_at_utc": "2026-08-03T00:09:59Z",
        "observation_evidence_sha256": "a" * 64,
        "observation_for_operation_id": None,
    }
    destination_snapshot_base = {
        "provider": "runpod",
        "resource_role": "destination",
        "resource_type": "pod",
        "resource_id": pod_id,
        "resource_name": "nano-h6-destination",
        "machine_id": machine_id,
        "storage_resource_id": network_volume_id,
        "storage_identity": storage_identity,
        "data_center_id": "EU-RO-1",
        "image_id": admission._EXPECTED_IMAGE_ID,
        "configured_gpu_count": 1,
        "gpu_model": admission._EXPECTED_SPEC_GPU_MODEL,
        "gpu_count": 1,
        "lifecycle_state": "running",
        "secure_cloud": True,
        "workspace_state": "accessible",
        "observed_at_utc": attestation["observed_at_utc"],
        "runtime_sha256": snapshot_runtime_sha256
        or admission._EXPECTED_RUNTIME_AUTHORITY_SHA256,
        "observation_evidence_sha256": attestation_sha256,
        "observation_for_operation_id": None,
    }
    create_volume_snapshot = {
        **volume_snapshot_base,
        "observation_for_operation_id": "create-volume-op",
    }
    create_pod_snapshot = {
        **destination_snapshot_base,
        "observation_evidence_sha256": "b" * 64,
        "observation_for_operation_id": creation_operation_id,
    }
    raw_events = [
        ("RUN_INITIALIZED", {"spec": spec}),
        (
            "OPERATION_STARTED",
            {
                "operation_id": "create-volume-op",
                "kind": "create_volume",
                "resource_role": "network_volume",
                "target_id": "nano-h6-volume",
            },
        ),
        ("PROVIDER_SNAPSHOT", create_volume_snapshot),
        (
            "OPERATION_FINISHED",
            {"operation_id": "create-volume-op", "outcome": "succeeded"},
        ),
        (
            "OPERATION_STARTED",
            {
                "operation_id": creation_operation_id,
                "kind": "create_pod",
                "resource_role": "destination",
                "target_id": "nano-h6-destination",
            },
        ),
        ("PROVIDER_SNAPSHOT", create_pod_snapshot),
        (
            "OPERATION_FINISHED",
            {"operation_id": creation_operation_id, "outcome": "succeeded"},
        ),
        (
            "OPERATION_STARTED",
            {
                "operation_id": rehydration_operation_id,
                "kind": "rehydrate_destination",
                "resource_role": "destination",
                "target_id": pod_id,
            },
        ),
        (
            "OPERATION_FINISHED",
            {
                "operation_id": rehydration_operation_id,
                "outcome": "succeeded",
                "evidence_sha256": attestation_sha256,
            },
        ),
        ("PROVIDER_SNAPSHOT", _guard_snapshot(volume_snapshot_base)),
        ("PROVIDER_SNAPSHOT", _guard_snapshot(destination_snapshot_base)),
    ]
    events: list[dict[str, object]] = []
    previous_sha256: str | None = None
    for index, (event_name, payload) in enumerate(raw_events, start=1):
        event = _signed_event(
            index,
            previous_sha256,
            event=event_name,
            payload=payload,
        )
        events.append(event)
        previous_sha256 = event["event_sha256"]
    final_volume_event = events[-2]
    final_destination_event = events[-1]
    prepared_payload = {
        "schema": admission._ADMISSION_SYNC_SCHEMA,
        "transaction_id": "admission-sync-1",
        "run_id": run_id,
        "phase": "REPORTS_FROZEN",
        "ledger_tail_before": final_destination_event["event_sha256"],
        "final_network_volume_snapshot_event_index": final_volume_event["index"],
        "final_network_volume_snapshot_event_sha256": final_volume_event[
            "event_sha256"
        ],
        "final_destination_snapshot_event_index": final_destination_event["index"],
        "final_destination_snapshot_event_sha256": final_destination_event[
            "event_sha256"
        ],
        "destination_resource_id": pod_id,
        "network_volume_resource_id": network_volume_id,
        "authorization_envelope_sha256": spec["authorization_envelope_sha256"],
        "upload_allowlist_sha256": spec["upload_allowlist_sha256"],
        "runtime_authority_sha256": spec["runtime_authority_sha256"],
        "isolated_execution_contract": copy.deepcopy(
            admission._ISOLATED_EXECUTION_CONTRACT
        ),
        "remote_root": tmp_path.as_posix(),
        "relative_paths": {
            "run_spec": "runops/current/RUN_SPEC.json",
            "run_events": "runops/current/RUN_EVENTS.jsonl",
            "destination_attestation": "controls/destination-attestation.json",
            "admission_receipt": "receipts/destination-admission.json",
        },
        "transfer_protocol": admission._ADMISSION_SYNC_PROTOCOL,
        "transfer_provider_state_mutated": False,
        "transfer_guard_state_mutated": False,
    }
    prepared_event = _signed_event(
        len(events) + 1,
        previous_sha256,
        event=admission._ADMISSION_SYNC_EVENT,
        payload=prepared_payload,
    )
    events.append(prepared_event)
    previous_sha256 = prepared_event["event_sha256"]
    authority = {
        "spec": spec,
        "ledger_events": tuple(events),
        "ledger": {"tail_sha256": previous_sha256},
    }
    controls = {
        "authorization": {
            "isolated_execution_contract": copy.deepcopy(
                admission._ISOLATED_EXECUTION_CONTRACT
            )
        },
        "runtime_authority": {"required_destination": required_destination},
        "identities": {
            "runtime_authority": {
                "sha256": admission._EXPECTED_RUNTIME_AUTHORITY_SHA256
            }
        },
    }
    return authority, controls, attestation_path


def test_static_upload_policy_binds_exact_semantic_roles() -> None:
    first = {"role": "package", "relative_path": "input.tar.gz", **_identity(1)}
    second = {
        "role": "runtime_authority",
        "relative_path": "runtime.json",
        **_identity(2),
    }
    expected = {"package": first, "runtime_authority": second}

    observed = admission._require_exact_static_policy_rows(
        [first, second], expected_by_role=expected
    )

    assert observed == {
        "input.tar.gz": _identity(1),
        "runtime.json": _identity(2),
    }


@pytest.mark.parametrize("mutation", ["swap", "relabel"])
def test_static_upload_policy_rejects_role_tampering(mutation: str) -> None:
    first = {"role": "package", "relative_path": "input.tar.gz", **_identity(1)}
    second = {
        "role": "runtime_authority",
        "relative_path": "runtime.json",
        **_identity(2),
    }
    expected = {"package": first, "runtime_authority": second}
    tampered = [dict(first), dict(second)]
    if mutation == "swap":
        tampered[0]["role"], tampered[1]["role"] = (
            tampered[1]["role"],
            tampered[0]["role"],
        )
    else:
        tampered[0]["role"] = "renamed_package"

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="bind exact static identities to roles",
    ):
        admission._require_exact_static_policy_rows(tampered, expected_by_role=expected)


def test_authorization_policy_hash_dag_is_one_way() -> None:
    authority, controls = _policy_context()
    policy = _policy()

    assert (
        admission._require_upload_policy_header(
            policy, authority=authority, controls=controls
        )
        == "nano-h6-development-free-upload-v1"
    )

    back_edge = copy.deepcopy(policy)
    back_edge["authorities"]["authorization_envelope_sha256"] = "e" * 64
    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="authority DAG",
    ):
        admission._require_upload_policy_header(
            back_edge, authority=authority, controls=controls
        )


def test_guard_native_destination_spec_needs_no_ad_hoc_control_fields() -> None:
    spec = {
        "allowed_data_center_ids": ["EU-RO-1"],
        "allowed_image_ids": [admission._EXPECTED_IMAGE_ID],
        "allowed_gpu_models": [admission._EXPECTED_SPEC_GPU_MODEL],
        "storage_class": "network",
        "upload_allowlist_sha256": "f" * 64,
    }
    authority_hashes = {
        "runtime_authority_sha256": admission._EXPECTED_RUNTIME_AUTHORITY_SHA256,
        "required_artifacts_manifest_sha256": (
            admission._EXPECTED_REQUIRED_ARTIFACTS_SHA256
        ),
    }

    admission._require_guard_destination_spec(spec, authority_hashes)

    assert "upload_policy_sha256" not in spec
    assert "predecessor_binding_sha256" not in spec
    assert "destination_selection_sha256" not in spec


def test_installed_guard_builds_a_spec_accepted_by_admission() -> None:
    guard_path = Path(
        os.environ.get(
            "NANO_RUN_GUARD_PATH",
            "/Volumes/Express4M2/Offload/codex/skills/"
            "nano-runpod-operator/scripts/nano_run_guard.py",
        )
    )
    if not guard_path.is_file():
        pytest.skip("installed Nano RunPod guard is unavailable")
    guard = runpy.run_path(str(guard_path))
    required_artifacts = (
        Path(__file__).resolve().parents[2]
        / "artifacts/nano_h6/runops/evaluation-controls/required-artifacts.json"
    )
    arguments = argparse.Namespace(
        run_id="h6-eval",
        max_total_usd="5",
        evaluation_authority_id="nano-h6-dev-one-shot-v1",
        authority_dir=str(guard["canonical_authority_dir"]()),
        authorization_envelope_sha256="1" * 64,
        preregistration_sha256="2" * 64,
        freeze_sha256="3" * 64,
        package_sha256="4" * 64,
        evaluator_sha256="5" * 64,
        upload_allowlist_sha256="6" * 64,
        runtime_authority_sha256=admission._EXPECTED_RUNTIME_AUTHORITY_SHA256,
        development_sha256="7" * 64,
        required_artifacts_manifest=required_artifacts,
        required_artifacts_manifest_sha256=(
            admission._EXPECTED_REQUIRED_ARTIFACTS_SHA256
        ),
        allowed_gpu_model=[admission._EXPECTED_SPEC_GPU_MODEL],
        allowed_image_id=[admission._EXPECTED_IMAGE_ID],
        allowed_data_center_id=["EU-RO-1"],
        allowed_cloud_tier="secure",
        storage_class="network",
    )

    spec = guard["build_spec"](arguments, mode="fresh", initial_phase="PLANNED")
    authority_hashes = {name: spec[name] for name in admission._AUTHORITY_HASH_FIELDS}

    admission._require_guard_destination_spec(spec, authority_hashes)
    assert spec["upload_allowlist_sha256"] == "6" * 64
    assert "upload_policy_sha256" not in spec


def test_guard_native_network_volume_identity_is_accepted() -> None:
    network_volume_id = "volume-123"
    storage_identity = {
        "kind": "runpod_network_volume_v1",
        "identity_id": network_volume_id,
        "provider_resource_id": network_volume_id,
        "descriptor": {
            "provider": "runpod",
            "resource_type": "network_volume",
            "resource_id": network_volume_id,
        },
    }

    admission._require_guard_network_volume_identity(
        storage_identity,
        network_volume_id=network_volume_id,
    )

    host_local_shape = copy.deepcopy(storage_identity)
    host_local_shape["descriptor"] = {
        "provider": "runpod",
        "pod_id": "pod-123",
        "machine_id": "machine-123",
        "mount_path": "/workspace",
    }
    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="network-volume identity is invalid",
    ):
        admission._require_guard_network_volume_identity(
            host_local_shape,
            network_volume_id=network_volume_id,
        )


def test_destination_attestation_binds_final_unbound_snapshots_and_operations(
    tmp_path: Path,
) -> None:
    authority, controls, attestation_path = _destination_attestation_context(tmp_path)

    result, tracked = admission._authenticate_destination_attestation(
        authority=authority,
        controls=controls,
        attestation_path=attestation_path,
        destination_id="pod-123",
    )

    assert result["observation_operation_id"] is None
    assert result["operation_lineage"]["create_volume"]["outcome"] == "succeeded"
    assert result["operation_lineage"]["create_pod"]["operation_id"] == (
        "create-pod-op"
    )
    assert (
        result["operation_lineage"]["rehydrate_destination"]["operation_id"]
        == "rehydrate-op"
    )
    assert result["network_volume_provider_snapshot_event_index"] == 10
    assert result["provider_snapshot_event_index"] == 11
    assert result["admission_sync"]["prepared_event_index"] == 12
    assert (
        result["admission_sync"]["prepared_event_sha256"]
        == authority["ledger"]["tail_sha256"]
    )
    assert result["admission_sync"]["isolated_execution_contract"] == (
        admission._ISOLATED_EXECUTION_CONTRACT
    )
    assert tracked[attestation_path] == result["identity"]

    admission._require_admission_sync_paths(
        result["admission_sync"],
        upload_root=tmp_path,
        run_spec_path=tmp_path / "runops/current/RUN_SPEC.json",
        run_events_path=tmp_path / "runops/current/RUN_EVENTS.jsonl",
        destination_attestation_path=(
            tmp_path / "controls/destination-attestation.json"
        ),
        output_path=tmp_path / "receipts/destination-admission.json",
    )


def test_destination_attestation_requires_terminal_admission_sync(
    tmp_path: Path,
) -> None:
    authority, controls, attestation_path = _destination_attestation_context(tmp_path)
    events = authority["ledger_events"][:-1]
    authority["ledger_events"] = events
    authority["ledger"] = {"tail_sha256": events[-1]["event_sha256"]}

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="admission-sync preparation",
    ):
        admission._authenticate_destination_attestation(
            authority=authority,
            controls=controls,
            attestation_path=attestation_path,
            destination_id="pod-123",
        )


def test_admission_sync_rejects_authority_substitution(tmp_path: Path) -> None:
    authority, controls, attestation_path = _destination_attestation_context(tmp_path)
    prepared = authority["ledger_events"][-1]
    prepared["payload"]["upload_allowlist_sha256"] = "f" * 64
    prepared["event_sha256"] = _signed_event(
        prepared["index"],
        prepared["previous_sha256"],
        event=prepared["event"],
        payload=prepared["payload"],
    )["event_sha256"]
    authority["ledger"]["tail_sha256"] = prepared["event_sha256"]

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="does not bind the final destination state",
    ):
        admission._authenticate_destination_attestation(
            authority=authority,
            controls=controls,
            attestation_path=attestation_path,
            destination_id="pod-123",
        )


@pytest.mark.parametrize("mutation", ["flags", "extra"])
def test_admission_sync_rejects_execution_contract_substitution(
    tmp_path: Path, mutation: str
) -> None:
    authority, controls, attestation_path = _destination_attestation_context(tmp_path)
    prepared = authority["ledger_events"][-1]
    contract = prepared["payload"]["isolated_execution_contract"]
    if mutation == "flags":
        contract["python_flags"] = ["-S", "-I"]
    else:
        contract["unbound"] = True
    prepared["event_sha256"] = _signed_event(
        prepared["index"],
        prepared["previous_sha256"],
        event=prepared["event"],
        payload=prepared["payload"],
    )["event_sha256"]
    authority["ledger"]["tail_sha256"] = prepared["event_sha256"]

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="does not bind the final destination state",
    ):
        admission._authenticate_destination_attestation(
            authority=authority,
            controls=controls,
            attestation_path=attestation_path,
            destination_id="pod-123",
        )


def test_admission_sync_rejects_suffix_after_prepared_event(tmp_path: Path) -> None:
    authority, controls, attestation_path = _destination_attestation_context(tmp_path)
    prepared = authority["ledger_events"][-1]
    suffix = _signed_event(
        prepared["index"] + 1,
        prepared["event_sha256"],
        event="UNRELATED_EVENT",
        payload={"reason": "must remain locked"},
    )
    authority["ledger_events"] = (*authority["ledger_events"], suffix)
    authority["ledger"] = {"tail_sha256": suffix["event_sha256"]}

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="admission-sync preparation",
    ):
        admission._authenticate_destination_attestation(
            authority=authority,
            controls=controls,
            attestation_path=attestation_path,
            destination_id="pod-123",
        )


def test_admission_sync_rejects_invalid_transaction_id(tmp_path: Path) -> None:
    authority, controls, attestation_path = _destination_attestation_context(tmp_path)
    prepared = authority["ledger_events"][-1]
    prepared["payload"]["transaction_id"] = "invalid transaction"
    prepared["event_sha256"] = _signed_event(
        prepared["index"],
        prepared["previous_sha256"],
        event=prepared["event"],
        payload=prepared["payload"],
    )["event_sha256"]
    authority["ledger"]["tail_sha256"] = prepared["event_sha256"]

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="transaction ID is invalid",
    ):
        admission._authenticate_destination_attestation(
            authority=authority,
            controls=controls,
            attestation_path=attestation_path,
            destination_id="pod-123",
        )


def test_admission_sync_rejects_wrong_prepared_tail(tmp_path: Path) -> None:
    authority, controls, attestation_path = _destination_attestation_context(tmp_path)
    authority["ledger"]["tail_sha256"] = "e" * 64

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="does not bind the final destination state",
    ):
        admission._authenticate_destination_attestation(
            authority=authority,
            controls=controls,
            attestation_path=attestation_path,
            destination_id="pod-123",
        )


def test_admission_sync_rejects_prepared_path_substitution(tmp_path: Path) -> None:
    authority, controls, attestation_path = _destination_attestation_context(tmp_path)
    result, _tracked = admission._authenticate_destination_attestation(
        authority=authority,
        controls=controls,
        attestation_path=attestation_path,
        destination_id="pod-123",
    )
    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="paths differ",
    ):
        admission._require_admission_sync_paths(
            result["admission_sync"],
            upload_root=tmp_path,
            run_spec_path=tmp_path / "runops/current/WRONG.json",
            run_events_path=tmp_path / "runops/current/RUN_EVENTS.jsonl",
            destination_attestation_path=(
                tmp_path / "controls/destination-attestation.json"
            ),
            output_path=tmp_path / "receipts/destination-admission.json",
        )


def test_destination_attestation_rejects_runtime_values_digest_substitution(
    tmp_path: Path,
) -> None:
    runtime_values = {
        "python": "3.12.3",
        "torch": "2.8.0+cu128",
        "tokenizers": "0.22.2",
        "cuda": "12.8",
        "gpu": admission._EXPECTED_CONFIGURED_GPU_ID,
        "gpu_count": 1,
        "cublas_workspace_config": ":4096:8",
        "platform": "Linux-6.8.0-90-generic-x86_64-with-glibc2.39",
    }
    incorrect_runtime_digest = hashlib.sha256(
        admission._canonical_json_bytes(runtime_values).removesuffix(b"\n")
    ).hexdigest()
    authority, controls, attestation_path = _destination_attestation_context(
        tmp_path,
        snapshot_runtime_sha256=incorrect_runtime_digest,
    )

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="provider snapshot does not match",
    ):
        admission._authenticate_destination_attestation(
            authority=authority,
            controls=controls,
            attestation_path=attestation_path,
            destination_id="pod-123",
        )


def test_ledger_replay_rejects_payload_tampering(tmp_path: Path) -> None:
    first = _signed_event(
        1,
        None,
        event="RUN_INITIALIZED",
        payload={"spec": {"run_id": "h6-eval"}},
    )
    second = _signed_event(
        2,
        first["event_sha256"],
        event="PROVIDER_SNAPSHOT",
        payload={"resource_id": "destination"},
    )
    ledger_path = tmp_path / "RUN_EVENTS.jsonl"
    ledger_path.write_bytes(
        admission._canonical_json_bytes(first) + admission._canonical_json_bytes(second)
    )

    _first, events, ledger, _identity_value = admission._verify_ledger(
        ledger_path, expected_tail_sha256=second["event_sha256"]
    )
    assert len(events) == 2
    assert ledger["tail_sha256"] == second["event_sha256"]

    second["payload"] = {"resource_id": "tampered"}
    ledger_path.write_bytes(
        admission._canonical_json_bytes(first) + admission._canonical_json_bytes(second)
    )
    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="ledger hash mismatch",
    ):
        admission._verify_ledger(
            ledger_path, expected_tail_sha256=second["event_sha256"]
        )


def test_ledger_replay_rejects_noncanonical_bytes(tmp_path: Path) -> None:
    event = _signed_event(
        1,
        None,
        event="RUN_INITIALIZED",
        payload={"spec": {"run_id": "h6-eval"}},
    )
    noncanonical = (json.dumps(event) + "\n").encode("utf-8")
    assert noncanonical != admission._canonical_json_bytes(event)
    ledger_path = tmp_path / "RUN_EVENTS.jsonl"
    ledger_path.write_bytes(noncanonical)

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="ledger record 1 is not canonical",
    ):
        admission._verify_ledger(
            ledger_path,
            expected_tail_sha256=event["event_sha256"],
        )


def test_prohibited_broken_symlink_is_present_and_refused(tmp_path: Path) -> None:
    broken_link = tmp_path / "development"
    broken_link.symlink_to(tmp_path / "missing-target")

    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="prohibited data path is present",
    ):
        admission._assert_paths_absent(
            [broken_link, tmp_path / "fresh", tmp_path / "private"]
        )


def test_receipt_is_canonical_and_no_clobber(tmp_path: Path) -> None:
    receipt_path = tmp_path / "admission.json"
    receipt = {"z": 1, "a": [True, None]}

    identity = admission._write_no_clobber(receipt_path, receipt)

    assert receipt_path.read_bytes() == b'{"a":[true,null],"z":1}\n'
    assert identity["sha256"] == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(
        admission.H6DestinationAdmissionError,
        match="already exists",
    ):
        admission._write_no_clobber(receipt_path, receipt)
