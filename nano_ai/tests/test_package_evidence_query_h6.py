from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from nano_ai.training import package_evidence_query_h6


def test_h6_packaging_contract_is_versioned_and_not_eager() -> None:
    assert (
        package_evidence_query_h6.H6_RUNPOD_BUNDLE_SCHEMA_VERSION
        == "nano.h6-runpod-bundle.v1"
    )
    assert (
        package_evidence_query_h6.H6_READINESS_SCHEMA_VERSION == "nano.h6-readiness.v1"
    )
    assert package_evidence_query_h6.BUNDLE_FILENAME == ("nano-h6-runpod-input.tar.gz")
    assert package_evidence_query_h6.READINESS_FILENAME == "READINESS.json"


def test_archive_bytes_are_deterministic_and_normalized() -> None:
    members = {"z.txt": b"last\n", "a.txt": b"first\n"}
    generated = {"RUN_H6_TRAIN.sh": b"#!/usr/bin/env bash\ntrue\n"}

    first = package_evidence_query_h6._archive_bytes(members, generated)
    second = package_evidence_query_h6._archive_bytes(members, generated)

    assert first == second
    with (
        gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as uncompressed,
        tarfile.open(fileobj=uncompressed, mode="r:") as archive,
    ):
        entries = archive.getmembers()
        assert [entry.name for entry in entries] == [
            "nano-h6-runpod/RUN_H6_TRAIN.sh",
            "nano-h6-runpod/a.txt",
            "nano-h6-runpod/z.txt",
        ]
        assert all(entry.mtime == 0 for entry in entries)
        assert all(entry.uid == entry.gid == 0 for entry in entries)
        assert entries[0].mode == 0o755
        assert all(entry.mode == 0o644 for entry in entries[1:])


def test_archive_self_verification_authenticates_every_member() -> None:
    members = {"input.txt": b"frozen input\n"}
    generated = {
        "BUNDLE_MANIFEST.json": b"{}\n",
        "RUN_H6.sh": b"#!/usr/bin/env bash\ntrue\n",
    }
    archive = package_evidence_query_h6._archive_bytes(members, generated)
    identities = {
        name: {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in {
            **members,
            "RUN_H6.sh": generated["RUN_H6.sh"],
        }.items()
    }

    package_evidence_query_h6._verify_archive_bytes(
        archive,
        member_identity=identities,
    )

    corrupted = {name: dict(identity) for name, identity in identities.items()}
    corrupted["input.txt"]["sha256"] = "0" * 64
    with pytest.raises(
        package_evidence_query_h6.H6PackagingError,
        match="member identity mismatch",
    ):
        package_evidence_query_h6._verify_archive_bytes(
            archive,
            member_identity=corrupted,
        )


def test_archive_rejects_generated_member_collision() -> None:
    with pytest.raises(
        package_evidence_query_h6.H6PackagingError,
        match="collide",
    ):
        package_evidence_query_h6._archive_bytes(
            {"RUN_H6.sh": b"input"},
            {"RUN_H6.sh": b"generated"},
        )


def test_run_scripts_freeze_training_before_development_and_evaluation() -> None:
    manifest_sha256 = "a" * 64
    training = package_evidence_query_h6._training_run_script(
        training_manifest_sha256=manifest_sha256
    ).decode("utf-8")
    evaluation = package_evidence_query_h6._evaluation_run_script(
        training_manifest_sha256=manifest_sha256
    ).decode("utf-8")

    assert training.count("train_evidence_query_h6 --data-dir") == 2
    assert "--seed 20260805" in training
    assert "--seed 20260806" in training
    assert "python -m nano_ai.training.evaluate_evidence_query_h6" not in training
    assert training.count("test ! -e h2_development") >= 4
    assert training.index("--seed 20260806") < training.rindex(
        "TRAINING_REPORT_SHA256SUMS.tmp"
    )
    assert "python -m nano_ai.training.evaluate_evidence_query_h6" in evaluation
    assert "--training-manifest-sha256 " + manifest_sha256 in evaluation
    assert evaluation.count("--training-report") == 2
    assert package_evidence_query_h6.H2_DEVELOPMENT_MANIFEST_SHA256 in evaluation
    assert package_evidence_query_h6.H2_DEVELOPMENT_SHA256 in evaluation
    assert evaluation.index("sha256sum -c") < evaluation.index(
        "sha256sum h2_development/manifest.json"
    )


def test_run_scripts_pin_matched_runtime_and_are_terminate_safe() -> None:
    training = package_evidence_query_h6._training_run_script(
        training_manifest_sha256="a" * 64
    ).decode("utf-8")
    evaluation = package_evidence_query_h6._evaluation_run_script(
        training_manifest_sha256="a" * 64
    ).decode("utf-8")

    assert "python -m venv --system-site-packages .h6-venv" in training
    assert 'export PATH="$PWD/.h6-venv/bin:$PATH"' in training
    assert 'export PATH="$PWD/.h6-venv/bin:$PATH"' in evaluation
    assert training.index("pip install") < training.index("import platform, torch")
    assert "requirements-h4-runpod.txt" in training
    assert 'platform.python_version().startswith("3.12.")' in training
    assert 'torch.__version__ == "2.8.0+cu128"' in training
    assert 'torch.version.cuda == "12.8"' in training
    assert 'tokenizers.__version__ == "0.22.2"' in training
    assert 'torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090"' in training
    assert "NVIDIA GeForce RTX 4090" not in training
    for script in (training, evaluation):
        assert "trap cleanup EXIT" in script
        assert "trap 'exit 130' INT" in script
        assert "trap 'exit 143' TERM" in script
        assert "trap 'exit 129' HUP" in script
        assert "trap - EXIT INT TERM HUP" in script
    assert "TRAINING_REPORT_SHA256SUMS.tmp" in training
    assert "nano-h6-results.tar.gz.tmp" in evaluation


def test_packager_rejects_existing_output_before_reading_inputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "already-there"
    output.mkdir()

    with pytest.raises(
        package_evidence_query_h6.H6PackagingError,
        match="already exists",
    ):
        package_evidence_query_h6.package_h6_runpod_bundle(
            repo_root=tmp_path / "missing-repo",
            data_dir=tmp_path / "missing-data",
            output_dir=output,
        )


def test_bundle_source_inventory_is_an_exact_safe_allowlist(tmp_path: Path) -> None:
    members = package_evidence_query_h6._source_members(tmp_path)

    assert set(members) == set(package_evidence_query_h6._SOURCE_ALLOWLIST)
    assert all(path == tmp_path / name for name, path in members.items())
    assert {
        "nano_ai/training/replay_mixture_data.py",
        "nano_ai/training/state_conditioned_evidence_query_model.py",
        "nano_ai/training/train_evidence_query_h6.py",
        "nano_ai/training/evaluate_evidence_query_h6.py",
        "nano_ai/training/package_evidence_query_h6.py",
        "nano_ai/tests/test_replay_mixture_data.py",
        "nano_ai/tests/test_state_conditioned_evidence_query_model.py",
        "nano_ai/tests/test_train_evidence_query_h6.py",
        "nano_ai/tests/test_evaluate_evidence_query_h6.py",
        "nano_ai/tests/test_package_evidence_query_h6.py",
        "requirements-h4-runpod.txt",
    } <= set(members)
    assert "nano_ai/training/evaluate_evidence_query_h4.py" not in members
    assert "nano_ai/training/package_evidence_query_h4.py" not in members
    assert not any(
        marker in name.casefold()
        for name in members
        for marker in (
            "artifact",
            "benchmark",
            "fresh_v0",
            "fresh_v1",
            "private",
            "sealed",
            "__pycache__",
        )
    )
