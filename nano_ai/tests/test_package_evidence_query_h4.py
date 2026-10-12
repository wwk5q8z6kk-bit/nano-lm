from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from nano_ai.training import package_evidence_query_h4


def test_archive_bytes_are_deterministic_and_normalized() -> None:
    members = {"z.txt": b"last\n", "a.txt": b"first\n"}
    generated = {"RUN_H4_TRAIN.sh": b"#!/usr/bin/env bash\ntrue\n"}

    first = package_evidence_query_h4._archive_bytes(members, generated)
    second = package_evidence_query_h4._archive_bytes(members, generated)

    assert first == second
    with (
        gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as uncompressed,
        tarfile.open(fileobj=uncompressed, mode="r:") as archive,
    ):
        entries = archive.getmembers()
        assert [entry.name for entry in entries] == [
            "nano-h4-runpod/RUN_H4_TRAIN.sh",
            "nano-h4-runpod/a.txt",
            "nano-h4-runpod/z.txt",
        ]
        assert all(entry.mtime == 0 for entry in entries)
        assert all(entry.uid == entry.gid == 0 for entry in entries)
        assert entries[0].mode == 0o755
        assert all(entry.mode == 0o644 for entry in entries[1:])


def test_archive_self_verification_authenticates_every_member() -> None:
    members = {"input.txt": b"frozen input\n"}
    generated = {
        "BUNDLE_MANIFEST.json": b"{}\n",
        "RUN_H4.sh": b"#!/usr/bin/env bash\ntrue\n",
    }
    archive = package_evidence_query_h4._archive_bytes(members, generated)
    identities = {
        name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in {**members, "RUN_H4.sh": generated["RUN_H4.sh"]}.items()
    }

    package_evidence_query_h4._verify_archive_bytes(
        archive,
        member_identity=identities,
    )

    corrupted = {name: dict(identity) for name, identity in identities.items()}
    corrupted["input.txt"]["sha256"] = "0" * 64
    with pytest.raises(
        package_evidence_query_h4.H4PackagingError,
        match="member identity mismatch",
    ):
        package_evidence_query_h4._verify_archive_bytes(
            archive,
            member_identity=corrupted,
        )


def test_archive_rejects_generated_member_collision() -> None:
    with pytest.raises(
        package_evidence_query_h4.H4PackagingError,
        match="collide",
    ):
        package_evidence_query_h4._archive_bytes(
            {"RUN_H4.sh": b"input"},
            {"RUN_H4.sh": b"generated"},
        )


def test_run_scripts_freeze_training_before_development_and_evaluation() -> None:
    training = package_evidence_query_h4._training_run_script(
        training_manifest_sha256="a" * 64
    ).decode("utf-8")
    evaluation = package_evidence_query_h4._evaluation_run_script(
        training_manifest_sha256="a" * 64
    ).decode("utf-8")

    assert training.count("train_evidence_query_h4 --data-dir") == 2
    assert "--seed 20260805" in training
    assert "--seed 20260806" in training
    assert "python -m nano_ai.training.evaluate_evidence_query_h4" not in training
    assert "test ! -e h2_development" in training
    assert "TRAINING_REPORT_SHA256SUMS" in training
    assert "python -m nano_ai.training.evaluate_evidence_query_h4" in evaluation
    assert "--training-manifest-sha256 " + "a" * 64 in evaluation
    assert package_evidence_query_h4.H2_DEVELOPMENT_SHA256 in evaluation
    assert "fresh_v1" not in training + evaluation
    assert "test ! -e results" in training
    assert "python -m venv --system-site-packages .h4-venv" in training
    assert 'export PATH="$PWD/.h4-venv/bin:$PATH"' in training
    assert 'export PATH="$PWD/.h4-venv/bin:$PATH"' in evaluation
    assert training.index("pip install") < training.index("import platform, torch")
    assert "requirements-h4-runpod.txt" in training
    assert "torch.__version__ == \"2.8.0+cu128\"" in training
    assert "torch.version.cuda == \"12.8\"" in training
    assert "torch.cuda.get_device_name(0) == \"NVIDIA GeForce RTX 5090\"" in training
    assert "NVIDIA GeForce RTX 4090" not in training


def test_packager_rejects_existing_output_before_reading_inputs(tmp_path: Path) -> None:
    output = tmp_path / "already-there"
    output.mkdir()

    with pytest.raises(
        package_evidence_query_h4.H4PackagingError,
        match="already exists",
    ):
        package_evidence_query_h4.package_h4_runpod_bundle(
            repo_root=tmp_path / "missing-repo",
            data_dir=tmp_path / "missing-data",
            output_dir=output,
        )


def test_bundle_source_inventory_is_an_exact_safe_allowlist(tmp_path: Path) -> None:
    members = package_evidence_query_h4._source_members(tmp_path)

    assert set(members) == set(package_evidence_query_h4._SOURCE_ALLOWLIST)
    assert all(path == tmp_path / name for name, path in members.items())
    assert not any(
        marker in name.casefold()
        for name in members
        for marker in (
            "benchmark",
            "fresh_v0",
            "fresh_v1",
            "sealed",
            "__pycache__",
        )
    )
