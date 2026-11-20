from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from nano_ai.training import package_evidence_query_h7


def test_h7_package_contract_is_versioned_for_two_isolated_stages() -> None:
    assert (
        package_evidence_query_h7.H7_KAGGLE_PACKAGE_SCHEMA_VERSION
        == "nano.h7-kaggle-package.v1"
    )
    assert (
        package_evidence_query_h7.H7_READINESS_SCHEMA_VERSION
        == "nano.h7-kaggle-readiness.v1"
    )
    assert package_evidence_query_h7.AUTHORIZED_COST_USD == 0
    assert package_evidence_query_h7.MAX_RUNTIME_HOURS == 4


def test_h7_archive_is_deterministic_normalized_and_stage_rooted() -> None:
    members = {"z.txt": b"last\n", "a.txt": b"first\n"}
    generated = {"RUN_H7_TRAIN.sh": b"#!/usr/bin/env bash\ntrue\n"}

    first = package_evidence_query_h7._archive_bytes(
        root="stage", members=members, generated=generated
    )
    second = package_evidence_query_h7._archive_bytes(
        root="stage", members=members, generated=generated
    )

    assert first == second
    with (
        gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as uncompressed,
        tarfile.open(fileobj=uncompressed, mode="r:") as archive,
    ):
        entries = archive.getmembers()
        assert [entry.name for entry in entries] == [
            "stage/RUN_H7_TRAIN.sh",
            "stage/a.txt",
            "stage/z.txt",
        ]
        assert all(entry.mtime == 0 for entry in entries)
        assert all(entry.uid == entry.gid == 0 for entry in entries)
        assert entries[0].mode == 0o755
        assert all(entry.mode == 0o644 for entry in entries[1:])


def test_h7_archive_verifier_authenticates_every_nonmanifest_member() -> None:
    members = {"input.txt": b"frozen\n"}
    run_script = b"#!/usr/bin/env bash\ntrue\n"
    generated = {
        "BUNDLE_MANIFEST.json": b"{}\n",
        "RUN_H7_TRAIN.sh": run_script,
    }
    archive = package_evidence_query_h7._archive_bytes(
        root="stage", members=members, generated=generated
    )
    identities = {
        name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in {**members, "RUN_H7_TRAIN.sh": run_script}.items()
    }

    package_evidence_query_h7._verify_archive_bytes(
        archive, root="stage", member_identity=identities
    )
    identities["input.txt"]["sha256"] = "0" * 64
    with pytest.raises(
        package_evidence_query_h7.H7PackagingError,
        match="member identity mismatch",
    ):
        package_evidence_query_h7._verify_archive_bytes(
            archive, root="stage", member_identity=identities
        )


def test_h7_source_allowlists_keep_screen_out_of_training(tmp_path: Path) -> None:
    training = package_evidence_query_h7._source_members(tmp_path, stage="training")
    evaluation = package_evidence_query_h7._source_members(
        tmp_path, stage="evaluation"
    )

    assert "nano_ai/training/train_evidence_query_h7.py" in training
    assert "nano_ai/training/boundary_margin.py" in training
    assert "nano_ai/training/h7_boundary_screen_data.py" not in training
    assert "nano_ai/training/evaluate_evidence_query_h7.py" not in training
    assert "nano_ai/training/h7_boundary_screen_data.py" in evaluation
    assert "nano_ai/training/evaluate_evidence_query_h7.py" in evaluation
    assert "nano_ai/training/diagnose_h6.py" in evaluation
    assert not any("artifact" in name.casefold() for name in training)


def test_h7_scripts_enforce_two_stage_free_cpu_execution() -> None:
    training = package_evidence_query_h7._training_run_script(
        training_manifest_sha256="a" * 64
    ).decode()
    evaluation = package_evidence_query_h7._evaluation_run_script().decode()

    assert training.count("train_evidence_query_h7 --data-dir") == 2
    assert "--seed 20260805 --device cpu" in training
    assert "--seed 20260806 --device cpu" in training
    assert "evaluate_evidence_query_h7" not in training
    assert training.count("test ! -e h7_screen") >= 3
    assert "test ! -e ../nano-h7-evaluation" in training
    assert "TRAINING_REPORT_SHA256SUMS" in training
    assert "nano-h7-treatment-results.tar.gz" in training

    assert evaluation.count("evaluate_evidence_query_h7") == 2
    assert evaluation.count("--treatment-report") == 2
    assert "--device cpu" in evaluation
    assert "test ! -e h5_data" in evaluation
    assert "test ! -e checkpoints/anchors/nano_v01_scribe.pt" in evaluation
    assert package_evidence_query_h7.H7_SCREEN_SHA256 in evaluation
    assert package_evidence_query_h7.CONTROL_CHECKPOINT_SHA256 in evaluation
    for script in (training, evaluation):
        assert "trap cleanup EXIT" in script
        assert "trap - EXIT INT TERM HUP" in script


def test_h7_packager_rejects_bad_git_sha_before_reading_inputs(tmp_path: Path) -> None:
    with pytest.raises(
        package_evidence_query_h7.H7PackagingError,
        match="full lowercase commit SHA",
    ):
        package_evidence_query_h7.package_h7_kaggle_bundles(
            repo_root=tmp_path,
            data_dir=tmp_path / "missing-data",
            control_checkpoint=tmp_path / "missing-control",
            output_dir=tmp_path / "output",
            git_sha="main",
        )


def test_h7_packager_rejects_existing_output_before_reading_inputs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(
        package_evidence_query_h7.H7PackagingError,
        match="already exists",
    ):
        package_evidence_query_h7.package_h7_kaggle_bundles(
            repo_root=tmp_path,
            data_dir=tmp_path / "missing-data",
            control_checkpoint=tmp_path / "missing-control",
            output_dir=output,
            git_sha="a" * 40,
        )
