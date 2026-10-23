"""Synthetic operational checks; never open an H7 screen or run a model."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from scripts import run_h7_kaggle as runner


def metadata(stage: str = "training") -> dict:
    return {"is_private": True, "enable_gpu": False, "enable_tpu": False,
            "dataset_sources": [f"owner/{stage}"] + (["owner/treatments"] if stage == "evaluation" else []),
            "kernel_sources": [], "competition_sources": [], "model_sources": []}


def canonical(stage: str = "training") -> str:
    suffix = "train" if stage == "training" else "eval"
    commands = (
        "python -m nano_ai.training.train_evidence_query_h7 --seed 20260805\n"
        "python -m nano_ai.training.train_evidence_query_h7 --seed 20260806\n"
        if stage == "training" else
        "sha256sum -c treatments/results/TRAINING_REPORT_SHA256SUMS\n"
        "python -m nano_ai.training.evaluate_evidence_query_h7 --batch-size 32\n"
    )
    return (f"python -m venv --system-site-packages .h7-{suffix}-venv\n"
            f'export PATH="$PWD/.h7-{suffix}-venv/bin:$PATH"\n'
            "python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-h7-kaggle.txt\n" + commands)


def reserve_args(tmp_path: Path, stage: str = "training") -> argparse.Namespace:
    path = tmp_path / f"{stage}-metadata.json"
    path.write_text(json.dumps(metadata(stage)))
    return argparse.Namespace(stage=stage, registry=tmp_path / "authority", kernel_metadata=path,
        stage_dataset=f"owner/{stage}", treatment_dataset="owner/treatments" if stage == "evaluation" else None,
        treatment_archive_sha256="a" * 64 if stage == "evaluation" else None,
        verified_private_dataset=metadata(stage)["dataset_sources"], training_terminal=None)


def synthetic_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "input/nano-h7-training"
    root.mkdir(parents=True)
    (root / "RUN_H7_TRAIN.sh").write_text(canonical())
    (root / "frozen.txt").write_text("synthetic data, not a scientific record\n")
    manifest = {
        "schema_version": "nano.h7-kaggle-package.v1", "stage": "training",
        "archive_root": "nano-h7-training", "git_sha": runner.GIT_SHA,
        "recipe": "nano-evidence-query-h7-boundary-margin-v1", "authorized_cost_usd": 0,
        "max_runtime_hours": 4, "training_seeds": [20260805, 20260806], "venue": "Kaggle free CPU",
        "members": {path.name: {"bytes": path.stat().st_size, "sha256": runner.digest(path)} for path in root.iterdir()},
    }
    path = root / "BUNDLE_MANIFEST.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setitem(runner.PACKAGES, "training", {**runner.PACKAGES["training"], "manifest_sha256": runner.digest(path)})
    return root


def test_reservation_is_create_only_and_bound_to_script(tmp_path: Path) -> None:
    args = reserve_args(tmp_path)
    path = runner.reserve(args)
    original = path.read_bytes()
    assert runner.validate_reservation(json.loads(original)) == "training"
    with pytest.raises(FileExistsError):
        runner.reserve(args)
    assert path.read_bytes() == original
    changed = json.loads(original)
    changed["runner_sha256"] = "0" * 64
    with pytest.raises(runner.H7OperationError, match="Runner differs"):
        runner.validate_reservation(changed)


def test_evaluation_receives_only_remaining_runtime(tmp_path: Path) -> None:
    training = json.loads(runner.reserve(reserve_args(tmp_path)).read_bytes())
    terminal = tmp_path / "terminal.json"
    terminal.write_text(json.dumps({"attempt_id": training["attempt_id"], "status": "completed", "exit_code": 0, "error": None, "elapsed_seconds": 5000.2}))
    args = reserve_args(tmp_path, "evaluation")
    args.training_terminal = terminal
    evaluation = json.loads(runner.reserve(args).read_bytes())
    assert evaluation["max_runtime_seconds"] == 9399
    assert runner.validate_reservation(evaluation) == "evaluation"
    evaluation["max_runtime_seconds"] = 14400
    with pytest.raises(runner.H7OperationError, match="remaining cumulative"):
        runner.validate_reservation(evaluation)


@pytest.mark.parametrize("change", [{"is_private": False}, {"enable_gpu": True}, {"dataset_sources": ["owner/training", "owner/screen"]}, {"kernel_sources": ["owner/other"]}])
def test_rejects_public_paid_or_mixed_inputs(change: dict) -> None:
    with pytest.raises(runner.H7OperationError):
        runner.validate_metadata({**metadata(), **change}, "training", "owner/training", None)


@pytest.mark.parametrize("damage", ["extra", "modified", "symlink"])
def test_full_tree_validation_rejects_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    bundle = synthetic_bundle(tmp_path, monkeypatch)
    runner.validate_bundle(bundle, "training")
    if damage == "extra":
        (bundle / "unexpected.txt").write_text("extra")
    elif damage == "modified":
        (bundle / "frozen.txt").write_text("changed")
    else:
        (bundle / "linked-dir").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(runner.H7OperationError):
        runner.validate_bundle(bundle, "training")


@pytest.mark.parametrize("stage", ["training", "evaluation"])
def test_derived_wrapper_preserves_scientific_commands(stage: str) -> None:
    original = canonical(stage)
    wrapper = runner.derive_wrapper(original, stage)
    commands = lambda script: [line for line in script.splitlines() if line.startswith("python -m nano_ai.training.")]
    assert commands(wrapper) == commands(original)
    assert "python -m venv" not in wrapper
    assert "--target .h7-" in wrapper
    assert wrapper.count('python "$H7_EVENT_MARKER"') == (2 if stage == "training" else 1)
    if stage == "evaluation":
        assert "(cd treatments && sha256sum -c results/TRAINING_REPORT_SHA256SUMS)" in wrapper
    with pytest.raises(runner.H7OperationError, match="exactly once"):
        runner.derive_wrapper(original + original, stage)


def test_additional_input_mount_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "input"
    stage = root / "training"
    stage.mkdir(parents=True)
    (stage / "file").write_text("allowed")
    runner.validate_mounts(root, stage, None)
    (root / "other").write_text("unexpected")
    with pytest.raises(runner.H7OperationError, match="Unexpected mounted input"):
        runner.validate_mounts(root, stage, None)


def test_treatment_archive_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "treatments.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("results/../../escape")
        info.size = 1
        handle.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(runner.H7OperationError, match="Unsafe treatment"):
        runner.extract_treatments(archive, runner.digest(archive), tmp_path / "extracted")
    assert not (tmp_path / "escape").exists()


def test_failure_preserves_markers_logs_and_prevents_reentry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = synthetic_bundle(tmp_path, monkeypatch)
    frozen_before = {p.name: p.read_bytes() for p in source.iterdir()}
    reservation = runner.reserve(reserve_args(tmp_path))
    def fail(command, cwd, env, log, remaining):
        assert 0 < remaining <= runner.MAX_SECONDS
        log.write_text("synthetic execution failure\n")
        (cwd / "results").mkdir()
        (cwd / "results/partial.log").write_text("partial evidence\n")
        return 9
    monkeypatch.setattr(runner, "run_process", fail)
    args = argparse.Namespace(reservation=reservation, source_dir=source, input_root=tmp_path / "input", output_dir=tmp_path / "output", marker_root=tmp_path / "markers", treatment_archive=None)
    assert runner.run(args) == 9
    evidence = args.output_dir / "evidence"
    terminal = json.loads((evidence / "terminal.json").read_bytes())
    assert terminal["status"] == "failed_or_interrupted"
    assert terminal["retry_authorized"] is False
    assert (evidence / "runner.log").read_text() == "synthetic execution failure\n"
    assert (args.output_dir / "nano-h7-training-attempt-evidence.tar.gz").is_file()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == frozen_before
    args.output_dir = tmp_path / "different-output"
    with pytest.raises(FileExistsError):
        runner.run(args)


def test_subprocess_timeout_keeps_partial_log(tmp_path: Path) -> None:
    log = tmp_path / "timeout.log"
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_process([sys.executable, "-c", "import time; print('started', flush=True); time.sleep(60)"], tmp_path, {}, log, 0.2)
    assert log.read_text() == "started\n"


def test_checksum_inventory_rejects_unlisted_treatment_file(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "report.json").write_text("{}")
    checksums = results / "SHA256SUMS"
    checksums.write_text(f"{runner.digest(results / 'report.json')}  results/report.json\n")
    runner.verify_checksum_file(tmp_path, checksums, complete=True)
    (results / "extra").write_text("unlisted")
    with pytest.raises(runner.H7OperationError, match="inventory"):
        runner.verify_checksum_file(tmp_path, checksums, complete=True)
