"""Synthetic operational checks; never open an H7 screen or run a model."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
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


@pytest.mark.parametrize("operation", [runner.reserve, runner.run])
@pytest.mark.parametrize("stage", ["training", "evaluation"])
def test_consumed_h7_blocks_all_stages_before_reading_or_writing(operation, stage, tmp_path):
    args = argparse.Namespace(stage=stage, registry=tmp_path / "registry",
                              marker_root=tmp_path / "fresh-kernel")
    with pytest.raises(runner.H7OperationError, match="H7 is consumed"):
        operation(args)
    assert list(tmp_path.iterdir()) == []


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


def test_archived_runtime_probe_rejects_mismatched_pins_under_optimization(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    monkeypatch.setenv("H7_RUNTIME_JSON", str(tmp_path / "runtime.json"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="wrong",
                        version=SimpleNamespace(cuda=None), cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(__version__="0.22.2"))
    with pytest.raises(SystemExit, match="runtime mismatch"):
        exec(compile(runner.RUNTIME_PROBE, "<runtime_probe>", "exec", optimize=2), {})
