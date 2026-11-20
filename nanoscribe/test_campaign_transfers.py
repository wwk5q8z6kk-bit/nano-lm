"""Offline checks for the campaign's local artifact and transport boundaries."""
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pull_native_metrics
import pull_native_weights
import pull_qlora_adapter
from nanoscribe import runpod_wallet


def test_wallet_only_persists_budget_fields():
    raw = {"clientBalance": 12.25, "currentSpendPerHr": 0.3, "email": "test@example.invalid", "id": "fixture"}
    with patch.object(runpod_wallet.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps(raw))):
        assert runpod_wallet.query_live_balance() == {
            "ok": True, "client_balance_usd": 12.25, "current_spend_per_hr": 0.3,
        }
    with patch.object(runpod_wallet.subprocess, "run", return_value=SimpleNamespace(returncode=1, stderr=json.dumps(raw))):
        assert runpod_wallet.query_live_balance() == {"ok": False, "error": "runpodctl user failed (exit 1)"}


def test_native_weights_use_restricted_reader(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    run = "smoke_s0"
    path = tmp_path / run / "latest.pt"
    path.parent.mkdir()
    torch.save({"model_state": {"weight": torch.tensor([1.0])}}, path)
    monkeypatch.setattr(pull_native_weights, "MIN_WEIGHT_BYTES", 1)
    with patch.object(torch, "load", wraps=torch.load) as read:
        assert pull_native_weights.verify_weights(run, tmp_path) == (True, str(path))
        read.assert_called_once_with(path, map_location="cpu", weights_only=True)


def test_native_metrics_match_operator_run(tmp_path):
    good = {"config": {"run_id": "smoke_s0"}, "step": 6}
    header = "===FILE=== step_000006.json\n"
    pull_native_metrics.save_metrics(io.StringIO(header + json.dumps(good)), "smoke_s0", tmp_path)
    assert json.loads((tmp_path / "smoke_s0" / "step_000006.json").read_text()) == good
    with pytest.raises(ValueError, match="unexpected run"):
        pull_native_metrics.save_metrics(io.StringIO(header + json.dumps(good)), "smoke_s1", tmp_path)
    assert not (tmp_path / "smoke_s1").exists()
    with pytest.raises(ValueError, match="no step metrics"):
        pull_native_metrics.save_metrics(io.StringIO(""), "smoke_s0", tmp_path)
    with pytest.raises(ValueError, match="invalid native run"):
        pull_native_weights.checkpoint_run_dir(tmp_path, "run with spaces")


def test_adapter_archive_installs_verified_directory(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    weights = safetensors.save({"adapter.weight": torch.ones(1)})
    monkeypatch.setattr(pull_qlora_adapter, "MIN_ADAPTER_BYTES", 1)
    archive = tmp_path / "adapter.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for filename, content in [("adapter_config.json", b"{}"), ("adapter_model.safetensors", weights)]:
            member = tarfile.TarInfo("qlora_canary_adapter/" + filename)
            member.size = len(content)
            tar.addfile(member, io.BytesIO(content))
    destination = tmp_path / "student_adapter"
    pull_qlora_adapter.install_adapter(archive, destination)
    assert (destination / "adapter_model.safetensors").read_bytes() == weights
    assert not (tmp_path / "qlora_canary_adapter").exists()
    with tarfile.open(archive, "w:gz") as tar:
        tar.addfile(tarfile.TarInfo("notes.txt"), io.BytesIO())
    with pytest.raises(ValueError, match="unexpected adapter archive path"):
        pull_qlora_adapter.install_adapter(archive, destination)
    assert (destination / "adapter_model.safetensors").read_bytes() == weights
