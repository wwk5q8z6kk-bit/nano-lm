import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def model(monkeypatch):
    path = Path(__file__).with_name("model_nano.py")
    spec = importlib.util.spec_from_file_location("sft_model_under_test", path)
    module = importlib.util.module_from_spec(spec)
    # This historical model opens its checked-in tokenizer relative to the cwd.
    with monkeypatch.context() as local:
        local.chdir(path.parent)
        spec.loader.exec_module(module)
    return module


def test_load_checkpoint(tmp_path, model, monkeypatch):
    monkeypatch.setattr(model, "dev", "cpu")
    source = model.GPT()
    with torch.no_grad():
        source.emb.weight.fill_(0.5)
    checkpoint = tmp_path / "model.pt"
    torch.save(source.state_dict(), checkpoint)

    restored = model.load(checkpoint)
    assert isinstance(restored, model.GPT)
    assert not restored.training
    assert next(restored.parameters()).device.type == "cpu"
    for name, tensor in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], tensor), name


def test_missing_checkpoint(tmp_path, model, monkeypatch):
    monkeypatch.setattr(model, "dev", "cpu")
    with pytest.raises(FileNotFoundError):
        model.load(tmp_path / "missing.pt")


def test_incompatible_state_dict(tmp_path, model, monkeypatch):
    monkeypatch.setattr(model, "dev", "cpu")
    checkpoint = tmp_path / "incompatible.pt"
    torch.save({"unrelated": torch.zeros(1)}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load(checkpoint)
