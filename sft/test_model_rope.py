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


def reference_rope(x):
    # A complex phase rotates each adjacent pair in the real-valued embedding.
    cpu = x.cpu().double()
    positions = torch.arange(cpu.shape[-2], dtype=torch.float64)
    columns = torch.arange(0, cpu.shape[-1], 2, dtype=torch.float64)
    angles = positions[:, None] * 10000 ** (-columns / cpu.shape[-1])
    phase = torch.polar(torch.ones_like(angles), angles)
    pairs = torch.view_as_complex(cpu.unflatten(-1, (-1, 2)))
    return torch.view_as_real(pairs * phase).flatten(-2).to(x)


def test_rope_standard(model):
    generator = torch.Generator().manual_seed(42)
    q = torch.randn(2, 4, model.S, model.hd, generator=generator).to(model.dev)
    k = torch.randn(2, 2, model.S, model.hd, generator=generator).to(model.dev)

    actual_q, actual_k = model.rope(q, k)
    torch.testing.assert_close(actual_q, reference_rope(q), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_k, reference_rope(k), rtol=1e-4, atol=1e-4)
