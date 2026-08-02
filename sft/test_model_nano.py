import pytest
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_nano import seq_logprob, S, V, dev

class MockModel(torch.nn.Module):
    def __init__(self, expected_logits):
        super().__init__()
        self.expected_logits = expected_logits.to(dev)

    def forward(self, x):
        return self.expected_logits.unsqueeze(0)

def test_seq_logprob_basic():
    prompt_ids = [10, 20]
    comp_ids = [30, 40, 50]

    logits = torch.randn(S, V)
    model = MockModel(logits)

    lp = F.log_softmax(logits.float(), -1).cpu()
    expected_tot = (
        lp[1, 30] + lp[2, 40] + lp[3, 50]
    ).item()

    tot = seq_logprob(model, prompt_ids, comp_ids)

    assert isinstance(tot, torch.Tensor)
    assert torch.isclose(tot.cpu(), torch.tensor(expected_tot), atol=1e-5)

def test_seq_logprob_truncation():
    prompt_ids = [10] * (S - 2)
    comp_ids = [20, 30, 40, 50]
    full = (prompt_ids + comp_ids)[:S]

    logits = torch.randn(S, V)
    model = MockModel(logits)

    lp = F.log_softmax(logits.float(), -1).cpu()

    comp_lo = len(prompt_ids)
    comp_hi = len(full)

    expected_tot = 0.0
    for pos in range(comp_lo, comp_hi):
        expected_tot += lp[pos - 1, full[pos]].item()

    tot = seq_logprob(model, prompt_ids, comp_ids)

    assert torch.isclose(tot.cpu(), torch.tensor(expected_tot), atol=1e-5)

def test_seq_logprob_empty_comp():
    prompt_ids = [10, 20]
    comp_ids = []

    logits = torch.randn(S, V)
    model = MockModel(logits)

    tot = seq_logprob(model, prompt_ids, comp_ids)

    assert tot == 0.0 or (isinstance(tot, torch.Tensor) and tot.item() == 0.0)

def test_seq_logprob_padding_logic():
    prompt_ids = [1]
    comp_ids = [2]

    class InspectModel(torch.nn.Module):
        def forward(self, x):
            self.x = x
            return torch.zeros(x.shape[0], S, V, device=dev)

    model = InspectModel()
    seq_logprob(model, prompt_ids, comp_ids)

    assert model.x.shape == (1, S)
    assert model.x[0, 0].item() == 1
    assert model.x[0, 1].item() == 2
    assert torch.all(model.x[0, 2:] == 0)

if __name__ == "__main__":
    pytest.main([__file__])
