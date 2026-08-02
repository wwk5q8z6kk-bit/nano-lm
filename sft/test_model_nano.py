import math
import sys
import os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_nano

class MockModel:
    def __init__(self):
        pass

    def __call__(self, x):
        # x shape: (1, S)
        S_dim = x.shape[1]
        V = model_nano.V

        pos_idx = torch.arange(S_dim, device=x.device).unsqueeze(1)
        tok_idx = torch.arange(V, device=x.device).unsqueeze(0)
        logits = (pos_idx + tok_idx).float()
        return (logits,)

def test_seq_logprob_happy_path():
    m = MockModel()
    prompt_ids = [1, 2, 3]
    comp_ids = [4, 5]

    res = model_nano.seq_logprob(m, prompt_ids, comp_ids)

    V = model_nano.V
    tot = 0.0
    for pos in range(3, 5):
        tok = comp_ids[pos - 3]
        pos_logits = torch.arange(V, device=model_nano.dev).float() + (pos - 1)
        lp = F.log_softmax(pos_logits, dim=-1)
        tot += lp[tok].item()

    assert math.isclose(res.item(), tot, rel_tol=1e-5), f"Expected {tot}, got {res.item()}"
    print("test_seq_logprob_happy_path passed")

def test_seq_logprob_empty_comp():
    m = MockModel()
    prompt_ids = [1, 2, 3]
    comp_ids = []

    res = model_nano.seq_logprob(m, prompt_ids, comp_ids)
    assert type(res) == float and res == 0.0, f"Expected 0.0, got {res}"
    print("test_seq_logprob_empty_comp passed")

def test_seq_logprob_truncation():
    m = MockModel()
    S = model_nano.S
    prompt_ids = list(range(S - 2)) # Length S - 2
    comp_ids = [10, 11, 12, 13] # Length 4, total S + 2

    res = model_nano.seq_logprob(m, prompt_ids, comp_ids)

    V = model_nano.V
    tot = 0.0
    for pos in range(S - 2, S):
        tok = comp_ids[pos - (S - 2)]
        pos_logits = torch.arange(V, device=model_nano.dev).float() + (pos - 1)
        lp = F.log_softmax(pos_logits, dim=-1)
        tot += lp[tok].item()

    assert math.isclose(res.item() if isinstance(res, torch.Tensor) else res, tot, rel_tol=1e-5), f"Expected {tot}, got {res}"
    print("test_seq_logprob_truncation passed")

def test_seq_logprob_partial_truncation():
    m = MockModel()
    S = model_nano.S
    prompt_ids = list(range(S + 10))
    comp_ids = [1, 2, 3]

    res = model_nano.seq_logprob(m, prompt_ids, comp_ids)
    assert type(res) == float and res == 0.0, f"Expected 0.0, got {res}"
    print("test_seq_logprob_partial_truncation passed")

if __name__ == "__main__":
    test_seq_logprob_happy_path()
    test_seq_logprob_empty_comp()
    test_seq_logprob_truncation()
    test_seq_logprob_partial_truncation()
    print("All tests passed!")
