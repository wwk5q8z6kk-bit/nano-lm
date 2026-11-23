"""Fast CPU tests: causal alignment, normalization, no-copy fallback, gradient."""
import math

import pytest
import torch
import torch.nn.functional as F

from causal_copy import CausalCopyMixture


def onehot(ids, vocab):
    return F.one_hot(ids, num_classes=vocab).float()


def identity_pointer(dim, bias=8.0, window=8):
    m = CausalCopyMixture(dim, window=window, initial_gate_logit=bias)
    with torch.no_grad():
        m.query.weight.copy_(torch.eye(dim) * 8.0)
        m.key.weight.copy_(torch.eye(dim) * 8.0)
    return m


def test_predict_follower_from_prefix_not_future():
    # Before the last 2, the known sequence contains '2 -> 7'.
    seq = torch.tensor([[2, 7, 4, 2]])
    head = identity_pointer(10)
    logits = torch.zeros(1, 4, 10)
    logprobs = head.full_log_probs(logits, seq, onehot(seq, 10))
    assert logprobs[0, 3, 7].exp() > 0.98
    # Appending arbitrary future tokens must not change the earlier positions.
    longer = torch.tensor([[2, 7, 4, 2, 9, 1, 8]])
    newer = head.full_log_probs(torch.zeros(1, 7, 10), longer, onehot(longer, 10))
    assert torch.allclose(newer[:, :4], logprobs, rtol=1e-6, atol=1e-6)


def test_first_position_has_exact_base_distribution():
    ids = torch.tensor([[1, 2, 3]])
    logits = torch.randn(1, 3, 7)
    head = identity_pointer(7)
    result = head.full_log_probs(logits, ids, onehot(ids, 7))
    assert torch.allclose(result[:, 0], logits.log_softmax(-1)[:, 0], atol=1e-6)


def test_sparse_target_nll_matches_full_distribution():
    torch.manual_seed(17)
    seq = torch.randint(0, 11, (2, 8))
    targets = torch.randint(0, 11, (2, 8))
    hidden = torch.randn(2, 8, 13)
    base = torch.randn(2, 8, 11)
    head = CausalCopyMixture(13, window=4)
    sparse = head.target_nll(base, seq, hidden, targets)
    full = -head.full_log_probs(base, seq, hidden).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(sparse, full, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(sparse).all()


def test_complete_distribution_normalizes_and_has_no_negative_mass():
    torch.manual_seed(7)
    seq = torch.randint(0, 13, (2, 11))
    head = CausalCopyMixture(9, window=5)
    probs = head.full_log_probs(torch.randn(2, 11, 13), seq, torch.randn(2, 11, 9)).exp()
    assert torch.allclose(probs.sum(-1), torch.ones(2, 11), atol=1e-6)
    assert torch.all(probs >= 0)


def test_target_absent_and_gate_near_one_falls_back_to_base_when_no_candidates():
    seq = torch.tensor([[1]])
    head = identity_pointer(8, bias=20)
    logits = torch.randn(1, 1, 8)
    loss = head.target_nll(logits, seq, onehot(seq, 8), torch.tensor([[2]]))
    expect = -logits.log_softmax(-1)[0, 0, 2]
    assert torch.allclose(loss[0, 0], expect, atol=1e-6)


def test_gradients_finite_for_pointer_and_gate():
    torch.manual_seed(5)
    ids = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2]])
    hidden = torch.randn(1, 8, 10, requires_grad=True)
    base = torch.randn(1, 8, 10, requires_grad=True)
    head = CausalCopyMixture(10, window=5, initial_gate_logit=0)
    nll = head.target_nll(base, ids, hidden, torch.tensor([[2, 3, 1, 2, 3, 1, 2, 3]])).mean()
    nll.backward()
    for v in [hidden.grad, base.grad, *[p.grad for p in head.parameters()]]:
        assert v is not None and torch.isfinite(v).all()
    assert head.query.weight.grad.abs().sum() > 0
    assert head.gate.bias.grad.abs().sum() > 0


def test_reject_shape_mismatch_and_invalid_window():
    with pytest.raises(ValueError):
        CausalCopyMixture(16, window=0)
    m = CausalCopyMixture(6)
    with pytest.raises(ValueError):
        m.target_nll(torch.randn(1, 4, 10), torch.ones(1, 3, dtype=torch.long),
                     torch.randn(1, 3, 6), torch.ones(1, 3, dtype=torch.long))


def test_high_gate_with_missing_target_stays_numerically_stable():
    # Sigmoid(25) rounds to 1.0 in FP32, so probability-space mixing
    # would incorrectly zero all non-copyable tokens. Log-space must not.
    seq = torch.tensor([[2, 7, 4, 2]])
    head = identity_pointer(10, bias=25.0)
    logits = torch.zeros(1, 4, 10)
    h = onehot(seq, 10)
    targets = torch.tensor([[1, 1, 1, 9]])
    sparse = head.target_nll(logits, seq, h, targets)
    dense = -head.full_log_probs(logits, seq, h).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert torch.isfinite(sparse).all()
    assert torch.allclose(sparse, dense, atol=1e-5)
    probs = head.full_log_probs(logits, seq, h).exp()
    assert torch.allclose(probs.sum(-1), torch.ones(1, 4), atol=1e-6)
