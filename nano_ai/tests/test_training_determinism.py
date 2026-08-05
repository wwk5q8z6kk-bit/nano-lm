"""B5: prove bit-identity of a real training step on CPU.

Artifacts record `"deterministic_algorithms": True` in their runtime block, but
verified 2026-08-05:

  * `train_evidence_query_h6.py` and `train_evidence_query.py` never call
    `torch.use_deterministic_algorithms` at all (only the H1-era
    `train_state_span.py` does); the flag in the artifact is written by the
    *evaluation* path.
  * Even where the flag is set, it does not cover
    `scaled_dot_product_attention` -- SDPA runs without raising under
    `use_deterministic_algorithms(True)`, and it dominates the backward pass.

And every trainer call site in `nano_ai/tests/` sits inside `pytest.raises`, so
no test had ever completed a training step. These tests do.
"""

from __future__ import annotations

import hashlib

import torch

from nano_ai.training.model import NanoGPT, initialize_weights


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _train_two_steps(seed: int) -> str:
    """One real forward/backward/optimizer cycle, twice, on CPU."""
    torch.manual_seed(seed)
    model = NanoGPT()
    initialize_weights(model, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(
        0, model.config.vocabulary_size, (2, 16),
        generator=torch.Generator().manual_seed(seed + 1),
    )
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        hidden = model(tokens)
        loss = hidden.float().pow(2).mean()
        loss.backward()
        optimizer.step()
    return _state_digest(model)


def test_identical_seeds_produce_byte_identical_weights_on_cpu():
    """The claim the artifacts assert, actually tested."""
    assert _train_two_steps(1234) == _train_two_steps(1234)


def test_different_seeds_produce_different_weights():
    """Guard against a digest that is constant for trivial reasons."""
    assert _train_two_steps(1234) != _train_two_steps(4321)


def test_deterministic_flag_does_not_cover_scaled_dot_product_attention():
    """Pin the gap: the flag is not a guarantee for the op that dominates training."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        query = torch.randn(1, 2, 8, 16)
        torch.nn.functional.scaled_dot_product_attention(query, query, query)
    finally:
        torch.use_deterministic_algorithms(previous)
