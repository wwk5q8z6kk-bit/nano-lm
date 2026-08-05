"""B4: explicit from-scratch weight initialization.

`NanoGPT.__init__` performs no initialization. Every current path warm-starts
from an anchor checkpoint, so `load_state_dict` overwrites PyTorch's defaults
and the omission is invisible -- but a from-scratch rung-1 pretrain would
silently use those defaults instead of the scheme `pretrain/AUDIT.md` records
("init 0.02 depth-scaled", step-1 loss 8.35 ~= ln(4096)).
"""

from __future__ import annotations

import pytest
import torch

from nano_ai.training.init_weights import initialize_weights
from nano_ai.training.model import NANO_MODEL_CONFIG, NanoGPT


def _init(seed: int) -> NanoGPT:
    model = NanoGPT()
    return initialize_weights(model, generator=torch.Generator().manual_seed(seed))


def test_initialization_changes_weights_and_matches_recorded_scheme():
    model = NanoGPT()
    baseline = model.emb.weight.detach().clone()
    initialize_weights(model, generator=torch.Generator().manual_seed(11))
    assert not torch.equal(baseline, model.emb.weight.detach())
    assert model.emb.weight.detach().std().item() == pytest.approx(0.02, abs=0.003)
    assert model.blocks[0].q.weight.detach().std().item() == pytest.approx(0.02, abs=0.003)


def test_residual_projections_are_depth_scaled():
    """o and dn carry 0.02 / sqrt(2 * layer_count) so residual variance is depth-stable."""
    model = _init(3)
    expected = 0.02 * (2 * NANO_MODEL_CONFIG.layer_count) ** -0.5
    for name in ("o", "dn"):
        std = getattr(model.blocks[0], name).weight.detach().std().item()
        assert std == pytest.approx(expected, rel=0.15), f"{name} not depth-scaled"


def test_initialization_is_deterministic_under_a_seeded_generator():
    a, b = _init(29), _init(29)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert torch.equal(pa.detach(), pb.detach()), f"{na} differs across identical seeds"


def test_initialization_is_opt_in_and_not_called_by_the_constructor():
    """No existing warm-start path or frozen recipe may change behaviour."""
    a, b = NanoGPT(), NanoGPT()
    # Untouched construction still uses PyTorch defaults; if a constructor-level
    # init were added, these would coincide only by luck.
    assert not torch.equal(a.emb.weight.detach(), b.emb.weight.detach())


def test_h6_zero_initialized_offsets_survive_initialization():
    """H6's state_boundary_query_offsets are deliberately zero; init must not disturb them.

    This is the safety-critical case: H6's whole identity argument is that at
    step 0 the model is functionally identical to a freshly initialized H5. A
    blanket initializer that touched raw nn.Parameter tensors would destroy
    that silently. `initialize_weights` only visits nn.Linear and nn.Embedding,
    so the offsets are preserved regardless of constructor ordering.
    """
    from nano_ai.training.state_conditioned_evidence_query_model import (
        NanoStateConditionedEvidenceQueryPointerModel,
    )

    model = NanoStateConditionedEvidenceQueryPointerModel()
    assert torch.count_nonzero(model.state_boundary_query_offsets).item() == 0
    initialize_weights(model, generator=torch.Generator().manual_seed(5))
    assert torch.count_nonzero(model.state_boundary_query_offsets).item() == 0, (
        "initialization destroyed H6's zero-initialized residual"
    )
