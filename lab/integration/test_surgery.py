"""Unit tests for Track A -> Track B progressive architecture surgery (`lab/integration/surgery.py`)."""

from __future__ import annotations

import copy
import torch

from lab.arch.nextgen import NextGenCfg, NextGenLM
from lab.integration.surgery import (
    build_surgical_hybrid,
    graft_zero_init_gated_vres,
    stage1_align_mixers,
    stage2_distill_end_to_end,
)


def _tiny_dense() -> NextGenLM:
    torch.manual_seed(7)
    cfg = NextGenCfg(vocab=256, d=64, n_head=4, n_kv=2, ffn=128, layout="AAAA", seq=64, gated_vres=False)
    return NextGenLM(cfg)


def test_zero_init_gated_vres_exact_step0_identity_and_grad():
    base = _tiny_dense()
    snap = copy.deepcopy(base.state_dict())
    grafted = graft_zero_init_gated_vres(base)

    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    with torch.no_grad():
        lg_base = base(x)
        lg_graft = grafted(x)
    assert torch.equal(lg_base, lg_graft), f"Expected exact step-0 identity, got max diff {(lg_base - lg_graft).abs().max()}"

    # Step-1 gradient flows into highway gates (layers 1..3 where v0 is already set by layer 0)
    loss = grafted(x, y)
    loss.backward()
    assert grafted.blocks[1].mix.highway_gate.weight.grad is not None
    assert grafted.blocks[1].mix.highway_gate.weight.grad.abs().sum() > 0
    # Original base weights completely untouched
    for k, v in base.state_dict().items():
        assert torch.equal(v, snap[k])


def test_progressive_hybrid_surgery_reduces_mse_and_kl():
    base = _tiny_dense()
    with torch.no_grad():
        base.emb.weight.mul_(6.0)  # non-uniform sharp teacher distribution
    snap = copy.deepcopy(base.state_dict())
    hybrid = build_surgical_hybrid(base, target_layout="DADA", warmstart_qkv=True)
    assert hybrid.c.layout == "DADA"

    torch.manual_seed(11)
    fixed_batch = torch.randint(0, 256, (8, 32))
    batches = [fixed_batch for _ in range(12)]
    mses = stage1_align_mixers(base, hybrid, batches, lr=5e-3)
    assert mses[-1] < mses[0], f"Stage-1 mixer MSE did not decrease: {mses[0]} -> {mses[-1]}"

    batches_xy = [(fixed_batch[:, :-1], fixed_batch[:, 1:]) for _ in range(12)]
    kls = stage2_distill_end_to_end(base, hybrid, batches_xy, lr=1e-3, alpha_kd=1.0, topk=32)
    assert kls[-1] < kls[0], f"Stage-2 Top-K+Tail KL did not decrease: {kls[0]} -> {kls[-1]}"

    for k, v in base.state_dict().items():
        assert torch.equal(v, snap[k])
