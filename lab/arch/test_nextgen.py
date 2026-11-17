"""Unit tests for Next-Gen Track A mechanisms (Gated-VRes, Sharp-Delta, SGRD)."""

import torch

from lab.arch.nextgen import NextGenCfg, NextGenLM


def test_nextgen_causality_all_mechanisms():
    torch.manual_seed(42)
    cfg = NextGenCfg(
        vocab=64,
        d=64,
        n_head=4,
        n_kv=2,
        ffn=128,
        layout="DDADDA",
        chunk=8,
        gated_vres=True,
        sharp_v=True,
        qk_canon_attn=True,
        loop=(1, 4, 1, 3),
        sgrd_prior_p=0.45,
    )
    m = NextGenLM(cfg).eval()
    x1 = torch.randint(0, 64, (2, 21))
    x2 = x1.clone()
    x2[:, 13:] = (x2[:, 13:] + 7) % 64
    y1, aux1 = m(x1, return_aux=True)
    y2, _ = m(x2, return_aux=True)
    assert torch.allclose(y1[:, :13], y2[:, :13], atol=2e-4, rtol=1e-3), (y1[:, :13] - y2[:, :13]).abs().max()
    assert 1.0 <= float(aux1["expected_loops"]) <= 3.0


def test_nextgen_backward_and_sgrd_gradients():
    torch.manual_seed(7)
    cfg = NextGenCfg(
        vocab=64,
        d=64,
        n_head=4,
        n_kv=2,
        ffn=128,
        layout="DDADDA",
        chunk=8,
        gated_vres=True,
        sharp_v=True,
        loop=(1, 4, 1, 3),
        sgrd_prior_p=0.5,
    )
    m = NextGenLM(cfg)
    x = torch.randint(0, 64, (2, 18))
    loss, aux = m(x[:, :-1], x[:, 1:], return_aux=True)
    loss.backward()
    assert m.halt_gamma.grad is not None and torch.isfinite(m.halt_gamma.grad)
    assert m.blocks[1].mix.v_gate.weight.grad is not None
    assert m.blocks[2].mix.v_gate.weight.grad is not None
    assert float(aux["sgrd_kl"].detach()) >= -1e-5
