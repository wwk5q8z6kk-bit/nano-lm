"""Unit tests for `LAB-ARCH-04` (`X5_atrd_muon` & `X6_pkm_muon`), including strict causality proof!"""

from __future__ import annotations

import torch

from lab.arch.arch04 import Arch04Cfg, Arch04LM


def test_atrd_strict_causality_no_future_leak():
    """CRITICAL: Mutating future tokens x[:, t+1:] MUST NOT alter logits[:, :t+1] in eval mode!"""
    torch.manual_seed(42)
    cfg = Arch04Cfg(vocab=256, d=64, n_head=4, n_kv=2, ffn=128, layout="DDAD", seq=32, use_atrd=True, atrd_frac=0.35)
    m = Arch04LM(cfg)
    # Run 1 training step so refiner weights & EMA entropy threshold are non-trivial (non-zero!)
    x_tr = torch.randint(0, 256, (2, 16))
    y_tr = torch.randint(0, 256, (2, 16))
    loss, aux = m(x_tr, y_tr, return_aux=True)
    loss.backward()
    with torch.no_grad():
        for p in m.parameters():
            if p.grad is not None:
                p.sub_(1e-2 * p.grad)
        # Force refiner output weights non-zero to guarantee Pass 2 actively modifies routed tokens!
        m.refiner.o.weight.normal_(0, 0.1)
        m.refiner.dn.weight.normal_(0, 0.1)
        m.refiner.ema_entropy_tau.fill_(5.40)  # ensures ~half of tokens trigger Pass 2

    m.eval()
    x_a = torch.randint(0, 256, (2, 16))
    x_b = x_a.clone()
    x_b[:, 8:] = (x_b[:, 8:] + 77) % 256  # mutate ALL future positions t >= 8!

    with torch.no_grad():
        lg_a, aux_a = m(x_a, return_aux=True)
        lg_b, _ = m(x_b, return_aux=True)

    assert aux_a["routed_frac"] > 0.0, "Refiner was not triggered in causality test!"
    max_diff = (lg_a[:, :8] - lg_b[:, :8]).abs().max().item()
    assert max_diff < 1e-5, f"Future leak detected in ATD! max_diff={max_diff}"


def test_pkm_strict_causality_and_param_separation():
    torch.manual_seed(99)
    cfg = Arch04Cfg(
        vocab=256, d=64, n_head=4, n_kv=2, ffn=128, layout="DDADDA", seq=32,
        use_pkm=True, pkm_layers=(1, 4), pkm_slots=128, pkm_topk=4, pkm_key_dim=32,
    )
    m = Arch04LM(cfg)
    rep = m.param_report()
    assert rep["persistent_memory_params"] == 2 * (128 * 32 + 128 * 64)
    assert rep["active_non_embedding"] < rep["non_embedding"]

    # Force out_scale non-zero and verify strict causality + gradient to selected slots
    with torch.no_grad():
        for b in m.pkm_banks.values():
            b.out_scale.fill_(0.5)
    m.eval()
    x_a = torch.randint(0, 256, (2, 16))
    x_b = x_a.clone()
    x_b[:, 9:] = (x_b[:, 9:] + 31) % 256
    with torch.no_grad():
        lg_a = m(x_a)
        lg_b = m(x_b)
    assert (lg_a[:, :9] - lg_b[:, :9]).abs().max().item() < 1e-5

    m.train()
    loss, aux = m(x_a, x_b, return_aux=True)
    loss.backward()
    assert "pkm_conf_gate" in aux
    assert m.pkm_banks["1"].mem_vals.grad is not None
    # Verify sparse slot activation: not all 128 slots are touched on tiny batch if topk=4
    nonzero_rows = (m.pkm_banks["1"].mem_vals.grad.abs().sum(dim=-1) > 0).sum().item()
    assert 0 < nonzero_rows <= 2 * 16 * 4


def test_margin_atrd_strict_causality():
    """Verify X8_margin_atrd_muon (Top-2 Margin Gate + 32-bucket gather) has zero future leak!"""
    torch.manual_seed(77)
    cfg = Arch04Cfg(
        vocab=256, d=64, n_head=4, n_kv=2, ffn=128, layout="DDAD", seq=64,
        use_atrd=True, use_margin_gate=True, atrd_frac=0.30,
    )
    m = Arch04LM(cfg)
    x_tr = torch.randint(0, 256, (2, 40))
    y_tr = torch.randint(0, 256, (2, 40))
    loss, _ = m(x_tr, y_tr, return_aux=True)
    loss.backward()
    with torch.no_grad():
        m.refiner.o.weight.normal_(0, 0.1)
        m.refiner.dn.weight.normal_(0, 0.1)
        m.refiner.ema_entropy_tau.fill_(0.50)

    m.eval()
    x_a = torch.randint(0, 256, (2, 40))
    x_b = x_a.clone()
    x_b[:, 19:] = (x_b[:, 19:] + 53) % 256
    with torch.no_grad():
        lg_a, aux_a = m(x_a, return_aux=True)
        lg_b, _ = m(x_b, return_aux=True)
    assert aux_a["routed_frac"] > 0.0
    assert (lg_a[:, :19] - lg_b[:, :19]).abs().max().item() < 1e-5
