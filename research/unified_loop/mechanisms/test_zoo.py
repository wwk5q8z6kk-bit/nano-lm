"""Verification & speed sanity tests for EXP-U02 cross-paradigm mechanisms."""
from __future__ import annotations

import torch

from research.unified_loop.mechanisms.zoo import (
    GDN1Mixer,
    MECHANISM_SPECS,
    SkewCayleyFlowMixer,
    UnifiedBlockLM,
    _center_l2,
    verify_strict_causality,
)


def test_gdn1_chunked_wy_matches_sequential() -> None:
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        for full_sphere in (False, True):
            for T in (48, 128):
                torch.manual_seed(17)
                B, D, H = 4, 32, 4
                mixer = GDN1Mixer(D, H, full_sphere=full_sphere).to(dev)
                x0 = torch.randn(B, T, D, device=dev)

                def run_mode(chunked: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
                    x = x0.clone().detach().requires_grad_(True)
                    mixer.zero_grad(set_to_none=True)
                    q, k, v = mixer.conv(mixer.qkv(x)).split(D, dim=-1)
                    q = _center_l2(q.view(B, T, H, D // H), center=full_sphere)
                    k = _center_l2(k.view(B, T, H, D // H), center=full_sphere)
                    v = v.view(B, T, H, D // H).float()
                    log_d = -torch.nn.functional.softplus(mixer.a(x.float())) * mixer.A_log.exp()
                    beta = (2.0 * torch.sigmoid(mixer.b(x.float()))) if full_sphere else torch.sigmoid(mixer.b(x.float()))
                    yh = (
                        mixer._forward_chunked_wy(q, k, v, log_d, beta, chunk_size=32)
                        if chunked
                        else mixer._forward_sequential(q, k, v, log_d.exp(), beta)
                    )
                    loss = (yh ** 2).sum()
                    loss.backward()
                    pgrads = {n: p.grad.detach().clone() for n, p in mixer.named_parameters() if p.grad is not None}
                    return yh.detach(), x.grad.detach().clone(), pgrads

                y_seq, xg_seq, pg_seq = run_mode(False)
                y_chk, xg_chk, pg_chk = run_mode(True)

                assert float((y_seq - y_chk).abs().max().item()) < 1e-4, f"{dev} T={T} fs={full_sphere} fwd diff"
                assert float((xg_seq - xg_chk).abs().max().item()) < 1e-4, f"{dev} T={T} fs={full_sphere} xgrad diff"
                for kname in pg_seq:
                    diff = float((pg_seq[kname] - pg_chk[kname]).abs().max().item())
                    assert diff < 1e-4, f"{dev} T={T} fs={full_sphere} param {kname} grad diff={diff}"


def test_woodbury_cayley_matches_linalg_solve() -> None:
    torch.manual_seed(42)
    BH, hd = 12, 8
    S = torch.randn(BH, hd, hd)
    u = torch.randn(BH, hd) / (hd ** 0.5)
    w = torch.randn(BH, hd) / (hd ** 0.5)
    gamma = torch.nn.functional.softplus(torch.randn(BH, hd))
    h_step = 0.5 * torch.sigmoid(torch.randn(BH, 1))

    out_wb = SkewCayleyFlowMixer._cayley_step_woodbury(S, u, w, gamma, h_step)

    # Exact reference via dense torch.linalg.solve:
    # A = u w^T - w u^T - diag(gamma), M = (I - h A)^-1 (I + h A), S_rot = S @ M^T
    A = torch.bmm(u.unsqueeze(-1), w.unsqueeze(-2)) - torch.bmm(w.unsqueeze(-1), u.unsqueeze(-2)) - torch.diag_embed(gamma)
    eye = torch.eye(hd).unsqueeze(0).expand(BH, hd, hd)
    hA = h_step.unsqueeze(-1) * A
    M_ref = torch.linalg.solve(eye - hA, eye + hA)
    out_ref = torch.bmm(S, M_ref.transpose(-1, -2))

    max_diff = float((out_wb - out_ref).abs().max().item())
    assert max_diff < 1e-5, f"Woodbury Cayley mismatch: {max_diff}"


def test_all_mechanisms_strictly_causal_param_parity_and_finite_grads() -> None:
    report = verify_strict_causality(vocab_size=72, d_model=32)
    for name, info in report.items():
        assert info["causal_leak_max"] == 0.0, f"{name} leaked future tokens: {info}"
        assert info["finite"], f"{name} produced non-finite logits"
        assert 54_500 <= info["params"] <= 57_500, f"{name} param parity out of range: {info['params']}"

    torch.manual_seed(0)
    x = torch.randint(0, 72, (4, 16))
    y = torch.randint(0, 72, (4, 16))
    for name, spec in MECHANISM_SPECS.items():
        m = UnifiedBlockLM(72, 32, name)
        if spec.loops > 1:
            _, loop_lgs = m(x, return_loop_logits=True)
            loss = sum(torch.nn.functional.cross_entropy(lg.reshape(-1, 72), y.reshape(-1)) for lg in loop_lgs) / len(loop_lgs)
        else:
            out = m(x)
            if spec.use_pointer:
                loss = torch.nn.functional.nll_loss(out.reshape(-1, 72), y.reshape(-1))
            else:
                loss = torch.nn.functional.cross_entropy(out.reshape(-1, 72), y.reshape(-1))
        m.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss), f"{name} non-finite loss"
        for pname, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name}.{pname} bad grad"
