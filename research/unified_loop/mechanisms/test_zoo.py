"""Verification & speed sanity tests for EXP-U02 cross-paradigm mechanisms."""
from __future__ import annotations

import math

import torch

from research.unified_loop.mechanisms.zoo import (
    GDN1Mixer,
    GaugeSU2HolonomyMixer,
    HSPALieSO3Mixer,
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


def test_hspa_parallel_scan_matches_sequential() -> None:
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        for T in (48, 128):
            torch.manual_seed(29)
            B, D, H = 4, 32, 4
            mixer = HSPALieSO3Mixer(D, H).to(dev)
            x0 = torch.randn(B, T, D, device=dev)

            def run_mode(parallel: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
                x = x0.clone().detach().requires_grad_(True)
                mixer.zero_grad(set_to_none=True)
                omega = mixer.w_omega(x).view(B, T, H, mixer.r, 3).float()
                u_in = mixer.w_in(x).view(B, T, H, mixer.r, 3).float()
                a_raw, eta_raw = mixer.w_gate(x).view(B, T, 2, H, mixer.r).unbind(2)
                decay = torch.sigmoid(a_raw.float()).unsqueeze(-1)
                eta = torch.sigmoid(eta_raw.float()).unsqueeze(-1)
                yh = (
                    mixer._forward_parallel_scan(omega, u_in, decay, eta)
                    if parallel
                    else mixer._forward_sequential(omega, u_in, decay, eta)
                )
                loss = (yh ** 2).mean()
                loss.backward()
                pgrads = {n: p.grad.detach().clone() for n, p in mixer.named_parameters() if p.grad is not None}
                return yh.detach(), x.grad.detach().clone(), pgrads

            y_seq, xg_seq, pg_seq = run_mode(False)
            y_par, xg_par, pg_par = run_mode(True)
            assert torch.allclose(y_seq, y_par, atol=1e-4, rtol=1e-4), (
                f"{dev} T={T} hspa fwd diff={float((y_seq - y_par).abs().max().item())}"
            )
            assert torch.allclose(xg_seq, xg_par, atol=1e-4, rtol=1e-4), (
                f"{dev} T={T} hspa xgrad diff={float((xg_seq - xg_par).abs().max().item())}"
            )
            for kname in pg_seq:
                assert torch.allclose(pg_seq[kname], pg_par[kname], atol=1e-4, rtol=1e-4), (
                    f"{dev} T={T} hspa {kname} grad diff={float((pg_seq[kname] - pg_par[kname]).abs().max().item())}"
                )


def test_sparse_ow_pointer_causality_and_overwrite_index() -> None:
    # Synthetic overwrite stream: t=0:K(5), t=1:V1(11), t=2..3:bg, t=4:K(5), t=5:V2(22), t=6..9:bg, t=10:K(5)
    seq = torch.tensor([[5, 11, 30, 31, 5, 22, 32, 33, 34, 35, 5, 0]], dtype=torch.long)
    cand_idx, cand_valid = UnifiedBlockLM._build_sparse_pointer_candidates(seq, n_exact=8, n_local=24)
    B, T, K = cand_idx.shape
    assert K == 32
    # Strict causality: every valid index j < t
    for t in range(T):
        val = cand_valid[0, t]
        if bool(val.any()):
            assert bool((cand_idx[0, t][val] < t).all())
            # Zero duplicate indices inside [t]'s candidate slots
            valid_indices = cand_idx[0, t][val].tolist()
            assert len(valid_indices) == len(set(valid_indices)), f"duplicate idx at t={t}: {valid_indices}"
    # t=10 queries K(5): rank 0 exact match MUST point to latest K at t=4 (next=V2), rank 1 -> older K at t=0 (next=V1)
    assert bool(cand_valid[0, 10, 0]) and int(cand_idx[0, 10, 0]) == 4
    assert bool(cand_valid[0, 10, 1]) and int(cand_idx[0, 10, 1]) == 0
    assert not bool(cand_valid[0, 10, 2])
    # t=0 has no prior context -> zero valid candidates
    assert not bool(cand_valid[0, 0].any())

    if torch.backends.mps.is_available():
        for mname in ("M10_HSPA_DensePointer", "M11_FullSphere_SparseOWPointer", "M12_HSPA_SparseOWPointer"):
            xm = torch.randint(0, 72, (2, 24), device="mps")
            ym = torch.randint(0, 72, (2, 24), device="mps")
            mm = UnifiedBlockLM(72, 32, mname).to("mps")
            loss_m = torch.nn.functional.nll_loss(mm(xm).reshape(-1, 72), ym.reshape(-1))
            mm.zero_grad(set_to_none=True)
            loss_m.backward()
            assert torch.isfinite(loss_m), f"{mname} MPS non-finite loss"
            for pn, p in mm.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), f"{mname} MPS bad grad {pn}"


def test_gdn1_aligned_keys_no_cancellation() -> None:
    """Invention A: aligned unit keys + beta=1.9 + decay=0.999 must not cancel like Neumann doubling."""
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        torch.manual_seed(7)
        B, T, H, hd = 2, 128, 4, 8
        mixer = GDN1Mixer(H * hd, H, full_sphere=True).to(dev)
        k_base = torch.nn.functional.normalize(torch.randn(B, 1, H, hd, device=dev), dim=-1)
        k = k_base.expand(B, T, H, hd).contiguous()
        q = torch.nn.functional.normalize(torch.randn(B, T, H, hd, device=dev), dim=-1)
        v0 = torch.randn(B, T, H, hd, device=dev)
        log_d0 = torch.full((B, T, H), math.log(0.999), device=dev)
        beta0 = torch.full((B, T, H), 1.9, device=dev)

        def run_mode(chunked: bool) -> tuple[torch.Tensor, torch.Tensor]:
            v = v0.clone().detach().requires_grad_(True)
            yh = (
                mixer._forward_chunked_wy(q, k, v, log_d0, beta0, chunk_size=32)
                if chunked
                else mixer._forward_sequential(q, k, v, log_d0.exp(), beta0)
            )
            (yh ** 2).mean().backward()
            return yh.detach(), v.grad.detach().clone()

        y_seq, vg_seq = run_mode(False)
        y_chk, vg_chk = run_mode(True)
        assert torch.isfinite(y_chk).all() and torch.isfinite(vg_chk).all(), f"{dev} non-finite aligned WY"
        y_rel = float((y_seq - y_chk).abs().max().item()) / max(1.0, float(y_seq.abs().max().item()))
        v_rel = float((vg_seq - vg_chk).abs().max().item()) / max(1e-6, float(vg_seq.abs().max().item()))
        assert y_rel < 1e-4 and float((y_seq - y_chk).abs().max().item()) < 5e-4, f"{dev} aligned WY fwd rel={y_rel}"
        assert v_rel < 1e-4 and float((vg_seq - vg_chk).abs().max().item()) < 5e-4, f"{dev} aligned WY grad rel={v_rel}"


def test_gauge_su2_parallel_scan_matches_sequential() -> None:
    """Invention C: Gauge-decoupled SU(2) quaternion scan == sequential recurrence, ||q||==1, Phi(-q)==Phi(q)."""
    q_rand = torch.nn.functional.normalize(torch.randn(8, 4), dim=-1)
    assert float((GaugeSU2HolonomyMixer._quat_to_so3_6d(q_rand) - GaugeSU2HolonomyMixer._quat_to_so3_6d(-q_rand)).abs().max().item()) < 1e-6

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        for T in (16, 128):
            torch.manual_seed(31)
            B, D, H = 3, 32, 4
            mixer = GaugeSU2HolonomyMixer(D, H).to(dev)
            x0 = torch.randn(B, T, D, device=dev)

            def run_mode(parallel: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
                x = x0.clone().detach().requires_grad_(True)
                mixer.zero_grad(set_to_none=True)
                xc = x + mixer.conv(x)
                omega = mixer.w_omega(xc).view(B, T, H, mixer.r, 3).float()
                u_in = mixer.w_in(xc).view(B, T, H, mixer.r, 3).float()
                a_raw, eta_raw = mixer.w_gate(xc.float()).view(B, T, 2, H, mixer.r).unbind(2)
                decay = torch.sigmoid(a_raw.float()).unsqueeze(-1)
                eta = torch.sigmoid(eta_raw.float()).unsqueeze(-1)
                dq = mixer._axis_angle_to_quat(omega)
                yh = (
                    mixer._forward_parallel_scan(dq, u_in, decay, eta)
                    if parallel
                    else mixer._forward_sequential(dq, u_in, decay, eta)
                )
                (yh ** 2).mean().backward()
                pgrads = {n: p.grad.detach().clone() for n, p in mixer.named_parameters() if p.grad is not None}
                return yh.detach(), x.grad.detach().clone(), pgrads

            y_seq, xg_seq, pg_seq = run_mode(False)
            y_par, xg_par, pg_par = run_mode(True)
            assert torch.allclose(y_seq, y_par, atol=1e-4, rtol=1e-4), (
                f"{dev} T={T} gauge_su2 fwd max={float((y_seq - y_par).abs().max().item())}"
            )
            assert torch.allclose(xg_seq, xg_par, atol=1e-4, rtol=1e-4), (
                f"{dev} T={T} gauge_su2 xgrad max={float((xg_seq - xg_par).abs().max().item())}"
            )
            for kname in pg_seq:
                assert torch.allclose(pg_seq[kname], pg_par[kname], atol=1e-4, rtol=1e-4), (
                    f"{dev} T={T} gauge_su2 {kname} grad max={float((pg_seq[kname] - pg_par[kname]).abs().max().item())}"
                )


def test_evid_cos_sparse_pointer_causality_and_unseen_suffix() -> None:
    """Invention B: unseen 4-gram suffix at distance 64 hits exact cos_lex == 1.0 at slot 0 + finite MPS grads."""
    m = UnifiedBlockLM(72, 32, "M13_HSPA_EvidCosSparsePtr").eval()
    torch.manual_seed(11)
    seq = torch.randint(4, 44, (1, 80), dtype=torch.long)
    suffix = torch.tensor([51, 52, 53, 54], dtype=torch.long)
    seq[0, 4:8] = suffix
    seq[0, 8] = 55
    seq[0, 68:72] = suffix  # distance = 71 - 7 = 64

    with torch.no_grad():
        emb_in = torch.nn.functional.normalize(m.emb(seq).float(), dim=-1)
        h_rand = torch.randn(1, 80, 32)
        probs_base = torch.softmax(m._readout(h_rand), dim=-1)
        _, cand_idx, cand_valid, attn_w, g, cos_lex = m._compute_sparse_evid_components(h_rand, seq, probs_base, emb_in)
    assert bool(cand_valid[0, 71, 0]) and int(cand_idx[0, 71, 0]) == 7
    assert abs(float(cos_lex[0, 71, 0].item()) - 1.0) < 1e-5, f"expected cos_lex=1.0, got {cos_lex[0, 71, 0].item()}"
    assert bool((cand_idx[0, 71][cand_valid[0, 71]] < 71).all())
    assert torch.isfinite(attn_w).all() and torch.isfinite(g).all()

    if torch.backends.mps.is_available():
        for mname in (
            "M13_HSPA_EvidCosSparsePtr",
            "M14_GaugeSU2_Pure",
            "M15_GaugeSU2_EvidCosSparsePtr",
            "M16_FullSphereTri_EvidCosSparsePtr",
        ):
            spec = MECHANISM_SPECS[mname]
            xm = torch.randint(0, 72, (2, 24), device="mps")
            ym = torch.randint(0, 72, (2, 24), device="mps")
            mm = UnifiedBlockLM(72, 32, mname).to("mps")
            out_m = mm(xm)
            loss_m = (
                torch.nn.functional.nll_loss(out_m.reshape(-1, 72), ym.reshape(-1))
                if spec.has_pointer
                else torch.nn.functional.cross_entropy(out_m.reshape(-1, 72), ym.reshape(-1))
            )
            mm.zero_grad(set_to_none=True)
            loss_m.backward()
            assert torch.isfinite(loss_m), f"{mname} MPS non-finite loss"
            for pn, p in mm.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), f"{mname} MPS bad grad {pn}"


def test_sparse_candidates_on_device_matches_hash() -> None:
    """Upgrade 1b: vectorized on-device candidate builder == CPU hash-list builder bit-for-bit."""
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        for B, T in ((4, 48), (2, 192)):
            torch.manual_seed(13 + T)
            x = torch.randint(0, 20, (B, T), device=dev)  # small vocab forces heavy matches & dedup
            idx_vec, val_vec = UnifiedBlockLM._build_sparse_pointer_candidates(x, n_exact=8, n_local=24)
            idx_h, val_h = UnifiedBlockLM._build_sparse_pointer_candidates_hash(x, n_exact=8, n_local=24)
            assert torch.equal(val_vec, val_h), f"{dev} T={T} valid mismatch"
            assert torch.equal(torch.where(val_vec, idx_vec, 0), torch.where(val_h, idx_h, 0)), f"{dev} T={T} idx mismatch"


def test_sparse_hybrid_causality_dedup_and_mps_grads() -> None:
    """Upgrades 2 & 3: M17/M18 strict causality, zero duplicate slots, streaming heap exactness, AuxBase & MPS grads."""
    torch.manual_seed(41)
    # 1. Verify M18 32-slot dedup & strict < t causality
    m18 = UnifiedBlockLM(72, 32, "M18_HSPA_HybridSemLexPtr").eval()
    x = torch.randint(0, 24, (2, 64))
    with torch.no_grad():
        emb_in = torch.nn.functional.normalize(m18.emb(x).float(), dim=-1)
        h = emb_in.clone()
        for blk in m18.blocks:
            h = blk(h)
        probs_base = torch.softmax(m18._readout(h), dim=-1)
        _, cand_idx, cand_valid, attn_w, g, _ = m18._compute_sparse_evid_components(h, x, probs_base, emb_in)
    assert cand_idx.shape == (2, 64, 32)
    for b in range(2):
        for t in range(64):
            val = cand_valid[b, t]
            if bool(val.any()):
                slots = cand_idx[b, t][val].tolist()
                assert all(j < t for j in slots), f"future slot leak at t={t}: {slots}"
                assert len(slots) == len(set(slots)), f"duplicate slots across exact+sem+local at t={t}: {slots}"
    assert torch.isfinite(attn_w).all() and torch.isfinite(g).all()

    # 2. Verify streaming min-heap T > 1024 path against brute-force dense top-k on T=1030
    torch.manual_seed(99)
    T_long = 1030
    x_long = torch.randint(0, 50, (1, T_long))
    base_i_long, base_v_long = UnifiedBlockLM._build_sparse_pointer_candidates(x_long, n_exact=8, n_local=16)
    sal_long = torch.randn(1, T_long)
    sem_i_stream, sem_v_stream = UnifiedBlockLM._select_causal_salience_slots(sal_long, base_i_long, base_v_long, n_sem=8)
    for t_chk in (32, 200, 1029):
        excl = {int(idx) for idx, ok in zip(base_i_long[0, t_chk].tolist(), base_v_long[0, t_chk].tolist()) if ok}
        cands = sorted([j for j in range(t_chk) if j not in excl], key=lambda j: (-float(sal_long[0, j]), -j))[:8]
        got = [int(idx) for idx, ok in zip(sem_i_stream[0, t_chk].tolist(), sem_v_stream[0, t_chk].tolist()) if ok]
        assert got == cands, f"heap mismatch at t={t_chk}: got={got} ref={cands}"

    # 3. Verify return_base_logits + finite MPS grads (including ptr_salience.grad != 0) on M17 & M18
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        for mname in ("M17_HSPA_MultiScale_EvidCosSparsePtr", "M18_HSPA_HybridSemLexPtr"):
            mm = UnifiedBlockLM(72, 32, mname).to(dev)
            xm = torch.randint(0, 72, (2, 32), device=dev)
            ym = torch.randint(0, 72, (2, 32), device=dev)
            mix_lp, base_lg = mm(xm, return_base_logits=True)
            assert mix_lp.shape == base_lg.shape == (2, 32, 72)
            assert torch.isfinite(mix_lp).all() and torch.isfinite(base_lg).all()
            loss = (
                torch.nn.functional.nll_loss(mix_lp.reshape(-1, 72), ym.reshape(-1))
                + 0.25 * torch.nn.functional.cross_entropy(base_lg.reshape(-1, 72), ym.reshape(-1))
            )
            mm.zero_grad(set_to_none=True)
            loss.backward()
            for pn, p in mm.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), f"{dev} {mname} bad grad {pn}"
            if mname == "M18_HSPA_HybridSemLexPtr":
                assert float(mm.ptr_salience.weight.grad.abs().max().item()) > 0.0, f"{dev} ptr_salience got zero grad"


def test_all_mechanisms_strictly_causal_param_parity_and_finite_grads() -> None:
    report = verify_strict_causality(vocab_size=72, d_model=32)
    for name, info in report.items():
        assert info["causal_leak_max"] == 0.0, f"{name} leaked future tokens: {info}"
        assert info["finite"], f"{name} produced non-finite logits"
        assert 54_500 <= info["params"] <= 56_500, f"{name} param parity out of range: {info['params']}"

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
            if spec.has_pointer:
                loss = torch.nn.functional.nll_loss(out.reshape(-1, 72), y.reshape(-1))
            else:
                loss = torch.nn.functional.cross_entropy(out.reshape(-1, 72), y.reshape(-1))
        m.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss), f"{name} non-finite loss"
        for pname, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"{name}.{pname} bad grad"
