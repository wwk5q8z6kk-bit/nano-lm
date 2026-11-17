"""PyTorch unit, parity, and spectral/Jacobian tests for 2026 mechanism lab (`X03`–`X05`)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from nano.mechanisms.recurrent_cells import (  # noqa: E402
    ComplexRotoSSMCell,
    ContractionDeltaCell,
    DenseGQACell,
    GDN2IndependentGateCell,
    GeoEraseWriteCell,
    HouseholderDeltaCell,
    ToySequenceModel,
    build_mixer_cell,
    compute_transition_and_jacobian_spectrum,
)


@pytest.mark.parametrize(
    "cell_name",
    [
        "dense_gqa",
        "diff_gqa",
        "contraction_delta",
        "householder_delta",
        "householder_rank2",
        "gdn2",
        "complex_roto_ssm",
        "geo_erase_write",
    ],
)
def test_sequence_vs_step_by_step_parity(cell_name: str):
    torch.manual_seed(7)
    cell = build_mixer_cell(cell_name, d_model=16, n_heads=2)
    cell.eval()
    x = torch.randn(2, 9, 16)
    with torch.no_grad():
        y_seq, _ = cell(x)
        state = cell.init_state(2, x.device, x.dtype)
        ys = []
        for t in range(x.shape[1]):
            yt, state = cell(x[:, t : t + 1, :], state)
            ys.append(yt)
        y_step = torch.cat(ys, dim=1)
    assert torch.allclose(y_seq, y_step, atol=1e-5, rtol=1e-4)


def test_householder_operator_vs_full_write_jacobian():
    torch.manual_seed(11)
    x = torch.randn(1, 16)
    s0 = torch.randn(1, 2, 8, 8)

    c_cell = ContractionDeltaCell(16, 2)
    h_cell = HouseholderDeltaCell(16, 2)
    g_cell = GeoEraseWriteCell(16, 2)
    with torch.no_grad():
        # Push b_proj positive to compare upper eigenvalue ranges
        c_cell.b_proj.bias.fill_(8.0)
        h_cell.b_proj.bias.fill_(8.0)
        g_cell.b_proj.bias.fill_(8.0)
        g_cell.eta_proj.bias.fill_(4.0)

    c_spec = compute_transition_and_jacobian_spectrum(c_cell, x, s0)
    h_spec = compute_transition_and_jacobian_spectrum(h_cell, x, s0)
    g_spec = compute_transition_and_jacobian_spectrum(g_cell, x, s0)

    assert c_spec["homogeneous_A_sigma_max"] <= 1.0 + 1e-5
    assert c_spec["homogeneous_A_min_real_eig"] >= -1e-5  # contraction cannot reflect negative!
    assert h_spec["homogeneous_A_sigma_max"] <= 1.0 + 1e-5
    assert h_spec["homogeneous_A_min_real_eig"] < -0.5  # Householder reflection crosses negative real axis
    assert g_spec["homogeneous_A_sigma_max"] <= 1.0 + 1e-5
    assert g_spec["full_step_jacobian_sigma_max"] > 0.0

    # Verify step_matrix_operator matches forward state step on GeoEraseWriteCell
    with torch.no_grad():
        _, s_fwd = g_cell(x.unsqueeze(1), s0)
        _, s_op, _ = g_cell.step_matrix_operator(x, s0)
    assert torch.allclose(s_fwd, s_op, atol=1e-5)


def test_complex_roto_ssm_rotation_and_memory_profile():
    cell = ComplexRotoSSMCell(16, 2)
    x = torch.randn(1, 16)
    s0 = torch.randn(1, 2, 8, 8)
    spec = compute_transition_and_jacobian_spectrum(cell, x, s0)
    assert spec["full_step_jacobian_sigma_max"] > 0.0

    model = ToySequenceModel(
        vocab_size=32,
        d_model=16,
        n_heads=2,
        layer_names=("geo_erase_write", "geo_erase_write", "geo_erase_write", "dense_gqa"),
        num_classes=4,
        learnable_init_state=True,
    )
    assert set(model.init_states.keys()) == {"0", "1", "2"}
    prof_short = model.memory_profile_bytes(seq_len=64, dtype_bytes=4)
    prof_long = model.memory_profile_bytes(seq_len=1024, dtype_bytes=4)
    assert prof_short["recurrent_state_bytes"] == prof_long["recurrent_state_bytes"]
    assert prof_long["kv_cache_bytes"] > prof_short["kv_cache_bytes"]
    assert not math.isnan(model(torch.randint(0, 32, (2, 12))).sum().item())
