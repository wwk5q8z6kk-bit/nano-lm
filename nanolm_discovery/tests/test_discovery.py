"""Unit and invariant tests for the NanoLM Discovery package."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nano_ai.training.model import NANO_MODEL_CONFIG, NanoGPT
from nanolm_discovery.config import (
    HARD_MAX_DEPLOYED_PARAMETERS,
    ArchitectureConfig,
    ParameterBudgetError,
    ParadigmKind,
    compute_parameter_census,
    make_iso_param_suite,
    retune_iso_param_variant,
)
from nanolm_discovery.eval_benchmarks import (
    generate_multi_needle_batch,
    generate_overwrite_batch,
    generate_s3_batch,
)
from nanolm_discovery.models import NanoLMDiscoveryModel
from nanolm_discovery.operators import HMWROperator, PortHamiltonianOperator, MPRNOperator
from nanolm_discovery.optim import build_optimizer
from nanolm_discovery.registry import InnovationScore, LineageRegistry


def test_original_baseline_and_exact_census_match():
    orig = NanoGPT(NANO_MODEL_CONFIG)
    assert sum(p.numel() for p in orig.parameters()) == 3_148_608

    for tier in ("mech", "scale_8m", "discovery_18m"):
        suite = make_iso_param_suite(tier)
        b0_count = compute_parameter_census(suite[ParadigmKind.B0_TRANSFORMER]).total_deployed_parameters
        for kind, cfg in suite.items():
            model = NanoLMDiscoveryModel(cfg)
            census = compute_parameter_census(cfg)
            actual = model.deployed_parameter_count()
            assert actual == census.total_deployed_parameters, f"{kind} mismatch: {actual} != {census.total_deployed_parameters}"
            rel_diff = abs(actual - b0_count) / float(b0_count)
            assert rel_diff <= 0.005, f"{kind} tier {tier} exceeded ±0.5% iso-param band: {rel_diff:.4f}"


def test_strict_4b_flagship_and_budget_enforcement():
    suite_4b = make_iso_param_suite("flagship_4b")
    targets = []
    for kind, cfg in suite_4b.items():
        census = compute_parameter_census(cfg, enforce_strict_4b=True)
        assert 3_750_000_000 <= census.total_deployed_parameters < HARD_MAX_DEPLOYED_PARAMETERS
        targets.append(census.total_deployed_parameters)
        with torch.device("meta"):
            meta_model = NanoLMDiscoveryModel(cfg, skip_init_weights=True)
        assert meta_model.deployed_parameter_count() == census.total_deployed_parameters
    assert (max(targets) - min(targets)) / float(targets[0]) <= 0.005

    too_big = ArchitectureConfig(
        name="over_budget",
        paradigm=ParadigmKind.B0_TRANSFORMER,
        vocab_size=65_536,
        d_model=3072,
        n_layers=36,
        n_heads=24,
        n_kv_heads=8,
        head_dim=128,
        ffn_hidden=12_288,
    )
    with pytest.raises(ParameterBudgetError):
        compute_parameter_census(too_big, enforce_strict_4b=True)


def test_training_only_jep_head_excluded_from_deployed_budget():
    suite = make_iso_param_suite("mech", jep_head_enabled=True)
    cfg = suite[ParadigmKind.G2_MPRN]
    model = NanoLMDiscoveryModel(cfg)
    assert model.deployed_parameter_count() == model.census.total_deployed_parameters
    assert model.training_only_parameter_count() == model.census.training_only_parameters > 0
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, targets=y, jep_weight=0.1)
    assert out.jep_loss is not None and torch.isfinite(out.loss)


@pytest.mark.parametrize("kind", list(ParadigmKind))
def test_exact_parallel_recurrent_duality(kind: ParadigmKind):
    torch.manual_seed(7)
    cfg = make_iso_param_suite("mech")[kind]
    model = NanoLMDiscoveryModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 48))
    with torch.no_grad():
        out_par = model(x)
        states = None
        step_logits = []
        for t in range(x.shape[1]):
            st = model.forward_step(x[:, t], states=states)
            step_logits.append(st.logits[:, 0])
            states = st.states
        logits_seq = torch.stack(step_logits, dim=1)
    max_err = (out_par.logits - logits_seq).abs().max().item()
    assert max_err < 1e-4, f"{kind} parallel vs step diverged: max_err={max_err:.3e}"


@pytest.mark.parametrize("kind", list(ParadigmKind))
def test_strict_causality_zero_future_leakage(kind: ParadigmKind):
    torch.manual_seed(11)
    cfg = make_iso_param_suite("mech")[kind]
    model = NanoLMDiscoveryModel(cfg).eval()
    x1 = torch.randint(0, cfg.vocab_size, (2, 40))
    x2 = x1.clone()
    t0 = 19
    x2[:, t0 + 1 :] = (x2[:, t0 + 1 :] + 7) % cfg.vocab_size
    with torch.no_grad():
        o1 = model(x1)
        o2 = model(x2)
    diff = (o1.logits[:, : t0 + 1] - o2.logits[:, : t0 + 1]).abs().max().item()
    assert diff == 0.0 or diff < 1e-6, f"{kind} leaked future tokens into <=t0: {diff}"
    if o1.token_uncertainty is not None and o2.token_uncertainty is not None:
        u_diff = (o1.token_uncertainty[:, : t0 + 1] - o2.token_uncertainty[:, : t0 + 1]).abs().max().item()
        assert u_diff == 0.0 or u_diff < 1e-6


def test_unconditional_spectral_stability_of_metriplectic_operators():
    torch.manual_seed(19)
    d, h, hd = 32, 4, 8
    eye = torch.eye(hd)
    for op in (
        PortHamiltonianOperator(d, h, hd),
        HMWROperator(d, h, hd),
        MPRNOperator(d, h, hd, use_symplectic_reflection=True, use_precision_norm=True, use_scam_memory=True),
    ):
        x = torch.randn(3, 12, d) * 10.0  # Extreme amplitude stress test
        with torch.no_grad():
            if isinstance(op, (PortHamiltonianOperator, HMWROperator)):
                u, k, _, _, beta, tau, log_gamma = op._project_terms(x)
            else:
                bsz, t_len, _ = x.shape
                u = torch.nn.functional.normalize(op.u_proj(x).view(bsz, t_len, h, hd), dim=-1).transpose(1, 2)
                k = torch.nn.functional.normalize(op.k_proj(x).view(bsz, t_len, h, hd), dim=-1).transpose(1, 2)
                gates = op.gate_proj(x).view(bsz, t_len, h, 3).permute(0, 2, 1, 3)
                beta = 1.98 * torch.sigmoid(gates[..., 0])
                tau = torch.sigmoid(gates[..., 1])
                log_gamma = -torch.nn.functional.softplus(gates[..., 2]) * 0.20
            # Explicit homogeneous transition A_t = gamma_t (I - tau k k^T)(I - beta u u^T)
            p_orth = eye - beta[..., None, None] * torch.einsum("bhtd,bhte->bhtde", u, u)
            p_diss = eye - tau[..., None, None] * torch.einsum("bhtd,bhte->bhtde", k, k)
            gamma = torch.exp(log_gamma)[..., None, None]
            a_mat = gamma * torch.matmul(p_diss, p_orth)
            max_sv = torch.linalg.matrix_norm(a_mat, ord=2).max().item()
            assert max_sv <= 1.0 + 1e-5, f"Spectral norm exceeded 1.0: {max_sv}"


def test_muon_adamw_and_ablation_iso_param_and_registry(tmp_path):
    suite = make_iso_param_suite("mech")
    g2 = suite[ParadigmKind.G2_MPRN]
    abl = retune_iso_param_variant(g2, suite[ParadigmKind.B0_TRANSFORMER], name="g2_no_refl", align=2, use_symplectic_reflection=False)
    b0_cnt = compute_parameter_census(suite[ParadigmKind.B0_TRANSFORMER]).total_deployed_parameters
    abl_cnt = NanoLMDiscoveryModel(abl).deployed_parameter_count()
    assert abs(abl_cnt - b0_cnt) / float(b0_cnt) <= 0.005

    g3 = suite[ParadigmKind.G3_HMWR]
    g3_abl = retune_iso_param_variant(
        g3, suite[ParadigmKind.B0_TRANSFORMER], name="g3_no_depth", align=2, use_depth_routing=False
    )
    g3_abl_cnt = NanoLMDiscoveryModel(g3_abl).deployed_parameter_count()
    assert abs(g3_abl_cnt - b0_cnt) / float(b0_cnt) <= 0.005

    model = NanoLMDiscoveryModel(g2)
    opt = build_optimizer(model, kind="muon", lr=1e-3, lr_muon=2e-3)
    x = torch.randint(0, g2.vocab_size, (2, 16))
    y = torch.randint(0, g2.vocab_size, (2, 16))
    out = model(x, targets=y)
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    assert all(torch.isfinite(p).all() for p in model.parameters())

    reg = LineageRegistry(tmp_path / "lineage.json")
    reg.register_baseline({"name": "b0", "composite_bpb": 1.50, "ood_composite_acc": 0.40})
    with pytest.raises(RuntimeError, match="immutable"):
        reg.register_baseline({"name": "b0_hack", "composite_bpb": 1.00})
    v_bad = reg.submit_experimental({"name": "bad", "composite_bpb": 1.65, "ood_composite_acc": 0.30})
    assert v_bad["decision"] == "REJECTED"
    assert reg.data["best"]["name"] == "b0"
    v_good = reg.submit_experimental({"name": "good", "composite_bpb": 1.45, "ood_composite_acc": 0.55})
    assert v_good["decision"] == "PROMOTED_TO_BEST"
    assert reg.data["best"]["name"] == "good"

    score = InnovationScore(8.5, 9.0, 9.0, 8.0, 7.5)
    assert score.mean_score == pytest.approx(8.4)


def test_synthetic_generators_valid_shapes():
    rng = np.random.default_rng(0)
    dev = torch.device("cpu")
    xs, ys, ms = generate_s3_batch(4, 32, rng=rng, device=dev)
    assert xs.shape == (4, 32) and ys.shape == (4, 32) and ms.all()
    xo, yo, mo = generate_overwrite_batch(4, 48, rng=rng, device=dev)
    assert xo.shape == (4, 48) and mo.sum().item() >= 4
    xn, yn, mn = generate_multi_needle_batch(4, 256, n_needles=8, rng=rng, device=dev)
    assert xn.shape == (4, 256) and mn.sum().item() == 4 * 8


@pytest.mark.parametrize("kind", list(ParadigmKind))
def test_mps_or_cpu_bfloat16_autocast_finite_grad(kind: ParadigmKind):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = make_iso_param_suite("mech")[kind]
    model = NanoLMDiscoveryModel(cfg).to(dev)
    x = torch.randint(0, cfg.vocab_size, (2, 128), device=dev)
    y = torch.randint(0, cfg.vocab_size, (2, 128), device=dev)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
        out = model(x, targets=y)
    assert torch.isfinite(out.logits).all()
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("kind", [ParadigmKind.I3_HRWN, ParadigmKind.G3_HMWR])
def test_recurrent_ring_storage_matches_logical_state(kind):
    cfg = make_iso_param_suite("mech")[kind]
    model = NanoLMDiscoveryModel(cfg).eval()
    with torch.no_grad():
        states = model(torch.randint(0, cfg.vocab_size, (2, 128)), return_states=True).states
    assert states
    checked = set()
    for layer in states:
        for name in ("shift", "wave_ring", "prev_wave"):
            tensor = layer.get(name)
            if isinstance(tensor, torch.Tensor):
                assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
                checked.add(name)
    assert {"wave_ring", "prev_wave"} <= checked
