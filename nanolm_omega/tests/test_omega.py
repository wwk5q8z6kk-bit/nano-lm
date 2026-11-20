"""Synthetic-only unit and invariant tests for the `nanolm_omega` package."""

from __future__ import annotations

import random

import pytest
import torch

from nano.contracts import EpistemicStatus
from nano.kernel import Identity
from nanolm_discovery.config import HARD_MAX_DEPLOYED_PARAMETERS, ParameterBudgetError, ParadigmKind, compute_parameter_census, make_iso_param_suite
from nanolm_omega.backbone import (
    G4OmegaModel,
    OmegaBackboneConfig,
    OmegaVariant,
    X8DualUnembedModel,
    build_omega_model,
    compute_omega_census,
    make_omega_suite,
)
from nanolm_omega.cognition_efe import CognitiveAgentModel, HardCausalSWEEnv, _collect_val_span
from nanolm_omega.unified_system import (
    OmegaComputeRouter,
    OmegaUnifiedCognitiveSystem,
    SLWContinualMemoryLedger,
    build_omega_bench_01,
    evaluate_omega_bench_01,
)


@pytest.mark.parametrize("tier", ["mech", "discovery_18m"])
def test_omega_exact_census_and_iso_param_parity(tier: str):
    b0_count = compute_parameter_census(make_iso_param_suite(tier)[ParadigmKind.B0_TRANSFORMER]).total_deployed_parameters
    suite = make_omega_suite(tier)
    for var, cfg in suite.items():
        c = compute_omega_census(cfg)
        m = build_omega_model(cfg)
        actual = m.deployed_parameter_count()
        assert actual == c.total_deployed_parameters, f"{var} census mismatch: {actual} != {c.total_deployed_parameters}"
        rel_diff = abs(actual - b0_count) / float(b0_count)
        assert rel_diff <= 0.005, f"{var} exceeded +-0.5% parity: {rel_diff:.4%}"


def test_omega_strict_4b_budget_and_honest_unembed_flop_accounting():
    suite_4b = make_omega_suite("flagship_4b")
    for var, cfg in suite_4b.items():
        c = compute_omega_census(cfg, enforce_strict_4b=True)
        assert 3_800_000_000 <= c.total_deployed_parameters < HARD_MAX_DEPLOYED_PARAMETERS
        with torch.device("meta"):
            m = build_omega_model(cfg)
        assert m.deployed_parameter_count() == c.total_deployed_parameters

    x8_c = compute_omega_census(suite_4b[OmegaVariant.X8_DUAL_UNEMBED_ATRD])
    g4_c = compute_omega_census(suite_4b[OmegaVariant.G4_OMEGA])
    assert x8_c.unembed_passes == 2
    assert g4_c.unembed_passes == 1
    assert g4_c.true_flops_per_token_4k < x8_c.true_flops_per_token_4k
    assert g4_c.state_bytes_at_32768_bf16 * 10 < x8_c.state_bytes_at_32768_bf16

    too_big = OmegaBackboneConfig(
        name="too_big",
        variant=OmegaVariant.G4_OMEGA,
        vocab_size=65_536,
        d_model=3200,
        n_layers=36,
        n_heads=25,
        n_kv_heads=5,
        head_dim=128,
        ffn_hidden=10_000,
        refiner_ffn=1024,
    )
    with pytest.raises(ParameterBudgetError):
        compute_omega_census(too_big, enforce_strict_4b=True)


@pytest.mark.parametrize("var", [OmegaVariant.G4_OMEGA, OmegaVariant.G4_NO_VRES, OmegaVariant.G4_NO_SPARSE_REFINER])
def test_g4_omega_parallel_recurrent_duality(var: OmegaVariant):
    torch.manual_seed(17)
    cfg = make_omega_suite("mech")[var]
    model = build_omega_model(cfg).eval()
    assert isinstance(model, G4OmegaModel)
    x = torch.randint(0, cfg.vocab_size, (2, 44))
    with torch.no_grad():
        par = model(x)
        states = None
        seq = []
        for t in range(x.shape[1]):
            st = model.forward_step(x[:, t], states=states)
            seq.append(st.logits[:, 0])
            states = st.states
        logits_seq = torch.stack(seq, dim=1)
    assert (par.logits - logits_seq).abs().max().item() < 1e-4


@pytest.mark.parametrize("var", list(OmegaVariant))
def test_omega_strict_causality_and_finite_bf16_grads(var: OmegaVariant):
    torch.manual_seed(23)
    cfg = make_omega_suite("mech")[var]
    model = build_omega_model(cfg).eval()
    x1 = torch.randint(0, cfg.vocab_size, (2, 36))
    x2 = x1.clone()
    x2[:, 18:] = (x2[:, 18:] + 11) % cfg.vocab_size
    with torch.no_grad():
        l1 = model(x1).logits[:, :18]
        l2 = model(x2).logits[:, :18]
    assert (l1 - l2).abs().max().item() < 1e-6

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_omega_model(cfg).to(dev).train()
    x = torch.randint(0, cfg.vocab_size, (2, 64), device=dev)
    y = torch.randint(0, cfg.vocab_size, (2, 64), device=dev)
    if dev.type == "mps":
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            out = model(x, targets=y)
    else:
        out = model(x, targets=y)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_g4_metriplectic_spectral_norm_bound():
    cfg = make_omega_suite("mech")[OmegaVariant.G4_OMEGA]
    m = G4OmegaModel(cfg).eval()
    eye = torch.eye(cfg.head_dim)
    x = torch.randn(2, 10, cfg.d_model) * 8.0
    for blk in m.blocks:
        x_addr, _ = blk.key_shift(x)
        u, k, _, _, beta, tau, log_gamma = blk._project_terms(x, x_addr)
        p_orth = eye - beta[..., None, None] * torch.einsum("bhtd,bhte->bhtde", u, u)
        p_diss = eye - tau[..., None, None] * torch.einsum("bhtd,bhte->bhtde", k, k)
        a_mat = torch.exp(log_gamma)[..., None, None] * torch.matmul(p_diss, p_orth)
        assert torch.linalg.matrix_norm(a_mat, ord=2).max().item() <= 1.0 + 1e-5


def test_hard_causal_swe_super_ood_24_and_omega_efe():
    rng = random.Random(101)
    for regime, lo, hi in (("id", 4, 6), ("ood", 9, 12), ("super_ood", 16, 24)):
        ep = HardCausalSWEEnv.sample_episode(rng, regime=regime)
        assert lo <= ep.n_modules <= hi
        val_mem: list[tuple[int, list[float]]] = []
        _collect_val_span(HardCausalSWEEnv.initial_observation(ep), val_mem, 0)
        # Verify oracle solves 100% of sampled hard episodes cleanly
        for st in range(HardCausalSWEEnv.MAX_STEPS):
            a, m, v = HardCausalSWEEnv.oracle_action(ep)
            obs, done, ok = HardCausalSWEEnv.step(ep, a, m, v)
            _collect_val_span(obs, val_mem, st + 1)
            if done:
                assert ok
                break
        agent = CognitiveAgentModel("omega_efe_planner", d=64)
        out = agent.forward_state(ep, val_mem, torch.device("cpu"))
        assert out["mod_logits"].shape == (ep.n_modules,)
        assert out["act_logits"].shape == (7,)


def test_tool_recovery_requires_successful_simulated_verification(monkeypatch):
    from nanolm_omega.cognition_efe import ACT_STOP, HardCausalSWEEnv
    from nanolm_omega.unified_system import _verify_tool_episode

    assert _verify_tool_episode({"verified_outcome": "RESOLVED_VERIFIED"})[0] == "UNVERIFIED"
    assert _verify_tool_episode({"episode_seed": 17000})[0] == "RESOLVED_VERIFIED"
    step = HardCausalSWEEnv.step

    def reject_stop(episode, action, module, value):
        obs, done, verified = step(episode, action, module, value)
        return obs, done, False if action == ACT_STOP else verified

    monkeypatch.setattr(HardCausalSWEEnv, "step", staticmethod(reject_stop))
    assert _verify_tool_episode({"episode_seed": 17000})[0] == "UNVERIFIED"


def test_slw_provenance_identity_guard_and_unified_bench():
    ledger = SLWContinualMemoryLedger(subject_id="omega_unit_subj")
    other = Identity(subject_id="wrong_subj", kind="subject")
    with pytest.raises(ValueError, match="Cross-subject contamination"):
        ledger.ingest_fact("svc", "timeout", "100", subject=other)

    ledger.ingest_fact("svc", "timeout", "100", version=1)
    ledger.ingest_fact("svc", "timeout", "450", version=2)
    ledger.ingest_fact("svc", "timeout", "125", version=1)
    assert ledger.query("svc", "timeout") == ("450", EpistemicStatus.DIRECT_DOCUMENTATION)
    assert ledger.query("svc", "missing") == ("INSUFFICIENT_EVIDENCE", EpistemicStatus.NOT_FOUND)
    ledger.ingest_fact("svc", "quorum", "3", version=2)
    ledger.ingest_fact("svc", "quorum", "5", version=2)
    assert ledger.query("svc", "quorum") == ("CONFLICT_UNRESOLVED", EpistemicStatus.CONFLICTING)

    router = OmegaComputeRouter()
    p_math = "A research team chooses 2 sensors out of 7 stations across 3 shifts and reserves 2 calibration slots. How many total active assignments exist?"
    assert router.route_legacy_v2(p_math) == "FAST_DIRECT"
    assert router.route_omega(p_math) == "DUAL_SPEC_MATH"

    tasks = build_omega_bench_01()
    assert len(tasks) == 60
    summary = evaluate_omega_bench_01()
    assert summary["omega_unified_system"]["accuracy"] > summary["legacy_keyword_system_v2"]["accuracy"]
    assert summary["omega_unified_system"]["avg_model_calls"] < summary["omega_no_early_halt"]["avg_model_calls"]
    assert summary["omega_unified_system"]["abstention_and_conflict_acc"] == pytest.approx(1.0)


@pytest.mark.parametrize("var", [OmegaVariant.X8_DUAL_UNEMBED_ATRD, OmegaVariant.G4_OMEGA])
def test_omega_fp16_scaled_grad_stability(var: OmegaVariant):
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(31)
    cfg = make_omega_suite("mech")[var]
    model = build_omega_model(cfg).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scale = 65536.0
    for step in range(8):
        x = torch.randint(0, cfg.vocab_size, (4, 48), device=dev)
        y = torch.randint(0, cfg.vocab_size, (4, 48), device=dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type == "mps")):
            out = model(x, targets=y)
            loss = out.loss
        assert torch.isfinite(loss), f"non-finite loss at step={step}"
        (loss * scale).backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite scaled grad at step={step}"
                p.grad.mul_(1.0 / scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def test_g4_self_calibrated_router_correlation_and_duality():
    torch.manual_seed(41)
    cfg = make_omega_suite("mech")[OmegaVariant.G4_OMEGA]
    model = build_omega_model(cfg).train()
    assert isinstance(model, G4OmegaModel)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    # Construct fixed easy vs hard token positions so self-calibrated router learns positive difficulty alignment
    x = torch.randint(4, cfg.vocab_size - 4, (8, 32))
    y = x.roll(-1, dims=1)
    y[:, 1::2] = (y[:, 1::2] + 37) % cfg.vocab_size  # odd positions require non-trivial shift; even wait
    x[:, ::2] = 2
    y[:, ::2] = 3  # predictable deterministic token -> lower NLL than random odd positions

    def _corr() -> float:
        model.eval()
        with torch.no_grad():
            o = model(x)
            ce = torch.nn.functional.cross_entropy(
                o.logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="none"
            )
            u = o.token_uncertainty.float().reshape(-1)
            uc = u - u.mean()
            cc = ce - ce.mean()
            return float((uc * cc).sum().item() / ((uc.norm() * cc.norm()).clamp_min(1e-8).item()))

    corr_before = _corr()
    model.train()
    for _ in range(18):
        opt.zero_grad(set_to_none=True)
        out = model(x, targets=y)
        out.loss.backward()
        opt.step()
    corr_after = _corr()
    assert corr_after > corr_before and corr_after > 0.1

    model.eval()
    xs = x[:2, :20]
    with torch.no_grad():
        par = model(xs).logits
        st = None
        seq = []
        for t in range(xs.shape[1]):
            so = model.forward_step(xs[:, t], states=st)
            seq.append(so.logits[:, 0])
            st = so.states
    assert (par - torch.stack(seq, dim=1)).abs().max().item() < 1e-4


@pytest.mark.parametrize("use_cross_attention", [False, True])
def test_training_refiner_uses_threshold_from_previous_batch(use_cross_attention):
    import copy
    from nanolm_omega.backbone import _LatentTokenRefiner, _SparseCausalCrossRefiner

    torch.manual_seed(29)
    hidden = torch.randn(1, 8, 8)
    if use_cross_attention:
        refiner = _SparseCausalCrossRefiner(8, 1, 8, 16)
        memory = hidden[:, None]
        args = (hidden, hidden, memory, memory)
    else:
        refiner = _LatentTokenRefiner(8, 16)
        args = (hidden, hidden, hidden)
    other = copy.deepcopy(refiner)
    first = torch.tensor([[0.6] * 4 + [0.1] * 4])
    second = torch.tensor([[0.6] * 4 + [0.9] * 4])
    with torch.no_grad():
        a = refiner(*args, first)[0]
        b = other(*args, second)[0]
    assert not torch.equal(a[:, :4], hidden[:, :4])
    torch.testing.assert_close(a[:, :4], b[:, :4])
    assert refiner.ema_tau < other.ema_tau


@pytest.mark.parametrize("fast,brute", [
    ("raise ValueError()", "raise ValueError()"),
    ("print(2)", "print(3)"),
    ("pass", "pass"),
])
def test_math_fallback_requires_verifier_agreement(fast, brute):
    from dataclasses import replace

    task = replace(build_omega_bench_01()[0], pass1_guess="1", fast_formula_code=fast, brute_spec_code=brute)
    answer, status, _, _ = OmegaUnifiedCognitiveSystem().solve_task(task)
    assert answer in {"1", "3"}
    assert status == EpistemicStatus.UNCERTAIN


def test_assertion_only_harness_and_missing_harness():
    from dataclasses import replace

    base = build_omega_bench_01()[0]
    task = replace(base, prompt="Implement def solve(xs)", code_candidates=(
        "def solve(xs): return 0", "def solve(xs): return sum(xs)"
    ), test_harness="assert solve([1, 2, 3]) == 6")
    system = OmegaUnifiedCognitiveSystem()
    assert system.solve_task(task) == ("PASS_VERIFIED", EpistemicStatus.DIRECT_MEASUREMENT, 2, 2)
    assert system.solve_task(replace(task, test_harness=""))[1] == EpistemicStatus.UNCERTAIN


@pytest.mark.parametrize("variant", list(OmegaVariant))
def test_untied_vocabulary_projection_matches_census(variant, monkeypatch):
    from dataclasses import replace
    from torch.nn import functional as F

    cfg = replace(make_omega_suite("mech")[variant], tie_embeddings=False)
    model = build_omega_model(cfg).eval()
    assert model.deployed_parameter_count() == compute_omega_census(cfg).total_deployed_parameters
    assert model.output_projection.weight.data_ptr() != model.tok_emb.weight.data_ptr()
    linear = F.linear
    projections = []

    def record_projection(x, weight, bias=None):
        if weight is model.output_projection.weight or weight is model.tok_emb.weight:
            projections.append(weight)
        return linear(x, weight, bias)

    monkeypatch.setattr(F, "linear", record_projection)
    model(torch.randint(cfg.vocab_size, (1, 8))).logits.sum().backward()
    assert len(projections) == compute_omega_census(cfg).unembed_passes
    assert all(weight is model.output_projection.weight for weight in projections)
    assert model.output_projection.weight.grad.abs().sum() > 0
