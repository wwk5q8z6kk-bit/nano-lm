"""Unit, algebraic-invariance, O(log T)/O(1) scan equivalence, and decontamination tests for nanolm_holonomy."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from nanolm_frontier.benchmark_15d import BenchmarkTask, build_benchmark_15d_suite, verify_benchmark_response
from nanolm_frontier.curriculum import verify_zero_contamination
from nanolm_holonomy.architecture import (
    ABSOLUTE_MAX_PARAMETERS,
    HSPA_2B_CONFIG,
    HSPA_4B_CONFIG,
    TRANSFORMER_4B_CONFIG,
    HSPAArchConfig,
    HSPAFamily,
    HSPAUnifiedLM,
    ParameterBudgetExceededError,
    build_suite_configs,
)
from nanolm_holonomy.registry import ExperimentRegistry, RegistryVersion
from nanolm_holonomy.sidecar_evolution import (
    HolonomicRegisterSidecar,
    _bypass_all_holonomy_adapters,
    apply_holonomic_graft,
    build_challenging_rlvr_curriculum,
)
from nanolm_holonomy.spinor_algebra import (
    abelian_complex_scan,
    cayley_rotor,
    complex_cayley_rotor,
    complex_product,
    hamilton_product,
    parallel_holonomy_scan,
    sequential_holonomy_reference,
    spinor_conjugate,
    step_holonomy,
)
from nanolm_posttrain.lora import LoRAConfig


def test_su2_spinor_algebra_identities() -> None:
    torch.manual_seed(0)
    a = torch.randn(4, 5, 4)
    b = torch.randn(4, 5, 4)
    c = torch.randn(4, 5, 4)
    # Strict non-commutativity
    assert (hamilton_product(a, b) - hamilton_product(b, a)).abs().max().item() > 1e-2
    # Strict associativity
    lhs = hamilton_product(hamilton_product(a, b), c)
    rhs = hamilton_product(a, hamilton_product(b, c))
    assert (lhs - rhs).abs().max().item() < 1e-5
    # Exact Cayley identity and unit norm
    w_zero = torch.zeros(3, 3)
    id_q = cayley_rotor(w_zero)
    assert torch.allclose(id_q, torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(id_q))
    u = cayley_rotor(torch.randn(5, 3))
    assert torch.allclose(u.norm(dim=-1), torch.ones(5), atol=1e-6)
    # Reverse-order conjugate unbinding: conj(H * q) * (H * q * v) == |H|^2 * v
    h = cayley_rotor(torch.randn(5, 3)) * 1.3
    hq = hamilton_product(h, u)
    bound = hamilton_product(hq, a[0])
    unbound = hamilton_product(spinor_conjugate(hq), bound)
    h_sq = (h * h).sum(dim=-1, keepdim=True)
    assert (unbound - h_sq * a[0]).abs().max().item() < 1e-4


def test_parallel_holonomy_scan_matches_sequential_and_chunks() -> None:
    torch.manual_seed(42)
    for seq_len in (1, 7, 19, 32, 47):
        g = cayley_rotor(torch.randn(2, 4, 6, seq_len, 3, requires_grad=True)) * 0.92
        s = torch.randn(2, 4, 6, seq_len, 4, requires_grad=True)
        init = torch.randn(2, 4, 6, 4)
        ref = sequential_holonomy_reference(g, s, init_state=init)
        par, last_p = parallel_holonomy_scan(g, s, init_state=init, chunk_size=None)
        chk, last_c = parallel_holonomy_scan(g, s, init_state=init, chunk_size=11)
        assert (ref - par).abs().max().item() < 1e-5
        assert (ref - chk).abs().max().item() < 1e-5
        assert (par[..., -1, :] - last_p).abs().max().item() < 1e-6
        assert (chk[..., -1, :] - last_c).abs().max().item() < 1e-6
        # Verify step_holonomy matches final recurrence step
        stepped = step_holonomy(g[..., -1, :], s[..., -1, :], ref[..., -2, :] if seq_len > 1 else init)
        assert (ref[..., -1, :] - stepped).abs().max().item() < 1e-5
        par.sum().backward()


def test_all_five_hspa_families_iso_param_decode_and_4b_cap() -> None:
    torch.manual_seed(9)
    for fam in HSPAFamily:
        cfg = HSPAArchConfig(
            name=f"test-{fam.value}",
            family=fam,
            vocab_size=256,
            d_model=64,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            d_head=16,
            d_ff=128,
            n_registers=4,
            d_fiber=32,
            max_seq_len=256,
        )
        m = HSPAUnifiedLM(cfg)
        assert cfg.parameter_count == m.num_parameters
        x = torch.randint(0, 256, (2, 18))
        full_lg = m(x)
        assert full_lg.shape == (2, 18, 256)
        full_lg.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)

        # Verify exact O(1) incremental step decode matches parallel prefill
        m.eval()
        with torch.no_grad():
            ref_lg = m(x)
            _, st = m(x[:, :10], use_state=True)
            incs = []
            for t in range(10, 18):
                lt, st = m(x[:, t : t + 1], past_states=st, use_state=True, past_len=t)
                incs.append(lt)
            assert (ref_lg[:, 10:] - torch.cat(incs, dim=1)).abs().max().item() < 1e-4

    # Check suite iso-parameter spreads and 4B ceiling configs
    suite = build_suite_configs(preset="micro")
    pcounts = [c.parameter_count for c in suite]
    assert (max(pcounts) - min(pcounts)) / min(pcounts) < 0.01

    assert 1_700_000_000 <= HSPA_2B_CONFIG.parameter_count <= 2_100_000_000
    assert 3_600_000_000 <= HSPA_4B_CONFIG.parameter_count <= ABSOLUTE_MAX_PARAMETERS
    assert 3_600_000_000 <= TRANSFORMER_4B_CONFIG.parameter_count <= ABSOLUTE_MAX_PARAMETERS
    assert HSPA_4B_CONFIG.state_bytes_per_sequence(16384, 2) < TRANSFORMER_4B_CONFIG.state_bytes_per_sequence(16384, 2) / 500

    with pytest.raises(ParameterBudgetExceededError):
        replace(HSPA_4B_CONFIG, n_layers=64)


def test_registry_promotion_and_rejection_policy() -> None:
    reg = ExperimentRegistry(higher_is_better=False)
    reg.register_baseline(RegistryVersion("b0", "b0_transformer", 14_000_000, 0.7000, [0.701, 0.699]))

    # Reject single seed
    dec1 = reg.evaluate_candidate(RegistryVersion("single", "x", 14_000_000, 0.6500, [0.6500]), min_seeds=2)
    assert not dec1["promoted"]
    assert reg.best and reg.best.name == "b0"

    # Reject worse candidate
    dec2 = reg.evaluate_candidate(RegistryVersion("worse", "x", 14_000_000, 0.7200, [0.721, 0.719]), min_seeds=2)
    assert not dec2["promoted"]
    assert reg.best.name == "b0"

    # Promote genuinely better multi-seed candidate
    dec3 = reg.evaluate_candidate(RegistryVersion("gen2", "gen2_h3", 14_000_000, 0.6800, [0.681, 0.679]), min_seeds=2)
    assert dec3["promoted"]
    assert reg.best.name == "gen2"
    assert reg.baseline.name == "b0"


class _MockLayer(nn.Module):
    def __init__(self, d: int = 32) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a = self.o_proj(self.q_proj(h) + self.k_proj(h) + self.v_proj(h))
        return h + self.down_proj(torch.nn.functional.silu(self.gate_proj(a)) * self.up_proj(a))


class _MockHFModel(nn.Module):
    def __init__(self, vocab: int = 64, d: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.config = SimpleNamespace(hidden_size=d, num_attention_heads=4, head_dim=d // 4)
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(_MockLayer(d) for _ in range(n_layers))
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(input_ids)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


def test_holonomic_sidecar_exact_step0_and_challenging_rl_decontam() -> None:
    torch.manual_seed(5)
    m = _MockHFModel()
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        base_lg = m(x).clone()

    info = apply_holonomic_graft(m, lora_config=LoRAConfig(rank=4, alpha=8.0), n_sidecar_layers=2, n_registers=4, d_fiber=16)
    assert info["holonomic_sidecar_layers"] == 2
    assert any(isinstance(mod, HolonomicRegisterSidecar) for mod in m.modules())

    with torch.no_grad():
        graft_lg = m(x)
    assert (base_lg - graft_lg).abs().max().item() < 1e-6

    m(x).sum().backward()
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(-0.05 * p.grad)
        mut_lg = m(x)
    assert (base_lg - mut_lg).abs().max().item() > 1e-3

    with _bypass_all_holonomy_adapters(m), torch.no_grad():
        rec_lg = m(x)
    assert (base_lg - rec_lg).abs().max().item() < 1e-6

    # Verify all generated RL prompts pass their own verifier and have 0 overlap with benchmark_15d
    rl_cur = build_challenging_rlvr_curriculum(seed=777)
    eval_tasks = build_benchmark_15d_suite()
    dec = verify_zero_contamination(rl_cur, eval_tasks)
    assert dec["overlaps"] == 0
    for ex in rl_cur:
        v = verify_benchmark_response(
            BenchmarkTask(ex.example_id, ex.dim, "hard", ex.system_prompt, ex.user_prompt, ex.verifier_spec),
            ex.reference_response,
        )
        assert v.passed and v.score >= 0.99, f"Failed reference on {ex.example_id}: {v}"


def test_holonomy_scan_fn_gradcheck_and_optional_cuda_triton() -> None:
    from nanolm_holonomy.triton_holonomy import HAS_TRITON, holonomy_scan_fast

    torch.manual_seed(17)
    g = torch.randn(1, 2, 2, 6, 4, dtype=torch.float64, requires_grad=True)
    s = torch.randn(1, 2, 2, 6, 4, dtype=torch.float64, requires_grad=True)
    init = torch.randn(1, 2, 2, 4, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(lambda gg, ss: holonomy_scan_fast(gg, ss, init_state=None)[0], (g, s), eps=1e-6, atol=1e-5)
    assert torch.autograd.gradcheck(lambda gg, ss, ii: holonomy_scan_fast(gg, ss, init_state=ii)[0], (g, s, init), eps=1e-6, atol=1e-5)

    # On RunPod CUDA hosts, explicitly prove fused Triton kernel matches sequential ref in forward & backward
    if HAS_TRITON and torch.cuda.is_available():
        for seq_len in (31, 300):  # single-tile and multi-tile (> 256)
            gc_t = cayley_rotor(torch.randn(2, 4, 4, seq_len, 3, device="cuda")).detach().requires_grad_(True)
            sc_t = torch.randn(2, 4, 4, seq_len, 4, device="cuda", requires_grad=True)
            ic_t = torch.randn(2, 4, 4, 4, device="cuda", requires_grad=True)

            out_tr, _ = holonomy_scan_fast(gc_t, sc_t, init_state=ic_t)
            out_tr.sum().backward()
            dg_tr, ds_tr, di_tr = gc_t.grad.clone(), sc_t.grad.clone(), ic_t.grad.clone()

            gc_r = gc_t.detach().clone().requires_grad_(True)
            sc_r = sc_t.detach().clone().requires_grad_(True)
            ic_r = ic_t.detach().clone().requires_grad_(True)
            out_ref = sequential_holonomy_reference(gc_r, sc_r, init_state=ic_r)
            out_ref.sum().backward()

            assert torch.allclose(out_tr.float(), out_ref.float(), rtol=1e-4, atol=1e-4)
            assert (dg_tr - gc_r.grad).abs().max().item() / gc_r.grad.abs().max().clamp_min(1.0).item() < 1e-4
            assert (ds_tr - sc_r.grad).abs().max().item() / sc_r.grad.abs().max().clamp_min(1.0).item() < 1e-4
            assert (di_tr - ic_r.grad).abs().max().item() / ic_r.grad.abs().max().clamp_min(1.0).item() < 1e-4


def test_curriculum_v2_and_zero_oracle_syntax_gate() -> None:
    from nanolm_holonomy.curriculum_v2 import build_curriculum_v2
    from nanolm_holonomy.cycle2_experiments import is_syntax_valid_zero_oracle

    eval_tasks = build_benchmark_15d_suite()
    cur_v2 = build_curriculum_v2()
    assert len(cur_v2) >= 60
    assert verify_zero_contamination(cur_v2, eval_tasks)["overlaps"] == 0
    for ex in cur_v2:
        v = verify_benchmark_response(
            BenchmarkTask(ex.example_id, ex.dim, "medium", ex.system_prompt, ex.user_prompt, ex.verifier_spec),
            ex.reference_response,
        )
        assert v.passed and v.score >= 0.99, f"curriculum_v2 reference failure {ex.example_id}: {v}"

    # Zero-oracle syntax gate only checks AST / JSON syntax without any task spec access
    assert is_syntax_valid_zero_oracle("Return only valid Python.", "```python\ndef f(x):\n    return x + 1\n```")[0]
    assert not is_syntax_valid_zero_oracle("Return only valid Python.", "```python\ndef broken(:\n```")[0]
    assert is_syntax_valid_zero_oracle("Return only valid JSON.", '{"a": 1}')[0]
    assert not is_syntax_valid_zero_oracle("Return only valid JSON.", '{"a": ')[0]
    assert is_syntax_valid_zero_oracle("Solve step by step.", "unstructured prose")[0]
