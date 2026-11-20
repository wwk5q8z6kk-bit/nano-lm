"""Unit & Mathematical Identity Tests for NANOLM Omega Modules."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import torch
from torch import nn

from nanolm_holonomy.compute_planner import (
    GPU_CATALOG,
    WorkloadSpec,
    estimate_workload,
    plan_omega_campaign,
    plan_omega_v2_campaign,
    select_optimal_gpu,
)
from nanolm_holonomy.omega_architecture import (
    SpectralNullspaceLoRALinear,
    apply_nullspace_lora,
    merge_omega_weights,
    merged_omega_inference,
    so3_projective_adjoint,
    su2_lie_exponential,
    unmerge_omega_weights,
)
from nanolm_holonomy.omega_system import (
    classify_task_mode,
    normalize_chained_mermaid_edges,
    omega_generate,
    preserve_valid_json_keys,
    preserve_valid_json_structure,
    verify_python_smoke_zero_oracle,
)
from nanolm_holonomy.spinor_algebra import hamilton_product
from nanolm_posttrain.lora import LoRAConfig


def test_su2_lie_exp_and_so3_adjoint_identities() -> None:
    torch.manual_seed(7)
    w = torch.randn(4, 5, 3, dtype=torch.float32) * 1.5
    q = su2_lie_exponential(w)
    norms = torch.linalg.vector_norm(q, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Exact 180-degree involution at ||omega|| = pi/2 gives scalar w = 0 (impossible under finite Cayley)
    inv_q = su2_lie_exponential(torch.tensor([[math.pi / 2.0, 0.0, 0.0]], dtype=torch.float32))
    assert torch.allclose(inv_q, torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32), atol=1e-5)

    # Z_2 projective invariance: R(-q) == R(q)
    assert torch.allclose(so3_projective_adjoint(-q), so3_projective_adjoint(q), atol=1e-6)

    # Homomorphism: R(q2 * q1) == R(q2) @ R(q1)
    q1 = su2_lie_exponential(torch.randn(3, 3))
    q2 = su2_lie_exponential(torch.randn(3, 3))
    r1 = so3_projective_adjoint(q1).view(3, 3, 3)
    r2 = so3_projective_adjoint(q2).view(3, 3, 3)
    r12 = so3_projective_adjoint(hamilton_product(q2, q1)).view(3, 3, 3)
    assert torch.allclose(r12, torch.bmm(r2, r1), atol=1e-5)


def test_spectral_nullspace_lora_orthogonality_and_step0() -> None:
    torch.manual_seed(11)
    lin = nn.Linear(64, 48, bias=False)
    toy = nn.Sequential()
    toy.add_module("q_proj", lin)

    x = torch.randn(3, 12, 64)
    with torch.no_grad():
        out_before = toy(x).clone()

    replaced = apply_nullspace_lora(toy, LoRAConfig(rank=8, alpha=16.0), nullspace_k=12)
    assert replaced == 1
    wrapped = toy.q_proj
    assert isinstance(wrapped, SpectralNullspaceLoRALinear)

    with torch.no_grad():
        out_step0 = toy(x)
    assert torch.allclose(out_before, out_step0, atol=1e-6)

    # Train 5 steps and verify effective DeltaW stays strictly orthogonal to V_k
    opt = torch.optim.SGD([wrapped.lora_a, wrapped.lora_b], lr=0.2)
    target = torch.randn(3, 12, 48)
    for _ in range(5):
        opt.zero_grad()
        loss = (toy(x) - target).pow(2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        vk = wrapped.v_proj_subspace.float()
        delta_w = wrapped.effective_delta_weight()
        assert float(delta_w.abs().max().item()) > 1e-4
        interference = (delta_w @ vk).abs().max().item()
        assert interference < 1e-5, f"Spectral leakage into top-k subspace: {interference}"


def test_compute_planner_preflight_and_selection() -> None:
    w_fit = WorkloadSpec(name="1.7b-sft", total_params=1_747_149_824, trainable_params=26_574_848, train_tokens=20_000, decode_tokens=5_000)
    est_l40s = estimate_workload(w_fit, GPU_CATALOG["runpod_l40s_48gb"])
    assert est_l40s["fits_vram"] is True
    assert est_l40s["estimated_peak_vram_mb"] < 15_000.0

    sel = select_optimal_gpu([w_fit], max_wall_minutes=20.0, allow_kaggle=False)
    assert sel["recommended"]["feasible"] is True

    omega_plan = plan_omega_campaign(allow_kaggle=False)
    rec = omega_plan["selection"]["recommended"]
    assert rec["fits_vram"] is True
    assert rec["estimated_usd"] < 1.0


def test_omega_cognitive_controller_zero_oracle_contract() -> None:
    sig = inspect.signature(omega_generate)
    param_names = set(sig.parameters)
    assert "verifier_spec" not in param_names
    assert "task" not in param_names
    assert {"system_prompt", "user_prompt", "max_new_tokens", "device"}.issubset(param_names)

    assert classify_task_mode("Return only Python inside ```python```", "Write shortest_path(n, edges, src, dst)") == "CODE_PYTHON"
    assert classify_task_mode("Output valid Mermaid", "Draw a flowchart LR with Ingress and Cache") == "DIAGRAM_MERMAID"
    assert classify_task_mode('Return JSON `{"order": list, "makespan": int}`', "Schedule DAG dependencies and makespan") == "SYMBOLIC_COMPUTE"

    ok_bad, err_bad = verify_python_smoke_zero_oracle(
        "```python\ndef f(n: int, edges: list[tuple[int, int, int]], src: int, dst: int) -> int:\n"
        "    pq = [(0, src)]\n    return heapq.heappop(pq)[0]\n```"
    )
    assert not ok_bad and "heapq" in err_bad

    ok_good, _ = verify_python_smoke_zero_oracle(
        "```python\nimport heapq\ndef f(n: int, edges: list[tuple[int, int, int]], src: int, dst: int) -> int:\n"
        "    pq = [(0, src)]\n    return heapq.heappop(pq)[0]\n```"
    )
    assert ok_good


def test_omega_experiments_smoke_cpu() -> None:
    from nanolm_holonomy.omega_experiments import (
        run_exp_omega1_s4_holonomy,
        run_exp_omega2_nullspace_continual,
    )

    cpu = torch.device("cpu")
    r1 = run_exp_omega1_s4_holonomy(device=cpu, seeds=(101,), steps=2, batch_size=4, train_len=8)
    assert len(r1["arms"]) == 5
    r2 = run_exp_omega2_nullspace_continual(device=cpu, seeds=(42,), pretrain_steps=1, adapt_steps=1, batch_size=4)
    assert len(r2["arms"]) == 3


def test_omega_schema_preserving_guard() -> None:
    assert classify_task_mode('Return JSON `{"calls": [...]}`.', "Available tools: balance_get,subtotal=5") == "STRUCTURED_JSON"
    assert not preserve_valid_json_keys('{"order":["A","B"]}', {"order": [{"name": "A"}, {}]})
    existing = '{"sum_total": 109, "label": "ok"}'
    assert preserve_valid_json_keys(existing, {"sum_total": 100, "label": "ok"})
    assert not preserve_valid_json_keys(existing, {"result": 100})
    assert preserve_valid_json_keys("invalid {json", {"sum_total": 100})


def test_gen4_config_budget_and_incremental_decode() -> None:
    from nanolm_holonomy.architecture import (
        HSPA_OMEGA_2B_CONFIG,
        HSPA_OMEGA_4B_CONFIG,
        TRANSFORMER_4B_CONFIG,
        HSPAArchConfig,
        HSPAFamily,
        HSPAUnifiedLM,
    )
    from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS

    assert 1_700_000_000 <= HSPA_OMEGA_2B_CONFIG.parameter_count <= 2_100_000_000
    assert 3_600_000_000 <= HSPA_OMEGA_4B_CONFIG.parameter_count <= ABSOLUTE_MAX_PARAMETERS
    assert abs(HSPA_OMEGA_4B_CONFIG.parameter_count - TRANSFORMER_4B_CONFIG.parameter_count) / TRANSFORMER_4B_CONFIG.parameter_count < 0.01

    cfg = HSPAArchConfig(
        name="test-gen4",
        family=HSPAFamily.GEN4_OMEGA_SO3_PREDCODE,
        vocab_size=128,
        d_model=64,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        d_ff=128,
        n_registers=4,
        d_fiber=32,
        max_seq_len=128,
    )
    m = HSPAUnifiedLM(cfg)
    assert cfg.parameter_count == m.num_parameters
    x = torch.randint(0, 128, (2, 16))
    lg = m(x)
    lg.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)

    m.eval()
    with torch.no_grad():
        ref = m(x)
        _, st = m(x[:, :8], use_state=True)
        incs = []
        for t in range(8, 16):
            lt, st = m(x[:, t : t + 1], past_states=st, use_state=True, past_len=t)
            incs.append(lt)
        assert (ref[:, 8:] - torch.cat(incs, dim=1)).abs().max().item() < 1e-4


def test_nullspace_lora_merge_unmerge_exact_equivalence() -> None:
    torch.manual_seed(29)
    lin = nn.Linear(64, 48, bias=False)
    w_orig = lin.weight.detach().clone()
    toy = nn.Sequential()
    toy.add_module("q_proj", lin)
    apply_nullspace_lora(toy, LoRAConfig(rank=8, alpha=16.0), nullspace_k=12)
    wrapped = toy.q_proj
    assert isinstance(wrapped, SpectralNullspaceLoRALinear)
    with torch.no_grad():
        nn.init.normal_(wrapped.lora_b, std=0.05)

    x = torch.randn(3, 10, 64)
    with torch.no_grad():
        y_unmerged = toy(x).clone()

    with merged_omega_inference(toy) as n_merged:
        assert n_merged == 1
        assert wrapped.merged is True
        with torch.no_grad():
            y_merged = toy(x).clone()
        assert torch.allclose(y_unmerged, y_merged, atol=1e-5, rtol=1e-5)

    assert wrapped.merged is False
    assert torch.allclose(wrapped.base.weight, w_orig, atol=1e-6)
    assert merge_omega_weights(toy) == 1
    assert unmerge_omega_weights(toy) == 1


def test_static_graph_hspa_decoder_greedy_parity() -> None:
    from nanolm_holonomy.architecture import HSPAArchConfig, HSPAFamily, HSPAUnifiedLM
    from nanolm_holonomy.cuda_graph_decode import StaticGraphHSPADecoder

    torch.manual_seed(33)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fam in (HSPAFamily.GEN3_H4_HSPA_WAVEPRED, HSPAFamily.GEN4_OMEGA_SO3_PREDCODE):
        cfg = HSPAArchConfig(
            name=f"graph-{fam.value}",
            family=fam,
            vocab_size=96,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_head=16,
            d_ff=128,
            n_registers=4,
            d_fiber=32,
            max_seq_len=128,
        )
        m = HSPAUnifiedLM(cfg).to(dev).eval()
        prompt = torch.randint(0, 96, (1, 14), device=dev)

        with torch.no_grad():
            lg_p, st_p = m(prompt, use_state=True, logits_to_keep=1)
            cur = lg_p[:, -1:, :].argmax(dim=-1)
            ref_toks = [cur]
            for s_i in range(6):
                lg_s, st_p = m(cur, past_states=st_p, use_state=True, past_len=14 + s_i, logits_to_keep=1)
                cur = lg_s[:, -1:, :].argmax(dim=-1)
                ref_toks.append(cur)
            ref_seq = torch.cat(ref_toks[:-1], dim=1)  # 6 generated tokens starting with first post-prompt

        dec = StaticGraphHSPADecoder(m, batch_size=1, device=dev)
        dec.capture_graph(warmup_steps=2)
        gen_seq = dec.generate_greedy(prompt, max_new_tokens=6, chunk_size=8)
        assert torch.equal(ref_seq, gen_seq)


def test_omega_controller_v2_guards_mermaid_and_probes() -> None:
    assert classify_task_mode("Return JSON.", "Init user Alice role viewer, update quota 35") == "STATE_JSON"
    assert classify_task_mode("Return JSON.", "Compute subtotal, discount, tax, grand_total from 5 items") == "SYMBOLIC_COMPUTE"

    ok1, _ = preserve_valid_json_structure('{"items": 5, "cost": 115}', {"items": 5, "cost": 150})
    assert ok1
    ok2, r2 = preserve_valid_json_structure('{"items": 5, "cost": 115}', {"items": [{"a": 1}], "cost": 150})
    assert not ok2 and "type_mismatch" in r2

    ok3, r3 = preserve_valid_json_structure(
        '{"order": ["T1", "T2", "T3", "T4", "T5"], "makespan": 18}',
        {"order": ["T1", "T2", "T4", "T5"], "makespan": 17},
    )
    assert not ok3 and "item_set_mismatch" in r3

    ok4, _ = preserve_valid_json_structure(
        '{"order": ["T1", "T2", "T3", "T4", "T5"], "makespan": 18}',
        {"order": ["T1", "T3", "T2", "T4", "T5"], "makespan": 13},
    )
    assert ok4

    chained = '```mermaid\nflowchart LR\n    Client["Browser (UI)"] --> Gateway["API Gateway"] --> Worker["Job Runner"]\n    Gateway --> Alert["Alert"]\n```'
    normed = normalize_chained_mermaid_edges(chained)
    assert normed.count("-->") == 3 and len(normed.splitlines()) == 6

    ok_none, err_none = verify_python_smoke_zero_oracle("```python\ndef f(values: list[int], k: int) -> int:\n    pass\n```")
    assert not ok_none and ("None" in err_none or "int" in err_none)

    ok_sent, err_sent = verify_python_smoke_zero_oracle("```python\ndef kth_smallest_bst(values: list[int], k: int) -> int:\n    return -1\n```")
    assert not ok_sent and "SentinelError" in err_sent


def test_cycle3_omega_v2_planner_and_cpu_smoke() -> None:
    from nanolm_holonomy.architecture import HSPAArchConfig, HSPAFamily
    from nanolm_holonomy.cycle3_omega_v2 import (
        run_exp_omega4_full_lm_s4,
        run_exp_omega6_cudagraph_and_merge_speedup,
    )

    plan_v2 = plan_omega_v2_campaign(allow_kaggle=True)
    rec = plan_v2["selection"]["recommended"]
    assert rec["backend"] == "kaggle"
    assert rec["estimated_usd"] == 0.0
    assert rec["fits_vram"] is True

    cpu = torch.device("cpu")
    r4 = run_exp_omega4_full_lm_s4(
        device=cpu,
        seeds=(101,),
        steps=2,
        batch_size=4,
        train_len=8,
        lengths=(8, 16),
        probe_batch=8,
    )
    assert len(r4["arms"]) == 4

    tiny_cfg = HSPAArchConfig(
        name="tiny-omega6",
        family=HSPAFamily.GEN4_OMEGA_SO3_PREDCODE,
        vocab_size=64,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        d_ff=128,
        n_registers=4,
        d_fiber=32,
        max_seq_len=128,
    )
    r6 = run_exp_omega6_cudagraph_and_merge_speedup(
        device=cpu,
        seq_len=16,
        n_decode=4,
        configs=(tiny_cfg,),
        merge_shapes=((64, 64),),
    )
    assert r6["decode_benchmarks"][0]["greedy_parity"] is True
    assert all(row["nullspace_orthogonality_error"] < 1e-5 for row in r6["nullspace_merge_benchmarks"])


def test_omega_wrapped_layer_gradient_checkpointing() -> None:
    from types import SimpleNamespace
    from torch.utils.checkpoint import checkpoint
    from nanolm_holonomy.omega_architecture import _OmegaWrappedLayer, apply_omega_graft

    class _Block(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + self.o_proj(torch.relu(self.q_proj(hidden_states)))

    class _ToyLM(nn.Module):
        def __init__(self, d: int = 32) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=d)
            self.layers = nn.ModuleList(_Block(d) for _ in range(2))

    torch.manual_seed(41)
    m = _ToyLM(d=32)
    apply_omega_graft(
        m,
        lora_config=LoRAConfig(rank=4, alpha=8.0, target_suffixes=("q_proj", "o_proj")),
        nullspace_k=4,
        n_sidecar_layers=2,
        n_registers=2,
        d_fiber=16,
    )
    m.train()
    x = torch.randn(2, 10, 32, requires_grad=True)
    h = x
    for layer in m.layers:
        assert isinstance(layer, _OmegaWrappedLayer)
        h_base = checkpoint(layer.base_layer, h, use_reentrant=False)
        h = h_base + layer.sidecar(h_base)
    h.sum().backward()
    trainable = [p for p in m.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)

def test_omega_cache_tracks_independent_requests_and_continues_each_stream():
    from nanolm_holonomy.omega_architecture import HSPAOmegaSidecar, _OmegaWrappedLayer

    class Cache:
        length = 0

        def get_seq_length(self, layer):
            assert layer == 2
            return self.length

    class Layer(nn.Module):
        def forward(self, h, past_key_values=None, use_cache=True):
            if use_cache and past_key_values is not None:
                past_key_values.length += h.shape[1]
            return (h, "base metadata")

    torch.manual_seed(921)
    sc = HSPAOmegaSidecar(16, n_registers=2, d_fiber=8)
    wrapper = _OmegaWrappedLayer(Layer(), sc, 2).eval()
    with torch.inference_mode():
        sc.sidecar_gate.fill_(0.7)
        a, b = torch.randn(2, 1, 6, 16)
        full_a = wrapper(a)[0]
        full_b = wrapper(b)[0]
        fresh, metadata = wrapper(a[:, -1:])
        assert metadata == "base metadata"
        ca, cb = Cache(), Cache()
        wrapper(a[:, :2], past_key_values=ca)
        wrapper(b[:, :4], past_key_values=cb)
        torch.testing.assert_close(wrapper(a[:, 2:5], past_key_values=ca)[0], full_a[:, 2:5])
        torch.testing.assert_close(wrapper(b[:, 4:], past_key_values=cb)[0], full_b[:, 4:])
        torch.testing.assert_close(wrapper(a[:, -1:])[0], fresh)
        torch.testing.assert_close(wrapper(a[:, -1:], past_key_values=ca, use_cache=False)[0], fresh)
        torch.testing.assert_close(wrapper(a[:, -1:], past_key_values=ca)[0], full_a[:, -1:])
        ca.length = 0
        torch.testing.assert_close(wrapper(a[:, -1:], past_key_values=ca)[0], fresh)
        sc.reset_cache()
        torch.testing.assert_close(wrapper(a[:, -1:], past_key_values=cb)[0], fresh)
