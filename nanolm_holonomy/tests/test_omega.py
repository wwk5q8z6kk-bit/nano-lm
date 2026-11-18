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
    select_optimal_gpu,
)
from nanolm_holonomy.omega_architecture import (
    SpectralNullspaceLoRALinear,
    apply_nullspace_lora,
    so3_projective_adjoint,
    su2_lie_exponential,
)
from nanolm_holonomy.omega_system import (
    classify_task_mode,
    omega_generate,
    preserve_valid_json_keys,
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
    existing = '{"sum_total": 109, "label": "ok"}'
    assert preserve_valid_json_keys(existing, {"sum_total": 100, "label": "ok"})
    assert not preserve_valid_json_keys(existing, {"result": 100})
    assert preserve_valid_json_keys("invalid {json", {"sum_total": 100})
