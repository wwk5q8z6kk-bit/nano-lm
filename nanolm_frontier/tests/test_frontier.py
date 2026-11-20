"""Unit, mathematical-equivalence, and fail-closed verifier tests for nanolm_frontier."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch
from torch import nn

from nanolm_frontier.architectures import (
    ABSOLUTE_MAX_PARAMETERS,
    FRONTIER_2B_CONFIG,
    FRONTIER_4B_CONFIG,
    ArchitectureFamily,
    FrontierArchConfig,
    FrontierCausalLM,
    ParameterBudgetExceededError,
    build_micro_tournament_configs,
    chunked_ema_key_energy,
    chunkwise_gated_delta_rule,
    reference_recurrent_gated_delta_rule,
)
from nanolm_frontier.benchmark_15d import (
    CAPABILITY_DIMENSIONS,
    BenchmarkDim,
    BenchmarkTask,
    bootstrap_accuracy_ci,
    build_benchmark_15d_suite,
    compute_efficiency_score,
    verify_benchmark_response,
)
from nanolm_frontier.curriculum import (
    ContaminationError,
    CurriculumExample,
    FrontierMicroTokenizer,
    build_decontaminated_curriculum,
    build_ood_recall_probes,
    build_pretrain_tournament_datasets,
    verify_zero_contamination,
)
from nanolm_frontier.surgery_and_rlvr import (
    DexDiffAttentionWrapper,
    ThalamocorticalVAnchor,
    _bypass_adapters,
    apply_frontier_surgery,
)
from nanolm_frontier.tournament import run_e1_architecture_tournament
from nanolm_posttrain.lora import LoRAConfig


def _tiny_cfg(fam: ArchitectureFamily) -> FrontierArchConfig:
    return FrontierArchConfig(
        name=f"tiny-{fam.value}",
        family=fam,
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        d_ff=128,
        max_seq_len=128,
        delta_chunk_size=8,
        pkm_side=8,
        pkm_top_k=4,
        pkm_d_val=16,
        max_loops=4,
        active_loops=2,
    )


def test_five_architectures_forward_backward_and_budget() -> None:
    torch.manual_seed(7)
    x = torch.randint(0, 256, (2, 24))
    for fam in ArchitectureFamily:
        cfg = _tiny_cfg(fam)
        model = FrontierCausalLM(cfg)
        assert cfg.parameter_count == model.num_parameters
        logits = model(x)
        assert logits.shape == (2, 24, 256)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if "lambda_k" in name:
                assert p.grad is not None and p.grad.abs().max() > 0
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)

    # Check 2B and 4B scale configs strictly obey <= 4B ceiling
    assert 1_800_000_000 <= FRONTIER_2B_CONFIG.parameter_count <= 2_200_000_000
    assert 3_500_000_000 <= FRONTIER_4B_CONFIG.parameter_count <= ABSOLUTE_MAX_PARAMETERS

    # Exceeding 4B must fail loudly
    with pytest.raises(ParameterBudgetExceededError):
        replace(FRONTIER_4B_CONFIG, n_layers=64)


def test_iso_param_tournament_configs_and_kv_reduction() -> None:
    cfgs = build_micro_tournament_configs()
    counts = [c.parameter_count for c in cfgs]
    # All 5 architectures stay within 1% parameter spread
    assert (max(counts) - min(counts)) / min(counts) < 0.015
    kv_c0 = cfgs[0].kv_cache_bytes_per_token()
    for c in cfgs[1:]:
        assert c.kv_cache_bytes_per_token() == kv_c0 // 4


def test_chunkwise_delta_and_ema_key_energy_match_reference() -> None:
    torch.manual_seed(11)
    q = torch.randn(2, 3, 40, 16)
    k = torch.nn.functional.normalize(torch.randn(2, 3, 40, 16), dim=-1)
    v = torch.randn(2, 3, 40, 20)
    a = torch.sigmoid(torch.randn(2, 3, 40) + 1.2)
    b = 2.0 * torch.sigmoid(torch.randn(2, 3, 40))

    ref = reference_recurrent_gated_delta_rule(q, k, v, a, b)
    chk = chunkwise_gated_delta_rule(q, k, v, torch.log(a), b, chunk_size=12)
    assert (ref - chk).abs().max().item() < 1e-4

    # Check EMA key energy against exact token-by-token formula
    gamma = torch.tensor([0.85, 0.90, 0.95])
    chk_ema = chunked_ema_key_energy(k, gamma, chunk_size=12, eps=1e-4)
    c = torch.full((2, 3, 16), 1.0 / 16.0)
    ref_steps = []
    for t in range(40):
        g = gamma.view(1, 3, 1)
        c = g * c + (1.0 - g) * (k[:, :, t, :] ** 2)
        ref_steps.append(torch.sqrt(c + 1e-4))
    ref_ema = torch.stack(ref_steps, dim=2)
    assert (chk_ema - ref_ema).abs().max().item() < 1e-5


def test_jcle_depth_scaling_and_entropy_early_exit() -> None:
    cfg = replace(_tiny_cfg(ArchitectureFamily.C4_JCLE_FRONTIER), entropy_halt_threshold=100.0)
    model = FrontierCausalLM(cfg)
    x = torch.randint(0, 256, (2, 16))
    logits_r1, st1 = model(x, active_loops=1, return_loop_stats=True)
    logits_r4, st4 = model(x, active_loops=4, return_loop_stats=True)
    assert logits_r1.shape == logits_r4.shape == (2, 16, 256)
    assert st1["mean_loops"] == 1.0 and st4["mean_loops"] == 4.0

    # With huge threshold, every token halts at loop 1
    logits_halt, st_halt = model(x, active_loops=4, adaptive_entropy_halt=True, return_loop_stats=True)
    assert logits_halt.shape == (2, 16, 256)
    assert st_halt["early_exit_rate"] == 1.0
    assert st_halt["mean_loops"] == 1.0


class _DummySelfAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.up_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.down_proj = nn.Linear(d_model * 2, d_model, bias=False)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = self.o_proj(q + k + v)
        mlp = self.down_proj(torch.nn.functional.silu(self.gate_proj(attn)) * self.up_proj(attn))
        return x + mlp


class _DummyDecoderModel(nn.Module):
    def __init__(self, vocab: int = 128, d_model: int = 32, n_heads: int = 4, n_layers: int = 4) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.config = SimpleNamespace(
            num_attention_heads=n_heads,
            head_dim=d_model // n_heads,
            hidden_size=d_model,
        )
        self.emb = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(nn.Module() for _ in range(n_layers))
        for layer in self.layers:
            layer.self_attn = _DummySelfAttn(d_model, n_heads)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.emb(input_ids)
        for layer in self.layers:
            h = layer.self_attn(h)
        return self.head(h)


def test_frontier_surgery_exact_step0_equivalence_and_bypass() -> None:
    torch.manual_seed(3)
    model = _DummyDecoderModel()
    x = torch.randint(0, 128, (2, 10))
    with torch.no_grad():
        base_logits = model(x).clone()

    info = apply_frontier_surgery(model, lora_config=LoRAConfig(rank=4, alpha=8.0))
    assert info["dex_diff_layers"] == 4
    assert info["v_anchor_layers"] == 2
    assert any(isinstance(m, DexDiffAttentionWrapper) for m in model.modules())
    assert any(isinstance(m, ThalamocorticalVAnchor) for m in model.modules())

    with torch.no_grad():
        graft_logits = model(x)
    assert (base_logits - graft_logits).abs().max().item() < 1e-6

    # Take a training step modifying adapters + gates; verify bypass still recovers exact base_logits
    loss = model(x).sum()
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.add_(-0.05 * p.grad)
        mutated_logits = model(x)
    assert (base_logits - mutated_logits).abs().max().item() > 1e-3

    with _bypass_adapters(model), torch.no_grad():
        recovered_logits = model(x)
    assert (base_logits - recovered_logits).abs().max().item() < 1e-6


def test_all_15_verifiers_positive_and_negative_controls() -> None:
    tasks = {t.task_id: t for t in build_benchmark_15d_suite()}
    assert len(tasks) == 56
    assert len({t.dim for t in tasks.values()}) == len(CAPABILITY_DIMENSIONS) == 14

    # Positive controls spanning capability dims
    assert verify_benchmark_response(
        tasks["cod-01"],
        "```python\ndef run_length_encode(s):\n    if not s: return []\n    out=[]; cur=s[0]; cnt=0\n    for c in s:\n        if c==cur: cnt+=1\n        else: out.append((cur,cnt)); cur=c; cnt=1\n    out.append((cur,cnt)); return out\n```",
    ).passed
    assert not verify_benchmark_response(tasks["cod-01"], "def bad(:\n pass").passed

    assert verify_benchmark_response(tasks["mth-03"], "Remainder solving gives x = 38.\nFINAL_ANSWER: 38").passed
    assert not verify_benchmark_response(tasks["mth-03"], "FINAL_ANSWER: 39").passed

    assert verify_benchmark_response(tasks["dia-03"], '```mermaid\nflowchart LR\n  A["Browser (UI)"] --> B["Gateway"]\n  B --> C["Job Runner (Async)"]\n  C --> D["Alert"]\n```').passed
    assert not verify_benchmark_response(tasks["dia-03"], "```mermaid\nflowchart LR\n  A[Browser (UI)] --> B[Gateway]\n  B --> C[Job Runner]\n  C --> D[Alert]\n```").passed

    assert verify_benchmark_response(tasks["pln-02"], json.dumps({"order": ["Spec", "Backend", "Frontend", "Deploy"], "makespan": 9})).passed
    assert not verify_benchmark_response(tasks["pln-02"], json.dumps({"order": ["Deploy", "Spec", "Backend", "Frontend"], "makespan": 9})).passed

    assert verify_benchmark_response(tasks["ood-02"], "Compute 3*4=7 then 2+7=14.\nFINAL_ANSWER: 14").passed
    assert not verify_benchmark_response(tasks["ood-02"], "FINAL_ANSWER: 14.0001").passed  # slight wrong check
    assert not verify_benchmark_response(tasks["ood-02"], "FINAL_ANSWER: 11").passed  # memorized precedence fails

    assert verify_benchmark_response(tasks["hal-01"], "ABSTAIN: Insufficient information because unit price is not in the invoice.").passed
    assert not verify_benchmark_response(tasks["hal-01"], "The unit price is $25.").passed

    assert verify_benchmark_response(tasks["tol-03"], json.dumps({"tool": "cache_lookup", "args": {"key": "cuda", "fallback": True}})).passed
    assert not verify_benchmark_response(tasks["tol-03"], json.dumps({"tool": "primary_search", "args": {"query": "cuda"}})).passed

    assert verify_benchmark_response(tasks["mem-03"], json.dumps({"A": 80, "B": 70})).passed
    assert not verify_benchmark_response(tasks["mem-03"], json.dumps({"A": 100, "B": 50})).passed

    # Dimension 15 inference_efficiency & bootstrap CI
    eff = compute_efficiency_score(
        capability_accuracy=0.80,
        total_parameters=1_750_000_000,
        tokens_per_sec=45.0,
        kv_cache_mb_at_4k=112.0,
    )
    assert 0.0 < eff["efficiency_composite_score"] <= 1.0
    assert eff["capability_per_billion_params"] > 0.4
    mean, lo, hi = bootstrap_accuracy_ci([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert 0.0 <= lo <= mean <= hi <= 1.0


def test_curriculum_decontamination_and_fail_closed_guard() -> None:
    eval_tasks = build_benchmark_15d_suite()
    cur = build_decontaminated_curriculum()
    res = verify_zero_contamination(cur, eval_tasks)
    assert res["overlaps"] == 0

    # Inject duplicate prompt from eval and ensure ContaminationError fires
    leaked = CurriculumExample(
        "leak-01",
        BenchmarkDim.CODING,
        "sys",
        eval_tasks[0].user_prompt,
        "resp",
        {},
    )
    with pytest.raises(ContaminationError):
        verify_zero_contamination([leaked], eval_tasks)


def test_local_e1_smoke_tournament() -> None:
    # Fast CPU smoke verify of end-to-end 5-arm tournament orchestration
    tiny_cfgs = tuple(replace(_tiny_cfg(fam), vocab_size=512, max_seq_len=512) for fam in ArchitectureFamily)
    res = run_e1_architecture_tournament(
        device=torch.device("cpu"),
        configs=tiny_cfgs,
        vocab_size=512,
        train_sequences=16,
        train_steps=2,
        batch_size=4,
        lr=3e-4,
        ood_lengths=(64, 128),
        ood_probes_per_length=4,
        max_val_batches=1,
    )
    assert len(res["arms"]) == 5
    c4_arm = res["arms"][-1]
    assert "jcle_test_time_scaling" in c4_arm
    assert len(c4_arm["jcle_test_time_scaling"]) == 4


def test_reviewed_verifier_boundaries() -> None:
    tasks = {task.task_id: task for task in build_benchmark_15d_suite()}

    def passes(task_id, text):
        return verify_benchmark_response(tasks[task_id], text).passed

    assert passes("dia-03", 'flowchart LR\nA["Browser"] --> B[Gateway] --> C["Job Runner"] --> Alert')
    assert not passes("dia-01", 'flowchart TD\nIngest -->\nValidate -->\nArchive -->')
    assert not passes("mth-03", "FINAL_ANSWER: 39, not 38")
    assert not passes("mth-03", "I reject 38")
    assert not passes("pln-02", json.dumps({"order": [{}, {}, {}, {}], "makespan": 9}))
    assert passes("hal-01", "ABSTAIN: $250 was for shipping.")
    assert passes("hal-04", "ABSTAIN: Warehouse East holds 4,200 pallets, West is unknown.")
    assert passes("hal-04", "ABSTAIN: The 4,200 pallets refer to Warehouse East; Warehouse West is not provided.")
    assert not passes("hal-04", "ABSTAIN: East is unknown and West held 4,200 pallets.")
    assert not passes("hal-04", "ABSTAIN: West held 4,200 pallets.")
    mean, lo, hi = bootstrap_accuracy_ci([0, 1] * 28)
    assert lo < mean < hi


def test_branch_topology_and_json_type_contracts() -> None:
    from nanolm_frontier.benchmark_15d import _verify_json_invariant

    tasks = {task.task_id: task for task in build_benchmark_15d_suite()}
    main = 'flowchart LR\nClient["Browser (UI)"] --> Gateway["API Gateway"] --> Worker["Job Runner (Async)"]\n'
    for edge in ("Gateway -->|error| Alert", "Worker --> Alert"):
        assert verify_benchmark_response(tasks["dia-03"], main + edge).passed
    for edge in ('Other["API Gateway"] --> Alert', "Alert --> Gateway", "Client --> Alert"):
        assert not verify_benchmark_response(tasks["dia-03"], main + edge).passed
    for task_id, key in (("jsn-01", "tls"), ("mem-01", "active")):
        task = tasks[task_id]
        expected = task.spec["expected_json"]
        assert verify_benchmark_response(task, json.dumps(expected)).passed
        assert not verify_benchmark_response(task, json.dumps({**expected, key: int(expected[key])})).passed
    spec = {"expected_json": {"values": [True, 0, 1.5]}}
    assert _verify_json_invariant('{"values": [true, 0, 1.5005]}', spec).passed
    for bad in ('{"values": [1, 0, 1.5]}', '{"values": [true, false, 1.5]}', '{"values": [true, 0, NaN]}'):
        assert not _verify_json_invariant(bad, spec).passed


def test_decoding_attention_flops_match_value_width_and_loop_count() -> None:
    for cfg in build_micro_tournament_configs():
        start, end = cfg.resolved_loop_span
        softmax_layers = sum(
            cfg.active_loops if cfg.uses_jcle and start <= i < end else 1
            for i in range(cfg.n_layers) if not cfg.is_delta_layer(i)
        )
        qk_width = cfg.n_heads * cfg.d_head
        value_width = cfg.d_head * (2 if cfg.uses_diff_and_whitening else 1)
        av_width = cfg.n_heads * value_width
        expected_delta = 2 * softmax_layers * (qk_width + av_width)
        assert cfg.active_flops_per_token(2048) - cfg.active_flops_per_token(2047) == expected_delta
