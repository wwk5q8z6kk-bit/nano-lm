"""Unit and integration tests for the NanoLM 2B-4B foundation model stack."""

from __future__ import annotations

import pytest
import torch

from nanolm_posttrain import (
    ABSOLUTE_MAX_PARAMETERS,
    FoundationDomain,
    LoRAConfig,
    NANO_2B_CONFIG,
    NANO_350M_CONFIG,
    NANO_4B_CONFIG,
    NANO_MICRO_CONFIG,
    NanoLM,
    NanoLMConfig,
    PREFERRED_TARGET_PARAMETERS,
    ParameterBudgetExceededError,
    apply_lora,
    build_foundation_eval_suite,
    build_foundation_train_curriculum,
    lora_state_dict,
    trainable_parameter_summary,
    verify_foundation_response,
)
from nanolm_posttrain.lora import train_masked_sft_step


def test_parameter_budgets_enforce_2b_target_and_4b_hard_ceiling():
    assert NANO_MICRO_CONFIG.parameter_count < 50_000_000
    assert 300_000_000 < NANO_350M_CONFIG.parameter_count < 400_000_000
    assert 1_800_000_000 < NANO_2B_CONFIG.parameter_count <= PREFERRED_TARGET_PARAMETERS
    assert 3_500_000_000 < NANO_4B_CONFIG.parameter_count <= ABSOLUTE_MAX_PARAMETERS
    assert NANO_2B_CONFIG.effective_depth == 54
    assert NANO_4B_CONFIG.effective_depth == 72

    with pytest.raises(ParameterBudgetExceededError, match="4B"):
        NanoLMConfig(
            name="oversized-5b",
            vocab_size=128_000,
            d_model=3_072,
            n_layers=44,
            n_heads=24,
            n_kv_heads=8,
            d_head=128,
            d_ff=8_192,
        )


def test_nanolm_analytical_parameter_count_matches_instantiated_module_and_loops():
    tiny_cfg = NanoLMConfig(
        name="nanolm-tiny-test",
        vocab_size=256,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        d_ff=128,
        max_seq_len=128,
        thought_loops=3,
    )
    model = NanoLM(tiny_cfg)
    assert model.num_parameters == tiny_cfg.parameter_count

    tokens = torch.randint(0, 256, (2, 16))
    # At init, loop_step_emb == 0 and loop_gates == 0, so 1 loop and 3 loops match identically!
    out_1 = model(tokens, active_thought_loops=1)
    out_3 = model(tokens, active_thought_loops=3)
    assert out_1.shape == (2, 16, 256)
    assert torch.allclose(out_1, out_3, atol=1e-5)

    # Once loop gate is non-zero, recurrent reasoning loop modulates hidden state.
    with torch.no_grad():
        model.loop_gates.fill_(0.5)
        model.loop_step_emb[1].fill_(0.2)
    out_modulated = model(tokens, active_thought_loops=3)
    assert not torch.allclose(out_1, out_modulated)


def test_foundation_eval_and_curriculum_suites_are_disjoint_and_verifiers_sound():
    eval_suite = build_foundation_eval_suite()
    train_suite = build_foundation_train_curriculum()
    assert len(eval_suite) == 12
    assert {t.domain for t in eval_suite} == set(FoundationDomain)
    assert {t.task_id for t in eval_suite}.isdisjoint({t.task_id for t in train_suite})
    assert {t.user_prompt for t in eval_suite}.isdisjoint({t.user_prompt for t in train_suite})

    # Negative controls: broken code, unquoted Mermaid parens, hallucinated summary number, wrong chart total.
    code_task = next(t for t in eval_suite if t.task_id == "eval-code-01")
    assert not verify_foundation_response(code_task, "def merge_intervals(x): return []").passed

    diag_task = next(t for t in eval_suite if t.task_id == "eval-diag-01")
    unquoted_mermaid = "```mermaid\nflowchart LR\n  A[Lexer] --> B[Parser (AST)] --> C[Optimizer (SSA)] --> D[Codegen]\n```"
    assert not verify_foundation_response(diag_task, unquoted_mermaid).passed
    quoted_mermaid = (
        "```mermaid\nflowchart LR\n"
        "  A[\"Lexer\"] --> B[\"Parser (AST)\"]\n"
        "  B --> C[\"Optimizer (SSA)\"]\n"
        "  C --> D[\"Codegen\"]\n```"
    )
    assert verify_foundation_response(diag_task, quoted_mermaid).passed

    sum_task = next(t for t in eval_suite if t.task_id == "eval-sum-01")
    hallucinated = "Helios LSM cut write amplification by 64% and read p99 to 6.2 ms with 99% less RAM and 12% CPU."
    assert not verify_foundation_response(sum_task, hallucinated).passed

    chart_task = next(t for t in eval_suite if t.task_id == "eval-chart-02")
    bad_total = '{"chart_type": "bar", "x_labels": ["Q1", "Q2", "Q3", "Q4"], "series": [120, 145, 110, 165], "total": 999}'
    assert not verify_foundation_response(chart_task, bad_total).passed


def test_pure_pytorch_lora_freezes_base_and_reduces_masked_sft_loss():
    torch.manual_seed(7)
    cfg = NanoLMConfig(
        name="nanolm-lora-test",
        vocab_size=128,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        d_head=16,
        d_ff=128,
        max_seq_len=64,
        thought_loops=2,
    )
    model = NanoLM(cfg)
    base_snapshot = model.tok_emb.weight.detach().clone()
    wrapped = apply_lora(model, LoRAConfig(rank=4, alpha=8.0))
    assert wrapped == 4 * 7  # 7 linear projections per transformer block
    summary = trainable_parameter_summary(model)
    assert 0 < summary["trainable_parameters"] < summary["total_parameters"] * 0.25

    input_ids = torch.tensor([[5, 11, 22, 33, 44, 55, 66]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, -100, 33, 44, 55, 66]], dtype=torch.long)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    initial_loss = train_masked_sft_step(model, opt, input_ids, labels)
    final_loss = initial_loss
    for _ in range(12):
        final_loss = train_masked_sft_step(model, opt, input_ids, labels)

    assert final_loss < initial_loss * 0.6
    assert torch.equal(model.tok_emb.weight, base_snapshot)
    assert len(lora_state_dict(model)) == wrapped * 2
