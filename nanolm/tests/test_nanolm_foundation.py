"""Comprehensive verification suite for the independent NanoLM 2B-4B foundation package."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from nanolm.config import (
    AttentionType,
    FFNType,
    HARD_MAX_PARAMETERS,
    NanoLMConfig,
    ParameterBudgetExceededError,
    get_preset_config,
    list_presets,
)
from nanolm.data import PackedDocumentDataset, decontaminate_corpus
from nanolm.eval import (
    collect_all_eval_prompts,
    run_capability_evaluation,
    verify_chart_spec,
    verify_math_or_logic,
    verify_mermaid_flowchart,
    verify_python_solution,
    verify_structured_json,
    verify_summary_faithfulness,
)
from nanolm.model import NanoLMForCausalLM
from nanolm.tokenizer import audit_tokenizer_efficiency, train_nanolm_tokenizer
from nanolm.train import TrainConfig, run_training


def test_hard_4b_parameter_ceiling_enforced() -> None:
    with pytest.raises(ParameterBudgetExceededError):
        NanoLMConfig(
            name="illegal-4.3b",
            tier="production",
            vocab_size=65_536,
            hidden_size=3072,
            num_layers=40,
            num_attention_heads=24,
            num_kv_heads=8,
            head_dim=128,
            intermediate_size=8192,
        )

    # Also verify deploy_mtp=True counts toward the 4B hard ceiling!
    with pytest.raises(ParameterBudgetExceededError):
        NanoLMConfig(
            name="illegal-moe-plus-deployed-mtp",
            tier="production",
            vocab_size=65_536,
            hidden_size=2048,
            num_layers=22,
            num_attention_heads=16,
            num_kv_heads=4,
            head_dim=128,
            intermediate_size=6912,
            ffn_type=FFNType.MOE_SWIGLU,
            num_experts=16,
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_expert_intermediate_size=1536,
            mtp_num_heads=2,
            deploy_mtp=True,
        )


def test_all_presets_exact_parameter_parity() -> None:
    for preset_name in list_presets():
        cfg = get_preset_config(preset_name)
        acct = cfg.parameter_accounting()
        assert acct.total_deployed_parameters <= HARD_MAX_PARAMETERS
        with torch.device("meta"):
            model = NanoLMForCausalLM(cfg)
        assert model.count_deployed_parameters() == acct.total_deployed_parameters
        assert model.count_total_parameters() == (
            acct.total_deployed_parameters + acct.training_only_parameters
        )


@pytest.mark.parametrize(
    "attn_type,ffn_type,loop_repeats",
    [
        (AttentionType.GQA, FFNType.SWIGLU, 1),
        (AttentionType.HYBRID_SWA, FFNType.SWIGLU, 1),
        (AttentionType.MLA, FFNType.SWIGLU, 1),
        (AttentionType.HYBRID_SWA, FFNType.MOE_SWIGLU, 1),
        (AttentionType.HYBRID_SWA, FFNType.SWIGLU, 2),
    ],
)
def test_kv_cache_matches_full_forward_across_architectures(
    attn_type: AttentionType,
    ffn_type: FFNType,
    loop_repeats: int,
) -> None:
    torch.manual_seed(7)
    cfg = NanoLMConfig(
        name="tiny-kv-parity",
        tier="research_rung",
        vocab_size=512,
        hidden_size=64,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        sliding_window_size=6,
        global_attn_every_n_layers=2,
        kv_lora_rank=32,
        q_lora_rank=48,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        attention_type=attn_type,
        ffn_type=ffn_type,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_expert_intermediate_size=64,
        loop_start_layer=1 if loop_repeats > 1 else 0,
        loop_end_layer=3 if loop_repeats > 1 else 0,
        loop_repeats=loop_repeats,
        mtp_num_heads=1,
        deploy_mtp=False,
    )
    model = NanoLMForCausalLM(cfg).eval()
    x = torch.randint(0, 512, (2, 10))
    full_logits = model(input_ids=x, use_cache=False).logits

    # Prefill 6 tokens, then decode 4 tokens one-by-one using KV cache
    prefill = model(input_ids=x[:, :6], use_cache=True)
    kv = prefill.past_key_values
    step_logits = [prefill.logits]
    for pos in range(6, 10):
        step_out = model(input_ids=x[:, pos : pos + 1], past_key_values=kv, use_cache=True)
        kv = step_out.past_key_values
        step_logits.append(step_out.logits)
    cached_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(full_logits, cached_logits, atol=1e-4, rtol=1e-4)


def test_packed_document_isolation_zero_cross_doc_leakage() -> None:
    cfg = NanoLMConfig(
        name="tiny-pack",
        tier="research_rung",
        vocab_size=512,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
    )
    model = NanoLMForCausalLM(cfg).eval()
    docs = [[10, 11, 12, 13], [20, 21, 22, 23, 24], [30, 31, 32, 33]]
    ds = PackedDocumentDataset(docs, seq_len=12, bos_id=0, eos_id=1)
    batch = ds.sample_batch(batch_size=1, step=0, seed=1)

    # Verify cross-doc target transition is masked with -100
    assert (batch.targets == -100).any()

    # Mutate tokens belonging exclusively to the first document in the block;
    # logits of subsequent documents MUST remain bit-identical!
    doc0_id = int(batch.doc_ids[0, 0].item())
    later_mask = batch.doc_ids[0] != doc0_id
    assert bool(later_mask.any())

    logits_a = model(
        input_ids=batch.input_ids,
        position_ids=batch.position_ids,
        doc_ids=batch.doc_ids,
    ).logits
    mutated_ids = batch.input_ids.clone()
    doc0_positions = torch.where(~later_mask)[0]
    mutated_ids[0, doc0_positions] = 99
    logits_b = model(
        input_ids=mutated_ids,
        position_ids=batch.position_ids,
        doc_ids=batch.doc_ids,
    ).logits
    assert torch.allclose(logits_a[0, later_mask], logits_b[0, later_mask], atol=1e-5)


def test_decontamination_removes_13gram_and_exact_leaks() -> None:
    eval_prompts = collect_all_eval_prompts()
    leaked_doc = "Intro note: " + eval_prompts[0] + " Trailing remark."
    clean_doc = "Completely independent mathematical derivation of Fourier series convergence on L2."
    kept, report = decontaminate_corpus([clean_doc, clean_doc, leaked_doc], eval_prompts, ngram_size=13)
    assert kept == [clean_doc]
    assert report.removed_exact_dups == 1
    assert report.removed_eval_contaminated == 1


def test_tokenizer_digit_splitting_and_roundtrip() -> None:
    corpus = [
        "def fib(n: int) -> int:\n    return n if n <= 1 else fib(n - 1) + fib(n - 2)\n",
        "Theorem: For x = 12345, y = 67890, x + y = 80235.",
        '{"cluster": "alpha", "latency_ms": 42.5}',
    ] * 40
    tok = train_nanolm_tokenizer(corpus, vocab_size=600, min_frequency=2, split_digits=True)
    rep = audit_tokenizer_efficiency(
        "test-600",
        lambda s: tok.encode(s).ids,
        tok.get_vocab_size(),
        {"code": corpus[:5]},
    )
    assert rep.individual_digit_splitting is True
    sample = "def calc(): return 2048"
    assert tok.decode(tok.encode(sample).ids) == sample


def test_checkpoint_save_and_resume_continuity(tmp_path: Path) -> None:
    cfg = NanoLMConfig(
        name="tiny-train",
        tier="research_rung",
        vocab_size=512,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        mtp_num_heads=1,
    )
    docs = [[(i * 7 + j) % 480 + 10 for j in range(20)] for i in range(60)]
    ds = PackedDocumentDataset(docs, seq_len=16)

    tcfg1 = TrainConfig(
        run_id="smoke-part1",
        max_steps=4,
        micro_batch_size=4,
        grad_accum_steps=1,
        warmup_steps=2,
        eval_every=2,
        save_every=2,
        precision="fp32",
        checkpoint_dir=str(tmp_path / "run"),
    )
    res1 = run_training(cfg, tcfg1, ds, ds, val_bytes_per_token=3.5, device="cpu")
    assert res1.steps_completed == 4
    assert Path(res1.final_checkpoint_path).is_file()

    tcfg2 = TrainConfig(
        run_id="smoke-part2",
        max_steps=8,
        micro_batch_size=4,
        grad_accum_steps=1,
        warmup_steps=2,
        eval_every=2,
        save_every=4,
        precision="fp32",
        checkpoint_dir=str(tmp_path / "run"),
    )
    res2 = run_training(
        cfg,
        tcfg2,
        ds,
        ds,
        val_bytes_per_token=3.5,
        device="cpu",
        resume_from=res1.final_checkpoint_path,
    )
    assert res2.steps_completed == 8
    assert res2.final_val_bpb < res1.initial_val_bpb


def test_programmatic_capability_validators() -> None:
    ok, _ = verify_python_solution(
        "```python\ndef add(a, b):\n    return a + b\n```",
        "assert add(2, 3) == 5",
    )
    assert ok is True
    bad, _ = verify_python_solution("```python\ndef add(a, b): return a - b\n```", "assert add(2, 3) == 5")
    assert bad is False

    assert verify_math_or_logic("<|think|>14*12-19=149<|end_think|>\nFINAL_ANSWER: 149", "149") is True
    s_res = verify_summary_faithfulness(
        "Enabling zero-copy in velox_stream reduced RAM from 64 GB to 38 GB, except on arm64-debug nodes.",
        required_facts=["velox_stream", "64", "38"],
        required_qualifications=["except", "arm64-debug"],
        forbidden_claims=["windows"],
    )
    assert s_res["passed"] is True

    v_syn, v_sem, _ = verify_structured_json(
        '{"cluster":"zephyr","nodes":[{"host":"n1","cores":16},{"host":"n2","cores":32}],"total_cores":48}',
        {"cluster": str, "nodes": list, "total_cores": int},
        lambda o: o["total_cores"] == 48,
    )
    assert v_syn and v_sem

    m_syn, m_sem, _ = verify_mermaid_flowchart(
        '```mermaid\nflowchart LR\n    A["api_ingress"] -->|"authenticated"| B["core_engine"]\n    A -->|"rejected"| C["audit_sink"]\n```',
        [("api_ingress", "core_engine", "authenticated"), ("api_ingress", "audit_sink", "rejected")],
    )
    assert m_syn and m_sem

    c_syn, c_sem, _ = verify_chart_spec(
        '{"chart_type":"line","x_label":"Quarter","y_label":"Req/s","x_values":["Q1","Q2"],"y_values":[120,155]}',
        {"line", "area"},
        ["Q1", "Q2"],
        [120.0, 155.0],
    )
    assert c_syn and c_sem

    empty_eval = run_capability_evaluation("null_baseline", lambda _: "")
    assert empty_eval.overall_pass_rate == 0.0
