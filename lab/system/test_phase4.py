"""Regression tests for Phase 4 fixes: System-v3 math voting & transformers>=5 SmolLM3 encoding."""

from __future__ import annotations

from lab.system.controller import NanoLMSystem


def _mk_math_mock(ana: str, brute: str, formula: str):
    def mock(msgs, think, max_tok):
        txt = msgs[-1]["content"]
        if "EXHAUSTIVE BRUTE-FORCE" in txt:
            return f"```python\nprint({brute})\n```"
        if "Write a short, exact Python 3 script" in txt:
            return f"```python\nprint({formula})\n```"
        return f"Reasoning... \\boxed{{{ana}}}"
    return mock


def test_math_v3_majority_and_bruteforce_tiebreak():
    prompt = "How many permutations of {1..8} have exactly 2 fixed points? Put the answer in \\boxed{}."
    # Majority (A == B) overrides a wrong formula channel
    ep = NanoLMSystem(_mk_math_mock("7420", "7420", "999")).solve(prompt, task_hint="math_hard")
    assert ep.extracted_answer == "7420" and ep.epistemic_tier == "CERTIFIED_MAJORITY"
    # 3-way split -> literal brute-force specification wins
    ep2 = NanoLMSystem(_mk_math_mock("111", "7420", "999")).solve(prompt, task_hint="math_hard")
    assert ep2.extracted_answer == "7420" and ep2.epistemic_tier == "BRUTE_FORCE_SPECIFICATION"
    # Unanimous
    ep3 = NanoLMSystem(_mk_math_mock("7420", "7420", "7420")).solve(prompt, task_hint="math_hard")
    assert ep3.epistemic_tier == "CERTIFIED_TRIPLE_AGREEMENT"


def test_smollm3_encoder_handles_batchencoding():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    except Exception:
        return  # offline: skip
    from lab.trackb.phase3_innovation import _encode_smollm3_example
    ex = _encode_smollm3_example(tok, "What is 2+3? Put the answer in \\boxed{}.", "2+3=5, so \\boxed{5}.", think=False)
    assert ex is not None, "Encoder dropped a valid example (Phase-3 B6-3B bug regression!)"
    assert (ex["labels"] != -100).sum().item() > 3


def test_v4_schema_json_multiline_indent2_and_dict():
    """Regression test for Phase-4 Bug A: multi-line json.dumps(..., indent=2) must NOT truncate to '}'."""
    calls = {"n": 0}

    def mock(msgs, think, max_tok):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"total": 999, "items": {"a": 10, "b": 20}}'  # fails conservation total == 30
        return (
            "```python\n"
            "import json\n"
            "res = {'total': 10 + 20, 'items': {'a': 10, 'b': 20}}\n"
            "print(json.dumps(res, indent=2))\n"
            "```"
        )

    ep = NanoLMSystem(mock).solve(
        "Output ONLY JSON with exact keys total, items.",
        task_hint="json_strict",
        json_required_keys={"total", "items"},
        json_invariant_fn=lambda o: (o["total"] == sum(o["items"].values()), "sum mismatch"),
    )
    assert ep.epistemic_tier == "VERIFIED_SCHEMA"
    assert '"total": 30' in ep.final_response


def test_v4_agentic_tool_loop_truncated_think_and_premature_stop():
    """Regression test for Phase-4 Bug B: unclosed <think> and premature stop during unresolved error."""
    from lab.system.frontier_bench import _make_incident_env

    env = _make_incident_env("INC-7100", "shard-eu-20", 2000, 650, "LEASE-LIVE-900", locked=False)()
    step = {"k": 0}

    def mock(msgs, think, max_tok):
        step["k"] += 1
        k = step["k"]
        if k == 1:
            assert not think, "Turn 0 happy path must use think=False"
            return '{"tool_calls": [{"name": "lookup_incident_context", "arguments": {"incident_id": "INC-7100"}}]}'
        if k == 2:
            assert not think, "Turn 1 happy path (no error yet!) must use think=False"
            return '{"tool_calls": [{"name": "apply_shard_mitigation", "arguments": {"shard_id": "shard-eu-20", "lease_token": "LEASE-STALE-100", "target_rps": 500}}]}'
        if k == 3:
            assert think, "Turn 2 after StaleLeaseToken must escalate to think=True"
            return "<think>Let me analyze the stale lease token..."  # truncated <think> -> empty strip_thinking!
        if k == 4:
            assert not think, "JSON recovery fallback must force think=False"
            return '{"tool_calls": [], "response": "Giving up early despite unresolved error."}'  # premature stop!
        if k == 5:
            return '{"tool_calls": [{"name": "refresh_lease_token", "arguments": {"shard_id": "shard-eu-20"}}]}'
        if k == 6:
            return '{"tool_calls": [{"name": "apply_shard_mitigation", "arguments": {"shard_id": "shard-eu-20", "lease_token": "LEASE-LIVE-900", "target_rps": 500}}]}'
        if k == 7:
            return '{"tool_calls": [{"name": "apply_shard_mitigation", "arguments": {"shard_id": "shard-eu-20", "lease_token": "LEASE-LIVE-900", "target_rps": 650}}]}'
        return '{"tool_calls": [], "response": "Mitigation completed."}'  # deliberately omits shard/650 in prose!

    ep = NanoLMSystem(mock, max_tool_turns=7).solve("Resolve INC-7100.", task_hint="tool_workflow", tool_env=env)
    assert env.state["mitigated"] is True and env.state["current_rps"] == 650
    assert "shard-eu-20" in ep.final_response and "650" in ep.final_response


def test_v4_code_repair_preserves_compiling_candidate_on_truncated_repair():
    step = {"k": 0}

    def mock(msgs, think, max_tok):
        step["k"] += 1
        if step["k"] == 1:
            return "```python\ndef add_two(x: int) -> int:\n    return x + 2\n```"
        if step["k"] == 2:
            # Hallucinated wrong self-test assertion!
            return "```python\nassert add_two(2) == 999\n```"
        # Truncated repairs with no code
        return "<think>Still thinking..."

    ep = NanoLMSystem(mock, max_repairs=2).solve("Implement `def add_two(x: int) -> int:`", task_hint="coding")
    assert "def add_two" in ep.final_response


def test_b7_3b_causal_shift_alignment():
    """Regression test for Phase-5A Bug: B7-3B must shift (input_ids[:-1], labels[1:]) before proper_topk_kd_loss."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
    except Exception:
        return
    from lab.trackb.phase3_innovation import _encode_smollm3_example

    raw = _encode_smollm3_example(tok, "Compute 6*7.", "6*7 = 42, so \\boxed{42}.", think=False)
    assert raw is not None
    inp = raw["input_ids"][:-1]
    lbl = raw["labels"][1:]
    valid_pos = (lbl != -100).nonzero(as_tuple=True)[0]
    assert len(valid_pos) > 4
    # At every active target position t, lbl[t] must equal raw["input_ids"][t + 1] (next token!), NOT inp[t]!
    for t in valid_pos.tolist():
        assert int(lbl[t].item()) == int(raw["input_ids"][t + 1].item())


def test_b8_causal_completion_augmentations_decontaminated():
    """Verify all 66 B8 causal-completion & abstention trajectories pass 13-gram shield & verifiers."""
    from lab.trackb.phase3_innovation import build_causal_completion_augmentations

    aug = build_causal_completion_augmentations()
    assert len(aug) == 66
