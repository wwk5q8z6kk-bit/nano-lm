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
