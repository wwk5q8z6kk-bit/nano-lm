"""Unit tests for Track B post-training: decontamination, verification, proper Top-K+Tail KD, rsLoRA, RLVR."""

import torch
import torch.nn as nn

from lab.trackb.eval_capabilities import build_math_logic_suite
from lab.trackb.posttrain_pipeline import (
    attach_rslora,
    collect_eval_shield_ngrams,
    extract_topk_teacher_targets,
    filter_verified_trajectories,
    grpo_verifiable_loss,
    proper_topk_kd_loss,
)


def test_decontamination_and_verifier_gate():
    shield = collect_eval_shield_ngrams()
    leaked_prompt = build_math_logic_suite()[0]["prompt"]
    cands = [
        # 1. Contaminated (exact eval prompt) -> must be dropped!
        {"domain": "math", "prompt": leaked_prompt, "response": "FINAL_ANSWER: 149", "verifier_spec": {"expected": "149"}},
        # 2. Uncontaminated but WRONG teacher answer -> must be rejected by verifier!
        {"domain": "math", "prompt": "What is 17 * 19?", "response": "FINAL_ANSWER: 999", "verifier_spec": {"expected": "323"}},
        # 3. Uncontaminated AND verified correct math -> must be kept!
        {"domain": "math", "prompt": "What is 17 * 19?", "response": "Step 1: 17*20 - 17 = 323.\nFINAL_ANSWER: 323", "verifier_spec": {"expected": "323"}},
        # 4. Uncontaminated AND verified Python code -> must be kept!
        {"domain": "code", "prompt": "Write add_one(x).", "response": "```python\ndef add_one(x):\n    return x + 1\n```", "verifier_spec": {"harness": "assert add_one(4) == 5"}},
        # 5. Verified tool call -> must be kept!
        {"domain": "tool", "prompt": "Scale web to 3.", "response": '{"tool_calls": [{"name": "scale", "arguments": {"n": 3}}]}',
         "verifier_spec": {"allowed_tools": ["scale"], "required_arg_keys": {"scale": ["n"]}}},
    ]
    kept, stats = filter_verified_trajectories(cands, shield)
    assert stats["contaminated_dropped"] == 1
    assert stats["verification_failed"] == 1
    assert stats["kept"] == 3 and len(kept) == 3


def test_proper_topk_kd_zero_at_optimum_and_strictly_positive_away():
    torch.manual_seed(0)
    teacher_logits = torch.randn(2, 5, 256)
    labels = torch.randint(0, 256, (2, 5))
    labels[0, 0] = -100
    idx, lp = extract_topk_teacher_targets(teacher_logits, k=16, temperature=1.5)

    # Matching logits -> KL must be ~0 (within fp16 quantization of cached teacher logprobs)
    same = proper_topk_kd_loss(teacher_logits.clone().requires_grad_(True), idx, lp, labels, temperature=1.5, alpha_kd=1.0)
    assert float(same["kd_kl"]) < 1e-4

    # Perturbed student logits -> KL must be strictly positive and differentiable
    student = (teacher_logits + torch.randn_like(teacher_logits)).requires_grad_(True)
    diff = proper_topk_kd_loss(student, idx, lp, labels, temperature=1.5, alpha_kd=0.7)
    assert float(diff["kd_kl"]) > 0.01
    diff["loss"].backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()


def test_rslora_and_grpo():
    class TinyHybrid(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkvz = nn.Linear(16, 32, bias=False)
            self.o_proj = nn.Linear(32, 16, bias=False)

        def forward(self, x):
            return self.o_proj(torch.relu(self.in_proj_qkvz(x)))

    m = TinyHybrid()
    orig_w = m.in_proj_qkvz.weight.clone()
    n_train = attach_rslora(m, rank=4, alpha=8.0)
    assert n_train == 4 * (16 + 32) + 4 * (32 + 16)
    x = torch.randn(2, 6, 16)
    y = m(x).sum()
    y.backward()
    # Base weights untouched; LoRA A/B receive gradients
    assert torch.equal(m.in_proj_qkvz.base.weight, orig_w)
    assert m.in_proj_qkvz.lora_A.grad is not None

    pol_lp = torch.randn(4, 8, requires_grad=True)
    ref_lp = pol_lp.detach() + 0.1
    mask = torch.ones(4, 8)
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    out = grpo_verifiable_loss(pol_lp, ref_lp, mask, rewards)
    out["loss"].backward()
    assert float(out["kl"]) >= 0.0 and pol_lp.grad is not None
