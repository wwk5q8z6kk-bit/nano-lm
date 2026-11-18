"""Function-Preserving Holonomic Spinor Sidecar & Curriculum-Fixed GRPO Evolution (Experiment X4).

Addresses the exact empirical failure modes uncovered in ``nanolm_frontier_v1``:
1. Over-training SFT (5 epochs -> 0.03 loss) collapsed Qwen3-1.7B Python coding from
   75% to 0% and starved Stage-2 GRPO of reward variance (71/72 zero-variance groups).
2. Replaces naive 5-epoch SFT with early-stopped 2-epoch SFT preserving coding skills,
   grafts a zero-initialized Non-Abelian SU(2) ``HolonomicRegisterSidecar`` onto upper
   decoder layers (exact 0.00e+00 step-0 equivalence), and feeds dynamically generated,
   strictly decontaminated multi-difficulty RLVR prompts into Stage-2 Verifier-Guided GRPO.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import gc
import json
import math
import random
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_frontier.benchmark_15d import (
    BenchmarkDim,
    BenchmarkTask,
    build_benchmark_15d_suite,
    verify_benchmark_response,
)
from nanolm_frontier.curriculum import (
    CurriculumExample,
    build_decontaminated_curriculum,
    verify_zero_contamination,
)
from nanolm_frontier.surgery_and_rlvr import (
    GRPOConfig,
    _bypass_adapters,
    _token_logprobs,
    apply_chat_template_safe,
    train_sft_curriculum,
)
from nanolm_frontier.tournament import _evaluate_model_15d, _load_hf_model_and_tokenizer
from nanolm_holonomy.spinor_algebra import (
    cayley_rotor,
    hamilton_product,
    spinor_conjugate,
    step_holonomy,
)
from nanolm_holonomy.triton_holonomy import holonomy_scan_fast
from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS, ParameterBudgetExceededError
from nanolm_posttrain.lora import LoRAConfig, apply_lora, lora_state_dict


class HolonomicRegisterSidecar(nn.Module):
    """Lightweight Non-Abelian SU(2) spinor register sidecar added to decoder layer residual.

    Zero-initialized output gate ``sidecar_gate = 0`` guarantees exact step-0
    bit-identity with the base pretrained model while giving upper Transformer layers
    a persistent non-abelian quaternionic state track.
    """

    def __init__(self, d_model: int, *, n_registers: int = 4, d_fiber: int = 128) -> None:
        super().__init__()
        if d_fiber % 4 != 0:
            raise ValueError("d_fiber must be divisible by 4")
        self.R = n_registers
        self.G = d_fiber // 4
        self.d_fiber = d_fiber
        self.bypass = False
        self._cached_scan_state: Tensor | None = None

        self.in_norm = nn.RMSNorm(d_model)
        self.phi_proj = nn.Linear(d_model, d_fiber, bias=False)
        self.gate_proj = nn.Linear(d_model, 3 * self.R, bias=False)  # write, beta, read
        self.spinor_proj = nn.Linear(d_model, 9 * self.G, bias=False)  # omega, rho, q
        self.o_proj = nn.Linear(d_fiber, d_model, bias=False)
        self.sidecar_gate = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

    def reset_cache(self) -> None:
        self._cached_scan_state = None

    def forward(self, h: Tensor) -> Tensor:
        if self.bypass:
            return torch.zeros_like(h)

        bsz, seq_len, _ = h.shape
        x = self.in_norm(h.float())
        phi = self.phi_proj(x).view(bsz, seq_len, self.G, 4).permute(0, 2, 1, 3).unsqueeze(1)
        w_l, b_l, r_l = self.gate_proj(x).chunk(3, dim=-1)
        w = F.softmax(w_l, dim=-1).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        beta = torch.sigmoid(b_l).transpose(1, 2).unsqueeze(2).unsqueeze(-1)
        read_pi = F.softmax(r_l, dim=-1).transpose(1, 2).unsqueeze(-1)

        o_gen, rho_gen, q_gen = self.spinor_proj(x).chunk(3, dim=-1)
        o_gen = o_gen.view(bsz, seq_len, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1)
        rho_gen = rho_gen.view(bsz, seq_len, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1)
        q_gen = q_gen.view(bsz, seq_len, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1)

        g = (1.0 - w * beta).clamp_min(1e-4) * cayley_rotor(o_gen)
        s = w * hamilton_product(cayley_rotor(rho_gen), phi)

        if (
            not self.training
            and seq_len == 1
            and self._cached_scan_state is not None
            and self._cached_scan_state.shape[0] == bsz
        ):
            nxt = step_holonomy(g.squeeze(-2), s.squeeze(-2), self._cached_scan_state)
            self._cached_scan_state = nxt.detach()
            all_omega = nxt.unsqueeze(-2)
        else:
            all_omega, nxt = holonomy_scan_fast(g, s, init_state=None, chunk_size=256)
            if not self.training:
                self._cached_scan_state = nxt.detach()

        unbound = hamilton_product(spinor_conjugate(cayley_rotor(q_gen)), all_omega)
        flat = unbound.permute(0, 1, 3, 2, 4).reshape(bsz, self.R, seq_len, self.d_fiber)
        out = self.o_proj((read_pi * flat).sum(dim=1))
        return (torch.tanh(self.sidecar_gate) * out).to(h.dtype)


class _SidecarWrappedDecoderLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, sidecar: HolonomicRegisterSidecar) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.sidecar = sidecar

    def forward(self, hidden_states: Tensor, *args: Any, **kwargs: Any) -> Any:
        out = self.base_layer(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            h_out = out[0]
            return (h_out + self.sidecar(h_out), *out[1:])
        return out + self.sidecar(out)


def apply_holonomic_graft(
    model: nn.Module,
    *,
    lora_config: LoRAConfig = LoRAConfig(rank=16, alpha=32.0),
    n_sidecar_layers: int = 8,
    n_registers: int = 4,
    d_fiber: int = 128,
) -> dict[str, Any]:
    """Attach LoRA + zero-init SU(2) HolonomicRegisterSidecar on upper decoder layers."""
    total_before = sum(p.numel() for p in model.parameters())
    if total_before > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(f"base model {total_before:,} > 4B limit")

    lora_n = apply_lora(model, lora_config)
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else getattr(model, "layers", None)
    d_model = int(getattr(model.config, "hidden_size", 0))
    ref_param = next(model.parameters())

    sidecars = 0
    if layers is not None and d_model > 0:
        n_layers = len(layers)
        start_idx = max(0, n_layers - n_sidecar_layers)
        for idx in range(start_idx, n_layers):
            sc = HolonomicRegisterSidecar(d_model, n_registers=n_registers, d_fiber=d_fiber).to(ref_param.device)
            layers[idx] = _SidecarWrappedDecoderLayer(layers[idx], sc)
            sidecars += 1

    total_after = sum(p.numel() for p in model.parameters())
    if total_after > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(f"grafted model {total_after:,} > 4B limit")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "lora_layers": lora_n,
        "holonomic_sidecar_layers": sidecars,
        "total_parameters": total_after,
        "trainable_parameters": trainable,
        "trainable_pct": round(100.0 * trainable / max(total_after, 1), 4),
    }


@contextmanager
def _bypass_all_holonomy_adapters(model: nn.Module) -> Iterator[None]:
    saved_sidecars: list[tuple[HolonomicRegisterSidecar, bool]] = []
    for mod in model.modules():
        if isinstance(mod, HolonomicRegisterSidecar):
            saved_sidecars.append((mod, mod.bypass))
            mod.bypass = True
    with _bypass_adapters(model):
        try:
            yield
        finally:
            for mod, flg in saved_sidecars:
                mod.bypass = flg


def build_challenging_rlvr_curriculum(seed: int = 777) -> tuple[CurriculumExample, ...]:
    """Generate parameterized, harder multi-step RLVR tasks engineered to produce nonzero GRPO variance.

    Strictly decontaminated against ``benchmark_15d`` via explicit check before use.
    """
    rng = random.Random(seed)
    items: list[CurriculumExample] = []

    # Harder Chinese Remainder Theorem / modular congruence problems
    for i in range(6):
        m1 = rng.choice([5, 7, 9, 11])
        m2 = rng.choice([8, 13, 15, 17])
        if math.gcd(m1, m2) != 1:
            m2 = 13 if m1 != 13 else 17
        target_x = rng.randint(25, m1 * m2 - 1)
        r1 = target_x % m1
        r2 = target_x % m2
        smallest = target_x
        for cand in range(1, m1 * m2 + 1):
            if cand % m1 == r1 and cand % m2 == r2:
                smallest = cand
                break
        items.append(
            CurriculumExample(
                f"rl-crt-{i}",
                BenchmarkDim.MATH_LOGIC,
                "Solve step by step and end with `FINAL_ANSWER: <integer>`.",
                f"Find the smallest positive integer `z` such that `z mod {m1} = {r1}` and `z mod {m2} = {r2}`.",
                f"Testing multiples gives {smallest}.\nFINAL_ANSWER: {smallest}",
                {"accepted_answers": [str(smallest)]},
            )
        )

    # Multi-branch DAG critical-path problems (specifically weak in E2: Base=0/4, LoRA_SFT=0/4!)
    for i in range(6):
        dA = rng.randint(2, 6)
        dB = rng.randint(3, 8)
        dC = rng.randint(2, 7)
        dD = rng.randint(2, 6)
        dE = rng.randint(1, 5)
        fA = dA
        fB = fA + dB
        fC = fA + dC
        fD = max(fB, fC) + dD
        makespan = max(fB, fD) + dE
        dag = {
            "Alpha": {"dur": dA, "deps": []},
            "Beta": {"dur": dB, "deps": ["Alpha"]},
            "Gamma": {"dur": dC, "deps": ["Alpha"]},
            "Delta": {"dur": dD, "deps": ["Beta", "Gamma"]},
            "Omega": {"dur": dE, "deps": ["Beta", "Delta"]},
        }
        items.append(
            CurriculumExample(
                f"rl-dag-{i}",
                BenchmarkDim.PLANNING_DECOMPOSITION,
                'Return only valid JSON `{"order": list[str], "makespan": int}`.',
                f"Schedule DAG: `Alpha` (dur {dA}, deps []), `Beta` (dur {dB}, deps [Alpha]), "
                f"`Gamma` (dur {dC}, deps [Alpha]), `Delta` (dur {dD}, deps [Beta, Gamma]), "
                f"`Omega` (dur {dE}, deps [Beta, Delta]). Return valid execution order and parallel makespan.",
                json.dumps({"order": ["Alpha", "Beta", "Gamma", "Delta", "Omega"], "makespan": makespan}),
                {"dag": dag},
            )
        )

    # Multi-turn stateful register/account transfers (weak in E2: 1/4 across all arms!)
    for i in range(6):
        a = rng.randint(40, 120)
        b = rng.randint(30, 90)
        t1 = rng.randint(10, 25)
        d1 = rng.randint(5, 20)
        t2 = rng.randint(5, 15)
        w1 = rng.randint(5, 15)
        # Execute exact sequence
        cur_a, cur_b = a, b
        cur_a -= t1
        cur_b += t1
        cur_a += d1
        cur_b -= t2
        cur_a += t2
        cur_b -= w1
        cur_a *= 2
        items.append(
            CurriculumExample(
                f"rl-mem-{i}",
                BenchmarkDim.MEMORY_RETENTION,
                "Return only valid JSON.",
                "\n".join([
                    f"S1: INIT X={a}, Y={b}",
                    f"S2: MOVE X->Y {t1}",
                    f"S3: ADD X {d1}",
                    f"S4: MOVE Y->X {t2}",
                    f"S5: SUB Y {w1}",
                    "S6: DOUBLE X",
                    'S7: Output JSON `{"X": int, "Y": int}`.',
                ]),
                json.dumps({"X": cur_a, "Y": cur_b}),
                {"expected_json": {"X": cur_a, "Y": cur_b}},
            )
        )

    # Counterfactual OOD rules + Coding problems
    for i in range(4):
        x, y, z = rng.randint(3, 9), rng.randint(2, 8), rng.randint(2, 7)
        val = (y + z) * x  # reversed precedence
        items.append(
            CurriculumExample(
                f"rl-ood-{i}",
                BenchmarkDim.OOD_ROBUSTNESS,
                "Apply the counterfactual operator rule strictly and end with `FINAL_ANSWER: <int>`.",
                f"In Dialectic-Q math, `~` means integer multiplication and `^` means integer addition with `^` evaluated BEFORE `~`. Evaluate `{x} ~ {y} ^ {z}`.",
                f"Evaluate {y} ^ {z} = {y + z}, then {x} ~ {y + z} = {val}.\nFINAL_ANSWER: {val}",
                {"accepted_answers": [str(val)]},
            )
        )

    items.append(
        CurriculumExample(
            "rl-cod-0",
            BenchmarkDim.CODING,
            "Return only valid Python inside a ```python``` block. No prose.",
            "Write `max_subarray_sum(nums: list[int]) -> int` returning maximum contiguous non-empty subarray sum (0 if list is empty).",
            "```python\ndef max_subarray_sum(nums: list[int]) -> int:\n    if not nums:\n        return 0\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best\n```",
            {"assertions": [
                "assert max_subarray_sum([]) == 0",
                "assert max_subarray_sum([-2,1,-3,4,-1,2,1,-5,4]) == 6",
                "assert max_subarray_sum([-5, -2, -8]) == -2",
            ]},
        )
    )
    items.append(
        CurriculumExample(
            "rl-cod-1",
            BenchmarkDim.CODING,
            "Return only valid Python inside a ```python``` block. No prose.",
            "Write `min_coins(coins: list[int], amount: int) -> int` returning fewest coins to make `amount`, or `-1` if impossible.",
            "```python\ndef min_coins(coins: list[int], amount: int) -> int:\n    dp = [amount + 1] * (amount + 1)\n    dp[0] = 0\n    for a in range(1, amount + 1):\n        for c in coins:\n            if c <= a:\n                dp[a] = min(dp[a], dp[a - c] + 1)\n    return dp[amount] if dp[amount] <= amount else -1\n```",
            {"assertions": [
                "assert min_coins([1, 2, 5], 11) == 3",
                "assert min_coins([2], 3) == -1",
                "assert min_coins([1], 0) == 0",
            ]},
        )
    )
    return tuple(items)


def _grpo_with_holonomy_ref(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[CurriculumExample],
    *,
    device: torch.device,
    config: GRPOConfig = GRPOConfig(rounds=3, group_size=4, max_new_tokens=224, lr=4e-5, kl_beta=0.03),
) -> dict[str, Any]:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    total_updates = 0
    skipped_zero_var = 0
    round_rewards: list[float] = []
    mean_kls: list[float] = []

    for _ in range(config.rounds):
        r_vals: list[float] = []
        for ex in prompts:
            msgs = [{"role": "system", "content": ex.system_prompt}, {"role": "user", "content": ex.user_prompt}]
            ptxt = apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True)
            pids = tokenizer(ptxt, add_special_tokens=False)["input_ids"]
            plen = len(pids)
            pt = torch.tensor([pids], dtype=torch.long, device=device)

            model.eval()
            rollouts: list[Tensor] = []
            rewards: list[float] = []
            tview = BenchmarkTask(ex.example_id, ex.dim, "hard", ex.system_prompt, ex.user_prompt, ex.verifier_spec)
            with torch.no_grad():
                bp = pt.repeat(config.group_size, 1)
                seqs = model.generate(
                    input_ids=bp,
                    attention_mask=torch.ones_like(bp),
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=0.95,
                    pad_token_id=pad_id,
                )
                for gi in range(config.group_size):
                    cids = seqs[gi, plen:].tolist()
                    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in cids:
                        cids = cids[: cids.index(tokenizer.eos_token_id) + 1]
                    if not cids:
                        rewards.append(0.0)
                        rollouts.append(seqs[gi : gi + 1, : plen + 1])
                        continue
                    txt = tokenizer.decode(cids, skip_special_tokens=True)
                    v = verify_benchmark_response(tview, txt)
                    rewards.append(1.0 if v.passed else 0.35 * float(v.score))
                    rollouts.append(torch.tensor([pids + cids], dtype=torch.long, device=device))

            mu = sum(rewards) / len(rewards)
            r_vals.append(mu)
            std = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards))
            if std < 1e-4:
                skipped_zero_var += 1
                continue

            model.train()
            optimizer.zero_grad(set_to_none=True)
            for gi, fg in enumerate(rollouts):
                if fg.shape[1] <= plen:
                    continue
                adv = max(-2.5, min(2.5, (rewards[gi] - mu) / (std + 1e-5)))
                out_pi = model(input_ids=fg)
                logits_pi = out_pi.logits if hasattr(out_pi, "logits") else out_pi
                logp_pi = _token_logprobs(logits_pi, fg, plen)
                with torch.no_grad(), _bypass_all_holonomy_adapters(model):
                    out_ref = model(input_ids=fg)
                    logits_ref = out_ref.logits if hasattr(out_ref, "logits") else out_ref
                    logp_ref = _token_logprobs(logits_ref, fg, plen)
                lr_t = (logp_ref - logp_pi).clamp(-10.0, 10.0)
                kl = torch.exp(lr_t) - lr_t - 1.0
                loss_g = (-adv * logp_pi.mean() + config.kl_beta * kl.mean()) / float(len(rollouts))
                loss_g.backward()
                mean_kls.append(float(kl.mean().detach().cpu().item()))
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            total_updates += 1
        round_rewards.append(round(sum(r_vals) / max(len(r_vals), 1), 4))

    model.eval()
    return {
        "rounds": config.rounds,
        "total_prompt_groups": config.rounds * len(prompts),
        "gradient_updates": total_updates,
        "skipped_zero_variance_groups": skipped_zero_var,
        "round_mean_rewards": round_rewards,
        "mean_kl": round(sum(mean_kls) / max(len(mean_kls), 1), 6),
    }


def run_x4_foundation_holonomy_evolution(
    *,
    model_id: str = "Qwen/Qwen3-1.7B",
    device: torch.device,
) -> dict[str, Any]:
    """Run Experiment X4 validating Holonomic Sidecar + Early-Stop SFT + High-Variance RLVR."""
    eval_tasks = build_benchmark_15d_suite()
    sft_curriculum = build_decontaminated_curriculum(seed=2026)
    rl_curriculum = build_challenging_rlvr_curriculum(seed=777)

    dec1 = verify_zero_contamination(sft_curriculum, eval_tasks)
    dec2 = verify_zero_contamination(rl_curriculum, eval_tasks)

    model, tok = _load_hf_model_and_tokenizer(model_id, device)
    sample = tok("def dijkstra(n, edges, src, dst):\n    import", return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        pre_logits = model(**sample).logits.float().clone()

    graft_info = apply_holonomic_graft(
        model,
        lora_config=LoRAConfig(rank=16, alpha=32.0),
        n_sidecar_layers=8,
        n_registers=4,
        d_fiber=128,
    )
    model.eval()
    with torch.no_grad():
        post_logits = model(**sample).logits.float()
    step0_diff = float((pre_logits - post_logits).abs().max().item())
    if step0_diff > 1e-4:
        raise RuntimeError(f"Holonomic graft step-0 equivalence broken: {step0_diff}")
    graft_info["step0_max_abs_logit_diff"] = step0_diff

    # Early-stopped 2-epoch SFT (vs 5-epoch overfit in frontier_v1)
    sft_stats = train_sft_curriculum(model, tok, sft_curriculum, device=device, epochs=2, lr=1.2e-4)
    # High-variance parameterized RLVR GRPO
    grpo_stats = _grpo_with_holonomy_ref(model, tok, list(rl_curriculum), device=device)

    gate_norms = [
        float(m.sidecar_gate.detach().abs().mean().cpu().item())
        for m in model.modules()
        if isinstance(m, HolonomicRegisterSidecar)
    ]

    arm_res = _evaluate_model_15d(model, tok, eval_tasks, device, arm_name="Qwen3_1.7B_Holonomic_SFT2_GRPO")
    arm_res["training_stats"] = {
        "graft": graft_info,
        "sft_early_stopped": sft_stats,
        "grpo_challenging": grpo_stats,
        "mean_abs_sidecar_gate": round(sum(gate_norms) / max(len(gate_norms), 1), 6),
        "trainable_tensor_count": len(lora_state_dict(model)),
    }
    print(
        f"[X4] Holonomic_SFT2_GRPO passed={arm_res['passed']}/{arm_res['total_tasks']} "
        f"acc={arm_res['accuracy']:.4f} grpo_updates={grpo_stats['gradient_updates']} "
        f"coding={arm_res['by_dimension']['coding']['passed']}/4 "
        f"planning={arm_res['by_dimension']['planning_decomposition']['passed']}/4",
        flush=True,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": "X4_holonomic_sidecar_and_grpo_evolution",
        "model_id": model_id,
        "decontamination": {"sft": dec1, "rl_challenging": dec2},
        "arm": arm_res,
    }
