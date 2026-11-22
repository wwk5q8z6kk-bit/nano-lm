"""Function-Preserving Architecture Surgery & Verifier-Guided GRPO Post-Trainer.

Implements:
1. ``DexDiffAttentionWrapper`` — Zero-initialized cross-head differential noise
   cancellation over GQA attention outputs (``tanh(gamma) == 0`` at step 0 -> exact
   ``KL(base || grafted) == 0.0`` bit-level preservation).
2. ``ThalamocorticalVAnchor`` — Zero-initialized upper-layer Layer-0 Value Residual
   Anchor preserving lexical/key fidelity over long horizons.
3. ``grpo_verifier_posttrain`` — On-policy Verifier-Guided GRPO using zero-memory
   frozen-base reference logprobs (temporarily bypassing adapters) and deterministic
   per-task verifiers.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_frontier.benchmark_15d import BenchmarkTask, verify_benchmark_response
from nanolm_frontier.curriculum import CurriculumExample
from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS, ParameterBudgetExceededError
from nanolm_posttrain.lora import LoRAConfig, LoRALinear, apply_lora, build_chat_sft_tensors, train_masked_sft_step


class AnchorContext:
    """Shared per-forward holder carrying Layer-0 value features to upper layers."""

    def __init__(self) -> None:
        self.layer0_v: Tensor | None = None

    def clear(self) -> None:
        self.layer0_v = None


class DexDiffAttentionWrapper(nn.Module):
    """Wraps attention ``o_proj`` with zero-init cross-head differential noise cancellation.

    Splits multi-head inputs ``[B, T, n_heads * d_head]`` into adjacent head pairs
    ``(h_even, h_odd)``. Computes a paired contrast signal ``pair_proj(h_odd)`` modulated
    by zero-initialized ``tanh(gamma)`` subtracted from ``h_even`` prior to ``o_proj``.
    At initialization ``gamma == 0 => exact identity`` with zero logit perturbation.
    """

    def __init__(
        self,
        base_o_proj: nn.Module,
        *,
        n_heads: int,
        d_head: int,
    ) -> None:
        super().__init__()
        if n_heads < 2 or n_heads % 2 != 0:
            raise ValueError(f"DexDiffAttentionWrapper requires even n_heads >= 2, got {n_heads}")
        self.base_o_proj = base_o_proj
        self.n_heads = n_heads
        self.d_head = d_head
        self.n_pairs = n_heads // 2
        self.bypass = False

        ref_param = next(base_o_proj.parameters())
        device = ref_param.device
        # Keep adapter parameters in FP32 for stable mixed-precision optimizer steps
        self.gamma = nn.Parameter(torch.zeros((self.n_pairs, self.d_head), device=device, dtype=torch.float32))
        self.pair_proj = nn.Linear(self.d_head, self.d_head, bias=False, device=device, dtype=torch.float32)
        nn.init.normal_(self.pair_proj.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if self.bypass:
            return self.base_o_proj(x)
        # x shape: [..., n_heads * d_head]
        orig_shape = x.shape
        h = x.view(*orig_shape[:-1], self.n_pairs, 2, self.d_head)
        h_even = h[..., 0, :]
        h_odd = h[..., 1, :]
        scale = torch.tanh(self.gamma).view(*([1] * (x.ndim - 1)), self.n_pairs, self.d_head)
        corr = scale * self.pair_proj(h_odd.float())
        h_mod_even = h_even - corr.to(h.dtype)
        h_mod = torch.stack([h_mod_even, h_odd], dim=-2).reshape(*orig_shape)
        return self.base_o_proj(h_mod)


class Layer0ValueCapture(nn.Module):
    """Wraps Layer-0 ``v_proj`` cleanly to broadcast its output to ``AnchorContext``."""

    def __init__(self, base_v_proj: nn.Module, ctx: AnchorContext) -> None:
        super().__init__()
        self.base_v_proj = base_v_proj
        self.ctx = ctx

    def forward(self, x: Tensor) -> Tensor:
        v = self.base_v_proj(x)
        self.ctx.layer0_v = v
        return v


class ThalamocorticalVAnchor(nn.Module):
    """Wraps upper-layer ``v_proj`` with a zero-initialized residual anchor from Layer-0 values."""

    def __init__(self, base_v_proj: nn.Module, ctx: AnchorContext, *, out_features: int) -> None:
        super().__init__()
        self.base_v_proj = base_v_proj
        self.ctx = ctx
        self.bypass = False
        ref_param = next(base_v_proj.parameters())
        device = ref_param.device
        self.anchor_gate = nn.Parameter(torch.zeros((out_features,), device=device, dtype=torch.float32))
        self.anchor_proj = nn.Linear(out_features, out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.eye_(self.anchor_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        v = self.base_v_proj(x)
        if self.bypass or self.ctx.layer0_v is None:
            return v
        l0 = self.ctx.layer0_v
        if l0.shape != v.shape or l0.device != v.device:
            return v
        gate = torch.tanh(self.anchor_gate).to(v.dtype)
        anchored = self.anchor_proj(l0.float()).to(v.dtype)
        return v + gate * anchored


def _infer_attn_geometry(model: nn.Module) -> tuple[int, int]:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_heads", None)
        d_head = getattr(cfg, "head_dim", None) or getattr(cfg, "d_head", None)
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        if n_heads and d_head:
            return int(n_heads), int(d_head)
        if n_heads and hidden:
            return int(n_heads), int(hidden) // int(n_heads)
    raise ValueError("Cannot infer (n_heads, d_head) from model.config")


def _find_decoder_layers(model: nn.Module) -> list[nn.Module] | None:
    # Support HuggingFace Llama/Qwen/Mistral (`model.model.layers`) and native NanoLM (`model.blocks`)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return list(inner.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    if hasattr(model, "blocks"):
        return list(model.blocks)
    return None


def apply_frontier_surgery(
    model: nn.Module,
    *,
    lora_config: LoRAConfig = LoRAConfig(rank=16, alpha=32.0),
    enable_dex_diff: bool = True,
    enable_v_anchor: bool = True,
) -> dict[str, Any]:
    """Apply LoRA + DexDiffAttentionWrapper + ThalamocorticalVAnchor while preserving exact step-0 logits."""
    total_before = sum(p.numel() for p in model.parameters())
    if total_before > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(
            f"base model has {total_before:,} params > 4B limit"
        )

    lora_count = apply_lora(model, lora_config)
    layers = _find_decoder_layers(model)
    n_heads, d_head = _infer_attn_geometry(model)

    ctx = AnchorContext()
    diff_wrapped = 0
    v_anchored = 0

    if layers:
        n_layers = len(layers)
        for idx, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                continue
            if enable_dex_diff and hasattr(attn, "o_proj") and (n_heads % 2 == 0):
                attn.o_proj = DexDiffAttentionWrapper(attn.o_proj, n_heads=n_heads, d_head=d_head)
                diff_wrapped += 1
            if enable_v_anchor and hasattr(attn, "v_proj"):
                if idx == 0:
                    attn.v_proj = Layer0ValueCapture(attn.v_proj, ctx)
                elif idx >= n_layers // 2:
                    v_mod = attn.v_proj
                    out_feat = (
                        v_mod.base.out_features if isinstance(v_mod, LoRALinear) else v_mod.out_features
                    )
                    attn.v_proj = ThalamocorticalVAnchor(v_mod, ctx, out_features=out_feat)
                    v_anchored += 1

    total_after = sum(p.numel() for p in model.parameters())
    if total_after > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(
            f"grafted model has {total_after:,} params > 4B limit"
        )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "lora_layers": lora_count,
        "dex_diff_layers": diff_wrapped,
        "v_anchor_layers": v_anchored,
        "total_parameters": total_after,
        "trainable_parameters": trainable,
        "trainable_pct": round(100.0 * trainable / max(total_after, 1), 4),
    }


@contextmanager
def _bypass_adapters(model: nn.Module) -> Iterator[None]:
    """Temporarily disable LoRA/DexDiff/VAnchor adapters to evaluate the exact frozen base policy."""
    saved_scales: list[tuple[LoRALinear, float]] = []
    saved_diff: list[tuple[DexDiffAttentionWrapper, bool]] = []
    saved_anchor: list[tuple[ThalamocorticalVAnchor, bool]] = []
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            saved_scales.append((mod, mod.scaling))
            mod.scaling = 0.0
        elif isinstance(mod, DexDiffAttentionWrapper):
            saved_diff.append((mod, mod.bypass))
            mod.bypass = True
        elif isinstance(mod, ThalamocorticalVAnchor):
            saved_anchor.append((mod, mod.bypass))
            mod.bypass = True
    try:
        yield
    finally:
        for mod, sc in saved_scales:
            mod.scaling = sc
        for mod, flg in saved_diff:
            mod.bypass = flg
        for mod, flg in saved_anchor:
            mod.bypass = flg


def apply_chat_template_safe(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    """Apply tokenizer chat template cleanly, disabling Qwen3 CoT think tags when supported."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def build_sft_chat_tensors_safe(
    tokenizer: Any,
    examples: Sequence[tuple[str, str, str]],
    *,
    device: str | torch.device,
) -> list[tuple[Tensor, Tensor]]:
    """Build assistant-only masked SFT tensors using ``apply_chat_template_safe``."""
    if not hasattr(tokenizer, "apply_chat_template") or getattr(tokenizer, "chat_template", None) is None:
        return build_chat_sft_tensors(tokenizer, examples, device=device)

    batches: list[tuple[Tensor, Tensor]] = []
    for sys_p, usr_p, reply in examples:
        prompt_msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}]
        full_msgs = [*prompt_msgs, {"role": "assistant", "content": reply}]
        prompt_text = apply_chat_template_safe(tokenizer, prompt_msgs, add_generation_prompt=True)
        full_text = apply_chat_template_safe(tokenizer, full_msgs, add_generation_prompt=False)
        p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        if len(f_ids) <= len(p_ids) or f_ids[: len(p_ids)] != p_ids:
            eos_tok = getattr(tokenizer, "eos_token", "") or ""
            r_ids = tokenizer(reply + eos_tok, add_special_tokens=False)["input_ids"]
            f_ids = list(p_ids) + list(r_ids)
        labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
        batches.append(
            (
                torch.tensor([f_ids], dtype=torch.long, device=device),
                torch.tensor([labels], dtype=torch.long, device=device),
            )
        )
    return batches


def train_sft_curriculum(
    model: nn.Module,
    tokenizer: Any,
    curriculum: Sequence[CurriculumExample],
    *,
    device: str | torch.device,
    epochs: int = 4,
    lr: float = 2e-4,
    max_grad_norm: float = 1.0,
) -> dict[str, Any]:
    """Run clean prompt-masked SFT across the decontaminated curriculum with cosine decay."""
    triples = [(ex.system_prompt, ex.user_prompt, ex.reference_response) for ex in curriculum]
    batches = build_sft_chat_tensors_safe(tokenizer, triples, device=device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    total_steps = epochs * len(batches)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    epoch_losses: list[float] = []
    for _ in range(epochs):
        losses: list[float] = []
        for x, y in batches:
            loss = train_masked_sft_step(model, optimizer, x, y, max_grad_norm=max_grad_norm)
            scheduler.step()
            losses.append(loss)
        epoch_losses.append(round(sum(losses) / max(len(losses), 1), 5))
    return {"epochs": epochs, "steps": total_steps, "epoch_losses": epoch_losses}


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    rounds: int = 2
    group_size: int = 4
    temperature: float = 0.75
    max_new_tokens: int = 192
    lr: float = 4e-5
    kl_beta: float = 0.03
    max_grad_norm: float = 1.0


def _token_logprobs(logits: Tensor, full_ids: Tensor, prompt_len: int) -> Tensor:
    """Extract per-token log-probabilities for generated completion tokens ``[1, T_gen]``."""
    # Next-token prediction: logits[:, t] predicts full_ids[:, t + 1]
    gen_logits = logits[:, prompt_len - 1 : -1, :].float()
    gen_targets = full_ids[:, prompt_len:]
    log_probs = F.log_softmax(gen_logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=gen_targets.unsqueeze(-1)).squeeze(-1)


def grpo_verifier_posttrain(
    model: nn.Module,
    tokenizer: Any,
    curriculum: Sequence[CurriculumExample],
    *,
    device: str | torch.device,
    config: GRPOConfig = GRPOConfig(),
) -> dict[str, Any]:
    """On-policy Verifier-Guided GRPO focusing gradient steps strictly on mixed-reward groups."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    total_updates = 0
    skipped_zero_var = 0
    mean_rewards: list[float] = []
    mean_kls: list[float] = []

    for rnd in range(config.rounds):
        round_rewards: list[float] = []
        for ex in curriculum:
            prompt_msgs = [
                {"role": "system", "content": ex.system_prompt},
                {"role": "user", "content": ex.user_prompt},
            ]
            prompt_text = apply_chat_template_safe(tokenizer, prompt_msgs, add_generation_prompt=True)
            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)
            prompt_t = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # 1. Sample G on-policy rollouts cleanly under eval mode
            model.eval()
            rollouts: list[Tensor] = []
            rewards: list[float] = []
            task_view = BenchmarkTask(
                ex.example_id,
                ex.dim,
                "medium",
                ex.system_prompt,
                ex.user_prompt,
                ex.verifier_spec,
            )
            with torch.no_grad():
                batched_prompt = prompt_t.repeat(config.group_size, 1)
                seqs = model.generate(
                    input_ids=batched_prompt,
                    attention_mask=torch.ones_like(batched_prompt),
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=0.95,
                    pad_token_id=pad_id,
                )
                for gi in range(config.group_size):
                    seq_g = seqs[gi]
                    # Trim trailing pad tokens after first EOS if present
                    comp_ids = seq_g[prompt_len:].tolist()
                    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in comp_ids:
                        eos_pos = comp_ids.index(tokenizer.eos_token_id)
                        comp_ids = comp_ids[: eos_pos + 1]
                    if not comp_ids:
                        rewards.append(0.0)
                        rollouts.append(seq_g[: prompt_len + 1].unsqueeze(0))
                        continue
                    text_g = tokenizer.decode(comp_ids, skip_special_tokens=True)
                    verdict = verify_benchmark_response(task_view, text_g)
                    reward = 1.0 if verdict.passed else 0.35 * float(verdict.score)
                    rewards.append(reward)
                    full_g = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
                    rollouts.append(full_g)

            mean_r = sum(rewards) / len(rewards)
            round_rewards.append(mean_r)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = math.sqrt(var_r)
            if std_r < 1e-4:
                skipped_zero_var += 1
                continue

            # 2. Group-normalized policy gradient + exact reference KL using frozen base weights
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss_scalar = 0.0
            for gi, full_g in enumerate(rollouts):
                if full_g.shape[1] <= prompt_len:
                    continue
                adv = max(-2.5, min(2.5, (rewards[gi] - mean_r) / (std_r + 1e-5)))
                out_pi = model(input_ids=full_g)
                logits_pi = out_pi.logits if hasattr(out_pi, "logits") else out_pi
                logp_pi = _token_logprobs(logits_pi, full_g, prompt_len)

                with torch.no_grad(), _bypass_adapters(model):
                    out_ref = model(input_ids=full_g)
                    logits_ref = out_ref.logits if hasattr(out_ref, "logits") else out_ref
                    logp_ref = _token_logprobs(logits_ref, full_g, prompt_len)

                # Non-negative k3 KL penalty
                log_ratio = (logp_ref - logp_pi).clamp(-10.0, 10.0)
                kl = torch.exp(log_ratio) - log_ratio - 1.0
                pg_term = -adv * logp_pi.mean()
                kl_term = config.kl_beta * kl.mean()
                loss_g = (pg_term + kl_term) / float(len(rollouts))
                loss_g.backward()
                total_loss_scalar += float(loss_g.detach().cpu().item())
                mean_kls.append(float(kl.mean().detach().cpu().item()))

            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            total_updates += 1

        mean_rewards.append(round(sum(round_rewards) / max(len(round_rewards), 1), 4))

    model.eval()
    return {
        "rounds": config.rounds,
        "gradient_updates": total_updates,
        "skipped_zero_variance_groups": skipped_zero_var,
        "round_mean_rewards": mean_rewards,
        "mean_kl": round(sum(mean_kls) / max(len(mean_kls), 1), 6),
    }
