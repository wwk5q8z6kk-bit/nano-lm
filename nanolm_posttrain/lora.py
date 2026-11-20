"""Pure-PyTorch LoRA adaptation and verifier-guided post-training for NanoLM (<= 4B).

Zero external PEFT dependency required: wraps target ``nn.Linear`` layers in
frozen base models (both native ``NanoLM`` and HuggingFace ``<= 4B`` backbones),
keeping base weights frozen in FP16/BF16 while training FP32 low-rank ``A`` and
``B`` adapters on assistant-only target tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS, ParameterBudgetExceededError


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_suffixes: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class LoRALinear(nn.Module):
    """Frozen base ``nn.Linear`` augmented with FP32 low-rank ``B @ A`` residual."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        in_features = base.in_features
        out_features = base.out_features
        self.rank = rank
        self.scaling = alpha / float(rank)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty((rank, in_features), device=device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros((out_features, rank), device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(self.drop(x.float()), self.lora_a), self.lora_b) * self.scaling
        return base_out + delta.to(base_out.dtype)


def apply_lora(model: nn.Module, config: LoRAConfig = LoRAConfig()) -> int:
    """Freeze base model (enforcing <= 4B ceiling) and attach LoRA adapters."""

    total_base = sum(p.numel() for p in model.parameters())
    targets = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(
            name == suffix or name.endswith("." + suffix) for suffix in config.target_suffixes
        )
    ]
    adapter_parameters = sum(config.rank * (module.in_features + module.out_features) for _, module in targets)
    total = total_base + adapter_parameters
    if total > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(f"adapted model has {total:,} parameters, exceeding NanoLM's 4B ceiling")
    if not targets:
        raise ValueError(f"no linear layers matched target_suffixes={config.target_suffixes}")
    for param in model.parameters():
        param.requires_grad_(False)

    for full_name, module in targets:
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(
            parent,
            parts[-1],
            LoRALinear(module, rank=config.rank, alpha=config.alpha, dropout=config.dropout),
        )
    return len(targets)


def trainable_parameter_summary(model: nn.Module) -> dict[str, int | float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": round(trainable / max(total, 1), 6),
    }


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def train_masked_sft_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: Tensor,
    labels: Tensor,
    *,
    max_grad_norm: float = 1.0,
) -> float:
    """Single causal SFT step computing loss strictly on unmasked (``!= -100``) labels."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    loss.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()
    return float(loss.detach().cpu().item())


def build_chat_sft_tensors(
    tokenizer: Any,
    examples: Sequence[tuple[str, str, str]],
    *,
    device: str | torch.device,
) -> list[tuple[Tensor, Tensor]]:
    """Tokenize ``(system, user, assistant)`` triples with prompt tokens masked as ``-100``."""

    batches: list[tuple[Tensor, Tensor]] = []
    for system_prompt, user_prompt, assistant_reply in examples:
        prompt_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        full_msgs = [*prompt_msgs, {"role": "assistant", "content": assistant_reply}]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if len(labels) != len(full_ids):
            raise ValueError("chat template tokenization prefix mismatch")
        batches.append(
            (
                torch.tensor([full_ids], dtype=torch.long, device=device),
                torch.tensor([labels], dtype=torch.long, device=device),
            )
        )
    return batches
