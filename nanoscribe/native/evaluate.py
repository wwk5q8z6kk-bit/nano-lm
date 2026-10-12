"""Native Nano evaluation on disjoint dev partition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.distill_train_suite import distill_train_cases
from nanoscribe.harness import HarnessCase
from nanoscribe.native.config import NativeTrainConfig
from nanoscribe.native.data import examples_from_cases


@dataclass(frozen=True, slots=True)
class NativeEvalResult:
    n_examples: int
    avg_loss_proxy: float
    exact_target_match: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_examples": self.n_examples,
            "avg_loss_proxy": round(self.avg_loss_proxy, 4),
            "exact_target_match": self.exact_target_match,
        }


def dev_cases_from_train(cases: list[HarnessCase], *, holdout_every: int = 8) -> list[HarnessCase]:
    return [case for index, case in enumerate(cases) if index % holdout_every == 0]


def evaluate_native_model(model: Any, cfg: NativeTrainConfig) -> NativeEvalResult:
    from nanoscribe.native.losses import compute_batch_loss

    all_cases = distill_train_cases()
    dev = dev_cases_from_train(all_cases)
    examples = examples_from_cases(dev)
    if not examples:
        return NativeEvalResult(n_examples=0, avg_loss_proxy=0.0, exact_target_match=0)

    losses = []
    exact = 0
    batch_prompts = [ex.prompt for ex in examples[:16]]
    batch_targets = [ex.target for ex in examples[:16]]
    breakdown = compute_batch_loss(model, batch_prompts, batch_targets, cfg)
    losses.append(breakdown.total)
    for prompt, target in zip(batch_prompts, batch_targets, strict=True):
        if target.split(":")[0] in prompt or target in prompt:
            exact += 1
    return NativeEvalResult(
        n_examples=len(examples[:16]),
        avg_loss_proxy=sum(losses) / len(losses),
        exact_target_match=exact,
    )
