"""Causal selection policies for the recovered ACE-X9.2 checkpoints."""

import torch


def cap_prefix(eligible: torch.Tensor) -> torch.Tensor:
    """Accept the first eligible token allowed by the current row's prefix budget.

    Prefix length t may request max(1, floor(0.3 t)) refinements. Rejections
    are irrevocable. The largest past excess of offers over allowance is the
    count of irrevocable rejections necessary through the current position.
    """
    steps = torch.arange(1, eligible.shape[-1] + 1, device=eligible.device)
    allowance = (3 * steps // 10).clamp_min(1)
    offered = eligible.long().cumsum(-1)
    rejected = (offered - allowance).clamp_min(0).cummax(-1).values
    accepted = offered - rejected
    return torch.diff(accepted, prepend=torch.zeros_like(accepted[..., :1])) > 0


def stock_prefix_end(voc: torch.Tensor, eligible: torch.Tensor, shield: torch.Tensor) -> torch.Tensor:
    """Replay the archived gate's final decision for every single-row prefix.

    This quadratic-time audit reference deliberately preserves the old >= tie
    convention and ranking over all token scores when the budget binds.
    """
    steps = torch.arange(1, voc.shape[-1] + 1, device=voc.device)
    earlier_or_equal = steps[None, :] <= steps[:, None]
    greater = ((voc.unsqueeze(-2) > voc.unsqueeze(-1)) & earlier_or_equal).sum(-1)
    # Match the original int(prefix_length * 0.30) arithmetic exactly.
    budget = torch.tensor([max(1, int(n * 0.30)) for n in range(1, voc.shape[-1] + 1)], device=voc.device)
    return torch.where(eligible.long().cumsum(-1) > budget, (greater < budget) & ~shield, eligible)
