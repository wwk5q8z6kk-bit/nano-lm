"""Native Nano inference — autoregressive target generation for span-port eval."""

from __future__ import annotations

from typing import Any

from nanoscribe.native.config import NativeTrainConfig
from nanoscribe.native.tokenize import detokenize, hash_tokens


def generate_target_line(
    model: Any,
    prompt: str,
    cfg: NativeTrainConfig,
    *,
    max_new_tokens: int = 64,
) -> str:
    import torch

    device = next(model.parameters()).device
    vocab = cfg.vocab_size
    prompt_ids = hash_tokens(prompt, vocab)
    seq = list(prompt_ids)
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = seq[-(cfg.max_seq - 1) :]
            x = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(x)
            next_id = int(logits[0, -1].argmax().item())
            if next_id <= 0:
                break
            seq.append(next_id)
            decoded = detokenize(seq[len(prompt_ids) :], vocab)
            if "\n" in decoded:
                break
    return detokenize(seq[len(prompt_ids) :], vocab).strip()
