"""Native Nano model — NATIVE-A decoder baseline, NATIVE-B evidence-aware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanoscribe.native.config import NativeTrainConfig, NativeVariant


def estimate_param_count(cfg: NativeTrainConfig) -> int:
    v, d, layers = cfg.vocab_size, cfg.d_model, cfg.n_layers
    emb = v * d
    block = layers * (4 * d * d + 2 * d * (4 * d))
    evidence = 2 * d * d if cfg.evidence_aware else 0
    return emb + block + evidence


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("torch required for native model") from exc
    return torch, nn, F


def build_model(cfg: NativeTrainConfig):
    torch, nn, F = _require_torch()
    d = cfg.d_model
    layers = cfg.n_layers
    heads = cfg.n_heads
    vocab = cfg.vocab_size
    head_dim = max(1, d // heads)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.ln2 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
            self.mlp = nn.Sequential(
                nn.Linear(d, 4 * d),
                nn.GELU(),
                nn.Linear(4 * d, d),
            )

        def forward(self, x):
            h = self.ln1(x)
            attn_out, _ = self.attn(h, h, h, need_weights=False)
            x = x + attn_out
            x = x + self.mlp(self.ln2(x))
            return x

    class NativeNano(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = cfg
            self.token_emb = nn.Embedding(vocab, d)
            self.pos_emb = nn.Embedding(cfg.max_seq, d)
            self.blocks = nn.ModuleList(Block() for _ in range(layers))
            self.ln_f = nn.LayerNorm(d)
            self.evidence_gate = (
                nn.Linear(d, d, bias=False) if cfg.evidence_aware else None
            )

        def forward(self, input_ids):
            bsz, seqlen = input_ids.shape
            positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            h = self.token_emb(input_ids) + self.pos_emb(positions)
            if self.evidence_gate is not None:
                h = h + 0.1 * self.evidence_gate(h)
            for block in self.blocks:
                h = block(h)
            h = self.ln_f(h)
            logits = torch.matmul(h, self.token_emb.weight.t())
            return logits

    return NativeNano()


@dataclass(frozen=True, slots=True)
class ModelBuildResult:
    model: Any
    param_count: int
    variant: NativeVariant

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_count": self.param_count,
            "params_m": round(self.param_count / 1e6, 2),
            "variant": self.variant.value,
        }


def build_native_model(cfg: NativeTrainConfig) -> ModelBuildResult:
    model = build_model(cfg)
    param_count = estimate_param_count(cfg)
    return ModelBuildResult(model=model, param_count=param_count, variant=cfg.variant)
