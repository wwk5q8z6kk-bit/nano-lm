#!/usr/bin/env python3
"""Re-derive the BPB plausibility floor for a given tokenizer. PREREG D3.3 §7.3.

The current floor was calibrated on character-level runs. D6's lesson was that a
threshold must be calibrated against the **actual failure population it claims
to detect**, not against an anecdote — so this measures both populations under
each instrument rather than assuming BPB transfers.

It runs a 2x2: {hash, bpe} x {honest, leaking}. The leaking arm reconstructs
D1.1 by making every block bidirectional, which is the defect the floor exists
to catch.

This deliberately drives the training internals rather than `train_native`, so
the leaking arm does not require bypassing the startup gate — the gate stays
un-weakened in the production path. Nothing here writes to any run namespace.

    python3 scripts/derive_bpb_floor.py [--steps 40]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.native.config import NativeTrainConfig, NativeVariant
from nanoscribe.native.data import NativeBatchIterator, load_train_examples
from nanoscribe.native.integrity import bits_per_byte
from nanoscribe.native.losses import compute_batch_loss
from nanoscribe.native.model import build_model


def _make_bidirectional(model) -> None:
    """Reconstruct D1.1: strip the causal mask from every block."""
    for block in model.blocks:
        original = block.attn.forward

        def bidirectional(q, k, v, _orig=original, **kwargs):
            kwargs.pop("attn_mask", None)
            kwargs.pop("is_causal", None)
            return _orig(q, k, v, **kwargs)

        block.attn.forward = bidirectional


def run_cell(tokenizer: str, *, leaking: bool, steps: int) -> dict:
    import torch

    cfg = NativeTrainConfig(
        run_id=f"bpbcal_{tokenizer}_{'leak' if leaking else 'honest'}",
        variant=NativeVariant.NATIVE_A,
        cell=None,
        seed=0,
        d_model=192,
        n_layers=4,
        n_heads=4,
        max_seq=512,
        max_steps=steps,
        batch_size=8,
        cpu_smoke=True,
        tokenizer=tokenizer,
    )
    model = build_model(cfg)
    if leaking:
        _make_bidirectional(model)

    examples = load_train_examples(cfg.dataset_path)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr)

    first_bpb = None
    bpb = float("nan")
    lm_nats = float("nan")
    step = 0
    while step < steps:
        for batch in NativeBatchIterator(examples, cfg.batch_size, seed=cfg.seed):
            if step >= steps:
                break
            prompts = [b.prompt for b in batch]
            targets = [b.target for b in batch]
            opt.zero_grad(set_to_none=True)
            bd = compute_batch_loss(model, prompts, targets, cfg)
            total = bd.lm + bd.span_port
            total.backward()
            opt.step()
            lm_nats = float(bd.lm.detach())
            n_sup = bd.n_supervised
            n_bytes = sum(len(t.encode("utf-8")) for t in targets)
            bpb = bits_per_byte(lm_nats * n_sup, max(1, n_bytes))
            if first_bpb is None:
                first_bpb = bpb
            step += 1

    return {
        "tokenizer": tokenizer,
        "condition": "leaking" if leaking else "honest",
        "steps": steps,
        "init_bpb": round(first_bpb, 5) if first_bpb is not None else None,
        "final_bpb": round(bpb, 5),
        "final_lm_nats_per_token": round(lm_nats, 5),
        "tokens_per_supervised_byte": round(n_sup / max(1, n_bytes), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/campaign/bpb_floor_derivation_v1.json")
    args = ap.parse_args()

    cells = [
        run_cell(tok, leaking=leak, steps=args.steps)
        for tok in ("hash", "bpe")
        for leak in (False, True)
    ]

    by = {(c["tokenizer"], c["condition"]): c for c in cells}
    report = {
        "schema": "nano.native.bpb_floor_derivation.v1",
        "purpose": "PREREG D3.3 §7.3 — re-derive the BPB floor for the BPE instrument.",
        "method": ("2x2 over {hash,bpe} x {honest,leaking}. The leaking arm "
                   "reconstructs D1.1 by making every block bidirectional, i.e. the "
                   "actual failure population the floor claims to detect (D6's lesson)."),
        "steps": args.steps,
        "cells": cells,
    }
    for tok in ("hash", "bpe"):
        h, l = by[(tok, "honest")]["final_bpb"], by[(tok, "leaking")]["final_bpb"]
        sep = h / l if l > 0 else float("inf")
        report[f"separation_{tok}"] = {
            "honest_final_bpb": h,
            "leaking_final_bpb": l,
            "separation_ratio": round(sep, 1) if math.isfinite(sep) else None,
            "geometric_midpoint": round(math.sqrt(h * l), 5) if l > 0 else None,
        }
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'tokenizer':10s} {'condition':10s} {'init_bpb':>12s} {'final_bpb':>12s} {'lm/tok':>10s} {'tok/byte':>9s}")
    print("-" * 70)
    for c in cells:
        print(f"{c['tokenizer']:10s} {c['condition']:10s} {c['init_bpb']:>12.4f} "
              f"{c['final_bpb']:>12.5f} {c['final_lm_nats_per_token']:>10.4f} "
              f"{c['tokens_per_supervised_byte']:>9.3f}")
    print()
    for tok in ("hash", "bpe"):
        s = report[f"separation_{tok}"]
        print(f"  {tok}: honest {s['honest_final_bpb']} vs leaking {s['leaking_final_bpb']}"
              f"  -> separation {s['separation_ratio']}x, geometric midpoint {s['geometric_midpoint']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
