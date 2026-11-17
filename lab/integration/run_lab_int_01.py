"""Run `LAB-INT-01` Controlled Architecture Surgery Experiment on a Pretrained Dense GQA Checkpoint.

Evaluates all three preregistered conversion arms (`PREREG_LAB-INT-01.md`) on held-out real
`LAB-DATA-01` validation shards (`val_web`, `val_textbook`, `val_math`, `val_code`) + 512-tok OOD copy:
  - `Base`: Unmodified pretrained Dense GQA checkpoint
  - `I0_naive_swap`: Naive 3:1 Sharp-Delta swap + standard CE fine-tuning (matched steps)
  - `I1_progressive_rad_kd`: QKV-warm-started 3:1 Sharp-Delta + Stage-1 Mixer MSE + Stage-2 Top-64+Tail KD
  - `I2_zero_init_gated_vres`: Exact step-0 identity Gated-VRes graft + Top-64+Tail KD
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lab.arch.nextgen import NextGenCfg, NextGenLM
from lab.arch.train import evaluate
from lab.integration.surgery import (
    build_surgical_hybrid,
    graft_zero_init_gated_vres,
    stage1_align_mixers,
    stage2_distill_end_to_end,
)


def load_pretrained_nextgen(ckpt_path: Path, dev: torch.device) -> NextGenLM:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = dict(raw["cfg"])
    if isinstance(cfg_dict.get("loop"), list):
        cfg_dict["loop"] = tuple(cfg_dict["loop"])
    cfg = NextGenCfg(**cfg_dict)
    # Strip torch.compile `_orig_mod.` prefix if present
    sd = {k.replace("_orig_mod.", ""): v for k, v in raw["model"].items()}
    m = NextGenLM(cfg)
    m.load_state_dict(sd, strict=True)
    return m.to(dev)


def sample_train_batches(train_mm: np.memmap, n_steps: int, bsz: int, seq: int, dev: torch.device, seed: int = 123):
    rng = np.random.default_rng(seed)
    max_off = len(train_mm) - bsz * (seq + 1) - 1
    out = []
    for _ in range(n_steps):
        off = int(rng.integers(605_000_000, max(605_000_001, max_off)))  # unseen tail of 800M train.bin!
        chunk = np.asarray(train_mm[off : off + bsz * (seq + 1)], dtype=np.int64).reshape(bsz, seq + 1)
        out.append(torch.from_numpy(chunk).to(dev))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to pretrained S0_dense_125m final.pt")
    ap.add_argument("--data", default="/Volumes/Express1M2/nanolm-lab/data/v1")
    ap.add_argument("--out", default="/Users/mac/Projects/nano-lm/.worktrees/nanolm-lab/lab/reports/lab_int_01_results.json")
    ap.add_argument("--steps", type=int, default=80, help="Matched adaptation steps per surgical arm")
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type in ("cuda", "mps") else torch.float32
    data_dir = Path(args.data)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    train_mm = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")

    base = load_pretrained_nextgen(Path(args.ckpt), dev)
    base.eval()
    n_layers = len(base.c.layout)
    hybrid_layout = "".join("A" if (i + 1) % 4 == 0 else "D" for i in range(n_layers))

    print(f"Loaded pretrained dense model ({n_layers} layers) on {dev}; target hybrid layout={hybrid_layout}", flush=True)
    eval_base = evaluate(base, data_dir, manifest, dev, dtype, args.seq)
    print("BASE_EVAL:", json.dumps(eval_base), flush=True)

    batches = sample_train_batches(train_mm, args.steps, args.bsz, args.seq, dev)
    batches_xy = [(b[:, :-1], b[:, 1:]) for b in batches]

    # Arm I0: Naive cold swap + standard CE fine-tuning (matched total steps)
    t0 = time.perf_counter()
    m_i0 = build_surgical_hybrid(base, hybrid_layout, warmstart_qkv=False)
    eval_i0_step0 = evaluate(m_i0, data_dir, manifest, dev, dtype, args.seq)
    m_i0.train()
    opt_i0 = torch.optim.AdamW(m_i0.parameters(), lr=5e-4, weight_decay=0.01)
    for xb, yb in batches_xy:
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
            loss = m_i0(xb, yb)
        opt_i0.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m_i0.parameters(), 1.0)
        opt_i0.step()
    eval_i0_final = evaluate(m_i0, data_dir, manifest, dev, dtype, args.seq)
    s_i0 = round(time.perf_counter() - t0, 2)
    print("I0_NAIVE_SWAP:", json.dumps({"step0": eval_i0_step0, "final": eval_i0_final}), flush=True)
    del m_i0

    # Arm I1: QKV warm-start + Stage-1 mixer MSE (half steps) + Stage-2 Top-64+Tail KD (half steps)
    t0 = time.perf_counter()
    m_i1 = build_surgical_hybrid(base, hybrid_layout, warmstart_qkv=True)
    eval_i1_step0 = evaluate(m_i1, data_dir, manifest, dev, dtype, args.seq)
    half = args.steps // 2
    mses = stage1_align_mixers(base, m_i1, [b[:, :-1] for b in batches[:half]], lr=2e-3)
    kls = stage2_distill_end_to_end(base, m_i1, batches_xy[half:], lr=4e-4, alpha_kd=0.90, topk=64)
    eval_i1_final = evaluate(m_i1, data_dir, manifest, dev, dtype, args.seq)
    s_i1 = round(time.perf_counter() - t0, 2)
    print("I1_PROGRESSIVE_RAD_KD:", json.dumps({"step0": eval_i1_step0, "final": eval_i1_final}), flush=True)
    del m_i1

    # Arm I2: Exact Zero-Init Gated-VRes Highway Graft (100% frozen backbone, highway_gate only!)
    t0 = time.perf_counter()
    m_i2 = graft_zero_init_gated_vres(base)
    eval_i2_step0 = evaluate(m_i2, data_dir, manifest, dev, dtype, args.seq)
    for name, p in m_i2.named_parameters():
        p.requires_grad_("highway_gate" in name)
    opt_i2 = torch.optim.AdamW([p for p in m_i2.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.0)
    m_i2.train()
    for xb, yb in batches_xy:
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
            loss_i2 = m_i2(xb, yb)
        opt_i2.zero_grad(set_to_none=True)
        loss_i2.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m_i2.parameters() if p.requires_grad], 1.0)
        opt_i2.step()
    eval_i2_final = evaluate(m_i2, data_dir, manifest, dev, dtype, args.seq)
    s_i2 = round(time.perf_counter() - t0, 2)
    print("I2_ZERO_INIT_GATED_VRES:", json.dumps({"step0": eval_i2_step0, "final": eval_i2_final}), flush=True)

    summary = {
        "schema": "nanolm.lab_int_01.v1",
        "pretrained_ckpt": str(args.ckpt),
        "hybrid_layout": hybrid_layout,
        "adaptation_tokens": args.steps * args.bsz * args.seq,
        "base": eval_base,
        "I0_naive_swap": {"step0": eval_i0_step0, "final": eval_i0_final, "wall_s": s_i0},
        "I1_progressive_rad_kd": {
            "step0": eval_i1_step0, "final": eval_i1_final,
            "stage1_mse_start_end": [round(mses[0], 5), round(mses[-1], 5)],
            "stage2_kl_start_end": [round(kls[0], 5), round(kls[-1], 5)],
            "wall_s": s_i1,
        },
        "I2_zero_init_gated_vres": {"step0": eval_i2_step0, "final": eval_i2_final, "wall_s": s_i2},
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("SAVED_LAB_INT_01", args.out, flush=True)


if __name__ == "__main__":
    main()
