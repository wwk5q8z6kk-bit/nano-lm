"""LAB-ARCH-03 & LAB-SCALE-01 trainer for Next-Gen first-principles architectures (Track A).

Uses the exact same data stream (train.bin), validation shards (val_*.bin), 512-tok OOD copy
probe, WSD schedule, and AdamW / Muon optimizers as lab/arch/train.py so every number is
strictly comparable to LAB-ARCH-02.

Arms:
  X1_gated_vres       46M (29.5M non-emb): DDDADDDADA + universal per-head input-gated V0 highway
  X2_sharp_delta_vres 46M (29.5M non-emb): X1 + Sharp-V (QK-only conv) + QK-Canon Attention
  X3_sgrd_delta       46M (29.5M non-emb): DDDADDDADA loop(2,6,2,2) + Surprise-Gated Recurrent Depth
  X4_nanolm_v2_muon   46M (29.5M non-emb): X2_sharp_delta_vres + Muon orthogonalized momentum
  S0_dense_125m      121M (96.0M non-emb): 16-layer Dense GQA baseline at 2.6x scale (d=768, H=12, KV=4)
  S1_nanolm_v2_125m  121M (96.0M non-emb): 16-layer NextGen (DDDADDDADDDADDDA + Gated-VRes + Sharp-V)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from lab.arch.nextgen import NextGenCfg, NextGenLM
from lab.arch.train import Muon, evaluate, wsd

NEXTGEN_SPECS: dict[str, dict] = {
    "X1_gated_vres": {
        "scale": "46m", "d": 512, "n_head": 8, "n_kv": 4, "layout": "DDDADDDADA",
        "gated_vres": True, "sharp_v": False, "qk_canon_attn": False, "loop": None, "sgrd_prior_p": 0.0,
        "optim": "adamw",
    },
    "X2_sharp_delta_vres": {
        "scale": "46m", "d": 512, "n_head": 8, "n_kv": 4, "layout": "DDDADDDADA",
        "gated_vres": True, "sharp_v": True, "qk_canon_attn": True, "loop": None, "sgrd_prior_p": 0.0,
        "optim": "adamw",
    },
    "X3_sgrd_delta": {
        "scale": "46m", "d": 512, "n_head": 8, "n_kv": 4, "layout": "DDDADDDADA",
        "gated_vres": True, "sharp_v": True, "qk_canon_attn": False, "loop": (2, 6, 2, 2), "sgrd_prior_p": 0.55,
        "optim": "adamw",
    },
    "X4_nanolm_v2_muon": {
        "scale": "46m", "d": 512, "n_head": 8, "n_kv": 4, "layout": "DDDADDDADA",
        "gated_vres": True, "sharp_v": True, "qk_canon_attn": True, "loop": None, "sgrd_prior_p": 0.0,
        "optim": "muon",
    },
    # LAB-SCALE-01: 125M scale-up verification (d=768, 16 layers, ~96.3M non-embedding params)
    "S0_dense_125m": {
        "scale": "125m", "d": 768, "n_head": 12, "n_kv": 4, "layout": "A" * 16,
        "gated_vres": False, "sharp_v": False, "qk_canon_attn": False, "loop": None, "sgrd_prior_p": 0.0,
        "optim": "adamw",
    },
    "S1_nanolm_v2_125m": {
        "scale": "125m", "d": 768, "n_head": 12, "n_kv": 4, "layout": "DDDADDDADDDADDDA",
        "gated_vres": True, "sharp_v": True, "qk_canon_attn": True, "loop": None, "sgrd_prior_p": 0.0,
        "optim": "adamw",
    },
    # LAB-SCALE-03: 350M scale-up Dense GQA baseline (~289.5M non-emb / ~323M total, d=1024, L=24, H=16, KV=4)
    "S0_dense_350m": {
        "scale": "350m", "d": 1024, "n_head": 16, "n_kv": 4, "layout": "A" * 24,
        "gated_vres": False, "sharp_v": False, "qk_canon_attn": False, "loop": None, "sgrd_prior_p": 0.0,
        "optim": "muon", "muon_lr": 0.00913,
    },
}

TARGETS_NON_EMB = {
    "46m": 29_500_416,     # exact T0_dense non-embedding params
    "125m": 96_350_208,    # S0_dense_125m at ffn=2048
    "350m": 289_456_128,   # S0_dense_350m at ffn=3072
}


def _make_cfg(arm: str, vocab: int, seq: int, ffn: int) -> NextGenCfg:
    s = NEXTGEN_SPECS[arm]
    return NextGenCfg(
        vocab=vocab, d=s["d"], n_head=s["n_head"], n_kv=s["n_kv"], ffn=ffn,
        layout=s["layout"], seq=seq, gated_vres=s["gated_vres"], sharp_v=s["sharp_v"],
        qk_canon_attn=s["qk_canon_attn"], loop=s["loop"], sgrd_prior_p=s["sgrd_prior_p"],
    )


def build_nextgen(arm: str, vocab: int, seq: int) -> NextGenLM:
    target = TARGETS_NON_EMB[NEXTGEN_SPECS[arm]["scale"]]
    lo, hi = 256, 6144
    while hi - lo > 32:
        mid = (lo + hi) // 2 // 32 * 32
        if NextGenLM(_make_cfg(arm, vocab, seq, mid)).param_report()["non_embedding"] < target:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda f: abs(NextGenLM(_make_cfg(arm, vocab, seq, f)).param_report()["non_embedding"] - target))
    return NextGenLM(_make_cfg(arm, vocab, seq, best))


def make_nextgen_optim(model: NextGenLM, kind: str, lr: float, wd: float, muon_lr: float = 0.02):
    hidden2d = [p for n, p in model.named_parameters() if p.dim() == 2 and "emb" not in n and "head" not in n and "v_gate" not in n and "halt_proj" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() == 2 and "emb" not in n and "head" not in n and "v_gate" not in n and "halt_proj" not in n)]
    if kind == "adamw":
        decay = [p for p in model.parameters() if p.dim() >= 2]
        nod = [p for p in model.parameters() if p.dim() < 2]
        return [torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": nod, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95), eps=1e-8)]
    if kind == "muon":
        return [Muon(hidden2d, lr=muon_lr, wd=wd * 0.1), torch.optim.AdamW(other, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)]
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(NEXTGEN_SPECS))
    ap.add_argument("--data", default=os.environ.get("NANOLM_DATA_DIR", "artifacts/nanolm_lab/data/v1"))
    ap.add_argument("--out", default=os.environ.get("NANOLM_RUNS_DIR", "artifacts/nanolm_lab/runs"))
    ap.add_argument("--tokens", type=float, default=600e6)
    ap.add_argument("--batch-tokens", type=int, default=131072)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--optim", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", type=int, default=4)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    spec = NEXTGEN_SPECS[a.arm]
    default_lr = 1.5e-3 if spec["scale"] == "350m" else (2e-3 if spec["scale"] == "125m" else 3e-3)
    lr = a.lr if a.lr is not None else default_lr
    optim_kind = a.optim or spec["optim"]

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type in ("cuda", "mps") else torch.float32
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed)

    data = Path(a.data)
    manifest = json.loads((data / "manifest.json").read_text())
    model = build_nextgen(a.arm, vocab=manifest["vocab"], seq=a.seq).to(dev)
    opts = make_nextgen_optim(model, optim_kind, lr, a.wd, muon_lr=spec.get("muon_lr", 0.02))
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    # Note: SGRD uses dynamic auxiliary dict output in training; compile non-SGRD models
    fwd = torch.compile(model) if (a.compile and not model.use_sgrd) else model

    run = f"{a.arm}__{optim_kind}__tokens__{int(a.tokens/1e6)}M__s{a.seed}{('__' + a.tag) if a.tag else ''}"
    rdir = Path(a.out) / run
    rdir.mkdir(parents=True, exist_ok=True)
    train = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    accum = max(1, a.batch_tokens // (a.micro * a.seq))
    step_tokens = accum * a.micro * a.seq
    steps = int(a.tokens // step_tokens)
    rep = model.param_report()
    print(json.dumps({"run": run, "device": str(dev), "steps": steps, "accum": accum, **rep}), flush=True)

    hist, evals = [], []
    eval_at = {int(steps * (i + 1) / a.evals) - 1 for i in range(a.evals)}
    t0, ptr, tok_seen = time.time(), 0, 0

    for step in range(steps):
        f = wsd(step, steps)
        for o, bl in zip(opts, base_lrs):
            for g, base_lr in zip(o.param_groups, bl):
                g["lr"] = base_lr * f
        loss_acc = 0.0
        exp_loops = None
        h_gamma = None
        for _ in range(accum):
            n = a.micro * (a.seq + 1)
            xb = torch.from_numpy(np.asarray(train[ptr : ptr + n], dtype=np.int64).reshape(a.micro, a.seq + 1)).to(dev)
            ptr += n
            with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
                if model.use_sgrd:
                    loss, aux = fwd(xb[:, :-1], xb[:, 1:], return_aux=True)
                    exp_loops = float(aux["expected_loops"])
                    h_gamma = float(aux["halt_gamma"])
                else:
                    loss = fwd(xb[:, :-1], xb[:, 1:])
            (loss / accum).backward()
            loss_acc += loss.item() / accum
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        tok_seen += step_tokens
        if not math.isfinite(loss_acc):
            print("NON-FINITE LOSS; aborting", flush=True)
            break
        if step % 100 == 0 or step == steps - 1:
            el = time.time() - t0
            rec = {"step": step, "tokens": tok_seen, "loss": round(loss_acc, 4), "gnorm": round(gn, 3), "tok_s": round(tok_seen / el, 1)}
            if exp_loops is not None:
                rec["expected_loops"] = round(exp_loops, 3)
                rec["halt_gamma"] = round(h_gamma, 4)
            hist.append(rec)
            print(json.dumps(rec), flush=True)
        if step in eval_at:
            e = evaluate(model, data, manifest, dev, dtype, a.seq)
            e.update(step=step, tokens=tok_seen)
            if exp_loops is not None:
                e["expected_loops"] = round(exp_loops, 3)
                e["halt_gamma"] = round(h_gamma, 4)
            evals.append(e)
            print("EVAL", json.dumps(e), flush=True)

    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
    hd = model.c.d // model.c.n_head
    kv_per_tok = sum(2 * model.c.n_kv * hd * 2 for k in model.c.layout if k == "A")
    res = {
        "run": run, "arm": a.arm, "optim": optim_kind, "match": "tokens", "lr": lr, "seed": a.seed,
        "device": str(dev), "params": rep, "tokens": tok_seen, "steps": steps,
        "wall_s": round(wall, 1), "train_tok_s": round(tok_seen / wall, 1),
        "peak_mem_gb": round(peak / 2**30, 2), "global_attn_kv_bytes_per_token": kv_per_tok,
        "train_flops_total": 6 * rep["non_embedding"] * tok_seen,
        "final": evals[-1] if evals else None, "evals": evals, "history": hist,
        "data_manifest_train_sha256": manifest["train_sha256"],
    }
    (rdir / "result.json").write_text(json.dumps(res, indent=2))
    torch.save({"model": model.state_dict(), "cfg": vars(model.c)}, rdir / "final.pt")
    print("DONE", json.dumps(res["final"]), flush=True)


if __name__ == "__main__":
    main()
