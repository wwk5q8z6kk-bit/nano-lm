"""Trainer for Track A Phase-2 Breakthrough Arms (`LAB-ARCH-04`: `X5_atrd_muon` & `X6_pkm_muon`).

Matches exact active non-embedding parameters (`~29.4M`), WSD schedule, validation shards,
and 512-token OOD copy probe against `X4_nanolm_v2_muon` (`0.9575 ± 0.0008` BPB).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from lab.arch.arch04 import Arch04Cfg, Arch04LM
from lab.arch.train import Muon, evaluate, wsd

ARCH04_SPECS = {
    "X5_atrd_muon": {"use_atrd": True, "atrd_frac": 0.25, "use_pkm": False, "optim": "muon"},
    "X6_pkm_muon": {"use_atrd": False, "atrd_frac": 0.0, "use_pkm": True, "optim": "muon"},
    "X7_atrd_pkm_muon": {"use_atrd": True, "atrd_frac": 0.25, "use_pkm": True, "optim": "muon"},
}

TARGET_ACTIVE_NON_EMB = 29_377_080  # exact match to X4_nanolm_v2_muon!


def build_arch04(arm: str, vocab: int, seq: int) -> Arch04LM:
    s = ARCH04_SPECS[arm]
    lo, hi = 512, 2048
    while hi - lo > 32:
        mid = (lo + hi) // 2 // 32 * 32
        cfg = Arch04Cfg(vocab=vocab, seq=seq, ffn=mid, use_atrd=s["use_atrd"], atrd_frac=s["atrd_frac"], use_pkm=s["use_pkm"])
        if Arch04LM(cfg).param_report()["active_non_embedding"] < TARGET_ACTIVE_NON_EMB:
            lo = mid
        else:
            hi = mid
    best = min(
        (lo, hi),
        key=lambda f: abs(
            Arch04LM(Arch04Cfg(vocab=vocab, seq=seq, ffn=f, use_atrd=s["use_atrd"], atrd_frac=s["atrd_frac"], use_pkm=s["use_pkm"]))
            .param_report()["active_non_embedding"]
            - TARGET_ACTIVE_NON_EMB
        ),
    )
    return Arch04LM(Arch04Cfg(vocab=vocab, seq=seq, ffn=best, use_atrd=s["use_atrd"], atrd_frac=s["atrd_frac"], use_pkm=s["use_pkm"]))


def make_arch04_optim(model: Arch04LM, lr: float, wd: float):
    # Sparse persistent memory tables & gates use AdamW (sparse row gradients should NOT be Newton-Schulz orthogonalized!)
    skip_muon = ("emb", "head", "v_gate", "pkm_banks", "step_emb")
    hidden2d = [p for n, p in model.named_parameters() if p.dim() == 2 and not any(t in n for t in skip_muon)]
    other = [p for n, p in model.named_parameters() if not (p.dim() == 2 and not any(t in n for t in skip_muon))]
    return [
        Muon(hidden2d, lr=0.02, wd=wd * 0.1),
        torch.optim.AdamW(other, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARCH04_SPECS))
    ap.add_argument("--data", default="/Volumes/Express1M2/nanolm-lab/data/v1")
    ap.add_argument("--out", default="/Volumes/Express1M2/nanolm-lab/runs")
    ap.add_argument("--tokens", type=float, default=600e6)
    ap.add_argument("--batch-tokens", type=int, default=131072)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", type=int, default=4)
    ap.add_argument("--compile", action="store_true")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type in ("cuda", "mps") else torch.float32
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed)

    data = Path(a.data)
    manifest = json.loads((data / "manifest.json").read_text())
    model = build_arch04(a.arm, vocab=manifest["vocab"], seq=a.seq).to(dev)
    opts = make_arch04_optim(model, a.lr, a.wd)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    # Compile PKM arm (static graph); ATD uses dynamic sparse K_max gather so run eager+Triton FLA
    fwd = torch.compile(model) if (a.compile and not model.c.use_atrd) else model

    run = f"{a.arm}__muon__tokens__{int(a.tokens / 1e6)}M__s{a.seed}"
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
        last_aux: dict = {}
        for _ in range(accum):
            n = a.micro * (a.seq + 1)
            xb = torch.from_numpy(np.asarray(train[ptr : ptr + n], dtype=np.int64).reshape(a.micro, a.seq + 1)).to(dev)
            ptr += n
            with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
                loss, aux = fwd(xb[:, :-1], xb[:, 1:], return_aux=True)
                last_aux = {k: (float(v.detach()) if isinstance(v, torch.Tensor) else v) for k, v in aux.items()}
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
            rec = {"step": step, "tokens": tok_seen, "loss": round(loss_acc, 4), "gnorm": round(gn, 3), "tok_s": round(tok_seen / el, 1), **last_aux}
            hist.append(rec)
            print(json.dumps(rec), flush=True)
        if step in eval_at:
            e = evaluate(model, data, manifest, dev, dtype, a.seq)
            e.update(step=step, tokens=tok_seen, **last_aux)
            evals.append(e)
            print("EVAL", json.dumps(e), flush=True)

    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if dev.type == "cuda" else 0
    hd = model.c.d // model.c.n_head
    kv_per_tok = sum(2 * model.c.n_kv * hd * 2 for k in model.c.layout if k == "A")
    res = {
        "run": run, "arm": a.arm, "optim": "muon", "match": "tokens", "lr": a.lr, "seed": a.seed,
        "device": str(dev), "params": rep, "tokens": tok_seen, "steps": steps,
        "wall_s": round(wall, 1), "train_tok_s": round(tok_seen / wall, 1),
        "peak_mem_gb": round(peak / 2**30, 2), "global_attn_kv_bytes_per_token": kv_per_tok,
        "train_flops_total": 6 * rep["active_non_embedding"] * tok_seen,
        "final": evals[-1] if evals else None, "evals": evals, "history": hist,
        "data_manifest_train_sha256": manifest["train_sha256"],
    }
    (rdir / "result.json").write_text(json.dumps(res, indent=2))
    torch.save({"model": model.state_dict(), "cfg": vars(model.c)}, rdir / "final.pt")
    print("DONE", json.dumps(res["final"]), flush=True)


if __name__ == "__main__":
    main()
