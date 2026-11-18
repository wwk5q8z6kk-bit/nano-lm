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
    "X5_atrd_muon": {"scale": "46m", "d": 512, "n_head": 8, "n_kv": 4, "layout": "DDDADDDADA", "use_atrd": True, "use_margin_gate": False, "atrd_frac": 0.25, "use_pkm": False, "optim": "muon", "muon_lr": 0.02},
    "X6_pkm_muon": {"scale": "46m", "d": 512, "n_head": 8, "n_kv": 4, "layout": "DDDADDDADA", "use_atrd": False, "use_margin_gate": False, "atrd_frac": 0.0, "use_pkm": True, "optim": "muon", "muon_lr": 0.02},
    "X7_atrd_pkm_muon": {"scale": "46m", "d": 512, "n_head": 8, "n_kv": 4, "layout": "DDDADDDADA", "use_atrd": True, "use_margin_gate": False, "atrd_frac": 0.25, "use_pkm": True, "optim": "muon", "muon_lr": 0.02},
    "X8_margin_atrd_muon": {"scale": "46m", "d": 512, "n_head": 8, "n_kv": 4, "layout": "DDDADDDADA", "use_atrd": True, "use_margin_gate": True, "atrd_frac": 0.25, "use_pkm": False, "optim": "muon", "muon_lr": 0.02},
    # LAB-SCALE-02: 125M scale-up of NanoLM-v3 (X8_margin_atrd_muon) with spectral width/depth-scaled Muon LR!
    # eta_Muon(d=768, L=16) = 0.02 * sqrt(512/768) * sqrt(10/16) = 0.01291
    "S2_nanolm_v3_125m": {"scale": "125m", "d": 768, "n_head": 12, "n_kv": 4, "layout": "DDDADDDADDDADDDA", "use_atrd": True, "use_margin_gate": True, "atrd_frac": 0.25, "use_pkm": False, "optim": "muon", "muon_lr": 0.01291},
}

TARGETS_ACTIVE_NON_EMB = {
    "46m": 29_377_080,   # exact match to X4_nanolm_v2_muon
    "125m": 96_350_208,  # exact match to S0_dense_125m & S1_nanolm_v2_125m!
}


def build_arch04(arm: str, vocab: int, seq: int) -> Arch04LM:
    s = ARCH04_SPECS[arm]
    target = TARGETS_ACTIVE_NON_EMB[s["scale"]]
    lo, hi = 512, 4096
    while hi - lo > 32:
        mid = (lo + hi) // 2 // 32 * 32
        cfg = Arch04Cfg(
            vocab=vocab, d=s["d"], n_head=s["n_head"], n_kv=s["n_kv"], layout=s["layout"],
            seq=seq, ffn=mid, use_atrd=s["use_atrd"], use_margin_gate=s["use_margin_gate"],
            atrd_frac=s["atrd_frac"], use_pkm=s["use_pkm"],
        )
        if Arch04LM(cfg).param_report()["active_non_embedding"] < target:
            lo = mid
        else:
            hi = mid
    best = min(
        (lo, hi),
        key=lambda f: abs(
            Arch04LM(Arch04Cfg(
                vocab=vocab, d=s["d"], n_head=s["n_head"], n_kv=s["n_kv"], layout=s["layout"],
                seq=seq, ffn=f, use_atrd=s["use_atrd"], use_margin_gate=s["use_margin_gate"],
                atrd_frac=s["atrd_frac"], use_pkm=s["use_pkm"],
            )).param_report()["active_non_embedding"]
            - target
        ),
    )
    return Arch04LM(Arch04Cfg(
        vocab=vocab, d=s["d"], n_head=s["n_head"], n_kv=s["n_kv"], layout=s["layout"],
        seq=seq, ffn=best, use_atrd=s["use_atrd"], use_margin_gate=s["use_margin_gate"],
        atrd_frac=s["atrd_frac"], use_pkm=s["use_pkm"],
    ))


def make_arch04_optim(model: Arch04LM, lr: float, wd: float, muon_lr: float = 0.02):
    # Sparse persistent memory tables & gates use AdamW (sparse row gradients should NOT be Newton-Schulz orthogonalized!)
    skip_muon = ("emb", "head", "v_gate", "pkm_banks", "step_emb")
    hidden2d = [p for n, p in model.named_parameters() if p.dim() == 2 and not any(t in n for t in skip_muon)]
    other = [p for n, p in model.named_parameters() if not (p.dim() == 2 and not any(t in n for t in skip_muon))]
    return [
        Muon(hidden2d, lr=muon_lr, wd=wd * 0.1),
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
    spec_cfg = ARCH04_SPECS[a.arm]
    eff_adam_lr = a.lr * math.sqrt(512.0 / spec_cfg["d"]) if spec_cfg["scale"] == "125m" else a.lr
    opts = make_arch04_optim(model, eff_adam_lr, a.wd, muon_lr=spec_cfg["muon_lr"])
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
