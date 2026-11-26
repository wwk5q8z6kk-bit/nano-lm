"""Discovery-lab trainer: identical data order, budget, and evaluation for every arm.

Usage:
    python -m lab.arch.train --arm T0_dense --tokens 200e6 --data /Volumes/.../data/v1
Writes <out>/<run>/result.json and appends to <out>/results.jsonl.
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

from lab.arch.arms import ARMS, build
from lab.arch.flops import train_flops_per_token
from lab.arch.model import LM


# ---------------------------------------------------------------- optimizers
def newton_schulz(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tr = G.size(-2) > G.size(-1)
    if tr:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return (X.mT if tr else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz (Jordan et al. 2024), for 2-D hidden weights."""

    def __init__(self, params, lr=0.02, momentum=0.95, wd=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                upd = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))
                upd *= max(1.0, p.size(0) / p.size(1)) ** 0.5
                if g["wd"]:
                    p.mul_(1 - g["lr"] * g["wd"])
                p.add_(upd, alpha=-g["lr"])


def make_optim(model: LM, kind: str, lr: float, wd: float):
    hidden2d = [p for n, p in model.named_parameters() if p.dim() == 2 and "emb" not in n and "head" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() == 2 and "emb" not in n and "head" not in n)]
    if kind == "adamw":
        decay = [p for p in model.parameters() if p.dim() >= 2]
        nod = [p for p in model.parameters() if p.dim() < 2]
        return [torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": nod, "weight_decay": 0}],
                                  lr=lr, betas=(0.9, 0.95), eps=1e-8)]
    if kind == "muon":
        return [Muon(hidden2d, lr=0.02, wd=wd * 0.1),
                torch.optim.AdamW(other, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)]
    raise ValueError(kind)


def wsd(step: int, total: int, warm: float = 0.02, decay: float = 0.2) -> float:
    w, d = max(1, int(total * warm)), int(total * decay)
    if step < w:
        return (step + 1) / w
    if step >= total - d:
        return max(0.0, (total - step) / d)
    return 1.0


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model: LM, data: Path, manifest: dict, dev, dtype, seq: int, windows: int = 128, bs: int = 16):
    model.eval()
    out: dict = {"bpb": {}}
    for dom, meta in manifest["domains"].items():
        arr = np.fromfile(data / f"val_{dom}.bin", dtype=np.uint16)
        n = min(windows, (len(arr) - 1) // (seq + 1))
        idx = np.arange(n) * ((len(arr) - seq - 1) // n)
        tot, cnt = 0.0, 0
        pos_loss = torch.zeros(seq, device=dev)
        for i in range(0, n, bs):
            xb = np.stack([arr[j:j + seq + 1] for j in idx[i:i + bs]]).astype(np.int64)
            x = torch.from_numpy(xb).to(dev)
            with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
                lg = model(x[:, :-1])
            l = torch.nn.functional.cross_entropy(lg.float().transpose(1, 2), x[:, 1:], reduction="none")
            pos_loss += l.sum(0)
            tot += l.sum().item()
            cnt += l.numel()
        nats = tot / cnt
        bpt = meta["val"]["bytes_per_token"]
        out["bpb"][dom] = round(nats / math.log(2) / bpt, 5)
        pl = (pos_loss / n).cpu()
        if dom == "web":
            # context utilisation: how much later positions benefit from longer context
            out["web_loss_pos_0_64"] = round(pl[:64].mean().item(), 4)
            out["web_loss_pos_last_256"] = round(pl[-256:].mean().item(), 4)
    w = {"web": 0.4, "textbook": 0.2, "math": 0.2, "code": 0.2}
    out["bpb_composite"] = round(sum(out["bpb"][k] * w[k] for k in w), 5)

    # in-context copy probe: random token string repeated; loss on the second copy (nats).
    g = torch.Generator().manual_seed(1234)
    L = seq // 2
    xs = torch.randint(64, manifest["vocab"] - 64, (32, L), generator=g)
    x = torch.cat([xs, xs], 1).to(dev)
    with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
        lg = model(x[:, :-1])
    l = torch.nn.functional.cross_entropy(lg.float().transpose(1, 2), x[:, 1:], reduction="none")
    out["copy_loss_second_half"] = round(l[:, L:].mean().item(), 4)
    out["copy_acc_second_half"] = round((lg[:, L:].argmax(-1) == x[:, L + 1:]).float().mean().item(), 4)
    model.train()
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--data", default=os.environ.get("NANOLM_DATA_DIR", "artifacts/nanolm_lab/data/v1"))
    ap.add_argument("--out", default=os.environ.get("NANOLM_RUNS_DIR", "artifacts/nanolm_lab/runs"))
    ap.add_argument("--tokens", type=float, default=200e6)
    ap.add_argument("--batch-tokens", type=int, default=131072)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--optim", default=None, help="override arm optimizer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evals", type=int, default=4)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--match", choices=["tokens", "flops"], default="tokens",
                    help="flops: scale this arm's token budget so total train FLOPs equal T0_dense at --tokens")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.type in ("cuda", "mps") else torch.float32
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed)
    data = Path(a.data)
    manifest = json.loads((data / "manifest.json").read_text())
    spec = ARMS[a.arm]
    model = build(a.arm, vocab=manifest["vocab"], seq=a.seq).to(dev)
    optim_kind = a.optim or spec.get("optim", "adamw")
    opts = make_optim(model, optim_kind, a.lr, a.wd)
    base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]
    fwd = torch.compile(model) if a.compile else model
    flops = train_flops_per_token(model, a.seq)
    if a.match == "flops":
        ref = train_flops_per_token(build("T0_dense", vocab=manifest["vocab"], seq=a.seq), a.seq)
        a.tokens = a.tokens * ref["total"] / flops["total"]

    run = f"{a.arm}__{optim_kind}__{a.match}__{int(a.tokens/1e6)}M__s{a.seed}{('__' + a.tag) if a.tag else ''}"
    rdir = Path(a.out) / run
    rdir.mkdir(parents=True, exist_ok=True)
    train = np.memmap(data / "train.bin", dtype=np.uint16, mode="r")
    accum = max(1, a.batch_tokens // (a.micro * a.seq))
    step_tokens = accum * a.micro * a.seq
    steps = int(a.tokens // step_tokens)
    assert steps * step_tokens + 1 < len(train), "not enough training tokens"
    rep = model.param_report()
    print(json.dumps({"run": run, "device": str(dev), "steps": steps, "accum": accum, **rep}), flush=True)

    hist, evals = [], []
    eval_at = {int(steps * (i + 1) / a.evals) - 1 for i in range(a.evals)}
    t0, ptr, tok_seen = time.time(), 0, 0
    for step in range(steps):
        f = wsd(step, steps)
        for o, bl in zip(opts, base_lrs):
            for g, lr in zip(o.param_groups, bl):
                g["lr"] = lr * f
        loss_acc = 0.0
        for _ in range(accum):
            n = a.micro * (a.seq + 1)
            xb = torch.from_numpy(np.asarray(train[ptr:ptr + n], dtype=np.int64).reshape(a.micro, a.seq + 1)).to(dev)
            ptr += n  # sequential, never repeated, identical order for all arms
            with torch.autocast(dev.type, dtype=dtype, enabled=dtype != torch.float32):
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
        if step % 20 == 0 or step == steps - 1:
            el = time.time() - t0
            hist.append({"step": step, "tokens": tok_seen, "loss": round(loss_acc, 4), "gnorm": round(gn, 3),
                         "tok_s": round(tok_seen / el, 1)})
            print(json.dumps(hist[-1]), flush=True)
        if step in eval_at:
            e = evaluate(model, data, manifest, dev, dtype, a.seq)
            e.update(step=step, tokens=tok_seen)
            evals.append(e)
            print("EVAL", json.dumps(e), flush=True)
    wall = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() if dev.type == "cuda"
            else torch.mps.driver_allocated_memory() if dev.type == "mps" else 0)
    # decode-state memory per token-position (bf16): attention KV grows with T; linear layers are O(1).
    hd = model.c.d // model.c.n_head
    kv_per_tok = sum(2 * model.c.n_kv * hd * 2 for k in model.c.layout if k == "A")
    res = {"run": run, "arm": a.arm, "spec": {k: v for k, v in spec.items() if k != "cfg"},
           "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(model.c).items()},
           "optim": optim_kind, "lr": a.lr, "seed": a.seed, "device": str(dev), "params": rep,
           "tokens": tok_seen, "steps": steps, "wall_s": round(wall, 1), "train_tok_s": round(tok_seen / wall, 1),
           "peak_mem_gb": round(peak / 2**30, 2), "global_attn_kv_bytes_per_token": kv_per_tok,
           "flops_per_token": flops, "train_flops_total": flops["total"] * tok_seen, "match": a.match,
           "final": evals[-1] if evals else None, "evals": evals, "history": hist,
           "data_manifest_train_sha256": manifest["train_sha256"], "git": os.popen("git rev-parse HEAD").read().strip()}
    (rdir / "result.json").write_text(json.dumps(res, indent=2))
    with open(Path(a.out) / "results.jsonl", "a") as fh:
        fh.write(json.dumps({k: res[k] for k in ("run", "arm", "optim", "seed", "tokens", "wall_s", "train_tok_s",
                                                  "params", "final", "peak_mem_gb", "match", "train_flops_total")}) + "\n")
    torch.save({"model": model.state_dict(), "cfg": vars(model.c)}, rdir / "final.pt")
    print("DONE", json.dumps(res["final"]), flush=True)


if __name__ == "__main__":
    main()
