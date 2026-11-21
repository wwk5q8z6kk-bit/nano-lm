"""Small language-model comparison on the v1 corpus: dense against weight-tied looping.

Uses the dense model already on master (research/frontier/m21_dense_code/dense.py), which has
a loop option: (prelude, core, coda, r) applies the core blocks r times.
Step-matched. Reports bits per byte on the four held-out domains.
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m21_dense_code"))
from dense import Cfg, LM  # noqa: E402

ARMS = {  # every arm runs 8 block applications per token
    "dense8": dict(layout="A" * 8, loop=None, d=256, ffn=704),
    "loop_1_2_1x3": dict(layout="A" * 4, loop=(1, 2, 1, 3), d=256, ffn=704),          # fewer parameters, same compute
    "loop_wide": dict(layout="A" * 4, loop=(1, 2, 1, 3), d=320, ffn=880),             # more compute per step; closer in block parameters
}


def batch(data, n, seq, rng, device):
    ix = rng.integers(0, len(data) - seq - 1, n)
    x = np.stack([data[i:i + seq + 1] for i in ix]).astype(np.int64)
    t = torch.from_numpy(x).to(device)
    return t[:, :-1], t[:, 1:]


@torch.no_grad()
def evaluate(model, root, manifest, seq, device, n_batches=20, bs=16):
    out = {}; model.eval()
    for dom in ("web", "textbook", "math", "code"):
        d = np.memmap(root / f"val_{dom}.bin", dtype=np.uint16, mode="r")
        rng = np.random.default_rng(0); nll = 0.0
        for _ in range(n_batches):
            x, y = batch(d, bs, seq, rng, device); nll += float(model(x, y))
        bits_tok = nll / n_batches / math.log(2)
        out[dom] = bits_tok / manifest["domains"][dom]["val"]["bytes_per_token"]
    mix = manifest["mix"]; out["mix"] = sum(mix[k] * out[k] for k in mix); model.train()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--data", default="/Volumes/Express1M2/nanolm-lab/data/v1")
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--device", default="mps"); ap.add_argument("--out", default="")
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()
    root = Path(a.data); manifest = json.load(open(root / "manifest.json"))
    data = np.memmap(root / "train.bin", dtype=np.uint16, mode="r")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    cfg = Cfg(n_head=4, n_kv=2, seq=a.seq, **ARMS[a.arm]); model = LM(cfg).to(a.device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    t0, curve = time.time(), []
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / a.steps))))
        for g in opt.param_groups: g["lr"] = lr
        x, y = batch(data, a.bs, a.seq, rng, a.device)
        loss = model(x, y); opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            e = evaluate(model, root, manifest, a.seq, a.device)
            curve.append({"step": step, "tokens": step * a.bs * a.seq, "seconds": round(time.time() - t0, 1), "train_loss": float(loss), **e})
            print(a.arm, curve[-1], flush=True)
    res = {"arm": a.arm, "seed": a.seed, "params": params, "steps": a.steps, "tokens": a.steps * a.bs * a.seq,
           "tokens_per_second": round(a.steps * a.bs * a.seq / (time.time() - t0)), "final": curve[-1], "curve": curve, "cfg": ARMS[a.arm]}
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}))
    if a.out: json.dump(res, open(a.out, "w"))
