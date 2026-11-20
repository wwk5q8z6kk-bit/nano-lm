"""Fit one arm on BINDINGS for a fixed wall-clock budget and report accuracy by hop count.

Pilot use only until PROTOCOL.md is committed; pilots use seeds >= 1000.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from bindings import Spec, make, guessers, NULL
from demos import Demos
from baselines import EventTransformer
from structured import GraphMP, KeyStore


def collate(spec, rng, batch, device):
    ds = [make(spec, rng) for _ in range(batch)]
    Ns = max(len(b) for d in ds for b in d["blocks"])
    st = np.zeros((batch, spec.groups, Ns, 3), np.int64)
    q = np.zeros((batch, spec.groups, spec.queries, 2), np.int64)
    y = np.zeros((batch, spec.groups, spec.queries), np.int64)
    T = max(len(d["events"]) for d in ds)
    ev = np.zeros((batch, T, 5), np.int64)
    for i, d in enumerate(ds):
        for g, (blk, qs) in enumerate(zip(d["blocks"], d["queries"])):
            if blk: st[i, g, :len(blk)] = blk
            q[i, g] = [(k, h) for k, h, _ in qs]; y[i, g] = [a for _, _, a in qs]
        ev[i, :len(d["events"])] = d["events"]
    t = lambda a: torch.from_numpy(a).to(device)
    return {"st": t(st), "q": t(q), "y": t(y), "ev": t(ev), "N": np.mean([d["N"] for d in ds]),
            "M": np.mean([d["M"] for d in ds])}


BLOCK_ARMS = ("demos", "graph", "keystore")


def step_loss(model, arm, b):
    """Returns loss, and per-query (correct, hops) tensors."""
    if arm.startswith(BLOCK_ARMS):
        logits = model(b)
        y, hops = b["y"], b["q"][..., 1]
        loss = F.cross_entropy(logits.flatten(0, 2), y.flatten())
        return loss, (logits.argmax(-1) == y).flatten(), hops.flatten()
    ev = b["ev"]; logits = model(ev)
    mask = ev[..., 4] == 1
    loss = F.cross_entropy(logits[mask], ev[..., 3][mask])
    # hop count of each scored position = 1 + number of HOP events directly before it
    hop = torch.zeros_like(ev[..., 0]); run = torch.zeros_like(ev[:, 0, 0])
    for tpos in range(ev.shape[1]):
        run = torch.where(ev[:, tpos, 0] == 4, run + 1, torch.zeros_like(run)); hop[:, tpos] = run + 1
    return loss, (logits.argmax(-1) == ev[..., 3])[mask], hop[mask]


def build(arm, spec, d, device):
    if arm == "demos": m = Demos(spec.symbols, d=d)
    elif arm == "demos_nomass": m = Demos(spec.symbols, d=d, mass_dynamics=False)
    elif arm == "demos_noflow": m = Demos(spec.symbols, d=d, store_steps=0)
    elif arm == "graph": m = GraphMP(spec.symbols, d=d)
    elif arm == "keystore_token": m = KeyStore(spec.symbols, d=d, mode="token")
    elif arm == "keystore_learned": m = KeyStore(spec.symbols, d=d, mode="learned")
    elif arm == "dense": m = EventTransformer(spec.symbols, d=d, layers=4)
    elif arm == "looped": m = EventTransformer(spec.symbols, d=d, layers=1, loops=4)
    else: raise SystemExit(arm)
    return m.to(device)


def evaluate(model, arm, spec, device, seed, batches=8, batch=16):
    rng = np.random.default_rng(seed); model.eval()
    ok, hp = [], []
    for _ in range(batches):
        b = collate(spec, rng, batch, device)
        with torch.enable_grad() if arm.startswith("demos") else torch.no_grad():
            _, c, h = step_loss(model, arm, b)
        ok.append(c.detach().float().cpu()); hp.append(h.detach().cpu())
    ok, hp = torch.cat(ok), torch.cat(hp); model.train()
    return {"all": ok.mean().item(), **{f"h{int(k)}": ok[hp == k].mean().item() for k in hp.unique()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--symbols", type=int, default=256); ap.add_argument("--live", type=int, default=32)
    ap.add_argument("--churn", type=float, default=0.0); ap.add_argument("--hops", default="1")
    ap.add_argument("--groups", type=int, default=2); ap.add_argument("--queries", type=int, default=16)
    ap.add_argument("--device", default="mps"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    spec = Spec(symbols=a.symbols, live=a.live, churn=a.churn, hops=tuple(int(x) for x in a.hops.split(",")),
                groups=a.groups, queries=a.queries)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    model = build(a.arm, spec, a.d, a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    params = sum(p.numel() for p in model.parameters())
    g = guessers(spec, np.random.default_rng(a.seed + 1), 100)
    t0, steps, log = time.time(), 0, []
    while time.time() - t0 < a.seconds:
        b = collate(spec, rng, a.batch, a.device)
        loss, c, _ = step_loss(model, a.arm, b)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); steps += 1
        if steps % 50 == 0:
            log.append((steps, round(time.time() - t0, 1), round(loss.item(), 4), round(c.float().mean().item(), 4)))
            print(*log[-1], flush=True)
    res = {"arm": a.arm, "seed": a.seed, "params": params, "steps": steps, "seconds": a.seconds,
           "spec": spec.__dict__, "guessers": g, "eval": evaluate(model, a.arm, spec, a.device, a.seed + 7),
           "live_particles": getattr(model, "last_live", None), "log": log[-5:]}
    print(json.dumps(res, default=str))
    if a.out: open(a.out, "w").write(json.dumps(res, default=str))
