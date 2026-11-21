"""PLASTICITY v0: does a network keep its ability to learn across a long sequence of tasks?

Each task: memorise 256 random (symbol -> class) facts, tagged with a task token.
The same network is trained on tasks one after another, with a fixed step cap per task.
Measured per task: steps to reach 95% on that task (cap if never). Measured at the end:
accuracy on every earlier task without retraining (retention).

Predictions written before the first run:
  P1 a dense MLP needs more steps on later tasks than on early ones (loss of plasticity),
     and retains little of earlier tasks.
  P2 a sparse key-value memory (top-k of many slots) keeps steps roughly flat and retains more,
     because each fact writes to few slots.
Either prediction can fail. If the MLP shows no slowdown, this version is underpowered and says so.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

S, C, FACTS = 512, 64, 256


class MLP(nn.Module):
    def __init__(self, tasks, d=64, h=256, norm=False):
        super().__init__()
        self.sym, self.task = nn.Embedding(S, d), nn.Embedding(tasks, d)
        layers = [nn.Linear(2 * d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU()]
        if norm: layers.insert(1, nn.LayerNorm(h)); layers.insert(4, nn.LayerNorm(h))
        self.net = nn.Sequential(*layers); self.out = nn.Linear(h, C)

    def forward(self, t, x): return self.out(self.net(torch.cat([self.task(t), self.sym(x)], -1)))


class SparseMemory(nn.Module):
    """Query -> top-k of `slots` learned keys -> weighted sum of slot values -> classes."""
    def __init__(self, tasks, d=64, slots=8192, k=8, frozen_address=False):
        super().__init__()
        self.sym, self.task = nn.Embedding(S, d), nn.Embedding(tasks, d)
        self.q = nn.Linear(2 * d, d)
        self.keys = nn.Parameter(torch.randn(slots, d) / d ** 0.5)
        self.values = nn.Embedding(slots, C); nn.init.zeros_(self.values.weight)
        self.k = k
        if frozen_address:      # addresses never change: embeddings, query map and keys are fixed at initialisation
            for prm in (self.sym.weight, self.task.weight, self.q.weight, self.q.bias, self.keys): prm.requires_grad_(False)

    def forward(self, t, x):
        q = F.normalize(self.q(torch.cat([self.task(t), self.sym(x)], -1)), dim=-1)
        s = q @ F.normalize(self.keys, dim=-1).T
        top, idx = s.topk(self.k, -1)
        return (torch.softmax(top * 8.0, -1).unsqueeze(-1) * self.values(idx)).sum(1)


def run(arm, tasks, cap, seed, lr):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = {"mlp": lambda: MLP(tasks), "mlp_norm_wd": lambda: MLP(tasks, norm=True),
             "sparse_memory": lambda: SparseMemory(tasks),
             "sparse_memory_frozen_address": lambda: SparseMemory(tasks, frozen_address=True)}[arm]()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01 if arm == "mlp_norm_wd" else 0.0)
    data, steps_needed = [], []
    for t in range(tasks):
        x = torch.from_numpy(rng.choice(S, FACTS, replace=False)); y = torch.from_numpy(rng.integers(0, C, FACTS))
        tt = torch.full((FACTS,), t); data.append((tt, x, y)); need = cap
        for step in range(1, cap + 1):
            logits = model(tt, x); loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 5 == 0 and (model(tt, x).argmax(-1) == y).float().mean() >= 0.95: need = step; break
        steps_needed.append(need)
    with torch.no_grad():
        retention = [float((model(a, b).argmax(-1) == c).float().mean()) for a, b, c in data]
    q = tasks // 4
    return {"arm": arm, "seed": seed, "tasks": tasks, "cap": cap, "lr": lr,
            "params": sum(p.numel() for p in model.parameters()),
            "steps_first_quarter": float(np.mean(steps_needed[:q])), "steps_last_quarter": float(np.mean(steps_needed[-q:])),
            "never_reached_last_quarter": float(np.mean(np.array(steps_needed[-q:]) >= cap)),
            "retention_earliest_quarter": float(np.mean(retention[:q])), "retention_all": float(np.mean(retention[:-1])),
            "steps": steps_needed}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=200); ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=""); ap.add_argument("--arms", default="mlp,mlp_norm_wd,sparse_memory")
    a = ap.parse_args(); rows = []
    ap2 = a.arms.split(",")
    for arm in ap2:
        for seed in range(a.seeds):
            t0 = time.time(); r = run(arm, a.tasks, a.cap, 2000 + seed, a.lr); r["seconds"] = round(time.time() - t0, 1)
            rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "steps"}, flush=True)
    if a.out: json.dump(rows, open(a.out, "w"))
