"""What was lost when the learned-address memory forgot? Three notions of retention, measured separately.

Setup: the PLASTICITY task sequence (fresh 256 facts per task), fewer tasks.
For the sparse memory with learned addresses we snapshot the addressing parameters
(embeddings, query map, keys) at the end of each task. After the whole sequence:

  current      accuracy on an old task with today's addressing and today's values
  old_address  accuracy with that task's own snapshot of addressing, but today's values
  savings      steps to relearn an old task, against steps to learn a brand-new one

Readings stated before the run:
  drift      old_address is high while current is at chance: the values survived, the path to them moved
  overwrite  old_address is also near chance: the stored values themselves were destroyed
  both       old_address is well above current but well below 100%
For the plain network (with normalisation and weight decay) only `current` and `savings` apply, plus
  probe      accuracy after refitting only the output layer on the old task (were the features kept?)
"""
from __future__ import annotations
import copy, json, sys
import numpy as np, torch, torch.nn.functional as F
from plasticity import MLP, SparseMemory, S, C, FACTS

TASKS, CAP = 80, 300
ADDR = ("sym.weight", "task.weight", "q.weight", "q.bias", "keys")


def fit(model, opt, tt, x, y, cap=CAP):
    for step in range(1, cap + 1):
        loss = F.cross_entropy(model(tt, x), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 5 == 0 and (model(tt, x).argmax(-1) == y).float().mean() >= 0.95: return step
    return cap


def acc(model, tt, x, y):
    with torch.no_grad(): return float((model(tt, x).argmax(-1) == y).float().mean())


def run(kind, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = SparseMemory(TASKS + 8) if kind == "memory" else MLP(TASKS + 8, norm=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0 if kind == "memory" else 0.01)
    data, snaps, steps = [], [], []
    for t in range(TASKS):
        x = torch.from_numpy(rng.choice(S, FACTS, replace=False)); y = torch.from_numpy(rng.integers(0, C, FACTS)); tt = torch.full((FACTS,), t)
        data.append((tt, x, y)); steps.append(fit(model, opt, tt, x, y))
        if kind == "memory": snaps.append({k: v.clone() for k, v in model.state_dict().items() if k in ADDR})
    old = list(range(0, TASKS // 2, 4))                                            # ten old tasks
    out = {"kind": kind, "seed": seed, "steps_last_ten": float(np.mean(steps[-10:])),
           "current": float(np.mean([acc(model, *data[t]) for t in old]))}
    if kind == "memory":
        now = copy.deepcopy(model.state_dict()); vals = []
        for t in old:
            model.load_state_dict({**now, **snaps[t]}); vals.append(acc(model, *data[t]))
        model.load_state_dict(now); out["old_address"] = float(np.mean(vals))
    else:
        vals = []
        for t in old:                                                              # refit only the output layer
            m2 = copy.deepcopy(model); o2 = torch.optim.AdamW(m2.out.parameters(), lr=1e-3)
            fit(m2, o2, *data[t], cap=100); vals.append(acc(m2, *data[t]))
        out["probe"] = float(np.mean(vals))
    relearn, fresh = [], []
    for i, t in enumerate(old[:5]):                                                # savings: relearn old, learn new, on copies
        m2 = copy.deepcopy(model); o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3); relearn.append(fit(m2, o2, *data[t]))
        m3 = copy.deepcopy(model); o3 = torch.optim.AdamW(m3.parameters(), lr=1e-3)
        x = torch.from_numpy(rng.choice(S, FACTS, replace=False)); y = torch.from_numpy(rng.integers(0, C, FACTS))
        fresh.append(fit(m3, o3, torch.full((FACTS,), TASKS + i), x, y))
    out["relearn_steps"], out["fresh_steps"] = float(np.mean(relearn)), float(np.mean(fresh))
    return out


if __name__ == "__main__":
    rows = []
    for kind in ("memory", "mlp"):
        for seed in (4001, 4002):
            r = run(kind, seed); rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    if len(sys.argv) > 1: json.dump(rows, open(sys.argv[1], "w"), indent=1)
