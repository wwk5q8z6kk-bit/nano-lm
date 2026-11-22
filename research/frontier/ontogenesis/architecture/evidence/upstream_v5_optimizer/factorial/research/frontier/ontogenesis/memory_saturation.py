"""Why does the fixed-address memory in cab.py stop acquiring with most slots unused?

Separate diagnostic; cab.py and its results are not changed.

Measured first, no training (seed 3001, the memory exactly as cab.py builds it):
  8,192 possible queries, 16,384 slots. Slots ever addressed: 4,619. Effective number of slots
  (1 / sum of squared address probabilities): 526. Two queries with the same relation share 0.60
  of their 8 slots on average; a uniform hash would give 0.004. Read weights are almost flat
  (top slot 0.164 of 1.0). Rank of the address matrix within one relation: about 840 of 1,024.
So the nominal capacity is misleading: the address function concentrates on a small part of the store.

Competing explanations of the acquisition collapse:
  M1 address concentration   queries pile onto the same slots; each write disturbs many others
  M2 optimiser               a solution exists (the rank is high) but 200 Adam steps cannot reach it
  M3 loss of plasticity      the value table stops responding regardless of addresses
Arms (same stream as cab.py: 1,024 entities, 8 relations, 60 phases, 128 new facts + 16 corrections,
200 Adam steps at 1e-3 on the value table; the one-pass chain loss of cab.py is omitted here):
  original   cab.py's frozen projection addresses
  hash       each (entity, relation) gets 8 distinct uniformly random slots, equal weights
  original_x10  original addresses, learning rate 1e-2
  original_reset  original addresses, optimiser state reset every phase
Seed 3001 only (diagnostic). Recorded per phase: accuracy on this phase's facts after training,
retention of all earlier facts, and the fraction of this phase's facts whose slots were all unused.

Predictions, written before any run of this file:
  D1 original reproduces the collapse: late acquisition below 0.2.
  D2 hash keeps late acquisition above 0.9 and retention above 0.8. If so, M1 is the main cause.
  D3 original_x10 raises late acquisition above original but stays below hash by at least 0.2.
  D4 original_reset does not restore acquisition (within 0.1 of original): M3 in the sense of
     optimiser state is not the cause.
  If hash also collapses, M1 is wrong.
"""
import json, sys
import numpy as np, torch, torch.nn.functional as F
from cab import Net, E, R

def run(arm, seed=3001, phases=60, new=128, fix=16, steps=200, slots=16384, k=8):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = Net("memory")
    s = torch.arange(E).repeat_interleave(R); r = torch.arange(R).repeat(E)
    with torch.no_grad():
        if arm == "hash":
            g = torch.Generator().manual_seed(seed)
            idx = torch.stack([torch.randperm(slots, generator=g)[:k] for _ in range(E * R)]); w = torch.full((E * R, k), 1.0 / k)
        else:
            x = torch.cat([m.ent(s), m.rel(r), m.rel(torch.full_like(r, R))], -1)
            top, idx = (F.normalize(m.q(x), dim=-1) @ F.normalize(m.keys, dim=-1).T).topk(k, -1); w = torch.softmax(top * 8.0, -1)
    V = torch.nn.Parameter(torch.zeros(slots, E)); lr = 1e-2 if arm == "original_x10" else 1e-3
    opt = torch.optim.AdamW([V], lr=lr, weight_decay=0.0)
    read = lambda q: (w[q][..., None] * V[idx[q]]).sum(1)
    table, used, log = {}, torch.zeros(slots, dtype=torch.bool), []
    for ph in range(phases):
        fresh = {}
        while len(fresh) < new:
            key = int(rng.integers(E * R))
            if key not in table and key not in fresh: fresh[key] = int(rng.integers(E))
        fixed = {}
        if table:
            olds = list(table); rng.shuffle(olds)
            for key in olds[:fix]: fixed[key] = int(rng.integers(E))
        old = dict(table); table.update(fresh); table.update(fixed)
        q = torch.tensor(list(fresh) + list(fixed)); y = torch.tensor(list(fresh.values()) + list(fixed.values()))
        virgin = float((~used[idx[torch.tensor(list(fresh))]]).all(1).float().mean())
        if arm == "original_reset": opt = torch.optim.AdamW([V], lr=lr, weight_decay=0.0)
        for _ in range(steps):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        used[idx[q].flatten()] = True
        with torch.no_grad():
            row = {"phase": ph, "acquire": float((read(q[:new]).argmax(-1) == y[:new]).float().mean()), "virgin": round(virgin, 3), "slots_used": int(used.sum())}
            if old:
                oq = torch.tensor(list(old)); oy = torch.tensor([table[kk] for kk in old]); row["retain"] = float((read(oq).argmax(-1) == oy).float().mean())
        log.append(row)
        if ph % 10 == 9: print(arm, row, flush=True)
    late = log[-15:]; mean = lambda kk: round(float(np.mean([x[kk] for x in late if kk in x])), 3)
    return {"arm": arm, "seed": seed, "late_acquire": mean("acquire"), "late_retain": mean("retain"), "early_acquire": round(float(np.mean([x["acquire"] for x in log[1:16]])), 3), "log": log}

if __name__ == "__main__":
    res = run(sys.argv[1]); print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
    json.dump(res, open(f"results/memory_saturation/{sys.argv[1]}.json", "w"))
