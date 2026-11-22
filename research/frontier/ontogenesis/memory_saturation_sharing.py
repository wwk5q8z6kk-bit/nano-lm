"""Does slot sharing explain the acquisition gap that remains without carried optimiser state?

Protocol: PROTOCOL-MEMSAT-SHARING.md (committed before any confirmatory run).
Follow-up to memory_saturation_factorial.py. Dose-response: start from cab.py's projection
addresses and replace M of each query's 8 slots (random positions) with uniformly random slots,
keeping the query's read weights. M=0 is the original address function; M=8 keeps only its weights.

usage:
  memory_saturation_sharing.py measure SEED            # no training; prints address statistics
  memory_saturation_sharing.py train M SEED OUT.json   # Adam reset each phase, lr 1e-3
Result files are created exclusively and never overwritten.
"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from cab import Net, E, R

torch.set_num_threads(2)
SLOTS, K = 16384, 8


def addresses(seed, m):
    """Identical to memory_saturation.run for m == 0 (same RNG use before this point)."""
    torch.manual_seed(seed); net = Net("memory")
    s = torch.arange(E).repeat_interleave(R); r = torch.arange(R).repeat(E)
    with torch.no_grad():
        x = torch.cat([net.ent(s), net.rel(r), net.rel(torch.full_like(r, R))], -1)
        top, idx = (F.normalize(net.q(x), dim=-1) @ F.normalize(net.keys, dim=-1).T).topk(K, -1)
        w = torch.softmax(top * 8.0, -1)
    if m:
        g = torch.Generator().manual_seed(seed + 10_000)  # separate stream: the fact stream is untouched
        idx = idx.clone()
        for i in range(E * R):
            pos = torch.randperm(K, generator=g)[:m]
            keep = set(idx[i].tolist()) - set(idx[i, pos].tolist())
            cand = [c for c in torch.randperm(SLOTS, generator=g)[:K + m].tolist() if c not in keep][:m]
            idx[i, pos] = torch.tensor(cand)
    return idx, w, r


def measure(seed, m, pairs=20000):
    idx, w, r = addresses(seed, m)
    mass = torch.zeros(SLOTS).index_add_(0, idx.flatten(), w.flatten()); p = mass / mass.sum()
    rng = np.random.default_rng(seed + 20_000)
    a = rng.integers(E * R, size=pairs); b = rng.integers(E * R, size=pairs); ok = a != b
    a, b = a[ok], b[ok]
    share = (idx[a][:, :, None] == idx[b][:, None, :]).any(-1).float().mean(1).numpy()
    same = (r[a] == r[b]).numpy()
    return {"seed": seed, "m": m, "slots_addressed": int((mass > 0).sum()),
            "effective_slots": round(float(1.0 / (p ** 2).sum()), 1),
            "overlap_same_relation": round(float(share[same].mean()), 4),
            "overlap_other_relation": round(float(share[~same].mean()), 4),
            "top_weight_mean": round(float(w.max(1).values.mean()), 4)}


def train(m, seed, phases=60, new=128, fix=16, steps=200, lr=1e-3):
    idx, w, _ = addresses(seed, m)
    rng = np.random.default_rng(seed)
    V = torch.nn.Parameter(torch.zeros(SLOTS, E))
    read = lambda q: (w[q][..., None] * V[idx[q]]).sum(1)
    table, log = {}, []
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
        opt = torch.optim.AdamW([V], lr=lr, weight_decay=0.0)  # reset each phase
        loss = torch.zeros(())
        for _ in range(steps):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            row = {"phase": ph, "acquire": float((read(q[:new]).argmax(-1) == y[:new]).float().mean()), "final_loss": float(loss)}
            if old:
                oq = torch.tensor(list(old)); oy = torch.tensor([table[kk] for kk in old])
                row["retain"] = float((read(oq).argmax(-1) == oy).float().mean())
        log.append(row)
    late = log[-15:]; mean = lambda kk: round(float(np.mean([x[kk] for x in late if kk in x])), 4)
    return {"m": m, "opt": "adam_reset", "seed": seed, "lr": lr, "late_acquire": mean("acquire"), "late_retain": mean("retain"),
            "early_acquire": round(float(np.mean([x["acquire"] for x in log[1:16]])), 4), "address_stats": measure(seed, m), "log": log}


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        for m in (0, 2, 4, 6, 8): print(json.dumps(measure(int(sys.argv[2]), m)), flush=True)
    else:
        m, seed, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        f = open(out, "x"); t = time.time(); res = train(m, seed)
        res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__
        json.dump(res, f); f.close()
        print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
