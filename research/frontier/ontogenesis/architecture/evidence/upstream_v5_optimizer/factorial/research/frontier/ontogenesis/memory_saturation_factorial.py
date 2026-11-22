"""Address x optimiser factorial for the fixed-address acquisition collapse.

Protocol: PROTOCOL-MEMSAT-FACTORIAL.md (committed before any confirmatory run).
Follow-up to memory_saturation.py (one seed, 3001). That file and its results are not changed.
The training stream, model, step budget and metrics are copied from memory_saturation.run so the
"adam" and "adam_reset" cells are the same procedure as its "original"/"hash" and
"original_reset" arms. Result files are created exclusively and never overwritten.

usage: memory_saturation_factorial.py ADDRESS OPT SEED OUT.json [--lr LR]
  ADDRESS  original | hash
  OPT      adam | adam_reset | adam_b2_09 | sgd
"""
import argparse, json, time
import numpy as np, torch, torch.nn.functional as F
from cab import Net, E, R

torch.set_num_threads(2)


def make_opt(kind, V, lr):
    if kind in ("adam", "adam_reset"):
        return torch.optim.AdamW([V], lr=lr, weight_decay=0.0)
    if kind == "adam_b2_09":
        return torch.optim.AdamW([V], lr=lr, betas=(0.9, 0.9), weight_decay=0.0)
    if kind == "sgd":
        return torch.optim.SGD([V], lr=lr)
    raise ValueError(kind)


def run(address, opt_kind, seed, lr, phases=60, new=128, fix=16, steps=200, slots=16384, k=8):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); m = Net("memory")
    s = torch.arange(E).repeat_interleave(R); r = torch.arange(R).repeat(E)
    with torch.no_grad():
        if address == "hash":
            g = torch.Generator().manual_seed(seed)
            idx = torch.stack([torch.randperm(slots, generator=g)[:k] for _ in range(E * R)]); w = torch.full((E * R, k), 1.0 / k)
        elif address == "original":
            x = torch.cat([m.ent(s), m.rel(r), m.rel(torch.full_like(r, R))], -1)
            top, idx = (F.normalize(m.q(x), dim=-1) @ F.normalize(m.keys, dim=-1).T).topk(k, -1); w = torch.softmax(top * 8.0, -1)
        else:
            raise ValueError(address)
    V = torch.nn.Parameter(torch.zeros(slots, E))
    opt = make_opt(opt_kind, V, lr)
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
        if opt_kind == "adam_reset": opt = make_opt(opt_kind, V, lr)
        for _ in range(steps):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        used[idx[q].flatten()] = True
        with torch.no_grad():
            row = {"phase": ph, "acquire": float((read(q[:new]).argmax(-1) == y[:new]).float().mean()),
                   "final_loss": float(loss), "slots_used": int(used.sum())}
            if old:
                oq = torch.tensor(list(old)); oy = torch.tensor([table[kk] for kk in old])
                row["retain"] = float((read(oq).argmax(-1) == oy).float().mean())
        log.append(row)
    late = log[-15:]; mean = lambda kk: round(float(np.mean([x[kk] for x in late if kk in x])), 4)
    return {"address": address, "opt": opt_kind, "seed": seed, "lr": lr,
            "late_acquire": mean("acquire"), "late_retain": mean("retain"),
            "early_acquire": round(float(np.mean([x["acquire"] for x in log[1:16]])), 4), "log": log}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("address"); ap.add_argument("opt"); ap.add_argument("seed", type=int); ap.add_argument("out")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    f = open(a.out, "x")  # refuse to overwrite an existing result before spending the compute
    t = time.time(); res = run(a.address, a.opt, a.seed, a.lr)
    res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__; res["threads"] = torch.get_num_threads()
    json.dump(res, f); f.close()
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
