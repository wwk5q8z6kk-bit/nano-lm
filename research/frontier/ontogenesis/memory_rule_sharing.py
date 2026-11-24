"""Does spreading a fixed-address memory's reads cost generalisation when its addresses carry structure?

Protocol: PROTOCOL-MEMRULE-SHARING.md (committed before any confirmatory run).
The memory is cab.py's fixed-address value table (frozen projection addresses, 16,384 slots, read 8 at a
time, Adam reset each phase, 200 steps at 1e-3). The world is cab_s.py's (hidden types, rules, 10%
exceptions, 3 of 8 relations shown, 16 corrections per phase). In cab.py the entity vectors that feed the
addresses are random, so addresses carry no type information and there is nothing for spreading to destroy.
Here entity vectors are either random ("random") or a type prototype plus noise of equal expected norm
("typed"). "typed" gives the memory type information for free; it is a mechanism probe, not a learner to
compare with cab_s.py's.

usage: memory_rule_sharing.py CODES M SEED OUT.json     CODES in {random, typed}; M in 0..8
Result files are created exclusively and never overwritten.
"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from cab_s import world, Z, R, V

torch.set_num_threads(2)
SLOTS, K, D = 16384, 8, 128
PHASES, NEW, FIX, STEPS, LR = 40, 64, 16, 200, 1e-3


def addresses(codes, m, seed, n):
    rng0 = np.random.default_rng(seed); rng0.integers(V, size=(Z, R)); typ = rng0.integers(Z, size=n)  # same draws as cab_s.world
    g = torch.Generator().manual_seed(seed + 30_000)
    proto = torch.randn(Z, D, generator=g) / D ** 0.5; noise = torch.randn(n, D, generator=g) / D ** 0.5
    ent = proto[torch.from_numpy(typ)] + noise if codes == "typed" else noise * 2 ** 0.5   # equal expected norm
    rel = torch.randn(R, D, generator=g) / D ** 0.5
    q = torch.randn(2 * D, D, generator=g) / (2 * D) ** 0.5; keys = torch.randn(SLOTS, D, generator=g) / D ** 0.5
    e = torch.arange(n).repeat_interleave(R); r = torch.arange(R).repeat(n)
    x = torch.cat([ent[e], rel[r]], -1) @ q
    idx_parts, top_parts = [], []
    for c in torch.split(torch.arange(n * R), 8192):
        t, i = (F.normalize(x[c], dim=-1) @ F.normalize(keys, dim=-1).T).topk(K, -1); idx_parts.append(i); top_parts.append(t)
    idx, w = torch.cat(idx_parts), torch.softmax(torch.cat(top_parts) * 8.0, -1)
    if m:
        h = torch.Generator().manual_seed(seed + 10_000)
        idx = idx.clone()
        for i in range(n * R):
            pos = torch.randperm(K, generator=h)[:m]
            keep = set(idx[i].tolist()) - set(idx[i, pos].tolist())
            idx[i, pos] = torch.tensor([c for c in torch.randperm(SLOTS, generator=h)[:K + m].tolist() if c not in keep][:m])
    mass = torch.zeros(SLOTS).index_add_(0, idx.flatten(), w.flatten()); p = mass / mass.sum()
    return idx, w, typ, round(float(1.0 / (p ** 2).sum()), 1)


def run(codes, m, seed):
    rng, truth, shown = world(seed, PHASES, NEW, FIX)
    n = PHASES * NEW
    idx, w, typ, eff = addresses(codes, m, seed, n)
    key = lambda e, r: torch.as_tensor(np.asarray(e) * R + np.asarray(r))
    Vt = torch.nn.Parameter(torch.zeros(SLOTS, V))
    read = lambda q: (w[q][..., None] * Vt[idx[q]]).sum(1)
    pred = lambda e, r: read(key(e, r)).argmax(-1).numpy()
    acc = lambda e, r: float((pred(e, r) == truth[e, r]).mean()) if len(e) else float("nan")
    log = []
    for ph in range(PHASES):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        se, sr = np.nonzero(shown[ents]); se = ents[se]
        fe = fr = np.zeros(0, int)
        if ph:
            oe, orr = np.nonzero(shown[:ph * NEW]); pick = rng.choice(len(oe), FIX, replace=False)
            fe, fr = oe[pick], orr[pick]; truth[fe, fr] = rng.integers(V, size=FIX)
        qe, qr = np.concatenate([se, fe]), np.concatenate([sr, fr])
        q, y = key(qe, qr), torch.from_numpy(truth[qe, qr])
        opt = torch.optim.AdamW([Vt], lr=LR, weight_decay=0.0)   # reset each phase
        for _ in range(STEPS):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            row = {"phase": ph, "acquire": acc(se, sr)}
            ne, nr = np.nonzero(~shown[ents]); row["infer_new"] = acc(ents[ne], nr)
            if ph:
                oe, orr = np.nonzero(shown[:ph * NEW]); row["retain"] = acc(oe, orr); row["correct"] = acc(fe, fr)
                ue, ur = np.nonzero(~shown[:ph * NEW]); row["infer_old"] = acc(ue, ur)
        log.append(row)
    late = log[-(PHASES // 4):]; mean = lambda kk: round(float(np.nanmean([x.get(kk, np.nan) for x in late])), 4)
    rule = np.random.default_rng(seed).integers(V, size=(Z, R))
    return {"codes": codes, "m": m, "seed": seed, "effective_slots": eff,
            "late": {k: mean(k) for k in ("acquire", "infer_new", "infer_old", "retain", "correct")},
            "rule_ceiling_note": "infer ceiling about 1 - EPS = 0.90; chance 1/32", "log": log}


if __name__ == "__main__":
    codes, m, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    f = open(out, "x"); t = time.time(); res = run(codes, m, seed)
    res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__
    json.dump(res, f); f.close()
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
