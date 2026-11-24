"""Can the memory build type-like addresses from the data instead of being given the types?

Protocol: PROTOCOL-MEMRULE-DERIVED.md (committed before any confirmatory run).
Every earlier test of the typed fixed-address memory gave it each entity's hidden type (a type prototype in
the entity vector). Here the entity vector is built only from what the memory is shown about the entity at
creation: the sum of a fixed random vector for each shown (relation, value) pair. Entities of the same type
share values on the relations they both had shown, so their vectors, and hence their addresses, are similar
without any type label. The vector is fixed at creation; later corrections do not change it.

Arms (single shared path, as memory_rule_sharing.py M = 0, learning rate 1e-3, 200 steps, Adam reset):
  derived   entity vector from shown facts at creation
  typed     the earlier oracle (memory_rule_sharing.py typed M = 0); reused, not rerun
  random    the earlier random vectors (memory_rule_sharing.py random M = 0); reused, not rerun

usage: memory_rule_derived.py SEED OUT.json
"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from cab_s import world, V
from memory_rule_sharing import SLOTS, K, D, PHASES, NEW, FIX, STEPS, LR

torch.set_num_threads(2)


def cluster_online(truth, shown, alpha=0.5, new_cost=2.0):
    """Label-free online grouping, in arrival order. Each cluster keeps counts of (relation, value) among its
    members' shown facts. A new entity joins the cluster with the best smoothed log-likelihood of its shown
    facts, unless a fresh cluster (uniform over values, minus NEW_COST nats) scores better. Uses only facts
    shown at creation; never the hidden type."""
    n, R = truth.shape
    counts, totals, assign = [], [], np.empty(n, int)
    for e in range(n):
        rs = np.flatnonzero(shown[e]); vs = truth[e, rs]
        best, best_s = -1, len(rs) * np.log(1.0 / V) - new_cost
        for c in range(len(counts)):
            s = float(np.sum(np.log((counts[c][rs, vs] + alpha) / (totals[c][rs] + alpha * V))))
            if s > best_s: best, best_s = c, s
        if best < 0:
            counts.append(np.zeros((R, V))); totals.append(np.zeros(R)); best = len(counts) - 1
        counts[best][rs, vs] += 1; totals[best][rs] += 1; assign[e] = best
    return assign


def derived_addresses(seed, truth, shown, codes="derived"):
    n, R = truth.shape
    g = torch.Generator().manual_seed(seed + 30_000)
    pair = torch.randn(R, V, D, generator=g) / D ** 0.5           # one random vector per (relation, value)
    rel = torch.randn(R, D, generator=g) / D ** 0.5
    q = torch.randn(2 * D, D, generator=g) / (2 * D) ** 0.5; keys = torch.randn(SLOTS, D, generator=g) / D ** 0.5
    if codes == "derived":
        ent = torch.zeros(n, D)
        for e in range(n):                                        # truth at creation: no corrections have happened yet
            for r in np.flatnonzero(shown[e]): ent[e] += pair[r, truth[e, r]]
        ent = ent / ent.norm(dim=-1, keepdim=True).clamp_min(1e-9) * (2 ** 0.5)   # same expected norm as earlier arms
    else:                                                         # "clustered": prototype of the learned cluster plus noise
        assign = cluster_online(truth, shown)
        h = torch.Generator().manual_seed(seed + 40_000)
        proto = torch.randn(int(assign.max()) + 1, D, generator=h) / D ** 0.5
        ent = proto[torch.from_numpy(assign)] + torch.randn(n, D, generator=h) / D ** 0.5
    e = torch.arange(n).repeat_interleave(R); r = torch.arange(R).repeat(n)
    x = torch.cat([ent[e], rel[r]], -1) @ q
    idx_parts, top_parts = [], []
    for c in torch.split(torch.arange(n * R), 8192):
        t, i = (F.normalize(x[c], dim=-1) @ F.normalize(keys, dim=-1).T).topk(K, -1); idx_parts.append(i); top_parts.append(t)
    idx, w = torch.cat(idx_parts), torch.softmax(torch.cat(top_parts) * 8.0, -1)
    mass = torch.zeros(SLOTS).index_add_(0, idx.flatten(), w.flatten()); p = mass / mass.sum()
    return idx, w, round(float(1.0 / (p ** 2).sum()), 1)


def run(seed, codes="derived", lr=LR, steps=STEPS):
    rng, truth, shown = world(seed, PHASES, NEW, FIX)
    idx, w, eff = derived_addresses(seed, truth.copy(), shown, codes)
    R = truth.shape[1]
    T = torch.nn.Parameter(torch.zeros(SLOTS, V))
    key = lambda e, r: torch.as_tensor(np.asarray(e) * R + np.asarray(r))
    read = lambda q: (w[q][..., None] * T[idx[q]]).sum(1)
    acc = lambda e, r: float((read(key(e, r)).argmax(-1).numpy() == truth[e, r]).mean()) if len(e) else float("nan")
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
        opt = torch.optim.AdamW([T], lr=lr, weight_decay=0.0)
        for _ in range(steps):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            row = {"phase": ph, "acquire": acc(se, sr)}
            ne, nr = np.nonzero(~shown[ents]); row["infer_new"] = acc(ents[ne], nr)
            if ph:
                oe, orr = np.nonzero(shown[:ph * NEW]); row["retain"] = acc(oe, orr); row["correct"] = acc(fe, fr)
                ue, ur = np.nonzero(~shown[:ph * NEW]); row["infer_old"] = acc(ue, ur)
        log.append(row)
    late = log[-(PHASES // 4):]; mean = lambda kk: round(float(np.nanmean([x.get(kk, np.nan) for x in late])), 4)
    return {"codes": codes, "seed": seed, "effective_slots": eff,
            "late": {k: mean(k) for k in ("acquire", "infer_new", "infer_old", "retain", "correct")}, "log": log}


if __name__ == "__main__":
    seed, out = int(sys.argv[1]), sys.argv[2]
    codes = sys.argv[3] if len(sys.argv) > 3 else "derived"
    f = open(out, "x"); t = time.time(); res = run(seed, codes)
    res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__
    json.dump(res, f); f.close()
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
