"""Revisable addresses: does periodic re-clustering with a full re-index recover the given-type generalisation?

Protocol: PROTOCOL-MEMRULE-RECLUSTER.md (committed before any confirmatory run).
Follows RESULTS-MEMRULE-DERIVED.md: addresses built from shown facts and frozen at creation recover a third
of the given-type inference, and greedy online clustering collapses. On the development seed, batch EM over all
entities' shown facts recovers the hidden types with purity 0.91 after 5 phases and 0.99 after 40.

Memory: single shared path as memory_rule_sharing.py (16,384 slots read 8 at a time, Adam 1e-3, 200 steps per
phase on that phase's facts, reset each phase). Entity vectors = learned-cluster prototype + noise, the same
construction as the typed arm with the hidden type replaced by an inferred cluster. Every EVERY phases:
  1. re-cluster all entities seen so far by EM (mixture of categoricals over their facts shown at creation;
     KC clusters, 5 restarts, no type label), seeded from the previous assignment's best restart;
  2. rebuild every address from the new assignment;
  3. re-index: reset the value table and retrain it on the exact store of all current facts written so far
     (shown facts with their latest values), REINDEX_STEPS full-batch steps.
Between re-indexes, a new entity takes the cluster with the highest posterior under the current EM parameters.
The exact store and the full retrain are a cost, reported as facts replayed; this is the "full re-index"
baseline of the blueprint's Family A, not an online transport.

usage: memory_rule_recluster.py SEED OUT.json [KC] [EVERY] [REINDEX_STEPS] [typed]
  "typed" as the sixth argument runs the control: given hidden types in place of EM, same re-index schedule.
"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from cab_s import world, V
from memory_rule_sharing import SLOTS, K, D, PHASES, NEW, FIX, STEPS, LR
from memory_rule_factsplit import fact_split

torch.set_num_threads(2)


def features(truth0, shown, ents):
    R = truth0.shape[1]
    X = np.zeros((len(ents), R, V)); M = shown[ents].astype(float)
    for i, e in enumerate(ents):
        for r in np.flatnonzero(shown[e]): X[i, r, truth0[e, r]] = 1
    return X, M


def em(X, M, KC, restarts=5, iters=60, a=0.05):
    best = None
    for rs in range(restarts):
        g = np.random.default_rng(rs); resp = g.dirichlet(np.ones(KC), size=len(X))
        for _ in range(iters):
            pi = resp.mean(0) + 1e-9
            th = (np.einsum('nk,nrv->krv', resp, X) + a) / (np.einsum('nk,nr->kr', resp, M)[..., None] + a * V)
            ll = np.log(pi)[None] + np.einsum('nrv,krv->nk', X, np.log(th))
            mx = ll.max(1, keepdims=True); resp = np.exp(ll - mx); resp /= resp.sum(1, keepdims=True)
        tot = float(np.sum(mx[:, 0] + np.log(np.exp(ll - mx).sum(1))))
        if best is None or tot > best[0]: best = (tot, resp.argmax(1), pi, th)
    return best[1], best[2], best[3]


def run(seed, KC=16, every=5, reindex_steps=1000, lr=LR, oracle=False):
    rng, truth, shown = world(seed, PHASES, NEW, FIX)
    n, R = truth.shape
    truth0 = truth.copy()                                         # values at creation: the only thing clustering sees
    g = torch.Generator().manual_seed(seed + 30_000)
    rel = torch.randn(R, D, generator=g) / D ** 0.5
    q = torch.randn(2 * D, D, generator=g) / (2 * D) ** 0.5; keys = torch.randn(SLOTS, D, generator=g) / D ** 0.5
    h = torch.Generator().manual_seed(seed + 40_000)
    proto = torch.randn(KC, D, generator=h) / D ** 0.5; noise = torch.randn(n, D, generator=h) / D ** 0.5
    assign = np.zeros(n, int); pi = np.ones(KC) / KC; th = np.full((KC, R, V), 1.0 / V)
    idx = torch.zeros(n * R, K, dtype=torch.long); w = torch.zeros(n * R, K)

    def address(ents):
        ents = np.asarray(ents)
        if not len(ents): return
        ent = proto[torch.from_numpy(assign[ents])] + noise[torch.from_numpy(ents)]
        e = torch.from_numpy(ents).repeat_interleave(R); r = torch.arange(R).repeat(len(ents))
        x = torch.cat([ent.repeat_interleave(R, 0), rel[r]], -1) @ q
        t, i = (F.normalize(x, dim=-1) @ F.normalize(keys, dim=-1).T).topk(K, -1)
        rows = e * R + r; idx[rows] = i; w[rows] = torch.softmax(t * 8.0, -1)

    T = torch.nn.Parameter(torch.zeros(SLOTS, V))
    key = lambda e, r: torch.as_tensor(np.asarray(e) * R + np.asarray(r))
    read = lambda qq: (w[qq][..., None] * T[idx[qq]]).sum(1)
    acc = lambda e, r: float((read(key(e, r)).argmax(-1).numpy() == truth[e, r]).mean()) if len(e) else float("nan")

    def fit(qe, qr, n_steps):
        qq, y = key(qe, qr), torch.from_numpy(truth[qe, qr])
        opt = torch.optim.AdamW([T], lr=lr, weight_decay=0.0)
        for _ in range(n_steps):
            loss = F.cross_entropy(read(qq), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    log, replayed, purity_log = [], 0, []
    r0 = np.random.default_rng(seed); rule = r0.integers(V, size=(16, R)); typ = r0.integers(16, size=n)
    corrected = np.zeros((n, R), bool)
    if oracle:
        assert KC >= 16, "the typed control needs one prototype per hidden type"
    for ph in range(PHASES):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        if oracle:
            assign[ents] = typ[ents]                                 # control: given types, same re-index schedule
        else:
            X, M = features(truth0, shown, ents)
            ll = np.log(pi)[None] + np.einsum('nrv,krv->nk', X, np.log(th)); assign[ents] = ll.argmax(1)
        address(ents)
        se, sr = np.nonzero(shown[ents]); se = ents[se]
        fe = fr = np.zeros(0, int)
        if ph:
            oe, orr = np.nonzero(shown[:ph * NEW]); pick = rng.choice(len(oe), FIX, replace=False)
            fe, fr = oe[pick], orr[pick]; truth[fe, fr] = rng.integers(V, size=FIX); corrected[fe, fr] = True
        fit(np.concatenate([se, fe]), np.concatenate([sr, fr]), STEPS)
        row = {"phase": ph}
        if (ph + 1) % every == 0:
            seen = np.arange((ph + 1) * NEW)
            if not oracle:
                Xs, Ms = features(truth0, shown, seen)
                assign[seen], pi, th = em(Xs, Ms, KC)
            address(seen)
            with torch.no_grad(): T.zero_()
            ae, ar = np.nonzero(shown[seen]); fit(ae, ar, reindex_steps); replayed += len(ae)
            pur = sum(np.bincount(typ[seen][assign[seen] == c]).max() for c in set(assign[seen])) / len(seen)
            row["purity"] = round(float(pur), 4); purity_log.append(row["purity"])
        with torch.no_grad():
            row["acquire"] = acc(se, sr)
            ne, nr = np.nonzero(~shown[ents]); row["infer_new"] = acc(ents[ne], nr)
            if ph:
                oe, orr = np.nonzero(shown[:ph * NEW]); row["retain"] = acc(oe, orr); row["correct"] = acc(fe, fr)
                ue, ur = np.nonzero(~shown[:ph * NEW]); row["infer_old"] = acc(ue, ur)
        log.append(row)
    late = log[-(PHASES // 4):]; mean = lambda kk: round(float(np.nanmean([x.get(kk, np.nan) for x in late])), 4)
    with torch.no_grad(): final_split = fact_split(acc, truth0, rule, typ, shown, corrected)
    return {"codes": "typed_reindexed" if oracle else "reclustered", "seed": seed, "KC": KC, "every": every, "reindex_steps": reindex_steps,
            "facts_replayed": int(replayed), "final_purity": purity_log[-1] if purity_log else None,
            "late": {k: mean(k) for k in ("acquire", "infer_new", "infer_old", "retain", "correct")},
            "final_split": final_split, "log": log}


if __name__ == "__main__":
    seed, out = int(sys.argv[1]), sys.argv[2]
    KC = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    every = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    rs = int(sys.argv[5]) if len(sys.argv) > 5 else 1000
    oracle = len(sys.argv) > 6 and sys.argv[6] == "typed"
    f = open(out, "x"); t = time.time(); res = run(seed, KC, every, rs, oracle=oracle)
    res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__
    json.dump(res, f); f.close()
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
