"""Revisable addresses without full replay: move only what changed, using the memory's own predictions.

Protocol: PROTOCOL-MEMRULE-TRANSPORT.md (committed before any confirmatory run).
Follows RESULTS-MEMRULE-RECLUSTER.md: re-clustering with a full re-index (value table reset, then retrained on
an exact store of every current fact) recovers the given-type inference, at the cost of replaying 34,560 facts.
Here the table is never reset and no exact store of values is kept. Every EVERY phases:
  1. re-cluster all entities seen so far by EM (as memory_rule_recluster.py, KC clusters);
  2. match the new cluster labels to the old ones (Hungarian assignment on co-membership), so that an entity
     keeps its prototype unless its group really changed;
  3. for entities whose matched cluster changed ("moved"), rebuild their addresses, and:
       transport  write into the new addresses the memory's own soft prediction read from the old addresses
                  (KL to the old read's softmax), for the keys the memory has written (an index of written
                  (entity, relation) keys, no values), TRANSPORT_STEPS full-batch steps on moved keys only
       drop       rebuild the addresses and write nothing: moved entities' facts are lost from the read
Unmoved entities keep their addresses and values untouched.

usage: memory_rule_transport.py MODE SEED OUT.json [KC] [EVERY] [TRANSPORT_STEPS]     MODE in {transport, drop}
"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from cab_s import world, V
from memory_rule_sharing import SLOTS, K, D, PHASES, NEW, FIX, STEPS, LR
from memory_rule_recluster import features, em

torch.set_num_threads(2)


def match(old, new, KC):
    """Relabel NEW to agree with OLD as much as possible (max co-membership)."""
    C = np.zeros((KC, KC))
    np.add.at(C, (new, old), 1)
    rows, cols = linear_sum_assignment(-C)
    perm = np.empty(KC, int); perm[rows] = cols
    return perm[new]


def run(mode, seed, KC=32, every=5, transport_steps=1000, lr=LR):
    rng, truth, shown = world(seed, PHASES, NEW, FIX)
    n, R = truth.shape
    truth0 = truth.copy()
    g = torch.Generator().manual_seed(seed + 30_000)
    rel = torch.randn(R, D, generator=g) / D ** 0.5
    q = torch.randn(2 * D, D, generator=g) / (2 * D) ** 0.5; keys = torch.randn(SLOTS, D, generator=g) / D ** 0.5
    h = torch.Generator().manual_seed(seed + 40_000)
    proto = torch.randn(KC, D, generator=h) / D ** 0.5; noise = torch.randn(n, D, generator=h) / D ** 0.5
    assign = np.zeros(n, int); pi = np.ones(KC) / KC; th = np.full((KC, R, V), 1.0 / V)
    idx = torch.zeros(n * R, K, dtype=torch.long); w = torch.zeros(n * R, K)
    written = np.zeros((n, R), bool)                               # index of written keys; no values kept

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

    r0 = np.random.default_rng(seed); r0.integers(V, size=(16, R)); typ = r0.integers(16, size=n)
    log, moved_total, keys_moved, purity_log = [], 0, 0, []
    for ph in range(PHASES):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        X, M = features(truth0, shown, ents)
        ll = np.log(pi)[None] + np.einsum('nrv,krv->nk', X, np.log(th)); assign[ents] = ll.argmax(1)
        address(ents)
        se, sr = np.nonzero(shown[ents]); se = ents[se]
        fe = fr = np.zeros(0, int)
        if ph:
            oe, orr = np.nonzero(shown[:ph * NEW]); pick = rng.choice(len(oe), FIX, replace=False)
            fe, fr = oe[pick], orr[pick]; truth[fe, fr] = rng.integers(V, size=FIX)
        qe, qr = np.concatenate([se, fe]), np.concatenate([sr, fr]); fit(qe, qr, STEPS); written[qe, qr] = True
        row = {"phase": ph}
        if (ph + 1) % every == 0:
            seen = np.arange((ph + 1) * NEW)
            Xs, Ms = features(truth0, shown, seen)
            new, pi, th = em(Xs, Ms, KC)
            new = match(assign[seen], new, KC)
            moved = seen[new != assign[seen]]
            row["moved"] = int(len(moved)); moved_total += len(moved)
            # re-estimate pi and th under the matched labels, so later assignments use the same label space
            resp = np.eye(KC)[new]
            pi = resp.mean(0) + 1e-9
            th = (np.einsum('nk,nrv->krv', resp, Xs) + 0.05) / (np.einsum('nk,nr->kr', resp, Ms)[..., None] + 0.05 * V)
            if len(moved):
                me, mr = np.nonzero(written[moved]); me = moved[me]
                with torch.no_grad(): target = torch.softmax(read(key(me, mr)), -1)   # old addresses' own prediction
                assign[moved] = new[np.isin(seen, moved)]
                address(moved)
                if mode == "transport" and len(me):
                    qq = key(me, mr); opt = torch.optim.AdamW([T], lr=lr, weight_decay=0.0)
                    for _ in range(transport_steps):
                        loss = F.kl_div(F.log_softmax(read(qq), -1), target, reduction="batchmean")
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    keys_moved += len(me)
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
    return {"mode": mode, "seed": seed, "KC": KC, "every": every, "transport_steps": transport_steps,
            "entities_moved": int(moved_total), "keys_transported": int(keys_moved),
            "final_purity": purity_log[-1] if purity_log else None,
            "late": {k: mean(k) for k in ("acquire", "infer_new", "infer_old", "retain", "correct")}, "log": log}


if __name__ == "__main__":
    mode, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    KC = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    every = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    ts = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
    f = open(out, "x"); t = time.time(); res = run(mode, seed, KC, every, ts)
    res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__
    json.dump(res, f); f.close()
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
