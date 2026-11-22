"""G04, second study: a label-free rule-change detector. Can it close the gap to the ceiling of the first study?

NOT YET RUN AND NOT YET REGISTERED. Predictions are added after a mechanics check on development seed 1,
and before any run on registered seeds 7201-7203 (new seeds: the first study's seeds are not reused,
because its results are known).

Background (RULE-CHANGE-20261122.md). In a world where rules change, a seen-twice policy keeps exceptions
(100%) and resists noise (96%) without labels, but revises changed rules no better than a plain learner
(52% of old entities answer the new value). An oracle that knows which (type, relation) changed and
retrains the shared decoder with old addresses frozen reaches 97-100%. It needs the hidden types.

Detector (label-free; uses only observations and the model's own answers).
  1. Train the phase's new entities' facts as usual (the seen-twice policy handles old-entity observations).
  2. Signature of an entity = the model's current answers on every relation except r. Entities of one
     hidden type share most of these, so a signature stands in for the type. Two signatures match when
     they agree on at least SIG_AGREE of the R - 1 relations.
  3. Candidate change on (signature cluster, r, v): at least Q new entities of the phase that match the
     cluster each SHOW value v on r, while the cluster's older members (matching entities created in
     earlier phases) are answered with a different value on r by the model.
  4. On a candidate: retrain the decoder with addresses frozen on the cluster's older members, with targets
     v on r and the model's own current answers on the other relations (which keeps their other answers in
     place). Entities with an override-store entry on r are skipped (their exception stands).
Measures and world: exactly as rule_change.py. Arms: seen2 (the first study's label-free arm, rerun on the
new seeds), detect (seen2 plus the detector), oracle_dec (the ceiling, rerun on the new seeds).
Reported in addition: false_alarms (candidates that match no true change), missed (true changes never
flagged), and delay (phases from a change to its first flag).
"""
from __future__ import annotations
import json, sys
import numpy as np, torch
import rule_change as rc
from cab_s import Z, R, V, K

Q, SIG_AGREE = None, None          # fixed by the mechanics check


def run(arm, seed, q=None, sig_agree=None, phases=rc.PHASES, m=24):
    """A copy of rule_change.run (which stays unchanged) with the detector added for arm 'detect'."""
    q = q or Q; sig_agree = sig_agree or SIG_AGREE; NEW = rc.NEW
    rng = np.random.default_rng(seed); n = phases * NEW
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n)
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    exc = np.full((n, R), -1)
    for e in range(n):
        for r in np.flatnonzero(shown[e]):
            if rng.random() < rc.P_EXC: exc[e, r] = (rule[typ[e], r] + 1 + rng.integers(V - 1)) % V
    changes = []
    for p in rc.CHANGE_AT:
        if p >= phases: continue
        pairs = set()
        while len(pairs) < rc.N_CHANGE: pairs.add((int(rng.integers(Z)), int(rng.integers(R))))
        for z, r in sorted(pairs): changes.append((p, z, r, int((rule[z, r] + 1 + rng.integers(V - 1)) % V)))
    rule_at = np.repeat(rule[None], phases, 0).copy()
    for p, z, r, v in changes: rule_at[p:, z, r] = v
    truth_at = lambda ph, e, r: int(exc[e, r]) if exc[e, r] >= 0 else int(rule_at[ph, typ[e], r])
    L = rc.Model(n, m, seed); store = {}; noisy = {}; log = []; flags = []
    born = np.repeat(np.arange(phases), NEW)

    def pred(e, r):
        out = L.predict(e, r)
        return np.array([store.get((int(a), int(b)), int(o)) for a, b, o in zip(e, r, out)])

    def all_preds(upto):
        ee = np.repeat(np.arange(upto), R); rr = np.tile(np.arange(R), upto)
        return pred(ee, rr).reshape(upto, R)

    for ph in range(phases):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        new_facts = [(int(e), int(r), truth_at(ph, e, r)) for e in ents for r in np.flatnonzero(shown[e])]
        old_obs = []
        if ph:
            prev = np.arange((ph - 1) * NEW, ph * NEW)
            old_obs += [(int(e), int(r), int(exc[e, r]), "exception") for e in prev for r in np.flatnonzero(shown[e]) if exc[e, r] >= 0]
            changed_pairs = {(z, r) for p, z, r, v in changes if p <= ph}
            oe, orr = np.nonzero(shown[:ph * NEW])
            ok = np.array([exc[a, b] < 0 and (typ[a], b) not in changed_pairs for a, b in zip(oe, orr)])
            for i in rng.choice(np.flatnonzero(ok), rc.N_NOISE, replace=False):
                e, r = int(oe[i]), int(orr[i]); wrong = (truth_at(ph, e, r) + 1 + int(rng.integers(V - 1))) % V
                old_obs.append((e, r, wrong, "noise")); noisy.setdefault((e, r), ph)
        if arm in ("seen2", "detect"):
            L.fit(new_facts)
            for e, r, v, _ in old_obs:
                if shown[e, r] and truth_at(int(born[e]), e, r) == v: store[(e, r)] = v
        elif arm == "oracle_dec":
            fit = [f for f in new_facts if exc[f[0], f[1]] < 0]
            for e, r, v in new_facts:
                if exc[e, r] >= 0: store[(e, r)] = v
            bulk = []
            for p, z, r, v in changes:
                if p == ph:
                    olds = np.flatnonzero((typ[:ph * NEW] == z) & shown[:ph * NEW, r] & (exc[:ph * NEW, r] < 0))
                    bulk += [(int(e), r, v) for e in olds]
            L.fit(fit)
            if bulk: L.fit(bulk, frozen=torch.tensor(sorted({f[0] for f in bulk}), dtype=torch.long))
        else: raise SystemExit(arm)
        if arm == "detect" and ph:
            P = all_preds((ph + 1) * NEW); done = set()
            for r in range(R):
                others = [x for x in range(R) if x != r]
                newr = [int(e) for e in ents if shown[e, r]]
                for e in newr:
                    v = truth_at(ph, e, r)                                   # what was observed
                    sig = P[e, others]
                    peers = [f for f in newr if f != e and truth_at(ph, f, r) == v and (P[f, others] == sig).sum() >= sig_agree]
                    if len(peers) + 1 < q: continue
                    olds = np.flatnonzero((P[:ph * NEW][:, others] == sig).sum(1) >= sig_agree)
                    olds = np.array([o for o in olds if (int(o), r) not in store])
                    if len(olds) == 0 or (r, v, tuple(sig)) in done: continue
                    if np.mean(P[olds, r] == v) >= 0.5: continue                # the cluster already answers v
                    done.add((r, v, tuple(sig)))
                    z_major = int(np.bincount(typ[olds], minlength=Z).argmax())
                    true = [(p, z, rr_, vv) for p, z, rr_, vv in changes if p <= ph and z == z_major and rr_ == r and vv == v]
                    flags.append({"phase": ph, "r": r, "v": v, "z": z_major, "cluster": int(len(olds)), "true_change": bool(true),
                                  "change_phase": true[0][0] if true else None})
                    targets = [(int(o), r, v) for o in olds] + [(int(o), int(x), int(P[o, x])) for o in olds for x in others]
                    L.fit(targets, frozen=torch.tensor(sorted(int(o) for o in olds), dtype=torch.long))
        nf = np.array(new_facts); log.append({"phase": ph, "acquire_now": float((pred(nf[:, 0], nf[:, 1]) == nf[:, 2]).mean())})

    last = phases - 1
    T = lambda e, r: np.array([truth_at(last, a, b) for a, b in zip(e, r)])
    def acc(e, r): return float((pred(e, r) == T(e, r)).mean()) if len(e) else float("nan")
    changed = {(z, r) for p, z, r, v in changes}
    late = np.arange((phases - 10) * NEW, n); ue, ur = np.nonzero(~shown[late]); ue = late[ue]
    st = np.array([(typ[a], b) not in changed for a, b in zip(ue, ur)])
    out = {"arm": arm, "seed": seed, "acquire_now": round(float(np.mean([x["acquire_now"] for x in log])), 4), "infer_stable": round(acc(ue[st], ur[st]), 4)}
    r5, r10, ro = [], [], []
    for p, z, r, v in changes:
        for lo, hi, bucket in ((p + 1, p + 6, r5), (p + 6, p + 11, r10)):
            es = np.flatnonzero((typ == z) & (born >= lo) & (born < min(hi, phases)) & ~shown[:, r] & (exc[:, r] < 0))
            bucket += list(pred(es, np.full(len(es), r)) == v)
        es = np.flatnonzero((typ == z) & (born < p) & (exc[:, r] < 0)); ro += list(pred(es, np.full(len(es), r)) == v)
    out.update({"revise_next5": round(float(np.mean(r5)), 4), "revise_next10": round(float(np.mean(r10)), 4), "revise_old": round(float(np.mean(ro)), 4)})
    ee, er = np.nonzero((exc >= 0) & (born[:, None] <= last - 10)); out["exception_kept"] = round(acc(ee, er), 4)
    nk = [(e, r) for (e, r), p in noisy.items() if p <= last - 10]
    out["noise_resisted"] = round(acc(np.array([a for a, _ in nk]), np.array([b for _, b in nk])), 4)
    out["store_size"] = len(store)
    found = {(f["change_phase"], f["z"], f["r"]) for f in flags if f["true_change"]}
    out["flags"] = len(flags); out["false_alarms"] = sum(1 for f in flags if not f["true_change"])
    out["missed"] = sum(1 for p, z, r, v in changes if (p, z, r) not in found)
    delays = [min(f["phase"] - p for f in flags if f["true_change"] and (f["change_phase"], f["z"], f["r"]) == (p, z, r)) for p, z, r, v in changes if (p, z, r) in found]
    out["delay_mean"] = round(float(np.mean(delays)), 2) if delays else None
    return out


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [7201, 7202, 7203]
    assert Q is not None and SIG_AGREE is not None, "fixed by the mechanics check"
    rows = [run(arm, s) for s in seeds]
    for x in rows: print(json.dumps(x), flush=True)
    json.dump(rows, open(f"results/rule_change_detect/{arm}.json", "w"), indent=1)
