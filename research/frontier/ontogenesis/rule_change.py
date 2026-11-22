"""Gate G04, first study: a world where rules really change. How large is the gap between a plain learner,
a label-free policy that separates noise from exceptions, and an oracle that knows each surprise's kind?

Predictions X1-X6 are at the end of this docstring, written after a mechanics check on development seed
1 and before any run on the registered seeds 7101-7103.

Why. Every earlier world here is stationary: a surprising fact is an exception or a correction, never a
change of the rule itself. A learner that sends every surprise to private storage (the memory-line
session's detector; INVESTIGATION.md Q8's detector) is then right by construction. In a world where rules
change, the same policy would keep answering the old rule. This study builds that world and measures
the gap before anyone builds a change detector.

World (cab_s.py rule world): Z = 16 hidden types, R = 8 relations, V = 32 values; rule[z, r] is a value.
40 phases; each brings 64 new entities with K = 3 shown relations. Three kinds of surprise:
  rule change  at phases 10, 20 and 30, four random (type, relation) pairs get a new value. From then
               on it is the truth for every entity of that type on that relation (old and new), except
               entities with an exception there. Evidence: new entities of that type showing it.
  exception    each new entity's shown fact is, with probability 0.05, an entity-specific value (not the
               rule). It is shown at creation and shown again, unchanged, in the next phase.
  noise        each phase from phase 1, 8 old shown facts (no exception, not on a changed pair) are
               re-observed once with a wrong random value. Their truth does not change.
Model: the bilinear learner of correction_interference.py (m = 24, Adam 0.003, 200 steps per phase).

Arms
  plain    every observation of the phase is trained (new facts, exception repeats, noisy re-observations)
  seen2    label-free. New entities' facts are trained (nothing to judge them against). An observation
           about an OLD entity is never trained: if it repeats the value shown when the entity was created
           (seen twice, consistently) it goes to an exact override store; otherwise it is held as pending.
           It can separate noise (seen once) from exceptions (seen twice). It cannot detect a rule change.
  oracle   knows each surprise's kind. Noise: discarded. Exceptions: exact override store, not trained.
           Rule change: the new value is trained for every previously shown entity of that type on that
           relation (a bulk revision only possible with the hidden types), plus the new facts as usual.
  oracle_dec  as oracle, but the bulk revision is trained with the old entities' addresses frozen, so the
           change must go into the shared decoder (the rule) rather than into each entity's address.
The override store answers before the model; its size is reported.

Deviations, recorded before any registered run (development seed 1 only).
  1. seen2 as first written ignored a repeated observation whenever the model already answered it, so
     exceptions were later forgotten (stored 1 of ~390). Changed to the policy stated above.
  2. oracle as first written was not a ceiling: its bulk revision went into the old entities' addresses and
     revised the rule worse than plain (0.52 against 0.67 for new entities). oracle is kept as a measured
     arm; oracle_dec was added as the ceiling. Its form was chosen on seed 1 by revision accuracy, so it is
     declared tuned and serves only as an upper reference.

Measures (end of stream unless stated)
  acquire_now      the phase's new facts right after the phase
  infer_stable     unshown relations of the last 10 phases' entities, on pairs never changed
  revise_next5     unshown relations on a changed pair, for entities created 1-5 phases after the change
  revise_next10    the same, entities created 6-10 phases after the change
  revise_old       entities created before a change, on the changed pair, answer the new value
  exception_kept   exception facts 10+ phases old
  noise_resisted   facts that received a noisy re-observation 10+ phases ago, answered with their truth
  store_size       entries in the override store

Predictions (seeds 7101-7103; informed by development seed 1, so they test replication, not discovery):
  X1 plain absorbs noise and loses exceptions: noise_resisted at most 0.30, exception_kept at most 0.85,
     and revises partially: revise_next5 between 0.50 and 0.85.
  X2 seen2 separates noise from exceptions: noise_resisted and exception_kept at least 0.95; but has no
     change mechanism: revise_next5 within 0.10 of plain, revise_old at most 0.70.
  X3 oracle_dec is a ceiling: revise_next5, revise_old, exception_kept and noise_resisted all at least 0.95.
  X4 knowing the kinds is not enough when the revision goes into addresses: oracle revise_old at most 0.70.
  X5 the cost of having no change detector: oracle_dec minus seen2 on revise_old is at least 0.30.
  X6 the stable rule is kept: infer_stable at least 0.90 in every arm.
  Reading. X2 and X5 holding: a label-free policy that handles noise and exceptions still leaves rule
  changes unrevised; detecting a change (and writing it into the shared rule) is the open part of G04.
  X4 holding as well: the write location matters as much as the label.
"""
from __future__ import annotations
import json, sys
import numpy as np, torch
from cab_s import Z, R, V, K
from correction_interference import Model

PHASES, NEW, CHANGE_AT, N_CHANGE, P_EXC, N_NOISE = 40, 64, (10, 20, 30), 4, 0.05, 8


def run(arm, seed, phases=PHASES, m=24):
    rng = np.random.default_rng(seed); n = phases * NEW
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n)
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    exc = np.full((n, R), -1)                                    # exception value, or -1
    for e in range(n):
        for r in np.flatnonzero(shown[e]):
            if rng.random() < P_EXC: exc[e, r] = (rule[typ[e], r] + 1 + rng.integers(V - 1)) % V
    changes, used = [], set()                                   # (phase, z, r, new value); each pair once
    for p in CHANGE_AT:
        if p >= phases: continue
        pairs = set()
        while len(pairs) < N_CHANGE:
            pair = (int(rng.integers(Z)), int(rng.integers(R)))
            if pair not in used: pairs.add(pair)                # a later phase must not redraw it: its v would stop being the truth
        used |= pairs
        for z, r in sorted(pairs): changes.append((p, z, r, int((rule[z, r] + 1 + rng.integers(V - 1)) % V)))
    rule_at = np.repeat(rule[None], phases, 0).copy()             # rule in force at each phase
    for p, z, r, v in changes: rule_at[p:, z, r] = v
    truth_at = lambda ph, e, r: int(exc[e, r]) if exc[e, r] >= 0 else int(rule_at[ph, typ[e], r])
    L = Model(n, m, seed); store = {}; pending = {}; noisy = {}; log = []
    born = np.repeat(np.arange(phases), NEW)

    def pred(e, r):
        out = L.predict(e, r)
        return np.array([store.get((int(a), int(b)), int(o)) for a, b, o in zip(e, r, out)])

    for ph in range(phases):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        new_facts = [(int(e), int(r), truth_at(ph, e, r)) for e in ents for r in np.flatnonzero(shown[e])]
        old_obs = []                                             # (e, r, value, kind)
        if ph:
            prev = np.arange((ph - 1) * NEW, ph * NEW)
            old_obs += [(int(e), int(r), int(exc[e, r]), "exception") for e in prev for r in np.flatnonzero(shown[e]) if exc[e, r] >= 0]
            changed_pairs = {(z, r) for p, z, r, v in changes if p <= ph}
            oe, orr = np.nonzero(shown[:ph * NEW])
            ok = np.array([exc[a, b] < 0 and (typ[a], b) not in changed_pairs for a, b in zip(oe, orr)])
            pick = rng.choice(np.flatnonzero(ok), N_NOISE, replace=False)
            for i in pick:
                e, r = int(oe[i]), int(orr[i]); wrong = (truth_at(ph, e, r) + 1 + int(rng.integers(V - 1))) % V
                old_obs.append((e, r, wrong, "noise")); noisy.setdefault((e, r), ph)
        if arm == "plain":
            L.fit(new_facts + [(e, r, v) for e, r, v, _ in old_obs])
        elif arm == "seen2":
            L.fit(new_facts)
            for e, r, v, _ in old_obs:
                created_value = truth_at(int(born[e]), e, r) if shown[e, r] else None
                if created_value == v: store[(e, r)] = v          # seen twice, consistently: keep exactly
                else: pending[(e, r, v)] = pending.get((e, r, v), 0) + 1   # seen once: hold, do not train
        elif arm in ("oracle", "oracle_dec"):
            exc_keys = {(e, r) for e, r, _, k in old_obs if k == "exception"}
            fit = [f for f in new_facts if exc[f[0], f[1]] < 0]
            for e, r, v in new_facts:
                if exc[e, r] >= 0: store[(e, r)] = v
            for p, z, r, v in changes:
                if p == ph:
                    olds = np.flatnonzero((typ[:ph * NEW] == z) & shown[:ph * NEW, r] & (exc[:ph * NEW, r] < 0))
                    fit += [(int(e), r, v) for e in olds]
            if arm == "oracle": L.fit(fit)
            else:
                bulk = [f for f in fit if f[0] < ph * NEW]; L.fit([f for f in fit if f[0] >= ph * NEW])
                if bulk: L.fit(bulk, frozen=torch.tensor(sorted({f[0] for f in bulk}), dtype=torch.long))
        else: raise SystemExit(arm)
        nf = np.array(new_facts); log.append({"phase": ph, "acquire_now": float((pred(nf[:, 0], nf[:, 1]) == nf[:, 2]).mean())})

    last = phases - 1
    T = lambda e, r: np.array([truth_at(last, a, b) for a, b in zip(e, r)])
    def acc(e, r): return float((pred(e, r) == T(e, r)).mean()) if len(e) else float("nan")
    changed = {(z, r) for p, z, r, v in changes}
    late = np.arange((phases - 10) * NEW, n); ue, ur = np.nonzero(~shown[late]); ue = late[ue]
    st = np.array([(typ[a], b) not in changed for a, b in zip(ue, ur)])
    out = {"arm": arm, "seed": seed, "acquire_now": round(float(np.mean([x["acquire_now"] for x in log])), 4),
           "infer_stable": round(acc(ue[st], ur[st]), 4)}
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
    return out


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or [7101, 7102, 7103]
    rows = [run(arm, s) for s in seeds]
    for x in rows: print(json.dumps(x), flush=True)
    if seeds == [7101, 7102, 7103]: json.dump(rows, open(f"results/rule_change/{arm}.json", "w"), indent=1)
