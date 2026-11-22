"""G04 third study: a three-way router in a world with exceptions, noise and rule changes.

Protocol: PROTOCOL-THREEWAY.md (reviewed by the memory-line and ontogenesis sessions). Predictions are added
to that protocol after the development check on seed 1 and before any run on seeds 7301-7305.

World: rule_change.py's, regenerated here with the same draws. Learner: the memory-line session's two-table
memory (memory_rule_sharing.addresses, copied from research/memory-saturation-factorial and unchanged):
a shared table read through typed addresses, a private table read through fully spread addresses, reads
summed; joint training 200 steps at 1e-3 (AdamW, as memory_rule_routing.py); private-only pass 2,000 steps.

usage: threeway.py ARM [SEED ...]
ARM in none, private, private_t02, change_only, threeway, threeway_t02, threeway_rev200, threeway_privfirst,
       threeway_random, oracle
"""
from __future__ import annotations
import json, sys
import numpy as np, torch, torch.nn.functional as F
from cab_s import Z, R, V, K
import rule_change as rc
from memory_rule_sharing import addresses, SLOTS

torch.set_num_threads(2)
JOINT_STEPS, PRIVATE_STEPS, LR = 200, 2000, 1e-3
Q_DET, SIG_AGREE = 2, 5                                         # as registered in rule_change_detect.py
CONF, WARMUP, MAX_OLDS = 0.12, 3, 64                            # development-stage fixes (PROTOCOL-THREEWAY.md)
GATE_SIG = False                                                # gate only the older members' answer on r


def build_world(seed, phases=rc.PHASES):
    NEW = rc.NEW; rng = np.random.default_rng(seed); n = phases * NEW
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
    return rng, rule, typ, shown, exc, changes, rule_at


def run(arm, seed, phases=rc.PHASES):
    NEW = rc.NEW; n = phases * NEW
    tau = 0.2 if arm in ("private_t02", "threeway_t02") else 0.1
    rev_steps = 200 if arm == "threeway_rev200" else PRIVATE_STEPS
    rng, rule, typ, shown, exc, changes, rule_at = build_world(seed, phases)
    truth_at = lambda ph, e, r: int(exc[e, r]) if exc[e, r] >= 0 else int(rule_at[ph, typ[e], r])
    i1, w1, typ_m, _ = addresses("typed", 0, seed, n)
    i2, w2, _, _ = addresses("typed", 8, seed, n)
    assert np.array_equal(typ_m, typ), "memory addresses must use the world's types"
    T1 = torch.nn.Parameter(torch.zeros(SLOTS, V)); T2 = torch.nn.Parameter(torch.zeros(SLOTS, V))
    key = lambda e, r: torch.as_tensor(np.asarray(e) * R + np.asarray(r))
    read = lambda q: (w1[q][..., None] * T1[i1[q]]).sum(1) + (w2[q][..., None] * T2[i2[q]]).sum(1)
    sel = np.random.default_rng(seed + 50_000)

    @torch.no_grad()
    def predict(e, r):
        lg = read(key(e, r)); p = lg.softmax(-1); top = p.max(-1)
        return top.indices.numpy(), top.values.numpy()

    def fit(facts, params, steps):
        if not facts: return
        a = np.array(facts); q = key(a[:, 0], a[:, 1]); y = torch.as_tensor(a[:, 2])
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
        for _ in range(steps):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        for p_ in (T1, T2): p_.grad = None

    born = np.repeat(np.arange(phases), NEW); noisy = {}; log = []; flags = []; n_private = 0
    exc_in_false_group = []                                              # (entity, relation) exceptions inside falsely flagged groups
    fa_pairs = set()                                                     # (type, relation) hit by a false alarm

    def detect(facts, ph):
        """Group-agreement change detector on the given new-entity facts. Returns flags as (r, v, members, olds, true)."""
        if not facts or ph < WARMUP: return []
        upto = (ph + 1) * NEW; ee = np.repeat(np.arange(upto), R); rr = np.tile(np.arange(R), upto)
        P, C = predict(ee, rr); P = P.reshape(upto, R); C = C.reshape(upto, R) >= CONF; out = []; done = set()
        by_r = {}
        for e, r, v in facts: by_r.setdefault(r, []).append((e, v))
        for r, lst in by_r.items():
            others = [x for x in range(R) if x != r]
            for e, v in lst:
                sig = P[e, others]; csig = C[e, others]
                if GATE_SIG and csig.sum() < SIG_AGREE: continue        # signature must rest on confident answers
                if GATE_SIG: agree = lambda rows: ((P[rows][:, others] == sig) & C[rows][:, others] & csig).sum(1) >= SIG_AGREE
                else: agree = lambda rows: (P[rows][:, others] == sig).sum(1) >= SIG_AGREE
                members = [f for f, vf in lst if vf == v and agree(np.array([f]))[0]]
                if len(members) < Q_DET or (r, v, tuple(sig)) in done: continue
                olds = np.flatnonzero(agree(np.arange(ph * NEW)))
                olds = olds[C[olds, r]]                                  # older members confidently answering r
                if len(olds) < 3 or np.mean(P[olds, r] == v) >= 0.5: continue
                done.add((r, v, tuple(sig)))
                z_major = int(np.bincount(typ[olds], minlength=Z).argmax())
                true = any(p <= ph and z == z_major and r_ == r and v_ == v for p, z, r_, v_ in changes)
                out.append((r, v, members, olds, true, z_major, P))
        return out

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

        joint, private, revision = [], [], []
        gate = lambda facts: [f for f, (pv, pp) in zip(facts, zip(*predict(np.array([x[0] for x in facts]), np.array([x[1] for x in facts])))) if pv != f[2] and pp >= tau] if facts else []

        if arm == "none":
            joint = new_facts + [(e, r, v) for e, r, v, _ in old_obs]
        elif arm in ("private", "private_t02"):
            facts = new_facts + [(e, r, v) for e, r, v, _ in old_obs]
            flagged = set(gate(facts)); private = list(flagged); joint = [f for f in facts if f not in flagged]
        elif arm == "oracle":
            private = [f for f in new_facts if exc[f[0], f[1]] >= 0] + [(e, r, v) for e, r, v, k in old_obs if k == "exception"]
            joint = [f for f in new_facts if exc[f[0], f[1]] < 0]
            for p, z, r, v in changes:
                if p == ph:
                    olds = np.flatnonzero((typ[:ph * NEW] == z) & (exc[:ph * NEW, r] < 0))
                    revision += [(int(e), r, v) for e in olds]
        else:                                                            # change_only and the threeway family
            if arm == "threeway_privfirst":
                flagged = set(gate(new_facts)); rest = [f for f in new_facts if f not in flagged]
                found = detect(rest, ph); private += list(flagged)
            else:
                found = detect(new_facts, ph)
            exempt = {(e, r) for r, v, members, *_ in found for e in members}
            for r, v, members, olds, true, z_major, P in found:
                flags.append({"phase": ph, "r": r, "v": v, "z": z_major, "true_change": true})
                others = [x for x in range(R) if x != r]
                cap = olds if len(olds) <= MAX_OLDS else sel.choice(olds, MAX_OLDS, replace=False)
                revision += [(int(o), r, v) for o in cap]
                if not true:
                    fa_pairs.add((z_major, r))
                    exc_in_false_group += [(int(o), r) for o in olds if exc[o, r] >= 0] + [(int(e), r) for e in members if exc[e, r] >= 0]
            if arm == "change_only":
                joint = new_facts + [(e, r, v) for e, r, v, _ in old_obs]
            else:
                cand = [f for f in new_facts if (f[0], f[1]) not in exempt and f not in set(private)]
                if arm == "threeway_privfirst": flagged_new = []
                elif arm == "threeway_random":
                    k = len(gate(cand)); flagged_new = [cand[i] for i in sel.choice(len(cand), k, replace=False)] if k else []
                else: flagged_new = gate(cand)
                private += flagged_new
                joint = [f for f in new_facts if f not in set(private)]
                for e, r, v, _ in old_obs:                               # seen twice, consistently -> private; else held
                    if shown[e, r] and truth_at(int(born[e]), e, r) == v: private.append((e, r, v))
        fit(joint, [T1, T2], JOINT_STEPS)
        fit(private, [T2], PRIVATE_STEPS); n_private += len(private)
        fit(revision, [T1], rev_steps)
        nf = np.array(new_facts); log.append({"phase": ph, "acquire_now": float((predict(nf[:, 0], nf[:, 1])[0] == nf[:, 2]).mean())})

    last = phases - 1
    pred = lambda e, r: predict(np.asarray(e), np.asarray(r))[0]
    T = lambda e, r: np.array([truth_at(last, a, b) for a, b in zip(e, r)])
    def acc(e, r): return float((pred(e, r) == T(e, r)).mean()) if len(e) else float("nan")
    changed = {(z, r) for p, z, r, v in changes}
    late = np.arange((phases - 10) * NEW, n); ue, ur = np.nonzero(~shown[late]); ue = late[ue]
    st = np.array([(typ[a], b) not in changed for a, b in zip(ue, ur)])
    fa = np.array([(typ[a], b) in fa_pairs for a, b in zip(ue, ur)])
    out = {"arm": arm, "seed": seed, "acquire_now": round(float(np.mean([x["acquire_now"] for x in log])), 4),
           "infer_stable": round(acc(ue[st], ur[st]), 4),
           "infer_stable_fa_pairs": round(acc(ue[st & fa], ur[st & fa]), 4), "infer_stable_other_pairs": round(acc(ue[st & ~fa], ur[st & ~fa]), 4)}
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
    found = {(f["z"], f["r"]) for f in flags if f["true_change"]}
    out.update({"routed_private": n_private, "flags": len(flags), "false_alarms": sum(1 for f in flags if not f["true_change"]),
                "missed": sum(1 for p, z, r, v in changes if (z, r) not in found)})
    eg = sorted(set(exc_in_false_group))
    out["exceptions_in_false_groups"] = len(eg)
    out["exceptions_in_false_groups_kept"] = round(acc(np.array([a for a, _ in eg]), np.array([b for _, b in eg])), 4) if eg else None
    return out


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or [7301, 7302, 7303, 7304, 7305]
    rows = [run(arm, s) for s in seeds]
    for x in rows: print(json.dumps(x), flush=True)
    if seeds == [7301, 7302, 7303, 7304, 7305]: json.dump(rows, open(f"results/threeway/{arm}.json", "w"), indent=1)
