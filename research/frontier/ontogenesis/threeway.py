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
import memory_rule_sharing as msr
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


def run(arm, seed, phases=rc.PHASES, p_slots=None, private_steps=None, changes_on=True, narrow=False, narrow_w=0.02, share=None,
        codes="typed", KC=32, every=5, detector=True, cooldown=False):
    """Version 2 (2026-11-23): Bugbot fixes on PR #204, see CORRECTIONS-G04-20261123.md.
    codes: 'typed' (registered: memory_rule_sharing typed addresses), 'given32' or 'em' (THREEWAY-EM; the memory-line
    session's construction: cluster prototype + noise, KC prototypes; 'given32' uses the hidden types, 'em' re-clusters
    by EM over creation-time facts every `every` phases, labels matched, no replay). detector=False skips change detection."""
    """p_slots, private_steps, changes_on: exploration settings added after the registered run (THREEWAY-COST);
    their defaults reproduce the registered behaviour exactly."""
    NEW = rc.NEW; n = phases * NEW
    p_slots = p_slots or SLOTS; private_steps = PRIVATE_STEPS if private_steps is None else private_steps
    tau = 0.2 if arm in ("private_t02", "threeway_t02") else 0.1
    rev_steps = 200 if arm == "threeway_rev200" else PRIVATE_STEPS
    rng, rule, typ, shown, exc, changes, rule_at = build_world(seed, phases)
    if not changes_on: changes = []; rule_at = np.repeat(rule[None], phases, 0).copy()
    truth_at = lambda ph, e, r: int(exc[e, r]) if exc[e, r] >= 0 else int(rule_at[ph, typ[e], r])
    i1, w1, typ_m, _ = addresses("typed", 0, seed, n)
    msr.SLOTS = p_slots
    try: i2, w2, _, _ = addresses("typed", 8, seed, n)
    finally: msr.SLOTS = SLOTS
    assert np.array_equal(typ_m, typ), "memory addresses must use the world's types"
    assign = typ.copy(); split_log = []
    if codes != "typed":
        from memory_rule_recluster import features, em
        from memory_rule_transport import match
        import torch.nn.functional as F_
        Dm, Km = msr.D, msr.K
        gq = torch.Generator().manual_seed(seed + 30_000)
        relv = torch.randn(R, Dm, generator=gq) / Dm ** 0.5
        qm = torch.randn(2 * Dm, Dm, generator=gq) / (2 * Dm) ** 0.5; keys = torch.randn(SLOTS, Dm, generator=gq) / Dm ** 0.5
        hq = torch.Generator().manual_seed(seed + 40_000)
        proto = torch.randn(KC, Dm, generator=hq) / Dm ** 0.5; noise_v = torch.randn(n, Dm, generator=hq) / Dm ** 0.5
        i1 = torch.zeros(n * R, Km, dtype=torch.long); w1 = torch.zeros(n * R, Km)
        em_pi = np.ones(KC) / KC; em_th = np.full((KC, R, V), 1.0 / V)
        truth0 = np.zeros((n, R), int)                                   # values shown at creation (what clustering sees)
        def address(ents):
            ents = np.asarray(ents)
            if not len(ents): return
            ent = proto[torch.from_numpy(assign[ents])] + noise_v[torch.from_numpy(ents)]
            e_ = torch.from_numpy(ents).repeat_interleave(R); r_ = torch.arange(R).repeat(len(ents))
            x = torch.cat([ent.repeat_interleave(R, 0), relv[r_]], -1) @ qm
            tt, ii = (F_.normalize(x, dim=-1) @ F_.normalize(keys, dim=-1).T).topk(Km, -1)
            rows = e_ * R + r_; i1[rows] = ii; w1[rows] = torch.softmax(tt * 8.0, -1)
    T1 = torch.nn.Parameter(torch.zeros(SLOTS, V)); T2 = torch.nn.Parameter(torch.zeros(p_slots, V))
    key = lambda e, r: torch.as_tensor(np.asarray(e) * R + np.asarray(r))
    read = lambda q: (w1[q][..., None] * T1[i1[q]]).sum(1) + (w2[q][..., None] * T2[i2[q]]).sum(1)
    sel = np.random.default_rng(seed + 50_000)

    @torch.no_grad()
    def predict(e, r):
        lg = read(key(e, r)); p = lg.softmax(-1); top = p.max(-1)
        return top.indices.numpy(), top.values.numpy()

    # narrow revision (exploration setting, THREEWAY-NARROW): a shared slot may be revised for (type z, relation r)
    # only if no query of any other (type, relation) pair reads it with weight above narrow_w
    pair_of_query = torch.as_tensor(assign).repeat_interleave(R) * R + torch.arange(R).repeat(n)
    def allowed_slots(z, r):
        if share is not None:                                            # soft: the changed pair carries at least `share` of the slot's read weight
            own_q = (pair_of_query == z * R + r)[:, None].expand_as(i1)
            tot = torch.zeros(SLOTS).index_add_(0, i1.flatten(), w1.flatten())
            mine = torch.zeros(SLOTS).index_add_(0, i1[own_q], w1[own_q])
            return (mine > 0) & (mine >= share * tot)
        heavy = w1 > narrow_w
        own = (pair_of_query == z * R + r)[:, None].expand_as(i1) & heavy
        other = (pair_of_query != z * R + r)[:, None].expand_as(i1) & heavy
        mask = torch.zeros(SLOTS, dtype=torch.bool); mask[i1[own]] = True
        blocked = torch.zeros(SLOTS, dtype=torch.bool); blocked[i1[other]] = True
        return mask & ~blocked

    def fit(facts, params, steps, row_mask=None):
        if not facts: return
        a = np.array(facts); q = key(a[:, 0], a[:, 1]); y = torch.as_tensor(a[:, 2])
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
        for _ in range(steps):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward()
            if row_mask is not None: T1.grad.mul_(row_mask[:, None].float())
            opt.step()
        for p_ in (T1, T2): p_.grad = None

    born = np.repeat(np.arange(phases), NEW); noisy = {}; log = []; flags = []; n_private = 0
    exc_in_false_group = []                                              # (entity, relation) exceptions inside falsely flagged groups
    fa_pairs = set()                                                     # (type, relation) hit by a false alarm

    fired = set()                                                        # cooldown: (group key, relation, value) already flagged
    stage_of = {}                                                        # diagnostic: furthest detector stage per true change
    def note_stage(e, r, v, ph, stage):
        for p, z, r_, v_ in changes:
            if p <= ph and z == typ[e] and r_ == r and v_ == v:
                stage_of[(p, z, r)] = max(stage_of.get((p, z, r), 0), stage)

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
                note_stage(e, r, v, ph, 1)                                   # stage 1: the new value was observed
                if len(members) < Q_DET or (r, v, tuple(sig)) in done: continue
                note_stage(e, r, v, ph, 2)                                   # stage 2: enough agreeing new members
                olds = np.flatnonzero(agree(np.arange(ph * NEW)))
                olds = olds[C[olds, r]]                                  # older members confidently answering r
                if len(olds) < 3: continue
                note_stage(e, r, v, ph, 3)                                   # stage 3: enough confident older members
                if np.mean(P[olds, r] == v) >= 0.5: continue
                note_stage(e, r, v, ph, 4)                                   # stage 4: older members disagree -> flag
                done.add((r, v, tuple(sig)))
                z_major = int(np.bincount(typ[olds], minlength=Z).argmax())
                active = [(p, v_) for p, z, r_, v_ in changes if z == z_major and r_ == r and p <= ph]
                cp = max(active)[0] if active else None                  # the change in force now (fix, Bugbot #204)
                true = bool(active) and max(active)[1] == v
                pkey = int(np.bincount(assign[olds], minlength=max(KC, Z) if codes != "typed" else Z).argmax())
                if cooldown and (pkey, r, v) in fired: continue                  # this group already flagged this value
                fired.add((pkey, r, v))
                out.append((r, v, members, olds, true, z_major, P, cp if true else None, pkey))
        return out

    for ph in range(phases):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        new_facts = [(int(e), int(r), truth_at(ph, e, r)) for e in ents for r in np.flatnonzero(shown[e])]
        if codes != "typed":
            for e, r, v in new_facts: truth0[e, r] = v
            if codes == "em":
                X, M = features(truth0, shown, ents)
                ll = np.log(em_pi)[None] + np.einsum('nrv,krv->nk', X, np.log(em_th)); assign[ents] = ll.argmax(1)
            address(ents); pair_of_query[:] = torch.as_tensor(assign).repeat_interleave(R) * R + torch.arange(R).repeat(n)
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

        joint, private, revision, rev_groups = [], [], [], []
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
                    rev_groups.append(((z, r), [(int(e), r, v) for e in olds]))
        else:                                                            # change_only and the threeway family
            if not detector: found = []
            elif arm == "threeway_privfirst":
                flagged = set(gate(new_facts)); rest = [f for f in new_facts if f not in flagged]
                found = detect(rest, ph); private += list(flagged)
            else:
                found = detect(new_facts, ph)
            exempt = {(e, r) for r, v, members, *_ in found for e in members}
            for r, v, members, olds, true, z_major, P, cp, pkey in found:
                flags.append({"phase": ph, "r": r, "v": v, "z": z_major, "true_change": true, "change_phase": cp, "pkey": int(pkey)})
                others = [x for x in range(R) if x != r]
                cap = olds if len(olds) <= MAX_OLDS else sel.choice(olds, MAX_OLDS, replace=False)
                revision += [(int(o), r, v) for o in cap]
                rev_groups.append(((pkey, r), [(int(o), r, v) for o in cap]))
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
        fit(private, [T2], private_steps); n_private += len(private)
        if narrow and revision:
            for (zz, rr_), facts_ in rev_groups:
                fit(facts_, [T1], rev_steps, row_mask=allowed_slots(int(zz), int(rr_)))
        else:
            fit(revision, [T1], rev_steps)
        if codes == "em" and (ph + 1) % every == 0:                      # re-cluster, match labels, re-address; no replay
            seen = np.arange((ph + 1) * NEW)
            Xs, Ms = features(truth0, shown, seen)
            new_assign, em_pi, em_th = em(Xs, Ms, KC)
            perm_assign = match(assign[seen], new_assign, KC)
            inv = np.arange(KC)                                          # reorder EM parameters to the matched labels
            for old_label, new_label in zip(perm_assign, new_assign): inv[old_label] = new_label
            em_pi, em_th = em_pi[inv], em_th[inv]
            moved = np.flatnonzero(perm_assign != assign[seen]); assign[seen] = perm_assign
            address(seen[moved]); pair_of_query[:] = torch.as_tensor(assign).repeat_interleave(R) * R + torch.arange(R).repeat(n)
            for p, z, r, v in changes:                                   # does EM split a changed type into before / after?
                if p <= ph:
                    mem = np.flatnonzero(typ[:(ph + 1) * NEW] == z); before = mem[born[mem] < p]; after = mem[born[mem] >= p]
                    if len(before) and len(after):
                        top_b = np.bincount(assign[before], minlength=KC).argmax(); top_a = np.bincount(assign[after], minlength=KC).argmax()
                        split_log.append({"phase": ph, "change_phase": p, "z": int(z), "r": int(r), "split": bool(top_b != top_a),
                                          "clusters_of_type": int(len(set(assign[mem])))})
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
    with torch.no_grad():                                                # diagnostic (memory-line session): unseen keys were never
        q = key(ue[st], ur[st])                                          # written privately, so their private read is pure leakage
        shared_only = (w1[q][..., None] * T1[i1[q]]).sum(1).argmax(-1).numpy()
        out["infer_stable_shared_only"] = round(float((shared_only == T(ue[st], ur[st])).mean()), 4)
    r5, r10, ro = [], [], []
    for p, z, r, v in changes:                                           # fix (Bugbot #204): a pair may change twice
        nxt = min([p2 for p2, z2, r2, _ in changes if (z2, r2) == (z, r) and p2 > p] or [phases])
        for lo, hi, bucket in ((p + 1, p + 6, r5), (p + 6, p + 11, r10)):
            es = np.flatnonzero((typ == z) & (born >= lo) & (born < min(hi, phases, nxt)) & ~shown[:, r] & (exc[:, r] < 0))
            bucket += list(pred(es, np.full(len(es), r)) == v)
        if nxt == phases:                                                # only the change still in force is scored on old entities
            es = np.flatnonzero((typ == z) & (born < p) & (exc[:, r] < 0)); ro += list(pred(es, np.full(len(es), r)) == v)
    out.update({"revise_next5": round(float(np.mean(r5)), 4), "revise_next10": round(float(np.mean(r10)), 4), "revise_old": round(float(np.mean(ro)), 4)})
    ee, er = np.nonzero((exc >= 0) & (born[:, None] <= last - 10)); out["exception_kept"] = round(acc(ee, er), 4)
    nk = [(e, r) for (e, r), p in noisy.items() if p <= last - 10]
    out["noise_resisted"] = round(acc(np.array([a for a, _ in nk]), np.array([b for _, b in nk])), 4)
    found = {(f["change_phase"], f["z"], f["r"]) for f in flags if f["true_change"]}
    out.update({"routed_private": n_private, "flags": len(flags), "false_alarms": sum(1 for f in flags if not f["true_change"]),
                "missed": sum(1 for p, z, r, v in changes if (p, z, r) not in found)})
    if codes == "em":
        out["purity"] = round(float(sum(np.bincount(typ[assign == c]).max() for c in set(assign)) / n), 4)
        out["clusters_used"] = int(len(set(assign)))
        out["changed_types_split"] = round(float(np.mean([s_["split"] for s_ in split_log])), 4) if split_log else None
        out["per_change"] = [{"phase": p, "z": int(z), "r": int(r), "detected": (p, int(z), int(r)) in found,
                              "ever_split": any(s_["split"] for s_ in split_log if (s_["z"], s_["r"], s_["change_phase"]) == (z, r, p)),
                              "clusters_of_type": max([s_["clusters_of_type"] for s_ in split_log if (s_["z"], s_["r"], s_["change_phase"]) == (z, r, p)] or [0])}
                             for p, z, r, v in changes]
    out["detector_stage"] = [{"phase": p, "z": int(z), "r": int(r), "stage": int(stage_of.get((p, z, r), 0))} for p, z, r, v in changes]
    out["flag_log"] = flags                                               # diagnostic: every flag with its group key
    eg = sorted(set((a, b) for a, b in exc_in_false_group if born[a] <= last - 10))
    out["exceptions_in_false_groups"] = len(eg)
    out["exceptions_in_false_groups_kept"] = round(acc(np.array([a for a, _ in eg]), np.array([b for _, b in eg])), 4) if eg else None
    return out


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or [7301, 7302, 7303, 7304, 7305]
    rows = [run(arm, s) for s in seeds]
    for x in rows: print(json.dumps(x), flush=True)
    if seeds == [7301, 7302, 7303, 7304, 7305]: json.dump(rows, open(f"results/threeway/{arm}.json", "w"), indent=1)
