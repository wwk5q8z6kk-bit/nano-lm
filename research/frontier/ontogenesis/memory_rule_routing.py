"""Can the memory choose private or shared writes itself, without being told which facts are corrections?

Protocol: PROTOCOL-MEMRULE-ROUTING.md (committed before any confirmatory run).
Follows RESULTS-MEMRULE-PRIVATEFIX.md, where corrections given an explicit label and written to a private table
with a 2,000-step pass took 0.89 of the time at a cost of 0.02 later inference. Here nothing is labelled.
Each phase the memory receives the phase's facts (new entities' shown facts and the corrections, mixed) and:
  none      trains both tables on all of them jointly, 200 steps (no routing)
  detector  flags every fact whose current prediction (shared + private read, before this phase's training)
            differs from the given value. Unflagged facts: joint training on both tables, 200 steps. Flagged
            facts: then written alone to the private table only, 2,000 steps.
  random    as detector, but the flagged set is the same number of facts drawn at random from the phase's facts
A fact is flagged when the argmax disagrees with the given value AND the read's top softmax probability is at
least TAU. TAU = 0 is pure disagreement; on the development seed that collapses (an empty memory disagrees with
everything in phase 0, everything goes private, the shared path never learns the rule). TAU is chosen on the
development seed only, as the protocol records.
It is the same idea as routing_loo.py's leave-one-out detector on research/ontogenesis-phase1 (INVESTIGATION.md,
Question 8), applied to this memory; that file is not imported or changed.
The labelled ceiling is RESULTS-MEMRULE-PRIVATEFIX.md's private 2,000-step arm (same seeds and world).

usage: memory_rule_routing.py ARM SEED OUT.json [TAU]     ARM in {none, detector, random}
"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from cab_s import world, Z, R as NREL, V
from memory_rule_sharing import addresses, SLOTS, PHASES, NEW, FIX, STEPS, LR

torch.set_num_threads(2)
FIX_STEPS = 2000


def run(arm, seed, tau=0.0, lr=LR):
    rng, truth, shown = world(seed, PHASES, NEW, FIX)
    n = PHASES * NEW
    rule = np.random.default_rng(seed).integers(V, size=(Z, NREL))          # same first draw as cab_s.world
    i1, w1, typ, _ = addresses("typed", 0, seed, n)
    i2, w2, _, _ = addresses("typed", 8, seed, n)
    regular0 = truth == rule[typ]                                           # before any correction
    sel = np.random.default_rng(seed + 50_000)                              # random router's own stream
    T1 = torch.nn.Parameter(torch.zeros(SLOTS, V)); T2 = torch.nn.Parameter(torch.zeros(SLOTS, V))
    key = lambda e, r: torch.as_tensor(np.asarray(e) * NREL + np.asarray(r))
    read = lambda q: (w1[q][..., None] * T1[i1[q]]).sum(1) + (w2[q][..., None] * T2[i2[q]]).sum(1)
    pred = lambda e, r: read(key(e, r)).argmax(-1).numpy()
    acc = lambda e, r: float((pred(e, r) == truth[e, r]).mean()) if len(e) else float("nan")

    def fit(qe, qr, params, n_steps):
        if not len(qe): return
        q, y = key(qe, qr), torch.from_numpy(truth[qe, qr])
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        for _ in range(n_steps):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        for p in (T1, T2): p.grad = None

    log = []
    for ph in range(PHASES):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        se, sr = np.nonzero(shown[ents]); se = ents[se]
        fe = fr = np.zeros(0, int)
        if ph:
            oe, orr = np.nonzero(shown[:ph * NEW]); pick = rng.choice(len(oe), FIX, replace=False)
            fe, fr = oe[pick], orr[pick]; truth[fe, fr] = rng.integers(V, size=FIX)
        qe, qr = np.concatenate([se, fe]), np.concatenate([sr, fr])
        is_fix = np.r_[np.zeros(len(se), bool), np.ones(len(fe), bool)]
        with torch.no_grad():
            pr = torch.softmax(read(key(qe, qr)), -1)
            conf, guess = pr.max(-1)
            flag = (guess.numpy() != truth[qe, qr]) & (conf.numpy() >= tau)
        n_flag = int(flag.sum())
        if arm == "random":
            flag = np.zeros(len(qe), bool); flag[sel.choice(len(qe), n_flag, replace=False)] = True
        row = {"phase": ph, "flagged": n_flag / len(qe)}
        newreg = (~is_fix) & regular0[qe, qr]; newexc = (~is_fix) & ~regular0[qe, qr]
        if newreg.any(): row["flag_regular_new"] = float(flag[newreg].mean())      # false positives
        if newexc.any(): row["flag_exception_new"] = float(flag[newexc].mean())
        if is_fix.any(): row["flag_correction"] = float(flag[is_fix].mean())
        if arm == "none":
            fit(qe, qr, [T1, T2], STEPS)
        else:
            fit(qe[~flag], qr[~flag], [T1, T2], STEPS)
            fit(qe[flag], qr[flag], [T2], FIX_STEPS)
        with torch.no_grad():
            row["acquire"] = acc(se, sr)
            ne, nr = np.nonzero(~shown[ents]); row["infer_new"] = acc(ents[ne], nr)
            if ph:
                row["correct"] = acc(fe, fr)
                oe, orr = np.nonzero(shown[:ph * NEW]); row["retain"] = acc(oe, orr)
                ue, ur = np.nonzero(~shown[:ph * NEW]); row["infer_old"] = acc(ue, ur)
        log.append(row)
    late = log[-(PHASES // 4):]; mean = lambda kk: round(float(np.nanmean([x.get(kk, np.nan) for x in late])), 4)
    keys = ("acquire", "infer_new", "infer_old", "retain", "correct", "flagged", "flag_regular_new", "flag_exception_new", "flag_correction")
    return {"arm": arm, "seed": seed, "tau": tau, "lr": lr, "steps": STEPS, "fix_steps": FIX_STEPS, "late": {k: mean(k) for k in keys}, "log": log}


if __name__ == "__main__":
    arm, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    tau = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    f = open(out, "x"); t = time.time(); res = run(arm, seed, tau)
    res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__
    json.dump(res, f); f.close()
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
