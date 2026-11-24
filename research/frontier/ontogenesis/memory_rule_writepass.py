"""Do corrections take when written alone, outside the phase's majority? And at what collateral cost?

Protocol: PROTOCOL-MEMRULE-WRITEPASS.md (committed before any confirmatory run).
Falsifier proposed by another session (nano-lm-d5) for the "conflict inside one objective" account in
RESULTS-MEMRULE-DUAL.md. Shared (typed) path only, as memory_rule_dual.py shared_only, except that each phase:
  1. trains on the phase's new shown facts only (corrections left out), Adam reset, STEPS steps;
  2. measures, then writes the phase's corrections in a separate pass with only the corrections in the loss,
     a fresh Adam, the same lr and the same number of steps;
  3. measures again.
Collateral is measured at writing (between 2 and 3): the corrected entities' other shown facts, shown facts of
other entities of the corrected entities' types, and inference on this phase's new entities.
Deviation from the proposal: the 16 corrections of a phase are written together in one pass, not one pass each.
They share no new-fact majority; 16 separate passes per phase would cost sixteen times the compute.

usage: memory_rule_writepass.py SEED OUT.json [LR] [STEPS] [FIX_STEPS]
  FIX_STEPS: steps of the correction pass only (default STEPS); the relative-budget test raises it alone.
"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from cab_s import world, V
from memory_rule_sharing import addresses, SLOTS, PHASES, NEW, FIX, STEPS, LR

torch.set_num_threads(2)


def run(seed, lr=LR, steps=STEPS, fix_steps=None):
    fix_steps = steps if fix_steps is None else fix_steps
    rng, truth, shown = world(seed, PHASES, NEW, FIX)
    n = PHASES * NEW
    idx, w, typ, eff = addresses("typed", 0, seed, n)
    T = torch.nn.Parameter(torch.zeros(SLOTS, V))
    R = truth.shape[1]
    key = lambda e, r: torch.as_tensor(np.asarray(e) * R + np.asarray(r))
    read = lambda q: (w[q][..., None] * T[idx[q]]).sum(1)
    acc = lambda e, r: float((read(key(e, r)).argmax(-1).numpy() == truth[e, r]).mean()) if len(e) else float("nan")

    def fit(qe, qr, n_steps):
        q, y = key(qe, qr), torch.from_numpy(truth[qe, qr])
        opt = torch.optim.AdamW([T], lr=lr, weight_decay=0.0)
        for _ in range(n_steps):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    log = []
    for ph in range(PHASES):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        se, sr = np.nonzero(shown[ents]); se = ents[se]
        fe = fr = np.zeros(0, int)
        if ph:
            oe, orr = np.nonzero(shown[:ph * NEW]); pick = rng.choice(len(oe), FIX, replace=False)
            fe, fr = oe[pick], orr[pick]; truth[fe, fr] = rng.integers(V, size=FIX)
        fit(se, sr, steps)                                            # 1. new facts only
        row = {"phase": ph}
        with torch.no_grad():
            row["acquire"] = acc(se, sr)
            ne, nr = np.nonzero(~shown[ents]); row["infer_new_before"] = acc(ents[ne], nr)
            if ph:
                row["correct_before"] = acc(fe, fr)
                cset = set(fe.tolist()); fixed = set(zip(fe.tolist(), fr.tolist()))
                oe, orr = np.nonzero(shown[:ph * NEW])
                own = np.array([(a in cset) and ((a, b) not in fixed) for a, b in zip(oe, orr)])
                ctyp = set(typ[fe].tolist())
                peer = np.array([(a not in cset) and (typ[a] in ctyp) for a in oe])
                row["own_before"] = acc(oe[own], orr[own]); row["peer_before"] = acc(oe[peer], orr[peer])
        if ph:
            fit(fe, fr, fix_steps)                                    # 2. corrections alone in the loss
            with torch.no_grad():
                row["correct_after"] = acc(fe, fr)
                row["own_after"] = acc(oe[own], orr[own]); row["peer_after"] = acc(oe[peer], orr[peer])
                row["infer_new_after"] = acc(ents[ne], nr)
                row["retain"] = acc(oe, orr)
                ue, ur = np.nonzero(~shown[:ph * NEW]); row["infer_old"] = acc(ue, ur)
                row["acquire_after"] = acc(se, sr)
        log.append(row)
    late = log[-(PHASES // 4):]; mean = lambda kk: round(float(np.nanmean([x.get(kk, np.nan) for x in late])), 4)
    keys = ("acquire", "acquire_after", "infer_new_before", "infer_new_after", "infer_old", "retain",
            "correct_before", "correct_after", "own_before", "own_after", "peer_before", "peer_after")
    return {"arm": "shared_writepass", "seed": seed, "lr": lr, "steps": steps, "fix_steps": fix_steps, "effective_slots": eff,
            "late": {k: mean(k) for k in keys}, "log": log}


if __name__ == "__main__":
    seed, out = int(sys.argv[1]), sys.argv[2]
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else LR
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else STEPS
    fix_steps = int(sys.argv[5]) if len(sys.argv) > 5 else steps
    f = open(out, "x"); t = time.time(); res = run(seed, lr, steps, fix_steps)
    res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__
    json.dump(res, f); f.close()
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
