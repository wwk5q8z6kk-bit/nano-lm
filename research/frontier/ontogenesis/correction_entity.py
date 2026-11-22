"""Two checks on Question 9 requested by the independent review (registered in INVESTIGATION.md before any run).

correction_interference.py is left unchanged. This file repeats its world, model, seeds and arms exactly, and adds:

1. Whole-entity scoring. For every corrected entity, the model's answers on all eight relations are read just
   before the phase's write and just after it. For the seven relations that were not corrected it reports the share
   of answers kept, split into the other shown facts (what `collateral_now` measured) and the unshown, rule-implied
   ones (never scored before), and accuracy against the truth before and after.
2. single_rowscale: the original damaging arm `single`, changed in one thing only. Before each phase's training,
   the Adam state of the corrected entities' address rows is set to that of recently trained rows: first moment
   0, second moment the per-dimension median over the previous phase's new entities. A row idle for many phases
   has a decayed second moment, which makes its first steps several times larger than usual. If that drives
   the damage, this arm keeps much more.

usage: correction_entity.py <arm>     (arms: single entity_replay random_replay frozen_address store constrained single_rowscale)
"""
from __future__ import annotations
import json, sys
import numpy as np, torch
from cab_s import Z, R, V, K
from correction_interference import Model, address_edit


def run(arm, seed, phases=40, new=64, fix=16, m=24):
    rng = np.random.default_rng(seed); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n); truth = rule[typ].copy()
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    corrected = np.zeros((n, R), bool); when = np.full((n, R), -1)
    L = Model(n, m, seed); store = {}; log = []
    acc = lambda e, r: float((L.predict(e, r) == truth[e, r]).mean()) if len(e) else float("nan")
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]; corr = []
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False); fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True; when[fe, fr] = ph
            corr = [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        ce = sorted({c[0] for c in corr})
        cr = {(c[0], c[1]) for c in corr}
        if ce:                                                            # answers on all relations, just before the write
            E0 = np.repeat(ce, R); R0 = np.tile(np.arange(R), len(ce)); P0 = L.predict(E0, R0)
        arm_fit = "single" if arm == "single_rowscale" else arm
        if arm == "single_rowscale" and ce and L.u in L.opt.state:
            st = L.opt.state[L.u]; prev = np.arange((ph - 1) * new, ph * new)
            with torch.no_grad():
                st["exp_avg"][ce] = 0.0
                st["exp_avg_sq"][ce] = st["exp_avg_sq"][prev].median(0).values
        if arm_fit == "single": L.fit(facts + corr)
        elif arm_fit == "entity_replay":
            extra = [(e, int(r), int(truth[e, r])) for e in ce for r in np.flatnonzero(shown[e]) if (e, int(r)) not in cr]
            L.fit(facts + corr + extra)
        elif arm_fit == "random_replay":
            k = sum(int(shown[e].sum()) for e in ce) - len(corr)
            if k > 0:
                oe2, or2 = np.nonzero(shown[:ph * new]); ok = ~np.isin(oe2, ce); oe2, or2 = oe2[ok], or2[ok]
                pk = rng.choice(len(oe2), k, replace=False)
                L.fit(facts + corr + [(int(a), int(b), int(truth[a, b])) for a, b in zip(oe2[pk], or2[pk])])
            else: L.fit(facts + corr)
        elif arm_fit == "frozen_address": L.fit(facts + corr, frozen=torch.tensor(ce, dtype=torch.long) if ce else None)
        elif arm_fit == "store":
            L.fit(facts)
            for c in corr: store[(c[0], c[1])] = c[2]
        elif arm_fit == "constrained":
            L.fit(facts)
            Wn = L.W.detach().numpy().astype(np.float64)
            for e, r, y in corr:
                d, ok = address_edit(Wn, L.u.detach().numpy()[e].astype(np.float64), r, y)
                if ok:
                    with torch.no_grad(): L.u[e] += torch.tensor(d, dtype=torch.float32)
                else: store[(e, r)] = y
        else: raise SystemExit(arm)
        if corr:
            c = np.array(corr); row = {"phase": ph, "corrected_now": acc(c[:, 0], c[:, 1])}
            P1 = L.predict(E0, R0); other = np.array([(e, r) not in cr for e, r in zip(E0, R0)])
            sh = shown[E0, R0] & other; un = ~shown[E0, R0] & other
            row.update(kept_shown=float((P1 == P0)[sh].mean()), kept_unshown=float((P1 == P0)[un].mean()), kept_all7=float((P1 == P0)[other].mean()),
                       acc_shown_before=float((P0 == truth[E0, R0])[sh].mean()), acc_shown_after=float((P1 == truth[E0, R0])[sh].mean()),
                       acc_unshown_before=float((P0 == truth[E0, R0])[un].mean()), acc_unshown_after=float((P1 == truth[E0, R0])[un].mean()))
            oe3, or3 = np.nonzero(shown[ce]); oe3 = np.array(ce)[oe3]; keep = ~corrected[oe3, or3]
            row["collateral_now"] = acc(oe3[keep], or3[keep]); log.append(row)
    mean = lambda k: round(float(np.nanmean([x[k] for x in log])), 4)
    keys = ("corrected_now", "collateral_now", "kept_shown", "kept_unshown", "kept_all7", "acc_shown_before", "acc_shown_after", "acc_unshown_before", "acc_unshown_after")
    e, r = np.nonzero(shown); ok = L.predict(e, r) == truth[e, r]; cold = corrected[e, r] & ((phases - 1) - when[e, r] >= 20)
    return {"arm": arm, "seed": seed, **{k: mean(k) for k in keys}, "corrected_old": round(float(ok[cold].mean()), 4)}


if __name__ == "__main__":
    arm = sys.argv[1]; rows = [run(arm, s) for s in (8001, 8002, 8003)]
    for x in rows: print(json.dumps(x), flush=True)
    json.dump(rows, open(f"results/correction_interference/entity-{arm}.json", "w"))
