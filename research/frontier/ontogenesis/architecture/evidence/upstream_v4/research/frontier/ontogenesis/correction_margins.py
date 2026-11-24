"""Follow-up to correction_interference.py (which is not modified): size of the write, or missing constraints?

Established there: an address-only edit that changes none of the entity's other answers exists for 99.8%
of corrections; training the correction alone does not find it; rehearsing the entity's own facts does;
the computed edit is harmless at once but fragile later (old regular facts on those entities 83.5%,
the correction itself 21.5% after 20+ phases).

Two open readings of "training alone does damage":
  E2  the write is simply too large or goes the wrong way; a smallest-possible write would be harmless
      without any knowledge of what else must hold.
  E3  the write lacks constraints; even the smallest write that installs the correction damages the
      entity's other answers unless those answers are imposed as constraints.

Arms (all write the correction by a computed address-only edit; nothing else changes):
  minimal          smallest (L1) edit that gives the new value a 1-logit margin; no other constraint
  half_m1          the registered `constrained` arm (margin 1; others keep at least half their margin) - must reproduce it
  half_m3, half_m6 same with a 3- and 6-logit margin for the correction
  keep_m3          margin 3, and every other relation keeps at least 90% of its present margin

Predictions, written before any run of this file (seeds 8001-8003, same streams as the registered arms):
  M1 (E3 over E2) minimal: collateral_now at most 0.90. If it is 0.97 or more, the damage in the
     `single` arm was mostly the size or direction of the step, not missing constraints.
  M2 a larger margin makes the correction last: corrected_old rises with margin (1 < 3 < 6) and reaches 0.60 at 6.
  M3 keeping the neighbours' margins makes them last: keep_m3 ends with old regular facts on corrected
     entities at 0.93 or more (against 0.835 for half_m1), with feasibility still at least 0.90.
  M4 no margin makes a written correction permanent: corrected_old stays below 0.90 in every arm.
"""
from __future__ import annotations
import json, sys
import numpy as np, torch
from scipy.optimize import linprog
from cab_s import Z, R, V, K
from correction_interference import Model


def edit(W, u, r, y_new, margin, preserve):
    m = u.shape[0]; rows, rhs = [], []
    for rr in range(R):
        lg = W[rr] @ u
        if rr == r: t, g = y_new, margin
        elif preserve == "none": continue
        else:
            t = int(lg.argmax()); srt = np.sort(lg); gap = float(srt[-1] - srt[-2])
            g = min(1.0, 0.5 * gap) if preserve == "half" else 0.9 * gap
        D = np.delete(W[rr], t, axis=0) - W[rr][t]; rows.append(D); rhs.append(-g - D @ u)
    A = np.concatenate(rows); b = np.concatenate(rhs)
    res = linprog(np.ones(2 * m), A_ub=np.concatenate([A, -A], 1), b_ub=b, bounds=[(0, None)] * (2 * m), method="highs")
    return (res.x[:m] - res.x[m:], True) if res.status == 0 else (None, False)


def run(margin, preserve, seed, phases=40, new=64, fix=16, m=24):
    rng = np.random.default_rng(seed); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n); truth = rule[typ].copy()
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    corrected = np.zeros((n, R), bool); when = np.full((n, R), -1)
    L = Model(n, m, seed); log = []; n_corr = n_ok = 0; norms = []
    acc = lambda e, r: float((L.predict(e, r) == truth[e, r]).mean()) if len(e) else float("nan")
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]; corr = []
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False); fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True; when[fe, fr] = ph
            corr = [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        L.fit(facts)
        Wn = L.W.detach().numpy().astype(np.float64)
        for e, r, y in corr:
            d, ok = edit(Wn, L.u.detach().numpy()[e].astype(np.float64), r, y, margin, preserve); n_corr += 1; n_ok += ok
            if ok:
                norms.append(float(np.abs(d).sum()))
                with torch.no_grad(): L.u[e] += torch.tensor(d, dtype=torch.float32)
        if corr:
            c = np.array(corr); ce = sorted({x[0] for x in corr}); row = {"corrected_now": acc(c[:, 0], c[:, 1])}
            oe3, or3 = np.nonzero(shown[ce]); oe3 = np.array(ce)[oe3]; keep = ~corrected[oe3, or3]
            row["collateral_now"] = acc(oe3[keep], or3[keep]); log.append(row)
    e, r = np.nonzero(shown); ok = L.predict(e, r) == truth[e, r]
    age = (phases - 1) - np.repeat(np.arange(phases), new)[e]; isreg = ~corrected[e, r]; hit = corrected.any(1)[e]
    cold = corrected[e, r] & ((phases - 1) - when[e, r] >= 20)
    mean = lambda k: round(float(np.nanmean([x[k] for x in log])), 4)
    return {"arm": f"{preserve}_m{margin:g}", "seed": seed, "feasible": round(n_ok / max(1, n_corr), 4), "edit_l1": round(float(np.mean(norms)), 3),
            "corrected_now": mean("corrected_now"), "collateral_now": mean("collateral_now"),
            "rule_old_corrected": round(float(ok[isreg & (age >= 20) & hit].mean()), 4), "corrected_old": round(float(ok[cold].mean()), 4)}


if __name__ == "__main__":
    margin, preserve = float(sys.argv[1]), sys.argv[2]; rows = [run(margin, preserve, s) for s in (8001, 8002, 8003)]
    for x in rows: print(json.dumps(x), flush=True)
    json.dump(rows, open(f"results/correction_interference/margins-{preserve}-m{margin:g}.json", "w"))
