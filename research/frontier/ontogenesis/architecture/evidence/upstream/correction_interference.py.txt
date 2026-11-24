"""How should a correction be written, and does a write that harms nothing else exist?

Background (established by the two controls in results/forgetting_split/): in this world, old regular
facts are lost only on entities that later received a correction; exceptions alone cause no such loss.
A correction was trained on that one fact, without the entity's other facts.

Question. Is the damage a property of the write rule (a better write exists and can be found), or of
what the model can represent (no write inside the model can hold the correction without harming
something, so separate storage is necessary)?

World: as cab_s.py with no exceptions (EPS 0) and 16 corrections per phase. Model: bilinear,
logits = W[r] @ u[e], 24 dimensions, Adam 0.003, 200 steps per phase (as in every earlier run).
A correction replaces the value of one shown fact of an earlier entity by an arbitrary value.

Arms (how a correction is written):
  single          trained with the phase's new facts, alone among that entity's facts (as before)
  entity_replay   same, plus the corrected entity's other shown facts in the batch
  random_replay   same, plus an equal number of facts from other, uncorrected entities (control)
  frozen_address  same as single, but the corrected entity's address row is held fixed
  store           never trained; kept in an exact store (the model alone will not know it)
  constrained     never trained; written by a computed edit of that entity's address only:
                  the smallest change (L1) that gives the new value a margin of 1 logit and keeps the
                  model's current answer on the entity's other seven relations. It uses the model
                  and the correction only, no stored facts. Found by linear programming, which also
                  certifies when no such edit exists; then the correction goes to the exact store.

Measured, model alone unless stated:
  corrected_now      corrections answered correctly right after the phase
  collateral_now     the corrected entities' other shown facts right after the phase
  same_type_now      relation-r facts of other entities of the same hidden type as a corrected entity
  At the end: rule_old_corrected (regular facts, 20+ phases old, on entities that received a
  correction), rule_old_clean (same on other entities), corrected_old (corrections 20+ phases old),
  infer_late (unseen relations of the last ten phases' entities), lp_feasible (constrained arm).

Predictions, written before any run of this file (seeds 8001-8003):
  K1 single reproduces the damage: collateral_now at most 0.85, rule_old_corrected at most 0.80.
  K2 an address-only edit that harms none of the entity's other answers exists for at least 80% of
     corrections. Below 50% would mean the model cannot hold such corrections: a representation limit.
  K3 constrained: collateral_now at least 0.98, corrected_now at least 0.80, rule_old_corrected at least 0.95.
  K4 entity_replay: collateral_now at least 0.97 and corrected_now at least 0.95; random_replay gives
     no such protection (collateral_now within 0.05 of single).
  K5 frozen_address protects the entity (collateral_now at least 0.97) but then either the write
     fails (corrected_now at most 0.5) or the rule is damaged (same_type_now at least 0.02 below single).
  K6 a correction written into the model decays under later training: corrected_old at least 0.10
     below corrected_now in every arm that writes into the model.
  If K2 and K3 hold, write-time damage is a property of the write rule, not of the representation.
  If K2 fails, the model class cannot absorb corrections and separate storage is necessary.
  If K4 holds but K3 fails, rehearsing the entity's own facts is needed; an edit from the model alone is not enough.
"""
from __future__ import annotations
import json, sys
import numpy as np, torch, torch.nn.functional as F
from scipy.optimize import linprog
from cab_s import Z, R, V, K


class Model:
    def __init__(self, n, m, seed, lr=0.003):
        torch.manual_seed(seed)
        self.u = torch.nn.Parameter(torch.randn(n, m) * 0.1); self.W = torch.nn.Parameter(torch.randn(R, V, m) * 0.1)
        self.opt = torch.optim.Adam([self.u, self.W], lr=lr)

    def fit(self, facts, steps=200, frozen=None):
        a = torch.tensor(facts); e, r, y = a[:, 0], a[:, 1], a[:, 2]
        keep = self.u.data[frozen].clone() if frozen is not None else None
        for _ in range(steps):
            loss = F.cross_entropy(torch.einsum("bvm,bm->bv", self.W[r], self.u[e]), y)
            self.opt.zero_grad(); loss.backward()
            if frozen is not None: self.u.grad[frozen] = 0
            self.opt.step()
            if frozen is not None: self.u.data[frozen] = keep
    @torch.no_grad()
    def predict(self, e, r):
        return torch.einsum("bvm,bm->bv", self.W[torch.as_tensor(r)], self.u[torch.as_tensor(e)]).argmax(-1).numpy()


def address_edit(W, u, r, y_new, margin=1.0):
    """Smallest L1 change d to u such that relation r answers y_new with `margin`, and every other
    relation keeps its current answer (with half its current margin, capped at `margin`).
    Returns (d, True) or (None, False) when the linear programme is infeasible."""
    m = u.shape[0]; rows, rhs = [], []
    for rr in range(R):
        lg = W[rr] @ u
        t = y_new if rr == r else int(lg.argmax())
        if rr == r: g = margin
        else:
            srt = np.sort(lg); g = min(margin, 0.5 * float(srt[-1] - srt[-2]))
        D = np.delete(W[rr], t, axis=0) - W[rr][t]                       # competitor minus target, [V-1, m]
        rows.append(D); rhs.append(-g - D @ u)
    A = np.concatenate(rows); b = np.concatenate(rhs)
    res = linprog(np.ones(2 * m), A_ub=np.concatenate([A, -A], 1), b_ub=b, bounds=[(0, None)] * (2 * m), method="highs")
    if res.status != 0: return None, False
    return res.x[:m] - res.x[m:], True


def run(arm, seed, phases=40, new=64, fix=16, m=24):
    rng = np.random.default_rng(seed); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n); truth = rule[typ].copy()
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    corrected = np.zeros((n, R), bool); when = np.full((n, R), -1)
    L = Model(n, m, seed); store = {}; log = []; n_corr = n_feasible = 0
    acc = lambda e, r: float((L.predict(e, r) == truth[e, r]).mean()) if len(e) else float("nan")
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]; corr = []
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False); fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True; when[fe, fr] = ph
            corr = [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        ce = sorted({c[0] for c in corr})
        if arm == "single": L.fit(facts + corr)
        elif arm == "entity_replay":
            extra = [(e, int(r), int(truth[e, r])) for e in ce for r in np.flatnonzero(shown[e]) if (e, int(r)) not in {(c[0], c[1]) for c in corr}]
            L.fit(facts + corr + extra)
        elif arm == "random_replay":
            k = sum(int(shown[e].sum()) for e in ce) - len(corr)
            if k > 0:
                oe2, or2 = np.nonzero(shown[:ph * new]); ok = ~np.isin(oe2, ce); oe2, or2 = oe2[ok], or2[ok]
                pk = rng.choice(len(oe2), k, replace=False)
                L.fit(facts + corr + [(int(a), int(b), int(truth[a, b])) for a, b in zip(oe2[pk], or2[pk])])
            else: L.fit(facts + corr)
        elif arm == "frozen_address": L.fit(facts + corr, frozen=torch.tensor(ce, dtype=torch.long) if ce else None)
        elif arm == "store":
            L.fit(facts)
            for c in corr: store[(c[0], c[1])] = c[2]
        elif arm == "constrained":
            L.fit(facts)
            Wn = L.W.detach().numpy().astype(np.float64)
            for e, r, y in corr:
                d, ok = address_edit(Wn, L.u.detach().numpy()[e].astype(np.float64), r, y); n_corr += 1; n_feasible += ok
                if ok:
                    with torch.no_grad(): L.u[e] += torch.tensor(d, dtype=torch.float32)
                else: store[(e, r)] = y
        else: raise SystemExit(arm)
        if corr:
            c = np.array(corr); row = {"phase": ph, "corrected_now": acc(c[:, 0], c[:, 1])}
            oe3, or3 = np.nonzero(shown[ce]); oe3 = np.array(ce)[oe3]; keep = ~corrected[oe3, or3]
            row["collateral_now"] = acc(oe3[keep], or3[keep])
            se, sr = [], []
            for e, r, _ in corr:
                mates = np.flatnonzero((typ[:(ph + 1) * new] == typ[e]) & shown[:(ph + 1) * new, r] & ~corrected[:(ph + 1) * new, r]); mates = mates[mates != e]
                se += list(mates); sr += [r] * len(mates)
            row["same_type_now"] = acc(np.array(se, dtype=int), np.array(sr, dtype=int)); log.append(row)
    e, r = np.nonzero(shown); ok = L.predict(e, r) == truth[e, r]
    age = (phases - 1) - np.repeat(np.arange(phases), new)[e]; isreg = ~corrected[e, r]; hit = corrected.any(1)[e]
    cold = corrected[e, r] & ((phases - 1) - when[e, r] >= 20)
    late = np.arange((phases - 10) * new, n); ue, ur = np.nonzero(~shown[late])
    mean = lambda k: round(float(np.nanmean([x[k] for x in log])), 4)
    out = {"arm": arm, "seed": seed, "corrected_now": mean("corrected_now"), "collateral_now": mean("collateral_now"), "same_type_now": mean("same_type_now"),
           "rule_old_corrected": round(float(ok[isreg & (age >= 20) & hit].mean()), 4), "rule_old_clean": round(float(ok[isreg & (age >= 20) & ~hit].mean()), 4),
           "corrected_old": round(float(ok[cold].mean()), 4), "infer_late": round(acc(late[ue], ur), 4), "in_store": len(store)}
    if arm == "constrained": out["lp_feasible"] = round(n_feasible / max(1, n_corr), 4)
    return out


if __name__ == "__main__":
    arm = sys.argv[1]; rows = [run(arm, s) for s in (8001, 8002, 8003)]
    for x in rows: print(json.dumps(x), flush=True)
    json.dump(rows, open(f"results/correction_interference/{arm}.json", "w"))
