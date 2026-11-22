"""G06b: can protection be localised so corrections last AND new entities still use the rule?

Background (NULLSPACE-DURABILITY-20261122.md). Projecting later decoder updates off the addresses of
ALL old entities keeps 98.5% of corrections after 20+ phases (21.5% without), but new entities' unshown
relations are inferred only 56-66% of the time (99.7% without). In 24 dimensions the projector leaves
no decoder direction free, so the decoder stops carrying the rule forward. A peer session's rule-world
test found the same split from the memory side: shared structure generalises but resists corrections,
private structure corrects but cannot generalise.

Question. Is the inference cost a property of protection itself, or of protecting too much in too
few dimensions? Two ways of protecting less, crossed with width:
  owm_all   (as G06, alpha 1)  project every relation's decoder update off all old entities' addresses
  owm_ent   project every relation's update off the addresses of entities that received a correction
  owm_rel   project relation r's update off the addresses of entities corrected on relation r only
            (the corrected answer's own output; about 2 entities per relation per phase)
  none      no projection
each at m = 24 (as G06) and m = 96. Soft projector P = alpha (A A^T + alpha I)^(-1), alpha = 1.
To trace the curve between G06's two ends (suggested by the ontogenesis session), owm_all is also run
at m = 24 with alpha = 10 and alpha = 100 (weaker projection; alpha -> infinity is no projection).
Arm names: <mode>[-a<alpha>]_m<width>, e.g. owm_rel_m96, owm_all-a10_m24.
Everything else is nullspace_durability.py unchanged (world, write rule, seeds 8001-8003, measures).

Predictions, written before any run of this file:
  L1 none at m = 24 reproduces nullspace_durability.py `none` exactly; owm_all at m = 24 reproduces its
     `owm_1` within 0.01 on corrected_old and infer_late (same projector, computed per relation, so
     floating-point rounding may differ).
  L2 owm_ent at m = 24 behaves like owm_all: unshown relations of new entities (infer_late) at most
     0.80. Sixteen corrected entities per phase span 24 dimensions within two phases.
  L3 owm_rel at m = 24: corrected_old at least 0.90 and infer_late at least 0.85.
  L4 owm_rel at m = 96: corrected_old at least 0.90 and infer_late within 0.03 of none at m = 96.
  L5 owm_all at m = 96 still costs inference: infer_late at most 0.85 (2,500 old entities span 96
     dimensions too).
  L6 the weaker projectors trace a trade-off: from alpha 1 to 10 to 100, corrected_old falls and
     infer_late rises, each monotonically.
  Reading. L3 or L4 holding: protecting only the corrected outputs separates durability from rule use,
  and the G06 cost came from protecting everything. L4 holding with L3 failing: the separation needs
  room, and at 24 dimensions crowding forces the trade-off. L3 and L4 both failing: protection of a
  shared decoder costs the rule even when localised, which supports keeping corrections outside it.
"""
from __future__ import annotations
import json, sys
import numpy as np, torch
import torch.nn.functional as F
from cab_s import Z, R, V, K
from correction_interference import Model, address_edit


class LocalProjected(Model):
    def __init__(self, n, m, seed, mode, alpha=1.0):
        super().__init__(n, m, seed); self.mode = mode; self.P = None; self.m = m; self.alpha = alpha

    def set_protected(self, rows_by_rel):
        """rows_by_rel: list of R arrays of entity rows to protect for that relation (or None)."""
        if self.mode == "none": self.P = None; return
        U = self.u.detach().double(); I = torch.eye(self.m, dtype=torch.float64); Ps = []
        for r in range(R):
            rows = rows_by_rel[r]
            if len(rows) == 0: Ps.append(I.clone()); continue
            A = U[torch.as_tensor(rows)].T
            Ps.append(self.alpha * torch.linalg.inv(A @ A.T + self.alpha * I))
        self.P = torch.stack(Ps).float()                                                   # R x m x m

    def fit(self, facts, steps=200, frozen=None):
        if self.P is None: return super().fit(facts, steps, frozen)
        a = torch.tensor(facts); e, r, y = a[:, 0], a[:, 1], a[:, 2]
        for _ in range(steps):
            loss = F.cross_entropy(torch.einsum("bvm,bm->bv", self.W[r], self.u[e]), y)
            W0 = self.W.detach().clone()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            with torch.no_grad(): self.W.copy_(W0 + torch.einsum("rvm,rmk->rvk", self.W - W0, self.P))


def run(arm, seed, phases=40, new=64, fix=16):
    mode, m = arm.rsplit("_m", 1); m = int(m); alpha = 1.0
    if "-a" in mode: mode, a = mode.split("-a"); alpha = float(a)
    rng = np.random.default_rng(seed); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n); truth = rule[typ].copy()
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    corrected = np.zeros((n, R), bool); when = np.full((n, R), -1)
    L = LocalProjected(n, m, seed, mode, alpha); store = {}; log = []
    corrected_before = np.zeros(n, bool)       # entity received a correction in an EARLIER phase (fix, see below)
    acc = lambda e, r: float((L.predict(e, r) == truth[e, r]).mean()) if len(e) else float("nan")
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]; corr = []
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False); fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True; when[fe, fr] = ph
            corr = [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        old = np.arange(ph * new)
        if mode == "owm_all": L.set_protected([old] * R)
        elif mode == "owm_ent":
            # Fixed 2026-11-22 (Bugbot on PR #202, confirmed by the ontogenesis session): the original test
            # `corrected.any(1) & (when.max(1) < ph)` dropped an entity re-corrected in this phase from
            # protection for the whole fit, unshielding its earlier corrections.
            ce_rows = np.flatnonzero(corrected_before[:ph * new]); L.set_protected([ce_rows] * R)
        elif mode == "owm_rel":
            L.set_protected([np.flatnonzero(corrected[:ph * new, r] & (when[:ph * new, r] < ph)) for r in range(R)])
        else: L.set_protected(None)
        L.fit(facts)
        fe_new = np.array([f[0] for f in facts]); fr_new = np.array([f[1] for f in facts])
        row = {"phase": ph, "acquire_now": acc(fe_new, fr_new)}
        Wn = L.W.detach().numpy().astype(np.float64)
        for e, r, y in corr:
            d, ok = address_edit(Wn, L.u.detach().numpy()[e].astype(np.float64), r, y)
            if ok:
                with torch.no_grad(): L.u[e] += torch.tensor(d, dtype=torch.float32)
            else: store[(e, r)] = y
        if corr:
            c = np.array(corr); row["corrected_now"] = acc(c[:, 0], c[:, 1]); corrected_before[c[:, 0]] = True
        log.append(row)
    e, r = np.nonzero(shown); ok = L.predict(e, r) == truth[e, r]
    age = (phases - 1) - np.repeat(np.arange(phases), new)[e]; isreg = ~corrected[e, r]; hit = corrected.any(1)[e]
    cold = corrected[e, r] & ((phases - 1) - when[e, r] >= 20)
    late = np.arange((phases - 10) * new, n); ue, ur = np.nonzero(~shown[late])
    mean = lambda k: round(float(np.nanmean([x[k] for x in log if k in x])), 4)
    room = None if L.P is None else round(float(L.P.diagonal(dim1=1, dim2=2).sum(-1).mean()) / m, 4)
    return {"arm": arm, "seed": seed, "acquire_now": mean("acquire_now"), "corrected_now": mean("corrected_now"),
            "rule_old_corrected": round(float(ok[isreg & (age >= 20) & hit].mean()), 4), "rule_old_clean": round(float(ok[isreg & (age >= 20) & ~hit].mean()), 4),
            "corrected_old": round(float(ok[cold].mean()), 4), "infer_late": round(acc(late[ue], ur), 4), "room_end": room, "in_store": len(store)}


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:] if s.isdigit()] or [8001, 8002, 8003]
    rows = [run(arm, s) for s in seeds]
    for x in rows: print(json.dumps(x), flush=True)
    out = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else "results/nullspace_local"
    if seeds == [8001, 8002, 8003]: json.dump(rows, open(f"{out}/{arm}.json", "w"), indent=1)
