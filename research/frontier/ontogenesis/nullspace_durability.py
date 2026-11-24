"""Gate G06: can later training be kept out of the directions old answers use, and what does that cost?

Background (INVESTIGATION.md, Question 9 and its review checks; decision note section 10).
- The computed address edit (`constrained`) writes a correction while keeping 97% of the entity's other
  answers. Protection at write time is solved in this world.
- Durability is not: 20+ phases later only 21.5% of those corrections survive, although the edited
  address receives no gradient afterwards. The loss comes through the shared decoder W, which every
  later phase trains.
- The review struck "a projector can write a correction" (a projector only preserves). This file does
  not use a projector to write. It uses one to protect what was already written, during later training.

Mechanism tested (orthogonal weight modification, OWM; Zeng et al. 2019, published prior art).
logits for (entity e, relation r) are W[r] @ u[e]. A change dW[r] leaves every old answer's logits
unchanged when dW[r] @ u[e] = 0 for every old entity e. Let A (m x n_old) hold the address rows of all
entities seen in earlier phases. After each optimiser step the decoder change is replaced by
    dW[r] <- dW[r] @ P_alpha,   P_alpha = alpha (A A^T + alpha I)^(-1)            (m x m)
alpha -> infinity is no projection (P = I); alpha -> 0 projects onto the complement of span(A), which in
24 dimensions with hundreds of old entities is (nearly) nothing: the decoder is then effectively frozen.
Addresses of new entities still train freely. Whether new entities can be learned through their
addresses alone, with a decoder frozen after the first phases, is the open question.

World, model, seeds, write rule: exactly `constrained` in correction_interference.py (cab_s world, no
exceptions, 16 corrections per phase, bilinear model, 24 dimensions, Adam 0.003, 200 steps per phase,
corrections written by the computed address edit). Seeds 8001-8003.

Arms
  none        no projection; must reproduce the recorded `constrained` numbers exactly
  owm_1       alpha = 1
  owm_1e-2    alpha = 0.01
  owm_1e-4    alpha = 0.0001
  corr_replay no projection; every correction written so far is also trained in each later phase
              (the simplest durability baseline; costs one stored fact per correction)

Measures (model alone)
  corrected_now, corrected_old (20+ phases old), rule_old_corrected, rule_old_clean, infer_late as in
  correction_interference.py; plus
  acquire_now   the phase's new facts answered correctly right after the phase (learning new entities)
  room          trace(P_alpha) / m at the last phase: fraction of decoder directions still free (1 = all)

Predictions, written before any run of this file:
  N1 `none` reproduces correction_interference.py `constrained` exactly (corrected_old 0.215 mean over
     seeds as recorded in the review-check table; same code path and random draws).
  N2 owm_1e-4 makes written corrections durable: corrected_old at least 0.80, and rule_old_corrected
     at least 0.95.
  N3 owm_1e-4 keeps learning new entities: acquire_now within 0.05 of `none`. (The rival outcome, that
     projection leaves no room to learn, would show as acquire_now at least 0.20 below `none`.)
  N4 owm_1e-4 keeps the rule usable for new entities: infer_late within 0.05 of `none`.
  N5 corr_replay: corrected_old at least 0.90 with acquire_now within 0.02 of `none`.
  N6 room falls with alpha: owm_1e-4 leaves at most 0.05 of the decoder free at the end.
  Reading. N2 and N3 holding: once early phases have shaped the decoder, freezing its used directions
  costs nothing here, and correction decay is a property of unprotected shared weights, not of the
  write. N2 holding with N3 failing: durability and acquisition trade off through the shared decoder
  and the curve over alpha is the result. N5 holding: storing and rehearsing corrections is the
  simple baseline any in-model mechanism must beat on cost.
"""
from __future__ import annotations
import json, sys
import numpy as np, torch
from cab_s import Z, R, V, K
from correction_interference import Model, address_edit


class Projected(Model):
    def __init__(self, n, m, seed, alpha=None):
        super().__init__(n, m, seed); self.alpha = alpha; self.P = None

    def set_old(self, old_rows):
        if self.alpha is None or len(old_rows) == 0: self.P = None; return
        A = self.u.detach()[torch.as_tensor(old_rows)].double().T                      # m x n_old
        m = A.shape[0]
        self.P = (self.alpha * torch.linalg.inv(A @ A.T + self.alpha * torch.eye(m, dtype=torch.float64))).float()

    def fit(self, facts, steps=200, frozen=None):
        if self.P is None: return super().fit(facts, steps, frozen)
        import torch.nn.functional as F
        a = torch.tensor(facts); e, r, y = a[:, 0], a[:, 1], a[:, 2]
        for _ in range(steps):
            loss = F.cross_entropy(torch.einsum("bvm,bm->bv", self.W[r], self.u[e]), y)
            W0 = self.W.detach().clone()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            with torch.no_grad(): self.W.copy_(W0 + (self.W - W0) @ self.P)


def run(arm, seed, phases=40, new=64, fix=16, m=24):
    alpha = {"none": None, "corr_replay": None, "owm_1": 1.0, "owm_1e-2": 1e-2, "owm_1e-4": 1e-4}[arm]
    rng = np.random.default_rng(seed); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n); truth = rule[typ].copy()
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    corrected = np.zeros((n, R), bool); when = np.full((n, R), -1)
    L = Projected(n, m, seed, alpha); store = {}; log = []; written = []
    acc = lambda e, r: float((L.predict(e, r) == truth[e, r]).mean()) if len(e) else float("nan")
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]; corr = []
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False); fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True; when[fe, fr] = ph
            corr = [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        L.set_old(np.arange(ph * new))
        rehearse = [(e, r, int(truth[e, r])) for e, r, _ in written if truth[e, r] == _] if arm == "corr_replay" else []
        L.fit(facts + rehearse)
        fe_new = np.array([f[0] for f in facts]); fr_new = np.array([f[1] for f in facts])
        row = {"phase": ph, "acquire_now": acc(fe_new, fr_new)}
        Wn = L.W.detach().numpy().astype(np.float64)
        for e, r, y in corr:
            d, ok = address_edit(Wn, L.u.detach().numpy()[e].astype(np.float64), r, y)
            if ok:
                with torch.no_grad(): L.u[e] += torch.tensor(d, dtype=torch.float32)
                written.append((e, r, y))
            else: store[(e, r)] = y
        if corr:
            c = np.array(corr); row["corrected_now"] = acc(c[:, 0], c[:, 1])
        log.append(row)
    e, r = np.nonzero(shown); ok = L.predict(e, r) == truth[e, r]
    age = (phases - 1) - np.repeat(np.arange(phases), new)[e]; isreg = ~corrected[e, r]; hit = corrected.any(1)[e]
    cold = corrected[e, r] & ((phases - 1) - when[e, r] >= 20)
    late = np.arange((phases - 10) * new, n); ue, ur = np.nonzero(~shown[late])
    mean = lambda k: round(float(np.nanmean([x[k] for x in log if k in x])), 4)
    room = None if L.P is None else round(float(torch.trace(L.P)) / m, 4)
    return {"arm": arm, "seed": seed, "acquire_now": mean("acquire_now"), "corrected_now": mean("corrected_now"),
            "rule_old_corrected": round(float(ok[isreg & (age >= 20) & hit].mean()), 4), "rule_old_clean": round(float(ok[isreg & (age >= 20) & ~hit].mean()), 4),
            "corrected_old": round(float(ok[cold].mean()), 4), "infer_late": round(acc(late[ue], ur), 4), "room_end": room, "in_store": len(store)}


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or [8001, 8002, 8003]
    rows = [run(arm, s) for s in seeds]
    for x in rows: print(json.dumps(x), flush=True)
    if seeds == [8001, 8002, 8003]: json.dump(rows, open(f"results/nullspace_durability/{arm}.json", "w"), indent=1)
