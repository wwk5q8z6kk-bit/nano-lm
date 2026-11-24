"""G06c: does localised protection still separate durability from rule use with a NONLINEAR readout?

STATUS: NOT RUN AS REGISTERED. The mechanics check (below) showed the planned write rule cannot be used
with this model, so no predictions were written and no registered seed (8001-8003) was run.

Mechanics finding, development seeds 1, 2 and 3 only (exploratory, 12 phases, arm `none`):
with the two-layer ReLU decoder frozen, optimising the corrected entity's address alone, with no
constraint at all, makes the new value win in only 44%, 48% and 35% of corrections. That figure did not
change with the address step size (0.05 to 0.3), the hidden width (32 or 128), or more steps (up to
3000). With the constraint to keep the entity's other answers it is 26-40%. In the bilinear model the
same address-only edit is feasible for 99.8% (linear programme, correction_interference.py).
Likely cause, not proven: the hidden layer's outputs are non-negative, so for a fixed W2[r] some values
can never be the argmax for any address. Consequence: G06b's recipe (exact address edit, then localised
protection of the decoder) does not carry over to a nonlinear readout. There, the write itself must
change shared weights, so protection would be needed at write time as well as afterwards. That is a
different experiment, and it is not started here.

Background. G06b (NULLSPACE-LOCAL-20261122.md): in the bilinear model, projecting later decoder updates
off only the corrected outputs keeps corrections (94-99.6% after 20+ phases) and rule inference for new
entities (99.1-99.9%). There the projector is exact because logits are linear in the decoder. Here the
readout is a two-layer network, so protection can only be first order: gradient projection (orthogonal
gradient descent with ground-truth-logit gradients, OGD-GTL; Farajtabar et al. 2020, published).

Model. logits(e, r) = W2[r] @ relu(W1 @ u[e] + b1); m = 24, hidden width H. Adam 0.003, 200 steps per
phase, as in every earlier arm. World and stream: cab_s, no exceptions, 16 corrections per phase,
40 phases of 64 entities, as in correction_interference.py.

Write rule (the model has no exact edit). The corrected entity's address alone is optimised, decoder
frozen, fresh Adam, WRITE_STEPS steps, on: cross-entropy of the new value at the corrected relation
+ cross-entropy of the model's own current answers at the entity's other seven relations. It uses the
model and the correction only, like the computed edit; no stored facts.

Protection. After each phase's writes, protected outputs (e, r, y) contribute the gradient of logit y at
(e, r) with respect to the decoder (W1, b1, W2). These are orthonormalised into a basis Q; during every
later phase each decoder update is replaced by its component orthogonal to Q.
  none        no protection
  ogd_rel     protect each corrected output (e, r, new value)
  ogd_ent     protect every relation of each corrected entity (its current answers)
  ogd_rand    protect as many outputs as ogd_ent adds each phase, drawn at random from old shown facts
              (same number of constraints, not localised to corrections)

Measures: as nullspace_durability.py (acquire_now, corrected_now, corrected_old, rule_old_corrected,
rule_old_clean, infer_late) plus collateral_now (the corrected entities' other answers kept at writing)
and basis_size at the end.
"""
from __future__ import annotations
import json, sys
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from cab_s import Z, R, V, K

H, WRITE_STEPS = 32, None          # WRITE_STEPS fixed by the mechanics check
torch.set_num_threads(2)


class Net(nn.Module):
    def __init__(s, n, m, seed):
        super().__init__(); torch.manual_seed(seed)
        s.u = nn.Parameter(torch.randn(n, m) * 0.1); s.W1 = nn.Parameter(torch.randn(H, m) * (1 / m) ** 0.5)
        s.b1 = nn.Parameter(torch.zeros(H)); s.W2 = nn.Parameter(torch.randn(R, V, H) * 0.1)
        s.opt = torch.optim.Adam(s.parameters(), lr=0.003); s.Q = None
    def dec(s): return [s.W1, s.b1, s.W2]
    def logits(s, e, r, u=None):
        h = F.relu((s.u[e] if u is None else u) @ s.W1.T + s.b1)
        return torch.einsum("bvh,bh->bv", s.W2[r], h)
    @torch.no_grad()
    def predict(s, e, r): return s.logits(torch.as_tensor(e), torch.as_tensor(r)).argmax(-1).numpy()
    def flat_dec(s): return torch.cat([p.detach().reshape(-1) for p in s.dec()])
    def set_dec(s, v):
        i = 0
        with torch.no_grad():
            for p in s.dec(): p.copy_(v[i:i + p.numel()].view_as(p)); i += p.numel()
    def fit(s, facts, steps=200):
        a = torch.tensor(facts); e, r, y = a[:, 0], a[:, 1], a[:, 2]
        for _ in range(steps):
            loss = F.cross_entropy(s.logits(e, r), y)
            d0 = s.flat_dec() if s.Q is not None else None
            s.opt.zero_grad(); loss.backward(); s.opt.step()
            if d0 is not None:
                delta = s.flat_dec() - d0; delta = delta - s.Q @ (s.Q.T @ delta); s.set_dec(d0 + delta)
    def protect(s, outputs):
        """outputs: list of (e, r, y). Adds orthonormalised decoder gradients of logit y at (e, r)."""
        new = []
        for e, r, y in outputs:
            for p in s.dec(): p.grad = None
            s.logits(torch.tensor([e]), torch.tensor([r]))[0, y].backward()
            new.append(torch.cat([p.grad.reshape(-1) for p in s.dec()]).double())
        for p in s.parameters(): p.grad = None
        if not new: return
        G = torch.stack(new, 1)
        if s.Q is not None: G = G - s.Q.double() @ (s.Q.double().T @ G)
        q, rr = torch.linalg.qr(G); keep = rr.diagonal().abs() > 1e-6 * max(1.0, float(rr.diagonal().abs().max()))
        q = q[:, keep].float()
        s.Q = q if s.Q is None else torch.cat([s.Q, q], 1)

    def write(s, e, r, y, steps):
        with torch.no_grad(): cur = s.predict(np.full(R, e), np.arange(R))
        others = [rr for rr in range(R) if rr != r]
        u = s.u.detach()[e:e + 1].clone().requires_grad_(True); opt = torch.optim.Adam([u], lr=0.01)
        rs = torch.tensor([r] + others); ys = torch.tensor([y] + [int(cur[rr]) for rr in others])
        for _ in range(steps):
            loss = F.cross_entropy(s.logits(None, rs, u.expand(R, -1)), ys); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): s.u[e] = u[0]


def run(arm, seed, phases=40, new=64, fix=16, m=24, write_steps=None):
    write_steps = write_steps or WRITE_STEPS
    rng = np.random.default_rng(seed); prng = np.random.default_rng(seed + 99); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n); truth = rule[typ].copy()
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    corrected = np.zeros((n, R), bool); when = np.full((n, R), -1)
    L = Net(n, m, seed); log = []
    acc = lambda e, r: float((L.predict(e, r) == truth[e, r]).mean()) if len(e) else float("nan")
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]; corr = []
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False); fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True; when[fe, fr] = ph
            corr = [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        L.fit(facts)
        fe_new = np.array([f[0] for f in facts]); fr_new = np.array([f[1] for f in facts])
        row = {"phase": ph, "acquire_now": acc(fe_new, fr_new)}
        if corr:
            ce = sorted({c[0] for c in corr}); before = {e: L.predict(np.full(R, e), np.arange(R)) for e in ce}
            for e, r, y in corr: L.write(e, r, y, write_steps)
            c = np.array(corr); row["corrected_now"] = acc(c[:, 0], c[:, 1])
            kept = [float(L.predict(np.array([e]), np.array([rr]))[0] == before[e][rr]) for e in ce for rr in range(R) if not corrected[e, rr] or when[e, rr] != ph]
            row["collateral_now"] = float(np.mean(kept))
            prot = []
            if arm == "ogd_rel": prot = corr
            elif arm in ("ogd_ent", "ogd_rand"):
                ent_out = [(e, rr, int(L.predict(np.array([e]), np.array([rr]))[0])) for e in ce for rr in range(R)]
                if arm == "ogd_ent": prot = ent_out
                else:
                    oe2, or2 = np.nonzero(shown[:ph * new]); pk = prng.choice(len(oe2), len(ent_out), replace=False)
                    prot = [(int(a), int(b), int(L.predict(np.array([a]), np.array([b]))[0])) for a, b in zip(oe2[pk], or2[pk])]
            L.protect(prot)
        log.append(row)
    e, r = np.nonzero(shown); ok = L.predict(e, r) == truth[e, r]
    age = (phases - 1) - np.repeat(np.arange(phases), new)[e]; isreg = ~corrected[e, r]; hit = corrected.any(1)[e]
    cold = corrected[e, r] & ((phases - 1) - when[e, r] >= 20)
    late = np.arange((phases - 10) * new, n); ue, ur = np.nonzero(~shown[late])
    mean = lambda k: round(float(np.nanmean([x[k] for x in log if k in x])), 4)
    return {"arm": arm, "seed": seed, "acquire_now": mean("acquire_now"), "corrected_now": mean("corrected_now"), "collateral_now": mean("collateral_now"),
            "rule_old_corrected": round(float(ok[isreg & (age >= 20) & hit].mean()), 4), "rule_old_clean": round(float(ok[isreg & (age >= 20) & ~hit].mean()), 4),
            "corrected_old": round(float(ok[cold].mean()), 4), "infer_late": round(acc(late[ue], ur), 4),
            "basis_size": 0 if L.Q is None else int(L.Q.shape[1]), "decoder_params": int(sum(p.numel() for p in L.dec()))}


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [8001, 8002, 8003]
    assert WRITE_STEPS is not None, "fixed by the mechanics check"
    rows = [run(arm, s) for s in seeds]
    for x in rows: print(json.dumps(x), flush=True)
    json.dump(rows, open(f"results/nullspace_nonlinear/{arm}.json", "w"), indent=1)
