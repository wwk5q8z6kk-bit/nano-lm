"""Last rival explanation for the correction damage: is it the optimiser?

Established (correction_interference.py, correction_margins.py): a correction trained alone damages the
entity's other facts; a harmless address exists for 99.8% of corrections; imposing the entity's other
answers as constraints removes the damage; the smallest unconstrained edit is the most damaging.

Still open (named by the owner): is the damage produced by the optimiser - Adam's per-coordinate steps,
stale moment estimates on a long-untouched row, or simply training for too many steps?

Here the correction is written by gradient steps on the entity's address only (decoder frozen, so the
shared rule cannot be touched), using only the corrected fact, under three optimisers:
  gd_stop     plain gradient descent, stopped as soon as the new value leads by 1 logit
  adam_stop   Adam with fresh state, stopped as soon as the new value leads by 1 logit
  adam_200    Adam with fresh state, 200 steps (the number used everywhere else)
Same world, model, seeds (8001-8003) and measures as the other correction arms.

Prediction, written before any run of this file:
  O1 all three damage the entity's other facts: collateral_now at most 0.85 in each.
     If any of them reaches 0.95, the damage is an artefact of the optimiser or of over-training, and
     "missing constraints" is not the right explanation.
"""
from __future__ import annotations
import json, sys
import numpy as np, torch, torch.nn.functional as F
from cab_s import Z, R, V, K
from correction_interference import Model


def write(L, corr, arm):
    e = torch.tensor([c[0] for c in corr]); r = torch.tensor([c[1] for c in corr]); y = torch.tensor([c[2] for c in corr])
    W = L.W.detach()[r]; U = L.u.detach()[e].clone().requires_grad_(True)
    opt = torch.optim.SGD([U], lr=0.2) if arm == "gd_stop" else torch.optim.Adam([U], lr=0.003)
    done = torch.zeros(len(corr), dtype=torch.bool); frozen = U.detach().clone()
    for step in range(200 if arm == "adam_200" else 5000):
        lg = torch.einsum("bvm,bm->bv", W, U)
        if arm != "adam_200":
            top2 = lg.detach().topk(2, -1); lead = lg.detach().gather(1, y[:, None])[:, 0] - torch.where(top2.indices[:, 0] == y, top2.values[:, 1], top2.values[:, 0])
            newly = (lead >= 1.0) & ~done; frozen[newly] = U.detach()[newly]; done |= newly
            if done.all(): break
        loss = F.cross_entropy(lg, y, reduction="sum"); opt.zero_grad(); loss.backward()
        U.grad[done] = 0; opt.step()
        with torch.no_grad(): U[done] = frozen[done]
    with torch.no_grad(): L.u[e] = U.detach()
    return step


def run(arm, seed, phases=40, new=64, fix=16, m=24):
    rng = np.random.default_rng(seed); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n); truth = rule[typ].copy()
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    corrected = np.zeros((n, R), bool); when = np.full((n, R), -1)
    L = Model(n, m, seed); log = []; steps = []
    acc = lambda e, r: float((L.predict(e, r) == truth[e, r]).mean()) if len(e) else float("nan")
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]; corr = []
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False); fe, fr = oe[pick], orr[pick]
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True; when[fe, fr] = ph
            corr = [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        L.fit(facts)
        if corr:
            steps.append(write(L, corr, arm))
            c = np.array(corr); ce = sorted({x[0] for x in corr}); row = {"corrected_now": acc(c[:, 0], c[:, 1])}
            oe3, or3 = np.nonzero(shown[ce]); oe3 = np.array(ce)[oe3]; keep = ~corrected[oe3, or3]
            row["collateral_now"] = acc(oe3[keep], or3[keep]); log.append(row)
    e, r = np.nonzero(shown); ok = L.predict(e, r) == truth[e, r]
    age = (phases - 1) - np.repeat(np.arange(phases), new)[e]; isreg = ~corrected[e, r]; hit = corrected.any(1)[e]
    cold = corrected[e, r] & ((phases - 1) - when[e, r] >= 20)
    mean = lambda k: round(float(np.nanmean([x[k] for x in log])), 4)
    return {"arm": arm, "seed": seed, "write_steps": round(float(np.mean(steps)), 1), "corrected_now": mean("corrected_now"), "collateral_now": mean("collateral_now"),
            "rule_old_corrected": round(float(ok[isreg & (age >= 20) & hit].mean()), 4), "corrected_old": round(float(ok[cold].mean()), 4)}


if __name__ == "__main__":
    arm = sys.argv[1]; rows = [run(arm, s) for s in (8001, 8002, 8003)]
    for x in rows: print(json.dumps(x), flush=True)
    json.dump(rows, open(f"results/correction_interference/optimizer-{arm}.json", "w"))
