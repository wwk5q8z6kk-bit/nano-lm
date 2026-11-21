"""Does the routing gain need an oracle? A write-time detector that judges a fact before fitting it.

Separate from routing_test.py, whose arms and results are unchanged. Same world, learner and settings
(EPS 0.1, 16 corrections per phase, 40 phases, lr 0.003, 200 steps per phase).

Status of the earlier result, so this test is read correctly: the two controls showed that old regular
facts are lost because of late corrections trained on one fact, not because of exceptions; and the
inference gain from oracle routing is label-noise removal. So "routing" here means two things:
keeping arbitrary values out of address fitting, and not writing corrections into the model by gradient.

Detector (uses only the current model and the entity's own facts; no look-ahead, no labels):
  For a new entity with 3 shown facts: for each fact, fit a temporary address to the other two with
  the decoder frozen (60 Adam steps), and ask whether the held-out fact is predicted.
    all three predicted   -> train on all three
    otherwise             -> send to the exact store only the single fact with the lowest held-out
                             probability; train on the other two
  For a correction to an old entity: fit a temporary address to that entity's other shown facts and
  ask whether the new value is predicted; if not, send it to the exact store and do not train on it.
  Before phase 5 the model is too weak to judge, so everything trains.
Arms, seeds 7001-7003: mixed, oracle, loo (the detector above).
Reported: inference on unseen relations of the last ten phases' entities; old regular facts on
entities that received an irregular write; detector recall on truly irregular facts and false-alarm
rate on regular ones (phases 5 onward); fraction of facts sent to the store.

Predictions, written before any run of this file:
  L1 mixed and oracle reproduce the earlier gap in inference (at least 0.07).
  L2 loo recovers at least half of that gap.
  L3 loo keeps old regular facts on irregular-carrying entities at 0.95 or more.
  L4 detector recall at least 0.7, false-alarm rate at most 0.15.
  If loo is within 0.02 of mixed on inference, the gain needs an oracle and write-time detection
  by this method fails.
"""
import json, sys
import numpy as np, torch
from cab_s import Z, R, V, K, Grad

def held_out_prob(L, groups):
    """groups: list of (held_in facts [(r, y)...], held-out (r, y)). Returns probability of each held-out value."""
    W = L.W.detach(); n = len(groups); m = W.shape[-1]
    u = torch.zeros(n, m, requires_grad=True); opt = torch.optim.Adam([u], lr=0.05)
    gi = torch.tensor([i for i, (hin, _) in enumerate(groups) for _ in hin]); rr = torch.tensor([r for hin, _ in groups for r, _ in hin]); yy = torch.tensor([y for hin, _ in groups for _, y in hin])
    for _ in range(60):
        loss = torch.nn.functional.cross_entropy(torch.einsum("bvm,bm->bv", W[rr], u[gi]), yy); opt.zero_grad(); loss.backward(); opt.step()
    ro = torch.tensor([g[1][0] for g in groups]); yo = torch.tensor([g[1][1] for g in groups])
    with torch.no_grad(): p = torch.softmax(torch.einsum("bvm,bm->bv", W[ro], u), -1)
    return p.gather(1, yo[:, None]).squeeze(1).numpy(), (p.argmax(-1) == yo).numpy()

def run(arm, seed, eps=0.1, phases=40, new=64, fix=16, m=24):
    rng = np.random.default_rng(seed); n = phases * new
    rule = rng.integers(V, size=(Z, R)); typ = rng.integers(Z, size=n)
    truth = np.where(rng.random((n, R)) < eps, rng.integers(V, size=(n, R)), rule[typ])
    shown = np.zeros((n, R), bool)
    for e in range(n): shown[e, rng.choice(R, K, replace=False)] = True
    corrected = np.zeros((n, R), bool); L = Grad(n, m, seed=seed, lr=0.003, steps=200)
    stored = total = tp = fn = fp = tn = 0
    for ph in range(phases):
        ents = np.arange(ph * new, (ph + 1) * new)
        facts = [(int(e), int(r), int(truth[e, r])) for e in ents for r in np.flatnonzero(shown[e])]; corr = []
        if ph:
            oe, orr = np.nonzero(shown[:ph * new]); pick = rng.choice(len(oe), fix, replace=False); fe, fr = oe[pick], orr[pick]
            prev = {(int(a), int(b)): int(truth[a, b]) for a, b in zip(fe, fr)}
            truth[fe, fr] = rng.integers(V, size=fix); corrected[fe, fr] = True
            corr = [(int(a), int(b), int(truth[a, b])) for a, b in zip(fe, fr)]
        allf = facts + corr; total += len(allf); irregular = lambda f: f[2] != rule[typ[f[0]], f[1]]
        if arm == "oracle": keep = [f for f in allf if not irregular(f)]
        elif arm == "loo" and ph >= 5:
            drop = set(); groups, ref = [], []
            for e in ents:
                fs = [f for f in facts if f[0] == e]
                for i, f in enumerate(fs): groups.append(([(g[1], g[2]) for j, g in enumerate(fs) if j != i], (f[1], f[2]))); ref.append(f)
            p, ok = held_out_prob(L, groups)
            for b in range(0, len(ref), K):
                if not ok[b:b + K].all(): drop.add(ref[b + int(np.argmin(p[b:b + K]))])
            groups, ref = [], []
            for f in corr:
                others = [(int(r2), int(prev.get((f[0], int(r2)), truth[f[0], r2]))) for r2 in np.flatnonzero(shown[f[0]]) if r2 != f[1]]
                groups.append((others, (f[1], f[2]))); ref.append(f)
            if groups:
                p, ok = held_out_prob(L, groups)
                for f, o in zip(ref, ok):
                    if not o: drop.add(f)
            for f in allf:
                irr, d = irregular(f), f in drop
                tp += irr and d; fn += irr and not d; fp += (not irr) and d; tn += (not irr) and not d
            keep = [f for f in allf if f not in drop]
        else: keep = allf
        stored += len(allf) - len(keep); L.learn(keep, ph)
    e, r = np.nonzero(shown); ok = L.predict(e, r, False) == truth[e, r]
    isrule = (truth[e, r] == rule[typ[e], r]) & ~corrected[e, r]
    irr = ((shown & ((truth != rule[typ]) | corrected)).sum(1) > 0)[e]; old = (phases - 1) - np.repeat(np.arange(phases), new)[e] >= 20
    late = np.arange((phases - 10) * new, n); ue, ur = np.nonzero(~shown[late])
    out = {"arm": arm, "seed": seed, "infer_late": round(float((L.predict(late[ue], ur, False) == truth[late[ue], ur]).mean()), 4),
           "rule_old_irregular": round(float(ok[isrule & old & irr].mean()), 4), "rule_old_clean": round(float(ok[isrule & old & ~irr].mean()), 4), "stored_fraction": round(stored / total, 4)}
    if arm == "loo": out.update(recall=round(tp / max(1, tp + fn), 3), false_alarm=round(fp / max(1, fp + tn), 3))
    return out

if __name__ == "__main__":
    arm = sys.argv[1]; rows = [run(arm, s) for s in (7001, 7002, 7003)]
    for x in rows: print(json.dumps(x), flush=True)
    json.dump(rows, open(f"results/routing_loo/{arm}.json", "w"))
