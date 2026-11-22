"""Two read paths without a router: does a shared plus a private read keep generalisation and acquisition?

Protocol: PROTOCOL-MEMRULE-DUAL.md (committed before any confirmatory run).
Follows RESULTS-MEMRULE-SHARING.md: in cab_s.py's rule world a fixed-address memory with typed addresses
generalises but acquires and corrects poorly, and spreading its reads trades one for the other.
Here every query reads two sets of 8 slots and the logits are summed:
  shared   the typed projection addresses (memory_rule_sharing.py, M = 0)
  private  8 uniformly random slots per (entity, relation) (memory_rule_sharing.py, M = 8)
Both are trained on every written fact; there is no detector and no routing decision.
Unwritten private slots stay at zero, so an unseen (entity, relation) is answered by the shared path alone.

  dual_pool      both paths index one 16,384-slot value table (same storage as the single-path arms)
  dual_separate  each path has its own 16,384-slot table (twice the storage)

  shared_only    control: the typed path alone (memory_rule_sharing.py typed M = 0), so a learning-rate
                 change can be separated from the effect of the second path

usage: memory_rule_dual.py ARM SEED OUT.json [LR]     ARM in {dual_pool, dual_separate, shared_only}; LR defaults to 1e-3
"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from cab_s import world, V
from memory_rule_sharing import addresses, SLOTS, PHASES, NEW, FIX, STEPS, LR

torch.set_num_threads(2)


def run(arm, seed, lr=LR):
    rng, truth, shown = world(seed, PHASES, NEW, FIX)
    n = PHASES * NEW
    i1, w1, _, eff1 = addresses("typed", 0, seed, n)
    i2, w2, _, eff2 = addresses("typed", 8, seed, n)
    T1 = torch.nn.Parameter(torch.zeros(SLOTS, V))
    T2 = T1 if arm in ("dual_pool", "shared_only") else torch.nn.Parameter(torch.zeros(SLOTS, V))
    params = [T1] if arm in ("dual_pool", "shared_only") else [T1, T2]
    R = truth.shape[1]
    key = lambda e, r: torch.as_tensor(np.asarray(e) * R + np.asarray(r))
    if arm == "shared_only":   # control: the typed path alone, i.e. memory_rule_sharing.py typed M = 0, at this lr
        read = lambda q: (w1[q][..., None] * T1[i1[q]]).sum(1)
    else:
        read = lambda q: (w1[q][..., None] * T1[i1[q]]).sum(1) + (w2[q][..., None] * T2[i2[q]]).sum(1)
    pred = lambda e, r: read(key(e, r)).argmax(-1).numpy()
    acc = lambda e, r: float((pred(e, r) == truth[e, r]).mean()) if len(e) else float("nan")
    log = []
    for ph in range(PHASES):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        se, sr = np.nonzero(shown[ents]); se = ents[se]
        fe = fr = np.zeros(0, int)
        if ph:
            oe, orr = np.nonzero(shown[:ph * NEW]); pick = rng.choice(len(oe), FIX, replace=False)
            fe, fr = oe[pick], orr[pick]; truth[fe, fr] = rng.integers(V, size=FIX)
        qe, qr = np.concatenate([se, fe]), np.concatenate([sr, fr])
        q, y = key(qe, qr), torch.from_numpy(truth[qe, qr])
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)   # reset each phase, as in every earlier arm
        for _ in range(STEPS):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            row = {"phase": ph, "acquire": acc(se, sr)}
            ne, nr = np.nonzero(~shown[ents]); row["infer_new"] = acc(ents[ne], nr)
            if ph:
                oe, orr = np.nonzero(shown[:ph * NEW]); row["retain"] = acc(oe, orr); row["correct"] = acc(fe, fr)
                ue, ur = np.nonzero(~shown[:ph * NEW]); row["infer_old"] = acc(ue, ur)
        log.append(row)
    late = log[-(PHASES // 4):]; mean = lambda kk: round(float(np.nanmean([x.get(kk, np.nan) for x in late])), 4)
    return {"arm": arm, "seed": seed, "lr": lr, "effective_slots_shared": eff1, "effective_slots_private": eff2,
            "value_table_floats": sum(p.numel() for p in params),
            "late": {k: mean(k) for k in ("acquire", "infer_new", "infer_old", "retain", "correct")}, "log": log}


if __name__ == "__main__":
    arm, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else LR
    f = open(out, "x"); t = time.time(); res = run(arm, seed, lr)
    res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__
    json.dump(res, f); f.close()
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
