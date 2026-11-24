"""Does writing a correction into a private slot avoid the later cost of writing it into shared slots?

Protocol: PROTOCOL-MEMRULE-PRIVATEFIX.md (committed before any confirmatory run).
Follows RESULTS-MEMRULE-WRITEPASS.md: a correction forced into the shared typed slots takes (0.92) but
later inference on new entities falls from 0.74 to 0.61, because those slots are shared with same-type
entities. Here the memory has two separate value tables, read and summed for every query:
  shared   typed projection addresses (as memory_rule_sharing.py typed M = 0)
  private  8 uniformly random slots per (entity, relation), own table (as memory_rule_dual.py dual_separate)
Each phase trains both tables on the phase's new shown facts (Adam reset, STEPS steps). Then the phase's
corrections are written alone in the loss, with a fresh Adam, updating:
  both     both tables            (the shared-slot write of the write-pass test, with a private path added)
  private  the private table only (the shared table is not touched by corrections)
for FIX_STEPS steps. Measurements as memory_rule_writepass.py.

usage: memory_rule_privatefix.py UPDATE SEED OUT.json FIX_STEPS     UPDATE in {both, private}
"""
import json, sys, time
import numpy as np, torch, torch.nn.functional as F
from cab_s import world, V
from memory_rule_sharing import addresses, SLOTS, PHASES, NEW, FIX, STEPS, LR

torch.set_num_threads(2)


def run(update, seed, fix_steps, lr=LR, steps=STEPS):
    rng, truth, shown = world(seed, PHASES, NEW, FIX)
    n = PHASES * NEW
    i1, w1, typ, _ = addresses("typed", 0, seed, n)
    i2, w2, _, _ = addresses("typed", 8, seed, n)
    T1 = torch.nn.Parameter(torch.zeros(SLOTS, V)); T2 = torch.nn.Parameter(torch.zeros(SLOTS, V))
    R = truth.shape[1]
    key = lambda e, r: torch.as_tensor(np.asarray(e) * R + np.asarray(r))
    read = lambda q: (w1[q][..., None] * T1[i1[q]]).sum(1) + (w2[q][..., None] * T2[i2[q]]).sum(1)
    acc = lambda e, r: float((read(key(e, r)).argmax(-1).numpy() == truth[e, r]).mean()) if len(e) else float("nan")

    def fit(qe, qr, params, n_steps):
        q, y = key(qe, qr), torch.from_numpy(truth[qe, qr])
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        for _ in range(n_steps):
            loss = F.cross_entropy(read(q), y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        for p in (T1, T2): p.grad = None

    log = []
    for ph in range(PHASES):
        ents = np.arange(ph * NEW, (ph + 1) * NEW)
        se, sr = np.nonzero(shown[ents]); se = ents[se]
        fe = fr = np.zeros(0, int)
        if ph:
            oe, orr = np.nonzero(shown[:ph * NEW]); pick = rng.choice(len(oe), FIX, replace=False)
            fe, fr = oe[pick], orr[pick]; truth[fe, fr] = rng.integers(V, size=FIX)
        fit(se, sr, [T1, T2], steps)
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
            fit(fe, fr, [T1, T2] if update == "both" else [T2], fix_steps)
            with torch.no_grad():
                row["correct_after"] = acc(fe, fr)
                row["own_after"] = acc(oe[own], orr[own]); row["peer_after"] = acc(oe[peer], orr[peer])
                row["infer_new_after"] = acc(ents[ne], nr)
                row["retain"] = acc(oe, orr)
                ue, ur = np.nonzero(~shown[:ph * NEW]); row["infer_old"] = acc(ue, ur)
        log.append(row)
    late = log[-(PHASES // 4):]; mean = lambda kk: round(float(np.nanmean([x.get(kk, np.nan) for x in late])), 4)
    keys = ("acquire", "infer_new_before", "infer_new_after", "infer_old", "retain", "correct_before", "correct_after",
            "own_before", "own_after", "peer_before", "peer_after")
    return {"update": update, "seed": seed, "fix_steps": fix_steps, "lr": lr, "steps": steps,
            "late": {k: mean(k) for k in keys}, "log": log}


if __name__ == "__main__":
    update, seed, out, fix_steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    f = open(out, "x"); t = time.time(); res = run(update, seed, fix_steps)
    res["seconds"] = round(time.time() - t, 1); res["torch"] = torch.__version__
    json.dump(res, f); f.close()
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
