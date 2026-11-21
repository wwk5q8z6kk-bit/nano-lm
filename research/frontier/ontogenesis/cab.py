"""Continuous acquisition benchmark (CAB) v0.

A world of entities and relations. Facts (subject, relation) -> object arrive in phases.
In each phase the learner sees only that phase's new facts and corrections, for a fixed
number of gradient steps. Nothing from earlier phases is shown again unless the learner
keeps it itself (a replay buffer counts as memory and is reported).

After every phase, with no further training, four things are measured separately:
  acquire   accuracy on this phase's facts, and steps needed to reach 95%
  retain    accuracy on every earlier fact (current values)
  correct   on facts whose value was changed: accuracy on the new value, and the rate of
            answering the stale value
  compose   two-hop questions whose first fact is old and second is new (and the reverse),
            and three-hop questions; no such chain is ever trained

Two ways to answer a chain are compared for every learner:
  direct    a feed-forward read that takes (subject, r1, r2) and must compose in one pass;
            it is trained on chains made only of the current phase's facts
  iterate   the one-hop function applied repeatedly, feeding its predicted entity back in;
            it is never trained on chains

Predictions written before the first full run (seeds 3001-3003; seed 3000 was a 6-phase smoke test):
  P1 for every learner, `iterate` answers old-with-new chains far better than `direct`,
     and `direct` stays near chance on chains it was not trained on.
  P2 the plain network's retention falls to a low level; a 10% replay buffer raises it without
     fixing it; the fixed-address memory retains most facts but is worst at corrections
     (it keeps answering the stale value) and slowest to acquire.
  P3 no learner ends with acquisition, retention, correction and composition all above 80%.
"""
from __future__ import annotations
import argparse, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

E, R = 1024, 4


class Net(nn.Module):
    """One-hop: (entity, relation) -> entity logits. Optional second relation for the direct chain read."""
    def __init__(self, kind="mlp", d=128, h=512, slots=16384, k=8):
        super().__init__()
        self.kind, self.k = kind, k
        self.ent, self.rel = nn.Embedding(E, d), nn.Embedding(R + 1, d)       # relation R = "no second hop"
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(3 * d, h), nn.LayerNorm(h), nn.ReLU(), nn.Linear(h, h), nn.LayerNorm(h), nn.ReLU())
            self.out = nn.Linear(h, E)
        else:                                                                 # sparse memory with fixed addresses
            self.q = nn.Linear(3 * d, d); self.keys = nn.Parameter(torch.randn(slots, d) / d ** 0.5)
            self.values = nn.Embedding(slots, E); nn.init.zeros_(self.values.weight)
            for p in (self.ent.weight, self.rel.weight, self.q.weight, self.q.bias, self.keys): p.requires_grad_(False)

    def read(self, e_vec, r1, r2):
        x = torch.cat([e_vec, self.rel(r1), self.rel(r2)], -1)
        if self.kind == "mlp": return self.out(self.net(x))
        s = F.normalize(self.q(x), dim=-1) @ F.normalize(self.keys, dim=-1).T
        top, idx = s.topk(self.k, -1)
        return (torch.softmax(top * 8.0, -1).unsqueeze(-1) * self.values(idx)).sum(1)

    def hop(self, s, r):                     # one hop from entity ids
        return self.read(self.ent(s), r, torch.full_like(r, R))

    def direct(self, s, r1, r2):             # two hops in one pass
        return self.read(self.ent(s), r1, r2)

    def iterate(self, s, rels):              # apply the one-hop function repeatedly
        vec = self.ent(s)
        for r in rels:
            logits = self.read(vec, r, torch.full_like(r, R)); vec = self.ent(logits.argmax(-1))   # feed the predicted entity back
        return logits


def chains(table, firsts, seconds, rng, n, hops=2):
    """Sample chains whose first fact is in `firsts` and whose later facts are in `seconds` (dicts (s,r)->o)."""
    by_s = {}
    for (s, r), o in seconds.items(): by_s.setdefault(s, []).append(r)
    out, keys = [], list(firsts)
    rng.shuffle(keys)
    for (s, r1) in keys:
        o, rels, ok = table[(s, r1)], [r1], True
        for _ in range(hops - 1):
            if o not in by_s: ok = False; break
            r = by_s[o][int(rng.integers(len(by_s[o])))]; rels.append(r); o = table[(o, r)]
        if ok: out.append((s, *rels, o))
        if len(out) >= n: break
    return out


def acc(logits, y): return float((logits.argmax(-1) == y).float().mean()) if len(y) else float("nan")


def run(kind, replay, seed, phases, new, fix, steps, lr):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = Net(kind); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01 if kind == "mlp" else 0.0)
    table, birth, stale, buffer, log = {}, {}, {}, [], []
    T = lambda rows, j: torch.tensor([r[j] for r in rows], dtype=torch.long)
    for ph in range(phases):
        fresh = {}
        while len(fresh) < new:
            key = (int(rng.integers(E)), int(rng.integers(R)))
            if key not in table and key not in fresh: fresh[key] = int(rng.integers(E))
        fixed = {}
        if table:
            olds = list(table); rng.shuffle(olds)
            for key in olds[:fix]:
                v = int(rng.integers(E))
                if v != table[key]: stale[key] = table[key]; fixed[key] = v
        old_table = dict(table); table.update(fresh); table.update(fixed)
        for key in fresh: birth[key] = ph
        rows = [(s, r, o) for (s, r), o in {**fresh, **fixed}.items()]
        two = chains(table, fresh, fresh, rng, 64)                                   # chains made only of this phase's facts
        s, r, o = T(rows, 0), T(rows, 1), T(rows, 2); need = steps
        for step in range(1, steps + 1):
            bs, br, bo = s, r, o
            if replay and buffer:
                pick = [buffer[i] for i in rng.integers(len(buffer), size=min(len(buffer), len(rows)))]
                pick = [(ps, pr, table[(ps, pr)]) for ps, pr in pick]                # the buffer stores keys; values stay current
                bs, br, bo = torch.cat([s, T(pick, 0)]), torch.cat([r, T(pick, 1)]), torch.cat([o, T(pick, 2)])
            loss = F.cross_entropy(model.hop(bs, br), bo)
            if two: loss = loss + F.cross_entropy(model.direct(T(two, 0), T(two, 1), T(two, 2)), T(two, 3))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if need == steps and step % 10 == 0 and acc(model.hop(s, r), o) >= 0.95: need = step
        if replay:                                                                    # keep 10% of everything seen
            buffer += [k for k in fresh if rng.random() < 0.10]
        with torch.no_grad():
            m = {"phase": ph, "steps_to_95": need, "acquire": acc(model.hop(s, r), o)}
            if old_table:
                keep = [(a, b, table[(a, b)]) for (a, b) in old_table]
                m["retain"] = acc(model.hop(T(keep, 0), T(keep, 1)), T(keep, 2))
                st = [(a, b, table[(a, b)], stale[(a, b)]) for (a, b) in stale]
                if st:
                    pred = model.hop(T(st, 0), T(st, 1)).argmax(-1)
                    m["correct_new"] = float((pred == T(st, 2)).float().mean()); m["correct_stale"] = float((pred == T(st, 3)).float().mean())
                for name, a, b in (("old_new", old_table, fresh), ("new_old", fresh, old_table)):
                    c = chains(table, a, b, rng, 128)
                    if c:
                        m[f"direct_{name}"] = acc(model.direct(T(c, 0), T(c, 1), T(c, 2)), T(c, 3))
                        m[f"iterate_{name}"] = acc(model.iterate(T(c, 0), [T(c, 1), T(c, 2)]), T(c, 3))
                c3 = chains(table, table, table, rng, 128, hops=3)
                if c3: m["iterate_3hop"] = acc(model.iterate(T(c3, 0), [T(c3, 1), T(c3, 2), T(c3, 3)]), T(c3, 4))
        log.append(m)
    last = log[-max(1, phases // 4):]; first = log[1:1 + max(1, phases // 4)]
    mean = lambda rows, k: float(np.nanmean([r.get(k, np.nan) for r in rows]))
    keys = ["steps_to_95", "acquire", "retain", "correct_new", "correct_stale", "direct_old_new", "iterate_old_new", "direct_new_old", "iterate_new_old", "iterate_3hop"]
    return {"learner": kind + ("+replay" if replay else ""), "seed": seed, "facts_total": len(table), "buffer": len(buffer),
            "params": sum(p.numel() for p in model.parameters()),
            "early": {k: round(mean(first, k), 3) for k in keys}, "late": {k: round(mean(last, k), 3) for k in keys}, "log": log}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", type=int, default=60); ap.add_argument("--new", type=int, default=128); ap.add_argument("--fix", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="3000"); ap.add_argument("--out", default="")
    a = ap.parse_args(); rows = []
    for kind, replay in (("mlp", False), ("mlp", True), ("memory", False)):
        for seed in [int(x) for x in a.seeds.split(",")]:
            t0 = time.time(); r = run(kind, replay, seed, a.phases, a.new, a.fix, a.steps, a.lr); r["seconds"] = round(time.time() - t0, 1)
            rows.append(r); print(json.dumps({k: v for k, v in r.items() if k != "log"}), flush=True)
    if a.out: json.dump(rows, open(a.out, "w"))
