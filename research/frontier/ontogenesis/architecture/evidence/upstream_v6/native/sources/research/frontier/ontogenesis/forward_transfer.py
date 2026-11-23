"""Does experience make the next thing cheaper to learn? A development pilot for forward transfer.

Background. Every continual-learning run in this directory so far measures retention or the speed of
learning more of the same kind of fact (plasticity.py, cab_s.py, correction_*.py). None measures whether
what was learned earlier reduces the examples needed for a NEW family of tasks. That is the claim
"learning improves future learning", and it is a different quantity from retention or plasticity.

World. Inputs are the integers mod P = 23. There are 6 hidden primitive functions, each a random
permutation of the inputs, and 2 further primitives that the training stream never uses. A task family
is an ordered pair (a, b): the target is f_b(f_a(x)). A learner is never told a or b. Each family gets
its own learned 16-number task vector, initialised at zero, and nothing else identifies it.

Stream. 24 of the 36 ordered pairs of the 6 stream primitives, in random order, each trained on all 23
inputs for 300 steps. Examples are never kept in a context: whatever is retained is in the weights
(and, for the replay arms, in a stored copy of earlier families' examples, which is counted).

Probe. At 0, 2, 4, 8, 16 and 24 families of experience the learner is copied and the copy is tested on
families it has not trained on:
  novel_comp  8 held-out ordered pairs of the 6 stream primitives (new combinations of known parts)
  novel_prim  4 pairs that use at least one of the 2 never-used primitives (nothing to reuse)
For each probe family the copy sees k = 2, 4, 8 or 16 examples, adapts for 200 steps, and is scored on
the 23 - k inputs it was not shown. Two ways to adapt: `embed` changes only the new task vector;
`full` changes every weight. Guessing scores 1/23 = 0.043.

Arms.
  fresh         a new dense network per probe family, trained on the k examples only (no experience)
  dense         multilayer network on [input, task vector]; stream trained one family at a time
  dense_replay  the same, each phase trained on all families seen so far
  loop          one shared block applied twice; each application reads half of the task vector
  loop_replay   the same with replay
  menu          8 learned lookup tables; the task vector is two sets of mixing weights over them
  menu_replay   the same with replay
`menu` is given the structure of the problem by its designer. It is the ceiling for "select and compose
from a fixed menu", not evidence that structure was discovered. `loop` is given only "two steps".

Primary measure: accuracy on unshown inputs of novel_comp families, k = 8, `embed` adaptation, after
24 families. Secondary: the same against experience (0..24), against k, for novel_prim, and for `full`.

This is a development pilot: three seeds (9101-9103), one world, to learn the size and variance of the
effect so that a confirmatory rule can be set. It registers predictions but no pass/fail gate.

Predictions, written before any run of this file:
  T1 fresh cannot generalise to unshown inputs: at most 0.10 at every k.
  T2 dense without replay shows no forward transfer: primary measure at most 0.15.
  T3 dense_replay transfers weakly: primary measure between 0.15 and 0.50.
  T4 loop_replay transfers: primary measure at least 0.60, and at least 0.20 above dense_replay.
  T5 menu_replay is near the ceiling: at least 0.90 at k = 8 and at least 0.60 at k = 2.
     menu without replay keeps most of it: at least 0.70 at k = 8.
  T6 transfer grows with experience: for loop_replay and menu_replay the primary measure after 24
     families is at least 0.20 above the same measure after 4 families.
  T7 nothing transfers to never-used primitives: on novel_prim no arm exceeds 0.15 with `embed`
     adaptation at any k, and no arm with `full` adaptation beats fresh by more than 0.05.
  Reading. If T4 fails and only the menu arms transfer, forward transfer in this world needs structure
  that was handed over, and whether a learner can create it stays open. If dense_replay reaches 0.60,
  ordinary multi-task training already gives the effect and no new mechanism is called for.
  If T7 fails, the gain is not reuse of learned primitives and the probe is measuring something else.

Known limits, stated before the run. Every family is a two-step composition, which favours the loop
arm by construction. Replay arms see more examples per step at equal step counts. The arms differ in
parameter count (printed with the results). One synthetic world; nothing here is a language-model result.

Deviation 1, recorded before any registered run (registration commit bef6608). A mechanics check on
seed 1, which is not a registered seed, showed that the menu arm as registered could not fit its own
training stream: 29% of stream answers correct after 24 families with replay, and 33-36% when trained
on all 24 jointly, at every learning rate and step count tried. Dense and loop fitted 100%. An arm that
cannot learn the stream cannot serve as a ceiling. Cause: one shared set of 8 probability tables forces
the optimiser to discover that step-one and step-two tables coincide, and mixing probabilities blurs
the tables before the weights sharpen. Changed, using stream fit on seed 1 as the only criterion:
separate table sets for the two steps, mixing of logits instead of probabilities, stream task vectors
for the menu arms initialised at random (scale 0.5) and learning rate 0.02. Stream fit on seed 1 is
then 90%. No probe result was read for any arm during these checks, and no prediction is changed.
The menu arms' probe task vectors still start at zero.
"""
from __future__ import annotations
import copy, json, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

P, N_PRIM, N_NEW, D_TASK = 23, 6, 2, 16
STREAM, N_COMP, N_PRIMPROBE = 24, 8, 4
CHECKPOINTS, KS = (0, 2, 4, 8, 16, 24), (2, 4, 8, 16)
TRAIN_STEPS, ADAPT_STEPS = 300, 200
torch.set_num_threads(1)


def world(seed):
    rng = np.random.default_rng(seed)
    prim = np.stack([rng.permutation(P) for _ in range(N_PRIM + N_NEW)])
    pairs = [(a, b) for a in range(N_PRIM) for b in range(N_PRIM)]
    while True:  # every stream primitive must appear in both positions of the stream
        order = rng.permutation(len(pairs)); stream = [pairs[i] for i in order[:STREAM]]
        if {a for a, _ in stream} == set(range(N_PRIM)) == {b for _, b in stream}: break
    comp = [pairs[i] for i in order[STREAM:STREAM + N_COMP]]
    new = [(N_PRIM, 0), (1, N_PRIM + 1), (N_PRIM, N_PRIM + 1), (N_PRIM + 1, N_PRIM)]
    target = lambda fam: torch.tensor(prim[fam[1]][prim[fam[0]]])
    support = {(kind, i, k): torch.tensor(rng.permutation(P)[:k]) for kind, fams in (("novel_comp", comp), ("novel_prim", new))
               for i in range(len(fams)) for k in KS}
    return stream, comp, new, target, support


class Dense(nn.Module):
    def __init__(s, h=128):
        super().__init__(); s.net = nn.Sequential(nn.Linear(P + D_TASK, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, P))
    def forward(s, x, t): return s.net(torch.cat([F.one_hot(x, P).float(), t], -1))


class Loop(nn.Module):
    def __init__(s, w=64, h=128):
        super().__init__(); s.inp = nn.Linear(P, w); s.out = nn.Linear(w, P)
        s.block = nn.Sequential(nn.Linear(w + D_TASK // 2, h), nn.ReLU(), nn.Linear(h, w))
    def forward(s, x, t):
        z = s.inp(F.one_hot(x, P).float())
        for half in (t[:, :D_TASK // 2], t[:, D_TASK // 2:]): z = s.block(torch.cat([z, half], -1))
        return s.out(z)


class Menu(nn.Module):
    """Returns log-probabilities. A task mixes the logits of 8 step-one tables and of 8 step-two tables (see Deviation 1)."""
    def __init__(s, m=D_TASK // 2):
        super().__init__(); s.first = nn.Parameter(torch.randn(m, P, P) * 0.1); s.second = nn.Parameter(torch.randn(m, P, P) * 0.1)
    def forward(s, x, t):
        m = s.first.shape[0]; w1, w2 = t[:, :m], t[:, m:]
        p1 = torch.einsum("bm,mbv->bv", w1, s.first[:, x]).softmax(-1)      # distribution after step one
        step2 = torch.einsum("bm,muv->buv", w2, s.second).softmax(-1)       # this task's step-two table
        return (torch.einsum("bu,buv->bv", p1, step2) + 1e-9).log()


MAKE = {"dense": Dense, "loop": Loop, "menu": Menu}
XS = torch.arange(P)


def fit(model, vecs, data, steps, params, lr):
    """data: list of (task vector index, inputs, targets). One full batch per step."""
    ti = torch.cat([torch.full((len(x),), i) for i, x, _ in data]); x = torch.cat([d[1] for d in data]); y = torch.cat([d[2] for d in data])
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy(model(x, vecs[ti]), y); opt.zero_grad(); loss.backward(); opt.step()


def probe(model, fams, kind, target, support, kinds_of_adapt):
    out = {}
    for how in kinds_of_adapt:
        for k in KS:
            accs = []
            for i, fam in enumerate(fams):
                m = copy.deepcopy(model); vec = nn.Parameter(torch.zeros(1, D_TASK)); y = target(fam); sup = support[(kind, i, k)]
                params, lr = ([vec], 0.05) if how == "embed" else ([vec] + list(m.parameters()), 0.001)
                fit(m, vec, [(0, sup, y[sup])], ADAPT_STEPS, params, lr)
                query = torch.tensor([j for j in range(P) if j not in set(sup.tolist())])
                with torch.no_grad(): accs.append(float((m(query, vec.expand(len(query), -1)).argmax(-1) == y[query]).float().mean()))
            out[f"{kind}/{how}/k{k}"] = round(float(np.mean(accs)), 4)
    return out


def run(arm, seed):
    stream, comp, new, target, support = world(seed); torch.manual_seed(seed); t0 = time.time()
    if arm == "fresh":
        row = {}
        for kind, fams in (("novel_comp", comp), ("novel_prim", new)): row.update(probe(Dense(), fams, kind, target, support, ("full",)))
        return {"arm": arm, "seed": seed, "params": sum(p.numel() for p in Dense().parameters()), "by_experience": {"0": row}, "seconds": round(time.time() - t0, 1)}
    kind_of_model, replay = arm.split("_")[0], arm.endswith("_replay")
    model = MAKE[kind_of_model](); by = {}
    vecs = nn.Parameter(torch.randn(STREAM, D_TASK) * 0.5 if kind_of_model == "menu" else torch.zeros(STREAM, D_TASK))
    for n in range(STREAM + 1):
        if n in CHECKPOINTS:
            row = {}
            for kind, fams in (("novel_comp", comp), ("novel_prim", new)): row.update(probe(model, fams, kind, target, support, ("embed", "full")))
            if n:
                with torch.no_grad(): row["stream_retained"] = round(float(np.mean([float((model(XS, vecs[i].expand(P, -1)).argmax(-1) == target(stream[i])).float().mean()) for i in range(n)])), 4)
            by[str(n)] = row; print(json.dumps({"arm": arm, "seed": seed, "families": n, **row}), flush=True)
        if n == STREAM: break
        seen = range(n + 1) if replay else [n]
        fit(model, vecs, [(i, XS, target(stream[i])) for i in seen], TRAIN_STEPS, [vecs] + list(model.parameters()), 0.001 if kind_of_model != "menu" else 0.02)
    return {"arm": arm, "seed": seed, "params": sum(p.numel() for p in model.parameters()), "stored_examples": STREAM * P if replay else 0,
            "by_experience": by, "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]] or [9101, 9102, 9103]
    rows = [run(arm, s) for s in seeds]
    dest = f"results/forward_transfer/{arm}.json" if seeds == [9101, 9102, 9103] else f"/dev/stdout"
    json.dump(rows, open(dest, "w"))
