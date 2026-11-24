"""Forward transfer when the training signal rewards reuse: the follow-up to forward_transfer.py (gate G07).

Predictions V1-V5 are at the end of this docstring, written after the mechanics check and before any
registered run.

Established by forward_transfer.py (FORWARD-TRANSFER-20261121.md): dense, loop and menu learners that
fit 24 two-step task families, one at a time or with replay, gain nothing on new combinations of the
same 6 primitives. A task vector adapted on a new combination cannot even fit the examples it is shown.
Nothing in that training asked the network to be adaptable; it was only asked to fit.

Question. If the network is trained so that a task vector adapted from zero on a few examples must
answer the rest of a family (few-shot episodes; context adaptation in the style of CAVIA, Zintgraf et
al. 2019), does it then transfer to new combinations of known primitives?

Unchanged from forward_transfer.py: the world, the 24 stream families, the 8 new-combination and 4
new-primitive probe families, the k-shot support sets, the three seeds, the three architectures and
their sizes. Changed: only how the network is trained.

Training. Every family of the stream is available from the start (no sequence; this is the multi-task
setting of the `_replay` arms, which also saw every family). One outer step: 8 stream families; for
each, k drawn from {2, 4, 8, 16}, a random support set of k inputs and the other 23 - k as queries.
Inner loop: the family's task vector starts at one learned start vector shared by all tasks and takes INNER_STEPS plain gradient steps of size
INNER_LR on the support loss, differentiably. Outer loss: cross-entropy on the queries under the
adapted vector; Adam (lr 0.001) on the network weights and the start vector. OUTER_STEPS outer steps.

Evaluation, after training:
  meta_probe        the training inner loop, unchanged, from the learned start vector (what the network was trained for)
  registered_probe  forward_transfer.probe, unchanged (Adam, 200 steps, lr 0.05), for comparison with the pilot
on: trained (8 stream families, positive control), novel_comp, novel_prim; at k = 2, 4, 8, 16;
reporting accuracy on the unshown inputs and on the shown ones.
Primary measure: novel_comp, k = 8, meta_probe, accuracy on unshown inputs.
Comparator: the pilot's `_replay` arms on the same seeds (results/forward_transfer/*_replay.json).

Mechanics check (before predictions): INNER_LR, INNER_STEPS and OUTER_STEPS are chosen on seed 1, which
is not a registered seed, by query accuracy on the stream's own families only. No probe of new
combinations or new primitives is run on seed 1 or read during the check.

Mechanics check, as done. With a zero start vector the menu cannot learn at all (chance at every inner
step size tried): its output is uniform when the task vector is zero, so the first inner step sends no
gradient to step one. Every arm was therefore given one learned start vector shared by all tasks (as in
MAML), which is the only change from the first draft. With inner step 2.0, 5 inner steps and 3000 outer
steps, query accuracy on the stream's own families at the end of training on seed 1 was dense 100%,
loop 100%, menu 79%. Inner steps 0.5 and 5.0 were also tried for dense and loop (zero start, 1500
outer steps) and were no better. The registered probe still starts from zero, as in the pilot.

Predictions, written before any registered run (seeds 9101-9103):
  V1 positive control: meta_probe on trained families, k = 8, unshown inputs: at least 0.85 for dense
     and loop, at least 0.60 for menu. If this fails for an arm, that arm's other results are unread.
  V2 dense: novel_comp primary measure at most 0.15. Few-shot training teaches it to recognise the
     families it has seen, not to recombine their parts.
  V3 loop: novel_comp primary measure at least 0.30. Its two steps each read their own half of the
     task vector, so recognising a family can be done by recognising its two steps.
  V4 menu: novel_comp primary measure at least 0.40.
  V5 nothing transfers to never-used primitives: novel_prim, meta_probe, unshown inputs, at most 0.15
     for every arm at every k.
  Reading. V3 or V4 with V5: a training signal that rewards adaptation, plus a factored architecture,
  gives forward transfer here, and the pilot's null was missing signal, not missing capacity. V2 failing
  (dense transfers too): the factored architecture is not needed. V1 holding with V2-V4 all failing:
  few-shot training produces recognition without reusable parts, and compositional transfer needs
  something not tried here (for example, combinations held out during training, or explicit
  factorisation pressure). V5 failing: the probe measures something other than reuse of parts.
"""
from __future__ import annotations
import json, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import forward_transfer as ft

INNER_LR, INNER_STEPS, OUTER_STEPS, BATCH = 2.0, 5, 3000, 8   # fixed by the mechanics check (see docstring)
torch.set_num_threads(1)


class WithStart(nn.Module):
    """The architecture plus one learned start vector shared by every task (as in MAML's learned initialisation)."""
    def __init__(s, net):
        super().__init__(); s.net = net; s.start = nn.Parameter(torch.zeros(1, ft.D_TASK))
    def forward(s, x, t): return s.net(x, t)


def adapt(model, x, y, inner_lr, inner_steps, create_graph):
    vec = model.start if create_graph else model.start.detach().clone().requires_grad_(True)
    for _ in range(inner_steps):
        loss = F.cross_entropy(model(x, vec.expand(len(x), -1)), y)
        (g,) = torch.autograd.grad(loss, vec, create_graph=create_graph)
        vec = vec - inner_lr * g
    return vec


def train(model, stream, target, rng, inner_lr, inner_steps, outer_steps, log_every=0):
    opt = torch.optim.Adam(model.parameters(), lr=0.001); curve = []
    for step in range(outer_steps):
        loss, hits, n = 0.0, 0, 0
        for fam_i in rng.choice(len(stream), BATCH, replace=False):
            y = target(stream[fam_i]); k = int(rng.choice(ft.KS)); perm = torch.tensor(rng.permutation(ft.P))
            sup, qry = perm[:k], perm[k:]
            vec = adapt(model, sup, y[sup], inner_lr, inner_steps, True)
            out = model(qry, vec.expand(len(qry), -1)); loss = loss + F.cross_entropy(out, y[qry])
            hits += int((out.argmax(-1) == y[qry]).sum()); n += len(qry)
        opt.zero_grad(); (loss / BATCH).backward(); opt.step()
        if log_every and (step + 1) % log_every == 0: curve.append((step + 1, round(hits / n, 4)))
    return curve


def meta_probe(model, fams, sup_of, target, inner_lr, inner_steps):
    out = {}
    for k in ft.KS:
        q, s = [], []
        for i, fam in enumerate(fams):
            y = target(fam); sup = sup_of(i, k); qry = torch.tensor([j for j in range(ft.P) if j not in set(sup.tolist())])
            vec = adapt(model, sup, y[sup], inner_lr, inner_steps, False).detach()
            with torch.no_grad():
                q.append(float((model(qry, vec.expand(len(qry), -1)).argmax(-1) == y[qry]).float().mean()))
                s.append(float((model(sup, vec.expand(len(sup), -1)).argmax(-1) == y[sup]).float().mean()))
        out[f"k{k}"] = {"unshown": round(float(np.mean(q)), 4), "shown": round(float(np.mean(s)), 4)}
    return out


def run(arm, seed, inner_lr, inner_steps, outer_steps):
    stream, comp, new, target, support = ft.world(seed); torch.manual_seed(seed); rng = np.random.default_rng(seed + 7)
    model = WithStart(ft.MAKE[arm.split("_")[0]]()); t0 = time.time()
    curve = train(model, stream, target, rng, inner_lr, inner_steps, outer_steps, log_every=max(1, outer_steps // 10))
    sup_trained = {(i, k): torch.tensor(np.random.default_rng(seed * 100 + i * 10 + k).permutation(ft.P)[:k]) for i in range(8) for k in ft.KS}
    row = {"arm": arm, "seed": seed, "params": sum(p.numel() for p in model.parameters()), "train_curve": curve, "train_seconds": round(time.time() - t0, 1)}
    for name, fams, sup_of in (("trained", stream[:8], lambda i, k: sup_trained[(i, k)]),
                               ("novel_comp", comp, lambda i, k: support[("novel_comp", i, k)]),
                               ("novel_prim", new, lambda i, k: support[("novel_prim", i, k)])):
        row[f"meta_probe/{name}"] = meta_probe(model, fams, sup_of, target, inner_lr, inner_steps)
    for name, fams in (("novel_comp", comp), ("novel_prim", new)):
        row[f"registered_probe/{name}"] = ft.probe(model, fams, name, target, support, ("embed",))
    row["seconds"] = round(time.time() - t0, 1)
    return row


if __name__ == "__main__":
    arm = sys.argv[1]; seeds = [9101, 9102, 9103]
    assert None not in (INNER_LR, INNER_STEPS, OUTER_STEPS), "settings are fixed by the mechanics check"
    rows = []
    for s in seeds:
        rows.append(run(arm, s, INNER_LR, INNER_STEPS, OUTER_STEPS)); print(json.dumps(rows[-1]), flush=True)
    json.dump(rows, open(f"results/forward_transfer_meta/{arm}.json", "w"), indent=1)
