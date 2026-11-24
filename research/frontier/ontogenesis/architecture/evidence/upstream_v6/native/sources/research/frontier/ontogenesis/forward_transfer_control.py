"""Positive control for the forward_transfer.py probe. Written AFTER the registered results were read (post hoc).

The registered probe gives a new family a zero task vector, adapts only that vector on k examples and
scores the unshown inputs. Every trained arm scored at chance on new combinations. That means "no
reusable parts" only if the same procedure succeeds where success is certainly possible. Here each
replay arm is retrained exactly as registered (same seeds), and the unchanged probe is applied to
  trained     8 families the arm was trained on (a suitable task vector certainly exists)
  novel_comp  the registered new combinations, re-run
with one addition: accuracy on the k SHOWN examples is logged too, to see whether adaptation fits at all.
For the menu arms a second start for the probe vector is also tried (random, scale 0.5), because a zero
vector gives a uniform step-two table and so no gradient to the step-one half.
Finally the menu's learned tables are compared with the true primitives.
"""
import copy, json, sys
import numpy as np, torch, torch.nn as nn
import forward_transfer as ft


def probe2(model, fams, target, sup_of, start):
    out = {}
    for k in ft.KS:
        q, s = [], []
        for i, fam in enumerate(fams):
            torch.manual_seed(1000 + i)
            vec = nn.Parameter(torch.zeros(1, ft.D_TASK) if start == "zero" else torch.randn(1, ft.D_TASK) * 0.5)
            m = copy.deepcopy(model); y = target(fam); sup = sup_of(i, k)
            ft.fit(m, vec, [(0, sup, y[sup])], ft.ADAPT_STEPS, [vec], 0.05)
            query = torch.tensor([j for j in range(ft.P) if j not in set(sup.tolist())])
            with torch.no_grad():
                q.append(float((m(query, vec.expand(len(query), -1)).argmax(-1) == y[query]).float().mean()))
                s.append(float((m(sup, vec.expand(len(sup), -1)).argmax(-1) == y[sup]).float().mean()))
        out[f"k{k}"] = {"unshown": round(float(np.mean(q)), 4), "shown": round(float(np.mean(s)), 4)}
    return out


rows = []
for arm in ("dense_replay", "loop_replay", "menu_replay"):
    kind = arm.split("_")[0]
    for seed in (9101, 9102, 9103):
        stream, comp, new, target, support = ft.world(seed); torch.manual_seed(seed)
        rng = np.random.default_rng(seed); prim = np.stack([rng.permutation(ft.P) for _ in range(ft.N_PRIM + ft.N_NEW)])
        model = ft.MAKE[kind]()
        vecs = nn.Parameter(torch.randn(ft.STREAM, ft.D_TASK) * 0.5 if kind == "menu" else torch.zeros(ft.STREAM, ft.D_TASK))
        for n in range(ft.STREAM):
            ft.fit(model, vecs, [(i, ft.XS, target(stream[i])) for i in range(n + 1)], ft.TRAIN_STEPS, [vecs] + list(model.parameters()), 0.001 if kind != "menu" else 0.02)
        with torch.no_grad(): fit = float(np.mean([float((model(ft.XS, vecs[i].expand(ft.P, -1)).argmax(-1) == target(stream[i])).float().mean()) for i in range(ft.STREAM)]))
        row = {"arm": arm, "seed": seed, "stream_fit": round(fit, 4)}
        for start in (("zero", "random") if kind == "menu" else ("zero",)):
            row[f"trained/{start}"] = probe2(model, stream[:8], target, lambda i, k: support[("novel_comp", i, k)], start)
            row[f"novel_comp/{start}"] = probe2(model, comp, target, lambda i, k: support[("novel_comp", i, k)], start)
        if kind == "menu":
            with torch.no_grad():
                for name, T in (("first", model.first), ("second", model.second)):
                    arg = T.argmax(-1).numpy()                                           # [tables, inputs]
                    row[f"tables_{name}_best_match_to_a_primitive"] = [round(float(max((arg[m] == prim[p]).mean() for p in range(ft.N_PRIM))), 3) for m in range(arg.shape[0])]
        rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open("results/forward_transfer/control.json", "w"), indent=1)
