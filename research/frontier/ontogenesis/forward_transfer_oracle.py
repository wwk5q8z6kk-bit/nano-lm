"""Instrument check for forward_transfer.py, written AFTER the registered results were read (post hoc).

Every arm scored at chance on new combinations. That is only informative if the probe can detect
transfer when it exists. Here the menu's tables are set by hand to the true stream primitives
(step-one table m = primitive m, step-two likewise; two spare tables left random), nothing is trained,
and the registered probe is applied unchanged. If this does not score near 1.0 on novel_comp, the probe
is at fault and the registered result says nothing about the arms.
"""
import json, sys
import numpy as np, torch
import forward_transfer as ft

rows = []
for seed in (9101, 9102, 9103):
    rng = np.random.default_rng(seed); prim = np.stack([rng.permutation(ft.P) for _ in range(ft.N_PRIM + ft.N_NEW)])
    stream, comp, new, target, support = ft.world(seed); torch.manual_seed(seed); m = ft.Menu()
    with torch.no_grad():
        for i in range(ft.N_PRIM):
            t = torch.full((ft.P, ft.P), -8.0); t[torch.arange(ft.P), torch.tensor(prim[i])] = 8.0
            m.first[i] = t; m.second[i] = t
    row = {"seed": seed}
    for kind, fams in (("novel_comp", comp), ("novel_prim", new)): row.update(ft.probe(m, fams, kind, target, support, ("embed",)))
    rows.append(row); print(json.dumps(row), flush=True)
json.dump(rows, open("results/forward_transfer/oracle_menu.json", "w"))
