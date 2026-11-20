"""Check that trained weights predict a prefix identically when later tokens or batch neighbours change."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from study import ARMS, LENGTH, SEEDS, digest, make_model


@torch.inference_mode()
def audit(model, arm, tape):
    def row(start):
        return torch.from_numpy(tape[start:start + LENGTH].astype("int64"))

    def forward(batch):
        out = model(batch, return_base_logits=True) if arm == "m21" else (model(batch),)
        return [t.double() for t in out]

    totals = {}
    span = len(tape) - 4 * LENGTH
    for block in range(16):
        offset = (7331 * block + 137) % span
        first, second, other = row(offset), row((3 * offset + 19) % span), row((offset + 2 * LENGTH) % span)
        periodic = first[:8].repeat(LENGTH // 8)
        base = [t[0] for t in forward(torch.stack((first, second)))]
        half = LENGTH // 2
        cases = (("identical", first, second, LENGTH),
                 ("natural-future", torch.cat((first[:half], other[half:])), second, half),
                 ("repeated-future", torch.cat((first[:half], periodic[half:])), second, half),
                 ("natural-neighbor", first, other, LENGTH),
                 ("constant-neighbor", first, first[0].repeat(LENGTH), LENGTH),
                 ("periodic-neighbor", first, periodic, LENGTH),
                 ("alone", first, None, LENGTH))
        for label, a, b, n in cases:
            batch = a[None] if b is None else torch.stack((a, b))
            for head, before, after in zip(("primary", "base"), base, forward(batch)):
                cell = totals.setdefault(f"{head}/{label}", {"positions": 0, "logit_changes": 0, "greedy_changes": 0, "max_abs": 0.0})
                delta = (before[:n] - after[0, :n]).abs().amax(-1)
                cell["positions"] += n
                cell["logit_changes"] += int((delta > 1e-4).sum())
                cell["greedy_changes"] += int((before[:n].argmax(-1) != after[0, :n].argmax(-1)).sum())
                cell["max_abs"] = max(cell["max_abs"], float(delta.max()))
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--fits", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    tape = np.memmap(args.data / "val_code.bin", dtype="<u2", mode="r")
    cells = []
    for seed in SEEDS:
        for arm in ARMS:
            path = args.fits / f"seed-{seed}-{arm}.pt"
            model = make_model(arm, seed)
            model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
            cells.append({"seed": seed, "arm": arm, "weights_sha256": digest(path), "cases": audit(model.eval(), arm, tape)})
    worst = max(c["max_abs"] for cell in cells for c in cell["cases"].values())
    changed = sum(c["logit_changes"] for cell in cells for c in cell["cases"].values())
    print(json.dumps({"blocks": 16, "threshold": 1e-4, "maximum_absolute_logit_delta": worst,
                      "positions_over_threshold": changed, "cells": cells}, indent=2))


if __name__ == "__main__":
    main()
