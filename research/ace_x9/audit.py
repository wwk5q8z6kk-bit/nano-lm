"""Replay the historical and corrected fixed-shape prefix perturbations on CPU."""

import argparse
import json

import numpy as np
import torch

from .audit_contract import DOMAINS
from .score import CHECKPOINTS, WEIGHTS, load_model, sha256
from pathlib import Path
from unittest.mock import patch


def archived_batch_gate(voc, eligible, shield):
    """The exact historical whole-batch/whole-sequence selection for this audit only."""
    budget = max(1, int(eligible.numel() * .30))
    if eligible.sum().item() > budget:
        threshold = torch.topk(voc.reshape(-1), k=budget).values[-1]
        return (voc >= threshold) & ~shield
    return eligible


@torch.inference_mode()
def comparisons(model, streams):
    domains = tuple(WEIGHTS)
    rows = []
    for i in range(16):
        domain, other = domains[i % 4], domains[(i + 1) % 4]
        offset = (733 * i + 137) % 999800

        def row(source, start):
            return torch.tensor(streams[source][start:start + 128].astype("int64"))

        first, second = row(domain, offset), row(other, 3 * offset + 19)
        natural = row(other, 1000 + offset)
        periodic = first[:8].repeat(16)
        base = model(torch.stack((first, second)))[0].clone()
        cases = (
            ("identical", first, second, 128),
            ("natural-future", torch.cat((first[:64], natural[64:])), second, 64),
            ("repeated-future", torch.cat((first[:64], periodic[64:])), second, 64),
            ("natural-neighbor", first, natural, 128),
            ("constant-neighbor", first, first[0].repeat(128), 128),
            ("periodic-neighbor", first, periodic, 128),
        )
        for label, changed_first, changed_second, n in cases:
            changed = model(torch.stack((changed_first, changed_second)))[0, :n]
            delta = (base[:n] - changed).abs().amax(-1)
            rows.append({"block": i, "case": label, "positions": n,
                         "logit_changes": int((delta > 1e-4).sum()),
                         "greedy_changes": int((base[:n].argmax(-1) != changed.argmax(-1)).sum())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--seed", choices=CHECKPOINTS, required=True)
    parser.add_argument("--archived-batch-gate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    streams = {}
    for domain, (digest, _) in DOMAINS.items():
        path = args.data_dir / f"val_{domain}.bin"
        if sha256(path) != digest:
            raise ValueError(f"Validation split SHA-256 mismatch for {domain}")
        streams[domain] = np.memmap(path, mode="r", dtype="<u2")
    model = load_model(args.checkpoint, CHECKPOINTS[args.seed])
    if args.archived_batch_gate:
        model.refiner.eval_policy = "stock-serving"
        with patch(__package__ + ".model.stock_prefix_end", archived_batch_gate):
            rows = comparisons(model, streams)
    else:
        rows = comparisons(model, streams)
    print(json.dumps({"rows": rows, "seed": args.seed, "stock_batch": args.archived_batch_gate}, indent=2))


if __name__ == "__main__":
    main()
