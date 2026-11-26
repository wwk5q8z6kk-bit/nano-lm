"""Recompute the seed-zero macro BPB after excluding the window flagged by audit_overlap.py."""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from run import ARMS, digest, make_model, verify_corpus


@torch.inference_mode()
def sensitivity(data, weights, results):
    manifest = verify_corpus(data)
    overlaps = json.loads((results / "sampled-overlap.json").read_text())
    matches = overlaps["contiguous_token_matches_by_seed_and_length"]["0"]["32"]["matched_val_windows"]
    if len(matches) != 1 or matches[0]["length"] != 512:
        raise ValueError("Sensitivity calculation requires the single flagged primary window")
    domain, j = matches[0]["domain"], matches[0]["index"]
    window_count, length = 16, 512
    stream = np.memmap(data / f"val_{domain}.bin", dtype="<u2", mode="r")
    start = j * ((len(stream) - length - 1) // window_count)
    tokens = torch.from_numpy(stream[start:start + length + 1].astype(np.int64)).unsqueeze(0)
    rows = {}
    for arm in ARMS:
        stem = f"seed-0-{arm}"
        checkpoint = weights / f"{stem}.pt"
        receipt = json.loads((results / "cpu" / f"{stem}.json").read_text())
        if digest(checkpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint SHA-256 mismatch")
        model = make_model(arm, 0)
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        logp = model.eval()(tokens[:, :-1]).gather(-1, tokens[:, 1:, None])
        window_bpb = -logp.double().mean().item() / (math.log(2) * manifest["domains"][domain]["val"]["bytes_per_token"])
        domain_bp = {d: receipt["windows"][f"{d}/512"]["cpu_bpb"] for d in manifest["domains"]}
        domain_bp[domain] = (domain_bp[domain] * window_count - window_bpb) / (window_count - 1)
        rows[arm] = {"removed_window_bpb": round(window_bpb, 7), "remaining_domain_bpb": round(domain_bp[domain], 7),
                     "remaining_macro_bpb": round(statistics.mean(domain_bp.values()), 7)}
    return {"seed": 0, "omitted": matches[0], "arms": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(sensitivity(args.data, args.weights, args.results), indent=2))
