"""Recompute the validation BPB from a matching local checkpoint with a separate CPU loss calculation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from run import digest, make_model, save, verify_corpus


@torch.inference_mode()
def replay(receipt, weights, data, manifest):
    if digest(weights) != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch")
    model = make_model(receipt["arm"], receipt["seed"])
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"), strict=True)
    model.eval()
    rows = {}
    for length, windows in ((128, 24), (256, 24), (512, 16)):
        for domain, domain_manifest in manifest["domains"].items():
            stream = np.memmap(data / f"val_{domain}.bin", dtype="<u2", mode="r")
            stride = max(1, (len(stream) - length - 1) // windows)
            nll_sum = 0.0
            for j in range(windows):
                start = j * stride
                sequence = torch.from_numpy(stream[start:start + length + 1].astype(np.int64)).unsqueeze(0)
                predicted = model(sequence[:, :-1])
                chosen = predicted.gather(-1, sequence[:, 1:, None])
                nll_sum -= chosen.double().sum().item()
            value = nll_sum / (windows * length * math.log(2) * domain_manifest["val"]["bytes_per_token"])
            expected = receipt["validation_bpb"][f"L_{length}"][domain]
            if abs(value - expected) > 0.0001:
                raise ValueError(f"{domain}/{length}: CPU={value}, recorded={expected}")
            rows[f"{domain}/{length}"] = {"cpu_bpb": round(value, 7), "mps_bpb": expected, "windows": windows}
    return {"arm": receipt["arm"], "seed": receipt["seed"], "checkpoint_sha256": receipt["checkpoint_sha256"],
            "max_abs_delta": max(abs(r["cpu_bpb"] - r["mps_bpb"]) for r in rows.values()), "windows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--receipt", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    save(args.out, replay(json.loads(args.receipt.read_text()), args.weights, args.data, verify_corpus(args.data)))
