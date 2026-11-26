"""Score one restored Omega checkpoint on the four pinned validation token files."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import X9Config, X9LM


WEIGHTS = {"web": 0.4, "textbook": 0.2, "math": 0.2, "code": 0.2}
CHECKPOINTS = {
    "s0": "13287894bbb236776ef02f458d8491c9a336461354a09bdaea865f9e56bde81f",
    "s1": "9c481eecb0d7879e1c9c34fa7132c8844f0b33800e8216a940f7cfad1d720381",
    "x8": "0605687bdbda009b064d0a831ed493ed6a02fc3cd1252613d0436f9d90fa98e7",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def load_model(path, expected, policy="causal"):
    if sha256(path) != expected:
        raise ValueError("Checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = X9LM(X9Config(**checkpoint["cfg"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.refiner.eval_policy = policy
    return model.eval()


@torch.inference_mode()
def score(model, data_dir):
    from . import audit_contract as contract

    data_dir = Path(data_dir)
    rng = np.random.default_rng(84481)
    rows = []
    weighted = 0.0
    for domain, weight in WEIGHTS.items():
        path = data_dir / f"val_{domain}.bin"
        expected_hash, bytes_per_token = contract.DOMAINS[domain]
        if sha256(path) != expected_hash:
            raise ValueError(f"Validation split SHA-256 mismatch for {domain}")
        data = np.memmap(path, mode="r", dtype="<u2")
        offsets = sorted(rng.choice(np.arange(150000, 950000, 512), size=24, replace=False).tolist())
        losses = []
        for offset in offsets:
            sequence = torch.tensor(data[offset:offset + 129].astype("int64")).unsqueeze(0)
            logits = model(sequence[:, :-1])
            loss = F.cross_entropy(logits[0], sequence[0, 1:], reduction="mean").item()
            losses.append(loss)
            rows.append({"domain": domain, "offset": offset, "mean_nll": loss})
        weighted += weight * np.mean(losses) / (math.log(2) * bytes_per_token)
    return {"weighted_bits_per_byte": float(weighted), "windows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--seed", required=True, choices=CHECKPOINTS)
    parser.add_argument("--policy", choices=("causal", "stock-serving"), default="causal")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_model(args.checkpoint, CHECKPOINTS[args.seed], args.policy)
    print(json.dumps({"checkpoint_sha256": CHECKPOINTS[args.seed], "policy": args.policy,
                      "torch": str(torch.__version__), **score(model, args.data_dir)}, indent=2))


if __name__ == "__main__":
    main()
