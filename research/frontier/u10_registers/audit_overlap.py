"""Check the sampled training and BPB windows for shared, contiguous token sequences."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run import BATCH, LENGTH, STEPS, verify_corpus


def run(data, manifest):
    sample_lengths = (32, 128)
    lookup = {n: defaultdict(set) for n in sample_lengths}
    counts = {d: 0 for d in manifest["domains"]}
    for domain in counts:
        stream = np.memmap(data / f"val_{domain}.bin", dtype="<u2", mode="r")
        for length, windows in ((128, 24), (256, 24), (512, 16)):
            stride = (len(stream) - length - 1) // windows
            for j in range(windows):
                start = j * stride
                data_bytes = stream[start:start + length + 1].tobytes()
                counts[domain] += 1
                for n in sample_lengths:
                    for pos in range(length + 2 - n):
                        lookup[n][data_bytes[2 * pos:2 * (pos + n)]].add((domain, length, j))
    train = np.memmap(data / "train.bin", dtype="<u2", mode="r")
    results = {}
    for seed in (0, 1):
        rng = np.random.default_rng(70_000 + seed * 131)
        starts = rng.integers(0, len(train) - LENGTH - 1, size=(STEPS, BATCH), dtype=np.int64)
        by_n = {}
        for n in sample_lengths:
            matched_windows = set()
            matched_draws = 0
            for pos in starts.flat:
                sample = train[pos:pos + LENGTH + 1].tobytes()
                matched = False
                for index in range(LENGTH + 2 - n):
                    hit = lookup[n].get(sample[2 * index:2 * (index + n)])
                    if hit:
                        matched = True
                        matched_windows.update(hit)
                matched_draws += matched
            by_n[n] = {"train_draws_with_any_match": matched_draws,
                       "matched_val_windows": [dict(domain=d, length=L, index=j) for d, L, j in sorted(matched_windows)],
                       "val_windows_with_any_match": {d: sum(w[0] == d for w in matched_windows) for d in counts}}
        results[seed] = by_n
    return {"training_draws_per_seed": STEPS * BATCH, "validation_windows_by_domain": counts,
            "contiguous_token_matches_by_seed_and_length": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    data = ap.parse_args().data
    print(json.dumps(run(data, verify_corpus(data)), indent=2))
