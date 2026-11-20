"""Measure the effect of earlier answer exposure on native two-token associative recall."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from baselines import (
    ALIAS_OFFSET, BOS, K0, N_VALS, PRIMARY_KEYS, SEEN_VALS, SEP, V0,
    build_matched_arm, build_optimizer, make_batch,
)

ARMS = ("DA_dual_idcopy", "DA_gen", "IdCopy2Gram_Only")
SEEDS = (31271, 71339, 91823)
WORLD_SEED = 520241
N_RECORDS = 24
N_QUERIES = 16
N_WORLDS = 1024


def paired_worlds(count, seed=WORLD_SEED):
    """Hold the source, final query and its position fixed within each pair."""
    rng = np.random.default_rng(seed)
    unique, exposed, answers = [], [], []
    for _ in range(count):
        keys = rng.choice(PRIMARY_KEYS, N_RECORDS, replace=False) + K0
        values = rng.integers(SEEN_VALS, N_VALS, (N_RECORDS, 2)) + V0
        records = np.column_stack((keys, values))
        query_ids = rng.choice(N_RECORDS, N_QUERIES, replace=False)
        queries = records[query_ids].copy()
        queries[:, 0] += ALIAS_OFFSET
        target = queries[-1]
        # Omit both answer tokens from the final query.
        history = queries[:-1].copy()
        common = [BOS, *records.ravel(), SEP]
        unique.append([*common, *history.ravel(), target[0]])
        history[rng.integers(len(history))] = target
        exposed.append([*common, *history.ravel(), target[0]])
        answers.append(target[1:].tolist())
    return tuple(torch.tensor(rows, dtype=torch.long) for rows in (unique, exposed, answers))


@torch.inference_mode()
def predict(model, prefixes, device):
    """Generate both tokens greedily. The model receives its own first prediction."""
    outputs = []
    model.eval()
    for batch in prefixes.split(16):
        prefix = batch.to(device)
        tokens = []
        for _ in range(2):
            probs = model.forward_probs(prefix)[0][:, -1, :]
            token = probs.argmax(-1, keepdim=True)
            tokens.append(token)
            prefix = torch.cat([prefix, token], dim=1)
        outputs.append(torch.cat(tokens, dim=1).cpu())
    return torch.cat(outputs)


def train(arm, seed, steps, batch_size, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gate_rng = np.random.default_rng(seed + 1_000_000)
    model = build_matched_arm(arm).to(device)
    optimizer = build_optimizer(model, lr=0.05 if arm == "IdCopy2Gram_Only" else 0.002)
    warmup = max(40, steps // 10)
    def lr(step):
        warm = min(1.0, (step + 1) / warmup)
        return warm * 0.5 * (1.0 + math.cos(math.pi * min(1.0, step / steps)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr)
    for step in range(steps):
        limit = round(8 + 16 * min(1.0, step / (0.4 * steps)))
        num_records = int(rng.integers(2, limit + 1))
        inputs, targets = make_batch(rng, batch_size, num_records, 0, PRIMARY_KEYS, 0, SEEN_VALS,
                                     overwrite_prob=0.25, alias_prob=0.30)
        drop = gate_rng.random() < 0.5
        loss = model.loss(inputs.to(device), targets.to(device), force_copy=(arm == "DA_dual_idcopy" and drop))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return model


def score(arm, seed, steps, batch_size, worlds, device):
    model = train(arm, seed, steps, batch_size, device)
    unique, exposed, answers = worlds
    hits = []
    for prefixes in (unique, exposed):
        hits.append((predict(model, prefixes, device) == answers).all(dim=1))
    unique_hits, exposed_hits = hits
    return {
        "arm": arm, "seed": seed, "steps": steps, "batch_size": batch_size,
        "params": model.param_report(), "worlds": len(answers),
        "worlds_sha256": hashlib.sha256(b"".join(t.numpy().tobytes() for t in worlds)).hexdigest(),
        "unique_correct": int(unique_hits.sum()), "exposed_correct": int(exposed_hits.sum()),
        "both_correct": int((unique_hits & exposed_hits).sum()),
        "unique_only_correct": int((unique_hits & ~exposed_hits).sum()),
        "exposed_only_correct": int((~unique_hits & exposed_hits).sum()),
        "neither_correct": int((~unique_hits & ~exposed_hits).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--arm", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--worlds", type=int, default=N_WORLDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.worlds, args.steps, args.batch_size, args.threads) < 1:
        parser.error("worlds, steps, batch size and threads must be positive")
    torch.set_num_threads(args.threads)
    worlds = paired_worlds(args.worlds)
    with args.output.open("x", buffering=1) as stream:
        for seed in args.seed:
            for arm in args.arm:
                row = score(arm, seed, args.steps, args.batch_size, worlds, args.device)
                line = json.dumps(row, sort_keys=True)
                print(line, file=stream)
                print("ALIAS_STUDY_RESULT " + line, flush=True)


if __name__ == "__main__":
    main()
