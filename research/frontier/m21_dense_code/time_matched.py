"""Train and score the dense arm at M21's measured training time (TIME_MATCHED_PROTOCOL.md)."""
import argparse
import fcntl
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from evaluate import evaluate
from study import (BATCH, LENGTH, SEEDS, STEPS, digest, make_model, metrics, sample_offsets, save,
                   synchronize, tensor_rows, train_step, verified_corpus)

OUT = ROOT / "results" / "time-matched"


def dense_step(model, optimizer, tokens, step, total):
    optimizer.param_groups[0]["lr"] = metrics._cosine_lr(step, total, total // 10, 0.0015)
    loss = model(tokens[:, :-1], tgt=tokens[:, 1:])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def calibrate(docs, stream, device):
    target = OUT / "calibration.json"
    if target.exists():
        raise FileExistsError("Calibration is final once written")
    offsets = sample_offsets(docs, 9001, steps=340)
    runs = []
    for arm in ("m21", "gqa", "m21", "gqa"):
        model = make_model(arm, 0).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01)
        for step, starts in enumerate(offsets):
            if step == 40:
                synchronize(device)
                start = time.perf_counter()
            train_step(model, arm, opt, tensor_rows(stream, starts, device), step)
        synchronize(device)
        seconds = time.perf_counter() - start
        runs.append({"arm": arm, "tokens_per_second": 300 * BATCH * LENGTH / seconds})
        print(runs[-1], flush=True)
        del model, opt
    rate = {a: statistics.mean(r["tokens_per_second"] for r in runs if r["arm"] == a) for a in ("m21", "gqa")}
    ratio = rate["gqa"] / rate["m21"]
    save(target, {"device": str(device), "runs": runs, "mean_tokens_per_second": rate, "ratio": ratio,
                  "dense_updates": int(round(STEPS * ratio / 100) * 100), "torch": str(torch.__version__)})


def fit(docs, stream, device, fits):
    total = json.loads((OUT / "calibration.json").read_text())["dense_updates"]
    for seed in SEEDS:
        weights, receipt = fits / f"seed-{seed}-gqa-time.pt", OUT / f"seed-{seed}-gqa-time.json"
        if weights.exists() or receipt.exists():
            raise FileExistsError(f"Existing output for seed {seed}")
        offsets = sample_offsets(docs, seed, steps=total)
        model = make_model("gqa", seed).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.01)
        synchronize(device)
        start = time.perf_counter()
        trace = []
        for step, starts in enumerate(offsets):
            loss = dense_step(model, opt, tensor_rows(stream, starts, device), step, total)
            if (step + 1) % 1000 == 0 or step + 1 == total:
                synchronize(device)
                seconds = time.perf_counter() - start
                value = float(loss.cpu())
                if not np.isfinite(value):
                    raise FloatingPointError("Nonfinite loss")
                trace.append({"update": step + 1, "loss": value, "seconds": round(seconds, 2)})
                print(f"seed-{seed} {step + 1}/{total} loss={value:.4f}", flush=True)
        torch.save({k: t.detach().cpu() for k, t in model.state_dict().items()}, weights)
        save(receipt, {"seed": seed, "updates": total, "training_tokens": total * BATCH * LENGTH,
                       "sample_offsets_sha256": hashlib.sha256(offsets.astype("<i8", copy=False).tobytes()).hexdigest(),
                       "weights_sha256": digest(weights), "source_sha256": digest(Path(__file__)),
                       "device": str(device), "training_seconds": seconds, "trace": trace})
        del model, opt


def score(docs, stream, device, fits):
    ratio = lambda e: e["bits"] / e["targets"]
    receipts = [json.loads((OUT / f"seed-{s}-gqa-time.json").read_text()) for s in SEEDS]  # all three must exist first
    cells, per_repo, gains = [], [], []
    for seed, receipt in zip(SEEDS, receipts):
        path = fits / f"seed-{seed}-gqa-time.pt"
        if digest(path) != receipt["weights_sha256"]:
            raise ValueError("Weights changed after training")
        model = make_model("gqa", seed)
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        head = evaluate(model.to(device), "gqa", docs, stream, device)["primary"]
        m21 = json.loads((ROOT / "results" / f"eval-{seed}-m21.json").read_text())["heads"]["primary"]
        m21_fit = json.loads((ROOT / "results" / f"seed-{seed}-m21.json").read_text())
        if head["all"]["pooled"]["targets"] != m21["all"]["pooled"]["targets"] or set(head["repo"]) != set(m21["repo"]):
            raise ValueError("Populations differ from the stored M21 evaluation")
        save(OUT / f"eval-{seed}-gqa-time.json", {"seed": seed, "weights_sha256": receipt["weights_sha256"], "head": head})
        gains.append(ratio(head["all"]["pooled"]) - ratio(m21["all"]["pooled"]))
        per_repo.append({r: ratio(head["repo"][r]) - ratio(m21["repo"][r]) for r in head["repo"]})
        cells.append({"seed": seed, "dense_time_bits_per_token": round(ratio(head["all"]["pooled"]), 7),
                      "m21_bits_per_token": round(ratio(m21["all"]["pooled"]), 7), "gain": round(gains[-1], 7),
                      "dense_training_seconds": round(receipt["training_seconds"], 1),
                      "m21_training_seconds": round(m21_fit["training_seconds"], 1),
                      "pointer_strata": {k: {"dense_time": round(ratio(head["pointer_target"][k]), 7),
                                             "m21": round(ratio(m21["pointer_target"][k]), 7)} for k in head["pointer_target"]}})
    repos = sorted(per_repo[0], key=int)
    macro = np.array([statistics.mean(p[r] for p in per_repo) for r in repos])
    idx = np.random.default_rng(22026).integers(len(repos), size=(10000, len(repos)))
    low, high = (float(x) for x in np.quantile(macro[idx].mean(axis=1), [0.025, 0.975]))
    if min(gains) > 0 and statistics.mean(gains) >= 0.02 and low > 0:
        decision = "M21 lead survives equal training time"
    elif high < 0:
        decision = "dense leads at equal training time"
    else:
        decision = "inconclusive"
    summary = {"decision": decision, "dense_updates": receipts[0]["updates"], "cells": cells,
               "mean_gain_pooled": round(statistics.mean(gains), 7), "mean_gain_repo_macro": round(float(macro.mean()), 7),
               "repository_bootstrap_95pct": [round(low, 7), round(high, 7)]}
    save(OUT / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("calibrate", "fit", "score"))
    for key in ("data", "tokenizer", "fits"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    args.fits.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    docs, streams = verified_corpus(args.data, args.tokenizer)
    with (args.fits / ".time-matched.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == "calibrate":
            calibrate(docs["train"], streams["train"], device)
        elif args.phase == "fit":
            fit(docs["train"], streams["train"], device, args.fits)
        else:
            score(docs["val"], streams["val"], device, args.fits)


if __name__ == "__main__":
    main()
