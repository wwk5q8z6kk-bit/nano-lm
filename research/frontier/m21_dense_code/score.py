"""Apply the prespecified paired-repository analysis to compact public sums."""
import argparse
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (0, 1, 2)
ARMS = ("m21", "gqa")


def ratio(entry):
    return entry["bits"] / entry["targets"]


def validate_head(head, census):
    pooled = head["all"]["pooled"]
    if pooled["targets"] != census["targets"] or not math.isfinite(ratio(pooled)):
        raise ValueError("Invalid pooled denominator or loss")
    for groups in head.values():
        if any(v["targets"] <= 0 or not math.isfinite(v["bits"]) or v["bits"] < 0 for v in groups.values()):
            raise ValueError("Invalid groups")
        if sum(v["targets"] for v in groups.values()) != pooled["targets"]:
            raise ValueError("Grouped target counts do not match pooled count")
        if not math.isclose(sum(v["bits"] for v in groups.values()), pooled["bits"], abs_tol=1e-5, rel_tol=1e-10):
            raise ValueError("Grouped bits do not match pooled bits")
    if len(head["repo"]) != census["repositories"]:
        raise ValueError("Missing validation repository")


def summarize(directory):
    census = json.loads((directory / "validation-accounting.json").read_text())
    rows, heads, training = [], {}, {}
    for seed in SEEDS:
        offsets = set()
        for arm in ARMS:
            key = (seed, arm)
            fit = json.loads((directory / f"seed-{seed}-{arm}.json").read_text())
            metrics = json.loads((directory / f"eval-{seed}-{arm}.json").read_text())
            if fit["weights_sha256"] != metrics["weights_sha256"] or (metrics["seed"], metrics["arm"]) != key:
                raise ValueError("Weights or cell mismatch")
            if (fit["seed"], fit["arm"], fit["training_tokens"]) != (seed, arm, 4096000):
                raise ValueError("Unexpected training exposure")
            offsets.add(fit["sample_offsets_sha256"])
            for head in metrics["heads"].values():
                validate_head(head, census)
            heads[key] = metrics["heads"]
            training[key] = fit["training_tokens"] / fit["training_seconds"]
            h = heads[key]["primary"]
            rows.append({"seed": seed, "arm": arm, "pooled_bits_per_token": round(ratio(h["all"]["pooled"]), 7),
                         "repository_macro_bits_per_token": round(statistics.mean(map(ratio, h["repo"].values())), 7),
                         "train_tokens_per_second": round(training[key], 1),
                         "first_100_train_tokens_per_second": round(102400 / fit["trace"][1]["seconds"], 1),
                         "training_seconds": round(fit["training_seconds"], 2), "evaluation_seconds": round(metrics["seconds"], 2)})
        if len(offsets) != 1:
            raise ValueError("Training examples differ within a matched seed")
    counts = {part: {g: v["targets"] for g,v in groups.items()} for part,groups in heads[0,"m21"]["primary"].items()}
    for models in heads.values():
        for h in models.values():
            if any({g:v["targets"] for g,v in h[k].items()} != values for k,values in counts.items()):
                raise ValueError("Target populations differ across heads or seeds")
    gains = [ratio(heads[s,"gqa"]["primary"]["all"]["pooled"]) - ratio(heads[s,"m21"]["primary"]["all"]["pooled"]) for s in SEEDS]
    repos = sorted(counts["repo"], key=int)
    macro = np.array([statistics.mean(ratio(heads[s,"gqa"]["primary"]["repo"][r]) -
                                      ratio(heads[s,"m21"]["primary"]["repo"][r]) for s in SEEDS) for r in repos])
    indices = np.random.default_rng(22026).integers(len(repos), size=(10000, len(repos)))
    ci = np.quantile(macro[indices].mean(axis=1), [0.025, 0.975])
    stratified = {}
    for key in ("all", "row_license", "document_tokens", "pointer_target"):
        stratified[key] = {}
        for g,n in counts[key].items():
            stratified[key][g] = {"targets": n}
            for name, arm, output in (("m21", "m21", "primary"), ("gqa", "gqa", "primary"), ("m21_base", "m21", "base")):
                stratified[key][g][name] = round(statistics.mean(ratio(heads[s,arm][output][key][g]) for s in SEEDS), 7)
    advance = min(gains) > 0 and statistics.mean(gains) >= 0.02 and ci[0] > 0
    return {"decision": "investigate on another distribution" if advance else "hold",
            "total_train_predictions": 24576000, "validation": census,
            "cells": rows, "dense_minus_m21_pooled_bpt_by_seed": [round(x, 7) for x in gains],
            "mean_dense_minus_m21_pooled_bpt": round(statistics.mean(gains), 7),
            "mean_dense_minus_m21_repo_macro_bpt": round(float(macro.mean()), 7),
            "repository_bootstrap_95pct": [round(float(x), 7) for x in ci],
            "paired_repository_effect_sd": round(float(macro.std(ddof=1)), 7), "strata_bpt": stratified}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--check", action="store_true")
    args = p.parse_args()
    result = json.dumps(summarize(args.results), indent=2, allow_nan=False) + "\n"
    if args.check:
        if result != (args.results / "summary.json").read_text():
            raise ValueError("Offline score differs from committed summary")
        print("Six paired fits, score denominators, decision and repository interval agree.")
    else:
        print(result, end="")
