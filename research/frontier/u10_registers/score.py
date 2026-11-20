"""Summarize all six recorded runs and apply the advance rule with the Python standard library."""
import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = {"m21": (4, 460), "m22": (4, 460), "m22_wide": (11, 357)}
DOMAINS = ("web", "textbook", "math", "code")


def score(directory):
    manifest = json.loads((ROOT / "corpus.json").read_text())["manifest"]
    means = {}
    cells = []
    offsets = {}
    parameters = {}
    paired = {}
    advancement = {}
    for seed in (0, 1):
        for arm, config in ARMS.items():
            r = json.loads((directory / f"seed-{seed}-{arm}.json").read_text())
            if (r["seed"], r["arm"], r["registers"], r["ffn"]) != (seed, arm, *config):
                raise ValueError(f"Unexpected arm configuration: {seed}/{arm}")
            if (r["steps"], r["batch"], r["length"], r["train_tokens"]) != (1000, 8, 128, 1024000):
                raise ValueError("Unexpected training budget")
            if manifest["train_sha256"] != r["train_file_sha256"]:
                raise ValueError("Unexpected input corpus")
            if offsets.setdefault(seed, r["train_offsets_sha256"]) != r["train_offsets_sha256"]:
                raise ValueError(f"Unpaired training offsets in seed {seed}")
            if parameters.setdefault(arm, r["parameters"]) != r["parameters"]:
                raise ValueError(f"Parameter count changed for {arm}")
            bpb = r["validation_bpb"]
            if set(bpb) != {"L_128", "L_256", "L_512"} or any(set(row) != set(DOMAINS) for row in bpb.values()):
                raise ValueError("Incomplete domain or length coverage")
            if any(not math.isfinite(v) or v <= 0 for row in bpb.values() for v in row.values()):
                raise ValueError("Invalid BPB")
            mean = statistics.mean(bpb["L_512"].values())
            if not math.isclose(mean, r["mean_512_bpb"], rel_tol=0, abs_tol=1e-10):
                raise ValueError("Macro-mean BPB mismatch")
            means[arm, seed] = mean
            d3 = r["rank"]["val_code_256"][-1]
            if d3["layer"] != 3:
                raise ValueError("Missing fourth-layer code probe")
            cells.append({"seed": seed, "arm": arm, "macro_512_bpb": round(mean, 8), "domain_512_bpb": bpb["L_512"],
                          "code_l3_centered_rank": d3["mix_rank_cent"], "copy256_accuracy": r["copy"]["copy_256"]["acc"],
                          "tokens_per_second": r["training_tokens_per_second"]})
        wide = cells[-1]
        paired[seed] = {arm: round(means["m22_wide", seed] - means[arm, seed], 8) for arm in ("m21", "m22")}
        advancement[seed] = {"beats_both_controls_on_bpb": all(means["m22_wide", seed] < means[a, seed] for a in ("m21", "m22")),
                             "meets_rank_target": wide["code_l3_centered_rank"] >= 12,
                             "meets_copy_target": wide["copy256_accuracy"] >= 0.95}
    if offsets[0] == offsets[1]:
        raise ValueError("Expected two distinct seed streams")
    passed = all(all(criteria.values()) for criteria in advancement.values())
    return {"decision": "advance to another exploratory study" if passed else "hold",
            "total_training_tokens": 6 * 1024000, "parameters_by_arm": parameters, "cells": cells,
            "two_seed_mean_512_bpb": {arm: round(statistics.mean(means[arm, s] for s in (0, 1)), 8) for arm in ARMS},
            "two_seed_mean_512_bpb_by_domain": {arm: {d: round(statistics.mean(
                row["domain_512_bpb"][d] for row in cells if row["arm"] == arm), 8) for d in DOMAINS} for arm in ARMS},
            "wide_minus_control_macro_bpb": paired, "wide_passes_by_seed": advancement}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = json.dumps(score(args.results), indent=2, allow_nan=False) + "\n"
    if args.check:
        if output != (args.results / "summary.json").read_text():
            raise SystemExit("Scores differ from recorded summary")
        print("Six paired runs and their recorded decision verified.")
    else:
        print(output, end="")
