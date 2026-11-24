"""Read-only reconciliation of the two distinct V5 optimizer experiments.

No neural training, Torch import, historical source edit, network access, or
automatic result overwrite. NumPy is used only to reconstruct exogenous events.
All frozen inputs must match their Git blob and SHA-256 identities first.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

CORRECTION_COMMIT = "0afd5629990b4e442fb3aa8f7c700b030d8faede"
FACTORIAL_COMMIT = "767857e72bfa51e3fa6f32d047507aabef077fde"
CORRECTION_SEEDS = (8001, 8002, 8003)
FACTORIAL_SEEDS = (3101, 3102, 3103, 3104, 3105)
ARMS = ("adam_200", "adam_stop", "gd_stop")
CELLS = (("original", "adam"), ("original", "adam_reset"),
         ("hash", "adam"), ("hash", "adam_reset"),
         ("original", "adam_b2_09"), ("original", "sgd"))
SOURCE = Path("research/frontier/ontogenesis")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stats(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(x) for x in values), "nonfinite or empty sample")
    return {"mean": mean(values), "min": min(values), "max": max(values), "values": values}


def verify_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text())
    require(manifest["provenance"]["cutoffs"]["correction"] == CORRECTION_COMMIT, "wrong correction cutoff")
    require(manifest["provenance"]["cutoffs"]["factorial"] == FACTORIAL_COMMIT, "wrong factorial cutoff")
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["local_path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe manifest path")
        require(str(relative) not in seen, "duplicate manifest path")
        seen.add(str(relative))
        data = (root / relative).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(blob == entry["git_blob"], f"Git blob mismatch: {relative}")
        require(hashlib.sha256(data).hexdigest() == entry["sha256"], f"SHA-256 mismatch: {relative}")
        require(len(data) == entry["bytes"], f"byte count mismatch: {relative}")
    required = {"correction/" + str(SOURCE / name)
                for name in ("cab_s.py", "correction_optimizer.py", "correction_interference.py")}
    required |= {"correction/" + str(SOURCE / "results/correction_interference" / f"optimizer-{arm}.json")
                 for arm in ARMS}
    required |= {"factorial/" + str(SOURCE / "results/memory_saturation_factorial" / f"{a}-{o}-s{s}.json")
                 for a, o in CELLS for s in FACTORIAL_SEEDS}
    required |= {"factorial/" + str(SOURCE / "results/memory_saturation_factorial" / f"dev-hash-sgd-lr{lr}-s3001.json")
                 for lr in (1, 3, 10, 30, 100, 300)}
    required.add("factorial/" + str(SOURCE / "results/memory_saturation_factorial/dev-sgd-selection.json"))
    require(required <= seen, "manifest is missing a required analysis input")
    return manifest


def load_correction(root):
    result = {}
    keys = ("write_steps", "corrected_now", "collateral_now", "rule_old_corrected", "corrected_old")
    for arm in ARMS:
        path = root / "results/correction_interference" / f"optimizer-{arm}.json"
        rows = json.loads(path.read_text())
        require(len(rows) == 3 and sorted(x["seed"] for x in rows) == list(CORRECTION_SEEDS),
                f"missing or duplicate correction seeds: {arm}")
        rows.sort(key=lambda x: x["seed"])
        require(all(x["arm"] == arm for x in rows), "correction arm identity mismatch")
        for row in rows:
            require(all(math.isfinite(row[key]) for key in keys), "nonfinite correction result")
            require(all(0 <= row[key] <= 1 for key in keys if key != "write_steps"), "accuracy outside [0,1]")
        result[arm] = {"per_seed": rows, "metrics": {k: stats(row[k] for row in rows) for k in keys},
                       "O1_collateral_le_085_seed_count": sum(row["collateral_now"] <= .85 for row in rows),
                       "rival_trigger_collateral_ge_095_seed_count": sum(row["collateral_now"] >= .95 for row in rows)}
    return result


def summarize_factorial_run(row, address, optimizer, seed, lr):
    require((row["address"], row["opt"], row["seed"], row["lr"]) == (address, optimizer, seed, lr),
            "factorial identity mismatch")
    log = row["log"]
    require([x["phase"] for x in log] == list(range(60)), "expected exactly phases 0 through 59")
    for record in log:
        require(math.isfinite(record["acquire"]) and 0 <= record["acquire"] <= 1, "invalid acquisition")
        require(math.isfinite(record["final_loss"]), "nonfinite loss")
        require(isinstance(record["slots_used"], int) and 0 <= record["slots_used"] <= 16384, "invalid slots used")
        if record["phase"]:
            require(math.isfinite(record["retain"]) and 0 <= record["retain"] <= 1, "invalid retention")
    computed = {"late_acquire": mean(x["acquire"] for x in log[45:60]),
                "late_retain": mean(x["retain"] for x in log[45:60]),
                "early_acquire": mean(x["acquire"] for x in log[1:16])}
    for key, value in computed.items():
        require(abs(round(value, 4) - row[key]) < 1e-12, f"summary disagrees with phase log: {key}")
    return {"seed": seed, **computed, "stored_rounded": {k: row[k] for k in computed},
            "final_slots_used": log[-1]["slots_used"], "torch": row.get("torch"),
            "threads": row.get("threads"), "seconds_uncontrolled": row.get("seconds")}


def choose_sgd_lr(sweep):
    require(len(sweep) == 6 and sorted(lr for lr, _ in sweep) == [1, 3, 10, 30, 100, 300],
            "incomplete or duplicate development sweep")
    require(all(math.isfinite(value) and 0 <= value <= 1 for _, value in sweep), "invalid dev score")
    best = max(value for _, value in sweep)
    chosen = min(lr for lr, value in sweep if value >= best - .01)
    return {"best": best, "chosen_lr": chosen, "informative": best >= .9}


def prediction_counts(data, sgd_informative):
    for values in data.values():
        require(set(values) == set(FACTORIAL_SEEDS), "prediction scoring requires all five seeds")
    oa, reset, ha, hr, b2, gd = (data[f"{a}/{o}"] for a, o in CELLS)
    counts = {
        "F1": sum(v < .2 for v in oa.values()),
        "F2": sum(v > .9 for v in ha.values()),
        "F3": sum(v >= .7 for v in reset.values()),
        "F4": sum(abs(hr[s] - ha[s]) <= .05 for s in FACTORIAL_SEEDS),
        "F5_pass": sum(v >= .7 for v in b2.values()),
        "F5_refutation": sum(v < .3 for v in b2.values()),
        "F6": sum(v >= .7 for v in gd.values()),
    }
    verdicts = {k: ("holds" if counts[k] >= (5 if k == "F4" else 4) else "fails")
                for k in ("F1", "F2", "F3", "F4")}
    verdicts["F5"] = ("holds" if counts["F5_pass"] >= 4 else
                      "refuted_as_stated" if counts["F5_refutation"] >= 4 else "partial")
    verdicts["F6"] = ("uninformative" if not sgd_informative else "holds" if counts["F6"] >= 4 else "fails")
    return {"seed_count": 5, "counts": counts, "verdicts": verdicts}


def paired_contrasts(data):
    oa, reset, ha, hr = ([data[f"{a}/{o}"][s] for s in FACTORIAL_SEEDS] for a, o in CELLS[:4])
    original_effect = [r - a for r, a in zip(reset, oa)]
    hash_effect = [r - a for r, a in zip(hr, ha)]
    gaps = [h - a for h, a in zip(ha, oa)]
    require(all(g > 0 for g in gaps), "recovery fraction requires positive original-to-hash gaps")
    return {"reset_effect_original": stats(original_effect), "reset_effect_hash": stats(hash_effect),
            "interaction_reset_original_minus_hash": stats(a - b for a, b in zip(original_effect, hash_effect)),
            "reset_recovery_fraction_per_seed": stats(e / g for e, g in zip(original_effect, gaps)),
            "reset_recovery_ratio_of_means": mean(original_effect) / mean(gaps),
            "interpretation": "Conditional descriptive contrasts under these recipes; no unique causal allocation to optimizer state or collisions."}


def load_factorial(root):
    directory = root / "results/memory_saturation_factorial"
    sweep = []
    for lr in (1, 3, 10, 30, 100, 300):
        row = json.loads((directory / f"dev-hash-sgd-lr{lr}-s3001.json").read_text())
        value = summarize_factorial_run(row, "hash", "sgd", 3001, lr)
        sweep.append((lr, value["late_acquire"]))
    selection = choose_sgd_lr(sweep)
    saved = json.loads((directory / "dev-sgd-selection.json").read_text())
    require(all(saved[k] == v for k, v in selection.items()), "stored SGD selection disagrees")
    require(saved["sweep"] == [list(pair) for pair in sweep], "stored dev sweep disagrees")
    cells, acquisition, retention, rounded = {}, {}, {}, {}
    for address, optimizer in CELLS:
        name = f"{address}/{optimizer}"
        lr = selection["chosen_lr"] if optimizer == "sgd" else .001
        rows = [summarize_factorial_run(json.loads((directory / f"{address}-{optimizer}-s{s}.json").read_text()),
                                      address, optimizer, s, lr) for s in FACTORIAL_SEEDS]
        cells[name] = {"per_seed": rows, "metrics": {k: stats(r[k] for r in rows)
                       for k in ("late_acquire", "late_retain", "early_acquire", "final_slots_used")}}
        acquisition[name] = {r["seed"]: r["late_acquire"] for r in rows}
        retention[name] = {r["seed"]: r["late_retain"] for r in rows}
        rounded[name] = {r["seed"]: r["stored_rounded"]["late_acquire"] for r in rows}
    rules = prediction_counts(acquisition, selection["informative"])
    require(rules == prediction_counts(rounded, selection["informative"]), "rounded and raw-log verdicts diverge")
    retention_contrasts = {
        name: stats(retention[f"{name}/adam_reset"][s] - retention[f"{name}/adam"][s] for s in FACTORIAL_SEEDS)
        for name in ("original", "hash")}
    return {"selection": {**selection, "sweep": sweep}, "cells": cells,
            "registered_outcomes": rules, "acquisition_contrasts": paired_contrasts(acquisition),
            "secondary_retention_reset_contrasts": retention_contrasts,
            "unit": "Seed; 15 late phases are repeated measurements, not independent runs."}


def literal_world_constants(path):
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            names = [x.id for x in node.targets[0].elts if isinstance(x, ast.Name)]
            if names == ["Z", "R", "V", "EPS", "K"]:
                values = ast.literal_eval(node.value)
                return {k: v for k, v in zip(names, values) if k != "EPS"}
    raise ValueError("literal world constants missing")


def exogenous_stream(seed, c):
    import numpy as np
    rng = np.random.default_rng(seed)
    rule = rng.integers(c["V"], size=(c["Z"], c["R"]))
    truth = rule[rng.integers(c["Z"], size=2560)].copy()
    shown = np.zeros((2560, c["R"]), bool)
    for entity in range(2560):
        shown[entity, rng.choice(c["R"], c["K"], replace=False)] = True
    corrected = np.zeros_like(shown)
    when = np.full(shown.shape, -1)
    collisions, events, denominators, seen = [], [], [], set()
    noops = repeats = 0
    for phase in range(1, 40):
        oe, rr = np.nonzero(shown[:phase * 64])
        pick = rng.choice(len(oe), 16, replace=False)
        entities, relations = oe[pick], rr[pick]
        values = rng.integers(c["V"], size=16)
        noops += int((truth[entities, relations] == values).sum())
        truth[entities, relations] = values
        corrected[entities, relations] = True
        when[entities, relations] = phase
        counts = Counter(map(int, entities))
        if len(counts) < 16:
            collisions.append({"phase": phase, "repeated_entity_groups": {e: n for e, n in counts.items() if n > 1},
                               "candidates_beyond_distinct_entities": 16 - len(counts)})
        ce = sorted(counts)
        e, r = np.nonzero(shown[ce])
        e = np.asarray(ce)[e]
        denominators.append(int((~corrected[e, r]).sum()))
        for e, r, value in zip(entities, relations, values):
            key = (int(e), int(r))
            repeats += int(key in seen)
            seen.add(key)
            events.append({"phase": phase, "entity": key[0], "relation": key[1], "value": int(value)})
    e, r = np.nonzero(shown)
    age = 39 - np.repeat(np.arange(40), 64)[e]
    old_regular = ~corrected[e, r] & (age >= 20) & corrected.any(1)[e]
    old_correction = corrected[e, r] & (39 - when[e, r] >= 20)
    return {"seed": seed, "correction_event_count": len(events), "correction_phase_count": 39,
            "same_value_assignment_count": noops, "repeat_key_assignment_count": repeats,
            "unique_corrected_keys": len(seen), "phases_with_duplicate_entity_candidates": len(collisions),
            "candidates_beyond_distinct_entities": sum(x["candidates_beyond_distinct_entities"] for x in collisions),
            "collision_details": collisions, "collateral_phase_denominators": denominators,
            "final_old_correction_denominator": int(old_correction.sum()),
            "final_old_regular_corrected_entity_denominator": int(old_regular.sum()),
            "stream_sha256": hashlib.sha256(json.dumps(events, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "numpy_version": np.__version__,
            "scope": "Exogenous source stream only; does not reconstruct learned predictions, scatter order, or historical error causes."}


def independent_candidate_counterexample():
    # Relation-specific target-minus-competitor margins are u0 and u1.
    candidates = [[1., 0.], [0., 1.]]
    joint = [1., 1.]
    return {"candidates": candidates, "individual_target_margins": [candidates[0][0], candidates[1][1]],
            "joint_target_margins_for_each_candidate": candidates, "jointly_feasible_state": joint,
            "scope": "Two individually successful detached candidates need not preserve each other's target margin in one shared row; no assumed duplicate-index ordering and no reconstructed historical cause."}


def reconcile(root):
    manifest = verify_manifest(root)
    correction_source = root / "correction" / SOURCE
    c = literal_world_constants(correction_source / "cab_s.py")
    require(c == {"Z": 16, "R": 8, "V": 32, "K": 3}, "unexpected world constants")
    return {"scope": "Read-only source-and-result audit; no neural run, independent replication, or global causal theorem.",
            "provenance": manifest["provenance"], "verified_input_count": len(manifest["files"]),
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            "input_hashes": {e["local_path"]: e["sha256"] for e in manifest["files"]},
            "correction": {"arms": load_correction(correction_source),
                           "exogenous_streams": [exogenous_stream(s, c) for s in CORRECTION_SEEDS],
                           "counterexample": independent_candidate_counterexample()},
            "factorial": load_factorial(root / "factorial" / SOURCE)}


def write_report(report, target):
    # Validate/serialize before opening; exclusive creation refuses existing outputs.
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with target.open("x") as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = reconcile(args.evidence_dir)
    if args.out:
        write_report(report, args.out)
        print(f"Verified {report['verified_input_count']} frozen inputs; wrote {args.out}")
    else:
        print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
