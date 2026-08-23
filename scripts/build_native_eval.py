#!/usr/bin/env python3
"""Build p1_screening_eval_v2 — axis-stratified, leakage-safe screening suite.

The existing frozen suite has 150 gold atoms and can only resolve effects of
~0.133 at 80% power, with several capability axes at n=4. This builds a suite
where every critical axis clears a declared eligibility floor, so a screening
comparison is not silently underpowered on the axis that matters.

Values come from a slice of the INTERNAL_TEST pool, split off by a secondary
hash. That pool is already disjoint from TRAIN and DEV, so eval_v2 values were
never trained on, and carving from it leaves the existing corpus content_hash
untouched.

The historical p1_screening_eval_v1 is PRESERVED — this is an addition, not a
replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.native.corpus import vocab
from nanoscribe.native.corpus.adversarial import generate_adversarial
from nanoscribe.native.corpus.generators import generate_mechanism
from nanoscribe.native.corpus.schema import Axis, Partition

#: Axes a screening comparison must be able to resolve. From the order's
#: critical-axis list, mapped onto the generator's Axis vocabulary.
CRITICAL_AXES: tuple[Axis, ...] = (
    Axis.OPEN_VOCAB,
    Axis.EXACT_COPY,
    Axis.WRONG_SOURCE,
    Axis.SUPERSPAN,
    Axis.ASSERTION,
    Axis.NEGATION,
    Axis.UNCERTAINTY,
    Axis.TEMPORALITY,
    Axis.EXPERIENCER,
    Axis.CONFLICT,
    Axis.NOT_MENTIONED,
    Axis.SPURIOUS_TEMPTATION,
    Axis.ABSTENTION,
    Axis.MULTI_MENTION,
)

#: Screening floor, not a universal power guarantee. nanoscribe/eval/power.py
#: computes the confirmatory requirement from a declared effect size — for
#: assertion_state at 5pp paired that is 962 per arm, not 64.
AXIS_FLOOR = 64


def _eval_v2_values() -> set[str]:
    """Half of the INTERNAL_TEST pool, reserved for eval_v2."""
    pool = vocab.composed_values_for(Partition.INTERNAL_TEST)
    return {
        v
        for v in pool
        if int(hashlib.sha256(f"evalv2:{v}".encode()).hexdigest()[:8], 16) % 2 == 0
    }


def build(target_cases: int) -> tuple[list, dict]:
    reserved = _eval_v2_values()

    # Generate from the INTERNAL_TEST partition, then keep only rows whose value
    # is in the eval_v2 half.
    pool = list(generate_mechanism(Partition.INTERNAL_TEST, limit_composed=4000)) + list(
        generate_adversarial(Partition.INTERNAL_TEST, limit_composed=1500)
    )
    candidates = [e for e in pool if e.raw_value in reserved]

    # Greedy axis-balanced selection: repeatedly take the example that best
    # serves the axis furthest below the floor. Random sampling would leave rare
    # axes (conflict, superspan) starved exactly as v1 did.
    counts: Counter[str] = Counter()
    chosen: list = []
    remaining = candidates[:]
    while remaining and len(chosen) < target_cases:
        deficits = {a.value: max(0, AXIS_FLOOR - counts[a.value]) for a in CRITICAL_AXES}
        if not any(deficits.values()) and len(chosen) >= target_cases:
            break
        best, best_score = None, -1
        for idx, ex in enumerate(remaining):
            score = sum(deficits.get(a.value, 0) for a in ex.axes)
            if score > best_score:
                best, best_score = idx, score
        if best is None or best_score <= 0:
            # every floor met; top up to target with whatever remains
            chosen.extend(remaining[: target_cases - len(chosen)])
            break
        ex = remaining.pop(best)
        chosen.append(ex)
        for a in ex.axes:
            counts[a.value] += 1

    axis_hist = Counter()
    for ex in chosen:
        for a in ex.axes:
            axis_hist[a.value] += 1
    below = {a.value: axis_hist[a.value] for a in CRITICAL_AXES if axis_hist[a.value] < AXIS_FLOOR}

    train_vals = {v.lower() for v in vocab.composed_values_for(Partition.TRAIN)}
    dev_vals = {v.lower() for v in vocab.composed_values_for(Partition.DEV)}
    used = {e.raw_value.lower() for e in chosen}
    frozen_v1 = vocab.forbidden_values()

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=False
    ).stdout.strip()

    manifest = {
        "schema": "nano.eval.screening_suite.v2",
        "suite_id": "p1_screening_eval_v2",
        "built_at": datetime.now(UTC).isoformat(),
        "generator_commit_sha": sha,
        "preserves": "p1_screening_eval_v1 is retained unchanged; this is an addition",
        "n_cases": len(chosen),
        "axis_floor": AXIS_FLOOR,
        "axis_floor_note": (
            "screening floor, not a power guarantee — see nanoscribe/eval/power.py "
            "for confirmatory requirements from a declared effect size"
        ),
        "axis_histogram": dict(axis_hist.most_common()),
        "axes_below_floor": below,
        "target_label_histogram": dict(Counter(e.target.split(":")[0] for e in chosen)),
        "unique_values": len(used),
        "leakage": {
            "overlap_with_train_values": sorted(used & train_vals)[:5],
            "overlap_with_dev_values": sorted(used & dev_vals)[:5],
            "overlap_with_frozen_v1_values": sorted(used & frozen_v1)[:5],
            "pass": not (used & train_vals) and not (used & dev_vals) and not (used & frozen_v1),
        },
        "content_hash": hashlib.sha256(
            "".join(sorted(f"{e.encounter_id}{e.target}" for e in chosen)).encode()
        ).hexdigest(),
    }
    manifest["gates"] = {
        "leakage_pass": manifest["leakage"]["pass"],
        "axis_floor_pass": not below,
        "size_in_range": 256 <= len(chosen) <= 512,
    }
    manifest["gates"]["all_pass"] = all(manifest["gates"].values())
    return chosen, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build p1_screening_eval_v2")
    ap.add_argument("--cases", type=int, default=512)
    ap.add_argument("--out", type=Path, default=ROOT / "data/p1_screening_eval_v2.json")
    args = ap.parse_args()

    chosen, manifest = build(args.cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({**manifest, "entries": [e.to_entry() for e in chosen]}, indent=2) + "\n"
    )
    mpath = ROOT / "artifacts/campaign/p1_screening_eval_v2_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")

    print(json.dumps({k: v for k, v in manifest.items() if k != "axis_histogram"}, indent=2))
    print("\naxis histogram:")
    for axis, n in sorted(manifest["axis_histogram"].items(), key=lambda kv: -kv[1]):
        flag = "" if n >= AXIS_FLOOR else "  <-- BELOW FLOOR"
        print(f"  {axis:24s} {n:>5}{flag}")
    return 0 if manifest["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
