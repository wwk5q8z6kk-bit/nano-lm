#!/usr/bin/env python3
"""Score the Wave-1 revalidation arms against the controls that make it a test.

An arm "winning" means nothing unless it separates from BOTH:
  * a random-init control of the same architecture, and
  * the majority-class baseline.

The prior native round reported a winner without either, and neither arm in fact
separated from an untrained model. This computes the comparison that was missing.

Reports denominator-consistent yield per gold atom with 95% Wilson intervals,
plus seed variance, because an architecture effect smaller than seed spread is
not an architecture effect.

Usage:
    python3 scripts/analyze_revalidation.py --results-dir artifacts/campaign/reval_results
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD_ATOMS = 150  # p1_screening_eval_v1
MAJORITY_BASELINE = {"k": 103, "n": 129, "rate": 0.7984}  # always-ASSERTED


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def disjoint(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[1] < b[0] or b[1] < a[0]


def load(results_dir: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(results_dir.glob("reval30_*_*.json")):
        name = path.stem
        for mode in ("constrained", "unconstrained"):
            if name.endswith(f"_{mode}"):
                run_id = name[: -len(f"_{mode}")]
                try:
                    out[run_id][mode] = json.loads(path.read_text())
                except Exception as exc:  # pragma: no cover
                    print(f"  skip {path.name}: {exc}")
    return out


def arm_of(run_id: str) -> str:
    return run_id.rsplit("_s", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "artifacts/campaign/reval_results")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/campaign/native30_revalidation_summary_v1.json")
    args = ap.parse_args()

    runs = load(args.results_dir)
    if not runs:
        print(f"no results in {args.results_dir}")
        return 1

    per_run: dict[str, dict] = {}
    for run_id, modes in sorted(runs.items()):
        row: dict = {"arm": arm_of(run_id)}
        for mode, payload in modes.items():
            m = payload.get("suite_metrics", {})
            k = m.get("assertion_state_correct_count") or 0
            row[mode] = {
                "assertion_correct": k,
                "gold_atoms": GOLD_ATOMS,
                "yield": round(k / GOLD_ATOMS, 4),
                "wilson95": wilson(k, GOLD_ATOMS),
                "coverage_count": m.get("coverage_count"),
                "malformed": m.get("malformed"),
                "decoding_mode": payload.get("decoding_mode"),
                "span_metrics_are_tautological": payload.get("span_metrics_are_tautological"),
            }
        per_run[run_id] = row

    # Aggregate by arm, carrying seed spread.
    per_arm: dict[str, dict] = {}
    for arm in sorted({r["arm"] for r in per_run.values()}):
        entry: dict = {}
        for mode in ("constrained", "unconstrained"):
            ks = [r[mode]["assertion_correct"] for r in per_run.values()
                  if r["arm"] == arm and mode in r]
            if not ks:
                continue
            total_k, total_n = sum(ks), GOLD_ATOMS * len(ks)
            entry[mode] = {
                "seeds": len(ks),
                "per_seed_correct": ks,
                "pooled": {"k": total_k, "n": total_n, "rate": round(total_k / total_n, 4),
                           "wilson95": wilson(total_k, total_n)},
                "seed_spread": (max(ks) - min(ks)) / GOLD_ATOMS if len(ks) > 1 else None,
                "seed_stdev": round(statistics.stdev(ks) / GOLD_ATOMS, 4) if len(ks) > 1 else None,
            }
        per_arm[arm] = entry

    control = "reval30_decoder_control"
    verdicts: dict[str, dict] = {}
    for arm, entry in per_arm.items():
        if arm == control or control not in per_arm:
            continue
        v: dict = {}
        for mode in ("constrained", "unconstrained"):
            if mode not in entry or mode not in per_arm[control]:
                continue
            a, c = entry[mode]["pooled"], per_arm[control][mode]["pooled"]
            beats_control = disjoint(a["wilson95"], c["wilson95"]) and a["rate"] > c["rate"]
            beats_major = a["wilson95"][0] > MAJORITY_BASELINE["rate"]
            spread = entry[mode].get("seed_spread")
            effect = a["rate"] - c["rate"]
            v[mode] = {
                "arm_rate": a["rate"], "control_rate": c["rate"],
                "effect": round(effect, 4),
                "separates_from_control": beats_control,
                "exceeds_majority_baseline": beats_major,
                "seed_spread": spread,
                "effect_exceeds_seed_spread": (
                    None if spread is None else abs(effect) > spread
                ),
                "verdict": (
                    "PROMOTION_CANDIDATE"
                    if beats_control and beats_major
                    else "NOT_SEPARATED"
                ),
            }
        verdicts[arm] = v

    summary = {
        "schema": "nano.campaign.native30_revalidation.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "suite": "p1_screening_eval_v1",
        "gold_atoms": GOLD_ATOMS,
        "majority_class_baseline": MAJORITY_BASELINE,
        "control_arm": control,
        "per_run": per_run,
        "per_arm": per_arm,
        "verdicts": verdicts,
        "decision_rule": (
            "an arm is a PROMOTION_CANDIDATE only if its pooled Wilson interval is "
            "disjoint from the decoder control's AND its lower bound exceeds the "
            "majority-class baseline. An effect smaller than the seed spread is not "
            "an architecture effect."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{'arm':32s} {'mode':14s} {'k/n':>10s} {'rate':>7s} {'wilson95':>18s} {'seeds':>5s}")
    for arm, entry in per_arm.items():
        for mode, e in entry.items():
            p = e["pooled"]
            print(f"  {arm:30s} {mode:14s} {p['k']:>4}/{p['n']:<5} {p['rate']:>7.4f} "
                  f"{str(p['wilson95']):>18s} {e['seeds']:>5}")
    print(f"\nmajority-class baseline: {MAJORITY_BASELINE['rate']}")
    print("\nVERDICTS:")
    for arm, v in verdicts.items():
        for mode, d in v.items():
            print(f"  {arm:30s} {mode:14s} effect={d['effect']:+.4f}  {d['verdict']}")
    try:
        shown = args.out.relative_to(ROOT)
    except ValueError:
        shown = args.out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
