#!/usr/bin/env python3
"""NANO-SLW-001 runner — Synthetic Longitudinal World.

Emits the benchmark result plus the artifacts needed to audit it: the ground
truth the world generated, the corrupted observation channel the system was
allowed to see, and the two arms' snapshots. Auditability is the point — a
benchmark whose intermediate state cannot be inspected can only be trusted.

Synthetic, non-PHI data only. No model call, no training, no paid compute, no
network. Deterministic given `--seed`.

Usage:
    .venv/bin/python scripts/run_nano_slw_001.py [--out DIR] [--seed N]
                                                [--sweep] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nano.slw import (  # noqa: E402
    SyntheticWorld,
    WorldSpec,
    build_full,
    run_baseline_a,
    run_candidate_b,
    run_slw_001,
    state_signature,
)


def _enc(o):
    if is_dataclass(o) and not isinstance(o, type):
        return {k: _enc(v) for k, v in asdict(o).items()}
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (list, tuple, set)):
        return [_enc(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _enc(v) for k, v in o.items()}
    return o


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_enc(payload), indent=2, sort_keys=False) + "\n")


def _jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(_enc(row), sort_keys=True) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/nano_slw_001")
    ap.add_argument("--seed", type=int, default=WorldSpec().seed)
    ap.add_argument("--quick", action="store_true",
                    help="small world, for smoke-checking the harness")
    ap.add_argument("--sweep", action="store_true",
                    help="re-run across seeds; a result that only holds on one "
                         "seed is a coincidence, not a property")
    args = ap.parse_args()

    spec = (WorldSpec(seed=args.seed, n_sites=3, units_per_site=3, n_ticks=20,
                      checkpoint_every=5)
            if args.quick else WorldSpec(seed=args.seed))

    result = run_slw_001(spec)
    _json(args.out / "benchmark_results.json", result)

    # Auditable intermediates.
    world = SyntheticWorld.generate(spec)
    _jsonl(args.out / "ground_truth_changes.jsonl", world.changes)
    _jsonl(args.out / "observations.jsonl", world.observations)
    _json(args.out / "world_entities.json",
          {"entities": {k: v.value for k, v in sorted(world.entities.items())},
           "relations": {f"{s}|{r.value}": d
                         for (s, r), d in sorted(
                             world.relations.items(), key=lambda kv: kv[0][0])}})

    builder = build_full(world, world.observations, spec.n_ticks)
    _jsonl(args.out / "evidence_spans.jsonl", builder.ledger.spans)
    _jsonl(args.out / "assertions.jsonl", builder.ledger.assertions)
    _jsonl(args.out / "conflicts.jsonl", builder.conflict_records())
    _jsonl(args.out / "knowledge_gaps.jsonl", builder.ledger.gaps)

    a, b = run_baseline_a(world), run_candidate_b(world)
    _json(args.out / "arm_snapshots.json", {
        arm.name: {str(t): {"signature_hash": hash(state_signature(s)),
                            "ledger_version": s.evidence_ledger_version,
                            "facts": len(s.active_conditions)
                            + len(s.current_medications)
                            + len(s.laboratory_state),
                            "uncertain": len(s.uncertainties),
                            "recomputed": len(arm.recomputed[t])}
                   for t, s in arm.snapshots.items()}
        for arm in (a, b)})

    if args.sweep:
        sweep = []
        for seed in (args.seed + i for i in range(5)):
            r = run_slw_001(WorldSpec(seed=seed))
            sweep.append({
                "seed": seed,
                "final_state_identical": r["equivalence"]["final_state_identical"],
                "recomputation_ratio": r["cost"]["recomputation_ratio"],
                "precision": r["invalidation"]["precision"],
                "recall": r["invalidation"]["recall"],
                "isolation": r["branch_isolation"]["isolation"],
                "undeclared_error": r["faithfulness"]["nano"]["undeclared_error"],
                "control_undeclared_error":
                    r["faithfulness"]["silent_resolution_control"]["undeclared_error"],
            })
        _json(args.out / "seed_sweep.json", sweep)
        result["seed_sweep"] = sweep

    eq = result["equivalence"]
    print(f"NANO-SLW-001  seed={spec.seed}  spec={result['spec_fingerprint']}")
    print(f"  world              {result['world']['entities']} entities, "
          f"{result['world']['observations']} observations, "
          f"{result['world']['unobserved_changes']} unreported changes")
    print(f"  equivalence        final={eq['final_state_identical']} "
          f"history={eq['all_checkpoints_identical']} "
          f"conflicts={eq['conflict_sets_identical']}")
    print(f"  recompute ratio    {result['cost']['recomputation_ratio']}")
    print(f"  invalidation       P={result['invalidation']['precision']:.3f} "
          f"R={result['invalidation']['recall']:.3f}")
    print(f"  branch isolation   {result['branch_isolation']['isolation']:.3f}")
    print(f"  undeclared error   nano="
          f"{result['faithfulness']['nano']['undeclared_error']} "
          f"control="
          f"{result['faithfulness']['silent_resolution_control']['undeclared_error']}")
    print(f"  artifacts          {args.out}")

    # Exit non-zero when the arms disagree: a benchmark that reports a speedup
    # over a different answer is worse than no benchmark.
    return 0 if (eq["final_state_identical"]
                 and eq["all_checkpoints_identical"]
                 and eq["conflict_sets_identical"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
