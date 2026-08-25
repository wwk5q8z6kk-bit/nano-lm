#!/usr/bin/env python3
"""Build a Native training corpus with a full, auditable manifest.

Replaces the 4,516-character fixture as a TRAINING corpus. That fixture is
retained only as NATIVE_UNIT_OVERFIT_FIXTURE for trainer correctness tests.

Emits schema `nano.native.corpus.v1` — deliberately NOT `nano.distill.train.v1`,
because nanoscribe.native.data.load_train_examples ignores file contents and
regenerates from code for that schema, which would silently discard this corpus.

Usage:
    python3 scripts/build_native_corpus.py --stage screen
    python3 scripts/build_native_corpus.py --stage promotion --out <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.native.corpus.adversarial import generate_adversarial
from nanoscribe.native.corpus.generators import generate_mechanism
from nanoscribe.native.corpus.schema import CORPUS_SCHEMA, Partition
from nanoscribe.native.corpus.validate import (
    axis_coverage_floor,
    check_leakage,
    dedupe,
    sequence_budget,
    statistics,
)

# Stage targets are expressed as generation breadth, not as a hard row count:
# the honest budget unit is tokens, which the manifest reports.
STAGES = {
    "smoke": {"composed_mech": 40, "composed_adv": 20},
    "screen": {"composed_mech": 6000, "composed_adv": 2000},
    "promotion": {"composed_mech": 14386, "composed_adv": 6000},
}


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def build(stage: str) -> tuple[list, dict]:
    cfg = STAGES[stage]
    examples = []
    # DEV/INTERNAL_TEST are measurement sets, not training volume. Cap them so
    # TRAIN carries the token budget; an unbounded held-out split just burns
    # generation on values the model will never learn from.
    holdout_cap = max(60, cfg["composed_mech"] // 8)
    for partition in (Partition.TRAIN, Partition.DEV, Partition.INTERNAL_TEST):
        is_train = partition is Partition.TRAIN
        mech = cfg["composed_mech"] if is_train else holdout_cap
        adv = cfg["composed_adv"] if is_train else max(30, holdout_cap // 3)
        examples.extend(generate_mechanism(partition, limit_composed=mech))
        examples.extend(generate_adversarial(partition, limit_composed=adv))

    examples, dedupe_stats = dedupe(examples)
    leakage = check_leakage(examples)
    stats = statistics(examples)
    coverage = axis_coverage_floor(examples, minimum=64)
    budget = sequence_budget(examples)

    content = hashlib.sha256(
        "".join(sorted(f"{e.encounter_id}{e.target}" for e in examples)).encode()
    ).hexdigest()

    manifest = {
        "schema": CORPUS_SCHEMA,
        "corpus_id": f"native_corpus_{stage}_v1",
        "revision": f"native_corpus_{stage}_v1",
        "stage": stage,
        "built_at": datetime.now(UTC).isoformat(),
        "generator_commit_sha": _git_sha(),
        "seed_namespace": "native_corpus_v1",
        "no_phi": True,
        "content_hash": content,
        "generation_config": cfg,
        "statistics": stats,
        "dedupe": dedupe_stats,
        "leakage": leakage,
        "axis_coverage": coverage,
        "sequence_budget": budget,
        "gates": {
            "leakage_pass": leakage["pass"],
            "axis_coverage_pass": coverage["pass"],
            "sequence_budget_pass": budget["pass"],
            "nonzero": len(examples) > 0,
        },
    }
    manifest["gates"]["all_pass"] = all(manifest["gates"].values())
    return examples, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Native corpus")
    ap.add_argument("--stage", choices=sorted(STAGES), default="screen")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    examples, manifest = build(args.stage)
    out = args.out or ROOT / f"artifacts/campaign/native_corpus_{args.stage}_v1.json"
    manifest_path = out.with_name(out.stem + "_manifest.json")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    written: dict[str, str] = {}
    if not args.manifest_only:
        # One file PER PARTITION. nanoscribe.native.data.load_train_examples reads
        # every `entries` row it is given and does not filter on partition, so a
        # single mixed file would train the model on its own DEV and
        # INTERNAL_TEST splits. Separate files make that leak structurally
        # impossible rather than merely discouraged.
        for partition in (Partition.TRAIN, Partition.DEV, Partition.INTERNAL_TEST):
            subset = [e for e in examples if e.partition is partition]
            if not subset:
                continue
            path = out.with_name(f"{out.stem}_{partition.value.lower()}.json")
            payload = {
                **{k: v for k, v in manifest.items() if k != "statistics"},
                "n_cases": len(subset),
                "partition": partition.value,
                "held_out_eval": partition is not Partition.TRAIN,
                "entries": [e.to_entry() for e in subset],
            }
            path.write_text(json.dumps(payload) + "\n")
            written[partition.value] = str(path.relative_to(ROOT))
        manifest["partition_files"] = written
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        json.dumps(
            {
                "partition_files": written,
                "manifest": str(manifest_path.relative_to(ROOT)),
                "n_examples": manifest["statistics"]["n_examples"],
                "training_tokens_char_level": manifest["statistics"]["training_tokens_char_level"],
                "partition_sizes": manifest["statistics"]["partition_sizes"],
                "gates": manifest["gates"],
            },
            indent=2,
        )
    )
    return 0 if manifest["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
