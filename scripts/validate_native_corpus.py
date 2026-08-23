#!/usr/bin/env python3
"""Validate a built native corpus manifest and partition files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.native.corpus.manifest import load_manifest
from nanoscribe.native.corpus.schema import CORPUS_SCHEMA, Axis, CorpusExample, Layer, Partition
from nanoscribe.native.corpus.validate import axis_coverage_floor, check_leakage, dedupe, statistics


def _load_examples(path: Path) -> list[CorpusExample]:
    payload = json.loads(path.read_text())
    rows: list[CorpusExample] = []
    for entry in payload.get("entries", []):
        rows.append(
            CorpusExample(
                encounter_id=str(entry["encounter_id"]),
                atom_id=str(entry.get("atom_id", entry["encounter_id"])),
                prompt=str(entry.get("prompt", "")),
                target=str(entry.get("target", "")),
                raw_value=str(entry.get("raw_value", "")),
                axes=tuple(Axis(a) for a in entry.get("axes", [])),
                layer=Layer(str(entry.get("layer", Layer.MECHANISM.value))),
                template_id=str(entry.get("template_id", "unknown")),
                partition=Partition(str(entry.get("partition", Partition.TRAIN.value))),
            )
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate native corpus manifest")
    ap.add_argument("manifest", type=Path, nargs="?", default=ROOT / "artifacts/campaign/native_corpus_screen_v1_manifest.json")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    if manifest.payload.get("schema") != CORPUS_SCHEMA:
        print(json.dumps({"ok": False, "error": "wrong schema"}))
        return 1

    examples: list[CorpusExample] = []
    for rel in manifest.payload.get("partition_files", {}).values():
        examples.extend(_load_examples(ROOT / rel))

    deduped, dedupe_stats = dedupe(examples)
    leakage = check_leakage(deduped)
    stats = statistics(deduped)
    coverage = axis_coverage_floor(deduped, minimum=64)

    report = {
        "manifest": str(args.manifest.relative_to(ROOT)),
        "corpus_id": manifest.corpus_id,
        "content_hash": manifest.content_hash,
        "dedupe": dedupe_stats,
        "leakage": leakage,
        "statistics": stats,
        "axis_coverage": coverage,
        "gates": {
            "leakage_pass": leakage["pass"],
            "axis_coverage_pass": coverage["pass"],
            "nonzero": len(deduped) > 0,
        },
    }
    report["gates"]["all_pass"] = all(report["gates"].values())
    print(json.dumps(report, indent=2))
    return 0 if report["gates"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
