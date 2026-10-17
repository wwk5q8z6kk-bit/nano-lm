#!/usr/bin/env python3
"""Estimate char-level training tokens for a native corpus manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.native.corpus.manifest import load_manifest
from nanoscribe.native.corpus.registry import TIER_TARGETS


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate native corpus token budget")
    ap.add_argument("manifest", type=Path, nargs="?", default=ROOT / "artifacts/campaign/native_corpus_screen_v1_manifest.json")
    ap.add_argument("--tier", choices=sorted(TIER_TARGETS), default="R30")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    stats = manifest.payload.get("statistics", {})
    tokens = int(stats.get("training_tokens_char_level", 0))
    tier = TIER_TARGETS[args.tier]

    if "unique_tokens_char_level" in tier:
        target = tier["unique_tokens_char_level"]
        exposure = tier.get("exposure_tokens_char_level", target)
        report = {
            "tier": args.tier,
            "observed_char_level_tokens": tokens,
            "target_unique_char_level_tokens": target,
            "target_exposure_char_level_tokens": exposure,
            "unique_fraction_of_target": round(tokens / target, 4) if target else None,
            "meets_unique_target": tokens >= target,
            "meets_exposure_target": tokens >= exposure,
        }
    else:
        lo = tier["unique_tokens_char_level_min"]
        hi = tier["unique_tokens_char_level_max"]
        report = {
            "tier": args.tier,
            "observed_char_level_tokens": tokens,
            "target_unique_char_level_tokens_min": lo,
            "target_unique_char_level_tokens_max": hi,
            "within_target_band": lo <= tokens <= hi,
            "meets_min_target": tokens >= lo,
        }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
