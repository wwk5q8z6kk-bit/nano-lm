"""Read-only P0 audit of PR #203 tournament JSON; NOT a replacement scorer.

Usage: python audit_tournament_receipts.py seed_1000.json seed_1001.json seed_1002.json \
       --summary summary.json --scorer score_tournament.py
Or: python audit_tournament_receipts.py --self-test
Never writes result files, executes training, or changes registered verdicts.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ARMS = ("dense", "value_res", "block_attnres", "coda_refine", "dense10")
SEEDS = (1000, 1001, 1002)
STEP_TARGET = 3000
TOKEN_TARGET = 98_304_000


def audit(records: list[dict], summary: dict | None = None, scorer_source: str | None = None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    seeds = [d.get("seed") for d in records]
    if sorted(seeds) != list(SEEDS):
        errors.append(f"Required distinct seeds {SEEDS}, got {seeds}")

    for rec in records:
        seed = rec.get("seed")
        if rec.get("completed") is not True:
            errors.append(f"seed {seed}: completed is not true")
        recipe = rec.get("recipe", {})
        if recipe.get("steps") != STEP_TARGET or recipe.get("tokens_per_arm") != TOKEN_TARGET:
            errors.append(f"seed {seed}: recipe exposure differs from registered target")
        arm_map = rec.get("arms", {})
        if set(arm_map) != set(ARMS):
            errors.append(f"seed {seed}: missing/extra arms {sorted(set(ARMS) ^ set(arm_map))}")
        if set(rec.get("arm_order", [])) != set(ARMS) or len(rec.get("arm_order", [])) != len(ARMS):
            errors.append(f"seed {seed}: arm order incomplete or duplicated")
        for arm in ARMS:
            if arm not in arm_map:
                continue
            entry = arm_map[arm]
            curve = entry.get("curve") or []
            if not curve:
                errors.append(f"seed {seed}/{arm}: empty evaluation curve")
                continue
            prev_step, prev_tokens, prev_sec = -1, -1, -1.0
            for row in curve:
                try:
                    step, tokens, sec = int(row["step"]), int(row["tokens"]), float(row["train_sec"])
                    vals = [float(row[key]) for key in ("mix", "web", "textbook", "math", "code")]
                except (KeyError, TypeError, ValueError):
                    errors.append(f"seed {seed}/{arm}: incomplete or malformed curve row")
                    break
                if not all(map(math.isfinite, [sec, *vals])) or sec <= prev_sec:
                    errors.append(f"seed {seed}/{arm}: nonfinite metric or non-monotone time")
                if step <= prev_step or tokens <= prev_tokens or tokens != step * 64 * 512:
                    errors.append(f"seed {seed}/{arm}: non-monotone or inconsistent step/tokens")
                prev_step, prev_tokens, prev_sec = step, tokens, sec
            last = curve[-1]
            if last.get("step") != STEP_TARGET or last.get("tokens") != TOKEN_TARGET:
                errors.append(f"seed {seed}/{arm}: final step or tokens missing")
            if entry.get("skipped_amp_steps") not in (0, None):
                warnings.append(f"seed {seed}/{arm}: AMP updates skipped; assess exposure")

        # Repo-result blob hashes are Git custody, NOT run-time dataset/checkpoint custody.
        if not any(k in rec for k in ("dataset_sha256", "train_sha256", "code_sha256", "checkpoint_sha256")):
            warnings.append(f"seed {seed}: no immutable data/code/checkpoint hashes in receipt")

    value_gain = None
    if not errors and len(records) == 3:
        by_seed = {r["seed"]: r for r in records}
        diffs = [float(by_seed[s]["arms"]["dense"]["curve"][-1]["mix"])
                 - float(by_seed[s]["arms"]["value_res"]["curve"][-1]["mix"]) for s in SEEDS]
        mu = sum(diffs) / 3
        sd = math.sqrt(sum((x-mu)**2 for x in diffs) / 2)
        t_stat = mu / (sd / math.sqrt(3)) if sd > 1e-12 else None
        value_gain = {"per_seed": [round(x, 5) for x in diffs],
                      "mean_bpb": round(mu, 6), "sample_sd": round(sd, 6),
                      "paired_t_df2": round(t_stat, 3) if t_stat is not None else None,
                      "within_registered_P3_interval_0.001_to_0.006": 0.001 <= mu <= 0.006}
        if summary is not None:
            expected = summary.get("comparisons", {}).get("value_res_vs_dense", {}).get("equal_tokens_gain", {})
            if not expected or abs(float(expected.get("mean", 1e20)) - mu) > 0.000011:
                errors.append("committed summary's value-residual mean does not match source receipts")

    scorer_checks = None
    if scorer_source is not None:
        start = scorer_source.find("def score_files(")
        end = scorer_source.find('if __name__ == "__main__":', start)
        section = scorer_source[start:end if end >= 0 else None] if start >= 0 else ""
        scorer_checks = {
            "completion_guard_present": bool(re.search(r'\[\s*[\"\']completed[\"\']\s*\]', section)),
            "step_validation_present": bool(re.search(r'\[\s*[\"\']step[\"\']\s*\]', section)),
            "wallclock_significance_in_D1": 'ba_time["sig_95"]' in section and 'vr_time["sig_95"]' in section,
            "duplicate_seed_guard_present": bool(re.search(r'if\s+sd\s+in\s+by_seed', section)),
        }
        if not all(scorer_checks.values()):
            warnings.append("PR #203 scorer omits one or more prospective integrity checks; do not modify historical verdict")

    return {"receipt_status": "PASS_SOURCE_RECEIPTS_ONLY" if not errors else "REJECT_INCOMPLETE_OR_INCONSISTENT",
            "errors": errors, "warnings": warnings, "value_residual": value_gain,
            "scorer_source_checks": scorer_checks,
            "training_reproduced": False, "architecture_adopted": False}


def self_test() -> None:
    def row(step: int) -> dict:
        return {"step": step, "tokens": step*64*512, "train_sec": float(step),
                "mix": 1.25, "web": 1.2, "textbook": 1.0, "math": 1.4, "code": 1.1}
    def seed(n: int) -> dict:
        return {"seed": n, "completed": True, "recipe": {"steps": 3000, "tokens_per_arm": TOKEN_TARGET},
                "arm_order": list(ARMS), "arms": {a: {"curve": [row(500), row(3000)], "skipped_amp_steps": 0} for a in ARMS}}
    clean = [seed(n) for n in SEEDS]
    assert not audit(clean)["errors"]
    partial = json.loads(json.dumps(clean)); partial[1]["completed"] = False
    assert audit(partial)["errors"]
    missing_step = json.loads(json.dumps(clean)); missing_step[0]["arms"]["dense"]["curve"][-1] = row(2500)
    assert audit(missing_step)["errors"]
    duplicate = json.loads(json.dumps(clean)); duplicate[1]["seed"] = 1000
    assert audit(duplicate)["errors"]
    print("PASS: clean synthetic fixtures; incomplete, short and duplicate seed receipts rejected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--scorer", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    else:
        if not a.files:
            ap.error("provide three seed JSON files (or --self-test)")
        data = [json.loads(p.read_text()) for p in a.files]
        summary = json.loads(a.summary.read_text()) if a.summary else None
        source = a.scorer.read_text() if a.scorer else None
        result = audit(data, summary=summary, scorer_source=source)
        print(json.dumps(result, indent=2))
        if result["errors"]:
            raise SystemExit(1)
