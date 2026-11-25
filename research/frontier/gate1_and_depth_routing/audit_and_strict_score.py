"""Additive receipt validator and strict paired-significance scorer (P0 audit follow-up).

Preserves historic `score_tournament.py` and `PROTOCOL.md` unchanged while addressing both
Cursor Bugbot / PR #211 findings on future or re-audited runs:
  1. Rejects incomplete receipts (`completed != True`), truncated curves (`step != expected_steps`
     or `tokens != expected_tokens`), missing arms, or duplicate seed IDs.
  2. Enforces `sig_95` (`|t| > 4.303` for df=2) on BOTH `equal_tokens_gain` AND
     `equal_wallclock_gain` in the strict D1 predicate, and explicitly separates the quadrant
     where `value_res` passes both equal-token and equal-wall-clock hurdles while `block_attnres`
     fails equal-wall-clock (`QUALIFY_VALUE_RES_FOR_42M_GATE_REJECT_BLOCK_ATTNRES`).
"""
from __future__ import annotations
import glob
import hashlib
import json
import sys
from pathlib import Path

from score_tournament import score_files

EXPECTED_ARMS = ("dense", "value_res", "block_attnres", "coda_refine", "dense10")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_seed_receipts(
    paths: list[Path],
    expected_steps: int = 3000,
    expected_tokens: int = 98_304_000,
    expected_arms: tuple[str, ...] = EXPECTED_ARMS,
) -> dict:
    if not paths:
        raise ValueError("No seed JSON files supplied")
    seen_seeds: set[int] = set()
    file_hashes: dict[str, str] = {}
    for p in paths:
        d = json.loads(p.read_text())
        if d.get("completed") is not True:
            raise ValueError(f"{p.name}: receipt not marked completed=True (got {d.get('completed')!r})")
        seed = int(d["seed"])
        if seed in seen_seeds:
            raise ValueError(f"{p.name}: duplicate seed ID {seed}")
        seen_seeds.add(seed)
        arms = d.get("arms", {})
        for arm in expected_arms:
            if arm not in arms:
                raise ValueError(f"{p.name}: missing required arm {arm!r}")
            curve = arms[arm].get("curve", [])
            if not curve:
                raise ValueError(f"{p.name}: empty curve for arm {arm!r}")
            last = curve[-1]
            if int(last.get("step", -1)) != expected_steps:
                raise ValueError(
                    f"{p.name} arm {arm!r}: final step {last.get('step')} != {expected_steps}"
                )
            if int(last.get("tokens", -1)) != expected_tokens:
                raise ValueError(
                    f"{p.name} arm {arm!r}: final tokens {last.get('tokens')} != {expected_tokens}"
                )
        file_hashes[p.name] = sha256_file(p)
    return {"validated_seeds": sorted(seen_seeds), "receipt_sha256": file_hashes}


def strict_score_files(
    paths: list[Path],
    expected_steps: int = 3000,
    expected_tokens: int = 98_304_000,
) -> dict:
    prov = validate_seed_receipts(paths, expected_steps=expected_steps, expected_tokens=expected_tokens)
    base = score_files(paths)
    cmp_ = base["comparisons"]
    ba_tok = cmp_["block_attnres_vs_dense"]["equal_tokens_gain"]
    ba_time = cmp_["block_attnres_vs_dense"]["equal_wallclock_gain"]
    ba_vr = cmp_["block_attnres_vs_value_res"]["equal_tokens_gain"]
    vr_tok = cmp_["value_res_vs_dense"]["equal_tokens_gain"]
    vr_time = cmp_["value_res_vs_dense"]["equal_wallclock_gain"]

    ba_strict_pass = (
        ba_tok["mean"] >= 0.004
        and ba_tok["sig_95"]
        and ba_time["mean"] >= 0.004
        and ba_time["sig_95"]
        and ba_vr["mean"] >= 0.002
        and ba_vr["sig_95"]
    )
    vr_strict_pass = (
        vr_tok["mean"] >= 0.004
        and vr_tok["sig_95"]
        and vr_time["mean"] >= 0.004
        and vr_time["sig_95"]
    )
    if ba_strict_pass:
        strict_d1 = "ADOPT_BLOCK_ATTNRES"
    elif vr_strict_pass and ba_vr["mean"] < 0.002:
        strict_d1 = "ADOPT_VALUE_RES_REJECT_BLOCK_ATTNRES"
    elif vr_strict_pass:
        strict_d1 = "QUALIFY_VALUE_RES_FOR_42M_GATE_REJECT_BLOCK_ATTNRES"
    else:
        strict_d1 = "CLOSE_DEPTH_ROUTING_AT_14M"

    return {
        "provenance": prov,
        "historic_verdicts_unchanged": base["verdicts"],
        "strict_post_audit_verdicts": {
            "D1_strict": strict_d1,
            "D2_strict": base["verdicts"]["D2_coda_refinement"],
        },
    }


if __name__ == "__main__":
    files = [Path(p) for p in (sys.argv[1:] or sorted(glob.glob("results/tournament/seed_*.json")))]
    print(json.dumps(strict_score_files(files), indent=2))
