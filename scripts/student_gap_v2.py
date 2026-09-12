#!/usr/bin/env python3
"""student_gap_v2 — paired, per-axis capability gap with explicit denominators.

v1 was wrong in three ways: it averaged per-case COUNTS as if they were rates
(producing assertion_state_correct = 1.0078), clamped that impossible value to a
perfect 1.0 with min(1.0, x), and compared against a managed reference whose
numbers were hardcoded rather than measured.

v2 fixes the arithmetic and the design. The three models score the SAME 128
cases, so the comparison is PAIRED. An unpaired test throws away that structure
and needs far more cases for the same power — for assertion_state at 5pp, 962
per arm unpaired versus far fewer paired. Discordant pairs are what carry the
signal, so this reports McNemar counts alongside the rates.

Every metric carries numerator, denominator, rate and eligible unit, per the
metric contract.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "qwen38_27b": "artifacts/p1_runs/fanout_screening_p1_qwen_structured_*.json",
    "qwen3_32b_awq_managed_ref": "artifacts/p1_runs/managed_ref_qwen3_32b_awq_c2_*.json",
    "student_a": "artifacts/p1_runs/student_a_structured_c2_screening_*.json",
}

#: (axis, extractor) — each returns True/False per atom, or None when the atom
#: is not eligible for that axis.
AXES = {
    "assertion": lambda a: bool(a.get("assertion_state_correct")),
    "transport_exact_span": lambda a: bool(a.get("exact_gold_span")),
    "support_direct_exact": lambda a: a.get("support_relation") == "direct_exact",
    "coverage": lambda a: not a.get("omitted") and not a.get("abstained"),
    "not_malformed": lambda a: not a.get("malformed"),
    "not_spurious": lambda a: not a.get("spurious_atom"),
    "abstention": lambda a: bool(a.get("abstained")),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def mcnemar(b: int, c: int) -> dict:
    """Paired test on discordant pairs only.

    b = A right / B wrong, c = A wrong / B right. Concordant pairs carry no
    information about a difference, which is exactly why the paired design needs
    fewer cases than an unpaired one.
    """
    n = b + c
    if n == 0:
        return {"discordant": 0, "chi2": None, "p_approx": None, "significant_95": False}
    chi2 = (abs(b - c) - 1) ** 2 / n  # continuity-corrected
    # two-sided p from chi2 with 1 df, via the normal survival function
    p = math.erfc(math.sqrt(chi2 / 2)) if chi2 >= 0 else 1.0
    return {
        "discordant": n,
        "b_first_only": b,
        "c_second_only": c,
        "chi2": round(chi2, 4),
        "p_approx": round(p, 5),
        "significant_95": p < 0.05,
    }


def load(pattern: str) -> dict[str, dict]:
    """atom_id -> per_atom record, keyed so models can be aligned pairwise."""
    matches = sorted(glob.glob(str(ROOT / pattern)))
    if not matches:
        return {}
    data = json.loads(Path(matches[-1]).read_text())
    out: dict[str, dict] = {}
    for row in data.get("results", []):
        for atom_id, atom in (row.get("per_atom") or {}).items():
            out[atom_id] = atom
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/campaign/student_gap_v2.json")
    args = ap.parse_args()

    models = {name: load(pat) for name, pat in MODELS.items()}
    missing = [n for n, d in models.items() if not d]
    if missing:
        print(f"missing artifacts for: {missing}")
        return 1

    per_model: dict[str, dict] = {}
    for name, atoms in models.items():
        entry: dict = {"n_atoms": len(atoms), "axes": {}}
        for axis, fn in AXES.items():
            vals = [fn(a) for a in atoms.values()]
            k, n = sum(1 for v in vals if v), len(vals)
            entry["axes"][axis] = {
                "numerator": k, "denominator": n,
                "rate": round(k / n, 4) if n else None,
                "wilson95": wilson(k, n),
                "eligible_unit": "atom", "aggregation_level": "suite",
            }
        per_model[name] = entry

    # Paired comparisons on the atom ids present in BOTH models.
    comparisons: dict[str, dict] = {}
    for ref in ("qwen38_27b", "qwen3_32b_awq_managed_ref"):
        key = f"student_a_vs_{ref}"
        shared = sorted(set(models["student_a"]) & set(models[ref]))
        block: dict = {"paired_atoms": len(shared), "axes": {}}
        for axis, fn in AXES.items():
            b = c = 0
            for aid in shared:
                s, r = fn(models["student_a"][aid]), fn(models[ref][aid])
                if s and not r:
                    b += 1
                elif r and not s:
                    c += 1
            s_rate = per_model["student_a"]["axes"][axis]["rate"]
            r_rate = per_model[ref]["axes"][axis]["rate"]
            block["axes"][axis] = {
                "student_rate": s_rate,
                "reference_rate": r_rate,
                "delta": round((s_rate or 0) - (r_rate or 0), 4),
                "mcnemar": mcnemar(b, c),
            }
        comparisons[key] = block

    real_gaps = {
        k: [a for a, d in v["axes"].items()
            if d["mcnemar"]["significant_95"] and d["delta"] < 0]
        for k, v in comparisons.items()
    }

    summary = {
        "schema": "nano.campaign.student_gap.v2",
        "timestamp": datetime.now(UTC).isoformat(),
        "design": "PAIRED — all models scored the same atoms; McNemar on discordant pairs",
        "supersedes": "student_gap_v1 (counts averaged as rates; clamped; hardcoded reference)",
        "metric_contract": ["numerator", "denominator", "rate", "eligible_unit", "aggregation_level"],
        "per_model": per_model,
        "comparisons": comparisons,
        "statistically_established_deficits": real_gaps,
        "adaptation_verdict": (
            "JUSTIFIED" if any(real_gaps.values()) else "NOT_JUSTIFIED_BY_THIS_EVIDENCE"
        ),
        "note": (
            "Only axes where the student is significantly WORSE than a reference "
            "count as a capability gap. A gap that is not statistically established "
            "does not unlock paid adaptation."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{'axis':24s} {'student':>9s} {'qwen3.8':>9s} {'32b-awq':>9s}")
    for axis in AXES:
        s = per_model["student_a"]["axes"][axis]["rate"]
        q = per_model["qwen38_27b"]["axes"][axis]["rate"]
        m = per_model["qwen3_32b_awq_managed_ref"]["axes"][axis]["rate"]
        print(f"  {axis:22s} {s:>9.4f} {q:>9.4f} {m:>9.4f}")
    print("\nPAIRED (McNemar, significant at 95%):")
    for key, block in comparisons.items():
        print(f"  {key}  (paired atoms={block['paired_atoms']})")
        for axis, d in block["axes"].items():
            mc = d["mcnemar"]
            if mc["significant_95"]:
                direction = "WORSE" if d["delta"] < 0 else "better"
                print(f"    {axis:22s} delta={d['delta']:+.4f} {direction:6s} "
                      f"discordant={mc['discordant']} p={mc['p_approx']}")
    print(f"\nestablished deficits: {real_gaps}")
    print(f"adaptation verdict:  {summary['adaptation_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
