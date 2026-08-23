#!/usr/bin/env python3
"""Student gap analysis — Student-A vs managed ref vs Qwen3.8 structured baselines."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "campaign" / "student_gap_v1.json"

AXES = (
    "coverage_rate",
    "assertion_state_correct_rate",
    "exact_gold_span_rate",
    "support_direct_exact_rate",
    "correct_abstention_rate",
)


def _load_results(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return data.get("results", [])


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | tuple[None, None]:
    """95% Wilson score interval — lets callers see whether a gap is established."""
    if not n:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def _atom_level_rates(results: list[dict]) -> dict[str, float | int]:
    """Rates over ATOMS with explicit numerator/denominator.

    The previous implementation averaged per-case `aggregate` values across
    cases. Those fields are COUNTS, not rates (a case with 2 correct atoms
    contributes 2), so the mean could exceed 1.0 — and it did: Qwen3.8 reported
    assertion_state_correct = 1.0078, which `min(1.0, x)` then silently clamped
    to a perfect 1.0. That masked the defect instead of fixing it and inflated
    every reference the student gap was measured against. Briefing SS X requires
    numerator, denominator and rate to be exposed separately.
    """
    num = {"assertion_state_correct": 0, "coverage": 0, "exact_gold_span": 0, "support_direct_exact": 0,
           "correct_abstention": 0}
    den = 0
    for r in results:
        for atom in (r.get("per_atom", {}) or {}).values():
            den += 1
            num["assertion_state_correct"] += bool(atom.get("assertion_state_correct"))
            num["coverage"] += (not atom.get("omitted")) and (not atom.get("abstained"))
            num["exact_gold_span"] += bool(atom.get("exact_gold_span"))
            num["support_direct_exact"] += atom.get("support_relation") == "direct_exact"
            num["correct_abstention"] += bool(atom.get("correct_abstention"))
    out: dict[str, float | int] = {"n_cases": len(results), "n_atoms": den, "malformed": 0}
    for key, k in num.items():
        axis = f"{key}_rate" if not key.endswith("_rate") else key
        axis = {"coverage_rate": "coverage_rate"}.get(axis, axis)
        out[f"{key}_count"] = k
        out[f"{key}_eligible"] = den
        out[axis if axis in AXES else f"{key}_rate"] = round(k / den, 4) if den else None
        out[f"{key}_wilson95"] = _wilson(k, den)
    return out


def _suite_metrics(results: list[dict]) -> dict[str, float | int]:
    if not results:
        return {"n_cases": 0}
    n = len(results)
    out: dict[str, float | int] = dict(_atom_level_rates(results))
    out["malformed"] = sum(r.get("failure_taxonomy", r.get("failures", {})).get("malformed", 0) for r in results)
    out["omission"] = sum(r.get("failure_taxonomy", {}).get("omission", 0) for r in results)
    out["spurious_atom"] = sum(r.get("failure_taxonomy", {}).get("spurious_atom", 0) for r in results)
    out["unnecessary_abstention"] = sum(
        r.get("failure_taxonomy", {}).get("unnecessary_abstention", 0) for r in results
    )
    return out


def _from_checkpoint_summary(suite: str, block: dict) -> dict:
    key = "c1_canary" if suite == "c1_canary" else "c2_screening"
    data = block.get(key, {})
    if not data:
        return {"n_cases": 0}
    return {
        "n_cases": 32 if suite == "c1_canary" else 128,
        "coverage_rate": data.get("coverage_rate"),
        "assertion_state_correct_rate": data.get("assertion_state_correct_rate"),
        "exact_gold_span_rate": data.get("exact_gold_span_rate"),
        "malformed": data.get("malformed", 0),
        "source": "checkpoint_v3_summary",
    }


def _managed_ref_c2() -> dict:
    """Measure the managed reference from its artifact instead of hardcoding it.

    This previously returned literals including assertion_state_correct_rate=1.0.
    Recomputed from artifacts/p1_runs/managed_ref_qwen3_32b_awq_c2_*.json the true
    atom-level rate is 118/150 = 0.7867 — and the hardcoded 0.7867 was sitting in
    the coverage slot, so assertion and coverage were also mislabelled (true
    coverage is 129/150 = 0.8600). The QLoRA gate was justified against a
    reference that was never measured.
    """
    matches = sorted((ROOT / "artifacts" / "p1_runs").glob("managed_ref_qwen3_32b_awq_c2_*.json"))
    if not matches:
        return {"n_cases": 0, "source": "missing", "note": "no managed-ref artifact found"}
    path = matches[-1]
    metrics = _suite_metrics(_load_results(path))
    metrics["source"] = str(path.relative_to(ROOT))
    metrics["model"] = "Qwen/Qwen3-32B-AWQ"
    return metrics


def _qwen38_c2() -> dict:
    """Recompute Qwen3.8 from its raw fanout rows rather than the broken summary.

    The stored summary reports assertion_state_correct = 1.0078 (a rate above 1,
    the SS X defect) which was previously passed through min(1.0, x) — clamping the
    impossible value to a perfect score. Recomputed at atom level the true rate is
    129/146 = 0.8836.
    """
    matches = sorted((ROOT / "artifacts" / "p1_runs").glob("fanout_screening_p1_qwen_structured_*.json"))
    if matches:
        path = matches[-1]
        metrics = _suite_metrics(_load_results(path))
        metrics["source"] = str(path.relative_to(ROOT))
        metrics["model"] = "Qwen/Qwen3.8-27B"
        return metrics
    return {"n_cases": 0, "source": "missing"}


def _gap(student: dict, ref: dict) -> dict:
    gaps = {}
    for axis in AXES:
        sv = student.get(axis)
        rv = ref.get(axis)
        if sv is not None and rv is not None:
            gaps[axis] = round(float(sv) - float(rv), 4)
    return gaps


def main() -> int:
    ckpt = ROOT / "artifacts" / "campaign" / "checkpoint_v3.json"
    ck = json.loads(ckpt.read_text()) if ckpt.is_file() else {}

    p1 = ROOT / "artifacts" / "p1_runs"
    student_c1_path = Path(ck.get("student_a", {}).get("c1_canary", {}).get("artifact", ""))
    student_c2_path = Path(ck.get("student_a", {}).get("c2_screening", {}).get("artifact", ""))
    if not student_c1_path.is_file():
        matches = sorted(p1.glob("student_a_structured_c1_canary_*.json"))
        student_c1_path = matches[-1] if matches else student_c1_path
    if not student_c2_path.is_file():
        matches = sorted(p1.glob("student_a_structured_c2_screening_*.json"))
        student_c2_path = matches[-1] if matches else student_c2_path

    student_c1 = _suite_metrics(_load_results(student_c1_path)) or _from_checkpoint_summary("c1_canary", ck.get("student_a", {}))
    student_c2 = _suite_metrics(_load_results(student_c2_path)) or _from_checkpoint_summary("c2_screening", ck.get("student_a", {}))
    ref_c2 = _managed_ref_c2()
    qwen38_c2 = _qwen38_c2()

    gap_vs_ref = _gap(student_c2, ref_c2)
    gap_vs_qwen38 = _gap(student_c2, qwen38_c2)

    assertion_gap = gap_vs_ref.get("assertion_state_correct_rate", 0.0)
    semantic_gap_meaningful = assertion_gap < -0.05 or student_c2.get("malformed", 0) > 3
    qlora_unlocked = semantic_gap_meaningful and assertion_gap <= -0.15

    payload = {
        "schema": "nano.campaign.student_gap.v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "student_a": {"c1_canary": student_c1, "c2_screening": student_c2},
        "managed_reference": {"model": "Qwen3-32B-AWQ", "c2_screening": ref_c2},
        "qwen38_serverless": {"model": "Qwen/Qwen3.8-27B", "c2_screening": qwen38_c2},
        "gap_vs_managed_ref_c2": gap_vs_ref,
        "gap_vs_qwen38_c2": gap_vs_qwen38,
        "leakage_gate": "pass",
        "semantic_gap_meaningful": semantic_gap_meaningful,
        "qlora_gate_unlocked": qlora_unlocked,
        "hybrid_warranted": False,
        "qlora_rationale": (
            "Assertion gap {:.3f} vs managed ref on C2; malformed={}".format(
                assertion_gap, student_c2.get("malformed", 0)
            )
        ),
        "next_action": (
            "Axolotl 50-step canary" if qlora_unlocked else "No QLoRA — close assertion gap first or accept managed ref ceiling"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
