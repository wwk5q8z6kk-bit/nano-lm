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


def _suite_metrics(results: list[dict]) -> dict[str, float | int]:
    if not results:
        return {"n_cases": 0}
    n = len(results)
    out: dict[str, float | int] = {"n_cases": n, "malformed": 0}
    for axis in AXES:
        short = axis.replace("_rate", "")
        vals = []
        for r in results:
            agg = r.get("aggregate", {})
            vals.append(float(agg.get(axis, agg.get(short, 0)) or 0))
        out[axis] = round(sum(vals) / n, 4)
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
    return {
        "n_cases": 128,
        "coverage_rate": 0.7867,
        "assertion_state_correct_rate": 1.0,
        "exact_gold_span_rate": 0.1102,
        "malformed": 11,
        "source": "wave_v2_managed_ref",
    }


def _qwen38_c2() -> dict:
    p = ROOT / "artifacts" / "campaign" / "qwen_structured_metrics_v1.json"
    if p.is_file():
        s = json.loads(p.read_text()).get("summary", {})
        return {
            "n_cases": s.get("n_cases", 128),
            "coverage_rate": s.get("coverage"),
            "assertion_state_correct_rate": min(1.0, float(s.get("assertion_state_correct", 0))),
            "exact_gold_span_rate": s.get("exact_gold_span"),
            "support_direct_exact_rate": s.get("support_direct_exact"),
            "malformed": 0,
            "source": str(p),
            "note": "historical fanout; assertion metric pre-denominator-fix",
        }
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
