#!/usr/bin/env python3
"""NANO-CLIN-001 runner — emits the artifact set required by D-NANO-2026-08-25 §9.

Synthetic, non-PHI fixtures only. No model call, no paid compute, no training.

Usage:
    .venv/bin/python scripts/run_nano_clin_001.py [--out DIR]
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

from nano.fixtures import ALL_FIXTURES            # noqa: E402
from nano.pipeline import baseline_a, candidate_b  # noqa: E402

ABSENCE = {"not_found", "unavailable"}


def _enc(o):
    if is_dataclass(o):
        return {k: _enc(v) for k, v in asdict(o).items()}
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (list, tuple)):
        return [_enc(x) for x in o]
    if isinstance(o, dict):
        return {k: _enc(v) for k, v in o.items()}
    return o


def _jsonl(path: Path, rows) -> None:
    path.write_text("\n".join(json.dumps(_enc(r), ensure_ascii=False) for r in rows) + "\n")


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(_enc(obj), indent=2, ensure_ascii=False) + "\n")


def metrics(claims: list[dict]) -> dict:
    """Provenance coverage over CONTENT claims.

    Absence claims (NOT_FOUND / UNAVAILABLE) are excluded from the denominator
    rather than scored as uncited: there is no span to cite for something that
    is not in the record. They are reported separately so the exclusion is
    visible instead of silently improving the number.
    """
    content = [c for c in claims if c.get("epistemic_status") not in ABSENCE]
    absence = [c for c in claims if c.get("epistemic_status") in ABSENCE]
    cited = sum(1 for c in content if c["evidence_span_ids"])
    return {
        "claims_total": len(claims),
        "content_claims": len(content),
        "absence_claims": len(absence),
        "provenance_coverage": round(cited / len(content), 4) if content else 0.0,
        "unsupported_claims": sum(1 for c in content if not c["evidence_span_ids"]),
        "unsupported_rate": round(
            sum(1 for c in content if not c["evidence_span_ids"]) / len(content), 4
        ) if content else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "artifacts/nano_clin_001")
    args = ap.parse_args()

    results = []
    for fx in ALL_FIXTURES:
        d = args.out / fx.fixture_id
        d.mkdir(parents=True, exist_ok=True)

        note_a, claims_a = baseline_a(fx)
        r = candidate_b(fx)

        _json(d / "source_manifest.json", r["source"])
        _jsonl(d / "evidence_spans.jsonl", r["spans"])
        _jsonl(d / "assertions.jsonl", r["assertions"])
        _jsonl(d / "events.jsonl", r["events"])
        _json(d / "conflicts.json", r["conflicts"])
        _json(d / "knowledge_gaps.json", r["gaps"])
        _json(d / "patient_state.json", r["state"])
        _json(d / "timeline.json", [
            {"event_id": e.event_id, "type": e.event_type,
             "precision": e.temporal.precision.value,
             "event_time": e.temporal.event_time} for e in r["events"]])
        (d / "encounter_note.md").write_text(r["note"] + "\n")
        (d / "baseline_note.md").write_text(note_a + "\n")
        (d / "change_summary.md").write_text(
            f"# Change summary — {fx.fixture_id}\n\n"
            f"- ledger version: {r['ledger'].version}\n"
            f"- state snapshot: {r['state'].snapshot_id}\n"
            f"- new conflicts: {len(r['conflicts'])}\n"
            f"- new knowledge gaps: {len(r['gaps'])}\n")
        _json(d / "verification_receipt.json", r["receipt"])

        ma, mb = metrics(claims_a), metrics(r["claims"])
        results.append({
            "fixture": fx.fixture_id,
            "baseline_a": ma,
            "candidate_b": mb,
            "conflicts_detected": len(r["conflicts"]),
            "conflicts_gold": len(fx.gold_conflicts),
            "gaps_detected": len(r["gaps"]),
            "gaps_gold": len(fx.gold_gaps),
            "forbidden_assertions_in_baseline": sum(
                1 for f in fx.must_not_assert if f.lower() in note_a.lower()),
            "forbidden_assertions_in_candidate": sum(
                1 for f in fx.must_not_assert if f.lower() in r["note"].lower()),
            "state_rebuildable": r["state"].ledger_hash == r["ledger"].ledger_hash(),
        })

    summary = {
        "experiment": "NANO-CLIN-001",
        "decision": "D-NANO-2026-08-25",
        "data": "synthetic_non_phi",
        "model_calls": 0,
        "paid_compute": False,
        "per_fixture": results,
        "aggregate": {
            "baseline_provenance_coverage": round(sum(
                x["baseline_a"]["provenance_coverage"] for x in results) / len(results), 4),
            "candidate_provenance_coverage": round(sum(
                x["candidate_b"]["provenance_coverage"] for x in results) / len(results), 4),
            "conflict_recall": f"{sum(x['conflicts_detected'] for x in results)}"
                               f"/{sum(x['conflicts_gold'] for x in results)}",
            "gap_recall": f"{sum(x['gaps_detected'] for x in results)}"
                          f"/{sum(x['gaps_gold'] for x in results)}",
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    _json(args.out / "benchmark_results.json", summary)

    agg = summary["aggregate"]
    print(f"{'fixture':18s} {'A prov':>8s} {'B prov':>8s} {'conflicts':>12s} {'gaps':>8s}")
    for x in results:
        print(f"  {x['fixture']:16s} "
              f"{x['baseline_a']['provenance_coverage']:>8.2f} "
              f"{x['candidate_b']['provenance_coverage']:>8.2f} "
              f"{str(x['conflicts_detected'])+'/'+str(x['conflicts_gold']):>12s} "
              f"{str(x['gaps_detected'])+'/'+str(x['gaps_gold']):>8s}")
    print(f"\naggregate  A={agg['baseline_provenance_coverage']} "
          f"B={agg['candidate_provenance_coverage']} "
          f"conflicts={agg['conflict_recall']} gaps={agg['gap_recall']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
