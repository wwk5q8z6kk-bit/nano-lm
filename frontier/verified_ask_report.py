"""Verified Ask claim report — Active Frontier vertical slice.

Combines ask() + contradiction-relevant scan flags into one JSON report.
Does not touch Evidence Core, tags, or ledger rows. No LM.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wedge_v1.runtime import ask, scan


CONTRA_STATUSES = {"DISPUTED", "CONFLICT", "COLLISION"}
CONTRA_NOTES = ("conflict", "contradict", "collision", "disputed", "dose_change")


def _is_contradiction_claim(c: dict) -> bool:
    st = str(c.get("status", "")).upper()
    if st in CONTRA_STATUSES or st == "DISPUTED":
        return True
    notes = str(c.get("notes", "")).lower()
    task = str(c.get("task_id", "")).lower()
    blob = notes + " " + task
    return any(k in blob for k in CONTRA_NOTES)


def build_report(query: str, corpus_dir: Path | None = None) -> dict:
    t0 = time.perf_counter()
    ask_out = ask(query, corpus_dir=corpus_dir)
    scan_out = scan(corpus_dir=corpus_dir)

    claims = ask_out.get("claims") or []
    evidence_spans = []
    for c in claims:
        for e in c.get("evidence") or []:
            evidence_spans.append(
                {
                    "doc_id": c.get("doc_id"),
                    "task_id": c.get("task_id"),
                    **{k: e[k] for k in e},
                }
            )

    contradictions = [c for c in (scan_out.get("claims") or []) if _is_contradiction_claim(c)]
    ask_docs = {c.get("doc_id") for c in claims if c.get("doc_id")}
    nearby = [c for c in contradictions if c.get("doc_id") in ask_docs] or contradictions[:8]

    latency_ms = int(round((time.perf_counter() - t0) * 1000))
    status = ask_out.get("answer_status", "ABSTAIN")
    abstain_reason = None
    if status == "ABSTAIN":
        abstain_reason = ask_out.get("note") or (
            (ask_out.get("unsupported") or ["unsupported"])[0]
        )
    elif status == "NO_CORPUS":
        abstain_reason = "corpus empty or missing"

    return {
        "product": "verified_ask",
        "mandate": "BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1",
        "query": query,
        "corpus_dir": ask_out.get("corpus_dir"),
        "answer_status": status,
        "claims": claims,
        "evidence_spans": evidence_spans,
        "contradictions_nearby": nearby[:12],
        "solver": ask_out.get("solver_path") or [],
        "latency_ms": latency_ms,
        "abstain_reason": abstain_reason,
        "n_docs": ask_out.get("n_docs"),
        "lm_invoked": False,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verified Ask JSON claim report (frontier)")
    p.add_argument("query", nargs="+", help="Question text")
    p.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "wedge_v1" / "data" / "corpus",
        help="Local document folder",
    )
    p.add_argument("-o", "--output", type=Path, help="Write JSON to path")
    args = p.parse_args(argv)
    report = build_report(" ".join(args.query), corpus_dir=args.corpus)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if report["answer_status"] == "NO_CORPUS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
