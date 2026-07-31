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

    # Prefer ask()-computed query-relevant contradictions + banner when present
    ask_nearby = ask_out.get("contradictions_nearby") or []
    if ask_nearby:
        nearby = ask_nearby
    banner = ask_out.get("contradiction_banner")
    if not banner and nearby:
        kinds = sorted({
            (n.get("kind") or n.get("task_id") or "flag")
            for n in nearby
            if isinstance(n, dict)
        })
        banner = f"query-relevant contradictions: {', '.join(str(k) for k in kinds)}"
        if status == "SUPPORTED" and ask_nearby:
            status = "CONTRADICTED"

    return {
        "product": "verified_ask",
        "mandate": "BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1",
        "query": query,
        "corpus_dir": ask_out.get("corpus_dir"),
        "answer_status": status,
        "claims": claims,
        "evidence_spans": evidence_spans,
        "contradictions_nearby": nearby[:12],
        "contradictions_corpus": ask_out.get("contradictions_corpus") or [],
        "contradiction_banner": banner,
        "solver": ask_out.get("solver_path") or [],
        "solver_path": ask_out.get("solver_path") or [],
        "latency_ms": latency_ms,
        "abstain_reason": abstain_reason,
        "n_docs": ask_out.get("n_docs"),
        "lm_invoked": False,
    }




def format_report_md(report: dict, title: str | None = None) -> str:
    """Human-readable claim report (frontier UX)."""
    # Prefer shared runtime formatter when present (keeps CLI/report identical).
    try:
        from wedge_v1.runtime import format_report_md as _rt_md

        return _rt_md(report, title=title or "Verified Ask")
    except Exception:
        pass
    status = report.get("answer_status", "?")
    lines = [
        f"# {title or 'Verified Ask'}",
        "",
        f"**Status:** {status}",
    ]
    banner = report.get("contradiction_banner")
    if banner:
        lines += ["", f"> **Contradiction banner:** {banner}"]
    lines += [
        f"**Query:** {report.get('query', '')}",
        f"**Corpus:** {report.get('corpus_dir', '')}",
        f"**Docs:** {report.get('n_docs', 0)}",
        f"**Latency_ms:** {report.get('latency_ms', '')}",
        "",
        "## Claims",
    ]
    claims = report.get("claims") or []
    if not claims:
        lines.append("_none_")
    for c in claims:
        ev = c.get("evidence") or []
        span = ev[0].get("text", "") if ev else ""
        lines.append(f"- `{c.get('task_id')}` {c.get('value')} — _{span}_")
    nearby = report.get("contradictions_nearby") or []
    if nearby:
        lines += ["", "## Contradictions nearby"]
        for c in nearby[:8]:
            lines.append(f"- {c.get('doc_id')}: {c.get('value')} ({c.get('status')})")
    if report.get("abstain_reason"):
        lines += ["", f"**Abstain:** {report['abstain_reason']}"]
    lines.append("")
    return "\n".join(lines)

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verified Ask JSON claim report (frontier)")
    p.add_argument("query", nargs="+", help="Question text")
    p.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "wedge_v1" / "data" / "corpus",
        help="Local document folder",
    )
    p.add_argument("-o", "--output", type=Path, help="Write report to path")
    p.add_argument("--format", choices=["json", "md"], default="json")
    args = p.parse_args(argv)
    report = build_report(" ".join(args.query), corpus_dir=args.corpus)
    if args.format == "md":
        md = format_report_md(report)
        text = md if md.endswith("\n") else md + "\n"
    else:
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
