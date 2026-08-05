"""Faithful report projection of one authoritative Verified Ask result."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _evidence_spans(claims: list) -> list[dict]:
    """Project report spans only from evidence already bound to public claims."""
    spans = []
    for claim in claims:
        if not isinstance(claim, dict) or not claim.get("claim_id"):
            continue
        for evidence in claim.get("evidence") or []:
            if not isinstance(evidence, dict):
                continue
            spans.append(
                {
                    "claim_id": claim["claim_id"],
                    **{
                        key: evidence.get(key)
                        for key in (
                            "atom_id",
                            "doc_id",
                            "doc_digest",
                            "start",
                            "end",
                            "text",
                            "relation",
                        )
                    },
                }
            )
    return spans


def build_report(
    query: str,
    corpus_dir: Path | None = None,
    *,
    doc_ids: list[str] | None = None,
) -> dict:
    from wedge_v1.runtime import ask

    t0 = time.perf_counter()
    ask_out = ask(query, corpus_dir=corpus_dir, doc_ids=doc_ids)
    report = deepcopy(ask_out)
    status = report.get("answer_status", "ABSTAIN")
    abstain_reason = None
    if status == "ABSTAIN":
        abstain_reason = report.get("note") or (
            (report.get("unsupported") or ["unsupported"])[0]
        )
    elif status == "NO_CORPUS":
        abstain_reason = "corpus empty or missing"
    report["product"] = "verified_ask"
    report["solver"] = list(report.get("solver_path") or [])
    report["evidence_spans"] = _evidence_spans(report.get("claims") or [])
    report["latency_ms"] = int(round((time.perf_counter() - t0) * 1000))
    report["abstain_reason"] = abstain_reason
    return report




def format_report_md(report: dict, title: str | None = None) -> str:
    """Human-readable claim report (report UX)."""
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
    p = argparse.ArgumentParser(description="Verified Ask JSON claim report")
    p.add_argument("query", nargs="+", help="Question text")
    p.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "wedge_v1" / "data" / "corpus",
        help="Local document folder",
    )
    p.add_argument("--doc", dest="doc_ids", action="append", default=None)
    p.add_argument("-o", "--output", type=Path, help="Write report to path")
    p.add_argument("--format", choices=["json", "md"], default="json")
    args = p.parse_args(argv)
    report = build_report(" ".join(args.query), corpus_dir=args.corpus, doc_ids=args.doc_ids)
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
