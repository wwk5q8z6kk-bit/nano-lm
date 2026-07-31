"""CLI for Nano Runtime wedge slice (classical + E-class only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wedge_v1.ingest import corpus_stats
from wedge_v1.runtime import (
    DEFAULT_CORPUS,
    ask,
    compare,
    find_spans,
    format_report_md,
    load_corpus,
    scan,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="wedge_v1",
        description="Local verified research Q&A — classical + E-class solvers (no LM).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ask_p = sub.add_parser("ask", help="Ask a question; returns span-supported claims or ABSTAIN")
    ask_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ask_p.add_argument("query", nargs="+")

    find_p = sub.add_parser("find", help="Exact substring locate with spans")
    find_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    find_p.add_argument("needle", nargs="+")

    scan_p = sub.add_parser("scan", help="Inventory extract + contradictions over corpus")
    scan_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    cmp_p = sub.add_parser("compare", help="Cross-doc term compare; flags numeric disputes")
    cmp_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    cmp_p.add_argument("term", nargs="+")

    ingest_p = sub.add_parser("ingest", help="Index a folder (md/txt/pdf→text); write local manifest")
    ingest_p.add_argument("--corpus", type=Path, default=None, help="Alias for positional src")
    ingest_p.add_argument("src", nargs="?", type=Path, default=None)
    ingest_p.add_argument("--out", type=Path, default=None)

    rep = sub.add_parser("report", help="Markdown (default) or JSON report for ask/find/scan/compare")
    rep.add_argument("kind", choices=["ask", "find", "scan", "compare", "verified"])
    rep.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    rep.add_argument("text", nargs="*", help="Query / needle / term (unused for scan)")
    rep.add_argument("-o", "--output", type=Path, default=None)
    rep.add_argument("--json", action="store_true", help="Emit JSON instead of markdown")

    sub.add_parser("dogfood", help="Score dogfood tasks on papers/ corpus")
    sub.add_parser("smoke", help="Run runtime regression pins")

    args = p.parse_args(argv)

    if args.cmd == "ask":
        out = ask(" ".join(args.query), corpus_dir=args.corpus)
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    if args.cmd == "find":
        out = find_spans(" ".join(args.needle), corpus_dir=args.corpus)
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    if args.cmd == "scan":
        out = scan(corpus_dir=args.corpus)
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    if args.cmd == "compare":
        out = compare(" ".join(args.term), corpus_dir=args.corpus)
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    if args.cmd == "ingest":
        src = args.corpus or args.src or DEFAULT_CORPUS
        docs = load_corpus(src)
        stats = corpus_stats(src)
        man = {
            "schema": "nano-lm.wedge_v1.ingest_manifest.v1",
            "src": str(Path(src).resolve()),
            "n_docs": len(docs),
            "n_chars": stats.get("n_chars"),
            "doc_ids": sorted(docs.keys()),
            "n_pdf_files_on_disk": stats.get("n_pdf_files_on_disk"),
            "pypdf_available": stats.get("pypdf_available"),
            "note": stats.get("note") or "Local index only; not Layer-1 evidence.",
        }
        out_path = args.out or (Path(src) / ".wedge_manifest.json")
        out_path.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
        json.dump(man, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0 if docs else 2

    if args.cmd == "dogfood":
        from wedge_v1.run_dogfood import main as dogfood_main

        dogfood_main()
        return 0

    if args.cmd == "smoke":
        from wedge_v1 import test_runtime_smoke as smoke

        smoke.test_ttl_supported()
        smoke.test_oos_abstain()
        smoke.test_empty_corpus()
        smoke.test_scan_docs()
        smoke.test_find_ttl_phrase()
        smoke.test_bm25_hits_ttl_doc()
        smoke.test_bm25_span_supported()
        smoke.test_ingest_md_corpus()
        smoke.test_ingest_pdf_fixture()
        smoke.test_report_build()
        smoke.test_report_ask_markdown()
        print("WEDGE_V1_SMOKE_OK", file=sys.stderr)
        return 0

    if args.cmd == "report":
        kind = args.kind
        text = " ".join(args.text)
        if kind == "ask":
            out = ask(text, corpus_dir=args.corpus)
            title = f"ask: {text}"
        elif kind == "find":
            out = find_spans(text, corpus_dir=args.corpus)
            title = f"find: {text}"
        elif kind == "scan":
            out = scan(corpus_dir=args.corpus)
            title = "scan"
        elif kind == "compare":
            out = compare(text, corpus_dir=args.corpus)
            title = f"compare: {text}"
        else:
            from frontier.verified_ask_report import build_report

            out = build_report(text, corpus_dir=args.corpus)
            title = f"verified ask: {text}"
        body = (
            json.dumps(out, indent=2) + "\n"
            if args.json
            else format_report_md(out, title=title)
        )
        if args.output:
            args.output.write_text(body, encoding="utf-8")
        else:
            sys.stdout.write(body)
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
