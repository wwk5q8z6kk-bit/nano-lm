"""CLI for Nano Runtime wedge slice (classical + E-class only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wedge_v1.runtime import ask, scan, find_spans, format_report_md, DEFAULT_CORPUS, load_corpus


def _corpus(args) -> Path | None:
    return args.corpus if args.corpus is not None else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="wedge_v1",
        description="Local verified research Q&A — classical + E-class solvers (no LM).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ask_p = sub.add_parser("ask", help="Ask a question; returns span-supported claims or ABSTAIN")
    ask_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help=f"Local document folder (default: {DEFAULT_CORPUS})")
    ask_p.add_argument("query", nargs="+", help="Question text")

    find_p = sub.add_parser("find", help="Exact substring locate with spans")
    find_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    find_p.add_argument("needle", nargs="+", help="Exact text to find")

    scan_p = sub.add_parser("scan", help="Inventory extract + contradictions over corpus")
    scan_p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)

    sub.add_parser("dogfood", help="Score dogfood tasks on papers/ corpus")
    sub.add_parser("smoke", help="Run runtime regression pins")

    ingest_p = sub.add_parser("ingest", help="Index a folder (md/txt/pdf→text); writes local manifest")
    ingest_p.add_argument("src", type=Path, help="Source folder")
    ingest_p.add_argument("--out", type=Path, default=None, help="Manifest path (default: <src>/.wedge_manifest.json)")

    rep = sub.add_parser("report", help="Verified Ask JSON claim report (frontier)")
    rep.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    rep.add_argument("query", nargs="+", help="Question text")
    rep.add_argument("-o", "--output", type=Path, default=None)

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
    if args.cmd == "ingest":
        docs = load_corpus(args.src)
        man = {
            "schema": "nano-lm.wedge_v1.ingest_manifest.v1",
            "src": str(args.src.resolve()),
            "n_docs": len(docs),
            "doc_ids": sorted(docs.keys()),
            "note": "Local index only; not Layer-1 evidence.",
        }
        out_path = args.out or (args.src / ".wedge_manifest.json")
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
        smoke.test_bm25_span_supported()
        print("WEDGE_V1_SMOKE_OK", file=sys.stderr)
        return 0
    if args.cmd == "report":
        from frontier.verified_ask_report import build_report

        out = build_report(" ".join(args.query), corpus_dir=args.corpus)
        text = json.dumps(out, indent=2) + "\n"
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            sys.stdout.write(text)
        return 0 if out.get("answer_status") != "NO_CORPUS" else 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
