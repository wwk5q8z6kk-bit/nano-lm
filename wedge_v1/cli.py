"""CLI for Nano Runtime wedge slice (classical + E-class only)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wedge_v1.runtime import ask, scan, find_spans, DEFAULT_CORPUS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="wedge_v1",
        description="Local verified research Q&A — classical + E-class solvers (no LM).",
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"Local document folder (default: {DEFAULT_CORPUS})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ask_p = sub.add_parser("ask", help="Ask a question; returns span-supported claims or ABSTAIN")
    ask_p.add_argument("query", nargs="+", help="Question text")

    find_p = sub.add_parser("find", help="Exact substring locate with spans")
    find_p.add_argument("needle", nargs="+", help="Exact text to find")

    sub.add_parser("scan", help="Inventory extract + contradictions over corpus")
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
        print("WEDGE_V1_SMOKE_OK")
        return 0
    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
