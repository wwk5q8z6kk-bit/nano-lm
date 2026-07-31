"""Corpus contact protocol — labeled usefulness probe (no LM).

Corpus classes: SYNTHETIC_MINI | PAPERS_DOGFOOD | OWNER_PRIVATE
Not Layer-1 evidence. Product evaluation under ACTIVE_MANDATE.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from wedge_v1.ingest import corpus_stats
from wedge_v1.runtime import DEFAULT_CORPUS, ask, compare, find_spans, load_corpus, scan

PROBES = [
    ("ask", "How long before cached entries expire?"),
    ("ask", "What is the clinical accuracy of NanoScribe in hospitals?"),
    ("find", "0.925"),
    ("ask", "What is TTL?"),
    ("compare", "TTL"),
]


def run_contact(
    corpus: Path,
    *,
    corpus_class: str,
    useful_sentence: str | None = None,
    not_useful_sentence: str | None = None,
) -> dict:
    docs = load_corpus(corpus)
    stats = corpus_stats(corpus)
    rows = []
    for kind, text in PROBES:
        if kind == "ask":
            out = ask(text, corpus_dir=corpus)
        elif kind == "find":
            out = find_spans(text, corpus_dir=corpus)
        else:
            out = compare(text, corpus_dir=corpus)
        rows.append(
            {
                "kind": kind,
                "text": text,
                "answer_status": out.get("answer_status"),
                "n_claims": len(out.get("claims") or []),
                "note": out.get("note") or out.get("unsupported"),
            }
        )
    scan_out = scan(corpus_dir=corpus)
    n = len(docs)
    supported = sum(1 for r in rows if r["answer_status"] in {"SUPPORTED", "CONTRADICTED"})
    abstain = sum(1 for r in rows if r["answer_status"] == "ABSTAIN")
    return {
        "schema": "nano-lm.wedge_v1.corpus_contact.v1",
        "corpus_class": corpus_class,
        "corpus": str(Path(corpus).resolve()),
        "n_docs": n,
        "n_chars": stats.get("n_chars"),
        "probes": rows,
        "scan_status": scan_out.get("answer_status"),
        "scan_n_claims": len(scan_out.get("claims") or []),
        "summary": {
            "n_probes": len(rows),
            "n_supported_or_contradicted": supported,
            "n_abstain": abstain,
            "meets_n20": n >= 20,
        },
        "useful_sentence": useful_sentence,
        "not_useful_sentence": not_useful_sentence,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": "Product contact only. Not Layer-1. OWNER_PRIVATE requires owner folder path.",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Labeled corpus contact probe (no LM)")
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument(
        "--class",
        dest="corpus_class",
        choices=["SYNTHETIC_MINI", "PAPERS_DOGFOOD", "OWNER_PRIVATE"],
        required=True,
    )
    p.add_argument("--useful", default=None, help="One sentence: why this was useful")
    p.add_argument("--not-useful", dest="not_useful", default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args(argv)
    out = run_contact(
        args.corpus,
        corpus_class=args.corpus_class,
        useful_sentence=args.useful,
        not_useful_sentence=args.not_useful,
    )
    path = args.output or Path("wedge_v1/results_corpus_contact.json")
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"WROTE {path}", flush=True)
    return 0 if out["n_docs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
