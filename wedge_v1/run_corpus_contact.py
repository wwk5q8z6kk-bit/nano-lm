"""Corpus contact protocol — labeled usefulness probe (no LM).

Corpus classes: SYNTHETIC_MINI | PAPERS_DOGFOOD | OWNER_PRIVATE
Not Layer-1 evidence. This evaluates the internal Wedge v1 pipeline; it is not
validation of Nano's scribe intelligence.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from wedge_v1.ingest import corpus_stats
from wedge_v1.private_output import REPO_ROOT, require_private_output
from wedge_v1.runtime import DEFAULT_CORPUS, ask, compare, find_spans, load_corpus, scan

ROOT = Path(__file__).resolve().parent
PUBLIC_CONTACT_OUTPUT = ROOT / "results_corpus_contact.json"
OWNER_CONTACT_OUTPUT = ROOT / "results_owner_corpus_contact.json"
OWNER_CORPUS_DIR = ROOT / "data" / "owner_corpus"
PAPERS_CORPUS_DIR = REPO_ROOT / "papers"

PROBES = [
    ("ask", "How long before cached entries expire?"),
    ("ask", "What is the clinical accuracy of NanoScribe in hospitals?"),
    ("find", "0.925"),
    ("ask", "What is TTL?"),
    ("compare", "TTL"),
]


def resolve_contact_corpus(corpus: Path | None, corpus_class: str) -> Path:
    """Require a real, explicit corpus before applying the private label."""
    if corpus_class == "OWNER_PRIVATE" and corpus is None:
        raise ValueError("OWNER_PRIVATE requires an explicit --corpus path")
    resolved = Path(corpus if corpus is not None else DEFAULT_CORPUS).expanduser()
    if (
        corpus_class == "OWNER_PRIVATE"
        and resolved.resolve() == Path(DEFAULT_CORPUS).resolve()
    ):
        raise ValueError("OWNER_PRIVATE cannot use the synthetic default corpus")
    return resolved


def _known_owner_corpus(corpus: Path | None) -> bool:
    if corpus is None:
        return False
    try:
        Path(corpus).expanduser().resolve().relative_to(OWNER_CORPUS_DIR.resolve())
    except (OSError, ValueError):
        return False
    return True


def contact_is_private(corpus: Path | None, corpus_class: str) -> bool:
    """Derive storage sensitivity from both the declared class and corpus path."""
    if corpus_class == "OWNER_PRIVATE":
        return True
    if corpus is None:
        return False
    resolved = Path(corpus).expanduser().resolve()
    if _known_owner_corpus(resolved):
        return True
    if corpus_class == "SYNTHETIC_MINI" and resolved == DEFAULT_CORPUS.resolve():
        return False
    if corpus_class == "PAPERS_DOGFOOD":
        try:
            resolved.relative_to(PAPERS_CORPUS_DIR.resolve())
            return False
        except ValueError:
            pass
    # Unknown and external corpora are private by default, even if mislabeled.
    return True


def default_contact_output(corpus_class: str, corpus: Path | None = None) -> Path:
    """Keep private contact metadata in the repository's ignored owner namespace."""
    private = contact_is_private(corpus, corpus_class)
    return OWNER_CONTACT_OUTPUT if private else PUBLIC_CONTACT_OUTPUT


def run_contact(
    corpus: Path | None,
    *,
    corpus_class: str,
    useful_sentence: str | None = None,
    not_useful_sentence: str | None = None,
) -> dict:
    corpus = resolve_contact_corpus(corpus, corpus_class)
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
        "storage_class": (
            "OWNER_PRIVATE" if contact_is_private(corpus, corpus_class) else "PUBLIC"
        ),
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
        "note": (
            "Internal Wedge v1 corpus contact only. Not Nano AI validation or "
            "Layer-1 evidence. OWNER_PRIVATE requires owner folder path."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Labeled Wedge v1 corpus contact probe (no LM)")
    p.add_argument("--corpus", type=Path, default=None)
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
    try:
        out = run_contact(
            args.corpus,
            corpus_class=args.corpus_class,
            useful_sentence=args.useful,
            not_useful_sentence=args.not_useful,
        )
    except ValueError as exc:
        p.error(str(exc))
    path = args.output or default_contact_output(
        args.corpus_class,
        Path(out["corpus"]),
    )
    if out["storage_class"] == "OWNER_PRIVATE":
        try:
            require_private_output(path, purpose="private corpus-contact output")
        except ValueError as exc:
            p.error(str(exc))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"WROTE {path}", flush=True)
    return 0 if out["n_docs"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
