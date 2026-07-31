"""Regression pins for wedge_v1 runtime (no LM)."""
from __future__ import annotations

from pathlib import Path

from wedge_v1.classical.bm25 import top_paragraphs, tokenize
from wedge_v1.ingest import corpus_stats, load_corpus
from wedge_v1.runtime import DEFAULT_CORPUS, ask, find_spans, scan


def test_ttl_supported():
    r = ask("How long before cached entries expire?")
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED"}
    assert "query" in r
    assert "300" in str(r).lower()


def test_oos_abstain():
    r = ask("What is the clinical accuracy of NanoScribe in hospitals?")
    assert r["answer_status"] == "ABSTAIN"
    assert r["claims"] == []


def test_empty_corpus():
    r = ask("anything", corpus_dir=Path("/tmp/empty_wedge_test_xyz"))
    assert r["answer_status"] == "NO_CORPUS"


def test_scan_docs():
    r = scan()
    assert r["answer_status"] == "SUPPORTED"
    assert r["n_docs"] >= 1


def test_find_ttl_phrase():
    r = find_spans("TTL as 300 seconds")
    assert r["answer_status"] == "SUPPORTED"
    assert r["n_hits"] >= 1


def test_bm25_hits_ttl_doc():
    docs = {
        "a": "Unrelated chemistry notes about sodium chloride solutions.",
        "b": "We define cache TTL as 300 seconds for invalidation.",
    }
    hits = top_paragraphs(docs, "How long before cached entries expire?", k=3)
    assert hits
    assert hits[0]["doc_id"] == "b"
    assert tokenize("TTL seconds")


def test_bm25_span_supported():
    r = ask("cache TTL seconds", corpus_dir=DEFAULT_CORPUS)
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED"}


def test_ingest_md_corpus():
    docs = load_corpus(DEFAULT_CORPUS)
    assert len(docs) >= 1
    st = corpus_stats(DEFAULT_CORPUS)
    assert st["n_docs"] == len(docs)
    assert st["n_chars"] > 0


def test_ingest_pdf_fixture():
    try:
        import pypdf  # noqa: F401
    except Exception:
        return
    fix = Path(__file__).resolve().parent / "data" / "fixtures"
    if not fix.is_dir():
        return
    docs = load_corpus(fix)
    assert "plain" in docs


def test_report_build():
    from frontier.verified_ask_report import build_report, format_report_md

    r = build_report("How long before cached entries expire?", corpus_dir=DEFAULT_CORPUS)
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED", "ABSTAIN"}
    md = format_report_md(r)
    assert "**Status:**" in md


if __name__ == "__main__":
    test_ttl_supported()
    test_oos_abstain()
    test_empty_corpus()
    test_scan_docs()
    test_find_ttl_phrase()
    test_bm25_hits_ttl_doc()
    test_bm25_span_supported()
    test_ingest_md_corpus()
    test_ingest_pdf_fixture()
    test_report_build()
    print("WEDGE_V1_SMOKE_OK")
