"""Regression pins for wedge_v1 runtime (no LM)."""
from __future__ import annotations

from pathlib import Path

from wedge_v1.runtime import ask, scan, find_spans, compare, DEFAULT_CORPUS
from wedge_v1.ingest import load_corpus, corpus_stats
from wedge_v1.classical.bm25 import top_paragraphs, tokenize


def test_ttl_supported():
    r = ask("How long before cached entries expire?")
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED"}
    assert "query" in r and "latency_s" in r
    blob = str(r).lower()
    assert "300" in blob


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


def test_bm25_span_supported():
    r = ask("cache TTL seconds", corpus_dir=DEFAULT_CORPUS)
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED"}
    assert "latency_s" in r
    assert "query" in r
    assert any(
        c.get("task_id") == "BM25"
        or "bm25" in str(c.get("notes", "")).lower()
        or c.get("evidence")
        for c in r["claims"]
    )


def test_ask_schema_fields():
    r = ask("How long before cached entries expire?")
    for k in ("query", "corpus_dir", "claims", "answer_status", "latency_s"):
        assert k in r


def test_find_ttl_phrase():
    r = find_spans("TTL as 300 seconds")
    assert r["answer_status"] == "SUPPORTED"
    assert r["n_hits"] >= 1


def test_ask_surfaces_contradictions_on_synthetic():
    r = ask("What is the metformin dose?", corpus_dir=DEFAULT_CORPUS)
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED"}
    assert isinstance(r.get("contradictions_nearby"), list)
    assert any(c.get("kind") == "numeric_dose" for c in r["contradictions_nearby"])


def test_bm25_hits_ttl_doc():
    docs = {
        "a": "Unrelated chemistry notes about sodium chloride solutions.",
        "b": "We define cache TTL as 300 seconds for invalidation.",
    }
    hits = top_paragraphs(docs, "How long before cached entries expire?", k=3)
    assert hits
    assert hits[0]["doc_id"] == "b"
    assert tokenize("TTL seconds")


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
        return  # optional dep
    fix = Path(__file__).resolve().parent / "data" / "fixtures"
    docs = load_corpus(fix)
    assert "plain" in docs
    assert "ttl_note" in docs
    assert "300" in docs["ttl_note"]


def test_compare_ttl_contradicted():
    r = compare("TTL", corpus_dir=DEFAULT_CORPUS)
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED", "ABSTAIN"}
    assert "claims" in r or "hits" in r or "docs" in r


def test_report_ask_markdown():
    from wedge_v1.runtime import format_report_md

    payload = ask("How long before cached entries expire?")
    md = format_report_md(payload, title="ask smoke")
    assert "**Status:**" in md
    assert payload.get("answer_status") in {"SUPPORTED", "CONTRADICTED", "ABSTAIN", "NO_CORPUS"}


if __name__ == "__main__":
    test_ttl_supported()
    test_oos_abstain()
    test_empty_corpus()
    test_scan_docs()
    test_find_ttl_phrase()
    test_ask_surfaces_contradictions_on_synthetic()
    test_bm25_hits_ttl_doc()
    test_ingest_md_corpus()
    test_ingest_pdf_fixture()
    test_compare_ttl_contradicted()
    test_bm25_span_supported()
    test_ask_schema_fields()
    test_report_ask_markdown()
    print("WEDGE_V1_SMOKE_OK")
