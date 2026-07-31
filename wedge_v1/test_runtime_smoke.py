"""Regression pins for wedge_v1 runtime (no LM)."""
from __future__ import annotations

from pathlib import Path

from wedge_v1.classical.bm25 import top_paragraphs, tokenize
from wedge_v1.ingest import corpus_stats, load_corpus
from wedge_v1.runtime import DEFAULT_CORPUS, ask, compare, find_spans, scan


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


def test_compare_metformin_contradicted():
    r = compare("metformin", corpus_dir=DEFAULT_CORPUS)
    assert r["answer_status"] == "CONTRADICTED"
    assert r["n_docs_hit"] >= 2
    blob = str(r)
    assert "500" in blob and "850" in blob


def test_compare_literal_agree():
    r = compare("12000", corpus_dir=DEFAULT_CORPUS)
    assert r["answer_status"] == "SUPPORTED"
    assert "12000" in str(r)


def test_failure_gallery():
    from wedge_v1.failure_gallery import build_gallery, classify_outcome, gallery_to_markdown

    assert classify_outcome({"ok": True, "got_status": "SUPPORTED"}) == "ok_supported"
    assert classify_outcome({"ok": False, "expect_status": ["SUPPORTED"], "got_status": "ABSTAIN", "ok_status": False}) == "over_abstain"
    g = build_gallery(
        dogfood={
            "n_tasks": 2,
            "n_ok": 1,
            "accuracy": 0.5,
            "rows": [
                {"id": "a", "ok": True, "got_status": "SUPPORTED", "expect_status": ["SUPPORTED"]},
                {
                    "id": "b",
                    "ok": False,
                    "got_status": "ABSTAIN",
                    "expect_status": ["SUPPORTED"],
                    "ok_status": False,
                    "ok_needles": True,
                },
            ],
        },
        path=Path("synthetic"),
    )
    assert "over_abstain" in g["buckets"]
    assert "ok_supported" in g["buckets"]
    md = gallery_to_markdown(g)
    assert "failure gallery" in md


def test_report_build():
    from frontier.verified_ask_report import build_report, format_report_md

    r = build_report("How long before cached entries expire?", corpus_dir=DEFAULT_CORPUS)
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED", "ABSTAIN"}
    md = format_report_md(r)
    assert "**Status:**" in md




def test_report_ask_markdown():
    from wedge_v1.runtime import ask, format_report_md

    payload = ask("How long before cached entries expire?")
    md = format_report_md(payload, title="ask smoke")
    assert "**Status:**" in md
    assert payload.get("answer_status") in {"SUPPORTED", "CONTRADICTED", "ABSTAIN", "NO_CORPUS"}

def test_owner_dogfood_synthetic():
    from wedge_v1.run_owner_dogfood import main
    from pathlib import Path
    import tempfile
    out = Path(tempfile.mkdtemp()) / "results_owner_dogfood.json"
    rc = main(["--corpus", str(Path("wedge_v1/data/corpus")), "--out", str(out),
               "--tasks", "wedge_v1/data/owner_dogfood_tasks.example.json"])
    assert out.exists()
    assert rc in (0, 1)


