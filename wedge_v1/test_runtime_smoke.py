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
    from wedge_v1.run_owner_dogfood import main as owner_main

    rc = owner_main(["--demo"])
    assert rc in (0, 1)



def test_owner_dogfood_fixture():
    """Public fixture corpus proves owner-dogfood path (no PHI)."""
    from wedge_v1.run_owner_dogfood import EXAMPLE_CORPUS as FIXTURE_CORPUS, run

    out = run(FIXTURE_CORPUS)
    assert out.get("error") != "NO_CORPUS"
    assert out["n_tasks"] == 5
    assert out["n_ok"] == out["n_tasks"], out["rows"]
    assert Path(out["out"]).is_file()


def test_measure_dogfood_u():
    from wedge_v1.eval.dogfood_utility import measure_dogfood_u

    u = measure_dogfood_u(
        {
            "schema": "test",
            "n_tasks": 2,
            "n_ok": 2,
            "rows": [
                {"ok": True, "got_status": "SUPPORTED", "latency_s": 0.01},
                {"ok": True, "got_status": "ABSTAIN", "latency_s": 0.01},
            ],
        },
        corpus_class="SYNTHETIC_MINI",
    )
    assert u["Q"] == 1.0
    assert u["R"] == 0.5
    assert u["U_status"] == "DRAFT_NOT_SCORING_FROZEN"
    assert isinstance(u["U"], float)

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
    test_compare_metformin_contradicted()
    test_compare_literal_agree()
    test_failure_gallery()
    test_report_build()
    test_report_ask_markdown()
    test_owner_dogfood_synthetic()
    test_owner_dogfood_corpus_flag()
    test_measure_dogfood_u()
    print("WEDGE_V1_SMOKE_OK")
