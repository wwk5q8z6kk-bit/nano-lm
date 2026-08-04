"""Regression pins for wedge_v1 runtime (no LM)."""
from __future__ import annotations

from pathlib import Path

import pytest

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


def test_ingest_nested_same_stem_uses_relative_document_ids(tmp_path: Path):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "alpha" / "report.md").write_text("alpha report", encoding="utf-8")
    (tmp_path / "beta" / "report.md").write_text("beta report", encoding="utf-8")

    docs = load_corpus(tmp_path)

    assert docs == {
        "alpha/report": "alpha report",
        "beta/report": "beta report",
    }


def test_ingest_rejects_same_relative_identity_across_formats(tmp_path: Path):
    (tmp_path / "report.md").write_text("markdown", encoding="utf-8")
    (tmp_path / "report.txt").write_text("text", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate document identity 'report'"):
        load_corpus(tmp_path)


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
    assert classify_outcome(
        {"ok": False, "expect_status": ["SUPPORTED"], "got_status": "ABSTAIN", "ok_status": False}
    ) == "over_abstain"
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
    from wedge_v1.report import build_report, format_report_md

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


def test_owner_dogfood_synthetic(tmp_path: Path | None = None):
    import tempfile

    from wedge_v1.run_owner_dogfood import main as owner_main

    with tempfile.TemporaryDirectory(prefix="nano-lm-owner-smoke-") as fallback:
        output_dir = tmp_path or Path(fallback)
        out = output_dir / "results_owner_dogfood.json"
        gallery_md = output_dir / "failure_gallery.md"
        gallery_json = output_dir / "failure_gallery.json"
        smoke_out = output_dir / "results_owner_smoke.json"
        rc = owner_main(
            [
                "--demo",
                "--out",
                str(out),
                "--gallery",
                str(gallery_md),
                "--gallery-json",
                str(gallery_json),
                "--smoke-out",
                str(smoke_out),
            ]
        )
        assert rc == 0
        assert all(
            path.is_file() for path in (out, gallery_md, gallery_json, smoke_out)
        )


def test_owner_dogfood_corpus_flag(tmp_path: Path | None = None):
    import tempfile

    from wedge_v1.run_owner_dogfood import FIXTURE_CORPUS, main as owner_main

    with tempfile.TemporaryDirectory(prefix="nano-lm-owner-corpus-") as fallback:
        output_dir = tmp_path or Path(fallback)
        out = output_dir / "results_owner_dogfood.json"
        gallery_md = output_dir / "failure_gallery.md"
        gallery_json = output_dir / "failure_gallery.json"
        rc = owner_main(
            [
                "--corpus",
                str(FIXTURE_CORPUS),
                "--out",
                str(out),
                "--gallery",
                str(gallery_md),
                "--gallery-json",
                str(gallery_json),
            ]
        )
        assert all(path.is_file() for path in (out, gallery_md, gallery_json))
        assert rc == 0


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
    test_measure_dogfood_u()
    test_owner_dogfood_synthetic()
    test_owner_dogfood_corpus_flag()
    print("WEDGE_V1_SMOKE_OK")


def test_bm25_margin_fields():
    from wedge_v1.classical.bm25 import top_paragraphs
    from wedge_v1.runtime import DEFAULT_CORPUS, load_corpus

    docs = load_corpus(DEFAULT_CORPUS)
    hits = top_paragraphs(docs, "How long before cache entries expire?", k=3)
    assert hits
    assert "margin" in hits[0]
    assert "promote" in hits[0]


def test_ask_no_empty_evidence_present():
    from wedge_v1.runtime import ask

    r = ask("How long before cache entries expire?")
    for c in r.get("claims") or []:
        if c.get("status") in {"PRESENT", "CONFIRMED", "DISPUTED"}:
            assert c.get("evidence"), f"empty evidence: {c.get('task_id')}"


def test_evolve_recommends_workstreams():
    from wedge_v1.failure_to_architecture import recommend

    out = recommend({"tallies": {"low_margin_review": 2, "wrong_or_empty_span": 1}})
    assert "W1" in out["recommended_next"]
    assert "W2" in out["recommended_next"]


def test_evolve_does_not_invent_architecture_work_on_green_gallery():
    from wedge_v1.failure_to_architecture import recommend

    out = recommend({"tallies": {}})

    assert out["recommended_next"] == []
    assert out["next_product_action"] == "MEASURE_REPRESENTATIVE_USE"
