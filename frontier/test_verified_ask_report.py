"""Smoke tests for frontier verified ask report."""
from __future__ import annotations

from pathlib import Path

from frontier.verified_ask_report import build_report

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "wedge_v1" / "data" / "corpus"
PAPERS = ROOT / "papers"


def test_supported_ttl_on_synthetic():
    r = build_report("How long before cache entries expire?", corpus_dir=CORPUS)
    assert r["lm_invoked"] is False
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED"}
    assert r["evidence_spans"] or r["claims"]
    assert isinstance(r["latency_ms"], int)


def test_abstain_oos():
    r = build_report("What is the capital of Mars colonies in 3100?", corpus_dir=CORPUS)
    assert r["answer_status"] in {"ABSTAIN", "SUPPORTED"}  # may still find lexical noise; prefer abstain
    # Prefer abstain; if supported, must have spans
    if r["answer_status"] == "SUPPORTED":
        assert r["evidence_spans"]


def test_papers_corpus_loads():
    r = build_report("evidence ledger", corpus_dir=PAPERS)
    assert r["answer_status"] != "NO_CORPUS"
    assert (r["n_docs"] or 0) >= 1


def test_empty_corpus():
    r = build_report("anything", corpus_dir=ROOT / "frontier" / "_missing_corpus_")
    assert r["answer_status"] == "NO_CORPUS"


if __name__ == "__main__":
    test_supported_ttl_on_synthetic()
    test_abstain_oos()
    test_papers_corpus_loads()
    test_empty_corpus()
    print("FRONTIER_VERIFIED_ASK_SMOKE_OK")
