"""Regression pins for wedge_v1 runtime (no LM)."""
from __future__ import annotations

from wedge_v1.runtime import ask, scan, find_spans, DEFAULT_CORPUS
from pathlib import Path


def test_ttl_supported():
    r = ask("How long before cached entries expire?")
    assert r["answer_status"] == "SUPPORTED"
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


def test_find_ttl_phrase():
    r = find_spans("TTL as 300 seconds")
    assert r["answer_status"] == "SUPPORTED"
    assert r["n_hits"] >= 1


if __name__ == "__main__":
    test_ttl_supported()
    test_oos_abstain()
    test_empty_corpus()
    test_scan_docs()
    test_find_ttl_phrase()
    print("WEDGE_V1_SMOKE_OK")
