"""Regression pins for escalate-stub and document scope."""
from __future__ import annotations

import pytest

from wedge_v1.runtime import ask, find_spans, scan, DEFAULT_CORPUS

QUERY = "How long before cached entries expire?"
STUB_QUERY = "invalidation window for memoized responses"


def test_default_abstain_without_escalation():
    r = ask(STUB_QUERY)
    assert r["answer_status"] == "ABSTAIN"
    assert r.get("escalation_attempted") is None


def test_escalate_stub_recovers_ttl_paraphrase():
    r = ask(STUB_QUERY, escalate_stub=True)
    assert r["answer_status"] == "SUPPORTED"
    assert r.get("escalation_attempted") is True
    assert "hybrid_stub" in r.get("solver_path", [])
    assert "300" in str(r.get("claims"))


def test_env_escalate_stub_flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WEDGE_ESCALATE_STUB", "1")
    r = ask(STUB_QUERY)
    assert r["answer_status"] == "SUPPORTED"
    assert r.get("escalation_attempted") is True


def test_oos_stays_abstain_even_with_stub():
    q = "What is the clinical accuracy of NanoScribe in hospitals?"
    r = ask(q, escalate_stub=True)
    assert r["answer_status"] == "ABSTAIN"
    assert r.get("escalation") == "stub_no_recovery"


def test_unknown_doc_scope_abstains():
    r = ask(QUERY, doc_ids=["does-not-exist"])
    assert r["answer_status"] == "ABSTAIN"
    assert r["failure_codes"] == ["UNKNOWN_DOCUMENT_ID"]
    assert r["solver_path"] == ["document_scope"]


def test_empty_doc_scope_abstains():
    r = ask(QUERY, doc_ids=[])
    assert r["answer_status"] == "ABSTAIN"
    assert r["failure_codes"] == ["EMPTY_DOCUMENT_SCOPE"]


def test_exact_doc_scope_limits_sources():
    scoped = ask(QUERY, doc_ids=["tech_note_cache"])
    assert scoped["answer_status"] == "SUPPORTED"
    assert scoped["selected_doc_ids"] == ["tech_note_cache"]
    doc_ids = {c.get("doc_id") for c in scoped["claims"] if c.get("doc_id")}
    assert doc_ids <= {"tech_note_cache"}


def test_find_and_scan_reject_bad_scope():
    bad_find = find_spans("TTL", DEFAULT_CORPUS, doc_ids=["missing"])
    assert bad_find["answer_status"] == "ABSTAIN"
    assert bad_find["failure_codes"] == ["UNKNOWN_DOCUMENT_ID"]
    bad_scan = scan(DEFAULT_CORPUS, doc_ids=["missing"])
    assert bad_scan["answer_status"] == "ABSTAIN"
    assert bad_scan["failure_codes"] == ["UNKNOWN_DOCUMENT_ID"]
