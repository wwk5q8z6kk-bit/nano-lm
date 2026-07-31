"""W3 atomic predicates + multi-doc merge pins."""
from __future__ import annotations

from wedge_v1.classical.merge import merge_all
from wedge_v1.coe.predicates import decompose, evaluate_predicates, incomplete_conjunction
from wedge_v1.runtime import DEFAULT_CORPUS, ask, load_corpus
from wedge_v1.arch.failure_codes import FailureCode


def test_decompose_open_conjunct():
    preds = decompose("What is the cache TTL and the capital of Mars colonies in 3100?")
    assert len(preds) == 2
    assert preds[0].domain == "ttl_cache"
    assert preds[1].domain == "open"


def test_incomplete_conjunction_mars():
    r = ask("What is the cache TTL and the capital of Mars colonies in 3100?")
    assert r["answer_status"] == "ABSTAIN"
    assert r.get("abstain_class") == "coe_incomplete_conjunction"
    assert FailureCode.COE_INCOMPLETE_CONJUNCTION.value in (r.get("failure_codes") or [])
    support = r.get("predicate_support") or []
    assert any(s["domain"] == "ttl_cache" and s["supported"] for s in support)
    assert any(s["domain"] == "open" and not s["supported"] for s in support)


def test_complete_conjunction_ttl_dose():
    r = ask("What is the cache TTL and the metformin dose?")
    assert r["answer_status"] in {"SUPPORTED", "CONTRADICTED"}
    support = r.get("predicate_support") or []
    assert len(support) == 2
    assert all(s["supported"] for s in support)
    assert FailureCode.COE_INCOMPLETE_CONJUNCTION.value not in (r.get("failure_codes") or [])


def test_merge_detects_conflicts_without_fixture_ids():
    docs = load_corpus(DEFAULT_CORPUS)
    claims = merge_all(docs)
    notes = " ".join(c.notes for c in claims)
    assert "ttl_seconds:conflict" in notes
    assert "metformin_dose_mg:conflict" in notes
    for c in claims:
        if c.status == "DISPUTED":
            assert c.evidence and all("doc_id" in e for e in c.evidence)


def test_evaluate_incomplete_helper():
    docs = load_corpus(DEFAULT_CORPUS)
    preds = decompose("TTL and capital of Mars")
    # no claims → incomplete
    supports = evaluate_predicates(preds, docs, [])
    assert incomplete_conjunction(supports)


if __name__ == "__main__":
    test_decompose_open_conjunct()
    test_incomplete_conjunction_mars()
    test_complete_conjunction_ttl_dose()
    test_merge_detects_conflicts_without_fixture_ids()
    test_evaluate_incomplete_helper()
    print("W3_PREDICATES_OK")
