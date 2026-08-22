"""ΔU fixture arms + citation packing."""
from __future__ import annotations

from wedge_v1.eval.arms import DELTA, escalate_stub_ask, run_arms_eval
from wedge_v1.eval.cite_pack import format_citation_md, pack_claim, pack_claims
from wedge_v1.runtime import DEFAULT_CORPUS, ask, format_report_md, load_corpus


def test_cite_pack_compacts_evidence():
    claim = {
        "task_id": "T",
        "status": "PRESENT",
        "doc_id": "note",
        "value": {"answer": "300 seconds", "query": "ttl"},
        "evidence": [
            {
                "doc_id": "note",
                "start": 10,
                "end": 21,
                "text": "TTL as 300",
                "context": "Cache TTL as 300 seconds.",
            }
        ],
        "notes": "x",
    }
    packed = pack_claim(claim)
    assert packed["value"] == "300 seconds"
    assert packed["citations"][0]["quote"] == "TTL as 300"
    md = format_citation_md(packed["citations"][0])
    assert "note:10-21" in md
    assert pack_claims([claim])


def test_eval_arms_fixture_keep_classical():
    out = run_arms_eval(demo=True, persist=False)
    assert out["schema"] == "nano-lm.wedge_v1.eval_arms.v1"
    assert out["n_tasks"] >= 5
    assert out["classical"]["n_ok"] == out["n_tasks"]
    assert out["delta_threshold"] == DELTA
    assert out["verdict"] == "KEEP_CLASSICAL"
    assert out["admit_escalation"] is False
    assert out["delta_u"] <= DELTA


def test_escalate_stub_recovers_ttl_on_corpus():
    docs = load_corpus(DEFAULT_CORPUS)
    out = escalate_stub_ask("How long before cached entries expire?", docs)
    assert out["answer_status"] == "SUPPORTED"
    assert out["claims"]
    assert out["claims"][0].get("evidence")


def test_escalate_stub_refuses_oos():
    docs = load_corpus(DEFAULT_CORPUS)
    out = escalate_stub_ask("What is the clinical accuracy of NanoScribe in hospitals?", docs)
    assert out["answer_status"] == "ABSTAIN"


def test_ask_escalate_stub_default_off_keeps_oos():
    out = ask(
        "What is the clinical accuracy of NanoScribe in hospitals?",
        DEFAULT_CORPUS,
        persist_coe=False,
        escalate_stub=False,
    )
    assert out["answer_status"] == "ABSTAIN"
    assert out.get("escalation_attempted") is not True


def test_ask_escalate_stub_oos_still_abstains():
    out = ask(
        "What is the clinical accuracy of NanoScribe in hospitals?",
        DEFAULT_CORPUS,
        persist_coe=False,
        escalate_stub=True,
    )
    assert out["answer_status"] == "ABSTAIN"
    assert out.get("escalation_attempted") is True
    assert out.get("escalation") == "stub_miss"


def test_ask_escalate_stub_recovers_forced_classical_miss(monkeypatch, tmp_path):
    (tmp_path / "cache.md").write_text(
        "Cache TTL as 300 seconds for entries.\n",
        encoding="utf-8",
    )
    import wedge_v1.runtime as rt

    monkeypatch.setattr(rt, "_relevant_claim", lambda *a, **k: False)
    off = ask(
        "How long before cached entries expire?",
        tmp_path,
        persist_coe=False,
        escalate_stub=False,
    )
    assert off["answer_status"] == "ABSTAIN"
    on = ask(
        "How long before cached entries expire?",
        tmp_path,
        persist_coe=False,
        escalate_stub=True,
    )
    assert on["answer_status"] == "SUPPORTED"
    assert on.get("escalation_attempted") is True
    assert on.get("claims")
    assert any("hybrid_stub" in str(s) or "stub" in str(s) for s in on.get("solver_path") or [])


def test_format_report_uses_packed_citations():
    payload = {
        "query": "ttl",
        "answer_status": "SUPPORTED",
        "solver_path": ["ask"],
        "values_by_doc": {"a": ["300"], "b": ["600"]},
        "claims": [
            {
                "task_id": "COMPARE",
                "status": "DISPUTED",
                "doc_id": None,
                "value": {"term": "TTL", "all_values": ["300", "600"]},
                "evidence": [
                    {"doc_id": "a", "start": 0, "end": 3, "text": "TTL"},
                ],
            }
        ],
    }
    md = format_report_md(payload, title="test")
    assert "## Compare values" in md
    assert "`a`: 300" in md
    assert "cite:" in md
    assert "Claims (1)" in md


if __name__ == "__main__":
    test_cite_pack_compacts_evidence()
    test_eval_arms_fixture_keep_classical()
    test_escalate_stub_recovers_ttl_on_corpus()
    test_escalate_stub_refuses_oos()
    test_ask_escalate_stub_default_off_keeps_oos()
    test_ask_escalate_stub_oos_still_abstains()
    test_format_report_uses_packed_citations()
    print("EVAL_ARMS_OK")
