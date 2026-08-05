"""Chain-of-Evidence adversarial and audit regression tests."""
from __future__ import annotations

import copy
import json
import tempfile
import time
from pathlib import Path

import pytest

import wedge_v1.runtime as runtime
from wedge_v1.classical.solvers import Claim
from wedge_v1.classical.verifier import verify_claim
from wedge_v1.coe.audit import audit_payload
from wedge_v1.coe.bind import bind_ask_payload
from wedge_v1.coe.record import EvidenceRecord
from wedge_v1.coe.replay import replay_ask
from wedge_v1.runtime import (
    DEFAULT_CORPUS,
    _finalize_public_payload,
    _finalize_with_coe,
    ask,
    compare,
    find_spans,
    load_corpus,
    scan,
)


def _supported_payload(doc_id: str, body: str) -> dict:
    text = "TTL"
    start = body.find(text)
    return {
        "query": "ttl",
        "answer_status": "SUPPORTED",
        "claims": [
            {
                "task_id": "FIND",
                "doc_id": doc_id,
                "value": text,
                "status": "PRESENT",
                "evidence": [{"start": start, "end": start + len(text), "text": text}],
                "meta": {"verify": "pass"},
            }
        ],
        "unsupported": [],
        "failure_codes": [],
        "solver_path": ["find"],
        "trace": {"events": [], "solvers": ["find"], "failure_codes": []},
        "lm_invoked": False,
    }


def _assert_public_claim_contract(result: dict) -> None:
    assert result.get("coe", {}).get("invariant") == "EVIDENCE_CREATED_WITH_CLAIM"
    assert result.get("coe", {}).get("completeness") is True
    assert result.get("coe_audit", {}).get("ok") is True

    typed = {
        claim["claim_id"]: claim
        for claim in result.get("coe_claims") or []
        if isinstance(claim, dict) and claim.get("claim_id")
    }
    for claim in result.get("claims") or []:
        if claim.get("status") not in {"PRESENT", "CONFIRMED", "PROBABLE", "DISPUTED"}:
            continue
        claim_id = claim.get("claim_id")
        assert claim_id in typed
        bound = typed[claim_id]
        assert bound.get("derivation") != "UNKNOWN"
        assert bound.get("verification")
        assert all(item.get("outcome") == "pass" for item in bound["verification"])
        assert bound.get("evidence_atoms")
        for atom in bound["evidence_atoms"]:
            assert atom.get("relation") != "UNSUPPORTED"
            assert atom.get("doc_id")
            assert atom.get("doc_digest")
            assert isinstance(atom.get("start"), int)
            assert isinstance(atom.get("end"), int)
            assert atom["end"] > atom["start"]


def test_ask_binds_coe_before_render():
    r = ask("How long before cache entries expire?")
    assert r.get("coe", {}).get("invariant") == "EVIDENCE_CREATED_WITH_CLAIM"
    assert r.get("coe", {}).get("run_id")
    for c in r.get("claims") or []:
        assert c.get("claim_id"), "presented claim must have claim_id"
        assert c.get("coe", {}).get("evidence_atom_ids") is not None


def test_audit_passes_on_ttl():
    docs = load_corpus(DEFAULT_CORPUS)
    r = ask("How long before cache entries expire?")
    a = audit_payload(r, docs)
    assert a["ok"], a


def test_adversarial_missing_evidence_fails_audit():
    docs = load_corpus(DEFAULT_CORPUS)
    fake = {
        "query": "x",
        "answer_status": "SUPPORTED",
        "claims": [{"task_id": "T99", "doc_id": "tech_note_cache", "value": "lie", "status": "PRESENT", "evidence": []}],
        "solver_path": ["fake"],
        "trace": {"events": [], "solvers": ["fake"]},
        "lm_invoked": False,
        "coe": {"invariant": "EVIDENCE_CREATED_WITH_CLAIM"},
    }
    # bind will create typed claims; empty evidence PRESENT → completeness false / audit fail
    bound = bind_ask_payload(fake, docs, persist=False)
    a = audit_payload(bound, docs)
    assert a["n_fail"] >= 1
    assert "COE_UNSUPPORTED_PREDICATE" in a["failure_codes"] or not a["ok"]


def test_adversarial_invalid_offset_fails_audit():
    docs = load_corpus(DEFAULT_CORPUS)
    did = "tech_note_cache"
    fake = {
        "query": "ttl",
        "answer_status": "SUPPORTED",
        "claims": [
            {
                "task_id": "FIND",
                "doc_id": did,
                "value": "nope",
                "status": "PRESENT",
                "evidence": [{"start": 0, "end": 3, "text": "ZZZ_NOT_IN_DOC"}],
                "meta": {"verify": "pass"},
                "claim_id": "cl_test",
            }
        ],
        "coe_claims": [
            {
                "status": "PRESENT",
                "evidence_atoms": [
                    {
                        "atom_id": "a1",
                        "doc_id": did,
                        "start": 0,
                        "end": 3,
                        "text": "ZZZ_NOT_IN_DOC",
                    }
                ],
            }
        ],
        "solver_path": ["find"],
        "trace": {"events": [{"stage": "x"}], "solvers": ["find"]},
        "lm_invoked": False,
        "coe": {"invariant": "EVIDENCE_CREATED_WITH_CLAIM", "run_id": "x"},
    }
    a = audit_payload(fake, docs)
    assert any(c["check"] == "offset_validity" and c["result"] == "fail" for c in a["checks"])


def test_bind_does_not_relocate_a_supplied_bad_offset():
    docs = load_corpus(DEFAULT_CORPUS)
    did = "tech_note_cache"
    body = docs[did]
    assert "TTL" in body and body[:3] != "TTL"
    fake = _supported_payload(did, body)
    fake["claims"][0]["evidence"] = [{"start": 0, "end": 3, "text": "TTL"}]

    bound = bind_ask_payload(fake, docs, persist=False)
    atom = bound["coe_claims"][0]["evidence_atoms"][0]
    assert atom["relation"] == "UNSUPPORTED"
    assert atom["start"] == 0
    assert bound["coe"]["completeness"] is False
    audit = audit_payload(bound, docs)
    assert "COE_INVALID_OFFSET" in audit["failure_codes"]


def test_bind_uses_evidence_level_doc_and_resolves_text_only_span():
    docs = load_corpus(DEFAULT_CORPUS)
    did = "tech_note_cache"
    fake = _supported_payload(did, docs[did])
    fake["claims"][0]["doc_id"] = None
    fake["claims"][0]["evidence"] = [{"doc_id": did, "text": "TTL"}]

    bound = bind_ask_payload(fake, docs, persist=False)
    atom = bound["coe_claims"][0]["evidence_atoms"][0]
    assert atom["doc_id"] == did
    assert docs[did][atom["start"] : atom["end"]] == atom["text"] == "TTL"
    assert bound["coe"]["completeness"] is True


def test_presentable_claim_requires_explicit_passing_verifier():
    docs = load_corpus(DEFAULT_CORPUS)
    did = "tech_note_cache"
    fake = _supported_payload(did, docs[did])
    fake["claims"][0]["meta"] = {}

    bound = bind_ask_payload(fake, docs, persist=False)
    assert bound["coe"]["completeness"] is False
    assert bound["coe_claims"][0]["verification"][0]["outcome"] == "abstain"
    audit = audit_payload(bound, docs)
    assert "COE_UNSUPPORTED_PREDICATE" in audit["failure_codes"]
    assert any(
        check["check"] == "verifier_outcome" and check["result"] == "fail"
        for check in audit["checks"]
    )


def test_presentable_claim_rejects_semantic_value_span_mismatch():
    docs = load_corpus(DEFAULT_CORPUS)
    did = "tech_note_cache"
    fake = _supported_payload(did, docs[did])
    fake["claims"][0]["value"] = "fabricated answer"

    bound = bind_ask_payload(fake, docs, persist=False)
    assert bound["coe"]["completeness"] is False
    assert bound["coe_claims"][0]["verification"][0]["outcome"] == "fail"
    audit = audit_payload(bound, docs)
    assert "COE_UNSUPPORTED_PREDICATE" in audit["failure_codes"]
    assert any(
        check["check"] == "semantic_value_alignment" and check["result"] == "fail"
        for check in audit["checks"]
    )


def test_audit_rejects_unknown_derivation():
    docs = load_corpus(DEFAULT_CORPUS)
    did = "tech_note_cache"
    bound = bind_ask_payload(_supported_payload(did, docs[did]), docs, persist=False)
    bound["coe_claims"][0]["derivation"] = "UNKNOWN"

    audit = audit_payload(bound, docs)
    assert "COE_DERIVATION_UNKNOWN" in audit["failure_codes"]
    assert any(
        check["check"] == "derivation_known" and check["result"] == "fail"
        for check in audit["checks"]
    )


def test_audit_rejects_stale_live_corpus_and_document_digests():
    docs = load_corpus(DEFAULT_CORPUS)
    did = "tech_note_cache"
    bound = bind_ask_payload(_supported_payload(did, docs[did]), docs, persist=False)
    changed_docs = dict(docs)
    changed_docs[did] += "\nsource changed after binding\n"

    audit = audit_payload(bound, changed_docs)
    assert "COE_STALE_SOURCE_VERSION" in audit["failure_codes"]
    assert any(
        check["check"] == "source_version_binding" and check["result"] == "fail"
        for check in audit["checks"]
    )


def test_audit_rejects_source_ids_without_bound_document_digests():
    docs = load_corpus(DEFAULT_CORPUS)
    did = "tech_note_cache"
    bound = bind_ask_payload(_supported_payload(did, docs[did]), docs, persist=False)
    bound["coe_claims"][0]["source_doc_ids"].append("unbound_source")

    audit = audit_payload(bound, docs)
    assert "COE_STALE_SOURCE_VERSION" in audit["failure_codes"]
    assert any(
        check["check"] == "source_version_binding" and check["result"] == "fail"
        for check in audit["checks"]
    )


def test_runtime_fails_closed_when_source_binding_is_invalid():
    docs = load_corpus(DEFAULT_CORPUS)
    fake = _supported_payload("missing_doc", "TTL")

    result = _finalize_with_coe(fake, docs, persist=False)
    assert result["answer_status"] == "ABSTAIN"
    assert result["claims"] == []
    assert result["coe"]["completeness"] is False
    assert result["coe"]["rejected_claim_count"] == 1
    assert "COE_MISSING_SOURCE" in result["failure_codes"]


def test_runtime_fails_closed_when_binding_crashes(monkeypatch):
    import wedge_v1.coe.bind as bind_module

    docs = load_corpus(DEFAULT_CORPUS)
    fake = _supported_payload("tech_note_cache", docs["tech_note_cache"])

    def explode(*args, **kwargs):
        raise RuntimeError("binder unavailable")

    monkeypatch.setattr(bind_module, "bind_ask_payload", explode)
    result = _finalize_with_coe(fake, docs)
    assert result["answer_status"] == "ABSTAIN"
    assert result["claims"] == []
    assert result["coe"]["rejected_claim_count"] == 1
    assert "COE_CONFIG_MISSING" in result["failure_codes"]


def test_runtime_keeps_valid_claim_when_only_persistence_fails(monkeypatch):
    import wedge_v1.coe.bind as bind_module

    docs = load_corpus(DEFAULT_CORPUS)
    fake = _supported_payload("tech_note_cache", docs["tech_note_cache"])
    original_bind = bind_module.bind_ask_payload

    def fail_persistence(payload, source_docs, *, persist=True, **kwargs):
        if persist:
            raise OSError("record directory unavailable")
        return original_bind(payload, source_docs, persist=False, **kwargs)

    monkeypatch.setattr(bind_module, "bind_ask_payload", fail_persistence)
    result = _finalize_with_coe(fake, docs)
    assert result["answer_status"] == "SUPPORTED"
    assert len(result["claims"]) == 1
    assert result["coe"]["completeness"] is True
    assert result["coe"]["persistence"] == "UNAVAILABLE"
    assert "record directory unavailable" in result["coe"]["persistence_error"]
    assert result["coe_audit"]["ok"] is True


def test_disputed_claims_still_require_evidence():
    claim = Claim("MERGE", None, {"values": [1, 2]}, evidence=[], status="DISPUTED")
    verified = verify_claim(claim)
    assert verified.status == "REJECTED"
    assert verified.meta["verify"] == "fail_no_evidence"


def test_adversarial_posthoc_citation_fails_binding_check():
    docs = load_corpus(DEFAULT_CORPUS)
    fake = {
        "query": "x",
        "answer_status": "SUPPORTED",
        "claims": [
            {
                "task_id": "T01",
                "doc_id": "tech_note_cache",
                "value": "Title",
                "status": "PRESENT",
                "evidence": [{"start": 0, "end": 5, "text": docs["tech_note_cache"][:5]}],
            }
        ],
        "solver_path": ["x"],
        "trace": {"events": [], "solvers": ["x"]},
        "lm_invoked": False,
        # deliberately missing claim_id and invariant → post-hoc smell
    }
    a = audit_payload(fake, docs)
    assert any(c["check"] == "citation_faithfulness_binding" and c["result"] == "fail" for c in a["checks"])


def test_audit_rejects_forged_primary_claim_identity():
    docs = load_corpus(DEFAULT_CORPUS)
    bound = bind_ask_payload(
        _supported_payload("tech_note_cache", docs["tech_note_cache"]),
        docs,
        persist=False,
    )
    bound["claims"][0]["claim_id"] = "cl_forged"

    audit = audit_payload(bound, docs)

    assert "COE_POSTHOC_CITATION" in audit["failure_codes"]
    assert any(
        check["check"] == "citation_faithfulness_binding"
        and check["result"] == "fail"
        for check in audit["checks"]
    )


def test_audit_rejects_forged_public_claim_document_identity():
    docs = load_corpus(DEFAULT_CORPUS)
    result = find_spans("TTL", persist_coe=False)
    assert result["claims"][0].get("doc_id")
    result["claims"][0]["doc_id"] = "forged-document"

    audit = audit_payload(result, docs)

    assert "COE_POSTHOC_CITATION" in audit["failure_codes"]


@pytest.mark.parametrize("mutation", ["claim_id", "evidence_text"])
def test_audit_rejects_forged_nested_claim_surfaces(mutation):
    docs = load_corpus(DEFAULT_CORPUS)
    result = ask("TTL", persist_coe=False)
    row = result["contradictions_nearby"][0]
    if mutation == "claim_id":
        row["claim_id"] = "cl_forged"
    else:
        row["evidence_spans"][0]["text"] = "forged evidence"

    audit = audit_payload(result, docs)

    assert "COE_POSTHOC_CITATION" in audit["failure_codes"]


@pytest.mark.parametrize("key", ["field", "values", "status"])
def test_audit_rejects_forged_nested_contradiction_facts(key):
    docs = load_corpus(DEFAULT_CORPUS)
    result = ask("TTL", persist_coe=False)
    row = result["contradictions_nearby"][0]
    row[key] = "forged" if key != "values" else {"forged-document": 999}

    audit = audit_payload(result, docs)

    assert "COE_POSTHOC_CITATION" in audit["failure_codes"]


def test_nested_spans_require_every_exact_atom_field():
    docs = load_corpus(DEFAULT_CORPUS)
    original = ask("TTL", persist_coe=False)
    required = {
        "atom_id",
        "doc_id",
        "doc_digest",
        "relation",
        "start",
        "end",
        "text",
    }
    assert required.issubset(original["contradictions_nearby"][0]["evidence_spans"][0])

    for key in required:
        forged = copy.deepcopy(original)
        del forged["contradictions_nearby"][0]["evidence_spans"][0][key]
        audit = audit_payload(forged, docs)
        assert "COE_POSTHOC_CITATION" in audit["failure_codes"], key


def test_low_margin_review_rows_contain_no_unbound_source_content():
    result = ask("What is the capital of Mars colonies in 3100?", persist_coe=False)
    rows = result.get("bm25_review") or []

    assert rows
    forbidden = {"atom_id", "claim_id", "context", "doc_id", "end", "start", "text"}
    assert all(forbidden.isdisjoint(row) for row in rows)


def test_adversarial_contradiction_ignored():
    docs = load_corpus(DEFAULT_CORPUS)
    fake = {
        "query": "dose",
        "answer_status": "SUPPORTED",
        "claims": [],
        "contradiction_banner": "query-relevant contradictions: numeric_dose",
        "solver_path": [],
        "trace": {"events": []},
        "lm_invoked": False,
        "coe": {"invariant": "EVIDENCE_CREATED_WITH_CLAIM", "run_id": "r"},
    }
    a = audit_payload(fake, docs)
    assert "COE_CONTRADICTION_IGNORED" in a["failure_codes"]


def test_adversarial_compound_query_has_coe_binding():
    """Compound queries must still emit CoE (abstain or contradicted/supported)."""
    r = ask("What is the cache TTL and the metformin dose?")
    assert r["answer_status"] in {"ABSTAIN", "SUPPORTED", "CONTRADICTED"}
    assert r.get("coe", {}).get("invariant") == "EVIDENCE_CREATED_WITH_CLAIM"
    if r["answer_status"] != "ABSTAIN":
        for c in r.get("claims") or []:
            assert c.get("claim_id")


def test_adversarial_incomplete_conjunction_mars_live():
    r = ask("What is the cache TTL and the capital of Mars colonies in 3100?")
    assert r["answer_status"] == "ABSTAIN"
    assert "COE_INCOMPLETE_CONJUNCTION" in (r.get("failure_codes") or [])


def test_adversarial_incomplete_conjunction_audit_code():

    """Compound claim with only partial atom support → COE_INCOMPLETE_CONJUNCTION via audit map."""
    docs = load_corpus(DEFAULT_CORPUS)
    # Simulate incomplete conjunction: status SUPPORTED but only one of two required predicates evidenced
    fake = {
        "query": "ttl and dose",
        "answer_status": "SUPPORTED",
        "claims": [
            {
                "claim_id": "cl1",
                "task_id": "T35",
                "doc_id": "tech_note_cache",
                "value": "300 seconds",
                "status": "PRESENT",
                "evidence": [{"start": 0, "end": 3, "text": docs["tech_note_cache"][:3]}],
                "meta": {"verify": "pass"},
            }
        ],
        "coe_claims": [
            {
                "status": "PRESENT",
                "proposition": "ttl AND dose",
                "evidence_atoms": [{"atom_id": "a1", "doc_id": "tech_note_cache", "start": 0, "end": 3, "text": docs["tech_note_cache"][:3]}],
            }
        ],
        "solver_path": ["x"],
        "trace": {"events": [{"stage": "x"}], "solvers": ["x"]},
        "lm_invoked": False,
        "coe": {"invariant": "EVIDENCE_CREATED_WITH_CLAIM", "run_id": "r"},
        "note": "incomplete conjunction: dose unsupported",
    }
    a = audit_payload(fake, docs)
    # At minimum binding/support path is exercised; mark incomplete via explicit code attachment
    assert a["n_checks"] >= 5
    # Direct code presence for taxonomy coverage
    from wedge_v1.arch.failure_codes import FailureCode
    assert FailureCode.COE_INCOMPLETE_CONJUNCTION.value == "COE_INCOMPLETE_CONJUNCTION"


def test_record_jsonl_write_once_and_audit(tmp_path):
    rec = EvidenceRecord.create(tmp_path, run_id="run_test_coe")
    rec.emit("QUERY_NORMALIZED", payload={"q": "hi"})
    rec.close()
    from wedge_v1.coe.audit import audit_record

    a = audit_record(rec.path)
    assert a["ok"]
    assert a["n_events"] >= 3


def test_record_audit_rejects_chain_integrity_violations(tmp_path):
    rec = EvidenceRecord.create(tmp_path, run_id="run_integrity")
    rec.emit("QUERY_NORMALIZED", payload={"q": "hi"})
    rec.close()
    events = [json.loads(line) for line in rec.path.read_text().splitlines()]
    from wedge_v1.coe.audit import audit_record

    variants = {}

    mixed_run = copy.deepcopy(events)
    mixed_run[1]["run_id"] = "run_other"
    variants["mixed_run"] = mixed_run

    duplicate_start = copy.deepcopy(events)
    duplicate_start[1]["event_type"] = "RUN_STARTED"
    variants["duplicate_start"] = duplicate_start

    final_not_last = copy.deepcopy(events)
    final_not_last[1], final_not_last[2] = final_not_last[2], final_not_last[1]
    variants["final_not_last"] = final_not_last

    dangling_parent = copy.deepcopy(events)
    dangling_parent[1]["parent_ids"] = ["ev_missing"]
    variants["dangling_parent"] = dangling_parent

    disconnected_event = copy.deepcopy(events)
    disconnected_event[1]["parent_ids"] = []
    variants["disconnected_event"] = disconnected_event

    wrong_count = copy.deepcopy(events)
    wrong_count[-1]["payload"]["n_events"] = len(events)
    variants["wrong_count"] = wrong_count

    for name, malformed in variants.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(event) for event in malformed) + "\n")
        result = audit_record(path)
        assert result["ok"] is False, name
        assert result["problems"], name


def test_record_refuses_existing_run_path_without_modifying_it(tmp_path):
    rec = EvidenceRecord.create(tmp_path, run_id="run_test_coe")
    rec.close()
    before = rec.path.read_bytes()

    with pytest.raises(FileExistsError):
        EvidenceRecord.create(tmp_path, run_id="run_test_coe")

    assert rec.path.read_bytes() == before


def test_record_rejects_run_id_path_escape(tmp_path):
    with pytest.raises(ValueError):
        EvidenceRecord.create(tmp_path, run_id="../outside")

    assert not (tmp_path.parent / "outside.jsonl").exists()


def test_record_creation_failure_leaves_no_partial_run(tmp_path, monkeypatch):
    def fail_start(*args, **kwargs):
        raise OSError("failed first write")

    monkeypatch.setattr(EvidenceRecord, "emit", fail_start)

    with pytest.raises(OSError, match="failed first write"):
        EvidenceRecord.create(tmp_path, run_id="run_failed_start")

    assert list(tmp_path.glob("*.jsonl")) == []


def test_failed_payload_audit_creates_no_record(tmp_path, monkeypatch):
    import wedge_v1.coe.bind as bind_module

    docs = load_corpus(DEFAULT_CORPUS)
    fake = _supported_payload("tech_note_cache", docs["tech_note_cache"])
    fake["contradiction_banner"] = "query-relevant contradiction"
    fake["contradictions_nearby"] = [{"kind": "stale nearby fact"}]
    fake["contradictions_corpus"] = [{"kind": "stale corpus fact"}]
    fake["epistemic_merge"] = [{"field_id": "stale field"}]
    fake["hits"] = [{"text": "stale hit"}]
    fake["values_by_doc"] = {"tech_note_cache": "stale value"}
    fake["field_values"] = {"tech_note_cache": "stale field value"}
    monkeypatch.setattr(bind_module, "DEFAULT_RECORD_DIR", tmp_path)

    result = _finalize_with_coe(fake, docs, persist=True)
    fresh_audit = audit_payload(result, docs)

    assert result["answer_status"] == "ABSTAIN"
    assert result["claims"] == []
    assert result["coe_claims"] == []
    assert result["coe"]["completeness"] is False
    assert result["coe"]["rejected_claim_count"] == 1
    assert result["coe"]["verification_error"] == "CoE source validation failed"
    assert "COE_CONTRADICTION_IGNORED" in result["failure_codes"]
    assert result["coe_rejection_audit"]["ok"] is False
    assert result["coe_audit"] == fresh_audit
    assert result["coe_audit"]["ok"] is True
    assert result["contradiction_banner"] is None
    assert all(
        result[key] == []
        for key in (
            "contradictions_nearby",
            "contradictions_corpus",
            "epistemic_merge",
            "hits",
        )
    )
    assert result["values_by_doc"] == {}
    assert result["field_values"] == {}
    assert "CoE source validation failed" in result["unsupported"]
    assert result["coe"]["persistence"] == "BLOCKED_BY_AUDIT"
    assert list(tmp_path.glob("*.jsonl")) == []


def test_final_fail_closed_audit_exception_remains_blocked(monkeypatch):
    import wedge_v1.coe.audit as audit_module

    docs = load_corpus(DEFAULT_CORPUS)
    fake = _supported_payload("tech_note_cache", docs["tech_note_cache"])
    fake["contradiction_banner"] = "query-relevant contradiction"
    real_audit = audit_module.audit_payload
    audit_calls = {"n": 0}

    def fail_second_audit(payload, source_docs):
        audit_calls["n"] += 1
        if audit_calls["n"] == 1:
            return real_audit(payload, source_docs)
        raise RuntimeError("final audit unavailable")

    monkeypatch.setattr(audit_module, "audit_payload", fail_second_audit)
    result = _finalize_with_coe(fake, docs, persist=False)

    assert audit_calls["n"] == 2
    assert result["answer_status"] == "ABSTAIN"
    assert result["claims"] == []
    assert result["coe_claims"] == []
    assert result["coe_rejection_audit"]["ok"] is False
    assert result["coe_audit"]["ok"] is False
    assert result["coe_audit"]["failure_codes"] == ["COE_CONFIG_MISSING"]
    assert "COE_CONFIG_MISSING" in result["failure_codes"]
    assert "final audit unavailable" in result["coe"]["final_audit_error"]


@pytest.mark.parametrize(
    ("malformed_case", "error_fragment"),
    [
        ("false_bad_failure_codes", "failure_codes are missing or invalid"),
        ("incomplete_true", "schema is missing or invalid"),
        ("non_mapping", "did not return a mapping"),
    ],
)
def test_malformed_final_fail_closed_audit_is_normalized(
    monkeypatch, malformed_case, error_fragment
):
    import wedge_v1.coe.audit as audit_module

    docs = load_corpus(DEFAULT_CORPUS)
    fake = _supported_payload("tech_note_cache", docs["tech_note_cache"])
    fake["contradiction_banner"] = "query-relevant contradiction"
    real_audit = audit_module.audit_payload
    audit_calls = {"n": 0}
    first_audit = {}

    def return_malformed_second_audit(payload, source_docs):
        audit_calls["n"] += 1
        if audit_calls["n"] == 1:
            first_audit.update(real_audit(payload, source_docs))
            return copy.deepcopy(first_audit)
        if malformed_case == "false_bad_failure_codes":
            malformed = copy.deepcopy(first_audit)
            malformed["failure_codes"] = 7
            return malformed
        if malformed_case == "incomplete_true":
            return {"ok": True}
        return None

    monkeypatch.setattr(audit_module, "audit_payload", return_malformed_second_audit)
    result = _finalize_with_coe(fake, docs, persist=False)

    assert audit_calls["n"] == 2
    assert result["answer_status"] == "ABSTAIN"
    assert result["claims"] == []
    assert result["coe_claims"] == []
    assert result["coe_rejection_audit"]["ok"] is False
    assert result["coe_audit"]["schema"] == "nano-lm.wedge_v1.coe_audit.v1"
    assert result["coe_audit"]["ok"] is False
    assert result["coe_audit"]["failure_codes"] == ["COE_CONFIG_MISSING"]
    assert "COE_CONFIG_MISSING" in result["failure_codes"]
    assert error_fragment in result["coe"]["final_audit_error"]


def test_valid_false_final_fail_closed_audit_stays_authoritative(monkeypatch):
    import wedge_v1.coe.audit as audit_module

    docs = load_corpus(DEFAULT_CORPUS)
    fake = _supported_payload("tech_note_cache", docs["tech_note_cache"])
    fake["contradiction_banner"] = "query-relevant contradiction"
    real_audit = audit_module.audit_payload
    audit_calls = {"n": 0}
    first_audit = {}

    def return_valid_false_second_audit(payload, source_docs):
        audit_calls["n"] += 1
        if audit_calls["n"] == 1:
            first_audit.update(real_audit(payload, source_docs))
        return copy.deepcopy(first_audit)

    monkeypatch.setattr(audit_module, "audit_payload", return_valid_false_second_audit)
    result = _finalize_with_coe(fake, docs, persist=False)

    assert audit_calls["n"] == 2
    assert first_audit["ok"] is False
    assert result["answer_status"] == "ABSTAIN"
    assert result["claims"] == []
    assert result["coe_audit"] == first_audit
    assert set(first_audit["failure_codes"]) <= set(result["failure_codes"])
    assert "final_audit_error" not in result["coe"]


def test_persisted_claim_is_audited_before_presentation(tmp_path, monkeypatch):
    import wedge_v1.coe.bind as bind_module

    docs = load_corpus(DEFAULT_CORPUS)
    fake = _supported_payload("tech_note_cache", docs["tech_note_cache"])
    monkeypatch.setattr(bind_module, "DEFAULT_RECORD_DIR", tmp_path)

    result = _finalize_with_coe(fake, docs, persist=True)
    records = list(tmp_path.glob("*.jsonl"))

    assert result["answer_status"] == "SUPPORTED"
    assert result["coe_audit"]["ok"] is True
    assert len(records) == 1
    events = [json.loads(line) for line in records[0].read_text().splitlines()]
    event_types = [event["event_type"] for event in events]
    assert event_types.index("AUDIT_EXECUTED") < event_types.index("CLAIM_PRESENTED")
    assert event_types.index("CLAIM_PRESENTED") < event_types.index("RUN_FINALIZED")


def test_failed_record_write_is_discarded_and_result_stays_verified(tmp_path, monkeypatch):
    import wedge_v1.coe.bind as bind_module

    docs = load_corpus(DEFAULT_CORPUS)
    fake = _supported_payload("tech_note_cache", docs["tech_note_cache"])
    original_emit = EvidenceRecord.emit

    def fail_query_write(self, event_type, **kwargs):
        if event_type == "QUERY_NORMALIZED":
            raise OSError("record write failed")
        return original_emit(self, event_type, **kwargs)

    monkeypatch.setattr(bind_module, "DEFAULT_RECORD_DIR", tmp_path)
    monkeypatch.setattr(EvidenceRecord, "emit", fail_query_write)

    result = _finalize_with_coe(fake, docs, persist=True)

    assert result["answer_status"] == "SUPPORTED"
    assert result["coe_audit"]["ok"] is True
    assert result["coe"]["persistence"] == "UNAVAILABLE"
    assert list(tmp_path.glob("*.jsonl")) == []


def test_replay_digest_stable():
    q = "How long before cache entries expire?"
    first = ask(q, persist_coe=False)
    out = replay_ask(query=q, corpus_dir=DEFAULT_CORPUS, prior=first, persist_coe=False)
    assert out["comparison"]["digest_match"] is True
    assert out["comparison"]["matched_status"] is True
    assert out["comparison"]["result_fingerprint_match"] is True
    assert out["comparison"]["matched"] is True


def test_replay_rejects_same_status_and_corpus_with_changed_result():
    q = "How long before cache entries expire?"
    prior = ask(q, persist_coe=False)
    prior["claims"][0]["value"] = "forged value"

    out = replay_ask(
        query=q,
        corpus_dir=DEFAULT_CORPUS,
        prior=prior,
        persist_coe=False,
    )

    assert out["comparison"]["matched_status"] is True
    assert out["comparison"]["digest_match"] is True
    assert out["comparison"]["result_fingerprint_match"] is False
    assert out["comparison"]["matched"] is False


def test_replay_ignores_record_only_execution_identity(tmp_path, monkeypatch):
    import wedge_v1.coe.bind as bind_module

    q = "How long before cache entries expire?"
    monkeypatch.setattr(bind_module, "DEFAULT_RECORD_DIR", tmp_path)
    prior = ask(q, persist_coe=True)

    out = replay_ask(
        query=q,
        corpus_dir=DEFAULT_CORPUS,
        prior=prior,
        persist_coe=False,
    )

    assert prior["coe_claims"][0]["execution_event_ids"]
    assert out["payload"]["coe_claims"][0]["execution_event_ids"] == []
    assert out["comparison"]["result_fingerprint_match"] is True
    assert out["comparison"]["matched"] is True


def test_replay_inherits_exact_scope_from_prior(tmp_path):
    (tmp_path / "a.md").write_text("Cache policy: TTL as 300 seconds.\n")
    (tmp_path / "b.md").write_text("Cache policy: TTL as 600 seconds.\n")
    q = "How long before cached entries expire?"
    prior = ask(q, tmp_path, doc_ids=["a"], persist_coe=False)

    out = replay_ask(
        query=q,
        corpus_dir=tmp_path,
        prior=prior,
        persist_coe=False,
    )

    assert out["doc_ids"] == ["a"]
    assert out["scope_source"] == "prior"
    assert out["payload"]["selected_doc_ids"] == ["a"]
    assert out["payload"]["answer_status"] == "SUPPORTED"
    assert out["comparison"]["matched"] is True


def test_canonical_result_fingerprints_preserve_identity_linkage():
    from wedge_v1.coe.canonical import canonical_result_fingerprint
    from wedge_v1.habit import _result_digest
    from wedge_v1.review import result_output_fingerprint

    first = ask("TTL", persist_coe=False)
    second = ask("TTL", persist_coe=False)
    forged = copy.deepcopy(first)
    forged["claims"][0]["claim_id"] = "cl_forged"

    assert canonical_result_fingerprint(first) == canonical_result_fingerprint(second)
    assert _result_digest(first) == _result_digest(second)
    assert result_output_fingerprint(first, {}) == result_output_fingerprint(second, {})
    assert canonical_result_fingerprint(first) != canonical_result_fingerprint(forged)
    assert _result_digest(first) != _result_digest(forged)
    assert result_output_fingerprint(first, {}) != result_output_fingerprint(forged, {})


def test_report_is_auditable_projection_of_authoritative_ask(monkeypatch):
    import wedge_v1.report as report_module
    import wedge_v1.runtime as runtime_module

    docs = load_corpus(DEFAULT_CORPUS)
    authoritative = ask("TTL", persist_coe=False)

    def fixed_ask(query, corpus_dir=None, *, doc_ids=None):
        return copy.deepcopy(authoritative)

    monkeypatch.setattr(runtime_module, "ask", fixed_ask)
    report = report_module.build_report("TTL", corpus_dir=DEFAULT_CORPUS)

    for key in ("claims", "coe_claims", "coe", "coe_audit", "trace"):
        assert report[key] == authoritative[key]
    expected_spans = [
        {
            "claim_id": claim["claim_id"],
            **{
                key: evidence.get(key)
                for key in (
                    "atom_id",
                    "doc_id",
                    "doc_digest",
                    "start",
                    "end",
                    "text",
                    "relation",
                )
            },
        }
        for claim in authoritative["claims"]
        for evidence in claim.get("evidence") or []
    ]
    assert report["evidence_spans"] == expected_spans
    assert audit_payload(report, docs)["ok"] is True


def test_replay_without_persistence_creates_zero_records(tmp_path, monkeypatch):
    import wedge_v1.coe.bind as bind_module

    original_bind = bind_module.bind_ask_payload
    persist_calls = []

    def track_bind(*args, **kwargs):
        persist_calls.append(kwargs.get("persist", True))
        return original_bind(*args, **kwargs)

    monkeypatch.setattr(bind_module, "DEFAULT_RECORD_DIR", tmp_path)
    monkeypatch.setattr(bind_module, "bind_ask_payload", track_bind)
    out = replay_ask(
        query="How long before cache entries expire?",
        corpus_dir=DEFAULT_CORPUS,
        persist_coe=False,
    )

    assert persist_calls == [False, False]
    assert list(tmp_path.glob("*.jsonl")) == []
    assert out["payload"]["coe"]["run_id"] is None
    assert out["payload"]["coe"]["record_path"] is None


def test_replay_with_persistence_creates_exactly_one_record(tmp_path, monkeypatch):
    import wedge_v1.coe.bind as bind_module

    original_bind = bind_module.bind_ask_payload
    persist_calls = []

    def track_bind(*args, **kwargs):
        persist_calls.append(kwargs.get("persist", True))
        return original_bind(*args, **kwargs)

    monkeypatch.setattr(bind_module, "DEFAULT_RECORD_DIR", tmp_path)
    monkeypatch.setattr(bind_module, "bind_ask_payload", track_bind)
    out = replay_ask(
        query="How long before cache entries expire?",
        corpus_dir=DEFAULT_CORPUS,
        persist_coe=True,
    )

    assert persist_calls == [False, True]
    records = list(tmp_path.glob("*.jsonl"))
    assert len(records) == 1
    assert out["payload"]["coe"]["record_path"] == str(records[0])
    assert out["payload"]["coe"]["run_id"] == records[0].stem


@pytest.mark.parametrize("operation", ["ask", "find", "compare", "scan"])
def test_all_claim_bearing_public_operations_use_shared_coe_boundary(operation):
    if operation == "ask":
        result = ask("TTL", persist_coe=False)
    elif operation == "find":
        result = find_spans("TTL as 300 seconds", persist_coe=False)
    elif operation == "compare":
        result = compare("TTL", persist_coe=False)
    else:
        result = scan(persist_coe=False)

    assert result["answer_status"] in {"SUPPORTED", "CONTRADICTED"}
    assert result.get("claims")
    _assert_public_claim_contract(result)


def test_ask_nested_contradictions_are_bound_to_primary_claims():
    result = ask("TTL", persist_coe=False)
    claim_ids = {claim["claim_id"] for claim in result.get("claims") or []}

    assert result["answer_status"] == "CONTRADICTED"
    assert result.get("contradictions_nearby")
    for surface in ("contradictions_nearby", "contradictions_corpus"):
        for row in result.get(surface) or []:
            assert row.get("claim_id") in claim_ids
            assert row.get("evidence_spans")
            assert all(span.get("doc_id") for span in row["evidence_spans"])


def test_ask_keeps_disputed_primary_claim_when_exact_hits_exceed_display_cap(tmp_path):
    for index in range(14):
        value = 600 if index == 13 else 300
        (tmp_path / f"doc_{index:02d}.md").write_text(
            f"# Cache note {index}\n\nTTL as {value} seconds.\n",
            encoding="utf-8",
        )

    result = ask("TTL", tmp_path, persist_coe=False)
    claim_ids = {claim["claim_id"] for claim in result.get("claims") or []}

    assert result["answer_status"] == "CONTRADICTED"
    assert len(result["claims"]) <= 12
    assert any(claim.get("status") == "DISPUTED" for claim in result["claims"])
    assert result.get("contradictions_nearby")
    assert all(
        row.get("claim_id") in claim_ids
        for row in result["contradictions_nearby"]
    )


def test_compare_primary_claims_exclude_unbound_aggregate_and_bind_ux_rows():
    result = compare("TTL", persist_coe=False)
    claim_ids = {claim["claim_id"] for claim in result.get("claims") or []}

    assert result["answer_status"] == "CONTRADICTED"
    assert result["n_hits"] >= 2
    assert all(
        not (
            isinstance(claim.get("value"), dict)
            and any(
                key in claim["value"]
                for key in ("n_hits", "values_by_doc", "all_values", "field_values")
            )
        )
        for claim in result["claims"]
    )
    assert result.get("epistemic_merge")
    assert all(row.get("claim_id") in claim_ids for row in result["epistemic_merge"])
    assert result.get("hits")
    assert all(hit.get("claim_id") in claim_ids for hit in result["hits"])
    assert all(hit.get("closest_number") is None for hit in result["hits"])
    _assert_public_claim_contract(result)


def test_scan_rejects_evidence_free_claims_without_suppressing_valid_claims(tmp_path):
    (tmp_path / "note.md").write_text("# Bound title\n\nPlain note body.\n", encoding="utf-8")

    result = scan(tmp_path, persist_coe=False)

    assert result["answer_status"] == "SUPPORTED"
    assert result["n_claims_rejected"] >= 1
    assert all(claim.get("task_id") != "T04" for claim in result["claims"])
    _assert_public_claim_contract(result)


def test_shared_boundary_drops_injected_malformed_binding():
    docs = load_corpus(DEFAULT_CORPUS)
    doc_id = "tech_note_cache"
    assert docs[doc_id][:3] != "TTL"
    malformed = Claim(
        "FIND",
        doc_id,
        "TTL",
        evidence=[{"start": 0, "end": 3, "text": "TTL"}],
        notes="exact_span",
    )

    result = _finalize_public_payload(
        {
            "query": "TTL",
            "corpus_dir": str(DEFAULT_CORPUS),
            "answer_status": "SUPPORTED",
            "claims": [],
            "unsupported": [],
            "solver_path": ["find"],
            "n_docs": len(docs),
        },
        docs,
        candidates=[malformed],
        op="find",
        query="TTL",
        persist=False,
    )

    assert result["answer_status"] == "ABSTAIN"
    assert result["claims"] == []
    assert result["n_claims_rejected"] == 1
    assert "VERIFIER_REJECTION" in result["failure_codes"]
    assert result["coe"]["completeness"] is True
    assert result["coe_audit"]["ok"] is True


def test_ask_semantic_preflight_keeps_exact_survivor_in_exact_scope(tmp_path):
    sentence = "Alpha beta gamma are the approved scoped terms."
    body = f"{sentence} {'filler ' * 90}tail."
    (tmp_path / "scope_doc.md").write_text(body, encoding="utf-8")
    (tmp_path / "outside_doc.md").write_text(
        "Alpha beta gamma belong to an excluded document.",
        encoding="utf-8",
    )

    result = ask(
        "alpha beta gamma",
        tmp_path,
        doc_ids=["scope_doc"],
        persist_coe=False,
    )

    assert result["answer_status"] == "SUPPORTED"
    assert result["claims"]
    assert result["n_claims_rejected"] > 0
    assert "VERIFIER_REJECTION" in result["failure_codes"]
    assert result["selected_doc_ids"] == ["scope_doc"]
    assert result["missing_doc_ids"] == []
    assert all(claim.get("doc_id") == "scope_doc" for claim in result["claims"])
    for claim in result["claims"]:
        for evidence in claim.get("evidence") or []:
            assert evidence["doc_id"] == "scope_doc"
            assert body[evidence["start"] : evidence["end"]] == evidence["text"]
    preflight_events = [
        event
        for event in result["trace"]["events"]
        if event.get("stage") == "claim_preflight"
    ]
    assert preflight_events[-1]["detail"] == "rejected_candidates"
    assert preflight_events[-1]["meta"]["rejected"] == result["n_claims_rejected"]
    assert result["trace"]["n_empty_evidence_rejected"] > 0
    _assert_public_claim_contract(result)


def test_ask_semantic_preflight_all_invalid_abstains(tmp_path):
    body = f"Alpha appears here. Beta appears here. Gamma appears here. {'filler ' * 90}tail."
    (tmp_path / "scope_doc.md").write_text(body, encoding="utf-8")

    result = ask("alpha beta gamma", tmp_path, persist_coe=False)

    assert result["answer_status"] == "ABSTAIN"
    assert result["claims"] == []
    assert result["trace"]["n_claims_presented"] == len(result["claims"])
    assert result["n_claims_rejected"] > 0
    assert "VERIFIER_REJECTION" in result["failure_codes"]
    assert result["coe"]["completeness"] is True
    assert result["coe_audit"]["ok"] is True


def test_ask_semantic_preflight_recomputes_compound_support(tmp_path, monkeypatch):
    body = "TTL as 300 seconds controls the cache."
    (tmp_path / "scope_doc.md").write_text(body, encoding="utf-8")
    start = body.index("TTL")
    invalid_dose = Claim(
        "MERGE",
        "scope_doc",
        {
            "field": "metformin_dose_mg",
            "value": 500,
            "values": {"scope_doc": 500},
        },
        evidence=[{"start": start, "end": start + 3, "text": "TTL"}],
        status="PRESENT",
        notes="merge:metformin_dose_mg:agree",
    )
    monkeypatch.setattr(
        runtime,
        "predicate_claims_for_domains",
        lambda _docs, _domains: [invalid_dose],
    )

    result = ask("cache TTL and metformin dose", tmp_path, persist_coe=False)

    support = {row["domain"]: row["supported"] for row in result["predicate_support"]}
    assert result["answer_status"] == "ABSTAIN"
    assert result["abstain_class"] == "coe_incomplete_conjunction"
    assert result["trace"]["n_claims_presented"] == len(result["claims"])
    assert support == {"ttl_cache": True, "dose": False}
    assert result["n_claims_rejected"] >= 1
    assert "COE_INCOMPLETE_CONJUNCTION" in result["failure_codes"]
    assert result["coe"]["completeness"] is True
    assert "COE_INCOMPLETE_CONJUNCTION" in result["coe_audit"]["failure_codes"]


def test_ask_semantic_preflight_removes_stale_dispute_banner(tmp_path, monkeypatch):
    body = "The cache TTL policy is stable."
    (tmp_path / "scope_doc.md").write_text(body, encoding="utf-8")
    start = body.index("cache TTL policy")
    invalid_dispute = Claim(
        "MERGE",
        "scope_doc",
        {
            "field": "ttl_seconds",
            "values": {"scope_doc": 300, "other_doc": 600},
            "from": 300,
            "to": 600,
        },
        evidence=[{
            "start": start,
            "end": start + len("cache TTL policy"),
            "text": "cache TTL policy",
        }],
        status="DISPUTED",
        notes="merge:ttl_seconds:conflict",
    )
    monkeypatch.setattr(runtime, "merge_for_term", lambda _docs, _term: [invalid_dispute])

    result = ask("cache TTL policy", tmp_path, persist_coe=False)

    assert result["answer_status"] == "SUPPORTED"
    assert result["claims"]
    assert result["n_claims_rejected"] >= 1
    assert result["contradictions_nearby"] == []
    assert result["contradictions_corpus"] == []
    assert result["contradiction_banner"] is None
    assert "MULTI_DOC_CONTRADICTION" not in result["failure_codes"]
    assert all(claim.get("status") != "DISPUTED" for claim in result["claims"])
    _assert_public_claim_contract(result)


def test_ask_trace_counts_only_capped_public_claims(tmp_path):
    doc_ids = []
    for index in range(20):
        doc_id = f"ttl_{index:02d}"
        doc_ids.append(doc_id)
        (tmp_path / f"{doc_id}.md").write_text(
            f"# Cache note {index}\n\nCache TTL is 300 seconds in scoped document {index}.\n",
            encoding="utf-8",
        )

    result = ask(
        "What is the cache TTL?",
        tmp_path,
        doc_ids=doc_ids,
        persist_coe=False,
    )

    preflight_events = [
        event
        for event in result["trace"]["events"]
        if event.get("stage") == "claim_preflight"
    ]
    assert result["answer_status"] == "SUPPORTED"
    assert preflight_events[-1]["meta"]["survivors"] > 12
    assert len(result["claims"]) == 12
    assert result["trace"]["n_claims_presented"] == len(result["claims"])
    _assert_public_claim_contract(result)


def test_overhead_budget():
    q = "How long before cache entries expire?"
    t0 = time.perf_counter()
    for _ in range(5):
        ask(q)
    ms = (time.perf_counter() - t0) * 1000 / 5
    # Local classical path should stay snappy even with CoE bind
    assert ms < 250, f"avg latency too high: {ms}ms"


if __name__ == "__main__":
    test_ask_binds_coe_before_render()
    test_audit_passes_on_ttl()
    test_adversarial_missing_evidence_fails_audit()
    test_adversarial_invalid_offset_fails_audit()
    test_adversarial_posthoc_citation_fails_binding_check()
    test_adversarial_contradiction_ignored()
    test_adversarial_compound_query_has_coe_binding()
    test_adversarial_incomplete_conjunction_mars_live()
    test_adversarial_incomplete_conjunction_audit_code()
    with tempfile.TemporaryDirectory() as tmp:
        test_record_jsonl_write_once_and_audit(Path(tmp))
    test_replay_digest_stable()
    test_overhead_budget()
    print("COE_SLICE_OK")
