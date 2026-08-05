"""Replay verified-ask from corpus + query; compare to prior CoE payload."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from wedge_v1.coe.canonical import canonical_result_fingerprint


def replay_ask(
    *,
    query: str,
    corpus_dir: Path,
    prior: dict | None = None,
    doc_ids: list[str] | None = None,
    persist_coe: bool = False,
) -> dict:
    """Re-run ask and compare the complete canonical public result to prior."""
    from wedge_v1.coe.audit import audit_payload
    from wedge_v1.runtime import ask, load_corpus, normalize_doc_ids, select_documents

    scope_source = "explicit" if doc_ids is not None else "unscoped"
    effective_doc_ids = normalize_doc_ids(doc_ids)
    if doc_ids is None and prior is not None and (
        "selected_doc_ids" in prior or "missing_doc_ids" in prior
    ):
        effective_doc_ids = normalize_doc_ids(
            [
                *(prior.get("selected_doc_ids") or []),
                *(prior.get("missing_doc_ids") or []),
            ]
        )
        scope_source = "prior"

    docs, _, _ = select_documents(load_corpus(corpus_dir), effective_doc_ids)
    # ask() owns CoE binding. Passing persistence through avoids both an orphan
    # record and a second binding pass that would mint different claim IDs.
    out = ask(
        query,
        corpus_dir=corpus_dir,
        doc_ids=effective_doc_ids,
        persist_coe=persist_coe,
    )
    audit = audit_payload(out, docs)
    cmp: dict[str, Any] = {"matched": None}
    if prior is not None:
        prior_fingerprint = canonical_result_fingerprint(prior)
        replay_fingerprint = canonical_result_fingerprint(out)
        matched_status = out.get("answer_status") == prior.get("answer_status")
        digest_match = (prior.get("coe") or {}).get("corpus_digest") == (
            out.get("coe") or {}
        ).get("corpus_digest")
        result_fingerprint_match = prior_fingerprint == replay_fingerprint
        cmp = {
            "matched": matched_status and digest_match and result_fingerprint_match,
            "matched_status": matched_status,
            "prior_status": prior.get("answer_status"),
            "replay_status": out.get("answer_status"),
            "prior_n_claims": len(prior.get("claims") or []),
            "replay_n_claims": len(out.get("claims") or []),
            "corpus_digest_prior": (prior.get("coe") or {}).get("corpus_digest"),
            "corpus_digest_replay": (out.get("coe") or {}).get("corpus_digest"),
            "digest_match": digest_match,
            "prior_result_fingerprint": prior_fingerprint,
            "replay_result_fingerprint": replay_fingerprint,
            "result_fingerprint_match": result_fingerprint_match,
        }
    return {
        "schema": "nano-lm.wedge_v1.coe_replay.v1",
        "query": query,
        "corpus_dir": str(corpus_dir),
        "doc_ids": effective_doc_ids,
        "scope_source": scope_source,
        "payload": out,
        "audit": audit,
        "comparison": cmp,
    }
