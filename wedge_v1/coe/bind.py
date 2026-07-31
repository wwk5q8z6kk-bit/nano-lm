"""Bind ask() payloads to typed CoE claims + evidence record (claim-before-render)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from wedge_v1.coe.record import EvidenceRecord
from wedge_v1.coe.schema import (
    SOLVER_VERSION,
    DerivationKind,
    EvidenceAtom,
    EvidenceRelation,
    TypedClaim,
    VerificationRecord,
    digest_docs,
    digest_text,
    infer_derivation,
    new_id,
)

DEFAULT_RECORD_DIR = Path(__file__).resolve().parents[1] / ".coe_runs"


def _atom_from_evidence(doc_id: str | None, docs: dict[str, str], e: dict) -> EvidenceAtom | None:
    text = str(e.get("text") or e.get("line") or "")
    if not text and e.get("start") is None:
        return None
    body = docs.get(doc_id or "", "") if doc_id else ""
    start = e.get("start")
    end = e.get("end")
    # Prefer exact offsets; validate if possible
    if body and start is not None and end is not None:
        try:
            start_i, end_i = int(start), int(end)
            slice_ = body[start_i:end_i]
            if text and slice_ != text:
                if text in body:
                    start_i = body.find(text)
                    end_i = start_i + len(text)
                    slice_ = body[start_i:end_i]
                elif text not in body:
                    # invalid offset — still record atom with failure relation
                    return EvidenceAtom(
                        atom_id=new_id("atom"),
                        doc_id=doc_id,
                        doc_digest=digest_text(body) if body else None,
                        start=start_i,
                        end=end_i,
                        text=text or slice_,
                        relation=EvidenceRelation.UNSUPPORTED,
                    )
            if not text:
                text = slice_
        except (TypeError, ValueError):
            pass
    return EvidenceAtom(
        atom_id=new_id("atom"),
        doc_id=doc_id,
        doc_digest=digest_text(body) if body else None,
        start=int(start) if start is not None else None,
        end=int(end) if end is not None else None,
        text=text,
        relation=EvidenceRelation.EXACTLY_STATED if text else EvidenceRelation.UNSUPPORTED,
    )


def claim_dict_to_typed(
    c: dict,
    *,
    docs: dict[str, str],
    corpus_digest: str,
    query: str,
    event_ids: list[str],
) -> TypedClaim:
    atoms: list[EvidenceAtom] = []
    for e in c.get("evidence") or []:
        if isinstance(e, dict):
            a = _atom_from_evidence(c.get("doc_id"), docs, e)
            if a:
                atoms.append(a)
    notes = str(c.get("notes") or "")
    deriv = infer_derivation(str(c.get("task_id") or ""), notes)
    ver = c.get("meta", {}).get("verify") if isinstance(c.get("meta"), dict) else None
    vrecs = [
        VerificationRecord(
            verifier_id="classical.verifier.verify_claim",
            independent=True,
            outcome="pass" if ver == "pass" else ("fail" if ver and str(ver).startswith("fail") else "abstain"),
            checks=["nonempty_evidence", "offset_or_text"],
            detail=str(ver or ""),
        )
    ]
    prop = f"{c.get('task_id')}: {c.get('value')}"
    status = str(c.get("status") or "PRESENT")
    return TypedClaim(
        claim_id=new_id("cl"),
        claim_type=str(c.get("task_id") or "UNKNOWN"),
        proposition=prop,
        status=status,
        derivation=deriv,
        solver_id=notes or str(c.get("task_id") or "solver"),
        solver_version=SOLVER_VERSION,
        source_doc_ids=[d for d in [c.get("doc_id")] if d],
        corpus_digest=corpus_digest,
        evidence_atoms=atoms,
        verification=vrecs,
        abstention_state=None if status not in {"ABSTAIN", "REJECTED"} else status,
        execution_event_ids=list(event_ids),
        reproducibility={
            "query": query,
            "replay": "python -m wedge_v1 coe-replay --run RUN_ID",
        },
        raw_value=c.get("value"),
    )


def bind_ask_payload(
    payload: dict,
    docs: dict[str, str],
    *,
    record_dir: Path | None = None,
    persist: bool = True,
) -> dict:
    """Attach typed CoE claims + optional JSONL record. Mutates and returns payload."""
    corpus_digest = digest_docs(docs) if docs else None
    query = str(payload.get("query") or "")
    record = None
    event_ids: list[str] = []
    if persist:
        record = EvidenceRecord.create(record_dir or DEFAULT_RECORD_DIR)
        event_ids.append(
            record.emit(
                "CORPUS_OPENED",
                payload={
                    "corpus_dir": payload.get("corpus_dir"),
                    "n_docs": payload.get("n_docs") or len(docs),
                    "corpus_digest": corpus_digest,
                },
                output_digest=corpus_digest,
            )
        )
        event_ids.append(
            record.emit(
                "QUERY_NORMALIZED",
                payload={"query": query},
                input_digest=digest_text(query),
            )
        )
        # Mirror coarse stages from ask trace if present
        for ev in (payload.get("trace") or {}).get("events") or []:
            stage = str(ev.get("stage") or "CANDIDATE_RETURNED")
            et = {
                "ingest": "CORPUS_OPENED",
                "bm25_margin_gate": "RETRIEVER_EXECUTED",
                "contradiction": "CONTRADICTION_FOUND",
                "composition_gate": "CONDITION_DECOMPOSED",
            }.get(stage, "CANDIDATE_RETURNED")
            event_ids.append(record.emit(et, payload=ev))

    typed: list[TypedClaim] = []
    enriched_claims = []
    for c in payload.get("claims") or []:
        if not isinstance(c, dict):
            continue
        if record:
            eid = record.emit(
                "CLAIM_CONSTRUCTED",
                payload={"task_id": c.get("task_id"), "status": c.get("status")},
            )
            event_ids.append(eid)
            for e in c.get("evidence") or []:
                if isinstance(e, dict):
                    event_ids.append(
                        record.emit("SPAN_SELECTED", payload={"doc_id": c.get("doc_id"), **{k: e.get(k) for k in ("start", "end", "text")}})
                    )
            event_ids.append(
                record.emit(
                    "VERIFIER_EXECUTED",
                    payload={"verify": (c.get("meta") or {}).get("verify")},
                )
            )
        tc = claim_dict_to_typed(
            c, docs=docs, corpus_digest=corpus_digest or "", query=query, event_ids=event_ids[-3:]
        )
        if not tc.evidence_atoms and tc.status in {"PRESENT", "CONFIRMED"}:
            if record:
                record.emit("CLAIM_REJECTED", failure_code="COE_MISSING_SOURCE", payload={"claim_id": tc.claim_id})
        else:
            if record and payload.get("answer_status") in {"SUPPORTED", "CONTRADICTED"}:
                record.emit("CLAIM_PRESENTED", payload={"claim_id": tc.claim_id, "status": tc.status})
        typed.append(tc)
        cd = dict(c)
        cd["claim_id"] = tc.claim_id
        cd["coe"] = {
            "derivation": tc.derivation.value,
            "evidence_atom_ids": [a.atom_id for a in tc.evidence_atoms],
            "corpus_digest": corpus_digest,
        }
        enriched_claims.append(cd)

    if payload.get("contradiction_banner") and record:
        record.emit(
            "CONTRADICTION_FOUND",
            payload={"banner": payload.get("contradiction_banner")},
            failure_code="COE_CONTRADICTION_IGNORED" if payload.get("answer_status") == "SUPPORTED" else None,
        )

    payload["claims"] = enriched_claims
    payload["coe_claims"] = [t.to_dict() for t in typed]
    payload["coe"] = {
        "schema": "nano-lm.wedge_v1.coe.v1",
        "invariant": "EVIDENCE_CREATED_WITH_CLAIM",
        "corpus_digest": corpus_digest,
        "run_id": record.run_id if record else None,
        "record_path": str(record.path) if record else None,
        "n_typed_claims": len(typed),
        "completeness": all(
            (t.evidence_atoms or t.status in {"ABSTAIN", "REJECTED", "DISPUTED"})
            for t in typed
        )
        if typed
        else payload.get("answer_status") in {"ABSTAIN", "NO_CORPUS"},
    }
    if record:
        record.close()
    return payload
