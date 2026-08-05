"""Bind ask() payloads to typed CoE claims + evidence record (claim-before-render)."""
from __future__ import annotations

import re
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

_PRESENTABLE_STATUSES = {"PRESENT", "CONFIRMED", "PROBABLE", "DISPUTED"}
_NON_SEMANTIC_KEYS = {
    "doc_id",
    "end",
    "field",
    "method",
    "query",
    "relation",
    "start",
}


def _as_offset(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unsupported_atom(
    *,
    doc_id: str | None,
    body: str,
    start: Any,
    end: Any,
    text: str,
) -> EvidenceAtom:
    return EvidenceAtom(
        atom_id=new_id("atom"),
        doc_id=doc_id,
        doc_digest=digest_text(body) if body else None,
        start=_as_offset(start),
        end=_as_offset(end),
        text=text,
        relation=EvidenceRelation.UNSUPPORTED,
    )


def _has_bound_evidence(claim: TypedClaim) -> bool:
    return bool(claim.evidence_atoms) and all(
        atom.relation is not EvidenceRelation.UNSUPPORTED
        and atom.doc_id is not None
        and atom.doc_digest is not None
        and atom.start is not None
        and atom.end is not None
        for atom in claim.evidence_atoms
    )


def _semantic_fragments(value: Any) -> list[tuple[str, bool]]:
    """Return claim-value fragments that source text can directly support.

    Booleans encode an existence decision rather than literal source text, so the
    current span verifier cannot independently compare them. Structural metadata
    is likewise excluded; all remaining scalar values must occur in an atom.
    """
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [(str(value), True)]
    if isinstance(value, str):
        normalized = " ".join(value.casefold().split())
        return [(normalized, False)] if normalized else []
    if isinstance(value, dict):
        out: list[tuple[str, bool]] = []
        for child_key, child in value.items():
            if str(child_key).casefold() in _NON_SEMANTIC_KEYS:
                continue
            out.extend(_semantic_fragments(child))
        return out
    if isinstance(value, (list, tuple, set)):
        out = []
        for child in value:
            out.extend(_semantic_fragments(child))
        return out
    return []


def _fragment_in_text(fragment: str, numeric: bool, text: str) -> bool:
    normalized_text = " ".join(text.casefold().split())
    if numeric:
        return bool(re.search(rf"(?<![\w.]){re.escape(fragment)}(?![\w.])", normalized_text))
    return fragment in normalized_text


def _semantic_value_supported(value: Any, atoms: list[EvidenceAtom]) -> bool:
    if value is None:
        return False
    fragments = _semantic_fragments(value)
    # A boolean is a verifier decision about presence/absence; exact spans can
    # support the decision but do not literally contain "true" or "false".
    if isinstance(value, bool):
        return bool(atoms)
    if not fragments:
        return False
    texts = [atom.text for atom in atoms if atom.text]
    return bool(texts) and all(
        any(_fragment_in_text(fragment, numeric, text) for text in texts)
        for fragment, numeric in fragments
    )


def _is_presentable_ready(claim: TypedClaim) -> bool:
    return (
        _has_bound_evidence(claim)
        and claim.derivation is not DerivationKind.UNKNOWN
        and bool(claim.verification)
        and all(record.outcome == "pass" for record in claim.verification)
        and _semantic_value_supported(claim.raw_value, claim.evidence_atoms)
    )


def _atom_from_evidence(doc_id: str | None, docs: dict[str, str], e: dict) -> EvidenceAtom | None:
    text = str(e.get("text") or e.get("line") or "")
    if not text and e.get("start") is None:
        return None
    evidence_doc_id = e.get("doc_id") or doc_id
    if not isinstance(evidence_doc_id, str) or not evidence_doc_id:
        evidence_doc_id = None
    body = docs.get(evidence_doc_id, "") if evidence_doc_id is not None else ""
    start = e.get("start")
    end = e.get("end")

    if evidence_doc_id not in docs:
        return _unsupported_atom(
            doc_id=evidence_doc_id,
            body=body,
            start=start,
            end=end,
            text=text,
        )

    # Text-only evidence is bound deterministically to a real source span. Supplied
    # offsets, however, are assertions and must match exactly; they are never moved.
    if start is None and end is None:
        if not text:
            return None
        start_i = body.find(text)
        if start_i < 0:
            return _unsupported_atom(
                doc_id=evidence_doc_id,
                body=body,
                start=None,
                end=None,
                text=text,
            )
        end_i = start_i + len(text)
    else:
        start_i = _as_offset(start)
        end_i = _as_offset(end)
        if (
            start_i is None
            or end_i is None
            or start_i < 0
            or end_i <= start_i
            or end_i > len(body)
        ):
            return _unsupported_atom(
                doc_id=evidence_doc_id,
                body=body,
                start=start,
                end=end,
                text=text,
            )
        slice_ = body[start_i:end_i]
        if text and slice_ != text:
            return _unsupported_atom(
                doc_id=evidence_doc_id,
                body=body,
                start=start_i,
                end=end_i,
                text=text,
            )
        if not text:
            text = slice_

    return EvidenceAtom(
        atom_id=new_id("atom"),
        doc_id=evidence_doc_id,
        doc_digest=digest_text(body),
        start=start_i,
        end=end_i,
        text=text,
        relation=EvidenceRelation.EXACTLY_STATED,
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
    evidence_verified = bool(atoms) and all(
        atom.relation is not EvidenceRelation.UNSUPPORTED
        and atom.doc_id is not None
        and atom.doc_digest is not None
        and atom.start is not None
        and atom.end is not None
        for atom in atoms
    )
    semantic_verified = evidence_verified and _semantic_value_supported(c.get("value"), atoms)
    verifier_passed = ver == "pass"
    derivation_known = deriv is not DerivationKind.UNKNOWN
    if (ver and str(ver).startswith("fail")) or not evidence_verified:
        outcome = "fail"
        detail = str(ver or "source evidence did not bind")
    elif not verifier_passed:
        outcome = "abstain"
        detail = str(ver or "missing passing verifier outcome")
    elif not derivation_known:
        outcome = "fail"
        detail = "unknown derivation"
    elif not semantic_verified:
        outcome = "fail"
        detail = "claim value is not supported by bound evidence"
    else:
        outcome = "pass"
        detail = "pass"
    vrecs = [
        VerificationRecord(
            verifier_id="classical.verifier.verify_claim",
            independent=True,
            outcome=outcome,
            checks=[
                "nonempty_evidence",
                "source_doc_exists",
                "exact_span_match",
                "known_derivation",
                "semantic_value_evidence_alignment",
            ],
            detail=detail,
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
        source_doc_ids=list(
            dict.fromkeys(
                atom.doc_id
                for atom in atoms
                if isinstance(atom.doc_id, str) and atom.doc_id
            )
        ),
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


def _persist_bound_payload(
    payload: dict,
    docs: dict[str, str],
    *,
    audit: dict,
    record_dir: Path | None = None,
) -> dict:
    """Persist an already-bound payload after its presentation audit."""
    record = None
    try:
        record = EvidenceRecord.create(record_dir or DEFAULT_RECORD_DIR)
        corpus_digest = (payload.get("coe") or {}).get("corpus_digest")
        query = str(payload.get("query") or "")
        record.emit(
            "CORPUS_OPENED",
            payload={
                "corpus_dir": payload.get("corpus_dir"),
                "n_docs": payload.get("n_docs") or len(docs),
                "corpus_digest": corpus_digest,
            },
            output_digest=corpus_digest,
        )
        record.emit(
            "QUERY_NORMALIZED",
            payload={"query": query},
            input_digest=digest_text(query),
        )
        for event in (payload.get("trace") or {}).get("events") or []:
            stage = str(event.get("stage") or "CANDIDATE_RETURNED")
            event_type = {
                "ingest": "CORPUS_OPENED",
                "bm25_margin_gate": "RETRIEVER_EXECUTED",
                "contradiction": "CONTRADICTION_FOUND",
                "composition_gate": "CONDITION_DECOMPOSED",
            }.get(stage, "CANDIDATE_RETURNED")
            record.emit(event_type, payload=event)

        claims = payload.get("claims") or []
        typed_claims = payload.get("coe_claims") or []
        invalid_ids = set((payload.get("coe") or {}).get("invalid_claim_ids") or [])
        for claim, typed in zip(claims, typed_claims, strict=True):
            claim_events = [
                record.emit(
                    "CLAIM_CONSTRUCTED",
                    payload={
                        "claim_id": typed.get("claim_id"),
                        "task_id": claim.get("task_id"),
                        "status": claim.get("status"),
                    },
                )
            ]
            for evidence in claim.get("evidence") or []:
                claim_events.append(
                    record.emit(
                        "SPAN_SELECTED",
                        payload={
                            "atom_id": evidence.get("atom_id"),
                            "doc_id": evidence.get("doc_id") or claim.get("doc_id"),
                            **{key: evidence.get(key) for key in ("start", "end", "text")},
                        },
                    )
                )
            claim_events.append(
                record.emit(
                    "VERIFIER_EXECUTED",
                    payload={
                        "claim_id": typed.get("claim_id"),
                        "verification": typed.get("verification") or [],
                    },
                )
            )
            typed["execution_event_ids"] = claim_events

        payload.setdefault("coe", {})["run_id"] = record.run_id
        payload["coe"]["record_path"] = str(record.path)
        audit["run_id"] = record.run_id
        record.emit(
            "AUDIT_EXECUTED",
            payload={
                "ok": bool(audit.get("ok")),
                "failure_codes": list(audit.get("failure_codes") or []),
                "n_checks": audit.get("n_checks"),
            },
        )

        for typed in typed_claims:
            claim_id = typed.get("claim_id")
            if claim_id in invalid_ids:
                record.emit(
                    "CLAIM_REJECTED",
                    failure_code="COE_UNSUPPORTED_PREDICATE",
                    payload={"claim_id": claim_id},
                )
            elif payload.get("answer_status") in {"SUPPORTED", "CONTRADICTED"}:
                record.emit(
                    "CLAIM_PRESENTED",
                    payload={"claim_id": claim_id, "status": typed.get("status")},
                )

        if payload.get("contradiction_banner"):
            record.emit(
                "CONTRADICTION_FOUND",
                payload={"banner": payload.get("contradiction_banner")},
                failure_code=(
                    "COE_CONTRADICTION_IGNORED"
                    if payload.get("answer_status") == "SUPPORTED"
                    else None
                ),
            )
        record.close()
        return payload
    except Exception:
        if record is not None:
            record.discard()
        raise


def bind_ask_payload(
    payload: dict,
    docs: dict[str, str],
    *,
    record_dir: Path | None = None,
    persist: bool = True,
) -> dict:
    """Attach typed CoE claims + optional JSONL record. Mutates and returns payload."""
    corpus_digest = digest_docs(docs)
    query = str(payload.get("query") or "")
    typed: list[TypedClaim] = []
    enriched_claims = []
    for c in payload.get("claims") or []:
        if not isinstance(c, dict):
            continue
        tc = claim_dict_to_typed(
            c, docs=docs, corpus_digest=corpus_digest or "", query=query, event_ids=[]
        )
        typed.append(tc)
        cd = dict(c)
        cd["claim_id"] = tc.claim_id
        # Public evidence is projected only from bound atoms. Source context and
        # other raw retrieval fields have no independent offsets and must not
        # survive as unaudited factual text.
        cd["evidence"] = [
            {
                "atom_id": atom.atom_id,
                "doc_id": atom.doc_id,
                "doc_digest": atom.doc_digest,
                "start": atom.start,
                "end": atom.end,
                "text": atom.text,
                "relation": atom.relation.value,
            }
            for atom in tc.evidence_atoms
        ]
        cd["coe"] = {
            "derivation": tc.derivation.value,
            "evidence_atom_ids": [a.atom_id for a in tc.evidence_atoms],
            "corpus_digest": corpus_digest,
        }
        enriched_claims.append(cd)

    payload["claims"] = enriched_claims
    payload["coe_claims"] = [t.to_dict() for t in typed]
    invalid_claim_ids = [
        t.claim_id
        for t in typed
        if t.status in _PRESENTABLE_STATUSES and not _is_presentable_ready(t)
    ]
    payload["coe"] = {
        "schema": "nano-lm.wedge_v1.coe.v1",
        "invariant": "EVIDENCE_CREATED_WITH_CLAIM",
        "corpus_digest": corpus_digest,
        "run_id": None,
        "record_path": None,
        "n_typed_claims": len(typed),
        "n_invalid_claims": len(invalid_claim_ids),
        "invalid_claim_ids": invalid_claim_ids,
        "completeness": all(
            t.status not in _PRESENTABLE_STATUSES or _is_presentable_ready(t)
            for t in typed
        )
        if typed
        else payload.get("answer_status") in {"ABSTAIN", "NO_CORPUS"},
    }
    if persist:
        from wedge_v1.coe.audit import audit_payload

        audit = audit_payload(payload, docs)
        presentation_requested = payload.get("answer_status") in {"SUPPORTED", "CONTRADICTED"}
        presentation_authorized = (
            (not presentation_requested or bool(payload["coe"]["completeness"]))
            and bool(audit.get("ok"))
        )
        if presentation_authorized:
            _persist_bound_payload(
                payload,
                docs,
                audit=audit,
                record_dir=record_dir,
            )
        else:
            payload["coe"]["persistence"] = "BLOCKED_BY_AUDIT"
    return payload
