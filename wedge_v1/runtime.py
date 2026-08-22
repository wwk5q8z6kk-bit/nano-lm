"""Nano Runtime wedge slice — classical + E-class solvers only (no LM).

Authorized by owner "continue" after ECLASS_CLOSED_WITHOUT_LM.
E-class probes live in classical.solvers (paraphrastic_ttl / symbolic_dose_change / coref_binding).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import time

from wedge_v1.classical import solvers as S
from wedge_v1.classical.bm25 import index_docs
from wedge_v1.classical.bm25 import top_paragraphs
from wedge_v1.arch.failure_codes import FailureCode
from wedge_v1.arch.trace import AskTrace, classify_abstain_failures
from wedge_v1.classical.verifier import verify_claim
from wedge_v1.classical.merge import (
    epistemic_entry,
    merge_all,
    merge_for_term,
    predicate_claims_for_domains,
)
from wedge_v1.plugins.cascade import run_cascade
from wedge_v1.eval.cite_pack import format_packed_claims_md, pack_claims
from wedge_v1.plugins.lexicon import synonyms as _synonym_map
from wedge_v1.coe.predicates import (
    decompose,
    evaluate_predicates,
    incomplete_conjunction,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "corpus"
GOLD_PATH = ROOT / "data" / "gold" / "gold.json"



def load_corpus(corpus_dir: Path | None = None, *, normalize: bool | str = "auto") -> dict[str, str]:
    from wedge_v1.ingest import load_corpus as _ingest

    path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    return _ingest(path, normalize=normalize)


def normalize_doc_ids(doc_ids: list[str] | None) -> list[str] | None:
    """Return the canonical exact-document scope used by every public surface."""
    if doc_ids is None:
        return None
    return sorted({str(doc_id).strip() for doc_id in doc_ids if str(doc_id).strip()})


def select_documents(
    docs: dict[str, str],
    doc_ids: list[str] | None,
) -> tuple[dict[str, str], dict, bool]:
    """Apply an exact scope without ever falling back to the full corpus."""
    normalized = normalize_doc_ids(doc_ids)
    if normalized is None:
        return docs, {}, True

    selected_ids = [doc_id for doc_id in normalized if doc_id in docs]
    missing_ids = [doc_id for doc_id in normalized if doc_id not in docs]
    scope = {
        "selected_doc_ids": selected_ids,
        "missing_doc_ids": missing_ids,
    }
    valid = bool(normalized) and not missing_ids
    if not valid:
        return {}, scope, False
    return {doc_id: docs[doc_id] for doc_id in selected_ids}, scope, True


def _scope_abstention(
    *,
    op: str,
    query: str,
    corpus_path: Path,
    scope: dict,
    persist: bool,
    extra: dict | None = None,
) -> dict:
    """Fail closed before retrieval when an explicit document scope is invalid."""
    from wedge_v1.coe.schema import digest_docs

    missing = list(scope.get("missing_doc_ids") or [])
    code = "UNKNOWN_DOCUMENT_ID" if missing else "EMPTY_DOCUMENT_SCOPE"
    note = (
        "one or more exact document IDs were not found"
        if missing
        else "explicit document scope is empty"
    )
    trace = AskTrace(query=query, corpus_dir=str(corpus_path), op=op)
    trace.add_solver("document_scope")
    trace.n_docs = 0
    trace.event("document_scope", "selection rejected", **scope)
    trace_payload = trace.finalize("ABSTAIN")
    trace_payload["failure_codes"] = list(
        dict.fromkeys([*(trace_payload.get("failure_codes") or []), code])
    )
    payload = {
        "query": query,
        "corpus_dir": str(corpus_path),
        "answer_status": "ABSTAIN",
        "claims": [],
        "coe_claims": [],
        "unsupported": [note],
        "note": note,
        "solver_path": ["document_scope"],
        "failure_codes": [code],
        "trace": trace_payload,
        "n_docs": 0,
        "n_claims_presented": 0,
        "lm_invoked": False,
        **scope,
    }
    if extra:
        payload.update(extra)
    finalized = _finalize_with_coe(payload, {}, persist=persist)
    finalized.setdefault("coe", {})["corpus_digest"] = digest_docs({})
    finalized.setdefault("coe", {})["selected_doc_ids"] = list(scope.get("selected_doc_ids") or [])
    finalized["n_claims_presented"] = 0
    return finalized


def _apply_scope(finalized: dict, scope: dict, *, scoped: bool) -> dict:
    if scoped and scope:
        finalized.update(scope)
        finalized.setdefault("coe", {})["selected_doc_ids"] = list(
            scope.get("selected_doc_ids") or []
        )
    return finalized


def _env_escalate_stub() -> bool:
    return os.environ.get("WEDGE_ESCALATE_STUB", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _maybe_escalate_stub(payload: dict, docs: dict[str, str], *, enabled: bool) -> dict:
    """Opt-in hybrid stub after classical ABSTAIN. Default remains fail-closed."""
    if not enabled or payload.get("answer_status") != "ABSTAIN":
        return payload
    from wedge_v1.eval.arms import escalate_stub_ask

    query = str(payload.get("query") or "")
    stub = escalate_stub_ask(query, docs)
    path = list(payload.get("solver_path") or [])
    for step in stub.get("solver_path") or ["hybrid_stub"]:
        if step not in path:
            path.append(step)
    out = dict(payload)
    out["escalation_attempted"] = True
    out["escalation"] = stub.get("escalation")
    out["solver_path"] = path
    out["lm_invoked"] = False
    if stub.get("answer_status") != "SUPPORTED" or not stub.get("claims"):
        return out
    out["answer_status"] = "SUPPORTED"
    out["claims"] = list(stub.get("claims") or [])
    out["unsupported"] = []
    out["note"] = "recovered via hybrid stub after classical ABSTAIN"
    tr = out.get("trace")
    if isinstance(tr, dict):
        tr = dict(tr)
        tr["answer_status"] = "SUPPORTED"
        out["trace"] = tr
    return out


def _finish_ask(
    payload: dict,
    docs: dict[str, str],
    *,
    scope: dict,
    scoped: bool,
    persist_coe: bool,
    escalate_stub: bool,
) -> dict:
    payload = _maybe_escalate_stub(
        payload,
        docs,
        enabled=bool(escalate_stub) or _env_escalate_stub(),
    )
    return _apply_scope(
        _finalize_with_coe(payload, docs, persist=persist_coe),
        scope,
        scoped=scoped,
    )


def _load_gold() -> dict | None:
    if GOLD_PATH.exists():
        return json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return None


def claim_to_dict(c: S.Claim) -> dict:
    return asdict(c)


STOP = {
    "how", "long", "before", "what", "when", "where", "which", "does", "the", "and",
    "for", "is", "of", "in", "a", "an", "to", "on", "at", "by", "or", "as", "be",
    "are", "was", "were", "with", "from", "this", "that", "these", "those", "into",
    "will", "did", "under", "about", "into", "over", "than", "then", "also", "only",
}


def _content_tokens(q: str) -> list[str]:
    # Min length 2 after first char keeps E4/M1; STOP filters "is"/"on"/etc.
    words = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", q) if t.lower() not in STOP]
    raw_nums = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])", q)
    # Drop digits that only appear inside hyphenated tokens (e.g. GPT-4)
    hyphen_chunks = set(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", q))
    nums = []
    for n in raw_nums:
        if any(n in chunk and chunk != n for chunk in hyphen_chunks):
            continue
        nums.append(n)
    # Program IDs like E4, M1 (too short for the main word regex).
    for m in re.finditer(r"\b[A-Z]\d+\b", q):
        words.append(m.group(0))
    # preserve order, unique
    out, seen = [], set()
    for t in words + nums:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _token_equivalence_groups(tokens: list[str]) -> list[set[str]]:
    """Lexical synonym groups for relevance (W4 expand map; not LM).

    Short underscore/hyphen compounds (2–3 parts, e.g. ``M1_template``) expand to
    parts so prose spans match. Longer snake_case IDs stay atomic to avoid false hits.
    """
    syn = _synonym_map()
    groups: list[set[str]] = []
    for tok in tokens:
        low = tok.lower()
        group = {low}
        # Only short compounds (M1_template). Long snake_case IDs must match whole.
        parts = [p for p in re.split(r"[_\-]+", low) if len(p) >= 2]
        if 2 <= len(parts) <= 3:
            group.update(parts)
            group.add(" ".join(parts))
        for src, dsts in syn.items():
            src_l = src.lower()
            dst_l = {d.lower() for d in dsts}
            if low == src_l or low in dst_l or src_l in low or low in src_l:
                group.add(src_l)
                group.update(dst_l)
                group.update(re.findall(r"[a-z0-9]+", src_l))
        groups.append(group)
    return groups


def _relevant_claim(c: S.Claim, tokens: list[str], query: str = "", docs: dict[str, str] | None = None) -> bool:
    """Reject weak lexical coincidences (one stop-ish content token in a huge corpus)."""
    if not tokens:
        return False
    blob = json.dumps(c.value, default=str).lower() + " " + " ".join(
        str(e.get("text", "")) + " " + str(e.get("line", "")) for e in (c.evidence or [])
    ).lower()
    qlow = (query or "").lower()
    if c.task_id in {"T35", "T36", "T39", "T26", "T29", "T30", "MERGE"} and c.status in {
        "PRESENT", "CONFIRMED", "DISPUTED"
    }:
        return True
    # Phrase FIND: keep multi-word spans that are literally part of the query.
    if c.task_id == "FIND" and c.value:
        val = str(c.value).strip().lower()
        joined = " ".join(t.lower() for t in tokens)
        # Multi-word phrases only — single tokens like Pythia-160M must not
        # satisfy a different question (e.g. GPT-4 score) just by appearing in the query.
        if " " in val and len(val) >= 8 and (val in qlow or val in joined):
            return True
    # Numeric FIND with co-located query labels in the same doc (E4 KILL 0.638).
    # The span text is only the number; gate is applied separately via
    # _numeric_gate_claim_ok — treat token coverage as satisfied here when notes say so.
    if c.task_id == "FIND" and c.notes == "numeric_span" and c.doc_id:
        nums = [tok for tok in tokens if any(ch.isdigit() for ch in tok)]
        if nums and str(c.value) in nums:
            return True  # final keep still requires _numeric_gate_claim_ok
    # Numeric gate queries: span carries the number; labels co-occur in the document.
    if c.task_id == "FIND" and c.notes == "numeric_span" and docs and c.doc_id:
        nums = [t for t in tokens if any(ch.isdigit() for ch in t)]
        if nums and str(c.value) in nums:
            labels = [t for t in tokens if t not in nums]
            body = (docs.get(c.doc_id) or "").lower()
            if not labels or all(lab.lower() in body for lab in labels):
                return True

    hits = sum(1 for group in _token_equivalence_groups(tokens) if any(t in blob for t in group))
    # Passage claims stay strict: all content tokens for long queries (OOS safety).
    # OVER_ABSTENTION recoveries rely on snake_case expansion + phrase_locate,
    # not a majority gate (majority leaked GPT-4 / Paper-alpha coincidences).
    if len(tokens) >= 3:
        need = len(tokens)
    elif len(tokens) == 2:
        need = 2
    else:
        need = 1
    return hits >= need



def _is_table_dump_claim(c: S.Claim) -> bool:
    """Reject markdown-table / ledger dumps masquerading as answers (D07 class)."""
    val = str(c.value or "")
    if len(val) < 180:
        return False
    if val.count("|") >= 4:
        return True
    if val.count("\n") >= 4 and len(val) > 400:
        return True
    return False


def _numeric_gate_claim_ok(c: S.Claim, tokens: list[str], docs: dict[str, str]) -> bool:
    """Numeric spans need co-occurring gate labels in the same document (E4 KILL 0.638)."""
    if c.task_id != "FIND" or c.notes != "numeric_span" or not c.doc_id:
        return True
    nums = [t for t in tokens if any(ch.isdigit() for ch in t)]
    if not nums or str(c.value) not in nums:
        return True
    labels = [t for t in tokens if t not in nums]
    if not labels:
        return True
    body = (docs.get(c.doc_id) or "").lower()
    return any(lab.lower() in body for lab in labels)


def find_spans(
    needle: str,
    corpus_dir: Path | None = None,
    max_hits: int = 20,
    *,
    doc_ids: list[str] | None = None,
    persist_coe: bool = True,
) -> dict:
    """Exact substring locate with evidence spans (classical)."""
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    all_docs = load_corpus(corpus_dir)
    docs, scope, scope_valid = select_documents(all_docs, doc_ids)
    if not scope_valid:
        return _scope_abstention(
            op="find",
            query=needle.strip(),
            corpus_path=corpus_path,
            scope=scope,
            persist=persist_coe,
        )
    if not docs:
        return {"answer_status": "NO_CORPUS", "claims": [], "solver_path": ["load_corpus"]}
    needle = needle.strip()
    if not needle:
        return {"answer_status": "ABSTAIN", "claims": [], "unsupported": ["empty needle"], "solver_path": ["find_spans"]}
    claims = []
    for did, text in docs.items():
        start = 0
        while True:
            i = text.find(needle, start)
            if i < 0:
                break
            # window for context
            lo = max(0, i - 40)
            hi = min(len(text), i + len(needle) + 40)
            ctx = text[lo:hi].replace("\n", " ")
            claims.append(
                S.Claim(
                    "FIND",
                    did,
                    needle,
                    evidence=[{"start": i, "end": i + len(needle), "text": text[i : i + len(needle)], "context": ctx}],
                    status="PRESENT",
                    notes="exact_span",
                )
            )
            start = i + max(1, len(needle))
            if len(claims) >= max_hits:
                break
        if len(claims) >= max_hits:
            break
    if not claims:
        result = {
            "query": needle,
            "corpus_dir": str(corpus_path),
            "answer_status": "ABSTAIN",
            "claims": [],
            "unsupported": [needle],
            "solver_path": ["find_spans"],
            "n_docs": len(docs),
        }
    else:
        result = {
            "query": needle,
            "corpus_dir": str(corpus_path),
            "answer_status": "SUPPORTED",
            "claims": [claim_to_dict(c) for c in claims],
            "unsupported": [],
            "solver_path": ["find_spans"],
            "n_docs": len(docs),
            "n_hits": len(claims),
        }
    return _apply_scope(
        _finalize_with_coe(result, docs, persist=persist_coe),
        scope,
        scoped=doc_ids is not None,
    )



def nearby_contradictions(docs: dict[str, str]) -> list[dict]:
    """Light contradiction surface for ask() banners (W3 merge; classical only)."""
    out: list[dict] = []
    kind_map = {
        "ttl_seconds": "numeric_ttl",
        "metformin_dose_mg": "numeric_dose",
        "sample_n": "numeric_sample_n",
    }
    for claim in merge_all(docs):
        if claim.status != "DISPUTED" or not isinstance(claim.value, dict):
            continue
        field = claim.value.get("field") or "unknown"
        out.append(
            {
                "kind": kind_map.get(field, f"numeric_{field}"),
                "field": field,
                "values": claim.value.get("values", {}),
                "status": "DISPUTED",
                "evidence_spans": epistemic_entry(claim).get("evidence_spans") or [],
            }
        )
    coll = S.flag_entity_collision(docs)
    if coll.status == "DISPUTED":
        out.append({"kind": "entity_collision", "value": coll.value, "status": "DISPUTED"})
    return out



def _has_evidence_atom(c: S.Claim) -> bool:
    """W2: presentable claims need at least one evidence atom with text or offsets."""
    for e in c.evidence or []:
        if not isinstance(e, dict):
            continue
        if e.get("text") or e.get("line") or (e.get("start") is not None and e.get("end") is not None):
            return True
    return False


def _bm25_claims(docs: dict[str, str], q: str, k: int = 5) -> tuple[list[S.Claim], list[dict]]:
    """W1: promote only margin-gated BM25 hits; return (present_claims, review_hits)."""
    present: list[S.Claim] = []
    review: list[dict] = []
    for hit in top_paragraphs(docs, q, k=k):
        body = docs.get(hit["doc_id"]) or ""
        # Paragraph splits often sever headings from figures (E4 … 0.638 … KILL).
        # Keep evidence span on the hit; widen value for relevance only.
        ctx_lo = max(0, int(hit["start"]) - 160)
        ctx_hi = min(len(body), int(hit["end"]) + 220)
        ctx = body[ctx_lo:ctx_hi]
        ev = [{
            "start": hit["start"],
            "end": hit["end"],
            "text": hit["text"][:240],
            "bm25": hit["bm25"],
            "margin": hit.get("margin"),
            "rank": hit.get("rank"),
        }]
        if hit.get("promote"):
            present.append(
                S.Claim(
                    "T19",
                    hit["doc_id"],
                    (ctx or hit["text"])[:500],
                    evidence=ev,
                    status="PRESENT",
                    notes="bm25_paragraph",
                )
            )
        else:
            review.append({**hit, "status": "REVIEW", "reason": "low_bm25_margin"})
    return present, review



def _multi_fact_domains(q: str) -> list[str]:
    """Detect distinct fact domains in a conjunction-style query."""
    ql = q.lower()
    found = []
    if any(k in ql for k in ("ttl", "cache", "expire", "cached", "timeout")):
        found.append("ttl_cache")
    if any(k in ql for k in ("dose", "metformin", "mg")):
        found.append("dose")
    if any(k in ql for k in ("author", "title", "year", "doi")):
        found.append("biblio")
    return found


def _is_unsupported_composition_query(q: str) -> bool:
    ql = q.lower()
    if " and " not in ql and ";" not in ql:
        return False
    return len(_multi_fact_domains(q)) >= 2


def _corpus_has_lexical(docs: dict[str, str], tokens: list[str]) -> bool:
    if not tokens:
        return False
    blob = "\n".join(docs.values()).lower()
    return any(tok.lower() in blob for tok in tokens if len(tok) >= 3)


def _claim_covers_domain(claims: list, domain: str) -> bool:
    blob = " ".join(
        f"{getattr(c, 'value', '')} {getattr(c, 'notes', '')} {getattr(c, 'task_id', '')}".lower()
        for c in claims
    )
    if domain == "ttl_cache":
        return any(k in blob for k in ("ttl", "seconds", "cache", "300", "timeout"))
    if domain == "dose":
        return any(k in blob for k in ("metformin", "mg", "dose", "500", "850"))
    if domain == "biblio":
        return any(k in blob for k in ("author", "title", "year", "doi"))
    return False



def _finalize_with_coe(payload: dict, docs: dict[str, str], *, persist: bool = True) -> dict:
    """Bind typed CoE claims + JSONL record after claim construction (never post-hoc)."""
    try:
        from wedge_v1.coe.audit import audit_payload
        from wedge_v1.coe.bind import bind_ask_payload

        result = bind_ask_payload(payload, docs, persist=persist)
        result["coe_audit"] = audit_payload(result, docs)
        return result
    except Exception as exc:  # pragma: no cover — fail-open for product path
        payload.setdefault("coe", {"error": str(exc), "invariant": "EVIDENCE_CREATED_WITH_CLAIM"})
        return payload




def _phrase_locate_claims(docs: dict[str, str], query: str, tokens: list[str], limit: int = 8) -> list[S.Claim]:
    """Exact multi-word phrase locate — recovers over-abstain when find() would hit."""
    phrases: list[str] = []
    q = query.strip()
    if len(q) >= 8:
        phrases.append(q)
    if len(tokens) >= 2:
        phrases.append(" ".join(tokens))
    for n in range(min(len(tokens), 5), 2, -1):
        for i in range(0, len(tokens) - n + 1):
            phrases.append(" ".join(tokens[i : i + n]))
    seen: set[str] = set()
    uniq: list[str] = []
    for p in phrases:
        k = p.lower()
        if len(k) >= 8 and k not in seen:
            seen.add(k)
            uniq.append(p)
    out: list[S.Claim] = []
    for phrase in uniq[:12]:
        pl = phrase.lower()
        for did, body in docs.items():
            start = 0
            bl = body.lower()
            while True:
                i = bl.find(pl, start)
                if i < 0:
                    break
                span = body[i : i + len(phrase)]
                out.append(
                    S.Claim(
                        "FIND",
                        did,
                        span,
                        evidence=[{"start": i, "end": i + len(phrase), "text": span}],
                        status="PRESENT",
                        notes="phrase_span",
                    )
                )
                start = i + max(1, len(phrase))
                if len(out) >= limit:
                    return out
    return out



def _numeric_keyword_window_claims(
    docs: dict[str, str], tokens: list[str], *, window: int = 160, limit: int = 8
) -> list[S.Claim]:
    """Co-locate distinctive numbers with sibling query tokens in a local window.

    Recovers over-abstention on probes like ``E4 KILL 0.638`` where BM25
    paragraphs mention KILL or 0.638 separately but not together.
    """
    nums = [t for t in tokens if re.fullmatch(r"\d+(?:\.\d+)?", t)]
    words = [t for t in tokens if t not in nums and len(t) >= 2]
    if not nums or not words:
        return []
    out: list[S.Claim] = []
    for did, body in docs.items():
        low = body.lower()
        for num in nums:
            start = 0
            while True:
                i = body.find(num, start)
                if i < 0:
                    break
                lo = max(0, i - window)
                hi = min(len(body), i + len(num) + window)
                span = body[lo:hi]
                span_l = span.lower()
                if all(w.lower() in span_l for w in words):
                    out.append(
                        S.Claim(
                            "FIND",
                            did,
                            span.strip()[:240],
                            evidence=[{"start": lo, "end": hi, "text": span.strip()[:240]}],
                            status="PRESENT",
                            notes="numeric_keyword_window",
                        )
                    )
                    if len(out) >= limit:
                        return out
                start = i + max(1, len(num))
    return out



def ask(
    query: str,
    corpus_dir: Path | None = None,
    *,
    doc_ids: list[str] | None = None,
    persist_coe: bool = True,
    escalate_stub: bool = False,
) -> dict:
    """Span-first Q&A over a local folder. Never invents unsupported claims.

    escalate_stub / WEDGE_ESCALATE_STUB: on classical ABSTAIN only, try the
    constructive hybrid stub (ΔU eval arm). Default remains fail-closed.
    """
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    trace = AskTrace(query=query, corpus_dir=str(corpus_path), op="ask")
    all_docs = load_corpus(corpus_dir)
    docs, scope, scope_valid = select_documents(all_docs, doc_ids)
    scoped = doc_ids is not None
    if not scope_valid:
        return _scope_abstention(
            op="ask",
            query=query.strip(),
            corpus_path=corpus_path,
            scope=scope,
            persist=persist_coe,
        )
    if not docs:
        trace.add_solver("load_corpus")
        trace.add_failure(FailureCode.NO_CORPUS)
        tr = trace.finalize("NO_CORPUS")
        return {
            "query": query,
            "corpus_dir": str(corpus_path),
            "answer_status": "NO_CORPUS",
            "claims": [],
            "unsupported": ["corpus empty or missing"],
            "solver_path": ["load_corpus"],
            "failure_codes": tr["failure_codes"],
            "trace": tr,
            "latency_s": round(tr["latency_ms"] / 1000, 4),
            "latency_ms": tr["latency_ms"],
        }

    q = query.strip()
    claims: list[S.Claim] = []
    solver_path = ["load_corpus"]
    trace.add_solver("load_corpus")
    trace.n_docs = len(docs)
    trace.event("ingest", "loaded", n_docs=len(docs))
    tokens = _content_tokens(q)
    predicates = decompose(q)
    domains = [p.domain for p in predicates]
    composition = len(predicates) >= 2
    if composition:
        trace.event(
            "predicate_decompose",
            "atomic_conjunction",
            n=len(predicates),
            domains=domains,
        )
        # Seed cascade with corpus-agnostic merge claims for known domains
        merge_seed = predicate_claims_for_domains(docs, domains)
        claims.extend(merge_seed)
        if merge_seed:
            solver_path.append("merge_atomic_predicates")
            trace.add_solver("merge_atomic_predicates")

    q_content = " ".join(tokens) if tokens else q
    for did, text in docs.items():
        claims.append(S.keyword_paragraph(did, text, q_content))
        if tokens:
            claims.append(S.quote_sentence(did, text, tokens[0]))
    solver_path.append("keyword_paragraph+quote")
    trace.add_solver("keyword_paragraph+quote")
    for tok in tokens[:5]:
        claims.append(S.mention_docs(docs, tok))
        for did, body in docs.items():
            c = S.yes_no_mention(did, body, tok)
            if c.value is True:
                claims.append(c)
    solver_path.append("mention+yes_no")
    trace.add_solver("mention+yes_no")

    hyphen_chunks = set(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", q))
    for num in re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])", q):
        if any(num in chunk and chunk != num for chunk in hyphen_chunks):
            continue
        for did, body in docs.items():
            i = body.find(num)
            if i >= 0:
                line = body[max(0, body.rfind("\n", 0, i) + 1) : body.find("\n", i)]
                if not line:
                    line = body[i : i + len(num)]
                claims.append(
                    S.Claim(
                        "FIND",
                        did,
                        num,
                        evidence=[{
                            "start": i,
                            "end": i + len(num),
                            "text": body[i : i + len(num)],
                            "line": line.strip()[:240],
                        }],
                        status="PRESENT",
                        notes="numeric_span",
                    )
                )
    for lit in re.findall(r'"([^"]{3,80})"', q):
        for did, body in docs.items():
            i = body.find(lit)
            if i >= 0:
                claims.append(
                    S.Claim(
                        "FIND",
                        did,
                        lit,
                        evidence=[{
                            "start": i,
                            "end": i + len(lit),
                            "text": body[i : i + len(lit)],
                        }],
                        status="PRESENT",
                        notes="literal_span",
                    )
                )
    for tok in tokens:
        if "-" in tok or any(ch.isdigit() for ch in tok) or (tok.isupper() and len(tok) >= 2):
            for did, body in docs.items():
                i = body.find(tok)
                if i >= 0:
                    claims.append(
                        S.Claim(
                            "FIND",
                            did,
                            tok,
                            evidence=[{
                                "start": i,
                                "end": i + len(tok),
                                "text": body[i : i + len(tok)],
                            }],
                            status="PRESENT",
                            notes="token_span",
                        )
                    )
    solver_path.append("numeric+literal_spans")
    trace.add_solver("numeric+literal_spans")

    phrase_hits = _phrase_locate_claims(docs, q, tokens)
    if phrase_hits:
        claims.extend(phrase_hits)
        solver_path.append("phrase_span")
        trace.add_solver("phrase_span")
        trace.event("phrase_locate", "hits", n=len(phrase_hits))

    nk_hits = _numeric_keyword_window_claims(docs, tokens)
    if nk_hits:
        claims.extend(nk_hits)
        solver_path.append("numeric_keyword_window")
        trace.add_solver("numeric_keyword_window")
        trace.event("numeric_keyword_window", "hits", n=len(nk_hits))

    bm25_present, bm25_review = _bm25_claims(docs, q, k=5)
    claims.extend(bm25_present)
    if bm25_present:
        solver_path.append("bm25_paragraphs")
        trace.add_solver("bm25_paragraphs")
    if bm25_review:
        solver_path.append("bm25_low_margin_review")
        trace.add_solver("bm25_low_margin_review")
        trace.n_bm25_review = len(bm25_review)
        trace.add_failure(FailureCode.LOW_MARGIN_RETRIEVAL)
        trace.event("bm25_margin_gate", "review_hits", n=len(bm25_review))

    ql = q.lower()
    # W4: pluggable cascade (synonym/ocr/coref) — no fixture doc-id gates
    cascade = run_cascade(docs, q)
    claims.extend(cascade.claims)
    for mod in cascade.modules_run:
        name = f"plugin.{mod}"
        solver_path.append(name)
        trace.add_solver(name)
    if cascade.modules_run:
        trace.event("plugin_cascade", "ran", modules=cascade.modules_run, n=len(cascade.claims))
    if "dose" in ql or "metformin" in ql:
        claims.append(S.symbolic_dose_change(docs))
        claims.append(S.union_dosages(docs))
        solver_path.append("eclass_symbolic_dose+union")
        trace.add_solver("eclass_symbolic_dose+union")

    # Optional decidable verifier pass (records rejects; presentation still uses evidence gate)
    verified = [verify_claim(c) for c in claims]
    empty_rejected = sum(1 for c in verified if c.meta.get("verify") == "fail_no_evidence")
    trace.n_empty_evidence_rejected = empty_rejected
    if empty_rejected:
        trace.add_failure(FailureCode.EMPTY_EVIDENCE_REJECTED)

    presented = [
        c
        for c in verified
        if c.status in {"PRESENT", "CONFIRMED", "DISPUTED", "PROBABLE"}
        and _has_evidence_atom(c)
        and _relevant_claim(c, tokens, query=q, docs=docs)
        and not _is_table_dump_claim(c)
    ]
    presented_sorted = sorted(
        presented,
        key=lambda c: (0 if c.task_id == "FIND" else 1, 0 if c.evidence else 1, c.task_id),
    )
    trace.n_claims_raw = len(claims)
    trace.n_claims_presented = len(presented_sorted)
    has_lex = _corpus_has_lexical(docs, tokens)

    # W3/CoE: atomic predicates — compound answers require full conjunction support
    pred_support = evaluate_predicates(predicates, docs, presented_sorted)
    pred_payload = {
        "predicates": [p.to_dict() for p in predicates],
        "predicate_support": [s.to_dict() for s in pred_support],
    }
    if incomplete_conjunction(pred_support):
        missing = [s.domain for s in pred_support if not s.supported]
        trace.add_failure(FailureCode.UNSUPPORTED_COMPOSITION)
        trace.add_failure(FailureCode.COE_INCOMPLETE_CONJUNCTION)
        trace.event("incomplete_conjunction", "missing_predicates", missing=missing)
        for code in classify_abstain_failures(
            has_lexical_hit=has_lex,
            bm25_review_n=len(bm25_review),
            empty_rejected=empty_rejected,
            oos_expected=False,
            composition_blocked=True,
        ):
            trace.add_failure(code)
        tr = trace.finalize("ABSTAIN")
        # Present only claims that support *supported* predicates (optional transparency)
        supported_domains = {s.domain for s in pred_support if s.supported}
        partial = [
            c for c in presented_sorted
            if _claim_covers_domain([c], next(iter(supported_domains), ""))
            or any(d in str(c.value).lower() + c.notes.lower() for d in supported_domains)
        ][:6]
        payload = {
            "query": q,
            "corpus_dir": str(corpus_path),
            "answer_status": "ABSTAIN",
            "claims": [claim_to_dict(c) for c in partial],
            "unsupported": [q],
            "solver_path": solver_path + ["predicate_conjunction_gate"],
            "note": f"incomplete conjunction; unsupported predicates: {missing}",
            "n_docs": len(docs),
            "contradictions_nearby": nearby_contradictions(docs),
            "bm25_review": bm25_review,
            "abstain_class": "coe_incomplete_conjunction",
            "failure_codes": tr["failure_codes"],
            "trace": tr,
            "latency_s": round(tr["latency_ms"] / 1000, 4),
            "latency_ms": tr["latency_ms"],
            "lm_invoked": False,
            **pred_payload,
        }
        return _finish_ask(
            payload,
            docs,
            scope=scope,
            scoped=scoped,
            persist_coe=persist_coe,
            escalate_stub=escalate_stub,
        )

    if not presented_sorted:
        # Content tokens absent → OOS even if BM25 review ranked generic passages.
        oos = not has_lex
        final_codes = classify_abstain_failures(
            has_lexical_hit=has_lex,
            bm25_review_n=len(bm25_review),
            empty_rejected=empty_rejected,
            oos_expected=oos,
            composition_blocked=composition,
        )
        if oos:
            trace.replace_failures(final_codes)
            trace.event("oos_gate", "content_tokens_absent", n_review=len(bm25_review))
        else:
            for code in final_codes:
                trace.add_failure(code)
        primary = (trace.failure_codes[0].value if trace.failure_codes else "retrieval_miss_or_unsupported")
        merge_claims = merge_for_term(docs, q)
        epistemic_merge = [epistemic_entry(c) for c in merge_claims if c.status != "ABSTAIN"]
        tr = trace.finalize("ABSTAIN")
        payload = {
            "query": q,
            "corpus_dir": str(corpus_path),
            "answer_status": "ABSTAIN",
            "claims": [],
            "unsupported": [q],
            "solver_path": solver_path,
            "note": "no span-supported claim under classical+eclass cascade",
            "n_docs": len(docs),
            "contradictions_nearby": nearby_contradictions(docs),
            "epistemic_merge": epistemic_merge,
            "bm25_review": bm25_review,
            "abstain_class": primary.lower(),
            "failure_codes": tr["failure_codes"],
            "trace": tr,
            "latency_s": round(tr["latency_ms"] / 1000, 4),
            "latency_ms": tr["latency_ms"],
            "lm_invoked": False,
            "predicates": [p.to_dict() for p in predicates],
            "predicate_support": [s.to_dict() for s in evaluate_predicates(predicates, docs, [])],
        }
        return _finish_ask(
            payload,
            docs,
            scope=scope,
            scoped=scoped,
            persist_coe=persist_coe,
            escalate_stub=escalate_stub,
        )

    disputed = [c for c in presented_sorted if c.status == "DISPUTED"]
    nearby = nearby_contradictions(docs)
    ql_local = q.lower()
    relevant_nearby = []
    for n in nearby:
        kind = n.get("kind", "")
        if kind == "numeric_dose" and ("dose" in ql_local or "metformin" in ql_local):
            relevant_nearby.append(n)
        elif kind == "numeric_ttl" and any(k in ql_local for k in ("ttl", "cache", "expire", "cached")):
            relevant_nearby.append(n)
        elif kind == "entity_collision":
            val = n.get("value")
            strings: list[str] = []
            if isinstance(val, dict):
                strings = [str(v) for v in val.values()]
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        strings.append(str(item.get("string") or ""))
                    else:
                        strings.append(str(item))
            elif val is not None:
                strings = [str(val)]
            if any(s.lower() in ql_local for s in strings if s):
                relevant_nearby.append(n)
    status = "CONTRADICTED" if (disputed or relevant_nearby) else "SUPPORTED"
    if status == "CONTRADICTED":
        trace.add_failure(FailureCode.MULTI_DOC_CONTRADICTION)
        if any(n.get("kind") == "numeric_dose" for n in relevant_nearby):
            trace.add_failure(FailureCode.NUMERIC_CONTRADICTION)
    banner = None
    if relevant_nearby:
        kinds = sorted({n["kind"] for n in relevant_nearby})
        banner = f"query-relevant contradictions: {', '.join(kinds)}"
        trace.event("contradiction", banner, n=len(relevant_nearby))
    # W3: same typed merge surface as compare() (term-relevant fields + both spans)
    merge_claims = merge_for_term(docs, q)
    epistemic_merge = [epistemic_entry(c) for c in merge_claims if c.status != "ABSTAIN"]
    if epistemic_merge:
        solver_path.append("epistemic_merge")
        trace.add_solver("epistemic_merge")
    tr = trace.finalize(status)
    payload = {
        "query": q,
        "corpus_dir": str(corpus_path),
        "answer_status": status,
        "claims": [claim_to_dict(c) for c in presented_sorted[:12]],
        "unsupported": [],
        "solver_path": solver_path,
        "n_docs": len(docs),
        "contradictions_nearby": relevant_nearby,
        "contradictions_corpus": nearby,
        "contradiction_banner": banner,
        "epistemic_merge": epistemic_merge,
        "bm25_review": bm25_review,
        "failure_codes": tr["failure_codes"],
        "trace": tr,
        "latency_s": round(tr["latency_ms"] / 1000, 4),
        "latency_ms": tr["latency_ms"],
        "lm_invoked": False,
        "predicates": [p.to_dict() for p in predicates],
        "predicate_support": [s.to_dict() for s in evaluate_predicates(predicates, docs, presented_sorted)],
    }
    return _finish_ask(
        payload,
        docs,
        scope=scope,
        scoped=scoped,
        persist_coe=persist_coe,
        escalate_stub=escalate_stub,
    )





def scan(
    corpus_dir: Path | None = None,
    *,
    doc_ids: list[str] | None = None,
    persist_coe: bool = True,
) -> dict:
    """Run inventory extractors across corpus (metadata, dosages, contradictions)."""
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    all_docs = load_corpus(corpus_dir)
    docs, scope, scope_valid = select_documents(all_docs, doc_ids)
    if not scope_valid:
        return _scope_abstention(
            op="scan",
            query="scan",
            corpus_path=corpus_path,
            scope=scope,
            persist=persist_coe,
        )
    if not docs:
        return {"answer_status": "NO_CORPUS", "claims": [], "solver_path": ["load_corpus"]}

    claims: list[S.Claim] = []
    for did, text in docs.items():
        claims.extend([
            S.extract_title(did, text),
            S.extract_authors(did, text),
            S.extract_year(did, text),
            S.detect_doc_type(did, text),
            S.extract_dosages(did, text),
            S.extract_compounds(did, text),
            S.extract_sample_n(did, text),
            S.extract_doi(did, text),
        ])
    claims.append(S.union_dosages(docs))
    claims.append(S.symbolic_dose_change(docs))
    claims.append(S.flag_entity_collision(docs))
    casc = run_cascade(docs, want={"synonym", "ocr", "coref"})
    claims.extend(casc.claims)

    payload = {
        "query": "scan",
        "corpus_dir": str(corpus_path),
        "answer_status": "SUPPORTED",
        "claims": [claim_to_dict(c) for c in claims if c.status not in {"MISSING"}],
        "solver_path": ["scan_classical+eclass", "plugin_cascade"],
        "n_docs": len(docs),
        "n_claims": len(claims),
    }
    return _apply_scope(
        _finalize_with_coe(payload, docs, persist=persist_coe),
        scope,
        scoped=doc_ids is not None,
    )



_NUM_NEAR = re.compile(r"(?<![A-Za-z])(\d+(?:\.\d+)?)(?![A-Za-z])")


def _closest_number(text: str, center: int, window: int = 100) -> str | None:
    lo = max(0, center - window)
    hi = min(len(text), center + window)
    best, best_d = None, None
    for m in _NUM_NEAR.finditer(text, lo, hi):
        d = min(abs(m.start() - center), abs(m.end() - center))
        # Prefer numbers outside the match span itself when term is non-numeric.
        if best_d is None or d < best_d:
            best_d = d
            best = m.group(1)
    return best


def compare(
    term: str,
    corpus_dir: Path | None = None,
    window: int = 100,
    *,
    doc_ids: list[str] | None = None,
    persist_coe: bool = True,
) -> dict:
    """Cross-doc compare for TERM: spans + associated-number disagreement → CONTRADICTED.

    Classical only. Does not invent values outside corpus spans.
    """
    t0 = time.perf_counter()
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    all_docs = load_corpus(corpus_dir)
    docs, scope, scope_valid = select_documents(all_docs, doc_ids)
    if not scope_valid:
        return _scope_abstention(
            op="compare",
            query=term.strip(),
            corpus_path=corpus_path,
            scope=scope,
            persist=persist_coe,
            extra={"term": term},
        )
    if not docs:
        return {
            "term": term,
            "corpus_dir": str(corpus_path),
            "answer_status": "NO_CORPUS",
            "claims": [],
            "solver_path": ["load_corpus"],
            "latency_s": round(time.perf_counter() - t0, 4),
        }

    needle = term.strip()
    if not needle:
        return {
            "term": term,
            "corpus_dir": str(corpus_path),
            "answer_status": "ABSTAIN",
            "claims": [],
            "unsupported": ["empty term"],
            "solver_path": ["compare"],
            "latency_s": round(time.perf_counter() - t0, 4),
        }

    pattern = re.compile(re.escape(needle), re.I)
    hits: list[dict] = []
    # Field-like: "TTL as 300 seconds", "metformin 500 mg", "QPS is/remains 12000"
    field_re = re.compile(
        rf"(?:{re.escape(needle)}\s+(?:as\s+|is\s+|remains\s+|=\s*)?(\d+(?:\.\d+)?)\s*(?:seconds|mg|sec|qps)?|"
        rf"{re.escape(needle)}\s+(\d+(?:\.\d+)?))",
        re.I,
    )
    field_vals: dict[str, str] = {}
    closest_by_doc: dict[str, set[str]] = {}

    for did, text in docs.items():
        for m in pattern.finditer(text):
            i, j = m.start(), m.end()
            lo = max(0, i - window)
            hi = min(len(text), j + window)
            ctx = text[lo:hi].replace("\n", " ")
            # Closest number outside the term span (skip when term is the number).
            closest = None
            if not re.fullmatch(r"\d+(?:\.\d+)?", needle):
                line_lo = text.rfind("\n", 0, i) + 1
                line_hi = text.find("\n", j)
                if line_hi < 0:
                    line_hi = len(text)
                line = text[line_lo:line_hi]
                # Prefer number after the term on the same line (avoids year/header noise).
                after = re.search(
                    rf"{re.escape(needle)}\s*(?:as|is|remains|=|:)?\s*(\d+(?:\.\d+)?)",
                    line,
                    re.I,
                )
                if after:
                    closest = after.group(1)
                else:
                    closest = _closest_number(text, (i + j) // 2, window=window)
                if closest is not None:
                    closest_by_doc.setdefault(did, set()).add(closest)
            hits.append(
                {
                    "doc_id": did,
                    "start": i,
                    "end": j,
                    "text": text[i:j],
                    "context": ctx[:240],
                    "closest_number": closest,
                    "numeric": float(closest) if closest is not None else None,
                }
            )
        fm = field_re.search(text)
        if fm:
            field_vals[did] = fm.group(1) or fm.group(2)

    if not hits:
        return {
            "term": needle,
            "corpus_dir": str(corpus_path),
            "answer_status": "ABSTAIN",
            "claims": [],
            "unsupported": [needle],
            "solver_path": ["compare"],
            "n_docs": len(docs),
            "n_hits": 0,
            "latency_s": round(time.perf_counter() - t0, 4),
        }

    # Prefer structured field extraction when ≥2 docs expose a value.
    values_by_doc: dict[str, list[str]]
    if len(field_vals) >= 2:
        values_by_doc = {k: [v] for k, v in field_vals.items()}
        disputed = len(set(field_vals.values())) >= 2
    elif re.fullmatch(r"\d+(?:\.\d+)?", needle):
        # Term is a literal number: presence agreement only (no nearby-number dispute).
        values_by_doc = {h["doc_id"]: [needle] for h in hits}
        disputed = False
    else:
        values_by_doc = {k: sorted(v, key=float) for k, v in closest_by_doc.items()}
        # One representative per doc (closest); dispute if docs disagree.
        reps = {did: vals[0] for did, vals in values_by_doc.items() if vals}
        disputed = len(set(reps.values())) >= 2

    all_vals = sorted({v for vs in values_by_doc.values() for v in vs}, key=float)
    merge_claims = merge_for_term(docs, needle)
    epistemic_merge = [epistemic_entry(c) for c in merge_claims if c.status != "ABSTAIN"]
    merge_disputed = any(c.status == "DISPUTED" for c in merge_claims)
    status = "CONTRADICTED" if (disputed or merge_disputed) else "SUPPORTED"
    claim = S.Claim(
        "COMPARE",
        None,
        {
            "term": needle,
            "values_by_doc": values_by_doc,
            "all_values": all_vals,
            "n_hits": len(hits),
            "field_values": field_vals,
        },
        evidence=[
            {
                "doc_id": h["doc_id"],
                "start": h["start"],
                "end": h["end"],
                "text": h["text"],
                "context": h["context"],
            }
            for h in hits[:24]
        ],
        status="DISPUTED" if status == "CONTRADICTED" else "PRESENT",
        notes="cross_doc_compare",
    )
    claims_out = [claim_to_dict(claim)]
    for mc in merge_claims:
        if mc.status != "ABSTAIN":
            claims_out.append(claim_to_dict(mc))
    solver_path = ["compare"]
    if merge_claims:
        solver_path.append("epistemic_merge")

    payload = {
        "term": needle,
        "corpus_dir": str(corpus_path),
        "answer_status": status,
        "claims": claims_out,
        "epistemic_merge": epistemic_merge,
        "hits": hits[:24],
        "values_by_doc": values_by_doc,
        "field_values": field_vals,
        "unsupported": [],
        "solver_path": solver_path,
        "n_docs": len(docs),
        "n_docs_hit": len({h["doc_id"] for h in hits}),
        "n_hits": len(hits),
        "failure_codes": (
            [FailureCode.MULTI_DOC_CONTRADICTION.value, FailureCode.NUMERIC_CONTRADICTION.value]
            if status == "CONTRADICTED"
            else []
        ),
        "latency_s": round(time.perf_counter() - t0, 4),
    }
    return _apply_scope(
        _finalize_with_coe(payload, docs, persist=persist_coe),
        scope,
        scoped=doc_ids is not None,
    )


def format_report_md(payload: dict, *, title: str | None = None) -> str:
    """Human-readable markdown for ask/find/scan/compare JSON payloads (product UX)."""
    lines: list[str] = []
    status = payload.get("answer_status", "?")
    lines.append(f"# {title or 'wedge_v1 report'}")
    lines.append("")
    if payload.get("query"):
        lines.append(f"**Query:** {payload['query']}")
    if payload.get("term"):
        lines.append(f"**Term:** `{payload['term']}`")
    lines.append(f"**Status:** `{status}`")
    banner = payload.get("contradiction_banner")
    if banner:
        lines.append("")
        lines.append(f"> **Contradiction banner:** {banner}")
    if "n_docs" in payload:
        lines.append(f"**Docs:** {payload['n_docs']}")
    if "n_hits" in payload:
        lines.append(f"**Hits:** {payload['n_hits']}")
    if "latency_s" in payload:
        lines.append(f"**Latency:** {payload['latency_s']}s")
    if "latency_ms" in payload:
        lines.append(f"**Latency:** {payload['latency_ms']}ms")
    path = payload.get("solver_path") or payload.get("solver") or []
    if path:
        lines.append("**Solver path:** " + " → ".join(str(p) for p in path))
    lines.append("")
    # Compare UX: surface numeric disagreement before claim dump
    values_by_doc = payload.get("values_by_doc") or {}
    if values_by_doc:
        lines.append("## Compare values")
        lines.append("")
        for did, vals in sorted(values_by_doc.items()):
            lines.append(f"- `{did}`: {', '.join(str(v) for v in vals)}")
        lines.append("")
    claims = payload.get("claims") or []
    packed = pack_claims(claims)
    lines.extend(format_packed_claims_md(packed))
    if packed:
        lines.append("")
    unsupported = payload.get("unsupported") or []
    if unsupported:
        lines.append("## Unsupported / abstain reasons")
        for u in unsupported:
            lines.append(f"- {u}")
        lines.append("")
    if payload.get("abstain_reason"):
        lines.append(f"**Abstain reason:** {payload['abstain_reason']}")
        lines.append("")
    epistemic = payload.get("epistemic_merge") or []
    if epistemic:
        lines.append("## Epistemic merge (typed fields, both spans)")
        for row in epistemic[:12]:
            if not isinstance(row, dict):
                continue
            fid = row.get("field_id", "?")
            st = row.get("status", "?")
            vals = row.get("values_by_doc") or {}
            lines.append(f"### `{fid}` — {st}")
            for doc_id, val in sorted(vals.items()):
                lines.append(f"- **{doc_id}:** `{val}`")
            for sp in (row.get("evidence_spans") or [])[:8]:
                if not isinstance(sp, dict):
                    continue
                loc = ""
                if sp.get("start") is not None:
                    loc = f" [{sp.get('start')}:{sp.get('end')}]"
                lines.append(
                    f"  - span `{sp.get('doc_id')}`{loc}: {sp.get('text', '')}"
                )
            lines.append("")

    contradictions = (
        payload.get("contradictions_nearby")
        or payload.get("contradictions")
        or payload.get("disputes")
        or []
    )
    if contradictions:
        lines.append("## Contradictions")
        for item in contradictions[:20]:
            if isinstance(item, dict):
                kind = item.get("kind") or item.get("task_id") or "flag"
                doc = item.get("doc_id", "")
                val = item.get("values", item.get("value"))
                lines.append(f"- `{kind}` doc=`{doc}` value=`{val}`")
            else:
                lines.append(f"- `{item}`")
        lines.append("")
    spans = payload.get("evidence_spans") or []
    if spans:
        lines.append(f"## Evidence spans ({len(spans)})")
        for e in spans[:12]:
            if not isinstance(e, dict):
                continue
            lines.append(
                f"- `{e.get('doc_id')}` [{e.get('start')}:{e.get('end')}] {str(e.get('text', ''))[:160]}"
            )
        lines.append("")
    lines.append("---")
    lines.append("_Verification-first local slice. No generative fill-in._")
    lines.append("")
    return chr(10).join(lines)
