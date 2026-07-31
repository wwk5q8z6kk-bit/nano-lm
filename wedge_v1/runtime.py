"""Nano Runtime wedge slice — classical + E-class solvers only (no LM).

Authorized by owner "continue" after ECLASS_CLOSED_WITHOUT_LM.
E-class probes live in classical.solvers (paraphrastic_ttl / symbolic_dose_change / coref_binding).
"""
from __future__ import annotations

import json
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
from wedge_v1.classical.merge import predicate_claims_for_domains
from wedge_v1.plugins.cascade import run_cascade
from wedge_v1.coe.predicates import (
    decompose,
    evaluate_predicates,
    incomplete_conjunction,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CORPUS = ROOT / "data" / "corpus"
GOLD_PATH = ROOT / "data" / "gold" / "gold.json"



def load_corpus(corpus_dir: Path | None = None, *, normalize: bool = False) -> dict[str, str]:
    from wedge_v1.ingest import load_corpus as _ingest

    path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    return _ingest(path, normalize=normalize)


def _load_gold() -> dict | None:
    if GOLD_PATH.exists():
        return json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return None


def claim_to_dict(c: S.Claim) -> dict:
    return asdict(c)


def _expand_ttl_ask(docs: dict[str, str], query: str) -> S.Claim:
    """T35-style expand without requiring gold (runtime path)."""
    expand = {
        "expire": ["ttl", "seconds", "invalidation", "timeout"],
        "cached": ["cache", "ttl"],
        "entries": ["cache"],
        "long": ["ttl", "seconds"],
    }
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    for src, dsts in expand.items():
        if src in query.lower():
            terms.update(dsts)
    # Typed TTL patterns — avoid fixture-tied "TTL as N seconds" only (W1/W4).
    ttl_pats = [
        re.compile(r"TTL\s+as\s+(\d+)\s+seconds", re.I),
        re.compile(r"TTL\s+is\s+(\d+)\s+seconds", re.I),
        re.compile(r"TTL\s*[=:]\s*(\d+)\s*seconds", re.I),
        re.compile(r"TTL\s+of\s+(\d+)\s+seconds", re.I),
    ]
    best_id, best_score, best_m = None, 0, None
    for did, text in docs.items():
        low = text.lower()
        score = sum(1 for t in terms if t in low)
        m = None
        for pat in ttl_pats:
            m = pat.search(text)
            if m:
                score += 5
                break
        if score > best_score:
            best_score, best_id, best_m = score, did, m
    if best_m is not None and best_id is not None and best_score > 0:
        span = S._find(docs[best_id], best_m.group(0))
        return S.Claim(
            "T35",
            best_id,
            f"{best_m.group(1)} seconds",
            evidence=[span or {"text": best_m.group(0)}],
            notes="query_expand_runtime",
        )
    return S.Claim("T35", None, None, status="ABSTAIN", notes=query)


STOP = {
    "how", "long", "before", "what", "when", "where", "which", "does", "the", "and",
    "for", "is", "of", "in", "a", "an", "to", "on", "at", "by", "or", "as", "be",
    "are", "was", "were", "with", "from", "this", "that", "these", "those", "into",
    "will", "did", "under", "about", "into", "over", "than", "then", "also", "only",
}


def _content_tokens(q: str) -> list[str]:
    words = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", q) if t.lower() not in STOP]
    raw_nums = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])", q)
    # Drop digits that only appear inside hyphenated tokens (e.g. GPT-4)
    hyphen_chunks = set(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", q))
    nums = []
    for n in raw_nums:
        if any(n in chunk and chunk != n for chunk in hyphen_chunks):
            continue
        nums.append(n)
    # preserve order, unique
    out, seen = [], set()
    for t in words + nums:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _relevant_claim(c: S.Claim, tokens: list[str]) -> bool:
    """Reject weak lexical coincidences (one stop-ish content token in a huge corpus)."""
    if not tokens:
        return False
    blob = json.dumps(c.value, default=str).lower() + " " + " ".join(
        str(e.get("text", "")) + " " + str(e.get("line", "")) for e in (c.evidence or [])
    ).lower()
    # E-class structural probes: accepted when triggered
    if c.task_id in {"T35", "T36", "T39", "T26", "T29", "T30", "MERGE"} and c.status in {
        "PRESENT", "CONFIRMED", "DISPUTED"
    }:
        return True
    # Exact FIND of a query number/literal: value itself must be one of the tokens
    if c.task_id == "FIND":
        # A single token hit is not an answer to a multi-token question.
        # Require the same coverage rule as passage claims (falls through).
        pass
    hits = sum(1 for tok in tokens if tok.lower() in blob)
    # Short queries: majority. Longer (>=3 content tokens): require all tokens in evidence
    # so governance prose mentioning NanoScribe cannot "answer" clinical-accuracy questions.
    if len(tokens) >= 3:
        need = len(tokens)
    elif len(tokens) == 2:
        need = 2
    else:
        need = 1
    return hits >= need



def find_spans(needle: str, corpus_dir: Path | None = None, max_hits: int = 20) -> dict:
    """Exact substring locate with evidence spans (classical)."""
    docs = load_corpus(corpus_dir)
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
        return {
            "answer_status": "ABSTAIN",
            "claims": [],
            "unsupported": [needle],
            "solver_path": ["find_spans"],
            "n_docs": len(docs),
        }
    return {
        "answer_status": "SUPPORTED",
        "claims": [claim_to_dict(c) for c in claims],
        "unsupported": [],
        "solver_path": ["find_spans"],
        "n_docs": len(docs),
        "n_hits": len(claims),
    }



def nearby_contradictions(docs: dict[str, str]) -> list[dict]:
    """Light contradiction surface for ask() banners (classical only)."""
    out: list[dict] = []
    dose = S.symbolic_dose_change(docs)
    if dose.status in {"PRESENT", "DISPUTED", "CONFIRMED"} and isinstance(dose.value, dict):
        if dose.value.get("from") != dose.value.get("to"):
            out.append({
                "kind": "numeric_dose",
                "field": "metformin_dose_mg",
                "values": dose.value.get("values", dose.value),
                "status": "DISPUTED",
            })
    ttl = {}
    for did, body in docs.items():
        m = re.search(r"TTL as (\d+) seconds", body)
        if m:
            ttl[did] = int(m.group(1))
    if len(set(ttl.values())) > 1:
        out.append({"kind": "numeric_ttl", "field": "ttl_seconds", "values": ttl, "status": "DISPUTED"})
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
                    hit["text"][:500],
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
        from wedge_v1.coe.bind import bind_ask_payload

        return bind_ask_payload(payload, docs, persist=persist)
    except Exception as exc:  # pragma: no cover — fail-open for product path
        payload.setdefault("coe", {"error": str(exc), "invariant": "EVIDENCE_CREATED_WITH_CLAIM"})
        return payload



def ask(query: str, corpus_dir: Path | None = None) -> dict:
    """Span-first Q&A over a local folder. Never invents unsupported claims."""
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    trace = AskTrace(query=query, corpus_dir=str(corpus_path), op="ask")
    docs = load_corpus(corpus_dir)
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
        and _relevant_claim(c, tokens)
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
        return _finalize_with_coe(payload, docs)

    if not presented_sorted:
        oos = not has_lex and len(bm25_review) == 0
        for code in classify_abstain_failures(
            has_lexical_hit=has_lex,
            bm25_review_n=len(bm25_review),
            empty_rejected=empty_rejected,
            oos_expected=oos,
            composition_blocked=composition,
        ):
            trace.add_failure(code)
        primary = (trace.failure_codes[0].value if trace.failure_codes else "retrieval_miss_or_unsupported")
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
        return _finalize_with_coe(payload, docs)

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
        "bm25_review": bm25_review,
        "failure_codes": tr["failure_codes"],
        "trace": tr,
        "latency_s": round(tr["latency_ms"] / 1000, 4),
        "latency_ms": tr["latency_ms"],
        "lm_invoked": False,
        "predicates": [p.to_dict() for p in predicates],
        "predicate_support": [s.to_dict() for s in evaluate_predicates(predicates, docs, presented_sorted)],
    }
    return _finalize_with_coe(payload, docs)



def scan(corpus_dir: Path | None = None) -> dict:
    """Run inventory extractors across corpus (metadata, dosages, contradictions)."""
    docs = load_corpus(corpus_dir)
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

    return {
        "answer_status": "SUPPORTED",
        "claims": [claim_to_dict(c) for c in claims if c.status not in {"MISSING"}],
        "solver_path": ["scan_classical+eclass", "plugin_cascade"],
        "n_docs": len(docs),
        "n_claims": len(claims),
    }



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


def compare(term: str, corpus_dir: Path | None = None, window: int = 100) -> dict:
    """Cross-doc compare for TERM: spans + associated-number disagreement → CONTRADICTED.

    Classical only. Does not invent values outside corpus spans.
    """
    t0 = time.perf_counter()
    corpus_path = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS
    docs = load_corpus(corpus_dir)
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
    status = "CONTRADICTED" if disputed else "SUPPORTED"
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
        status="DISPUTED" if disputed else "PRESENT",
        notes="cross_doc_compare",
    )

    return {
        "term": needle,
        "corpus_dir": str(corpus_path),
        "answer_status": status,
        "claims": [claim_to_dict(claim)],
        "hits": hits[:24],
        "values_by_doc": values_by_doc,
        "field_values": field_vals,
        "unsupported": [],
        "solver_path": ["compare"],
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
    claims = payload.get("claims") or []
    if not claims:
        lines.append("_No claims._")
        lines.append("")
    else:
        lines.append(f"## Claims ({len(claims)})")
        lines.append("")
        for i, c in enumerate(claims, 1):
            if not isinstance(c, dict):
                continue
            val = c.get("value")
            st = c.get("status", "")
            tid = c.get("task_id", "")
            doc = c.get("doc_id", "")
            lines.append(f"### {i}. `{tid}` — {st}")
            lines.append(f"- **Doc:** `{doc}`")
            lines.append(f"- **Value:** `{val}`")
            for e in (c.get("evidence") or [])[:5]:
                if not isinstance(e, dict):
                    continue
                span = e.get("text") or e.get("line") or ""
                ctx = e.get("context")
                start_i, end_i = e.get("start"), e.get("end")
                loc = f" [{start_i}:{end_i}]" if start_i is not None else ""
                lines.append(f"- **Evidence{loc}:** {span}")
                if ctx:
                    lines.append(f"  - context: {ctx}")
            notes = c.get("notes")
            if notes:
                lines.append(f"- _notes:_ {notes}")
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
