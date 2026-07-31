"""Stdlib Okapi BM25 over paragraphs (no deps). Product retrieval, not science claim."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass


_TOKEN = re.compile(r"[a-z0-9]+", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


_EXPAND = {
    "expire": ["ttl", "timeout", "seconds", "invalidation"],
    "expiration": ["ttl", "timeout", "seconds"],
    "cached": ["cache", "ttl"],
    "entries": ["cache"],
    "long": ["ttl", "seconds"],
}


def expand_query(query: str) -> list[str]:
    """Light lexical expand so product asks match definitional prose."""
    q = tokenize(query)
    out = list(q)
    low = query.lower()
    for src, dsts in _EXPAND.items():
        if src in low:
            out.extend(tokenize(" ".join(dsts)))
    for t in list(out):
        if t.endswith("s") and len(t) > 4:
            out.append(t[:-1])
    seen: set[str] = set()
    uniq: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


@dataclass
class Passage:
    doc_id: str
    start: int
    end: int
    text: str
    tokens: list[str]


def split_paragraphs(doc_id: str, text: str, min_len: int = 40) -> list[Passage]:
    out: list[Passage] = []
    pos = 0
    for part in re.split(r"\n\s*\n+", text):
        idx = text.find(part, pos)
        if idx < 0:
            idx = pos
        end = idx + len(part)
        pos = end
        body = part.strip()
        if len(body) < min_len:
            continue
        out.append(Passage(doc_id, idx, end, body, tokenize(body)))
    if not out and text.strip():
        t = text.strip()
        out.append(Passage(doc_id, 0, len(text), t, tokenize(t)))
    return out


class BM25Index:
    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.N = len(passages) or 1
        self.avgdl = (sum(len(p.tokens) for p in passages) / self.N) if passages else 0.0
        df: dict[str, int] = {}
        for p in passages:
            for t in set(p.tokens):
                df[t] = df.get(t, 0) + 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for t, n in df.items()
        }

    def score(self, query_tokens: list[str], passage: Passage) -> float:
        if not passage.tokens:
            return 0.0
        tf: dict[str, int] = {}
        for t in passage.tokens:
            tf[t] = tf.get(t, 0) + 1
        dl = len(passage.tokens)
        s = 0.0
        for t in query_tokens:
            if t not in tf:
                continue
            idf = self.idf.get(t, 0.0)
            f = tf[t]
            s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
        return s

    def top_k(self, query: str, k: int = 5) -> list[tuple[float, Passage]]:
        q = expand_query(query)
        if not q or not self.passages:
            return []
        scored = [(self.score(q, p), p) for p in self.passages]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


def index_docs(docs: dict[str, str]) -> BM25Index:
    passages: list[Passage] = []
    for did, text in docs.items():
        passages.extend(split_paragraphs(did, text))
    return BM25Index(passages)


# Default margin below which BM25 hits are REVIEW, not PRESENT (W1).
BM25_MARGIN_TAU = 0.5


def top_paragraphs(
    corpus: dict[str, str],
    query: str,
    k: int = 5,
    *,
    margin_tau: float | None = None,
) -> list[dict]:
    """Product-facing BM25 hits as dicts for Claim evidence.

    Each hit includes bm25, rank, top2_bm25, margin, and promote flag
    (margin >= tau → eligible for PRESENT).
    """
    tau = BM25_MARGIN_TAU if margin_tau is None else margin_tau
    idx = index_docs(corpus)
    # Fetch k+1 so margin vs next is defined for the k-th hit when possible
    ranked = idx.top_k(query, k=max(k + 1, 2))
    out: list[dict] = []
    for i, (score, p) in enumerate(ranked[:k]):
        next_score = ranked[i + 1][0] if i + 1 < len(ranked) else 0.0
        margin = float(score) - float(next_score)
        out.append(
            {
                "doc_id": p.doc_id,
                "start": p.start,
                "end": p.end,
                "text": p.text,
                "bm25": round(float(score), 4),
                "top2_bm25": round(float(next_score), 4),
                "margin": round(margin, 4),
                "rank": i + 1,
                "promote": margin >= tau,
            }
        )
    return out
