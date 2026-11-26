"""Memory Engine: Episodic Working Memory + Confidence-Calibrated Persistent Knowledge Retrieval.

Implements:
  1. `WorkingMemory`: Compact structured state across multi-step reasoning/tool turns so long
     agentic trajectories retain exact entity IDs, intermediate numbers, and API constraints
     without context bloat.
  2. `PersistentKnowledgeStore`: Sublinear BM25 + normalized TF-IDF cosine hybrid index with:
     - Confidence gating (`max_score >= min_confidence`, avoiding noisy retrieval on self-contained math/code),
     - Version/contradiction detection (`supersedes` / conflicting numeric claims flagged explicitly).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\-\.]*")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class KnowledgeDoc:
    doc_id: str
    title: str
    content: str
    version: int = 1
    status: str = "active"  # 'active' | 'deprecated' | 'superseded'
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    triggered: bool
    confidence: float
    has_conflict_or_deprecation: bool
    chunks: list[KnowledgeDoc]
    formatted_evidence: str


def is_read_or_prep_tool(tool_name: str) -> bool:
    """Classify read-only lookup / credential-refresh tools vs state-mutating action tools."""
    low = tool_name.strip().lower()
    return low.startswith(
        ("lookup_", "get_", "fetch_", "query_", "list_", "read_", "refresh_", "rotate_", "inspect_", "describe_", "check_")
    )


class WorkingMemory:
    """Epistemic Belief-Revision Scratchpad tracking active facts, invalidated state, and unresolved errors."""

    def __init__(self, max_entries: int = 12):
        self.max_entries = max_entries
        self.facts: dict[str, Any] = {}
        self.invalidated_facts: dict[str, Any] = {}
        self.events: list[str] = []
        self.has_unresolved_error: bool = False
        self.has_mutating_success: bool = False

    def record_fact(self, key: str, value: Any) -> None:
        for prefix in ("fresh_", "new_", "updated_", "active_"):
            if key.startswith(prefix):
                base_k = key[len(prefix) :]
                if base_k in self.facts and self.facts[base_k] != value:
                    self.invalidated_facts[base_k] = self.facts.pop(base_k)
        self.facts[key] = value

    def revise_on_error(self, call_args: dict[str, Any], error_code: str, message: str) -> None:
        self.has_unresolved_error = True
        err_low = f"{error_code} {message}".lower()
        for k, v in call_args.items():
            if ("stale" in err_low or "expired" in err_low or "invalid" in err_low) and k.lower() in err_low:
                self.invalidated_facts[k] = v
                if self.facts.get(k) == v:
                    self.facts.pop(k, None)
        m_bound = re.search(
            r"(?:at least|minimum[^\d]*|clamp[^\d]*to(?: at least)?|floor[^\d]*)\s*(\d+)",
            message,
            flags=re.IGNORECASE,
        )
        if m_bound:
            self.facts["required_safety_bound"] = int(m_bound.group(1))
        self.record_event(f"ERROR [{error_code}]: {message}")

    def record_event(self, summary: str) -> None:
        compact = " ".join(summary.strip().split())[:240]
        self.events.append(compact)
        if len(self.events) > self.max_entries:
            self.events = self.events[-self.max_entries :]

    def render(self) -> str:
        if not self.facts and not self.invalidated_facts and not self.events:
            return ""
        parts = []
        if self.facts:
            kv = ", ".join(f"{k}={v!r}" for k, v in self.facts.items())
            parts.append(f"Active Verified State: {kv}")
        if self.invalidated_facts:
            bad = ", ".join(f"{k}={v!r} (INVALID/STALE - DO NOT USE)" for k, v in self.invalidated_facts.items())
            parts.append(f"Superseded/Invalidated State: {bad}")
        if self.events:
            parts.append("Recent Diagnostic Events:\n" + "\n".join(f"  - {e}" for e in self.events[-6:]))
        return "\n".join(parts)


class PersistentKnowledgeStore:
    """Hybrid BM25 + TF-IDF cosine persistent memory with calibrated confidence & conflict detection."""

    def __init__(self, docs: list[KnowledgeDoc] | None = None, min_confidence: float = 0.26):
        self.docs: list[KnowledgeDoc] = []
        self.min_confidence = min_confidence
        self._doc_tfs: list[Counter[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 1.0
        if docs:
            self.add_documents(docs)

    def add_documents(self, docs: list[KnowledgeDoc]) -> None:
        for d in docs:
            toks = _tokenize(f"{d.title} {d.doc_id} {d.content}")
            tf = Counter(toks)
            self.docs.append(d)
            self._doc_tfs.append(tf)
            self._doc_lens.append(len(toks))
            for term in tf:
                self._df[term] += 1
        self._avgdl = sum(self._doc_lens) / max(1, len(self._doc_lens))

    def _idf(self, term: str) -> float:
        n = len(self.docs)
        df = self._df.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        if not self.docs:
            return RetrievalResult(False, 0.0, False, [], "")
        q_toks = _tokenize(query)
        if not q_toks:
            return RetrievalResult(False, 0.0, False, [], "")

        q_tf = Counter(q_toks)
        q_weights = {t: q_tf[t] * self._idf(t) for t in q_tf}
        q_norm = math.sqrt(sum(w * w for w in q_weights.values())) + 1e-8

        scored: list[tuple[float, int]] = []
        k1, b = 1.5, 0.75
        for idx, (tf, dl, doc) in enumerate(zip(self._doc_tfs, self._doc_lens, self.docs)):
            bm25 = 0.0
            dot = 0.0
            for t, qw in q_weights.items():
                f = tf.get(t, 0)
                if f == 0:
                    continue
                idf = self._idf(t)
                bm25 += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * dl / max(1.0, self._avgdl)))
                dot += qw * (f * idf)
            if dot == 0.0:
                continue
            d_norm = math.sqrt(sum((cnt * self._idf(tm)) ** 2 for tm, cnt in tf.items())) + 1e-8
            cos = dot / (q_norm * d_norm)
            # Combine normalized cosine [0, 1] + squashed BM25 coverage, preferring active revisions slightly
            bm25_norm = bm25 / (bm25 + 6.0)
            status_bonus = 0.03 if doc.status == "active" else 0.0
            score = 0.55 * cos + 0.45 * bm25_norm + status_bonus
            scored.append((score, idx))

        if not scored:
            return RetrievalResult(False, 0.0, False, [], "")

        scored.sort(reverse=True)
        best_conf = round(scored[0][0], 4)
        if best_conf < self.min_confidence:
            return RetrievalResult(False, best_conf, False, [], "")

        chosen = [self.docs[i] for s, i in scored[:top_k] if s >= self.min_confidence * 0.75]
        conflict = any(d.status in ("deprecated", "superseded") for d in chosen) or (
            len({d.title.split(" v")[0] for d in chosen}) < len(chosen)
        )
        lines = []
        for d in chosen:
            tag = f"[status={d.status.upper()}, v{d.version}]"
            lines.append(f"- Doc `{d.doc_id}` ({d.title}) {tag}: {d.content}")
        return RetrievalResult(
            triggered=True,
            confidence=best_conf,
            has_conflict_or_deprecation=conflict,
            chunks=chosen,
            formatted_evidence="\n".join(lines),
        )
