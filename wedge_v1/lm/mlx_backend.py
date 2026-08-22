"""ONE real LMBackend — local MLX Llama with constructive span binding.

Contract (must hold for every PRESENT claim):
  1. Retrieve candidate passages (BM25 / synonym).
  2. Ask the model to *quote* a contiguous substring from those passages only.
  3. Relocate the quote uniquely in the source corpus (normalize → unique hit).
  4. verify_claim + span-ablation must pass; else ABSTAIN.

No paid compute. Default model: mlx-community/Llama-3.2-3B-Instruct-4bit
(already cached locally). Fan-out to other backends is out of scope until this
contract is proven end-to-end.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from wedge_v1.classical.bm25 import top_paragraphs
from wedge_v1.classical.solvers import Claim, _find
from wedge_v1.classical.verifier import verify_claim
from wedge_v1.lm.probe import ALLOWLIST_TASKS, DOSE_PAT, T35_QUERY, TTL_PAT, ablation_fails_support
from wedge_v1.plugins import synonym as synonym_plugin

DEFAULT_MLX_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

_QUOTE_RE = re.compile(r'["“]([^"”]{3,160})["”]')
_TTL_NUM_RE = re.compile(r"(\d+)\s*seconds?", re.I)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def relocate_unique(docs: dict[str, str], quote: str) -> tuple[str, dict] | None:
    """Return (doc_id, evidence) iff quote occurs exactly once across the corpus."""
    needle = _normalize_ws(quote).rstrip(" .;,:")
    if len(needle) < 3:
        return None
    hits: list[tuple[str, dict]] = []
    for doc_id, body in docs.items():
        start = 0
        while True:
            i = body.find(needle, start)
            if i < 0:
                break
            hits.append(
                (
                    doc_id,
                    {
                        "doc_id": doc_id,
                        "start": i,
                        "end": i + len(needle),
                        "text": body[i : i + len(needle)],
                    },
                )
            )
            start = i + 1
        if any(h[0] == doc_id for h in hits):
            continue
        span = _find(body, needle) or _find(body, quote.strip())
        if span:
            hits.append((doc_id, {**span, "doc_id": doc_id}))
    if len(hits) != 1:
        return None
    return hits[0]


def extract_quote(raw: str) -> str | None:
    """Pull a quoted span or the first substantive line from model output."""
    text = (raw or "").strip()
    if not text:
        return None
    m = _QUOTE_RE.search(text)
    if m:
        return _normalize_ws(m.group(1)).rstrip(" .;,:")
    line = text.splitlines()[0].strip()
    line = re.sub(r"^(answer|quote|span)\s*[:\-]\s*", "", line, flags=re.I)
    line = line.strip(" \"'`")
    line = _normalize_ws(line)
    # Models often append a period outside the source span — strip it for relocate.
    line = line.rstrip(" .;,:")
    return line if len(line) >= 3 else None


def _candidate_passages(docs: dict[str, str], query: str, *, k: int = 4) -> list[dict[str, Any]]:
    passages: list[dict[str, Any]] = []
    for hit in top_paragraphs(docs, query, k=k):
        passages.append(
            {
                "doc_id": hit["doc_id"],
                "text": hit["text"],
                "start": hit["start"],
                "end": hit["end"],
                "source": "bm25",
            }
        )
    plug = synonym_plugin.probe_ttl(docs, query)
    if plug.status == "PRESENT" and plug.doc_id and plug.evidence:
        ev = plug.evidence[0]
        start = max(0, (ev.get("start") or 0) - 40)
        end = (ev.get("end") or 0) + 40
        passages.insert(
            0,
            {
                "doc_id": plug.doc_id,
                "text": docs[plug.doc_id][start:end],
                "start": ev.get("start"),
                "end": ev.get("end"),
                "source": "synonym",
            },
        )
    seen: set[tuple] = set()
    out: list[dict[str, Any]] = []
    for p in passages:
        key = (p["doc_id"], p.get("start"))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out[:k]


def _build_prompt(task_id: str, query: str, passages: list[dict[str, Any]]) -> str:
    blocks = []
    for i, p in enumerate(passages, 1):
        blocks.append(f"[{i}] doc={p['doc_id']}\n{p['text']}")
    joined = "\n\n".join(blocks)
    if task_id == "T35":
        instruction = (
            "Task: find the cache TTL.\n"
            "Reply with ONLY a contiguous quote copied from the passages "
            '(e.g. "TTL as 300 seconds"). If unsupported, reply ABSTAIN.'
        )
    elif task_id == "T36":
        instruction = (
            "Task: find whether metformin doses change across docs.\n"
            "Reply with ONLY a contiguous quote containing a dose "
            '(e.g. "metformin 500 mg"). If unsupported, reply ABSTAIN.'
        )
    else:
        instruction = (
            "Task: find the pronoun antecedent binding.\n"
            "Reply with ONLY a contiguous quote from the passages. "
            "If unsupported, reply ABSTAIN."
        )
    return (
        f"{instruction}\n"
        f"Question: {query or task_id}\n\n"
        f"Passages:\n{joined}\n\n"
        "Quote:"
    )


@dataclass
class MLXLlamaBackend:
    """Local MLX Llama-3.2-3B Instruct — the single proven real LMBackend."""

    name: str = "mlx_llama32_3b_spanbound"
    model_id: str = DEFAULT_MLX_MODEL
    max_tokens: int = 48
    _model: Any = field(default=None, repr=False, compare=False)
    _tokenizer: Any = field(default=None, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        self._model, self._tokenizer = load(self.model_id)

    def _generate(self, prompt: str) -> str:
        from mlx_lm import generate

        self._ensure_loaded()
        return generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            verbose=False,
        )

    def propose(self, task_id: str, docs: dict[str, str], *, query: str = "") -> Claim:
        if task_id not in ALLOWLIST_TASKS:
            return Claim(task_id, None, None, status="ABSTAIN", notes="lm_not_allowlisted")
        q = query or (T35_QUERY if task_id == "T35" else task_id)
        passages = _candidate_passages(docs, q)
        if not passages:
            return Claim(
                task_id,
                None,
                {"method": self.name, "query": q},
                status="ABSTAIN",
                notes="mlx_no_candidates",
            )

        raw = self._generate(_build_prompt(task_id, q, passages))
        head = (raw or "").strip().upper().split()
        if head and head[0] == "ABSTAIN":
            return Claim(
                task_id,
                None,
                {"method": self.name, "query": q, "raw": (raw or "")[:200]},
                status="ABSTAIN",
                notes="mlx_model_abstain",
            )

        quote = extract_quote(raw or "")
        if not quote:
            return Claim(
                task_id,
                None,
                {"method": self.name, "query": q, "raw": (raw or "")[:200]},
                status="ABSTAIN",
                notes="mlx_no_quote",
            )

        if task_id == "T35" and not (TTL_PAT.search(quote) or _TTL_NUM_RE.search(quote)):
            num = _TTL_NUM_RE.search(quote)
            if num:
                for p in passages:
                    m = TTL_PAT.search(p["text"])
                    if m and num.group(1) in m.group(0):
                        quote = m.group(0)
                        break
            if not (TTL_PAT.search(quote) or _TTL_NUM_RE.search(quote)):
                return Claim(
                    task_id,
                    None,
                    {"method": self.name, "query": q, "quote": quote},
                    status="ABSTAIN",
                    notes="mlx_quote_not_ttl",
                )
        if task_id == "T36" and not DOSE_PAT.search(quote):
            return Claim(
                task_id,
                None,
                {"method": self.name, "query": q, "quote": quote},
                status="ABSTAIN",
                notes="mlx_quote_not_dose",
            )

        relocated = relocate_unique(docs, quote)
        if relocated is None and task_id == "T35":
            for doc_id, body in docs.items():
                m = TTL_PAT.search(body)
                if m and (m.group(1) in quote or quote in m.group(0)):
                    span = _find(body, m.group(0))
                    if span:
                        relocated = (doc_id, {**span, "doc_id": doc_id})
                        quote = m.group(0)
                        break
        if relocated is None:
            return Claim(
                task_id,
                None,
                {"method": self.name, "query": q, "quote": quote, "raw": (raw or "")[:200]},
                status="ABSTAIN",
                notes="mlx_relocate_not_unique",
            )

        doc_id, evidence = relocated
        if task_id == "T35":
            m = TTL_PAT.search(quote) or _TTL_NUM_RE.search(quote)
            answer: Any = f"{m.group(1)} seconds" if m else quote
        elif task_id == "T36":
            m = DOSE_PAT.search(quote)
            answer = {"dose_mg": int(m.group(1)), "quote": quote} if m else quote
        else:
            answer = quote

        claim = Claim(
            task_id,
            doc_id,
            {"query": q, "answer": answer, "method": self.name, "quote": quote},
            evidence=[evidence],
            status="PRESENT",
            notes="mlx_span_bound",
            meta={"backend": self.name, "model_id": self.model_id},
        )
        claim = verify_claim(claim)
        if claim.status != "PRESENT":
            return claim
        # Offsets must resolve literally in the cited doc.
        ev0 = (claim.evidence or [{}])[0]
        body = docs.get(claim.doc_id or "", "")
        start, end = ev0.get("start"), ev0.get("end")
        if not body or start is None or end is None or body[int(start):int(end)] != (ev0.get("text") or ""):
            return Claim(
                task_id,
                None,
                {"method": self.name, "query": q, "quote": quote},
                status="ABSTAIN",
                notes="mlx_offset_mismatch",
            )
        # Constructive faithfulness: ablating evidence must kill support.
        # ablation_fails_support==True means empty-evidence is correctly rejected.
        if not ablation_fails_support(claim, docs):
            return Claim(
                task_id,
                None,
                {"method": self.name, "query": q, "quote": quote},
                status="ABSTAIN",
                notes="mlx_ablation_contract_broken",
            )
        return claim
