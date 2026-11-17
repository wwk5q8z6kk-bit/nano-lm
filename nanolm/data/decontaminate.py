"""Exact and n-gram decontamination engine + SHA-256 dataset provenance tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Iterable

_WS_RE = re.compile(r"\s+")


def normalize_for_ngram(text: str) -> list[str]:
    cleaned = _WS_RE.sub(" ", text.lower().strip())
    return cleaned.split(" ") if cleaned else []


def extract_ngrams(words: list[str], n: int = 13) -> set[tuple[str, ...]]:
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


@dataclass(frozen=True, slots=True)
class DecontaminationReport:
    total_candidates: int
    kept_documents: int
    removed_exact_dups: int
    removed_eval_contaminated: int
    ngram_size: int
    corpus_sha256: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decontaminate_corpus(
    candidate_docs: Iterable[str],
    eval_texts: Iterable[str],
    ngram_size: int = 13,
) -> tuple[list[str], DecontaminationReport]:
    """Deduplicate training docs and scrub any doc sharing an exact match or 13-gram with eval sets."""
    eval_ngrams: set[tuple[str, ...]] = set()
    eval_exact: set[str] = set()
    for et in eval_texts:
        norm_words = normalize_for_ngram(et)
        if not norm_words:
            continue
        eval_exact.add(" ".join(norm_words))
        if len(norm_words) >= ngram_size:
            eval_ngrams.update(extract_ngrams(norm_words, n=ngram_size))

    seen_hashes: set[bytes] = set()
    kept: list[str] = []
    total = 0
    dup_count = 0
    contam_count = 0
    running_hash = hashlib.sha256()

    for doc in candidate_docs:
        total += 1
        norm_words = normalize_for_ngram(doc)
        norm_str = " ".join(norm_words)
        digest = hashlib.sha1(norm_str.encode("utf-8")).digest()
        if digest in seen_hashes:
            dup_count += 1
            continue

        is_contaminated = norm_str in eval_exact
        if not is_contaminated and len(norm_words) >= ngram_size and eval_ngrams:
            doc_ngrams = extract_ngrams(norm_words, n=ngram_size)
            if not doc_ngrams.isdisjoint(eval_ngrams):
                is_contaminated = True

        if is_contaminated:
            contam_count += 1
            continue

        seen_hashes.add(digest)
        kept.append(doc)
        running_hash.update(doc.encode("utf-8"))
        running_hash.update(b"\n")

    report = DecontaminationReport(
        total_candidates=total,
        kept_documents=len(kept),
        removed_exact_dups=dup_count,
        removed_eval_contaminated=contam_count,
        ngram_size=ngram_size,
        corpus_sha256=running_hash.hexdigest(),
    )
    return kept, report
