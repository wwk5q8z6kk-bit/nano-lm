"""Constrained evidence selection — SOFTWARE.

A selector may only emit a span that is an exact substring of one turn.
Paraphrase cannot become evidence. Ambiguous quotes abstain.
"""

from __future__ import annotations

import re

from nanoscribe.encounter import EncounterError, EvidenceSpan, Source, Speaker

_WS_RE = re.compile(r"\s+")

_UNICODE_MAP = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    "\u00a0": " ",
}


def _fail(code: str, message: str, path: str = "$") -> None:
    raise EncounterError(code, message, path=path)


def _normalize_surface(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    pending_space_from: int | None = None
    for i, raw in enumerate(text):
        ch = _UNICODE_MAP.get(raw, raw)
        if ch.isspace():
            if chars and pending_space_from is None:
                pending_space_from = i
            continue
        if pending_space_from is not None:
            chars.append(" ")
            index_map.append(pending_space_from)
            pending_space_from = None
        for folded in ch.casefold():
            chars.append(folded)
            index_map.append(i)
    return "".join(chars), index_map


def _span_in_turn(source: Source, start: int, end: int, text: str, evidence_id: str) -> EvidenceSpan:
    if start < 0 or end <= start or end > len(source.text):
        _fail("evidence_bounds", "offsets are outside the source")
    if source.text[start:end] != text:
        _fail("evidence_text_mismatch", "claimed text does not match source offsets")
    host: tuple[int, Speaker, str] | None = None
    for turn in source.turns:
        if turn.start <= start and end <= turn.end:
            host = (turn.start, turn.speaker, turn.turn_id)
            break
    if host is None:
        _fail("evidence_crosses_turn", "span cannot silently cross turn boundaries")
    _, speaker, turn_id = host
    span = EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source.source_id,
        turn_id=turn_id,
        speaker=speaker,
        start=start,
        end=end,
        text=text,
    )
    if source.text[span.start : span.end] != span.text:
        _fail("evidence_text_mismatch", "evidence text does not match its offsets")
    return span


def copy_span(
    source: Source,
    start: int,
    end: int,
    *,
    evidence_id: str,
    text: str | None = None,
) -> EvidenceSpan:
    claimed = source.text[start:end] if text is None else text
    return _span_in_turn(source, start, end, claimed, evidence_id)


def _exact_hits(source: Source, quote: str) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    if not quote:
        return hits
    for turn in source.turns:
        found = source.text.find(quote, turn.start, turn.end)
        while found >= 0:
            end = found + len(quote)
            if end <= turn.end:
                hits.append((found, end))
            found = source.text.find(quote, found + 1, turn.end)
    return hits


def match_count(source: Source, quote: str) -> int:
    return len(_exact_hits(source, quote))


def relocate(source: Source, quote: str, *, evidence_id: str) -> EvidenceSpan | None:
    hits = _exact_hits(source, quote)
    if len(hits) != 1:
        return None
    start, end = hits[0]
    return _span_in_turn(source, start, end, quote, evidence_id)


def _surface_hits(source: Source, quote: str) -> list[tuple[int, int]]:
    norm_quote, _ = _normalize_surface(quote)
    if not norm_quote:
        return []
    hits: list[tuple[int, int]] = []
    for turn in source.turns:
        segment = source.text[turn.start : turn.end]
        norm_segment, index_map = _normalize_surface(segment)
        found = norm_segment.find(norm_quote)
        while found >= 0:
            last = found + len(norm_quote) - 1
            start = turn.start + index_map[found]
            end = turn.start + index_map[last] + 1
            hits.append((start, end))
            found = norm_segment.find(norm_quote, found + 1)
    return hits


def snap_relocate(source: Source, quote: str, *, evidence_id: str) -> EvidenceSpan | None:
    hits = _surface_hits(source, quote)
    if len(hits) != 1:
        return None
    start, end = hits[0]
    return _span_in_turn(source, start, end, source.text[start:end], evidence_id)


def _quote_variants(quote: str, raw_value: str | None = None) -> tuple[str, ...]:
    """Ordered transport attempts — still fail-closed on ambiguous or invented text."""
    candidates: list[str] = []
    seen: set[str] = set()

    def add(text: str | None) -> None:
        if not text:
            return
        cleaned = text.strip()
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        candidates.append(cleaned)

    add(quote)
    add(quote.strip("\"'“”‘’"))
    collapsed = _WS_RE.sub(" ", quote).strip()
    add(collapsed)

    # raw_value is a LAST-RESORT variant, and only when the model supplied no
    # usable quote at all.
    #
    # Adding it unconditionally made the selector rescue every failed
    # generation: a paraphrase ("cervicalgia" for "neck") or an outright
    # hallucination ("totally unrelated words") would fail relocation, fall
    # through to raw_value, and return a valid span with abstained=False. The
    # offset invariant still held, so nothing fabricated reached the record —
    # but the ABSTENTION SIGNAL was destroyed and every such failure scored as a
    # success, inflating transport/coverage and zeroing correct_abstention.
    #
    # Briefing SS VI: "Never perform semantic paraphrase relocation. Ambiguous or
    # missing evidence: ABSTAIN / REVIEW, not guessed evidence." A model that
    # offered a quote and got it wrong must abstain.
    if not candidates:
        add(raw_value)
    return tuple(candidates)


def select_quote_variants(
    source: Source,
    quote: str,
    *,
    evidence_id: str,
    raw_value: str | None = None,
) -> EvidenceSpan | None:
    """Try exact then surface-normalized relocation for each quote variant."""
    for variant in _quote_variants(quote, raw_value=raw_value):
        span = relocate(source, variant, evidence_id=evidence_id)
        if span is not None:
            return span
        span = snap_relocate(source, variant, evidence_id=evidence_id)
        if span is not None:
            return span
    return None


class ConstrainedSelector:
    """Fail-closed interface: only exact, single-turn source spans may be emitted."""

    def copy_span(
        self,
        source: Source,
        start: int,
        end: int,
        *,
        evidence_id: str,
        text: str | None = None,
    ) -> EvidenceSpan:
        return copy_span(source, start, end, evidence_id=evidence_id, text=text)

    def select_quote(
        self,
        source: Source,
        quote: str,
        *,
        evidence_id: str,
        raw_value: str | None = None,
    ) -> EvidenceSpan | None:
        return select_quote_variants(
            source,
            quote,
            evidence_id=evidence_id,
            raw_value=raw_value,
        )
