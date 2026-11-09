"""Constrained evidence selection — SOFTWARE.

A selector may only emit a span that is an exact substring of one turn.
Paraphrase cannot become evidence. Ambiguous quotes abstain.
"""

from __future__ import annotations

from nanoscribe.encounter import EncounterError, EvidenceSpan, Source, Speaker

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


def _span_in_turn(
    source: Source, start: int, end: int, text: str, evidence_id: str
) -> EvidenceSpan:
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

    quote_len = len(quote)
    source_text = source.text
    found = source_text.find(quote)
    if found < 0:
        return hits

    found_positions: list[int] = []
    while found >= 0:
        found_positions.append(found)
        found = source_text.find(quote, found + 1)

    num_turns = len(source.turns)
    turns = source.turns
    turn_idx = 0

    for f_start in found_positions:
        f_end = f_start + quote_len
        while turn_idx < num_turns and turns[turn_idx].end < f_end:
            turn_idx += 1

        temp_idx = turn_idx
        while temp_idx < num_turns and turns[temp_idx].start <= f_start:
            if f_end <= turns[temp_idx].end:
                hits.append((f_start, f_end))
            temp_idx += 1

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

    norm_segment, index_map = _normalize_surface(source.text)
    found = norm_segment.find(norm_quote)
    hits: list[tuple[int, int]] = []
    if found < 0:
        return hits

    found_positions: list[int] = []
    while found >= 0:
        found_positions.append(found)
        found = norm_segment.find(norm_quote, found + 1)

    quote_len = len(norm_quote)
    num_turns = len(source.turns)
    turns = source.turns
    turn_idx = 0

    for f_start_norm in found_positions:
        f_end_norm = f_start_norm + quote_len - 1
        start = index_map[f_start_norm]
        end = index_map[f_end_norm] + 1

        while turn_idx < num_turns and turns[turn_idx].end < end:
            turn_idx += 1

        temp_idx = turn_idx
        while temp_idx < num_turns and turns[temp_idx].start <= start:
            if end <= turns[temp_idx].end:
                hits.append((start, end))
            temp_idx += 1

    return hits


def snap_relocate(
    source: Source, quote: str, *, evidence_id: str
) -> EvidenceSpan | None:
    hits = _surface_hits(source, quote)
    if len(hits) != 1:
        return None
    start, end = hits[0]
    return _span_in_turn(source, start, end, source.text[start:end], evidence_id)


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
        self, source: Source, quote: str, *, evidence_id: str
    ) -> EvidenceSpan | None:
        return relocate(source, quote, evidence_id=evidence_id) or snap_relocate(
            source, quote, evidence_id=evidence_id
        )
