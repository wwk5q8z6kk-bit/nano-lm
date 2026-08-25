"""Output-format module for E-DELIMIT — the ONLY thing the three arms vary.

E-DELIMIT asks whether the span-port loss is a *delimitation* failure. The
measured shape it responds to: with all leak channels closed the model selects
the correct conversational turn for 97 of 120 gold-bearing slots and delimits
the gold span within it for 2. Every one of the 95 non-exact located quotes is
*over*-extended; there are zero under-extended. Median quote/enclosing-turn
ratio is 1.000 — the model quotes the whole turn.

H5 says the model addresses content at *unit* granularity and cannot resolve
sub-unit boundaries. If that is right, removing the need to *generate* a
boundary — while leaving retrieval exactly as hard — should convert located
slots into grounded ones.

Three arms:

``free_form`` (A, control)
    The existing format. The model writes a verbatim quote and chooses its
    extent. This is the arm that produced 2/192.

``menu`` (B, the discriminator)
    Candidate sub-spans are enumerated and the model picks an index. Boundary
    *generation* is removed; boundary *selection* remains. Retrieval is
    untouched — see MENU SCOPE below.

``offsets`` (C, secondary)
    The model emits character offsets into the transcript. Secondary by
    pre-registration: it requires index arithmetic, a known weakness
    independent of delimitation, so a failure here is not evidence about H5.

R1 (contrast hygiene) requires the arms to differ only in the output-format
module and the *question* to be byte-identical. That is why this file exists
separately from ``prompt.py``: an arm branch changes ``OUTPUT_FORMAT`` and
nothing else, exactly as the leakage grid changes one line of ``leakage.py``.

MENU SCOPE — a pre-commitment, recorded because it departs from the literal
prereg text. `BOTTLENECK_2026-08-25_delimitation.md` §5 says "the located
turn's candidate sub-spans enumerated". Taken literally that would enumerate
only the gold-bearing turn, which hands the model the location and collapses
retrieval — contradicting the same paragraph's "leaving retrieval exactly as
hard", and failing guard R5, under which an index-0 parrot must score at
chance rather than at ceiling. The menu is therefore built over the WHOLE
transcript. R5 is the reason, and R5 is the check.

MENU ORDER — candidates are ordered by a keyed digest of the candidate text,
never by position. Transcript order would make a candidate's index correlate
with its turn, which is the one thing that would let the R5 parrot pick up
real signal and stop being a chance baseline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from nanoscribe.encounter import Source

# The one line an arm branch changes. "free_form" | "menu" | "offsets"
OUTPUT_FORMAT = "free_form"

# Longest candidate, in whitespace tokens. Pre-committed at 5: the longest gold
# span in campaign_v2_multi is 5 tokens ("No allergies in the notes."), and
# 87 of 120 are a single token. Raising it inflates the menu without adding
# reachable gold; lowering it would make gold unreachable for 15 slots.
MAX_CANDIDATE_TOKENS = 5

_TOKEN = re.compile(r"\S+")
_TRIM = ".,;:!?\"'()"

ARMS = ("free_form", "menu", "offsets")


@dataclass(frozen=True, slots=True)
class Candidate:
    start: int
    end: int
    text: str


def _trimmed(text: str, start: int, end: int) -> tuple[int, int]:
    while end > start and text[end - 1] in _TRIM:
        end -= 1
    while start < end and text[start] in _TRIM:
        start += 1
    return start, end


def candidates_for_source(source: Source) -> tuple[Candidate, ...]:
    """Every contiguous 1..MAX_CANDIDATE_TOKENS token run inside a turn.

    Both the raw run and its punctuation-trimmed form are emitted: gold spans
    are frequently the bare word inside a token that carries a trailing period
    ("smoked" inside "smoked."). Without the trimmed variant 36 of 120 gold
    spans are unreachable and arm B would score them 0 for a reason that has
    nothing to do with the model.

    Candidates never cross a turn boundary — the same invariant
    ``select._span_in_turn`` enforces on the scoring side.
    """
    seen: set[tuple[int, int]] = set()
    out: list[Candidate] = []
    for turn in source.turns:
        spans = [
            (m.start() + turn.start, m.end() + turn.start)
            for m in _TOKEN.finditer(turn.text)
        ]
        for i in range(len(spans)):
            for n in range(1, MAX_CANDIDATE_TOKENS + 1):
                if i + n > len(spans):
                    break
                start, end = spans[i][0], spans[i + n - 1][1]
                for pair in ((start, end), _trimmed(source.text, start, end)):
                    if pair[1] > pair[0] and pair not in seen:
                        seen.add(pair)
                        out.append(
                            Candidate(pair[0], pair[1], source.text[pair[0] : pair[1]])
                        )
    return tuple(out)


def menu_for_slot(source: Source, slot_id: str) -> tuple[Candidate, ...]:
    """Candidates in a gold-independent, slot-keyed, deterministic order."""
    cands = candidates_for_source(source)

    def key(c: Candidate) -> str:
        digest = hashlib.sha256(f"{slot_id}\x00{c.text}".encode()).hexdigest()
        return digest

    return tuple(sorted(cands, key=key))


def gold_in_menu(source: Source, slot_id: str, gold_start: int, gold_end: int) -> bool:
    """Is the gold span reachable at all in this slot's menu?

    Emitted per slot. A slot whose gold is absent from the menu is
    INVALID_NO_SIGNAL for arm B, not a miss — otherwise a candidate-generator
    bug reads as H5 REFUTED, which is the expensive wrong conclusion.
    """
    return any(
        c.start == gold_start and c.end == gold_end
        for c in candidates_for_source(source)
    )


def _numbered_transcript(source: Source) -> str:
    lines = []
    for turn in source.turns:
        lines.append(f"[{turn.start}] {turn.speaker.value}: {turn.text}")
    return "\n".join(lines)


def transcript_block(source: Source) -> str:
    """The transcript as the arm shows it.

    ``offsets`` needs absolute indices to be well-defined, so it prefixes each
    turn with its start offset in the source text. The turn text itself is
    byte-identical across arms.
    """
    if OUTPUT_FORMAT == "offsets":
        return _numbered_transcript(source)
    lines = [f"{t.speaker.value}: {t.text}" for t in source.turns]
    return "\n".join(lines)


def format_instruction(source: Source, slot_id: str) -> str:
    """The output-format tail. This is the whole contrast."""
    if OUTPUT_FORMAT == "free_form":
        return (
            "Reply with exactly one line: "
            "STATED, DENIED, UNCERTAIN, or NOT_MENTIONED with a verbatim quote."
        )
    if OUTPUT_FORMAT == "menu":
        menu = menu_for_slot(source, slot_id)
        listing = "\n".join(f"  [{i}] {c.text}" for i, c in enumerate(menu))
        return (
            "Candidate quotes:\n"
            f"{listing}\n"
            "Reply with exactly one line: STATED, DENIED, UNCERTAIN, or "
            "NOT_MENTIONED followed by the number of the candidate quote in "
            'square brackets, for example: STATED: [7]. '
            "Use NOT_MENTIONED alone if no candidate applies."
        )
    if OUTPUT_FORMAT == "offsets":
        return (
            "Each transcript line is prefixed with the character offset of its "
            "first character. Reply with exactly one line: STATED, DENIED, "
            "UNCERTAIN, or NOT_MENTIONED followed by the start and end character "
            'offsets of the evidence in square brackets, for example: '
            "STATED: [104,113]. Use NOT_MENTIONED alone if nothing applies."
        )
    raise ValueError(f"unknown OUTPUT_FORMAT: {OUTPUT_FORMAT}")


_INDEX_RE = re.compile(r"\[\s*(\d+)\s*\]")
_PAIR_RE = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]")


def resolve_quotes(
    raw_line: str,
    source: Source,
    slot_id: str,
    parsed_quotes: tuple[str, ...],
) -> tuple[str, ...]:
    """Turn the arm's raw answer into quote strings for the shared scorer.

    Downstream is deliberately untouched: whatever this returns is bound by
    ``ConstrainedSelector`` exactly as arm A's quotes are, so the scoring path
    is identical across arms and only the elicitation differs.
    """
    if OUTPUT_FORMAT == "free_form":
        return parsed_quotes
    if OUTPUT_FORMAT == "menu":
        match = _INDEX_RE.search(raw_line)
        if match is None:
            return ()
        menu = menu_for_slot(source, slot_id)
        index = int(match.group(1))
        if index < 0 or index >= len(menu):
            return ()
        return (menu[index].text,)
    if OUTPUT_FORMAT == "offsets":
        match = _PAIR_RE.search(raw_line)
        if match is None:
            return ()
        start, end = int(match.group(1)), int(match.group(2))
        if start < 0 or end <= start or end > len(source.text):
            return ()
        return (source.text[start:end],)
    raise ValueError(f"unknown OUTPUT_FORMAT: {OUTPUT_FORMAT}")


def parrot_quotes(source: Source, slot_id: str) -> tuple[str, ...]:
    """R5 manipulation check — always pick index 0.

    Must score at chance. If the index-0 parrot scores near ceiling the menu is
    ordered by something correlated with gold, and arm B is measuring menu
    construction rather than the model.
    """
    menu = menu_for_slot(source, slot_id)
    return (menu[0].text,) if menu else ()


def output_format_hash() -> str:
    """Identifies the elicitation. MUST differ across arms."""
    payload = f"{OUTPUT_FORMAT}\x00{MAX_CANDIDATE_TOKENS}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def delimit_config() -> dict[str, object]:
    """Recorded in every run report."""
    return {
        "output_format": OUTPUT_FORMAT,
        "max_candidate_tokens": MAX_CANDIDATE_TOKENS,
        "output_format_hash": output_format_hash(),
        "menu_scope": "whole_transcript",
        "menu_order": "sha256(slot_id||candidate_text)",
    }
