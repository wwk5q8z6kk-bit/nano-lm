"""Leakage-isolated supervision for Nano's native state/pointer heads.

The module consumes only the sealed H1 ``StateSpanExample`` contract.  It does
not generate examples, inspect benchmarks, or infer new labels.  Each existing
character evidence span is aligned to Nano v0.1's exact tokenizer and retained
as an inclusive token start/end pair.  Token offsets are kept only for the
transcript segment so a pointer can never silently bind to ChatML or prompt
tokens.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from tokenizers import Tokenizer

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FIELD_ORDER, EvidenceSpan, FieldState
from nano_ai.training.model import NANO_MODEL_CONFIG
from nano_ai.training.state_span_data import StateSpanExample, canonical_json_bytes
from nano_ai.training.train_state_span import load_frozen_tokenizer

POINTER_SUPERVISION_VERSION = "nano-native-pointer-supervision-v0"
POINTER_PROMPT_TEMPLATE_ID = "chatml-pointer-scribe-v0"
POINTER_PROMPT_INSTRUCTION = "Summarize the visit."

MAX_SPANS_PER_FIELD = 2
IGNORE_SPAN_INDEX = -100
STATE_ORDER: tuple[FieldState, ...] = tuple(FieldState)
STATE_POINTER_COUNTS: tuple[int, ...] = tuple(
    {
        FieldState.SUPPORTED: 1,
        FieldState.ABSENT: 1,
        FieldState.MISSING: 0,
        FieldState.UNCERTAIN: 1,
        FieldState.CONFLICTING: 2,
    }[state]
    for state in STATE_ORDER
)

_PATIENT_PREFIX = re.compile(r"^\s*patient\s*:\s*", flags=re.IGNORECASE)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

TokenOffset = tuple[int, int] | None
SpanLabels = tuple[tuple[int, int], ...]
SpanMask = tuple[tuple[bool, bool], ...]


class PointerDataError(ValueError):
    """A source, tokenizer, alignment, or partition invariant is invalid."""


@dataclass(frozen=True, slots=True)
class PointerSupervision:
    """One H1 source record encoded for Nano's direct pointer objective.

    Span starts and ends are inclusive token indices into ``token_ids``.
    Inactive positions are explicit in ``span_mask`` and carry
    ``IGNORE_SPAN_INDEX`` rather than a plausible token index.
    """

    split: str
    example_id: str
    world_id: str
    source_record_sha256: str
    token_ids: tuple[int, ...]
    attention_mask: tuple[bool, ...]
    transcript_mask: tuple[bool, ...]
    pointer_mask: tuple[bool, ...]
    token_offsets: tuple[TokenOffset, ...]
    transcript_token_start: int
    transcript_token_end: int
    state_labels: tuple[int, ...]
    span_starts: SpanLabels
    span_ends: SpanLabels
    span_mask: SpanMask

    def __post_init__(self) -> None:
        token_count = len(self.token_ids)
        if not 0 < token_count <= NANO_MODEL_CONFIG.sequence_length:
            raise PointerDataError("pointer input must contain at most 512 tokens")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in self.token_ids
        ):
            raise PointerDataError("token_ids must contain non-negative integers")
        for name, mask in (
            ("attention_mask", self.attention_mask),
            ("transcript_mask", self.transcript_mask),
            ("pointer_mask", self.pointer_mask),
        ):
            if len(mask) != token_count or any(
                type(value) is not bool for value in mask
            ):
                raise PointerDataError(f"{name} must be a boolean mask over token_ids")
        if not all(self.attention_mask):
            raise PointerDataError("individual pointer examples cannot contain padding")
        if len(self.token_offsets) != token_count:
            raise PointerDataError("token_offsets must align one-to-one with token_ids")
        if not (
            0 <= self.transcript_token_start < self.transcript_token_end <= token_count
        ):
            raise PointerDataError("transcript token range is invalid")

        expected_transcript_mask = tuple(
            self.transcript_token_start <= index < self.transcript_token_end
            for index in range(token_count)
        )
        if self.transcript_mask != expected_transcript_mask:
            raise PointerDataError(
                "transcript_mask disagrees with transcript token range"
            )
        for index, offset in enumerate(self.token_offsets):
            if self.transcript_mask[index] != (offset is not None):
                raise PointerDataError(
                    "only transcript tokens may carry character offsets"
                )
            if offset is not None:
                start, end = offset
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 0
                    or end <= start
                ):
                    raise PointerDataError("transcript token offset is invalid")
        if any(
            pointer and not transcript
            for pointer, transcript in zip(
                self.pointer_mask, self.transcript_mask, strict=True
            )
        ):
            raise PointerDataError("pointer_mask must be contained in transcript_mask")

        if self.split not in {"train", "dev"}:
            raise PointerDataError("pointer source split must be train or dev")
        if not self.example_id or not self.world_id:
            raise PointerDataError("pointer source identity cannot be empty")
        if _SHA256.fullmatch(self.source_record_sha256) is None:
            raise PointerDataError("source record SHA-256 is invalid")
        if len(self.state_labels) != len(FIELD_ORDER) or any(
            isinstance(label, bool)
            or not isinstance(label, int)
            or not 0 <= label < len(STATE_ORDER)
            for label in self.state_labels
        ):
            raise PointerDataError("state_labels must contain five valid state indices")

        for name, labels in (
            ("span_starts", self.span_starts),
            ("span_ends", self.span_ends),
            ("span_mask", self.span_mask),
        ):
            if len(labels) != len(FIELD_ORDER) or any(
                len(row) != MAX_SPANS_PER_FIELD for row in labels
            ):
                raise PointerDataError(f"{name} must have shape [5, 2]")

        for field_index in range(len(FIELD_ORDER)):
            seen_inactive = False
            active_count = 0
            for slot in range(MAX_SPANS_PER_FIELD):
                active = self.span_mask[field_index][slot]
                if type(active) is not bool:
                    raise PointerDataError("span_mask must contain booleans")
                start = self.span_starts[field_index][slot]
                end = self.span_ends[field_index][slot]
                if not active:
                    seen_inactive = True
                    if start != IGNORE_SPAN_INDEX or end != IGNORE_SPAN_INDEX:
                        raise PointerDataError(
                            "inactive span slots must carry the ignore index"
                        )
                    continue
                if seen_inactive:
                    raise PointerDataError("active span slots must be left-packed")
                active_count += 1
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start <= end < token_count
                ):
                    raise PointerDataError("active token span is out of bounds")
                if not all(self.pointer_mask[start : end + 1]):
                    raise PointerDataError(
                        "active token spans must remain wholly inside Patient content"
                    )
            expected_count = STATE_POINTER_COUNTS[self.state_labels[field_index]]
            if active_count != expected_count:
                raise PointerDataError(
                    "field state and active evidence-span count disagree"
                )


def load_pointer_tokenizer(path: Path) -> Tokenizer:
    """Load Nano v0.1's hash-verified tokenizer through the frozen H1 loader."""

    return load_frozen_tokenizer(Path(path))


def _require_exact_tokenizer(tokenizer: Tokenizer) -> None:
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must be a tokenizers.Tokenizer")
    try:
        snapshot = tokenizer.to_str(pretty=True).encode("utf-8")
    except Exception as exc:  # pragma: no cover - defensive native binding seam
        raise PointerDataError("tokenizer could not be serialized") from exc
    observed = hashlib.sha256(snapshot).hexdigest()
    if observed != FROZEN_NANO_V01.tokenizer_sha256:
        raise PointerDataError("tokenizer identity is not frozen Nano v0.1")
    if (
        tokenizer.get_vocab_size(with_added_tokens=True)
        != NANO_MODEL_CONFIG.vocabulary_size
    ):
        raise PointerDataError("tokenizer vocabulary does not match Nano v0.1")


def _patient_content_ranges(transcript: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in transcript.splitlines(keepends=True):
        visible = line.rstrip("\r\n")
        match = _PATIENT_PREFIX.match(visible)
        if match is not None and match.end() < len(visible):
            ranges.append((offset + match.end(), offset + len(visible)))
        offset += len(line)
    return tuple(ranges)


def _trim_outer_whitespace(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def token_span_to_evidence(
    transcript: str,
    token_offsets: Sequence[TokenOffset],
    start_token: int,
    end_token: int,
) -> EvidenceSpan:
    """Reconstruct one exact Patient span from inclusive token boundaries.

    Byte-level BPE tokens may absorb whitespace immediately outside a labeled
    character span.  Only Unicode whitespace at the outer token envelope is
    removed; punctuation and other content are never normalized or guessed.
    """

    if not isinstance(transcript, str) or not transcript:
        raise PointerDataError("transcript must contain text")
    if (
        isinstance(start_token, bool)
        or isinstance(end_token, bool)
        or not isinstance(start_token, int)
        or not isinstance(end_token, int)
        or not 0 <= start_token <= end_token < len(token_offsets)
    ):
        raise PointerDataError("token span is out of bounds")
    selected = token_offsets[start_token : end_token + 1]
    if not selected or any(offset is None for offset in selected):
        raise PointerDataError("token span is not wholly inside the transcript")
    concrete = tuple(offset for offset in selected if offset is not None)
    raw_start = concrete[0][0]
    raw_end = concrete[-1][1]
    if raw_start < 0 or raw_end > len(transcript) or raw_end <= raw_start:
        raise PointerDataError("token span has invalid character offsets")
    if any(
        left[0] > right[0] or left[1] > right[1] for left, right in pairwise(concrete)
    ):
        raise PointerDataError("token offsets are not monotonic")
    start, end = _trim_outer_whitespace(transcript, raw_start, raw_end)
    if start >= end:
        raise PointerDataError("token span contains only whitespace")
    evidence = EvidenceSpan(
        start=start,
        end=end,
        text=transcript[start:end],
        speaker="patient",
    )
    try:
        evidence.validate_against(transcript)
    except ValueError as exc:
        raise PointerDataError("token span is not exact Patient evidence") from exc
    return evidence


def character_span_to_token_span(
    transcript: str,
    token_offsets: Sequence[TokenOffset],
    evidence: EvidenceSpan,
) -> tuple[int, int]:
    """Map an exact character annotation to inclusive tokenizer boundaries."""

    if not isinstance(evidence, EvidenceSpan):
        raise TypeError("evidence must be an EvidenceSpan")
    try:
        evidence.validate_against(transcript)
    except ValueError as exc:
        raise PointerDataError("source evidence is not exact Patient text") from exc
    intersecting = [
        index
        for index, offset in enumerate(token_offsets)
        if offset is not None
        and offset[1] > evidence.start
        and offset[0] < evidence.end
    ]
    if not intersecting:
        raise PointerDataError("source evidence has no tokenizer coverage")
    if intersecting != list(range(intersecting[0], intersecting[-1] + 1)):
        raise PointerDataError("source evidence token coverage is not contiguous")
    start_token, end_token = intersecting[0], intersecting[-1]
    reconstructed = token_span_to_evidence(
        transcript,
        token_offsets,
        start_token,
        end_token,
    )
    if reconstructed != evidence:
        raise PointerDataError(
            "token boundaries do not reconstruct the exact source evidence"
        )
    return start_token, end_token


def _pointer_mask(
    transcript: str,
    token_offsets: Sequence[TokenOffset],
) -> tuple[bool, ...]:
    patient_ranges = _patient_content_ranges(transcript)
    mask: list[bool] = []
    for offset in token_offsets:
        if offset is None:
            mask.append(False)
            continue
        raw_start, raw_end = offset
        start, end = _trim_outer_whitespace(transcript, raw_start, raw_end)
        if start == end:
            start, end = raw_start, raw_end
        mask.append(
            start < end
            and any(
                patient_start <= start and end <= patient_end
                for patient_start, patient_end in patient_ranges
            )
        )
    return tuple(mask)


def _encode_pointer_example(
    tokenizer: Tokenizer,
    example: StateSpanExample,
) -> PointerSupervision:
    start_token = tokenizer.token_to_id("<|im_start|>")
    end_token = tokenizer.token_to_id("<|im_end|>")
    if start_token is None or end_token is None:
        raise PointerDataError("tokenizer is missing required ChatML tokens")

    # Encode transcript independently.  This deliberately prevents a BPE merge
    # with the ChatML header or instruction from changing transcript offsets.
    user_header = tokenizer.encode("user\n", add_special_tokens=False).ids
    transcript_encoding = tokenizer.encode(
        example.transcript,
        add_special_tokens=False,
    )
    instruction = tokenizer.encode(
        f"\n{POINTER_PROMPT_INSTRUCTION}",
        add_special_tokens=False,
    ).ids
    chatml_separator = tokenizer.encode("\n", add_special_tokens=False).ids
    assistant_header = tokenizer.encode(
        "assistant\n",
        add_special_tokens=False,
    ).ids

    prefix = [start_token, *user_header]
    suffix = [
        *instruction,
        end_token,
        *chatml_separator,
        start_token,
        *assistant_header,
    ]
    token_ids = (*prefix, *transcript_encoding.ids, *suffix)
    if len(token_ids) > NANO_MODEL_CONFIG.sequence_length:
        raise PointerDataError(
            f"encoded example {example.example_id} exceeds Nano's 512-token context"
        )
    transcript_start = len(prefix)
    transcript_end = transcript_start + len(transcript_encoding.ids)
    if transcript_start == transcript_end:
        raise PointerDataError("pointer transcript produced no tokens")
    offsets: tuple[TokenOffset, ...] = (
        *((None,) * transcript_start),
        *tuple(transcript_encoding.offsets),
        *((None,) * (len(token_ids) - transcript_end)),
    )
    transcript_mask = tuple(offset is not None for offset in offsets)
    pointer_mask = _pointer_mask(example.transcript, offsets)

    proposals = parse_state_span_summary(example.target, example.transcript)
    state_labels: list[int] = []
    span_starts: list[tuple[int, int]] = []
    span_ends: list[tuple[int, int]] = []
    span_masks: list[tuple[bool, bool]] = []
    for expected_field, proposal in zip(FIELD_ORDER, proposals, strict=True):
        if proposal.field is not expected_field:
            raise PointerDataError("source target fields are not in canonical order")
        state_labels.append(STATE_ORDER.index(proposal.state))
        if len(proposal.spans) > MAX_SPANS_PER_FIELD:
            raise PointerDataError("source field exceeds the two-span pointer bound")
        starts = [IGNORE_SPAN_INDEX] * MAX_SPANS_PER_FIELD
        ends = [IGNORE_SPAN_INDEX] * MAX_SPANS_PER_FIELD
        active = [False] * MAX_SPANS_PER_FIELD
        for slot, evidence in enumerate(proposal.spans):
            token_start, token_end = character_span_to_token_span(
                example.transcript,
                offsets,
                evidence,
            )
            if not all(pointer_mask[token_start : token_end + 1]):
                raise PointerDataError("source evidence escapes Patient pointer mask")
            starts[slot] = token_start
            ends[slot] = token_end
            active[slot] = True
        span_starts.append((starts[0], starts[1]))
        span_ends.append((ends[0], ends[1]))
        span_masks.append((active[0], active[1]))

    return PointerSupervision(
        split=example.split,
        example_id=example.example_id,
        world_id=example.world_id,
        source_record_sha256=hashlib.sha256(
            canonical_json_bytes(example.to_dict())
        ).hexdigest(),
        token_ids=token_ids,
        attention_mask=(True,) * len(token_ids),
        transcript_mask=transcript_mask,
        pointer_mask=pointer_mask,
        token_offsets=offsets,
        transcript_token_start=transcript_start,
        transcript_token_end=transcript_end,
        state_labels=tuple(state_labels),
        span_starts=tuple(span_starts),
        span_ends=tuple(span_ends),
        span_mask=tuple(span_masks),
    )


def encode_pointer_example(
    tokenizer: Tokenizer,
    example: StateSpanExample,
) -> PointerSupervision:
    """Encode one sealed H1 record after verifying exact tokenizer identity."""

    _require_exact_tokenizer(tokenizer)
    if not isinstance(example, StateSpanExample):
        raise TypeError("example must be a StateSpanExample")
    return _encode_pointer_example(tokenizer, example)


def encode_pointer_partition(
    tokenizer: Tokenizer,
    examples: Sequence[StateSpanExample],
    *,
    expected_split: str,
) -> tuple[PointerSupervision, ...]:
    """Encode one source partition without reordering, mixing, or relabeling it."""

    _require_exact_tokenizer(tokenizer)
    if expected_split not in {"train", "dev"}:
        raise PointerDataError("expected_split must be train or dev")
    records = tuple(examples)
    if not records:
        raise PointerDataError("pointer partition cannot be empty")
    if any(not isinstance(example, StateSpanExample) for example in records):
        raise TypeError("examples must contain only StateSpanExample records")
    if any(example.split != expected_split for example in records):
        raise PointerDataError("pointer partition contains a cross-split record")
    if any(
        not example.example_id.startswith(f"{expected_split}-")
        or not example.world_id.startswith(f"{expected_split}-world-")
        for example in records
    ):
        raise PointerDataError("source identity disagrees with its partition")
    example_ids = [example.example_id for example in records]
    if len(set(example_ids)) != len(example_ids):
        raise PointerDataError("pointer partition contains duplicate example IDs")
    return tuple(_encode_pointer_example(tokenizer, example) for example in records)


__all__ = [
    "IGNORE_SPAN_INDEX",
    "MAX_SPANS_PER_FIELD",
    "POINTER_PROMPT_INSTRUCTION",
    "POINTER_PROMPT_TEMPLATE_ID",
    "POINTER_SUPERVISION_VERSION",
    "STATE_ORDER",
    "STATE_POINTER_COUNTS",
    "PointerDataError",
    "PointerSupervision",
    "TokenOffset",
    "character_span_to_token_span",
    "encode_pointer_example",
    "encode_pointer_partition",
    "load_pointer_tokenizer",
    "token_span_to_evidence",
]
