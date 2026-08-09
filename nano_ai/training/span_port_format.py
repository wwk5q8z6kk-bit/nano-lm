"""Shared prompt and parse helpers for the route-(b) span-port falsifier.

Preregistration: `papers/PREREG_SPAN_PORT_B.md`.
Decision: `papers/DECISION_SPAN_PORT.md`.
"""

from __future__ import annotations

import re
from typing import Literal

from nano_ai.adapters.state_span import classify_patient_span_relocation
from nano_ai.contract import FieldState

RelocationClass = Literal["relocated", "no_match", "ambiguous"]

FIELD_QUESTION = {
    "medication": "any medication the patient is taking",
    "allergy": "any allergy the patient has",
}

# Prompt must match the training target. The v2 data build used the state-only
# "exactly one word" prompt while writing `STATED: "..."` targets — a silent
# format contradiction. Span-port training and probing both use this text.
PROMPT_WITH_SPANS = """You are reading a clinic transcript.

{transcript}

Question: regarding {topic}, which of these does the patient's own words do?

STATED       - the patient names a specific one
DENIED       - the patient says there is none / denies having any
NOT_MENTIONED - the topic never comes up

Answer on one line:
- STATED: "exact patient words that name it"
- DENIED: "exact patient words that deny it"
- NOT_MENTIONED
Quotes must copy the patient's words exactly. If the topic never comes up, write
only NOT_MENTIONED with no quotes."""

EXPECTED_LABEL = {
    FieldState.SUPPORTED: "STATED",
    FieldState.ABSENT: "DENIED",
    FieldState.MISSING: "NOT_MENTIONED",
}

_LABEL_RE = re.compile(
    r"\b(STATED|DENIED|NOT_MENTIONED)\b(?:\s*:\s*(.*))?",
    re.IGNORECASE | re.DOTALL,
)
_QUOTED_RE = re.compile(r'"([^"]+)"')


def parse_label_and_spans(raw: str) -> tuple[str | None, tuple[str, ...]]:
    """Extract the final label and any quoted span strings from a generation."""
    if not raw or not raw.strip():
        return None, ()
    matches = list(_LABEL_RE.finditer(raw))
    if not matches:
        return None, ()
    match = matches[-1]
    label = match.group(1).upper()
    rest = match.group(2) or ""
    spans = tuple(part.strip() for part in _QUOTED_RE.findall(rest) if part.strip())
    return label, spans


def classify_span_relocation(transcript: str, text: str) -> RelocationClass:
    """Classify one emitted span against the unmodified locator."""
    result = classify_patient_span_relocation(transcript, text)
    if result == "relocated":
        return "relocated"
    if result == "no_match":
        return "no_match"
    if result == "ambiguous":
        return "ambiguous"
    raise RuntimeError(f"unexpected relocation class: {result!r}")


def format_span_answer(label: str, span_texts: tuple[str, ...]) -> str:
    """Build the assistant target string used in span-bearing LoRA data."""
    if not span_texts:
        return label
    quoted = " ".join(f'"{text}"' for text in span_texts)
    return f"{label}: {quoted}"
