"""Gold-value leakage channels in the span-port measurement path.

The P1 span-port harness scores a model on whether it recovers the gold
evidence span for a slot described by an ``AtomSpec``. That spec carries the
gold ``raw_value``. Two places in the current path put that gold string back
into the measurement:

``PROMPT_INCLUDES_GOLD_VALUE`` (channel C1)
    ``prompt.build_span_port_prompt`` interpolates ``spec.raw_value`` into both
    the task question and the answer template, so the prompt can read
    ``Does the patient mention 'neck'? ... reply STATED: "neck"`` for the atom
    whose gold answer is ``STATED: "neck"``. A model that echoes the
    instruction scores the same as a model that read the transcript.

``PARSER_RAW_VALUE_FALLBACK`` (channel C2)
    ``adapt.candidate_from_span_port_line`` substitutes ``raw_value`` as the
    model's quote when the model emitted a label with no quote, so a bare
    ``STATED`` resolves to the exact gold span.

Both were introduced together in ``dc3b310`` to lift campaign_v1 coverage off
zero. They are separable, so each experiment branch sets these two flags and
changes nothing else; the run report echoes them under ``leakage_config``.

Flipping a flag changes measurement semantics, never model weights or the run
command — that is the point of keeping them here rather than in an env var.
"""

from __future__ import annotations

# Channel C1 — gold raw_value interpolated into the prompt.
PROMPT_INCLUDES_GOLD_VALUE = True

# Channel C2 — gold raw_value substituted for a missing model quote.
PARSER_RAW_VALUE_FALLBACK = True


def leakage_config() -> dict[str, bool]:
    """Active leakage channels, recorded in every run report."""
    return {
        "prompt_includes_gold_value": PROMPT_INCLUDES_GOLD_VALUE,
        "parser_raw_value_fallback": PARSER_RAW_VALUE_FALLBACK,
    }


def condition_label() -> str:
    """Short factorial cell name: C1/C2 on-off."""
    c1 = "C1on" if PROMPT_INCLUDES_GOLD_VALUE else "C1off"
    c2 = "C2on" if PARSER_RAW_VALUE_FALLBACK else "C2off"
    return f"{c1}_{c2}"
