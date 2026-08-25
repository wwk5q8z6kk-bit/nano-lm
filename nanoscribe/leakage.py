"""Gold-value leakage channels in the span-port measurement path.

The P1 span-port harness scores a model on whether it recovers the gold
evidence span for a slot described by an ``AtomSpec``. That spec carries the
gold ``raw_value``, and several places in the current path put that gold string
back into the measurement.

Not every mention of the value is leakage, and the distinction is the whole
design of this ablation:

``PROMPT_QUESTION_NAMES_CONCEPT`` (Q — MEASURED TO LEAK; see the hold note
below, this flag's framing is wrong and is being replaced by Q_SURFACE)
    The question names the concept being asked about ("Does the patient mention
    'migraines'?"). Span-port legitimately *is* "given a clinical concept,
    locate its assertion and its evidence", so naming the concept is how the
    slot is identified at all. Turning Q off does not produce a clean
    measurement — it produces an **underdetermined** one: slots that share an
    ``atom_type`` collapse to byte-identical prompts (enc-4's ``atom-throat``
    and ``atom-absent-fever`` are both SYMPTOM), so the model cannot tell which
    slot it is answering. Q-off is kept as a labelled diagnostic cell only; no
    verdict rests on it. ``test_prompts_stay_distinct`` pins this.

``PROMPT_ANSWER_TEMPLATE_GOLD_VALUE`` (C1 — leakage)
    The answer template hands over the exact string to emit — ``reply STATED:
    "{raw_value}"`` for the atom whose gold answer is ``STATED: "neck"`` — and
    the system prompt's format examples are themselves gold answers for
    enc-1/atom-neck, enc-1/atom-alg and enc-2/atom-chest. A model that copies
    its instructions scores the same as one that read the transcript. Removing
    this leaves the task fully specified: the question still says which concept
    to look for, it just no longer dictates the answer.

``PARSER_RAW_VALUE_FALLBACK`` (C2 — leakage, inside the scorer)
    ``adapt.candidate_from_span_port_line`` substitutes ``raw_value`` as the
    model's quote when the model emitted a label with no quote, so a bare
    ``STATED`` resolves to the exact gold span. Note this only fires on
    quote-less output: read ``quote_absent`` from the run report before making
    any claim about C2, since if it is 0 the C2 cells carry no information.

Provenance (corrected 2026-08-25 — the ledger audit turns on this). C1 and C2
did NOT arrive together. C2 is new in ``dc3b310`` (``adapt.py``: the
``not quotes and raw_value`` fallback, absent at ``09745ec``). A
gold-value-in-prompt channel predates it: ``09745ec``'s ``prompt.py`` already
interpolated ``{spec.raw_value!r}`` into the question and already shipped
``Example: STATED: "neck"`` in the system prompt. ``dc3b310`` widened the
prompt channel and added the scorer channel.

Each experiment branch sets these flags and changes nothing else; the run
report echoes them under ``leakage_config``.

Flipping a flag changes measurement semantics, never model weights or the run
command — that is the point of keeping them here rather than in an env var.

HOLD (2026-08-25) — do not launch the 2x2 on these flags as they stand.
Splitting Q out of C1 was meant to keep the task specified while closing the
leak. It does not: the question identifies the slot BY ITS GOLD SURFACE STRING,
so with Q pinned on the answer is in the prompt in all four cells, including the
one labelled leakage-free. Measured on i0, fixture-only, with a parrot that
discards the transcript and asserts the first surface string the instructions
name (perfect-reader ceiling is exact_span 10/16):

    C1on_C2on   10/16   C1on_C2off  10/16
    C1off_C2on   9/16   C1off_C2off  9/16

Zero transcript access lands within one slot of the ceiling in every cell, so
``exact_gold_span`` is saturated and the REFUTED branch of the prereg is
uninterpretable. The joint table still separates parrot from reader
(state_ok 6/16 vs 10/16; correct_abstention 0-3/6 vs 6/6; unbound_assertion 3-6
vs 0), which is the argument for making the discrimination the primary endpoint
rather than a single accuracy scalar.

Fix in progress: a per-slot ``concept_label`` distinct from ``raw_value``, so
the question can specify the slot without naming the surface form, turning Q
into a genuinely ablatable ``Q_SURFACE`` channel.
"""

from __future__ import annotations

# Q — question names the concept. Task specification; off = underdetermined.
PROMPT_QUESTION_NAMES_CONCEPT = True

# C1 — answer template and system-prompt examples carry the gold value.
PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = True

# C2 — gold raw_value substituted for a missing model quote.
PARSER_RAW_VALUE_FALLBACK = True


def leakage_config() -> dict[str, bool]:
    """Active channels, recorded in every run report."""
    return {
        "prompt_question_names_concept": PROMPT_QUESTION_NAMES_CONCEPT,
        "prompt_answer_template_gold_value": PROMPT_ANSWER_TEMPLATE_GOLD_VALUE,
        "parser_raw_value_fallback": PARSER_RAW_VALUE_FALLBACK,
    }


def condition_label() -> str:
    """Short factorial cell name."""
    return "_".join(
        (
            "C1on" if PROMPT_ANSWER_TEMPLATE_GOLD_VALUE else "C1off",
            "C2on" if PARSER_RAW_VALUE_FALLBACK else "C2off",
            "Qon" if PROMPT_QUESTION_NAMES_CONCEPT else "Qoff",
        )
    )
