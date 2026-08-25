"""BLOCKING manipulation check: does the prompt hand over the answer?

PREREG section 7 item 1. This is the check the previous instrument lacked.

``test_leakage_ablation.py::test_pure_echo_model_is_caught`` builds its echo
from ``spec.raw_value`` directly, so it never reads a prompt surface and is
cell-invariant by construction: it passes identically in every cell and cannot
detect *which* surface leaked. It tests the scorer, not the prompt.

The parrot here reads ONLY the prompt text. It discards the transcript, then
asserts the first properly-delimited quoted string the instructions name. If it
can score, the instrument is handing over the answer through some surface — and
a null result from the ablation is uninterpretable rather than reassuring.

Delimiter handling matters. The question is built with ``!r``, so surface
strings arrive as ``'value'`` (or ``"value"`` when the value itself contains an
apostrophe), while the answer template uses ``"value"``. A naive ``'([^']+)'``
also matches from the apostrophe inside a possessive concept label — e.g.
"the clinician's stated impression" — and spans to the next apostrophe,
manufacturing garbage that fails to bind and shows up as spurious
``unbound_assertion``. The lookarounds below require the delimiters to sit at
token boundaries, so possessives cannot open a match.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from nanoscribe import leakage
from nanoscribe.adapt import ModelCandidateBatch, candidate_from_span_port_line, run_pipeline
from nanoscribe.campaign_datasets import campaign_cases
from nanoscribe.prompt import build_span_port_prompt

# A quote delimiter may not be flanked by a word character, so the apostrophe in
# "clinician's" cannot open a match.
_QUOTED = re.compile(r"(?<!\w)'([^']+)'(?!\w)|(?<!\w)\"([^\"]+)\"(?!\w)")


def parrot_line(source, spec) -> str:
    """Answer built ONLY from the prompt's instructions; transcript discarded."""
    prompt = build_span_port_prompt(source, spec)
    # Everything before the first blank line is the transcript. Throw it away:
    # whatever the parrot scores after this point, it scored without reading.
    _, _, instructions = prompt.partition("\n\n")
    match = _QUOTED.search(instructions)
    if not match:
        return "NOT_MENTIONED"
    return f'STATED: "{match.group(1) or match.group(2)}"'


def parrot_totals() -> dict[str, int]:
    totals = {
        "exact_gold_span": 0,
        "assertion_state_correct": 0,
        "correct_abstention": 0,
        "unbound_assertion": 0,
        "slots": 0,
    }
    for case in campaign_cases("campaign_v2"):
        atoms = []
        for spec in case.atom_specs:
            atoms.append(
                candidate_from_span_port_line(
                    atom_id=spec.atom_id,
                    atom_type=spec.atom_type,
                    raw_value=spec.raw_value,
                    raw_line=parrot_line(case.model_input.source, spec),
                    speaker=spec.speaker,
                    experiencer=spec.experiencer,
                    temporality=spec.temporality,
                )
            )
            totals["slots"] += 1
        _, report = run_pipeline(
            case.model_input, ModelCandidateBatch(atoms=tuple(atoms)), gold=case.gold
        )
        for key in ("exact_gold_span", "assertion_state_correct", "correct_abstention", "unbound_assertion"):
            totals[key] += getattr(report, key)
    return totals


class PromptSurfaceParrotTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (
            leakage.PROMPT_QUESTION_NAMES_CONCEPT,
            leakage.PROMPT_QUESTION_USES_GOLD_SURFACE,
            leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE,
            leakage.PARSER_RAW_VALUE_FALLBACK,
        )
        leakage.PROMPT_QUESTION_NAMES_CONCEPT = True

    def tearDown(self) -> None:
        (
            leakage.PROMPT_QUESTION_NAMES_CONCEPT,
            leakage.PROMPT_QUESTION_USES_GOLD_SURFACE,
            leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE,
            leakage.PARSER_RAW_VALUE_FALLBACK,
        ) = self._saved

    def _set(self, *, q_surface: bool, c1: bool, c2: bool = True) -> None:
        leakage.PROMPT_QUESTION_USES_GOLD_SURFACE = q_surface
        leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = c1
        leakage.PARSER_RAW_VALUE_FALLBACK = c2

    def test_parrot_scores_when_any_surface_carries_the_gold_value(self) -> None:
        """The check must be able to FIRE, or its silence means nothing."""
        for q_surface, c1 in ((True, True), (True, False), (False, True)):
            with self.subTest(q_surface=q_surface, c1=c1):
                self._set(q_surface=q_surface, c1=c1)
                totals = parrot_totals()
                self.assertGreater(
                    totals["exact_gold_span"],
                    totals["slots"] // 4,
                    "a parrot with a gold surface available should score well "
                    "above chance; if it does not, this check has gone blind",
                )

    def test_parrot_collapses_when_every_gold_surface_is_closed(self) -> None:
        """PREREG section 7 item 1 — the condition L000 must be honest.

        With the question naming only the concept label and the answer template
        carrying no gold value, nothing in the prompt names the surface string.
        A parrot must then score exactly zero: no span, no assertion state, and
        no unbound assertion either (it has nothing to assert, so it abstains).
        """
        self._set(q_surface=False, c1=False)
        totals = parrot_totals()
        self.assertEqual(totals["exact_gold_span"], 0)
        self.assertEqual(totals["assertion_state_correct"], 0)
        self.assertEqual(
            totals["unbound_assertion"],
            0,
            "a correct parrot abstains everywhere here; residual unbound "
            "assertions mean the extractor is manufacturing spans",
        )

    def test_closing_the_question_channel_is_what_collapses_it(self) -> None:
        """Attribution: the collapse is caused by Q_SURFACE, not by C1 alone."""
        self._set(q_surface=True, c1=False)
        q_open = parrot_totals()["exact_gold_span"]
        self._set(q_surface=False, c1=False)
        q_closed = parrot_totals()["exact_gold_span"]
        self.assertGreater(q_open, 0)
        self.assertEqual(q_closed, 0)


if __name__ == "__main__":
    unittest.main()
