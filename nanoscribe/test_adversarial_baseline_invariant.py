"""Program invariant: an adversarial baseline must consume the channel under test.

Generalised from a defect in this repo's own first manipulation check. That
check built its answers from ``spec.raw_value`` — the gold field — so its output
was identical no matter which leakage channel was open or closed. It could never
fail, and it read as reassurance: the pre-registered REFUTED branch was "moves
<=1 slot across all four cells", which is precisely the signature of a channel
open in all four. A decision rule whose null branch is indistinguishable from
total confound is worse than no rule.

The invariant, stated so it can be checked rather than remembered:

    An adversarial baseline must consume EXACTLY the channel under test and no
    other source. Concretely, its score must VARY across the ablation cells. A
    baseline that scores identically everywhere is not measuring the channel —
    it is reporting its own construction.

The parrot in test_prompt_surface_parrot.py satisfies this: it reads the prompt
and nothing else, so closing a prompt channel provably starves it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from nanoscribe import leakage
from nanoscribe.adapt import (
    ModelCandidateBatch,
    candidate_from_span_port_line,
    run_pipeline,
)
from nanoscribe.campaign_datasets import instance_cases
from nanoscribe.campaign_instances import INSTANCE_IDS
from nanoscribe.test_prompt_surface_parrot import parrot_line

CELLS = [
    (c1, c2, c3)
    for c1 in (True, False)
    for c2 in (True, False)
    for c3 in (True, False)
]


def _score(line_for) -> tuple[int, int, int]:
    """(exact_gold_span, correct_abstention, unbound_assertion) over one instance."""
    exact = abstain = unbound = 0
    for case in instance_cases(INSTANCE_IDS[0]):
        atoms = [
            candidate_from_span_port_line(
                atom_id=spec.atom_id,
                atom_type=spec.atom_type,
                raw_value=spec.raw_value,
                raw_line=line_for(case, spec),
                speaker=spec.speaker,
                experiencer=spec.experiencer,
                temporality=spec.temporality,
            )
            for spec in case.atom_specs
        ]
        _, report = run_pipeline(
            case.model_input, ModelCandidateBatch(atoms=tuple(atoms)), gold=case.gold
        )
        exact += report.exact_gold_span
        abstain += report.correct_abstention
        unbound += report.unbound_assertion
    return exact, abstain, unbound


def _from_prompt(case, spec) -> str:
    """Legitimate: consumes only what the model would be shown."""
    return parrot_line(case.model_input.source, spec)


def _from_gold(case, spec) -> str:
    """Banned shape, kept executable as the counter-example."""
    return f'STATED: "{spec.raw_value}"'


class AdversarialBaselineInvariantTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (
            leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE,
            leakage.PARSER_RAW_VALUE_FALLBACK,
            leakage.PROMPT_QUESTION_USES_GOLD_SURFACE,
        )

    def tearDown(self) -> None:
        (
            leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE,
            leakage.PARSER_RAW_VALUE_FALLBACK,
            leakage.PROMPT_QUESTION_USES_GOLD_SURFACE,
        ) = self._saved

    def _sweep(self, line_for) -> set[tuple[int, int, int]]:
        seen = set()
        for c1, c2, c3 in CELLS:
            leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = c1
            leakage.PARSER_RAW_VALUE_FALLBACK = c2
            leakage.PROMPT_QUESTION_USES_GOLD_SURFACE = c3
            seen.add(_score(line_for))
        return seen

    def test_prompt_consuming_baseline_varies_across_cells(self) -> None:
        """THE INVARIANT. If this ever collapses to one score, the check is dead."""
        seen = self._sweep(_from_prompt)
        self.assertGreater(
            len(seen),
            1,
            "parrot scores identically in all 8 cells — it is no longer "
            "consuming the prompt, so it cannot detect a prompt channel",
        )

    def test_prompt_consuming_baseline_collapses_when_channels_close(self) -> None:
        """It must actually reach the floor, not merely wobble."""
        leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = False
        leakage.PROMPT_QUESTION_USES_GOLD_SURFACE = False
        leakage.PARSER_RAW_VALUE_FALLBACK = False
        exact, _abstain, _unbound = _score(_from_prompt)
        self.assertEqual(exact, 0, "closed channels must starve the parrot")

    def test_gold_constructed_baseline_is_cell_invariant(self) -> None:
        """Counter-example, executable: why construction-from-gold is banned.

        Identical in all eight cells. Any ablation using this shape would report
        'no effect' with total confidence and zero information.
        """
        seen = self._sweep(_from_gold)
        self.assertEqual(
            len(seen),
            1,
            "expected the banned shape to be cell-invariant; if this now varies "
            "the counter-example has drifted and the docstring is misleading",
        )


class FormEquivalenceInvariantTest(unittest.TestCase):
    """SECOND INVARIANT: a cell must differ in exactly the manipulated variable.

    Generalised from a defect in this repo's own C3 arm. C3-off was meant to
    remove the gold surface string and nothing else; it also changed the
    question FORM, from yes/no ("Does the patient mention 'migraines'?") to
    wh-extraction ("What is a condition ...?"). The measured effect was
    dominated by form — under yes/no the model answers the yes/no question,
    which the harness reads as NOT_MENTIONED — so the LEAKIER cells scored
    worse and the arm was void.

    The prompt-distinctness guard did not catch this, and could not: it asserts
    the prompts are DIFFERENT, and they were. Distinctness is necessary, not
    sufficient. The rule this pins:

        An ablation cell must differ in exactly the manipulated variable —
        same task form, same answer space, same response mode.

    Enforced constructively: both arms are rendered from one template with the
    identifier substituted, so replacing the identifier with a placeholder must
    yield byte-identical strings.
    """

    def setUp(self) -> None:
        self._saved = leakage.PROMPT_QUESTION_USES_GOLD_SURFACE

    def tearDown(self) -> None:
        leakage.PROMPT_QUESTION_USES_GOLD_SURFACE = self._saved

    def _questions(self, c3: bool):
        from nanoscribe.prompt import identifier_for_spec, topic_for_spec

        leakage.PROMPT_QUESTION_USES_GOLD_SURFACE = c3
        out = {}
        for instance_id in INSTANCE_IDS:
            for case in instance_cases(instance_id):
                for spec in case.atom_specs:
                    key = f"{instance_id}/{spec.atom_id}"
                    out[key] = (topic_for_spec(spec), identifier_for_spec(spec))
        return out

    def test_both_c3_arms_share_one_question_form(self) -> None:
        from nanoscribe.prompt import IDENTIFIER_PLACEHOLDER, topic_with_placeholder

        on = self._questions(True)
        off = self._questions(False)
        self.assertEqual(set(on), set(off))
        for key in on:
            q_on, ident_on = on[key]
            q_off, ident_off = off[key]
            # Removing the identifier must leave the SAME sentence in both arms.
            self.assertEqual(
                q_on.replace(ident_on, IDENTIFIER_PLACEHOLDER),
                q_off.replace(ident_off, IDENTIFIER_PLACEHOLDER),
                f"{key}: C3 arms differ by more than the identifier",
            )
            self.assertNotEqual(ident_on, ident_off, key)

    def test_the_placeholder_render_matches_both_arms(self) -> None:
        """Guards the guard: the template must actually be what gets rendered."""
        from nanoscribe.prompt import IDENTIFIER_PLACEHOLDER, topic_with_placeholder

        for c3 in (True, False):
            from nanoscribe.prompt import identifier_for_spec, topic_for_spec

            leakage.PROMPT_QUESTION_USES_GOLD_SURFACE = c3
            for case in instance_cases(INSTANCE_IDS[0]):
                for spec in case.atom_specs:
                    rendered = topic_for_spec(spec).replace(
                        identifier_for_spec(spec), IDENTIFIER_PLACEHOLDER
                    )
                    self.assertEqual(rendered, topic_with_placeholder(spec), spec.atom_id)

    def test_response_mode_is_wh_extraction_in_both_arms(self) -> None:
        """Not yes/no. Pins the specific form that broke the first C3 arm."""
        from nanoscribe.prompt import topic_for_spec

        for c3 in (True, False):
            leakage.PROMPT_QUESTION_USES_GOLD_SURFACE = c3
            for case in instance_cases(INSTANCE_IDS[0]):
                for spec in case.atom_specs:
                    question = topic_for_spec(spec)
                    self.assertTrue(question.startswith("What "), question)
                    self.assertFalse(question.startswith("Does "), question)


if __name__ == "__main__":
    unittest.main()
