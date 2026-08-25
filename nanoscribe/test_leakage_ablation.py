"""Pins for the span-port leakage ablation instrument.

Two load-bearing tests here:

``test_pure_echo_model_is_caught`` — the manipulation check. If a model that
does nothing but echo the value named in its prompt still scored clean, a null
result from the ablation would be uninterpretable (Stage P/P1 shipped a VOID
for exactly this reason).

``test_prompts_stay_distinct_in_every_scoring_cell`` — the confound guard. An
ablation cell that makes the task underdetermined measures prompt ambiguity,
not leakage, and would fire the CONFIRMED branch on an artifact.
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
from nanoscribe.campaign_datasets import campaign_cases, instance_cases
from nanoscribe.campaign_instances import INSTANCE_IDS, instance
from nanoscribe.run_eval import run_campaign_eval

SLOTS_PER_INSTANCE = 16
N_INSTANCES = len(INSTANCE_IDS)
SUITE_SLOTS = SLOTS_PER_INSTANCE * N_INSTANCES
ENC4_ABSENT_SLOTS = len(instance("i0").e4_absent)


def _case(encounter_id: str):
    for case in campaign_cases("campaign_v2"):
        if case.encounter_id == encounter_id:
            return case
    raise AssertionError(f"missing {encounter_id}")


def _adapter_lines(case, lines: dict[str, str]) -> ModelCandidateBatch:
    """Build a candidate batch from explicit per-slot span-port answers."""
    atoms = [
        candidate_from_span_port_line(
            atom_id=spec.atom_id,
            atom_type=spec.atom_type,
            raw_value=spec.raw_value,
            raw_line=lines[spec.atom_id],
            speaker=spec.speaker,
            experiencer=spec.experiencer,
            temporality=spec.temporality,
        )
        for spec in case.atom_specs
    ]
    return ModelCandidateBatch(atoms=tuple(atoms))


def _enc4_lines(absent_answer: str | None) -> dict[str, str]:
    """enc-4 answers: honest abstention, or (None) assert the absent value."""
    lines = {"atom-throat": 'STATED: "sore"'}
    for atom_id, _atom_type, value in instance("i0").e4_absent:
        lines[atom_id] = absent_answer or f'STATED: "{value}"'
    return lines


def _instance_prompts(instance_id: str) -> list[str]:
    from nanoscribe.prompt import build_span_port_prompt

    return [
        build_span_port_prompt(case.model_input.source, spec)
        for case in instance_cases(instance_id)
        for spec in case.atom_specs
    ]


def _all_prompts() -> list[str]:
    from nanoscribe.prompt import build_span_port_prompt

    return [
        build_span_port_prompt(case.model_input.source, spec)
        for case in campaign_cases("campaign_v2")
        for spec in case.atom_specs
    ]


class LeakageInstrumentTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (
            leakage.PROMPT_QUESTION_NAMES_CONCEPT,
            leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE,
            leakage.PARSER_RAW_VALUE_FALLBACK,
        )

    def tearDown(self) -> None:
        (
            leakage.PROMPT_QUESTION_NAMES_CONCEPT,
            leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE,
            leakage.PARSER_RAW_VALUE_FALLBACK,
        ) = self._saved

    def test_campaign_v2_is_a_superset_of_v1(self) -> None:
        v1 = [case.encounter_id for case in campaign_cases("campaign_v1")]
        v2 = [case.encounter_id for case in campaign_cases("campaign_v2")]
        # v2 leads with instance i0, whose encounter ids are unsuffixed, so the
        # historical partition appears verbatim at the head of the suite.
        self.assertEqual(v2[: len(v1)], v1)
        self.assertEqual(v2[len(v1)], "enc-4")
        self.assertEqual(len(v2), N_INSTANCES * 5)

    def test_fixture_ceiling_is_stable_across_scoring_cells(self) -> None:
        """A perfect reader scores the same in every cell — only real models differ."""
        leakage.PROMPT_QUESTION_NAMES_CONCEPT = True
        seen = set()
        for c1 in (True, False):
            for c2 in (True, False):
                leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = c1
                leakage.PARSER_RAW_VALUE_FALLBACK = c2
                agg = run_campaign_eval("campaign_v2", fixture_only=True)[
                    "suite_aggregate"
                ]
                seen.add(
                    (
                        agg["atoms"],
                        agg["exact_gold_span"],
                        agg["assertion_state_correct"],
                        agg["correct_abstention"],
                        agg["spurious_atom"],
                        agg["critical_error"],
                    )
                )
        self.assertEqual(
            seen,
            {(SUITE_SLOTS, 10 * N_INSTANCES, 10 * N_INSTANCES, 6 * N_INSTANCES, 0, 0)},
        )

    def test_prompts_stay_distinct_within_every_instance_and_cell(self) -> None:
        """Distinctness is a PER-INSTANCE property; pooling makes it vacuous.

        A model only ever sees one instance's prompts, so two labels colliding
        inside an instance is the failure that matters. Pooled over 12
        instances the count is trivially satisfied even when that happens,
        because the resampled surface values keep the pooled strings apart.

        Checked under C3-off specifically (prereg s7 item 2): with the gold
        surface string removed from the question, the concept labels are the
        only thing carrying slot identity, so that is the cell where a
        collision would actually make the task underdetermined.
        """
        leakage.PROMPT_QUESTION_NAMES_CONCEPT = True
        for c3 in (True, False):
            for c1 in (True, False):
                leakage.PROMPT_QUESTION_USES_GOLD_SURFACE = c3
                leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = c1
                for instance_id in INSTANCE_IDS:
                    prompts = _instance_prompts(instance_id)
                    where = f"instance={instance_id} C3={c3} C1={c1}"
                    self.assertEqual(len(prompts), SLOTS_PER_INSTANCE, where)
                    self.assertEqual(
                        len(set(prompts)), SLOTS_PER_INSTANCE, f"collision: {where}"
                    )

    def test_concept_labels_alone_are_distinct_within_an_instance(self) -> None:
        """Under C3-off the label is the whole of slot identity — pin it directly."""
        for instance_id in INSTANCE_IDS:
            labels = [
                spec.concept_label
                for case in instance_cases(instance_id)
                for spec in case.atom_specs
            ]
            self.assertEqual(len(labels), SLOTS_PER_INSTANCE, instance_id)
            self.assertEqual(len(set(labels)), SLOTS_PER_INSTANCE, instance_id)
            self.assertTrue(all(labels), f"{instance_id}: a slot has no label")

    def test_slot_type_only_questions_would_collide(self) -> None:
        """Why Q_ID is never flipped: the rejected alternative is degenerate.

        Measured with the leakage channels CLOSED, which is the only
        configuration where the comparison means anything — with C1 on, the
        answer template still carries the gold value and keeps the prompts
        superficially distinct, hiding the collision rather than removing it.
        Kept executable so the reason the 2x2x2 has no Q_ID arm stays checkable
        rather than remembered.
        """
        leakage.PROMPT_QUESTION_NAMES_CONCEPT = False
        leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = False
        for instance_id in INSTANCE_IDS:
            prompts = _instance_prompts(instance_id)
            self.assertLess(
                len(set(prompts)),
                SLOTS_PER_INSTANCE,
                f"{instance_id}: slot-type-only should be underdetermined",
            )

    def test_instance_count_meets_the_prereg_floor(self) -> None:
        """Prereg s10: fewer than 12 instances voids the round.

        The paired-interaction MDE at n=5 is ~3.7 slots, wider than H-leak's own
        3-slot threshold — the round could miss an effect at exactly the size it
        exists to call. Guarded here so the count cannot drift down unnoticed.
        """
        self.assertGreaterEqual(N_INSTANCES, 12)

    def test_c1_removes_the_gold_value_from_the_answer_template(self) -> None:
        leakage.PROMPT_QUESTION_NAMES_CONCEPT = True
        leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = True
        on = run_campaign_eval("campaign_v2", fixture_only=True)["suite_aggregate"]
        leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = False
        off = run_campaign_eval("campaign_v2", fixture_only=True)["suite_aggregate"]
        self.assertGreaterEqual(on["gold_in_answer_template"], 13 * N_INSTANCES)
        self.assertEqual(off["gold_in_answer_template"], 0)
        # Q is untouched: the question still identifies the concept in both.
        self.assertEqual(on["gold_in_question"], off["gold_in_question"])

    def test_c2_fallback_only_fires_without_a_quote(self) -> None:
        spec = _case("enc-4").atom_specs[0]
        leakage.PARSER_RAW_VALUE_FALLBACK = True
        on = candidate_from_span_port_line(
            atom_id="a", atom_type=spec.atom_type, raw_value="sore", raw_line="STATED"
        )
        leakage.PARSER_RAW_VALUE_FALLBACK = False
        off = candidate_from_span_port_line(
            atom_id="a", atom_type=spec.atom_type, raw_value="sore", raw_line="STATED"
        )
        self.assertEqual(on.quotes, ("sore",))
        self.assertEqual(off.quotes, ())
        # With a quote present the flag is irrelevant.
        for flag in (True, False):
            leakage.PARSER_RAW_VALUE_FALLBACK = flag
            got = candidate_from_span_port_line(
                atom_id="a",
                atom_type=spec.atom_type,
                raw_value="sore",
                raw_line='STATED: "sore"',
            )
            self.assertEqual(got.quotes, ("sore",))

    def test_pure_echo_model_is_caught(self) -> None:
        """MANIPULATION CHECK — the instrument must punish prompt-echoing.

        On campaign_v1 an echo model looks near-perfect, because every slot's
        gold value is present. On the added enc-4 slots the echoed value cannot
        bind to the source, and the two outcomes must separate cleanly:

            honest abstainer  -> correct_abstention 5, unbound_assertion 0
            prompt parrot     -> correct_abstention 0, unbound_assertion 5

        The constrained selector still protects the record either way — that is
        a system-level safety property. The point of ``unbound_assertion`` is
        that model quality must not be credited for the binder's save. If this
        separation ever collapses, a null result from the ablation is
        uninterpretable rather than reassuring.

        The echo answers are built by reading the PROMPT, never from
        ``spec.raw_value``. A gold-constructed baseline is byte-identical in all
        eight ablation cells and therefore cannot detect a channel at all; see
        test_adversarial_baseline_invariant, which pins that rule and keeps the
        banned shape executable as a counter-example.
        """
        from nanoscribe.test_prompt_surface_parrot import parrot_line

        leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = True
        leakage.PROMPT_QUESTION_USES_GOLD_SURFACE = True

        v1_clean = 0
        for encounter_id in ("enc-1", "enc-2", "enc-3"):
            case = _case(encounter_id)
            lines = {
                spec.atom_id: parrot_line(case.model_input.source, spec)
                for spec in case.atom_specs
            }
            batch = _adapter_lines(case, lines)
            _, report = run_pipeline(case.model_input, batch, gold=case.gold)
            v1_clean += report.exact_gold_span
        # Echoing looks good on the suite that has no absent slots.
        self.assertGreaterEqual(v1_clean, 5)

        case = _case("enc-4")
        parrot_lines = {
            spec.atom_id: parrot_line(case.model_input.source, spec)
            for spec in case.atom_specs
        }
        _, echo_report = run_pipeline(
            case.model_input, _adapter_lines(case, parrot_lines), gold=case.gold
        )
        self.assertEqual(echo_report.unbound_assertion, ENC4_ABSENT_SLOTS)
        self.assertEqual(echo_report.correct_abstention, 0)

        _, honest_report = run_pipeline(
            case.model_input,
            _adapter_lines(case, _enc4_lines("NOT_MENTIONED")),
            gold=case.gold,
        )
        self.assertEqual(honest_report.correct_abstention, ENC4_ABSENT_SLOTS)
        self.assertEqual(honest_report.unbound_assertion, 0)

    def test_binder_no_longer_launders_hallucination_into_abstention(self) -> None:
        """Regression pin for the defect this instrument exposed.

        Before the ``unbound_assertion`` split, a model asserting a value that
        occurs nowhere in the source was scored as a *correct abstention* — the
        harness credited a hallucination as safe behaviour.
        """
        case = _case("enc-4")
        predicted, report = run_pipeline(
            case.model_input, _adapter_lines(case, _enc4_lines(None)), gold=case.gold
        )
        hallucinated = [
            atom for atom in predicted.atoms if atom.atom_id != "atom-throat"
        ]
        self.assertEqual(len(hallucinated), ENC4_ABSENT_SLOTS)
        self.assertTrue(all(atom.unbound_assertion for atom in hallucinated))
        self.assertEqual(report.correct_abstention, 0)


if __name__ == "__main__":
    unittest.main()
