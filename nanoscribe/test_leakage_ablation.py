"""Pins for the span-port leakage ablation instrument.

The load-bearing test here is ``test_pure_echo_model_is_caught`` — the
manipulation check. If a model that does nothing but echo the value named in
its prompt still scored clean, then a null result from the ablation would be
uninterpretable (Stage P/P1 shipped a VOID for exactly this reason). These
tests assert the instrument has the discriminating power the experiment needs.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from nanoscribe import leakage
from nanoscribe.adapt import run_pipeline
from nanoscribe.adapters import AtomSpec, CandidateAtom, ModelCandidateBatch
from nanoscribe.adapt import candidate_from_span_port_line
from nanoscribe.campaign_datasets import campaign_cases
from nanoscribe.evaluate import evaluate
from nanoscribe.run_eval import run_campaign_eval


class _EchoAdapter:
    """Worst case: answers STATED with the value its own prompt named."""

    model_id = "echo/prompt-parrot"

    def __init__(self, raw_line_sink: dict[str, str] | None = None) -> None:
        self.raw_line_sink = raw_line_sink

    def propose(self, model_input, atom_specs) -> ModelCandidateBatch:
        del model_input
        atoms: list[CandidateAtom] = []
        for spec in atom_specs:
            raw_line = f'STATED: "{spec.raw_value}"'
            if self.raw_line_sink is not None:
                self.raw_line_sink[spec.atom_id] = raw_line
            atoms.append(
                candidate_from_span_port_line(
                    atom_id=spec.atom_id,
                    atom_type=spec.atom_type,
                    raw_value=spec.raw_value,
                    raw_line=raw_line,
                    speaker=spec.speaker,
                    experiencer=spec.experiencer,
                    temporality=spec.temporality,
                )
            )
        return ModelCandidateBatch(atoms=tuple(atoms))


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


class LeakageInstrumentTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (
            leakage.PROMPT_INCLUDES_GOLD_VALUE,
            leakage.PARSER_RAW_VALUE_FALLBACK,
        )

    def tearDown(self) -> None:
        (
            leakage.PROMPT_INCLUDES_GOLD_VALUE,
            leakage.PARSER_RAW_VALUE_FALLBACK,
        ) = self._saved

    def test_campaign_v2_is_a_superset_of_v1(self) -> None:
        v1 = [case.encounter_id for case in campaign_cases("campaign_v1")]
        v2 = [case.encounter_id for case in campaign_cases("campaign_v2")]
        self.assertEqual(v2[: len(v1)], v1)
        self.assertEqual(v2[len(v1) :], ["enc-4", "enc-5"])

    def test_fixture_ceiling_is_stable_across_all_four_cells(self) -> None:
        """A perfect reader scores the same in every cell — only real models differ."""
        seen = set()
        for c1 in (True, False):
            for c2 in (True, False):
                leakage.PROMPT_INCLUDES_GOLD_VALUE = c1
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
        self.assertEqual(seen, {(13, 10, 10, 3, 0, 0)})

    def test_c1_removes_the_gold_value_from_the_instructions(self) -> None:
        leakage.PROMPT_INCLUDES_GOLD_VALUE = True
        on = run_campaign_eval("campaign_v2", fixture_only=True)["suite_aggregate"]
        leakage.PROMPT_INCLUDES_GOLD_VALUE = False
        off = run_campaign_eval("campaign_v2", fixture_only=True)["suite_aggregate"]
        self.assertEqual(on["gold_value_in_prompt"], 13)
        # enc-1's medication slot has raw_value "medication", which the
        # slot-type question unavoidably contains. Documented, not hidden.
        self.assertEqual(off["gold_value_in_prompt"], 1)

    def test_c2_fallback_only_fires_without_a_quote(self) -> None:
        spec = AtomSpec(atom_id="a", atom_type=_case("enc-4").atom_specs[0].atom_type,
                        raw_value="sore")
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

            honest abstainer  -> correct_abstention 2, unbound_assertion 0
            prompt parrot     -> correct_abstention 0, unbound_assertion 2

        The constrained selector still protects the record either way — that is
        a system-level safety property. The point of ``unbound_assertion`` is
        that model quality must not be credited for the binder's save. If this
        separation ever collapses, a null result from the ablation is
        uninterpretable rather than reassuring.
        """
        v1_clean = 0
        for encounter_id in ("enc-1", "enc-2", "enc-3"):
            case = _case(encounter_id)
            batch = _EchoAdapter().propose(case.model_input, case.atom_specs)
            _, report = run_pipeline(case.model_input, batch, gold=case.gold)
            v1_clean += report.exact_gold_span
        # Echoing looks good on the suite that has no absent slots.
        self.assertGreaterEqual(v1_clean, 5)

        case = _case("enc-4")
        echo = _EchoAdapter().propose(case.model_input, case.atom_specs)
        _, echo_report = run_pipeline(case.model_input, echo, gold=case.gold)
        self.assertEqual(echo_report.unbound_assertion, 2)
        self.assertEqual(echo_report.correct_abstention, 0)

        honest = _adapter_lines(
            case,
            {
                "atom-throat": 'STATED: "sore"',
                "atom-absent-med": "NOT_MENTIONED",
                "atom-absent-fever": "NOT_MENTIONED",
            },
        )
        _, honest_report = run_pipeline(case.model_input, honest, gold=case.gold)
        self.assertEqual(honest_report.correct_abstention, 2)
        self.assertEqual(honest_report.unbound_assertion, 0)

    def test_binder_no_longer_launders_hallucination_into_abstention(self) -> None:
        """Regression pin for the defect this instrument exposed.

        Before the ``unbound_assertion`` split, a model asserting a value that
        occurs nowhere in the source was scored as a *correct abstention* — the
        harness credited a hallucination as safe behaviour.
        """
        case = _case("enc-4")
        batch = _adapter_lines(
            case,
            {
                "atom-throat": 'STATED: "sore"',
                "atom-absent-med": 'STATED: "lisinopril"',
                "atom-absent-fever": 'STATED: "fever"',
            },
        )
        predicted, report = run_pipeline(case.model_input, batch, gold=case.gold)
        hallucinated = [
            atom for atom in predicted.atoms if atom.atom_id != "atom-throat"
        ]
        self.assertTrue(all(atom.unbound_assertion for atom in hallucinated))
        self.assertEqual(report.correct_abstention, 0)


if __name__ == "__main__":
    unittest.main()
