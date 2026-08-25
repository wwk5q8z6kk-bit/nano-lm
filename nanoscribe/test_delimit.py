"""Invariants for the E-DELIMIT output-format module.

These pin the guards the pre-registration rests on (PREREG_E_DELIMIT.md §P3–P7).
Each one reconstructs a way arm B could silently measure something other than
the model, and asserts it does not happen.
"""

from __future__ import annotations

import pytest

from nanoscribe import delimit, leakage
from nanoscribe.adapt import run_pipeline
from nanoscribe.adapters import FixtureSpanPortAdapter
from nanoscribe.campaign_datasets import campaign_cases
from nanoscribe.prompt import question_template_hash

_LABEL = {"ASSERTED": "STATED", "DENIED": "DENIED", "UNCERTAIN": "UNCERTAIN"}


def _gold_spans_by_atom(case) -> dict[str, tuple[int, int]]:
    """Local copy — importing run_eval would re-exec the interpreter via
    ``venv_boot.ensure_venv`` and silently terminate the pytest process."""
    by_id = {ev.evidence_id: ev for ev in case.gold.evidence}
    out: dict[str, tuple[int, int]] = {}
    for atom in case.gold.atoms:
        spans = [by_id[e] for e in atom.evidence_ids if e in by_id]
        if spans:
            out[atom.atom_id] = (spans[0].start, spans[0].end)
    return out


@pytest.fixture
def menu_arm(monkeypatch):
    monkeypatch.setattr(delimit, "OUTPUT_FORMAT", "menu")
    monkeypatch.setattr(leakage, "PROMPT_ANSWER_TEMPLATE_GOLD_VALUE", False)
    monkeypatch.setattr(leakage, "PARSER_RAW_VALUE_FALLBACK", False)


def _cases():
    return campaign_cases("campaign_v2")


def _score(mode: str) -> tuple[int, int]:
    """exact_gold_span over gold-bearing slots, for a synthetic menu picker."""
    hits = total = 0
    for case in _cases():
        source = case.model_input.source
        gold = _gold_spans_by_atom(case)
        states = {a.atom_id: a.assertion_state.value.upper() for a in case.gold.atoms}
        lines = {}
        for spec in case.atom_specs:
            menu = delimit.menu_for_slot(source, spec.atom_id)
            label = _LABEL.get(states.get(spec.atom_id, ""), "STATED")
            if mode == "oracle" and spec.atom_id in gold:
                start, end = gold[spec.atom_id]
                index = next(
                    (i for i, c in enumerate(menu) if c.start == start and c.end == end),
                    None,
                )
                lines[spec.atom_id] = (
                    f"{label}: [{index}]" if index is not None else "NOT_MENTIONED"
                )
            else:
                lines[spec.atom_id] = "STATED: [0]"
        batch = FixtureSpanPortAdapter(lines=lines).propose(
            case.model_input, case.atom_specs
        )
        _, report = run_pipeline(case.model_input, batch, gold=case.gold)
        for item in report.atom_results:
            if item.atom_id in gold:
                total += 1
                hits += bool(item.exact_gold_span)
    return hits, total


def test_every_gold_span_is_reachable_in_its_menu(menu_arm):
    """P5. A generator miss would read as H5 REFUTED — the expensive wrong call."""
    missing = []
    for case in _cases():
        source = case.model_input.source
        for atom_id, (start, end) in _gold_spans_by_atom(case).items():
            if not delimit.gold_in_menu(source, atom_id, start, end):
                missing.append((case.encounter_id, atom_id))
    assert missing == [], f"gold unreachable in menu for {len(missing)} slots: {missing[:5]}"


def test_oracle_picker_reaches_ceiling(menu_arm):
    """The arm can express a win: nothing downstream of the menu blocks scoring."""
    hits, total = _score("oracle")
    assert (hits, total) == (120, 120)


def test_index_zero_parrot_scores_at_chance(menu_arm):
    """R5. If the parrot scored high, arm B would be measuring menu construction."""
    hits, total = _score("parrot")
    assert hits / total < 0.10, f"index-0 parrot scored {hits}/{total} — menu order leaks gold"


def test_question_is_byte_identical_across_arms(monkeypatch):
    """R2. The contrast is legal only if the question template matches."""
    specs = [s for c in _cases() for s in c.atom_specs]
    hashes = set()
    for arm in delimit.ARMS:
        monkeypatch.setattr(delimit, "OUTPUT_FORMAT", arm)
        hashes.add(question_template_hash(specs))
    assert len(hashes) == 1, f"question template varies across arms: {hashes}"


def test_output_format_hash_differs_across_arms(monkeypatch):
    """R1. The arms must differ where they are supposed to, and be told apart."""
    hashes = set()
    for arm in delimit.ARMS:
        monkeypatch.setattr(delimit, "OUTPUT_FORMAT", arm)
        hashes.add(delimit.output_format_hash())
    assert len(hashes) == len(delimit.ARMS)


def test_candidates_never_cross_a_turn_boundary():
    """Same invariant ``select._span_in_turn`` enforces on the scoring side."""
    for case in _cases():
        source = case.model_input.source
        for cand in delimit.candidates_for_source(source):
            assert any(
                t.start <= cand.start and cand.end <= t.end for t in source.turns
            ), f"candidate {cand.text!r} crosses a turn boundary"


def test_free_form_arm_is_the_identity(monkeypatch):
    """Arm A's behaviour must be unchanged by the existence of this module."""
    monkeypatch.setattr(delimit, "OUTPUT_FORMAT", "free_form")
    source = _cases()[0].model_input.source
    quotes = ("My neck has been hurting.",)
    assert delimit.resolve_quotes('STATED: "x"', source, "atom-neck", quotes) == quotes


def test_menu_order_is_deterministic_and_slot_keyed():
    """P4. Same slot reproduces; different slots permute."""
    source = _cases()[0].model_input.source
    first = [c.text for c in delimit.menu_for_slot(source, "atom-neck")]
    assert first == [c.text for c in delimit.menu_for_slot(source, "atom-neck")]
    other = [c.text for c in delimit.menu_for_slot(source, "atom-alg")]
    assert set(first) == set(other)
    assert first != other
