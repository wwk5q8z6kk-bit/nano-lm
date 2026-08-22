"""Span / evidence evaluator for Encounter Representation v0.

Does not collapse quality into one utility number. Exact gold span, character
F1, wrong-source-span, state/support, abstention, ambiguity, and critical
errors are counted separately. SOFTWARE only.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanoscribe.encounter import (
    AssertionState,
    EncounterError,
    EncounterRecord,
    EvidenceSpan,
    Source,
)
from nanoscribe.select import match_count


def char_span_f1(gold: tuple[int, int], pred: tuple[int, int]) -> float:
    gs, ge = gold
    ps, pe = pred
    inter = max(0, min(ge, pe) - max(gs, ps))
    if inter == 0 or ge == gs or pe == ps:
        return 0.0
    precision = inter / (pe - ps)
    recall = inter / (ge - gs)
    return 2 * precision * recall / (precision + recall)


def _span_in_source(source: Source, span: EvidenceSpan) -> bool:
    if span.end > len(source.text) or span.start < 0:
        return False
    return source.text[span.start : span.end] == span.text


@dataclass(frozen=True, slots=True)
class PredictedAtom:
    atom_id: str
    assertion_state: AssertionState | None = None
    spans: tuple[EvidenceSpan, ...] = ()
    abstained: bool = False
    malformed: bool = False
    quote: str | None = None


@dataclass(frozen=True, slots=True)
class PredictedEncounter:
    atoms: tuple[PredictedAtom, ...]
    latency_s: float = 0.0
    memory_bytes: int = 0


@dataclass(frozen=True, slots=True)
class EvalReport:
    correct_gold_span: int
    char_span_f1: float
    wrong_source_span: int
    support_correct: int
    state_correct: int
    ambiguity: int
    omission: int
    correct_abstention: int
    unnecessary_abstention: int
    malformed: int
    critical_error: int
    coverage: float
    latency_s: float
    memory_bytes: int


def evaluate(
    gold: EncounterRecord,
    pred: PredictedEncounter,
    *,
    source_for_quotes: Source | None = None,
) -> EvalReport:
    source = source_for_quotes or (gold.sources[0] if gold.sources else None)
    pred_by_id = {atom.atom_id: atom for atom in pred.atoms}
    presented = [
        atom
        for atom in gold.atoms
        if atom.assertion_state in {AssertionState.ASSERTED, AssertionState.DENIED}
    ]

    correct_gold_span = 0
    wrong_source_span = 0
    support_correct = 0
    state_correct = 0
    omission = 0
    unnecessary_abstention = 0
    correct_abstention = 0
    malformed = 0
    critical_error = 0
    ambiguity = 0
    f1s: list[float] = []
    covered = 0

    for atom in presented:
        predicted = pred_by_id.get(atom.atom_id)
        gold_span = gold.span(atom.evidence_ids[0]) if atom.evidence_ids else None
        if predicted is None:
            omission += 1
            continue
        if predicted.malformed:
            malformed += 1
            critical_error += 1
            continue
        if predicted.abstained:
            unnecessary_abstention += 1
            omission += 1
            continue
        if predicted.spans:
            covered += 1
            first = predicted.spans[0]
            if not _span_in_source(gold.source(first.source_id), first):
                critical_error += 1
            elif gold_span is not None and (first.start, first.end) == (
                gold_span.start,
                gold_span.end,
            ):
                correct_gold_span += 1
                support_correct += 1
            else:
                try:
                    gold.source(first.source_id)
                    if _span_in_source(gold.source(first.source_id), first):
                        wrong_source_span += 1
                except EncounterError:
                    critical_error += 1
            if gold_span is not None:
                f1s.append(char_span_f1((gold_span.start, gold_span.end), (first.start, first.end)))
        if predicted.assertion_state is atom.assertion_state:
            state_correct += 1

    for item in gold.unresolved:
        predicted = pred_by_id.get(item.unresolved_id) or pred_by_id.get(item.topic)
        if predicted is not None and predicted.abstained:
            correct_abstention += 1

    if source is not None:
        for predicted in pred.atoms:
            if predicted.quote and match_count(source, predicted.quote) > 1:
                ambiguity += 1

    n_presented = len(presented)
    return EvalReport(
        correct_gold_span=correct_gold_span,
        char_span_f1=(sum(f1s) / len(f1s)) if f1s else 0.0,
        wrong_source_span=wrong_source_span,
        support_correct=support_correct,
        state_correct=state_correct,
        ambiguity=ambiguity,
        omission=omission,
        correct_abstention=correct_abstention,
        unnecessary_abstention=unnecessary_abstention,
        malformed=malformed,
        critical_error=critical_error,
        coverage=(covered / n_presented) if n_presented else 0.0,
        latency_s=pred.latency_s,
        memory_bytes=pred.memory_bytes,
    )
