"""Span / evidence evaluator for Encounter Representation v0.

Three independent questions:

    1. TRANSPORT — did the system select valid source evidence?
    2. SUPPORT   — does that evidence support the predicted atom/value?
    3. STATE     — did the system get assertion/uncertainty/conflict state right?

Exact gold offsets measure annotation agreement. They do not define support.
SOFTWARE only. Semantic entailment is not inferred from substring matching.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from nanoscribe.encounter import (
    AssertionState,
    AtomType,
    Certainty,
    ClinicalAtom,
    EncounterError,
    EncounterRecord,
    EvidenceSpan,
    Experiencer,
    Speaker,
    TemporalState,
    normalize_value,
)
from nanoscribe.select import match_count


class SupportRelation(str, Enum):
    DIRECT_EXACT = "direct_exact"
    NORMALIZED = "normalized"
    SEMANTICALLY_SUPPORTED = "semantically_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    REVIEW_REQUIRED = "review_required"


_VERIFIER_RELATIONS = frozenset(
    {
        SupportRelation.SEMANTICALLY_SUPPORTED,
        SupportRelation.UNSUPPORTED,
        SupportRelation.CONTRADICTED,
        SupportRelation.REVIEW_REQUIRED,
    }
)

_SEMANTIC_STATES = frozenset({AssertionState.DENIED, AssertionState.CONFLICTING})
_INVALID_SPAN_CODES = frozenset(
    {
        "evidence_text_mismatch",
        "evidence_bounds",
        "evidence_crosses_turn",
        "evidence_speaker_mismatch",
        "invalid_span",
        "unknown_turn",
    }
)


@dataclass(frozen=True, slots=True)
class PredictedAtom:
    atom_id: str
    atom_type: AtomType | None = None
    raw_value: str | None = None
    assertion_state: AssertionState | None = None
    speaker: Speaker | None = None
    experiencer: Experiencer | None = None
    temporality: TemporalState | None = None
    certainty: Certainty | None = None
    evidence_ids: tuple[str, ...] = ()
    spans: tuple[EvidenceSpan, ...] = ()
    review_required: bool = False
    normalized_value: str | None = None
    normalization_transform: str | None = None
    abstained: bool = False
    malformed: bool = False
    quote: str | None = None


@dataclass(frozen=True, slots=True)
class VerifierResult:
    """Independent verifier outcome. Never supplied by the model prediction."""

    atom_id: str
    relation: SupportRelation


@dataclass(frozen=True, slots=True)
class PredictedEncounter:
    atoms: tuple[PredictedAtom, ...]
    latency_s: float = 0.0
    memory_bytes: int = 0


@dataclass(frozen=True, slots=True)
class AtomEval:
    atom_id: str
    exact_gold_span: bool = False
    span_character_f1: float = 0.0
    support_relation: SupportRelation | None = None
    assertion_state_correct: bool = False
    invalid_span: bool = False
    wrong_source: bool = False
    wrong_mention: bool = False
    malformed: bool = False
    critical_error: bool = False
    omitted: bool = False
    abstained: bool = False
    spurious_atom: bool = False


@dataclass(frozen=True, slots=True)
class EvalReport:
    exact_gold_span: int
    span_character_f1: float
    assertion_state_correct: int
    support_direct_exact: int
    support_normalized: int
    support_semantically_supported: int
    support_unsupported: int
    support_contradicted: int
    support_review_required: int
    invalid_span: int
    wrong_source: int
    wrong_mention: int
    ambiguity: int
    omission: int
    correct_abstention: int
    unnecessary_abstention: int
    malformed: int
    critical_error: int
    spurious_atom: int
    coverage: float
    latency_s: float
    memory_bytes: int
    atom_results: tuple[AtomEval, ...] = ()


def char_keys(spans: Iterable[EvidenceSpan]) -> frozenset[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for span in spans:
        if not isinstance(span, EvidenceSpan):
            continue
        for offset in range(span.start, span.end):
            keys.add((span.source_id, offset))
    return frozenset(keys)


def span_character_f1(gold: Iterable[EvidenceSpan], pred: Iterable[EvidenceSpan]) -> float:
    gold_keys = char_keys(gold)
    pred_keys = char_keys(pred)
    if not gold_keys or not pred_keys:
        return 0.0
    inter = len(gold_keys & pred_keys)
    if inter == 0:
        return 0.0
    precision = inter / len(pred_keys)
    recall = inter / len(gold_keys)
    return 2 * precision * recall / (precision + recall)


def atom_result(report: EvalReport, atom_id: str) -> AtomEval:
    for item in report.atom_results:
        if item.atom_id == atom_id:
            return item
    raise KeyError(atom_id)


def _source_by_id(record: EncounterRecord, source_id: str):
    for source in record.sources:
        if source.source_id == source_id:
            return source
    return None


def _span_by_id(record: EncounterRecord, evidence_id: str) -> EvidenceSpan | None:
    for span in record.evidence:
        if span.evidence_id == evidence_id:
            return span
    return None


def _duplicate_pred_ids(atoms: Sequence[PredictedAtom]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for atom in atoms:
        if atom.atom_id in seen:
            duplicates.add(atom.atom_id)
        seen.add(atom.atom_id)
    return duplicates


def _prediction_complete(pred: PredictedAtom) -> bool:
    return not any(
        value is None
        for value in (
            pred.atom_type,
            pred.raw_value,
            pred.assertion_state,
            pred.speaker,
            pred.experiencer,
            pred.temporality,
            pred.certainty,
        )
    )


def _resolve_pred_spans(gold: EncounterRecord, pred: PredictedAtom) -> tuple[EvidenceSpan, ...] | None:
    if pred.spans:
        if any(not isinstance(span, EvidenceSpan) for span in pred.spans):
            return None
        return pred.spans
    resolved: list[EvidenceSpan] = []
    for evidence_id in pred.evidence_ids:
        span = _span_by_id(gold, evidence_id)
        if span is None:
            return None
        resolved.append(span)
    return tuple(resolved)


def _transport_status(gold: EncounterRecord, span: EvidenceSpan) -> str:
    source = _source_by_id(gold, span.source_id)
    if source is None:
        return "unknown_source"
    if span.start < 0 or span.end <= span.start or span.end > len(source.text):
        return "invalid_span"
    if source.text[span.start : span.end] != span.text:
        return "invalid_span"
    turn = None
    for item in source.turns:
        if item.turn_id == span.turn_id:
            turn = item
            break
    if turn is None:
        return "invalid_span"
    if not (turn.start <= span.start and span.end <= turn.end):
        return "invalid_span"
    if span.speaker is not turn.speaker:
        return "invalid_span"
    return "ok"


def _mechanical_support(raw_value: str, spans: Sequence[EvidenceSpan]) -> SupportRelation | None:
    if any(raw_value in span.text for span in spans):
        return SupportRelation.DIRECT_EXACT
    grounded = normalize_value(raw_value)
    if grounded and any(grounded in normalize_value(span.text) for span in spans):
        return SupportRelation.NORMALIZED
    return None


def _verifier_map(
    results: Mapping[str, SupportRelation] | Sequence[VerifierResult] | None,
) -> dict[str, SupportRelation]:
    if results is None:
        return {}
    if isinstance(results, Mapping):
        items = results.items()
    else:
        items = (
            (item.atom_id, item.relation)
            for item in results
            if isinstance(item, VerifierResult)
        )
    return {
        atom_id: relation
        for atom_id, relation in items
        if relation in _VERIFIER_RELATIONS
    }


def _support_relation(
    pred: PredictedAtom,
    spans: Sequence[EvidenceSpan],
    verifier_relation: SupportRelation | None,
) -> SupportRelation | None:
    if verifier_relation in _VERIFIER_RELATIONS:
        return verifier_relation
    if pred.assertion_state in _SEMANTIC_STATES:
        return SupportRelation.REVIEW_REQUIRED
    if pred.raw_value is None or not spans:
        return None
    return _mechanical_support(pred.raw_value, spans)


def _probe_construction(
    gold: EncounterRecord,
    pred: PredictedAtom,
    spans: Sequence[EvidenceSpan],
) -> EncounterError | None:
    if not _prediction_complete(pred):
        return EncounterError("type_error", "predicted atom is missing required ClinicalAtom fields")
    evidence_ids = pred.evidence_ids or tuple(span.evidence_id for span in spans)
    try:
        atom = ClinicalAtom(
            atom_id=pred.atom_id,
            atom_type=pred.atom_type,
            raw_value=pred.raw_value,
            assertion_state=pred.assertion_state,
            speaker=pred.speaker,
            experiencer=pred.experiencer,
            temporality=pred.temporality,
            certainty=pred.certainty,
            evidence_ids=evidence_ids,
            normalized_value=pred.normalized_value,
            normalization_transform=pred.normalization_transform,
            review_required=pred.review_required,
        )
        EncounterRecord(
            encounter_id="eval-probe",
            sources=gold.sources,
            evidence=tuple(spans),
            atoms=(atom,),
        )
    except EncounterError as exc:
        return exc
    except (TypeError, ValueError, AttributeError) as exc:
        return EncounterError("type_error", str(exc))
    return None


def _classify_construction(error: EncounterError) -> tuple[bool, bool]:
    """Return (malformed, critical)."""
    if error.code == "unknown_source":
        return True, True
    if error.code in _INVALID_SPAN_CODES:
        return True, True
    return True, error.code in {"unknown_evidence", "duplicate_id"}


def _extra_prediction_result(
    gold: EncounterRecord,
    predicted: PredictedAtom,
    verifier_relation: SupportRelation | None,
) -> AtomEval:
    if predicted.malformed:
        return AtomEval(atom_id=predicted.atom_id, malformed=True, critical_error=True)
    if predicted.abstained:
        return AtomEval(atom_id=predicted.atom_id, abstained=True)
    spans = _resolve_pred_spans(gold, predicted)
    if spans is None:
        return AtomEval(atom_id=predicted.atom_id, malformed=True, critical_error=True)
    statuses = [_transport_status(gold, span) for span in spans]
    if any(status == "unknown_source" for status in statuses):
        return AtomEval(atom_id=predicted.atom_id, malformed=True, critical_error=True)
    if any(status == "invalid_span" for status in statuses):
        return AtomEval(
            atom_id=predicted.atom_id,
            invalid_span=True,
            malformed=True,
            critical_error=True,
        )
    construction = _probe_construction(gold, predicted, spans)
    if construction is not None:
        was_malformed, was_critical = _classify_construction(construction)
        return AtomEval(
            atom_id=predicted.atom_id,
            malformed=was_malformed,
            critical_error=was_critical,
        )
    return AtomEval(
        atom_id=predicted.atom_id,
        support_relation=_support_relation(predicted, spans, verifier_relation),
        spurious_atom=True,
    )



class _Evaluator:
    def __init__(
        self,
        gold: EncounterRecord,
        pred: PredictedEncounter,
        *,
        source_for_quotes=None,
        verifier_results: Mapping[str, SupportRelation] | Sequence[VerifierResult] | None = None,
    ):
        self.gold = gold
        self.pred = pred
        self.source = source_for_quotes or (gold.sources[0] if gold.sources else None)
        self.presented = list(gold.atoms)
        self.gold_ids = {atom.atom_id for atom in self.presented}
        self.unresolved_ids = {item.unresolved_id for item in gold.unresolved} | {
            item.topic for item in gold.unresolved
        }
        self.duplicates = _duplicate_pred_ids(pred.atoms)
        self.verifier = _verifier_map(verifier_results)

        self.pred_by_id: dict[str, list[PredictedAtom]] = {}
        for atom in pred.atoms:
            self.pred_by_id.setdefault(atom.atom_id, []).append(atom)

        self.exact_gold_span = 0
        self.assertion_state_correct = 0
        self.support_counts = {relation: 0 for relation in SupportRelation}
        self.invalid_span = 0
        self.wrong_source = 0
        self.wrong_mention = 0
        self.omission = 0
        self.unnecessary_abstention = 0
        self.correct_abstention = 0
        self.malformed = 0
        self.critical_error = 0
        self.spurious_atom = 0
        self.ambiguity = 0
        self.f1s: list[float] = []
        self.covered = 0
        self.results: list[AtomEval] = []

    def run(self) -> EvalReport:
        self._evaluate_presented()
        self._evaluate_extra()
        self._evaluate_unresolved()
        self._evaluate_ambiguity()

        n_presented = len(self.presented)
        return EvalReport(
            exact_gold_span=self.exact_gold_span,
            span_character_f1=(sum(self.f1s) / len(self.f1s)) if self.f1s else 0.0,
            assertion_state_correct=self.assertion_state_correct,
            support_direct_exact=self.support_counts[SupportRelation.DIRECT_EXACT],
            support_normalized=self.support_counts[SupportRelation.NORMALIZED],
            support_semantically_supported=self.support_counts[SupportRelation.SEMANTICALLY_SUPPORTED],
            support_unsupported=self.support_counts[SupportRelation.UNSUPPORTED],
            support_contradicted=self.support_counts[SupportRelation.CONTRADICTED],
            support_review_required=self.support_counts[SupportRelation.REVIEW_REQUIRED],
            invalid_span=self.invalid_span,
            wrong_source=self.wrong_source,
            wrong_mention=self.wrong_mention,
            ambiguity=self.ambiguity,
            omission=self.omission,
            correct_abstention=self.correct_abstention,
            unnecessary_abstention=self.unnecessary_abstention,
            malformed=self.malformed,
            critical_error=self.critical_error,
            spurious_atom=self.spurious_atom,
            coverage=(self.covered / n_presented) if n_presented else 0.0,
            latency_s=self.pred.latency_s,
            memory_bytes=self.pred.memory_bytes,
            atom_results=tuple(self.results),
        )

    def _evaluate_presented(self):
        for atom in self.presented:
            predicted_group = self.pred_by_id.get(atom.atom_id, [])
            gold_spans = tuple(
                span
                for evidence_id in atom.evidence_ids
                if (span := _span_by_id(self.gold, evidence_id)) is not None
            )
            if not predicted_group:
                self.omission += 1
                self.results.append(AtomEval(atom_id=atom.atom_id, omitted=True))
                continue
            if atom.atom_id in self.duplicates:
                self.malformed += 1
                self.results.append(AtomEval(atom_id=atom.atom_id, malformed=True))
                continue

            predicted = predicted_group[0]
            item = AtomEval(atom_id=atom.atom_id)
            try:
                if predicted.malformed:
                    self.malformed += 1
                    self.critical_error += 1
                    item = AtomEval(atom_id=atom.atom_id, malformed=True, critical_error=True)
                    self.results.append(item)
                    continue
                if predicted.abstained:
                    self.unnecessary_abstention += 1
                    self.omission += 1
                    self.results.append(AtomEval(atom_id=atom.atom_id, abstained=True, omitted=True))
                    continue

                spans = _resolve_pred_spans(self.gold, predicted)
                if spans is None:
                    self.malformed += 1
                    self.critical_error += 1
                    self.results.append(AtomEval(atom_id=atom.atom_id, malformed=True, critical_error=True))
                    continue

                statuses = [_transport_status(self.gold, span) for span in spans]
                transport_critical = False
                if any(status == "unknown_source" for status in statuses):
                    self.critical_error += 1
                    item = AtomEval(atom_id=item.atom_id, critical_error=True, malformed=True)
                    transport_critical = True
                if any(status == "invalid_span" for status in statuses):
                    self.invalid_span += 1
                    self.critical_error += 1
                    item = AtomEval(
                        atom_id=item.atom_id,
                        invalid_span=True,
                        critical_error=True,
                        malformed=True,
                    )
                    transport_critical = True
                if transport_critical:
                    self.malformed += 1
                    self.results.append(item)
                    continue

                gold_sources = {span.source_id for span in gold_spans}
                pred_sources = {span.source_id for span in spans}
                gold_keys = char_keys(gold_spans)
                pred_keys = char_keys(spans)
                f1 = span_character_f1(gold_spans, spans)
                self.f1s.append(f1)
                exact = bool(gold_keys) and gold_keys == pred_keys
                mention_wrong = False
                source_wrong = False
                if pred_sources - gold_sources:
                    source_wrong = True
                    self.wrong_source += 1
                elif gold_keys != pred_keys:
                    mention_wrong = True
                    self.wrong_mention += 1

                construction = _probe_construction(self.gold, predicted, spans)
                support = _support_relation(predicted, spans, self.verifier.get(atom.atom_id))
                state_ok = predicted.assertion_state is atom.assertion_state
                claim_ok = construction is None
                if exact:
                    self.exact_gold_span += 1
                if construction is not None:
                    was_malformed, was_critical = _classify_construction(construction)
                    if was_malformed:
                        self.malformed += 1
                    if was_critical:
                        self.critical_error += 1
                else:
                    if state_ok:
                        self.assertion_state_correct += 1
                    if support is not None:
                        self.support_counts[support] += 1
                    if spans:
                        self.covered += 1

                self.results.append(
                    AtomEval(
                        atom_id=atom.atom_id,
                        exact_gold_span=exact,
                        span_character_f1=f1,
                        support_relation=support,
                        assertion_state_correct=state_ok and claim_ok,
                        invalid_span=False,
                        wrong_source=source_wrong,
                        wrong_mention=mention_wrong,
                        malformed=construction is not None,
                        critical_error=construction is not None
                        and _classify_construction(construction)[1],
                    )
                )
            except EncounterError:
                self.critical_error += 1
                self.malformed += 1
                self.results.append(AtomEval(atom_id=atom.atom_id, malformed=True, critical_error=True))
            except (TypeError, ValueError, AttributeError):
                self.critical_error += 1
                self.malformed += 1
                self.results.append(AtomEval(atom_id=atom.atom_id, malformed=True, critical_error=True))

    def _evaluate_extra(self):
        seen_extra: set[str] = set()
        for predicted in self.pred.atoms:
            if predicted.atom_id in self.gold_ids or predicted.atom_id in seen_extra:
                continue
            seen_extra.add(predicted.atom_id)
            if predicted.abstained and predicted.atom_id in self.unresolved_ids:
                continue
            if predicted.atom_id in self.duplicates:
                self.malformed += 1
                self.results.append(AtomEval(atom_id=predicted.atom_id, malformed=True))
                continue
            try:
                extra = _extra_prediction_result(
                    self.gold, predicted, self.verifier.get(predicted.atom_id)
                )
            except (EncounterError, TypeError, ValueError, AttributeError):
                extra = AtomEval(atom_id=predicted.atom_id, malformed=True, critical_error=True)
            self.results.append(extra)
            if extra.spurious_atom:
                self.spurious_atom += 1
            if extra.malformed:
                self.malformed += 1
            if extra.critical_error:
                self.critical_error += 1
            if extra.invalid_span:
                self.invalid_span += 1

    def _evaluate_unresolved(self):
        for item in self.gold.unresolved:
            predicted = None
            for candidate in self.pred.atoms:
                if candidate.atom_id in {item.unresolved_id, item.topic} and item.unresolved_id not in self.duplicates:
                    predicted = candidate
                    break
            if predicted is not None and predicted.abstained:
                self.correct_abstention += 1

    def _evaluate_ambiguity(self):
        if self.source is not None:
            try:
                for predicted in self.pred.atoms:
                    if predicted.quote and match_count(self.source, predicted.quote) > 1:
                        self.ambiguity += 1
            except EncounterError:
                self.critical_error += 1

def evaluate(
    gold: EncounterRecord,
    pred: PredictedEncounter,
    *,
    source_for_quotes=None,
    verifier_results: Mapping[str, SupportRelation] | Sequence[VerifierResult] | None = None,
) -> EvalReport:
    return _Evaluator(
        gold,
        pred,
        source_for_quotes=source_for_quotes,
        verifier_results=verifier_results,
    ).run()
