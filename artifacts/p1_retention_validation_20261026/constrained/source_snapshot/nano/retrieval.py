"""Retrieval over raw observations — the control arm for constitution §L.8.

Two of Nano's laws are empirical claims, not slogans:

    Retrieval is not memory.
    Context is not persistent state.

They are only worth asserting if a competent retrieval system, given the same
evidence, measurably fails at something the state layer does. This module is
that system. It is built to be **strong**, because a strawman baseline would
make the laws look proven while proving nothing:

* it indexes the same `EvidenceSpanV2` objects the ledger holds — the real
  artifacts, not a degraded copy;
* it ranks by *event* time rather than arrival time, which is the more
  competent choice and the one a careful RAG implementation would make;
* it makes the same three-way output decision (§XXX) rather than being allowed
  to blurt, so `undeclared_error` compares like with like;
* its budget `k` sweeps up to reading the entire history, which is the
  context-stuffing limit.

What it deliberately does not have is a **fold**: no maintained resolution, no
supersession set, no conflict registry, no gap record, no dependency graph, no
prior state. That absence is the independent variable. Anything it recovers
anyway is a point against the laws; anything it cannot recover at any `k` is the
mechanism behind them.

The honest framing, stated up front so the scorer cannot quietly overclaim: a
retrieval system that *adds* a fold over its results has become the state layer.
The comparison is not retrieval-versus-state as engineering rivals — it is
whether the fold is load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nano.contracts import EpistemicStatus
from nano.kernel import Identity
from nano.runtime import Answer, Disposition, Question, QuestionKind

#: Cost model. A search scans the index once, then pays per span actually read.
#: Reading more evidence is not free, which is the whole reason a budget exists.
SCAN_COST = 1.0
SPAN_READ_COST = 0.25

#: What folding one observation into maintained state costs the other arm. Set
#: equal to a span read so the comparison is not rigged by pricing the fold
#: cheaply — the state arm pays full freight for every observation it ingests.
FOLD_COST_PER_OBSERVATION = SPAN_READ_COST


@dataclass(frozen=True)
class IndexedSpan:
    """One retrievable unit: exactly what the ledger stored, nothing added."""
    span_id: str
    entity: str
    attribute: str
    value: str
    verbatim: str
    documentation_time: str
    event_time: str


@dataclass
class ObservationIndex:
    """A retrieval index over raw evidence. No fold, by construction.

    `searches` and `spans_read` are counters rather than estimates: the cost
    comparison against maintained state has to be measured, not asserted, or it
    becomes an argument about which is *obviously* cheaper.
    """
    spans: list = field(default_factory=list)
    searches: int = 0
    spans_read: int = 0

    @classmethod
    def build(cls, ledger_spans, parse) -> "ObservationIndex":
        indexed = []
        for span in ledger_spans:
            parsed = parse(span.verbatim)
            if parsed is None:
                continue
            entity, attribute, value = parsed
            indexed.append(IndexedSpan(
                span_id=span.evidence_span_id, entity=entity,
                attribute=attribute, value=value, verbatim=span.verbatim,
                documentation_time=span.documentation_time,
                event_time=span.candidate_event_time))
        return cls(spans=indexed)

    @property
    def cost(self) -> float:
        return self.searches * SCAN_COST + self.spans_read * SPAN_READ_COST

    def search(self, entity: str, attribute: str, k: int) -> list:
        """Top-k spans about a key, most recent event first.

        Ranking by `event_time` rather than `documentation_time` is the
        competent choice: a delayed report about an old event should not
        outrank a timely report about a recent one. Note what this cannot do —
        `event_time` here is a raw string, so a month-precision timestamp sorts
        against a day-precision one lexicographically. The state layer fixed
        that with interval comparison inside its fold; an index has no fold to
        put the fix in.
        """
        self.searches += 1
        hits = [s for s in self.spans
                if s.entity == entity and s.attribute == attribute]
        hits.sort(key=lambda s: (s.event_time, s.documentation_time,
                                 s.span_id), reverse=True)
        taken = hits[:k] if k > 0 else hits
        self.spans_read += len(taken)
        return taken

    def scan_all(self, k: int) -> list:
        """Whole-index read, for questions that are inherently global."""
        self.searches += 1
        taken = self.spans if k <= 0 else self.spans[:k]
        self.spans_read += len(taken)
        return taken


#: Question kinds a fold-free index cannot attempt at any budget, and why.
#: Reported as a distinct category rather than scored as wrong answers —
#: counting a structural impossibility as a failed attempt would inflate the
#: result and obscure the actual mechanism.
STRUCTURALLY_UNAVAILABLE = {
    QuestionKind.WHAT_CHANGED:
        "no prior state to diff against; raw spans record what was reported, "
        "not what the system previously believed",
    QuestionKind.WHY_DIFFERENT:
        "an explanation of a belief change requires the earlier belief, which "
        "was never maintained",
    QuestionKind.WHAT_IS_STALE:
        "staleness is a property of a dependency graph over derived objects; "
        "an index has no derived objects and no lineage",
    QuestionKind.WHAT_WOULD_HELP:
        "an information need is computed from resolved/conflicted/gap state; "
        "an index has none of those, and cannot know a report is *missing* "
        "because a missing report leaves nothing to retrieve",
}


def retrieval_answer(question: Question, index: ObservationIndex,
                     subject: Identity, *, k: int = 16) -> Answer:
    """Answer from retrieved evidence alone, with the same output decision.

    The decision rule is the best one available from retrieved text: disagree
    at the top event time -> REVIEW; nothing retrieved -> ABSTAIN; otherwise
    PRESENT. Denying the baseline an output decision would make the
    undeclared-error comparison meaningless.
    """
    if question.kind in STRUCTURALLY_UNAVAILABLE:
        return Answer(
            question=question, disposition=Disposition.ABSTAIN, content="",
            reason=f"unavailable without persistent state: "
                   f"{STRUCTURALLY_UNAVAILABLE[question.kind]}",
            epistemic_status=EpistemicStatus.UNAVAILABLE,
            capability_used=f"retrieval_k{k}", cost_spent=0.0)

    before = index.cost
    if question.kind is QuestionKind.WHAT_CONFLICTS:
        hits = index.scan_all(k)
        groups: dict = {}
        for s in hits:
            groups.setdefault((s.entity, s.attribute, s.event_time),
                              set()).add(s.value)
        disputed = {g: v for g, v in groups.items() if len(v) > 1}
        spent = index.cost - before
        if not disputed:
            # Honest about scope: a budgeted scan that found nothing has not
            # established that nothing exists.
            return Answer(
                question=question, disposition=Disposition.ABSTAIN, content="",
                reason=f"no disagreement in the {len(hits)} spans read, but a "
                       "budgeted scan cannot establish that none exists",
                epistemic_status=EpistemicStatus.NOT_FOUND,
                capability_used=f"retrieval_k{k}", cost_spent=spent)
        listed = "; ".join(f"{e}.{a}: {'/'.join(sorted(v))}"
                           for (e, a, _), v in sorted(disputed.items())[:5])
        return Answer(
            question=question, disposition=Disposition.REVIEW,
            content=f"{len(disputed)} disputed: {listed}",
            reason="retrieved spans disagree at the same event time",
            epistemic_status=EpistemicStatus.CONFLICTING,
            evidence_span_ids=tuple(sorted(s.span_id for s in hits))[:8],
            capability_used=f"retrieval_k{k}", cost_spent=spent)

    hits = index.search(question.entity, question.attribute, k)
    spent = index.cost - before
    if not hits:
        return Answer(
            question=question, disposition=Disposition.ABSTAIN, content="",
            reason="nothing retrieved for this key",
            epistemic_status=EpistemicStatus.NOT_FOUND,
            capability_used=f"retrieval_k{k}", cost_spent=spent)

    if question.kind is QuestionKind.WHAT_SUPPORTS:
        return Answer(
            question=question, disposition=Disposition.PRESENT,
            content=f"{len(hits)} evidence span(s): "
                    f"{', '.join(sorted(s.span_id for s in hits))}",
            reason="citing retrieved spans directly",
            epistemic_status=EpistemicStatus.DIRECT_DOCUMENTATION,
            evidence_span_ids=tuple(sorted(s.span_id for s in hits)),
            capability_used=f"retrieval_k{k}", cost_spent=spent)

    latest = hits[0].event_time
    top = [s for s in hits if s.event_time == latest]
    values = {s.value for s in top}
    spans = tuple(sorted(s.span_id for s in top))
    if len(values) > 1:
        return Answer(
            question=question, disposition=Disposition.REVIEW,
            content=f"{question.entity}.{question.attribute} is disputed: "
                    f"{', '.join(sorted(values))}",
            reason="retrieved reports disagree at the latest event time",
            epistemic_status=EpistemicStatus.CONFLICTING,
            evidence_span_ids=spans, capability_used=f"retrieval_k{k}",
            cost_spent=spent)
    return Answer(
        question=question, disposition=Disposition.PRESENT,
        content=f"{question.entity}.{question.attribute} = "
                f"{next(iter(values))}",
        reason=f"most recent of {len(hits)} retrieved report(s) by event time",
        epistemic_status=EpistemicStatus.DIRECT_MEASUREMENT,
        evidence_span_ids=spans, capability_used=f"retrieval_k{k}",
        cost_spent=spent)
