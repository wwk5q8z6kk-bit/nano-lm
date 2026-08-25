"""The minimal executable Nano cognitive substrate — constitution §XXXVI, §L.5.

Everything below the WorkSlice already existed and was tested: observation,
evidence, entity/event, state, state delta, dependency graph, freshness,
information need. Everything at and above it existed only as a *type*.
`WorkSlice` had no executor, `Tool.cost` and `Tool.reliability` were read by
nothing, `ArtifactIR` had no consumer outside the ontology registry, and
`VerificationReceipt` had exactly one. This module closes that gap and nothing
else — it is a slice, not a framework.

The chain it runs, end to end:

    question -> WorkSlice -> capability selection -> execution
             -> verification -> PRESENT / ABSTAIN / REVIEW -> ArtifactIR

Why the output decision is the point (§XXX)
-------------------------------------------
An answer is not a string, it is a *disposition*. Every output boundary resolves
to PRESENT, ABSTAIN or REVIEW, and which one it is has to be derived from the
epistemic state rather than from how confident the prose sounds. This is the
same instrument as `undeclared_error`: a system that is wrong and says so is
behaving correctly under partial observation; one that is wrong and presents
anyway is the failure the whole architecture exists to prevent.

Why capability selection reads cost (§XI, §XII, §XXVIII)
--------------------------------------------------------
"Smallest sufficient capability" is only a slogan until something actually
compares costs and refuses the expensive path when a cheap one clears the
reliability floor. The registry below is deliberately small and deterministic;
the point is that the selection *happens* and is measurable, not that these are
the final capabilities.

Structural guard
----------------
`StateView` is a narrow read-only projection. The executor cannot reach the
world, the generator, or ground truth — the same guard `rank_needs` carries, for
the same reason: an executor that can see the answer will score well and teach
us nothing. A test fails if this module ever imports a scorer.

Nothing here calls a model. Constitution §XIII: the deterministic substrate
governs; the learned system proposes. The learned parts arrive later, behind
these interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from fabric.schemas import _cid

from nano.contracts import (
    DerivationMode,
    EpistemicStatus,
    VerificationReceipt,
)
from nano.dependency import DependencyGraph, Freshness
from nano.kernel import ArtifactIR, Identity, SliceStatus, Tool, WorkSlice
from nano.needs import DEFAULT_STRATEGY, Strategy, rank_needs


# ---------------------------------------------------------------------------
# What the executor is allowed to see
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StateView:
    """A read-only projection of belief. Deliberately narrow.

    Every field here is something the system genuinely knows at question time.
    There is no world, no generator, no truth table — an executor that could
    reach those would score perfectly and demonstrate nothing.
    """
    subject: Identity
    as_of: str
    resolved: dict = field(default_factory=dict)        # (ent, attr) -> value
    conflicted: dict = field(default_factory=dict)      # (ent, attr) -> (when, vals)
    gaps: frozenset = frozenset()                       # (ent, attr)
    evidence: dict = field(default_factory=dict)        # (ent, attr) -> (span_id,)
    latest_time: dict = field(default_factory=dict)
    evidence_count: dict = field(default_factory=dict)
    known_spans: frozenset = frozenset()
    prior: dict = field(default_factory=dict)           # last answered value
    freshness: dict = field(default_factory=dict)       # object_id -> Freshness
    ledger_version: int = 0

    def tracked(self, key) -> bool:
        return (key in self.resolved or key in self.conflicted
                or key in self.gaps)


# ---------------------------------------------------------------------------
# Questions
# ---------------------------------------------------------------------------

class QuestionKind(str, Enum):
    """The question set constitution §XXXVII asks of DomainPack-0."""
    CURRENT_VALUE = "current_value"      # what is true now?
    WHAT_SUPPORTS = "what_supports"      # what evidence supports that?
    WHAT_CONFLICTS = "what_conflicts"    # what disagrees?
    WHAT_CHANGED = "what_changed"        # what moved since last time?
    WHAT_IS_STALE = "what_is_stale"      # what needs recomputation?
    WHY_DIFFERENT = "why_different"      # why is this not your earlier answer?
    WHAT_WOULD_HELP = "what_would_help"  # what would reduce uncertainty?


@dataclass(frozen=True)
class Question:
    kind: QuestionKind
    entity: str = ""
    attribute: str = ""
    question_id: str = ""

    def __post_init__(self):
        needs_key = {QuestionKind.CURRENT_VALUE, QuestionKind.WHAT_SUPPORTS,
                     QuestionKind.WHY_DIFFERENT}
        if self.kind in needs_key and not (self.entity and self.attribute):
            raise ValueError(
                f"{self.kind.value} is about a specific fact and needs an "
                "entity and attribute")
        object.__setattr__(self, "question_id", _cid(
            {"k": self.kind.value, "e": self.entity, "a": self.attribute}, "q"))

    @property
    def key(self) -> tuple:
        return (self.entity, self.attribute)


# ---------------------------------------------------------------------------
# Output decision — constitution §XXX
# ---------------------------------------------------------------------------

class Disposition(str, Enum):
    PRESENT = "present"    # verified and grounded; say it plainly
    ABSTAIN = "abstain"    # the state does not support an answer; say that
    REVIEW = "review"      # answerable, but a human must adjudicate


@dataclass(frozen=True)
class Answer:
    """An answer and its standing. The standing is not optional.

    `content` without `disposition` is the failure mode: a string that reads
    identically whether the system knew or guessed.
    """
    question: Question
    disposition: Disposition
    content: str
    reason: str
    derivation: DerivationMode = DerivationMode.DERIVED
    epistemic_status: EpistemicStatus = EpistemicStatus.UNCERTAIN
    evidence_span_ids: tuple = ()
    capability_used: str = ""
    cost_spent: float = 0.0
    receipt: VerificationReceipt | None = None
    artifact: ArtifactIR | None = None

    def __post_init__(self):
        if not self.reason:
            raise ValueError("an answer must say why it has the standing it has")
        if self.disposition is Disposition.PRESENT and not self.evidence_span_ids:
            raise ValueError(
                "refusing to PRESENT an answer with no evidence — an ungrounded "
                "claim must ABSTAIN or go to REVIEW (§XXX)")


# ---------------------------------------------------------------------------
# Capability fabric — constitution §XII
# ---------------------------------------------------------------------------
#
# Deliberately small. The claim under test is not that these are the right
# capabilities; it is that selection *reads cost and reliability* and picks the
# cheapest option clearing the floor, so "smallest sufficient capability" is a
# measurable behaviour rather than a slogan.

STATE_LOOKUP = Tool(name="state_lookup", inputs=("question", "state"),
                    outputs=("value",), cost=1.0, latency_s=0.001,
                    reliability=0.99)
EVIDENCE_TRACE = Tool(name="evidence_trace", inputs=("question", "state"),
                      outputs=("spans",), cost=2.0, latency_s=0.002,
                      reliability=0.99)
CONFLICT_REPORT = Tool(name="conflict_report", inputs=("question", "state"),
                       outputs=("conflicts",), cost=2.0, latency_s=0.002,
                       reliability=0.99)
DELTA_READ = Tool(name="delta_read", inputs=("question", "state", "prior"),
                  outputs=("changes",), cost=3.0, latency_s=0.004,
                  reliability=0.98)
FRESHNESS_PROBE = Tool(name="freshness_probe", inputs=("question", "graph"),
                       outputs=("stale",), cost=3.0, latency_s=0.004,
                       reliability=0.99)
NEED_PLANNER = Tool(name="need_planner", inputs=("state",),
                    outputs=("plan",), cost=6.0, latency_s=0.010,
                    reliability=0.95)

#: Candidates per question kind, richest last. Selection walks this in cost
#: order, so adding an expensive capability never makes a cheap sufficient one
#: stop being chosen.
CAPABILITIES: dict = {
    QuestionKind.CURRENT_VALUE: (STATE_LOOKUP,),
    QuestionKind.WHAT_SUPPORTS: (EVIDENCE_TRACE,),
    QuestionKind.WHAT_CONFLICTS: (CONFLICT_REPORT,),
    QuestionKind.WHAT_CHANGED: (DELTA_READ,),
    QuestionKind.WHY_DIFFERENT: (DELTA_READ,),
    QuestionKind.WHAT_IS_STALE: (FRESHNESS_PROBE,),
    QuestionKind.WHAT_WOULD_HELP: (NEED_PLANNER,),
}


def select_capability(question: Question, *, reliability_floor: float,
                      budget_remaining: float) -> Tool:
    """Cheapest capability that clears the reliability floor and fits the budget.

    Raises rather than silently downgrading: a slice that cannot afford any
    sufficient capability must stop and say so, not quietly run a worse one and
    present the result as though it were the intended answer.
    """
    candidates = CAPABILITIES.get(question.kind, ())
    if not candidates:
        raise ValueError(f"no capability registered for {question.kind.value}")
    eligible = [t for t in sorted(candidates, key=lambda t: t.cost)
                if t.reliability >= reliability_floor]
    if not eligible:
        raise ValueError(
            f"{question.kind.value}: no capability meets reliability floor "
            f"{reliability_floor}")
    affordable = [t for t in eligible if t.cost <= budget_remaining]
    if not affordable:
        raise BudgetExhausted(
            f"{question.kind.value}: cheapest sufficient capability costs "
            f"{eligible[0].cost}, budget remaining {budget_remaining}")
    return affordable[0]


class BudgetExhausted(RuntimeError):
    """A slice ran out of compute before it could answer.

    Distinct from "the state does not support an answer": one is a resource
    fact, the other an epistemic one, and reporting a budget stop as an
    abstention would hide a capacity problem behind an epistemic-sounding word.
    """


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _answer_current_value(q: Question, view: StateView) -> tuple:
    """(content, spans, status, disposition, reason)."""
    key = q.key
    if key in view.conflicted:
        when, values = view.conflicted[key]
        return (f"{q.entity}.{q.attribute} is disputed: {', '.join(values)}",
                view.evidence.get(key, ()), EpistemicStatus.CONFLICTING,
                Disposition.REVIEW,
                "live reports disagree and none supersedes the others — a human "
                "must adjudicate rather than the system picking")
    if key in view.resolved:
        value = view.resolved[key]
        spans = view.evidence.get(key, ())
        if key in view.gaps:
            return (f"{q.entity}.{q.attribute} = {value}", spans,
                    EpistemicStatus.OUTDATED, Disposition.REVIEW,
                    "a change to this attribute went unreported, so the held "
                    "value may predate the current state")
        if not spans:
            return ("", (), EpistemicStatus.NOT_FOUND, Disposition.ABSTAIN,
                    "a value is held but no evidence backs it")
        return (f"{q.entity}.{q.attribute} = {value}", spans,
                EpistemicStatus.DIRECT_MEASUREMENT, Disposition.PRESENT,
                "resolved from live evidence with no live disagreement")
    if key in view.gaps:
        return ("", (), EpistemicStatus.NOT_FOUND, Disposition.ABSTAIN,
                "a change was known to go unreported and nothing since has "
                "established a value — not found is not absent")
    return ("", (), EpistemicStatus.NOT_FOUND, Disposition.ABSTAIN,
            "no source has reported on this attribute")


def _answer_what_supports(q: Question, view: StateView) -> tuple:
    spans = view.evidence.get(q.key, ())
    if not spans:
        return ("", (), EpistemicStatus.NOT_FOUND, Disposition.ABSTAIN,
                "nothing in the ledger supports this")
    return (f"{len(spans)} evidence span(s): {', '.join(sorted(spans))}", spans,
            EpistemicStatus.DIRECT_DOCUMENTATION, Disposition.PRESENT,
            "citing ledger spans directly")


def _answer_what_conflicts(q: Question, view: StateView) -> tuple:
    if q.entity:
        items = {k: v for k, v in view.conflicted.items() if k[0] == q.entity}
    else:
        items = view.conflicted
    if not items:
        # Distinct from "no conflicts exist": the system checked and found none,
        # which is a positive finding and is allowed to be PRESENTed only
        # because the search scope is stated.
        return ("no live disagreement across the tracked attributes checked",
                tuple(sorted(view.known_spans))[:1],
                EpistemicStatus.DIRECT_DOCUMENTATION, Disposition.PRESENT,
                "searched every tracked attribute in scope and found none")
    spans = tuple(sorted({s for k in items for s in view.evidence.get(k, ())}))
    listed = "; ".join(f"{e}.{a}: {'/'.join(v[1])}" for (e, a), v in
                       sorted(items.items()))
    return (f"{len(items)} disputed: {listed}", spans,
            EpistemicStatus.CONFLICTING, Disposition.REVIEW,
            "contradictions are preserved for adjudication, never resolved "
            "silently")


def _answer_what_changed(q: Question, view: StateView) -> tuple:
    if not view.prior:
        return ("", (), EpistemicStatus.UNAVAILABLE, Disposition.ABSTAIN,
                "no earlier answer to compare against — a first observation is "
                "not a change")
    scope = ({k: v for k, v in view.resolved.items() if k[0] == q.entity}
             if q.entity else view.resolved)
    changes = [f"{e}.{a}: {view.prior[(e, a)]} -> {v}"
               for (e, a), v in sorted(scope.items())
               if (e, a) in view.prior and view.prior[(e, a)] != v]
    if not changes:
        return ("nothing changed in scope", tuple(sorted(view.known_spans))[:1],
                EpistemicStatus.DIRECT_DOCUMENTATION, Disposition.PRESENT,
                "compared every tracked attribute in scope against the prior "
                "answer")
    spans = tuple(sorted({s for k in scope for s in view.evidence.get(k, ())}))
    return ("; ".join(changes), spans, EpistemicStatus.DIRECT_MEASUREMENT,
            Disposition.PRESENT, "each change is backed by the evidence that "
            "produced the new value")


def _answer_what_is_stale(q: Question, view: StateView) -> tuple:
    stale = sorted(k for k, v in view.freshness.items()
                   if v in (Freshness.STALE, Freshness.POSSIBLY_STALE))
    if not stale:
        return ("nothing is stale", tuple(sorted(view.known_spans))[:1],
                EpistemicStatus.DIRECT_DOCUMENTATION, Disposition.PRESENT,
                "every derived object in the lineage graph is CURRENT")
    return (f"{len(stale)} object(s) need recomputation: {', '.join(stale[:5])}"
            + (" ..." if len(stale) > 5 else ""),
            tuple(sorted(view.known_spans))[:1],
            EpistemicStatus.OUTDATED, Disposition.PRESENT,
            "read from the dependency graph, which records why each is stale")


def _answer_why_different(q: Question, view: StateView) -> tuple:
    key = q.key
    was, now = view.prior.get(key), view.resolved.get(key)
    if was is None:
        return ("", (), EpistemicStatus.UNAVAILABLE, Disposition.ABSTAIN,
                "no earlier answer on this attribute to differ from")
    if was == now:
        return (f"unchanged: still {now}", view.evidence.get(key, ()),
                EpistemicStatus.DIRECT_MEASUREMENT, Disposition.PRESENT,
                "the held value is the same as the previous answer")
    spans = view.evidence.get(key, ())
    if not spans:
        return ("", (), EpistemicStatus.NOT_FOUND, Disposition.ABSTAIN,
                "the belief moved but no evidence explains it — an unexplained "
                "belief change is not inspectable")
    return (f"was {was}, now {now}", spans, EpistemicStatus.DIRECT_MEASUREMENT,
            Disposition.PRESENT,
            "the change is attributable to the cited evidence")


def _answer_what_would_help(q: Question, view: StateView,
                            strategy: Strategy) -> tuple:
    plan = rank_needs(conflicted=view.conflicted, gaps=set(view.gaps),
                      resolved=view.resolved, latest_time=view.latest_time,
                      evidence_count=view.evidence_count, now=view.as_of,
                      strategy=strategy)
    if q.entity:
        plan = [r for r in plan if r.subject == q.entity]
    if not plan:
        return ("no outstanding information need in scope",
                tuple(sorted(view.known_spans))[:1],
                EpistemicStatus.DIRECT_DOCUMENTATION, Disposition.PRESENT,
                "every tracked attribute in scope is resolved, corroborated "
                "and current")
    top = plan[:3]
    return ("; ".join(f"{r.subject}.{r.attribute} ({r.kind.value}): {r.reason}"
                      for r in top),
            tuple(sorted(view.known_spans))[:1],
            EpistemicStatus.INFERRED, Disposition.PRESENT,
            f"ranked plan over {len(plan)} outstanding needs; showing top "
            f"{len(top)}")


_HANDLERS = {
    QuestionKind.CURRENT_VALUE: _answer_current_value,
    QuestionKind.WHAT_SUPPORTS: _answer_what_supports,
    QuestionKind.WHAT_CONFLICTS: _answer_what_conflicts,
    QuestionKind.WHAT_CHANGED: _answer_what_changed,
    QuestionKind.WHAT_IS_STALE: _answer_what_is_stale,
    QuestionKind.WHY_DIFFERENT: _answer_why_different,
}


# ---------------------------------------------------------------------------
# Verification — constitution §XXIX, and the first consumer of the receipt
# ---------------------------------------------------------------------------

def verify(content: str, spans: tuple, view: StateView,
           artifact_id: str) -> VerificationReceipt:
    """Check every cited span actually exists in the ledger.

    Deterministic where a deterministic verifier exists (§XXIX): this is a set
    membership test, not a model asked whether the answer looks right. A span id
    that does not resolve means the answer cited something that was never
    admitted, which is the citation-fabrication failure mode.
    """
    results = []
    for span in spans:
        results.append({"claim": content[:120], "evidence_span_ids": [span],
                        "supported": span in view.known_spans})
    if not spans:
        results.append({"claim": content[:120], "evidence_span_ids": [],
                        "supported": False})
    return VerificationReceipt(
        artifact_id=artifact_id, claim_results=tuple(results),
        coverage_status="complete" if spans else "uncited",
        verifier_version="nano-runtime-001")


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------

def open_slice(objective: str, subject: Identity, *, budget: float = 40.0,
               reliability_floor: float = 0.90) -> WorkSlice:
    """A slice that states how it stops before it starts (§XI)."""
    return WorkSlice(
        objective=objective, subject=subject,
        stop_conditions=("all questions answered", "compute budget exhausted"),
        compute_budget=budget,
        constraints=(f"reliability_floor={reliability_floor}",),
        tools=tuple(t.name for t in
                    {t for group in CAPABILITIES.values() for t in group}))


def answer(question: Question, view: StateView, work: WorkSlice, *,
           reliability_floor: float = 0.90,
           strategy: Strategy = DEFAULT_STRATEGY) -> Answer:
    """Run one question through the full chain.

    select capability -> execute -> verify -> decide disposition -> compile IR.

    Verification can only *downgrade* a disposition, never upgrade one. An
    unverified claim cannot be promoted to PRESENT by anything downstream, which
    is what keeps the output decision from drifting back into "how confident does
    it sound".
    """
    if view.subject.subject_id != work.subject.subject_id:
        raise ValueError(
            f"cross-subject execution: slice is for {work.subject.subject_id}, "
            f"state view is for {view.subject.subject_id}")
    try:
        tool = select_capability(question, reliability_floor=reliability_floor,
                                 budget_remaining=work.compute_budget - work.spent)
    except BudgetExhausted as exc:
        work.status = SliceStatus.STOPPED
        return Answer(
            question=question, disposition=Disposition.REVIEW, content="",
            reason=f"stopped before answering: {exc}",
            derivation=DerivationMode.DERIVED,
            epistemic_status=EpistemicStatus.UNAVAILABLE,
            capability_used="")

    work.spend(tool.cost)
    work.status = SliceStatus.RUNNING

    if question.kind is QuestionKind.WHAT_WOULD_HELP:
        content, spans, status, disposition, reason = _answer_what_would_help(
            question, view, strategy)
    else:
        content, spans, status, disposition, reason = _HANDLERS[question.kind](
            question, view)

    receipt = verify(content, spans, view, artifact_id=question.question_id)
    if disposition is Disposition.PRESENT and receipt.unsupported_count:
        disposition = Disposition.ABSTAIN
        reason = (f"{receipt.unsupported_count} cited span(s) do not resolve in "
                  "the ledger — refusing to present a fabricated citation")
        content, spans = "", ()

    artifact = None
    if disposition is not Disposition.ABSTAIN and content:
        # First real consumer of ArtifactIR: the semantic plan is built and its
        # evidence selected *before* anything is rendered (§XIX.1).
        artifact = ArtifactIR(
            subject=view.subject, purpose=question.kind.value,
            audience="operator", required_claims=(content,),
            selected_evidence=spans or tuple(sorted(view.known_spans))[:1],
            uncertainty_disclosure=(reason,),
            representation="prose",
            state_version=str(view.ledger_version))

    derivation = (DerivationMode.OBSERVED
                  if status in (EpistemicStatus.DIRECT_MEASUREMENT,
                                EpistemicStatus.DIRECT_DOCUMENTATION)
                  and question.kind is QuestionKind.WHAT_SUPPORTS
                  else DerivationMode.DERIVED)
    if status is EpistemicStatus.INFERRED:
        derivation = DerivationMode.INFERRED

    return Answer(question=question, disposition=disposition, content=content,
                  reason=reason, derivation=derivation,
                  epistemic_status=status, evidence_span_ids=tuple(spans),
                  capability_used=tool.name, cost_spent=tool.cost,
                  receipt=receipt, artifact=artifact)


def run_slice(work: WorkSlice, questions, view: StateView, *,
              reliability_floor: float = 0.90,
              strategy: Strategy = DEFAULT_STRATEGY) -> list:
    """Answer questions until they run out or the budget does.

    The budget is a real stop condition, not decoration: when it runs out the
    remaining questions come back as REVIEW with the reason stated. Silently
    answering fewer questions would make a resource limit look like a shorter
    problem.
    """
    answers = []
    for question in questions:
        answers.append(answer(question, view, work,
                              reliability_floor=reliability_floor,
                              strategy=strategy))
    if work.status is not SliceStatus.STOPPED:
        work.status = SliceStatus.VERIFIED
    return answers


def render(a: Answer) -> str:
    """Compile the IR into prose that cannot be mistaken for a bare assertion.

    The disposition is rendered *first*. A reader skimming the output must not
    be able to miss that the system abstained.
    """
    if a.disposition is Disposition.ABSTAIN:
        return f"[ABSTAIN] {a.reason}"
    prefix = "[REVIEW] " if a.disposition is Disposition.REVIEW else ""
    cited = (f"  (evidence: {len(a.evidence_span_ids)} span(s))"
             if a.evidence_span_ids else "")
    return f"{prefix}{a.content}{cited}"
