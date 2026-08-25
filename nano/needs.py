"""Information need — the "needed" axis of MTA-EPISTEMIC, Layer XII.

`PatientStateSnapshot` already carries what is *known*, what is *unknown* and
what is *conflicting*. The fourth axis, **needed**, was a flat list: gaps sorted
alphabetically, which is an inventory, not a plan. An inventory says "here are
forty things I don't know"; a plan says "ask about these three first, and here
is why".

The distinction matters because acquiring information is never free. A system
that emits every gap it has has pushed the whole triage problem onto whoever
reads it, while appearing thorough.

What makes a request legitimate
-------------------------------
Every request must name a **reason** and what it **would resolve**. An
unexplained "go find out more" cannot be acted on, cannot be prioritised against
anything else, and cannot be checked afterwards. So both are required at
construction, in the same spirit as `Dependency` refusing an inputless derived
object.

Domain-neutral by construction
------------------------------
`rank_needs` takes plain dicts and sets, never a ledger or a world. A need for a
missing lab and a need for a missing sensor reading rank by the same rules, and
the ranking can be unit-tested without building either.

Choosing the ranking by measurement, not by preference
------------------------------------------------------
Four strategies are provided, from a deliberate no-signal control upward, so the
benchmark can *measure* which one earns its place rather than assuming the most
elaborate is best. It did, and the answer was humbling: cause alone carries the
result, while scarcity-weighting and age-weighting — both of which sounded
obviously useful — add nothing distinguishable from zero across ten worlds. The
numbers are pinned next to `DEFAULT_STRATEGY` below, and the default is the
simplest strategy that beats the control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum

from fabric.schemas import _cid


class NeedKind(str, Enum):
    """Why information is needed. These are not severities — they are *causes*,
    and collapsing them into one number is the overloading failure the ontology
    guards against. Weighting happens in the strategy, separately."""
    CONTRADICTION = "contradiction"   # live reports disagree at the same time
    UNORDERABLE = "unorderable"       # reports disagree and cannot be ordered
    GAP = "gap"                       # a report is known to be missing
    THIN = "thin"                     # believed on a single uncorroborated report
    STALE = "stale"                   # nothing reported for a long time


class Strategy(str, Enum):
    """Ranking strategies, kept in increasing complexity so the comparison is
    honest. The losers stay in the enum on purpose: deleting a strategy that was
    measured and found not to help turns a recorded negative result into a
    silent one, and the next person re-adds it."""
    ARBITRARY = "arbitrary"                    # control: no signal at all
    KIND = "kind"                              # cause only — the default, see below
    KIND_SCARCITY = "kind_scarcity"            # cause / corroboration
    KIND_SCARCITY_AGE = "kind_scarcity_age"    # cause / corroboration * age


#: Measured on NANO-SLW-001, 10 seeds, paired per seed at a 25% budget
#: (`nano.slw.compare_need_strategies`, re-derivable with the recheck command in
#: the docstring there):
#:
#:     kind              - arbitrary      +0.6314  95% CI [+0.5929, +0.6698]  ***
#:     kind_scarcity     - kind           +0.0137  95% CI [-0.0086, +0.0360]  ns
#:     kind_scarcity_age - kind_scarcity  -0.0070  95% CI [-0.0308, +0.0169]  ns
#:
#: So: knowing *why* information is missing does essentially all the work, and
#: the two refinements I expected to help do not — one of them is directionally
#: negative. `DEFAULT_STRATEGY` is therefore the simplest strategy that beats the
#: control, not the most elaborate one available. Promote a refinement only when
#: its paired interval clears zero.
DEFAULT_STRATEGY = Strategy.KIND


#: Cause weights. Ordered by how little the system knows, not by how alarming
#: the word sounds: a contradiction means it holds *no* usable value, a gap
#: means it holds a possibly-stale one, thinness means it holds an uncorroborated
#: one. Provisional — the benchmark measures whether the ordering pays.
KIND_WEIGHT = {
    NeedKind.CONTRADICTION: 4.0,
    NeedKind.UNORDERABLE: 4.0,
    NeedKind.GAP: 3.0,
    NeedKind.THIN: 2.0,
    NeedKind.STALE: 1.0,
}


@dataclass(frozen=True)
class InformationRequest:
    """One actionable question, with its justification attached.

    `reason` and `would_resolve` are required. A request that cannot say why it
    exists is indistinguishable from noise once it reaches a queue, and a
    request that cannot say what would settle it can never be closed.
    """
    subject: str
    attribute: str
    kind: NeedKind
    reason: str
    would_resolve: tuple
    evidence_count: int = 0
    downstream_count: int = 0
    age_days: int = 0
    priority: float = 0.0
    request_id: str = ""

    def __post_init__(self):
        if not self.reason:
            raise ValueError(
                f"{self.subject}.{self.attribute}: an information request without "
                "a reason cannot be prioritised or audited")
        if not self.would_resolve:
            raise ValueError(
                f"{self.subject}.{self.attribute}: a request that names nothing "
                "it would resolve can never be closed")
        if self.evidence_count < 0 or self.age_days < 0:
            raise ValueError("negative evidence count or age")
        object.__setattr__(self, "request_id", _cid(
            {"s": self.subject, "a": self.attribute, "k": self.kind.value},
            "need"))

    @property
    def key(self) -> tuple:
        return (self.subject, self.attribute)


def _days_between(earlier: str, later: str) -> int:
    """Age in days, tolerant of month-precision timestamps.

    A month-precision time is treated as its first day — the *oldest* it could
    be — because underestimating staleness is the error that leaves a stale
    value unquestioned, and that is the direction that hurts.
    """
    def parse(t: str) -> date | None:
        try:
            if len(t) == 7 and t[4] == "-":
                return date(int(t[:4]), int(t[5:7]), 1)
            return date.fromisoformat(t[:10])
        except (ValueError, IndexError):
            return None

    a, b = parse(earlier), parse(later)
    return max(0, (b - a).days) if a and b else 0


def priority_of(request: InformationRequest, strategy: Strategy,
                *, age_window: int = 30) -> float:
    """Score a request under one strategy.

    Documented per `rules/scientific-calibration.md`:

        Equation:   see per-branch below
        Purpose:    order information requests under a finite acquisition budget
        Inputs:     cause weight, live-evidence count, age in days
        Output:     non-negative float; larger = ask sooner
        Assumptions the benchmark tests, rather than assumes:
                    (a) cause ordering predicts which keys are actually wrong
                    (b) sparse corroboration predicts error
                    (c) age predicts error
        Why not simpler:  ARBITRARY is included precisely so "simpler" is on the
                    board and can win.
        Why not more complex: a learned ranker needs labels this system does not
                    have at request time, and would be unfalsifiable at n=1 world.
        Calibration: precision@K against keys measurably wrong, plus simulated
                    acquisition. See `nano.slw.score_information_need`.
    """
    if strategy is Strategy.ARBITRARY:
        return 0.0
    weight = KIND_WEIGHT[request.kind]
    if strategy is Strategy.KIND:
        return weight
    # 1/sqrt(n) rather than 1/n: corroboration has diminishing returns, and the
    # harsher 1/n makes evidence count dominate the cause entirely.
    scarcity = weight / math.sqrt(1 + request.evidence_count)
    if strategy is Strategy.KIND_SCARCITY:
        return scarcity
    return scarcity * (1.0 + request.age_days / age_window)


def rank_needs(
    *,
    conflicted: dict,
    gaps: set,
    resolved: dict,
    latest_time: dict,
    evidence_count: dict,
    now: str,
    downstream: dict | None = None,
    strategy: Strategy = DEFAULT_STRATEGY,
    stale_after_days: int = 21,
    thin_below: int = 2,
) -> list:
    """Turn an epistemic state into a ranked, justified list of questions.

    Deterministic: ties break on the key, never on dict order, so two runs over
    the same state produce the same plan. A plan that reshuffles under an
    unchanged world cannot be reviewed.
    """
    downstream = downstream or {}
    requests: list[InformationRequest] = []

    def add(key, kind, reason, would_resolve):
        subject, attribute = key
        requests.append(InformationRequest(
            subject=subject, attribute=attribute, kind=kind, reason=reason,
            would_resolve=tuple(would_resolve),
            evidence_count=evidence_count.get(key, 0),
            downstream_count=downstream.get(key, 0),
            age_days=_days_between(latest_time.get(key, now), now)))

    for key in sorted(conflicted):
        when, values = conflicted[key]
        add(key, NeedKind.CONTRADICTION,
            f"{len(values)} live reports disagree ({', '.join(values)}) and none "
            "supersedes the others",
            (f"resolve {key[0]}.{key[1]} to one of {', '.join(values)}",))

    for key in sorted(gaps):
        if key in conflicted:
            continue  # already asked about, for a stronger reason
        held = resolved.get(key)
        add(key, NeedKind.GAP,
            "a change to this attribute went unreported; the held value "
            + (f"({held}) may be stale" if held else "is unknown"),
            (f"confirm the current value of {key[0]}.{key[1]}",))

    for key, value in sorted(resolved.items()):
        if key in conflicted or key in gaps:
            continue
        age = _days_between(latest_time.get(key, now), now)
        count = evidence_count.get(key, 0)
        if age >= stale_after_days:
            add(key, NeedKind.STALE,
                f"nothing reported for {age} days; the world may have moved",
                (f"refresh {key[0]}.{key[1]} (held: {value})",))
        elif count < thin_below:
            add(key, NeedKind.THIN,
                f"believed on {count} uncorroborated report",
                (f"corroborate {key[0]}.{key[1]} = {value}",))

    scored = [
        InformationRequest(
            subject=r.subject, attribute=r.attribute, kind=r.kind,
            reason=r.reason, would_resolve=r.would_resolve,
            evidence_count=r.evidence_count,
            downstream_count=r.downstream_count, age_days=r.age_days,
            priority=priority_of(r, strategy))
        for r in requests]
    # Descending priority; ties broken deterministically on the key.
    return sorted(scored, key=lambda r: (-r.priority, r.subject, r.attribute))


def explain_plan(requests: list, budget: int) -> dict:
    """What the plan says, and — as important — what it defers.

    A plan that reports only what it selected reads as though nothing was left
    out. Naming the deferred count and the cause mix is what makes a truncated
    plan honest rather than merely short.
    """
    chosen, deferred = requests[:budget], requests[budget:]
    def mix(rs):
        out: dict = {}
        for r in rs:
            out[r.kind.value] = out.get(r.kind.value, 0) + 1
        return dict(sorted(out.items()))
    return {
        "total_requests": len(requests),
        "budget": budget,
        "selected": len(chosen),
        "deferred": len(deferred),
        "selected_mix": mix(chosen),
        "deferred_mix": mix(deferred),
        "top": [{"key": f"{r.subject}.{r.attribute}", "kind": r.kind.value,
                 "why": r.reason, "priority": round(r.priority, 4)}
                for r in chosen[:5]],
    }
