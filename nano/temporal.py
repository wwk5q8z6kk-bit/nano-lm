"""Interval relations — RSN-TEMPORAL's during / overlap / recurrence half.

`strictly_after` in `nano.slw` settled before/after over mixed precision. That is
the easy half. The rest of the capability — *during*, *overlap*, *meets*,
*starts*, *finishes*, and recurrence — needs intervals, and intervals under
uncertain endpoints behave differently from points in a way that matters:

    When endpoints are imprecise, the honest answer is a SET of relations
    consistent with what is known, not one relation.

A month-precision start against a day-precision start inside that month does not
determine which came first. Naming one anyway is the same defect
`strictly_after` was built to stop, one dimension up.

Two metrics that pull against each other
----------------------------------------
* **Soundness** — is the true relation inside the reported set? A system that is
  unsound has stated something false about time.
* **Sharpness** — how small is the set? A system that always returns all
  thirteen relations is perfectly sound and perfectly useless.

Reporting either alone is meaningless, so both are always reported together. The
control (`document_order_relate`) is RSN-TEMPORAL's own named failure mode —
ordering by document order rather than event time — and it always answers with
exactly one relation, which makes it maximally sharp and measurably unsound.

Satisfiability is a documented over-approximation
-------------------------------------------------
Each Allen relation is a conjunction of order constraints over four endpoints.
`relate` checks each constraint against the endpoint ranges *independently*
rather than solving for a joint assignment. That is a relaxation: it can admit a
relation no single assignment realises, so the answer may be less sharp than
necessary. It can never *exclude* a realisable one, so it is never unsound.

Erring toward a larger set is the safe direction — an over-wide answer is
uninformative, an over-narrow one is wrong — and because sharpness is measured,
the cost of the relaxation shows up in the number rather than hiding.

Nothing here is clinical, and nothing here calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class AllenRelation(str, Enum):
    """The thirteen basic relations between two intervals."""
    BEFORE = "before"
    MEETS = "meets"
    OVERLAPS = "overlaps"
    STARTS = "starts"
    DURING = "during"
    FINISHES = "finishes"
    EQUALS = "equals"
    FINISHED_BY = "finished_by"
    CONTAINS = "contains"
    STARTED_BY = "started_by"
    OVERLAPPED_BY = "overlapped_by"
    MET_BY = "met_by"
    AFTER = "after"


#: Relations that assert temporal containment or coincidence rather than mere
#: sequence. Kept as a named set because "did A happen during B" is the question
#: a delayed effect turns on, and it must not be answered by a before/after test.
CONCURRENT = frozenset({
    AllenRelation.OVERLAPS, AllenRelation.STARTS, AllenRelation.DURING,
    AllenRelation.FINISHES, AllenRelation.EQUALS, AllenRelation.FINISHED_BY,
    AllenRelation.CONTAINS, AllenRelation.STARTED_BY,
    AllenRelation.OVERLAPPED_BY,
})

INVERSE = {
    AllenRelation.BEFORE: AllenRelation.AFTER,
    AllenRelation.MEETS: AllenRelation.MET_BY,
    AllenRelation.OVERLAPS: AllenRelation.OVERLAPPED_BY,
    AllenRelation.STARTS: AllenRelation.STARTED_BY,
    AllenRelation.DURING: AllenRelation.CONTAINS,
    AllenRelation.FINISHES: AllenRelation.FINISHED_BY,
    AllenRelation.EQUALS: AllenRelation.EQUALS,
}
INVERSE.update({v: k for k, v in INVERSE.items()})


def _bounds(stamp: str) -> tuple:
    """A timestamp's earliest and latest possible day.

    Month precision is an interval, not a point — the same reading
    `nano.slw.time_range` uses, restated here so this module stands alone and
    cannot drift from a copy.
    """
    if len(stamp) == 7 and stamp[4] == "-":
        y, m = int(stamp[:4]), int(stamp[5:7])
        nxt = date(y + (m == 12), (m % 12) + 1, 1)
        return (date(y, m, 1), nxt - timedelta(days=1))
    d = date.fromisoformat(stamp[:10])
    return (d, d)


@dataclass(frozen=True)
class Interval:
    """A span with possibly-imprecise endpoints.

    `end=""` means the span is still open — the value has not been observed to
    change. Open is distinct from unknown: the system knows the tenure has not
    ended, which is different from not knowing when it did.
    """
    label: str
    start: str
    end: str = ""
    open_ended: bool = False
    horizon: str = ""          # latest date the system knows about

    def __post_init__(self):
        if not self.start:
            raise ValueError(f"{self.label}: an interval needs a start")
        if not self.end and not self.open_ended:
            raise ValueError(
                f"{self.label}: an interval with no end must say it is open — "
                "a missing end and an unfinished span are different facts")
        object.__setattr__(self, "_s", _bounds(self.start))
        end_stamp = self.end or self.horizon or self.start
        object.__setattr__(self, "_e", _bounds(end_stamp))

    @property
    def start_bounds(self) -> tuple:
        return getattr(self, "_s")

    @property
    def end_bounds(self) -> tuple:
        return getattr(self, "_e")

    @property
    def precise(self) -> bool:
        return (self.start_bounds[0] == self.start_bounds[1]
                and self.end_bounds[0] == self.end_bounds[1])


def _lt(a: tuple, b: tuple) -> bool:
    """Could a < b hold for some assignment inside the ranges?"""
    return a[0] < b[1]


def _gt(a: tuple, b: tuple) -> bool:
    return a[1] > b[0]


def _eq(a: tuple, b: tuple) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


#: Each relation as constraints over (a.start, a.end, b.start, b.end).
#: Written out rather than derived so the definition is auditable against a
#: reference; a clever derivation that is subtly wrong would be invisible.
_CONSTRAINTS = {
    AllenRelation.BEFORE:        [("ae", _lt, "bs")],
    AllenRelation.MEETS:         [("ae", _eq, "bs")],
    AllenRelation.OVERLAPS:      [("as", _lt, "bs"), ("ae", _gt, "bs"),
                                  ("ae", _lt, "be")],
    AllenRelation.STARTS:        [("as", _eq, "bs"), ("ae", _lt, "be")],
    AllenRelation.DURING:        [("as", _gt, "bs"), ("ae", _lt, "be")],
    AllenRelation.FINISHES:      [("as", _gt, "bs"), ("ae", _eq, "be")],
    AllenRelation.EQUALS:        [("as", _eq, "bs"), ("ae", _eq, "be")],
    AllenRelation.FINISHED_BY:   [("as", _lt, "bs"), ("ae", _eq, "be")],
    AllenRelation.CONTAINS:      [("as", _lt, "bs"), ("ae", _gt, "be")],
    AllenRelation.STARTED_BY:    [("as", _eq, "bs"), ("ae", _gt, "be")],
    AllenRelation.OVERLAPPED_BY: [("as", _gt, "bs"), ("as", _lt, "be"),
                                  ("ae", _gt, "be")],
    AllenRelation.MET_BY:        [("as", _eq, "be")],
    AllenRelation.AFTER:         [("as", _gt, "be")],
}


def relate(a: Interval, b: Interval) -> frozenset:
    """Every Allen relation consistent with what is known about the endpoints.

    A singleton means the relation is determined. A larger set means the
    evidence genuinely does not decide, and saying so is the correct answer —
    not a hedge.
    """
    slots = {"as": a.start_bounds, "ae": a.end_bounds,
             "bs": b.start_bounds, "be": b.end_bounds}
    out = set()
    for relation, constraints in _CONSTRAINTS.items():
        if all(test(slots[left], slots[right])
               for left, test, right in constraints):
            out.add(relation)
    if not out:
        # Cannot happen for well-formed intervals; if it ever does, an empty
        # set would read as "no relation holds", which is false of any two
        # intervals. Fail loudly rather than return a lie.
        raise ValueError(f"no relation consistent for {a.label} vs {b.label}")
    return frozenset(out)


def document_order_relate(a: Interval, b: Interval, *,
                          a_documented: str, b_documented: str) -> frozenset:
    """Control arm: order by when it was written down, not when it happened.

    This is RSN-TEMPORAL's named failure mode, implemented faithfully rather
    than as a caricature: it produces exactly one relation, which is what makes
    it look decisive and what makes it wrong.
    """
    if a_documented < b_documented:
        return frozenset({AllenRelation.BEFORE})
    if a_documented > b_documented:
        return frozenset({AllenRelation.AFTER})
    return frozenset({AllenRelation.EQUALS})


def sharpness(reported: frozenset) -> float:
    """1.0 when the relation is pinned, 0.0 when nothing is ruled out.

        sharpness = (13 - |reported|) / 12

    Reported next to soundness, never alone: a set of all thirteen scores 0.0
    here and 1.0 on soundness, which is exactly the useless-but-safe corner the
    pair exists to expose.
    """
    return (len(AllenRelation) - len(reported)) / (len(AllenRelation) - 1)


def is_sound(reported: frozenset, actual: AllenRelation) -> bool:
    return actual in reported


# ---------------------------------------------------------------------------
# Recurrence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecurrenceCount:
    """How many times a value came back — as a range when reports are missing.

    A point count over a record with known gaps is a guess wearing a number.
    `low` is what the evidence establishes; `high` adds the episodes a gap could
    be hiding. When they differ the system does not know, and says so.
    """
    label: str
    low: int
    high: int
    gaps_in_window: int = 0

    def __post_init__(self):
        if self.low > self.high:
            raise ValueError(f"{self.label}: low {self.low} > high {self.high}")
        if self.gaps_in_window == 0 and self.low != self.high:
            raise ValueError(
                f"{self.label}: an uncertain count with no gap to explain it "
                "is a guess, not a range")

    @property
    def certain(self) -> bool:
        return self.low == self.high

    def describe(self) -> str:
        if self.certain:
            return f"{self.label}: {self.low}"
        return (f"{self.label}: between {self.low} and {self.high} "
                f"({self.gaps_in_window} unreported change(s) in window)")


def count_recurrences(label: str, sequence, target: str,
                      *, gaps: int = 0, contested: int = 0) -> RecurrenceCount:
    """Episodes in which `target` was held, from an ordered value sequence.

    An episode is a maximal run, so a value reported twice in a row is one
    episode, not two — the distinction between a repeated report and a genuine
    return is the whole point of recurrence.

    The range extends in both directions, and the two inputs are not
    symmetric. An unreported change can only HIDE an episode, so `gaps` raises
    the ceiling alone. A contested or unorderable report is ambiguous both ways —
    picking the wrong value can invent a boundary *or* mask one — so `contested`
    moves both bounds. Extending only upward leaves every invented episode
    uncountable and the count stays confidently wrong.

    Both bounds are deliberately loose in the safe direction: understating what
    a corruption could do is how a recurrence gets silently missed.
    """
    episodes = 0
    previous = None
    for value in sequence:
        if value == target and previous != target:
            episodes += 1
        previous = value
    return RecurrenceCount(label=label, low=max(0, episodes - contested),
                           high=episodes + gaps + contested,
                           gaps_in_window=gaps + contested)
