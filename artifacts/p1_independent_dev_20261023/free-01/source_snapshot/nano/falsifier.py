"""What would change the conclusion — MTA-WOULDCHANGE.

The runtime can already say what it believes (`Answer`) and what it needs
(`nano.needs`). It cannot say what would change its mind. That is the third
thing an operator actually wants, and it is the one that makes the other two
auditable: a belief whose falsifier is unnamed cannot be checked by anyone, and
a belief whose falsifier is *vague* cannot be checked by anyone either.

Executable or it isn't a falsifier
----------------------------------
"New evidence to the contrary would change this" is true of every claim ever
made and actionable for none of them. A falsifier here has to name a concrete
observation — which entity, which attribute, which value, at which time — so
that it can be **executed**: construct it, admit it through the normal ingest
path, and check the answer actually moves.

That turns the capability into something measurable rather than rhetorical:

* **validity** — when the named observation arrives, does the answer change?
  A falsifier that does not fire when satisfied is false.
* **discrimination** — when a *near miss* arrives (the same key, corroborating
  rather than contradicting), does the answer correctly stay put? A falsifier
  satisfied by anything is useless even when it is technically valid.

Both are needed. Validity alone is maximised by naming something so broad that
any new evidence satisfies it; discrimination alone is maximised by naming
something impossible.

Derived from lineage, not invented
----------------------------------
A falsifier is a function of what the claim *rests on*, so `rests_on` is
required whenever the claim has evidence. If a falsifier could be constructed
without reference to the claim's dependencies it would be a guess about what
might matter, which is exactly the thing this module exists to replace.

Domain-neutral, deterministic, no model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from fabric.schemas import _cid


class FalsifierKind(str, Enum):
    """How the named observation would do its work."""
    SUPERSEDING_REPORT = "superseding_report"      # later, different value
    CONTRADICTING_REPORT = "contradicting_report"  # same time, different value
    FIRST_REPORT = "first_report"                  # anything, where nothing is known
    RESOLVING_REPORT = "resolving_report"          # settles an existing dispute


#: What the answer becomes if the falsifier is satisfied. Naming the destination
#: is part of the contract: "this could change" is not a falsifier, "this would
#: become X" is.
class Outcome(str, Enum):
    NEW_VALUE = "the held value is replaced"
    BECOMES_DISPUTED = "the key becomes contested and escalates to review"
    BECOMES_KNOWN = "an unknown key acquires a value"
    BECOMES_SETTLED = "a contested key resolves"


def _next_day(stamp: str) -> str:
    """The first day strictly after a possibly-imprecise timestamp.

    Month precision resolves to the day after that month ends — the earliest
    date that is unambiguously later. Picking a day *inside* the month would
    produce a falsifier that cannot be ordered against the claim it is meant to
    overturn, which is the mixed-precision defect one more time.
    """
    if len(stamp) == 7 and stamp[4] == "-":
        y, m = int(stamp[:4]), int(stamp[5:7])
        nxt = date(y + (m == 12), (m % 12) + 1, 1)
        return nxt.isoformat()
    return (date.fromisoformat(stamp[:10]) + timedelta(days=1)).isoformat()


@dataclass(frozen=True)
class Falsifier:
    """A concrete observation that would change a specific conclusion."""
    claim: str
    entity: str
    attribute: str
    value: str
    event_time: str
    kind: FalsifierKind
    outcome: Outcome
    rests_on: tuple = ()
    reason: str = ""
    falsifier_id: str = ""

    def __post_init__(self):
        if not (self.entity and self.attribute and self.value
                and self.event_time):
            raise ValueError(
                f"{self.claim}: a falsifier must name a concrete observation — "
                "entity, attribute, value and time. 'Evidence to the contrary' "
                "is true of every claim and actionable for none")
        if not self.reason:
            raise ValueError(f"{self.claim}: a falsifier must say why it bites")
        if self.kind is not FalsifierKind.FIRST_REPORT and not self.rests_on:
            raise ValueError(
                f"{self.claim}: a falsifier for an evidenced claim must cite "
                "what the claim rests on — otherwise it is a guess about what "
                "might matter, not a derivation from what does")
        object.__setattr__(self, "falsifier_id", _cid(
            {"c": self.claim, "e": self.entity, "a": self.attribute,
             "v": self.value, "t": self.event_time}, "fals"))

    def describe(self) -> str:
        return (f"if {self.entity}.{self.attribute} were reported as "
                f"{self.value} on {self.event_time}, {self.outcome.value}")


def _other_value(current: str) -> str:
    """A value that is definitely not the current one.

    Deliberately synthetic rather than drawn from the world's vocabulary: the
    falsifier is a hypothetical, and borrowing a real value would make it look
    like a prediction about what will be reported.
    """
    return f"not:{current}"


def falsifiers_for(entity: str, attribute: str, *, resolved, conflicted,
                   latest_time, evidence, as_of: str) -> tuple:
    """Every named observation that would change the conclusion about one key.

    Derived from the resolution rule itself, which is what makes these
    exhaustive-by-construction for the mechanisms the resolver actually uses:
    a later differing report replaces the value; a same-time differing report
    contests it; where nothing is known, anything settles it.
    """
    key = (entity, attribute)
    spans = tuple(evidence.get(key, ()))

    if key in conflicted:
        when, _values = conflicted[key]
        return (Falsifier(
            claim=f"{entity}.{attribute} is disputed",
            entity=entity, attribute=attribute,
            value=_other_value("dispute"), event_time=_next_day(when),
            kind=FalsifierKind.RESOLVING_REPORT,
            outcome=Outcome.BECOMES_SETTLED, rests_on=spans,
            reason="a report strictly later than the disputed time would "
                   "supersede both sides and settle the key"),)

    if key in resolved:
        current = resolved[key]
        when = latest_time.get(key, as_of)
        return (
            Falsifier(
                claim=f"{entity}.{attribute} = {current}",
                entity=entity, attribute=attribute,
                value=_other_value(current), event_time=_next_day(when),
                kind=FalsifierKind.SUPERSEDING_REPORT,
                outcome=Outcome.NEW_VALUE, rests_on=spans,
                reason="a later report of a different value wins the "
                       "resolution outright"),
            Falsifier(
                claim=f"{entity}.{attribute} = {current}",
                entity=entity, attribute=attribute,
                value=_other_value(current), event_time=when,
                kind=FalsifierKind.CONTRADICTING_REPORT,
                outcome=Outcome.BECOMES_DISPUTED, rests_on=spans,
                reason="a differing report at the same event time cannot be "
                       "ordered against the current one, so the key becomes "
                       "contested rather than replaced"),
        )

    return (Falsifier(
        claim=f"{entity}.{attribute} is unknown",
        entity=entity, attribute=attribute, value="any", event_time=as_of,
        kind=FalsifierKind.FIRST_REPORT, outcome=Outcome.BECOMES_KNOWN,
        reason="nothing is known, so any report establishes a value"),)


def broad_falsifier(entity: str, attribute: str, *, resolved, conflicted,
                    latest_time, evidence, as_of: str) -> tuple:
    """Control arm: name the key, but constrain nothing about it.

    "A new report about this key would change the answer" is what a system
    writes when it has not derived the falsifier from its own resolution rule.
    It sounds specific — it names an entity and an attribute — and it is
    satisfied by evidence that changes nothing at all.

    Not a strawman: it is strictly more informative than the usual "evidence to
    the contrary", and it is still under-specified in exactly the way that
    matters. The measured gap between this and `falsifiers_for` is the value of
    deriving rather than guessing.
    """
    key = (entity, attribute)
    current = resolved.get(key, "unknown")
    return (Falsifier(
        claim=f"{entity}.{attribute} = {current}",
        entity=entity, attribute=attribute,
        # Any report at all, so the earliest thing that satisfies the wording is
        # a corroboration of what is already believed.
        value=current if key in resolved else "any",
        event_time=latest_time.get(key, as_of),
        kind=(FalsifierKind.SUPERSEDING_REPORT if key in resolved
              else FalsifierKind.FIRST_REPORT),
        outcome=Outcome.NEW_VALUE if key in resolved else Outcome.BECOMES_KNOWN,
        rests_on=tuple(evidence.get(key, ())) or ("unspecified",),
        reason="any new report about this key might change it"),)
