"""Causal attribution — constitution §III derivation axis, RSN-CAUSAL.

The named failure mode is precise and it is a safety failure, not a formatting
one:

    temporal association reported as cause

An intervention happened at t; the world changed at t+1. That is co-occurrence.
It becomes a causal claim only when a source *says so*, and a system that
cannot tell the difference will report the two identically — which is how
"the medication was started, then the cough resolved" becomes "the medication
resolved the cough".

Three bases, kept apart
-----------------------
* **DOCUMENTED_RATIONALE** — a source explicitly stated the link. This is the
  only basis that may be presented as a cause, and it is really an *observation
  of an assertion*, not a causal inference.
* **TEMPORAL_ASSOCIATION** — an action and a change co-occur. Reportable, but
  never as a cause, and never without the competing explanations alongside it.
* **NONE** — nothing to say.

Why competing explanations are required
---------------------------------------
A temporal association with no alternatives considered is a cause wearing a
hedge. `CausalLink` therefore refuses to construct a TEMPORAL_ASSOCIATION
without an explicit `competing_explanations` tuple — which may be empty only
when the field `searched_for_alternatives` records that a search happened and
returned nothing. "None found" and "never looked" must not render identically.

Domain-neutral: an actor setting a unit's status and a clinician starting a
medication are the same shape. Nothing here is clinical, and nothing here calls
a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from fabric.schemas import _cid

from nano.contracts import DerivationMode, EpistemicStatus


class CausalBasis(str, Enum):
    DOCUMENTED_RATIONALE = "documented_rationale"
    TEMPORAL_ASSOCIATION = "temporal_association"
    NONE = "none"


#: Which derivation mode each basis is allowed to carry. A documented rationale
#: is OBSERVED because the *statement* was observed; an association is INFERRED
#: because nothing observed the link. Mapping this explicitly stops the two
#: collapsing the first time someone writes a convenience wrapper.
BASIS_DERIVATION = {
    CausalBasis.DOCUMENTED_RATIONALE: DerivationMode.OBSERVED,
    CausalBasis.TEMPORAL_ASSOCIATION: DerivationMode.INFERRED,
    CausalBasis.NONE: DerivationMode.DERIVED,
}

BASIS_STATUS = {
    CausalBasis.DOCUMENTED_RATIONALE: EpistemicStatus.DIRECT_DOCUMENTATION,
    CausalBasis.TEMPORAL_ASSOCIATION: EpistemicStatus.UNCERTAIN,
    CausalBasis.NONE: EpistemicStatus.NOT_FOUND,
}


@dataclass(frozen=True)
class CausalLink:
    """A candidate explanation for one change, with its basis attached.

    The basis is not metadata. It determines what may be said, and the
    invariants below make an over-claim unconstructible rather than merely
    discouraged.
    """
    entity: str
    attribute: str
    effect_value: str
    effect_time: str
    basis: CausalBasis
    candidate_cause: str = ""            # intervention id
    actor: str = ""
    evidence_span_ids: tuple = ()
    competing_explanations: tuple = ()
    searched_for_alternatives: bool = False
    lag_days: int = 0
    link_id: str = ""

    def __post_init__(self):
        if self.basis is CausalBasis.NONE and self.candidate_cause:
            raise ValueError(
                "basis NONE cannot name a candidate cause — that is an "
                "association being smuggled in as an absence")
        if self.basis is not CausalBasis.NONE and not self.candidate_cause:
            raise ValueError(f"basis {self.basis.value} names no cause")
        if self.basis is CausalBasis.DOCUMENTED_RATIONALE and \
                not self.evidence_span_ids:
            raise ValueError(
                "a documented rationale must cite the span that documents it — "
                "otherwise it is an inference claiming to be a quotation")
        if self.basis is CausalBasis.TEMPORAL_ASSOCIATION and \
                not (self.competing_explanations or self.searched_for_alternatives):
            raise ValueError(
                "a temporal association must either list competing explanations "
                "or record that a search found none — 'none found' and 'never "
                "looked' must not render identically")
        object.__setattr__(self, "link_id", _cid(
            {"e": self.entity, "a": self.attribute, "v": self.effect_value,
             "t": self.effect_time, "c": self.candidate_cause}, "cause"))

    @property
    def derivation(self) -> DerivationMode:
        return BASIS_DERIVATION[self.basis]

    @property
    def epistemic_status(self) -> EpistemicStatus:
        return BASIS_STATUS[self.basis]

    @property
    def may_be_stated_as_cause(self) -> bool:
        """Only a documented rationale. This is the whole guard."""
        return self.basis is CausalBasis.DOCUMENTED_RATIONALE

    def describe(self) -> str:
        """Wording is load-bearing here, so it lives with the type rather than
        at each call site where it could drift into something stronger."""
        what = f"{self.entity}.{self.attribute} = {self.effect_value}"
        if self.basis is CausalBasis.NONE:
            return f"{what}: no candidate explanation"
        if self.basis is CausalBasis.DOCUMENTED_RATIONALE:
            return (f"{what}: attributed to {self.candidate_cause} by "
                    f"{self.actor or 'a source'}, as documented")
        alts = (f"{len(self.competing_explanations)} competing explanation(s)"
                if self.competing_explanations
                else "no competing explanation found")
        return (f"{what}: followed {self.candidate_cause} by "
                f"{self.actor or 'an actor'} after {self.lag_days} day(s) — "
                f"association only, not established as the cause; {alts}")


def attribute_cause(entity: str, attribute: str, effect_value: str,
                    effect_time: str, *, actions, documented_causes,
                    window_days: int = 7) -> CausalLink:
    """Explain one change, at the strongest basis the evidence actually supports.

    Documented first, association second, nothing third. The order matters: a
    documented link should never be downgraded to an association, and an
    association must never be upgraded because no documentation was found.

    `actions` is a sequence of `(intervention_id, actor, entity, attribute,
    time, lag_days)`; `documented_causes` maps `(entity, attribute, time,
    value)` to `(intervention_id, span_id)`. The value is part of the key
    deliberately: a rationale documents a specific outcome, and matching on time
    alone would attach it to a different value reported at the same moment.
    Both are plain data — this function cannot reach a world or a ground truth.
    """
    documented = documented_causes.get(
        (entity, attribute, effect_time, effect_value))
    if documented:
        intervention_id, span_id = documented
        actor = next((a[1] for a in actions if a[0] == intervention_id), "")
        return CausalLink(
            entity=entity, attribute=attribute, effect_value=effect_value,
            effect_time=effect_time,
            basis=CausalBasis.DOCUMENTED_RATIONALE,
            candidate_cause=intervention_id, actor=actor,
            evidence_span_ids=(span_id,))

    nearby = [a for a in actions
              if a[2] == entity and a[3] == attribute
              and 0 <= a[5] <= window_days]
    if not nearby:
        return CausalLink(
            entity=entity, attribute=attribute, effect_value=effect_value,
            effect_time=effect_time, basis=CausalBasis.NONE,
            searched_for_alternatives=True)

    nearby.sort(key=lambda a: a[5])
    best = nearby[0]
    return CausalLink(
        entity=entity, attribute=attribute, effect_value=effect_value,
        effect_time=effect_time, basis=CausalBasis.TEMPORAL_ASSOCIATION,
        candidate_cause=best[0], actor=best[1], lag_days=best[5],
        competing_explanations=tuple(a[0] for a in nearby[1:]),
        searched_for_alternatives=True)


def post_hoc_attribute(entity: str, attribute: str, effect_value: str,
                       effect_time: str, *, actions, documented_causes,
                       window_days: int = 7) -> CausalLink:
    """Control arm: *post hoc ergo propter hoc*.

    Identical inputs, identical search — the single difference is that a
    temporal association is reported at documented strength. This is not a
    strawman; it is what a system that has no basis distinction necessarily
    does, and naming it as the control makes the cost of that missing
    distinction measurable rather than assumed.

    It fabricates a citation to the action record in order to satisfy the
    documented-rationale invariant, which is itself the tell: the only way to
    state an association as a cause is to cite something that does not document
    it.
    """
    documented = documented_causes.get(
        (entity, attribute, effect_time, effect_value))
    if documented:
        intervention_id, span_id = documented
        actor = next((a[1] for a in actions if a[0] == intervention_id), "")
        return CausalLink(
            entity=entity, attribute=attribute, effect_value=effect_value,
            effect_time=effect_time,
            basis=CausalBasis.DOCUMENTED_RATIONALE,
            candidate_cause=intervention_id, actor=actor,
            evidence_span_ids=(span_id,))

    nearby = sorted([a for a in actions
                     if a[2] == entity and a[3] == attribute
                     and 0 <= a[5] <= window_days], key=lambda a: a[5])
    if not nearby:
        return CausalLink(
            entity=entity, attribute=attribute, effect_value=effect_value,
            effect_time=effect_time, basis=CausalBasis.NONE,
            searched_for_alternatives=True)
    best = nearby[0]
    return CausalLink(
        entity=entity, attribute=attribute, effect_value=effect_value,
        effect_time=effect_time, basis=CausalBasis.DOCUMENTED_RATIONALE,
        candidate_cause=best[0], actor=best[1], lag_days=best[5],
        evidence_span_ids=(f"action_record:{best[0]}",))
