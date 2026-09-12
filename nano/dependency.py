"""Dependency lineage and invalidation — Layer XIII, domain-independent.

The problem (XXX): when evidence is corrected, everything derived from it must
become *inspectably stale* rather than silently wrong.

    E1 -> C1 -> E2 -> P1 -> T1 -> S1 -> R1 -> V1

Correcting E1 must mark the downstream chain stale, retain the prior versions,
and record why the belief changed. Without this, "continual updating" means
recomputing everything from scratch, which is not an architectural property —
it is an admission that lineage was never tracked.

Two failure modes are guarded, and they pull in opposite directions:

  * **Under-invalidation** — a stale artifact keeps being served as current.
    This is the dangerous one: the artifact is confidently wrong.
  * **Over-invalidation** — marking everything stale on any change. Safe but
    useless; it destroys the incrementality that motivates the graph. A test
    pins precision, not just recall.

Nothing here is clinical. A dependency between a chart and its series behaves
identically to one between a proof step and its lemma.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from fabric.schemas import _cid


class Freshness(str, Enum):
    CURRENT = "current"
    STALE = "stale"                  # an input changed; this must be recomputed
    POSSIBLY_STALE = "possibly_stale"  # an ancestor changed; effect unconfirmed
    SUPERSEDED = "superseded"        # replaced by a corrected version


@dataclass(frozen=True)
class Dependency:
    """A derived object and the inputs it was computed from."""
    derived_id: str
    input_ids: tuple
    kind: str = ""            # e.g. "claim<-evidence", "artifact<-state"
    producer: str = ""        # what computed it, for recomputation
    dependency_id: str = ""

    def __post_init__(self):
        if not self.input_ids:
            raise ValueError(
                f"{self.derived_id}: a derived object with no inputs is not derived — "
                "record it as a source instead")
        if self.derived_id in self.input_ids:
            raise ValueError(f"{self.derived_id}: self-dependency")
        object.__setattr__(self, "dependency_id", _cid(
            {"d": self.derived_id, "i": sorted(self.input_ids)}, "dep"))


@dataclass
class DependencyGraph:
    """Lineage over derived objects, with precise invalidation.

    Append-only in the same sense as the ledger: registering a correction does
    not delete the prior node, it marks it SUPERSEDED and records the reason.
    """
    edges: dict = field(default_factory=dict)          # derived_id -> Dependency
    _dependents: dict = field(default_factory=lambda: defaultdict(set))
    freshness: dict = field(default_factory=dict)      # object_id -> Freshness
    reasons: dict = field(default_factory=dict)        # object_id -> why it changed

    def register(self, dep: Dependency) -> "DependencyGraph":
        if dep.derived_id in self.edges:
            raise ValueError(
                f"{dep.derived_id} already has lineage — a recomputation "
                "produces a NEW id rather than rewriting the old one")
        self._detect_cycle(dep)
        self.edges[dep.derived_id] = dep
        for i in dep.input_ids:
            self._dependents[i].add(dep.derived_id)
        self.freshness.setdefault(dep.derived_id, Freshness.CURRENT)
        for i in dep.input_ids:
            self.freshness.setdefault(i, Freshness.CURRENT)
        return self

    def _detect_cycle(self, dep: Dependency) -> None:
        """A cycle means lineage is not a derivation order and cannot be replayed."""
        seen, stack = set(), list(dep.input_ids)
        while stack:
            node = stack.pop()
            if node == dep.derived_id:
                raise ValueError(f"cycle through {dep.derived_id}")
            if node in seen:
                continue
            seen.add(node)
            if node in self.edges:
                stack.extend(self.edges[node].input_ids)

    def dependents_of(self, object_id: str) -> set:
        """Transitive closure of things derived from `object_id`."""
        out, stack = set(), [object_id]
        while stack:
            for d in self._dependents.get(stack.pop(), ()):
                if d not in out:
                    out.add(d)
                    stack.append(d)
        return out

    def invalidate(self, changed_id: str, *, reason: str,
                   superseded: bool = False) -> dict:
        """Mark `changed_id` and only its descendants.

        Direct dependents are STALE (an input demonstrably changed). Deeper
        descendants are POSSIBLY_STALE — the change may not propagate, and
        claiming certainty would be over-invalidation. Distinguishing them is
        what makes selective recomputation possible.
        """
        if not reason:
            raise ValueError("invalidation requires a reason — an unexplained "
                             "belief change is not inspectable")
        self.freshness[changed_id] = (
            Freshness.SUPERSEDED if superseded else Freshness.STALE)
        self.reasons[changed_id] = reason

        direct = set(self._dependents.get(changed_id, ()))
        all_desc = self.dependents_of(changed_id)
        for d in all_desc:
            self.freshness[d] = (
                Freshness.STALE if d in direct else Freshness.POSSIBLY_STALE)
            self.reasons[d] = f"ancestor {changed_id} changed: {reason}"
        return {
            "changed": changed_id,
            "direct": sorted(direct),
            "transitive": sorted(all_desc - direct),
            "unaffected": sorted(set(self.freshness) - all_desc - {changed_id}),
        }

    def stale(self) -> list:
        return sorted(k for k, v in self.freshness.items()
                      if v in (Freshness.STALE, Freshness.POSSIBLY_STALE))

    def current(self) -> list:
        return sorted(k for k, v in self.freshness.items()
                      if v is Freshness.CURRENT)

    def explain(self, object_id: str) -> dict:
        """Why is this object in its current freshness state?"""
        return {
            "object": object_id,
            "freshness": self.freshness.get(object_id, Freshness.CURRENT).value,
            "reason": self.reasons.get(object_id, ""),
            "inputs": list(self.edges[object_id].input_ids)
            if object_id in self.edges else [],
        }

    def recompute_order(self) -> list:
        """Stale objects in dependency order, so inputs are rebuilt first."""
        stale = set(self.stale())
        out, seen = [], set()

        def visit(node):
            if node in seen:
                return
            seen.add(node)
            for i in self.edges.get(node, Dependency("x", ("y",))).input_ids \
                    if node in self.edges else ():
                if i in stale:
                    visit(i)
            if node in stale:
                out.append(node)

        for node in sorted(stale):
            visit(node)
        return out
