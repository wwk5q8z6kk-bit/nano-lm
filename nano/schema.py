"""Dynamic Hierarchical Ontology & Constraint Graph — Layer I/VI, domain-independent.

Replaces hardcoded flat domain enums (`CHIEF_COMPLAINT`, `MEDICATION`, etc.) with
a composable, runtime-extensible schema DAG supporting:
  - arbitrary nesting (`is_a`, `part_of`, `component_of`, `attribute_of`)
  - sibling topic competition & mutual exclusion without hardcoded domain branches
  - denomination-aware hierarchical rollup & coboundary consistency (`sum`, `max`, `min`)
  - depth-invariant relative tree-gauge addressing `(up_hops, branch_idx)`
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Iterable

from nano.units import DenomRelation, DenominationRegistry, QuantityValue


class NodeRelation(str, Enum):
    ATTRIBUTE_OF = "attribute_of"
    PART_OF = "part_of"
    IS_A = "is_a"
    COMPONENT_OF = "component_of"
    ALTERNATIVE_OF = "alternative_of"


class ValueKind(str, Enum):
    TEXT_LABEL = "text_label"
    QUANTITY = "quantity"
    ENTITY = "entity"
    ORDINAL = "ordinal"
    BOOLEAN = "boolean"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class OntologyNode:
    """Declarative specification for one node in a dynamic hierarchical schema."""
    node_id: str
    label: str = ""
    parent_id: str | None = None
    relation_to_parent: NodeRelation = NodeRelation.ATTRIBUTE_OF
    value_kind: ValueKind = ValueKind.TEXT_LABEL
    dimension: str = ""
    allowed_roles: tuple[str, ...] = ()
    default_denomination: str = ""
    ordinal_scale: tuple[tuple[str, float, float], ...] = ()
    topic_cues: tuple[str, ...] = ()         # Regexes indicating topic in questions or clauses
    predicate_cues: tuple[str, ...] = ()     # Strong clause predicate cues assigning ownership
    denial_cues: tuple[str, ...] = ()        # Domain-specific explicit absence patterns
    value_extract_patterns: tuple[str, ...] = ()
    lead_strip_patterns: tuple[str, ...] = ()
    trail_strip_patterns: tuple[str, ...] = ()
    disallowed_value_patterns: tuple[str, ...] = ()
    episode_disclaimer_cues: tuple[str, ...] = ()
    coexisting_list_patterns: tuple[str, ...] = ()
    contrast_disclaimer_patterns: tuple[str, ...] = ()
    exception_cues: tuple[str, ...] = (r"\bexcept\b",)
    exempt_leading_articles: tuple[str, ...] = ()
    require_topic_cue: bool = True
    rollup_op: str = ""                      # "" | "sum" | "max" | "min" over children
    rollup_tolerance: float = 1e-3
    exclusive_siblings: bool = True
    exclusive_values: bool | None = None
    supports_absence: bool = False
    reject_leading_articles: bool = False
    allow_quoted_or_self_ref: bool = False
    max_words: int = 14
    max_chars: int = 120
    aliases: tuple[str, ...] = ()

    @property
    def is_single_scalar_value(self) -> bool:
        if self.exclusive_values is not None:
            return self.exclusive_values
        return self.value_kind in (ValueKind.QUANTITY, ValueKind.ORDINAL, ValueKind.BOOLEAN)

    def __post_init__(self) -> None:
        if not self.node_id or "." in self.node_id and self.node_id.startswith("."):
            raise ValueError(f"invalid node_id: {self.node_id!r}")
        if not self.label:
            object.__setattr__(self, "label", self.node_id.split(".")[-1])
        if self.rollup_op not in ("", "sum", "max", "min"):
            raise ValueError(f"unsupported rollup_op: {self.rollup_op!r}")


@dataclass(frozen=True)
class HierarchyCheckResult:
    parent_id: str
    status: str  # consistent | conflict | incomparable | incomplete
    reason: str
    expected_quantity: QuantityValue | None = None
    observed_quantity: QuantityValue | None = None
    child_ids: tuple[str, ...] = ()


@dataclass
class OntologyGraph:
    """Runtime-dynamic hierarchical ontology graph shared across symbolic and neural paths."""
    nodes: dict[str, OntologyNode] = field(default_factory=dict)
    _order: list[str] = field(default_factory=list)
    _children: dict[str | None, list[str]] = field(default_factory=lambda: defaultdict(list))
    _compiled: dict[str, dict[str, tuple[re.Pattern[str], ...]]] = field(default_factory=dict)

    def __init__(self, initial_nodes: Iterable[OntologyNode] = ()) -> None:
        self.nodes = {}
        self._order = []
        self._children = defaultdict(list)
        self._compiled = {}
        for node in initial_nodes:
            self.register(node)
        self._validate_acyclic()

    def register(self, node: OntologyNode) -> "OntologyGraph":
        """Add or dynamically extend the schema graph with a new typed node."""
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate OntologyNode {node.node_id!r}")
        if node.parent_id == node.node_id:
            raise ValueError(f"self-parent cycle on {node.node_id!r}")
        self.nodes[node.node_id] = node
        self._order.append(node.node_id)
        self._children[node.parent_id].append(node.node_id)
        self._compiled[node.node_id] = {
            "topic": tuple(re.compile(p, re.IGNORECASE) for p in node.topic_cues),
            "predicate": tuple(re.compile(p, re.IGNORECASE) for p in node.predicate_cues),
            "denial": tuple(re.compile(p, re.IGNORECASE) for p in node.denial_cues),
            "value_extract": tuple(re.compile(p, re.IGNORECASE) for p in node.value_extract_patterns),
            "lead_strip": tuple(re.compile(p, re.IGNORECASE) for p in node.lead_strip_patterns),
            "trail_strip": tuple(re.compile(p, re.IGNORECASE) for p in node.trail_strip_patterns),
            "disallowed": tuple(re.compile(p, re.IGNORECASE) for p in node.disallowed_value_patterns),
            "episode_disclaimer": tuple(re.compile(p, re.IGNORECASE) for p in node.episode_disclaimer_cues),
            "coexisting_list": tuple(re.compile(p, re.IGNORECASE) for p in node.coexisting_list_patterns),
            "contrast_disclaimer": tuple(re.compile(p, re.IGNORECASE) for p in node.contrast_disclaimer_patterns),
            "exception": tuple(re.compile(p, re.IGNORECASE) for p in node.exception_cues),
        }
        self._validate_acyclic()
        return self

    def _validate_acyclic(self) -> None:
        for nid in self.nodes:
            seen: set[str] = set()
            cur: str | None = nid
            while cur is not None:
                if cur in seen:
                    raise ValueError(f"cycle detected in OntologyGraph at {cur!r}")
                seen.add(cur)
                cur = self.nodes[cur].parent_id if cur in self.nodes else None

    @property
    def ordered_nodes(self) -> tuple[OntologyNode, ...]:
        return tuple(self.nodes[nid] for nid in self._order)

    def children_of(self, node_id: str | None) -> tuple[OntologyNode, ...]:
        return tuple(self.nodes[cid] for cid in self._children.get(node_id, ()))

    def siblings_of(self, node_id: str) -> tuple[OntologyNode, ...]:
        node = self.nodes[node_id]
        return tuple(self.nodes[cid] for cid in self._children.get(node.parent_id, ()) if cid != node_id)

    def depth_of(self, node_id: str) -> int:
        d = 0
        cur = self.nodes[node_id].parent_id
        while cur is not None and cur in self.nodes:
            d += 1
            cur = self.nodes[cur].parent_id
        return d

    def path_from_root(self, node_id: str) -> tuple[str, ...]:
        chain: list[str] = []
        cur: str | None = node_id
        while cur is not None and cur in self.nodes:
            chain.append(cur)
            cur = self.nodes[cur].parent_id
        chain.reverse()
        return tuple(chain)

    def relative_address(self, from_node_id: str, to_node_id: str) -> tuple[int, tuple[int, ...]]:
        """Depth-invariant relative tree gauge: `(up_hops, child_branch_indices_down)`."""
        p_from = self.path_from_root(from_node_id)
        p_to = self.path_from_root(to_node_id)
        lca_len = 0
        for a, b in zip(p_from, p_to):
            if a == b:
                lca_len += 1
            else:
                break
        up_hops = len(p_from) - lca_len
        down_indices: list[int] = []
        for idx in range(lca_len, len(p_to)):
            cid = p_to[idx]
            parent_id = self.nodes[cid].parent_id
            down_indices.append(self._children[parent_id].index(cid))
        return up_hops, tuple(down_indices)

    def patterns(self, node_id: str) -> dict[str, tuple[re.Pattern[str], ...]]:
        return self._compiled[node_id]

    def score_clause_ownership(self, clause_text: str, prev_question: str = "") -> dict[str, float]:
        """Score every node on `(predicate, topic, denial, question)` cues without hardcoded branches.

        Resolves sibling topic collisions structurally: a clause explicitly mentioning
        allergy reactions scores higher on `sensitivity.allergy` than a sibling `medication` node.
        """
        scores: dict[str, float] = {}
        for nid, node in self.nodes.items():
            pats = self._compiled[nid]
            s = 0.0
            for p in pats["predicate"]:
                if p.search(clause_text):
                    s += 3.0
            for p in pats["denial"]:
                if p.search(clause_text):
                    s += 2.5
            for p in pats["topic"]:
                if p.search(clause_text):
                    s += 1.5
            # Context question alignment weighs meaningfully unless clause explicitly switches topic
            q_hits = sum(1 for p in pats["topic"] if p.search(prev_question)) if prev_question else 0
            if q_hits:
                s += 2.0
            scores[nid] = s
        return scores

    def owning_siblings_win(self, node_id: str, clause_text: str, prev_question: str = "") -> bool:
        """True unless another non-ancestor/non-descendant node owns `clause_text` more strongly.

        Checks across the full OntologyGraph hierarchy (not only direct siblings) so
        a top-level node (e.g., `debt`) never steals values from a deeper child node
        (e.g., `assets.current.cash`) inside an answer to a multi-topic question.
        """
        node = self.nodes[node_id]
        if not node.exclusive_siblings:
            return True
        scores = self.score_clause_ownership(clause_text, prev_question)
        my_score = scores[node_id]
        my_path = set(self.path_from_root(node_id))
        clause_has_me = any(
            p.search(clause_text) for k in ("predicate", "denial", "topic") for p in self._compiled[node_id][k]
        )

        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue
            other_path = set(self.path_from_root(other_id))
            # Ancestor / descendant nodes: allow multi-attribute sharing (`ATTRIBUTE_OF` across distinct kinds/dims
            # or when the clause explicitly names `node_id`), while preventing parent summary nodes from stealing child PART_OF spans.
            if other_id in my_path or node_id in other_path:
                is_attr_pair = (
                    (other_id in my_path and node.relation_to_parent is NodeRelation.ATTRIBUTE_OF)
                    or (node_id in other_path and other_node.relation_to_parent is NodeRelation.ATTRIBUTE_OF)
                )
                if clause_has_me or (is_attr_pair and (other_node.value_kind != node.value_kind or other_node.dimension != node.dimension)):
                    continue
            # Orthogonal attribute siblings under the same parent with distinct non-empty dimensions
            # (e.g., `duration` vs `severity` inside "Moderate pain for three days") can coexist in one clause.
            if (
                other_node.parent_id == node.parent_id
                and node.relation_to_parent is NodeRelation.ATTRIBUTE_OF
                and other_node.relation_to_parent is NodeRelation.ATTRIBUTE_OF
                and node.dimension
                and other_node.dimension
                and node.dimension != other_node.dimension
            ):
                continue
            other_pats = self._compiled[other_id]
            clause_has_other = any(
                p.search(clause_text) for k in ("predicate", "denial", "topic") for p in other_pats[k]
            )
            if clause_has_other and not clause_has_me and scores[other_id] >= my_score:
                return False
            if clause_has_other and scores[other_id] > my_score + 0.5:
                return False
            # Also if question explicitly asked about any non-ancestor/non-descendant node and clause didn't mention me
            if other_id not in my_path and node_id not in other_path and prev_question and not clause_has_me:
                q_has_other = any(p.search(prev_question) for p in other_pats["topic"])
                q_has_me = any(p.search(prev_question) for p in self._compiled[node_id]["topic"])
                if q_has_other and not q_has_me:
                    return False
        return True

    def _postorder_nodes(self) -> tuple[OntologyNode, ...]:
        visited: set[str] = set()
        ordered: list[OntologyNode] = []

        def dfs(nid: str) -> None:
            if nid in visited:
                return
            visited.add(nid)
            for cid in self._children.get(nid, ()):
                dfs(cid)
            ordered.append(self.nodes[nid])

        for nid in self._order:
            dfs(nid)
        return tuple(ordered)

    def verify_rollups(
        self,
        resolved_quantities: dict[str, QuantityValue],
        registry: DenominationRegistry,
    ) -> list[HierarchyCheckResult]:
        """Verify conservation and parent-child consistency across hierarchical trees (bottom-up post-order)."""
        results: list[HierarchyCheckResult] = []
        working = dict(resolved_quantities)
        for node in self._postorder_nodes():
            if not node.rollup_op:
                continue
            children = [
                c for c in self.children_of(node.node_id)
                if c.relation_to_parent in (NodeRelation.PART_OF, NodeRelation.COMPONENT_OF)
            ]
            if not children:
                continue
            child_ids = tuple(c.node_id for c in children)
            present = [working[c.node_id] for c in children if c.node_id in working]
            if len(present) < len(children):
                if present or node.node_id in working:
                    results.append(HierarchyCheckResult(
                        parent_id=node.node_id,
                        status="incomplete",
                        reason=f"only {len(present)}/{len(children)} child components resolved",
                        observed_quantity=working.get(node.node_id),
                        child_ids=child_ids,
                    ))
                continue

            target_denom = node.default_denomination or (
                working[node.node_id].denomination if node.node_id in working else ""
            )
            if node.rollup_op == "sum":
                agg, status = registry.sum_quantities(present, target_denomination=target_denom)
                if agg is None:
                    results.append(HierarchyCheckResult(
                        parent_id=node.node_id,
                        status="incomparable",
                        reason=status,
                        observed_quantity=working.get(node.node_id),
                        child_ids=child_ids,
                    ))
                    continue
            else:
                # max / min across comparable children
                ref_q = present[0]
                conv_list: list[QuantityValue] = []
                err_reason = ""
                for q in present:
                    if q.dimension != ref_q.dimension:
                        err_reason = f"dimension_mismatch:{ref_q.dimension}:{q.dimension}"
                        break
                    qc = registry.convert(q, target_denom or ref_q.denomination) if q.denomination != (target_denom or ref_q.denomination) else q
                    if qc is None:
                        err_reason = f"missing_conversion:{q.denomination}->{target_denom or ref_q.denomination}"
                        break
                    conv_list.append(qc)
                if err_reason:
                    results.append(HierarchyCheckResult(
                        parent_id=node.node_id, status="incomparable", reason=err_reason,
                        observed_quantity=working.get(node.node_id), child_ids=child_ids,
                    ))
                    continue
                fn = max if node.rollup_op == "max" else min
                agg_min = fn(q.base_min for q in conv_list)
                agg_max = fn(q.base_max for q in conv_list)
                agg = QuantityValue(
                    raw_text=f"{node.rollup_op}({len(conv_list)})",
                    dimension=conv_list[0].dimension,
                    unit=conv_list[0].unit,
                    denomination=conv_list[0].denomination,
                    role=conv_list[0].role,
                    base_unit=conv_list[0].base_unit,
                    base_min=agg_min,
                    base_max=agg_max,
                )

            parent_obs = working.get(node.node_id)
            if parent_obs is None:
                working[node.node_id] = agg
                results.append(HierarchyCheckResult(
                    parent_id=node.node_id, status="consistent",
                    reason="derived_from_children", expected_quantity=agg,
                    observed_quantity=None, child_ids=child_ids,
                ))
                continue

            cmp_res = registry.compare(agg, parent_obs, rel_tol=node.rollup_tolerance)
            if cmp_res.relation in (DenomRelation.EQUIVALENT, DenomRelation.OVERLAPPING):
                results.append(HierarchyCheckResult(
                    parent_id=node.node_id, status="consistent", reason=cmp_res.reason,
                    expected_quantity=agg, observed_quantity=parent_obs, child_ids=child_ids,
                ))
            elif cmp_res.relation in (DenomRelation.INCOMPARABLE_DENOMINATION, DenomRelation.INCOMPARABLE_DIMENSION, DenomRelation.ROLE_MISMATCH):
                results.append(HierarchyCheckResult(
                    parent_id=node.node_id, status="incomparable", reason=cmp_res.reason,
                    expected_quantity=agg, observed_quantity=parent_obs, child_ids=child_ids,
                ))
            else:
                results.append(HierarchyCheckResult(
                    parent_id=node.node_id, status="conflict",
                    reason=f"hierarchy_{node.rollup_op}_mismatch: {cmp_res.reason}",
                    expected_quantity=agg, observed_quantity=parent_obs, child_ids=child_ids,
                ))
        return results
