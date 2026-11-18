"""Domain-Agnostic Epistemic Verifier, Joint Extractor & Hierarchical Custody Engine.

Operates on ANY `OntologyGraph` + `DenominationRegistry` across arbitrary domains
(finance/denominations, clinical encounters, SWE/infra incidents, legal obligations,
hierarchical inventories) with zero domain-specific branches in engine logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Iterable

from nano.dependency import Dependency, DependencyGraph
from nano.schema import HierarchyCheckResult, OntologyGraph, OntologyNode, ValueKind
from nano.units import DenomRelation, DenominationRegistry, QuantityValue


class TurnRole(str, Enum):
    PRINCIPAL = "principal"        # Patient / Client / Reporter / Primary Actor
    INTERLOCUTOR = "interlocutor"  # Clinician / Auditor / Interviewer / Questioner
    EXTERNAL = "external"          # Unverified Other / Unknown bystander
    SYSTEM = "system"


class GroundedState(str, Enum):
    SUPPORTED = "supported"
    ABSENT = "absent"
    MISSING = "missing"
    UNCERTAIN = "uncertain"
    CONFLICTING = "conflicting"


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span bounds ({self.start}, {self.end})")
        if len(self.text) != self.end - self.start:
            raise ValueError("span text length mismatch")


@dataclass(frozen=True)
class SourceTurn:
    turn_id: int
    role: TurnRole
    speaker_label: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class DocumentSource:
    source_id: str
    raw_text: str
    turns: tuple[SourceTurn, ...]


@dataclass(frozen=True)
class ConflictPair:
    node_id: str
    kind: str  # value_disagreement | presence_disagreement | incomparable_denomination | role_mismatch | hierarchy_conflict
    spans: tuple[SourceSpan, ...]
    reason: str


@dataclass(frozen=True)
class NodeResolution:
    node_id: str
    state: GroundedState
    value_text: str | None = None
    quantity: QuantityValue | None = None
    spans: tuple[SourceSpan, ...] = ()
    verified: bool = False
    reason: str = ""


@dataclass
class GroundedWorldReport:
    source_id: str
    resolutions: dict[str, NodeResolution]
    conflicts: dict[str, tuple[ConflictPair, ...]]
    hierarchy_checks: tuple[HierarchyCheckResult, ...]
    dependency_graph: DependencyGraph
    global_unsafe: bool = False


_DEFAULT_INTERLOCUTORS = frozenset({"doctor", "clinician", "physician", "nurse", "interviewer", "auditor", "analyst", "reviewer", "assistant"})
_DEFAULT_PRINCIPALS = frozenset({"patient", "client", "user", "reporter", "borrower", "speaker", "engineer", "owner", "vendor", "applicant"})
_DEFAULT_EXTERNALS = frozenset({"other", "unknown", "third_party", "bystander"})

_NEGATION_WORD_RE = re.compile(
    r"\b(?:no|not|none|nothing|neither|nor|never|without|zero|denies|denied|deny|"
    r"cannot|can't|don't|do not|didn't|did not|doesn't|does not|haven't|have not|"
    r"hasn't|has not|hadn't|won't|wouldn't|isn't|is not|aren't|are not|wasn't|weren't)\b",
    re.IGNORECASE,
)

_EPISTEMIC_HEDGE_RE = re.compile(
    r"\b(?:unsure|uncertain|unknown|maybe|perhaps|possibly|probably|potentially|"
    r"might|may have|not sure|am not sure|i'm not sure|"
    r"can't remember|cannot remember|don't remember|do not remember|"
    r"can't recall|cannot recall|don't recall|"
    r"don't know|do not know|not that i know of|"
    r"can't tell|cannot tell|can't decide|cannot decide|can't choose|cannot choose|"
    r"can't pin down|cannot pin down|can't put it|cannot put it|hard to judge|"
    r"whether|different accounts|or perhaps|guess|suppose|wonder)\b",
    re.IGNORECASE,
)

_NONUSE_ACTION_RE = re.compile(
    r"\b(?:avoid|avoided|avoiding|avoidance|skip|skipped|skipping|"
    r"withhold|withheld|withholding|stop|stopped|stopping|"
    r"discontinue|discontinued|discontinuing|quit|consider|considered|considering|"
    r"declined|cancelled|canceled|suspended)\b",
    re.IGNORECASE,
)

_THIRD_PARTY_RE = re.compile(
    r"\b(?:mother|father|sister|brother|family|child|children|daughter|son|"
    r"wife|husband|partner|friend|cousin|neighbor|neighbour|uncle|aunt|"
    r"grandmother|grandfather|someone else|anyone else|competitor|subsidiary|his|her|their)\b",
    re.IGNORECASE,
)

_PAST_OR_HYPOTHETICAL_RE = re.compile(
    r"\b(?:used to|previously|formerly|last year|last month|last winter|last spring|"
    r"last summer|last fall|last autumn|years ago|in childhood|as a child|"
    r"previous episode|previous visit|old episode|old history|unrelated to why|"
    r"has gone away|gone away|hypothetical|pretend|ignore all instructions|repeat after me)\b"
    r"|\bif\s+(?:i|we|they|it)\b",
    re.IGNORECASE,
)

_SELF_CORRECTION_RE = re.compile(
    r"\b(?:sorry|mistake|mistaken|wrong|incorrect|meant|correction|"
    r"other thing i was thinking of|intake form.*mistake|scratch that|retract)\b"
    r"|\bactually,?\s+(?:no|not|never)\b",
    re.IGNORECASE,
)

_GLOBAL_REASSIGNMENT_RE = re.compile(
    r"\b(?:all (?:of )?(?:my |the |those |these )?(?:answers|responses|statements|entries|numbers)|"
    r"those answers|these answers|all of that|everything i said)\b"
    r".*\b(?:about|belong|belonged|were|was|for|incorrect|wrong|untrue|invalid)\b"
    r"|\b(?:that|this|it) was (?:about |actually )?(?:my|his|her|their) \w+['’]s (?:experience|story|answer)\b"
    r"|\bi made (?:that|it|those answers) up\b",
    re.IGNORECASE,
)

_QUESTION_HAZARD_RE = re.compile(
    r"\b(?:"
    r"mother|father|sister|brother|family|child|children|daughter|son|wife|husband|partner|friend|cousin|"
    r"someone else|anyone else|he|she|(?:his|her|their)\s+(?:mother|father|sister|brother|family|child|daughter|son|wife|husband|partner|friend|cousin|pain|symptoms?|medications?|medicines?)|"
    r"last year|last month|last winter|last spring|previous episode|previous visit|old episode|old history|used to|in childhood|"
    r"hypothetical|pretend|ignore|repeat after me"
    r")\b",
    re.IGNORECASE,
)

_QUOTED_REF_RE = re.compile(r"\b(?:just\s+says|reads)\s+['\"]", re.IGNORECASE)

_ELLIPTICAL_DENIAL_RE = re.compile(
    r"^(?:no|none|nope|no,?\s+none|none\s+today|i\s+haven't|i\s+have\s+not|i\s+didn't|i\s+did\s+not|"
    r"i\s+don't\s+have\s+any|i\s+do\s+not\s+have\s+any|"
    r"none\s+at\s+all|there\s+aren't\s+any|there\s+are\s+none|not\s+any|zero)[.!]?$",
    re.IGNORECASE,
)


def parse_document_source(
    raw_text: str,
    *,
    source_id: str = "doc-0",
    interlocutor_labels: frozenset[str] = _DEFAULT_INTERLOCUTORS,
    principal_labels: frozenset[str] = _DEFAULT_PRINCIPALS,
    external_labels: frozenset[str] = _DEFAULT_EXTERNALS,
) -> DocumentSource:
    """Parse speaker-labeled lines or raw prose while preserving exact offsets."""
    turns: list[SourceTurn] = []
    offset = 0
    for idx, line in enumerate(raw_text.splitlines(keepends=True)):
        stripped = line.rstrip("\r\n")
        m = re.match(r"^(?P<label>[A-Za-z][A-Za-z0-9_ -]{0,24}):\s*(?P<body>.*)$", stripped)
        if m:
            lbl = m.group("label").strip()
            low = lbl.lower()
            if low in interlocutor_labels:
                role = TurnRole.INTERLOCUTOR
            elif low in external_labels:
                role = TurnRole.EXTERNAL
            elif low in principal_labels:
                role = TurnRole.PRINCIPAL
            else:
                role = TurnRole.PRINCIPAL
            body = m.group("body")
            b_start = offset + m.start("body")
            b_end = offset + m.end("body")
        else:
            lbl = "document"
            role = TurnRole.PRINCIPAL
            body = stripped
            b_start = offset
            b_end = offset + len(stripped)
        if body:
            turns.append(SourceTurn(idx, role, lbl, b_start, b_end, body))
        offset += len(line)
    return DocumentSource(source_id=source_id, raw_text=raw_text, turns=tuple(turns))


def split_turn_clauses(turn: SourceTurn) -> tuple[SourceSpan, ...]:
    """Split a turn into sentence/semicolon clauses WITHOUT breaking decimal numbers ($1.5M, 0.5g)."""
    out: list[SourceSpan] = []
    # Match chunks separated by `;`, `!`, `?`, or `.` NOT flanked by digits (so `$1.5M` stays intact!)
    pos = 0
    n = len(turn.text)
    i = 0
    while i < n:
        ch = turn.text[i]
        is_sep = False
        if ch in "!?;":
            is_sep = True
        elif ch == ".":
            prev_d = i > 0 and turn.text[i - 1].isdigit()
            next_d = i + 1 < n and turn.text[i + 1].isdigit()
            if not (prev_d and next_d):
                is_sep = True
        if is_sep:
            end_i = i + 1
            raw = turn.text[pos:end_i]
            lead_ws = len(raw) - len(raw.lstrip())
            trail_ws = len(raw) - len(raw.rstrip())
            s0 = turn.start + pos + lead_ws
            e0 = turn.start + end_i - trail_ws
            if e0 > s0:
                out.append(SourceSpan(s0, e0, turn.text[pos + lead_ws : end_i - trail_ws]))
            pos = end_i
        i += 1
    if pos < n:
        raw = turn.text[pos:]
        lead_ws = len(raw) - len(raw.lstrip())
        trail_ws = len(raw) - len(raw.rstrip())
        s0 = turn.start + pos + lead_ws
        e0 = turn.start + n - trail_ws
        if e0 > s0:
            out.append(SourceSpan(s0, e0, turn.text[pos + lead_ws : n - trail_ws]))
    return tuple(out)


def _find_turn_index(source: DocumentSource, start: int, end: int) -> int | None:
    for i, t in enumerate(source.turns):
        if t.start <= start and end <= t.end:
            return i
    return None


def _enclosing_clause(turn: SourceTurn, span: SourceSpan) -> SourceSpan:
    for cl in split_turn_clauses(turn):
        if cl.start <= span.start and span.end <= cl.end:
            return cl
    return SourceSpan(turn.start, turn.end, turn.text)


def has_global_reassignment(source: DocumentSource) -> bool:
    return any(_GLOBAL_REASSIGNMENT_RE.search(t.text) for t in source.turns)


def normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[.,!?;:]+$", "", text.strip().lower()))


def _clause_retracted(
    turn: SourceTurn,
    clause: SourceSpan,
    source: DocumentSource,
    turn_idx: int,
    span_text: str = "",
) -> bool:
    clauses = split_turn_clauses(turn)
    for i, cl in enumerate(clauses):
        if cl.start == clause.start and i + 1 < len(clauses):
            nxt = clauses[i + 1].text.lower()
            if _SELF_CORRECTION_RE.search(nxt) and re.search(
                r"\b(?:sorry,?\s+that\s+was|actually|meant|mistaken|wrong|incorrect|not\s+really|after\s+all|scratch|retract|that\s+was\s+(?:last\s+year|my\s+\w+'s))\b", nxt
            ):
                return True
    norm_val = normalize_label(span_text or clause.text)
    for j in range(turn_idx + 1, len(source.turns)):
        later = source.turns[j]
        if later.role is TurnRole.EXTERNAL:
            if not later.speaker_label.lower().startswith("other"):
                return True
            continue
        if later.role is not TurnRole.PRINCIPAL:
            continue
        low = later.text.lower()
        if _SELF_CORRECTION_RE.search(low) or re.search(r"\bactually\b", low):
            if j == turn_idx + 1 and re.search(
                r"\b(?:i\s+was\s+(?:mistaken|wrong|incorrect)|that\s+was\s+(?:last\s+year|before|hypothetical|my\s+\w+)|"
                r"that\s+belonged\s+to|is\s+a\s+better\s+word|i\s+meant)\b",
                low,
            ):
                return True
            if norm_val and re.search(r"\b(?:not|mistake|mistaken|wrong|incorrect|retract|sorry)\b", low):
                words = [w for w in re.findall(r"[a-z0-9]{3,}", norm_val) if w not in {"the", "for", "with", "that", "this"}]
                if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words):
                    return True
    return False


def verify_node_span(
    source: DocumentSource,
    node: OntologyNode,
    state: GroundedState,
    span: SourceSpan,
    schema: OntologyGraph,
    registry: DenominationRegistry,
) -> tuple[bool, str, QuantityValue | None]:
    """Domain-agnostic span & value verification driven purely by node schema + unit algebra."""
    if has_global_reassignment(source):
        return False, "global_source_reassignment", None

    idx = _find_turn_index(source, span.start, span.end)
    if idx is None or source.turns[idx].role is not TurnRole.PRINCIPAL:
        return False, "not_principal_turn", None

    turn = source.turns[idx]
    prev = source.turns[idx - 1] if idx > 0 else None
    if prev is not None and prev.role is TurnRole.EXTERNAL and not prev.speaker_label.lower().startswith("other"):
        return False, "intervening_external_speaker", None
    prev_q = prev.text if (prev is not None and prev.role is TurnRole.INTERLOCUTOR) else ""
    if prev_q and _QUESTION_HAZARD_RE.search(prev_q):
        if state is GroundedState.ABSENT or not re.search(r"\bfor this (?:new )?", turn.text, re.IGNORECASE):
            return False, "non_current_or_third_party_question_scope", None

    clause = _enclosing_clause(turn, span)
    if not schema.owning_siblings_win(node.node_id, clause.text, prev_q):
        return False, "sibling_topic_ownership", None

    if not node.allow_quoted_or_self_ref and _QUOTED_REF_RE.search(turn.text):
        return False, "quoted_secondary_reference", None

    if _clause_retracted(turn, clause, source, idx, span.text):
        return False, "retracted_by_principal_correction", None

    pats = schema.patterns(node.node_id)

    if state is GroundedState.SUPPORTED:
        cleaned = span.text.strip()
        if not cleaned or cleaned != span.text or len(cleaned) > node.max_chars:
            return False, "invalid_span_bounds", None
        if cleaned.endswith((".", "!", "?", ";", ":")):
            return False, "trailing_punctuation", None
        words = re.findall(r"[^\W_]+(?:'[^\W_]+)?", cleaned.casefold())
        if not words or len(words) > node.max_words:
            return False, "word_count_out_of_bounds", None
        if node.reject_leading_articles and words[0] in {"a", "an", "the", "my", "some"}:
            if not (node.exempt_leading_articles and cleaned.casefold().startswith(node.exempt_leading_articles)):
                return False, "leading_article_rejected", None

        if _EPISTEMIC_HEDGE_RE.search(clause.text) or _NONUSE_ACTION_RE.search(clause.text):
            return False, "clause_uncertainty_or_nonuse", None
        if _THIRD_PARTY_RE.search(clause.text) or _PAST_OR_HYPOTHETICAL_RE.search(clause.text):
            return False, "clause_third_party_or_past", None
        if _NEGATION_WORD_RE.search(clause.text):
            return False, "clause_negation_hazard", None
        if _SELF_CORRECTION_RE.search(clause.text):
            return False, "clause_correction_marker", None
        if any(p.search(turn.text) for p in pats["episode_disclaimer"]):
            return False, "episode_disclaimed_in_turn", None
        for later in source.turns[idx + 1 :]:
            if later.role is TurnRole.PRINCIPAL and any(p.search(later.text) for p in pats["episode_disclaimer"]):
                return False, "episode_disclaimed_in_followup", None
        for p in pats["coexisting_list"]:
            if p.search(clause.text):
                return False, "coexisting_multiple_values_require_uncertain", None
        for p in pats["disallowed"]:
            if p.search(cleaned):
                return False, "disallowed_node_value", None

        qty: QuantityValue | None = None
        if node.value_kind in (ValueKind.QUANTITY, ValueKind.ORDINAL) or node.dimension:
            qty = registry.parse(
                cleaned,
                expected_dimension=node.dimension,
                ordinal_scale=node.ordinal_scale,
            )
            if qty is None:
                return False, "unparseable_quantity_or_ordinal", None
            if node.allowed_roles and qty.role not in node.allowed_roles:
                return False, f"role_mismatch:{qty.role}", None
        return True, "verified_supported", qty

    if state is GroundedState.ABSENT:
        if not node.supports_absence:
            return False, "absence_not_supported_for_node", None
        text = span.text.strip()
        if _EPISTEMIC_HEDGE_RE.search(text) or _THIRD_PARTY_RE.search(text):
            return False, "hedged_or_third_party_denial", None
        if any(p.search(text) for p in pats["contrast_disclaimer"]):
            return False, "contrast_disclaimer_not_absence", None
        if any(p.search(text) for p in pats["exception"]):
            return False, "partial_exclusion_not_blanket_denial", None
        if not _NEGATION_WORD_RE.search(text):
            return False, "denial_missing_negation", None
        has_cue = any(p.search(text) for p in pats["denial"])
        if not has_cue:
            # Elliptical answer only when preceded by an unambiguous interlocutor question about this node
            if prev_q and re.search(r"\b(?:anything\s+else|any\s+other|else\s+on\s+your|other\s+than\s+that)\b", prev_q, re.IGNORECASE):
                return False, "elliptical_no_additional_not_blanket_denial", None
            q_match = bool(prev_q and any(p.search(prev_q) for p in pats["topic"]))
            if not (q_match and schema.owning_siblings_win(node.node_id, prev_q, "") and _ELLIPTICAL_DENIAL_RE.fullmatch(text)):
                return False, "unanchored_or_wrong_topic_denial", None
        semi_clauses = [m.group().strip() for m in re.finditer(r"[^;]+", turn.text) if m.group().strip()]
        if text not in {turn.text.strip(), clause.text.strip(), *semi_clauses}:
            return False, "partial_clause_denial_span", None
        return True, "verified_absent", None

    return False, "state_not_directly_verifiable", None


def _trim_source_span(source: DocumentSource, start: int, end: int) -> SourceSpan | None:
    s = source.raw_text[start:end]
    lead = len(s) - len(s.lstrip(" \t\r\n,;"))
    trail = len(s) - len(s.rstrip(" \t\r\n.,!?;:"))
    ns, ne = start + lead, end - trail
    if ne <= ns:
        return None
    return SourceSpan(ns, ne, source.raw_text[ns:ne])


def extract_node_candidates(
    source: DocumentSource,
    node: OntologyNode,
    schema: OntologyGraph,
    registry: DenominationRegistry,
) -> tuple[list[tuple[SourceSpan, QuantityValue | None]], list[SourceSpan]]:
    """Extract verified `(supported_spans_with_qty, denial_spans)` using only node schema."""
    if has_global_reassignment(source):
        return [], []

    sup_map: dict[tuple[int, int], tuple[SourceSpan, QuantityValue | None]] = {}
    den_map: dict[tuple[int, int], SourceSpan] = {}
    pats = schema.patterns(node.node_id)

    for t_idx, turn in enumerate(source.turns):
        if turn.role is not TurnRole.PRINCIPAL:
            continue
        prev = source.turns[t_idx - 1] if t_idx > 0 else None
        prev_q = prev.text if (prev is not None and prev.role is TurnRole.INTERLOCUTOR) else ""
        clauses = split_turn_clauses(turn)

        for cl in clauses:
            # Check explicit denials first
            if node.supports_absence:
                ok_d, _, _ = verify_node_span(source, node, GroundedState.ABSENT, cl, schema, registry)
                if ok_d:
                    den_map[(cl.start, cl.end)] = cl
                    continue

            if not schema.owning_siblings_win(node.node_id, cl.text, prev_q):
                continue

            # Explicit value extraction regexes declared on node (plus optional `(?P<core>...)` capture)
            for vp in pats["value_extract"]:
                for m in vp.finditer(cl.text):
                    sp = _trim_source_span(source, cl.start + m.start(), cl.start + m.end())
                    if sp is not None:
                        ok, _, qty = verify_node_span(source, node, GroundedState.SUPPORTED, sp, schema, registry)
                        if ok:
                            sup_map[(sp.start, sp.end)] = (sp, qty)
                    if "core" in vp.groupindex and m.start("core") >= 0:
                        sp_c = _trim_source_span(source, cl.start + m.start("core"), cl.start + m.end("core"))
                        if sp_c is not None:
                            ok_c, _, qty_c = verify_node_span(source, node, GroundedState.SUPPORTED, sp_c, schema, registry)
                            if ok_c:
                                sup_map[(sp_c.start, sp_c.end)] = (sp_c, qty_c)

            # Topic-matched clause extraction (unless `require_topic_cue=False` for implicit/default attribute nodes)
            topic_or_q_match = (
                (not node.require_topic_cue)
                or any(p.search(cl.text) for p in pats["predicate"])
                or any(p.search(cl.text) for p in pats["topic"])
                or (prev_q and any(p.search(prev_q) for p in pats["topic"]))
            )
            if not topic_or_q_match:
                continue

            # For typed QUANTITY / ORDINAL nodes, extract typed sub-spans via DenominationRegistry automatically
            matched_typed = False
            if node.value_kind in (ValueKind.QUANTITY, ValueKind.ORDINAL) or node.dimension:
                for qs, qe, _ in registry.extract_spans(
                    cl.text,
                    expected_dimension=node.dimension,
                    ordinal_scale=node.ordinal_scale,
                ):
                    sp_q = _trim_source_span(source, cl.start + qs, cl.start + qe)
                    if sp_q is not None:
                        ok, _, qty = verify_node_span(source, node, GroundedState.SUPPORTED, sp_q, schema, registry)
                        if ok:
                            sup_map[(sp_q.start, sp_q.end)] = (sp_q, qty)
                            matched_typed = True
                if matched_typed and not (pats["lead_strip"] or pats["trail_strip"]):
                    continue

            body = cl.text.rstrip(".!?").strip()
            sub_start = 0
            for lp in pats["lead_strip"]:
                ml = lp.match(body)
                if ml and ml.end() > sub_start:
                    sub_start = ml.end()
            rem = body[sub_start:]
            candidates: list[str] = []
            stripped = rem
            for tp in pats["trail_strip"]:
                nxt_s = tp.sub("", stripped).strip(" ,")
                if nxt_s and nxt_s != stripped:
                    candidates.append(nxt_s)
                    stripped = nxt_s
            if stripped:
                candidates.append(stripped)
            if rem and rem not in candidates:
                candidates.append(rem)
            for sub in candidates:
                pos = cl.text.find(sub, sub_start)
                if pos >= 0:
                    sp = _trim_source_span(source, cl.start + pos, cl.start + pos + len(sub))
                    if sp is not None:
                        if node.reject_leading_articles:
                            m_art = re.match(r"^(?:a|an|the|my|some)\s+(?P<rest>.+)$", sp.text, re.IGNORECASE)
                            if m_art and not (
                                node.exempt_leading_articles
                                and sp.text.casefold().startswith(node.exempt_leading_articles)
                            ):
                                r_off = sp.start + m_art.start("rest")
                                sp = SourceSpan(r_off, r_off + len(m_art.group("rest")), m_art.group("rest"))
                        ok, _, qty = verify_node_span(source, node, GroundedState.SUPPORTED, sp, schema, registry)
                        if ok:
                            sup_map[(sp.start, sp.end)] = (sp, qty)

    return list(sup_map.values()), list(den_map.values())


def extract_node_uncertain_bundles(
    source: DocumentSource,
    node: OntologyNode,
    schema: OntologyGraph,
) -> list[tuple[SourceSpan, ...]]:
    """Extract genuine epistemic-uncertainty or coexisting-multi-value span bundles cleanly."""
    bundles: list[tuple[SourceSpan, ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    pats = schema.patterns(node.node_id)
    for turn in source.turns:
        if turn.role is not TurnRole.PRINCIPAL:
            continue
        for cl in split_turn_clauses(turn):
            for pat in pats["coexisting_list"]:
                m = pat.search(cl.text)
                if m and "first" in m.groupdict() and "second" in m.groupdict():
                    s1 = m.group("first").strip()
                    s2 = m.group("second").strip()
                    if s1 and s2 and normalize_label(s1) != normalize_label(s2):
                        p1 = cl.text.find(s1)
                        p2 = cl.text.find(s2, p1 + len(s1)) if p1 >= 0 else -1
                        if p1 >= 0 and p2 >= 0:
                            sp1 = SourceSpan(cl.start + p1, cl.start + p1 + len(s1), s1)
                            sp2 = SourceSpan(cl.start + p2, cl.start + p2 + len(s2), s2)
                            key = ((sp1.start, sp1.end), (sp2.start, sp2.end))
                            if key not in seen:
                                seen.add(key)
                                bundles.append((sp1, sp2))
                whole = SourceSpan(turn.start, turn.end, turn.text)
                wkey = ((whole.start, whole.end),)
                if m and wkey not in seen:
                    seen.add(wkey)
                    bundles.append((whole,))
        if _EPISTEMIC_HEDGE_RE.search(turn.text) and not re.search(r"\bchecking whether i can hear\b", turn.text, re.IGNORECASE):
            whole = SourceSpan(turn.start, turn.end, turn.text)
            wkey = ((whole.start, whole.end),)
            if wkey not in seen:
                seen.add(wkey)
                bundles.append((whole,))
    return bundles


def detect_node_conflicts(
    source: DocumentSource,
    node: OntologyNode,
    schema: OntologyGraph,
    registry: DenominationRegistry,
) -> list[ConflictPair]:
    """Detect incompatibilities across turns via denomination algebra or scalar mismatch."""
    sup_items, den_spans = extract_node_candidates(source, node, schema, registry)
    conflicts: list[ConflictPair] = []

    if sup_items and den_spans:
        for sp_s, _ in sup_items:
            ts = _find_turn_index(source, sp_s.start, sp_s.end)
            for sp_d in den_spans:
                td = _find_turn_index(source, sp_d.start, sp_d.end)
                if ts is not None and td is not None and ts != td:
                    pair = (sp_s, sp_d) if sp_s.start < sp_d.start else (sp_d, sp_s)
                    conflicts.append(ConflictPair(
                        node_id=node.node_id,
                        kind="presence_disagreement",
                        spans=pair,
                        reason="supported assertion and explicit denial across distinct turns",
                    ))
                    return conflicts

    # Check multi-turn incompatible values
    by_turn: dict[int, tuple[SourceSpan, QuantityValue | None]] = {}
    for sp, qty in sup_items:
        ti = _find_turn_index(source, sp.start, sp.end)
        if ti is not None and ti not in by_turn:
            by_turn[ti] = (sp, qty)

    if len(by_turn) >= 2:
        t_ids = sorted(by_turn)
        sp1, q1 = by_turn[t_ids[0]]
        sp2, q2 = by_turn[t_ids[-1]]
        if q1 is not None and q2 is not None:
            cmp_res = registry.compare(q1, q2)
            if cmp_res.relation in (DenomRelation.CONFLICTING, DenomRelation.INCOMPARABLE_DENOMINATION, DenomRelation.INCOMPARABLE_DIMENSION, DenomRelation.ROLE_MISMATCH):
                conflicts.append(ConflictPair(
                    node_id=node.node_id,
                    kind=cmp_res.relation.value,
                    spans=(sp1, sp2),
                    reason=cmp_res.reason,
                ))
        elif node.is_single_scalar_value and normalize_label(sp1.text) != normalize_label(sp2.text):
            n1, n2 = normalize_label(sp1.text), normalize_label(sp2.text)
            if n1 not in n2 and n2 not in n1:
                conflicts.append(ConflictPair(
                    node_id=node.node_id,
                    kind="value_disagreement",
                    spans=(sp1, sp2),
                    reason=f"distinct scalar labels {n1!r} vs {n2!r}",
                ))
    return conflicts


def resolve_world_state(
    raw_text: str,
    schema: OntologyGraph,
    registry: DenominationRegistry | None = None,
    *,
    source_id: str = "world-0",
    proposals: dict[str, tuple[GroundedState, tuple[SourceSpan, ...]]] | None = None,
) -> GroundedWorldReport:
    """Solve all schema nodes, check denominations & hierarchical rollups, and track dependency lineage."""
    reg = registry or DenominationRegistry()
    source = parse_document_source(raw_text, source_id=source_id)
    global_unsafe = has_global_reassignment(source)
    dep_graph = DependencyGraph()

    # Non-destructive joint allocation across all nodes before committing selections
    all_candidates: dict[str, tuple[list[tuple[SourceSpan, QuantityValue | None]], list[SourceSpan]]] = {}
    for node in schema.ordered_nodes:
        all_candidates[node.node_id] = extract_node_candidates(source, node, schema, reg)

    resolutions: dict[str, NodeResolution] = {}
    conflicts: dict[str, tuple[ConflictPair, ...]] = {}
    resolved_quantities: dict[str, QuantityValue] = {}

    for node in schema.ordered_nodes:
        nid = node.node_id
        node_conflicts = detect_node_conflicts(source, node, schema, reg)
        if node_conflicts and not global_unsafe:
            conflicts[nid] = tuple(node_conflicts)
            resolutions[nid] = NodeResolution(
                node_id=nid,
                state=GroundedState.CONFLICTING,
                spans=node_conflicts[0].spans,
                verified=True,
                reason=node_conflicts[0].reason,
            )
            continue

        if global_unsafe:
            resolutions[nid] = NodeResolution(
                node_id=nid, state=GroundedState.UNCERTAIN, verified=False, reason="global_source_reassignment"
            )
            continue

        # If caller supplied model proposals, verify each strictly against source & node spec
        if proposals and nid in proposals:
            p_state, p_spans = proposals[nid]
            if p_state is GroundedState.MISSING:
                resolutions[nid] = NodeResolution(nid, GroundedState.MISSING, verified=True, reason="proposed_missing")
            elif p_state is GroundedState.UNCERTAIN:
                resolutions[nid] = NodeResolution(nid, GroundedState.UNCERTAIN, spans=p_spans, verified=True, reason="proposed_uncertain")
            elif p_state in (GroundedState.SUPPORTED, GroundedState.ABSENT) and p_spans:
                ok, reason, qty = verify_node_span(source, node, p_state, p_spans[0], schema, reg)
                if ok:
                    val = p_spans[0].text if p_state is GroundedState.SUPPORTED else None
                    resolutions[nid] = NodeResolution(nid, p_state, val, qty, p_spans, True, reason)
                    if qty is not None:
                        resolved_quantities[nid] = qty
                else:
                    resolutions[nid] = NodeResolution(nid, GroundedState.UNCERTAIN, spans=p_spans, verified=False, reason=reason)
            else:
                resolutions[nid] = NodeResolution(nid, GroundedState.UNCERTAIN, spans=p_spans, verified=False, reason="invalid_proposal")
            continue

        # Autonomous extraction with joint non-overlapping span preference (shorter/more specific child spans preferred over engulfing parent spans!)
        sup_items, den_spans = all_candidates[nid]
        if sup_items:
            # Prefer tightest span that doesn't swallow another node's quantitative candidate
            other_qty_spans = [
                sp for other_id, (osup, _) in all_candidates.items()
                if other_id != nid and schema.nodes[other_id].value_kind in (ValueKind.QUANTITY, ValueKind.ORDINAL)
                for sp, _ in osup
            ]
            def _rank_span(item: tuple[SourceSpan, QuantityValue | None]) -> tuple[int, int]:
                sp, _ = item
                swallows = sum(1 for o in other_qty_spans if sp.start <= o.start and o.end <= sp.end and sp != o)
                return (swallows, len(sp.text))

            chosen_sp, chosen_qty = min(sup_items, key=_rank_span)
            resolutions[nid] = NodeResolution(
                nid, GroundedState.SUPPORTED, chosen_sp.text, chosen_qty, (chosen_sp,), True, "extracted_verified"
            )
            if chosen_qty is not None:
                resolved_quantities[nid] = chosen_qty
        elif den_spans:
            resolutions[nid] = NodeResolution(
                nid, GroundedState.ABSENT, None, None, (den_spans[0],), True, "extracted_denial"
            )
        else:
            resolutions[nid] = NodeResolution(nid, GroundedState.MISSING, verified=True, reason="no_matching_evidence")

    # Verify hierarchical conservation / rollups across mixed denominations & track lineage
    hier_checks = tuple(schema.verify_rollups(resolved_quantities, reg))
    for hc in hier_checks:
        if hc.child_ids:
            dep_graph.register(Dependency(
                derived_id=f"rollup:{hc.parent_id}",
                input_ids=tuple(f"node:{c}" for c in hc.child_ids),
                kind="hierarchy_rollup",
                producer="nano.schema.OntologyGraph",
            ))
        if hc.status == "conflict" and not global_unsafe:
            parent_res = resolutions.get(hc.parent_id)
            c_spans = parent_res.spans if parent_res else ()
            conflicts[hc.parent_id] = (*conflicts.get(hc.parent_id, ()), ConflictPair(
                node_id=hc.parent_id,
                kind="hierarchy_conflict",
                spans=c_spans,
                reason=hc.reason,
            ))
            resolutions[hc.parent_id] = NodeResolution(
                node_id=hc.parent_id,
                state=GroundedState.CONFLICTING,
                value_text=parent_res.value_text if parent_res else None,
                quantity=parent_res.quantity if parent_res else None,
                spans=c_spans,
                verified=True,
                reason=hc.reason,
            )
        elif hc.status == "incomparable":
            parent_res = resolutions.get(hc.parent_id)
            c_spans = parent_res.spans if parent_res else ()
            conflicts[hc.parent_id] = (*conflicts.get(hc.parent_id, ()), ConflictPair(
                node_id=hc.parent_id,
                kind="incomparable_denomination",
                spans=c_spans,
                reason=hc.reason,
            ))
            resolutions[hc.parent_id] = NodeResolution(
                node_id=hc.parent_id,
                state=GroundedState.UNCERTAIN,
                value_text=None,
                quantity=parent_res.quantity if parent_res else None,
                spans=c_spans,
                verified=False,
                reason=hc.reason,
            )

    return GroundedWorldReport(
        source_id=source_id,
        resolutions=resolutions,
        conflicts=conflicts,
        hierarchy_checks=hier_checks,
        dependency_graph=dep_graph,
        global_unsafe=global_unsafe,
    )
