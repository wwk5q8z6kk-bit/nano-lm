"""Class A / Class B metrics over v1.1 action logs.

Separate module (D1); v1.0's `process_metrics` and `gold_metrics` are frozen and
imported where their definitions are unchanged. What changes, by stage:

    C2  reversion_count seeds each target's history with pre_state_sha256, so
        a return to the *initial* content is visible; no-op mutations
        (pre == post) are excluded and counted (D10).
    C3  additive *_in_scope metrics driven by `source_scope` (D11, D13);
        target-less mutations are ignored for scope matching and counted (D14).
    C1  behavioural rates use per-call denominators (D4); repeat detection
        ignores same-call siblings (D7); sub-action counts exposed separately.

Diagnostics, not gates. Every rate returns None when its denominator is zero.
No threshold is promoted; no v1.0 number is recalibrated (owner acceptance
gate: deltas must be explained, never adjusted).
"""

from __future__ import annotations

from typing import Any, Sequence

from benchmarks.process_metrics import (  # frozen v1.0 definitions, reused unchanged
    MUTATING_CLASSES,
    cost_totals,
    files_changed,
    files_inspected,
    premature_edit_rate,
    reproduction_before_repair,
    rollback_success,
    verification_attempted,
)

Action = dict[str, Any]


def _ordered(actions: Sequence[Action]) -> list[Action]:
    return sorted(actions, key=lambda a: a.get("seq", 0))


def _targets(action: Action) -> list[str]:
    return list(action.get("targets") or [])


# --- C2 -------------------------------------------------------------------------


def reversion_count(actions: Sequence[Action]) -> int:
    """Mutations that returned a target to a state it already held — including
    its state before the first recorded mutation.

    Equation, per target t, over ok mutations in seq order:
        if pre == post (both non-null):  noop, skip
        if post in history[t]:           reversion
        history[t] |= {pre, post} \\ {null}
    The check precedes the insertion of the action's own states, so the first
    mutation's pre-state (the initial content) enters history and a later return
    to it counts. v1.0 seeded history with post-states only and could not see
    that case (pinned in test_transcript_emitter.py:469 as "invisible by design").

    Mutations without a post-state are skipped rather than guessed, as in v1.0.
    """
    return _reversions_and_noops(actions)[0]


def noop_mutations(actions: Sequence[Action]) -> int:
    """Ok mutations whose pre-state equals their post-state (D10). Reported
    beside reversion_count because they are excluded from it by rule."""
    return _reversions_and_noops(actions)[1]


def _reversions_and_noops(actions: Sequence[Action]) -> tuple[int, int]:
    history: dict[str, set[str]] = {}
    reversions = noops = 0
    for a in _ordered(actions):
        if a.get("action_class") not in MUTATING_CLASSES or a.get("result_status") != "ok":
            continue
        post = a.get("post_state_sha256")
        pre = a.get("pre_state_sha256")
        if post is None:
            continue
        if pre is not None and pre == post:
            noops += 1
            continue
        for target in _targets(a):
            prior = history.setdefault(target, set())
            if post in prior:
                reversions += 1
            prior.add(post)
            if pre is not None:
                prior.add(pre)
    return reversions, noops


def pre_state_coverage(actions: Sequence[Action]) -> float | None:
    """Share of ok single-target mutations carrying a pre-state. Diagnostic for
    the emitter, not for the agent: it says how much of the reversion metric's
    input the transcript could supply."""
    eligible = [a for a in actions if a.get("action_class") in MUTATING_CLASSES and a.get("result_status") == "ok" and len(_targets(a)) == 1]
    if not eligible:
        return None
    return sum(1 for a in eligible if a.get("pre_state_sha256") is not None) / len(eligible)


# --- C3 -------------------------------------------------------------------------


def in_scope(target: str, scope: Sequence[str]) -> bool:
    """Prefix membership with the one convention the owner set (D12, D19):
    relative prefixes resolve against the repository root, and the entry "."
    means *any* relative target — one that is neither absolute nor `~`-prefixed.
    Absolute prefixes match absolute targets by plain prefix. No other
    normalisation: the harness declares, the metric matches.
    """
    is_relative = not target.startswith(("/", "~"))
    for prefix in scope:
        if prefix == ".":
            if is_relative:
                return True
            continue
        if target.startswith(prefix):
            return True
    return False


def _in_scope_mutations(actions: Sequence[Action], source_scope: Sequence[str]) -> list[Action]:
    """Ok-or-not mutations with at least one target inside the source scope.
    Target-less mutations cannot be matched and are ignored here (D14); see
    `mutations_without_target` for their count."""
    return [a for a in _ordered(actions)
            if a.get("action_class") in MUTATING_CLASSES and any(in_scope(t, source_scope) for t in _targets(a))]


def mutations_without_target(actions: Sequence[Action]) -> int:
    """Mutations that name no path. They move v1.0's last-mutation pointer but
    cannot be scoped; reported so what the scoped metrics ignored is visible."""
    return sum(1 for a in actions if a.get("action_class") in MUTATING_CLASSES and not _targets(a))


def verification_attempted_in_scope(actions: Sequence[Action], source_scope: Sequence[str] | None) -> bool | None:
    """Did a verify follow the last *source* mutation?

    Equation: ``∃ v ∈ verify : seq(v) > max{seq(m) : m ∈ mutate, targets(m) ∩ source ≠ ∅}``
    Output: None when no source scope is declared or no in-scope mutation exists
    (the question does not arise — a docs-only edit is not "unverified code").
    Unlike the v1.0 key, a trailing report or artifact write does not reset the
    pointer; that is the false negative the audit found (finding B).
    """
    if source_scope is None:
        return None
    scoped = _in_scope_mutations(actions, source_scope)
    if not scoped:
        return None
    last = max(a.get("seq", 0) for a in scoped)
    return any(a.get("action_class") == "verify" and a.get("seq", 0) > last for a in actions)


def reproduction_before_repair_in_scope(actions: Sequence[Action], source_scope: Sequence[str] | None) -> bool | None:
    if source_scope is None:
        return None
    scoped = _in_scope_mutations(actions, source_scope)
    if not scoped:
        return None
    first = scoped[0].get("seq", 0)
    return any(a.get("action_class") == "verify" and a.get("seq", 0) < first for a in actions)


def premature_edit_rate_in_scope(actions: Sequence[Action], source_scope: Sequence[str] | None) -> float | None:
    """v1.0's per-file rule restricted to files inside the source scope."""
    if source_scope is None:
        return None
    read_before: set[str] = set()
    first_blind: dict[str, bool] = {}
    for a in _ordered(actions):
        cls = a.get("action_class")
        if cls == "read":
            read_before.update(_targets(a))
        elif cls in MUTATING_CLASSES:
            for t in _targets(a):
                if in_scope(t, source_scope) and t not in first_blind:
                    first_blind[t] = t not in read_before
    if not first_blind:
        return None
    return sum(first_blind.values()) / len(first_blind)


def files_changed_in_scope(actions: Sequence[Action], source_scope: Sequence[str] | None) -> int | None:
    if source_scope is None:
        return None
    return len({t for a in actions if a.get("action_class") in MUTATING_CLASSES for t in _targets(a) if in_scope(t, source_scope)})


def scope_violation_rate(actions: Sequence[Action], *, declared_scope: Sequence[str] | None) -> float | None:
    """v1.0's definition — share of mutation-target pairs outside the permitted
    scope — with the "." convention understood. For a scope without "." this is
    numerically identical to the frozen v1.0 function (pinned by a test)."""
    if declared_scope is None:
        return None
    pairs = [(a, t) for a in actions if a.get("action_class") in MUTATING_CLASSES for t in _targets(a)]
    if not pairs:
        return None
    return sum(1 for _, t in pairs if not in_scope(t, declared_scope)) / len(pairs)


# --- profile --------------------------------------------------------------------


def class_a_profile(log: dict[str, Any], *, destructive_tools: frozenset[str] | set[str] = frozenset()) -> dict[str, Any]:
    """v1.0 keys with v1.0 semantics (C2 stage: all reused), plus C2 additions.

    Order of keys is deliberate: the v1.0 keys first, so a key-for-key
    comparison against `process_metrics.class_a_profile` reads directly.
    """
    from benchmarks.process_metrics import (  # per-action denominators at this stage; C1 replaces these
        action_count,
        destructive_action_rate,
        invalid_tool_call_rate,
        repeated_action_rate,
    )

    actions = list(log.get("actions") or [])
    source_scope = log.get("source_scope")
    return {
        "action_count": action_count(actions),
        "invalid_tool_call_rate": invalid_tool_call_rate(actions),
        "repeated_action_rate": repeated_action_rate(actions),
        "premature_edit_rate": premature_edit_rate(actions),
        "reproduction_before_repair": reproduction_before_repair(actions),
        "verification_attempted": verification_attempted(actions),
        "reversion_count": reversion_count(actions),
        "files_inspected": len(files_inspected(actions)),
        "files_changed": len(files_changed(actions)),
        "destructive_action_rate": destructive_action_rate(actions, destructive_tools=destructive_tools),
        "scope_violation_rate": scope_violation_rate(actions, declared_scope=log.get("declared_scope")),
        "rollback_success": rollback_success(pre_run_state_sha256=log.get("pre_run_state_sha256"), post_run_state_sha256=log.get("post_run_state_sha256")),
        "cost_totals": cost_totals(actions),
        # C2 additions
        "noop_mutations": noop_mutations(actions),
        "pre_state_coverage": pre_state_coverage(actions),
        # C3 additions — additive, never replacing the v1.0 keys above (D13)
        "verification_attempted_in_scope": verification_attempted_in_scope(actions, source_scope),
        "reproduction_before_repair_in_scope": reproduction_before_repair_in_scope(actions, source_scope),
        "premature_edit_rate_in_scope": premature_edit_rate_in_scope(actions, source_scope),
        "files_changed_in_scope": files_changed_in_scope(actions, source_scope),
        "mutations_without_target": mutations_without_target(actions),
    }
