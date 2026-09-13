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


# --- C1 -------------------------------------------------------------------------

INVALID_STATUSES = frozenset({"tool_error"})


def _calls(actions: Sequence[Action]) -> list[list[Action]]:
    """Sub-actions grouped by call, in seq order. Siblings are consecutive by
    contract, so a call is a maximal run of equal call_id."""
    groups: list[list[Action]] = []
    for a in _ordered(actions):
        if groups and groups[-1][0].get("call_id") == a.get("call_id"):
            groups[-1].append(a)
        else:
            groups.append([a])
    return groups


def action_count(actions: Sequence[Action]) -> int:
    """Sub-actions. Exposed separately from call_count (D4)."""
    return len(actions)


def call_count(actions: Sequence[Action]) -> int:
    """Tool calls the agent made — the v1.0 notion of an action."""
    return len(_calls(actions))


def invalid_tool_call_rate(actions: Sequence[Action]) -> float | None:
    """Calls the harness rejected as malformed ÷ calls (D4). Siblings share the
    call's status, so a rejected call counts once however many effects it had."""
    calls = _calls(actions)
    if not calls:
        return None
    return sum(1 for c in calls if c[0].get("result_status") in INVALID_STATUSES) / len(calls)


def destructive_action_rate(actions: Sequence[Action], *, destructive_tools: frozenset[str] | set[str]) -> float | None:
    calls = _calls(actions)
    if not calls:
        return None
    return sum(1 for c in calls if c[0].get("tool") in destructive_tools) / len(calls)


def repeated_action_rate(actions: Sequence[Action]) -> float | None:
    """Calls that repeat an earlier *call* with unchanged result ÷ calls (D4, D7).

    Equation: a call is a repeat iff an earlier call with a DIFFERENT call_id had
    the same (tool, args_sha256) and the same non-null result_sha256. Siblings
    share all three by contract and must never count against each other.
    Because siblings also share every field the rule reads, this equals v1.0's
    rule applied to the sequence of first siblings — the migration module checks
    that identity mechanically.
    """
    calls = _calls(actions)
    if not calls:
        return None
    seen: dict[tuple[str, str], tuple[str | None, str]] = {}
    repeats = 0
    for c in calls:
        first = c[0]
        key = (first.get("tool", ""), first.get("args_sha256", ""))
        result = first.get("result_sha256")
        if key in seen and result is not None and seen[key][0] == result and seen[key][1] != first.get("call_id"):
            repeats += 1
        seen[key] = (result, str(first.get("call_id")))
    return repeats / len(calls)


def effects_per_call(actions: Sequence[Action]) -> dict[int, int]:
    """Histogram: how many calls decomposed into 1, 2, 3 … sub-actions."""
    out: dict[int, int] = {}
    for c in _calls(actions):
        out[len(c)] = out.get(len(c), 0) + 1
    return dict(sorted(out.items()))


# --- Class B (v1.1) --------------------------------------------------------------


def calls_to_first_correct_file(actions: Sequence[Action], gold_patch_files: frozenset[str]) -> int | None:
    """v1.0's actions_to_first_correct_file with a per-call index (D4)."""
    if not gold_patch_files:
        return None
    for index, c in enumerate(_calls(actions)):
        for a in c:
            if a.get("action_class") in MUTATING_CLASSES and set(_targets(a)) & gold_patch_files:
                return index
    return None


def class_b_profile(log: dict[str, Any], gold: Any, outcome: Any) -> dict[str, Any]:
    """v1.0 Class B keys reused from the frozen module (they read targets,
    tests_run and grader results, all preserved on sub-actions), plus the
    per-call index."""
    from benchmarks.gold_metrics import class_b_profile as class_b_v10

    actions = list(log.get("actions") or [])
    profile = class_b_v10(log, gold, outcome)
    profile["calls_to_first_correct_file"] = calls_to_first_correct_file(actions, gold.patch_files)
    return profile


# --- profile --------------------------------------------------------------------


def class_a_profile(log: dict[str, Any], *, destructive_tools: frozenset[str] | set[str] = frozenset()) -> dict[str, Any]:
    """v1.0 keys first, in v1.0 order, so a key-for-key comparison against
    `process_metrics.class_a_profile` reads directly; then the v1.1 additions.

    v1.0 keys whose definition v1.1 changes (D4, D7): action_count (now
    sub-actions), invalid_tool_call_rate, repeated_action_rate,
    destructive_action_rate (now per call). On a log with no compound calls
    every one of them equals its v1.0 value — spec T3, pinned by a test.
    """
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
        # C1 additions
        "call_count": call_count(actions),
        "effects_per_call": effects_per_call(actions),
    }
