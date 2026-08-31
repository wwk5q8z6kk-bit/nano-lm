"""Class A engineering-process metrics — computable from an agent action log alone.

Scope
-----
These are the Tier 4 / Class A metrics of
`papers/SWE_CAPABILITY_EVALUATION_PROPOSAL.md` §4.1: everything derivable from an
agent action log with **no gold labels and no annotation**. Class B (needs gold
patches and fail-to-pass tests), Class C (needs expert annotation), and Class D
(needs a protocol change so the agent emits a probability) are deliberately not
implemented here.

These are **diagnostics, not gates.** No threshold is promoted. Promotion to a
gate requires the criteria in `papers/SWE_CAPABILITY_EVALUATION_PROPOSAL.md` §5.3
— sufficient n, tested false-positive and false-negative modes, documented
baseline, justified threshold, recheck command. None of that exists yet because
no corpus has been run.

Undefined versus zero
---------------------
Every rate returns ``None`` when its denominator is zero. A run that mutated no
file has an *undefined* premature-edit rate, not a perfect one; collapsing the
two would let an agent that did nothing outscore one that did the work. Counts
return ``0`` because zero occurrences is a real observation.

Input contract: `benchmarks/AGENT_ACTION_LOG.schema.json`. Records are plain
dicts; this module does not import the schema, so it stays usable on partial logs
during harness development. Validate separately when validation is what you want.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

Action = dict[str, Any]

MUTATING_CLASSES = frozenset({"mutate"})
READING_CLASSES = frozenset({"read"})
VERIFYING_CLASSES = frozenset({"verify"})
ERROR_STATUSES = frozenset({"error", "tool_error", "timeout"})
INVALID_STATUSES = frozenset({"tool_error"})


def _ordered(actions: Iterable[Action]) -> list[Action]:
    """Sort by ``seq``. Ordering is the substrate for every sequence metric."""
    return sorted(actions, key=lambda a: a.get("seq", 0))


def _targets(action: Action) -> list[str]:
    return list(action.get("targets") or [])


def action_count(actions: Sequence[Action]) -> int:
    """Total actions.

    Equation: ``|A|``. Output: integer ≥ 0. Interpretation: raw effort, not
    efficiency — comparable only at fixed task difficulty.
    """
    return len(actions)


def invalid_tool_call_rate(actions: Sequence[Action]) -> float | None:
    """Fraction of actions the harness rejected as malformed.

    Equation: ``|{a : result_status = tool_error}| / |A|``
    Output: [0, 1], or None when ``|A| = 0``.
    Why this form: separates *schema/parse* failures (the agent misused the tool)
    from ``error`` (the tool ran and the world said no), which is a competence
    signal rather than a task signal.
    False positives: a harness that reports genuine tool outages as ``tool_error``.
    False negatives: a harness that coerces malformed args instead of rejecting.
    """
    if not actions:
        return None
    invalid = sum(1 for a in actions if a.get("result_status") in INVALID_STATUSES)
    return invalid / len(actions)


def repeated_action_rate(actions: Sequence[Action]) -> float | None:
    """Fraction of actions that repeat an earlier action whose result was unchanged.

    Equation: ``|{a : ∃ a' < a, (tool, args_sha256)_a = (tool, args_sha256)_a'
                 ∧ result_sha256_a = result_sha256_a' ≠ null}| / |A|``
    Output: [0, 1], or None when ``|A| = 0``.

    Why the unchanged-result condition: re-running a test after an edit is
    correct behavior, not a loop. Only a repeat that returns the *same* result is
    evidence the agent is not learning from its own actions.

    Why args_sha256 rather than raw args: raw dicts compare unequal on key order,
    path spelling, and whitespace. Normalization is the emitter's obligation (see
    the schema); comparing digests here keeps this function from silently
    depending on an unstated normalization.

    False negatives: an action repeated against a nondeterministic tool whose
    result digest differs each call is not counted, by construction — the metric
    cannot distinguish it from a genuine retry against new information.
    False positives: an idempotent read the agent legitimately re-issues after a
    long gap; this metric does not model forgetting.
    """
    if not actions:
        return None
    seen: dict[tuple[str, str], str | None] = {}
    repeats = 0
    for a in _ordered(actions):
        key = (a.get("tool", ""), a.get("args_sha256", ""))
        result = a.get("result_sha256")
        if key in seen and result is not None and seen[key] == result:
            repeats += 1
        seen[key] = result
    return repeats / len(actions)


def premature_edit_rate(actions: Sequence[Action]) -> float | None:
    """Fraction of mutated files whose first mutation was not preceded by a read.

    Equation: ``|{f ∈ M : ¬∃ read(f) before first_mutation(f)}| / |M|``
    where ``M`` is the set of files mutated.
    Output: [0, 1], or None when ``|M| = 0``.

    Why per-file rather than per-run: a run that reads and edits ``a.py``, then
    blind-edits ``b.py``, is premature on ``b.py``. A first-action-only run-level
    definition scores that run clean, which is the wrong answer and the reason
    this definition is stated explicitly rather than inherited from prose.

    False positives: a file whose content arrived through a tool the harness did
    not classify as ``read`` (e.g. a grep that returned the full body).
    False negatives: a token read — opening a file and ignoring it counts as
    evidence gathering here. This metric measures sequence, not comprehension.
    """
    ordered = _ordered(actions)
    read_before: set[str] = set()
    first_mutation_blind: dict[str, bool] = {}
    for a in ordered:
        cls = a.get("action_class")
        if cls in READING_CLASSES:
            read_before.update(_targets(a))
        elif cls in MUTATING_CLASSES:
            for target in _targets(a):
                if target not in first_mutation_blind:
                    first_mutation_blind[target] = target not in read_before
    if not first_mutation_blind:
        return None
    return sum(first_mutation_blind.values()) / len(first_mutation_blind)


def reproduction_before_repair(actions: Sequence[Action]) -> bool | None:
    """Did a verification action precede the first mutation?

    Equation: ``∃ v ∈ verify : seq(v) < min{seq(m) : m ∈ mutate}``
    Output: bool, or None when the run mutated nothing (the question does not arise).

    Interpretation: the cheapest available proxy for "reproduced the failure
    before trying to fix it". It cannot tell whether the verification actually
    reproduced the reported problem — that needs gold tests and is Class B.
    """
    ordered = _ordered(actions)
    mutations = [a for a in ordered if a.get("action_class") in MUTATING_CLASSES]
    if not mutations:
        return None
    first_mutation_seq = mutations[0].get("seq", 0)
    return any(
        a.get("action_class") in VERIFYING_CLASSES and a.get("seq", 0) < first_mutation_seq
        for a in ordered
    )


def verification_attempted(actions: Sequence[Action]) -> bool | None:
    """Did any verification action follow the last mutation?

    Equation: ``∃ v ∈ verify : seq(v) > max{seq(m) : m ∈ mutate}``
    Output: bool, or None when the run mutated nothing.

    This is the highest-value member of Class A: it targets the failure mode of
    an apparently valid patch shipped without ever being run. It is structural
    and cheap, and it says nothing about whether the verification was *sufficient*
    — that is Class B (affected-test recall) and Class C (adversarial-test
    effectiveness).

    False positives: a harness that labels a trivial action ``verify``. The
    classification is declared at emission and is not checkable from the log.
    """
    ordered = _ordered(actions)
    mutations = [a for a in ordered if a.get("action_class") in MUTATING_CLASSES]
    if not mutations:
        return None
    last_mutation_seq = max(a.get("seq", 0) for a in mutations)
    return any(
        a.get("action_class") in VERIFYING_CLASSES and a.get("seq", 0) > last_mutation_seq
        for a in ordered
    )


def reversion_count(actions: Sequence[Action]) -> int:
    """Mutations that restored a target to a state it already held.

    Equation: ``|{m ∈ mutate : post_state_sha256(m) ∈ prior_states(target(m))}|``
    Output: integer ≥ 0.

    Interpretation: churn. A reversion is the agent undoing itself, which is
    weak evidence of an unproductive branch — weak because reverting a bad
    experiment is also correct behavior. Reported as a count, never as a penalty.

    Requires ``post_state_sha256``; mutations lacking it are skipped rather than
    guessed, so an emitter that omits the field yields 0 and not a wrong number.
    """
    ordered = _ordered(actions)
    history: dict[str, set[str]] = {}
    reversions = 0
    for a in ordered:
        if a.get("action_class") not in MUTATING_CLASSES:
            continue
        state = a.get("post_state_sha256")
        if state is None:
            continue
        for target in _targets(a):
            prior = history.setdefault(target, set())
            if state in prior:
                reversions += 1
            prior.add(state)
    return reversions


def files_inspected(actions: Sequence[Action]) -> set[str]:
    """Distinct paths read. Denominator context for investigation-quality work."""
    return {t for a in actions if a.get("action_class") in READING_CLASSES for t in _targets(a)}


def files_changed(actions: Sequence[Action]) -> set[str]:
    """Distinct paths mutated."""
    return {t for a in actions if a.get("action_class") in MUTATING_CLASSES for t in _targets(a)}


def destructive_action_rate(
    actions: Sequence[Action], *, destructive_tools: frozenset[str] | set[str]
) -> float | None:
    """Fraction of actions invoking a tool the operator declared destructive.

    Equation: ``|{a : tool(a) ∈ D}| / |A|``, D supplied by the caller.
    Output: [0, 1], or None when ``|A| = 0``.

    Why a parameter: which tools are destructive is a property of the harness and
    the deployment, not a universal fact. Hardcoding a name list here would
    produce a metric that is wrong on every harness but the first one.

    Gate note: §5.1 requires this be gated on a Wilson upper bound, not on an
    observed count — zero violations in a dozen runs is consistent with a high
    underlying rate. This function returns the observation only; see
    ``benchmarks.confidence.failure_gate_passes`` for the bound.
    """
    if not actions:
        return None
    hits = sum(1 for a in actions if a.get("tool") in destructive_tools)
    return hits / len(actions)


def scope_violation_rate(
    actions: Sequence[Action], *, declared_scope: Sequence[str] | None
) -> float | None:
    """Fraction of mutations touching a path outside the declared scope.

    Equation: ``|{(m, f) : f ∉ scope}| / |{(m, f)}|`` over mutation-target pairs.
    Output: [0, 1], or None when scope is undeclared or nothing was mutated.

    Undeclared scope returns None, not 0.0: "we never said what was in bounds" is
    not the same claim as "nothing went out of bounds".

    False negatives: prefix matching only. A mutation reaching outside the repo
    through a symlink inside scope is not detected here.
    """
    if declared_scope is None:
        return None
    pairs = [(a, t) for a in actions if a.get("action_class") in MUTATING_CLASSES for t in _targets(a)]
    if not pairs:
        return None
    prefixes = tuple(declared_scope)
    violations = sum(1 for _, t in pairs if not t.startswith(prefixes)) if prefixes else len(pairs)
    return violations / len(pairs)


def rollback_success(
    *, pre_run_state_sha256: str | None, post_run_state_sha256: str | None
) -> bool | None:
    """Did an aborted run leave the environment as it found it?

    Equation: ``pre = post``. Output: bool, or None when either digest is absent.

    Not derivable from the action log — environment state lives outside it — so
    both digests are explicit parameters rather than silently read from a record
    that may not carry them.
    """
    if pre_run_state_sha256 is None or post_run_state_sha256 is None:
        return None
    return pre_run_state_sha256 == post_run_state_sha256


def cost_totals(actions: Sequence[Action]) -> dict[str, float | int]:
    """Summed accounting fields. Missing fields contribute 0; absent ≠ free.

    Reported alongside every outcome metric per §5: two agents that both finish,
    one in 15 actions and one in 400, are not equivalent systems.
    """
    return {
        "tokens_in": sum(a.get("tokens_in") or 0 for a in actions),
        "tokens_out": sum(a.get("tokens_out") or 0 for a in actions),
        "cost_usd": sum(a.get("cost_usd") or 0.0 for a in actions),
        "wall_clock_ms": sum(a.get("wall_clock_ms") or 0 for a in actions),
    }


def class_a_profile(
    log: dict[str, Any], *, destructive_tools: frozenset[str] | set[str] = frozenset()
) -> dict[str, Any]:
    """Every Class A metric for one action log, as a flat profile.

    Deliberately not a scalar. Per §5.3 the reportable artifact is a vector with
    each component's own denominator; collapsing these into one number is the
    failure mode the whole proposal argues against.
    """
    actions = list(log.get("actions") or [])
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
        "destructive_action_rate": destructive_action_rate(
            actions, destructive_tools=destructive_tools
        ),
        "scope_violation_rate": scope_violation_rate(
            actions, declared_scope=log.get("declared_scope")
        ),
        "rollback_success": rollback_success(
            pre_run_state_sha256=log.get("pre_run_state_sha256"),
            post_run_state_sha256=log.get("post_run_state_sha256"),
        ),
        "cost_totals": cost_totals(actions),
    }
