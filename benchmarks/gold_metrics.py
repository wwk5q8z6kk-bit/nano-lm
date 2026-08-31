"""Class B engineering-process metrics — action log plus gold artifacts, no annotation.

Relationship to `benchmarks/process_metrics.py`
-----------------------------------------------
Class A (there) needs nothing but the action log. Class B (here) additionally needs
the gold artifacts that issue-resolution tasks **already ship**: the reference patch
and the fail-to-pass / pass-to-pass test sets. No new expert annotation is required,
which is what separates Class B from Class C
(`papers/SWE_CAPABILITY_EVALUATION_PROPOSAL.md` §4.2 vs §4.3).

The gold artifacts are function **parameters**, never a bundled dataset. That is why
this module exists before any corpus does: the instrument can be defined and tested
against synthetic tasks, and the corpus decision stays open.

Diagnostics, not gates. No threshold is promoted. Promotion requires §5.3's criteria
(sufficient n, tested failure modes, documented baseline, justified threshold), and
§5.1 requires that any *safety* gate be stated as a Wilson upper bound over runs
rather than an observed count — this module returns observations only; the bound
lives in `benchmarks/confidence.py`.

Undefined versus zero, as in Class A: every rate returns ``None`` when its
denominator is zero. A task with no pass-to-pass tests has an undefined regression
escape rate, not a perfect one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

Action = dict[str, Any]

MUTATING_CLASSES = frozenset({"mutate"})


@dataclass(frozen=True)
class GoldTask:
    """The reference artifacts an issue-resolution task ships with.

    `patch_files` and `patch_lines_changed` come from the reference patch;
    `fail_to_pass` are the tests the fix must turn green; `pass_to_pass` are the
    tests that must stay green (the regression surface).
    """

    patch_files: frozenset[str] = frozenset()
    patch_lines_changed: int = 0
    fail_to_pass: frozenset[str] = frozenset()
    pass_to_pass: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RunOutcome:
    """Test results the harness observed around the run.

    Kept separate from the action log because these come from the *grader*
    executing the gold suite, not from anything the agent did. Conflating the two
    is how an agent's own claim about its tests becomes the score.
    """

    f2p_after: dict[str, bool] = field(default_factory=dict)
    p2p_before: dict[str, bool] = field(default_factory=dict)
    p2p_after: dict[str, bool] = field(default_factory=dict)


def _ordered(actions: Sequence[Action]) -> list[Action]:
    return sorted(actions, key=lambda a: a.get("seq", 0))


def _mutated_files(actions: Sequence[Action]) -> set[str]:
    return {
        t
        for a in actions
        if a.get("action_class") in MUTATING_CLASSES
        for t in (a.get("targets") or [])
    }


def localization_recall(actions: Sequence[Action], gold: GoldTask) -> float | None:
    """Share of gold-patch files the agent actually touched.

    Equation: ``|touched ∩ gold| / |gold|``. Output: [0, 1], or None when the gold
    patch names no file.

    Interpretation: did the agent find the right place? Recall is the primary
    localization signal because missing a required file guarantees an incomplete
    fix, whereas touching an extra file may be harmless.

    False positives: a file touched incidentally (formatting, import reorder)
    counts as located. This metric sees paths, not intent.
    """
    if not gold.patch_files:
        return None
    touched = _mutated_files(actions)
    return len(touched & gold.patch_files) / len(gold.patch_files)


def localization_precision(actions: Sequence[Action], gold: GoldTask) -> float | None:
    """Share of touched files that were in the gold patch.

    Equation: ``|touched ∩ gold| / |touched|``. Output: [0, 1], or None when the
    agent mutated nothing.

    Reported beside recall, never merged into one number: an agent that rewrites
    the repository achieves perfect recall, and precision is what exposes it.

    False negatives: a legitimate fix that differs from the reference patch scores
    poorly. The gold patch is *a* solution, not the only one — which is why this is
    a diagnostic and not a gate.
    """
    touched = _mutated_files(actions)
    if not touched:
        return None
    return len(touched & gold.patch_files) / len(touched)


def actions_to_first_correct_file(actions: Sequence[Action], gold: GoldTask) -> int | None:
    """Number of actions taken before first mutating a gold-patch file.

    Equation: ``index of first m ∈ mutate with target ∈ gold``, 0-based over the
    ordered log. Output: integer ≥ 0, or None if no gold file was ever mutated.

    Interpretation: search efficiency. Only comparable across runs on the same
    task — a harder task legitimately costs more actions.
    """
    if not gold.patch_files:
        return None
    for index, action in enumerate(_ordered(actions)):
        if action.get("action_class") not in MUTATING_CLASSES:
            continue
        if set(action.get("targets") or []) & gold.patch_files:
            return index
    return None


def affected_test_recall(actions: Sequence[Action], gold: GoldTask) -> float | None:
    """Share of the gold test surface the agent chose to run itself.

    Equation: ``|ran ∩ (F2P ∪ P2P)| / |F2P ∪ P2P|``. Output: [0, 1], or None when
    the task ships no tests.

    Interpretation: change-impact awareness — did the agent know what its edit
    could break? Distinct from whether the tests *passed*, which the grader
    determines; an agent gets no credit here for tests it never ran.

    False negatives: an agent that runs the whole suite by one command may not
    enumerate individual test ids, scoring low while having covered everything.
    That is an emitter contract issue: `tests_run` must list what actually ran.
    """
    surface = gold.fail_to_pass | gold.pass_to_pass
    if not surface:
        return None
    ran: set[str] = set()
    for action in actions:
        ran.update(action.get("tests_run") or [])
    return len(ran & surface) / len(surface)


def regression_escape_rate(outcome: RunOutcome) -> float | None:
    """Share of pass-to-pass tests the run broke.

    Equation: ``|{t : before(t) ∧ ¬after(t)}| / |{t : before(t)}|``.
    Output: [0, 1], or None when no test passed beforehand.

    Only tests that were green *before* count in the denominator: a test already
    failing at the start is not a regression this run caused.

    This is the collateral-damage measure — the failure named in the original
    critique, where a patch looks valid and quietly breaks something else.
    """
    green_before = [t for t, passed in outcome.p2p_before.items() if passed]
    if not green_before:
        return None
    broken = sum(1 for t in green_before if not outcome.p2p_after.get(t, False))
    return broken / len(green_before)


def resolved(gold: GoldTask, outcome: RunOutcome) -> bool | None:
    """Did every fail-to-pass test end green?

    Equation: ``∀ t ∈ F2P : after(t)``. Output: bool, or None when F2P is empty.
    A test in F2P with no recorded result counts as **not** passing: an unreported
    result is not a pass.
    """
    if not gold.fail_to_pass:
        return None
    return all(outcome.f2p_after.get(t, False) for t in gold.fail_to_pass)


def claimed_complete(actions: Sequence[Action]) -> bool:
    """Did the agent assert completion at any action?"""
    return any(bool(a.get("claimed_complete")) for a in actions)


def claimed_complete_but_failing(
    actions: Sequence[Action], gold: GoldTask, outcome: RunOutcome
) -> bool | None:
    """The agent declared done while the gold tests still fail.

    Equation: ``claimed_complete ∧ ¬resolved``. Output: bool, or None when
    resolution is undefined (no F2P tests).

    This is the most diagnostic single number in the Class A/B set: it is the
    direct measure of the opening failure — an apparently valid patch, damage
    unnoticed, verification skipped, and a confident claim of success. It costs
    nothing beyond `claimed_complete` on the log and the gold test results.

    Note the asymmetry with `verification_attempted` (Class A): an agent can run
    tests and still misread them. Class A asks whether it looked; this asks
    whether it was right.
    """
    is_resolved = resolved(gold, outcome)
    if is_resolved is None:
        return None
    return claimed_complete(actions) and not is_resolved


def fix_minimality(actions: Sequence[Action], gold: GoldTask) -> float | None:
    """Lines changed by the agent relative to the reference patch.

    Equation: ``Σ lines_changed / gold.patch_lines_changed``.
    Output: ≥ 0, or None when the gold patch records no line count.
    1.0 means the same size as the reference; 5.0 means five times larger.

    Not a quality score in either direction. A larger fix may be a worse one or a
    more thorough one, and a smaller fix may be elegant or incomplete. It is
    reported so that a patch an order of magnitude off the reference is visible
    rather than hidden inside a binary pass.

    Mutating actions without `lines_changed` contribute 0, so a partial emitter
    understates rather than invents.
    """
    if not gold.patch_lines_changed:
        return None
    changed = sum(
        a.get("lines_changed") or 0
        for a in actions
        if a.get("action_class") in MUTATING_CLASSES
    )
    return changed / gold.patch_lines_changed


def class_b_profile(
    log: dict[str, Any], gold: GoldTask, outcome: RunOutcome
) -> dict[str, Any]:
    """Every Class B metric for one run. A vector, never a scalar (§5.3)."""
    actions = list(log.get("actions") or [])
    return {
        "localization_recall": localization_recall(actions, gold),
        "localization_precision": localization_precision(actions, gold),
        "actions_to_first_correct_file": actions_to_first_correct_file(actions, gold),
        "affected_test_recall": affected_test_recall(actions, gold),
        "regression_escape_rate": regression_escape_rate(outcome),
        "resolved": resolved(gold, outcome),
        "claimed_complete": claimed_complete(actions),
        "claimed_complete_but_failing": claimed_complete_but_failing(actions, gold, outcome),
        "fix_minimality": fix_minimality(actions, gold),
    }


def aggregate_rate(values: Sequence[bool | None]) -> tuple[int, int] | None:
    """Successes and denominator over many runs, as counts — deliberately not a rate.

    Returns ``(count_true, n_defined)`` or None when no run had a defined value.

    Why counts rather than a float: §5.1 requires any safety-relevant gate to be
    stated as a Wilson upper bound, which needs the numerator and denominator, not
    a collapsed ratio. Zero violations in twelve runs is consistent with a high
    underlying rate, and returning ``0.0`` here would invite exactly the reading
    the bound exists to prevent. Callers that want a point estimate can divide;
    callers building a gate pass both terms to
    ``benchmarks.confidence.failure_gate_passes``.
    """
    defined = [v for v in values if v is not None]
    if not defined:
        return None
    return sum(1 for v in defined if v), len(defined)
