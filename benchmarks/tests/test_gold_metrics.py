"""Class B metric tests — gold artifacts as parameters, no corpus required.

Run explicitly (``benchmarks`` is not in ``pytest.ini`` testpaths):

    pytest benchmarks/ -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.gold_metrics import (  # noqa: E402
    GoldTask,
    RunOutcome,
    actions_to_first_correct_file,
    affected_test_recall,
    aggregate_rate,
    claimed_complete_but_failing,
    class_b_profile,
    fix_minimality,
    localization_precision,
    localization_recall,
    regression_escape_rate,
    resolved,
)
from benchmarks.tests.test_process_metrics import act, LOG_SCHEMA_VERSION  # noqa: E402

SCHEMA_PATH = REPO / "benchmarks/AGENT_ACTION_LOG.schema.json"

GOLD = GoldTask(
    patch_files=frozenset({"src/a.py", "src/b.py"}),
    patch_lines_changed=10,
    fail_to_pass=frozenset({"test_bug"}),
    pass_to_pass=frozenset({"test_x", "test_y"}),
)


# --- undefined vs zero ------------------------------------------------------


def test_everything_is_undefined_against_an_empty_gold_task():
    empty = GoldTask()
    assert localization_recall([act(0, "mutate", targets=["a.py"])], empty) is None
    assert actions_to_first_correct_file([], empty) is None
    assert affected_test_recall([], empty) is None
    assert resolved(empty, RunOutcome()) is None
    assert fix_minimality([], empty) is None


def test_regression_escape_is_undefined_when_nothing_was_green_before():
    """A suite that was already red cannot be regressed by this run."""
    assert regression_escape_rate(RunOutcome(p2p_before={}, p2p_after={})) is None
    assert regression_escape_rate(
        RunOutcome(p2p_before={"t": False}, p2p_after={"t": False})
    ) is None


def test_localization_precision_undefined_when_the_agent_mutated_nothing():
    assert localization_precision([act(0, "read", targets=["src/a.py"])], GOLD) is None


# --- localization -----------------------------------------------------------


def test_localization_recall_and_precision_are_reported_separately():
    """An agent that rewrites the repo gets perfect recall; precision exposes it."""
    shotgun = [act(i, "mutate", targets=[f"src/{n}"]) for i, n in enumerate(
        ["a.py", "b.py", "c.py", "d.py", "e.py", "f.py", "g.py", "h.py", "i.py", "j.py"]
    )]
    assert localization_recall(shotgun, GOLD) == 1.0
    assert localization_precision(shotgun, GOLD) == 0.2


def test_localization_recall_partial_hit():
    actions = [act(0, "mutate", targets=["src/a.py"])]
    assert localization_recall(actions, GOLD) == 0.5
    assert localization_precision(actions, GOLD) == 1.0


def test_reads_do_not_count_as_localization():
    """Only mutations localize; reading the right file is Class A investigation."""
    reads_only = [act(0, "read", targets=["src/a.py", "src/b.py"])]
    assert localization_recall(reads_only, GOLD) == 0.0


def test_actions_to_first_correct_file_uses_seq_order():
    shuffled = [
        act(3, "mutate", targets=["src/a.py"]),
        act(0, "read", targets=["src/a.py"]),
        act(1, "mutate", targets=["src/wrong.py"]),
        act(2, "verify"),
    ]
    assert actions_to_first_correct_file(shuffled, GOLD) == 3


def test_actions_to_first_correct_file_none_when_never_touched():
    assert actions_to_first_correct_file([act(0, "mutate", targets=["z.py"])], GOLD) is None


# --- test surface -----------------------------------------------------------


def test_affected_test_recall_counts_only_tests_the_agent_ran():
    actions = [act(0, "verify", tests_run=["test_bug", "test_x"])]
    assert affected_test_recall(actions, GOLD) == 2 / 3


def test_affected_test_recall_ignores_tests_outside_the_gold_surface():
    actions = [act(0, "verify", tests_run=["test_unrelated"])]
    assert affected_test_recall(actions, GOLD) == 0.0


def test_affected_test_recall_is_about_running_not_passing():
    """Running the tests earns recall regardless of outcome; the grader judges results."""
    actions = [act(0, "verify", tests_run=["test_bug", "test_x", "test_y"])]
    assert affected_test_recall(actions, GOLD) == 1.0
    assert resolved(GOLD, RunOutcome(f2p_after={"test_bug": False})) is False


# --- regression and resolution ---------------------------------------------


def test_regression_escape_counts_only_tests_green_before():
    outcome = RunOutcome(
        p2p_before={"test_x": True, "test_y": True, "test_z": False},
        p2p_after={"test_x": True, "test_y": False, "test_z": False},
    )
    assert regression_escape_rate(outcome) == 0.5


def test_missing_after_result_counts_as_broken_not_as_passing():
    outcome = RunOutcome(p2p_before={"test_x": True}, p2p_after={})
    assert regression_escape_rate(outcome) == 1.0


def test_resolved_requires_every_f2p_test_and_missing_is_not_a_pass():
    gold = GoldTask(fail_to_pass=frozenset({"a", "b"}))
    assert resolved(gold, RunOutcome(f2p_after={"a": True, "b": True})) is True
    assert resolved(gold, RunOutcome(f2p_after={"a": True, "b": False})) is False
    assert resolved(gold, RunOutcome(f2p_after={"a": True})) is False


# --- the headline metric ----------------------------------------------------


def test_claimed_complete_but_failing_is_the_opening_failure():
    """Confident completion on a patch that does not resolve the task."""
    actions = [
        act(0, "mutate", targets=["src/a.py"]),
        act(1, "verify", claimed_complete=True),
    ]
    failing = RunOutcome(f2p_after={"test_bug": False})
    assert claimed_complete_but_failing(actions, GOLD, failing) is True

    passing = RunOutcome(f2p_after={"test_bug": True})
    assert claimed_complete_but_failing(actions, GOLD, passing) is False


def test_silent_failure_is_not_counted_as_a_false_claim():
    """Failing without claiming success is a different, lesser failure."""
    quiet = [act(0, "mutate", targets=["src/a.py"])]
    assert claimed_complete_but_failing(quiet, GOLD, RunOutcome(f2p_after={"test_bug": False})) is False


def test_class_a_asks_whether_it_looked_class_b_asks_whether_it_was_right():
    """An agent can run the tests, misread them, and still claim completion."""
    from benchmarks.process_metrics import verification_attempted

    actions = [
        act(0, "mutate", targets=["src/a.py"]),
        act(1, "verify", tests_run=["test_bug"], claimed_complete=True),
    ]
    assert verification_attempted(actions) is True
    assert claimed_complete_but_failing(actions, GOLD, RunOutcome(f2p_after={"test_bug": False})) is True


# --- minimality -------------------------------------------------------------


def test_fix_minimality_is_a_ratio_to_the_reference_patch():
    actions = [
        act(0, "mutate", targets=["src/a.py"], lines_changed=30),
        act(1, "mutate", targets=["src/b.py"], lines_changed=20),
    ]
    assert fix_minimality(actions, GOLD) == 5.0


def test_fix_minimality_understates_rather_than_invents_when_field_absent():
    actions = [act(0, "mutate", targets=["src/a.py"])]
    assert fix_minimality(actions, GOLD) == 0.0


# --- aggregation ------------------------------------------------------------


def test_aggregate_returns_counts_not_a_collapsed_rate():
    """A gate needs numerator and denominator; 0.0 invites the wrong reading."""
    assert aggregate_rate([True, False, False]) == (1, 3)
    assert aggregate_rate([False, False]) == (0, 2)


def test_aggregate_skips_undefined_runs_and_is_none_when_all_undefined():
    assert aggregate_rate([True, None, False]) == (1, 2)
    assert aggregate_rate([None, None]) is None


# --- profile and schema -----------------------------------------------------


def test_class_b_profile_is_a_vector():
    log = {
        "schema_version": LOG_SCHEMA_VERSION,
        "actions": [
            act(0, "read", targets=["src/a.py"]),
            act(1, "mutate", targets=["src/a.py"], lines_changed=10),
            act(2, "verify", tests_run=["test_bug", "test_x", "test_y"], claimed_complete=True),
        ],
    }
    outcome = RunOutcome(
        f2p_after={"test_bug": True},
        p2p_before={"test_x": True, "test_y": True},
        p2p_after={"test_x": True, "test_y": True},
    )
    profile = class_b_profile(log, GOLD, outcome)
    assert profile["resolved"] is True
    assert profile["claimed_complete_but_failing"] is False
    assert profile["regression_escape_rate"] == 0.0
    assert profile["affected_test_recall"] == 1.0
    assert profile["localization_recall"] == 0.5
    assert not isinstance(profile, (int, float))


def test_schema_accepts_the_class_b_fields():
    schema = json.loads(SCHEMA_PATH.read_text())
    log = {
        "schema_version": LOG_SCHEMA_VERSION,
        "actions": [
            act(0, "mutate", targets=["src/a.py"], lines_changed=12),
            act(1, "verify", tests_run=["test_bug"], claimed_complete=True),
        ],
    }
    jsonschema.validate(instance=log, schema=schema)


def test_schema_rejects_duplicate_test_ids_and_negative_line_counts():
    schema = json.loads(SCHEMA_PATH.read_text())
    for bad in (
        [act(0, "verify", tests_run=["t", "t"])],
        [act(0, "mutate", targets=["a.py"], lines_changed=-1)],
    ):
        try:
            jsonschema.validate(
                instance={"schema_version": LOG_SCHEMA_VERSION, "actions": bad}, schema=schema
            )
        except jsonschema.ValidationError:
            continue
        raise AssertionError(f"schema accepted invalid action: {bad}")
