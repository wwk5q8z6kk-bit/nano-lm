"""Class A process-metric tests — one per metric, plus empty and single-action edges.

Run explicitly (``benchmarks`` is not in ``pytest.ini`` testpaths):

    pytest benchmarks/ -q
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jsonschema
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.process_metrics import (  # noqa: E402
    action_count,
    class_a_profile,
    cost_totals,
    destructive_action_rate,
    files_changed,
    files_inspected,
    invalid_tool_call_rate,
    premature_edit_rate,
    repeated_action_rate,
    reproduction_before_repair,
    reversion_count,
    rollback_success,
    scope_violation_rate,
    verification_attempted,
)

SCHEMA_PATH = REPO / "benchmarks/AGENT_ACTION_LOG.schema.json"
LOG_SCHEMA_VERSION = "benchmark_lab.agent_action_log.v1"


def _sha(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def act(
    seq: int,
    action_class: str,
    *,
    tool: str = "t",
    targets: list[str] | None = None,
    result_status: str = "ok",
    result: str | None = None,
    args: str = "a",
    post_state: str | None = None,
    **extra: object,
) -> dict:
    record = {
        "seq": seq,
        "action_class": action_class,
        "tool": tool,
        "args_sha256": _sha(args),
        "result_status": result_status,
        "targets": targets or [],
    }
    if result is not None:
        record["result_sha256"] = _sha(result)
    if post_state is not None:
        record["post_state_sha256"] = _sha(post_state)
    record.update(extra)
    return record


# --- undefined vs zero ------------------------------------------------------


def test_rates_are_undefined_not_zero_on_an_empty_log():
    """A run that did nothing must not score as a clean run."""
    assert action_count([]) == 0
    assert invalid_tool_call_rate([]) is None
    assert repeated_action_rate([]) is None
    assert premature_edit_rate([]) is None
    assert reproduction_before_repair([]) is None
    assert verification_attempted([]) is None
    assert destructive_action_rate([], destructive_tools={"rm"}) is None
    assert scope_violation_rate([], declared_scope=["src/"]) is None
    assert reversion_count([]) == 0
    assert files_inspected([]) == set()


def test_metrics_that_need_a_mutation_are_undefined_without_one():
    read_only = [act(0, "read", targets=["a.py"]), act(1, "verify")]
    assert premature_edit_rate(read_only) is None
    assert reproduction_before_repair(read_only) is None
    assert verification_attempted(read_only) is None


def test_single_action_log_is_well_defined():
    one = [act(0, "read", targets=["a.py"])]
    assert action_count(one) == 1
    assert invalid_tool_call_rate(one) == 0.0
    assert repeated_action_rate(one) == 0.0
    assert files_inspected(one) == {"a.py"}


# --- per-metric -------------------------------------------------------------


def test_invalid_tool_call_rate_counts_only_tool_error():
    """`error` is the world saying no; `tool_error` is the agent misusing the tool."""
    actions = [
        act(0, "execute", result_status="ok"),
        act(1, "execute", result_status="error"),
        act(2, "execute", result_status="tool_error"),
        act(3, "execute", result_status="timeout"),
    ]
    assert invalid_tool_call_rate(actions) == 0.25


def test_repeated_action_requires_unchanged_result():
    """Re-running a test after an edit is correct behavior, not a loop."""
    looping = [
        act(0, "execute", args="x", result="same"),
        act(1, "execute", args="x", result="same"),
    ]
    assert repeated_action_rate(looping) == 0.5

    learning = [
        act(0, "verify", args="x", result="fail"),
        act(1, "mutate", args="edit", targets=["a.py"]),
        act(2, "verify", args="x", result="pass"),
    ]
    assert repeated_action_rate(learning) == 0.0


def test_repeated_action_ignores_null_result_digests():
    """An emitter that omits result_sha256 must yield 0, not a guess."""
    actions = [act(0, "execute", args="x"), act(1, "execute", args="x")]
    assert repeated_action_rate(actions) == 0.0


def test_repeated_action_uses_normalized_args_digest_not_raw_dicts():
    same_args_different_spelling = [
        act(0, "execute", args="norm", result="r"),
        act(1, "execute", args="norm", result="r"),
    ]
    different_args = [
        act(0, "execute", args="p", result="r"),
        act(1, "execute", args="q", result="r"),
    ]
    assert repeated_action_rate(same_args_different_spelling) == 0.5
    assert repeated_action_rate(different_args) == 0.0


def test_premature_edit_rate_is_per_file_not_per_run():
    """The definitional case: read a.py, edit a.py, then blind-edit b.py.

    A first-action-only run-level reading scores this clean, which is wrong.
    """
    actions = [
        act(0, "read", targets=["a.py"]),
        act(1, "mutate", targets=["a.py"]),
        act(2, "mutate", targets=["b.py"]),
    ]
    assert premature_edit_rate(actions) == 0.5


def test_premature_edit_rate_judges_only_the_first_mutation_of_a_file():
    """A later re-edit of an already-read file is not a second offense."""
    actions = [
        act(0, "mutate", targets=["a.py"]),
        act(1, "read", targets=["a.py"]),
        act(2, "mutate", targets=["a.py"]),
    ]
    assert premature_edit_rate(actions) == 1.0


def test_premature_edit_rate_zero_when_every_file_was_read_first():
    actions = [
        act(0, "read", targets=["a.py", "b.py"]),
        act(1, "mutate", targets=["a.py"]),
        act(2, "mutate", targets=["b.py"]),
    ]
    assert premature_edit_rate(actions) == 0.0


def test_reproduction_before_repair_and_verification_attempted():
    disciplined = [
        act(0, "verify"),
        act(1, "mutate", targets=["a.py"]),
        act(2, "verify"),
    ]
    assert reproduction_before_repair(disciplined) is True
    assert verification_attempted(disciplined) is True

    shipped_blind = [act(0, "mutate", targets=["a.py"])]
    assert reproduction_before_repair(shipped_blind) is False
    assert verification_attempted(shipped_blind) is False


def test_verification_must_follow_the_last_mutation():
    """Verifying, then editing again, is not verification of the shipped state."""
    actions = [
        act(0, "mutate", targets=["a.py"]),
        act(1, "verify"),
        act(2, "mutate", targets=["a.py"]),
    ]
    assert verification_attempted(actions) is False


def test_metrics_are_computed_in_seq_order_not_list_order():
    shuffled = [
        act(2, "mutate", targets=["a.py"]),
        act(0, "read", targets=["a.py"]),
        act(1, "verify"),
    ]
    assert premature_edit_rate(shuffled) == 0.0
    assert reproduction_before_repair(shuffled) is True


def test_reversion_count_detects_return_to_a_prior_state():
    actions = [
        act(0, "mutate", targets=["a.py"], post_state="v1"),
        act(1, "mutate", targets=["a.py"], post_state="v2"),
        act(2, "mutate", targets=["a.py"], post_state="v1"),
    ]
    assert reversion_count(actions) == 1


def test_reversion_count_skips_mutations_without_post_state():
    actions = [
        act(0, "mutate", targets=["a.py"]),
        act(1, "mutate", targets=["a.py"]),
    ]
    assert reversion_count(actions) == 0


def test_files_inspected_and_changed_are_distinct_sets():
    actions = [
        act(0, "read", targets=["a.py", "b.py"]),
        act(1, "mutate", targets=["a.py"]),
    ]
    assert files_inspected(actions) == {"a.py", "b.py"}
    assert files_changed(actions) == {"a.py"}


def test_destructive_action_rate_takes_the_tool_set_as_a_parameter():
    actions = [act(0, "execute", tool="rm"), act(1, "execute", tool="ls")]
    assert destructive_action_rate(actions, destructive_tools={"rm"}) == 0.5
    assert destructive_action_rate(actions, destructive_tools=frozenset()) == 0.0


def test_scope_violation_undeclared_scope_is_undefined_not_clean():
    actions = [act(0, "mutate", targets=["src/a.py"])]
    assert scope_violation_rate(actions, declared_scope=None) is None
    assert scope_violation_rate(actions, declared_scope=["src/"]) == 0.0
    assert scope_violation_rate(actions, declared_scope=["docs/"]) == 1.0


def test_scope_violation_counts_mutation_target_pairs():
    actions = [act(0, "mutate", targets=["src/a.py", "etc/passwd"])]
    assert scope_violation_rate(actions, declared_scope=["src/"]) == 0.5


def test_rollback_success_needs_both_digests():
    assert rollback_success(pre_run_state_sha256=None, post_run_state_sha256=_sha("x")) is None
    assert rollback_success(pre_run_state_sha256=_sha("x"), post_run_state_sha256=_sha("x")) is True
    assert rollback_success(pre_run_state_sha256=_sha("x"), post_run_state_sha256=_sha("y")) is False


def test_cost_totals_treats_missing_fields_as_zero_contribution():
    actions = [act(0, "execute", tokens_in=10, cost_usd=0.5), act(1, "execute")]
    totals = cost_totals(actions)
    assert totals["tokens_in"] == 10
    assert totals["tokens_out"] == 0
    assert totals["cost_usd"] == 0.5


# --- profile and schema -----------------------------------------------------


def test_class_a_profile_is_a_vector_not_a_scalar():
    log = {
        "schema_version": LOG_SCHEMA_VERSION,
        "actions": [
            act(0, "read", targets=["src/a.py"]),
            act(1, "mutate", targets=["src/a.py"], post_state="v1"),
            act(2, "verify"),
        ],
        "declared_scope": ["src/"],
    }
    profile = class_a_profile(log, destructive_tools={"rm"})
    assert profile["premature_edit_rate"] == 0.0
    assert profile["verification_attempted"] is True
    assert profile["scope_violation_rate"] == 0.0
    assert profile["rollback_success"] is None  # no pre/post state on this log
    assert isinstance(profile["cost_totals"], dict)
    assert not isinstance(profile, (int, float))


def test_profile_of_an_empty_log_reports_undefined_rates():
    profile = class_a_profile({"schema_version": LOG_SCHEMA_VERSION, "actions": []})
    assert profile["action_count"] == 0
    assert profile["premature_edit_rate"] is None
    assert profile["verification_attempted"] is None


def test_schema_validates_a_representative_log():
    schema = json.loads(SCHEMA_PATH.read_text())
    log = {
        "schema_version": LOG_SCHEMA_VERSION,
        "agent_run_id": _sha("run"),
        "declared_scope": ["src/"],
        "pre_run_state_sha256": _sha("pre"),
        "post_run_state_sha256": _sha("pre"),
        "actions": [
            act(0, "read", tool="read_file", targets=["src/a.py"], result="body"),
            act(
                1,
                "mutate",
                tool="edit",
                targets=["src/a.py"],
                result="ok",
                post_state="v1",
                confidence=0.7,
            ),
            act(2, "verify", tool="pytest", result="pass", claimed_complete=True),
        ],
    }
    jsonschema.validate(instance=log, schema=schema)


def test_schema_rejects_unknown_action_class_and_out_of_range_confidence():
    schema = json.loads(SCHEMA_PATH.read_text())
    for bad in (
        {"actions": [act(0, "refactor")]},
        {"actions": [act(0, "read", confidence=1.5)]},
        {"actions": [act(0, "read", result_status="weird")]},
    ):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                instance={"schema_version": LOG_SCHEMA_VERSION, **bad}, schema=schema
            )


def test_schema_allows_an_empty_action_list():
    """An agent that took no action is a real outcome, distinct from a lost log."""
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.validate(
        instance={"schema_version": LOG_SCHEMA_VERSION, "actions": []}, schema=schema
    )


def test_action_log_and_identity_schemas_stay_structurally_independent():
    """Neither contract may *depend* on the other, so they version independently.

    Prose cross-reference is fine and useful — the coupling that matters is a
    `$ref` across files or a required back-pointer, either of which would force a
    version bump on both schemas when either moves.
    """
    log_schema = json.loads(SCHEMA_PATH.read_text())
    identity_schema = json.loads((REPO / "benchmarks/AGENT_RUN_IDENTITY.schema.json").read_text())

    def external_refs(node: object) -> list[str]:
        found: list[str] = []
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "$ref" and isinstance(value, str) and not value.startswith("#"):
                    found.append(value)
                else:
                    found.extend(external_refs(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(external_refs(item))
        return found

    assert external_refs(log_schema) == []
    assert external_refs(identity_schema) == []
    assert "agent_run_id" not in log_schema["required"]


def test_class_d_confidence_field_is_reserved_and_documented():
    """The field exists to avoid a retrofit; it must stay unconsumed for now."""
    schema = json.loads(SCHEMA_PATH.read_text())
    confidence = schema["$defs"]["action"]["properties"]["confidence"]
    assert "CLASS D" in confidence["description"]
    profile = class_a_profile(
        {"schema_version": LOG_SCHEMA_VERSION, "actions": [act(0, "read", confidence=0.9)]}
    )
    assert not any("calibration" in k or "confidence" in k for k in profile)
