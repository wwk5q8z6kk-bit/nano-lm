"""v1.1 stage C2 — schema exclusion (T1), fixture/generator match, pre_state
semantics, and reversion with pre-state seeding (T8: F08, F09, F10, F11).

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

from benchmarks.process_metrics import class_a_profile as class_a_profile_v10  # noqa: E402
from benchmarks.process_metrics import reversion_count as reversion_count_v10  # noqa: E402
from benchmarks.v1_1.emitter import LOG_SCHEMA_VERSION, emit_file  # noqa: E402
from benchmarks.v1_1.metrics import class_a_profile, noop_mutations, pre_state_coverage, reversion_count  # noqa: E402

FIX = REPO / "benchmarks/v1_1/fixtures/synthetic"
SCHEMA_V10 = json.loads((REPO / "benchmarks/AGENT_ACTION_LOG.schema.json").read_text())
SCHEMA_V11 = json.loads((REPO / "benchmarks/v1_1/AGENT_ACTION_LOG.v1_1.schema.json").read_text())


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _log(name: str) -> dict:
    return emit_file(FIX / f"{name}.jsonl").log


# --- T1: schema versioning discipline ------------------------------------------


def test_schema_versions_are_mutually_exclusive():
    assert SCHEMA_V10["properties"]["schema_version"]["const"] != SCHEMA_V11["properties"]["schema_version"]["const"]
    v11 = _log("F08_edit_revert")
    jsonschema.validate(instance=v11, schema=SCHEMA_V11)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=v11, schema=SCHEMA_V10)  # rejected on schema_version AND on call_id
    v10_shaped = {"schema_version": SCHEMA_V10["properties"]["schema_version"]["const"], "actions": []}
    jsonschema.validate(instance=v10_shaped, schema=SCHEMA_V10)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=v10_shaped, schema=SCHEMA_V11)


def test_v1_0_validator_rejects_call_id_specifically():
    """Even with the right version string, a v1.0 action object rejects the new field."""
    action = {"seq": 0, "call_id": "t0", "action_class": "read", "tool": "Read", "args_sha256": "a" * 64, "result_status": "ok"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={"schema_version": SCHEMA_V10["properties"]["schema_version"]["const"], "actions": [action]}, schema=SCHEMA_V10)


def test_v1_1_requires_call_id_on_every_action():
    action = {"seq": 0, "action_class": "read", "tool": "Read", "args_sha256": "a" * 64, "result_status": "ok"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance={"schema_version": LOG_SCHEMA_VERSION, "actions": [action]}, schema=SCHEMA_V11)


def test_v1_1_schema_differs_from_v1_0_by_exactly_the_three_additions():
    a10, a11 = SCHEMA_V10["$defs"]["action"]["properties"], SCHEMA_V11["$defs"]["action"]["properties"]
    assert set(a11) - set(a10) == {"call_id", "pre_state_sha256"}
    assert set(a10) - set(a11) == set()
    assert set(SCHEMA_V11["properties"]) - set(SCHEMA_V10["properties"]) == {"source_scope"}
    assert SCHEMA_V11["$defs"]["action"]["additionalProperties"] is False


# --- fixtures match their generator ---------------------------------------------


def test_v1_1_fixtures_match_generator():
    sys.path.insert(0, str(REPO / "benchmarks/v1_1/fixtures"))
    import make_synthetic_v1_1 as gen  # noqa: E402

    expected = {f"{name}.jsonl": gen.render(s) for name, s in gen.scenarios().items()}
    on_disk = {p.name: p.read_text() for p in FIX.glob("*.jsonl")}
    assert on_disk == expected, "regenerate with make_synthetic_v1_1.py"
    assert len(expected) == 15


@pytest.mark.parametrize("path", sorted(FIX.glob("*.jsonl")), ids=lambda p: p.stem)
def test_every_v1_1_fixture_emits_a_valid_v1_1_log(path):
    r = emit_file(path)
    jsonschema.validate(instance=r.log, schema=SCHEMA_V11)
    assert all(a["call_id"] for a in r.log["actions"])


# --- C2: pre_state semantics ------------------------------------------------------


def test_pre_state_present_on_ok_edit_and_write_update_null_on_create():
    edits = {a["seq"]: a for a in _log("F08_edit_revert")["actions"] if a["action_class"] == "mutate"}
    assert edits[1]["pre_state_sha256"] == _sha("x = 1\ny = 2\n")
    assert edits[1]["post_state_sha256"] == _sha("x = 2\ny = 2\n")
    assert edits[2]["pre_state_sha256"] == _sha("x = 2\ny = 2\n")
    assert edits[2]["post_state_sha256"] == _sha("x = 1\ny = 2\n")

    writes = [a for a in _log("F10_write_create_update_revert")["actions"] if a["action_class"] == "mutate"]
    assert writes[0]["pre_state_sha256"] is None                      # create: no prior content, no sentinel
    assert writes[0]["post_state_sha256"] == _sha("print(1)\n")
    assert writes[1]["pre_state_sha256"] == _sha("print(1)\n")
    assert writes[2]["pre_state_sha256"] == _sha("print(2)\n")


def test_pre_state_null_on_bash_and_multi_target(tmp_path):
    bash = [a for a in _log("F11_multi_target_copy")["actions"] if a["action_class"] == "mutate"]
    assert bash and all(a["pre_state_sha256"] is None and a["post_state_sha256"] is None for a in bash)


def test_pre_state_null_on_non_ok_mutation():
    errored = [a for a in _log("F06_errored_compound")["actions"] if a["action_class"] == "mutate" or a["result_status"] != "ok"]
    assert errored and all(a["pre_state_sha256"] is None for a in errored)


def test_pre_state_never_a_sentinel():
    for path in FIX.glob("*.jsonl"):
        for a in emit_file(path).log["actions"]:
            assert a["pre_state_sha256"] is None or (len(a["pre_state_sha256"]) == 64 and a["pre_state_sha256"] != "0" * 64)


# --- T8: reversion with pre-state seeding ------------------------------------------


def test_f08_return_to_initial_state_is_now_a_reversion():
    """The case v1.0 pinned as invisible by design."""
    log = _log("F08_edit_revert")
    assert reversion_count_v10(log["actions"]) == 0
    assert reversion_count(log["actions"]) == 1
    assert noop_mutations(log["actions"]) == 0


def test_f09_noop_edits_are_excluded_and_counted():
    log = _log("F09_noop_edits")
    assert reversion_count(log["actions"]) == 0
    assert noop_mutations(log["actions"]) == 2


def test_f10_write_create_update_revert_detected_via_pre_state():
    log = _log("F10_write_create_update_revert")
    assert reversion_count_v10(log["actions"]) == 1  # v1.0 sees it too: C1 was a recorded post-state
    assert reversion_count(log["actions"]) == 1
    assert pre_state_coverage(log["actions"]) == pytest.approx(2 / 3)


def test_reversion_naive_rule_would_count_consecutive_noops():
    """Documents why D10 exists: without the noop exclusion the second A→A counts."""
    log = _log("F09_noop_edits")
    acts = [dict(a, pre_state_sha256=None) for a in log["actions"]]  # drop pre → naive post-only history
    assert reversion_count(acts) == 1  # post A seen before → naive reversion; the real rule reports 0


# --- backward compatibility at this stage (T3, trivially) ---------------------------


@pytest.mark.parametrize("name", ["session_small_reads", "session_small_errors", "session_trivial"])
def test_v1_0_keys_identical_on_frozen_session_fixtures(name):
    from benchmarks.transcript_emitter import emit_file as emit_v10

    path = REPO / "benchmarks/fixtures/transcripts" / f"{name}.jsonl"
    p10 = class_a_profile_v10(emit_v10(path).log)
    p11 = class_a_profile(emit_file(path).log)
    for key, value in p10.items():
        assert p11[key] == value, key
