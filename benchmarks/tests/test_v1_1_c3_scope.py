"""v1.1 stage C3 — scope declarations and additive *_in_scope metrics (T10:
F12, F13, F14; scope_violation_rate identity with v1.0).

Run explicitly (``benchmarks`` is not in ``pytest.ini`` testpaths):

    pytest benchmarks/ -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.process_metrics import scope_violation_rate as scope_violation_rate_v10  # noqa: E402
from benchmarks.transcript_emitter import emit_file as emit_v10  # noqa: E402
from benchmarks.v1_1.emitter import emit_file  # noqa: E402
from benchmarks.v1_1.metrics import (  # noqa: E402
    class_a_profile,
    files_changed_in_scope,
    in_scope,
    mutations_without_target,
    premature_edit_rate_in_scope,
    reproduction_before_repair_in_scope,
    scope_violation_rate,
    verification_attempted_in_scope,
)

FIX = REPO / "benchmarks/v1_1/fixtures/synthetic"
V10_FIX = REPO / "benchmarks/fixtures/transcripts"
SCHEMA_V11 = json.loads((REPO / "benchmarks/v1_1/AGENT_ACTION_LOG.v1_1.schema.json").read_text())
SRC = ["src/", "tests/"]


def _log(name: str, **kw):
    return emit_file(FIX / f"{name}.jsonl", **kw).log


# --- declarations are recorded verbatim, never inferred -----------------------------


def test_scopes_are_written_verbatim_and_omitted_when_absent():
    bare = _log("F12_trailing_artifact_write")
    assert "declared_scope" not in bare and "source_scope" not in bare
    odd = ["src/", ".", "/tmp/scratch/", "  weird/ ", "artifacts/../src/"]  # no normalisation, by design
    log = _log("F12_trailing_artifact_write", declared_scope=odd, source_scope=["src/"])
    assert log["declared_scope"] == odd
    assert log["source_scope"] == ["src/"]
    jsonschema.validate(instance=log, schema=SCHEMA_V11)


def test_emitter_does_not_infer_scope_from_transcript_content():
    """A transcript full of src/ and artifacts/ paths yields no scope unless declared."""
    for name in ("F12_trailing_artifact_write", "F14_docs_only", "F04_test_output_redirect"):
        assert "source_scope" not in _log(name)


# --- in_scope: the one convention ------------------------------------------------------


@pytest.mark.parametrize(
    "target,scope,expected",
    [
        ("src/a.py", ["src/"], True),
        ("srcx/a.py", ["src/"], False),           # prefix, not path component — as in v1.0
        ("artifacts/r.md", ["src/"], False),
        ("src/a.py", ["."], True),
        ("artifacts/r.md", ["."], True),           # "." is any relative target
        ("/tmp/out.txt", ["."], False),
        ("~/.local/x", ["."], False),
        ("/tmp/out.txt", ["/tmp/"], True),
        ("src/a.py", [], False),
    ],
)
def test_in_scope_convention(target, scope, expected):
    assert in_scope(target, scope) is expected


# --- T10: the three scoped scenarios ------------------------------------------------------


def test_f12_trailing_artifact_write_is_ignored_by_the_scoped_metric():
    """Audit finding B: a report write after a real verify made v1.0 say 'no'."""
    log = _log("F12_trailing_artifact_write", source_scope=SRC)
    p = class_a_profile(log)
    assert p["verification_attempted"] is False            # v1.0 key, v1.0 semantics, unchanged
    assert p["verification_attempted_in_scope"] is True    # the scoped answer
    assert p["files_changed"] == 2 and p["files_changed_in_scope"] == 1
    assert p["mutations_without_target"] == 0


def test_f13_targetless_mutation_is_ignored_for_scope_and_counted():
    log = _log("F13_trailing_targetless_mutate", source_scope=SRC)
    p = class_a_profile(log)
    assert p["verification_attempted"] is False
    assert p["verification_attempted_in_scope"] is True
    assert p["mutations_without_target"] == 1


def test_f14_docs_only_edit_is_undefined_in_scope_not_false():
    log = _log("F14_docs_only", source_scope=SRC)
    p = class_a_profile(log)
    assert p["verification_attempted"] is False
    assert p["verification_attempted_in_scope"] is None
    assert p["reproduction_before_repair_in_scope"] is None
    assert p["premature_edit_rate_in_scope"] is None
    assert p["files_changed_in_scope"] == 0


def test_scoped_keys_are_undefined_without_a_source_scope():
    p = class_a_profile(_log("F12_trailing_artifact_write"))
    for key in ("verification_attempted_in_scope", "reproduction_before_repair_in_scope", "premature_edit_rate_in_scope", "files_changed_in_scope"):
        assert p[key] is None, key
    assert p["mutations_without_target"] == 0  # a count, defined regardless


def test_scoped_premature_and_reproduction_follow_v1_0_rules_within_scope():
    log = _log("F12_trailing_artifact_write", source_scope=SRC)
    acts = log["actions"]
    assert premature_edit_rate_in_scope(acts, SRC) == 0.0       # src/a.py was read first
    assert reproduction_before_repair_in_scope(acts, SRC) is False
    assert files_changed_in_scope(acts, SRC) == 1
    assert verification_attempted_in_scope(acts, None) is None


# --- scope_violation_rate: v1.0 semantics, "." understood --------------------------------


@pytest.mark.parametrize("name", ["session_small_errors", "session_small_reads", "session_trivial", "synthetic_minimal"])
@pytest.mark.parametrize("scope", [["benchmarks/", "nanoscribe/"], ["src/"], [], ["/tmp/"]])
def test_scope_violation_rate_identical_to_v1_0_without_the_dot_entry(name, scope):
    """On a log whose mutation-target pairs are the same under both contracts,
    the two functions must agree exactly on every scope that does not use '.'.
    The large fixture is excluded on purpose: C1 splits `pytest > /tmp/x` into
    a verify and a mutate, adding pairs v1.0 never had — see the next test."""
    path = V10_FIX / f"{name}.jsonl"
    v10 = scope_violation_rate_v10(emit_v10(path).log["actions"], declared_scope=scope)
    v11 = scope_violation_rate(emit_file(path).log["actions"], declared_scope=scope)
    assert v10 == v11


def test_scope_violation_rate_delta_on_large_fixture_is_exactly_the_split_pairs():
    """Spec §3.4 said this key would be unchanged with the same declared_scope.
    That was wrong once C1 landed: split writes add mutation-target pairs. The
    migration module explains the delta as `scope-pairs`; here the explanation
    is checked directly — the pair sets differ by exactly the pairs whose call
    v1.0 did not label `mutate`."""
    from benchmarks.v1_1.migration import compare_file

    c = compare_file(V10_FIX / "session_large_varied.jsonl", declared_scope=["benchmarks/", "nanoscribe/"])
    d = next(x for x in c.deltas if x.key == "scope_violation_rate")
    assert d.category == "scope-pairs" and d.explained, d


def test_scope_violation_rate_dot_means_in_repo():
    acts = _log("F04_test_output_redirect")["actions"]  # [read, mutate(src/a.py), verify, mutate(/tmp/out.txt)] under C1
    assert scope_violation_rate(acts, declared_scope=["."]) == 0.5   # the /tmp pair is the violation
    assert scope_violation_rate_v10(acts, declared_scope=["."]) == 1.0  # v1.0 cannot read "."; documented difference
    assert scope_violation_rate(acts, declared_scope=None) is None


def test_mutations_without_target_counts_bash_heredocs_with_computed_paths():
    assert mutations_without_target(_log("F13_trailing_targetless_mutate")["actions"]) == 1
    assert mutations_without_target(_log("F08_edit_revert")["actions"]) == 0
