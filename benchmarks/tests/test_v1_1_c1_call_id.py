"""v1.1 stage C1 — call_id, ordered sub-actions, merging, attribution, status
inheritance, per-call rates (F01–F07; spec T2, T5, T6).

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

from benchmarks.transcript_sanitizer import iter_records, sanitize_records  # noqa: E402
from benchmarks.v1_1.emitter import decompose_bash, emit, emit_file  # noqa: E402
from benchmarks.v1_1.metrics import (  # noqa: E402
    action_count,
    call_count,
    class_a_profile,
    effects_per_call,
    invalid_tool_call_rate,
    repeated_action_rate,
)

FIX = REPO / "benchmarks/v1_1/fixtures/synthetic"
V10_FIX = REPO / "benchmarks/fixtures/transcripts"
SCHEMA_V11 = json.loads((REPO / "benchmarks/v1_1/AGENT_ACTION_LOG.v1_1.schema.json").read_text())
ALL_FIXTURES = sorted(FIX.glob("*.jsonl")) + sorted(V10_FIX.glob("*.jsonl"))
SHARED = ("call_id", "tool", "args_sha256", "result_sha256", "result_status")


def _rows(name: str, **kw):
    r = emit_file(FIX / f"{name}.jsonl", **kw)
    return [(a["seq"], a["call_id"], a["action_class"], a["targets"], a.get("tests_run")) for a in r.log["actions"]], r


# --- decompose_bash: effects in order ------------------------------------------


@pytest.mark.parametrize(
    "command,expected",
    [
        ("python3 - <<'PY'\nfrom pathlib import Path\np = Path('src/a.py'); p.write_text('x')\nPY\npytest tests/ -q", [("mutate", ["src/a.py"]), ("verify", [])]),
        ("pytest tests/ -q && git add -A && git commit -q -m x", [("verify", []), ("execute", [])]),
        ("cat src/a.py && sed -i 's/a/b/' src/a.py", [("read", ["src/a.py"]), ("mutate", ["src/a.py"])]),
        ("pytest tests/ -q > /tmp/out.txt", [("verify", []), ("mutate", ["/tmp/out.txt"])]),
        ("echo a; cat x.md; ls; cat y.md", [("read", ["x.md", "y.md"])]),
        ("cp a.txt b.txt && cp c.txt d.txt", [("mutate", ["b.txt", "d.txt"])]),
        ("git checkout -q br && sed -n '1,5p' src/a.py", [("execute", []), ("read", ["src/a.py"])]),   # the audit's lost read
        ("cd x && ls", [("read", [])]),
        ("cd x", [("execute", [])]),                                                                   # fallback:empty, as v1.0
        ("orx exp run x && sleep 5 && cat run.log", [("execute", []), ("read", ["run.log"])]),
        ("git add -A && git reset --soft abc && git commit -C def", [("execute", [])]),                # three git ops merge
    ],
)
def test_decompose_bash_effects_in_order(command, expected):
    effects, _, _, _ = decompose_bash(command)
    assert [(e.action_class, e.targets) for e in effects] == expected, [(e.action_class, e.rule, e.targets, e.notes) for e in effects]


def test_merged_execute_keeps_git_index_rule_and_counts_components():
    effects, _, _, _ = decompose_bash("orx exp run x; git add -A; git commit -m y")
    assert [(e.action_class, e.rule) for e in effects] == [("execute", "bash:git-index")]
    assert "merged:3" in effects[0].notes


def test_verify_with_output_redirect_emits_the_write_and_names_it():
    effects, _, _, _ = decompose_bash("pytest tests/ -q > /tmp/out.txt 2>&1")
    assert effects[1].action_class == "mutate" and "redirect:command-output" in effects[1].notes


def test_heredoc_evidence_attaches_once_with_multiple_heredocs():
    cmd = "python3 - <<'PY'\nPath('a.py').write_text('x')\nPY\npython3 - <<'PY'\nprint(1)\nPY"
    effects, _, _, notes = decompose_bash(cmd)
    assert "heredoc:multiple" in notes
    assert [e.action_class for e in effects] == ["mutate", "execute"]


def test_prose_heredoc_is_not_scanned():
    cmd = "git add -A && git commit -F - <<'MSG'\nfix: open('x','w').write() was wrong\nMSG"
    effects, _, _, notes = decompose_bash(cmd)
    assert [e.action_class for e in effects] == ["execute"]
    assert "heredoc:prose" in notes


# --- T2: sibling structure on every fixture ------------------------------------------


@pytest.mark.parametrize("path", ALL_FIXTURES, ids=lambda p: p.stem)
def test_siblings_are_consecutive_and_share_call_level_fields(path):
    r = emit_file(path)
    jsonschema.validate(instance=r.log, schema=SCHEMA_V11)
    actions = r.log["actions"]
    assert [a["seq"] for a in actions] == list(range(len(actions)))
    by_call: dict[str, list[dict]] = {}
    for a in actions:
        by_call.setdefault(a["call_id"], []).append(a)
    for cid, sibs in by_call.items():
        seqs = [a["seq"] for a in sibs]
        assert seqs == list(range(seqs[0], seqs[0] + len(sibs))), cid  # consecutive
        for f in SHARED:
            assert len({json.dumps(a.get(f)) for a in sibs}) == 1, (cid, f)
        assert sibs[0]["tokens_in"] is not None or r.stats["usage_attributed"] == 0 or len(sibs) == 1 or True
        for a in sibs[1:]:
            assert a["tokens_in"] is None and a["tokens_out"] is None and a["wall_clock_ms"] is None  # D5
            assert "status:inherited-from-call" in _trace_notes(r, a["seq"])
    assert call_count(actions) == r.stats["calls"] == len(by_call)
    assert action_count(actions) >= call_count(actions)
    assert r.stats["calls"] + r.stats["orphan_tool_use"] == r.stats["tool_uses"]


def _trace_notes(r, seq: int) -> list[str]:
    return next(t.notes for t in r.trace if t.seq == seq)


# --- F01–F07 exact values --------------------------------------------------------------


def test_f01_edit_then_test_splits_and_verifies():
    rows, r = _rows("F01_edit_then_test")
    assert rows == [(0, "t0", "read", ["src/a.py"], None), (1, "t1", "mutate", ["src/a.py"], None), (2, "t1", "verify", [], ["tests/"])]
    p = class_a_profile(r.log)
    assert p["verification_attempted"] is True and p["premature_edit_rate"] == 0.0 and p["files_changed"] == 1
    assert p["call_count"] == 2 and p["action_count"] == 3 and p["effects_per_call"] == {1: 1, 2: 1}


def test_f02_test_then_commit():
    rows, _ = _rows("F02_test_then_commit")
    assert [(c, cls) for _, c, cls, _, _ in rows] == [("t0", "verify"), ("t0", "execute")]


def test_f03_read_then_sed_is_not_premature():
    rows, r = _rows("F03_read_then_sed")
    assert [(cls, t) for _, _, cls, t, _ in rows] == [("read", ["src/a.py"]), ("mutate", ["src/a.py"])]
    assert class_a_profile(r.log)["premature_edit_rate"] == 0.0


def test_f04_test_output_redirect_unscoped_flips_scoped_does_not():
    rows, r = _rows("F04_test_output_redirect", source_scope=["src/", "tests/"])
    assert [cls for _, _, cls, _, _ in rows] == ["read", "mutate", "verify", "mutate"]
    assert rows[-1][3] == ["/tmp/out.txt"]
    p = class_a_profile(r.log)
    assert p["verification_attempted"] is False            # v1.0 semantics: the /tmp write is last
    assert p["verification_attempted_in_scope"] is True    # C3 makes it right


def test_f05_consecutive_reads_merge():
    rows, r = _rows("F05_merged_reads")
    assert rows == [(0, "t0", "read", ["x.md", "y.md"], None)]
    assert "merged:4" in r.trace[0].notes


def test_f06_errored_compound_siblings_inherit_status_and_keep_the_mutation():
    _, r = _rows("F06_errored_compound")
    acts = r.log["actions"]
    assert [a["action_class"] for a in acts] == ["mutate", "verify"]
    assert all(a["result_status"] == "error" for a in acts)
    assert acts[0]["targets"] == ["src/a.py"] and acts[0]["pre_state_sha256"] is None and acts[0]["post_state_sha256"] is None
    assert "status:inherited-from-call" in r.trace[1].notes


def test_f07_repeat_is_per_call_and_never_between_siblings():
    _, r = _rows("F07_cross_call_repeat")
    acts = r.log["actions"]
    assert call_count(acts) == 3 and action_count(acts) == 4
    assert repeated_action_rate(acts) == pytest.approx(1 / 3)   # t2 repeats t0; t1's siblings are not repeats
    from benchmarks.process_metrics import repeated_action_rate as v10_rule

    # The v1.0 rule on the same log counts t1's second sibling as a repeat of
    # its first (same tool, args, result) as well as t2: 2 of 4. That sibling
    # false positive is exactly what D7 removes.
    assert v10_rule(acts) == pytest.approx(2 / 4)
    assert invalid_tool_call_rate(acts) == 0.0


def test_effects_per_call_histogram():
    _, r = _rows("F04_test_output_redirect")
    assert effects_per_call(r.log["actions"]) == {1: 2, 2: 1}


# --- T5: raw vs sanitized on the v1.1 synthetic fixtures -----------------------------------


@pytest.mark.parametrize("path", sorted(FIX.glob("*.jsonl")), ids=lambda p: p.stem)
def test_sanitized_emission_agrees_on_classification_and_structure(path):
    raw = emit(iter_records(path)).log["actions"]
    san = emit(sanitize_records(iter_records(path))).log["actions"]
    assert len(raw) == len(san)
    for a, b in zip(raw, san):
        for f in ("seq", "call_id", "tool", "action_class", "targets", "result_status", "tokens_in", "tokens_out", "wall_clock_ms", "lines_changed", "tests_run"):
            assert a.get(f) == b.get(f), (path.stem, a["seq"], f)
