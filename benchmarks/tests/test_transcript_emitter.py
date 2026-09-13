"""Transcript emitter tests.

Three layers, in order of how much they pin:

1. `classify_bash` table — the aggressive `action_class` tests the owner asked
   for. Bash is a compound tool and the schema says its class is declared,
   unverifiable, and therefore the weakest assumption. Every row here is a
   command shape observed in this project's own sessions or a near-miss that a
   naive classifier gets wrong (`echo pytest`, `ls test_data/`, a heredoc
   followed by pytest).
2. Synthetic fixture — exact values where the fixture has real fake content.
3. Sanitized fixtures — structural smoke: every emitted log validates, every
   trace line has a rule, no tool is unmapped.

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

from benchmarks.process_metrics import class_a_profile  # noqa: E402
from benchmarks.transcript_emitter import (  # noqa: E402
    TOOL_CLASS,
    classify_bash,
    emit,
    emit_file,
    relativize,
    result_status,
)

FIXTURES = REPO / "benchmarks/fixtures/transcripts"
SYNTHETIC = FIXTURES / "synthetic_minimal.jsonl"
SESSION_FIXTURES = sorted(FIXTURES.glob("session_*.jsonl"))
SCHEMA = json.loads((REPO / "benchmarks/AGENT_ACTION_LOG.schema.json").read_text())
ACTION_CLASSES = set(SCHEMA["$defs"]["action"]["properties"]["action_class"]["enum"])


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# --- 1. classify_bash: the aggressive table ------------------------------------

VERIFY = [
    ("pytest benchmarks/ -q", ["benchmarks/"]),
    ('python3 -m pytest benchmarks/ -o addopts="" --tb=short -p no:cacheprovider 2>&1 | tail -3', ["benchmarks/"]),
    ("python3 fabric/test_fabric.py", ["fabric/test_fabric.py"]),
    ('cd "$(git rev-parse --show-toplevel)" && python3 -m pytest benchmarks/ -q 2>&1 | tail -2', ["benchmarks/"]),
    (".venv/bin/python3 -m pytest -q tests/", ["tests/"]),
    (".venv/bin/pytest tests/test_x.py::test_a", ["tests/test_x.py::test_a"]),
    ("PYTHONPATH=. pytest x/", ["x/"]),
    ("python -m unittest discover", []),
    ("npm test", []),
    ("yarn run test", []),
    ("make test", []),
    ("make check", []),
    ("cargo test", []),
    ("go test ./...", []),  # tests_run extraction is pytest-shaped by design
    ("pytest x || true", ["x"]),
    ("python3 thing_test.py", ["thing_test.py"]),
    # runner-in-program-position must survive prefixes and env
    ("cd /x && FOO=1 pytest -q", []),
]


@pytest.mark.parametrize("command,tests_run", VERIFY)
def test_verify_commands(command, tests_run):
    c = classify_bash(command)
    assert c["action_class"] == "verify", (command, c)
    assert c["rule"] == "bash:test-runner"
    assert c["tests_run"] == tests_run, (command, c["tests_run"])


NOT_VERIFY = [
    "echo pytest",                       # runner name in an argument, not program position
    "ls test_data/",                     # test_ prefix on a directory
    "cat tests/test_x.py",               # reading a test file is not running it
    "grep -rn pytest benchmarks/",
    "python3 -m benchmarks.adapters.lm_eval.cli smoke --mode mock",
    "git log --grep=test",
    "pip install pytest",
]


@pytest.mark.parametrize("command", NOT_VERIFY)
def test_not_verify_commands(command):
    assert classify_bash(command)["action_class"] != "verify", command


def test_verify_outranks_write_in_a_compound_and_says_so():
    """Documented precedence: a target-less mutate would move the last-mutation
    pointer and turn a call that did run tests into a false 'shipped unverified'."""
    c = classify_bash("cp a.py /tmp/a.bak && python3 -m pytest x/ -q")
    assert c["action_class"] == "verify"
    assert "compound:verify+write" in c["notes"]
    assert c["compound"] is True


def test_verify_outranks_git_in_a_compound():
    c = classify_bash("pytest tests/ -q && git add -A && git commit -q -m 'fix'")
    assert c["action_class"] == "verify"
    assert "compound:verify+git" in c["notes"]


def test_heredoc_then_pytest_is_verify_not_mutate():
    """The bug the first splitter had: everything after the first newline was
    'body', so a pytest after the terminator was invisible."""
    cmd = "python3 - <<'PY'\nfrom pathlib import Path\nPath('a.py').write_text('x')\nPY\npytest tests/ -q"
    c = classify_bash(cmd)
    assert c["action_class"] == "verify", c
    assert "heredoc:write-evidence" in c["notes"]
    assert "compound:verify+write" in c["notes"]
    assert c["tests_run"] == ["tests/"]


def test_heredoc_body_operators_are_not_segmented():
    """`;` and `|` inside a Python heredoc are Python, not shell."""
    cmd = "python3 - <<'PY'\nx = 1; y = 2 | 3\nprint(x)\nPY"
    c = classify_bash(cmd)
    assert c["segments"] == 1
    assert c["action_class"] == "execute"  # no write evidence, not a test, not read-only


MUTATE = [
    ("sed -i 's/a/b/' benchmarks/x.py", ["benchmarks/x.py"]),
    ("cp a b", ["b"]),
    ("mv a b", ["b", "a"]),
    ("rm -rf /tmp/x", ["/tmp/x"]),
    ("mkdir -p out/dir", ["out/dir"]),
    ("touch a.txt", ["a.txt"]),
    ("echo hi > out.txt", ["out.txt"]),
    ("printf 'x' >> log.txt", ["log.txt"]),
    ("cat > file.md <<'EOF'\nbody\nEOF", ["file.md"]),
    ("python3 - <<'PY'\nimport pathlib\npathlib.Path('a/b.py').write_text(t)\nPY", ["a/b.py"]),
    ("python3 - <<'PY'\nopen('x.txt', 'w').write('a')\nPY", ["x.txt"]),
    ("python3 -c \"open('x','w').write('1')\"", ["x"]),
    ("cat a.py; rm a.py", ["a.py"]),
    ("git add -A && cp x y", ["y"]),
    ("tee out.txt", ["out.txt"]),
    ("cp benchmarks/gold_metrics.py /tmp/gm.bak && cp benchmarks/process_metrics.py /tmp/pm.bak", ["/tmp/gm.bak", "/tmp/pm.bak"]),
]


@pytest.mark.parametrize("command,targets", MUTATE)
def test_mutate_commands(command, targets):
    c = classify_bash(command)
    assert c["action_class"] == "mutate", (command, c)
    assert c["rule"] == "bash:file-write"
    assert c["targets"] == targets, (command, c["targets"])


def test_write_outranks_git_and_notes_it():
    c = classify_bash("git add -A && cp x y")
    assert "compound:write+git" in c["notes"]


def test_heredoc_write_without_path_literal_is_mutate_with_no_target():
    c = classify_bash("python3 - <<'PY'\nfor p in paths: p.write_text('x')\nPY")
    assert c["action_class"] == "mutate"
    assert c["targets"] == []
    assert "mutate:no-target" in c["notes"]


READ = [
    ("cat benchmarks/README.md", ["benchmarks/README.md"]),
    ("grep -rn foo src/ 2>/dev/null | head -5", ["src/"]),
    ("wc -l a.py b.py 2>/dev/null", ["a.py", "b.py"]),
    ("sed -n '10,20p' x.py", ["x.py"]),
    ("echo '=== x ===' && cat a.md", ["a.md"]),
    ("cd /x && ls", []),
    ("ls -la && git log --oneline -5", []),
    ("git status --porcelain && git diff --stat master..HEAD", []),
    ("git stash list", []),
    ("git branch -a", []),
    ("git branch", []),
    ("git tag -l", []),
    ("git worktree list", []),
    ("git show --stat abc1234 | head", []),
    ('python3 -c "import json; print(1)"', []),
    ("find . -name '*.py' | xargs wc -l", []),
    ("diff a b", []),  # bare names without / or . are not recognised as paths
    ("cat a.py 2>&1", ["a.py"]),
    ("head -c 400 pretrain/tokenizer.json; echo; wc -c pretrain/tokenizer.json", ["pretrain/tokenizer.json"]),
]


@pytest.mark.parametrize("command,targets", READ)
def test_read_commands(command, targets):
    c = classify_bash(command)
    assert c["action_class"] == "read", (command, c)
    assert c["rule"] == "bash:read-only"
    assert c["targets"] == targets, (command, c["targets"])


def test_read_target_extraction_ignores_redirect_targets():
    """`/dev/null` leaked in as a read target on the first synthetic run."""
    assert "/dev/null" not in classify_bash("grep -rn foo src/ 2>/dev/null | head")["targets"]
    assert classify_bash("cat a.py > /dev/null")["action_class"] == "read"


GIT_INDEX = [
    "git add -A && git commit -m x",
    "git reset --soft abc && git commit -C def",
    "git switch -c br",
    "git stash push -u -m x",
    "git stash",
    "git worktree add ../x",
    "git tag v1",
    "git branch -D old",
    "git checkout -- file",
]


@pytest.mark.parametrize("command", GIT_INDEX)
def test_git_index_operations_are_execute_not_mutate(command):
    c = classify_bash(command)
    assert c["action_class"] == "execute", (command, c)
    assert c["rule"] == "bash:git-index"
    assert c["targets"] == []


FALLBACK = [
    "python3 scripts/run_thing.py",
    "sleep 100",
    "make build",
    "npm install",
    "docker run x",
    "~/.claude/bin/loop-state init --objective x",
    "python3 -m benchmarks.adapters.lm_eval.cli smoke --mode mock",
    "",
]


@pytest.mark.parametrize("command", FALLBACK)
def test_fallback_is_execute_and_names_the_unmatched_program(command):
    c = classify_bash(command)
    assert c["action_class"] == "execute", (command, c)
    assert c["rule"] == "bash:fallback"
    assert any(n.startswith("fallback:") for n in c["notes"])


# --- multi-line, control flow, wrappers: the shapes real sessions actually use ----


MULTILINE = [
    # newline is a separator; a leading `cd` line must not swallow the rest
    ("cd ~/proj\ngit add a.py b.py\ngit -c user.name=x commit -q -m 'msg'", "execute", "bash:git-index", []),
    ("cd ~/proj\necho \"=== a ===\"\ncat a.md\nsed -n '1,5p' b.md", "read", "bash:read-only", ["a.md", "b.md"]),
    ("cd ~/proj\ngit show abc:artifacts/x.md > artifacts/x.md", "mutate", "bash:file-write", ["artifacts/x.md"]),
    # heredoc opener on line 2, read-only body -> investigation
    ("cd ~/proj\npython3 - <<'PY'\nimport json, glob\ng = json.load(open(\"artifacts/s.json\"))\nprint(g)\nPY", "read", "bash:read-only", ["artifacts/s.json"]),
    # heredoc opener on line 2, write body, then a test after the terminator
    ("cd ~/proj\npython3 - <<'PY'\nfrom pathlib import Path\nPath('a.py').write_text('x')\nPY\npytest tests/ -q", "verify", "bash:test-runner", []),
    # backslash continuation joins lines
    ("for row in \"a|b\" \\\n           \"c|d\"; do echo $row; done", "read", "bash:read-only", []),
    # control flow: loop body is the action
    ("for f in a.py b.py; do cat $f; done", "read", "bash:read-only", []),
    ("for f in a b; do rm -f \"$f\"; done", "mutate", "bash:file-write", []),
    ("if [ -f x.md ]; then cat x.md; fi", "read", "bash:read-only", ["x.md"]),
    ("while read -r l; do echo \"$l\"; done < list.txt", "read", "bash:read-only", []),
    ("{ echo a; echo b; } > out.txt", "mutate", "bash:file-write", ["out.txt"]),
    ("if grep -q x f.txt; then pytest t/; fi", "verify", "bash:test-runner", []),
    # wrappers
    ("timeout 300 python3 -m pytest tests/a.py -q", "verify", "bash:test-runner", []),
    ("find . -name '*.pyc' | xargs rm", "mutate", "bash:file-write", []),
    ("FOO=1 nice -n 5 cat a.py", "read", "bash:read-only", ["a.py"]),
    ("env PYTHONPATH=. python3 -m pytest x/", "verify", "bash:test-runner", []),
    # python -c with read evidence and a literal
    ("python3 -c \"import json; print(json.load(open('a.json')))\"", "read", "bash:read-only", ["a.json"]),
    # system-information programs are reads
    ("sysctl -n hw.memsize | awk '{print $1}'; df -h .; ps aux | head", "read", "bash:read-only", []),
    # subshells, arithmetic, expansion fragments, loop-control builtins: no world effect
    ("( cd sub && ls )", "read", "bash:read-only", []),
    ("n=0; ((n++)); echo $n", "read", "bash:read-only", []),
    ("rest=${row#*|}; br=${row%%|*}; echo \"$br\"", "read", "bash:read-only", []),
    ("for f in a b; do [ -f \"$f\" ] || continue; cat \"$f\"; done", "read", "bash:read-only", []),
    ("( cd sub && rm -f x.txt )", "mutate", "bash:file-write", ["x.txt"]),
]


@pytest.mark.parametrize("command,cls,rule,targets", MULTILINE, ids=[m[0][:40] for m in MULTILINE])
def test_multiline_control_flow_and_wrappers(command, cls, rule, targets):
    c = classify_bash(command)
    assert c["action_class"] == cls, (command, c)
    assert c["rule"] == rule, (command, c)
    assert c["targets"] == targets, (command, c["targets"])


def test_read_only_patterns_extend_the_default():
    import re

    cmd = "orx experiment list --project x 2>&1 | head -60"
    assert classify_bash(cmd)["action_class"] == "execute"
    extra = (re.compile(r"^orx (experiment list|project view|runs|logs)\b"),)
    assert classify_bash(cmd, read_only_patterns=extra)["action_class"] == "read"


def test_fd_digit_of_a_redirection_is_not_a_test_target():
    c = classify_bash("python3 -m pytest benchmarks/ -q 2>&1 | tail -2")
    assert c["tests_run"] == ["benchmarks/"]


def test_heredoc_with_neither_read_nor_write_evidence_is_execute():
    c = classify_bash("python3 - <<'PY'\nimport math\nprint(math.log(2))\nPY")
    assert c["action_class"] == "execute"
    assert "heredoc" in c["notes"]
    assert not any(n.startswith("heredoc:") and "evidence" in n for n in c["notes"])


def test_unbalanced_quotes_fall_back_to_whitespace_split_and_say_so():
    c = classify_bash("echo 'unbalanced")
    assert "shlex:fallback-whitespace-split" in c["notes"]
    assert c["action_class"] == "read"


def test_unterminated_heredoc_is_noted():
    c = classify_bash("python3 - <<'PY'\nprint(1)\n")
    assert "heredoc:unterminated" in c["notes"]


def test_caller_supplied_verify_patterns_extend_the_default():
    import re

    cmd = "python3 scripts/check_docs_integrity.py"
    assert classify_bash(cmd)["action_class"] == "execute"
    extra = (re.compile(r"scripts/check_\w+\.py"),)
    assert classify_bash(cmd, verify_patterns=extra)["action_class"] == "verify"


def test_every_bash_class_is_in_the_schema_enum():
    for cmd, *_ in VERIFY + MUTATE + READ:
        assert classify_bash(cmd)["action_class"] in ACTION_CLASSES
    for cmd in NOT_VERIFY + GIT_INDEX + FALLBACK:
        assert classify_bash(cmd)["action_class"] in ACTION_CLASSES


def test_tool_class_table_only_uses_schema_classes():
    for name, (cls, rule, _) in TOOL_CLASS.items():
        assert cls in ACTION_CLASSES, name
        assert rule.startswith("tool:"), name


# --- result status ------------------------------------------------------------


@pytest.mark.parametrize(
    "block,tur,expected",
    [
        ({"content": "ok"}, {"interrupted": False}, "ok"),
        ({"content": "Exit code 1\nboom", "is_error": True}, {"interrupted": False}, "error"),
        ({"content": "File does not exist.", "is_error": True}, None, "error"),
        ({"content": "<tool_use_error>String to replace not found", "is_error": True}, "x", "tool_error"),
        ({"content": "Output does not match required schema", "is_error": True}, None, "tool_error"),
        ({"content": "InputValidationError: bad", "is_error": True}, None, "tool_error"),
        ({"content": "Command timed out", "is_error": True}, {"interrupted": True}, "timeout"),
        ({"content": [{"type": "text", "text": "x"}]}, None, "ok"),
    ],
)
def test_result_status(block, tur, expected):
    assert result_status(block, tur) == expected


def test_relativize_keeps_out_of_repo_paths_visibly_absolute():
    assert relativize("/repo/src/a.py", "/repo") == "src/a.py"
    assert relativize("/repo", "/repo") == "."
    assert relativize("/tmp/out.txt", "/repo") == "/tmp/out.txt"
    assert relativize("/repo2/x", "/repo") == "/repo2/x"  # prefix, not path component


# --- 2. synthetic fixture: exact values ----------------------------------------


@pytest.fixture(scope="module")
def synthetic():
    return emit_file(SYNTHETIC)


def test_synthetic_fixture_matches_its_generator():
    sys.path.insert(0, str(FIXTURES))
    import make_synthetic  # noqa: E402

    assert SYNTHETIC.read_text() == make_synthetic.render(), "regenerate with make_synthetic.py"


def test_synthetic_action_table(synthetic):
    rows = [(a["seq"], a["tool"], a["action_class"], a["result_status"], t.rule, a["targets"])
            for a, t in zip(synthetic.log["actions"], synthetic.trace)]
    assert rows == [
        (0, "Read", "read", "ok", "tool:Read", ["src/a.py"]),
        (1, "Bash", "verify", "error", "bash:test-runner", []),
        (2, "Edit", "mutate", "ok", "tool:Edit", ["src/a.py"]),
        (3, "Bash", "verify", "ok", "bash:test-runner", []),
        (4, "Edit", "mutate", "ok", "tool:Edit", ["src/a.py"]),
        (5, "Edit", "mutate", "ok", "tool:Edit", ["src/a.py"]),
        (6, "Bash", "mutate", "ok", "bash:file-write", ["src/b.py"]),
        (7, "Edit", "mutate", "tool_error", "tool:Edit", ["src/a.py"]),
        (8, "Bash", "read", "ok", "bash:read-only", ["src/b.py"]),
        (9, "Write", "mutate", "ok", "tool:Write", ["src/c.py"]),
        (10, "Bash", "read", "ok", "bash:read-only", ["src/"]),
        (11, "Bash", "mutate", "ok", "bash:file-write", ["/tmp/out.txt"]),
        (12, "Bash", "execute", "timeout", "bash:fallback", []),
    ]


def test_synthetic_orphan_is_counted_not_emitted(synthetic):
    assert synthetic.stats["tool_uses"] == 14
    assert synthetic.stats["answered"] == 13
    assert synthetic.stats["orphan_tool_use"] == 1
    assert len(synthetic.log["actions"]) == 13


def test_synthetic_post_state_reconstruction_and_reversion(synthetic):
    a = {x["seq"]: x for x in synthetic.log["actions"]}
    assert a[2]["post_state_sha256"] == _sha("x = 2\ny = 2\n")
    assert a[4]["post_state_sha256"] == _sha("x = 3\ny = 2\n")
    assert a[5]["post_state_sha256"] == a[2]["post_state_sha256"]  # back to a seen state
    assert a[7]["post_state_sha256"] is None  # tool_error: no mutation happened
    assert a[9]["post_state_sha256"] == _sha("print(1)\n")
    assert not any("post_state:derived" in t.notes for t in synthetic.trace)


def test_synthetic_lines_changed_and_tests_run(synthetic):
    a = {x["seq"]: x for x in synthetic.log["actions"]}
    assert a[2]["lines_changed"] == 2
    assert a[9]["lines_changed"] == 1
    assert "lines_changed" not in a[7]
    assert a[1]["tests_run"] == ["tests/"]
    assert a[3]["tests_run"] == ["tests/"]
    assert "tests_run" not in a[2]


def test_synthetic_token_and_timing_attribution(synthetic):
    for a in synthetic.log["actions"]:
        assert a["tokens_in"] == 1100  # input + cache_read + cache_creation
        assert a["tokens_out"] == 20
        assert a["wall_clock_ms"] == 2000
        assert a["cost_usd"] is None


def test_synthetic_log_validates_against_schema(synthetic):
    jsonschema.validate(instance=synthetic.log, schema=SCHEMA)


def test_synthetic_class_a_profile_is_the_hand_computed_one(synthetic):
    p = class_a_profile(synthetic.log)
    # mutated: a.py (read at 0 ✓), b.py (read only after, at 8), c.py (never), /tmp/out.txt (never)
    assert p["premature_edit_rate"] == 0.75
    assert p["reproduction_before_repair"] is True   # pytest at 1 precedes first mutate at 2
    assert p["verification_attempted"] is False      # last mutate is 11, nothing verifies after
    assert p["invalid_tool_call_rate"] == pytest.approx(1 / 13)
    assert p["reversion_count"] == 1
    assert p["files_changed"] == 4 and p["files_inspected"] == 3  # a.py, b.py, src/


def test_synthetic_reversion_to_initial_state_is_invisible_by_design(synthetic):
    """Documented limitation for the contract review: the log carries only
    post-states, so returning a file to its *initial* content (never a
    post-state) is not a reversion the metric can see. seq 2 does exactly that
    in reverse — the initial content is never recorded."""
    initial = _sha("x = 1\ny = 2\n")
    assert initial not in {a["post_state_sha256"] for a in synthetic.log["actions"]}


def test_claimed_complete_and_declared_scope_are_never_emitted(synthetic):
    assert "declared_scope" not in synthetic.log
    assert not any("claimed_complete" in a for a in synthetic.log["actions"])


def test_emit_is_pure_and_repeatable():
    a = emit_file(SYNTHETIC).log
    b = emit_file(SYNTHETIC).log
    assert a == b


# --- 3. sanitized fixtures: structural smoke -----------------------------------


def test_session_fixtures_exist():
    assert len(SESSION_FIXTURES) >= 3, [p.name for p in SESSION_FIXTURES]


@pytest.mark.parametrize("path", SESSION_FIXTURES, ids=lambda p: p.stem)
def test_session_fixture_emits_a_valid_log(path):
    r = emit_file(path)
    jsonschema.validate(instance=r.log, schema=SCHEMA)
    assert len(r.trace) == len(r.log["actions"])
    assert r.stats["answered"] == len(r.log["actions"])
    assert r.stats["answered"] + r.stats["orphan_tool_use"] == r.stats["tool_uses"]
    assert r.stats["unknown_tools"] == {}, r.stats["unknown_tools"]
    assert set(r.stats["by_class"]) <= ACTION_CLASSES
    for t in r.trace:
        assert t.rule
        assert t.action_class in ACTION_CLASSES


@pytest.mark.parametrize("path", SESSION_FIXTURES, ids=lambda p: p.stem)
def test_session_fixture_seq_is_dense_and_ordered(path):
    seqs = [a["seq"] for a in emit_file(path).log["actions"]]
    assert seqs == list(range(len(seqs)))


def test_large_fixture_derived_post_states_are_edits_only():
    """On sanitized fixtures Edit reconstruction cannot succeed (both sides are
    markers); Write can. Pin that the fallback fires only where expected."""
    big = [p for p in SESSION_FIXTURES if "large" in p.name]
    if not big:
        pytest.skip("no large fixture")
    r = emit_file(big[0])
    for t in r.trace:
        if "post_state:derived" in t.notes:
            assert t.tool in ("Edit", "MultiEdit"), t


# --- 4. characterization: pinned distributions -----------------------------------
#
# These pin the classifier's *current* behaviour on the sanitized fixtures. They
# are not correctness claims — the manual audit (validation experiment, step 7)
# is where correctness is judged. Their job is to make any classifier change
# visible: if a rule count moves, re-audit and update the pin deliberately.

EXPECTED_BY_RULE = {
    "session_large_varied": {
        "bash:fallback": 29, "bash:file-write": 37, "bash:git-index": 16, "bash:read-only": 100,
        "bash:test-runner": 38, "tool:Edit": 15, "tool:ListAgents": 2, "tool:Read": 5, "tool:Skill": 3,
        "tool:ToolSearch": 3, "tool:Write": 12, "tool:message-passing": 7, "tool:peer-discovery": 2,
    },
    "session_small_errors": {"bash:read-only": 4, "tool:Glob": 1, "tool:Grep": 2, "tool:Read": 5, "tool:StructuredOutput": 2},
    "session_small_reads": {"bash:read-only": 1, "tool:Read": 8, "tool:StructuredOutput": 1},
    "session_trivial": {"tool:StructuredOutput": 1},
}

#: What still falls through on the large fixture, by program. Every entry is a
#: genuine external program or script with no read/write evidence — the honest
#: meaning of `execute`. A new name here, or one disappearing, is a classifier
#: change to look at.
EXPECTED_FALLBACK_PROGRAMS = {"python3", "orx", "sleep", "attach", "check", "peer-claims", "kill"}


@pytest.mark.parametrize("name", sorted(EXPECTED_BY_RULE), ids=lambda n: n)
def test_characterization_by_rule(name):
    path = FIXTURES / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{name} not present")
    got = emit_file(path).stats["by_rule"]
    assert got == EXPECTED_BY_RULE[name], (
        f"classifier behaviour changed on {name}; re-audit before updating this pin"
    )


def test_characterization_fallback_residue_is_all_named_programs():
    r = emit_file(FIXTURES / "session_large_varied.jsonl")
    programs = set()
    for t in r.trace:
        for n in t.notes:
            if n.startswith("fallback:"):
                programs.update(p for p in n.split(":", 1)[1].split(",") if p)
    assert programs == EXPECTED_FALLBACK_PROGRAMS, programs
    assert "empty" not in programs  # the multi-line/cd bug, fixed
