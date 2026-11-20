"""Generate `synthetic_minimal.jsonl` — a hand-built transcript with *real* fake
content, so exact-value emitter behaviour (post-state reconstruction, line
counts, timeouts, orphans) can be tested where sanitized fixtures cannot.

Regenerate:  python3 benchmarks/fixtures/transcripts/make_synthetic.py
The test suite asserts the checked-in file matches this generator's output.

Scenario (12 answered calls + 1 orphan), chosen to exercise one thing each:

  seq  tool   what                                   exercises
  0    Read   src/a.py                                read target
  1    Bash   pytest tests/ -q  (fails)               verify + error + tests_run
  2    Edit   a.py: x = 1 -> x = 2                    mutate, reconstruction, lines_changed
  3    Bash   pytest && git add && git commit         compound verify+git
  4    Edit   a.py: x = 2 -> x = 3                    second state
  5    Edit   a.py: x = 3 -> x = 2                    reversion to a seen state
  6    Bash   heredoc python writing src/b.py         mutate via heredoc, target extraction
  7    Edit   a.py with stale old_string              tool_error (agent misuse)
  8    Bash   cat src/b.py                            read after mutate (premature on b.py)
  9    Write  src/c.py (create)                       mutate, lines_changed from content
  10   Bash   grep -rn foo src/ 2>/dev/null | head    read-only pipeline, /dev/null ignored
  11   Bash   echo hi > /tmp/out.txt                  mutate outside repo (absolute target)
  12   Bash   sleep 100  (interrupted)                timeout
  -    Bash   orphan tool_use with no result          skipped, counted
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

CWD = "/repo"
T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
USAGE = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}

A0 = "x = 1\ny = 2\n"
A1 = "x = 2\ny = 2\n"
A2 = "x = 3\ny = 2\n"
C_CONTENT = "print(1)\n"


def _ts(i: int) -> str:
    return (T0 + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def records() -> list[dict]:
    out: list[dict] = []
    clock = [0]
    uuid_n = [0]

    def nxt():
        uuid_n[0] += 1
        return f"u{uuid_n[0]:03d}"

    def assistant(tool: str, tool_input: dict, tid: str) -> None:
        out.append({
            "type": "assistant", "uuid": nxt(), "parentUuid": out[-1]["uuid"] if out else None,
            "timestamp": _ts(clock[0]), "cwd": CWD, "gitBranch": "main", "isSidechain": False,
            "message": {"role": "assistant", "model": "claude-synthetic", "usage": dict(USAGE),
                        "content": [{"type": "tool_use", "id": tid, "name": tool, "input": tool_input}]},
        })
        clock[0] += 2

    def result(tid: str, content: str, tur, *, is_error: bool = False) -> None:
        rec = {
            "type": "user", "uuid": nxt(), "parentUuid": out[-1]["uuid"], "timestamp": _ts(clock[0]),
            "cwd": CWD, "isSidechain": False,
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": content, "is_error": is_error}]},
        }
        if tur is not None:
            rec["toolUseResult"] = tur
        out.append(rec)
        clock[0] += 1

    def bash_result(stdout: str, *, interrupted: bool = False) -> dict:
        return {"stdout": stdout, "stderr": "", "interrupted": interrupted, "isImage": False, "noOutputExpected": False}

    def edit_result(original: str, old: str, new: str, patch_lines: list[str]) -> dict:
        return {"filePath": f"{CWD}/src/a.py", "oldString": old, "newString": new, "originalFile": original,
                "replaceAll": False, "userModified": False,
                "structuredPatch": [{"oldStart": 1, "oldLines": 2, "newStart": 1, "newLines": 2, "lines": patch_lines}]}

    out.append({"type": "user", "uuid": nxt(), "parentUuid": None, "timestamp": _ts(clock[0]), "cwd": CWD,
                "isSidechain": False, "message": {"role": "user", "content": "fix the bug in a.py"}})
    clock[0] += 1

    assistant("Read", {"file_path": f"{CWD}/src/a.py"}, "t00")
    result("t00", "1\tx = 1\n2\ty = 2\n", {"type": "text", "file": {"filePath": f"{CWD}/src/a.py", "content": A0, "numLines": 2, "startLine": 1, "totalLines": 2}})

    assistant("Bash", {"command": "pytest tests/ -q", "description": "Run tests"}, "t01")
    result("t01", "Exit code 1\n1 failed", bash_result("1 failed"), is_error=True)

    assistant("Edit", {"file_path": f"{CWD}/src/a.py", "old_string": "x = 1", "new_string": "x = 2", "replace_all": False}, "t02")
    result("t02", "The file /repo/src/a.py has been updated.", edit_result(A0, "x = 1", "x = 2", ["-x = 1", "+x = 2", " y = 2"]))

    assistant("Bash", {"command": "pytest tests/ -q && git add -A && git commit -q -m 'fix'"}, "t03")
    result("t03", "1 passed", bash_result("1 passed"))

    assistant("Edit", {"file_path": f"{CWD}/src/a.py", "old_string": "x = 2", "new_string": "x = 3", "replace_all": False}, "t04")
    result("t04", "The file /repo/src/a.py has been updated.", edit_result(A1, "x = 2", "x = 3", ["-x = 2", "+x = 3", " y = 2"]))

    assistant("Edit", {"file_path": f"{CWD}/src/a.py", "old_string": "x = 3", "new_string": "x = 2", "replace_all": False}, "t05")
    result("t05", "The file /repo/src/a.py has been updated.", edit_result(A2, "x = 3", "x = 2", ["-x = 3", "+x = 2", " y = 2"]))

    assistant("Bash", {"command": "python3 - <<'PY'\nfrom pathlib import Path\np = Path('src/b.py'); p.write_text('z = 0\\n')\nPY"}, "t06")
    result("t06", "", bash_result(""))

    assistant("Edit", {"file_path": f"{CWD}/src/a.py", "old_string": "nope", "new_string": "x = 9", "replace_all": False}, "t07")
    result("t07", "<tool_use_error>String to replace not found in file.\nString: nope", "String to replace not found in file.", is_error=True)

    assistant("Bash", {"command": "cat src/b.py"}, "t08")
    result("t08", "z = 0", bash_result("z = 0"))

    assistant("Write", {"file_path": f"{CWD}/src/c.py", "content": C_CONTENT}, "t09")
    result("t09", "File created successfully at: /repo/src/c.py",
           {"type": "create", "filePath": f"{CWD}/src/c.py", "content": C_CONTENT, "structuredPatch": [], "userModified": False})

    assistant("Bash", {"command": "grep -rn foo src/ 2>/dev/null | head -5"}, "t10")
    result("t10", "", bash_result(""))

    assistant("Bash", {"command": "echo hi > /tmp/out.txt"}, "t11")
    result("t11", "", bash_result(""))

    assistant("Bash", {"command": "sleep 100"}, "t12")
    result("t12", "Command timed out", bash_result("", interrupted=True), is_error=True)

    assistant("Bash", {"command": "ls"}, "t13")  # orphan: no result follows
    return out


def render() -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records())


if __name__ == "__main__":
    target = Path(__file__).with_name("synthetic_minimal.jsonl")
    target.write_text(render())
    print(f"wrote {target} ({len(records())} records)")
