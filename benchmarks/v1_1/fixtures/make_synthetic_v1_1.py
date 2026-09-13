"""Generate the fourteen v1.1 semantic fixtures (migration spec §5.1, F1–F14).

One transcript file per scenario under `synthetic/`, each with real fake content
so exact values can be asserted. The v1.0 synthetic fixture is untouched.

Regenerate:  python3 benchmarks/v1_1/fixtures/make_synthetic_v1_1.py
A test asserts every checked-in file matches this generator's output.

  id   scenario                                                    exercises
  F01  heredoc edits src/a.py (Path literal) then pytest, one call  C1 split [mutate, verify]
  F02  pytest && git add && git commit                              C1 [verify, execute(git-index)]
  F03  cat src/a.py && sed -i … src/a.py                            C1 [read, mutate]; not premature
  F04  pytest tests/ > /tmp/out.txt                                 C1×C3 [verify, mutate(/tmp)]; scoped vs unscoped
  F05  echo a; cat x.md; ls; cat y.md                               D3b one merged read, targets [x.md, y.md]
  F06  errored call: heredoc write then pytest                      D6 siblings inherit `error`
  F07  same read command in two calls, same result                  D7 cross-call repeat; siblings never
  F08  Edit A→B, Edit B→A                                           C2 reversion 1 (v1.0 sees 0)
  F09  Edit A→A twice                                               D10 noop 2, reversions 0
  F10  Write create C1, update C2, update back to C1                C2 pre null on create; reversion via pre-state
  F11  cp a b && cp c d                                             D9 merged mutate, two targets, pre/post null
  F12  verify, then Write artifacts/report.md, end                  C3 unscoped false, in_scope true
  F13  verify, then target-less heredoc mutate, end                 D14 scoped ignores; mutations_without_target 1
  F14  mutate docs/x.md only, no verify                             C3 in_scope None (no in-scope mutation)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

CWD = "/repo"
T0 = datetime(2026, 2, 1, 9, 0, 0, tzinfo=timezone.utc)
USAGE = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}

A0, A1, A2 = "x = 1\ny = 2\n", "x = 2\ny = 2\n", "x = 3\ny = 2\n"
C1, C2 = "print(1)\n", "print(2)\n"


class Session:
    def __init__(self) -> None:
        self.out: list[dict] = []
        self.clock = 0
        self.n = 0

    def _uuid(self) -> str:
        self.n += 1
        return f"u{self.n:03d}"

    def _ts(self) -> str:
        return (T0 + timedelta(seconds=self.clock)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def prompt(self, text: str) -> None:
        self.out.append({"type": "user", "uuid": self._uuid(), "parentUuid": self.out[-1]["uuid"] if self.out else None,
                         "timestamp": self._ts(), "cwd": CWD, "isSidechain": False,
                         "message": {"role": "user", "content": text}})
        self.clock += 1

    def call(self, tool: str, tool_input: dict, tid: str, content: str, tur, *, is_error: bool = False) -> None:
        self.out.append({"type": "assistant", "uuid": self._uuid(), "parentUuid": self.out[-1]["uuid"] if self.out else None,
                         "timestamp": self._ts(), "cwd": CWD, "gitBranch": "main", "isSidechain": False,
                         "message": {"role": "assistant", "model": "claude-synthetic", "usage": dict(USAGE),
                                     "content": [{"type": "tool_use", "id": tid, "name": tool, "input": tool_input}]}})
        self.clock += 2
        rec = {"type": "user", "uuid": self._uuid(), "parentUuid": self.out[-1]["uuid"], "timestamp": self._ts(),
               "cwd": CWD, "isSidechain": False,
               "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": content, "is_error": is_error}]}}
        if tur is not None:
            rec["toolUseResult"] = tur
        self.out.append(rec)
        self.clock += 1

    # -- tool-specific helpers ------------------------------------------------
    def bash(self, tid: str, command: str, stdout: str = "", *, is_error: bool = False, content: str | None = None) -> None:
        self.call("Bash", {"command": command}, tid, content if content is not None else stdout,
                  {"stdout": stdout, "stderr": "", "interrupted": False, "isImage": False, "noOutputExpected": False}, is_error=is_error)

    def edit(self, tid: str, path: str, original: str, old: str, new: str) -> None:
        self.call("Edit", {"file_path": f"{CWD}/{path}", "old_string": old, "new_string": new, "replace_all": False}, tid,
                  f"The file {CWD}/{path} has been updated.",
                  {"filePath": f"{CWD}/{path}", "oldString": old, "newString": new, "originalFile": original, "replaceAll": False,
                   "userModified": False,
                   "structuredPatch": [{"oldStart": 1, "oldLines": 2, "newStart": 1, "newLines": 2, "lines": [f"-{old}", f"+{new}", " y = 2"]}]})

    def write(self, tid: str, path: str, content: str, original: str | None) -> None:
        tur = {"type": "update" if original is not None else "create", "filePath": f"{CWD}/{path}", "content": content,
               "structuredPatch": [] if original is None else [{"oldStart": 1, "oldLines": 1, "newStart": 1, "newLines": 1, "lines": ["-old", "+new"]}],
               "userModified": False}
        if original is not None:
            tur["originalFile"] = original
        self.call("Write", {"file_path": f"{CWD}/{path}", "content": content}, tid,
                  f"File {'updated' if original is not None else 'created'} successfully at: {CWD}/{path}", tur)

    def read(self, tid: str, path: str, content: str) -> None:
        self.call("Read", {"file_path": f"{CWD}/{path}"}, tid, content,
                  {"type": "text", "file": {"filePath": f"{CWD}/{path}", "content": content, "numLines": content.count("\n"), "startLine": 1, "totalLines": content.count("\n")}})


HEREDOC_EDIT = "python3 - <<'PY'\nfrom pathlib import Path\np = Path('src/a.py'); s = p.read_text(); p.write_text(s.replace('x = 1', 'x = 2'))\nPY\npytest tests/ -q"


def scenarios() -> dict[str, Session]:
    s: dict[str, Session] = {}

    f = s["F01_edit_then_test"] = Session(); f.prompt("fix a.py")
    f.read("t0", "src/a.py", A0)
    f.bash("t1", HEREDOC_EDIT, "1 passed")

    f = s["F02_test_then_commit"] = Session(); f.prompt("commit")
    f.bash("t0", "pytest tests/ -q && git add -A && git commit -q -m 'fix'", "1 passed")

    f = s["F03_read_then_sed"] = Session(); f.prompt("edit")
    f.bash("t0", "cat src/a.py && sed -i 's/x = 1/x = 2/' src/a.py", A0)

    f = s["F04_test_output_redirect"] = Session(); f.prompt("test")
    f.read("t0", "src/a.py", A0)
    f.edit("t1", "src/a.py", A0, "x = 1", "x = 2")
    f.bash("t2", "pytest tests/ -q > /tmp/out.txt", "")

    f = s["F05_merged_reads"] = Session(); f.prompt("look")
    f.bash("t0", "echo a; cat x.md; ls; cat y.md", "a\n...")

    f = s["F06_errored_compound"] = Session(); f.prompt("fix")
    f.bash("t0", HEREDOC_EDIT, "", is_error=True, content="Exit code 1\n1 failed")

    f = s["F07_cross_call_repeat"] = Session(); f.prompt("read twice")
    f.bash("t0", "cat src/a.py", A0)
    f.bash("t1", HEREDOC_EDIT, "1 passed")
    f.bash("t2", "cat src/a.py", A0)

    f = s["F08_edit_revert"] = Session(); f.prompt("edit and revert")
    f.read("t0", "src/a.py", A0)
    f.edit("t1", "src/a.py", A0, "x = 1", "x = 2")
    f.edit("t2", "src/a.py", A1, "x = 2", "x = 1")

    f = s["F09_noop_edits"] = Session(); f.prompt("noop")
    f.read("t0", "src/a.py", A0)
    f.edit("t1", "src/a.py", A0, "x = 1", "x = 1")
    f.edit("t2", "src/a.py", A0, "x = 1", "x = 1")

    f = s["F10_write_create_update_revert"] = Session(); f.prompt("write")
    f.write("t0", "src/c.py", C1, None)
    f.write("t1", "src/c.py", C2, C1)
    f.write("t2", "src/c.py", C1, C2)

    f = s["F11_multi_target_copy"] = Session(); f.prompt("copy")
    f.bash("t0", "cp a.txt b.txt && cp c.txt d.txt", "")

    f = s["F12_trailing_artifact_write"] = Session(); f.prompt("finish")
    f.read("t0", "src/a.py", A0)
    f.edit("t1", "src/a.py", A0, "x = 1", "x = 2")
    f.bash("t2", "pytest tests/ -q", "1 passed")
    f.write("t3", "artifacts/report.md", "# done\n", None)

    f = s["F13_trailing_targetless_mutate"] = Session(); f.prompt("finish")
    f.read("t0", "src/a.py", A0)
    f.edit("t1", "src/a.py", A0, "x = 1", "x = 2")
    f.bash("t2", "pytest tests/ -q", "1 passed")
    f.bash("t3", "python3 - <<'PY'\nfor p in paths: p.write_text('x')\nPY", "")

    f = s["F14_docs_only"] = Session(); f.prompt("docs")
    f.read("t0", "docs/x.md", "# x\n")
    f.edit("t1", "docs/x.md", "# x\n", "# x", "# y")
    return s


def render(session: Session) -> str:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in session.out)


def write_all(target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, session in scenarios().items():
        path = target_dir / f"{name}.jsonl"
        path.write_text(render(session))
        written.append(path)
    return written


if __name__ == "__main__":
    paths = write_all(Path(__file__).with_name("synthetic"))
    print(f"wrote {len(paths)} fixtures under {paths[0].parent}")
