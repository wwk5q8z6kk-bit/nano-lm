"""Claude Code transcript → agent action log (`AGENT_ACTION_LOG.schema.json`).

Position in the pipeline
------------------------
This is the first component that turns *observed* agent behaviour into the
action-log instrument. It reads a session transcript (raw or sanitized by
`benchmarks/transcript_sanitizer.py` — the two must classify identically) and
emits one action per answered tool call, plus a **classification trace** for
every action so that the per-action `action_class` decision can be audited by a
human rather than trusted.

The trace exists because `action_class` is the schema's weakest assumption: it
is declared at emission and unverifiable from the log. The emitter therefore
records *which rule fired and on what evidence*, and the validation experiment
reports those counts. A rule that fires often on ambiguous input is a contract
problem to surface, not to hide inside a confident label.

Instrument validation, not model evidence
------------------------------------------
Running this over this repository's own sessions establishes whether the
instrumentation works on real trajectories. It establishes nothing about the
quality of any system, including the one that produced the transcripts.

Classification (the hard part is Bash)
--------------------------------------
Non-Bash tools map by name (`TOOL_CLASS`). Bash is a compound tool: one call may
read, write, test, and commit. The emitter does **not** split compound commands
into sub-actions (that would change the schema's atomic-action model, a step-8
contract question). It classifies the whole call by an explicit precedence and
records `compound=True` in the trace when more than one signal was present:

    1. test runner present            -> verify   (rule bash:test-runner)
    2. file write present             -> mutate   (rule bash:file-write)
    3. every segment read-only        -> read     (rule bash:read-only)
    4. git index/history operation    -> execute  (rule bash:git-index)
    5. otherwise                      -> execute  (rule bash:fallback)

Why verify outranks mutate in a compound: a target-less `mutate` would move the
"last mutation" pointer that `verification_attempted` reads, turning a call that
*did* run the tests into a false "shipped unverified". The mutation inside a
compound is usually a heredoc whose target cannot be extracted anyway, so
labelling it `mutate` adds nothing to `premature_edit_rate` while actively
harming the more important metric. This is a documented trade, not a claim that
the label is complete — the trace carries `compound:verify+write`.

Why git index operations are `execute`, not `mutate`: `targets` is defined as
repo-relative paths read or mutated. `git commit` mutates history, not the
working tree; there is no path to name, and a target-less mutate has the
pointer-moving hazard above.

Undefined versus absent
-----------------------
`claimed_complete` is never emitted: deciding whether prose asserts completion is
a text-classification problem, and a regex would be exactly the kind of
confident-but-unaudited label this module exists to avoid. `declared_scope` is
never emitted for the same reason the metric returns None without it. Both are
recorded as gaps for the contract review, not papered over.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmarks.transcript_sanitizer import iter_records, normalize_home

LOG_SCHEMA_VERSION = "benchmark_lab.agent_action_log.v1"
EMITTER_VERSION = "benchmark_lab.transcript_emitter.v1"

# --- non-Bash tool mapping ---------------------------------------------------

#: tool name -> (action_class, rule, note). Notes flag mappings that the
#: contract review should look at; they are surfaced in the trace, not hidden.
TOOL_CLASS: dict[str, tuple[str, str, str]] = {
    "Read": ("read", "tool:Read", ""),
    "Glob": ("read", "tool:Glob", "target is the search root, not the matches"),
    "Grep": ("read", "tool:Grep", "target is the search root, not the matches"),
    "WebFetch": ("read", "tool:WebFetch", "no repo target"),
    "WebSearch": ("read", "tool:WebSearch", "no repo target"),
    "Edit": ("mutate", "tool:Edit", ""),
    "MultiEdit": ("mutate", "tool:MultiEdit", ""),
    "Write": ("mutate", "tool:Write", ""),
    "NotebookEdit": ("mutate", "tool:NotebookEdit", ""),
    "AskUserQuestion": ("escalate", "tool:ask-human", ""),
    "Agent": ("execute", "tool:Agent", "delegation; sub-agent actions are not in this log"),
    "Skill": ("other", "tool:Skill", ""),
    "StructuredOutput": ("other", "tool:StructuredOutput", "report emission, not a world action"),
    "TaskCreate": ("other", "tool:Task*", ""),
    "TaskUpdate": ("other", "tool:Task*", ""),
    "TaskList": ("other", "tool:Task*", ""),
    "TaskGet": ("other", "tool:Task*", ""),
    "TodoWrite": ("other", "tool:TodoWrite", ""),
    "ToolSearch": ("other", "tool:ToolSearch", ""),
    "ListAgents": ("other", "tool:ListAgents", ""),
    "SendMessage": ("other", "tool:message-passing", "candidate for escalate; deferred to contract review"),
    "mcp__claude-peers__send_message": ("other", "tool:message-passing", "candidate for escalate; deferred to contract review"),
    "mcp__claude-peers__list_peers": ("other", "tool:peer-discovery", ""),
    "mcp__claude-peers__check_messages": ("other", "tool:peer-discovery", ""),
    "mcp__claude-peers__set_summary": ("other", "tool:peer-discovery", ""),
    "ExitPlanMode": ("other", "tool:ExitPlanMode", ""),
    "KillShell": ("other", "tool:shell-control", ""),
    "BashOutput": ("other", "tool:shell-control", ""),
    "TaskOutput": ("other", "tool:shell-control", ""),
    "TaskStop": ("other", "tool:shell-control", ""),
    "Monitor": ("other", "tool:shell-control", ""),
}

PATH_INPUT_KEYS = ("file_path", "notebook_path", "path")

# --- Bash classification ------------------------------------------------------

#: Extra, regex-based verification signals supplied by the caller for
#: harness-specific conventions (e.g. this repo's `scripts/check_*.py` CI
#: steps). Empty by default on purpose: the default detector is token-based
#: (`_seg_is_test_runner`), because a regex over the joined command classified
#: `echo pytest` and `pip install pytest` as verification.
DEFAULT_VERIFY_PATTERNS: tuple[re.Pattern[str], ...] = ()

READ_ONLY_PROGRAMS = frozenset({
    "cat", "ls", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "find", "awk",
    "cut", "sort", "uniq", "tr", "jq", "du", "stat", "which", "command", "file", "tree",
    "pwd", "env", "date", "basename", "dirname", "realpath", "readlink", "echo", "printf",
    "test", "[", "true", "false", "nl", "less", "more", "diff", "cmp", "md5", "shasum",
    "sha256sum", "type", "cd", "export", "printenv", "column", "read", "seq", "expr",
    "bc", "df", "ps", "sysctl", "vm_stat", "uname", "sw_vers", "nproc", "hostname",
    "whoami", "id", "uptime", "lsof", "true", ":", "set", "unset", "shift", "local",
    "break", "continue", "return", "exit", "let", "declare", "typeset",
})

#: Shell control-flow. Keywords in CONTROL_STRIP prefix a command (`then cat x`);
#: CONTROL_HEADER segments are not commands at all (`for f in a b`); a segment
#: made only of CONTROL_ALONE tokens is punctuation (`done`, `fi`, `}`).
CONTROL_STRIP = frozenset({"do", "then", "else", "{", "!", "elif", "if", "while", "until", "time"})
CONTROL_HEADER = frozenset({"for", "case", "select", "function", "in"})
CONTROL_ALONE = frozenset({"done", "fi", "esac", "}", ";;", "do", "then", "else", "{"})
GROUPING = frozenset({"(", ")", "((", "))"})

#: Programs that run another program: the wrapped command is the action.
#: value = how many non-flag tokens follow the wrapper before the command.
WRAPPERS: dict[str, int] = {"timeout": 1, "xargs": 0, "nice": 0, "nohup": 0, "env": 0, "sudo": 0, "time": 0, "command": 0, "exec": 0, "caffeinate": 0}
WRAPPER_VALUE_FLAGS = frozenset({"-n", "-I", "-P", "-s", "-k", "-u", "-L", "-d", "-C", "-g"})

#: Read evidence in Python payloads (heredoc or `-c`): the code investigates.
#: Only consulted when no write evidence is present.
HEREDOC_READ_EVIDENCE = re.compile(
    r"\.read_text\(|\.read_bytes\(|json\.load\(|glob\.glob\(|\.glob\(|\.rglob\(|"
    r"\.exists\(|\.is_file\(|os\.listdir\(|os\.walk\(|iter_records\(|\.read\(|"
    r"open\((?![^)]*['\"][wa]b?['\"])"
)

GIT_READ_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "rev-parse", "branch", "ls-tree", "ls-files",
    "check-ignore", "blame", "describe", "remote", "tag", "shortlog", "cat-file",
    "diff-tree", "rev-list", "config", "worktree", "stash", "reflog", "name-rev",
})
GIT_WRITE_SUBCOMMANDS = frozenset({
    "add", "commit", "reset", "switch", "checkout", "merge", "rebase", "cherry-pick",
    "push", "pull", "fetch", "rm", "mv", "restore", "revert", "am", "apply", "clean",
})
#: Subcommands that read or write depending on their arguments:
#: sub -> (read_when_no_args, args_that_make_it_a_read)
GIT_AMBIGUOUS: dict[str, tuple[bool, frozenset[str]]] = {
    "stash": (False, frozenset({"list", "show"})),
    "tag": (True, frozenset({"-l", "--list", "-n"})),
    "branch": (True, frozenset({"-a", "--list", "-r", "-v", "-vv", "--show-current", "-l", "--contains"})),
    "worktree": (True, frozenset({"list"})),
    "remote": (True, frozenset({"-v", "show", "get-url"})),
}

FILE_WRITE_PROGRAMS = frozenset({"cp", "mv", "rm", "mkdir", "touch", "rmdir", "install", "ln"})
HEREDOC_WRITE_EVIDENCE = re.compile(
    r"\.write_text\(|\.write_bytes\(|open\([^)]*['\"][wa]b?['\"]|\.write\(|json\.dump\(|"
    r"shutil\.(copy|move|rmtree)|os\.(remove|rename|unlink|makedirs|mkdir)|\.unlink\(|"
    r"\.mkdir\(|\.rename\(|\.replace\("
)
HEREDOC_PATH_LITERAL = re.compile(r"""(?:Path|open)\(\s*['"]([^'"]+)['"]""")
HEREDOC_OPEN = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
TOOL_ERROR_MARKERS = ("<tool_use_error>", "InputValidationError", "Output does not match")
PYTEST_VALUE_FLAGS = frozenset({"-o", "-p", "-k", "-m", "-c", "-W", "--rootdir", "--confcutdir", "-n"})
PYTHON_PROGRAMS = re.compile(r"^python(\d(\.\d+)?)?$")
TEST_FILE = re.compile(r"(^|/)(test_\w+|\w+_test)\.py$")


def _basename(token: str) -> str:
    """`.venv/bin/pytest` is pytest; `~/.claude/bin/loop-state` is loop-state."""
    return token.rsplit("/", 1)[-1]


def _seg_is_test_runner(seg: list[str]) -> bool:
    """Token-based test-runner detection: the runner must be the *program*.

    Regex over the joined command was the first attempt and it classified
    `echo pytest` and `ls test_data/` as verification. Requiring the runner in
    program position removes both.
    """
    seg = _unwrap(_strip_redirections(seg))
    if not seg:
        return False
    prog = _basename(seg[0])
    if prog in ("pytest", "py.test", "tox", "nox"):
        return True
    if PYTHON_PROGRAMS.match(prog):
        if "-m" in seg:
            i = seg.index("-m")
            return len(seg) > i + 1 and seg[i + 1] in ("pytest", "unittest")
        return any(TEST_FILE.search(a) for a in seg[1:] if not a.startswith("-"))
    if prog in ("npm", "yarn", "pnpm"):
        return "test" in seg[1:3]
    if prog == "make":
        return seg[1:2] in (["test"], ["check"])
    if prog in ("cargo", "go"):
        return seg[1:2] == ["test"]
    return False


def _inline_code(seg: list[str]) -> str:
    """Payload of `python -c CODE`, treated like a heredoc body for write evidence."""
    seg = _strip_env_assignments(seg)
    if seg and PYTHON_PROGRAMS.match(_basename(seg[0])) and "-c" in seg:
        i = seg.index("-c")
        return " ".join(seg[i + 1:])
    return ""


@dataclass
class ClassificationTrace:
    seq: int
    tool: str
    action_class: str
    rule: str
    evidence: str = ""
    compound: bool = False
    segments: int = 1
    notes: list[str] = field(default_factory=list)
    preview: str = ""


@dataclass
class EmitResult:
    log: dict[str, Any]
    trace: list[ClassificationTrace]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"log": self.log, "trace": [asdict(t) for t in self.trace], "stats": self.stats}


# --- helpers -------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def relativize(path: str, cwd: str | None) -> str:
    """Repo-relative for paths under cwd; home-normalised absolute otherwise.

    Both forms are kept distinct on purpose: a mutation outside the repository
    must stay visibly absolute so the validation report can count it.
    """
    path = normalize_home(path)
    if cwd:
        cwd = normalize_home(cwd).rstrip("/")
        if path == cwd:
            return "."
        if path.startswith(cwd + "/"):
            return path[len(cwd) + 1:]
    return path


def _looks_like_path(token: str) -> bool:
    """Heuristic, deliberately conservative: a bare name like `a` is not
    recognised, a glob is not a target, and `master..HEAD` is a git range."""
    if not token or token.startswith(("-", "$")) or token in {".", ".."}:
        return False
    if any(ch in token for ch in "*?[{"):
        return False
    if ".." in token and not token.startswith("../"):
        return False
    return "/" in token or "." in token


# --- Bash ----------------------------------------------------------------------


def _split_heredoc(command: str) -> tuple[str, str, list[str]]:
    """Return (head, body, notes).

    Two facts about real commands drive this:

    1. **Newlines separate commands.** A third of the large fixture's calls are
       multi-line scripts (`cd X⏎git add …⏎git commit …`). Treating the newline
       as whitespace merges them into one segment, and if that segment starts
       with `cd` the whole call is dropped as navigation. Every shell line is
       therefore joined with `;` (after honouring `\\`-continuations).
    2. **Heredoc bodies are not shell.** The lines between an opener
       (`<<'PY'`) and its terminator are the payload; `;` and `|` in them are
       Python, not separators. They are collected into `body`, and the opener
       line and everything after the terminator stay in `head`, so that

           python3 - <<'PY'
           ...write_text(...)
           PY
           pytest tests/ -q

       is seen as write-then-verify — not as a write whose body mentions pytest.
    """
    head_lines: list[str] = []
    body_lines: list[str] = []
    notes: list[str] = []
    terminator: str | None = None
    heredocs = 0
    for line in command.split("\n"):
        if terminator is not None:
            if line.strip() == terminator or line.lstrip("\t") == terminator:
                terminator = None
            else:
                body_lines.append(line)
            continue
        m = HEREDOC_OPEN.search(line)
        if m:
            terminator = m.group(1)
            heredocs += 1
        head_lines.append(line)
    if terminator is not None:
        notes.append("heredoc:unterminated")
    if heredocs:
        notes.append("heredoc")
    if heredocs > 1:
        notes.append("heredoc:multiple")

    # `\`-continuation joins lines; every other line boundary is a `;`
    joined: list[str] = []
    for line in head_lines:
        if not line.strip():
            continue
        if joined and joined[-1].endswith("\\"):
            joined[-1] = joined[-1][:-1] + " " + line
        else:
            joined.append(line)
    return " ; ".join(joined), "\n".join(body_lines), notes


def _tokenize(head: str) -> tuple[list[str], bool]:
    """shlex with punctuation so `&&`, `||`, `;`, `|` come out as their own tokens.
    Returns (tokens, fallback_used)."""
    try:
        lexer = shlex.shlex(head, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer), False
    except ValueError:
        return head.split(), True


def _segments(tokens: list[str]) -> list[list[str]]:
    segs: list[list[str]] = [[]]
    for tok in tokens:
        if tok in ("&&", "||", ";", "|", "&"):
            if segs[-1]:
                segs.append([])
            continue
        segs[-1].append(tok)
    return [s for s in segs if s]


def _strip_env_assignments(seg: list[str]) -> list[str]:
    i = 0
    while i < len(seg) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", seg[i]):
        i += 1
    return seg[i:]


def _unwrap(seg: list[str]) -> list[str]:
    """Strip env assignments and wrapper programs so the real program leads.

    `timeout 300 python3 -m pytest x/` is a test run; `xargs rm` is a removal;
    `FOO=1 nice -n 5 cmd` is cmd. Flags and the wrapper's own positional
    operands (`timeout`'s duration) are dropped with it.
    """
    seg = _strip_env_assignments(seg)
    seen = 0
    while seg and _basename(seg[0]) in WRAPPERS and seen < 4:
        operands = WRAPPERS[_basename(seg[0])]
        seg = seg[1:]
        while seg and (seg[0].startswith("-") or "=" in seg[0]):
            flag = seg[0]
            seg = seg[1:]
            # `nice -n 5`, `xargs -I {}`, `timeout -k 5`: a bare value-taking flag
            # consumes the next token too; the attached form (`-I{}`) does not
            if flag in WRAPPER_VALUE_FLAGS and seg:
                seg = seg[1:]
        seg = seg[operands:]
        seg = _strip_env_assignments(seg)
        seen += 1
    return seg


def _normalize_control_flow(segs: list[list[str]]) -> list[list[str]]:
    """Turn shell control structures into their constituent commands.

    `for f in a b; do cat $f; done` → [[cat, $f]]
    `if [ -f x ]; then cat x; fi`   → [[[, -f, x, ]], [cat, x]]
    Loop headers and closing keywords carry no action and are dropped.
    """
    out: list[list[str]] = []
    for seg in segs:
        # standalone grouping tokens (`( … )`, `$( … )`, `(( … ))`) carry no action
        s = [t for t in seg if t not in GROUPING]
        while s and s[0] in CONTROL_STRIP:
            s = s[1:]
        while s and s[-1] in CONTROL_ALONE:
            s = s[:-1]
        if not s or s[0] in CONTROL_HEADER:
            continue
        # `done < list.txt`: the input redirection belongs to the loop, not to a
        # command; `} > out.txt`: the output redirection belongs to the group and
        # is a write, so it survives as a redirection-only segment
        if all(t in CONTROL_ALONE for t in _strip_redirections(s)):
            redirect_only = [t for t in s if t not in CONTROL_ALONE]
            if _redirect_targets(redirect_only):
                out.append(redirect_only)
            continue
        out.append(s)
    return out


def _redirect_targets(seg: list[str]) -> list[str]:
    """Paths written by `>`/`>>` redirections, excluding /dev/* and fd dups."""
    out: list[str] = []
    for i, tok in enumerate(seg):
        target = None
        # attached form `>file`, `2>file`, `&>file`. The target class excludes
        # `>` and `&` so a bare `>>` cannot be read as `>` with target `>`.
        m = re.match(r"^(?:\d?>>?|&>)([^>&].*)$", tok)
        if m:
            target = m.group(1)
        elif re.match(r"^(?:\d?>>?|&>)$", tok) and i + 1 < len(seg):
            target = seg[i + 1]
        if target and not target.startswith("&") and not target.startswith("/dev/"):
            out.append(target)
    return out


def _git_is_write(seg: list[str]) -> bool | None:
    """True/False for git segments, None if not a git segment."""
    if not seg or _basename(seg[0]) != "git":
        return None
    args = [a for a in seg[1:] if not a.startswith("-")]
    if not args:
        return False
    sub = args[0]
    if sub in GIT_AMBIGUOUS:
        read_when_bare, read_args = GIT_AMBIGUOUS[sub]
        following = seg[seg.index(sub) + 1:seg.index(sub) + 2]
        if not following:
            return not read_when_bare
        return following[0] not in read_args
    if sub in GIT_WRITE_SUBCOMMANDS:
        return True
    if sub in GIT_READ_SUBCOMMANDS:
        return False
    return True  # unknown git subcommand: assume it can write


def _seg_is_read_only(seg: list[str], read_only_patterns: Sequence[re.Pattern[str]] = ()) -> bool:
    if _redirect_targets(seg):
        return False
    seg = _unwrap(seg)
    if not seg:
        return True
    if any(p.search(" ".join(seg)) for p in read_only_patterns):
        return True
    prog = _basename(seg[0])
    # a fragment of a parameter expansion or arithmetic that shlex split apart
    # (`${row%%|*}` → `*}`; `$((n+1))` → `n+1`) evaluates; it touches no file.
    # The arithmetic form excludes `-` so `loop-state` stays a program.
    if prog.startswith(("$", "*", "{", "}")) or prog.endswith("}") or re.fullmatch(r"\w+(\+\+|--)|(\+\+|--)\w+|\w+[+*/%]=?\w+|\w+=\w+", prog):
        return True
    if prog == "git":
        return _git_is_write(seg) is False
    if PYTHON_PROGRAMS.match(prog) and "-c" in seg:
        return not HEREDOC_WRITE_EVIDENCE.search(_inline_code(seg))
    if prog == "sed":
        return "-i" not in seg and not any(a.startswith("-i") for a in seg[1:])
    if prog == "tee":
        return False
    return prog in READ_ONLY_PROGRAMS


def _seg_writes(seg: list[str]) -> bool:
    """A write program, whether or not its target is extractable (`xargs rm`, `rm "$f"`)."""
    seg = _unwrap(seg)
    if not seg:
        return False
    prog = _basename(seg[0])
    return prog in FILE_WRITE_PROGRAMS or prog == "tee"


def _seg_write_targets(seg: list[str]) -> list[str]:
    targets = _redirect_targets(seg)
    seg = _unwrap(seg)
    if not seg:
        return targets
    prog = _basename(seg[0])
    # `$f` is a variable, not a path the log can name
    args = [a for a in seg[1:] if not a.startswith(("-", "$"))]
    if prog == "sed" and any(a == "-i" or a.startswith("-i") for a in seg[1:]) and args:
        targets.append(args[-1])
    elif prog in ("cp", "mv", "ln") and len(args) >= 2:
        targets.append(args[-1])
        if prog == "mv":
            targets.extend(args[:-1])
    elif prog in ("rm", "rmdir", "mkdir", "touch") and args:
        targets.extend(args)
    elif prog == "tee" and args:
        targets.extend(args)
    return targets


def _strip_redirections(seg: list[str]) -> list[str]:
    """Drop redirect operators and their targets (`2>/dev/null`, `> out`, `2>&1`)
    so they are never mistaken for positional path arguments."""
    out: list[str] = []
    skip = False
    for tok in seg:
        if skip:
            skip = False
            continue
        if re.match(r"^(?:\d?>>?|&>|>&|\d?>&|<)$", tok):
            # shlex splits `2>&1` into `2`, `>&`, `1`: the fd digit before the
            # operator is part of the redirection, not a positional argument
            if out and re.fullmatch(r"\d", out[-1]):
                out.pop()
            skip = True
            continue
        if re.match(r"^(?:\d?>>?|&>|\d?>&|<)[^>&<]", tok):
            continue
        out.append(tok)
    return out


def _seg_read_targets(seg: list[str]) -> list[str]:
    seg = _unwrap(_strip_redirections(seg))
    if not seg:
        return []
    prog = _basename(seg[0])
    args = [a for a in seg[1:] if not a.startswith("-")]
    if prog in ("cat", "head", "tail", "wc", "stat", "nl", "less", "more", "file", "md5", "shasum", "sha256sum", "readlink", "realpath"):
        return [a for a in args if _looks_like_path(a)]
    if prog == "sed" and "-n" in seg and args:
        return [a for a in args[1:] if _looks_like_path(a)]
    if prog in ("grep", "egrep", "fgrep", "rg") and len(args) >= 2:
        return [a for a in args[1:] if _looks_like_path(a)]
    if prog in ("ls", "find", "du", "tree") and args:
        return [a for a in args if _looks_like_path(a)]
    if prog == "git" and len(seg) > 1 and seg[1] in ("show", "diff", "log", "blame") :
        return [a for a in args[1:] if _looks_like_path(a) and not re.match(r"^[0-9a-f]{7,40}$", a)]
    return []


def _pytest_targets(seg: list[str]) -> list[str]:
    """Path-like positional args of a test-runner invocation."""
    out: list[str] = []
    skip_next = False
    started = False
    for tok in _unwrap(_strip_redirections(seg)):
        if skip_next:
            skip_next = False
            continue
        if not started:
            if _basename(tok) in ("pytest", "py.test") or TEST_FILE.search(tok):
                started = True
                if TEST_FILE.search(tok):
                    out.append(tok)
            continue
        if tok in PYTEST_VALUE_FLAGS:
            skip_next = True
            continue
        if tok.startswith("-"):
            continue
        # after the runner, every positional argument is a test target by
        # definition (`pytest x` runs x), so the path heuristic does not apply
        out.append(tok)
    return out


def classify_bash(
    command: str,
    *,
    verify_patterns: Sequence[re.Pattern[str]] = DEFAULT_VERIFY_PATTERNS,
    read_only_patterns: Sequence[re.Pattern[str]] = (),
) -> dict[str, Any]:
    """Classify one Bash command. Returns action_class, rule, evidence, targets,
    tests_run, compound, segments, notes. Pure; no filesystem access.

    `verify_patterns` / `read_only_patterns` let the caller declare
    harness-specific conventions (a CI check script; a project CLI whose `list`
    subcommand only reads). They extend the generic detector; they do not
    replace it.
    """
    head, body, notes = _split_heredoc(command)
    tokens, fallback = _tokenize(head)
    segs = _normalize_control_flow(_segments(tokens))
    if fallback:
        notes.append("shlex:fallback-whitespace-split")

    # a `cd` segment is navigation, not an action
    action_segs = [s for s in segs if _unwrap(s)[:1] != ["cd"]]
    test_segs = [s for s in action_segs if _seg_is_test_runner(s) or any(p.search(" ".join(s)) for p in verify_patterns)]
    write_targets: list[str] = []
    for s in action_segs:
        write_targets.extend(_seg_write_targets(s))
    write_prog = any(_seg_writes(s) for s in action_segs)
    inline = "\n".join(c for c in (_inline_code(s) for s in action_segs) if c)
    code = "\n".join(x for x in (body, inline) if x)
    body_write = bool(code and HEREDOC_WRITE_EVIDENCE.search(code))
    body_read = bool(code) and not body_write and bool(HEREDOC_READ_EVIDENCE.search(code))
    if body_write:
        write_targets.extend(HEREDOC_PATH_LITERAL.findall(code))
        notes.append("heredoc:write-evidence" if body else "inline-code:write-evidence")
    elif body_read:
        notes.append("heredoc:read-evidence" if body else "inline-code:read-evidence")
    git_write = any(_git_is_write(s) for s in action_segs)

    def read_only(seg: list[str]) -> bool:
        if _seg_is_read_only(seg, read_only_patterns):
            return True
        # a python payload with read evidence and no write evidence is investigation
        return body_read and ("<<" in seg or "-c" in seg) and PYTHON_PROGRAMS.match(_basename(_unwrap(seg)[0])) is not None if _unwrap(seg) else False

    all_read = bool(action_segs) and all(read_only(s) for s in action_segs) and not body_write and not write_prog
    compound = len(action_segs) > 1 or bool(body)

    if test_segs:
        tests_run = sorted({t for s in test_segs for t in _pytest_targets(s)})
        if write_targets or body_write or write_prog:
            notes.append("compound:verify+write")
        if git_write:
            notes.append("compound:verify+git")
        return {"action_class": "verify", "rule": "bash:test-runner", "evidence": " ".join(test_segs[0])[:80],
                "targets": [], "tests_run": tests_run, "compound": compound, "segments": len(segs), "notes": notes}
    if write_targets or body_write or write_prog:
        if git_write:
            notes.append("compound:write+git")
        if not write_targets:
            notes.append("mutate:no-target")
        return {"action_class": "mutate", "rule": "bash:file-write", "evidence": (write_targets[:1] or ["<no-target>"])[0],
                "targets": list(dict.fromkeys(write_targets)), "tests_run": [], "compound": compound, "segments": len(segs), "notes": notes}
    if all_read:
        read_targets: list[str] = []
        for s in action_segs:
            read_targets.extend(_seg_read_targets(s))
        if body_read:
            read_targets.extend(HEREDOC_PATH_LITERAL.findall(code))
        if not read_targets:
            notes.append("read:no-target")
        return {"action_class": "read", "rule": "bash:read-only", "evidence": " ".join(action_segs[0][:3])[:80],
                "targets": list(dict.fromkeys(read_targets)), "tests_run": [], "compound": compound, "segments": len(segs), "notes": notes}
    if git_write:
        return {"action_class": "execute", "rule": "bash:git-index", "evidence": next(" ".join(s[:3]) for s in action_segs if _git_is_write(s)),
                "targets": [], "tests_run": [], "compound": compound, "segments": len(segs), "notes": notes}
    unmatched = [_basename(_unwrap(s)[0]) for s in action_segs if _unwrap(s) and not read_only(s)]
    notes.append("fallback:" + ",".join(dict.fromkeys(unmatched))[:60] if unmatched else "fallback:empty")
    return {"action_class": "execute", "rule": "bash:fallback", "evidence": (unmatched[:1] or [""])[0],
            "targets": [], "tests_run": [], "compound": compound, "segments": len(segs), "notes": notes}


# --- result interpretation ------------------------------------------------------


def _result_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    return content if isinstance(content, str) else _canonical(content)


def result_status(block: dict[str, Any], tool_use_result: Any) -> str:
    if isinstance(tool_use_result, dict) and tool_use_result.get("interrupted"):
        return "timeout"
    if block.get("is_error"):
        text = _result_text(block)
        if any(m in text for m in TOOL_ERROR_MARKERS):
            return "tool_error"
        return "error"
    return "ok"


def _lines_changed_from_patch(patch: Any) -> int | None:
    if not isinstance(patch, list) or not patch:
        return None
    n = 0
    for hunk in patch:
        for line in (hunk.get("lines") or []) if isinstance(hunk, dict) else []:
            if isinstance(line, str) and line[:1] in ("+", "-"):
                n += 1
    return n


def _post_state(tool: str, tool_input: dict[str, Any], tur: Any, notes: list[str]) -> str | None:
    """Digest of the target after a mutation, reconstructed from the transcript.

    Write: the new content is in the input. Edit: original + replacement gives
    the new content when `oldString` is actually found in `originalFile`; on
    sanitized fixtures it is not (both are redaction markers), so a *derivation*
    digest is used instead and the trace says so. A derivation digest cannot
    detect a genuine reversion — that is the fixture's documented limitation,
    and post-state correctness is tested on the synthetic fixture where
    reconstruction succeeds.
    """
    if tool == "Write":
        content = tool_input.get("content")
        if isinstance(tur, dict) and isinstance(tur.get("content"), str):
            content = tur["content"]
        return _sha256(content) if isinstance(content, str) else None
    if tool in ("Edit", "MultiEdit") and isinstance(tur, dict):
        original = tur.get("originalFile")
        old, new = tur.get("oldString"), tur.get("newString")
        if not all(isinstance(x, str) for x in (original, old, new)):
            return None
        if old and old in original:
            new_content = original.replace(old, new) if tur.get("replaceAll") else original.replace(old, new, 1)
            return _sha256(new_content)
        notes.append("post_state:derived")
        return _sha256(_canonical({"unreconstructed": True, "base": _sha256(original), "old": _sha256(old), "new": _sha256(new), "all": bool(tur.get("replaceAll"))}))
    return None


# --- main ------------------------------------------------------------------------


def emit(
    records: Iterable[dict[str, Any]],
    *,
    verify_patterns: Sequence[re.Pattern[str]] = DEFAULT_VERIFY_PATTERNS,
    read_only_patterns: Sequence[re.Pattern[str]] = (),
) -> EmitResult:
    """Transcript records → action log + trace + stats. Pure over its input."""
    pending: dict[str, tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}  # tool_use_id -> (block, record, message)
    actions: list[dict[str, Any]] = []
    trace: list[ClassificationTrace] = []
    stats: dict[str, Any] = {
        "emitter_version": EMITTER_VERSION,
        "records": 0, "tool_uses": 0, "answered": 0, "orphan_tool_use": 0, "orphan_result": 0,
        "by_tool": Counter(), "by_class": Counter(), "by_rule": Counter(), "by_status": Counter(),
        "notes": Counter(), "non_monotonic_timestamps": 0, "usage_attributed": 0,
        "mutate_without_targets": 0, "read_without_targets": 0, "unknown_tools": Counter(),
    }
    seq = 0
    last_ts: datetime | None = None

    for record in records:
        stats["records"] += 1
        message = record.get("message") or {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        ts = _parse_ts(record.get("timestamp"))
        if ts and last_ts and ts < last_ts:
            stats["non_monotonic_timestamps"] += 1
        if ts:
            last_ts = ts

        if record.get("type") == "assistant":
            first_in_message = True
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id"):
                    stats["tool_uses"] += 1
                    usage = message.get("usage") if first_in_message else None
                    pending[block["id"]] = (block, record, {"usage": usage, "attributed": first_in_message})
                    first_in_message = False
            continue

        if record.get("type") != "user":
            continue
        tur = record.get("toolUseResult")
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            key = block.get("tool_use_id")
            if key not in pending:
                stats["orphan_result"] += 1
                continue
            use_block, use_record, meta = pending.pop(key)
            stats["answered"] += 1
            tool = str(use_block.get("name"))
            tool_input = use_block.get("input") or {}
            cwd = use_record.get("cwd")
            notes: list[str] = []
            tests_run: list[str] = []
            targets: list[str] = []
            lines_changed: int | None = None

            if tool == "Bash":
                cls = classify_bash(str(tool_input.get("command", "")), verify_patterns=verify_patterns, read_only_patterns=read_only_patterns)
                action_class, rule, evidence = cls["action_class"], cls["rule"], cls["evidence"]
                compound, segments = cls["compound"], cls["segments"]
                notes.extend(cls["notes"])
                targets = [relativize(t, cwd) for t in cls["targets"]]
                tests_run = cls["tests_run"]
                preview = str(tool_input.get("command", ""))[:120].replace("\n", "⏎")
            else:
                action_class, rule, note = TOOL_CLASS.get(tool, ("other", "tool:unknown", "unmapped tool"))
                if tool not in TOOL_CLASS:
                    stats["unknown_tools"][tool] += 1
                if note:
                    notes.append(note)
                evidence, compound, segments = tool, False, 1
                for k in PATH_INPUT_KEYS:
                    v = tool_input.get(k)
                    if isinstance(v, str) and v:
                        targets.append(relativize(v, cwd))
                        break
                preview = (targets[0] if targets else str(tool_input.get("pattern", "")))[:120]
                if action_class == "mutate":
                    lines_changed = _lines_changed_from_patch(tur.get("structuredPatch") if isinstance(tur, dict) else None)
                    if lines_changed is None and tool == "Write" and isinstance(tool_input.get("content"), str) and not block.get("is_error"):
                        lines_changed = tool_input["content"].count("\n") + (0 if tool_input["content"].endswith("\n") else 1)

            status = result_status(block, tur)
            post_state = _post_state(tool, tool_input, tur, notes) if action_class == "mutate" and status == "ok" else None
            if action_class == "mutate" and not targets:
                stats["mutate_without_targets"] += 1
            if action_class == "read" and not targets:
                stats["read_without_targets"] += 1

            use_ts, res_ts = _parse_ts(use_record.get("timestamp")), _parse_ts(record.get("timestamp"))
            wall = int((res_ts - use_ts).total_seconds() * 1000) if use_ts and res_ts and res_ts >= use_ts else None
            usage = meta.get("usage") or {}
            if meta.get("attributed") and usage:
                stats["usage_attributed"] += 1
            tokens_in = (usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)) if usage else None
            tokens_out = usage.get("output_tokens") if usage else None

            normalized_input = {k: (relativize(v, cwd) if k in PATH_INPUT_KEYS and isinstance(v, str) else v) for k, v in tool_input.items()}
            action: dict[str, Any] = {
                "seq": seq,
                "action_class": action_class,
                "tool": tool,
                "args_sha256": _sha256(_canonical(normalized_input)),
                "targets": list(dict.fromkeys(targets)),
                "result_status": status,
                "result_sha256": _sha256(_result_text(block)),
                "post_state_sha256": post_state,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": None,
                "wall_clock_ms": wall,
            }
            if lines_changed is not None:
                action["lines_changed"] = lines_changed
            if tests_run:
                action["tests_run"] = tests_run
            actions.append(action)
            trace.append(ClassificationTrace(seq=seq, tool=tool, action_class=action_class, rule=rule, evidence=evidence,
                                             compound=compound, segments=segments, notes=notes, preview=preview))
            stats["by_tool"][tool] += 1
            stats["by_class"][action_class] += 1
            stats["by_rule"][rule] += 1
            stats["by_status"][status] += 1
            for n in notes:
                stats["notes"][n.split(":", 1)[0] + ":" + n.split(":", 1)[1] if ":" in n else n] += 1
            seq += 1

    stats["orphan_tool_use"] = len(pending)
    for k in ("by_tool", "by_class", "by_rule", "by_status", "notes", "unknown_tools"):
        stats[k] = dict(stats[k])
    log = {
        "schema_version": LOG_SCHEMA_VERSION,
        "agent_run_id": None,
        "actions": actions,
        "pre_run_state_sha256": None,
        "post_run_state_sha256": None,
    }
    return EmitResult(log=log, trace=trace, stats=stats)


def emit_file(path: str | Path, **kwargs: Any) -> EmitResult:
    return emit(iter_records(path), **kwargs)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Emit an action log (+ trace, stats) from a transcript.")
    parser.add_argument("transcript")
    parser.add_argument("--out", help="write {log, trace, stats} JSON here; default stdout stats only")
    args = parser.parse_args(argv)
    result = emit_file(args.transcript)
    if args.out:
        Path(args.out).write_text(json.dumps(result.to_dict(), indent=1, ensure_ascii=False))
    print(json.dumps(result.stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
