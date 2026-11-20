"""Sanitizer tests: what is preserved, what is replaced, and that the checked-in
fixtures obey both.

Run explicitly (``benchmarks`` is not in ``pytest.ini`` testpaths):

    pytest benchmarks/ -q
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.transcript_sanitizer import (  # noqa: E402
    COMMAND_KEYS,
    KEPT_INPUT_KEYS,
    normalize_home,
    redact_secrets,
    redaction_marker,
    sanitize_record,
    sanitize_tool_use_result,
)

FIXTURES = sorted((REPO / "benchmarks/fixtures/transcripts").glob("session_*.jsonl"))
MARKER = re.compile(r"^<redacted sha256=[0-9a-f]{16} len=\d+>\n*$")
HOME = os.path.expanduser("~")


# --- marker properties --------------------------------------------------------


def test_marker_preserves_equality_and_line_count():
    a, b = redaction_marker("one\ntwo\nthree"), redaction_marker("one\ntwo\nthree")
    assert a == b
    assert a.count("\n") == 2
    assert MARKER.match(a)
    assert redaction_marker("different") != a


def test_marker_reveals_nothing_of_the_text():
    text = "SECRET_CONTENT_12345"
    assert "SECRET" not in redaction_marker(text)
    assert "12345" not in redaction_marker(text)


# --- secrets ------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "curl -H 'Authorization: Bearer abcdefghijklmnop1234' x",
        "export OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz",
        "git push https://ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ12@github.com/x",
        "aws configure set aws_access_key_id AKIAIOSFODNN7EXAMPLE",
        "python x.py --token=abcdef123456",
        "echo password: hunter2hunter2",
    ],
)
def test_secret_shapes_are_redacted(command):
    out = redact_secrets(command)
    assert "<secret>" in out, out
    for needle in ("abcdefghijklmnop1234", "sk-abcdef", "ghp_ABC", "AKIAIOSFODNN7EXAMPLE", "abcdef123456", "hunter2"):
        assert needle not in out


def test_word_boundary_keeps_tokenizer_and_secret_key_names_intact():
    """`tokenizer=` is not `token=`; the first scan of real fixtures matched it."""
    assert redact_secrets("tokenizer = AutoTokenizer.from_pretrained(x)") == "tokenizer = AutoTokenizer.from_pretrained(x)"
    assert redact_secrets("model.tokenizer_sha256: abc123") == "model.tokenizer_sha256: abc123"


# --- home normalisation -------------------------------------------------------


def test_home_is_normalised_everywhere_a_path_can_appear():
    if HOME == "/":
        pytest.skip("no home dir")
    rec = {
        "type": "user", "uuid": "u", "cwd": f"{HOME}/proj", "timestamp": "t",
        "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                    "content": f'Working directory "{HOME}/proj/x" does not exist', "is_error": True}]},
        "toolUseResult": {"filePath": f"{HOME}/proj/a.py", "filenames": [f"{HOME}/proj/b.py"],
                          "file": {"filePath": f"{HOME}/proj/c.py", "content": "body"}},
    }
    out = json.dumps(sanitize_record(rec))
    assert HOME not in out
    assert "~/proj" in out


def test_command_paths_are_home_normalised_but_command_is_kept():
    if HOME == "/":
        pytest.skip("no home dir")
    rec = {"type": "assistant", "uuid": "u", "timestamp": "t", "cwd": "/x",
           "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                       "input": {"command": f"cat {HOME}/proj/a.py && pytest x/", "description": "read then test"}}]}}
    inp = sanitize_record(rec)["message"]["content"][0]["input"]
    assert inp["command"] == "cat ~/proj/a.py && pytest x/"
    assert inp["description"] == "read then test"


# --- per-key policy -----------------------------------------------------------


def test_free_text_input_keys_become_markers_paths_stay():
    rec = {"type": "assistant", "uuid": "u", "timestamp": "t", "cwd": "/x",
           "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Edit",
                       "input": {"file_path": "/x/a.py", "old_string": "x = 1", "new_string": "x = 2", "replace_all": False}}]}}
    inp = sanitize_record(rec)["message"]["content"][0]["input"]
    assert inp["file_path"] == "/x/a.py"
    assert inp["replace_all"] is False
    assert MARKER.match(inp["old_string"]) and MARKER.match(inp["new_string"])
    assert "x = 1" not in json.dumps(inp)


def test_unknown_string_keys_are_redacted_by_default():
    rec = {"type": "assistant", "uuid": "u", "timestamp": "t", "cwd": "/x",
           "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Weird",
                       "input": {"mystery": "should not survive", "nested": {"deep": "nor this", "n": 3}}}]}}
    inp = sanitize_record(rec)["message"]["content"][0]["input"]
    assert MARKER.match(inp["mystery"])
    assert MARKER.match(inp["nested"]["deep"])
    assert inp["nested"]["n"] == 3


# --- results ------------------------------------------------------------------


def test_error_header_first_line_is_kept_rest_is_marker():
    rec = {"type": "user", "uuid": "u", "timestamp": "t", "cwd": "/x",
           "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                       "content": "<tool_use_error>String to replace not found in file.\nString: the secret old string"}]}}
    content = sanitize_record(rec)["message"]["content"][0]["content"]
    first, rest = content.split("\n", 1)
    assert first == "<tool_use_error>String to replace not found in file."
    assert MARKER.match(rest)
    assert "secret old string" not in content


def test_non_error_result_content_is_fully_replaced():
    rec = {"type": "user", "uuid": "u", "timestamp": "t", "cwd": "/x",
           "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                       "content": "1\tdef f():\n2\t    return 42\n"}]}}
    content = sanitize_record(rec)["message"]["content"][0]["content"]
    assert MARKER.match(content)
    assert "return 42" not in content


def test_tool_use_result_structure_is_kept_content_replaced():
    tur = {"filePath": "/x/a.py", "oldString": "a", "newString": "b", "originalFile": "a\nc\n",
           "replaceAll": False, "userModified": False,
           "structuredPatch": [{"oldStart": 1, "oldLines": 2, "newStart": 1, "newLines": 2, "lines": ["-a", "+b", " c"]}]}
    out = sanitize_tool_use_result(tur)
    assert out["filePath"] == "/x/a.py" and out["replaceAll"] is False
    assert MARKER.match(out["originalFile"]) and out["originalFile"].count("\n") == 2
    assert out["structuredPatch"][0]["lines"] == ["-", "+", " "]
    assert out["structuredPatch"][0]["oldStart"] == 1


def test_bash_result_keeps_flags_replaces_streams():
    out = sanitize_tool_use_result({"stdout": "hello", "stderr": "", "interrupted": True, "isImage": False, "noOutputExpected": False})
    assert out["interrupted"] is True
    assert MARKER.match(out["stdout"])


def test_list_results_are_collapsed():
    assert sanitize_tool_use_result([1, 2, 3]) == {"_redacted": "list", "len": 3}


# --- record filter --------------------------------------------------------------


@pytest.mark.parametrize("kind", ["attachment", "queue-operation", "last-prompt", "atis-latch", "ai-title", "mode", "system"])
def test_non_conversation_records_are_dropped(kind):
    assert sanitize_record({"type": kind, "sessionId": "s"}) is None


def test_thinking_blocks_are_dropped_text_blocks_become_markers():
    rec = {"type": "assistant", "uuid": "u", "timestamp": "t", "cwd": "/x",
           "message": {"role": "assistant", "content": [
               {"type": "thinking", "thinking": "private"},
               {"type": "text", "text": "visible prose"},
               {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/x/a"}}]}}
    blocks = sanitize_record(rec)["message"]["content"]
    assert [b["type"] for b in blocks] == ["text", "tool_use"]
    assert MARKER.match(blocks[0]["text"])
    assert "private" not in json.dumps(blocks) and "visible prose" not in json.dumps(blocks)


def test_usage_keeps_only_the_four_counters_and_session_id_is_hashed():
    rec = {"type": "assistant", "uuid": "u", "timestamp": "t", "cwd": "/x", "sessionId": "real-session-id",
           "message": {"role": "assistant", "usage": {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": 3,
                                                       "cache_creation_input_tokens": 4, "service_tier": "x", "inference_geo": "y"},
                       "content": []}}
    out = sanitize_record(rec)
    assert out["message"]["usage"] == {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": 3, "cache_creation_input_tokens": 4}
    assert out["sessionId"].startswith("sha256:") and "real-session-id" not in json.dumps(out)


# --- the checked-in fixtures obey the policy -------------------------------------


def _walk_strings(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_strings(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_strings(v, path + (i,))
    elif isinstance(obj, str):
        yield path, obj


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_has_no_home_path_and_no_secret_shapes(path):
    text = path.read_text()
    if HOME != "/":
        assert HOME not in text
    assert "/Users/" not in text
    assert redact_secrets(text) == text, "a secret-shaped string survived in a fixture"


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixture_free_text_is_all_markers(path):
    """Every string under a non-kept key must be a marker: the fixture's
    trustworthiness rests on this being mechanically true, not reviewed once."""
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        for block in rec["message"]["content"]:
            if block["type"] == "tool_use":
                for p, s in _walk_strings(block.get("input", {})):
                    key = next((k for k in reversed(p) if isinstance(k, str)), "")
                    if key in KEPT_INPUT_KEYS or key in COMMAND_KEYS:
                        continue
                    assert MARKER.match(s), (path.name, p, s[:60])
            elif block["type"] == "tool_result":
                c = block.get("content")
                if isinstance(c, str):
                    first, _, rest = c.partition("\n")
                    assert MARKER.match(c) or (not rest or MARKER.match(rest)), (path.name, c[:80])
            elif block["type"] == "text":
                assert MARKER.match(block["text"]), (path.name, block["text"][:60])
