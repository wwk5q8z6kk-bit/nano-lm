"""v1.0 is frozen permanently (owner decision D1, 2026-09-13).

Every file that defines, produces, pins, or tests the v1.0 action-log contract
must be byte-identical to its content at the freeze commit. This test compares
the working tree against `git show <freeze>:<path>` so that a change to any
frozen file fails the suite even if every other test still passes — the way
the v1.1 work is kept from contaminating v1.0 by construction rather than by
care.

Skips only when git is unavailable; a missing frozen path is a failure, not a
skip, because deletion is also a modification.

Run explicitly (``benchmarks`` is not in ``pytest.ini`` testpaths):

    pytest benchmarks/ -q
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FREEZE_COMMIT = "18e410c"

FROZEN_FILES = [
    "benchmarks/AGENT_ACTION_LOG.schema.json",
    "benchmarks/AGENT_RUN_IDENTITY.schema.json",
    "benchmarks/agent_identity.py",
    "benchmarks/transcript_emitter.py",
    "benchmarks/transcript_sanitizer.py",
    "benchmarks/process_metrics.py",
    "benchmarks/gold_metrics.py",
    "benchmarks/confidence.py",
    "benchmarks/validation/instrument_validation_transcripts.py",
    "benchmarks/tests/test_agent_identity.py",
    "benchmarks/tests/test_process_metrics.py",
    "benchmarks/tests/test_gold_metrics.py",
    "benchmarks/tests/test_confidence.py",
    "benchmarks/tests/test_contract_consistency.py",
    "benchmarks/tests/test_transcript_emitter.py",
    "benchmarks/tests/test_transcript_sanitizer.py",
]
FROZEN_DIRS = ["benchmarks/fixtures/transcripts"]

#: The five fixture corpora were NOT in git at the freeze commit: `.gitignore`'s
#: `*.jsonl` data rule silently excluded them from both v1.0 fixture commits, so
#: `git show 18e410c:<path>` cannot serve as their reference. They are pinned
#: here by the sha256 of the on-disk content that the 62-transcript validation
#: and every v1.0 characterization pin were computed against, and are tracked
#: from the commit that adds this block onward (`.gitignore` override).
FROZEN_FIXTURE_DIGESTS = {
    "benchmarks/fixtures/transcripts/session_large_varied.jsonl": "ccff09f6aa2245b3cf73d6bc8858525c80b00600bb38ff8a1c804d0790045d65",
    "benchmarks/fixtures/transcripts/session_small_errors.jsonl": "d480c3ab6bd4c826da38761d3c4e35ae0098f628469f570e0c23e3491f664de9",
    "benchmarks/fixtures/transcripts/session_small_reads.jsonl": "11895f4617793f8353931a2713af4f1aa454c0b00385040673482dda9da53223",
    "benchmarks/fixtures/transcripts/session_trivial.jsonl": "766668691ae480a711829080a4db7515b8fde28aea5e4378553d1eb7fe86acf4",
    "benchmarks/fixtures/transcripts/synthetic_minimal.jsonl": "b069b1dae789e90fe420caf89a61e02660f217f6460ccda3d4e8a26a5e456fc0",
}


def _git_available() -> bool:
    if shutil.which("git") is None:
        return False
    probe = subprocess.run(["git", "cat-file", "-e", f"{FREEZE_COMMIT}^{{commit}}"], cwd=REPO, capture_output=True)
    return probe.returncode == 0


def _tracked_at_freeze(d: str) -> set[str]:
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", FREEZE_COMMIT, d], cwd=REPO, capture_output=True, text=True)
    return {line for line in out.stdout.splitlines() if line}


def _frozen_paths() -> list[str]:
    paths = list(FROZEN_FILES)
    if _git_available():
        for d in FROZEN_DIRS:
            paths.extend(sorted(_tracked_at_freeze(d)))
    return paths


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("path", _frozen_paths(), ids=lambda p: p)
def test_frozen_file_is_byte_identical_to_freeze_commit(path):
    if not _git_available():
        pytest.skip("git or the freeze commit is unavailable")
    frozen = subprocess.run(["git", "show", f"{FREEZE_COMMIT}:{path}"], cwd=REPO, capture_output=True)
    assert frozen.returncode == 0, f"{path} does not exist at {FREEZE_COMMIT}"
    current = REPO / path
    assert current.exists(), f"{path} was deleted; deletion is a modification of the frozen v1.0 path"
    assert _sha(current.read_bytes()) == _sha(frozen.stdout), (
        f"{path} differs from {FREEZE_COMMIT}. v1.0 is frozen; put the change in benchmarks/v1_1/."
    )


def test_frozen_fixture_directory_has_no_additions():
    """A new file under the frozen fixture directory is also a modification."""
    if not _git_available():
        pytest.skip("git or the freeze commit is unavailable")
    for d in FROZEN_DIRS:
        expected = _tracked_at_freeze(d) | {p for p in FROZEN_FIXTURE_DIGESTS if p.startswith(d)}
        now = {str(p.relative_to(REPO)) for p in (REPO / d).rglob("*") if p.is_file() and "__pycache__" not in p.parts}
        assert now == expected, f"frozen directory {d} changed: added={sorted(now - expected)} removed={sorted(expected - now)}"


@pytest.mark.parametrize("path,digest", sorted(FROZEN_FIXTURE_DIGESTS.items()), ids=lambda x: x if isinstance(x, str) and "/" in x else "")
def test_frozen_fixture_corpus_matches_pinned_digest(path, digest):
    """Content pin for the fixture files the freeze commit could not carry."""
    current = REPO / path
    assert current.exists(), f"{path} is missing; the v1.0 fixture corpus is frozen"
    assert _sha(current.read_bytes()) == digest, f"{path} content changed; v1.0 fixtures are frozen"


def test_v1_0_schema_version_const_is_unchanged():
    import json

    schema = json.loads((REPO / "benchmarks/AGENT_ACTION_LOG.schema.json").read_text())
    assert schema["properties"]["schema_version"]["const"] == "benchmark_lab.agent_action_log.v1"
