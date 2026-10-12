"""CI guard: fail if test collection silently shrinks.

The default pytest configuration once limited collection to fabric/ and
trajectory/, so 15 of 19 nanoscribe test files never ran — including the
regression pins for defects that invalidated a full round of native results. A
directory dropping out of collection is silent: the suite still passes, it just
tests less. This pins a floor so that failure becomes loud.

Raise TEST_FLOOR deliberately when real tests are added; never lower it to make
a red build green.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Verified by running: python3 -m pytest nanoscribe -q
# Set below the observed count so ordinary additions do not trip it, but high
# enough that losing a test FILE does.
TEST_FLOOR = 180

# Files that must always be collected. These hold the regression pins for the
# tokenizer truncation and corrupt-checkpoint defects.
REQUIRED_FILES = (
    "test_native_p1_eval.py",
    "test_agent_canary.py",
    "test_native_training.py",
    "test_campaign_datasets.py",
    "test_evidence_transport.py",
)


def _collect_count(target: str) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", target, "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    total = 0
    for line in proc.stdout.splitlines():
        # `-q --collect-only` prints "<path>: <n>" per file.
        if ":" in line and line.rsplit(":", 1)[-1].strip().isdigit():
            total += int(line.rsplit(":", 1)[-1].strip())
    return total


def test_nanoscribe_collection_meets_floor() -> None:
    count = _collect_count("nanoscribe")
    assert count >= TEST_FLOOR, (
        f"nanoscribe collection dropped to {count}, below floor {TEST_FLOOR}. "
        "A test directory or file has likely fallen out of discovery. Do not lower "
        "the floor to make this pass — find what stopped being collected."
    )


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_required_test_files_present(filename: str) -> None:
    path = ROOT / "nanoscribe" / filename
    assert path.is_file(), f"{filename} is missing; it holds load-bearing regression pins"
