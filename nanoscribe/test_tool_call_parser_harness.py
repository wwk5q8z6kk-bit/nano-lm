# Parser harness fixture tests.
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanoscribe.tool_parsing.harness import assert_all_fixtures_pass, run_all_fixtures


def test_all_tool_call_fixtures_pass() -> None:
    count = assert_all_fixtures_pass()
    assert count >= 8


def test_fixture_results_all_named() -> None:
    results = run_all_fixtures()
    assert all(result.name for result in results)
    assert all(result.passed for result in results)
