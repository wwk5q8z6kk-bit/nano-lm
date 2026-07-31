"""Owner-corpus contact pins (public fixture / demo path)."""
from __future__ import annotations

import json
from pathlib import Path

from wedge_v1.run_owner_dogfood import DEFAULT_TASKS, FIXTURE_CORPUS, SMOKE_OUT, main, run
from wedge_v1.run_owner_smoke import EXAMPLE, run as smoke_run


def test_example_corpus_present():
    assert FIXTURE_CORPUS.is_dir() or EXAMPLE.is_dir()
    assert DEFAULT_TASKS.is_file()


def test_owner_smoke_example_pass():
    out = smoke_run(EXAMPLE)
    assert out["n_tasks"] >= 5
    assert out["n_ok"] == 5
    assert out["rows"][1]["got_status"] == "CONTRADICTED"
    assert out["rows"][4]["got_status"] == "ABSTAIN"


def test_owner_smoke_example_all_pass(tmp_path: Path | None = None):
    test_owner_smoke_example_pass()


def test_owner_dogfood_demo_pass():
    rc = main(["--demo", "--smoke"])
    assert rc == 0
    assert SMOKE_OUT.is_file()
    data = json.loads(SMOKE_OUT.read_text())
    assert data["n_tasks"] >= 5
    assert data["n_ok"] == 5


def test_owner_dogfood_via_run(tmp_path):
    out = tmp_path / "out.json"
    corpus = FIXTURE_CORPUS if FIXTURE_CORPUS.is_dir() else EXAMPLE
    result = run(corpus, DEFAULT_TASKS, out_json=out)
    assert out.exists()
    assert result.get("error") != "NO_CORPUS"
    assert result["n_ok"] == result["n_tasks"]
