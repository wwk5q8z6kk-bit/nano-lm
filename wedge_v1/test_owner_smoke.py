"""Owner-corpus contact pins (fixture / demo path)."""
from __future__ import annotations

import json
from pathlib import Path

from wedge_v1.run_owner_dogfood import DEFAULT_TASKS, FIXTURE_CORPUS, main, run


def test_example_corpus_present():
    assert FIXTURE_CORPUS.is_dir()
    assert any(FIXTURE_CORPUS.glob("*.md"))
    assert DEFAULT_TASKS.is_file()


def test_owner_dogfood_demo_pass():
    rc = main(["--demo"])
    assert rc == 0
    out = Path(__file__).resolve().parent / "results_owner_dogfood.json"
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["n_tasks"] >= 5
    assert data["n_ok"] == data["n_tasks"]


def test_owner_smoke_example_pass(tmp_path: Path | None = None):
    """CLI smoke entry — fixture pack via run()."""
    dest = (tmp_path or Path("/tmp")) / "results_owner_dogfood_smoke.json"
    result = run(FIXTURE_CORPUS, DEFAULT_TASKS, out_json=dest)
    assert dest.is_file()
    assert result["n_ok"] == result["n_tasks"]
    assert result["n_tasks"] >= 5
