"""Owner-corpus contact pins (fixture / demo path)."""
from __future__ import annotations

from pathlib import Path

from wedge_v1.run_owner_dogfood import DEFAULT_TASKS, EXAMPLE_TASKS, FIXTURE_CORPUS, run


def _tasks() -> Path:
    return EXAMPLE_TASKS if EXAMPLE_TASKS.is_file() else DEFAULT_TASKS


def _run(base: Path):
    out = base / "results_owner_dogfood.json"
    gal = base / "gallery.md"
    gal_json = base / "gallery.json"
    result = run(
        FIXTURE_CORPUS,
        _tasks(),
        out_json=out,
        gallery_md=gal,
        gallery_json=gal_json,
    )
    assert out.is_file()
    assert result["n_tasks"] >= 5
    assert result["n_ok"] == result["n_tasks"], [r for r in result.get("rows", []) if not r.get("ok")]
    return result


def test_example_corpus_present():
    assert FIXTURE_CORPUS.is_dir()
    assert any(FIXTURE_CORPUS.glob("*.md"))
    assert _tasks().is_file()


def test_owner_dogfood_demo_pass(tmp_path: Path | None = None):
    _run(Path(tmp_path) if tmp_path is not None else Path("/tmp"))


def test_owner_smoke_example_pass(tmp_path: Path | None = None):
    _run(Path(tmp_path) if tmp_path is not None else Path("/tmp"))
