"""Owner-corpus contact pins (fixture / demo path)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import wedge_v1.run_corpus_contact as corpus_contact
import wedge_v1.run_owner_dogfood as owner_dogfood
from wedge_v1.cli import main as cli_main
from wedge_v1.owner_ready import check as readiness_check
from wedge_v1.private_output import PRIVATE_CORPUS_ROOT, PRIVATE_EXPORT_ROOT
from wedge_v1.run_owner_dogfood import (
    DEFAULT_GALLERY_JSON,
    DEFAULT_GALLERY_MD,
    DEFAULT_OUT,
    DEFAULT_TASKS,
    FIXTURE_CORPUS,
    SMOKE_OUT,
    main,
    run,
)


def test_example_corpus_present():
    assert FIXTURE_CORPUS.is_dir()
    assert any(FIXTURE_CORPUS.glob("*.md"))
    assert DEFAULT_TASKS.is_file()


def test_default_owner_outputs_stay_out_of_the_document_corpus():
    for path in (DEFAULT_OUT, DEFAULT_GALLERY_MD, DEFAULT_GALLERY_JSON, SMOKE_OUT):
        assert path.is_relative_to(PRIVATE_EXPORT_ROOT)
        assert not path.is_relative_to(PRIVATE_CORPUS_ROOT)


def test_owner_dogfood_demo_pass(tmp_path: Path):
    out = tmp_path / "results_owner_dogfood.json"
    gallery_md = tmp_path / "results_owner_failure_gallery.md"
    gallery_json = tmp_path / "results_owner_failure_gallery.json"
    smoke_out = tmp_path / "results_owner_smoke.json"
    rc = main(
        [
            "--demo",
            "--out",
            str(out),
            "--gallery",
            str(gallery_md),
            "--gallery-json",
            str(gallery_json),
            "--smoke-out",
            str(smoke_out),
        ]
    )
    assert rc == 0
    assert all(path.is_file() for path in (out, gallery_md, gallery_json, smoke_out))
    data = json.loads(out.read_text())
    assert data["n_tasks"] == 5
    assert data["n_ok"] == data["n_tasks"]
    assert data["out"] == str(out)
    assert data["gallery_md"] == str(gallery_md)
    assert data["gallery_json"] == str(gallery_json)
    assert data["smoke_out"] == str(smoke_out)
    assert data["written"] == [
        str(out),
        str(gallery_md),
        str(gallery_json),
        str(smoke_out),
    ]
    assert json.loads(gallery_json.read_text())["source"] == str(out)
    assert smoke_out.read_text() == out.read_text()


def test_fully_injected_cli_run_leaves_default_owner_artifacts_untouched(
    tmp_path: Path, monkeypatch
):
    default_paths = {
        "out": tmp_path / "default-results.json",
        "gallery_md": tmp_path / "default-gallery.md",
        "gallery_json": tmp_path / "default-gallery.json",
        "smoke": tmp_path / "default-smoke.json",
    }
    for name, path in default_paths.items():
        path.write_text(f"sentinel:{name}\n", encoding="utf-8")
    before = {name: path.read_bytes() for name, path in default_paths.items()}

    monkeypatch.setattr(owner_dogfood, "DEFAULT_OUT", default_paths["out"])
    monkeypatch.setattr(owner_dogfood, "DEFAULT_GALLERY_MD", default_paths["gallery_md"])
    monkeypatch.setattr(owner_dogfood, "DEFAULT_GALLERY_JSON", default_paths["gallery_json"])
    monkeypatch.setattr(owner_dogfood, "SMOKE_OUT", default_paths["smoke"])

    for command, prefix, out_flag in (
        (["owner-dogfood", "--demo"], "dogfood", "--out"),
        (["owner-smoke"], "smoke", "--output"),
    ):
        injected = {
            "out": tmp_path / f"{prefix}-results.json",
            "gallery_md": tmp_path / f"{prefix}-gallery.md",
            "gallery_json": tmp_path / f"{prefix}-gallery.json",
            "smoke": tmp_path / f"{prefix}-smoke.json",
        }
        rc = cli_main(
            command
            + [
                out_flag,
                str(injected["out"]),
                "--gallery",
                str(injected["gallery_md"]),
                "--gallery-json",
                str(injected["gallery_json"]),
                "--smoke-out",
                str(injected["smoke"]),
            ]
        )

        assert rc == 0
        assert all(path.is_file() for path in injected.values())
    assert {name: path.read_bytes() for name, path in default_paths.items()} == before


def test_each_single_output_override_routes_unspecified_companions_beside_it(
    tmp_path: Path, monkeypatch
):
    fixed_dir = tmp_path / "fixed"
    fixed_dir.mkdir()
    default_paths = {
        "out": fixed_dir / "default-results.json",
        "gallery_md": fixed_dir / "default-gallery.md",
        "gallery_json": fixed_dir / "default-gallery.json",
        "smoke": fixed_dir / "default-smoke.json",
    }
    for name, path in default_paths.items():
        path.write_text(f"sentinel:{name}\n", encoding="utf-8")
    before = {name: path.read_bytes() for name, path in default_paths.items()}

    monkeypatch.setattr(owner_dogfood, "DEFAULT_OUT", default_paths["out"])
    monkeypatch.setattr(owner_dogfood, "DEFAULT_GALLERY_MD", default_paths["gallery_md"])
    monkeypatch.setattr(owner_dogfood, "DEFAULT_GALLERY_JSON", default_paths["gallery_json"])
    monkeypatch.setattr(owner_dogfood, "SMOKE_OUT", default_paths["smoke"])

    cases = (
        ("out", "--out", "custom-result.json", False),
        ("gallery_md", "--gallery", "custom-gallery.md", False),
        ("gallery_json", "--gallery-json", "custom-gallery.json", False),
        ("smoke", "--smoke-out", "custom-smoke.json", True),
    )
    for case_name, flag, filename, writes_smoke in cases:
        output_dir = tmp_path / case_name
        explicit = output_dir / filename
        rc = main(["--demo", flag, str(explicit)])

        assert rc == 0
        expected = {
            "out": explicit if case_name == "out" else output_dir / default_paths["out"].name,
            "gallery_md": (
                explicit
                if case_name == "gallery_md"
                else output_dir / default_paths["gallery_md"].name
            ),
            "gallery_json": (
                explicit
                if case_name == "gallery_json"
                else output_dir / default_paths["gallery_json"].name
            ),
        }
        if writes_smoke:
            expected["smoke"] = explicit
        assert all(path.is_file() for path in expected.values())

        result = json.loads(expected["out"].read_text(encoding="utf-8"))
        assert result["out"] == str(expected["out"])
        assert result["gallery_md"] == str(expected["gallery_md"])
        assert result["gallery_json"] == str(expected["gallery_json"])
        assert result["smoke_out"] == (
            str(expected["smoke"]) if writes_smoke else None
        )

    assert {name: path.read_bytes() for name, path in default_paths.items()} == before


def test_owner_smoke_single_output_override_routes_every_written_artifact(
    tmp_path: Path, monkeypatch
):
    fixed_dir = tmp_path / "fixed"
    fixed_dir.mkdir()
    default_paths = {
        "out": fixed_dir / "default-results.json",
        "gallery_md": fixed_dir / "default-gallery.md",
        "gallery_json": fixed_dir / "default-gallery.json",
        "smoke": fixed_dir / "default-smoke.json",
    }
    for name, path in default_paths.items():
        path.write_text(f"sentinel:{name}\n", encoding="utf-8")
    before = {name: path.read_bytes() for name, path in default_paths.items()}

    monkeypatch.setattr(owner_dogfood, "DEFAULT_OUT", default_paths["out"])
    monkeypatch.setattr(owner_dogfood, "DEFAULT_GALLERY_MD", default_paths["gallery_md"])
    monkeypatch.setattr(owner_dogfood, "DEFAULT_GALLERY_JSON", default_paths["gallery_json"])
    monkeypatch.setattr(owner_dogfood, "SMOKE_OUT", default_paths["smoke"])

    output_dir = tmp_path / "owner-smoke"
    explicit = output_dir / "custom-result.json"
    rc = cli_main(["owner-smoke", "--output", str(explicit)])

    assert rc == 0
    expected = (
        explicit,
        output_dir / default_paths["gallery_md"].name,
        output_dir / default_paths["gallery_json"].name,
        output_dir / default_paths["smoke"].name,
    )
    assert all(path.is_file() for path in expected)
    assert {name: path.read_bytes() for name, path in default_paths.items()} == before


def test_cli_smoke_scopes_coe_records(tmp_path: Path, monkeypatch):
    import wedge_v1.coe.bind as bind_module

    fixed_record_dir = tmp_path / "fixed-coe-records"
    fixed_record_dir.mkdir()
    sentinel = fixed_record_dir / "sentinel.txt"
    sentinel.write_text("protected\n", encoding="utf-8")
    monkeypatch.setattr(bind_module, "DEFAULT_RECORD_DIR", fixed_record_dir)

    assert cli_main(["smoke"]) == 0

    assert sorted(path.name for path in fixed_record_dir.iterdir()) == [sentinel.name]
    assert sentinel.read_text(encoding="utf-8") == "protected\n"


def test_owner_smoke_example_pass(tmp_path: Path):
    """CLI smoke entry — fixture pack via run()."""
    dest = tmp_path / "results_owner_dogfood_smoke.json"
    gallery_md = tmp_path / "failure_gallery.md"
    gallery_json = tmp_path / "failure_gallery.json"
    result = run(
        FIXTURE_CORPUS,
        DEFAULT_TASKS,
        out_json=dest,
        gallery_md=gallery_md,
        gallery_json=gallery_json,
    )
    assert all(path.is_file() for path in (dest, gallery_md, gallery_json))
    assert result["n_ok"] == result["n_tasks"]
    assert result["n_tasks"] == 5
    assert result["written"] == [str(dest), str(gallery_md), str(gallery_json)]
    assert list((tmp_path / "coe_runs").glob("*.jsonl"))


def test_owner_dogfood_honors_task_document_scope(tmp_path: Path):
    tasks = tmp_path / "scoped_tasks.json"
    tasks.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.owner_dogfood.v1",
                "tasks": [
                    {
                        "id": "SCOPED_TTL",
                        "mode": "compare",
                        "query": "TTL",
                        "doc_ids": ["note_cache_policy"],
                        "expect_status": ["SUPPORTED"],
                        "must_contain_any": ["300"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "scoped_result.json"
    gallery_md = tmp_path / "scoped_gallery.md"
    gallery_json = tmp_path / "scoped_gallery.json"

    result = run(
        FIXTURE_CORPUS,
        tasks,
        out_json=out,
        gallery_md=gallery_md,
        gallery_json=gallery_json,
    )

    row = result["rows"][0]
    assert row["got_status"] == "SUPPORTED"
    assert row["doc_ids"] == ["note_cache_policy"]
    assert row["selected_doc_ids"] == ["note_cache_policy"]
    assert row["missing_doc_ids"] == []
    assert row["ok"] is True


def test_explicit_small_local_corpus_is_smoke_ready_not_representative(tmp_path: Path):
    corpus = tmp_path / "private-corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("Local note with evidence.\n", encoding="utf-8")

    report = readiness_check(corpus, tasks=tmp_path / "missing-tasks.json")

    assert report["smoke_ready"] is True
    assert report["representative_ready"] is False
    assert report["ready_for_private_run"] is False
    assert "TOO_FEW_DOCUMENTS" in report["blockers"]
    assert "TASK_PACK_MISSING" in report["blockers"]
    assert "OWNER_CORPUS_PENDING" not in report["blockers"]


def test_private_contact_requires_explicit_nondefault_corpus():
    with pytest.raises(ValueError, match="explicit --corpus"):
        corpus_contact.run_contact(None, corpus_class="OWNER_PRIVATE")
    with pytest.raises(ValueError, match="synthetic default"):
        corpus_contact.run_contact(
            corpus_contact.DEFAULT_CORPUS,
            corpus_class="OWNER_PRIVATE",
        )


def test_private_contact_default_routes_to_ignored_owner_namespace(
    tmp_path: Path, monkeypatch
):
    corpus = tmp_path / "private-corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    owner_output = tmp_path / "results_owner_corpus_contact.json"
    public_output = tmp_path / "results_corpus_contact.json"
    monkeypatch.setattr(corpus_contact, "OWNER_CONTACT_OUTPUT", owner_output)
    monkeypatch.setattr(corpus_contact, "PUBLIC_CONTACT_OUTPUT", public_output)

    assert cli_main(
        [
            "contact",
            "--class",
            "OWNER_PRIVATE",
            "--corpus",
            str(corpus),
        ]
    ) == 0

    payload = json.loads(owner_output.read_text(encoding="utf-8"))
    assert payload["corpus_class"] == "OWNER_PRIVATE"
    assert payload["corpus"] == str(corpus.resolve())
    assert not public_output.exists()


def test_known_owner_contact_cannot_downgrade_to_public_output(
    tmp_path: Path, monkeypatch
):
    corpus = tmp_path / "owner-corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    owner_output = tmp_path / "results_owner_corpus_contact.json"
    public_output = tmp_path / "results_corpus_contact.json"
    monkeypatch.setattr(corpus_contact, "OWNER_CORPUS_DIR", corpus)
    monkeypatch.setattr(corpus_contact, "OWNER_CONTACT_OUTPUT", owner_output)
    monkeypatch.setattr(corpus_contact, "PUBLIC_CONTACT_OUTPUT", public_output)

    payload = corpus_contact.run_contact(corpus, corpus_class="PAPERS_DOGFOOD")

    assert payload["storage_class"] == "OWNER_PRIVATE"
    assert (
        corpus_contact.default_contact_output("PAPERS_DOGFOOD", corpus)
        == owner_output
    )
    assert not public_output.exists()


def test_private_contact_rejects_commit_visible_explicit_output(tmp_path: Path):
    corpus = tmp_path / "private-corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    unsafe = corpus_contact.ROOT.parent / "private-contact-leak-test.json"
    assert not unsafe.exists()

    with pytest.raises(SystemExit) as exc:
        cli_main(
            [
                "contact",
                "--class",
                "SYNTHETIC_MINI",
                "--corpus",
                str(corpus),
                "--output",
                str(unsafe),
            ]
        )

    assert exc.value.code == 2
    assert not unsafe.exists()


def test_private_owner_run_rejects_commit_visible_output(tmp_path: Path):
    corpus = tmp_path / "private-corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("Cache TTL is 300 seconds.\n", encoding="utf-8")
    unsafe = owner_dogfood.ROOT.parent / "private-owner-leak-test.json"
    assert not unsafe.exists()

    with pytest.raises(ValueError, match="ignored private namespace"):
        run(corpus, DEFAULT_TASKS, out_json=unsafe)

    assert not unsafe.exists()


def test_custom_tasks_make_fixture_run_private(tmp_path: Path):
    tasks = tmp_path / "private-tasks.json"
    tasks.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "PRIVATE_TASK",
                        "mode": "ask",
                        "query": "PRIVATE_QUERY",
                        "expect_status": ["ABSTAIN"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    unsafe = owner_dogfood.ROOT.parent / "private-task-leak-test.json"
    assert not unsafe.exists()

    with pytest.raises(ValueError, match="ignored private namespace"):
        run(FIXTURE_CORPUS, tasks, out_json=unsafe)

    assert not unsafe.exists()
