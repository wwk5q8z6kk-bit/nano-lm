"""Owner-corpus dogfood harness — classical only; results gitignored.

Accepts --corpus outside the repo (or gitignored). Never commits PHI.
Not Layer-1 evidence. No LM. No training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wedge_v1.ingest import corpus_stats, load_corpus
from wedge_v1.runtime import DEFAULT_CORPUS, ask, compare, find_spans

ROOT = Path(__file__).resolve().parent
FIXTURE_CORPUS = ROOT / "fixtures" / "owner_corpus"
EXAMPLE_CORPUS = FIXTURE_CORPUS  # public synthetic stand-in
LEGACY_EXAMPLE = ROOT / "data" / "owner_corpus.example"
DEFAULT_TASKS = ROOT / "data" / "owner_dogfood_tasks.json"
DEMO_TASKS = ROOT / "data" / "owner_dogfood_tasks_demo.json"
EXAMPLE_TASKS = ROOT / "data" / "owner_dogfood_tasks.example.json"
if not DEFAULT_TASKS.is_file():
    DEFAULT_TASKS = EXAMPLE_TASKS
DEFAULT_OUT = ROOT / "results_owner_dogfood.json"
SMOKE_OUT = ROOT / "results_owner_smoke.json"
DEFAULT_GALLERY_MD = ROOT / "results_owner_failure_gallery.md"
DEFAULT_GALLERY_JSON = ROOT / "results_owner_failure_gallery.json"
OWNER_DIR = ROOT / "data" / "owner_corpus"


def _blob(result: dict) -> str:
    return json.dumps(result, default=str).lower()


def _scoped_corpus_dir(corpus: Path, doc_ids: list[str] | None) -> tuple[Path, Path | None]:
    if not doc_ids:
        return corpus, None
    docs = load_corpus(corpus)
    wanted = [d for d in doc_ids if d in docs]
    if not wanted or len(wanted) == len(docs):
        return corpus, None
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="wedge_scope_"))
    for did in wanted:
        (tmp / f"{did}.md").write_text(docs[did], encoding="utf-8")
    return tmp, tmp


def resolve_corpus(*, corpus: Path | None = None, demo: bool = False) -> Path:
    """Prefer explicit path, then OWNER_CORPUS, then gitignored folder, then fixture."""
    if demo:
        if FIXTURE_CORPUS.is_dir() and any(FIXTURE_CORPUS.glob("*.md")):
            return FIXTURE_CORPUS
        if LEGACY_EXAMPLE.is_dir() and any(LEGACY_EXAMPLE.glob("*.md")):
            return LEGACY_EXAMPLE
        return DEFAULT_CORPUS
    if corpus is not None:
        return Path(corpus).expanduser()
    env = (os.environ.get("OWNER_CORPUS") or os.environ.get("WEDGE_OWNER_CORPUS") or "").strip()
    if env:
        return Path(env).expanduser()
    if OWNER_DIR.is_dir():
        files = [
            p
            for p in OWNER_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in {".md", ".txt", ".pdf"}
        ]
        if files:
            return OWNER_DIR
    if FIXTURE_CORPUS.is_dir() and any(FIXTURE_CORPUS.glob("*.md")):
        return FIXTURE_CORPUS
    return DEFAULT_CORPUS


def score_task(task: dict, corpus: Path) -> dict:
    q = task["query"]
    mode = task.get("mode") or "ask"
    scoped, tmp = _scoped_corpus_dir(corpus, task.get("doc_ids"))
    try:
        if mode == "compare":
            result = compare(q, corpus_dir=scoped)
        elif mode == "find":
            result = find_spans(q, corpus_dir=scoped)
        else:
            result = ask(q, corpus_dir=scoped)
    finally:
        if tmp is not None:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)

    status = result.get("answer_status")
    expect = task.get("expect_status") or ["any"]
    if "any" in expect:
        ok_status = status not in {None, "NO_CORPUS"}
    else:
        ok_status = status in expect
    needles = task.get("must_contain_any") or []
    if list(expect) == ["ABSTAIN"]:
        ok_needles = True
    elif not needles:
        ok_needles = True
    else:
        ok_needles = any(n.lower() in _blob(result) for n in needles)
    ok = bool(ok_status and ok_needles)

    fail_kind = None
    if not ok:
        if status == "ABSTAIN" and any(s in expect for s in ("SUPPORTED", "CONTRADICTED")):
            fail_kind = "over_abstain"
        elif status in {"SUPPORTED", "CONTRADICTED"} and list(expect) == ["ABSTAIN"]:
            fail_kind = "over_answer"
        elif not ok_needles:
            fail_kind = "wrong_span_or_miss"
        else:
            fail_kind = "status_mismatch"

    return {
        "id": task["id"],
        "ok": ok,
        "mode": mode,
        "query": q,
        "expect_status": expect,
        "got_status": status,
        "ok_status": ok_status,
        "ok_needles": ok_needles,
        "fail_kind": fail_kind,
        "n_claims": len(result.get("claims") or result.get("hits") or []),
        "solver_path": result.get("solver_path"),
        "latency_s": result.get("latency_s"),
        "note": result.get("note"),
        "scoped_doc_ids": task.get("doc_ids"),
    }


def failure_gallery_md(rows: list[dict], corpus: Path) -> str:
    fails = [r for r in rows if not r["ok"]]
    lines = [
        "# Owner dogfood failure gallery",
        "",
        f"Corpus: `{corpus}`",
        f"Failures: {len(fails)} / {len(rows)}",
        "",
        "_Local product eval — not Layer-1 evidence._",
        "",
    ]
    if not fails:
        lines.append("No failures.")
        return "\n".join(lines) + "\n"
    by: dict[str, list] = {}
    for r in fails:
        by.setdefault(r.get("fail_kind") or "other", []).append(r)
    for kind, items in sorted(by.items()):
        lines.append(f"## {kind} ({len(items)})")
        for r in items:
            lines.append(
                f"- **{r['id']}** `{r['mode']}` status={r['got_status']} expect={r['expect_status']}"
            )
            lines.append(f"  - query: {r['query']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def run(
    corpus: Path,
    tasks_path: Path = DEFAULT_TASKS,
    out_json: Path | None = None,
    *,
    write_smoke: bool = False,
    gallery_md: Path | None = None,
    gallery_json: Path | None = None,
) -> dict:
    docs = load_corpus(corpus)
    if not docs:
        return {
            "schema": "nano-lm.wedge_v1.owner_dogfood_result.v1",
            "error": "NO_CORPUS",
            "corpus": str(corpus),
            "n_tasks": 0,
            "n_ok": 0,
            "accuracy": 0.0,
            "rows": [],
        }

    pack = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    rows = [score_task(t, corpus) for t in pack.get("tasks") or []]
    n_ok = sum(1 for r in rows if r["ok"])
    stats = corpus_stats(corpus)
    out = {
        "schema": "nano-lm.wedge_v1.owner_dogfood_result.v1",
        "corpus": str(Path(corpus).resolve()),
        "tasks_path": str(tasks_path),
        "n_docs": len(docs),
        "ingest": stats,
        "n_tasks": len(rows),
        "n_ok": n_ok,
        "accuracy": n_ok / max(1, len(rows)),
        "rows": rows,
        "note": "Owner/local dogfood; gitignored results; not Layer-1 ledger claim.",
    }
    dest = out_json or DEFAULT_OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(out, indent=2) + "\n"
    dest.write_text(payload, encoding="utf-8")
    if write_smoke:
        SMOKE_OUT.write_text(payload, encoding="utf-8")

    gal_body = failure_gallery_md(rows, Path(corpus))
    gal_path = gallery_md or DEFAULT_GALLERY_MD
    gal_path.write_text(gal_body, encoding="utf-8")
    # When caller redirects markdown gallery (tests/tmp), keep JSON beside it
    # so we never touch the protected default owner gallery path unintentionally.
    if gallery_json is not None:
        gal_json_path = gallery_json
    elif gallery_md is not None:
        gal_json_path = Path(gallery_md).with_suffix(".json")
    else:
        gal_json_path = DEFAULT_GALLERY_JSON
    gal_json_path.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.failure_gallery.v1",
                "source": str(dest),
                "n_tasks": len(rows),
                "n_ok": n_ok,
                "accuracy": out["accuracy"],
                "rows": [
                    {
                        "id": r["id"],
                        "ok": r["ok"],
                        "fail_kind": r.get("fail_kind"),
                        "got_status": r["got_status"],
                        "expect_status": r["expect_status"],
                        "query": r["query"],
                    }
                    for r in rows
                ],
                "note": "Product failure gallery; not Layer-1.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    out["written"] = [str(dest), str(gal_path), str(gal_json_path)]
    if write_smoke:
        out["written"].append(str(SMOKE_OUT))
    out["out"] = str(dest)
    out["gallery_md"] = str(gal_path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Owner-corpus classical dogfood (gitignored outputs; no PHI in git)"
    )
    ap.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Private folder outside git (default: $OWNER_CORPUS or fixture)",
    )
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Force public synthetic fixture at wedge_v1/fixtures/owner_corpus",
    )
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--gallery", type=Path, default=None, help="Failure gallery markdown path")
    ap.add_argument("--smoke", action="store_true", help="Also write results_owner_smoke.json")
    args = ap.parse_args(argv)
    if args.demo and DEMO_TASKS.is_file() and Path(args.tasks) == DEFAULT_TASKS:
        args.tasks = DEMO_TASKS

    corpus = resolve_corpus(corpus=args.corpus, demo=args.demo)
    if args.demo and Path(args.tasks) == DEFAULT_TASKS and DEMO_TASKS.is_file():
        args.tasks = DEMO_TASKS
    out = run(
        corpus,
        args.tasks,
        out_json=args.out,
        write_smoke=bool(args.smoke),
        gallery_md=args.gallery,
    )
    if out.get("error") == "NO_CORPUS":
        print(
            json.dumps(
                {
                    "error": "NO_CORPUS",
                    "corpus": out["corpus"],
                    "hint": (
                        "Pass --corpus ~/path/to/private/docs, set OWNER_CORPUS, "
                        "or use --demo for wedge_v1/fixtures/owner_corpus"
                    ),
                },
                indent=2,
            )
        )
        return 2

    print(
        json.dumps(
            {
                "accuracy": out["accuracy"],
                "n_ok": out["n_ok"],
                "n_tasks": out["n_tasks"],
                "corpus": out["corpus"],
                "out": out.get("out"),
                "gallery": out.get("gallery_md"),
                "rows": [
                    {
                        "id": r["id"],
                        "ok": r["ok"],
                        "got": r["got_status"],
                        "fail": r.get("fail_kind"),
                    }
                    for r in out["rows"]
                ],
            },
            indent=2,
        )
    )
    print("WEDGE_V1_OWNER_DOGFOOD_DONE")
    return 0 if out["n_ok"] == out["n_tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
