"""Owner-corpus Wedge v1 dogfood harness — classical only; results gitignored.

Accepts --corpus outside the repo (or gitignored). Never commits PHI.
Not Layer-1 evidence. No LM. No training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wedge_v1.ingest import corpus_stats, load_corpus
from wedge_v1.private_output import PRIVATE_EXPORT_ROOT, require_private_output
from wedge_v1.runtime import DEFAULT_CORPUS, ask, compare, find_spans, normalize_doc_ids

ROOT = Path(__file__).resolve().parent
FIXTURE_CORPUS = ROOT / "fixtures" / "owner_corpus"
EXAMPLE_CORPUS = FIXTURE_CORPUS  # public synthetic stand-in
DEFAULT_TASKS = ROOT / "data" / "owner_dogfood_tasks.example.json"
DEFAULT_OUTPUT_DIR = PRIVATE_EXPORT_ROOT / "owner-dogfood"
DEFAULT_OUT = DEFAULT_OUTPUT_DIR / "results_owner_dogfood.json"
SMOKE_OUT = DEFAULT_OUTPUT_DIR / "results_owner_smoke.json"
DEFAULT_GALLERY_MD = DEFAULT_OUTPUT_DIR / "results_owner_failure_gallery.md"
DEFAULT_GALLERY_JSON = DEFAULT_OUTPUT_DIR / "results_owner_failure_gallery.json"
OWNER_DIR = ROOT / "data" / "owner_corpus"


def _blob(result: dict) -> str:
    return json.dumps(result, default=str).lower()


def resolve_corpus(*, corpus: Path | None = None, demo: bool = False) -> Path:
    """Prefer explicit path, then OWNER_CORPUS, then gitignored folder, then fixture."""
    if demo:
        if FIXTURE_CORPUS.is_dir() and any(FIXTURE_CORPUS.glob("*.md")):
            return FIXTURE_CORPUS
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
    raw_doc_ids = task.get("doc_ids")
    if raw_doc_ids is not None and not isinstance(raw_doc_ids, list):
        raise ValueError(f"task {task.get('id')} doc_ids must be a list")
    doc_ids = normalize_doc_ids(raw_doc_ids)
    if mode == "compare":
        result = compare(q, corpus_dir=corpus, doc_ids=doc_ids)
    elif mode == "find":
        result = find_spans(q, corpus_dir=corpus, doc_ids=doc_ids)
    else:
        result = ask(q, corpus_dir=corpus, doc_ids=doc_ids)

    status = result.get("answer_status")
    expect = task.get("expect_status") or ["any"]
    if "any" in expect:
        ok_status = status not in {None, "NO_CORPUS"}
    else:
        ok_status = status in expect
    needles = task.get("must_contain_any") or []
    if list(expect) == ["ABSTAIN"] or not needles:
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
        "doc_ids": doc_ids,
        "selected_doc_ids": result.get("selected_doc_ids"),
        "missing_doc_ids": result.get("missing_doc_ids"),
        "expect_status": expect,
        "got_status": status,
        "ok_status": ok_status,
        "ok_needles": ok_needles,
        "fail_kind": fail_kind,
        "n_claims": len(result.get("claims") or result.get("hits") or []),
        "solver_path": result.get("solver_path"),
        "latency_s": result.get("latency_s"),
        "note": result.get("note"),
    }


def failure_gallery_md(rows: list[dict], corpus: Path) -> str:
    fails = [r for r in rows if not r["ok"]]
    lines = [
        "# Owner dogfood failure gallery",
        "",
        f"Corpus: `{corpus}`",
        f"Failures: {len(fails)} / {len(rows)}",
        "",
        "_Internal Wedge v1 eval — not Nano AI validation or Layer-1 evidence._",
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
    smoke_out: Path | None = None,
) -> dict:
    docs = load_corpus(corpus)
    if not docs:
        return {
            "schema": "nano-lm.wedge_v1.owner_dogfood_result.v1",
            "corpus_class": "OWNER_PRIVATE",
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
        "corpus_class": "OWNER_PRIVATE",
        "corpus": str(Path(corpus).resolve()),
        "tasks_path": str(tasks_path),
        "n_docs": len(docs),
        "ingest": stats,
        "n_tasks": len(rows),
        "n_ok": n_ok,
        "accuracy": n_ok / max(1, len(rows)),
        "rows": rows,
        "note": (
            "Owner/local Wedge v1 dogfood; gitignored results; not Nano AI "
            "validation or a Layer-1 ledger claim."
        ),
    }
    explicit_outputs = (out_json, gallery_md, gallery_json, smoke_out)
    anchor = next(
        (Path(path).parent for path in explicit_outputs if path is not None),
        None,
    )

    def output_path(explicit: Path | None, default: Path) -> Path:
        if explicit is not None:
            return Path(explicit)
        if anchor is not None:
            return anchor / default.name
        return default

    dest = output_path(out_json, DEFAULT_OUT)
    gal_path = output_path(gallery_md, DEFAULT_GALLERY_MD)
    gal_json_path = output_path(gallery_json, DEFAULT_GALLERY_JSON)
    smoke_path = output_path(smoke_out, SMOKE_OUT)
    write_smoke = bool(write_smoke or smoke_out is not None)
    written = [dest, gal_path, gal_json_path]
    if write_smoke:
        written.append(smoke_path)
    private_input = (
        Path(corpus).expanduser().resolve() != FIXTURE_CORPUS.resolve()
        or Path(tasks_path).expanduser().resolve() != DEFAULT_TASKS.resolve()
    )
    if private_input:
        for path in written:
            require_private_output(path, purpose="owner dogfood output")
    for path in written:
        path.parent.mkdir(parents=True, exist_ok=True)
    out["written"] = [str(path) for path in written]
    out["out"] = str(dest)
    out["gallery_md"] = str(gal_path)
    out["gallery_json"] = str(gal_json_path)
    out["smoke_out"] = str(smoke_path) if write_smoke else None
    payload = json.dumps(out, indent=2) + "\n"
    dest.write_text(payload, encoding="utf-8")
    if write_smoke:
        smoke_path.write_text(payload, encoding="utf-8")

    gal_body = failure_gallery_md(rows, Path(corpus))
    gal_path.write_text(gal_body, encoding="utf-8")
    gal_json_path.write_text(
        json.dumps(
            {
                "schema": "nano-lm.wedge_v1.failure_gallery.v1",
                "corpus_class": "OWNER_PRIVATE",
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
                "note": (
                    "Internal Wedge v1 failure gallery; not Nano AI validation "
                    "or Layer-1 evidence."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Owner-corpus Wedge v1 classical dogfood "
            "(gitignored outputs; no PHI in git)"
        )
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
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--gallery", type=Path, default=None, help="Failure gallery markdown path"
    )
    ap.add_argument(
        "--gallery-json", type=Path, default=None, help="Failure gallery JSON path"
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Also write a private smoke-result copy",
    )
    ap.add_argument(
        "--smoke-out",
        type=Path,
        default=None,
        help="Smoke result path; providing it also enables the smoke copy",
    )
    args = ap.parse_args(argv)

    corpus = resolve_corpus(corpus=args.corpus, demo=args.demo)
    out = run(
        corpus,
        args.tasks,
        out_json=args.out,
        write_smoke=bool(args.smoke),
        gallery_md=args.gallery,
        gallery_json=args.gallery_json,
        smoke_out=args.smoke_out,
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
                "gallery_json": out.get("gallery_json"),
                "smoke_out": out.get("smoke_out"),
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
