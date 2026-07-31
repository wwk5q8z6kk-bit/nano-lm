"""Owner-corpus dogfood harness — private folder eval; results gitignored.

Active Frontier product measurement. Not Layer-1 evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wedge_v1.ingest import corpus_stats, load_corpus
from wedge_v1.runtime import DEFAULT_CORPUS, ask, compare, find_spans

ROOT = Path(__file__).resolve().parent
EXAMPLE_CORPUS = ROOT / "data" / "owner_corpus.example"
EXAMPLE_TASKS = ROOT / "data" / "owner_dogfood_tasks.example.json"
DEFAULT_TASKS = ROOT / "data" / "owner_dogfood_tasks.json"
if not DEFAULT_TASKS.is_file():
    DEFAULT_TASKS = EXAMPLE_TASKS
DEFAULT_OUT = ROOT / "results_owner_dogfood.json"
SMOKE_OUT = ROOT / "results_owner_smoke.json"
OWNER_DIR = ROOT / "data" / "owner_corpus"


def _blob(result: dict) -> str:
    return json.dumps(result, default=str).lower()


def resolve_corpus(*, corpus: Path | None = None, demo: bool = False) -> Path:
    if demo:
        return EXAMPLE_CORPUS if EXAMPLE_CORPUS.is_dir() else DEFAULT_CORPUS
    if corpus is not None:
        return Path(corpus)
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
    if EXAMPLE_CORPUS.is_dir() and any(EXAMPLE_CORPUS.glob("*.md")):
        return EXAMPLE_CORPUS
    return DEFAULT_CORPUS


def score_task(task: dict, corpus: Path) -> dict:
    q = task["query"]
    mode = task.get("mode") or "ask"
    if mode == "compare":
        result = compare(q, corpus_dir=corpus)
    elif mode == "find":
        result = find_spans(q, corpus_dir=corpus)
    else:
        result = ask(q, corpus_dir=corpus)

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
    tasks_path: Path,
    out_json: Path | None = None,
    *,
    write_smoke: bool = False,
) -> dict:
    docs = load_corpus(corpus)
    pack = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    rows = [score_task(t, corpus) for t in pack.get("tasks") or []]
    n_ok = sum(1 for r in rows if r["ok"])
    stats = corpus_stats(corpus)
    out = {
        "schema": "nano-lm.wedge_v1.owner_dogfood_result.v1",
        "corpus": str(Path(corpus).resolve()),
        "n_docs": len(docs),
        "ingest": stats,
        "n_tasks": len(rows),
        "n_ok": n_ok,
        "accuracy": n_ok / max(1, len(rows)),
        "rows": rows,
        "note": "Owner/local corpus product eval; gitignored; not Layer-1 ledger claim.",
    }
    dest = out_json or DEFAULT_OUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(out, indent=2) + "\n"
    dest.write_text(payload, encoding="utf-8")
    if write_smoke:
        SMOKE_OUT.write_text(payload, encoding="utf-8")
    gal_path = ROOT / "results_owner_failure_gallery.md"
    gal_path.write_text(failure_gallery_md(rows, Path(corpus)), encoding="utf-8")
    out["written"] = [str(dest), str(gal_path)]
    if write_smoke:
        out["written"].append(str(SMOKE_OUT))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Owner-corpus dogfood (gitignored results)")
    ap.add_argument("--corpus", type=Path, default=None)
    ap.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--gallery", type=Path, default=None)
    ap.add_argument(
        "--demo",
        action="store_true",
        help="Use tracked owner_corpus.example stand-in",
    )
    ap.add_argument("--smoke", action="store_true", help="Also write results_owner_smoke.json")
    args = ap.parse_args(argv)

    corpus = resolve_corpus(corpus=args.corpus, demo=args.demo)
    docs = load_corpus(corpus)
    if not docs:
        print(json.dumps({"error": "NO_CORPUS", "corpus": str(corpus)}, indent=2))
        return 2

    tasks_path = EXAMPLE_TASKS if args.demo and args.tasks == DEFAULT_TASKS else args.tasks
    out = run(
        corpus,
        tasks_path,
        out_json=args.out,
        write_smoke=bool(args.smoke or args.demo),
    )
    if args.gallery:
        Path(args.gallery).write_text(
            failure_gallery_md(out["rows"], Path(corpus)), encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "accuracy": out["accuracy"],
                "n_ok": out["n_ok"],
                "n_tasks": out["n_tasks"],
                "corpus": out["corpus"],
                "out": str(args.out),
                "rows": [
                    {"id": r["id"], "ok": r["ok"], "got": r["got_status"], "fail": r["fail_kind"]}
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
