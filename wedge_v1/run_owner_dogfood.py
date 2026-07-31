"""Owner-corpus dogfood harness — classical only; results gitignored.

Does not touch Evidence Core. No LM. No PHI commit path.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wedge_v1.failure_gallery import build_gallery, gallery_to_markdown
from wedge_v1.runtime import ask, compare, find_spans

ROOT = Path(__file__).resolve().parent
EXAMPLE_TASKS = ROOT / "data" / "owner_dogfood_tasks.example.json"
OUT_JSON = ROOT / "results_owner_dogfood.json"
OUT_MD = ROOT / "results_owner_dogfood.md"
OUT_GALLERY_JSON = ROOT / "results_owner_failure_gallery.json"
OUT_GALLERY_MD = ROOT / "results_owner_failure_gallery.md"
DEFAULT_OWNER_CORPUS = ROOT / "data" / "owner_corpus"


def _blob(result: dict) -> str:
    return json.dumps(result, default=str).lower()


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
    expect = task.get("expect_status") or [
        "SUPPORTED",
        "ABSTAIN",
        "CONTRADICTED",
        "NO_CORPUS",
    ]
    ok_status = status in expect
    needles = task.get("must_contain_any") or []
    blob = _blob(result)
    if "ABSTAIN" in expect and len(expect) == 1:
        ok_needles = True
    else:
        ok_needles = any(n.lower() in blob for n in needles) if needles else True
    ok = bool(ok_status and ok_needles)
    return {
        "id": task.get("id"),
        "query": q,
        "mode": mode,
        "ok": ok,
        "expect_status": expect,
        "got_status": status,
        "ok_status": ok_status,
        "ok_needles": ok_needles,
        "n_claims": len(result.get("claims") or result.get("hits") or []),
        "solver_path": result.get("solver_path"),
        "note": result.get("note") or task.get("note"),
        "latency_s": result.get("latency_s"),
    }


def run(
    corpus: Path,
    tasks_path: Path,
    out_json: Path = OUT_JSON,
    out_md: Path = OUT_MD,
) -> dict:
    pack = json.loads(tasks_path.read_text(encoding="utf-8"))
    rows = [score_task(t, corpus) for t in pack.get("tasks") or []]
    n_ok = sum(1 for r in rows if r["ok"])
    out = {
        "schema": "nano-lm.wedge_v1.owner_dogfood_result.v1",
        "corpus": str(corpus.resolve()) if corpus.exists() else str(corpus),
        "tasks_path": str(tasks_path),
        "n_tasks": len(rows),
        "n_ok": n_ok,
        "accuracy": n_ok / max(1, len(rows)),
        "rows": rows,
        "note": "Owner/local dogfood; gitignored results; not Layer-1 evidence.",
    }
    out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    g = build_gallery(dogfood=out, path=out_json)
    OUT_GALLERY_JSON.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
    OUT_GALLERY_MD.write_text(gallery_to_markdown(g), encoding="utf-8")
    lines = [
        "# Owner dogfood results",
        "",
        f"**Corpus:** `{out['corpus']}`",
        f"**Accuracy:** {out['accuracy']} ({n_ok}/{len(rows)})",
        "",
    ]
    for r in rows:
        mark = "OK" if r["ok"] else "FAIL"
        lines.append(
            f"- `{r['id']}` [{mark}] {r['mode']} expect={r['expect_status']} "
            f"got=`{r['got_status']}` — {r['query']}"
        )
    lines += ["", "_Private results — do not commit._", ""]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Owner-corpus classical dogfood (gitignored outputs)"
    )
    p.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Private folder (default: OWNER_CORPUS env or wedge_v1/data/owner_corpus)",
    )
    p.add_argument("--tasks", type=Path, default=EXAMPLE_TASKS)
    p.add_argument("--out", type=Path, default=OUT_JSON)
    args = p.parse_args(argv)

    corpus = args.corpus
    if corpus is None:
        env = os.environ.get("OWNER_CORPUS")
        corpus = Path(env) if env else DEFAULT_OWNER_CORPUS

    if not corpus.is_dir():
        print(
            json.dumps(
                {
                    "error": "NO_CORPUS",
                    "corpus": str(corpus),
                    "hint": (
                        "Create wedge_v1/data/owner_corpus/ or pass --corpus / "
                        "set OWNER_CORPUS"
                    ),
                },
                indent=2,
            )
        )
        return 2

    out = run(corpus, args.tasks, out_json=args.out)
    print(
        json.dumps(
            {
                "accuracy": out["accuracy"],
                "n_ok": out["n_ok"],
                "n_tasks": out["n_tasks"],
                "out": str(args.out),
            },
            indent=2,
        )
    )
    print("WEDGE_V1_OWNER_DOGFOOD_DONE")
    return 0 if out["n_tasks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
