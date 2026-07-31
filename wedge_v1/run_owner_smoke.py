"""Owner-corpus contact smoke — Active Frontier (not Layer-1 evidence)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from wedge_v1.ingest import corpus_stats
from wedge_v1.runtime import ask, compare, find_spans, load_corpus

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DEFAULT_TASKS = ROOT / "data" / "owner_smoke_tasks.json"
EXAMPLE = ROOT / "data" / "owner_corpus.example"
OWNER = ROOT / "data" / "owner_corpus"
OUT_NAME = "results_owner_smoke.json"


def resolve_corpus(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("WEDGE_OWNER_CORPUS", "").strip()
    if env:
        return Path(env)
    # Prefer private owner_corpus when present; else committed example fixture
    if OWNER.is_dir() and any(p.suffix in {".md", ".txt", ".pdf"} for p in OWNER.iterdir() if p.is_file()):
        return OWNER
    return EXAMPLE


def _blob(result: dict) -> str:
    return json.dumps(result, default=str).lower()


def score_task(task: dict, corpus: Path) -> dict:
    q = task["query"]
    mode = task.get("mode", "ask")
    if mode == "compare":
        result = compare(q, corpus_dir=corpus)
    elif mode == "find":
        result = find_spans(q, corpus_dir=corpus)
    else:
        result = ask(q, corpus_dir=corpus)
    status = result.get("answer_status")
    expect = task.get("expect_status") or []
    ok_status = status in expect if expect else True
    needles = task.get("must_contain_any") or []
    blob = _blob(result)
    if expect == ["ABSTAIN"]:
        ok_needles = True
    else:
        ok_needles = any(n.lower() in blob for n in needles) if needles else True
    return {
        "id": task["id"],
        "mode": mode,
        "query": q,
        "ok": bool(ok_status and ok_needles),
        "expect_status": expect,
        "got_status": status,
        "ok_status": ok_status,
        "ok_needles": ok_needles,
        "n_claims": len(result.get("claims") or []),
        "n_hits": result.get("n_hits"),
        "solver_path": result.get("solver_path"),
        "latency_s": result.get("latency_s"),
        "values_by_doc": result.get("values_by_doc"),
    }


def run(
    corpus_dir: Path | None = None,
    out_path: Path | None = None,
    tasks_path: Path | None = None,
) -> dict:
    corpus = resolve_corpus(corpus_dir)
    docs = load_corpus(corpus)
    stats = corpus_stats(corpus)
    pack = json.loads((tasks_path or DEFAULT_TASKS).read_text(encoding="utf-8"))
    rows = [score_task(t, corpus) for t in pack["tasks"]]
    n_ok = sum(1 for r in rows if r["ok"])
    by_status: dict[str, int] = {}
    for r in rows:
        by_status[r["got_status"]] = by_status.get(r["got_status"], 0) + 1
    abstain_rate = by_status.get("ABSTAIN", 0) / max(1, len(rows))
    out = {
        "schema": "nano-lm.wedge_v1.owner_smoke_result.v1",
        "NONCLAIM": "Product contact smoke. Not Layer-1 evidence. Not a public claim.",
        "corpus": str(corpus.resolve()),
        "n_docs": len(docs),
        "ingest": stats,
        "n_tasks": len(rows),
        "n_ok": n_ok,
        "accuracy": round(n_ok / max(1, len(rows)), 4),
        "abstain_rate": round(abstain_rate, 4),
        "status_counts": by_status,
        "rows": rows,
        "note": "Active Frontier owner contact. Not Layer-1. Do not commit private corpora.",
        "claim_level": "LAB.PRODUCT_SMOKE",
    }
    dest = out_path or (ROOT / OUT_NAME)
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, indent=2) + chr(10)
    dest.write_text(text, encoding="utf-8")
    # Also drop a copy beside the corpus when it is a writable local folder
    written = [str(dest)]
    # Avoid polluting tracked example fixture with result JSON
    try:
        if "owner_corpus.example" not in str(corpus.resolve()):
            side = corpus / OUT_NAME
            if corpus.resolve() != ROOT.resolve():
                side.write_text(text, encoding="utf-8")
                written.append(str(side))
    except OSError:
        pass
    out["written"] = written
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Owner-corpus ask+compare smoke")
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--tasks", type=Path, default=None)
    p.add_argument("-o", "--output", type=Path, default=None)
    args = p.parse_args(argv)
    out = run(args.corpus, args.output, args.tasks)
    print(
        json.dumps(
            {
                "n_ok": out["n_ok"],
                "n_tasks": out["n_tasks"],
                "accuracy": out["accuracy"],
                "abstain_rate": out["abstain_rate"],
                "status_counts": out["status_counts"],
                "corpus": out["corpus"],
                "written": out["written"],
                "task_ok": {r["id"]: r["ok"] for r in out["rows"]},
            },
            indent=2,
        )
    )
    print("WEDGE_V1_OWNER_SMOKE_DONE", flush=True)
    return 0 if out["n_ok"] == out["n_tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
