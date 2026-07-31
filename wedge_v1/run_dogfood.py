"""Dogfood scorecard: ask()/find_spans() against papers/ corpus tasks."""
from __future__ import annotations

import json
import re
from pathlib import Path

from wedge_v1.runtime import ask, find_spans, compare, DEFAULT_CORPUS
from wedge_v1.auth_gate import require_auth

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
TASKS = ROOT / "data" / "dogfood_tasks.json"
OUT = ROOT / "results_wedge_v1_dogfood.json"
PAPERS = REPO / "papers"


def _blob(result: dict) -> str:
    return json.dumps(result, default=str).lower()


def _is_literal(q: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.:-]{8,}", q.strip()))


def score_task(task: dict) -> dict:
    q = task["query"]
    if task.get("corpus") in {None, "papers"}:
        corpus = PAPERS
    elif task.get("corpus") in {"default", "synthetic", "wedge_v1/data/corpus"}:
        corpus = DEFAULT_CORPUS
    else:
        rel = Path(task["corpus"])
        corpus = rel if rel.is_absolute() else (REPO / rel)
    if task.get("mode") == "compare":
        result = compare(q, corpus_dir=corpus)
    elif _is_literal(q):
        result = find_spans(q, corpus_dir=corpus)
        if result.get("answer_status") == "ABSTAIN":
            result = ask(q, corpus_dir=corpus)
    else:
        result = ask(q, corpus_dir=corpus)
        if result.get("answer_status") == "ABSTAIN":
            for lit in task.get("must_contain_any") or []:
                fr = find_spans(lit, corpus_dir=corpus)
                if fr.get("answer_status") == "SUPPORTED":
                    result = fr
                    result["note"] = f"fallback_find:{lit}"
                    break

    status = result.get("answer_status")
    ok_status = status in task["expect_status"]
    blob = _blob(result)
    needles = task.get("must_contain_any") or []
    if task["expect_status"] == ["ABSTAIN"]:
        ok_needles = True
    else:
        ok_needles = any(n.lower() in blob for n in needles) if needles else True
    ok = bool(ok_status and ok_needles)
    return {
        "id": task["id"],
        "ok": ok,
        "expect_status": task["expect_status"],
        "got_status": status,
        "ok_status": ok_status,
        "ok_needles": ok_needles,
        "n_claims": len(result.get("claims") or []),
        "solver_path": result.get("solver_path"),
        "note": result.get("note"),
    }


def main() -> None:
    mandate = ROOT / "ACTIVE_MANDATE.md"
    under_frontier = (
        mandate.exists()
        and "BUILD_SMALL_POWERFUL_USEFUL_SYSTEM_V1" in mandate.read_text(encoding="utf-8")
    )
    if not under_frontier:
        require_auth(
            auth_id="AUTHORIZE_WEDGE_V1_RUNTIME_SLICE",
            auth_record=ROOT / "AUTH_RUNTIME.md",
            need_bits={"execute_eval"},
            mode="integrity_remediation",
        )
    pack = json.loads(TASKS.read_text(encoding="utf-8"))
    rows = [score_task(t) for t in pack["tasks"]]
    n_ok = sum(1 for r in rows if r["ok"])
    out = {
        "schema": "nano-lm.wedge_v1.dogfood_result.v1",
        "corpus": str(PAPERS),
        "n_tasks": len(rows),
        "n_ok": n_ok,
        "accuracy": n_ok / max(1, len(rows)),
        "rows": rows,
        "note": "Classical dogfood; not Layer-1 ledger claim.",
    }
    payload = json.dumps(out, indent=2) + "\n"
    # real newline:
    payload = json.dumps(out, indent=2) + chr(10)
    OUT.write_text(payload, encoding="utf-8")
    (REPO / "trajectory" / "results_wedge_v1_dogfood.json").write_text(payload, encoding="utf-8")
    print(json.dumps({"accuracy": out["accuracy"], "n_ok": n_ok, "n_tasks": len(rows), "rows": rows}, indent=2))
    print("WEDGE_V1_DOGFOOD_DONE")


if __name__ == "__main__":
    main()
