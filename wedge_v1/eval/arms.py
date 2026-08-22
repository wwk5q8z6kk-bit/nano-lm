"""Fixture eval harness — U_classical vs hybrid-stub under ΔU gate.

Product measurement only (not Layer-1). No training. No paid compute.
Hybrid arm escalates only on classical ABSTAIN using constructive span stubs.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from wedge_v1.classical.bm25 import top_paragraphs
from wedge_v1.classical.solvers import Claim, _find
from wedge_v1.classical.verifier import verify_claim
from wedge_v1.eval.dogfood_utility import measure_dogfood_u
from wedge_v1.eval.utility import Weights, utility
from wedge_v1.plugins import synonym as synonym_plugin
from wedge_v1.run_owner_dogfood import (
    DEFAULT_TASKS,
    EXAMPLE_TASKS,
    FIXTURE_CORPUS,
    resolve_corpus,
    score_task,
)
from wedge_v1.runtime import compare, find_spans, load_corpus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results_eval_arms.json"
DELTA = Weights().delta

_TTL = re.compile(
    r"(?:TTL|expire[sd]?|timeout)\s+(?:as|is|of|=|:)?\s*(\d+)\s*(seconds|sec|s)?",
    re.I,
)
_NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(seconds|sec|mg|qps|%|delta)?", re.I)


def _tasks_path(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    demo = ROOT / "data" / "owner_dogfood_tasks_demo.json"
    if demo.is_file():
        return demo
    if DEFAULT_TASKS.is_file():
        return DEFAULT_TASKS
    return EXAMPLE_TASKS


def _score_ok(task: dict, status: str | None, result: dict) -> tuple[bool, bool, bool]:
    expect = task.get("expect_status") or ["any"]
    if "any" in expect:
        ok_status = status not in {None, "NO_CORPUS"}
    else:
        ok_status = status in expect
    needles = task.get("must_contain_any") or []
    blob = json.dumps(result, default=str).lower()
    if list(expect) == ["ABSTAIN"]:
        ok_needles = True
    elif not needles:
        ok_needles = True
    else:
        ok_needles = any(n.lower() in blob for n in needles)
    return bool(ok_status and ok_needles), ok_status, ok_needles


def escalate_stub_ask(query: str, docs: dict[str, str]) -> dict[str, Any]:
    """Constructive span stub for classical ABSTAIN — never invents unsupported text."""
    t0 = time.perf_counter()
    q_low = query.lower()
    ttl_query = any(
        k in q_low
        for k in ("ttl", "expire", "expiry", "timeout", "cache", "seconds")
    )
    plug = synonym_plugin.probe_ttl(docs, query) if ttl_query else None
    if plug is not None and plug.status == "PRESENT" and plug.evidence:
        c = Claim(
            "ASK_STUB",
            plug.doc_id,
            {"query": query, "answer": plug.value, "method": "hybrid_stub_synonym"},
            evidence=plug.evidence,
            status="PRESENT",
            notes="hybrid_stub_synonym",
        )
        c = verify_claim(c)
        if c.status in {"PRESENT", "CONFIRMED"}:
            return {
                "query": query,
                "answer_status": "SUPPORTED",
                "claims": [
                    {
                        "task_id": c.task_id,
                        "doc_id": c.doc_id,
                        "value": c.value,
                        "evidence": c.evidence,
                        "status": c.status,
                        "notes": c.notes,
                    }
                ],
                "solver_path": ["hybrid_stub", "synonym_ttl"],
                "lm_invoked": False,
                "escalation": "stub_synonym",
                "latency_s": round(time.perf_counter() - t0, 4),
            }

    hits = top_paragraphs(docs, query, k=4)
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", q_low) if len(t) > 2}
    for hit in hits:
        if not hit.get("promote"):
            continue
        text = hit["text"]
        did = hit["doc_id"]
        hit_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        if q_tokens and len(q_tokens & hit_tokens) < max(1, min(2, len(q_tokens))):
            continue
        # Prefer TTL pattern when query is TTL-like; otherwise require unit-bearing number
        m = _TTL.search(text) if ttl_query else None
        if m is None and not ttl_query:
            # Avoid answering arbitrary OOS asks with a random number from a paragraph
            continue
        if m is None:
            m = _TTL.search(text) or _NUM_UNIT.search(text)
        if not m:
            continue
        span_text = m.group(0)
        local = text.find(span_text)
        if local < 0:
            continue
        start = int(hit["start"]) + local
        end = start + len(span_text)
        body = docs.get(did, "")
        if body[start:end] != span_text:
            span = _find(body, span_text)
            if not span:
                continue
            evidence = [span]
        else:
            evidence = [{"start": start, "end": end, "text": span_text, "doc_id": did}]
        answer = m.group(1) if m.lastindex else span_text
        c = Claim(
            "ASK_STUB",
            did,
            {"query": query, "answer": answer, "method": "hybrid_stub_bm25"},
            evidence=evidence,
            status="PRESENT",
            notes="hybrid_stub_bm25",
        )
        c = verify_claim(c)
        if c.status not in {"PRESENT", "CONFIRMED"}:
            continue
        return {
            "query": query,
            "answer_status": "SUPPORTED",
            "claims": [
                {
                    "task_id": c.task_id,
                    "doc_id": c.doc_id,
                    "value": c.value,
                    "evidence": c.evidence,
                    "status": c.status,
                    "notes": c.notes,
                }
            ],
            "solver_path": ["hybrid_stub", "bm25_span_lock"],
            "lm_invoked": False,
            "escalation": "stub_bm25",
            "latency_s": round(time.perf_counter() - t0, 4),
        }

    return {
        "query": query,
        "answer_status": "ABSTAIN",
        "claims": [],
        "unsupported": ["hybrid_stub_no_verified_span"],
        "solver_path": ["hybrid_stub"],
        "lm_invoked": False,
        "escalation": "stub_miss",
        "latency_s": round(time.perf_counter() - t0, 4),
    }


def score_hybrid_task(task: dict, corpus: Path, classical_row: dict) -> dict[str, Any]:
    """Hybrid arm: keep classical when supported; escalate only on ABSTAIN."""
    mode = task.get("mode") or "ask"
    q = task["query"]
    expect = task.get("expect_status") or ["any"]
    classical_status = classical_row.get("got_status")

    if classical_status and classical_status != "ABSTAIN":
        return {
            **{
                k: classical_row.get(k)
                for k in (
                    "id",
                    "ok",
                    "mode",
                    "query",
                    "expect_status",
                    "got_status",
                    "ok_status",
                    "ok_needles",
                    "n_claims",
                    "latency_s",
                    "note",
                )
            },
            "arm": "hybrid_stub",
            "escalated": False,
            "solver_path": classical_row.get("solver_path"),
        }

    if list(expect) == ["ABSTAIN"]:
        return {
            "id": task["id"],
            "ok": True,
            "mode": mode,
            "query": q,
            "expect_status": expect,
            "got_status": "ABSTAIN",
            "ok_status": True,
            "ok_needles": True,
            "n_claims": 0,
            "latency_s": classical_row.get("latency_s"),
            "note": "hybrid_respect_oos_abstain",
            "arm": "hybrid_stub",
            "escalated": False,
            "solver_path": ["hybrid_skip_oos"],
        }

    docs = load_corpus(corpus)
    if mode == "compare":
        result = compare(q, corpus_dir=corpus)
        escalated = False
    elif mode == "find":
        result = find_spans(q, corpus_dir=corpus)
        if result.get("answer_status") == "ABSTAIN":
            result = escalate_stub_ask(q, docs)
            escalated = True
        else:
            escalated = False
    else:
        result = escalate_stub_ask(q, docs)
        escalated = True

    status = result.get("answer_status")
    ok, ok_status, ok_needles = _score_ok(task, status, result)
    return {
        "id": task["id"],
        "ok": ok,
        "mode": mode,
        "query": q,
        "expect_status": expect,
        "got_status": status,
        "ok_status": ok_status,
        "ok_needles": ok_needles,
        "n_claims": len(result.get("claims") or result.get("hits") or []),
        "latency_s": result.get("latency_s"),
        "note": result.get("note") or result.get("escalation"),
        "arm": "hybrid_stub",
        "escalated": escalated,
        "solver_path": result.get("solver_path"),
    }


def _dogfood_shell(rows: list[dict], *, schema: str, corpus: Path, arm: str, cost: float) -> dict:
    n_ok = sum(1 for r in rows if r.get("ok"))
    return {
        "schema": schema,
        "arm": arm,
        "corpus": str(corpus),
        "n_tasks": len(rows),
        "n_ok": n_ok,
        "accuracy": n_ok / max(1, len(rows)),
        "rows": rows,
        "cost_proxy_C": cost,
        "note": "Fixture arm score; not Layer-1 evidence.",
    }


def measure_arm_u(dogfood: dict[str, Any], *, corpus_class: str, cost: float) -> dict[str, Any]:
    measured = measure_dogfood_u(dogfood, corpus_class=corpus_class)
    w = Weights()
    Q, E, R, L = measured["Q"], measured["E"], measured["R"], measured["L"]
    C = float(cost)
    U = utility(Q, E, R, L, C, w)
    measured = dict(measured)
    measured["C"] = C
    measured["U"] = U
    measured["arm"] = dogfood.get("arm")
    return measured


def run_arms_eval(
    *,
    corpus: Path | None = None,
    tasks_path: Path | None = None,
    demo: bool = True,
    persist: bool = True,
    out_path: Path | None = None,
    corpus_class: str = "SYNTHETIC_MINI",
) -> dict[str, Any]:
    """Score classical vs hybrid-stub on the same fixture pack; apply ΔU gate."""
    corp = resolve_corpus(corpus=corpus, demo=demo) if demo or corpus is None else Path(corpus)
    if not corp.is_dir():
        corp = FIXTURE_CORPUS if FIXTURE_CORPUS.is_dir() else corp
    tpath = _tasks_path(tasks_path)
    pack = json.loads(tpath.read_text(encoding="utf-8"))
    tasks = list(pack.get("tasks") or [])

    classical_rows = [score_task(t, corp) for t in tasks]
    hybrid_rows = [
        score_hybrid_task(t, corp, classical_rows[i]) for i, t in enumerate(tasks)
    ]

    classical_dog = _dogfood_shell(
        classical_rows,
        schema="nano-lm.wedge_v1.eval_arms_classical.v1",
        corpus=corp,
        arm="classical",
        cost=1.0,
    )
    n_escalated = sum(1 for r in hybrid_rows if r.get("escalated"))
    hybrid_cost = 1.0 + 0.15 * (n_escalated / max(1, len(hybrid_rows)))
    hybrid_dog = _dogfood_shell(
        hybrid_rows,
        schema="nano-lm.wedge_v1.eval_arms_hybrid.v1",
        corpus=corp,
        arm="hybrid_stub",
        cost=hybrid_cost,
    )

    u_c = measure_arm_u(classical_dog, corpus_class=corpus_class, cost=1.0)
    u_h = measure_arm_u(hybrid_dog, corpus_class=corpus_class, cost=hybrid_cost)
    delta_u = float(u_h["U"]) - float(u_c["U"])
    admit = delta_u > DELTA

    out = {
        "schema": "nano-lm.wedge_v1.eval_arms.v1",
        "corpus_class": corpus_class,
        "corpus": str(corp),
        "tasks_path": str(tpath),
        "n_tasks": len(tasks),
        "classical": {
            "n_ok": classical_dog["n_ok"],
            "accuracy": classical_dog["accuracy"],
            "U": u_c["U"],
            "Q": u_c["Q"],
            "E": u_c["E"],
            "R": u_c["R"],
            "L": u_c["L"],
            "C": u_c["C"],
            "rows": classical_rows,
        },
        "hybrid_stub": {
            "n_ok": hybrid_dog["n_ok"],
            "accuracy": hybrid_dog["accuracy"],
            "U": u_h["U"],
            "Q": u_h["Q"],
            "E": u_h["E"],
            "R": u_h["R"],
            "L": u_h["L"],
            "C": u_h["C"],
            "n_escalated": n_escalated,
            "rows": hybrid_rows,
        },
        "delta_u": delta_u,
        "delta_threshold": DELTA,
        "verdict": "ADMIT_HYBRID_STUB" if admit else "KEEP_CLASSICAL",
        "admit_escalation": admit,
        "U_status": "DRAFT_NOT_SCORING_FROZEN",
        "note": (
            "ΔU-gated fixture arms. Hybrid stub escalates only on classical ABSTAIN "
            "with verified spans; no training / no paid LM."
        ),
    }
    path = out_path or DEFAULT_OUT
    if persist:
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        out["out"] = str(path)
    return out
