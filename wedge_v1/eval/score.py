"""Score classical claims against gold; compute U components."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from wedge_v1.classical.solvers import Claim
from wedge_v1.classical.verifier import present, verify_all
from wedge_v1.eval.utility import Weights, utility


ROOT = Path(__file__).resolve().parents[1]


def _load_gold() -> dict:
    return json.loads((ROOT / "data" / "gold" / "gold.json").read_text(encoding="utf-8"))


def _task_checks(claims: list[Claim], gold: dict) -> dict[str, dict[str, Any]]:
    by = defaultdict(list)
    for c in claims:
        by[c.task_id].append(c)

    results = {}

    def ok(tid: str, passed: bool, detail: dict | None = None):
        results[tid] = {"pass": passed, **(detail or {})}

    # T01 titles
    titles = {c.value["doc_id"]: c.value["title"] for c in by["T01"] if c.status != "REJECTED"}
    gtitles = {d: v["title"] for d, v in gold["docs"].items()}
    ok("T01", titles == gtitles, {"n": len(titles)})

    # T02 authors
    authors = {c.value["doc_id"]: c.value["authors"] for c in by["T02"] if c.status != "REJECTED"}
    gauth = {d: v["authors"] for d, v in gold["docs"].items()}
    ok("T02", authors == gauth, {"n": len(authors)})

    # T03 years
    years = {c.value["doc_id"]: c.value["year"] for c in by["T03"] if c.status != "REJECTED"}
    gyears = {d: v["year"] for d, v in gold["docs"].items()}
    ok("T03", years == gyears, {"n": len(years)})

    # T04 doc types — allow note/abstract/table_dump
    types = {c.value["doc_id"]: c.value["doc_type"] for c in by["T04"] if c.status != "REJECTED"}
    gtypes = {d: v["doc_type"] for d, v in gold["docs"].items()}
    ok("T04", types == gtypes, {"got": types, "gold": gtypes})

    # T06 DOIs where present
    dois = {c.value["doc_id"]: c.value["doi"] for c in by["T06"] if c.status != "REJECTED"}
    gdoi = {d: v["doi"] for d, v in gold["docs"].items() if v["doi"]}
    ok("T06", dois == gdoi, {"n": len(dois)})

    # T09 dosages contain planted
    got_doses = {(c.value["doc_id"], c.value["dosage"].lower()) for c in by["T09"] if c.status != "REJECTED"}
    need = {(d["doc_id"], d["text"].lower()) for d in gold["planted"]["dosages"]}
    ok("T09", need.issubset(got_doses), {"need": list(need), "got": list(got_doses)})

    # T10 compounds
    got_c = {(c.value["doc_id"], c.value["compound"]) for c in by["T10"] if c.status != "REJECTED"}
    ok("T10", ("bio_abs_metformin", "metformin") in got_c, {"n": len(got_c)})

    # T13 sample sizes
    ns = {c.value["doc_id"]: c.value["n"] for c in by["T13"] if c.status != "REJECTED"}
    need_n = gold["planted"]["sample_sizes"]
    ok("T13", all(ns.get(k) == v for k, v in need_n.items()), {"got": ns, "need": need_n})

    # T15 email
    emails = {c.value["doc_id"]: c.value["email"] for c in by["T15"] if c.status != "REJECTED"}
    ok("T15", emails.get("tech_note_cache") == gold["planted"]["emails"]["tech_note_cache"])

    # T17 kv
    kv_claims = [c for c in by["T17"] if c.value.get("doc_id") == "semi_structured_lab" and c.status != "REJECTED"]
    ok("T17", bool(kv_claims) and kv_claims[0].value["kv"].get("device") == "spectrometer-7")

    # T21 unicornium abstain + metformin mention
    t21 = by["T21"]
    uni = [c for c in t21 if c.value.get("entity") == "unicornium"]
    met = [c for c in t21 if c.value.get("entity") == "metformin"]
    ok("T21", bool(uni) and uni[0].status == "ABSTAIN" and bool(met) and met[0].value.get("mentioned") is True)

    # T25 metformin docs
    t25 = by["T25"][0] if by["T25"] else None
    need_docs = set(gold["planted"]["mentions"]["metformin"])
    ok("T25", t25 is not None and need_docs.issubset(set(t25.value.get("docs", []))),
       {"got": None if t25 is None else t25.value.get("docs")})

    # T28 TTL differ
    t28 = by["T28"][0] if by["T28"] else None
    ok("T28", t28 is not None and t28.value.get("relation") == "differ")

    # T29 contradiction
    t29 = by["T29"][0] if by["T29"] else None
    ok("T29", t29 is not None and t29.status == "DISPUTED" and t29.value.get("contradiction") is True)

    # T30 collision disputed
    t30 = by["T30"][0] if by["T30"] else None
    ok("T30", t30 is not None and t30.status == "DISPUTED")

    # T33 rejected
    t33 = by["T33"][0] if by["T33"] else None
    ok("T33", t33 is not None and t33.status == "REJECTED")

    # T34 abstains exist
    ok("T34", any(c.status == "ABSTAIN" for c in by["T34"]))

    # T35 paraphrastic abstain (classical)
    t35 = by["T35"][0] if by["T35"] else None
    ok("T35", t35 is not None and t35.status == "ABSTAIN")

    # T36 abstain
    t36 = by["T36"][0] if by["T36"] else None
    ok("T36", t36 is not None and t36.status == "ABSTAIN")

    # T38 table rows
    t38 = by["T38"][0] if by["T38"] else None
    need_rows = gold["planted"]["table_rows"]["tableish_throughput"]
    ok("T38", t38 is not None and len(t38.value.get("rows", [])) == len(need_rows))

    # T40 extractive
    t40 = by["T40"][0] if by["T40"] else None
    ok("T40", t40 is not None and "Latency is" in t40.value.get("explanation", ""))

    # Remaining tasks: presence smoke (ran without crash / produced claims)
    for tid in ["T05", "T07", "T08", "T11", "T12", "T14", "T16", "T18", "T19", "T20",
                "T22", "T23", "T24", "T26", "T27", "T31", "T32", "T37", "T39"]:
        ok(tid, len(by.get(tid, [])) > 0, {"n": len(by.get(tid, []))})

    return results


def score(claims: list[Claim]) -> dict:
    gold = _load_gold()
    verified = verify_all(claims)
    presented = present(verified)
    checks = _task_checks(verified, gold)

    # Primary: task-level exact checks on the discriminating subset + smoke on rest
    task_ids = sorted(checks)
    n = len(task_ids)
    n_pass = sum(1 for t in task_ids if checks[t]["pass"])
    n_fail = n - n_pass

    # Presented precision proxy: among presented claims, fraction not REJECTED path and checks
    # Use task pass rate as Q for v1 classical (each task is a unit of useful work)
    Q = n_pass / n if n else 0.0
    E = n_fail / n if n else 1.0

    # Review load: ABSTAIN + DISPUTED + REVIEW over all claims
    n_claims = len(verified) or 1
    n_review = sum(1 for c in verified if c.status in {"ABSTAIN", "DISPUTED", "REVIEW"})
    R = n_review / n_claims

    L = float(verified[0].meta.get("suite_latency_s", 0.0)) if verified else 0.0
    C = 1.0  # classical baseline reference

    w = Weights()
    U = utility(Q, E, R, L, C, w)

    liability = sum(1 for c in presented if c.meta.get("expect_reject"))

    return {
        "schema": "nano-lm.wedge_v1.classical.v1",
        "track": "clean",
        "weights": w.__dict__,
        "U": U,
        "components": {"Q": Q, "E": E, "R": R, "L": L, "C": C},
        "n_tasks_scored": n,
        "n_tasks_pass": n_pass,
        "n_tasks_fail": n_fail,
        "n_claims": len(verified),
        "n_presented": len(presented),
        "n_review_routed": n_review,
        "liability_presented_bad": liability,
        "probe_flags": gold["probe_flags"],
        "task_checks": checks,
        "failed_tasks": [t for t, v in checks.items() if not v["pass"]],
        "note": "Phase 2 classical-only; no LM. Q/E from task-check pass rate on frozen gold.",
    }
