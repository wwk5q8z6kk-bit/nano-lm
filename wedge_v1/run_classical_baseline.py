"""Run Wedge v1 classical baseline under AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from wedge_v1.build_corpus import build
from wedge_v1.classical import solvers as S
from wedge_v1.auth_gate import require_auth
from wedge_v1.eval.claim_report import claim_level_report

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_wedge_v1_classical.json"
AUTH = "AUTHORIZE_WEDGE_V1_CLASSICAL_BASELINE"


def claim_ok_evidence(c: S.Claim) -> bool:
    if c.status in {"ABSTAIN", "MISSING"}:
        return True
    if c.status == "DISPUTED":
        return True
    if c.task_id in {"T04", "T05", "T07", "T08", "T16", "T17", "T33", "T29", "T30", "T25", "T26", "T28", "T37", "T38", "T39"}:
        return True  # structural / multi-doc / derived
    if c.task_id == "T21" and c.value is False:
        return True  # grounded negative: term absent
    return bool(c.evidence)


def score(claims: list[S.Claim], gold: dict, docs: dict) -> dict:
    checks = []

    def add(task, ok, detail=""):
        checks.append({"task_id": task, "ok": bool(ok), "detail": detail})

    by = {}
    for c in claims:
        by.setdefault(c.task_id, []).append(c)

    # T01 titles vs gold
    for did, meta in gold["docs"].items():
        cs = [c for c in by.get("T01", []) if c.doc_id == did]
        add("T01", cs and cs[0].value == meta["title"], did)

    for did, meta in gold["docs"].items():
        cs = [c for c in by.get("T02", []) if c.doc_id == did]
        add("T02", cs and cs[0].value == meta["authors"], did)

    for did, meta in gold["docs"].items():
        cs = [c for c in by.get("T03", []) if c.doc_id == did]
        add("T03", cs and cs[0].value == meta["year"], did)

    for did, meta in gold["docs"].items():
        cs = [c for c in by.get("T04", []) if c.doc_id == did]
        add("T04", cs and cs[0].value == meta["doc_type"], did)

    # dosages planted
    for item in gold["planted"]["dosages"]:
        cs = [c for c in by.get("T09", []) if c.doc_id == item["doc_id"]]
        texts = []
        if cs and cs[0].value:
            texts = [x["text"] for x in cs[0].value]
        add("T09", item["text"] in texts, item["doc_id"])

    for did, n in gold["planted"]["sample_sizes"].items():
        cs = [c for c in by.get("T13", []) if c.doc_id == did]
        add("T13", cs and cs[0].value == n, did)

    email = gold["planted"]["emails"]["tech_note_cache"]
    cs = [c for c in by.get("T15", []) if c.doc_id == "tech_note_cache"]
    add("T15", cs and cs[0].value == email, "tech_note_cache")

    # kv
    for k, v in gold["planted"]["kv"]["semi_structured_lab"].items():
        cs = [c for c in by.get("T17", []) if c.doc_id == "semi_structured_lab"]
        add("T17", cs and str(cs[0].value.get(k)) == str(v), k)

    # table
    cs = [c for c in by.get("T38", []) if c.doc_id == "tableish_throughput"]
    add("T38", cs and len(cs[0].value) == 3, "rows")

    # contradictions / collisions / abstain
    cs = by.get("T29", [])
    add("T29", cs and cs[0].status == "DISPUTED", "numeric")
    cs = by.get("T30", [])
    add("T30", cs and cs[0].status == "DISPUTED", "ibuprofen reagent")
    cs = [c for c in by.get("T34", []) if c.doc_id == "semi_structured_lab"]
    add("T34", cs and cs[0].status == "ABSTAIN", "patient_id")

    # mentions metformin
    cs = by.get("T25", [])
    exp = set(gold["planted"]["mentions"]["metformin"])
    add("T25", cs and set(cs[0].value) >= exp, "metformin docs")

    # T35 paraphrastic — ok if PRESENT correct or ABSTAIN (classical allowed to abstain)
    cs = by.get("T35", [])
    ok35 = False
    if cs:
        if cs[0].status == "ABSTAIN":
            ok35 = True
        elif cs[0].value == gold["planted"]["B4_paraphrastic"]["answer_span"]:
            ok35 = True
    add("T35", ok35, "paraphrastic")

    # OCR
    cs = [c for c in by.get("T37", []) if c.doc_id == "noisy_ocr_line"]
    add("T37", cs and isinstance(cs[0].value, dict) and cs[0].value.get("ttl_seconds") == 250, "ocr")

    # evidence rule on PRESENT atomic claims
    evid_viol = [
        c.task_id
        for c in claims
        if c.status == "PRESENT" and not claim_ok_evidence(c)
    ]
    add("T33", len(evid_viol) == 0, f"violations={evid_viol[:5]}")

    n = len(checks)
    n_ok = sum(1 for c in checks if c["ok"])
    # Presented precision proxy: among PRESENT claims that were checked, fraction ok
    # Simple utility draft from WEDGE_V1
    Q = n_ok / n if n else 0.0
    E = 1.0 - Q
    R = sum(1 for c in claims if c.status in {"ABSTAIN", "DISPUTED"}) / max(1, len(claims))
    # L/C placeholders for classical local
    L = 0.05
    C = 1.0
    U = Q - 0.5 * E - 0.3 * R - 0.02 * L - 0.05 * C
    return {
        "n_checks": n,
        "n_ok": n_ok,
        "Q": Q,
        "E": E,
        "R": R,
        "L": L,
        "C": C,
        "U": U,
        "checks": checks,
        "evidence_violations": evid_viol,
    }


def main():
    gate = require_auth(
        auth_id=AUTH,
        auth_record=ROOT / "AUTH_RECORD.md",
        need_bits={"execute_eval"},
        mode="integrity_remediation",
    )
    t0 = time.perf_counter()
    manifest = build()
    gold = json.loads((ROOT / "data" / "gold" / "gold.json").read_text(encoding="utf-8"))
    docs = S.load_docs(ROOT / "data" / "corpus")

    claims: list[S.Claim] = []
    for did, text in docs.items():
        claims += [
            S.extract_title(did, text),
            S.extract_authors(did, text),
            S.extract_year(did, text),
            S.detect_doc_type(did, text),
            S.list_headings(did, text),
            S.extract_doi(did, text),
            S.word_count(did, text),
            S.build_toc(did, text),
            S.extract_dosages(did, text),
            S.extract_compounds(did, text),
            S.extract_sample_n(did, text),
            S.extract_definition(did, text),
            S.extract_email(did, text),
            S.extract_urls(did, text),
            S.extract_kv(did, text),
            S.extract_captions(did, text),
            S.parse_table(did, text),
            S.yes_no_mention(did, text, "metformin"),
            S.quote_sentence(did, text, "metformin"),
            S.keyword_paragraph(did, text, "cache TTL seconds"),
            S.ocr_normalize(did, text),
            S.coref_binding(did, text),
        ]
        if did == "semi_structured_lab":
            claims.append(S.missing_patient_id(did, text))

    claims.append(S.mention_docs(docs, "metformin"))
    claims.append(S.union_dosages(docs))
    claims.append(S.flag_numeric_contradiction(docs))
    claims.append(S.flag_entity_collision(docs))
    claims.append(S.reject_ungrounded())
    claims.append(S.paraphrastic_ttl(docs, gold))
    # T28 compare years tech notes
    ymap = {}
    for did in ("tech_note_cache", "tech_note_cache_v2"):
        c = S.extract_year(did, docs[did])
        if c.status == "PRESENT":
            ymap[did] = c.value
    claims.append(
        S.Claim(
            "T28",
            None,
            ymap,
            status="DISPUTED" if len(set(ymap.values())) > 1 else "PRESENT",
            notes="year compare",
        )
    )

    elapsed = time.perf_counter() - t0
    summary = score(claims, gold, docs)
    summary["L"] = round(elapsed, 4)

    # recompute U with measured L
    Q, E, R, L, C = summary["Q"], summary["E"], summary["R"], summary["L"], summary["C"]
    summary["U"] = Q - 0.5 * E - 0.3 * R - 0.02 * L - 0.05 * C

    out = {
        "schema": "nano-lm.wedge_v1.classical_baseline.v1",
        "auth": AUTH,
        "track": "clean",
        "seed": gold["seed"],
        "manifest": manifest,
        "n_docs": len(docs),
        "n_claims": len(claims),
        "summary": summary,
        "claims": [asdict(c) for c in claims],
        "probe_flags": gold["probe_flags"],
        "auth_gate": gate,
        "claim_level": claim_level_report(claims),
        "corpus_class": "SYNTHETIC_MINI",
        "U_status": "DRAFT_NOT_SCORING_FROZEN",
        "notes": [
            "Classical-only; no LM solvers.",
            "U uses draft WEDGE_V1 weights; L is wall-clock for this harness run.",
            "Not a Layer-1 Evidence Ledger claim until owner promotes.",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # also repo-root convenience copy under trajectory? keep in wedge_v1/
    print(json.dumps({"ok": True, "U": summary["U"], "Q": summary["Q"], "n_ok": summary["n_ok"], "n_checks": summary["n_checks"], "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
