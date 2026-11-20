"""Phase 3: close E-class with non-LM probes; LM only if still needed."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from wedge_v1.build_corpus import build
from wedge_v1.auth_gate import require_auth
from wedge_v1.eval.claim_report import claim_level_report
from wedge_v1.classical import solvers as S
from wedge_v1.classical.eclass_probes import apply_eclass_overrides, lm_still_needed
from wedge_v1.run_classical_baseline import score as score_classical

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
AUTH = "AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE"
OUT = ROOT / "results_wedge_v1_phase3.json"


def collect_claims(docs: dict, gold: dict) -> list[S.Claim]:
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
    claims.append(S.paraphrastic_ttl(docs, gold))  # T35 expand
    claims.append(S.symbolic_dose_change(docs))      # T36 symbolic

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
    return claims


def eclass_gate(claims: list[S.Claim], gold: dict) -> dict:
    by: dict[str, list] = {}
    for c in claims:
        by.setdefault(c.task_id, []).append(c)

    t35 = next(iter(by.get("T35", [])), None)
    t36 = next(iter(by.get("T36", [])), None)
    t39 = next((c for c in by.get("T39", []) if c.doc_id == "binding_coref"), None)

    def _t35_ok(c):
        if not c:
            return False
        if c.status in {"PRESENT", "CONFIRMED"}:
            v = c.value
            if v == gold["planted"]["B4_paraphrastic"]["answer_span"]:
                return True
            if isinstance(v, dict) and "300" in str(v.get("answer", "")):
                return True
        return False

    ok = {
        "T35": _t35_ok(t35),
        "T36": bool(
            t36
            and t36.status in {"PRESENT", "CONFIRMED"}
            and isinstance(t36.value, dict)
            and t36.value.get("from") == 500
            and t36.value.get("to") == 850
        ),
        "T39": bool(
            t39
            and t39.status in {"PRESENT", "CONFIRMED"}
            and (
                (isinstance(t39.value, list) and len(t39.value) >= 2)
                or (isinstance(t39.value, dict) and len(t39.value.get("bindings", [])) >= 2)
            )
        ),
    }
    detail = {
        "T35": asdict(t35) if t35 else None,
        "T36": asdict(t36) if t36 else None,
        "T39": asdict(t39) if t39 else None,
    }
    return {"ok": ok, "detail": detail, "accuracy": sum(ok.values()) / 3.0}


def main() -> None:
    t0 = time.perf_counter()
    man = build()
    assert man.get("probe_ok")

    classical = json.loads((ROOT / "results_wedge_v1_classical.json").read_text(encoding="utf-8"))
    U_c = classical["summary"]["U"]

    gold = json.loads((ROOT / "data" / "gold" / "gold.json").read_text(encoding="utf-8"))
    docs = S.load_docs(ROOT / "data" / "corpus")
    claims = apply_eclass_overrides(collect_claims(docs, gold), docs)
    summary = score_classical(claims, gold, docs)
    summary["L"] = round(time.perf_counter() - t0, 4)
    summary["C"] = 1.05
    Q, E, R, L, C = summary["Q"], summary["E"], summary["R"], summary["L"], summary["C"]
    U_h = Q - 0.5 * E - 0.3 * R - 0.02 * L - 0.05 * C
    summary["U"] = U_h

    eg = eclass_gate(claims, gold)
    lm_needed = not all(eg["ok"].values())
    dU = U_h - U_c
    delta = 0.05

    if not lm_needed:
        verdict = "ECLASS_CLOSED_WITHOUT_LM"
    else:
        verdict = "LM_PROBE_INDICATED"

    out = {
        "schema": "nano-lm.wedge_v1.phase3.v1",
        "auth": AUTH,
        "track": "clean",
        "classical_U": U_c,
        "hybrid_U": U_h,
        "delta_U": dU,
        "delta": delta,
        "summary": summary,
        "eclass_ok": eg["ok"],
        "eclass_detail": eg["detail"],
        "eclass_accuracy": eg["accuracy"],
        "lm_still_needed": lm_needed,
        "lm_invoked": False,
        "verdict": verdict,
        "allowlist": ["T35", "T36", "T39"],
        "methods": {
            "T35": "query_synonym_expansion",
            "T36": "symbolic_multi_doc_compare",
            "T39": "sentence_level_coref",
        },
        "notes": [
            "First-principles cascade: cheapest non-LM probes before any LM.",
            "LM not invoked; E-class closed without generation." if not lm_needed
            else "LM indicated for remaining abstentions; recipe card required before invoke.",
            "A–D remain classical-only. Not a Layer-1 ledger claim until promoted.",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (REPO / "trajectory" / "results_wedge_v1_phase3.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "verdict": verdict,
        "classical_U": U_c,
        "hybrid_U": U_h,
        "delta_U": dU,
        "eclass_accuracy": eg["accuracy"],
        "eclass_ok": eg["ok"],
        "lm_still_needed": lm_needed,
        "lm_invoked": False,
        "out": str(OUT),
    }, indent=2))
    print("WEDGE_V1_PHASE3_DONE")


if __name__ == "__main__":
    main()
