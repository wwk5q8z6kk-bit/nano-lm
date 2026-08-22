"""Marginal value measurement — classical vs +LM on irreducible abstain only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wedge_v1.classical.eclass_probes import lm_still_needed, probe_t35, probe_t36, probe_t39
from wedge_v1.eval.utility import Weights, utility
from wedge_v1.lm.admission import evaluate_admission
from wedge_v1.lm.probe import ALLOWLIST_TASKS, ablation_fails_support, get_backend, probe_eclass_task
from wedge_v1.runtime import DEFAULT_CORPUS, load_corpus

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_w6_marginal_probe.json"
CLASSICAL_RESULT = ROOT / "results_wedge_v1_classical.json"
DELTA = Weights().delta


def _eclass_classical(docs: dict[str, str]) -> dict[str, str]:
    return {
        "T35": probe_t35(docs).status,
        "T36": probe_t36(docs).status,
        "T39": probe_t39(docs).status,
    }


def _load_classical_u() -> float | None:
    if not CLASSICAL_RESULT.is_file():
        return None
    data = json.loads(CLASSICAL_RESULT.read_text(encoding="utf-8"))
    u = data.get("utility") or {}
    return u.get("U_dep") or u.get("U")


def run_marginal_probe(
    *,
    gallery: dict[str, Any] | None = None,
    corpus_dir: Path | None = None,
    dry_run: bool = True,
    backend_name: str = "stub",
    min_irreducible: int = 2,
    owner_corpus_contact: bool = False,
    persist: bool = True,
) -> dict[str, Any]:
    docs = load_corpus(corpus_dir or DEFAULT_CORPUS)
    eclass = _eclass_classical(docs)
    still_needed = lm_still_needed([probe_t35(docs), probe_t36(docs), probe_t39(docs)])

    admission = evaluate_admission(
        gallery or {},
        eclass_lm_still_needed=still_needed,
        min_irreducible=min_irreducible,
        owner_corpus_contact=owner_corpus_contact,
    )

    backend = get_backend(backend_name)
    rows = []
    lm_present_gain = 0
    for tid in sorted(ALLOWLIST_TASKS):
        classical = eclass.get(tid, "ABSTAIN")
        row = probe_eclass_task(tid, docs, backend=backend, classical_status=classical)
        rows.append(row)
        if classical in {"ABSTAIN", "MISSING", "REJECTED"} and row.get("claim_status") == "PRESENT":
            lm_present_gain += 1

    classical_u = _load_classical_u()
    # Diagnostic only on clean synthetic: LM cannot beat closed E-class
    u_lm_diagnostic = None
    delta_u = None
    if classical_u is not None and still_needed:
        # Rough proxy: each recovered E task worth ~0.02 U on this pack
        u_lm_diagnostic = classical_u + 0.02 * lm_present_gain
        delta_u = u_lm_diagnostic - classical_u

    if still_needed and delta_u is not None and delta_u >= DELTA:
        product_verdict = "SURVIVE"
    elif still_needed:
        product_verdict = "KILL"
    else:
        product_verdict = "NOT_APPLICABLE"

    out = {
        "schema": "nano-lm.wedge_v1.w6_marginal_probe.v1",
        "workstream": "W6",
        "dry_run": dry_run,
        "lm_invoked": any(r.get("lm_invoked") for r in rows),
        "backend": backend.name,
        "admission": admission,
        "eclass_classical": eclass,
        "eclass_lm_still_needed": still_needed,
        "probe_rows": rows,
        "lm_present_gain": lm_present_gain,
        "classical_u_dep": classical_u,
        "u_lm_diagnostic": u_lm_diagnostic,
        "delta_u_diagnostic": delta_u,
        "delta_threshold": DELTA,
        "product_verdict": product_verdict,
        "execute_auth": admission["execute_auth"],
        "corpus_dir": str(corpus_dir or DEFAULT_CORPUS),
        "note": (
            "Use backend_name=mlx for local MLXLlamaBackend (span-bound). "
            "NOT_APPLICABLE when E-class closed without LM on this corpus."
        ),
    }
    if persist:
        OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out
