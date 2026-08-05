"""Draft U from ask/find dogfood packs (engineering measurement; not Layer-1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wedge_v1.eval.utility import Weights, utility


def measure_dogfood_u(
    dogfood: dict[str, Any],
    *,
    corpus_class: str = "UNKNOWN",
    w: Weights | None = None,
) -> dict[str, Any]:
    """Score draft U from a dogfood/owner-dogfood result object.

    Q = task accuracy
    E = 1 - Q
    R = fraction of rows with ABSTAIN (review/abstain load proxy)
    L = mean latency_s when present, else 0.05 placeholder
    C = 1.0 (classical local reference)
    """
    w = w or Weights()
    rows = list(dogfood.get("rows") or [])
    n = int(dogfood.get("n_tasks") or len(rows))
    n_ok = int(
        dogfood.get("n_ok")
        if dogfood.get("n_ok") is not None
        else sum(1 for r in rows if r.get("ok"))
    )
    Q = (n_ok / n) if n else 0.0
    E = 1.0 - Q
    n_abstain = sum(1 for r in rows if str(r.get("got_status") or "") == "ABSTAIN")
    R = (n_abstain / n) if n else 0.0
    latencies = [float(r["latency_s"]) for r in rows if r.get("latency_s") is not None]
    L = (sum(latencies) / len(latencies)) if latencies else 0.05
    C = 1.0
    U = utility(Q, E, R, L, C, w)
    return {
        "schema": "nano-lm.wedge_v1.dogfood_utility.v1",
        "corpus_class": corpus_class,
        "source_schema": dogfood.get("schema"),
        "n_tasks": n,
        "n_ok": n_ok,
        "Q": Q,
        "E": E,
        "R": R,
        "L": L,
        "C": C,
        "U": U,
        "delta": w.delta,
        "weights": {
            "beta_E": w.beta_E,
            "gamma_R": w.gamma_R,
            "lam_L": w.lam_L,
            "kappa_C": w.kappa_C,
        },
        "U_status": "DRAFT_NOT_SCORING_FROZEN",
        "note": "Wedge draft U from dogfood pack; not Evidence Ledger.",
    }


def measure_path(path: Path, *, corpus_class: str = "UNKNOWN") -> dict[str, Any]:
    dogfood = json.loads(Path(path).read_text(encoding="utf-8"))
    out = measure_dogfood_u(dogfood, corpus_class=corpus_class)
    out["source"] = str(Path(path).resolve())
    return out
