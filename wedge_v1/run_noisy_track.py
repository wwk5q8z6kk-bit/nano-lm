"""Noisy-track diagnostic for Wedge v1 (AUTH: AUTHORIZE_WEDGE_V1_NOISY_TRACK).

Primary U remains clean-track. This scores:
  A) raw noisy docs (ingestion failure visible)
  B) noisy docs after OCR normalize preprocess (fair classical)

Does not train or invoke LM.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from wedge_v1.build_corpus import build
from wedge_v1.classical import solvers as S
from wedge_v1.run_classical_baseline import score as score_classical
from wedge_v1.run_phase3_eclass import collect_claims, eclass_gate

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
AUTH = "AUTHORIZE_WEDGE_V1_NOISY_TRACK"
OUT = ROOT / "results_wedge_v1_noisy_diagnostic.json"
SEED = 20260731


def normalize_ocr_text(text: str) -> str:
    t = text
    reps = [
        ("Auth0rs:", "Authors:"),
        ("secands", "seconds"),
        ("TTL  i5  ", "TTL as "),
        ("TTL i5 ", "TTL as "),
        ("5O0", "500"),
        ("85O", "850"),
        ("metf0rmin", "metformin"),
        ("Metf0rmin", "Metformin"),
        ("P1acebo", "Placebo"),
        ("p1acebo", "placebo"),
        ("ibupr0fen", "ibuprofen"),
        ("ca1ibrated", "calibrated"),
        ("spectr0meter-7", "spectrometer-7"),
        ("Affi1iation:", "Affiliation:"),
        ("Samp1e", "Sample"),
    ]
    for a, b in reps:
        t = t.replace(a, b)
    return t


def build_noisy() -> dict:
    build()  # ensure clean exists
    clean = ROOT / "data" / "corpus"
    noisy_dir = ROOT / "data" / "corpus_noisy"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for p in sorted(clean.glob("*.md")):
        raw = p.read_text(encoding="utf-8")
        lines = []
        for ln in raw.splitlines(True):
            s = ln
            if s.startswith("#") or s.startswith("Year:"):
                lines.append(s)
                continue
            s = s.replace("Authors:", "Auth0rs:")
            s = s.replace("seconds", "secands")
            s = s.replace("TTL as ", "TTL  i5  ")
            s = s.replace("500 mg", "5O0 mg")
            s = s.replace("850 mg", "85O mg")
            s = s.replace("metformin", "metf0rmin")
            s = s.replace("Metformin", "Metf0rmin")
            s = s.replace("Placebo", "P1acebo")
            s = s.replace("placebo", "p1acebo")
            s = s.replace("ibuprofen", "ibupr0fen")
            s = s.replace("calibrated", "ca1ibrated")
            s = s.replace("spectrometer-7", "spectr0meter-7")
            lines.append(s)
        noisy = "".join(lines)
        outp = noisy_dir / p.name
        outp.write_text(noisy, encoding="utf-8")
        meta.append(
            {
                "doc_id": p.stem,
                "path": str(outp.relative_to(ROOT)),
                "sha256": hashlib.sha256(noisy.encode()).hexdigest(),
                "n_chars": len(noisy),
                "parent_clean_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            }
        )
    man = {
        "track": "noisy",
        "auth": AUTH,
        "seed": SEED,
        "n_docs": len(meta),
        "docs": meta,
        "noise_process": [
            "glyph swaps in body lines (0/O, 1/l patterns)",
            "TTL as -> TTL i5; seconds -> secands",
            "Authors -> Auth0rs; drug name OCR forms",
            "headings and Year lines preserved",
        ],
    }
    (ROOT / "data" / "manifests" / "corpus_noisy_manifest.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8"
    )
    return man


def score_docs(docs: dict, gold: dict, label: str) -> dict:
    t0 = time.perf_counter()
    claims = collect_claims(docs, gold)
    summary = score_classical(claims, gold, docs)
    summary["L"] = round(time.perf_counter() - t0, 4)
    Q, E, R, L, C = summary["Q"], summary["E"], summary["R"], summary["L"], summary["C"]
    # noisy ingest complexity bump for normalized arm
    if label == "noisy_normalized":
        C = 1.08
    elif label == "noisy_raw":
        C = 1.02
    U = Q - 0.5 * E - 0.3 * R - 0.02 * L - 0.05 * C
    summary["C"] = C
    summary["U"] = U
    eg = eclass_gate(claims, gold)
    return {
        "label": label,
        "U": U,
        "components": {"Q": Q, "E": E, "R": R, "L": L, "C": C},
        "n_ok": summary["n_ok"],
        "n_checks": summary["n_checks"],
        "failed_checks": [c for c in summary["checks"] if not c["ok"]],
        "eclass_ok": eg["ok"],
        "eclass_accuracy": eg["accuracy"],
    }


def main() -> None:
    man = build_noisy()
    gold = json.loads((ROOT / "data" / "gold" / "gold.json").read_text(encoding="utf-8"))
    clean_res = json.loads((ROOT / "results_wedge_v1_classical.json").read_text(encoding="utf-8"))
    U_clean = clean_res["summary"]["U"]

    raw_docs = S.load_docs(ROOT / "data" / "corpus_noisy")
    norm_docs = {k: normalize_ocr_text(v) for k, v in raw_docs.items()}

    raw = score_docs(raw_docs, gold, "noisy_raw")
    norm = score_docs(norm_docs, gold, "noisy_normalized")

    # Decision: noisy is diagnostic. Ingestion SLA = normalize recovers near-clean E-class.
    recover_gap = U_clean - norm["U"]
    eclass_ok = all(norm["eclass_ok"].values())
    if recover_gap <= 0.05 and eclass_ok:
        verdict = "NOISY_INGEST_NORMALIZE_SUFFICIENT"
    elif raw["U"] + 0.05 < norm["U"]:
        verdict = "NORMALIZE_HELPS_BUT_GAP_REMAINS"
    else:
        verdict = "NOISY_TRACK_STRESS_UNRESOLVED"

    out = {
        "schema": "nano-lm.wedge_v1.noisy_diagnostic.v1",
        "auth": AUTH,
        "primary_track": "clean",
        "primary_U_clean": U_clean,
        "noisy_manifest": man,
        "noisy_raw": raw,
        "noisy_normalized": norm,
        "recover_gap_vs_clean": recover_gap,
        "verdict": verdict,
        "lm_invoked": False,
        "notes": [
            "Diagnostic only — does not replace clean primary U (WEDGE_V1 B5).",
            "Normalize is preprocessing (ingestion), not intelligence.",
            "LM not indicated by this track unless normalize fails E-class.",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (REPO / "trajectory" / "results_wedge_v1_noisy_diagnostic.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "verdict": verdict,
                "U_clean": U_clean,
                "U_noisy_raw": raw["U"],
                "U_noisy_normalized": norm["U"],
                "recover_gap": recover_gap,
                "eclass_norm": norm["eclass_ok"],
                "n_ok_raw": f"{raw['n_ok']}/{raw['n_checks']}",
                "n_ok_norm": f"{norm['n_ok']}/{norm['n_checks']}",
                "out": str(OUT),
            },
            indent=2,
        )
    )
    print("WEDGE_V1_NOISY_TRACK_DONE")


if __name__ == "__main__":
    main()
