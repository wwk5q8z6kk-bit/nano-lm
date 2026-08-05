"""Ingest SLA / recover_gap measurement (W5).

Normalize is preprocessing, not intelligence. Measures whether OCR/layout
normalize recovers field extractability vs clean and vs raw noisy.

Not Layer-1 evidence. No LM.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from wedge_v1.plugins.ocr import normalize_text
from wedge_v1.ingest import load_corpus

ROOT = Path(__file__).resolve().parent

# SLA thresholds (Wedge engineering — amendable)
MAX_RECOVER_GAP_U = 0.05  # when full U available
MIN_FIELD_RECOVERY = 0.90  # fraction of clean fields recovered after normalize
MAX_CHAR_CORRUPTION = 0.35  # raw vs clean char disagreement before normalize expected


FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "ttl_seconds": re.compile(r"TTL(?:\s+as|\s+is|\s*[=:]\s*|\s+of)\s+(\d+)\s+seconds", re.I),
    "metformin_dose_mg": re.compile(r"metformin\s+(\d+)\s*mg", re.I),
    "authors": re.compile(r"^Authors:\s*(.+)$", re.M | re.I),
    "year": re.compile(r"^Year:\s*(\d{4})\s*$", re.M),
}


def _fix_field_patterns() -> None:
    # Ensure patterns compiled correctly even if escapes got doubled in writers
    global FIELD_PATTERNS
    FIELD_PATTERNS = {
        "ttl_seconds": re.compile(r"TTL(?:\s+as|\s+is|\s*[=:]\s*|\s+of)\s+(\d+)\s+seconds", re.I),
        "metformin_dose_mg": re.compile(r"metformin\s+(\d+)\s*mg", re.I),
        "authors": re.compile(r"^Authors:\s*(.+)$", re.M | re.I),
        "year": re.compile(r"^Year:\s*(\d{4})\s*$", re.M),
    }


_fix_field_patterns()


def extract_fields(docs: dict[str, str]) -> dict[str, dict[str, str]]:
    """doc_id -> {field_id: value}."""
    out: dict[str, dict[str, str]] = {}
    for did, text in docs.items():
        found = {}
        for fid, pat in FIELD_PATTERNS.items():
            m = pat.search(text)
            if m:
                found[fid] = m.group(1).strip()
        if found:
            out[did] = found
    return out


def normalize_corpus(docs: dict[str, str]) -> tuple[dict[str, str], int]:
    n_edits = 0
    out = {}
    for did, text in docs.items():
        fixed, edits = normalize_text(text)
        n_edits += len(edits)
        out[did] = fixed
    return out, n_edits


def field_recovery(
    clean_fields: dict[str, dict[str, str]],
    other_fields: dict[str, dict[str, str]],
) -> dict[str, Any]:
    total = 0
    hit = 0
    missing = []
    for did, fields in clean_fields.items():
        for fid, val in fields.items():
            total += 1
            got = (other_fields.get(did) or {}).get(fid)
            if got == val:
                hit += 1
            else:
                missing.append({"doc_id": did, "field": fid, "clean": val, "got": got})
    rate = (hit / total) if total else 1.0
    return {"n_fields": total, "n_recovered": hit, "recovery_rate": round(rate, 4), "misses": missing[:40]}


def char_agreement(a: dict[str, str], b: dict[str, str]) -> float:
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0
    scores = []
    for k in keys:
        sa, sb = a[k], b[k]
        if not sa and not sb:
            scores.append(1.0)
            continue
        # simple ratio of matching prefix/char equality
        n = max(len(sa), len(sb), 1)
        same = sum(1 for i in range(min(len(sa), len(sb))) if sa[i] == sb[i])
        scores.append(same / n)
    return round(sum(scores) / len(scores), 4)


def measure_ingest_sla(
    *,
    clean_dir: Path | None = None,
    noisy_dir: Path | None = None,
    u_clean: float | None = None,
    u_noisy_raw: float | None = None,
    u_noisy_norm: float | None = None,
) -> dict:
    t0 = time.perf_counter()
    clean_dir = clean_dir or (ROOT / "data" / "corpus")
    noisy_dir = noisy_dir or (ROOT / "data" / "corpus_noisy")

    clean = load_corpus(clean_dir)
    if not clean:
        return {"ok": False, "error": "NO_CLEAN_CORPUS", "clean_dir": str(clean_dir)}

    noisy = load_corpus(noisy_dir) if Path(noisy_dir).is_dir() else {}
    if not noisy:
        # synthesize minimal noisy from clean for offline SLA pin
        noisy = {}
        for did, text in clean.items():
            s = text
            s = s.replace("seconds", "secands").replace("TTL as ", "TTL  i5  ")
            s = s.replace("500 mg", "5O0 mg").replace("Authors:", "Auth0rs:")
            noisy[did] = s
        synth = True
    else:
        synth = False

    clean_fields = extract_fields(clean)
    raw_fields = extract_fields(noisy)
    norm_docs, n_edits = normalize_corpus(noisy)
    norm_fields = extract_fields(norm_docs)

    raw_rec = field_recovery(clean_fields, raw_fields)
    norm_rec = field_recovery(clean_fields, norm_fields)
    agree_raw = char_agreement(clean, noisy)
    agree_norm = char_agreement(clean, norm_docs)

    field_gap = round(1.0 - norm_rec["recovery_rate"], 4)
    sla_field_ok = norm_rec["recovery_rate"] >= MIN_FIELD_RECOVERY
    improve = norm_rec["recovery_rate"] - raw_rec["recovery_rate"]

    recover_gap_u = None
    verdict = "FIELD_SLA_PASS" if sla_field_ok else "FIELD_SLA_FAIL"
    if u_clean is not None and u_noisy_norm is not None:
        recover_gap_u = round(u_clean - u_noisy_norm, 4)
        if recover_gap_u <= MAX_RECOVER_GAP_U and sla_field_ok:
            verdict = "NOISY_INGEST_NORMALIZE_SUFFICIENT"
        elif improve > 0.05 and not sla_field_ok:
            verdict = "NORMALIZE_HELPS_BUT_GAP_REMAINS"
        elif not sla_field_ok:
            verdict = "NOISY_TRACK_STRESS_UNRESOLVED"

    latency_ms = int(round((time.perf_counter() - t0) * 1000))
    return {
        "schema": "nano-lm.wedge_v1.ingest_sla.v1",
        "workstream": "W5",
        "clean_dir": str(Path(clean_dir).resolve()),
        "noisy_dir": str(Path(noisy_dir).resolve()) if Path(noisy_dir).is_dir() else "synthetic_from_clean",
        "synthetic_noisy": synth,
        "n_clean_docs": len(clean),
        "n_noisy_docs": len(noisy),
        "n_ocr_edits": n_edits,
        "char_agreement_raw_vs_clean": agree_raw,
        "char_agreement_norm_vs_clean": agree_norm,
        "fields_raw": raw_rec,
        "fields_normalized": norm_rec,
        "field_recovery_improve": round(improve, 4),
        "field_gap": field_gap,
        "thresholds": {
            "MIN_FIELD_RECOVERY": MIN_FIELD_RECOVERY,
            "MAX_RECOVER_GAP_U": MAX_RECOVER_GAP_U,
        },
        "u_clean": u_clean,
        "u_noisy_raw": u_noisy_raw,
        "u_noisy_norm": u_noisy_norm,
        "recover_gap_vs_clean_U": recover_gap_u,
        "verdict": verdict,
        "sla_field_ok": sla_field_ok,
        "lm_invoked": False,
        "latency_ms": latency_ms,
        "lexicon_sha256": hashlib.sha256(
            (ROOT / "plugins" / "data" / "ocr_substitutions.json").read_bytes()
        ).hexdigest()[:16],
        "note": "Normalize is ingest preprocessing; primary U remains clean-track.",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Ingest SLA / recover_gap (W5)")
    p.add_argument("--clean", type=Path, default=ROOT / "data" / "corpus")
    p.add_argument("--noisy", type=Path, default=ROOT / "data" / "corpus_noisy")
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--with-u", action="store_true", help="Attach U from prior noisy diagnostic JSON if present")
    args = p.parse_args(argv)

    u_clean = u_raw = u_norm = None
    if args.with_u:
        prev = ROOT / "results_wedge_v1_noisy_diagnostic.json"
        if prev.exists():
            d = json.loads(prev.read_text(encoding="utf-8"))
            u_clean = d.get("primary_U_clean")
            u_raw = (d.get("noisy_raw") or {}).get("U")
            u_norm = (d.get("noisy_normalized") or {}).get("U")

    out = measure_ingest_sla(
        clean_dir=args.clean,
        noisy_dir=args.noisy,
        u_clean=u_clean,
        u_noisy_raw=u_raw,
        u_noisy_norm=u_norm,
    )
    text = json.dumps(out, indent=2) + chr(10)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        # Default write is a local engineering result, not evidence core.
        default_out = ROOT / "results_ingest_sla.json"
        default_out.write_text(text, encoding="utf-8")
        sys.stdout.write(text)
        print(f"WROTE {default_out}", file=sys.stderr)
    return 0 if out.get("sla_field_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
