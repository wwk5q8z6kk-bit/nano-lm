"""OCR normalize plugin — table-driven substitutions with evidence."""
from __future__ import annotations

import re

from wedge_v1.classical.solvers import Claim
from wedge_v1.plugins.lexicon import ocr_subs


def normalize_text(text: str) -> tuple[str, list[dict]]:
    """Apply enabled substitutions; return (fixed_text, edit_log)."""
    fixed = text
    edits = []
    for row in ocr_subs():
        src = str(row.get("from") or "")
        dst = str(row.get("to") or "")
        if not src or src not in fixed:
            continue
        i = fixed.find(src)
        fixed = fixed.replace(src, dst)
        edits.append({"from": src, "to": dst, "at": i})
    return fixed, edits


def probe_docs(docs: dict[str, str]) -> list[Claim]:
    """Extract TTL/dose from OCR-normalized text for any doc."""
    ttl_pat = re.compile("TTL" + r"\s+(?:is|as)\s+(\d+)\s+seconds", re.I)
    dose_pat = re.compile(r"dose:\s*(\d+)\s*mg", re.I)
    out: list[Claim] = []
    for did, text in docs.items():
        fixed, edits = normalize_text(text)
        if not edits:
            continue
        val = {}
        evidence = []
        m = ttl_pat.search(fixed)
        if m:
            val["ttl_seconds"] = int(m.group(1))
            evidence.append({"doc_id": did, "start": m.start(1), "end": m.end(1), "text": m.group(1), "channel": "ocr_normalized"})
        d = dose_pat.search(fixed)
        if d:
            val["dose_mg"] = int(d.group(1))
            evidence.append({"doc_id": did, "start": d.start(1), "end": d.end(1), "text": d.group(1), "channel": "ocr_normalized"})
        if not val:
            out.append(
                Claim(
                    "T37",
                    did,
                    {"edits": edits},
                    evidence=[{"text": e["from"] + "→" + e["to"]} for e in edits[:5]],
                    status="PRESENT",
                    notes="plugin.ocr.edits_only",
                    meta={"plugin": "ocr", "edits": edits},
                )
            )
            continue
        out.append(
            Claim(
                "T37",
                did,
                val,
                evidence=evidence,
                status="PRESENT",
                notes="plugin.ocr.normalize",
                meta={"plugin": "ocr", "edits": edits},
            )
        )
    return out
