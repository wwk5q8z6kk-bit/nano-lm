"""E4 classical + generative predictors for R★."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from trajectory.e1.common import (
    FIELDS,
    FieldPred,
    ItemPred,
    format_summary,
    normalize_value,
    parse_summary,
    pred_from_values,
)

E4 = Path(__file__).resolve().parent
DATA = E4 / "data"
RULES = json.loads((E4 / "c_m1_rules.json").read_text())
LEX = json.loads((DATA / "rstar_train_lexicon.json").read_text())


def dialogue_of(item: dict) -> str:
    return item["convo"][0]["content"].rsplit("\nSummarize the visit.", 1)[0]


def patientish_text(dialogue: str) -> str:
    lines = []
    for ln in dialogue.split("\n"):
        if ln.startswith("Patient: "):
            lines.append(ln[len("Patient: "):])
        elif ln.startswith("Doctor: "):
            continue
        else:
            lines.append(ln)
    return " ".join(lines).lower()


def cue_hit(dialogue: str, field: str) -> bool:
    low = dialogue.lower()
    return any(c in low for c in RULES["cues"].get(field, []))


def predict_c_m1(item: dict, source_id: str) -> ItemPred:
    """Frozen template/regex slot filler — only canonical QA cues."""
    t0 = time.perf_counter()
    dialogue = dialogue_of(item)
    values = {f: "none" for f in FIELDS}
    spans = {}

    # Only extract when cue fires (budget frozen)
    for field, patterns in RULES["capture_patterns"].items():
        if field.endswith("_yes") or field.endswith("_no"):
            continue
        if not cue_hit(dialogue, field if field in FIELDS else field.split("_")[0]):
            continue
        for pat in patterns:
            m = re.search(pat, dialogue, re.I)
            if not m:
                continue
            if field == "dur":
                n, unit = m.group(1), m.group(2)
                unit = "days" if unit.lower().startswith("day") else "weeks"
                values["dur"] = f"{n} {unit}"
            elif field == "sev":
                values["sev"] = m.group(1).lower()
            elif field == "cc":
                frag = m.group(1).strip().rstrip(".")
                # strip articles
                frag = re.sub(r"^(a|an|the)\s+", "", frag, flags=re.I)
                values["cc"] = frag
            break

    # med/alg
    if cue_hit(dialogue, "med"):
        hit_no = any(re.search(p, dialogue, re.I) for p in RULES["capture_patterns"]["med_no"])
        if hit_no:
            values["med"] = "none"
        else:
            for pat in RULES["capture_patterns"]["med_yes"]:
                m = re.search(pat, dialogue, re.I)
                if m:
                    values["med"] = m.group(1).strip().rstrip(".")
                    break
    if cue_hit(dialogue, "alg"):
        hit_no = any(re.search(p, dialogue, re.I) for p in RULES["capture_patterns"]["alg_no"])
        if hit_no:
            values["alg"] = "none"
        else:
            for pat in RULES["capture_patterns"]["alg_yes"]:
                m = re.search(pat, dialogue, re.I)
                if m:
                    values["alg"] = m.group(1).strip().rstrip(".")
                    break

    return pred_from_values(values, spans, latency_s=time.perf_counter() - t0)


def predict_c_m2(item: dict, source_id: str) -> ItemPred:
    """Train-dict + span; no eval lexicon."""
    t0 = time.perf_counter()
    dialogue = dialogue_of(item)
    p = patientish_text(dialogue)
    values = {f: "none" for f in FIELDS}
    spans = {}

    for sev in ("mild", "moderate", "severe"):
        if re.search(rf"\b{sev}\b", p):
            values["sev"] = sev
            break

    m = re.search(r"\b(\d+)\s+(days?|weeks?)\b", p)
    if m:
        unit = "days" if m.group(2).startswith("day") else "weeks"
        values["dur"] = f"{m.group(1)} {unit}"

    for field, vocab in (("cc", LEX["cc"]), ("med", LEX["med"]), ("alg", LEX["alg"])):
        for term in sorted(vocab, key=len, reverse=True):
            if re.search(rf"\b{re.escape(term)}\b", p, re.I):
                values[field] = term
                break
        else:
            # denial heuristics
            if field == "med" and any(x in p for x in ("no med", "denies medication", "nothing yet", "haven't taken", "not taking")):
                values["med"] = "none"
            if field == "alg" and any(x in p for x in ("no allerg", "denies allerg", "nkda", "none whatsoever", "not that i know")):
                values["alg"] = "none"

    return pred_from_values(values, spans, latency_s=time.perf_counter() - t0)


def predict_c_m4(item: dict, source_id: str) -> ItemPred:
    """Constrained copy-only open slots from patientish text; closed from sets."""
    t0 = time.perf_counter()
    dialogue = dialogue_of(item)
    p = patientish_text(dialogue)
    values = {f: "none" for f in FIELDS}

    for sev in ("mild", "moderate", "severe"):
        if re.search(rf"\b{sev}\b", p):
            values["sev"] = sev
            break
    m = re.search(r"\b(\d+)\s+(days?|weeks?)\b", p)
    if m:
        unit = "days" if m.group(2).startswith("day") else "weeks"
        values["dur"] = f"{m.group(1)} {unit}"

    # Prefer train vocab hit; else longest plausible token span after keywords — still must be substring
    for field, vocab, keys in (
        ("cc", LEX["cc"], ["reports", "having", "bothering", "thought it was", "actually it's", "visit:", "say"]),
        ("med", LEX["med"], ["taking", "tried", "using", "meds", "switched to", "only"]),
        ("alg", LEX["alg"], ["allergic to", "allergy", "allerg"]),
    ):
        hit = None
        for term in sorted(vocab, key=len, reverse=True):
            if re.search(rf"\b{re.escape(term)}\b", p, re.I):
                hit = term
                break
        if hit is None:
            # copy after key without lexicon (open) — take up to 4 words
            for key in keys:
                m2 = re.search(rf"{re.escape(key)}\s+([a-z0-9][\w\s-]{{1,40}})", p, re.I)
                if m2:
                    frag = m2.group(1).strip().rstrip(".")
                    frag = re.split(r"[.,;]", frag)[0].strip()
                    words = frag.split()
                    frag = " ".join(words[:4])
                    if frag and frag in p:
                        # normalize mild article strip for cc
                        if field == "cc":
                            frag = re.sub(r"^(a|an|the|bit of|some)\s+", "", frag).strip()
                        hit = frag
                        break
        if hit is None and field in ("med", "alg"):
            if field == "med" and any(x in p for x in ("no med", "denies medication", "haven't taken", "nothing")):
                hit = "none"
            if field == "alg" and any(x in p for x in ("no allerg", "nkda", "denies allerg", "none")):
                hit = "none"
        values[field] = hit or "none"

    return pred_from_values(values, latency_s=time.perf_counter() - t0)


# Relative compute vs C-M1 on same class (design assignments; refined at runtime if measured)
COST_C = {
    "C-M1": 1.0,
    "C-M2": 1.2,
    "C-M4": 1.5,
    "G-ref": 40.0,
}
