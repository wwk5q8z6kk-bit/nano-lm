"""R★ verify-on/off presenter — grounding + absence adapted to free-text notes.

Fabric harness pattern (literal patientish grounding); ≠ NanoScribe product.
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

from trajectory.e1.common import FIELDS, ItemPred
from trajectory.e4.methods import dialogue_of, patientish_text

DENIAL_MED = ("no med", "denies medication", "haven't taken", "nothing yet", "not taking", "nothing at all")
DENIAL_ALG = ("no allerg", "denies allerg", "nkda", "none whatsoever", "not that i know", "nothing on record")


def stage_flag(field: str, pred_val: str, dialogue: str) -> bool:
    """True => route to review (abstain from presentation)."""
    p = patientish_text(dialogue)
    if pred_val != "none":
        # grounding: normalized token presence
        tokens = normalize_loose(pred_val)
        if not tokens:
            return True
        # all content tokens should appear
        return not all(re.search(rf"\b{re.escape(t)}\b", p, re.I) for t in tokens if len(t) > 1)
    # absence claims
    if field == "med":
        # if any med-like cue without denial → flag
        if any(x in p for x in DENIAL_MED):
            return False
        # presence of "taking/tried/using" without denial → should not claim none confidently
        if any(x in p for x in ("taking", "tried", "using", "meds ")):
            return True
        return False
    if field == "alg":
        if any(x in p for x in DENIAL_ALG):
            return False
        if "allergic" in p or "allergy" in p:
            return True
        return False
    # cc/dur/sev claimed none → always review
    return True


def normalize_loose(s: str):
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for art in ("a ", "an ", "the ", "some ", "bit of "):
        if s.startswith(art):
            s = s[len(art):]
    return [t for t in s.split() if t not in {"of", "the", "a", "an"}]


def apply_verify(item_pred: ItemPred, dialogue: str, verify_on: bool) -> Tuple[Dict[str, str | None], int]:
    presented = {}
    flagged = 0
    for f in FIELDS:
        val = item_pred.fields[f].value
        if not verify_on:
            presented[f] = val
            continue
        if stage_flag(f, val, dialogue):
            flagged += 1
            presented[f] = None
        else:
            presented[f] = val
    return presented, flagged
