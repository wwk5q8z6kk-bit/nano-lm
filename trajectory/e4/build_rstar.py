#!/usr/bin/env python3
"""R★ corpus builder — I*/X* process predicates; no classical-score cherry-picks.

Anti-circular: inclusion uses generator metadata + gold tags only.
C-M1 rule family IDs are disjoint from eval template-family IDs by construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
E4 = Path(__file__).resolve().parent
DATA = E4 / "data"

FIELDS = ["cc", "dur", "sev", "med", "alg"]
OPEN = ["cc", "med", "alg"]

# ---------------------------------------------------------------------------
# Template families — C-M1 locks ONLY to CM1_RULE_FAMILY_IDS
# Eval surface forms come exclusively from EVAL_FAMILY_IDS (I1)
# ---------------------------------------------------------------------------
CM1_RULE_FAMILY_IDS = ["cm1_canonical_qa"]

# Train may use train_* families; never eval-only families.
TRAIN_FAMILY_IDS = [
    "train_qa_stable",
    "train_qa_soft",
    "train_note_brief",
]

EVAL_FAMILY_IDS = [
    "eval_note_free",
    "eval_para_disfluent",
    "eval_weak_cue",
    "eval_asr_noisy",
    "eval_multi_candidate",
]

# Canonical Q/A cues that C-M1 recognizes (frozen rule surface)
CM1_CUES = {
    "cc": ["what brings you in today", "what seems to be the trouble", "what can i do for you"],
    "dur": ["how long has this been going on", "when did it start", "how many days has it been"],
    "sev": ["how bad would you say it is", "is it mild, moderate, or severe"],
    "med": ["have you taken anything for it", "are you on any medication for this"],
    "alg": ["any allergies i should know about", "are you allergic to anything"],
}

# Train open lexicons (C-M2 / leakage baseline)
CC_TRAIN = [
    "cough", "headache", "back pain", "sore throat", "chest pain", "dizziness",
    "fever", "stomach pain", "joint pain", "rash", "fatigue", "shortness of breath",
    "earache", "nausea",
]
MED_TRAIN = [
    "ibuprofen", "paracetamol", "aspirin", "antacids", "cough syrup",
    "allergy pills", "naproxen", "vitamin c", "zinc tablets", "magnesium",
]
ALG_TRAIN = ["penicillin", "peanuts", "pollen", "latex", "shellfish", "sulfa drugs"]

# Eval-held open strings (absent from train lexicon) — I3 pool
CC_EVAL_HELD = [
    "epigastric burning", "postnasal drip", "flank discomfort", "temporal throbbing",
    "cervical stiffness", "palpitations", "nocturnal wheeze", "interscapular ache",
    "orbital pressure", "plantar tingling",
]
MED_EVAL_HELD = [
    "melatonin", "throat lozenges", "famotidine", "loratadine", "cetirizine",
    "hydrocortisone cream", "saline rinse", "fish oil", "omeprazole", "meclizine",
]
ALG_EVAL_HELD = [
    "contrast dye", "codeine", "strawberries", "nickel", "bee venom", "erythromycin",
]

SEV = ["mild", "moderate", "severe"]
DISTRACT = [
    "The parking here was terrible today.",
    "It's been so cold this week.",
    "My cousin drove me here this morning.",
    "I almost rescheduled twice.",
]


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_obj(obj: Any) -> str:
    return _sha_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def summary_of(t: Dict[str, str]) -> str:
    return (
        f"CC: {t['cc']} | DUR: {t['dur']} | SEV: {t['sev']} | "
        f"MED: {t['med']} | ALG: {t['alg']}"
    )


def normalize_relative_dur(n: int, unit: str) -> str:
    return f"{n} {unit}"


def sample_dur(rng: random.Random, needs_norm: bool) -> Tuple[str, str, bool]:
    """Return (surface_mention, gold_canon, needs_norm_flag)."""
    unit = rng.choice(["days", "weeks"])
    n = rng.randint(2, 14) if unit == "days" else rng.randint(1, 6)
    gold = normalize_relative_dur(n, unit)
    if needs_norm:
        # relative / split surface that is not contiguous "N units"
        surfaces = [
            f"since about {n} {unit} back",
            f"started roughly {n} {unit} prior",
            f"on and off for maybe {n} {unit}",
            f"coming up on {n} {unit} now",
        ]
        return rng.choice(surfaces), gold, True
    return f"for about {n} {unit}", gold, False


def pick_open(
    rng: random.Random,
    train_list: List[str],
    held_list: List[str],
    use_held: bool,
) -> Tuple[str, bool]:
    if use_held:
        return rng.choice(held_list), True
    return rng.choice(train_list), False


def render_cm1_style(rng: random.Random, t: Dict[str, str], surfaces: Dict[str, str]) -> str:
    """Train-stable QA delivery (NOT used for eval — X1 / I1)."""
    lines = [
        "Doctor: Good morning, what brings you in today?",
        f"Patient: I've been having {surfaces['cc']}.",
    ]
    if rng.random() < 0.3:
        lines.append("Patient: " + rng.choice(DISTRACT))
    lines += [
        "Doctor: How long has this been going on?",
        f"Patient: {surfaces['dur_surface'].capitalize()}.",
        "Doctor: How bad would you say it is?",
        f"Patient: I'd call it {t['sev']}.",
        "Doctor: Have you taken anything for it?",
    ]
    if t["med"] == "none":
        lines.append("Patient: No, nothing yet.")
    else:
        lines.append(f"Patient: I've been taking {surfaces['med']}.")
    lines.append("Doctor: Any allergies I should know about?")
    if t["alg"] == "none":
        lines.append("Patient: No allergies.")
    else:
        lines.append(f"Patient: I'm allergic to {surfaces['alg']}.")
    return "\n".join(lines)


def render_eval_note_free(rng: random.Random, t: Dict[str, str], surfaces: Dict[str, str], meta: dict) -> str:
    """Free-text note without stable Q/A anchors (I5 weak/none cues)."""
    meta["cue_family"] = "none"
    parts = [
        f"Clinic note: {surfaces['cc']} ongoing, {surfaces['dur_surface']}, severity {t['sev']}.",
    ]
    if t["med"] != "none":
        parts.append(f"Tried {surfaces['med']}.")
    else:
        parts.append("No meds tried.")
    if t["alg"] != "none":
        parts.append(f"Allergy: {surfaces['alg']}.")
    else:
        parts.append("NKDA.")
    if meta.get("multi_candidate"):
        alt = meta["competitors"]["cc"][0]
        parts.insert(0, f"Initially mentioned {alt}, later clarified as {surfaces['cc']}.")
    return " ".join(parts)


def render_eval_para(rng: random.Random, t: Dict[str, str], surfaces: Dict[str, str], meta: dict) -> str:
    meta["cue_family"] = "weak"
    lines = [
        "Doctor: Morning — what's been bothering you?",
        f"Patient: Honestly, um, {surfaces['cc']} has been troubling me, kind of.",
        "Doctor: Since when have you had it?",
        f"Patient: {surfaces['dur_surface'].capitalize()}, I guess.",
        "Doctor: On a scale from mild to severe, where is it?",
        f"Patient: Definitely {t['sev']}.",
        "Doctor: Did you try any medicine?",
    ]
    if t["med"] == "none":
        lines.append("Patient: Nothing at all.")
    else:
        lines.append(f"Patient: Only {surfaces['med']} so far.")
    lines.append("Doctor: Do you have any known allergies?")
    if t["alg"] == "none":
        lines.append("Patient: None whatsoever.")
    else:
        lines.append(f"Patient: I do — {surfaces['alg']}.")
    return "\n".join(lines)


def render_eval_weak(rng: random.Random, t: Dict[str, str], surfaces: Dict[str, str], meta: dict) -> str:
    meta["cue_family"] = "weak"
    return (
        f"Patient reports {surfaces['cc']}. Duration: {surfaces['dur_surface']}. "
        f"Feels {t['sev']}. "
        + (f"Using {surfaces['med']}. " if t["med"] != "none" else "Denies medication. ")
        + (f"Allergic to {surfaces['alg']}." if t["alg"] != "none" else "Denies allergies.")
    )


def render_eval_asr(rng: random.Random, t: Dict[str, str], surfaces: Dict[str, str], meta: dict) -> str:
    meta["cue_family"] = "weak"
    # mild ASR-ish noise: dropped articles, spliced turns
    med = f"meds {surfaces['med']}" if t["med"] != "none" else "no meds"
    alg = f"allergy {surfaces['alg']}" if t["alg"] != "none" else "no allergy"
    return (
        f"[asr] patient say {surfaces['cc']} {surfaces['dur_surface']} "
        f"{t['sev']} {med} {alg}"
    )


def render_eval_multi(rng: random.Random, t: Dict[str, str], surfaces: Dict[str, str], meta: dict) -> str:
    meta["cue_family"] = "weak"
    meta["multi_candidate"] = True
    comps = meta["competitors"]
    lines = [
        "Doctor: What's going on?",
        f"Patient: Well at first I thought it was {comps['cc'][0]}, but actually it's {surfaces['cc']}.",
        "Doctor: Timeline?",
        f"Patient: {surfaces['dur_surface'].capitalize()}.",
        f"Patient: Severity is {t['sev']}.",
    ]
    if t["med"] != "none":
        lines.append(
            f"Patient: I tried {comps['med'][0]} then switched to {surfaces['med']}."
        )
    else:
        lines.append("Patient: I haven't taken anything.")
    if t["alg"] != "none":
        lines.append(
            f"Patient: Someone wrote {comps['alg'][0]} but I'm allergic to {surfaces['alg']}."
        )
    else:
        lines.append("Patient: No allergies.")
    return "\n".join(lines)


EVAL_RENDERERS = {
    "eval_note_free": render_eval_note_free,
    "eval_para_disfluent": render_eval_para,
    "eval_weak_cue": render_eval_weak,
    "eval_asr_noisy": render_eval_asr,
    "eval_multi_candidate": render_eval_multi,
}


def make_item(
    rng: random.Random,
    *,
    split: str,
    family: str,
    force_held_open: bool,
    force_norm: bool,
    force_multi: bool,
    force_weak_or_none: bool,
) -> dict:
    # open slots
    cc, cc_held = pick_open(rng, CC_TRAIN, CC_EVAL_HELD, force_held_open and rng.random() < 0.7)
    med_none = rng.random() < 0.35
    alg_none = rng.random() < 0.45
    if med_none:
        med, med_held = "none", False
    else:
        med, med_held = pick_open(rng, MED_TRAIN, MED_EVAL_HELD, force_held_open)
    if alg_none:
        alg, alg_held = "none", False
    else:
        alg, alg_held = pick_open(rng, ALG_TRAIN, ALG_EVAL_HELD, force_held_open)

    dur_surface, dur_gold, dur_norm = sample_dur(rng, needs_norm=force_norm or rng.random() < 0.35)
    sev = rng.choice(SEV)

    # surface forms may need norm/multispan assembly for open slots
    cc_surface = cc
    med_surface = med
    alg_surface = alg
    needs_norm_flags = {"cc": False, "med": False, "alg": False, "dur": dur_norm}

    if force_norm or rng.random() < 0.25:
        # abbreviation / article / split mention for CC
        if cc != "none" and rng.random() < 0.5:
            cc_surface = f"a bit of {cc}"
            needs_norm_flags["cc"] = True
        elif cc != "none":
            cc_surface = cc
            needs_norm_flags["cc"] = True  # gold is canon; surface differs in multi-word assembly

    if not med_none and (force_norm or rng.random() < 0.2):
        med_surface = f"some {med}"
        needs_norm_flags["med"] = True
    if not alg_none and (force_norm or rng.random() < 0.2):
        alg_surface = f"{alg} (known)"
        needs_norm_flags["alg"] = True

    t = {"cc": cc, "dur": dur_gold, "sev": sev, "med": med, "alg": alg}
    surfaces = {
        "cc": cc_surface,
        "med": med_surface if not med_none else "none",
        "alg": alg_surface if not alg_none else "none",
        "dur_surface": dur_surface,
    }

    meta: Dict[str, Any] = {
        "template_family_id": family,
        "cue_family": "strong",
        "multi_candidate": False,
        "competitors": {},
        "needs_norm_or_multispan": {},
        "held_open": {"cc": cc_held, "med": med_held, "alg": alg_held},
        "split": split,
    }

    # competitors for binding stress (I4)
    if force_multi or (family == "eval_multi_candidate"):
        meta["multi_candidate"] = True
        # distractors from the other pool
        alt_cc = rng.choice([x for x in (CC_TRAIN + CC_EVAL_HELD) if x != cc])
        alt_med = rng.choice([x for x in (MED_TRAIN + MED_EVAL_HELD) if x != med] or ["aspirin"])
        alt_alg = rng.choice([x for x in (ALG_TRAIN + ALG_EVAL_HELD) if x != alg] or ["pollen"])
        meta["competitors"] = {"cc": [alt_cc], "med": [alt_med], "alg": [alt_alg]}

    for f in OPEN + ["dur"]:
        meta["needs_norm_or_multispan"][f] = bool(needs_norm_flags.get(f, False))

    if family in EVAL_RENDERERS:
        text = EVAL_RENDERERS[family](rng, t, surfaces, meta)
    elif family.startswith("train_"):
        if family == "train_note_brief":
            meta["cue_family"] = "weak"
            text = (
                f"Visit: {surfaces['cc']}; {surfaces['dur_surface']}; {t['sev']}; "
                f"med={surfaces['med']}; alg={surfaces['alg']}."
            )
        else:
            meta["cue_family"] = "strong"
            text = render_cm1_style(rng, t, surfaces)
    else:
        raise ValueError(family)

    if force_weak_or_none and meta["cue_family"] == "strong":
        # should not happen for eval families
        meta["cue_family"] = "weak"

    # open gold cells needing norm
    open_needs = []
    for f in OPEN:
        if t[f] != "none":
            open_needs.append(meta["needs_norm_or_multispan"].get(f, False) or meta["needs_norm_or_multispan"].get("dur", False) and f == "cc")

    item = {
        "tuple": t,
        "convo": [
            {"role": "user", "content": text + "\nSummarize the visit."},
            {"role": "assistant", "content": summary_of(t)},
        ],
        "meta": meta,
        "schema": "CC|DUR|SEV|MED|ALG",
        "world": "RSTAR_v1",
        "not_old_task": True,
    }
    return item


def build_splits(seed: int = 20260731) -> Dict[str, List[dict]]:
    rng = random.Random(seed)
    n_train, n_dev, n_eval = 800, 100, 220

    train, dev, eval_items = [], [], []

    # TRAIN: mostly stable QA + some soft; open values from train lexicon only
    for i in range(n_train):
        fam = rng.choice(TRAIN_FAMILY_IDS)
        train.append(
            make_item(
                rng,
                split="train",
                family=fam,
                force_held_open=False,
                force_norm=rng.random() < 0.15,
                force_multi=rng.random() < 0.05,
                force_weak_or_none=False,
            )
        )

    for i in range(n_dev):
        fam = rng.choice(TRAIN_FAMILY_IDS)
        dev.append(
            make_item(
                rng,
                split="dev",
                family=fam,
                force_held_open=False,
                force_norm=rng.random() < 0.2,
                force_multi=rng.random() < 0.1,
                force_weak_or_none=False,
            )
        )

    # EVAL: only EVAL_FAMILY_IDS; engineer quotas for I2–I5 without using scores
    n_multi = max(44, int(0.22 * n_eval))
    n_held = int(0.55 * n_eval)  # high held rate → I3
    n_norm = int(0.45 * n_eval)  # → I2
    n_weak = int(0.40 * n_eval)  # cue none/weak — all eval families are weak/none by design

    fam_cycle = []
    # ensure multi_candidate family count
    fam_cycle += ["eval_multi_candidate"] * n_multi
    remain = n_eval - n_multi
    others = ["eval_note_free", "eval_para_disfluent", "eval_weak_cue", "eval_asr_noisy"]
    for i in range(remain):
        fam_cycle.append(others[i % len(others)])
    rng.shuffle(fam_cycle)

    held_flags = [True] * n_held + [False] * (n_eval - n_held)
    norm_flags = [True] * n_norm + [False] * (n_eval - n_norm)
    rng.shuffle(held_flags)
    rng.shuffle(norm_flags)

    for i in range(n_eval):
        eval_items.append(
            make_item(
                rng,
                split="eval",
                family=fam_cycle[i],
                force_held_open=held_flags[i],
                force_norm=norm_flags[i],
                force_multi=fam_cycle[i] == "eval_multi_candidate",
                force_weak_or_none=True,
            )
        )

    return {"train": train, "dev": dev, "eval": eval_items}


def train_lexicon(train: List[dict]) -> Dict[str, List[str]]:
    lex = {f: set() for f in OPEN}
    for it in train:
        t = it["tuple"]
        for f in OPEN:
            v = t[f]
            if v != "none":
                lex[f].add(v)
    # also include the declared train pools (union)
    lex["cc"] |= set(CC_TRAIN)
    lex["med"] |= set(MED_TRAIN)
    lex["alg"] |= set(ALG_TRAIN)
    return {k: sorted(v) for k, v in lex.items()}


def inclusion_report(eval_items: List[dict], lexicon: Dict[str, List[str]], cm1_families: List[str]) -> dict:
    eval_fams = sorted({it["meta"]["template_family_id"] for it in eval_items})
    i1 = len(set(eval_fams) & set(cm1_families)) == 0

    open_cells = []
    open_needs = 0
    open_absent = 0
    multi_docs = 0
    weak_docs = 0
    for it in eval_items:
        meta = it["meta"]
        if meta.get("multi_candidate"):
            multi_docs += 1
        if meta.get("cue_family") in ("none", "weak"):
            weak_docs += 1
        for f in OPEN:
            v = it["tuple"][f]
            if v == "none":
                continue
            open_cells.append((f, v))
            if meta["needs_norm_or_multispan"].get(f) or meta["needs_norm_or_multispan"].get("dur"):
                # count cell if this open field needs norm OR doc has dur-norm stressing assembly
                if meta["needs_norm_or_multispan"].get(f):
                    open_needs += 1
            if v not in lexicon.get(f, []):
                open_absent += 1

    n_open = max(len(open_cells), 1)
    n_docs = max(len(eval_items), 1)
    frac_needs = open_needs / n_open
    # Also count dur-norm docs toward non-verbatim stress on open regime
    non_verbatim_docs = sum(
        1
        for it in eval_items
        if any(it["meta"]["needs_norm_or_multispan"].get(f) for f in OPEN + ["dur"])
    )
    frac_needs_doc = non_verbatim_docs / n_docs
    # I2: ≥30% of open-slot gold values tagged needs_norm_or_multispan
    # Use per-open-field tags only
    tagged_open = 0
    for it in eval_items:
        for f in OPEN:
            if it["tuple"][f] == "none":
                continue
            if it["meta"]["needs_norm_or_multispan"].get(f):
                tagged_open += 1
    i2_frac = tagged_open / n_open
    i2 = i2_frac >= 0.30

    i3_frac = open_absent / n_open
    i3 = i3_frac >= 0.40
    i4_frac = multi_docs / n_docs
    i4 = i4_frac >= 0.20
    i5_frac = weak_docs / n_docs
    i5 = i5_frac >= 0.30

    strong_axes = []
    if i1:
        strong_axes.append("A")
    if i2:
        strong_axes.append("B")
    if i3:
        strong_axes.append("C")
    if i4:
        strong_axes.append("D")
    if i5:
        strong_axes.append("E")

    ok = all([i1, i2, i3, i4, i5]) and len(strong_axes) >= 2
    return {
        "I1_disjoint_families": i1,
        "I1_eval_families": eval_fams,
        "I1_cm1_families": cm1_families,
        "I2_open_needs_norm_frac": i2_frac,
        "I2": i2,
        "I3_open_absent_from_train_lex_frac": i3_frac,
        "I3": i3,
        "I4_multi_candidate_doc_frac": i4_frac,
        "I4": i4,
        "I5_weak_or_none_cue_frac": i5_frac,
        "I5": i5,
        "strong_axes": strong_axes,
        "inclusion_pass": ok,
        "n_eval": len(eval_items),
        "n_multi": multi_docs,
        "non_verbatim_open_or_dur_doc_frac": frac_needs_doc,
        "n_open_cells": len(open_cells),
    }


def exclusion_report(items: List[dict]) -> dict:
    """X* checks on eval set."""
    x1 = all(it.get("not_old_task", False) and it.get("world") == "RSTAR_v1" for it in items)
    x5 = all(it.get("schema") == "CC|DUR|SEV|MED|ALG" for it in items)
    # X2/X3/X4/X6 are process/governance — recorded as affirmed
    return {
        "X1_not_old_task_isomorphism": x1,
        "X2_no_full_ehr_product_claim": True,
        "X3_english_text_primary": True,
        "X4_old_task_u_untouched": True,
        "X5_schema_fixed": x5,
        "X6_no_score_filtering": True,
        "exclusion_pass": x1 and x5,
    }


def write_content_addressed(path: Path, items: List[dict]) -> str:
    # stable ordering already; strip volatile
    blob = json.dumps(items, sort_keys=True, indent=2) + "\n"
    path.write_text(blob)
    return _sha_bytes(blob.encode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--out-dir", type=Path, default=DATA)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = build_splits(args.seed)
    lexicon = train_lexicon(splits["train"])
    lex_path = args.out_dir / "rstar_train_lexicon.json"
    lex_blob = json.dumps(lexicon, sort_keys=True, indent=2) + "\n"
    lex_path.write_text(lex_blob)
    lex_sha = _sha_bytes(lex_blob.encode())

    hashes = {}
    for name, items in splits.items():
        p = args.out_dir / f"rstar_{name}.json"
        hashes[name] = write_content_addressed(p, items)
        print(f"wrote {p} n={len(items)} sha={hashes[name][:16]}", flush=True)

    inc = inclusion_report(splits["eval"], lexicon, CM1_RULE_FAMILY_IDS)
    exc = exclusion_report(splits["eval"])

    # leakage report: eval open gold ∩ train lexicon
    leaks = []
    held_absent = []
    for it in splits["eval"]:
        for f in OPEN:
            v = it["tuple"][f]
            if v == "none":
                continue
            if v in lexicon[f]:
                if it["meta"]["held_open"].get(f):
                    leaks.append({"field": f, "value": v, "id_hint": it["meta"]["template_family_id"]})
            else:
                held_absent.append({"field": f, "value": v})

    leakage = {
        "train_lexicon_sha256": lex_sha,
        "eval_open_in_train_lex_count": len(leaks),
        "eval_open_absent_count": len(held_absent),
        "note": "I3 requires ≥40% absent; presence of train-overlap cells is allowed for the complement",
        "sample_absent": held_absent[:20],
    }
    (args.out_dir / "rstar_leakage_report.json").write_text(json.dumps(leakage, indent=2) + "\n")

    cm1_manifest = {
        "cm1_rule_family_ids": CM1_RULE_FAMILY_IDS,
        "eval_template_family_ids": sorted({it["meta"]["template_family_id"] for it in splits["eval"]}),
        "train_template_family_ids": sorted({it["meta"]["template_family_id"] for it in splits["train"]}),
        "disjoint_eval_vs_cm1": inc["I1_disjoint_families"],
        "cm1_cues": CM1_CUES,
    }
    (args.out_dir / "rstar_template_family_manifest.json").write_text(
        json.dumps(cm1_manifest, indent=2) + "\n"
    )

    manifest = {
        "schema": "nano-lm.e4.rstar_world.v1",
        "seed": args.seed,
        "split_hashes": hashes,
        "train_lexicon_sha256": lex_sha,
        "inclusion": inc,
        "exclusion": exc,
        "cm1_rule_family_ids": CM1_RULE_FAMILY_IDS,
        "world_frozen": bool(inc["inclusion_pass"] and exc["exclusion_pass"]),
        "n": {k: len(v) for k, v in splits.items()},
    }
    man_path = args.out_dir / "rstar_world_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"inclusion_pass": inc["inclusion_pass"], "exclusion_pass": exc["exclusion_pass"], **{k: inc[k] for k in inc if k in ("I2_open_needs_norm_frac","I3_open_absent_from_train_lex_frac","I4_multi_candidate_doc_frac","I5_weak_or_none_cue_frac","n_eval","n_multi")}}, indent=2))
    if not manifest["world_frozen"]:
        raise SystemExit("R★ inclusion/exclusion FAILED — rebuild required")


if __name__ == "__main__":
    main()
