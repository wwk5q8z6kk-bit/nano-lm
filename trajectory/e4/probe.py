#!/usr/bin/env python3
"""Classical probe B* — pre-generative only; writes results_e4_classical_probe.json."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from trajectory.e4.methods import (  # noqa: E402
    DATA,
    RULES,
    cue_hit,
    dialogue_of,
    patientish_text,
    predict_c_m1,
    predict_c_m2,
)
from trajectory.e1.common import FIELDS  # noqa: E402

E4 = Path(__file__).resolve().parent
TRAJ = REPO / "trajectory"
RECIPE = json.loads((E4 / "recipe_freeze.json").read_text())
TAU = RECIPE["probe_tau"]
LEX = json.loads((DATA / "rstar_train_lexicon.json").read_text())
OPEN = ["cc", "med", "alg"]


def verbatim_recoverable(dialogue: str, gold: str) -> bool:
    if gold == "none":
        return True
    p = patientish_text(dialogue)
    return gold.lower() in p


def run_probe(eval_items: list) -> dict:
    n = len(eval_items)
    cue_hits = 0
    cue_slots = 0
    span_ok = 0
    span_total = 0
    bind_err = 0
    bind_docs = 0
    dict_hit = 0
    dict_total = 0

    for idx, it in enumerate(eval_items):
        dia = dialogue_of(it)
        truth = it["tuple"]
        # B1: cue-hit rate of frozen C-M1
        for f in FIELDS:
            cue_slots += 1
            if cue_hit(dia, f):
                cue_hits += 1

        # B2: verbatim-recoverable among open gold
        for f in OPEN:
            g = truth[f]
            if g == "none":
                continue
            span_total += 1
            if verbatim_recoverable(dia, g):
                span_ok += 1

        # B3: binding — run C-M2; wrong competitor
        if it.get("meta", {}).get("multi_candidate"):
            bind_docs += 1
            pred = predict_c_m2(it, f"probe/{idx}")
            comps = it["meta"].get("competitors", {})
            for f in OPEN:
                pv = pred.fields[f].value
                if pv != "none" and pv in comps.get(f, []) and pv != truth[f]:
                    bind_err += 1
                    break

        # B4: train-dict coverage of gold open
        for f in OPEN:
            g = truth[f]
            if g == "none":
                continue
            dict_total += 1
            if g in LEX.get(f, []):
                dict_hit += 1

    cue_rate = cue_hits / max(cue_slots, 1)
    span_rate = span_ok / max(span_total, 1)
    bind_rate = bind_err / max(bind_docs, 1)
    dict_rate = dict_hit / max(dict_total, 1)

    B1 = cue_rate < TAU["cue"]
    B2 = span_rate < TAU["span"]
    B3 = bind_rate >= TAU["bind"]
    B4 = (dict_rate < TAU["dict"]) and (cue_rate < TAU["cue"])
    fired = [name for name, val in [("B1", B1), ("B2", B2), ("B3", B3), ("B4", B4)] if val]
    in_rstar = len(fired) >= 2

    # also record crude C-M1 accuracy for diagnostics (not for inclusion)
    cm1_correct = 0
    total_f = 0
    for idx, it in enumerate(eval_items):
        pred = predict_c_m1(it, f"diag/{idx}")
        for f in FIELDS:
            total_f += 1
            if pred.fields[f].value == it["tuple"][f]:
                cm1_correct += 1

    return {
        "schema": "nano-lm.e4.classical_probe.v1",
        "prereg": "trajectory/PREREG_E4_Rstar_killgate.md",
        "regime": "trajectory/REGIME_P1_where_classical_fails.md",
        "n_eval": n,
        "tau": TAU,
        "rates": {
            "cue_hit": cue_rate,
            "verbatim_span": span_rate,
            "binding_error": bind_rate,
            "train_dict_coverage": dict_rate,
        },
        "predicates": {"B1": B1, "B2": B2, "B3": B3, "B4": B4},
        "fired": fired,
        "in_Rstar": in_rstar,
        "diagnostic_cm1_field_accuracy": cm1_correct / max(total_f, 1),
        "cm1_rules": "trajectory/e4/c_m1_rules.json",
        "lexicon": "trajectory/e4/data/rstar_train_lexicon.json",
        "note": "Probe is pre-generative VOID/rebuild gate only; not a cherry-pick filter.",
    }


def main():
    eval_items = json.loads((DATA / "rstar_eval.json").read_text())
    report = run_probe(eval_items)
    out = TRAJ / "results_e4_classical_probe.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("in_Rstar", "fired", "rates", "predicates")}, indent=2))
    if not report["in_Rstar"]:
        raise SystemExit("PROBE FAIL: in_Rstar=false — STOP (rebuild or end product path)")


if __name__ == "__main__":
    main()
