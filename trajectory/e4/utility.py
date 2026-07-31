"""U_R★ scorer + decision rule (frozen at AUTHORIZE_E4_BUILDER_AND_EXECUTE)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from trajectory.e1.common import FIELDS, FieldPred, ItemPred, normalize_value, score_fields
from trajectory.e4.methods import COST_C, dialogue_of
from trajectory.e4.verify import apply_verify

E4 = Path(__file__).resolve().parent
RECIPE = json.loads((E4 / "recipe_freeze.json").read_text())
W = RECIPE["U_Rstar"]["weights"]
DELTA = RECIPE["U_Rstar"]["delta"]
M_ASSIGN = RECIPE["M_rubric_preassigned"]

SENSITIVITY = [
    {"name": "default", "w": dict(W)},
    {"name": "high_miss", "w": {**W, "E": 1.0}},
    {"name": "high_review", "w": {**W, "R": 0.6}},
    {"name": "no_maintenance", "w": {**W, "M": 0.0}},
    {"name": "binding_heavy", "w": dict(W), "beta_bind": 0.2},
    {"name": "e1_shaped", "w": {**W, "M": 0.0}},
]


@dataclass
class Metrics:
    n_items: int = 0
    n_fields: int = 0
    correct: int = 0
    omission: int = 0
    halluc: int = 0
    correct_norm: int = 0
    presented: int = 0
    presented_correct: int = 0
    flagged: int = 0
    liability_presented_bad: int = 0
    bind_errors: int = 0
    bind_docs: int = 0
    latencies: List[float] = field(default_factory=list)

    def rates(self, cost_c: float, M: float, beta_bind_weight: float = 0.0):
        nf = max(self.n_fields, 1)
        recall = self.correct / nf
        Q = (self.presented_correct / self.presented) if self.presented else 0.0
        E = 1.0 - recall
        R = self.flagged / nf
        L = sorted(self.latencies)[len(self.latencies) // 2] if self.latencies else 0.0
        C = cost_c
        beta = (self.bind_errors / self.bind_docs) if self.bind_docs else 0.0
        U = Q - W["E"] * E - W["R"] * R - W["L"] * L - W["C"] * C - W["M"] * M
        if beta_bind_weight:
            U -= beta_bind_weight * beta
        return {
            "Q": Q,
            "E": E,
            "R": R,
            "L_p50": L,
            "C": C,
            "M": M,
            "U": U,
            "recall": recall,
            "beta_bind": beta,
            "correct_norm_rate": self.correct_norm / nf,
            "liability_presented_bad": self.liability_presented_bad,
            "n_items": self.n_items,
            "n_fields": nf,
            "omission": self.omission / nf,
            "halluc": self.halluc / nf,
        }


def evaluate_method(
    name: str,
    predict_fn: Callable,
    items: List[dict],
    verify_on: bool,
    cost_c: float,
    M: float,
) -> dict:
    met = Metrics()
    logs = []
    for idx, it in enumerate(items):
        dia = dialogue_of(it)
        truth = {f: str(it["tuple"][f]).strip() for f in FIELDS}
        t0 = time.perf_counter()
        try:
            ip = predict_fn(it, f"eval/{idx}")
        except Exception as e:
            ip = ItemPred(
                fields={f: FieldPred("none") for f in FIELDS},
                latency_s=time.perf_counter() - t0,
                raw=f"ERROR:{e}",
                parsed=False,
            )
        if ip.latency_s <= 0:
            ip.latency_s = time.perf_counter() - t0
        met.n_items += 1
        met.latencies.append(ip.latency_s)
        met.n_fields += 5
        if not ip.parsed:
            met.flagged += 5
            logs.append({"id": f"eval/{idx}", "parsed": False, "raw": ip.raw})
            continue
        pred_vals = {f: ip.fields[f].value for f in FIELDS}
        labels = score_fields(pred_vals, truth, "exact")
        labels_n = score_fields(pred_vals, truth, "normalize")
        for f in FIELDS:
            lab = labels[f]
            if lab == "correct":
                met.correct += 1
            elif lab == "omission":
                met.omission += 1
            else:
                met.halluc += 1
            if labels_n[f] == "correct":
                met.correct_norm += 1

        # binding error on multi-candidate docs
        if it.get("meta", {}).get("multi_candidate"):
            met.bind_docs += 1
            comps = it["meta"].get("competitors", {})
            bad = False
            for f in ("cc", "med", "alg"):
                if pred_vals[f] != "none" and pred_vals[f] in comps.get(f, []) and pred_vals[f] != truth[f]:
                    bad = True
            if bad:
                met.bind_errors += 1

        presented, flagged = apply_verify(ip, dia, verify_on)
        met.flagged += flagged
        for f in FIELDS:
            pv = presented[f]
            if pv is None:
                continue
            met.presented += 1
            if pv == truth[f]:
                met.presented_correct += 1
            elif labels[f] == "hallucination":
                met.liability_presented_bad += 1

        logs.append({"id": f"eval/{idx}", "pred": pred_vals, "truth": truth, "labels": labels})

    summary = met.rates(cost_c, M)
    # sensitivity
    sens = []
    for cell in SENSITIVITY:
        w = cell["w"]
        beta_w = cell.get("beta_bind", 0.0)
        nf = max(met.n_fields, 1)
        recall = met.correct / nf
        Q = summary["Q"]
        E = 1.0 - recall
        R = met.flagged / nf
        L = summary["L_p50"]
        C = cost_c
        beta = summary["beta_bind"]
        U = Q - w["E"] * E - w["R"] * R - w["L"] * L - w["C"] * C - w["M"] * M
        if beta_w:
            U -= beta_w * beta
        sens.append({"name": cell["name"], "U": U})
    summary["U_sensitivity"] = sens
    return {
        "method": name,
        "verify_on": verify_on,
        "cost_C": cost_c,
        "M": M,
        "summary": summary,
        "n_item_logs": len(logs),
        "item_logs": logs,
    }


def decide(results: Dict[str, dict]) -> dict:
    """results keys are method names for verify-on primary arms."""
    classical = [k for k in results if k.startswith("C-M")]
    if "G-ref" not in results or not classical:
        return {"verdict": "VOID", "reason": "missing G-ref or classical"}
    u_class = {k: results[k]["summary"]["U"] for k in classical}
    best_c = max(u_class, key=u_class.get)
    u_c = u_class[best_c]
    u_g = results["G-ref"]["summary"]["U"]

    # sensitivity flip: whether kill/survive changes
    default_kill = u_c >= u_g - DELTA
    sens_flip = False
    for i, cell in enumerate(SENSITIVITY):
        # recompute from stored sensitivity
        scores = {}
        for name, res in results.items():
            scores[name] = res["summary"]["U_sensitivity"][i]["U"]
        c_best = max(scores[k] for k in classical)
        g = scores["G-ref"]
        this_kill = c_best >= g - DELTA
        if this_kill != default_kill:
            sens_flip = True

    if sens_flip or abs(u_g - u_c) <= DELTA:
        # margin inside delta OR flip
        if abs(u_g - u_c) <= DELTA or sens_flip:
            if u_g > u_c + DELTA and sens_flip:
                verdict = "GRADED"
            elif u_g > u_c + DELTA:
                verdict = "SURVIVE"
            elif default_kill and not (u_g > u_c + DELTA):
                verdict = "KILL" if not sens_flip else "GRADED"
            else:
                verdict = "GRADED"
        else:
            verdict = "GRADED"
    elif u_g > u_c + DELTA:
        verdict = "SURVIVE"
    else:
        verdict = "KILL"

    # clarify per prereg
    if sens_flip:
        verdict = "GRADED"
    elif u_g > u_c + DELTA:
        verdict = "SURVIVE"
    elif u_c >= u_g - DELTA:
        verdict = "KILL"
    else:
        verdict = "GRADED"

    return {
        "verdict": verdict,
        "U_star_class": u_c,
        "best_classical": best_c,
        "U_star_gen": u_g,
        "delta": DELTA,
        "margin_gen_minus_class": u_g - u_c,
        "sensitivity_flip": sens_flip,
        "U_all": {**u_class, "G-ref": u_g},
        "rule": "KILL if U_class >= U_gen - delta; SURVIVE if U_gen > U_class + delta and no sens flip; else GRADED",
    }
