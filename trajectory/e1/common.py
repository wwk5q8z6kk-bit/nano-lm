"""Shared E1 instrument: load m0-m4, exact/normalize score, Stage-A verify, utility."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
TRAJ = REPO / "trajectory"
FIELDS = ["cc", "dur", "sev", "med", "alg"]
RE_SUMMARY = re.compile(
    r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$"
)

# Stage A deployment lexicons (gate_absence.py) — includes held terms by design.
MED_LEX = [
    "ibuprofen", "paracetamol", "aspirin", "antacids", "cough syrup",
    "allergy pills", "naproxen", "vitamin c", "zinc tablets", "magnesium",
    "fish oil", "nasal spray", "eye drops", "hydrocortisone cream",
    "loratadine", "cetirizine", "famotidine", "saline rinse",
    "melatonin", "throat lozenges",
]
ALG_LEX = ["penicillin", "peanuts", "pollen", "latex", "shellfish", "sulfa drugs"]
LEX = {"med": MED_LEX, "alg": ALG_LEX}

# Frozen normalize-then-match plurals (E1 construct arm). Commit before scoring.
PLURAL_MAP = {
    "drugs": "drug",
    "lozenges": "lozenge",
    "pills": "pill",
    "tablets": "tablet",
    "allergies": "allergy",
}

DEFAULT_WEIGHTS = dict(alpha=1.0, beta=0.5, gamma=0.3, lam=0.02, kappa=0.05)
SENSITIVITY = [
    dict(alpha=1.0, beta=0.5, gamma=0.3, lam=0.02, kappa=0.05),
    dict(alpha=1.0, beta=1.0, gamma=0.3, lam=0.02, kappa=0.05),
    dict(alpha=1.0, beta=0.5, gamma=0.6, lam=0.02, kappa=0.05),
]


def load_instances() -> List[List[dict]]:
    return [json.loads((TRAJ / f"scribe_eval_m{k}.json").read_text()) for k in range(5)]


def dialogue_of(item: dict) -> str:
    return item["convo"][0]["content"].rsplit("\nSummarize the visit.", 1)[0]


def truth_of(item: dict) -> Dict[str, str]:
    t = item["tuple"]
    return {f: str(t[f]).strip() for f in FIELDS}


def format_summary(pred: Dict[str, str]) -> str:
    return (
        f"CC: {pred['cc']} | DUR: {pred['dur']} | SEV: {pred['sev']} | "
        f"MED: {pred['med']} | ALG: {pred['alg']}"
    )


def parse_summary(text: str) -> Optional[Dict[str, str]]:
    m = RE_SUMMARY.match(text.strip())
    if not m:
        return None
    return dict(zip(FIELDS, [g.strip() for g in m.groups()]))


def normalize_value(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for art in ("a ", "an ", "the "):
        if s.startswith(art):
            s = s[len(art):]
    toks = s.split()
    if toks and toks[-1] in PLURAL_MAP:
        toks[-1] = PLURAL_MAP[toks[-1]]
    elif toks and toks[-1].endswith("s") and len(toks[-1]) > 3 and not toks[-1].endswith("ss"):
        # trivial English plural strip (frozen heuristic)
        toks[-1] = toks[-1][:-1]
    return " ".join(toks)


def patient_text(dialogue: str) -> str:
    return " ".join(
        ln[len("Patient:"):].lower()
        for ln in dialogue.split("\n")
        if ln.startswith("Patient:")
    )


def patient_lines(content: str):
    out, off = [], 0
    for ln in content.splitlines(keepends=True):
        if ln.startswith("Patient: "):
            out.append((ln[9:].rstrip("\n"), off + 9))
        off += len(ln)
    return out


@dataclass
class FieldPred:
    value: str
    start: Optional[int] = None
    end: Optional[int] = None
    text: Optional[str] = None


@dataclass
class ItemPred:
    fields: Dict[str, FieldPred]
    latency_s: float = 0.0
    raw: str = ""
    parsed: bool = True


def pred_from_values(values: Dict[str, str], spans: Optional[Dict[str, Tuple[int, int, str]]] = None,
                     latency_s: float = 0.0) -> ItemPred:
    fields = {}
    for f in FIELDS:
        sp = (spans or {}).get(f)
        if sp:
            fields[f] = FieldPred(values[f], sp[0], sp[1], sp[2])
        else:
            fields[f] = FieldPred(values[f])
    return ItemPred(fields=fields, latency_s=latency_s, raw=format_summary(values), parsed=True)


def score_fields(pred: Dict[str, str], truth: Dict[str, str], mode: str = "exact"):
    """Return per-field labels: correct|omission|hallucination."""
    out = {}
    for f in FIELDS:
        p, t = pred[f], truth[f]
        if mode == "normalize":
            pe, te = normalize_value(p), normalize_value(t)
        else:
            pe, te = p, t
        if pe == te:
            out[f] = "correct"
        elif p == "none" and t != "none":
            out[f] = "omission"
        else:
            out[f] = "hallucination"
    return out


def stage_a_flag(field: str, pred_val: str, dialogue: str) -> bool:
    """True => route to review (flag). Mirrors scribe/gate_absence.py rules."""
    ptext = patient_text(dialogue)
    if pred_val != "none":
        return pred_val.lower() not in ptext
    # absence
    if field in ("med", "alg"):
        return any(term in ptext for term in LEX[field])
    # mandatory fields claimed none
    return True


def apply_verify(item_pred: ItemPred, dialogue: str, verify_on: bool):
    """
    Returns presented dict (field->value or None if abstained),
    n_flagged, n_presented, n_pres_correct_exact needs truth later.
    """
    presented = {}
    flagged = 0
    for f in FIELDS:
        val = item_pred.fields[f].value
        if not verify_on:
            presented[f] = val
            continue
        if stage_a_flag(f, val, dialogue):
            flagged += 1
            presented[f] = None  # abstain / review — not presented
        else:
            presented[f] = val
    return presented, flagged


@dataclass
class InstanceMetrics:
    n_items: int = 0
    n_fields: int = 0
    parse_ok: int = 0
    correct: int = 0
    omission: int = 0
    halluc: int = 0
    held_correct: int = 0
    held_total: int = 0
    seen_correct: int = 0
    seen_total: int = 0
    presented: int = 0
    presented_correct: int = 0
    flagged: int = 0
    liability_presented_bad: int = 0  # halluc that would present without verify
    latencies: List[float] = field(default_factory=list)
    # normalize arm
    correct_norm: int = 0

    def as_rates(self, cost_c: float, weights=None):
        weights = weights or DEFAULT_WEIGHTS
        nf = max(self.n_fields, 1)
        recall = self.correct / nf
        P = (self.presented_correct / self.presented) if self.presented else 0.0
        M = 1.0 - recall
        rho = self.flagged / nf
        L = sorted(self.latencies)[len(self.latencies) // 2] if self.latencies else 0.0
        C = cost_c
        U = (
            weights["alpha"] * P
            - weights["beta"] * M
            - weights["gamma"] * rho
            - weights["lam"] * L
            - weights["kappa"] * C
        )
        held_r = self.held_correct / self.held_total if self.held_total else 0.0
        seen_r = self.seen_correct / self.seen_total if self.seen_total else 0.0
        return {
            "parse": self.parse_ok / max(self.n_items, 1),
            "recall": recall,
            "halluc": self.halluc / nf,
            "omission": self.omission / nf,
            "held_recall": held_r,
            "seen_recall": seen_r,
            "gap_pts": (seen_r - held_r) * 100.0,
            "P": P,
            "M": M,
            "rho": rho,
            "L_p50": L,
            "C": C,
            "U": U,
            "liability_presented_bad": self.liability_presented_bad,
            "correct_norm_rate": self.correct_norm / nf,
            "n_items": self.n_items,
            "n_fields": nf,
        }


def evaluate_method(
    name: str,
    predict_fn: Callable[[dict, str], ItemPred],
    instances: List[List[dict]],
    verify_on: bool,
    cost_c: float,
) -> dict:
    per_inst = []
    all_item_logs = []
    for inst_i, items in enumerate(instances):
        met = InstanceMetrics()
        for idx, it in enumerate(items):
            dia = dialogue_of(it)
            truth = truth_of(it)
            t0 = time.perf_counter()
            try:
                ip = predict_fn(it, f"m{inst_i}/{idx}")
            except Exception as e:
                ip = ItemPred(fields={f: FieldPred("none") for f in FIELDS},
                              latency_s=time.perf_counter() - t0, raw=f"ERROR:{e}", parsed=False)
            if ip.latency_s <= 0:
                ip.latency_s = time.perf_counter() - t0
            met.n_items += 1
            met.latencies.append(ip.latency_s)
            met.n_fields += 5
            if not ip.parsed:
                met.flagged += 5
                all_item_logs.append({
                    "id": f"m{inst_i}/{idx}", "held": it["held_values"],
                    "parsed": False, "raw": ip.raw,
                })
                continue
            met.parse_ok += 1
            pred_vals = {f: ip.fields[f].value for f in FIELDS}
            labels = score_fields(pred_vals, truth, "exact")
            labels_n = score_fields(pred_vals, truth, "normalize")
            bucket = "held" if it["held_values"] else "seen"
            for f in FIELDS:
                lab = labels[f]
                if lab == "correct":
                    met.correct += 1
                    if bucket == "held":
                        met.held_correct += 1
                        met.held_total += 1
                    else:
                        met.seen_correct += 1
                        met.seen_total += 1
                else:
                    if bucket == "held":
                        met.held_total += 1
                    else:
                        met.seen_total += 1
                    if lab == "omission":
                        met.omission += 1
                    else:
                        met.halluc += 1
                        # liability: would present without verify
                        if not stage_a_flag(f, pred_vals[f], dia):
                            met.liability_presented_bad += 1
                if labels_n[f] == "correct":
                    met.correct_norm += 1

            presented, flagged = apply_verify(ip, dia, verify_on)
            met.flagged += flagged
            for f in FIELDS:
                pv = presented[f]
                if pv is None:
                    continue
                met.presented += 1
                if pv == truth[f]:
                    met.presented_correct += 1

            all_item_logs.append({
                "id": f"m{inst_i}/{idx}",
                "held": it["held_values"],
                "pred": pred_vals,
                "truth": truth,
                "labels": labels,
                "spans": {
                    f: None if ip.fields[f].start is None else
                    [ip.fields[f].start, ip.fields[f].end, ip.fields[f].text]
                    for f in FIELDS
                },
                "latency_s": ip.latency_s,
            })
        rates = met.as_rates(cost_c)
        rates["instance"] = f"m{inst_i}"
        per_inst.append(rates)

    def mean_key(k):
        return sum(r[k] for r in per_inst) / len(per_inst)

    summary = {k: mean_key(k) for k in per_inst[0] if k != "instance"}
    summary["U_sensitivity"] = []
    for w in SENSITIVITY:
        Us = []
        for r in per_inst:
            Us.append(
                w["alpha"] * r["P"] - w["beta"] * r["M"] - w["gamma"] * r["rho"]
                - w["lam"] * r["L_p50"] - w["kappa"] * r["C"]
            )
        summary["U_sensitivity"].append({"weights": w, "U_mean": sum(Us) / len(Us)})
    return {
        "method": name,
        "verify_on": verify_on,
        "cost_C": cost_c,
        "summary": summary,
        "per_instance": per_inst,
        "n_item_logs": len(all_item_logs),
        "item_logs": all_item_logs,
    }


def aggregate_decision(results_by_method: Dict[str, dict], m0_name: str) -> dict:
    """Kill/survive/graded on default U, verify-on arms."""
    u = {k: v["summary"]["U"] for k, v in results_by_method.items() if v["verify_on"]}
    if m0_name not in u:
        return {"verdict": "INCOMPLETE", "reason": f"missing M0 {m0_name}", "U": u}
    u0 = u[m0_name]
    non = {k: v for k, v in u.items() if k != m0_name}
    best_non, best_u = max(non.items(), key=lambda kv: kv[1])
    delta = 0.05
    # sensitivity flip check
    sens_flip = False
    for i, wset in enumerate(SENSITIVITY):
        scores = {}
        for name, res in results_by_method.items():
            if not res["verify_on"]:
                continue
            scores[name] = res["summary"]["U_sensitivity"][i]["U_mean"]
        order = sorted(scores, key=scores.get, reverse=True)
        if (order[0] == m0_name) != (best_u < u0 - delta):
            # compare whether non-LM beats m0 under this weight
            non_best = max(v for k, v in scores.items() if k != m0_name)
            default_kill = best_u >= u0 - delta
            this_kill = non_best >= scores[m0_name] - delta
            if this_kill != default_kill:
                sens_flip = True
    if sens_flip:
        verdict = "GRADED"
    elif best_u >= u0 - delta:
        verdict = "KILL"
    else:
        verdict = "SURVIVE"
    return {
        "verdict": verdict,
        "m0": m0_name,
        "U_m0": u0,
        "best_nonlm": best_non,
        "U_best_nonlm": best_u,
        "delta": delta,
        "margin": best_u - u0,
        "sensitivity_flip": sens_flip,
        "U_all": u,
    }
