#!/usr/bin/env python3
"""EXP-U04 Part 2 & 3: Grounded Verifier 72-Case + 7-Hazard Check & Live Qwen2.5-1.5B on Grounded Menu (C14)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
MAIN_REPO = Path("/Users/mac/Projects/nano-lm")
for p in (str(MAIN_REPO), str(MAIN_REPO / "scripts"), str(WORKTREE_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_e07_scribe_decomposition as e07  # noqa: E402
sys.path.insert(0, str(WORKTREE_ROOT))
from evaluations.v2 import scribe_custody  # noqa: E402
from nano_ai.contract import FIELD_ORDER, NanoInput, normalize_value  # noqa: E402
from nanoscribe.check_note import transcript_source  # noqa: E402
from nanoscribe.compact import LocalCompactPredictor, compact_solver  # noqa: E402
import nanoscribe.grounded as grounded_mod  # noqa: E402
from nanoscribe.grounded import build_grounded_english_menu, grounded_candidate_choices, serialize_grounded_slot  # noqa: E402
from nanoscribe.scribe import scribe  # noqa: E402

assert str(grounded_mod.__file__).startswith(str(WORKTREE_ROOT)), grounded_mod.__file__

QWEN_SNAP = "/Users/mac/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OUT_PATH = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U04_nanoscribe_verifier_qwen" / "results.json"
COHORTS = [
    ("C12_independent_dev_20261023", MAIN_REPO / "artifacts" / "p1_independent_dev_20261023"),
    ("C13_scope_validation_20261023", MAIN_REPO / "artifacts" / "p1_scope_validation_20261023"),
    ("C14_retention_validation_20261026", MAIN_REPO / "artifacts" / "p1_retention_validation_20261026"),
]


def deterministic_menu_baseline(transcript: str) -> str:
    """0-parameter deterministic rule choosing from build_grounded_english_menu (CONTRADICTED > UNCERTAIN > REPORTED/DENIED > MISSING)."""
    source = transcript_source(NanoInput("det-baseline", transcript))
    used_offsets: set[tuple[int, int]] = set()
    slots: list[str] = []
    for field in FIELD_ORDER:
        menu = build_grounded_english_menu(source, field, used_offsets)
        # Rank priority: CONTRADICTED -> UNCERTAIN -> first REPORTED/DENIED -> MISSING
        pick = next((k for k in menu if k.startswith(" CONTRADICTED")), None)
        if pick is None:
            pick = next((k for k in menu if k.startswith(" UNCERTAIN")), None)
        if pick is None:
            pick = next((k for k in menu if k.startswith(" REPORTED") or k.startswith(" DENIED")), None)
        if pick is None:
            pick = " MISSING"
        code, spans = menu[pick]
        for sp in spans:
            used_offsets.add((sp.start, sp.end))
        slots.append(serialize_grounded_slot(source, field, code, spans))
    return "|".join(slots)


def evaluate_predictor_on_c14(
    predict_fn,
    c14_dir: Path,
    *,
    label: str,
) -> dict[str, Any]:
    sources = json.loads((c14_dir / "sources.json").read_text(encoding="utf-8"))
    gold_by_id = {g["case_id"]: g["fields"] for g in json.loads((c14_dir / "gold.json").read_text(encoding="utf-8"))}

    t0 = time.perf_counter()
    asserted_total = 0
    raw_all_match = 0
    raw_asserted_match = 0
    verified_all_match = 0
    verified_asserted_match = 0
    useful_notes = 0
    custody_violations = 0
    n_cases = len(sources)
    per_case_errors: list[dict[str, Any]] = []

    for case in sources:
        cid = case["case_id"]
        transcript = case["transcript"]
        gfields = gold_by_id[cid]
        gold_spans = {
            lbl: scribe_custody.locate_gold_spans(transcript, gfields[lbl].get("spans", ()))
            for lbl in scribe_custody.LABELS
        }
        raw_out = predict_fn(transcript)
        ora = scribe_custody._raw_proposal_matches_gold(transcript, raw_out, gfields, policy="grounded")

        draft = scribe(
            transcript,
            encounter_id=cid,
            solver=compact_solver(lambda _, r=raw_out: r, binder="grounded"),
            context_policy="grounded",
        )
        cust = scribe_custody.audit_claim_custody(transcript, draft)
        if not cust.invariant_satisfied:
            custody_violations += len(cust.violations)
        if draft.get("disposition") == "review" and cust.asserted_count > 0:
            useful_notes += 1

        claims_by_field = {c["field"]: c for c in draft["note"]["claims"]} if draft.get("note") else {}
        from nanoscribe.grounded import parse_grounded_summary
        _, props, _, _ = parse_grounded_summary(raw_out, transcript)

        case_errs = []
        for idx_f, (lbl, field) in enumerate(scribe_custody.LABELS.items()):
            gspec = gfields[lbl]
            is_asserted = gspec["state"] in ("S", "A")
            if is_asserted:
                asserted_total += 1
            prop = props[idx_f]
            # Track 1 raw proposal match
            p_state_ok = prop is not None and prop.state_code == gspec["state"]
            p_val_ok = p_state_ok
            if p_state_ok and gspec["state"] == "S":
                p_norm = normalize_value(prop.spans[0].text) if prop.spans else ""
                g_norms = {normalize_value(s.text) for s in gold_spans[lbl]}
                # Strip leading determiner in comparison if menu offered stripped CC
                p_val_ok = (p_norm in g_norms) or any(p_norm == g.removeprefix("a ").removeprefix("an ").removeprefix("the ") for g in g_norms)
            if p_val_ok:
                raw_all_match += 1
                if is_asserted:
                    raw_asserted_match += 1

            # Track 2 verified claim match
            cl = claims_by_field.get(field.value)
            v_ok = cl is not None and scribe_custody._match_claim_to_gold(cl, gspec, gold_spans[lbl])
            if v_ok:
                verified_all_match += 1
                if is_asserted:
                    verified_asserted_match += 1
            elif is_asserted:
                case_errs.append({"field": lbl, "gold": gspec["state"], "raw": raw_out, "claim": cl})
        if case_errs:
            per_case_errors.append({"case_id": cid, "misses": case_errs[:2]})

    return {
        "label": label,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "cases": n_cases,
        "total_slots": 5 * n_cases,
        "asserted_total": asserted_total,
        "track1_raw_all_match": raw_all_match,
        "track1_raw_asserted_match": raw_asserted_match,
        "track2_verified_all_match": verified_all_match,
        "track2_verified_asserted_match": verified_asserted_match,
        "useful_notes": useful_notes,
        "custody_violations": custody_violations,
        "sample_misses": per_case_errors[:5],
    }


def main() -> int:
    t0 = time.perf_counter()
    # Part 2: 72-case Cohort 2x2 + 7 Adversarial Hazards
    cohorts_res = {}
    agg = {"asserted": 0, "gold_real_grd": 0, "menu_grd": 0, "e2e_grd": 0, "n03_grd": 0, "useful_notes_grd": 0}
    for cname, cdir in COHORTS:
        r = scribe_custody.evaluate_cohort_2x2(cdir)
        cohorts_res[cname] = r
        agg["asserted"] += r["total_asserted_gold_slots"]
        agg["gold_real_grd"] += r["gold_candidates_x_real_verifier"]["grounded_retained_asserted_slots"]
        agg["menu_grd"] += r["real_candidate_menu_x_oracle_adjudication"]["grounded_menu_covers_asserted_slots"]
        agg["e2e_grd"] += r["real_candidate_menu_oracle_x_real_verifier"]["grounded_end2end_retained_asserted_slots"]
        agg["useful_notes_grd"] += r["gold_candidates_x_real_verifier"]["grounded_useful_notes"]
        agg["n03_grd"] += r["n03_custody_violations"]["grounded"]

    hazards = e07.run_adversarial_hazards()
    print(f"[U04-P2] 72-case agg={agg} | hazards_false_admissions={hazards['false_admissions']}/{hazards['total_hazards']}", flush=True)

    # Part 3: Live Qwen2.5-1.5B-Instruct on C14 grounded menu vs 0-param deterministic baseline
    c14_dir = COHORTS[2][1]
    det_res = evaluate_predictor_on_c14(deterministic_menu_baseline, c14_dir, label="Deterministic_0param_MenuRule")
    print(f"[U04-P3] DetBaseline C14: raw={det_res['track1_raw_asserted_match']}/{det_res['asserted_total']} verified={det_res['track2_verified_asserted_match']}/{det_res['asserted_total']} all={det_res['track2_verified_all_match']}/120", flush=True)

    qwen_pred = LocalCompactPredictor(
        QWEN_SNAP,
        decoding="constrained",
        prefix_cache=True,
        candidate_policy="grounded",
    )
    qwen_res = evaluate_predictor_on_c14(qwen_pred, c14_dir, label="Live_Qwen2.5_1.5B_GroundedMenu_PrefixKV")
    rec_stats = {
        "total_forward_passes": sum(r.get("forward_passes", 0) for r in qwen_pred.records.values()),
        "total_coalesced_tokens": sum(r.get("coalesced_zero_entropy_tokens", 0) for r in qwen_pred.records.values()),
        "device": next(iter(qwen_pred.records.values())).get("device") if qwen_pred.records else None,
    }
    qwen_res["inference_stats"] = rec_stats
    print(
        f"[U04-P3] Live Qwen2.5-1.5B Grounded C14 ({qwen_res['elapsed_s']}s on {rec_stats['device']}): "
        f"raw_asserted={qwen_res['track1_raw_asserted_match']}/{qwen_res['asserted_total']} "
        f"verified_asserted={qwen_res['track2_verified_asserted_match']}/{qwen_res['asserted_total']} "
        f"verified_all={qwen_res['track2_verified_all_match']}/120 notes={qwen_res['useful_notes']}/24 "
        f"custody_viol={qwen_res['custody_violations']}",
        flush=True,
    )

    payload = {
        "schema": "nanolm.unified_loop.u04.v1",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "paid_compute_usd": 0.0,
        "part2_72_cases_aggregate": agg,
        "part2_adversarial_hazards": hazards,
        "part2_per_cohort": cohorts_res,
        "part3_c14_deterministic_baseline": det_res,
        "part3_c14_live_qwen_grounded": qwen_res,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[U-04] Wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
