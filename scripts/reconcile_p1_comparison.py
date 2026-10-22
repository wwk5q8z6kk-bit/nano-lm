#!/usr/bin/env python3
"""Verify preserved P1 evidence and summarize separate studies without running models.

Default: JSON to stdout. --format markdown renders a readable report.
--verify-git additionally authenticates every commit:path against the local Git
object database; --restore-sources restores missing copies from those objects.
Neither option changes branches, existing evidence, or experiment state.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/p1_comparison"
H7_BAR_NAMES = (
    "binding_relative_reduction_at_least_0_20",
    "shared_state_span_gain_at_least_0_05",
    "state_noninferiority_minus_0_01",
    "no_disjoint_or_speaker_mismatch_increase",
    "zero_decode_failures",
    "no_field_collapse",
)


class EvidenceError(ValueError):
    """Stored evidence does not match the recorded provenance or contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def parse_json(payload: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise EvidenceError(f"Non-finite JSON value: {value}")

    return json.loads(payload, parse_constant=reject_constant)


def git_blob_oid(payload: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def verify_bytes(payload: bytes, entry: dict[str, Any]) -> None:
    role = entry["id"]
    require(len(payload) == entry["bytes"], f"{role}: byte count mismatch")
    require(hashlib.sha256(payload).hexdigest() == entry["sha256"], f"{role}: SHA-256 mismatch")
    require(git_blob_oid(payload) == entry["git_blob_oid"], f"{role}: Git blob mismatch")


def git_bytes(entry: dict[str, Any], repo: Path) -> bytes:
    require(bool(re.fullmatch(r"[0-9a-f]{40}", entry["source_commit"])), "Full source commit required")
    return subprocess.check_output(
        ["git", "show", f"{entry['source_commit']}:{entry['source_path']}"],
        cwd=repo, stderr=subprocess.PIPE,
    )


def load_sources(
    evidence_dir: Path = EVIDENCE, *, verify_git: bool = False,
    restore_sources: bool = False, repo: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = parse_json((evidence_dir / "manifest.json").read_bytes())
    require(manifest["schema"] == "nano.p1-evidence-comparison-manifest.v1", "Unknown manifest schema")
    sources: dict[str, Any] = {}
    for entry in manifest["sources"]:
        require(entry["id"] not in sources, "Duplicate source ID")
        path = evidence_dir / entry["local_path"]
        require(path.resolve().is_relative_to(evidence_dir.resolve()), "Source copy must stay within evidence directory")
        if not path.exists() and restore_sources:
            payload = git_bytes(entry, repo)
            verify_bytes(payload, entry)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
        payload = path.read_bytes()
        verify_bytes(payload, entry)
        if verify_git:
            original = git_bytes(entry, repo)
            verify_bytes(original, entry)
            require(original == payload, f"{entry['id']}: source commit bytes differ")
        sources[entry["id"]] = parse_json(payload) if path.suffix == ".json" else payload.decode()
    return manifest, sources


def count_rate(numerator: int, denominator: int) -> dict[str, Any]:
    require(type(numerator) is int and type(denominator) is int, "Counts must be integers")
    require(0 <= numerator <= denominator and denominator > 0, "Invalid numerator/denominator")
    return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator}


def c2_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for lane, model in (
        ("student_a", "Qwen/Qwen2.5-32B-Instruct"),
        ("managed_reference", "Qwen/Qwen3-32B-AWQ"),
        ("qwen38_serverless", "Qwen/Qwen3.8-27B"),
    ):
        original = source[lane]["c2_screening"]
        row: dict[str, Any] = {"lane": lane, "model": model, "cases": original["n_cases"], "atoms": original["n_atoms"], "malformed": original["malformed"]}
        for metric in ("assertion_state_correct", "exact_gold_span", "coverage"):
            value = count_rate(original[f"{metric}_count"], original[f"{metric}_eligible"])
            require(value["denominator"] == row["atoms"], f"C2 {lane}: {metric} denominator differs from atoms")
            require(round(value["rate"], 4) == original[f"{metric}_rate"], f"C2 {lane}: {metric} stored rate disagrees with counts")
            row[metric] = value
        rows.append(row)
    return rows


ORX_ARMS = ("P0", "P1", "S", "V", "D", "DV")
ORX_CELLS = {"asserted_grounded", "asserted_bound_wrong", "asserted_unbound", "abstained_correct", "abstained_incorrect"}


def orx_prompt_hash(slots: list[dict[str, Any]]) -> str:
    rows = [[r["instance_id"], r["atom_id"], r["prompt"]] for r in sorted(slots, key=lambda r: (r["instance_id"], r["atom_id"]))]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def orx_located(row: dict[str, Any]) -> bool:
    # Exact semantics of preserved triaxial.py for these quote-based arms.
    def norm(text: str) -> str:
        return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).casefold().split())

    gold = norm(row.get("gold_value") or "")
    return row["gold_present"] and bool(gold) and gold in norm(" ".join(row["parsed_quotes"]))


def verify_orx_logs(src: dict[str, Any], audit: dict[str, Any], directory: Path) -> None:
    """Optionally repeat the capture audit against local logs, without an API call."""
    for arm in ORX_ARMS:
        capture, doc = audit["runs"][arm], src[f"orx_slots_{arm}"]
        raw = (directory / f"{capture['run_id']}.log").read_bytes()
        require(len(raw) == capture["log_bytes"] and hashlib.sha256(raw).hexdigest() == capture["log_sha256"], f"ORX {arm}: local log fingerprint mismatch")
        text = raw.decode()
        log, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
        for key in capture["matching_aggregate_fields"]:
            require(doc[key] == log[key], f"ORX {arm}: log aggregate mismatch: {key}")
        for key in ("calibration_condition", "calibration_config"):
            require(capture[key] == log[key], f"ORX {arm}: log clause configuration mismatch")
        encounters = {(enc["instance_id"], atom): (enc, rec) for enc in log["encounters"] for atom, rec in enc["per_atom"].items()}
        require(len(encounters) == len(log["per_atom"]) == len(doc["slots"]), f"ORX {arm}: log slot count mismatch")
        for row in doc["slots"]:
            key = row["instance_id"], row["atom_id"]
            enc, rec = encounters[key]
            require(row["prompt"] == enc["prompts"][row["atom_id"]] and row["raw_line"] == enc["raw_lines"][row["atom_id"]], f"ORX {arm}: log prompt/raw line mismatch")
            for record in (rec, log["per_atom"]["/".join(key)]):
                require(all(row["joint_cell" if k == "cell" else k] == v for k, v in record.items()), f"ORX {arm}: log per-slot metric mismatch")


def orx_study(src: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    require(audit["schema"] == "nano.p1-orx-capture-audit.v1" and set(audit["runs"]) == set(ORX_ARMS), "ORX capture audit invalid")
    rows, paired = [], {}
    gold_reference = None
    for arm in ORX_ARMS:
        doc, capture = src[f"orx_slots_{arm}"], audit["runs"][arm]
        require(doc["schema"] == "nano.p1.run_slots.v1" and doc["cell"] == f"conf-{arm}", f"ORX {arm}: export identity mismatch")
        require(doc["run_id"] == capture["run_id"] and len(doc["slots"]) == doc["n_slots"] == 576, f"ORX {arm}: run or slot count mismatch")
        require(doc["suite"] == "campaign_v2" and doc["dataset_revision"] == "campaign_v2_multi_20260825", f"ORX {arm}: dataset mismatch")
        require(doc["weights"] == {"mode": "weights", "revision": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306", "weights_path": "Qwen/Qwen2.5-1.5B-Instruct"}, f"ORX {arm}: model mismatch")
        require(doc["condition"] == "C1off_C2off_Qon_QSoff" and doc["leakage_config"] == {
            "parser_raw_value_fallback": False, "prompt_answer_template_gold_value": False,
            "prompt_question_names_concept": True, "prompt_question_uses_gold_surface": False,
        }, f"ORX {arm}: leakage controls mismatch")
        require(capture["calibration_config"] == {
            "delimit_check": arm in ("D", "DV"), "placebo_clauses": int(arm == "P1"),
            "sufficiency_check": arm == "S", "verbatim_check": arm in ("V", "DV"),
        }, f"ORX {arm}: clause assignment mismatch")
        slots = doc["slots"]
        require(orx_prompt_hash(slots) == capture["prompts_sha256"], f"ORX {arm}: prompt identity mismatch")
        require(Counter(r["instance_id"] for r in slots) == {f"i{i}": 16 for i in range(36)}, f"ORX {arm}: instance cohort mismatch")
        keys = {(r["instance_id"], r["atom_id"]) for r in slots}
        require(len(keys) == 576, f"ORX {arm}: duplicate slot identity")
        for r in slots:
            require(r["run_id"] == doc["run_id"] and r["cell"] == doc["cell"], f"ORX {arm}: row identity mismatch")
            require(r["joint_cell"] in ORX_CELLS and type(r["gold_present"]) is bool, f"ORX {arm}: invalid slot classification")
            require(r["unbound_assertion"] == (r["joint_cell"] == "asserted_unbound"), f"ORX {arm}: unbound classification mismatch")
        gold = {(r["instance_id"], r["atom_id"]): (r["encounter_id"], r["gold_present"], r["gold_value"]) for r in slots}
        if gold_reference is not None:
            require(gold == gold_reference, f"ORX {arm}: paired gold cohort mismatch")
        gold_reference = gold
        require(Counter(r["instance_id"] for r in slots if r["gold_present"]) == {f"i{i}": 10 for i in range(36)}, f"ORX {arm}: present/absent denominator mismatch")
        cells = Counter(r["joint_cell"] for r in slots)
        for key in ORX_CELLS:
            require(cells[key] == doc["suite_aggregate"][key], f"ORX {arm}: pooled joint counts mismatch")
        for key in ("malformed", "exact_gold_span", "assertion_state_correct"):
            require(sum(r[key] for r in slots) == doc["suite_aggregate"][key], f"ORX {arm}: pooled {key} mismatch")
        fresh = [r for r in slots if r["instance_id"] in {f"i{i}" for i in range(12, 36)}]
        paired[arm] = {(r["instance_id"], r["atom_id"]): r for r in fresh}
        asserted = [r for r in fresh if r["joint_cell"].startswith("asserted")]
        rows.append({
            "arm": arm, "run_id": doc["run_id"], "source_id": f"orx_slots_{arm}",
            "fresh_slots": len(fresh), "fresh_instances": 24,
            "grounded_present": count_rate(sum(r["joint_cell"] == "asserted_grounded" for r in fresh), 240),
            "located_present": count_rate(sum(orx_located(r) for r in fresh), 240),
            "correct_abstention_absent": count_rate(sum(not r["gold_present"] and r["joint_cell"] == "abstained_correct" for r in fresh), 144),
            "commission_absent": count_rate(sum(not r["gold_present"] and r["joint_cell"].startswith("asserted") for r in fresh), 144),
            "unbound_asserted": count_rate(sum(r["joint_cell"] == "asserted_unbound" for r in fresh), len(asserted)),
            "legacy_malformed_all_36_instances": count_rate(sum(r["malformed"] for r in slots), 576),
        })
    # Reconcile only the historical pre-specified paired endpoints. The pooled
    # slot test is preserved with its independence limitation, not re-promoted.
    paired_checks = []
    for left, right, endpoint in (("D", "P1", "grounded"), ("D", "P1", "located"), ("D", "P1", "abstention"), ("V", "P1", "unbound"), ("V", "P1", "located"), ("V", "P1", "grounded"), ("V", "P1", "abstention"), ("P1", "P0", "commission"), ("S", "P1", "abstention"), ("DV", "D", "unbound")):
        def outcome(row: dict[str, Any]) -> bool:
            if endpoint == "located":
                return orx_located(row)
            if endpoint == "commission":
                return not row["gold_present"] and row["joint_cell"].startswith("asserted")
            return row["joint_cell"] == {"grounded": "asserted_grounded", "abstention": "abstained_correct", "unbound": "asserted_unbound"}[endpoint]

        left_only = sum(outcome(r) and not outcome(paired[right][key]) for key, r in paired[left].items())
        right_only = sum(not outcome(r) and outcome(paired[right][key]) for key, r in paired[left].items())
        n = left_only + right_only
        p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1)) / 2**n)
        paired_checks.append({"left": left, "right": right, "endpoint": endpoint, "left_only": left_only, "right_only": right_only, "historical_exact_two_sided_mcnemar_p": p})
    return {
        "comparison_group": "orx_synthetic_prompt_clauses", "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "scope": "Six fixed-checkpoint prompt variants; fresh i12–i35 only: 24 synthetic instances, 384 slots = 240 gold-present + 144 absent. Historical i0–i11 excluded from inference.",
        "rows": rows, "paired_endpoint_reconciliation": paired_checks,
        "capture_audit": audit,
        "denominators": {
            "grounded_present": "Joint cell asserted_grounded / gold-present slots; bound assertion with exact span AND correct state. Not standalone exact_gold_span.",
            "located_present": "Normalized gold value contained in joined parsed quotes / gold-present slots. Generous selection upper bound, without a state-correctness requirement.",
            "correct_abstention_absent": "Joint cell abstained_correct / gold-absent slots.",
            "unbound_asserted": "Joint cell asserted_unbound / all asserted joint cells; the legacy abstained boolean is not its complement.",
        },
        "historical_verdicts": [
            "D versus P1: CONFIRMED under the stored prereg; grounded 15:1 discordances. Selection/abstention harm was not detected, which is not proof of no harm.",
            "V versus P1: MIGRATION; location/unbound improve, but correct abstention falls (0:6 discordances). Supersedes the earlier unchanged-abstention claim.",
            "P1 versus P0: higher absent-slot commission (21:0 discordances). One content-free clause comparison does not isolate a universal causal effect of prompt length.",
            "S versus P1 abstention cost: NOT CONFIRMED (0:2 discordances). No post-hoc pooled claim that all instructions to abstain reduce abstention.",
            "DV versus D: higher unbound assertions (50:15 discordances), historically CONFIRMED; 73 versus 56 words remains a length confound.",
        ],
        "limitations": [
            "Historical preregs and their slot-level McNemar calculations are preserved, not a new confirmatory claim. Slots repeat within 24 synthetic instances; these p-values do not adjust for that dependence or establish generalization to patients/documents.",
            "V's location and unbound endpoints were separate tests without Bonferroni correction, as its prereg states; no familywise-adjusted confirmation is claimed.",
            "Fresh values were authored after earlier results. Same template family, one model checkpoint and decoding setup; no independent clinical validation or training gain.",
            "The dataset_revision string was retained through the 36-instance expansion. Exact source commits and prompt hashes identify this cohort.",
            "Allowed response labels were reported for 576/576 slots per arm; legacy evaluator-malformed counts are nonzero and measure evidence/state construction defects. Format compliance is not transport or semantic correctness.",
            "D wording closely targets the exact-span metric and reaches only 17/240 grounded present slots. This does not establish delimitation as the sole bottleneck or recover the separate VOID menu experiment.",
        ],
    }


def h7_result(path: Path | None, digest: str | None, contract: dict[str, Any]) -> dict[str, Any]:
    require((path is None) == (digest is None), "--h7-result and --h7-sha256 must be supplied together")
    common = {
        "comparison_group": "h7_fresh_boundary_screen",
        "scope": "Fresh 1000-record/250-world synthetic screen, 5000 fields; control versus two treatment seeds on this screen only. A PASS is a mechanism result, not P1 completion.",
        "comparable_to_other_groups": False,
    }
    if path is None:
        return {**common, "status": "NO_RESULT_IMPORTED", "decision": None}
    payload = path.read_bytes()
    require(bool(re.fullmatch(r"[0-9a-f]{64}", digest or "")), "Expected H7 SHA-256 must be 64 lowercase hex characters")
    require(hashlib.sha256(payload).hexdigest() == digest, "H7 result SHA-256 mismatch")
    result = parse_json(payload)
    for key in ("schema_version", "experiment_id"):
        require(result.get(key) == contract[key], f"H7 {key} mismatch")
    require(result.get("status") == "complete", "H7 result is not complete")
    screen = result["screen"]
    for field, expected in (("manifest_sha256", contract["screen_manifest_sha256"]), ("records_sha256", contract["screen_records_sha256"]), ("records", contract["records"])):
        require(screen.get(field) == expected, f"H7 screen {field} mismatch")
    require(result["termination"] == {
        "control_scoring_passes": 1, "treatment_scoring_passes": 2,
        "third_seed_allowed": False, "threshold_sweep_performed": False,
        "h6_development_opened": False,
    }, "H7 termination contract mismatch")
    arms = result["arms"]
    require(len(arms) == 3, "H7 requires exactly three scored arms")
    control, *treatments = arms
    require(control["candidate"]["seed"] is None and control["candidate"]["checkpoint_sha256"] == contract["control_checkpoint_sha256"], "H7 frozen control mismatch")
    require(sorted(arm["candidate"]["seed"] for arm in treatments) == contract["treatment_seeds"], "H7 treatment seeds mismatch")
    for arm in treatments:
        for key in ("checkpoint_sha256", "training_report_sha256"):
            require(bool(re.fullmatch(r"[0-9a-f]{64}", arm["candidate"].get(key) or "")), f"H7 {key} missing")
    for arm in arms:
        metrics, mechanism = arm["metrics"], arm["mechanism"]
        for key in ("joint_exact_accuracy", "state_accuracy"):
            value = metrics[key]
            require(type(value) in (int, float) and 0 <= value <= 1, f"H7 {key} outside [0,1]")
        for mapping, key in ((metrics, "decode_failure_items"), (mechanism, "binding_error_fields"), (mechanism, "disjoint_or_speaker_mismatch_fields"), (mechanism, "field_collapse_count")):
            require(type(mapping[key]) is int and mapping[key] >= 0, f"H7 {key} must be a nonnegative count")
    comparisons = result["comparisons"]
    require(len(comparisons) == 2, "H7 requires two comparisons")
    require(sorted(row["treatment_seed"] for row in comparisons) == contract["treatment_seeds"], "H7 comparison seeds mismatch")
    for row in comparisons:
        require(set(row["bars"]) == set(H7_BAR_NAMES) and all(type(v) is bool for v in row["bars"].values()), "H7 comparison bars invalid")
        values = [row["bars"][name] for name in H7_BAR_NAMES]
        require(row["all_pass_bars"] is all(values), "H7 all-pass summary inconsistent")
        require(row["boundary_improvement_bars_met"] is all(values[:2]), "H7 improvement summary inconsistent")
        require(row["safety_or_state_violation"] is (not all(values[2:])), "H7 safety summary inconsistent")
        span = row["span_exact_on_shared_state_correct"]
        require(span["denominator"] <= contract["records"] * 5, "H7 shared-state cohort exceeds field count")
        for side in ("control", "treatment"):
            rate = count_rate(span[f"{side}_numerator"], span["denominator"])["rate"]
            require(math.isclose(rate, span[f"{side}_rate"], abs_tol=1e-12), "H7 shared-state span rate inconsistent")
        treatment = next(arm for arm in treatments if arm["candidate"]["seed"] == row["treatment_seed"])
        binding = row["binding_error_fields"]
        require(binding["control"] == control["mechanism"]["binding_error_fields"] and binding["treatment"] == treatment["mechanism"]["binding_error_fields"], "H7 binding counts disagree with arms")
        # The evaluator cannot serialize an infinite reduction; zero/zero is its
        # defined finite edge case. Do not turn it into an improvement.
        require(binding["control"] > 0 or binding["treatment"] == 0, "H7 zero-control binding denominator is invalid")
        reduction = (binding["control"] - binding["treatment"]) / binding["control"] if binding["control"] else 0.0
        gain = span["treatment_rate"] - span["control_rate"]
        state_delta = treatment["metrics"]["state_accuracy"] - control["metrics"]["state_accuracy"]
        safety_delta = treatment["mechanism"]["disjoint_or_speaker_mismatch_fields"] - control["mechanism"]["disjoint_or_speaker_mismatch_fields"]
        for value, expected in ((binding["relative_reduction"], reduction), (span["absolute_gain"], gain), (row["state_accuracy_delta"], state_delta), (row["disjoint_or_speaker_mismatch_delta"], safety_delta)):
            require(math.isclose(value, expected, rel_tol=0, abs_tol=1e-12), "H7 comparison arithmetic inconsistent")
        expected_bars = (
            reduction >= 0.20, gain >= 0.05,
            state_delta >= -0.01 or math.isclose(state_delta, -0.01, rel_tol=0, abs_tol=1e-12),
            safety_delta <= 0, treatment["metrics"]["decode_failure_items"] == 0,
            treatment["mechanism"]["field_collapse_count"] == 0,
        )
        require(values == list(expected_bars), "H7 stored bars disagree with comparison arithmetic")
    decision = "PASS" if all(row["all_pass_bars"] for row in comparisons) else (
        "REJECT" if any(row["safety_or_state_violation"] for row in comparisons) or not any(row["boundary_improvement_bars_met"] for row in comparisons) else "GRAY"
    )
    require(result["decision"] == decision, "H7 decision disagrees with stored bars")
    return {
        **common, "status": "STORED_RESULT_IMPORTED", "decision": decision,
        "sha256": digest, "bytes": len(payload),
        "verification": "Caller-pinned bytes and stored result contract checked. Model outputs, checkpoint bytes, and training reports were not independently replayed or authenticated here.",
        "arms": [{key: arm[key] for key in ("candidate", "metrics", "mechanism")} for arm in arms],
        "comparisons": comparisons,
    }


def reconcile(evidence_dir: Path = EVIDENCE, *, verify_git: bool = False, restore_sources: bool = False, h7_path: Path | None = None, h7_sha256: str | None = None, orx_log_dir: Path | None = None) -> dict[str, Any]:
    manifest, src = load_sources(evidence_dir, verify_git=verify_git, restore_sources=restore_sources)
    if orx_log_dir is not None:
        verify_orx_logs(src, manifest["orx_capture_audit"], orx_log_dir)
    c2 = c2_rows(src["large_model_c2"])
    h6 = src["h6_terminal"]
    h6_rows = []
    for name, gate in h6["gates_uncalibrated_development_n5000"].items():
        if isinstance(gate.get("observed"), str) and "/" in gate["observed"]:
            k, n = map(int, gate["observed"].split("/"))
            h6_rows.append({"metric": name, **count_rate(k, n), "floor_count": gate["floor"], "passed": gate["passed"]})
    return {
        "schema": "nano.p1-evidence-comparison.v1",
        "scope": manifest["scope"],
        "cross_study_ranking_permitted": False,
        "provenance": {"verification": "commit_bytes" if verify_git else "preserved_bytes", "sources": manifest["sources"], "local_orx_logs_verified": orx_log_dir is not None},
        "studies": {
            "compact_field_filter": {
                "source_id": "compact_field_filter", "comparison_group": "compact_two_field_denial_bundle",
                "model": src["compact_field_filter"]["model"], "decision": src["compact_field_filter"]["overall"],
                "metrics": src["compact_field_filter"]["seeds"], "gates": src["compact_field_filter"]["gates"],
                "denominators": {
                    "min_arm_state_accuracy": "Minimum of 17 arm means; 30 fields per arm, two-field medication/allergy task. Not all-field joint correctness.",
                    "post_snap_no_match_rate": "Unrelocated / emitted spans after field filtering. Filtering removes rejected quotes from this denominator.",
                    "agreement_rate": "(Exact OR overlapping gold evidence) / relocated spans. Conditional, not exact-boundary accuracy or coverage.",
                    "disjoint_rate": "Disjoint gold evidence / relocated spans.",
                    "missing_span_rate": "Fields with gold evidence but no proposed span after filtering / fields with gold evidence.",
                },
                "limitations": ["Synthetic dev bundle and field-pair-specific lexicon/filter; real documents unvalidated.", "Three seeds and adaptive mechanism development; no clinical or P1 mastery claim.", "Source summaries retain rounded rates, not complete raw denominators; no pooled accuracy or confidence interval reconstructed.", "Branch-local P3/P4 are study stages, not program capability milestones."],
            },
            "compact_predecessor": {
                "source_id": "compact_predecessor", "comparison_group": "compact_two_field_denial_bundle",
                "decision": src["compact_predecessor"]["overall"], "gates": src["compact_predecessor"]["gates"], "metrics": src["compact_predecessor"]["seeds"],
                "interpretation": "State/grounding passed; span replication failed in two of three seeds. The later study changes augmentation, filtering, and seeds, so the difference is not a paired causal estimate of the filter alone.",
            },
            "compact_transfer_curve": {
                "source_id": "compact_transfer_curve", "comparison_group": "compact_state_only_transfer",
                "status": src["compact_transfer_curve"]["status"], "points": src["compact_transfer_curve"]["points"],
                "denominator": "Mean over 12 external denial-arm means; state labels only. Control-failing points remain uninterpretable.",
                "interpretation": "Single seed per base. Size, family, pretraining and training recipe differences do not identify a causal pretraining or universal minimum-size effect. No span-contract result.",
            },
            "openresearch_prompt_mechanisms": orx_study(src, manifest["orx_capture_audit"]),
            "large_model_c2": {
                "source_id": "large_model_c2", "correction_source_id": "large_model_correction", "comparison_group": "c2_structured_screening",
                "rows": c2,
                "student_minus_managed_assertion_rate": c2[0]["assertion_state_correct"]["rate"] - c2[1]["assertion_state_correct"]["rate"],
                "qlora_gate_unlocked": src["large_model_c2"]["qlora_gate_unlocked"],
                "limitations": ["Same 128-case C2 suite; atom denominators differ (149/150/146), so no paired common-denominator effect or significance claim.", "Corrected from counts-as-rates, a clamped impossible rate, and a hardcoded reference. Use atom counts, not earlier inflated summaries.", "Managed reference is not a frontier capability ceiling; no measured QLoRA adaptation gain.", "Source correction's CI-overlap significance interpretation is not adopted: interval overlap alone is not a paired hypothesis test."],
            },
            "h6_terminal": {
                "source_id": "h6_terminal", "comparison_group": "h6_adaptive_development",
                "decision": h6["decision"], "one_shot_consumed": h6["one_shot_consumed"], "metrics": h6_rows,
                "scope": "Joint field correctness on known adaptive synthetic development: 1000 records, 5000 fields. Separate instrument from compact/C2/H7.",
                "unassessed": h6["staged_stop"],
            },
            "h7": h7_result(h7_path, h7_sha256, manifest["h7_contract"]),
        },
    }


def percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def markdown(report: dict[str, Any]) -> str:
    studies = report["studies"]
    compact = studies["compact_field_filter"]
    lines = ["# Preserved P1 comparison", "", "Generated by `python3 scripts/reconcile_p1_comparison.py --format markdown`. Stored evidence only; no model or screen execution.", "", "**No cross-study leaderboard:** datasets, tasks, selection history and denominators differ. This imports evidence, not executable adapters or a production model.", "", "## Compact generate-and-relocate route", "", f"Recorded decision: **{compact['decision']}**; Qwen2.5-1.5B, augmentation + normalization snap + cross-field filter.", "", "| Seed | Minimum-arm state | No match | Gold agreement | Disjoint | Missing span |", "|---|---:|---:|---:|---:|---:|"]
    for seed, row in compact["metrics"].items():
        vals = [percentage(row[key]) for key in ("min_arm_state_accuracy", "post_snap_no_match_rate", "agreement_rate", "disjoint_rate", "missing_span_rate")]
        lines.append(f"| {seed} | " + " | ".join(vals) + " |")
    lines += ["", *[f"- **{key}:** {value}" for key, value in compact["denominators"].items()], "", *[f"- {value}" for value in compact["limitations"]], "", "Predecessor: span replication failed in two of three seeds (no-match 9.13%, 20.79%, 13.81%); state and grounding passed. Augmentation, filtering and seeds change together, so this is study progression, not a paired filter effect.", "", "State-only transfer screen (one seed/base):"]
    for point in studies["compact_transfer_curve"]["points"]:
        status = "clears bar" if point["clears_bar"] else "below bar"
        if not point["interpretable"]:
            status = "UNINTERPRETABLE: control class failed"
        lines.append(f"- {point['base']}: {percentage(point['held_out_mean'])}; {status}.")
    lines += ["", studies["compact_transfer_curve"]["interpretation"], "", "## Recovered OpenResearch prompt mechanisms", ""]
    orx = studies["openresearch_prompt_mechanisms"]
    lines += [orx["scope"], "", "Qwen2.5-1.5B-Instruct. P0 = base prompt; P1 = content-free clause; S = sufficiency; V = verbatim; D = shortest span; DV = shortest span + verbatim.", "", "| Arm | Grounded / present | Located / present | Correct abstention / absent | Commission / absent | Unbound / asserted |", "|---|---:|---:|---:|---:|---:|"]
    for row in orx["rows"]:
        cells = [f"{row[k]['numerator']}/{row[k]['denominator']} ({percentage(row[k]['rate'])})" for k in ("grounded_present", "located_present", "correct_abstention_absent", "commission_absent", "unbound_asserted")]
        lines.append(f"| {row['arm']} | " + " | ".join(cells) + " |")
    lines += ["", *[f"- **{key}:** {value}" for key, value in orx["denominators"].items()], "", "Recorded decisions and corrections:", "", *[f"- {value}" for value in orx["historical_verdicts"]], "", *[f"- {value}" for value in orx["limitations"]], "", "The sanitized capture audit in `manifest.json` records each run ID, run source commit, log hash/size and full 576-slot metric/prompt agreement with the preserved exports. Raw CLI logs stay local. `--orx-log-dir` can repeat that audit offline; default operation needs only the preserved Git sources. JSON output includes the reconciled paired endpoints and historical slot-level p-values.", "", "## Corrected larger-model C2 screen", "", "| Model | Cases | State correct | Exact gold span | Coverage | Malformed |", "|---|---:|---:|---:|---:|---:|"]
    for row in studies["large_model_c2"]["rows"]:
        cells = [f"{row[k]['numerator']}/{row[k]['denominator']} ({percentage(row[k]['rate'])})" for k in ("assertion_state_correct", "exact_gold_span", "coverage")]
        lines.append(f"| {row['model']} | {row['cases']} | " + " | ".join(cells) + f" | {row['malformed']} |")
    gap_pp = 100 * studies["large_model_c2"]["student_minus_managed_assertion_rate"]
    lines += ["", f"Corrected student-minus-managed assertion gap: **{gap_pp:.2f} pp**, replacing the earlier -32 pp claim. Preserved adaptation gate: LOCKED.", "", *[f"- {value}" for value in studies["large_model_c2"]["limitations"]], "", "## H6 terminal and H7 boundary screen", "", "H6 remains **REJECTED**, one-shot consumed. Joint-correctness gates:", "", "| Gate | Correct | Frozen floor (count) | Passed |", "|---|---:|---:|---|"]
    for row in studies["h6_terminal"]["metrics"]:
        lines.append(f"| {row['metric']} | {row['numerator']}/{row['denominator']} ({percentage(row['rate'])}) | {row['floor_count']} | {row['passed']} |")
    h7 = studies["h7"]
    lines += ["", studies["h6_terminal"]["unassessed"], "", f"H7: **{h7['status']}**" + (f"; recorded decision **{h7['decision']}**." if h7["decision"] else "."), "", h7["scope"]]
    if h7["decision"]:
        lines += ["", f"Result SHA-256: `{h7['sha256']}`. {h7['verification']}", "", "| Treatment seed | Binding error reduction | Shared-state span gain | State delta | Decision bars all pass |", "|---|---:|---:|---:|---|"]
        for row in h7["comparisons"]:
            lines.append(f"| {row['treatment_seed']} | {percentage(row['binding_error_fields']['relative_reduction'])} | {percentage(row['span_exact_on_shared_state_correct']['absolute_gain'])} | {percentage(row['state_accuracy_delta'])} | {row['all_pass_bars']} |")
    lines += ["", "## Exact source provenance", "", "Every source copy is byte-preserved and pinned by full commit, Git blob ID, SHA-256 and size in [manifest.json](manifest.json). `--verify-git` additionally compares it with the local commit object; default verification needs no branch checkout or remote access.", ""]
    for item in report["provenance"]["sources"]:
        lines.append(f"- [{item['id']}]({item['local_path']}): `{item['source_commit']}:{item['source_path']}`; SHA-256 `{item['sha256']}`.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--verify-git", action="store_true")
    parser.add_argument("--restore-sources", action="store_true")
    parser.add_argument("--h7-result", type=Path)
    parser.add_argument("--h7-sha256")
    parser.add_argument("--orx-log-dir", type=Path, help="Optionally verify retained <run-id>.log files against the sanitized ORX capture audit; no API call")
    args = parser.parse_args(argv)
    try:
        report = reconcile(verify_git=args.verify_git, restore_sources=args.restore_sources, h7_path=args.h7_result, h7_sha256=args.h7_sha256, orx_log_dir=args.orx_log_dir)
        print(markdown(report) if args.format == "markdown" else json.dumps(report, indent=2, sort_keys=True, allow_nan=False), end="" if args.format == "markdown" else "\n")
    except (EvidenceError, OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"P1_COMPARISON_ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
