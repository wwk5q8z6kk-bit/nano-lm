"""Stored-evidence checks only: no model imports, provider calls or screen reads."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("reconcile_p1_comparison", ROOT / "scripts/reconcile_p1_comparison.py")
assert spec and spec.loader
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_corrected_c2_retains_atom_denominators_and_never_ranks_studies():
    result = comparison.reconcile()
    rows = result["studies"]["large_model_c2"]["rows"]
    # The historical defect averaged counts across 128 cases and/or conditioned
    # on successful outputs. Neither 128 nor 122 is the student's denominator.
    assert [row["atoms"] for row in rows] == [149, 150, 146]
    assert rows[0]["assertion_state_correct"] == {"numerator": 109, "denominator": 149, "rate": 109 / 149}
    assert rows[0]["coverage"] == {"numerator": 128, "denominator": 149, "rate": 128 / 149}
    assert rows[0]["exact_gold_span"]["rate"] == 13 / 149
    assert result["studies"]["large_model_c2"]["student_minus_managed_assertion_rate"] == pytest.approx(-0.0551230425)
    assert result["cross_study_ranking_permitted"] is False
    assert result["studies"]["h6_terminal"]["decision"] == "REJECTED"
    assert result["studies"]["h6_terminal"]["one_shot_consumed"] is True


@pytest.mark.parametrize("mutation", ["rate", "denominator", "count"])
def test_inflated_or_relabelled_c2_metrics_are_rejected(mutation):
    _, src = comparison.load_sources()
    data = copy.deepcopy(src["large_model_c2"])
    cell = data["student_a"]["c2_screening"]
    key = {"rate": "assertion_state_correct_rate", "denominator": "assertion_state_correct_eligible", "count": "assertion_state_correct_count"}[mutation]
    cell[key] = {"rate": 1.0, "denominator": 128, "count": 160}[mutation]
    with pytest.raises(comparison.EvidenceError):
        comparison.c2_rows(data)


def test_compact_agreement_is_not_relabelled_exactness():
    studies = comparison.reconcile()["studies"]
    compact = studies["compact_field_filter"]
    assert compact["decision"] == "accepted_all_floors"
    assert len(compact["metrics"]) == 3
    assert "Exact OR overlapping" in compact["denominators"]["agreement_rate"]
    assert "removes rejected quotes" in compact["denominators"]["post_snap_no_match_rate"]
    assert studies["compact_predecessor"]["gates"]["G4_span_replicated"] is False
    assert studies["compact_transfer_curve"]["points"][0]["interpretable"] is False
    groups = {study["comparison_group"] for study in studies.values()}
    assert len(groups) == 6  # Only the two compact span studies share a group.


def test_orx_fresh_cohort_and_joint_metric_denominators_are_preserved():
    study = comparison.reconcile()["studies"]["openresearch_prompt_mechanisms"]
    rows = {row["arm"]: row for row in study["rows"]}
    assert all(row["fresh_slots"] == 384 and row["fresh_instances"] == 24 for row in rows.values())
    assert rows["D"]["grounded_present"] == comparison.count_rate(17, 240)
    assert rows["P1"]["grounded_present"] == comparison.count_rate(3, 240)
    assert rows["D"]["correct_abstention_absent"] == comparison.count_rate(21, 144)
    assert rows["P1"]["correct_abstention_absent"] == comparison.count_rate(24, 144)
    assert rows["D"]["unbound_asserted"] == comparison.count_rate(50, 347)
    assert rows["D"]["legacy_malformed_all_36_instances"] == comparison.count_rate(161, 576)
    assert "without a state-correctness" in study["denominators"]["located_present"]
    assert any("dependence" in limit for limit in study["limitations"])


def test_orx_reconciles_historical_paired_endpoints_without_pooling_old_instances():
    study = comparison.reconcile()["studies"]["openresearch_prompt_mechanisms"]
    tests = {(r["left"], r["right"], r["endpoint"]): r for r in study["paired_endpoint_reconciliation"]}
    for key, expected in {
        ("D", "P1", "grounded"): (15, 1, 0.000518798828125),
        ("D", "P1", "abstention"): (0, 3, 0.25),
        ("V", "P1", "abstention"): (0, 6, 0.03125),
        ("P1", "P0", "commission"): (21, 0, 9.5367431640625e-7),
        ("S", "P1", "abstention"): (0, 2, 0.5),
    }.items():
        row = tests[key]
        assert (row["left_only"], row["right_only"], row["historical_exact_two_sided_mcnemar_p"]) == expected


@pytest.mark.parametrize("mutation", ["duplicate", "mismatched_gold", "wrong_present_count", "wrong_joint_count", "changed_prompt", "wrong_run", "leakage", "clause_assignment"])
def test_orx_rejects_bad_pairing_or_mixed_cohorts(mutation):
    manifest, sources = comparison.load_sources()
    doc = sources["orx_slots_D"]
    if mutation == "duplicate":
        doc["slots"][1] = copy.deepcopy(doc["slots"][0])
    elif mutation == "mismatched_gold":
        doc["slots"][0]["gold_value"] = "different gold value"
    elif mutation == "wrong_present_count":
        doc["slots"][0]["gold_present"] = not doc["slots"][0]["gold_present"]
    elif mutation == "wrong_joint_count":
        doc["suite_aggregate"]["asserted_grounded"] += 1
    elif mutation == "changed_prompt":
        doc["slots"][0]["prompt"] += "Changed prompt"
    elif mutation == "wrong_run":
        doc["run_id"] = "wrong-run-id"
    elif mutation == "leakage":
        doc["leakage_config"]["prompt_question_uses_gold_surface"] = True
    else:
        manifest["orx_capture_audit"]["runs"]["D"]["calibration_config"]["delimit_check"] = False
    with pytest.raises(comparison.EvidenceError):
        comparison.orx_study(sources, manifest["orx_capture_audit"])


@pytest.mark.parametrize("mutation", ["fingerprint", "configuration", "prompt", "metric"])
def test_orx_offline_capture_audit_detects_log_drift(tmp_path, mutation):
    manifest, sources = comparison.load_sources()
    audit = manifest["orx_capture_audit"]
    capture, doc = audit["runs"]["P0"], sources["orx_slots_P0"]
    log = {key: doc[key] for key in capture["matching_aggregate_fields"]}
    log.update({key: copy.deepcopy(capture[key]) for key in ("calibration_condition", "calibration_config")})
    log["per_atom"], log["encounters"] = {}, []
    for row in doc["slots"]:
        rec = {"raw_line": row["raw_line"], "cell": row["joint_cell"], "gold_present": row["gold_present"]}
        log["per_atom"][row["instance_id"] + "/" + row["atom_id"]] = rec.copy()
        log["encounters"].append({"instance_id": row["instance_id"], "per_atom": {row["atom_id"]: rec}, "prompts": {row["atom_id"]: row["prompt"]}, "raw_lines": {row["atom_id"]: row["raw_line"]}})
    if mutation == "configuration":
        log["calibration_config"]["delimit_check"] = True
    elif mutation == "prompt":
        first = log["encounters"][0]["prompts"]
        first[next(iter(first))] += "changed"
    elif mutation == "metric":
        log["per_atom"][next(iter(log["per_atom"]))]["cell"] = "invented_metric"
    payload = json.dumps(log).encode()
    (tmp_path / f"{capture['run_id']}.log").write_bytes(payload)
    if mutation != "fingerprint":
        # Even if a caller re-pins the log bytes, the preserved export must match.
        capture.update(log_bytes=len(payload), log_sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(comparison.EvidenceError):
        comparison.verify_orx_logs(sources, audit, tmp_path)


def test_modified_source_is_detected_even_when_json_is_still_valid(tmp_path):
    evidence = tmp_path / "comparison"
    shutil.copytree(comparison.EVIDENCE, evidence)
    manifest = json.loads((evidence / "manifest.json").read_text())
    source = next(item for item in manifest["sources"] if item["id"] == "compact_field_filter")
    path = evidence / source["local_path"]
    path.write_bytes(path.read_bytes().replace(b"0.8333", b"0.9333"))
    with pytest.raises(comparison.EvidenceError, match="SHA-256 mismatch"):
        comparison.reconcile(evidence)
    # Restore refuses to replace an existing altered source with convenient bytes.
    with pytest.raises(comparison.EvidenceError, match="SHA-256 mismatch"):
        comparison.reconcile(evidence, restore_sources=True)


def test_blob_id_is_checked_independently_from_sha256():
    payload = b'{"valid":true}\n'
    entry = {"id": "test", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "git_blob_oid": "0" * 40}
    with pytest.raises(comparison.EvidenceError, match="Git blob mismatch"):
        comparison.verify_bytes(payload, entry)


def synthetic_h7_result():
    """Tiny invented unit fixture, never persisted as empirical evidence."""
    manifest, _ = comparison.load_sources()
    contract = manifest["h7_contract"]
    metrics = {"joint_exact_accuracy": 0.8, "state_accuracy": 0.9, "decode_failure_items": 0}
    mechanism = {"binding_error_fields": 100, "disjoint_or_speaker_mismatch_fields": 5, "field_collapse_count": 0}
    arms = [{"candidate": {"seed": None, "checkpoint_sha256": contract["control_checkpoint_sha256"]}, "metrics": metrics, "mechanism": mechanism}]
    comparisons = []
    for seed in contract["treatment_seeds"]:
        arms.append({"candidate": {"seed": seed, "checkpoint_sha256": "a" * 64, "training_report_sha256": "b" * 64}, "metrics": metrics.copy(), "mechanism": {**mechanism, "binding_error_fields": 60}})
        comparisons.append({
            "treatment_seed": seed,
            "bars": dict.fromkeys(comparison.H7_BAR_NAMES, True),
            "all_pass_bars": True, "safety_or_state_violation": False,
            "boundary_improvement_bars_met": True,
            "span_exact_on_shared_state_correct": {"denominator": 1000, "control_numerator": 800, "control_rate": 0.8, "treatment_numerator": 900, "treatment_rate": 0.9, "absolute_gain": 0.1},
            "binding_error_fields": {"control": 100, "treatment": 60, "relative_reduction": 0.4},
            "state_accuracy_delta": 0.0,
            "disjoint_or_speaker_mismatch_delta": 0,
        })
    return contract, {
        "schema_version": contract["schema_version"], "experiment_id": contract["experiment_id"],
        "status": "complete", "decision": "PASS", "arms": arms, "comparisons": comparisons,
        "screen": {"manifest_sha256": contract["screen_manifest_sha256"], "records_sha256": contract["screen_records_sha256"], "records": 1000},
        "termination": {"control_scoring_passes": 1, "treatment_scoring_passes": 2, "third_seed_allowed": False, "threshold_sweep_performed": False, "h6_development_opened": False},
    }


def write_result(tmp_path, data):
    path = tmp_path / "synthetic-unit-fixture.json"
    payload = json.dumps(data).encode()
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def test_h7_import_is_hash_pinned_separate_and_does_not_replay(tmp_path):
    contract, data = synthetic_h7_result()
    path, digest = write_result(tmp_path, data)
    result = comparison.h7_result(path, digest, contract)
    assert result["status"] == "STORED_RESULT_IMPORTED"
    assert result["decision"] == "PASS"
    assert result["comparable_to_other_groups"] is False
    # The original three-argument API remains a record-only capture check.
    assert result["scientific_validity"] == "UNVERIFIED_GOLD_ANNOTATIONS"
    assert result["claim_scope"] == "RECORD_ONLY"
    assert "not independently replayed" in result["verification"]
    with pytest.raises(comparison.EvidenceError, match="SHA-256 mismatch"):
        comparison.h7_result(path, "0" * 64, contract)
    with pytest.raises(comparison.EvidenceError, match="supplied together"):
        comparison.h7_result(path, None, contract)


def test_h7_absent_pointer_stays_unimported(monkeypatch, tmp_path):
    manifest, sources = comparison.load_sources()
    manifest.pop("stored_h7_result", None)
    monkeypatch.setattr(comparison, "load_sources", lambda *args, **kwargs: (manifest, sources))
    result = comparison.reconcile(tmp_path)
    assert result["studies"]["h7"]["status"] == "NO_RESULT_IMPORTED"
    assert result["studies"]["h7"]["decision"] is None


def test_h7_manifest_pointer_makes_default_report_reproducible(monkeypatch, tmp_path):
    _, data = synthetic_h7_result()
    path, digest = write_result(tmp_path, data)
    manifest, sources = comparison.load_sources()
    manifest["stored_h7_result"] = {"local_path": path.name, "sha256": digest}
    monkeypatch.setattr(comparison, "load_sources", lambda *args, **kwargs: (manifest, sources))
    result = comparison.reconcile(tmp_path)
    h7 = result["studies"]["h7"]
    assert (h7["decision"], h7["sha256"]) == ("PASS", digest)
    assert h7["input_source"] == {"kind": "manifest", "manifest_key": "stored_h7_result", "local_path": path.name}
    assert "Generated by `python3 scripts/reconcile_p1_comparison.py --format markdown`." in comparison.markdown(result)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(comparison.EvidenceError, match="SHA-256 mismatch"):
        comparison.reconcile(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        comparison.reconcile(tmp_path)


def test_h7_explicit_pair_overrides_manifest_and_records_reproduction_command(monkeypatch, tmp_path):
    _, data = synthetic_h7_result()
    directory = tmp_path / "path with spaces"
    directory.mkdir()
    path, digest = write_result(directory, data)
    manifest, sources = comparison.load_sources()
    # Explicit callers do not need the default capture to be available or valid.
    manifest["stored_h7_result"] = {"local_path": "missing.json", "sha256": "0" * 64}
    monkeypatch.setattr(comparison, "load_sources", lambda *args, **kwargs: (manifest, sources))
    result = comparison.reconcile(tmp_path, h7_path=path, h7_sha256=digest)
    assert result["studies"]["h7"]["input_source"] == {"kind": "explicit", "path": str(path)}
    command = shlex.join(["python3", "scripts/reconcile_p1_comparison.py", "--format", "markdown", "--h7-result", str(path), "--h7-sha256", digest])
    assert f"Generated by `{command}`." in comparison.markdown(result)
    for kwargs in ({"h7_path": path}, {"h7_sha256": digest}):
        with pytest.raises(comparison.EvidenceError, match="supplied together"):
            comparison.reconcile(tmp_path, **kwargs)


@pytest.mark.parametrize("mutation", ["null", "missing_key", "extra_key", "empty_path", "absolute_path", "escape", "symlink_escape", "bad_hash", "hash_type"])
def test_h7_invalid_manifest_pointer_cannot_silently_drop_the_result(tmp_path, mutation):
    contract, data = synthetic_h7_result()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    path, digest = write_result(evidence, data)
    pointer = {"local_path": path.name, "sha256": digest}
    if mutation == "null":
        pointer = None
    elif mutation == "missing_key":
        del pointer["sha256"]
    elif mutation == "extra_key":
        pointer["ignored"] = True
    elif mutation == "empty_path":
        pointer["local_path"] = ""
    elif mutation == "absolute_path":
        pointer["local_path"] = str(path)
    elif mutation == "escape":
        pointer["local_path"] = "../outside.json"
    elif mutation == "symlink_escape":
        outside, _ = write_result(tmp_path, data)
        link = evidence / "external.json"
        link.symlink_to(outside)
        pointer["local_path"] = link.name
    elif mutation == "bad_hash":
        pointer["sha256"] = "not-a-sha256"
    else:
        pointer["sha256"] = 7
    manifest = {"h7_contract": contract, "stored_h7_result": pointer}
    with pytest.raises(comparison.EvidenceError):
        comparison.import_h7_result(manifest, evidence, None, None)


@pytest.mark.parametrize("mixed,decision", [(True, "GRAY"), (False, "REJECT")])
def test_h7_import_preserves_safe_mixed_and_negative_outcomes(tmp_path, mixed, decision):
    contract, data = synthetic_h7_result()
    for index in ([1] if mixed else [0, 1]):
        row = data["comparisons"][index]
        row["span_exact_on_shared_state_correct"].update(
            treatment_numerator=820, treatment_rate=0.82, absolute_gain=0.02
        )
        row["bars"]["shared_state_span_gain_at_least_0_05"] = False
        row["all_pass_bars"] = False
        row["boundary_improvement_bars_met"] = False
    data["decision"] = decision
    path, digest = write_result(tmp_path, data)
    assert comparison.h7_result(path, digest, contract)["decision"] == decision


@pytest.mark.parametrize("mutation", ["incomplete", "wrong_screen", "duplicate_seed", "third_pass", "opened_h6", "wrong_decision", "bad_counts", "fabricated_gain", "nan"])
def test_h7_attempts_or_inconsistent_results_cannot_become_a_completed_result(tmp_path, mutation):
    contract, data = synthetic_h7_result()
    if mutation == "incomplete":
        data["status"] = "running"
    elif mutation == "wrong_screen":
        data["screen"]["records_sha256"] = "c" * 64
    elif mutation == "duplicate_seed":
        data["arms"][2]["candidate"]["seed"] = data["arms"][1]["candidate"]["seed"]
    elif mutation == "third_pass":
        data["termination"]["treatment_scoring_passes"] = 3
    elif mutation == "opened_h6":
        data["termination"]["h6_development_opened"] = True
    elif mutation == "wrong_decision":
        data["decision"] = "REJECT"
    elif mutation == "bad_counts":
        data["comparisons"][0]["span_exact_on_shared_state_correct"]["treatment_numerator"] = 1001
    elif mutation == "fabricated_gain":
        data["comparisons"][0]["binding_error_fields"]["relative_reduction"] = 0.99
    else:
        data["comparisons"][0]["state_accuracy_delta"] = float("nan")
    path, digest = write_result(tmp_path, data)
    with pytest.raises(comparison.EvidenceError):
        comparison.h7_result(path, digest, contract)


def synthetic_h7_quota_result():
    """Invented gold-state-only rows for the quota audit, not measured outputs."""
    contract, data = synthetic_h7_result()
    quota, columns = {}, {}
    for index, field in enumerate(comparison.H7_FIELDS):
        counts = {"supported": 996 - index, "absent": 1, "uncertain": 1, "conflicting": 1, "missing": 1 + index}
        quota.update({f"{field}:{state}": count for state, count in counts.items()})
        columns[field] = [state for state, count in counts.items() for _ in range(count)]
    manifest = {"screen": {"records": contract["records"], "sha256": contract["screen_records_sha256"], "state_field_quota": quota}}
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    contract["screen_manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    data["screen"].update(manifest=manifest, manifest_sha256=contract["screen_manifest_sha256"])
    rows = [{"example_id": f"synthetic-{i}", "fields": [{"field": field, "gold_state": columns[field][i]} for field in comparison.H7_FIELDS]} for i in range(contract["records"])]
    totals = {state: {"total": sum(quota[f"{field}:{state}"] for field in comparison.H7_FIELDS)} for state in comparison.H7_STATES}
    for arm in data["arms"]:
        arm["raw"] = {"by_gold_state": copy.deepcopy(totals), "item_diagnostics": copy.deepcopy(rows)}
    return contract, data


def test_h7_matching_authenticated_quotas_are_necessary_not_scientific_validation(monkeypatch, tmp_path):
    contract, data = synthetic_h7_quota_result()
    path, digest = write_result(tmp_path, data)
    manifest, sources = comparison.load_sources()
    manifest["h7_contract"] = contract
    manifest["stored_h7_result"] = {"local_path": path.name, "sha256": digest}
    monkeypatch.setattr(comparison, "load_sources", lambda *args, **kwargs: (manifest, sources))
    result = comparison.reconcile(tmp_path)
    h7 = result["studies"]["h7"]
    assert h7["scientific_validity"] == "QUOTA_CONSISTENT_ONLY"
    assert h7["claim_scope"] == "QUOTA_CONSISTENT_STORED_RESULT"
    assert h7["decision"] == "PASS"
    assert all(not arm["issues"] for arm in h7["gold_annotation_audit"]["arms"])
    assert "scientific validity remain unestablished" in h7["interpretation"]
    rendered = comparison.markdown(result)
    assert "Scientific validity: QUOTA_CONSISTENT_ONLY" in rendered
    assert "quota agreement alone does not validate these performance estimates" in rendered


@pytest.mark.parametrize("mutation", ["all_missing", "aggregate_only", "rows_only", "field_quota_only", "paired_disagreement"])
def test_h7_gold_quota_mismatches_retain_decision_but_block_performance_claims(tmp_path, mutation):
    contract, data = synthetic_h7_quota_result()
    for arm in data["arms"]:
        raw, rows = arm["raw"], arm["raw"]["item_diagnostics"]
        if mutation == "all_missing":
            raw["by_gold_state"] = {state: {"total": 5000 if state == "missing" else 0} for state in comparison.H7_STATES}
            for row in rows:
                for field in row["fields"]:
                    field["gold_state"] = "missing"
        elif mutation == "aggregate_only":
            raw["by_gold_state"]["supported"]["total"] -= 1
            raw["by_gold_state"]["missing"]["total"] += 1
        elif mutation == "rows_only":
            rows[0]["fields"][0]["gold_state"] = "missing"
        elif mutation == "field_quota_only":
            # Preserve overall state totals; only field-specific quotas expose it.
            rows[0]["fields"][0]["gold_state"] = "missing"
            rows[-1]["fields"][-1]["gold_state"] = "supported"
        elif arm["candidate"]["seed"] is not None:
            # Even matching every marginal quota cannot excuse paired gold drift.
            rows[0]["fields"][0]["gold_state"], rows[-1]["fields"][0]["gold_state"] = rows[-1]["fields"][0]["gold_state"], rows[0]["fields"][0]["gold_state"]
    path, digest = write_result(tmp_path, data)
    h7 = comparison.h7_result(path, digest, contract, audit_gold=True)
    assert h7["decision"] == "PASS"  # Preserve the evaluator record, even a PASS.
    assert h7["scientific_validity"] == "INVALID_GOLD_ANNOTATIONS"
    assert h7["claim_scope"] == "RECORD_ONLY"
    result = comparison.reconcile()
    result["studies"]["h7"] = h7
    rendered = comparison.markdown(result).split("H7: **", 1)[1]
    assert "recorded decision **PASS**" in rendered
    assert "Scientific validity: INVALID_GOLD_ANNOTATIONS" in rendered
    assert "do not support a measured accuracy, mechanism effect, or treatment-rejection claim" in rendered
    assert "H7 performance tables withheld" in rendered
    assert "| Treatment seed |" not in rendered


@pytest.mark.parametrize("missing", ["manifest", "raw", "summary", "rows", "failed_decode"])
def test_h7_missing_gold_audit_evidence_never_becomes_a_valid_result(tmp_path, missing):
    contract, data = synthetic_h7_quota_result()
    if missing == "manifest":
        del data["screen"]["manifest"]
    elif missing == "raw":
        del data["arms"][0]["raw"]
    elif missing == "summary":
        del data["arms"][0]["raw"]["by_gold_state"]
    elif missing == "rows":
        del data["arms"][0]["raw"]["item_diagnostics"]
    else:
        data["arms"][0]["raw"]["item_diagnostics"][0]["fields"] = None
    path, digest = write_result(tmp_path, data)
    h7 = comparison.h7_result(path, digest, contract, audit_gold=True)
    assert h7["decision"] == "PASS"
    assert h7["scientific_validity"] == "UNVERIFIED_GOLD_ANNOTATIONS"
    assert h7["claim_scope"] == "RECORD_ONLY"


def test_h7_aggregate_quota_violation_stays_invalid_when_item_rows_are_unavailable(tmp_path):
    contract, data = synthetic_h7_quota_result()
    raw = data["arms"][0]["raw"]
    raw["by_gold_state"]["missing"]["total"] += 1
    del raw["item_diagnostics"]
    path, digest = write_result(tmp_path, data)
    assert comparison.h7_result(path, digest, contract, audit_gold=True)["scientific_validity"] == "INVALID_GOLD_ANNOTATIONS"


def test_h7_row_quota_violation_stays_invalid_when_aggregates_are_unavailable(tmp_path):
    contract, data = synthetic_h7_quota_result()
    raw = data["arms"][0]["raw"]
    raw["item_diagnostics"][0]["fields"][0]["gold_state"] = "missing"
    del raw["by_gold_state"]
    path, digest = write_result(tmp_path, data)
    assert comparison.h7_result(path, digest, contract, audit_gold=True)["scientific_validity"] == "INVALID_GOLD_ANNOTATIONS"


def test_h7_embedded_quota_manifest_must_itself_match_the_frozen_hash(tmp_path):
    contract, data = synthetic_h7_quota_result()
    data["screen"]["manifest"]["screen"]["state_field_quota"]["allergy:supported"] += 1
    path, digest = write_result(tmp_path, data)
    with pytest.raises(comparison.EvidenceError, match="embedded screen manifest SHA-256 mismatch"):
        comparison.h7_result(path, digest, contract, audit_gold=True)


def test_actual_preserved_h7_result_is_invalid_gold_and_recorded_reject_only():
    # Stored result and embedded manifest only. No screen records or model access.
    result = comparison.reconcile()
    h7 = result["studies"]["h7"]
    assert h7["sha256"] == "9d1b8035b87b8af9419ead92a0dffd66806df1e77720abdf6a1d1deb98951d49"
    assert h7["decision"] == "REJECT"
    assert h7["scientific_validity"] == "INVALID_GOLD_ANNOTATIONS"
    assert h7["claim_scope"] == "RECORD_ONLY"
    audit = h7["gold_annotation_audit"]
    assert audit["expected_by_gold_state"] == {"supported": 3837, "absent": 413, "uncertain": 250, "conflicting": 250, "missing": 250}
    for arm in audit["arms"]:
        assert arm["stored_by_gold_state"] == {"supported": 0, "absent": 0, "uncertain": 0, "conflicting": 0, "missing": 5000}
        assert arm["observed_by_gold_state"] == arm["stored_by_gold_state"]
        assert arm["rows_without_gold_fields"] == 0
    rendered = comparison.markdown(result).split("H7: **", 1)[1]
    assert "recorded decision **REJECT**" in rendered
    assert "Scientific validity: INVALID_GOLD_ANNOTATIONS" in rendered
    assert "| Treatment seed |" not in rendered


def test_checked_in_readable_report_is_reproducible():
    assert (comparison.EVIDENCE / "REPORT.md").read_text() == comparison.markdown(comparison.reconcile())
