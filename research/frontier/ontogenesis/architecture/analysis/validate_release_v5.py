"""Read-only V5 release gate; --out exclusively creates one new receipt.

The sixteen V4 command groups and their saved-result checks are reused without
editing V4 source. Eight V5 groups add source reanalysis and deterministic CPU
contracts. The native comparison executes only inside its fourteen-test suite;
its full JSON output must match the preserved lossless gzip and readable view.
This is local engineering verification, not a language benchmark or remote CI.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

# Prevent importing the unchanged V4 helper from creating bytecode by default.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT.parent / "LEDGER.md"
SPEC = importlib.util.spec_from_file_location("nanolm_v4_release_helpers", Path(__file__).with_name("validate_release_v4.py"))
v4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v4)

RESEARCH_CUTOFF = "0afd5629990b4e442fb3aa8f7c700b030d8faede"
FACTORIAL_CUTOFF = "767857e72bfa51e3fa6f32d047507aabef077fde"
PRIOR_PR_HEAD = "0a1d11f19b91e3d3efb2ab5cad4cbad617a56971"
LATE_CUTOFF = "c597d5fefdd378c3e5c66a51e7690cdb90853287"
INTEGRATION_BASE = LATE_CUTOFF
CORE_MANIFEST_SHA256 = "d99569a08eb00ae9264f5575746c99724d6fc1f5d8b84a40347ede040507a804"
CORE_LEDGER_PREFIX_SHA256 = "1a4f2750d855412afec1c2b47335948a4e9b29b0b0bd632317700509a052cdf0"
CORE_LEDGER_PREFIX_BYTES = 66297
NATIVE_PROTOCOL_SHA = "4cd9bdabedd2b507663f5d5d368dab34c037c2b078b386d872d9238b23f18425"
CACHE_FINAL_SOURCES = {
    "prototype/causal_ngram_cache.py": "8836cf8ab645628499a6b438963d798c6c064cf832bb67df8a42a8ac32325732",
    "prototype/test_causal_ngram_cache.py": "b22f37dc4fc9a9355683eec6d30f7b5a9471bc4b2a89a0178e2e3174c0abd23f",
}
TIMEOUT_SECONDS = 60
CACHE_REQUIRED = {
    "exhaustive_independent_oracle_and_positive_normalization",
    "sequential_prefix_recompute_and_score_before_observe",
    "query_target_and_future_suffix_noninterference_full_vectors",
    "explicit_crop_document_reset_and_no_precrop_context",
    "masked_loss_still_advances_observed_context",
    "immutable_causal_snapshot_invalid_input_is_atomic",
    "configuration_and_context_validation",
    "proper_native_mixture_zero_weight_and_no_match_exact",
}

# Explicit stdout contract, independent of prints inside an individual test.
# The unmodified suite currently also prints the report during its CLI I/O
# test; that incidental output is suppressed here, not treated as a defect.
NATIVE_SUITE_ADAPTER = r"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path("experiments").resolve()))
with redirect_stdout(io.StringIO()):
    from test_native_update_comparison import NativeUpdateComparisonTests
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeUpdateComparisonTests)
    if suite.countTestCases() != 14:
        print("Expected exactly 14 native comparison tests", file=sys.stderr)
        raise SystemExit(2)
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
if (result.testsRun != 14 or not result.wasSuccessful() or result.skipped
        or result.expectedFailures or result.unexpectedSuccesses):
    raise SystemExit(1)
# setUpClass already executed the comparison; reuse its cached report.
payload = json.dumps(NativeUpdateComparisonTests.report, indent=2, allow_nan=False)
sys.stdout.write(payload + "\n")
"""


def unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def parse_json(raw):
    data = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(data, dict):
        raise ValueError("Expected one JSON object")
    v4.canonical(data)  # Reject nonfinite values.
    return data


def unittest_receipt(stderr, expected):
    matches = re.findall(r"^Ran (\d+) tests? in [0-9.]+s$", stderr, re.MULTILINE)
    count = int(matches[0]) if len(matches) == 1 else 0
    ok = count == expected and stderr.rstrip().splitlines()[-1:] == ["OK"]
    return {"status": "PASS" if ok else "FAIL", "tests_run": count,
            "passed": count if ok else 0,
            "format": "Exact unittest count and terminal OK; skips are not accepted"}


def legacy_commands(baseline_ledger):
    return [
        ("package_regressions", ["checks/test_research_package.py"]),
        ("mathematical_examples", ["checks/repo_lightweight_checks.py"]),
        ("numpy_examples", ["checks/formal_sanity.py"]),
        ("transport_anchors", ["checks/transport_anchor_pilot.py"]),
        ("native_forward_reference", ["prototype/test_native_reference.py"]),
        ("repository_reanalysis", ["analysis/reconcile_repository_evidence.py"]),
        ("package_validator", ["checks/validate_research_package.py", "--baseline-ledger", str(baseline_ledger.resolve())]),
        ("native_training_contracts", ["prototype/test_trainable_native.py"]),
        ("correction_contracts", ["experiments/test_correction_protocol.py"]),
        ("margin_contracts", ["checks/test_margin_durability.py"]),
        ("native_bridge_contracts", ["experiments/test_native_correction_smoke.py"]),
        ("independent_correction_oracle", ["checks/review_correction_oracle.py"]),
        ("independent_native_bridge_replay", ["checks/review_native_bridge_replay.py"]),
        ("margin_reanalysis", ["analysis/reconcile_margin_results.py", "--source-dir",
            "evidence/upstream_v4/research/frontier/ontogenesis", "--prior-results",
            "evidence/upstream/results/correction_interference/constrained.json"]),
        ("margin_diagnostics", ["analysis/margin_durability.py"]),
        ("native_bridge_smoke", ["experiments/native_correction_smoke.py"]),
    ]


def run_legacy(name, arguments):
    if name != "package_validator":
        return v4.run_command(name, arguments)
    # Explicit runtime adaptation of V5 inventory expectations; V4 file and
    # all other V4 predicates stay unchanged. Restore even if execution fails.
    original = v4.EXPECTATIONS
    v4.EXPECTATIONS = {**original, "package_validator": {
        **original["package_validator"], "events_checked": 46, "decisions_checked": 39,
        "full_baseline_append_only": "PASS"}}
    try:
        return v4.run_command(name, arguments)
    finally:
        v4.EXPECTATIONS = original


def native_saved_comparison(actual):
    compressed_path = ROOT / "results/native_update_comparison_v5.json.gz"
    summary_path = ROOT / "results/native_update_comparison_summary_v5.json"
    compressed = compressed_path.read_bytes()
    raw = gzip.decompress(compressed)
    saved = parse_json(raw)
    summary_raw = summary_path.read_bytes()
    summary = parse_json(summary_raw)
    full = summary["full_receipt"]
    expected_top = set(saved) - {"runs"}
    checks = {
        "parsed_test_output_equals_full_gzip": actual == saved,
        "summary_has_complete_top_level_projection": set(summary) == expected_top | {"full_receipt", "per_seed"},
        "summary_top_level_values_equal_full": all(summary.get(k) == saved[k] for k in expected_top),
        "gzip_filename_matches_summary": full["path"] == compressed_path.name,
        "gzip_sha256_matches_summary": full["sha256_gzip"] == v4.digest(compressed),
        "uncompressed_sha256_matches_summary": full["sha256_uncompressed"] == v4.digest(raw),
        "compressed_bytes_match_summary": full["compressed_bytes"] == len(compressed),
        "uncompressed_bytes_match_summary": full["uncompressed_bytes"] == len(raw),
        "full_gzip_has_three_seeds": [r["seed"] for r in saved["runs"]] == [9201, 9202, 9203],
    }
    required_fields = {"branch_checkpoint", "final_checkpoint", "counts", "correction_metrics",
                       "post_prefix_new_fact_acquisition", "memory_scope"}
    views = summary["per_seed"]
    view_ok = [r["seed"] for r in views] == [r["seed"] for r in saved["runs"]]
    for view, source in zip(views, saved["runs"]):
        view_ok = view_ok and set(view) == {"seed", "arms"} and set(view["arms"]) == set(source["arms"])
        for arm, projection in view["arms"].items():
            target = source["arms"].get(arm, {})
            view_ok = view_ok and required_fields <= set(projection) <= set(target)
            view_ok = view_ok and all(target.get(key) == value for key, value in projection.items())
    checks["summary_per_seed_views_equal_full_subsets"] = view_ok and len(views) == 3
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "full_receipt": str(compressed_path.relative_to(ROOT)),
              "full_gzip_sha256": v4.digest(compressed), "full_uncompressed_sha256": v4.digest(raw),
              "full_uncompressed_bytes": len(raw), "canonical_parsed_sha256": v4.digest(v4.canonical(actual)),
              "summary_receipt": str(summary_path.relative_to(ROOT)),
              "summary_sha256": v4.digest(summary_raw),
              "scope": "Exact parsed equality against lossless saved JSON, complete top-level summary projection and every per-seed view field. No separate comparison experiment is executed."}
    return report


def v5_semantic_checks(name, data):
    if name == "optimizer_contracts":
        return {"eight_tests_passed": data.get("status") == "PASS" and data.get("tests_run") == data.get("passed") == 8}
    if name == "optimizer_reanalysis":
        rules = v4.field(data, "factorial.registered_outcomes")
        arms = v4.field(data, "correction.arms") or {}
        streams = v4.field(data, "correction.exogenous_streams") or []
        return {
            "fixed_research_cutoff": v4.field(data, "provenance.cutoffs.correction") == RESEARCH_CUTOFF,
            "fixed_factorial_cutoff": v4.field(data, "provenance.cutoffs.factorial") == FACTORIAL_CUTOFF,
            "fifty_three_inputs_verified": data.get("verified_input_count") == 53,
            "literal_finite_verdicts": rules == {"seed_count": 5,
                "counts": {"F1": 5, "F2": 5, "F3": 0, "F4": 5, "F5_pass": 0, "F5_refutation": 0, "F6": 0},
                "verdicts": {"F1": "holds", "F2": "holds", "F3": "fails", "F4": "holds", "F5": "partial", "F6": "fails"}},
            "three_complete_correction_arms": set(arms) == {"adam_200", "adam_stop", "gd_stop"}
                and all([r["seed"] for r in row["per_seed"]] == [8001, 8002, 8003]
                        and row["O1_collateral_le_085_seed_count"] == 3 for row in arms.values()),
            "exogenous_streams_only": [r.get("seed") for r in streams] == [8001, 8002, 8003]
                and all(r.get("correction_event_count") == 624 for r in streams),
        }
    if name == "geometry_contracts":
        return {"seven_tests_passed": data.get("status") == "PASS"
                and data.get("tests_run") == data.get("passed") == 7
                and data.get("failures") == data.get("errors") == 0}
    if name == "geometry_diagnostics":
        return {
            "target_only_cost_four": v4.field(data, "target_only_minimum_l1.value_exact") == "4",
            "full_minimizer_segment": v4.field(data, "target_only_minimum_l1.all_minimizers") == "(x,x+3), -2 <= x <= 2",
            "tie_not_universal_argmax_error": v4.field(data, "target_only_minimum_l1.tie_is_argmax_error_independent_of_tie_breaking") is False,
            "positive_margin_tradeoff": data.get("tradeoff_formula") == "minimum L1 = 4 + (10/3)*mu for mu >= 0",
            "strict_infimum_not_attained": data.get("strict_only_problem") == {"infimum_exact": "4", "minimum_attained": False},
        }
    if name == "causal_cache_contracts":
        rows = data.get("checks", [])
        names = [r.get("check") for r in rows]
        return {
            "reviewed_final_source_hashes": all(v4.digest((ROOT / path).read_bytes()) == sha
                                               for path, sha in CACHE_FINAL_SOURCES.items()),
            "synthetic_scope": data.get("schema") == "causal-cache-fixture-checks-v1"
                and data.get("language_model_evaluation") is False,
            "all_grouped_checks_pass": data.get("passed") is True and all(r.get("passed") is True for r in rows),
            "eight_required_groups_present_once": CACHE_REQUIRED <= set(names) and len(names) == len(set(names)),
            "independent_oracle_1093_prefixes": any(r.get("check") == "exhaustive_independent_oracle_and_positive_normalization"
                and r.get("prefixes") == 1093 for r in rows),
            "forty_future_suffix_variants": any(r.get("check") == "query_target_and_future_suffix_noninterference_full_vectors"
                and r.get("suffix_variants") == 40 for r in rows),
        }
    if name == "store_reanalysis":
        rows = v4.field(data, "store_S.rows") or []
        return {
            "fixed_cutoff": data.get("research_cutoff") == RESEARCH_CUTOFF,
            "ten_inputs_verified": data.get("verified_source_files") == 10,
            "no_language_model_run": data.get("language_model_evaluation") is False,
            "five_domains_arithmetic_agrees": [r.get("domain") for r in rows] == ["web", "textbook", "math", "code", "mix"]
                and all(r.get("gain_agrees") is True for r in rows),
            "primary_rows_preserved": v4.field(data, "store_S.original_primary_rows_preserved") is True,
            "crop_boundary_counterexample": v4.field(data, "source_crop_boundary_counterexample.violation_reproduced") is True
                and v4.field(data, "source_crop_boundary_counterexample.crop_local_repair_agrees") is True,
            "old_cache_identity_counterexample": v4.field(data, "source_cache_identity_counterexample.old_identity_bug_reproduced") is True
                and v4.field(data, "source_cache_identity_counterexample.repaired_equals_causal_old_path") is True,
        }
    if name == "late_store_cache_reanalysis":
        results = data.get("results", {})
        functions = data.get("unchanged_model_and_evaluation_function_asts", {})
        parameter_rows = data.get("source_configuration_parameter_counts", {})
        return {
            "bounded_core_and_late_cutoffs": data.get("schema") == "store-cache-late-reanalysis-v1"
                and data.get("core_cutoff") == RESEARCH_CUTOFF and data.get("late_cutoff") == LATE_CUTOFF,
            "five_late_ten_prior_files_fifteen_rows": data.get("verified_late_source_files") == 5
                and data.get("verified_prior_source_files") == 10 and data.get("late_rows_reconciled") == 15,
            "source_only_no_language_run": data.get("language_model_evaluation") is False,
            "all_three_results_five_domains_arithmetic_agrees": set(results) == {
                "cache-exchange-S.json", "cache-exchange-M.json", "store-exchange-M.json"}
                and all([r.get("domain") for r in result["rows"]] == ["web", "textbook", "math", "code", "mix"]
                        and all(r.get("gain_arithmetic_agrees") is True for r in result["rows"])
                        for result in results.values()),
            "eight_model_and_evaluation_asts_unchanged": set(functions) == {
                "init", "rms", "rope_tables", "apply_rope", "forward", "count", "loss_fn", "evaluate"}
                and all(value is True for value in functions.values()),
            "two_exchange_script_blobs_unchanged": data.get("unchanged_exchange_script_blobs") == 2,
            "three_source_derived_parameter_counts": {
                key: row.get("shape_derived_parameters") for key, row in parameter_rows.items()
                } == {"S": 14292224, "M": 40379904, "L": 100684032},
            "cache_40M_numeric_stop_keeps_qualifications":
                v4.field(data, "registered_cache_40M_gate.applicable_result") == "cache-exchange-M.json"
                and v4.field(data, "registered_cache_40M_gate.scale_parameters") == 40379904
                and v4.field(data, "registered_cache_40M_gate.all_four_domain_gains_below_0_01") is True
                and v4.field(data, "registered_cache_40M_gate.numeric_branch") == "close_explicit_copy_line_at_tested_scale"
                and v4.field(data, "registered_cache_40M_gate.source_controls_fully_satisfied") is False,
            "store_101M_gate_remains_pending": v4.field(data, "registered_store_101M_gate.required_size") == "L"
                and v4.field(data, "registered_store_101M_gate.required_scale_parameters") == 100684032
                and v4.field(data, "registered_store_101M_gate.new_result_size") == "M"
                and v4.field(data, "registered_store_101M_gate.decision") == "pending_registered_L_result",
            "no_repaired_sampler_training_result": v4.field(data, "training_sampler.no_repaired_sampler_training_result_in_this_addendum") is True,
        }
    if name == "native_update_contracts":
        return {
            "experiment_identity": data.get("experiment_id") == "native-update-allocation-v1",
            "protocol_hash_matches": data.get("protocol_sha256") == NATIVE_PROTOCOL_SHA,
            "three_paired_seeds_four_arms": [r["seed"] for r in data.get("runs", [])] == [9201, 9202, 9203]
                and all(set(r["arms"]) == {"current_only", "replay_matched_total", "replay_additive", "frozen"}
                        for r in data.get("runs", [])),
            "actual_training_accounting": data.get("total_executed") == {
                "common_pretraining_optimizer_steps": 120, "post_branch_optimizer_steps": 756,
                "common_pretraining_scored_targets": 480, "post_branch_scored_targets": 756},
        }
    raise ValueError(f"Unknown V5 command group: {name}")


def run_v5_command(name, arguments):
    command = [sys.executable, "-B", *arguments]
    start = time.perf_counter()
    receipt = {"command": command, "timeout_seconds": TIMEOUT_SECONDS, "status": "FAIL", "returncode": None}
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=TIMEOUT_SECONDS,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONOPTIMIZE": "0"})
        receipt.update(returncode=result.returncode, stderr=result.stderr,
                       stdout_bytes=len(result.stdout.encode()), stdout_sha256=v4.digest(result.stdout.encode()))
        data = unittest_receipt(result.stderr, 8) if name == "optimizer_contracts" else parse_json(result.stdout)
        checks = v5_semantic_checks(name, data)
        if name == "optimizer_contracts":
            checks["stdout_empty"] = not result.stdout.strip()
        if name == "geometry_contracts":
            checks["stderr_seven_tests_ok"] = unittest_receipt(result.stderr, 7)["status"] == "PASS"
        if name == "native_update_contracts":
            receipt["stdout_contract"] = (
                "Validator-owned adapter suppresses incidental suite stdout, retains unittest stderr, "
                "and emits the cached class report once after fourteen successful tests with no skips or expected failures.")
            test_summary = unittest_receipt(result.stderr, 14)
            checks["fourteen_tests_ok"] = test_summary["status"] == "PASS"
            comparison = native_saved_comparison(data)
            checks["lossless_saved_result_and_summary_match"] = comparison["status"] == "PASS"
            receipt["saved_result_comparison"] = comparison
            receipt["test_summary"] = test_summary
            receipt["full_output_canonical_sha256"] = v4.digest(v4.canonical(data))
            receipt["output_compaction"] = (
                "Full parsed output was compared before removing runs from this combined receipt. "
                "The saved gzip represents that complete output only when saved_result_comparison is PASS; "
                "any mismatch fails the gate. No separate full comparison command ran.")
            data = {key: data[key] for key in ("schema_version", "experiment_id", "scope", "protocol_sha256",
                "environment", "source_sha256", "parameter_inventory", "descriptive_seed_summary",
                "paired_seed_differences", "total_executed", "claim_boundaries")}
            data["seeds"] = [9201, 9202, 9203]
        receipt.update(output=data, semantic_checks=checks)
        if result.returncode == 0 and all(checks.values()):
            receipt["status"] = "PASS"
    except subprocess.TimeoutExpired as exc:
        receipt["error"] = f"Command exceeded {TIMEOUT_SECONDS} seconds"
        raw = exc.stdout or b""
        raw = raw.encode() if isinstance(raw, str) else raw
        receipt.update(stdout_bytes=len(raw), stdout_sha256=v4.digest(raw))
        receipt["stderr"] = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError) as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
    receipt["elapsed_seconds_diagnostic_only"] = round(time.perf_counter() - start, 4)
    return receipt


def compare_new_results(receipts, issues):
    report = {}
    for name, filename in [
        ("optimizer_reanalysis", "optimizer_factorial_reanalysis_v5.json"),
        ("geometry_diagnostics", "update_feasibility_geometry_v5.json"),
        ("store_reanalysis", "store_cache_reanalysis_v5.json"),
        ("causal_cache_contracts", "causal_cache_checks_v5_final.json"),
        ("late_store_cache_reanalysis", "store_cache_late_reanalysis_v5.json"),
    ]:
        row = {"command": name, "artifact": "results/" + filename, "status": "FAIL"}
        try:
            raw = (ROOT / "results" / filename).read_bytes()
            same = receipts[name].get("output") == parse_json(raw)
            row.update(status="PASS" if same else "FAIL", artifact_sha256=v4.digest(raw),
                       equal_to_saved_result=same, comparison_scope="Exact parsed JSON equality")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["error"] = str(exc)
        if row["status"] != "PASS":
            issues.append(f"Saved V5 result differs or is unreadable: {filename}")
        report[filename] = row
    return report


def verify_sources_v5(issues, *, addendum=False):
    expected_count = 5 if addendum else 85
    filename = "evidence/source_manifest_v5_addendum.json" if addendum else "evidence/source_manifest_v5.json"
    report = {"manifest": filename, "expected_file_count": expected_count, "files": [], "status": "FAIL"}
    try:
        raw = (ROOT / filename).read_bytes()
        manifest = parse_json(raw)
        entries = manifest["files"]
        paths = [row["local_path"] for row in entries]
        if addendum:
            meta_ok = (manifest.get("review_core_cutoff") == RESEARCH_CUTOFF
                       and manifest.get("late_cutoff") == LATE_CUTOFF)
        else:
            meta_ok = (manifest.get("research_cutoff") == RESEARCH_CUTOFF
                       and manifest.get("factorial_cutoff") == FACTORIAL_CUTOFF
                       and manifest.get("prior_pr_head") == PRIOR_PR_HEAD
                       and v4.digest(raw) == CORE_MANIFEST_SHA256)
        count_ok = len(entries) == len(set(paths)) == expected_count
        report.update(manifest_sha256=v4.digest(raw), observed_file_count=len(entries),
                      fixed_cutoffs_match=meta_ok, distinct_entry_count_matches=count_ok)
        for entry in entries:
            row = {"path": entry["local_path"], "source_commit": entry["source_commit"],
                   "matches_source_manifest": False}
            try:
                relative = Path(entry["local_path"])
                path = ROOT / relative
                if relative.is_absolute() or not path.resolve().is_relative_to(ROOT):
                    raise ValueError("Source path escapes package")
                if not re.fullmatch(r"[0-9a-f]{40}", entry["source_commit"]):
                    raise ValueError("Source commit must be a full immutable SHA")
                if addendum and entry["source_commit"] != LATE_CUTOFF:
                    raise ValueError("Addendum source does not match its fixed late cutoff")
                data = path.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                same = v4.digest(data) == entry["sha256"] and blob == entry["git_blob_sha"] and len(data) == entry["bytes"]
                row.update(sha256=v4.digest(data), git_blob_sha=blob, bytes=len(data), matches_source_manifest=same)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row["error"] = str(exc)
            report["files"].append(row)
        if meta_ok and count_ok and all(row["matches_source_manifest"] for row in report["files"]):
            report["status"] = "PASS"
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["error"] = str(exc)
    if report["status"] != "PASS":
        issues.append(f"{filename} failed cutoff/count/byte/hash verification")
    return report


def inventory_at(root):
    return {path.relative_to(root).as_posix(): v4.digest(path.read_bytes())
            for path in sorted(root.rglob("*")) if path.is_file()}


def ledger_inventory(baseline, issues):
    report = {"status": "FAIL"}
    try:
        human_ids = re.findall(r"^\|\s*(D-\d{3,6})\s*\|", (ROOT / "02_DECISION_LEDGER.md").read_text(), re.MULTILINE)
        old_rows = [json.loads(row, object_pairs_hook=unique_object) for row in baseline.read_text().splitlines()]
        raw_lines = (ROOT / "ledger_events.jsonl").read_bytes().splitlines(keepends=True)
        rows = [json.loads(row, object_pairs_hook=unique_object) for row in raw_lines]
        additions = rows[len(old_rows):]
        expected_new = {f"D-{i:03d}" for i in range(55, 63)}
        expected_additions = Counter({key: 1 for key in expected_new | {"D-054"}})
        expected_additions["D-058"] += 1
        counts = Counter(row["decision_id"] for row in additions)
        earlier_d054 = [row for row in old_rows if row["decision_id"] == "D-054"]
        revised_d054 = [row for row in additions if row["decision_id"] == "D-054"]
        core_d058 = [row for row in rows[:45] if row["decision_id"] == "D-058"]
        core_prefix = b"".join(raw_lines[:45])
        checks = {
            "sixty_two_distinct_human_rows": len(human_ids) == len(set(human_ids)) == 62,
            "complete_human_identifier_range": set(human_ids) == {f"D-{i:03d}" for i in range(1, 63)},
            "baseline_has_thirty_six_events": len(old_rows) == 36,
            "forty_six_events_thirty_nine_decisions": len(rows) == 46 and len({r["decision_id"] for r in rows}) == 39,
            "eight_new_decisions_D054_and_D058_revisions": counts == expected_additions,
            "D054_revision_links_previous_event": len(earlier_d054) > 0 and len(revised_d054) == 1
                and revised_d054[0].get("previous_event_id") == earlier_d054[-1]["event_id"],
            "complete_core_forty_five_event_prefix_preserved": len(raw_lines) >= 45
                and len(core_prefix) == CORE_LEDGER_PREFIX_BYTES and v4.digest(core_prefix) == CORE_LEDGER_PREFIX_SHA256,
            "final_D058_addendum_links_core_event": len(rows) == 46 and len(core_d058) == 1
                and rows[-1]["decision_id"] == "D-058"
                and rows[-1].get("previous_event_id") == core_d058[0]["event_id"],
        }
        report.update(human_decision_rows=len(human_ids), machine_events=len(rows),
                      machine_decisions=len({r["decision_id"] for r in rows}), new_events=len(additions),
                      core_prefix_sha256=v4.digest(core_prefix), core_prefix_bytes=len(core_prefix),
                      checks=checks, status="PASS" if all(checks.values()) else "FAIL")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["error"] = str(exc)
    if report["status"] != "PASS":
        issues.append("Human/machine V5 ledger inventory or D-054 revision contract failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", required=True, type=Path, help="Separate preserved V4 package; compare every discovered old result")
    parser.add_argument("--baseline-ledger", required=True, type=Path, help="Complete preserved 36-event ledger; exact prefix required")
    parser.add_argument("--baseline-canonical", required=True, type=Path, help="Prior canonical ledger including the external append; exact prefix required")
    parser.add_argument("--out", type=Path, help="Optional NEW receipt; exclusive creation after all before/after snapshots")
    args = parser.parse_args()
    if args.out is not None and (args.out.exists() or args.out.is_symlink()):
        raise FileExistsError(f"Existing receipt is preserved: {args.out}")
    started = datetime.now(timezone.utc).isoformat()
    before = v4.snapshot()
    canonical_before = v4.digest(CANONICAL.read_bytes()) if CANONICAL.is_file() else None
    baseline_before = inventory_at(args.baseline_package)
    baseline_files_before = {str(path.resolve()): v4.digest(path.read_bytes())
                             for path in (args.baseline_ledger, args.baseline_canonical)}
    issues, receipts = [], {}
    old_commands = legacy_commands(args.baseline_ledger)
    new_commands = [
        ("optimizer_contracts", ["-m", "unittest", "discover", "-s", "checks", "-p", "test_optimizer_reanalysis.py", "-v"]),
        ("optimizer_reanalysis", ["analysis/reconcile_optimizer_factorial.py", "--evidence-dir", "evidence/upstream_v5_optimizer"]),
        ("geometry_contracts", ["checks/test_update_feasibility_geometry.py"]),
        ("geometry_diagnostics", ["analysis/update_feasibility_geometry.py"]),
        ("causal_cache_contracts", ["prototype/test_causal_ngram_cache.py"]),
        ("store_reanalysis", ["analysis/reconcile_store_cache.py", "--evidence-root", "evidence/upstream_v5_store"]),
        ("native_update_contracts", ["-c", NATIVE_SUITE_ADAPTER]),
        ("late_store_cache_reanalysis", ["analysis/reconcile_store_cache_late.py",
            "--evidence-root", "evidence/upstream_v5_store_late",
            "--prior-evidence-root", "evidence/upstream_v5_store"]),
    ]
    for name, arguments in old_commands:
        receipts[name] = run_legacy(name, arguments)
        if receipts[name]["status"] != "PASS":
            issues.append(f"{name}: command or semantic checks failed")
    for name, arguments in new_commands:
        receipts[name] = run_v5_command(name, arguments)
        if receipts[name]["status"] != "PASS":
            issues.append(f"{name}: command, semantic checks, or saved native result comparison failed")
    old_comparisons = v4.compare_saved(receipts, issues)
    new_comparisons = compare_new_results(receipts, issues)
    copies_v4 = v4.verify_copies(issues)
    copies_v5 = verify_sources_v5(issues)
    copies_addendum = verify_sources_v5(issues, addendum=True)
    all_source_paths = [r["path"] for r in copies_v5["files"] + copies_addendum["files"]]
    distinct_source_copies = len(set(all_source_paths))
    if distinct_source_copies != 90 or len(all_source_paths) != 90:
        issues.append("Core and late-addendum manifests must identify 90 distinct local source copies in total")
    historical = v4.compare_history(args.baseline_package, issues)
    ledger_prefix = v4.compare_prefix(args.baseline_ledger, ROOT / "ledger_events.jsonl", "complete prior machine ledger", issues)
    canonical_prefix = v4.compare_prefix(args.baseline_canonical, CANONICAL, "complete prior canonical ledger", issues)
    ledgers = ledger_inventory(args.baseline_ledger, issues)
    after = v4.snapshot()
    canonical_after = v4.digest(CANONICAL.read_bytes()) if CANONICAL.is_file() else None
    baseline_after = inventory_at(args.baseline_package)
    baseline_files_after = {str(path.resolve()): v4.digest(path.read_bytes())
                            for path in (args.baseline_ledger, args.baseline_canonical)}
    unchanged = before == after
    baselines_unchanged = baseline_before == baseline_after and baseline_files_before == baseline_files_after
    if not unchanged:
        issues.append("Validation commands changed package inventory or bytes")
    if canonical_before != canonical_after:
        issues.append("Validation commands changed canonical LEDGER.md")
    if not baselines_unchanged:
        issues.append("Validation commands changed supplied historical baselines")
    dependencies = {}
    for name in ("numpy", "jsonschema", "scipy"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "NOT_INSTALLED"
    report = {
        "status": "PASS" if not issues else "FAIL", "errors": issues,
        "scope": "Local package contracts, tiny deterministic synthetic CPU training, causal measurement and pinned source reanalysis. No upstream neural rerun, corpus evaluation, GPU, language effectiveness result, remote CI or formal proof-assistant verification.",
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "dependencies": dependencies,
        "inspected_prior_pr_head": PRIOR_PR_HEAD, "inspected_research_cutoff": RESEARCH_CUTOFF,
        "review_core_cutoff": RESEARCH_CUTOFF, "late_addendum_cutoff": LATE_CUTOFF,
        "integration_base": INTEGRATION_BASE, "factorial_cutoff": FACTORIAL_CUTOFF,
        "commands": receipts, "commands_expected": 24,
        "commands_passed": sum(row["status"] == "PASS" for row in receipts.values()),
        "v4_groups": len(old_commands), "v5_groups": len(new_commands),
        "helper_reuse": "Unchanged V4 runner, semantic, saved-result, source-copy, history and prefix helpers. Only the in-memory package-validator expectation is adapted to 46 events/39 decisions and full supplied-prefix PASS, then restored.",
        "saved_v4_result_comparisons": old_comparisons, "saved_v5_result_comparisons": new_comparisons,
        "reviewed_final_cache_source_hashes": CACHE_FINAL_SOURCES,
        "upstream_v4_copies": copies_v4, "consolidated_v5_copies": copies_v5,
        "late_addendum_copies": copies_addendum, "distinct_core_and_addendum_source_copies": distinct_source_copies,
        "source_copy_count_note": "85 preserved core copies plus five bounded addendum copies; the old 21-copy set is included in the core85, not counted twice.",
        "historical_result_preservation": historical, "baseline_ledger": ledger_prefix,
        "baseline_canonical_ledger": canonical_prefix, "ledger_inventory": ledgers,
        "history_scope_note": "Every result found in the caller-supplied V4 baseline is compared byte-for-byte; the observed count is reported dynamically. Local baseline hashes do not authenticate a remote revision.",
        "default_commands_left_package_unchanged": unchanged,
        "canonical_ledger_unchanged_during_gate": canonical_before == canonical_after,
        "historical_baselines_unchanged_during_gate": baselines_unchanged,
        "canonical_ledger_sha256_before": canonical_before,
        "before_after_paths_changed": sorted(set(before) ^ set(after)),
        "before_after_bytes_changed": sorted(path for path in set(before) & set(after) if before[path] != after[path]),
        "source_hashes_before_receipt_creation": before,
        "source_hash_scope": "All package file bytes including existing bytecode are checked before/after; canonical and supplied baselines are separately checked. A NEW --out receipt is created only after these snapshots and is excluded from its own source hashes.",
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as output:
            output.write(text)
    print(text, end="")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
