"""Read-only local V4 gate; optional --out exclusively creates one NEW receipt.

Runs synthetic CPU training/measurement checks and pinned source reanalysis.
No corpus, GPU, remote CI, upstream neural experiment, or effectiveness claim.
Baseline arguments identify caller-supplied local history, not remote authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT.parent / "LEDGER.md"
RESEARCH_CUTOFF = "9d88a2c20233f42b4a49c4430264b20f842895ad"
PRIOR_PR_HEAD = "176ece70b17caffd71e2f0f522bb9462ce5f76a8"
V3_SOURCE = "2ef9ce5827cb8ea1fb6c8946df32fbcf5e06dec0"
TIMEOUT_SECONDS = 60

EXPECTATIONS = {
    "package_regressions": {"passed": 35, "tests_run": 35, "failures": 0, "errors": 0},
    "mathematical_examples": {"passed": 12, "total": 12},
    "numpy_examples": {"n_passed": 5, "n_tests": 5},
    "transport_anchors": {"reference": 1, "known_transform": 1},
    "native_forward_reference": {"passed": 13, "total": 13},
    "repository_reanalysis": {"upstream_commit": V3_SOURCE,
                              "solver_status_probe.distinguishes_status_2_and_4": False},
    "package_validator": {"status": "PASS", "errors": [], "events_checked": 36,
                          "decisions_checked": 31, "history_prefix_preserved": True},
    "native_training_contracts": {"passed": 17, "total": 17},
    "correction_contracts": {"passed": 26, "tests_run": 26, "failures": 0, "errors": 0},
    "margin_contracts": {"passed": 9, "tests_run": 9, "failures": 0, "errors": 0},
    "native_bridge_contracts": {"status": "PASS", "tests_run": 9, "passed": 9},
    "independent_correction_oracle": {"status": "PASS", "histories": 320,
                                      "cohort_rows_checked": 24000},
    "independent_native_bridge_replay": {
        "status": "PASS", "optimizer_updates_replayed": 16,
        "event_boundary_state_checks": 16, "tensor_displacement_arrays_checked": 176,
        "original_reference_predictions_checked": 23,
    },
    "margin_reanalysis": {"source_commit": RESEARCH_CUTOFF,
        "half_m1_matches_prior_four_common_rounded_metrics_and_success_flags": True,
        "within_phase_counterexample.first_solver_success": True,
        "within_phase_counterexample.second_solver_success": True,
        "within_phase_counterexample.first_target_correct_after_first_write": True,
        "within_phase_counterexample.first_target_correct_after_second_write": False,
        "within_phase_counterexample.second_target_correct_after_second_write": True},
    "margin_diagnostics": {
        "large_initial_margin_can_be_overturned.actual_strict_target_winner_after": False,
        "orthogonal_state_change_preserves_prediction.actual_strict_target_winner_after": True,
        "coherent_transport_preserves_prediction_despite_loose_bound.actual_strict_target_winner_after": True},
    "native_bridge_smoke": {
        "instrument": "native-correction-integration-smoke-v1",
        "counts.delivered_events": 8, "counts.initial_observations": 3,
        "counts.assignment_events": 5, "counts.optimizer_steps": 16,
        "counts.scored_target_token_exposures": 16,
        "counts.inference_forward_calls": 23,
        "initial_checkpoint.optimizer_step": 0, "final_checkpoint.optimizer_step": 16},
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def snapshot():
    # Include existing bytecode too: -B must not silently add/change any file.
    return {path.relative_to(ROOT).as_posix(): digest(path.read_bytes())
            for path in sorted(ROOT.rglob("*")) if path.is_file()}


def field(data, path):
    for key in path.split("."):
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def semantic_checks(name, data):
    checks = {f"{path} == {expected!r}": field(data, path) == expected
              for path, expected in EXPECTATIONS[name].items()}
    for rows_name in ("checks", "tests"):
        if rows_name in data:
            rows = data[rows_name]
            expected_count = data.get("total", data.get("n_tests"))
            checks[rows_name + "_complete"] = (
                isinstance(rows, list) and len(rows) == expected_count
                and all(not isinstance(row, dict) or row.get("passed") is True for row in rows))
    if name == "margin_reanalysis":
        arms = data.get("arms", {})
        checks["five_pinned_three_seed_arms"] = (
            set(arms) == {"half_m1", "half_m3", "half_m6", "keep_m3", "none_m1"}
            and all(row.get("seed_count") == 3 for row in arms.values()))
        rows = data.get("exogenous_event_diagnostics", [])
        checks["three_624_assignment_streams"] = (
            [row.get("seed") for row in rows] == [8001, 8002, 8003]
            and all(row.get("correction_events") == 624 for row in rows))
    if name == "native_bridge_smoke":
        checks["actual_weights_changed"] = (
            field(data, "initial_checkpoint.model_sha256")
            != field(data, "final_checkpoint.model_sha256"))
        cohorts = field(data, "metrics.cohorts") or {}
        checks["three_complete_assignment_cohorts"] = (
            set(cohorts) == {"all_assignments", "semantic_value_changes", "same_value_assignments"}
            and all(set(cohort["phase_end_horizons"]) == {"0", "1", "5", "10", "20"}
                    and all(row["total_events"] == row["eligible"] + row["censored"] + row["not_yet_due"]
                            and row["eligible"] == row["measured"] + row["missing_measurement"]
                            and row["missing_measurement"] == 0
                            for row in cohort["phase_end_horizons"].values())
                    for cohort in cohorts.values()))
    return checks


def run_command(name, arguments):
    command = [sys.executable, "-B", *arguments]
    began = time.perf_counter()
    receipt = {"command": command, "timeout_seconds": TIMEOUT_SECONDS,
               "status": "FAIL", "returncode": None}
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                timeout=TIMEOUT_SECONDS,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONOPTIMIZE": "0"})
        receipt.update(returncode=result.returncode, stderr=result.stderr)
        if name == "native_bridge_contracts":
            # This unmodified suite uses unittest's stderr protocol, not JSON.
            matches = re.findall(r"^Ran (\d+) tests? in [0-9.]+s$", result.stderr, re.MULTILINE)
            ok = (len(matches) == 1 and result.stderr.rstrip().splitlines()[-1:] == ["OK"]
                  and not result.stdout.strip())
            count = int(matches[0]) if len(matches) == 1 else 0
            data = {"status": "PASS" if ok else "FAIL", "tests_run": count,
                    "passed": count if ok else 0,
                    "format": "Parsed exact unittest count and terminal OK; skips are not accepted"}
            receipt["stdout"] = result.stdout
        else:
            data = json.loads(result.stdout)
            if not isinstance(data, dict):
                raise ValueError("Expected one JSON object receipt")
            canonical(data)  # Reject NaN/Infinity, even if json.loads accepted them.
        receipt["output"] = data
        checks = semantic_checks(name, data)
        receipt["semantic_checks"] = checks
        if result.returncode == 0 and all(checks.values()):
            receipt["status"] = "PASS"
    except subprocess.TimeoutExpired as exc:
        receipt["error"] = f"Command exceeded {TIMEOUT_SECONDS} seconds"
        receipt["stdout"] = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
        receipt["stderr"] = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        if "result" in locals():
            receipt["stdout"] = result.stdout
    receipt["elapsed_seconds_diagnostic_only"] = round(time.perf_counter() - began, 4)
    return receipt


def compare_saved(receipts, issues):
    comparisons = {}
    specifications = [
        ("transport_anchors", "transport_anchor_pilot_v0.json", "summary", True),
        ("native_forward_reference", "native_reference_v3_final.json", None, True),
        ("repository_reanalysis", "repository_reanalysis_v3.json", None, True),
        ("native_training_contracts", "trainable_native_v4.json", None, False),
        ("native_training_contracts", "trainable_independent_verification_v4.json", "author_suite", False),
        ("correction_contracts", "correction_protocol_v4.json", None, False),
        ("margin_contracts", "margin_durability_tests_v4.json", None, False),
        ("margin_diagnostics", "margin_durability_v4.json", None, False),
        ("margin_reanalysis", "margin_reanalysis_v4.json", None, False),
        ("native_bridge_smoke", "native_correction_smoke_v4.json", None, False),
        ("independent_correction_oracle", "correction_oracle_review_v4.json", None, False),
        ("independent_native_bridge_replay", "native_bridge_replay_review_v4.json", None, False),
    ]
    for name, filename, subkey, required in specifications:
        path = ROOT / "results" / filename
        row = {"command": name, "artifact": path.relative_to(ROOT).as_posix()}
        try:
            if not path.exists() and not required:
                row["status"] = "NOT_PRESENT"
            else:
                raw = path.read_bytes()
                expected = json.loads(raw)
                if filename == "trainable_independent_verification_v4.json":
                    sources_match = all(digest((ROOT / name).read_bytes()) == sha
                                        for name, sha in expected["hashes"].items())
                    row["recorded_review_source_hashes_match"] = sources_match
                    row["review_scope"] = "Reruns the embedded author suite; stored alias/F-order review probes are not independently rerun here"
                    if expected.get("status") != "PASS" or not sources_match:
                        issues.append("Stored independent training review has stale source hashes or non-PASS status")
                if subkey is not None:
                    expected = expected[subkey]
                actual = receipts[name].get("output")
                same = actual == expected
                row.update(status="PASS" if same else "FAIL", artifact_sha256=digest(raw),
                           equal_to_saved_result=same,
                           comparison_scope="Exact parsed JSON equality" + (f" at saved key {subkey}" if subkey else ""))
                if not same:
                    issues.append(f"{name}: differs from {row['artifact']}")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status="FAIL", error=str(exc))
            issues.append(f"{name}: saved-result comparison failed: {exc}")
        comparisons[filename] = row
    return comparisons


def verify_copies(issues):
    report = {"expected_file_count": 21, "files": [], "status": "FAIL"}
    try:
        raw = (ROOT / "evidence/source_manifest.json").read_bytes()
        manifest = json.loads(raw)
        entries = manifest["files"]
        report.update(manifest_sha256=digest(raw), observed_file_count=len(entries))
        paths = [entry["local_path"] for entry in entries]
        if len(entries) != 21 or len(set(paths)) != 21:
            issues.append("Source manifest must identify 21 distinct preserved copies")
        for entry in entries:
            row = {"path": entry["local_path"], "source_commit": entry["source_commit"]}
            try:
                relative = Path(entry["local_path"])
                path = ROOT / relative
                if relative.is_absolute() or not path.resolve().is_relative_to(ROOT):
                    raise ValueError("Source copy escapes package")
                data = path.read_bytes()
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                matched = (digest(data) == entry["sha256"] and blob == entry["git_blob_sha"]
                           and len(data) == entry["bytes"])
                row.update(sha256=digest(data), git_blob_sha=blob, bytes=len(data),
                           matches_source_manifest=matched)
                if not matched:
                    issues.append(f"Preserved source-copy mismatch: {entry['local_path']}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(matches_source_manifest=False, error=str(exc))
                issues.append(f"Source-copy verification failed: {entry['local_path']}: {exc}")
            report["files"].append(row)
        report["status"] = ("PASS" if len(entries) == 21 and len(set(paths)) == 21
                            and all(row["matches_source_manifest"] for row in report["files"]) else "FAIL")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["error"] = str(exc)
        issues.append(f"Unreadable source manifest: {exc}")
    return report


def compare_history(baseline, issues):
    if baseline is None:
        return {"status": "NOT_CHECKED", "reason": "--baseline-package not supplied"}
    baseline = baseline.resolve()
    report = {"baseline_package": str(baseline), "files": {}, "status": "FAIL",
              "scope": "Every file in the supplied prior package/results, including all V3 receipts"}
    try:
        if baseline == ROOT or (baseline / "results").resolve() == (ROOT / "results").resolve():
            raise ValueError("Baseline results must be a separate preserved package")
        files = sorted(path for path in (baseline / "results").rglob("*") if path.is_file())
        if not files:
            raise ValueError("Baseline results directory is missing or empty")
        for old in files:
            relative = old.relative_to(baseline)
            current = ROOT / relative
            previous = old.read_bytes()
            now = current.read_bytes() if current.is_file() else None
            same = now == previous
            report["files"][relative.as_posix()] = {
                "byte_identical": same, "baseline_sha256": digest(previous),
                "current_sha256": digest(now) if now is not None else None,
                "baseline_bytes": len(previous), "current_bytes": len(now) if now is not None else None}
            if not same:
                issues.append(f"Historical result missing or rewritten: {relative.as_posix()}")
        report["files_checked"] = len(files)
        report["status"] = "PASS" if all(row["byte_identical"] for row in report["files"].values()) else "FAIL"
    except (OSError, ValueError) as exc:
        report["error"] = str(exc)
        issues.append(f"Historical results comparison failed: {exc}")
    return report


def compare_prefix(baseline, current, label, issues):
    if baseline is None:
        return {"status": "NOT_CHECKED", "reason": f"No {label} baseline supplied"}
    row = {"baseline_path": str(baseline.resolve()), "current_path": str(current), "status": "FAIL"}
    try:
        if baseline.resolve() == current.resolve():
            raise ValueError("Baseline must be a separate preserved prior file")
        previous, now = baseline.read_bytes(), current.read_bytes()
        matched = bool(previous) and now.startswith(previous)
        row.update(baseline_sha256=digest(previous), baseline_bytes=len(previous),
                   current_sha256=digest(now), current_bytes=len(now),
                   complete_prior_bytes_preserved_as_prefix=matched,
                   status="PASS" if matched else "FAIL")
        if not matched:
            issues.append(f"{label}: complete prior prefix changed or baseline is empty")
    except (OSError, ValueError) as exc:
        row["error"] = str(exc)
        issues.append(f"{label}: prefix comparison failed: {exc}")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-package", type=Path, help="Preserved V3 package root; compare every results file")
    parser.add_argument("--baseline-ledger", type=Path, help="Complete prior ledger_events.jsonl; exact prefix required")
    parser.add_argument("--baseline-canonical", type=Path, help="Optional complete prior canonical LEDGER.md; exact prefix required")
    parser.add_argument("--out", type=Path, help="Optional NEW JSON receipt; never overwrite")
    args = parser.parse_args()
    if args.out is not None and (args.out.exists() or args.out.is_symlink()):
        raise FileExistsError(f"Existing receipt is preserved: {args.out}")
    started = datetime.now(timezone.utc).isoformat()
    before = snapshot()
    canonical_before = digest(CANONICAL.read_bytes()) if CANONICAL.is_file() else None
    issues = []
    commands = [
        ("package_regressions", ["checks/test_research_package.py"]),
        ("mathematical_examples", ["checks/repo_lightweight_checks.py"]),
        ("numpy_examples", ["checks/formal_sanity.py"]),
        ("transport_anchors", ["checks/transport_anchor_pilot.py"]),
        ("native_forward_reference", ["prototype/test_native_reference.py"]),
        ("repository_reanalysis", ["analysis/reconcile_repository_evidence.py"]),
        ("package_validator", ["checks/validate_research_package.py"]),
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
    if args.baseline_ledger is not None:
        commands[6][1].extend(["--baseline-ledger", str(args.baseline_ledger.resolve())])
    receipts = {}
    for name, arguments in commands:
        receipts[name] = run_command(name, arguments)
        if receipts[name]["status"] != "PASS":
            issues.append(f"{name}: command return code, receipt or semantic checks failed")
    comparisons = compare_saved(receipts, issues)
    copies = verify_copies(issues)
    historical = compare_history(args.baseline_package, issues)
    ledger = compare_prefix(args.baseline_ledger, ROOT / "ledger_events.jsonl", "machine ledger", issues)
    canonical_ledger = compare_prefix(args.baseline_canonical, CANONICAL, "canonical ledger", issues)
    human_decision_rows = None
    try:
        human_ids = re.findall(r"^\|\s*(D-\d{3,6})\s*\|", (ROOT / "02_DECISION_LEDGER.md").read_text(), re.MULTILINE)
        human_decision_rows = len(human_ids)
        if human_decision_rows != 54 or len(set(human_ids)) != 54:
            issues.append("Expected 54 distinct human decision rows in the V4 ledger")
    except OSError as exc:
        issues.append(f"Cannot count human ledger rows: {exc}")
    after = snapshot()
    canonical_after = digest(CANONICAL.read_bytes()) if CANONICAL.is_file() else None
    unchanged = before == after
    if not unchanged:
        issues.append("Default validation commands changed package file inventory or bytes")
    if canonical_before != canonical_after:
        issues.append("Default validation commands changed canonical LEDGER.md")
    dependencies = {}
    for name in ("numpy", "jsonschema", "scipy"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "NOT_INSTALLED"
    report = {
        "status": "PASS" if not issues else "FAIL", "errors": issues,
        "scope": "Local deterministic package contracts, tiny synthetic CPU training, causal measurement checks and engineering source reanalysis. No corpus, GPU, language effectiveness result, upstream neural experiment replication, formal verification or remote CI.",
        "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "dependencies": dependencies,
        "inspected_prior_pr_head": PRIOR_PR_HEAD, "inspected_research_cutoff": RESEARCH_CUTOFF,
        "commands": receipts, "commands_expected": len(commands),
        "commands_passed": sum(row["status"] == "PASS" for row in receipts.values()),
        "saved_result_comparisons": comparisons, "upstream_copies": copies,
        "historical_result_preservation": historical, "baseline_ledger": ledger,
        "baseline_canonical_ledger": canonical_ledger,
        "human_decision_rows": human_decision_rows,
        "history_scope_note": "NOT_CHECKED/NOT_PRESENT means no claim for that omitted comparison; local baselines do not authenticate a remote revision.",
        "default_commands_left_package_unchanged": unchanged,
        "canonical_ledger_unchanged_during_gate": canonical_before == canonical_after,
        "canonical_ledger_sha256_before": canonical_before,
        "before_after_paths_changed": sorted(set(before) ^ set(after)),
        "before_after_bytes_changed": sorted(path for path in set(before) & set(after) if before[path] != after[path]),
        "source_hashes_before_receipt_creation": before,
        "source_hash_scope": "All package file payloads, including existing bytecode, before and after checks. The NEW --out receipt is created only after both snapshots and has no self-reference. Canonical LEDGER.md is checked separately at ROOT.parent/LEDGER.md.",
    }
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as destination:
            destination.write(text)
    print(text, end="")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
