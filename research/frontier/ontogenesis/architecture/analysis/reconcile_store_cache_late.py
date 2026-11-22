"""Bounded V5 late-result audit: metadata, source ASTs and rounded arithmetic.

No model imports, training, corpus access, network access or historical writes.
The earlier V5 cutoff remains a separate preserved evidence snapshot.
"""
from __future__ import annotations

import argparse
import ast
from decimal import Decimal
import hashlib
import json
from pathlib import Path


CORE = "0afd5629990b4e442fb3aa8f7c700b030d8faede"
LATE = "c597d5fefdd378c3e5c66a51e7690cdb90853287"
BASE = "research/frontier/ontogenesis/"


def local_path(root, row):
    return root / (row.get("local_path") or row["path"])


def verify(root, expected_count):
    manifest = json.loads((root / "source_manifest.json").read_text())
    if len(manifest["files"]) != expected_count:
        raise ValueError("unexpected evidence-file count")
    for row in manifest["files"]:
        data = local_path(root, row).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if (len(data) != row["bytes"] or blob != row["github_blob_sha"]
                or hashlib.sha256(data).hexdigest() != row["sha256"]):
            raise ValueError(f"evidence identity mismatch: {row['source_path']}")
    return manifest


def source(root, manifest, name, commit):
    rows = [row for row in manifest["files"]
            if row["source_path"] == BASE + name and row["commit"] == commit]
    if len(rows) != 1:
        raise ValueError(f"ambiguous source identity: {name}")
    return local_path(root, rows[0]).read_text()


def numeric_result(root, manifest, name, commit):
    return json.loads(source(root, manifest, "results/lm/" + name, commit), parse_float=Decimal)


def shape_counts(train_source):
    """Parse literal SIZES and derive init tensor sizes without executing it."""
    module = ast.parse(train_source)
    assignments = {target.id: node.value for node in module.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}
    vocab = ast.literal_eval(assignments["VOCAB"])
    cfg = {ast.literal_eval(key): {kw.arg: ast.literal_eval(kw.value) for kw in value.keywords}
           for key, value in zip(assignments["SIZES"].keys, assignments["SIZES"].values)}
    result = {}
    for name in ("S", "M", "L"):
        c = cfg[name]
        d, layers, ffn, heads, kv = (c[k] for k in ("d", "layers", "ffn", "heads", "kv"))
        head_dim = d // heads
        # Tied embedding/output, final RMS, and each block's two RMS scales,
        # Q/K head scales, Q/K/V/O matrices and gated feed-forward matrices.
        count = vocab * d + d + layers * (
            2 * d + 2 * head_dim + 2 * d * d + 2 * d * kv * head_dim + 3 * d * ffn)
        result[name] = {"config": c, "shape_derived_parameters": count}
    return result


def function_asts(text):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}


def reconcile(evidence_root, prior_evidence_root):
    evidence_root, prior_evidence_root = Path(evidence_root), Path(prior_evidence_root)
    late, prior = verify(evidence_root, 5), verify(prior_evidence_root, 10)
    if (late["late_cutoff"] != LATE or late["core_cutoff"] != CORE
            or prior["research_cutoff"] != CORE
            or any(row["commit"] != LATE for row in late["files"])):
        raise ValueError("unexpected source cutoff")
    current_train = source(evidence_root, late, "tpu/train_jax.py", LATE)
    old_train = source(prior_evidence_root, prior, "tpu/train_jax.py", CORE)
    shapes = shape_counts(current_train)
    assert shapes == shape_counts(old_train)
    old_functions, new_functions = function_asts(old_train), function_asts(current_train)
    unchanged = {name: old_functions[name] == new_functions[name] for name in (
        "init", "rms", "rope_tables", "apply_rope", "forward", "count", "loss_fn", "evaluate")}
    if not all(unchanged.values()):
        raise ValueError("model or evaluation function changed")
    for item in late["unchanged_script_identities"]:
        prior_text = source(prior_evidence_root, prior,
                            item["source_path"].removeprefix(BASE), CORE)
        data = prior_text.encode("utf-8")
        if (hashlib.sha256(data).hexdigest() != item["sha256"]
                or hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                != item["github_blob_sha_at_both_cutoffs"]):
            raise ValueError("unchanged evaluator identity mismatch")
    gate_cache = source(prior_evidence_root, prior, "tpu/cache_exchange.py", CORE)
    gate_store = source(prior_evidence_root, prior, "theory/RESEARCH-DECISION-20261121.md", CORE)
    assert "Decision rule, read on the 40M checkpoint" in gate_cache
    assert "under 0.01 on every domain" in gate_cache
    assert "Gain at 101M of 0.02 or more" in gate_store

    cache_s = numeric_result(evidence_root, late, "cache-exchange-S.json", LATE)
    cache_m = numeric_result(evidence_root, late, "cache-exchange-M.json", LATE)
    store_m = numeric_result(evidence_root, late, "store-exchange-M.json", LATE)
    store_s = numeric_result(prior_evidence_root, prior, "store-exchange-S.json", CORE)
    training_baselines = {"S": store_s, "M": store_m}
    result_rows = {}
    rows_reconciled = 0
    for filename, data, augmented, control in (
            ("cache-exchange-S.json", cache_s, "with_cache", "foreign_cache"),
            ("cache-exchange-M.json", cache_m, "with_cache", "foreign_cache"),
            ("store-exchange-M.json", store_m, "with_store", "uninformed_store")):
        source_baseline = training_baselines[data["size"]]
        same_checkpoint_path = data["ckpt"] == source_baseline["ckpt"]
        same_model_rows = all(row["model"] == source_baseline["bits_per_byte"][domain]["model"]
                              for domain, row in data["bits_per_byte"].items())
        rows = []
        for domain, values in data["bits_per_byte"].items():
            gain = values["model"] - values[augmented]
            control_gain = values["model"] - values[control]
            baseline_difference = values["model"] - source_baseline["reported_by_training_run"][domain]
            assert gain == data["gain"][domain]
            if "foreign_cache_gain" in data:
                assert control_gain == data["foreign_cache_gain"][domain]
            rows.append({"domain": domain, "model_bpb": str(values["model"]),
                         "augmented_bpb": str(values[augmented]), "gain": str(gain),
                         "gain_arithmetic_agrees": True, "control_gain": str(control_gain),
                         "baseline_difference_from_cross_record_reference": str(baseline_difference)})
            rows_reconciled += 1
        discrepancy_count = sum(Decimal(row["baseline_difference_from_cross_record_reference"]) != 0
                                for row in rows)
        maximum = max(abs(Decimal(row["baseline_difference_from_cross_record_reference"])) for row in rows)
        assert maximum == data["max_abs_difference_from_reported"]
        result_rows[filename] = {
            "size": data["size"], "shape_derived_parameters": shapes[data["size"]]["shape_derived_parameters"],
            "fit_target_presentations": data["fit_tokens"], "rows": rows,
            "baseline_reference": "store-exchange-" + data["size"] + ".json reported_by_training_run",
            "baseline_reference_is_cross_record_inference": filename.startswith("cache"),
            "baseline_reference_checkpoint_path_agrees": same_checkpoint_path,
            "baseline_reference_model_rows_agree": same_model_rows,
            "baseline_discrepant_rows": discrepancy_count,
            "reported_max_baseline_difference": str(data["max_abs_difference_from_reported"]),
            "literal_baseline_equality_satisfied": maximum == 0,
            "reported_causality_check_passed": data.get("causality_check_passed"),
            "store_array_bytes": data.get("store_bytes"),
            "whole_script_seconds_reported": data["seconds"],
            "checkpoint_uses_host_sampler_name": "-hs" in Path(data["ckpt"]).stem,
        }

    domains = ("web", "textbook", "math", "code")
    cache_stop = all(cache_m["gain"][domain] < Decimal("0.01") for domain in domains)
    support_identical = cache_s["share_of_tokens_the_cache_could_copy"] == cache_m["share_of_tokens_the_cache_could_copy"]
    return {
        "schema": "store-cache-late-reanalysis-v1", "core_cutoff": CORE, "late_cutoff": LATE,
        "scope": "bounded late addendum; rounded JSON arithmetic and source AST comparison only",
        "language_model_evaluation": False,
        "verified_late_source_files": 5, "verified_prior_source_files": 10,
        "late_rows_reconciled": rows_reconciled,
        "source_configuration_parameter_counts": shapes,
        "unchanged_model_and_evaluation_function_asts": unchanged,
        "unchanged_exchange_script_blobs": len(late["unchanged_script_identities"]),
        "results": result_rows,
        "cache_support_share": {
            "S_M_identical": support_identical,
            "values": {k: str(v) for k, v in cache_m["share_of_tokens_the_cache_could_copy"].items()},
            "interpretation": "fraction of true targets with positive exact-cache mass; not accuracy"},
        "registered_cache_40M_gate": {
            "applicable_result": "cache-exchange-M.json", "scale_parameters": shapes["M"]["shape_derived_parameters"],
            "all_four_domain_gains_below_0_01": cache_stop,
            "numeric_branch": "close_explicit_copy_line_at_tested_scale" if cache_stop else "other",
            "source_controls_fully_satisfied": False,
            "classification": "registered numerical stopping branch met, with unresolved control/provenance qualifications",
            "not_a_proof_all_copy_mechanisms_are_useless": True},
        "registered_store_101M_gate": {
            "required_size": "L", "required_scale_parameters": shapes["L"]["shape_derived_parameters"],
            "new_result_size": "M", "decision": "pending_registered_L_result"},
        "training_sampler": {
            "source_now_uses_host_numpy_sampling": "rng.integers(0, n_start" in current_train,
            "prior_source_used_device_jax_randint": "jax.random.randint(sub" in old_train,
            "new_run_naming_uses_hs": 'return n + "-hs"' in current_train,
            "reported_old_sampling_bias_independently_verified": False,
            "late_checkpoint_sampler_attribution": "legacy naming suggests first-version checkpoints; payload/recipe not independently recovered",
            "no_repaired_sampler_training_result_in_this_addendum": True},
        "remaining_qualifications": [
            "no checkpoint payload/hash or executed command", "no raw per-target losses or frozen exact mixture weights",
            "corpus byte denominator and mix weights not independently reconstructed",
            "store retains demonstrated extra pre-crop context and no document exclusion",
            "cache resets crops; underlying document boundaries and foreign-window disjointness not established",
            "model baseline comparison differs by 0.0001; direct zero-weight identity remains untested; zero rounded foreign gain does not prove independence",
            "no matched resource comparison or uncertainty interval", "no new language model evaluation by this audit"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    package = Path(__file__).resolve().parents[1]
    parser.add_argument("--evidence-root", type=Path, default=package / "evidence/upstream_v5_store_late")
    parser.add_argument("--prior-evidence-root", type=Path, default=package / "evidence/upstream_v5_store")
    parser.add_argument("--out", type=Path, help="new JSON receipt; existing paths are never overwritten")
    args = parser.parse_args()
    report = reconcile(args.evidence_root, args.prior_evidence_root)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as target:
            target.write(rendered)
    print(rendered, end="")
