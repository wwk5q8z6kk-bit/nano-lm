"""Pinned-summary arithmetic and source-function counterexamples only.

No JAX/model import, training, corpus access, network, or historical result edits.
Only allowlisted definitions from hash-verified source copies execute, on tiny
constructed integer arrays. Counterexamples diagnose contracts, not BPB impact.
"""
from __future__ import annotations

import argparse
import ast
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CUTOFF = "0afd5629990b4e442fb3aa8f7c700b030d8faede"
BASE = Path("research/frontier/ontogenesis")


def verify_manifest(root):
    manifest = json.loads((root / "source_manifest.json").read_text())
    if manifest["research_cutoff"] != CUTOFF:
        raise ValueError("unexpected research cutoff")
    for row in manifest["files"]:
        local_path = row.get("local_path") or row["path"]
        data = (root / local_path).read_bytes()
        if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"SHA-256/length mismatch: {local_path}")
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != row["github_blob_sha"]:
            raise ValueError(f"Git blob mismatch: {local_path}")
    return manifest


def source_definitions(path, names, constants):
    """Run selected literal constants/functions/classes; no imports or main block."""
    namespace = {"np": np, "math": math}
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Tuple):
                    targets = [part.id for part in target.elts if isinstance(part, ast.Name)]
                    if set(targets) == set(constants):
                        values = ast.literal_eval(node.value)
                        namespace.update(zip(targets, values))
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
    if {node.name for node in nodes} != set(names) or not set(constants).issubset(namespace):
        raise ValueError("pinned definitions unavailable")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def reanalyse(root):
    root = Path(root)
    manifest = verify_manifest(root)
    current = root / "current" / BASE
    result = json.loads((current / "results/lm/store-exchange-S.json").read_text(),
                        parse_float=Decimal)
    initial = json.loads((root / "store_result_initial" / BASE /
                          "results/lm/store-exchange-S.json").read_text(), parse_float=Decimal)
    rows = []
    for domain, scores in result["bits_per_byte"].items():
        gain = scores["model"] - scores["with_store"]
        baseline_difference = scores["model"] - result["reported_by_training_run"][domain]
        rows.append({"domain": domain, "model_bpb": str(scores["model"]),
                     "with_store_bpb": str(scores["with_store"]),
                     "gain_from_rounded_rows": str(gain),
                     "reported_gain": str(result["gain"][domain]),
                     "gain_agrees": gain == result["gain"][domain],
                     "baseline_difference": str(baseline_difference),
                     "literal_four_decimal_baseline_equality": baseline_difference == 0})
    old_primary_equal = all(
        initial["bits_per_byte"][d][key] == result["bits_per_byte"][d][key]
        for d in result["bits_per_byte"] for key in ("model", "with_store"))

    store_source = source_definitions(current / "tpu/store_exchange.py", {"ctx_key", "Store"},
                                      {"BITS", "ORDERS", "BUCKETS"})
    store = store_source["Store"](np.array([1, 2, 3, 4, 7, 8, 3, 5], dtype=np.int32))
    left, right = np.array([1, 2, 3, 4]), np.array([7, 8, 3, 4])
    assert np.array_equal(left[2:], right[2:])
    p_left, _ = store.lookup(left, 0)
    p_right, _ = store.lookup(right, 0)
    p_crop_left, _ = store.lookup(left[2:], 0)
    p_crop_right, _ = store.lookup(right[2:], 0)
    boundary = {"visible_crop": left[2:].tolist(), "target": 4,
                "full_file_lookup_target_probabilities": [float(p_left[-1]), float(p_right[-1])],
                "crop_local_lookup_target_probabilities": [float(p_crop_left[-1]), float(p_crop_right[-1])],
                "violation_reproduced": bool(p_left[-1] != p_right[-1]),
                "crop_local_repair_agrees": bool(p_crop_left[-1] == p_crop_right[-1]),
                "classification": "causal extra pre-crop context; not future leakage"}

    old_cache = source_definitions(root / "cache_initial" / BASE / "tpu/cache_exchange.py",
                                   {"cache_probs"}, {"ORDERS", "BUCKETS"})["cache_probs"]
    new_cache = source_definitions(current / "tpu/cache_exchange.py", {"cache_probs"},
                                   {"ORDERS", "BUCKETS"})["cache_probs"]
    tokens = [0, 1, 2, 3]
    old_same, old_same_g = old_cache(tokens, tokens)
    old_copies, old_copies_g = old_cache(list(tokens), list(tokens))
    repaired, repaired_g = new_cache(tokens)
    leakage = {"tokens": tokens, "old_same_object_probabilities": old_same.tolist(),
               "old_distinct_equal_object_probabilities": old_copies.tolist(),
               "repaired_probabilities": repaired.tolist(),
               "old_identity_bug_reproduced": bool(not np.array_equal(old_same, old_copies)),
               "repaired_equals_causal_old_path": bool(np.array_equal(old_same, repaired)
                                                        and np.array_equal(old_same_g, repaired_g))}

    mix = result["bits_per_byte"]["mix"]
    return {"schema": "store-cache-reanalysis-v1", "research_cutoff": CUTOFF,
            "scope": "rounded committed summaries and synthetic source-function checks only",
            "language_model_evaluation": False, "verified_source_files": len(manifest["files"]),
            "store_S": {"rows": rows, "store_bytes": result["store_bytes"],
                        "store_GiB": result["store_bytes"] / 2 ** 30,
                        "fit_token_positions": result["fit_tokens"],
                        "relative_mix_bpb_reduction_percent": str(
                            100 * (mix["model"] - mix["with_store"]) / mix["model"]),
                        "original_primary_rows_preserved": old_primary_equal,
                        "invalidated_old_shuffled_gain_mix": str(initial["shuffled_gain_mix"]),
                        "current_shifted_control_gain_mix": str(result["uninformed_store_gain_mix"]),
                        "weighted_mix_independently_recomputed": False,
                        "reason_mix_not_recomputed": "source corpus mix/byte manifest is not in this evidence set"},
            "source_crop_boundary_counterexample": boundary,
            "source_cache_identity_counterexample": leakage,
            "causal_cache_language_result_at_cutoff": "no committed valid rerun located in pinned repository tree",
            "registered_store_101M_decision": "pending; S checkpoint cannot activate it",
            "registered_cache_40M_decision": "pending"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "evidence/upstream_v5_store")
    parser.add_argument("--out", type=Path, help="new JSON path; never overwrite an existing file")
    args = parser.parse_args()
    report = reanalyse(args.evidence_root)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out:
        with args.out.open("x", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered, end="")
