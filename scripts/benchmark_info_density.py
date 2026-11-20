#!/usr/bin/env python3
"""Measure epistemic information density, KV-eviction safety, and Apple Silicon UMA speedup.

Offline/stdlib mode authenticates preserved H6 (5,000 fields) and P1 compact
evaluations to compute epistemic operator mutual information per parameter/byte.
When ``--compact-checkpoint`` is supplied, also executes a live head-to-head
benchmark between ``typed-context-source-trie-v8`` (uncached per-slot generate)
and ``typed-context-source-trie-v9-prefix-kv`` (shared prefix DynamicCache +
zero-entropy unary-trie coalescing + commit-gated KV eviction), plus read-only
POSIX ``mmap`` ``.safetensors`` alignment inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nano_ai.evaluation import evaluate_solver
from nano_ai.training.diagnose_h6 import CANONICAL_H6_EVALUATION_SHA256
from nanoscribe.compact import STRUCTURED_PROMPT, LocalCompactPredictor, compact_solver
from nanoscribe.info_density import (
    APPLE_M3_MAX_40GPU_64GB,
    CommitGatedKVCacheLedger,
    NANO_H6_FP16,
    QWEN25_1_5B_FP16,
    compute_epistemic_density,
    compute_uma_roofline,
    inspect_safetensors_zero_copy,
    recommend_uma_tier,
    simulate_sliding_window_erasure,
)
from nanoscribe.scribe import scribe
from scripts.run_p1_pilot import DATA as PILOT_DATA, Replay, admit as admit_pilot, proposal_fields

ROOT = Path(__file__).resolve().parents[1]
H6_EVAL_PATH = ROOT / "artifacts/nano_h6/kaggle/eval-20260805/results/development_evaluation.json"
C14_COMPARISON_PATH = ROOT / "artifacts/p1_retention_validation_20261026/constrained/comparison.json"
C14_PAIRED_PATH = ROOT / "artifacts/p1_retention_validation_20261026/constrained/paired_fields.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def evaluate_h6_density(eval_path: Path = H6_EVAL_PATH) -> dict[str, Any]:
    raw_bytes = eval_path.read_bytes()
    digest = _sha256_bytes(raw_bytes)
    if eval_path == H6_EVAL_PATH and digest != CANONICAL_H6_EVALUATION_SHA256:
        raise ValueError(f"H6 evaluation SHA-256 mismatch: {digest}")
    payload = json.loads(raw_bytes)
    gold_states: list[str] = []
    pred_states: list[str] = []
    joint_exact = 0
    for item in payload["uncalibrated_raw"]["item_diagnostics"]:
        for field in item["fields"]:
            gold_states.append(field["gold_state"])
            pred_states.append(field["raw_state"])
            if field["joint_exact"]:
                joint_exact += 1
    report = compute_epistemic_density(
        gold_states,
        pred_states,
        grounded_exact_count=joint_exact,
        geometry=NANO_H6_FP16,
    )
    return {"source_sha256": digest, **report.to_dict()}


def evaluate_c14_compact_density(paired_path: Path = C14_PAIRED_PATH) -> dict[str, Any]:
    raw_bytes = paired_path.read_bytes()
    rows = json.loads(raw_bytes)
    gold_states = [row["gold"]["state"] for row in rows]
    checked_states = [row["checked"]["prediction"]["state"] for row in rows]
    grounded_exact = sum(1 for row in rows if row["checked"]["grounded_exact_correct"])
    report = compute_epistemic_density(
        gold_states,
        checked_states,
        grounded_exact_count=grounded_exact,
        geometry=QWEN25_1_5B_FP16,
    )
    return {"source_sha256": _sha256_bytes(raw_bytes), **report.to_dict()}


def evaluate_pilot_stored_density(run_name: str = "run-09") -> dict[str, Any]:
    rows, cases = admit_pilot()
    run_dir = PILOT_DATA / run_name
    records = json.loads((run_dir / "encounters.json").read_text())
    gold_states: list[str] = []
    checked_states: list[str] = []
    for case, record in zip(cases, records, strict=True):
        extracted = record["draft"]["extraction"]["fields"]
        for target_field, pred_field in zip(case.gold.fields, extracted, strict=True):
            gold_states.append(target_field.state.value)
            checked_states.append(pred_field["state"])
    checked_eval = json.loads((run_dir / "checked_evaluation.json").read_text())
    grounded_exact = int(
        round(checked_eval["quality"]["grounded_exact_field_accuracy"] * len(gold_states))
    )
    report = compute_epistemic_density(
        gold_states,
        checked_states,
        grounded_exact_count=grounded_exact,
        geometry=QWEN25_1_5B_FP16,
    )
    return {"pilot_run": run_name, **report.to_dict()}


def benchmark_live_compact_prefix_kv(checkpoint_dir: Path) -> dict[str, Any]:
    rows, cases = admit_pilot()
    safetensors_path = checkpoint_dir / "model.safetensors"
    mmap_audit = (
        inspect_safetensors_zero_copy(safetensors_path).to_dict()
        if safetensors_path.is_file()
        else None
    )

    uncached = LocalCompactPredictor(
        str(checkpoint_dir), system_prompt=STRUCTURED_PROMPT, decoding="constrained", prefix_cache=False
    )
    cached = LocalCompactPredictor(
        str(checkpoint_dir), system_prompt=STRUCTURED_PROMPT, decoding="constrained", prefix_cache=True
    )

    # Warm up weights/device once before timed comparison so model load time is separated.
    uncached._load()
    cached._loaded = uncached._loaded

    t0 = time.perf_counter()
    uncached_raw = {row["example_id"]: uncached(row["transcript"]) for row in rows}
    uncached_wall = time.perf_counter() - t0

    t1 = time.perf_counter()
    cached_raw = {row["example_id"]: cached(row["transcript"]) for row in rows}
    cached_wall = time.perf_counter() - t1

    exact_raw_matches = sum(
        1 for row in rows if uncached_raw[row["example_id"]] == cached_raw[row["example_id"]]
    )

    uncached_parsed = {
        row["example_id"]: proposal_fields(uncached_raw[row["example_id"]], row["transcript"])
        for row in rows
    }
    cached_parsed = {
        row["example_id"]: proposal_fields(cached_raw[row["example_id"]], row["transcript"])
        for row in rows
    }
    uncached_eval = evaluate_solver(Replay(uncached_parsed, "v8-uncached"), cases)
    cached_eval = evaluate_solver(Replay(cached_parsed, "v9-prefix-kv"), cases)

    uncached_checked = {}
    cached_checked = {}
    ledger_summaries = []
    for row in rows:
        ex_id = row["example_id"]
        transcript = row["transcript"]
        u_draft = scribe(
            transcript,
            encounter_id=ex_id,
            solver=compact_solver(lambda _, r=uncached_raw[ex_id]: r, binder="context"),
            context_policy="context",
        )
        c_draft = scribe(
            transcript,
            encounter_id=ex_id,
            solver=compact_solver(lambda _, r=cached_raw[ex_id]: r, binder="context"),
            context_policy="context",
        )
        from nano_ai.contract import NanoOutput

        uncached_checked[ex_id] = NanoOutput.from_dict(u_draft["extraction"]).fields
        cached_checked[ex_id] = NanoOutput.from_dict(c_draft["extraction"]).fields

        # Audit commit-gated KV eviction on each encounter.
        ledger = CommitGatedKVCacheLedger(transcript, QWEN25_1_5B_FP16)
        turns = c_draft["encounter"]["sources"][0]["turns"]
        for idx, turn in enumerate(turns):
            tok_est = max(1, len(turn["text"].split()) * 2)
            ledger.register_turn_kv(idx, tok_est)
            spans = [
                (ev["evidence_id"], "committed", ev["start"], ev["end"])
                for ev in c_draft["encounter"]["evidence"]
                if ev["turn_id"] == turn["turn_id"]
            ]
            ledger.commit_turn(idx, spans)
        ledger.evict_all_committed()
        ledger_summaries.append(ledger.summary())

    uncached_checked_eval = evaluate_solver(Replay(uncached_checked, "v8-checked"), cases)
    cached_checked_eval = evaluate_solver(Replay(cached_checked, "v9-checked"), cases)

    total_gen_tokens = sum(uncached.records[row["transcript"]]["generated_tokens"] for row in rows)
    total_slot_calls = sum(len(uncached.records[row["transcript"]]["steps"]) for row in rows)
    uncached_forward_est = total_gen_tokens  # model.generate runs 1 prefill + (gen_tokens - 1) steps = gen_tokens forwards per slot
    cached_forwards = sum(cached.records[row["transcript"]]["forward_passes"] for row in rows)
    coalesced_tokens = sum(
        cached.records[row["transcript"]]["coalesced_zero_entropy_tokens"] for row in rows
    )
    roofline = compute_uma_roofline(
        QWEN25_1_5B_FP16,
        hardware=APPLE_M3_MAX_40GPU_64GB,
        uncached_forward_calls=uncached_forward_est,
        cached_forward_calls=cached_forwards,
    )

    return {
        "device": uncached._loaded[2],
        "dialogue_count": len(rows),
        "slot_call_count": total_slot_calls,
        "exact_raw_slot_string_matches": exact_raw_matches,
        "uncached_v8_seconds": round(uncached_wall, 4),
        "cached_v9_seconds": round(cached_wall, 4),
        "wall_clock_speedup_x": round(uncached_wall / max(cached_wall, 1e-9), 3),
        "uncached_forward_passes": uncached_forward_est,
        "cached_forward_passes": cached_forwards,
        "coalesced_zero_entropy_tokens": coalesced_tokens,
        "uncached_raw_grounded_exact": uncached_eval.quality["grounded_exact_field_accuracy"],
        "cached_raw_grounded_exact": cached_eval.quality["grounded_exact_field_accuracy"],
        "uncached_checked_grounded_exact": uncached_checked_eval.quality["grounded_exact_field_accuracy"],
        "cached_checked_grounded_exact": cached_checked_eval.quality["grounded_exact_field_accuracy"],
        "safetensors_zero_copy_mmap": mmap_audit,
        "uma_roofline": roofline,
        "commit_gated_kv_eviction": {
            "encounters_audited": len(ledger_summaries),
            "total_unbounded_kv_bytes": sum(int(s["unbounded_kv_bytes"]) for s in ledger_summaries),
            "total_symbolic_ledger_bytes": sum(int(s["symbolic_ledger_bytes"]) for s in ledger_summaries),
            "mean_memory_reduction_ratio": round(
                sum(float(s["memory_reduction_ratio"]) for s in ledger_summaries) / len(ledger_summaries),
                6,
            ),
            "uncommitted_erasure_events": sum(int(s["uncommitted_erasure_events"]) for s in ledger_summaries),
        },
    }


def build_report_bundle(*, compact_checkpoint: Path | None = None) -> dict[str, Any]:
    h6 = evaluate_h6_density()
    pilot_qwen = evaluate_pilot_stored_density("run-09")
    c14_qwen = evaluate_c14_compact_density()

    density_ratio_vs_pilot = round(
        h6["mi_bits_per_field_per_megabyte"] / max(pilot_qwen["mi_bits_per_field_per_megabyte"], 1e-12),
        2,
    )
    erasure_demo = simulate_sliding_window_erasure(
        turn_token_counts=[48] * 20,
        evidence_turn_indices=[1, 3, 7, 12, 18],
        window_tokens=256,
    )
    uma_tiers = {
        "nominal_64gb_free_28gb": recommend_uma_tier(
            thermal="nominal", available_memory_bytes=28 * 1024**3
        ),
        "nominal_compile_pressure_4gb": recommend_uma_tier(
            thermal="nominal", available_memory_bytes=4 * 1024**3
        ),
        "serious_thermal_pressure": recommend_uma_tier(
            thermal="serious", available_memory_bytes=16 * 1024**3
        ),
        "critical_thermal_pressure": recommend_uma_tier(
            thermal="critical", available_memory_bytes=16 * 1024**3
        ),
    }
    bundle: dict[str, Any] = {
        "schema_version": "nano.info_density_uma_benchmark.v1",
        "platform": platform.platform(),
        "epistemic_density": {
            "native_nano_h6_5000_fields": h6,
            "compact_qwen_pilot_100_fields": pilot_qwen,
            "compact_qwen_c14_fresh_120_fields": c14_qwen,
            "native_vs_compact_pilot_mi_density_ratio_x": density_ratio_vs_pilot,
        },
        "sliding_window_kv_erasure_counterexample": erasure_demo,
        "uma_tier_routing": uma_tiers,
        "live_apple_silicon_benchmark": (
            benchmark_live_compact_prefix_kv(compact_checkpoint)
            if compact_checkpoint is not None
            else None
        ),
    }
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compact-checkpoint",
        type=Path,
        default=None,
        help="Optional local Qwen2.5-1.5B-Instruct snapshot for live Apple Silicon benchmark",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark bundle",
    )
    args = parser.parse_args(argv)
    bundle = build_report_bundle(compact_checkpoint=args.compact_checkpoint)
    rendered = json.dumps(bundle, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
