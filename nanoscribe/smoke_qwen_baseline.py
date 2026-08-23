#!/usr/bin/env python3
"""GPU smoke: real Qwen2.5-1.5B through adapter → selector → PR2 eval.

Requires torch + transformers and a CUDA/MPS/CPU device.
Run on RunPod: NANOSCIBE_QWEN_WEIGHTS=Qwen/Qwen2.5-1.5B-Instruct python3 nanoscribe/smoke_qwen_baseline.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
_script_dir = str(Path(__file__).resolve().parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# `python3 nanoscribe/smoke_qwen_baseline.py` prepends nanoscribe/ and shadows stdlib select.
sys.path[:] = [p for p in sys.path if p != _script_dir]

from nanoscribe.adapt import run_pipeline
from nanoscribe.adapters import Qwen25BaselineAdapter, default_baseline_specs
from nanoscribe.evaluate import atom_result
from nanoscribe.qwen_inference import DEFAULT_QWEN_MODEL
from nanoscribe.test_adapt import _gold, _model_input


def main() -> int:
    weights = os.environ.get("NANOSCIBE_QWEN_WEIGHTS", DEFAULT_QWEN_MODEL)
    adapter = Qwen25BaselineAdapter(weights_path=weights)
    gold = _gold()
    model_input = _model_input(gold.sources[0])
    batch = adapter.propose(model_input, default_baseline_specs())
    predicted, report = run_pipeline(model_input, batch, gold=gold)
    assert report is not None

    raw_lines = {
        atom.atom_id: (
            "NOT_MENTIONED"
            if atom.abstained
            else (
                f'{atom.assertion_state.value.upper() if atom.assertion_state else "MALFORMED"}: '
                f'{" ".join(f"\"{q}\"" for q in atom.quotes)}'
            )
        )
        for atom in batch.atoms
    }

    result = {
        "experiment": "p1_qwen_baseline_smoke_v0",
        "model": weights,
        "adapter": adapter.model_id,
        "latency_s": round(batch.latency_s, 3),
        "memory_bytes": batch.memory_bytes,
        "raw_lines": raw_lines,
        "aggregate": {
            "exact_gold_span": report.exact_gold_span,
            "span_character_f1": round(report.span_character_f1, 4),
            "assertion_state_correct": report.assertion_state_correct,
            "support_direct_exact": report.support_direct_exact,
            "coverage": round(report.coverage, 4),
            "correct_abstention": report.correct_abstention,
            "critical_error": report.critical_error,
        },
        "per_atom": {
            item.atom_id: {
                "exact_gold_span": item.exact_gold_span,
                "span_character_f1": round(item.span_character_f1, 4),
                "assertion_state_correct": item.assertion_state_correct,
                "abstained": item.abstained,
                "malformed": item.malformed,
            }
            for item in report.atom_results
        },
    }
    out_path = os.environ.get("NANOSCIBE_SMOKE_OUT")
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if out_path:
        Path(out_path).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
