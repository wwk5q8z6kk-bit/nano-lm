#!/usr/bin/env python3
"""Student-A structured baseline on local GPU — C1 then C2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nanoscribe.adapt import ModelCandidate, run_pipeline
from nanoscribe.campaign import CampaignLedger
from nanoscribe.campaign_datasets import campaign_cases
from nanoscribe.harness import HarnessResult, TrackConfig, _per_atom, _report_aggregate, write_results
from nanoscribe.prompt import build_structured_candidate_prompt, structured_candidate_system_prompt
from nanoscribe.qwen_inference import resolve_weights_path
from nanoscribe.tracks import ModelTrack, STUDENT_MODEL


def _load_model(weights: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(weights, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(weights, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _generate_structured(model, tokenizer, device: str, user_prompt: str) -> str:
    import torch

    messages = [
        {"role": "system", "content": structured_candidate_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, temperature=0.0, do_sample=False)
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_suite(weights: str, suite: str) -> tuple[list[HarnessResult], float]:
    model, tokenizer, device = _load_model(weights)
    cases = campaign_cases(suite)
    track = TrackConfig(
        track=ModelTrack.FRONTIER,
        model_id=f"student/{weights}-structured",
        adapter_factory=lambda: None,
        cost_class="experiment_scoped_a100_80gb",
        notes="Student-A structured baseline — QLoRA blocked",
    )
    started = time.perf_counter()
    results: list[HarnessResult] = []
    for case in cases:
        t0 = time.perf_counter()
        prompt = build_structured_candidate_prompt(case.model_input.source, case.atom_specs)
        raw = _generate_structured(model, tokenizer, device, prompt)
        try:
            batch = ModelCandidate.from_json(raw)
        except Exception:
            batch = ModelCandidate(atoms=())
        predicted, report = run_pipeline(case.model_input, batch, gold=case.gold)
        assert report is not None
        from nanoscribe.harness import FailureTaxonomy

        results.append(
            HarnessResult(
                track=track.track,
                model_id=track.model_id,
                test_set=case.test_set,
                encounter_id=case.encounter_id,
                cost_class=track.cost_class,
                aggregate=_report_aggregate(report),
                failures=FailureTaxonomy.from_report(report),
                per_atom=_per_atom(report),
                latency_s=time.perf_counter() - t0,
                memory_bytes=0,
            )
        )
    return results, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="Student-A structured eval")
    parser.add_argument("--weights", default=resolve_weights_path(STUDENT_MODEL) or STUDENT_MODEL)
    parser.add_argument("--suites", default="c1_canary,c2_screening")
    parser.add_argument("--record-spend", action="store_true")
    args = parser.parse_args()

    payload: dict[str, object] = {"weights": args.weights, "suites": {}}
    total_s = 0.0
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        results, elapsed = run_suite(args.weights, suite)
        total_s += elapsed
        cov = sum(r.aggregate.get("coverage", 0) for r in results) / max(1, len(results))
        malformed = sum(r.failures.malformed for r in results)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "artifacts" / "p1_runs" / f"student_a_structured_{suite}_{ts}.json"
        write_results(results, out, extra={"suite": suite, "lane": "student_a_structured", "qlora": "BLOCKED"})
        payload["suites"][suite] = {
            "n_cases": len(results),
            "avg_coverage": round(cov, 4),
            "malformed": malformed,
            "wall_s": round(elapsed, 1),
            "artifact": str(out),
        }

    if args.record_spend:
        hours = max(total_s / 3600.0, 0.1)
        amount = round(1.59 * hours, 4)
        ledger = CampaignLedger.load()
        allowed, reason = ledger.budget_gate(amount)
        if allowed:
            entry = ledger.commit(
                "student_a",
                f"Student-A structured C1+C2 wall_s={total_s:.0f}",
                amount,
                gpu="NVIDIA A100-SXM4-80GB",
                rate_per_hr=1.59,
            )
            ledger.actualize(entry, amount)
            ledger.save()

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
