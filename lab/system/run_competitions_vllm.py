"""vLLM Runner for Separated Competition 1 (Naked Foundation) & Competition 2 (Complete System).

Supports `<4B` NanoLM checkpoints (`NanoLM-B3`, `NanoLM-B4*`, `NanoLM-B5-RLVR`), medium/large
dense models (`Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`), and `~120B` MoE (`openai/gpt-oss-120b` MXFP4)
under strictly matched evaluation protocols.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from lab.system.frontier_bench import (
    WORKFLOW_SCHEMAS,
    build_frontier_100_suite,
    evaluate_competition_2_system,
)
from lab.system.verification_engine import extract_canonical_answer, verify_json_candidate
from lab.trackb.eval_capabilities import TOOL_SYSTEM_INSTRUCTION, parse_tool_calls, strip_thinking
from nanolm.eval.validators import verify_math_or_logic, verify_python_solution


def render_chat(tok, messages: list[dict[str, str]], think: bool) -> str:
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_batched_competition_1(llm: LLM, tok, tasks: list[dict[str, Any]], think: bool) -> dict[str, Any]:
    """Fast batched vLLM execution of Competition 1 (Naked Foundation Model) across all 100 tasks."""
    prompts_txt: list[str] = []
    for item in tasks:
        if item["track"] == "stateful_tool_agents":
            p = f"{TOOL_SYSTEM_INSTRUCTION}\n\nAvailable tools:\n```json\n{json.dumps(WORKFLOW_SCHEMAS, indent=2)}\n```\nUser Objective: {item['prompt']}"
        else:
            p = item["prompt"]
        prompts_txt.append(render_chat(tok, [{"role": "user", "content": p}], think))

    max_toks = 3072 if think else 1536
    t0 = time.perf_counter()
    outputs = llm.generate(prompts_txt, SamplingParams(temperature=0.0, max_tokens=max_toks))
    wall_s = time.perf_counter() - t0

    by_track: dict[str, list[int]] = {}
    out_toks = 0
    in_toks = 0
    for item, out in zip(tasks, outputs):
        tr = item["track"]
        in_toks += len(out.prompt_token_ids)
        out_toks += len(out.outputs[0].token_ids)
        raw = out.outputs[0].text
        vis = strip_thinking(raw)
        if tr == "stateful_tool_agents":
            calls, _ = parse_tool_calls(raw)
            exp_name, exp_args = item["naked_turn1_tool"]
            ok = bool(calls and len(calls) == 1 and calls[0]["name"] == exp_name and calls[0]["arguments"] == exp_args)
        elif tr in ("frontier_math", "memory_spec_qa"):
            ans = extract_canonical_answer(raw)
            ok = verify_math_or_logic(f"FINAL_ANSWER: {ans}", item["expected"]) or verify_math_or_logic(vis, item["expected"])
        elif tr == "frontier_coding":
            ok, _ = verify_python_solution(vis, item["test_harness"])
        elif tr == "adversarial_conservation_json":
            ok = verify_json_candidate(raw, required_keys=item["required_keys"], invariant_fn=item["invariant_fn"]).passed
        else:
            ok = False
        by_track.setdefault(tr, []).append(1 if ok else 0)

    summary_tracks = {k: round(sum(v) / len(v), 4) for k, v in by_track.items()}
    passed = sum(sum(v) for v in by_track.values())
    return {
        "competition": "Competition_1_Naked_Foundation",
        "think": think,
        "total_items": len(tasks),
        "passed": passed,
        "overall_score": round(passed / len(tasks), 4),
        "tracks": summary_tracks,
        "wall_seconds": round(wall_s, 2),
        "input_tokens": in_toks,
        "output_tokens": out_toks,
        "measured_output_tok_per_s": round(out_toks / max(1e-3, wall_s), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--skip-think1-naked", action="store_true", help="Set for models without dual think toggle")
    args = ap.parse_args()

    mem_store, tasks = build_frontier_100_suite()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "gpu_memory_utilization": args.gpu_mem,
        "max_model_len": args.max_model_len,
        "trust_remote_code": False,
        "enable_prefix_caching": True,
    }
    # Allow native quantization (e.g. MXFP4 on openai/gpt-oss-120b) when present; otherwise bf16
    if "gpt-oss" not in args.model.lower() and "awq" not in args.model.lower():
        llm_kwargs["dtype"] = "bfloat16"

    llm = LLM(**llm_kwargs)

    c1_fast = run_batched_competition_1(llm, tok, tasks, think=False)
    print("COMP1_FAST:", json.dumps(c1_fast), flush=True)

    c1_think = None
    if not args.skip_think1_naked:
        c1_think = run_batched_competition_1(llm, tok, tasks, think=True)
        print("COMP1_THINK:", json.dumps(c1_think), flush=True)

    # Competition 2: Complete 5-Engine System using exact same loaded vLLM instance!
    sys_toks = {"in": 0, "out": 0}

    def _vllm_gen(msgs: list[dict[str, str]], think: bool, max_tok: int) -> str:
        rendered = render_chat(tok, msgs, think)
        res = llm.generate([rendered], SamplingParams(temperature=0.0, max_tokens=max_tok), use_tqdm=False)[0]
        sys_toks["in"] += len(res.prompt_token_ids)
        sys_toks["out"] += len(res.outputs[0].token_ids)
        return res.outputs[0].text

    t_sys0 = time.perf_counter()
    c2_sys = evaluate_competition_2_system(_vllm_gen, mem_store, tasks)
    sys_wall = time.perf_counter() - t_sys0
    c2_sys["wall_seconds"] = round(sys_wall, 2)
    c2_sys["input_tokens"] = sys_toks["in"]
    c2_sys["output_tokens"] = sys_toks["out"]
    c2_sys["measured_output_tok_per_s"] = round(sys_toks["out"] / max(1e-3, sys_wall), 1)
    print("COMP2_SYSTEM:", json.dumps(c2_sys), flush=True)

    payload = {
        "schema": "nanolm.frontier_competitions_1_2.v1",
        "model": args.model,
        "competition_1_naked_fast": c1_fast,
        "competition_1_naked_think": c1_think,
        "competition_2_complete_system": c2_sys,
    }
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(payload, indent=2))
    print("SAVED_COMPETITIONS_EVAL", str(out_p), flush=True)


if __name__ == "__main__":
    main()
