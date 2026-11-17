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


def render_chat(tok, messages: list[dict[str, str]], think: bool, is_gpt_oss: bool = False) -> str:
    if is_gpt_oss:
        try:
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="medium" if think else "low",
            )
        except TypeError:
            pass
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=think
        )
    except TypeError:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_batched_competition_1(llm: LLM, tok, tasks: list[dict[str, Any]], think: bool, is_gpt_oss: bool = False) -> dict[str, Any]:
    """Fast batched vLLM execution of Competition 1 (Naked Foundation Model) across all 100 tasks."""
    prompts_txt: list[str] = []
    for item in tasks:
        if item["track"] == "stateful_tool_agents":
            p = f"{TOOL_SYSTEM_INSTRUCTION}\n\nAvailable tools:\n```json\n{json.dumps(WORKFLOW_SCHEMAS, indent=2)}\n```\nUser Objective: {item['prompt']}"
        else:
            p = item["prompt"]
        prompts_txt.append(render_chat(tok, [{"role": "user", "content": p}], think, is_gpt_oss=is_gpt_oss))

    max_toks = 3072 if (think or is_gpt_oss) else 1536
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

    is_gpt_oss = "gpt-oss" in args.model.lower()
    llm = LLM(**llm_kwargs)

    c1_fast = run_batched_competition_1(llm, tok, tasks, think=False, is_gpt_oss=is_gpt_oss)
    print("COMP1_FAST:", json.dumps(c1_fast), flush=True)

    c1_think = None
    if not args.skip_think1_naked:
        c1_think = run_batched_competition_1(llm, tok, tasks, think=True, is_gpt_oss=is_gpt_oss)
        print("COMP1_THINK:", json.dumps(c1_think), flush=True)

    # Competition 2: Complete 5-Engine System with main-thread dynamic micro-batching across all 100 tasks!
    from collections import defaultdict
    from concurrent.futures import Future, ThreadPoolExecutor
    import queue

    sys_toks = {"in": 0, "out": 0}
    req_q: queue.Queue[tuple[str, int, Future[str]]] = queue.Queue()

    def _vllm_gen(msgs: list[dict[str, str]], think: bool, max_tok: int) -> str:
        rendered = render_chat(tok, msgs, think, is_gpt_oss=is_gpt_oss)
        eff_max = max(max_tok, 2048) if is_gpt_oss else max_tok
        fut: Future[str] = Future()
        req_q.put((rendered, eff_max, fut))
        return fut.result()

    t_sys0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as coord:
        c2_fut = coord.submit(evaluate_competition_2_system, _vllm_gen, mem_store, tasks, 50)
        while not c2_fut.done() or not req_q.empty():
            try:
                first = req_q.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.perf_counter() + 0.03
            while len(batch) < 64 and time.perf_counter() < deadline:
                try:
                    batch.append(req_q.get_nowait())
                except queue.Empty:
                    time.sleep(0.004)
            by_mtok: dict[int, list[tuple[str, Future[str]]]] = defaultdict(list)
            for pr, mt, fu in batch:
                by_mtok[mt].append((pr, fu))
            for mt, grp in by_mtok.items():
                try:
                    outs = llm.generate([g[0] for g in grp], SamplingParams(temperature=0.0, max_tokens=mt), use_tqdm=False)
                    for (_, fu), res in zip(grp, outs):
                        sys_toks["in"] += len(res.prompt_token_ids)
                        sys_toks["out"] += len(res.outputs[0].token_ids)
                        fu.set_result(res.outputs[0].text)
                except Exception as exc:
                    for _, fu in grp:
                        if not fu.done():
                            fu.set_exception(exc)
        c2_sys = c2_fut.result()
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
