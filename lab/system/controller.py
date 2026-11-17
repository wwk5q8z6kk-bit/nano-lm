"""Adaptive Computation Controller & Unified 5-Engine `NanoLMSystem` (Competition 2).

Coordinates:
  1. Adaptive Computation Controller (decides FAST_DIRECT vs RETRIEVE_VERIFY vs COMPUTE_VERIFY vs
     CODE_REPAIR vs AGENTIC_TOOL_LOOP using query structure + Pass-1 predictive confidence),
  2. Reasoning Engine (bounded thinking + anti-circularity intervention),
  3. Memory Engine (WorkingMemory + confidence-gated PersistentKnowledgeStore),
  4. Verification Engine (sandboxed execution, independent solver reconciliation, schema critics),
  5. Tool Execution Engine (isolated Python computation + StatefulToolEnvironment).

Every episode logs exact `llm_calls`, `output_tokens_est`, `sandbox_runs`, `memory_retrievals`,
`route_taken`, and `epistemic_tier` so system intelligence is never conflated with naked weights.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from lab.system.memory_engine import PersistentKnowledgeStore, WorkingMemory
from lab.system.tool_engine import StatefulToolEnvironment, execute_python_computation
from lab.system.verification_engine import (
    detect_circular_reasoning,
    extract_canonical_answer,
    verify_code_candidate,
    verify_json_candidate,
)
from lab.trackb.eval_capabilities import TOOL_SYSTEM_INSTRUCTION, strip_thinking


@dataclass
class SystemEpisodeResult:
    final_response: str
    extracted_answer: str
    route_taken: str
    epistemic_tier: str
    llm_calls: int
    prompt_chars: int
    output_chars: int
    sandbox_runs: int
    memory_retrievals: int
    wall_ms: float
    trace: list[str] = field(default_factory=list)


# Generic generate signature: fn(messages: list[dict[str, str]], think: bool, max_tokens: int) -> str
GenerateFn = Callable[[list[dict[str, str]], bool, int], str]


class NanoLMSystem:
    """Model-agnostic 5-engine system wrapper enabling identical-condition Competition 2 evaluation."""

    def __init__(
        self,
        generate_fn: GenerateFn,
        knowledge_store: PersistentKnowledgeStore | None = None,
        max_repairs: int = 2,
        max_tool_turns: int = 7,
    ):
        self.generate = generate_fn
        self.knowledge = knowledge_store or PersistentKnowledgeStore()
        self.max_repairs = max_repairs
        self.max_tool_turns = max_tool_turns

    def select_route(self, prompt: str, task_hint: str | None = None, has_tool_env: bool = False) -> str:
        """Adaptive Computation Controller: map task structure & uncertainty cues to minimal sufficient engine."""
        if has_tool_env or task_hint == "tool_workflow":
            return "AGENTIC_TOOL_LOOP"
        if task_hint in ("code", "coding") or ("```python" in prompt and "def " in prompt) or "Implement " in prompt and "def " in prompt:
            return "CODE_VERIFY_REPAIR"
        if task_hint == "memory_qa":
            return "RETRIEVE_VERIFY"
        # Check whether persistent memory has a high-confidence match BEFORE deciding on retrieval
        if len(self.knowledge.docs) > 0:
            probe = self.knowledge.retrieve(prompt, top_k=1)
            if probe.triggered and probe.confidence >= 0.34:
                return "RETRIEVE_VERIFY"
        if task_hint in ("math_hard", "math") or any(
            kw in prompt.lower()
            for kw in ("\\boxed", "modulo", "remainder", "combinatorics", "probability", "diophantine", "eigenvalue", "derangement", "prime")
        ):
            return "COMPUTE_VERIFY_MATH"
        if task_hint == "json_strict" or ("json" in prompt.lower() and ("only" in prompt.lower() or "exact keys" in prompt.lower())):
            return "SCHEMA_VERIFY_JSON"
        return "FAST_DIRECT"

    def solve(
        self,
        prompt: str,
        task_hint: str | None = None,
        tool_env: StatefulToolEnvironment | None = None,
        self_test_harness: str = "",
        json_required_keys: set[str] | None = None,
        json_invariant_fn: Callable[[dict[str, Any]], tuple[bool, str]] | None = None,
    ) -> SystemEpisodeResult:
        t0 = time.perf_counter()
        wm = WorkingMemory()
        route = self.select_route(prompt, task_hint=task_hint, has_tool_env=(tool_env is not None))
        calls = 0
        p_chars = 0
        o_chars = 0
        sbx = 0
        mem_q = 0
        trace: list[str] = [f"route={route}"]

        def _ask(msgs: list[dict[str, str]], think: bool, max_tok: int) -> str:
            nonlocal calls, p_chars, o_chars
            calls += 1
            p_chars += sum(len(m["content"]) for m in msgs)
            out = self.generate(msgs, think, max_tok)
            o_chars += len(out)
            # Anti-circularity intervention: if thinking looped/truncated, immediately re-issue concise think=False!
            if think and detect_circular_reasoning(out):
                trace.append("anti_circular_fallback_triggered")
                calls += 1
                concise_msgs = msgs + [{"role": "user", "content": "Stop repeating steps. Give your concise final derivation and answer directly."}]
                p_chars += len(concise_msgs[-1]["content"])
                out = self.generate(concise_msgs, False, min(1024, max_tok))
                o_chars += len(out)
            return out

        # 1. FAST_DIRECT: Easy/general queries solved in a single fast pass (think=0)!
        if route == "FAST_DIRECT":
            resp = _ask([{"role": "user", "content": prompt}], False, 1024)
            vis = strip_thinking(resp)
            return SystemEpisodeResult(
                vis, extract_canonical_answer(resp), route, "PLAUSIBLE_UNVERIFIED",
                calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
            )

        # 2. RETRIEVE_VERIFY: Confidence-gated memory lookup + conflict/abstention verification
        if route == "RETRIEVE_VERIFY":
            mem_q += 1
            ret = self.knowledge.retrieve(prompt, top_k=3)
            trace.append(f"mem_triggered={ret.triggered},conf={ret.confidence},conflict={ret.has_conflict_or_deprecation}")
            if not ret.triggered:
                aug_prompt = (
                    "No verified document in persistent memory matched this query above the confidence threshold.\n"
                    "If the question asks for a specific private/unseen specification fact not in general math/science, "
                    "output `FINAL_ANSWER: INSUFFICIENT_EVIDENCE`; otherwise answer carefully.\n\n"
                    f"Question: {prompt}"
                )
                resp = _ask([{"role": "user", "content": aug_prompt}], False, 768)
                tier = "UNCERTAIN_ABSTAIN" if "INSUFFICIENT_EVIDENCE" in resp else "PLAUSIBLE_UNVERIFIED"
            else:
                conflict_note = (
                    "IMPORTANT: Retrieved documents include a DEPRECATED/SUPERSEDED revision! "
                    "Use ONLY the highest-version `status=ACTIVE` specification and ignore deprecated values.\n"
                    if ret.has_conflict_or_deprecation
                    else ""
                )
                aug_prompt = (
                    "Use ONLY the retrieved persistent-memory evidence below to answer. "
                    "If the retrieved evidence does NOT contain the exact answer requested, output `FINAL_ANSWER: INSUFFICIENT_EVIDENCE`.\n"
                    f"{conflict_note}\nRetrieved Evidence:\n{ret.formatted_evidence}\n\nQuestion: {prompt}"
                )
                resp = _ask([{"role": "user", "content": aug_prompt}], False, 768)
                tier = "UNCERTAIN_ABSTAIN" if "INSUFFICIENT_EVIDENCE" in resp else "SUPPORTED_EVIDENCE"
            vis = strip_thinking(resp)
            return SystemEpisodeResult(
                vis, extract_canonical_answer(resp), route, tier,
                calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
            )

        # 3. COMPUTE_VERIFY_MATH: Analytical derivation + independent Python verification & reconciliation!
        if route == "COMPUTE_VERIFY_MATH":
            # Step A: Ask model to write a self-contained Python solver printing the exact answer first!
            py_prompt = (
                "Write a short, exact Python 3 script (using `fractions`, `math`, `itertools`, or `sympy` if installed, "
                "preferring exact integer/rational standard-library arithmetic) that computes the exact answer to the problem "
                "below and prints ONLY the final answer on the last line via `print(...)`.\n\n"
                f"Problem:\n{prompt}"
            )
            py_resp = _ask([{"role": "user", "content": py_prompt}], False, 896)
            sbx += 1
            ok_py, py_out = execute_python_computation(strip_thinking(py_resp))
            trace.append(f"math_py_ok={ok_py},out={py_out[:60]}")
            if ok_py:
                val_line = py_out.splitlines()[-1].strip()
                wm.record_fact("python_verified_result", val_line)
                synth_prompt = (
                    f"{prompt}\n\n"
                    f"[Verification Engine Note: Independent Python execution produced `{val_line}`. "
                    "Verify this result analytically step-by-step; if your independent check confirms it, "
                    "present the concise proof and final answer in `\\boxed{...}` (or requested format). "
                    "If you find a bug in the script's formulation, explain and output the corrected final answer.]"
                )
                final_r = _ask([{"role": "user", "content": synth_prompt}], False, 1024)
                ans = extract_canonical_answer(final_r) or val_line
                tier = "VERIFIED_COMPUTATION" if val_line in final_r else "PLAUSIBLE_UNVERIFIED"
                return SystemEpisodeResult(
                    strip_thinking(final_r), ans, route, tier,
                    calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
                )
            # Fallback if Python script errored: repair script once and re-execute!
            rep_py = _ask(
                [{"role": "user", "content": f"Your Python solver failed with `{py_out}`. Fix the Python 3 script so it prints ONLY the exact answer:\n\n{prompt}"}],
                False, 1024,
            )
            sbx += 1
            ok_py2, py_out2 = execute_python_computation(strip_thinking(rep_py))
            if ok_py2:
                val2 = py_out2.splitlines()[-1].strip()
                return SystemEpisodeResult(
                    f"\\boxed{{{val2}}}", val2, route, "VERIFIED_COMPUTATION",
                    calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
                )
            final_r = _ask([{"role": "user", "content": f"Solve carefully and put the final answer in `\\boxed{{}}`:\n{prompt}"}], True, 2560)
            return SystemEpisodeResult(
                strip_thinking(final_r), extract_canonical_answer(final_r), route, "PLAUSIBLE_UNVERIFIED",
                calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
            )

        # 4. CODE_VERIFY_REPAIR: Generate -> synthesize self-tests if none provided -> execute -> repair!
        if route == "CODE_VERIFY_REPAIR":
            cur_msgs = [{"role": "user", "content": prompt}]
            resp = _ask(cur_msgs, False, 1280)
            harness = self_test_harness
            if not harness:
                # Self-generate deterministic edge-case assertions from the problem specification!
                test_p = (
                    "Write 4 concise Python `assert` statements testing normal and tricky edge cases "
                    "(empty inputs, negative numbers, ties, boundary indices) for the specification below. "
                    "Output ONLY Python `assert ...` lines inside ```python``` (do NOT redefine the function):\n\n"
                    f"{prompt}"
                )
                t_resp = strip_thinking(_ask([{"role": "user", "content": test_p}], False, 512))
                m = re.search(r"```(?:python)?\s*\n(.*?)```", t_resp, flags=re.DOTALL | re.IGNORECASE)
                cand_asserts = (m.group(1) if m else t_resp).strip()
                if "assert " in cand_asserts and "def " not in cand_asserts:
                    harness = cand_asserts

            for attempt in range(self.max_repairs + 1):
                sbx += 1
                rep = verify_code_candidate(resp, self_tests=harness)
                trace.append(f"code_try{attempt}:{rep.critique[:60]}")
                if rep.passed:
                    vis = f"```python\n{rep.extracted_answer}\n```"
                    return SystemEpisodeResult(
                        vis, rep.extracted_answer, route, "VERIFIED_COMPUTATION",
                        calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
                    )
                if attempt < self.max_repairs:
                    cur_msgs = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": strip_thinking(resp)},
                        {"role": "user", "content": f"Verification Engine Feedback: {rep.critique}"},
                    ]
                    resp = _ask(cur_msgs, True, 2560)
            return SystemEpisodeResult(
                strip_thinking(resp), rep.extracted_answer, route, "UNCERTAIN_OR_FAILED",
                calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
            )

        # 5. SCHEMA_VERIFY_JSON: Fast generation + Python-assisted arithmetic/schema critic & repair
        if route == "SCHEMA_VERIFY_JSON":
            cur_msgs = [{"role": "user", "content": prompt}]
            resp = _ask(cur_msgs, False, 896)
            for attempt in range(self.max_repairs + 1):
                rep = verify_json_candidate(resp, required_keys=json_required_keys, invariant_fn=json_invariant_fn)
                trace.append(f"json_try{attempt}:{rep.critique[:60]}")
                if rep.passed:
                    return SystemEpisodeResult(
                        rep.extracted_answer, rep.extracted_answer, route, "VERIFIED_SCHEMA",
                        calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
                    )
                if attempt < self.max_repairs:
                    # Let the model emit a Python script printing `json.dumps(...)` so arithmetic conservation is exact!
                    py_json_p = (
                        f"Schema Critic found an error in your previous JSON: `{rep.critique}`.\n"
                        "Write a short Python 3 script using `import json` that computes every field accurately "
                        "and prints ONLY `json.dumps(result_dict)` to stdout:\n\n"
                        f"{prompt}"
                    )
                    py_cand = _ask([{"role": "user", "content": py_json_p}], False, 1024)
                    sbx += 1
                    ok_j, out_j = execute_python_computation(strip_thinking(py_cand))
                    if ok_j:
                        resp = out_j.splitlines()[-1].strip()
                    else:
                        resp = _ask(
                            [{"role": "user", "content": f"{prompt}\n\nFix error: {rep.critique}. Output ONLY valid JSON."}],
                            True, 2560,
                        )
            return SystemEpisodeResult(
                strip_thinking(resp), rep.extracted_answer, route, "UNCERTAIN_OR_FAILED",
                calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
            )

        # 6. AGENTIC_TOOL_LOOP: Multi-step stateful tool orchestration with WorkingMemory & error recovery
        assert tool_env is not None
        schema_txt = json.dumps(tool_env.schemas, indent=2)
        history_lines: list[str] = []
        last_vis = ""
        for turn in range(self.max_tool_turns):
            wm_txt = wm.render()
            wm_block = f"\nWorking Memory:\n{wm_txt}\n" if wm_txt else ""
            hist_block = ("\nInteraction Log:\n" + "\n".join(history_lines) + "\n") if history_lines else ""
            turn_prompt = (
                f"{TOOL_SYSTEM_INSTRUCTION}\n"
                "When ALL required tool steps are verified complete (or if an API returns a permanent permission/revocation stop), "
                'output `{"tool_calls": [], "response": "<final grounded summary with exact IDs/values>"}`.\n\n'
                f"Available tools:\n```json\n{schema_txt}\n```{wm_block}{hist_block}\nUser Objective: {prompt}"
            )
            resp = _ask([{"role": "user", "content": turn_prompt}], False, 768)
            last_vis = strip_thinking(resp)
            sbx += 1
            done, obs_list, st = tool_env.step(last_vis)
            trace.append(f"tool_turn{turn}:done={done},obs={len(obs_list)},st={st}")
            if done:
                try:
                    parsed = json.loads(last_vis[last_vis.find("{") : last_vis.rfind("}") + 1])
                    final_text = str(parsed.get("response", last_vis))
                except Exception:
                    final_text = last_vis
                tier = "VERIFIED_COMPUTATION" if any(o.get("observation", {}).get("ok") for o in tool_env.call_history) else "SUPPORTED_EVIDENCE"
                return SystemEpisodeResult(
                    final_text, final_text, route, tier,
                    calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
                )
            history_lines.append(f"Turn {turn + 1} Assistant: {last_vis}")
            history_lines.append(f"Turn {turn + 1} Tool Observation: {json.dumps(obs_list)}")
            for ob in obs_list:
                if ob.get("ok"):
                    for k, v in ob.items():
                        if k not in ("tool", "ok"):
                            wm.record_fact(k, v)
                else:
                    wm.record_event(f"ERROR on {ob.get('tool')}: {ob.get('error')} - {ob.get('message')}")

        return SystemEpisodeResult(
            last_vis, last_vis, route, "UNCERTAIN_OR_FAILED",
            calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
        )
