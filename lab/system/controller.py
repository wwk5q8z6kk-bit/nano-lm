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

        # 3. COMPUTE_VERIFY_MATH v3: Executable-Specification Voting.
        #    Channel A = analytical think=1; Channel B = exhaustive brute-force/exact-enumeration script
        #    (directly encodes the problem's literal constraints; no closed formulas); Channel C = efficient
        #    formula script. Majority vote; on 3-way split prefer B (literal specification) over A/C.
        if route == "COMPUTE_VERIFY_MATH":
            ana_prompt = f"Solve this mathematical problem step-by-step with exact reasoning and put your final answer inside `\\boxed{{}}`:\n\n{prompt}"
            ana_resp = _ask([{"role": "user", "content": ana_prompt}], True, 2560)
            ana_ans = (extract_canonical_answer(ana_resp) or "").strip()

            def _run_channel(instr: str, timeout_s: float) -> tuple[bool, str]:
                nonlocal sbx
                resp = _ask([{"role": "user", "content": instr}], False, 1024)
                sbx += 1
                ok, out = execute_python_computation(strip_thinking(resp), timeout_sec=timeout_s)
                if not ok:
                    rep = _ask(
                        [{"role": "user", "content": f"Your script failed with `{out}`. Fix it so it runs within {int(timeout_s)} seconds and prints ONLY the exact final answer on the last line:\n\n{instr}"}],
                        False, 1024,
                    )
                    sbx += 1
                    ok, out = execute_python_computation(strip_thinking(rep), timeout_sec=timeout_s)
                return ok, (out.splitlines()[-1].strip() if ok and out.strip() else "")

            brute_instr = (
                "Write a Python 3 script that answers the problem below by EXHAUSTIVE BRUTE-FORCE ENUMERATION or "
                "exact linear-system solving with `fractions.Fraction` (enumerate every candidate integer / permutation / "
                "tuple / state directly from the problem's literal constraints; do NOT use any closed-form formula). "
                "Respect every qualifier ('second smallest', 'exactly', 'non-negative', 'lowest terms'). "
                "It must finish within 20 seconds and print ONLY the final answer on the last line.\n\n"
                f"Problem:\n{prompt}"
            )
            formula_instr = (
                "Write a short, exact Python 3 script (preferring exact integer/`fractions.Fraction`/`math` arithmetic; "
                "read every qualifier like 'second smallest', 'non-negative', or 'lowest terms' carefully) "
                "that computes the exact answer and prints ONLY the final answer on the last line via `print(...)`:\n\n"
                f"Problem:\n{prompt}"
            )
            ok_b, val_b = _run_channel(brute_instr, 20.0)
            ok_c, val_c = _run_channel(formula_instr, 5.0)

            votes = [v for v in (ana_ans, val_b if ok_b else "", val_c if ok_c else "") if v]
            trace.append(f"math_v3:A={ana_ans},B={val_b if ok_b else 'ERR'},C={val_c if ok_c else 'ERR'}")
            final_ans, tier = "", "UNCERTAIN_OR_FAILED"
            if votes:
                counts: dict[str, int] = {}
                for v in votes:
                    counts[v] = counts.get(v, 0) + 1
                top_v, top_n = max(counts.items(), key=lambda kv: kv[1])
                if top_n >= 2:
                    final_ans, tier = top_v, ("CERTIFIED_TRIPLE_AGREEMENT" if top_n == 3 else "CERTIFIED_MAJORITY")
                elif ok_b and val_b:
                    final_ans, tier = val_b, "BRUTE_FORCE_SPECIFICATION"
                elif ana_ans:
                    final_ans, tier = ana_ans, "ANALYTICAL_UNVERIFIED"
                else:
                    final_ans, tier = votes[0], "EXECUTABLE_UNVERIFIED"
            if final_ans:
                wm.record_fact("math_v3_answer", final_ans)
            return SystemEpisodeResult(
                f"\\boxed{{{final_ans}}}" if final_ans else strip_thinking(ana_resp), final_ans, route, tier,
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

        # 6. AGENTIC_TOOL_LOOP v2: Epistemic Belief-Revision WorkingMemory + Error-Triggered Thinking!
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
                "Read Working Memory & API error messages carefully: never reuse an invalidated/stale parameter, "
                "clamp numeric parameters if an API specifies a safety bound, and halt immediately without retrying "
                "if an API reports a compliance/permission freeze. "
                "When ALL required tool steps are verified complete (or halted due to freeze), "
                'output `{"tool_calls": [], "response": "<final grounded summary with exact IDs and final values>"}`.\n\n'
                f"Available tools:\n```json\n{schema_txt}\n```{wm_block}{hist_block}\nUser Objective: {prompt}"
            )
            # Adaptive compute escalation: Turn 0 happy path uses fast think=False; recovery/multi-step uses think=True!
            use_think = bool(wm.has_unresolved_error or turn >= 1)
            resp = _ask([{"role": "user", "content": turn_prompt}], use_think, 3584 if use_think else 768)
            last_vis = strip_thinking(resp)
            sbx += 1
            prev_hist_len = len(tool_env.call_history)
            done, obs_list, st = tool_env.step(last_vis)
            new_calls = tool_env.call_history[prev_hist_len:]
            trace.append(f"tool_turn{turn}:think={int(use_think)},done={done},obs={len(obs_list)},st={st}")
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
            for idx_ob, ob in enumerate(obs_list):
                c_args = new_calls[idx_ob]["arguments"] if idx_ob < len(new_calls) and isinstance(new_calls[idx_ob].get("arguments"), dict) else {}
                if ob.get("ok"):
                    wm.has_unresolved_error = False
                    for k, v in ob.items():
                        if k not in ("tool", "ok"):
                            wm.record_fact(k, v)
                else:
                    wm.revise_on_error(c_args, str(ob.get("error", "APIError")), str(ob.get("message", "")))

        return SystemEpisodeResult(
            last_vis, last_vis, route, "UNCERTAIN_OR_FAILED",
            calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
        )
