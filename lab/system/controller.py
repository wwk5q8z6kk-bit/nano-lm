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

import ast
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
from lab.trackb.eval_capabilities import TOOL_SYSTEM_INSTRUCTION, parse_tool_calls, strip_thinking


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

        # 3. COMPUTE_VERIFY_MATH v4: Executable-Specification Voting + Exact Rational State Elimination.
        #    Channel A = analytical think=1; Channel B = exhaustive brute-force/exact-enumeration script
        #    (directly encodes the problem's literal constraints; no closed formulas, NEVER Monte Carlo);
        #    Channel C = efficient exact formula script. Majority vote; on 3-way split prefer B over A/C.
        if route == "COMPUTE_VERIFY_MATH":
            ana_prompt = f"Solve this mathematical problem step-by-step with exact reasoning and put your final answer inside `\\boxed{{}}`:\n\n{prompt}"
            ana_resp = _ask([{"role": "user", "content": ana_prompt}], True, 3584)
            ana_ans = (extract_canonical_answer(ana_resp) or "").strip()

            def _run_channel(instr: str, timeout_s: float) -> tuple[bool, str]:
                nonlocal sbx
                resp = _ask([{"role": "user", "content": instr}], False, 1280)
                sbx += 1
                ok, out = execute_python_computation(strip_thinking(resp), timeout_sec=timeout_s)
                if not ok:
                    rep = _ask(
                        [{"role": "user", "content": f"Your script failed with `{out}`. Fix it so it runs within {int(timeout_s)} seconds using ONLY exact integer/`fractions.Fraction` arithmetic (NO random sampling) and prints ONLY the exact final answer on the last line:\n\n{instr}"}],
                        False, 1280,
                    )
                    sbx += 1
                    ok, out = execute_python_computation(strip_thinking(rep), timeout_sec=timeout_s)
                return ok, (out.splitlines()[-1].strip() if ok and out.strip() else "")

            brute_instr = (
                "Write a Python 3 script that answers the problem below by EXHAUSTIVE BRUTE-FORCE ENUMERATION or "
                "exact linear-system solving with `fractions.Fraction` (enumerate every candidate integer / permutation / "
                "tuple directly from the problem's literal constraints, or for Markov/random-walk/recurrence state transitions "
                "solve the exact linear equations over states with `fractions.Fraction` Gaussian elimination; NEVER use `random` "
                "Monte Carlo sampling or `float` approximations, and do NOT guess closed-form combinatorics formulas). "
                "Respect every qualifier ('second smallest', 'exactly', 'non-negative', 'lowest terms'). "
                "It must finish within 20 seconds and print ONLY the final answer on the last line.\n\n"
                f"Problem:\n{prompt}"
            )
            formula_instr = (
                "Write a short, exact Python 3 script (using ONLY exact `int`, `fractions.Fraction`, and `math` arithmetic—NEVER "
                "`random` Monte Carlo or `float` rounding; read every qualifier like 'second smallest', 'non-negative', or "
                "'lowest terms' carefully) that computes the exact answer and prints ONLY the final answer on the last line via `print(...)`:\n\n"
                f"Problem:\n{prompt}"
            )
            ok_b, val_b = _run_channel(brute_instr, 20.0)
            ok_c, val_c = _run_channel(formula_instr, 5.0)

            votes = [v for v in (ana_ans, val_b if ok_b else "", val_c if ok_c else "") if v]
            trace.append(f"math_v4:A={ana_ans},B={val_b if ok_b else 'ERR'},C={val_c if ok_c else 'ERR'}")
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
                wm.record_fact("math_v4_answer", final_ans)
            return SystemEpisodeResult(
                f"\\boxed{{{final_ans}}}" if final_ans else strip_thinking(ana_resp), final_ans, route, tier,
                calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
            )

        # 4. CODE_VERIFY_REPAIR v4: Differential Brute-Force/Smoke Self-Testing + Safe Best-Candidate Tracking!
        if route == "CODE_VERIFY_REPAIR":
            cur_msgs = [{"role": "user", "content": prompt}]
            resp = _ask(cur_msgs, False, 1536)
            harness = self_test_harness
            if not harness:
                # Synthesize AST-validated smoke & tiny-input brute-force differential checks (avoiding mental-math constants!)
                test_p = (
                    "Write a short Python self-test snippet inside ```python``` to test an implementation of the specification below "
                    "(do NOT redefine the target function/class). Prefer:\n"
                    "1) Calling the target function/class on 4 boundary inputs (empty/0, negative numbers, ties, small corner cases) "
                    "checking return types and obvious base cases (`n=0` or `n=1`), AND\n"
                    "2) Where possible, a 5-line naive brute-force helper `_naive_ref(...)` on TINY inputs (`n <= 5`, coords `-3..5`) "
                    "asserting `target_fn(x) == _naive_ref(x)` instead of guessing large mental-math numbers!\n\n"
                    f"{prompt}"
                )
                t_resp = strip_thinking(_ask([{"role": "user", "content": test_p}], False, 640))
                m = re.search(r"```(?:python)?\s*\n(.*?)```", t_resp, flags=re.DOTALL | re.IGNORECASE)
                cand_asserts = (m.group(1) if m else t_resp).strip()
                if "assert " in cand_asserts and "class " not in cand_asserts:
                    try:
                        ast.parse(cand_asserts)
                        harness = cand_asserts
                    except SyntaxError:
                        harness = ""

            best_code = ""
            best_raw = resp
            for attempt in range(self.max_repairs + 1):
                sbx += 1
                # Always check standalone syntax/compile validity first so we preserve a compiling fallback!
                syn_rep = verify_code_candidate(resp, self_tests="")
                if syn_rep.passed and not best_code:
                    best_code = syn_rep.extracted_answer
                    best_raw = resp
                rep = verify_code_candidate(resp, self_tests=harness) if syn_rep.passed and harness else syn_rep
                trace.append(f"code_try{attempt}:{rep.critique[:60]}")
                if rep.passed:
                    vis = f"```python\n{rep.extracted_answer}\n```"
                    return SystemEpisodeResult(
                        vis, rep.extracted_answer, route, "VERIFIED_COMPUTATION",
                        calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
                    )
                if syn_rep.passed:
                    best_code = syn_rep.extracted_answer
                    best_raw = resp
                if attempt < self.max_repairs:
                    # Attempt 0 repair uses fast think=False (avoids <think> truncation on <4B); Attempt 1 escalates to think=True!
                    use_t = attempt >= 1
                    cur_msgs = [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": strip_thinking(resp) or best_code},
                        {"role": "user", "content": f"Verification Engine Feedback: {rep.critique}"},
                    ]
                    resp = _ask(cur_msgs, use_t, 3584 if use_t else 1536)
            if best_code:
                return SystemEpisodeResult(
                    f"```python\n{best_code}\n```", best_code, route, "UNCERTAIN_OR_FAILED",
                    calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
                )
            return SystemEpisodeResult(
                strip_thinking(best_raw), rep.extracted_answer, route, "UNCERTAIN_OR_FAILED",
                calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
            )

        # 5. SCHEMA_VERIFY_JSON v4: Fast generation + Multi-line/Dict-Safe Python Conservation Synthesis
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
                    py_json_p = (
                        f"Schema Critic found an error in your previous JSON: `{rep.critique}`.\n"
                        "Write a short Python 3 script using `import json` that computes every field accurately "
                        "and prints ONLY `json.dumps(result_dict)` to stdout:\n\n"
                        f"{prompt}"
                    )
                    py_cand = _ask([{"role": "user", "content": py_json_p}], False, 1024)
                    sbx += 1
                    ok_j, out_j = execute_python_computation(strip_thinking(py_cand))
                    if ok_j and out_j.strip():
                        full_s = out_j.strip()
                        if "{" in full_s and "}" in full_s:
                            cand_s = full_s[full_s.find("{") : full_s.rfind("}") + 1]
                            try:
                                json.loads(cand_s)
                                resp = cand_s
                            except Exception:
                                try:
                                    py_obj = ast.literal_eval(cand_s)
                                    resp = json.dumps(py_obj) if isinstance(py_obj, dict) else cand_s
                                except Exception:
                                    resp = cand_s
                        else:
                            resp = full_s.splitlines()[-1].strip()
                    else:
                        use_t = attempt >= 1
                        resp = _ask(
                            [{"role": "user", "content": f"{prompt}\n\nFix error: {rep.critique}. Output ONLY valid JSON."}],
                            use_t, 3072 if use_t else 1280,
                        )
            return SystemEpisodeResult(
                strip_thinking(resp), rep.extracted_answer, route, "UNCERTAIN_OR_FAILED",
                calls, p_chars, o_chars, sbx, mem_q, round((time.perf_counter() - t0) * 1000, 1), trace,
            )

        # 6. AGENTIC_TOOL_LOOP v4: Error-Only Adaptive Thinking + Truncation/Premature-Stop Guards + Grounded State!
        assert tool_env is not None
        schema_txt = json.dumps(tool_env.schemas, indent=2)
        history_lines: list[str] = []
        last_vis = ""
        rejected_premature_stop = False
        terminal_block_kws = ("freeze", "hold", "revok", "forbid", "denied", "compliance", "security")
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
            # Adaptive compute: ONLY invoke deep thinking when recovering from an active non-terminal API error!
            saw_terminal_block = any(any(kw in ev.lower() for kw in terminal_block_kws) for ev in wm.events)
            use_think = bool(wm.has_unresolved_error and not saw_terminal_block)
            resp = _ask([{"role": "user", "content": turn_prompt}], use_think, 3584 if use_think else 1024)
            last_vis = strip_thinking(resp)

            # Guard 1: Recover immediately if <think>/Harmony analysis truncated to empty string or malformed JSON!
            chk_calls, chk_st = parse_tool_calls(last_vis)
            if not last_vis.strip() or chk_calls is None:
                trace.append(f"tool_turn{turn}_json_recovery:{chk_st}")
                rec_prompt = (
                    f"{turn_prompt}\n\n"
                    'IMPORTANT: Respond ONLY with a valid JSON object `{"tool_calls": [{"name": "...", "arguments": {...}}]}` '
                    'or `{"tool_calls": [], "response": "..."}` (no markdown prose outside JSON).'
                )
                resp = _ask([{"role": "user", "content": rec_prompt}], False, 1024)
                last_vis = strip_thinking(resp)

            sbx += 1
            prev_hist_len = len(tool_env.call_history)
            done, obs_list, st = tool_env.step(last_vis)
            new_calls = tool_env.call_history[prev_hist_len:]
            trace.append(f"tool_turn{turn}:think={int(use_think)},done={done},obs={len(obs_list)},st={st}")
            if done:
                # Guard 2: Reject premature stop (at most once) if 0 tools called in plain text OR recoverable error unresolved!
                premature_zero_call = len(tool_env.call_history) == 0 and st == "plain_text_no_tool"
                premature_unresolved = wm.has_unresolved_error and not saw_terminal_block
                if not rejected_premature_stop and (premature_zero_call or premature_unresolved):
                    rejected_premature_stop = True
                    trace.append(f"tool_turn{turn}_premature_stop_rejected")
                    history_lines.append(f"Turn {turn + 1} Assistant: {last_vis[:300]}")
                    history_lines.append(
                        f"Turn {turn + 1} Verification Critic: Objective not yet verified complete "
                        "(recoverable API error or missing initial tool call, and no compliance/security freeze reported). "
                        "Invoke the appropriate recovery/action tool call now."
                    )
                    continue

                try:
                    parsed = json.loads(last_vis[last_vis.find("{") : last_vis.rfind("}") + 1])
                    final_text = str(parsed.get("response", last_vis))
                except Exception:
                    final_text = last_vis
                # Grounded WorkingMemory State Receipt: preserve verified live API IDs/values/freeze codes!
                receipt_bits: list[str] = []
                for fk, fv in wm.facts.items():
                    if str(fv) not in final_text:
                        receipt_bits.append(f"{fk}={fv}")
                if saw_terminal_block:
                    for ev in wm.events:
                        if any(kw in ev.lower() for kw in terminal_block_kws) and ev not in final_text:
                            receipt_bits.append(ev)
                if receipt_bits:
                    final_text = f"{final_text}\n[Verified Working Memory: {'; '.join(receipt_bits)}]"
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
