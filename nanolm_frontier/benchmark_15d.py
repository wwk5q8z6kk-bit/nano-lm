"""15-Dimension Objective Benchmark & Deterministic Verifier Suite for NanoLM (<= 4B).

Zero subjective LLM judges. All 15 dimensions are graded strictly by sandboxed
code execution, topological DAG verification, JSON invariant checks, regex/constraint
checkers, exact counterfactual evaluation, calibrated abstention checks, and runtime
memory/throughput telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import random
import re
from typing import Any

from nanolm_posttrain.eval_suite import (
    TaskVerdict,
    _extract_fenced_or_raw,
    _mermaid_chain,
    _verify_coding,
    _verify_summarization,
)


class BenchmarkDim(str, Enum):
    CODING = "coding"
    MATH_LOGIC = "math_logic"
    SUMMARIZATION = "summarization"
    STRUCTURED_JSON = "structured_json"
    DIAGRAMS_CHARTS = "diagrams_charts"
    LONG_CONTEXT = "long_context"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    INSTRUCTION_FOLLOWING = "instruction_following"
    TOOL_USE = "tool_use"
    PLANNING_DECOMPOSITION = "planning_decomposition"
    OOD_ROBUSTNESS = "ood_robustness"
    HALLUCINATION_RESISTANCE = "hallucination_resistance"
    MEMORY_RETENTION = "memory_retention"
    INFERENCE_EFFICIENCY = "inference_efficiency"


CAPABILITY_DIMENSIONS: tuple[BenchmarkDim, ...] = tuple(
    d for d in BenchmarkDim if d != BenchmarkDim.INFERENCE_EFFICIENCY
)


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    task_id: str
    dim: BenchmarkDim
    difficulty: str  # "easy" | "medium" | "hard"
    system_prompt: str
    user_prompt: str
    spec: dict[str, Any]
    max_new_tokens: int = 256


_MERMAID_HEADERS = (
    "flowchart ",
    "graph ",
    "sequenceDiagram",
    "stateDiagram-v2",
    "stateDiagram",
    "classDiagram",
)


def _extract_final_answer(response: str) -> str:
    matches = re.findall(r"FINAL_ANSWER\s*:\s*([^\n]+)", response, flags=re.IGNORECASE)
    if matches:
        return matches[-1].strip().strip("`").rstrip(".").strip()
    lines = [ln.strip() for ln in response.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1].strip("`").rstrip(".").strip()


def _verify_exact_answer(response: str, spec: dict[str, Any]) -> TaskVerdict:
    aliases = {str(a).strip().casefold().replace(",", "") for a in spec["accepted_answers"]}
    cand_raw = _extract_final_answer(response)
    cand = cand_raw.casefold().replace(",", "")
    if cand in aliases:
        return TaskVerdict(True, 1.0, "exact_answer_match", {"candidate": cand_raw})
    return TaskVerdict(False, 0.0, f"answer_mismatch:{cand_raw!r}", {"candidate": cand_raw})


def _verify_json_invariant(response: str, spec: dict[str, Any]) -> TaskVerdict:
    raw = _extract_fenced_or_raw(response)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Try last JSON object substring as salvage check only if balanced
        m = re.findall(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return TaskVerdict(False, 0.0, f"invalid_json:{exc.msg}", {})
        try:
            parsed = json.loads(m[-1])
        except json.JSONDecodeError:
            return TaskVerdict(False, 0.0, f"invalid_json:{exc.msg}", {})

    if not isinstance(parsed, dict):
        return TaskVerdict(False, 0.0, "json_not_object", {})

    expected: dict[str, Any] = spec["expected_json"]
    if set(parsed.keys()) != set(expected.keys()):
        hits = sum(1 for k, v in expected.items() if parsed.get(k) == v)
        return TaskVerdict(False, round(0.5 * hits / max(len(expected), 1), 4), f"schema_keys_mismatch:{sorted(parsed.keys())}", {})

    for k, exp_val in expected.items():
        got = parsed.get(k)
        if isinstance(exp_val, float) and isinstance(got, (int, float)):
            if abs(float(got) - exp_val) > 1e-3:
                return TaskVerdict(False, 0.4, f"float_mismatch:{k}", {})
        elif got != exp_val:
            return TaskVerdict(False, 0.5, f"value_mismatch:{k}:{got!r}!={exp_val!r}", {})
    return TaskVerdict(True, 1.0, "exact_json_match", {})


def _verify_diagram_or_chart(response: str, spec: dict[str, Any]) -> TaskVerdict:
    subkind = spec["subkind"]
    body = _extract_fenced_or_raw(response)
    if subkind == "chart_json":
        return _verify_json_invariant(body, spec)

    lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("%%")]
    if not lines or not any(lines[0].startswith(h) for h in _MERMAID_HEADERS):
        return TaskVerdict(False, 0.0, "invalid_mermaid_header", {})
    if not lines[0].startswith(str(spec["expected_header"])):
        return TaskVerdict(False, 0.2, f"wrong_diagram_header:{lines[0]}", {})
    for ln in lines[1:]:
        for bracket_content in re.findall(r"\b[A-Za-z0-9_]+\[(?!\")([^\]]+)\]", ln):
            if "(" in bracket_content or ")" in bracket_content:
                return TaskVerdict(False, 0.3, "unquoted_special_chars_in_label", {"line": ln})
    return _mermaid_chain(lines[1:], list(spec["required_tokens"]), int(spec["min_edges"]))


def _verify_instruction_following(response: str, spec: dict[str, Any]) -> TaskVerdict:
    txt = response.strip()
    if not txt:
        return TaskVerdict(False, 0.0, "empty_response", {})
    checks_passed = 0
    total_checks = 0
    failures: list[str] = []

    if "exact_bullets" in spec:
        total_checks += 1
        bullets = [ln for ln in txt.splitlines() if re.match(r"^\s*[-*]\s+\S+", ln)]
        if len(bullets) == int(spec["exact_bullets"]):
            checks_passed += 1
        else:
            failures.append(f"bullets={len(bullets)}!={spec['exact_bullets']}")

    if "max_words" in spec:
        total_checks += 1
        wc = len(txt.split())
        if wc <= int(spec["max_words"]):
            checks_passed += 1
        else:
            failures.append(f"words={wc}>{spec['max_words']}")

    if "min_words" in spec:
        total_checks += 1
        wc = len(txt.split())
        if wc >= int(spec["min_words"]):
            checks_passed += 1
        else:
            failures.append(f"words={wc}<{spec['min_words']}")

    for fw in spec.get("forbidden_words", ()):
        total_checks += 1
        if fw.casefold() not in txt.casefold():
            checks_passed += 1
        else:
            failures.append(f"forbidden_present:{fw}")

    for rw in spec.get("required_words", ()):
        total_checks += 1
        if rw.casefold() in txt.casefold():
            checks_passed += 1
        else:
            failures.append(f"required_missing:{rw}")

    if spec.get("require_all_lowercase"):
        total_checks += 1
        if txt == txt.lower():
            checks_passed += 1
        else:
            failures.append("uppercase_found")

    if "required_end_token" in spec:
        total_checks += 1
        if txt.endswith(str(spec["required_end_token"])):
            checks_passed += 1
        else:
            failures.append("missing_end_token")

    passed = checks_passed == total_checks and total_checks > 0
    score = checks_passed / max(total_checks, 1)
    return TaskVerdict(
        passed,
        round(score, 4),
        "all_constraints_satisfied" if passed else ";".join(failures),
        {"passed": checks_passed, "total": total_checks},
    )


def _verify_tool_use(response: str, spec: dict[str, Any]) -> TaskVerdict:
    raw = _extract_fenced_or_raw(response)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return TaskVerdict(False, 0.0, f"invalid_tool_json:{exc.msg}", {})

    # Accepts single call object or {"calls": [...]}
    expected_calls: list[dict[str, Any]] = list(spec["expected_calls"])
    if len(expected_calls) == 1 and isinstance(parsed, dict) and "tool" in parsed:
        got_calls = [parsed]
    elif isinstance(parsed, dict) and isinstance(parsed.get("calls"), list):
        got_calls = parsed["calls"]
    elif isinstance(parsed, list):
        got_calls = parsed
    else:
        return TaskVerdict(False, 0.0, "unexpected_tool_call_shape", {})

    if len(got_calls) != len(expected_calls):
        return TaskVerdict(False, 0.2, f"call_count:{len(got_calls)}!={len(expected_calls)}", {})

    matched = 0
    for got, exp in zip(got_calls, expected_calls):
        if isinstance(got, dict) and got.get("tool") == exp["tool"] and got.get("args") == exp["args"]:
            matched += 1
    passed = matched == len(expected_calls)
    return TaskVerdict(
        passed,
        round(matched / max(len(expected_calls), 1), 4),
        "verified_tool_calls" if passed else f"tool_call_mismatch:{matched}/{len(expected_calls)}",
        {},
    )


def _verify_planning_dag(response: str, spec: dict[str, Any]) -> TaskVerdict:
    raw = _extract_fenced_or_raw(response)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return TaskVerdict(False, 0.0, f"invalid_plan_json:{exc.msg}", {})
    if not isinstance(parsed, dict):
        return TaskVerdict(False, 0.0, "plan_not_object", {})

    order = parsed.get("order")
    makespan = parsed.get("makespan")
    nodes: dict[str, dict[str, Any]] = spec["dag"]
    if not isinstance(order, list) or not all(isinstance(item, str) for item in order):
        return TaskVerdict(False, 0.2, "invalid_task_names", {})
    if len(order) != len(nodes) or set(order) != set(nodes):
        return TaskVerdict(False, 0.2, "invalid_task_set_in_order", {})

    pos = {name: i for i, name in enumerate(order)}
    for name, info in nodes.items():
        for dep in info["deps"]:
            if pos[dep] > pos[name]:
                return TaskVerdict(False, 0.3, f"dependency_violation:{dep}->{name}", {})

    # Compute exact earliest finish time (critical path) of DAG
    finish: dict[str, int] = {}
    for name in order:
        deps = nodes[name]["deps"]
        start_t = max((finish[d] for d in deps), default=0)
        finish[name] = start_t + int(nodes[name]["dur"])
    expected_makespan = max(finish.values(), default=0)
    if makespan != expected_makespan:
        return TaskVerdict(False, 0.7, f"wrong_makespan:{makespan}!={expected_makespan}", {})
    return TaskVerdict(True, 1.0, "verified_dag_schedule_and_makespan", {"makespan": expected_makespan})


def _verify_hallucination_resistance(response: str, spec: dict[str, Any]) -> TaskVerdict:
    low = response.strip().casefold()
    markers = ("abstain", "cannot be determined", "insufficient information", "not provided", "contradict", "impossible")
    has_abstention = any(m in low for m in markers)
    forbidden_claims: list[str] = list(spec.get("forbidden_confabulations", ()))
    assertions = low
    for premise_pattern in spec.get("citable_claims", ()):
        assertions = re.sub(premise_pattern, " ", assertions)
    confabulated = any(re.search(r"(?<!\w)" + re.escape(fc.casefold()) + r"(?!\w)", assertions)
                       for fc in forbidden_claims)
    if has_abstention and not confabulated:
        return TaskVerdict(True, 1.0, "calibrated_abstention", {})
    if confabulated:
        return TaskVerdict(False, 0.0, "confabulated_unsupported_answer", {})
    return TaskVerdict(False, 0.2, "failed_to_abstain_explicitly", {})


def verify_benchmark_response(task: BenchmarkTask, response: str) -> TaskVerdict:
    if task.dim == BenchmarkDim.CODING:
        return _verify_coding(response, task.spec)
    if task.dim in (BenchmarkDim.MATH_LOGIC, BenchmarkDim.GENERAL_KNOWLEDGE, BenchmarkDim.MULTI_STEP_REASONING, BenchmarkDim.OOD_ROBUSTNESS):
        return _verify_exact_answer(response, task.spec)
    if task.dim == BenchmarkDim.SUMMARIZATION:
        return _verify_summarization(response, task.spec)
    if task.dim in (BenchmarkDim.STRUCTURED_JSON, BenchmarkDim.LONG_CONTEXT, BenchmarkDim.MEMORY_RETENTION):
        return _verify_json_invariant(response, task.spec)
    if task.dim == BenchmarkDim.DIAGRAMS_CHARTS:
        return _verify_diagram_or_chart(response, task.spec)
    if task.dim == BenchmarkDim.INSTRUCTION_FOLLOWING:
        return _verify_instruction_following(response, task.spec)
    if task.dim == BenchmarkDim.TOOL_USE:
        return _verify_tool_use(response, task.spec)
    if task.dim == BenchmarkDim.PLANNING_DECOMPOSITION:
        return _verify_planning_dag(response, task.spec)
    if task.dim == BenchmarkDim.HALLUCINATION_RESISTANCE:
        return _verify_hallucination_resistance(response, task.spec)
    raise ValueError(f"Unsupported verifiable dimension: {task.dim}")


def _long_haystack(needles: list[tuple[int, str]], total_blocks: int = 80, topic: str = "seismic array log") -> str:
    nmap = dict(needles)
    chunks: list[str] = []
    for i in range(total_blocks):
        if i in nmap:
            chunks.append(f"[Block {i:02d}] {nmap[i]}")
        else:
            chunks.append(
                f"[Block {i:02d}] Background {topic} telemetry frame {4000 + i}: sensor bank {i % 7} "
                f"operated cleanly inside baseline thermal envelope 22.{i % 9} C with zero interrupts or overrides."
            )
    return "\n".join(chunks)


def build_benchmark_15d_suite() -> tuple[BenchmarkTask, ...]:
    """Build 56 held-out verifiable evaluation tasks (4 per capability dim across 14 capability dims)."""

    long_doc_a = _long_haystack(
        [
            (7, "Telemetry node NORTH-14 reported latency 42 ms and owner 'Vera'."),
            (39, "Telemetry node SOUTH-27 reported latency 19 ms and owner 'Kiran'."),
            (68, "Audit erratum: ignore NORTH-14's earlier latency; revised latency for NORTH-14 is 27 ms."),
        ],
        total_blocks=90,
        topic="distributed packet trace",
    )
    long_doc_b = _long_haystack(
        [
            (11, "Rack alpha allocated 120 GPU hours on Monday under ticket TKT-901."),
            (48, "Rack alpha allocated 85 additional GPU hours on Wednesday under ticket TKT-901."),
            (79, "Refund credit: 40 GPU hours returned on Thursday for ticket TKT-901."),
        ],
        total_blocks=90,
        topic="cluster accounting ledger",
    )
    long_doc_c = _long_haystack(
        [
            (5, "Turbine pair P1 operates at 310 RPM with coolant flow 18 L/s."),
            (44, "Valve V8 directs 100% of secondary coolant to Turbine pair P2 running at 440 RPM."),
            (74, "Contradiction resolution rule: active RPM reported must be the minimum across active pairs."),
        ],
        total_blocks=90,
        topic="hydroelectric governor scan",
    )
    long_doc_d = _long_haystack(
        [
            (9, "Courier flight CF-302 carried 18 crates totaling 540 kg."),
            (51, "Waypoint inspection offloaded 5 crates weighing 165 kg total in Oslo."),
            (83, "Security seal code updated at landing to 'BOREAL-719'."),
        ],
        total_blocks=90,
        topic="freight manifest telemetry",
    )

    tasks: tuple[BenchmarkTask, ...] = (
        # 1. CODING (4)
        BenchmarkTask(
            "cod-01",
            BenchmarkDim.CODING,
            "easy",
            "Return only valid Python inside a ```python``` block. No prose.",
            "Write `run_length_encode(s: str) -> list[tuple[str, int]]` encoding consecutive identical chars.",
            {"assertions": [
                "assert run_length_encode('') == []",
                "assert run_length_encode('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]",
                "assert run_length_encode('xyz') == [('x', 1), ('y', 1), ('z', 1)]",
            ]},
        ),
        BenchmarkTask(
            "cod-02",
            BenchmarkDim.CODING,
            "medium",
            "Return only valid Python inside a ```python``` block. No prose.",
            "Write `eval_rpn(tokens: list[str]) -> int` evaluating Reverse Polish Notation with integers and `+`, `-`, `*`, `/` (truncating division toward zero using `int(a / b)`).",
            {"assertions": [
                "assert eval_rpn(['2', '1', '+', '3', '*']) == 9",
                "assert eval_rpn(['4', '13', '5', '/', '+']) == 6",
                "assert eval_rpn(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']) == 22",
                "assert eval_rpn(['-7', '2', '/']) == -3",
            ]},
        ),
        BenchmarkTask(
            "cod-03",
            BenchmarkDim.CODING,
            "medium",
            "Return only valid Python inside a ```python``` block. No prose.",
            "Write `kth_smallest_bst(values: list[int], k: int) -> int` that builds or treats distinct ints and returns the 1-indexed k-th smallest element without modifying caller's list.",
            {"assertions": [
                "src = [7, 2, 9, 4, 1]",
                "assert kth_smallest_bst(src, 3) == 4",
                "assert src == [7, 2, 9, 4, 1]",
                "assert kth_smallest_bst([10], 1) == 10",
            ]},
        ),
        BenchmarkTask(
            "cod-04",
            BenchmarkDim.CODING,
            "hard",
            "Return only valid Python inside a ```python``` block. No prose.",
            "Write `dijkstra(n: int, edges: list[tuple[int, int, int]], src: int, dst: int) -> int` for a directed non-negative weighted graph on nodes `0..n-1`. Return shortest distance from `src` to `dst`, or `-1` if unreachable.",
            {"assertions": [
                "assert dijkstra(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1),(2,3,5)], 0, 3) == 4",
                "assert dijkstra(3, [(0,1,5)], 0, 2) == -1",
                "assert dijkstra(2, [(0,1,3)], 1, 1) == 0",
            ]},
        ),

        # 2. MATH_LOGIC (4)
        BenchmarkTask(
            "mth-01",
            BenchmarkDim.MATH_LOGIC,
            "easy",
            "Solve carefully and end with `FINAL_ANSWER: <value>`.",
            "Compute `(17 * 23 - 91) // 5`.",
            {"accepted_answers": ["60"]},
        ),
        BenchmarkTask(
            "mth-02",
            BenchmarkDim.MATH_LOGIC,
            "medium",
            "Solve step by step and end with `FINAL_ANSWER: <integer>`.",
            "How many 3-digit positive integers (100..999) are divisible by both 6 and 15?",
            {"accepted_answers": ["30"]},  # lcm(6,15)=30; floor(999/30)-floor(99/30) = 33 - 3 = 30
        ),
        BenchmarkTask(
            "mth-03",
            BenchmarkDim.MATH_LOGIC,
            "hard",
            "Solve step by step and end with `FINAL_ANSWER: <integer>`.",
            "Find the smallest positive integer `x` such that `x ≡ 3 (mod 7)` and `x ≡ 5 (mod 11)`.",
            {"accepted_answers": ["38"]},  # 38 mod 7 = 3, 38 mod 11 = 5
        ),
        BenchmarkTask(
            "mth-04",
            BenchmarkDim.MATH_LOGIC,
            "hard",
            "Solve step by step and end with `FINAL_ANSWER: <integer>`.",
            "A sequence satisfies `a_0 = 1, a_1 = 2, a_n = 2*a_{n-1} + a_{n-2}`. What is `a_6`?",
            # a0=1, a1=2, a2=5, a3=12, a4=29, a5=70, a6=169
            {"accepted_answers": ["169"]},
        ),

        # 3. SUMMARIZATION (4)
        BenchmarkTask(
            "sum-01",
            BenchmarkDim.SUMMARIZATION,
            "easy",
            "Summarize concisely without adding any external numbers.",
            "Summarize in <= 30 words while keeping 'Kestrel', '48 hours', and '3.2%': "
            "Orbiter Kestrel completed its 48 hours endurance thermal soak with thruster mass leakage measured at 3.2%.",
            {"source": "Orbiter Kestrel completed its 48 hours endurance thermal soak with thruster mass leakage measured at 3.2%.",
             "max_words": 30, "required_terms": ["Kestrel", "48 hours", "3.2%"]},
        ),
        BenchmarkTask(
            "sum-02",
            BenchmarkDim.SUMMARIZATION,
            "medium",
            "Summarize strictly from text. Do not invent numbers.",
            "Summarize in <= 35 words mentioning 'CinderDB', '14.5 ms', and '99.94%': "
            "In Q2, CinderDB replaced mutex-guarded buffers with wait-free ring slots. P99 commit latency dropped to 14.5 ms while replication durability held at 99.94% under chaos drills.",
            {"source": "In Q2, CinderDB replaced mutex-guarded buffers with wait-free ring slots. P99 commit latency dropped to 14.5 ms while replication durability held at 99.94% under chaos drills.",
             "max_words": 35, "required_terms": ["CinderDB", "14.5 ms", "99.94%"]},
        ),
        BenchmarkTask(
            "sum-03",
            BenchmarkDim.SUMMARIZATION,
            "medium",
            "Summarize strictly from source.",
            "Summarize in <= 32 words including 'borehole', '620 m', and 'basalt': "
            "Survey crew drilled borehole BH-9 down to 620 m through fractured basalt, recovering 91% core integrity.",
            {"source": "Survey crew drilled borehole BH-9 down to 620 m through fractured basalt, recovering 91% core integrity.",
             "max_words": 32, "required_terms": ["borehole", "620 m", "basalt"]},
        ),
        BenchmarkTask(
            "sum-04",
            BenchmarkDim.SUMMARIZATION,
            "hard",
            "Summarize faithfully in strict word limit.",
            "Summarize in <= 35 words mentioning 'Loom-7', '0.41 kWh', and '28%': "
            "Smart-grid router Loom-7 cut idle transformer losses to 0.41 kWh per feeder while shaving evening peak draw by 28% across 12 substations.",
            {"source": "Smart-grid router Loom-7 cut idle transformer losses to 0.41 kWh per feeder while shaving evening peak draw by 28% across 12 substations.",
             "max_words": 35, "required_terms": ["Loom-7", "0.41 kWh", "28%"]},
        ),

        # 4. STRUCTURED_JSON (4)
        BenchmarkTask(
            "jsn-01",
            BenchmarkDim.STRUCTURED_JSON,
            "easy",
            "Return only valid JSON matching the exact schema keys.",
            'Extract JSON with keys `{"host": str, "port": int, "tls": bool}` from: "Gateway proxy listens on host gw-04.internal port 8443 with TLS enabled."',
            {"expected_json": {"host": "gw-04.internal", "port": 8443, "tls": True}},
        ),
        BenchmarkTask(
            "jsn-02",
            BenchmarkDim.STRUCTURED_JSON,
            "medium",
            "Return only valid JSON matching the exact schema keys.",
            'Return `{"items": int, "total_cost_usd": int, "currency": "USD"}` from receipt: '
            '"Bought 3 switches at 40 USD each and 2 cables at 15 USD each." (items = total unit count 3+2).',
            {"expected_json": {"items": 5, "total_cost_usd": 150, "currency": "USD"}},
        ),
        BenchmarkTask(
            "jsn-03",
            BenchmarkDim.STRUCTURED_JSON,
            "medium",
            "Return only valid JSON matching exact schema.",
            'Compute `{"active_regions": list[str], "offline_count": int}` sorted alphabetically for active_regions from: '
            '"Regions us-east=UP, eu-central=DOWN, ap-south=UP, sa-east=DOWN, ca-central=UP."',
            {"expected_json": {"active_regions": ["ap-south", "ca-central", "us-east"], "offline_count": 2}},
        ),
        BenchmarkTask(
            "jsn-04",
            BenchmarkDim.STRUCTURED_JSON,
            "hard",
            "Return only valid JSON.",
            'Normalize invoice lines into `{"subtotal": int, "discount": int, "tax": int, "grand_total": int}`: '
            '"Base fee 400, compute add-on 120, bundle discount 40, flat tax 38 applied afterward."',
            {"expected_json": {"subtotal": 520, "discount": 40, "tax": 38, "grand_total": 518}},
        ),

        # 5. DIAGRAMS_CHARTS (4)
        BenchmarkTask(
            "dia-01",
            BenchmarkDim.DIAGRAMS_CHARTS,
            "easy",
            "Return only Mermaid inside ```mermaid``` fence. Quote any node labels containing parentheses.",
            "Write a `flowchart TD` diagram with nodes `Ingest`, `Validate`, and `Archive` connected in sequence with `-->`.",
            {"subkind": "mermaid", "expected_header": "flowchart TD", "required_tokens": ["Ingest", "Validate", "Archive"], "min_edges": 2},
        ),
        BenchmarkTask(
            "dia-02",
            BenchmarkDim.DIAGRAMS_CHARTS,
            "medium",
            "Return only Mermaid inside ```mermaid``` fence.",
            "Write a `stateDiagram-v2` showing transitions `Idle --> Queued`, `Queued --> Running`, and `Running --> Done`.",
            {"subkind": "mermaid", "expected_header": "stateDiagram-v2", "required_tokens": ["Idle", "Queued", "Running", "Done"], "min_edges": 3},
        ),
        BenchmarkTask(
            "dia-03",
            BenchmarkDim.DIAGRAMS_CHARTS,
            "medium",
            "Return only Mermaid inside ```mermaid``` fence. Quote labels containing parentheses.",
            'Write a `flowchart LR` showing `Client["Browser (UI)"]` --> `Gateway["API Gateway"]` --> `Worker["Job Runner (Async)"]` and error edge back to `Alert`.',
            {"subkind": "mermaid", "expected_header": "flowchart LR", "required_tokens": ["Browser", "Gateway", "Job Runner", "Alert"], "min_edges": 3},
        ),
        BenchmarkTask(
            "dia-04",
            BenchmarkDim.DIAGRAMS_CHARTS,
            "hard",
            "Return only valid JSON inside ```json``` block.",
            'Create quantitative chart JSON `{"chart_type": "bar", "x_labels": ["Q1","Q2","Q3","Q4"], "values": list[int], "sum_total": int}` '
            'from metrics: Q1=14, Q2=29, Q3=21, Q4=36.',
            {"subkind": "chart_json", "expected_json": {"chart_type": "bar", "x_labels": ["Q1", "Q2", "Q3", "Q4"], "values": [14, 29, 21, 36], "sum_total": 100}},
        ),

        # 6. LONG_CONTEXT (4)
        BenchmarkTask(
            "lng-01",
            BenchmarkDim.LONG_CONTEXT,
            "easy",
            "Return only valid JSON matching exact keys.",
            long_doc_a + '\n\nReturn `{"owner_south": str, "final_north_latency_ms": int}` after applying all audit errata.',
            {"expected_json": {"owner_south": "Kiran", "final_north_latency_ms": 27}},
            max_new_tokens=128,
        ),
        BenchmarkTask(
            "lng-02",
            BenchmarkDim.LONG_CONTEXT,
            "medium",
            "Return only valid JSON matching exact keys.",
            long_doc_b + '\n\nCompute net GPU hours for TKT-901 and return `{"ticket": "TKT-901", "net_hours": int}`.',
            {"expected_json": {"ticket": "TKT-901", "net_hours": 165}},  # 120 + 85 - 40 = 165
            max_new_tokens=128,
        ),
        BenchmarkTask(
            "lng-03",
            BenchmarkDim.LONG_CONTEXT,
            "hard",
            "Return only valid JSON matching exact keys.",
            long_doc_c + '\n\nReturn `{"p1_rpm": int, "p2_rpm": int, "reported_rpm": int}` applying the contradiction resolution rule.',
            {"expected_json": {"p1_rpm": 310, "p2_rpm": 440, "reported_rpm": 310}},
            max_new_tokens=128,
        ),
        BenchmarkTask(
            "lng-04",
            BenchmarkDim.LONG_CONTEXT,
            "hard",
            "Return only valid JSON matching exact keys.",
            long_doc_d + '\n\nReturn `{"remaining_crates": int, "remaining_kg": int, "seal_code": str}`.',
            {"expected_json": {"remaining_crates": 13, "remaining_kg": 375, "seal_code": "BOREAL-719"}},
            max_new_tokens=128,
        ),

        # 7. GENERAL_KNOWLEDGE (4)
        BenchmarkTask(
            "knw-01",
            BenchmarkDim.GENERAL_KNOWLEDGE,
            "easy",
            "Answer concisely and end with `FINAL_ANSWER: <answer>`.",
            "What HTTP status code means 'Too Many Requests' for rate limiting?",
            {"accepted_answers": ["429"]},
        ),
        BenchmarkTask(
            "knw-02",
            BenchmarkDim.GENERAL_KNOWLEDGE,
            "easy",
            "Answer concisely and end with `FINAL_ANSWER: <answer>`.",
            "In Python / IEEE-754, what is the bit width of a standard double-precision `float64`?",
            {"accepted_answers": ["64", "64 bits"]},
        ),
        BenchmarkTask(
            "knw-03",
            BenchmarkDim.GENERAL_KNOWLEDGE,
            "medium",
            "Answer concisely and end with `FINAL_ANSWER: <answer>`.",
            "In Big-O notation, what is the worst-case time complexity of binary search on a sorted array of `n` elements? Write `O(...)`.",
            {"accepted_answers": ["O(log n)", "O(logn)", "O(log2 n)"]},
        ),
        BenchmarkTask(
            "knw-04",
            BenchmarkDim.GENERAL_KNOWLEDGE,
            "hard",
            "Answer concisely and end with `FINAL_ANSWER: <answer>`.",
            "Which CPU/OS synchronization primitive pairs a condition counter to allow wait/wake without kernel calls unless contended: `futex`, `fork`, `mmap`, or `inotify`?",
            {"accepted_answers": ["futex"]},
        ),

        # 8. MULTI_STEP_REASONING (4)
        BenchmarkTask(
            "msr-01",
            BenchmarkDim.MULTI_STEP_REASONING,
            "easy",
            "Track state step by step and end with `FINAL_ANSWER: <int>`.",
            "A buffer starts at 10. Step 1: double it. Step 2: subtract 6. Step 3: multiply by 3. Step 4: add 8. What is the final value?",
            {"accepted_answers": ["50"]},  # ((10*2)-6)*3 + 8 = 14*3 + 8 = 50
        ),
        BenchmarkTask(
            "msr-02",
            BenchmarkDim.MULTI_STEP_REASONING,
            "medium",
            "Track all constraints and end with `FINAL_ANSWER: <name>`.",
            "Three boxes Red, Blue, Green contain exactly one key each: Silver, Gold, Brass. "
            "Clue 1: Red does not contain Gold. Clue 2: Green contains Silver. Clue 3: Blue does not contain Brass. "
            "Which key is inside the Red box?",
            {"accepted_answers": ["Brass"]},  # Green=Silver; Red!=Gold -> Red=Brass, Blue=Gold
        ),
        BenchmarkTask(
            "msr-03",
            BenchmarkDim.MULTI_STEP_REASONING,
            "hard",
            "Compute exact parity/count and end with `FINAL_ANSWER: <int>`.",
            "In a 4-register machine `[r0=3, r1=5, r2=2, r3=7]`: "
            "execute `r0 <- r0 + r1`; `r2 <- r2 * r0`; `r3 <- r3 XOR r1`; `r1 <- r2 - r3`. What is `r1`?",
            # r0 = 8; r2 = 2*8 = 16; r3 = 7 ^ 5 = 2; r1 = 16 - 2 = 14
            {"accepted_answers": ["14"]},
        ),
        BenchmarkTask(
            "msr-04",
            BenchmarkDim.MULTI_STEP_REASONING,
            "hard",
            "Solve step by step and end with `FINAL_ANSWER: <int>`.",
            "Five servers A, B, C, D, E sit in a line 1..5. B is at 2. D is strictly to the right of B and immediately next to E. "
            "A is at an odd position greater than B. C is immediately left of B. What position number is E at?",
            # C=1, B=2; A odd >2 => A=3 or 5; if A=3, D/E at 4,5; if A=5, D/E at 3,4. Wait let's make position unique!
            # Add clue: "D is to the left of A." -> then D,E are at 3,4 or 4,3 with A=5. Make D<E: not needed if E=4: wait if D=3, E=4, A=5!
            {"accepted_answers": ["4"]},
        ),

        # 9. INSTRUCTION_FOLLOWING (4)
        BenchmarkTask(
            "ins-01",
            BenchmarkDim.INSTRUCTION_FOLLOWING,
            "easy",
            "Follow all formatting constraints strictly.",
            "Write a response in all lowercase containing the word `kernel` using at most 18 words and do not use the word `cloud`.",
            {"require_all_lowercase": True, "required_words": ["kernel"], "forbidden_words": ["cloud"], "max_words": 18},
        ),
        BenchmarkTask(
            "ins-02",
            BenchmarkDim.INSTRUCTION_FOLLOWING,
            "medium",
            "Follow all formatting constraints strictly.",
            "List exactly 3 bullet lines starting with `- ` explaining unit testing benefits. Do not use the word `bug` or `crash`, and end the final character of the reply with `END_CHECK`.",
            {"exact_bullets": 3, "forbidden_words": ["bug", "crash"], "required_end_token": "END_CHECK"},
        ),
        BenchmarkTask(
            "ins-03",
            BenchmarkDim.INSTRUCTION_FOLLOWING,
            "medium",
            "Follow all formatting constraints strictly.",
            "Write a short paragraph of at least 15 words and at most 28 words about hash tables. Include both `collision` and `bucket`, and do NOT use the word `pointer`.",
            {"min_words": 15, "max_words": 28, "required_words": ["collision", "bucket"], "forbidden_words": ["pointer"]},
        ),
        BenchmarkTask(
            "ins-04",
            BenchmarkDim.INSTRUCTION_FOLLOWING,
            "hard",
            "Follow every single constraint strictly.",
            "Reply in ALL lowercase with exactly 2 bullet lines (`- ...`), max 22 words total, including `latency` and `cache`, excluding `slow`, ending with `::done`.",
            {"require_all_lowercase": True, "exact_bullets": 2, "max_words": 22, "required_words": ["latency", "cache"], "forbidden_words": ["slow"], "required_end_token": "::done"},
        ),

        # 10. TOOL_USE (4)
        BenchmarkTask(
            "tol-01",
            BenchmarkDim.TOOL_USE,
            "easy",
            "Return only a single JSON tool call object `{\"tool\": str, \"args\": dict}`.",
            "Available tools: `read_metric(service: str, window_min: int)`, `restart_pod(pod: str)`. "
            "Request: Read metrics for service `auth-api` over a 15-minute window.",
            {"expected_calls": [{"tool": "read_metric", "args": {"service": "auth-api", "window_min": 15}}]},
        ),
        BenchmarkTask(
            "tol-02",
            BenchmarkDim.TOOL_USE,
            "medium",
            "Return only JSON `{\"calls\": [...]}`.",
            "Tools: `fetch_balance(account_id: int)`, `convert_currency(amount: int, from_ccy: str, to_ccy: str)`. "
            "Goal: First fetch balance for account 402, then convert 250 USD to EUR.",
            {"expected_calls": [
                {"tool": "fetch_balance", "args": {"account_id": 402}},
                {"tool": "convert_currency", "args": {"amount": 250, "from_ccy": "USD", "to_ccy": "EUR"}},
            ]},
        ),
        BenchmarkTask(
            "tol-03",
            BenchmarkDim.TOOL_USE,
            "medium",
            "Return only a JSON tool call object `{\"tool\": str, \"args\": dict}`.",
            "Tool call `primary_search(query: 'cuda')` returned error `503 SERVICE_UNAVAILABLE`. Policy: immediately call `cache_lookup(key: str, fallback: true)`. Issue the recovery call for key `'cuda'`.",
            {"expected_calls": [{"tool": "cache_lookup", "args": {"key": "cuda", "fallback": True}}]},
        ),
        BenchmarkTask(
            "tol-04",
            BenchmarkDim.TOOL_USE,
            "hard",
            "Return only JSON `{\"calls\": [...]}`.",
            "Step 1 tool `read_file(path: '/etc/missing.conf')` failed with `ENOENT`. "
            "Recovery policy: first call `create_backup_dir(path: '/tmp/recover')`, then call `write_default_config(target: '/etc/missing.conf', mode: 644)`.",
            {"expected_calls": [
                {"tool": "create_backup_dir", "args": {"path": "/tmp/recover"}},
                {"tool": "write_default_config", "args": {"target": "/etc/missing.conf", "mode": 644}},
            ]},
        ),

        # 11. PLANNING_DECOMPOSITION (4)
        BenchmarkTask(
            "pln-01",
            BenchmarkDim.PLANNING_DECOMPOSITION,
            "easy",
            "Return only valid JSON `{\"order\": list[str], \"makespan\": int}`.",
            "Schedule tasks: `A` (dur 3, deps none), `B` (dur 4, deps [A]), `C` (dur 2, deps [B]). Output valid execution order and optimal parallel makespan.",
            {"dag": {"A": {"dur": 3, "deps": []}, "B": {"dur": 4, "deps": ["A"]}, "C": {"dur": 2, "deps": ["B"]}}},
        ),
        BenchmarkTask(
            "pln-02",
            BenchmarkDim.PLANNING_DECOMPOSITION,
            "medium",
            "Return only valid JSON `{\"order\": list[str], \"makespan\": int}`.",
            "DAG tasks: `Spec` (dur 2, deps []), `Backend` (dur 5, deps [Spec]), `Frontend` (dur 3, deps [Spec]), `Deploy` (dur 2, deps [Backend, Frontend]). Find topological order and minimum parallel makespan.",
            {"dag": {"Spec": {"dur": 2, "deps": []}, "Backend": {"dur": 5, "deps": ["Spec"]}, "Frontend": {"dur": 3, "deps": ["Spec"]}, "Deploy": {"dur": 2, "deps": ["Backend", "Frontend"]}}},
        ),
        BenchmarkTask(
            "pln-03",
            BenchmarkDim.PLANNING_DECOMPOSITION,
            "hard",
            "Return only valid JSON `{\"order\": list[str], \"makespan\": int}`.",
            "DAG: `T1` (dur 4, deps []), `T2` (dur 2, deps []), `T3` (dur 6, deps [T1]), `T4` (dur 5, deps [T1, T2]), `T5` (dur 3, deps [T3, T4]). Compute valid topological order and minimum makespan.",
            {"dag": {
                "T1": {"dur": 4, "deps": []},
                "T2": {"dur": 2, "deps": []},
                "T3": {"dur": 6, "deps": ["T1"]},
                "T4": {"dur": 5, "deps": ["T1", "T2"]},
                "T5": {"dur": 3, "deps": ["T3", "T4"]},
            }},
        ),
        BenchmarkTask(
            "pln-04",
            BenchmarkDim.PLANNING_DECOMPOSITION,
            "hard",
            "Return only valid JSON `{\"order\": list[str], \"makespan\": int}`.",
            "DAG: `Init` (dur 1, deps []), `Data` (dur 7, deps [Init]), `Kernel` (dur 4, deps [Init]), `Eval` (dur 3, deps [Kernel]), `Signoff` (dur 2, deps [Data, Eval]). Compute valid order and minimum makespan.",
            {"dag": {
                "Init": {"dur": 1, "deps": []},
                "Data": {"dur": 7, "deps": ["Init"]},
                "Kernel": {"dur": 4, "deps": ["Init"]},
                "Eval": {"dur": 3, "deps": ["Kernel"]},
                "Signoff": {"dur": 2, "deps": ["Data", "Eval"]},
            }},
        ),

        # 12. OOD_ROBUSTNESS (4)
        BenchmarkTask(
            "ood-01",
            BenchmarkDim.OOD_ROBUSTNESS,
            "easy",
            "Use the custom operator rule strictly and end with `FINAL_ANSWER: <int>`.",
            "Redefine binary operator `@` by `a @ b = (2 * a) + (3 * b)`. Compute `5 @ 4`.",
            {"accepted_answers": ["22"]},
        ),
        BenchmarkTask(
            "ood-02",
            BenchmarkDim.OOD_ROBUSTNESS,
            "medium",
            "Apply the counterfactual arithmetic rule strictly and end with `FINAL_ANSWER: <int>`.",
            "In Base-Z arithmetic, `+` is ordinary multiplication and `*` is ordinary addition, with `*` having HIGHER precedence than `+`. Compute `2 + 3 * 4`.",
            # 3 * 4 = 3 + 4 = 7; 2 + 7 = 2 * 7 = 14
            {"accepted_answers": ["14"]},
        ),
        BenchmarkTask(
            "ood-03",
            BenchmarkDim.OOD_ROBUSTNESS,
            "hard",
            "Apply the novel string rule carefully and end with `FINAL_ANSWER: <value>`.",
            "Operator `FLIP_ADD(a, b)` reverses decimal digits of `a`, reverses decimal digits of `b`, and sums those reversed integers normally. Compute `FLIP_ADD(14, 230)`.",
            # rev(14)=41, rev(230)=032=32 -> 41+32=73
            {"accepted_answers": ["73"]},
        ),
        BenchmarkTask(
            "ood-04",
            BenchmarkDim.OOD_ROBUSTNESS,
            "hard",
            "Follow the counterfactual modulo rule and end with `FINAL_ANSWER: <int>`.",
            "Define `f(x) = (x^2 + 1) mod 13`. Compute `f(f(4))`.",
            # f(4) = 17 mod 13 = 4 -> f(4) = 4
            {"accepted_answers": ["4"]},
        ),

        # 13. HALLUCINATION_RESISTANCE (4)
        BenchmarkTask(
            "hal-01",
            BenchmarkDim.HALLUCINATION_RESISTANCE,
            "easy",
            "Never guess missing facts. If the question cannot be answered from provided facts or contains an impossible premise, respond starting with `ABSTAIN:` and explain why.",
            "Invoice #88 says the buyer paid $250 in shipping, and zero taxes were itemized. What was the exact price per unit of item SKU-9?",
            {"forbidden_confabulations": ["25", "50", "$100"]},
        ),
        BenchmarkTask(
            "hal-02",
            BenchmarkDim.HALLUCINATION_RESISTANCE,
            "medium",
            "Never fabricate data. Start with `ABSTAIN:` if premises are missing or contradictory.",
            "Log A states server uptime on May 4 was 100.00% with zero downtime. Log B states server uptime on May 4 was 94.2% due to a 50-minute power trip. Without an authoritative override source, what is the confirmed single exact number of outage minutes?",
            {"forbidden_confabulations": ["50 minutes is confirmed", "exact outage is 50", "0 minutes"]},
        ),
        BenchmarkTask(
            "hal-03",
            BenchmarkDim.HALLUCINATION_RESISTANCE,
            "medium",
            "Never guess missing facts. Output `ABSTAIN:` if unanswerable.",
            "Find a prime integer `p > 5` whose square `p^2` is even.",
            {"forbidden_confabulations": ["p = 7", "p = 11", "p = 13"]},
        ),
        BenchmarkTask(
            "hal-04",
            BenchmarkDim.HALLUCINATION_RESISTANCE,
            "hard",
            "Output `ABSTAIN:` cleanly if the document omits the requested quantity.",
            "Report excerpt: 'Warehouse East holds 4,200 pallets after Q1 consolidation, up from Q4 baseline.' Question: How many pallets did Warehouse West hold in Q4?",
            {"forbidden_confabulations": ["4200", "4,200"],
             "citable_claims": [r"\b(?:warehouse\s+)?east(?:\s+(?:warehouse|holds?|held|has|had|stores?|stored|reported|contained|with|only|currently|accounts?\s+for)|[\s,:-])+4,?200\b",
                               r"\b4,?200\s+pallets?\s+(?:at|in|for|refer(?:s)? to|belong(?:ing)? to|(?:is|are|were) (?:at|in|for))\s+(?:warehouse\s+)?east\b"]},
        ),

        # 14. MEMORY_RETENTION (4)
        BenchmarkTask(
            "mem-01",
            BenchmarkDim.MEMORY_RETENTION,
            "easy",
            "Return only valid JSON representing the FINAL state.",
            "\n".join([
                "Turn 1: SET user=alice",
                "Turn 2: SET role=viewer",
                "Turn 3: SET region=us-west",
                "Turn 4: OVERRIDE role=editor",
                "Turn 5: SET active=true",
                "Turn 6: OVERRIDE region=eu-west",
                "Turn 7: OVERRIDE active=false",
                "Turn 8: SET quota=20",
                "Turn 9: OVERRIDE quota=35",
                "Turn 10: Query final state `{\"user\": str, \"role\": str, \"region\": str, \"active\": bool, \"quota\": int}`.",
            ]),
            {"expected_json": {"user": "alice", "role": "editor", "region": "eu-west", "active": False, "quota": 35}},
        ),
        BenchmarkTask(
            "mem-02",
            BenchmarkDim.MEMORY_RETENTION,
            "medium",
            "Return only valid JSON representing the FINAL cart state.",
            "\n".join([
                "Turn 1: ADD gpu=2",
                "Turn 2: ADD ssd=4",
                "Turn 3: ADD cable=6",
                "Turn 4: REMOVE cable=2",
                "Turn 5: ADD gpu=1",
                "Turn 6: RESET ssd=1",
                "Turn 7: DELETE cable",
                "Turn 8: ADD fan=5",
                "Turn 9: REMOVE fan=2",
                "Turn 10: Return JSON `{\"gpu\": int, \"ssd\": int, \"fan\": int}` (exclude deleted keys).",
            ]),
            {"expected_json": {"gpu": 3, "ssd": 1, "fan": 3}},
        ),
        BenchmarkTask(
            "mem-03",
            BenchmarkDim.MEMORY_RETENTION,
            "hard",
            "Return only valid JSON representing the FINAL account balances.",
            "\n".join([
                "Turn 1: INIT A=100, B=50",
                "Turn 2: TRANSFER A->B 30",
                "Turn 3: DEPOSIT A 20",
                "Turn 4: TRANSFER B->A 15",
                "Turn 5: WITHDRAW B 25",
                "Turn 6: INTEREST A +10% (integer floor)",  # A=100-30+20+15=105 -> +10 -> 115
                "Turn 7: TRANSFER A->B 25",                 # A=90, B=50+30-15-25+25=65
                "Turn 8: WITHDRAW A 10",                    # A=80
                "Turn 9: DEPOSIT B 5",                      # B=70
                "Turn 10: Return JSON `{\"A\": int, \"B\": int}`.",
            ]),
            {"expected_json": {"A": 80, "B": 70}},
        ),
        BenchmarkTask(
            "mem-04",
            BenchmarkDim.MEMORY_RETENTION,
            "hard",
            "Return only valid JSON.",
            "\n".join([
                "Turn 1: OPEN branch=main, status=draft",
                "Turn 2: ASSIGN owner=maya",
                "Turn 3: PRIORITY=high",
                "Turn 4: REASSIGN owner=liam",
                "Turn 5: NOTE=ci_green",
                "Turn 6: STATUS=review",
                "Turn 7: PRIORITY=critical",
                "Turn 8: MERGE branch=release-1",
                "Turn 9: STATUS=merged",
                "Turn 10: Return JSON `{\"branch\": str, \"owner\": str, \"priority\": str, \"status\": str}`.",
            ]),
            {"expected_json": {"branch": "release-1", "owner": "liam", "priority": "critical", "status": "merged"}},
        ),
    )

    # Fix msr-04 ambiguity right here so msr-04 is 100% uniquely determined!
    fixed = list(tasks)
    for i, t in enumerate(fixed):
        if t.task_id == "msr-04":
            fixed[i] = BenchmarkTask(
                "msr-04",
                BenchmarkDim.MULTI_STEP_REASONING,
                "hard",
                "Solve step by step and end with `FINAL_ANSWER: <int>`.",
                "Five servers A, B, C, D, E sit in positions 1..5. "
                "B is at 2. C is immediately left of B. A is at 5. D is immediately right of B. "
                "Which position number is E at?",
                {"accepted_answers": ["4"]},
            )
    return tuple(fixed)


def compute_efficiency_score(
    *,
    capability_accuracy: float,
    total_parameters: int,
    tokens_per_sec: float,
    kv_cache_mb_at_4k: float,
    reference_tps: float = 40.0,
) -> dict[str, float]:
    """Compute Dimension 15 inference_efficiency summary + Capability-per-Billion-Params."""
    param_b = max(total_parameters / 1e9, 1e-6)
    cap_per_b = capability_accuracy / param_b
    speed_ratio = min(1.5, tokens_per_sec / max(reference_tps, 1e-6))
    # Normalized [0, 1] composite efficiency index combining accuracy, speed, and compact parameter count
    eff_score = min(1.0, max(0.0, 0.65 * capability_accuracy + 0.35 * min(1.0, speed_ratio / 1.25)))
    return {
        "capability_accuracy": round(capability_accuracy, 4),
        "capability_per_billion_params": round(cap_per_b, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "kv_cache_mb_at_4k": round(kv_cache_mb_at_4k, 3),
        "efficiency_composite_score": round(eff_score, 4),
    }


def bootstrap_accuracy_ci(
    outcomes: list[float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Deterministic seeded percentile bootstrap 95% confidence interval."""
    if not outcomes:
        return (0.0, 0.0, 0.0)
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    rng = random.Random(seed)
    n = len(outcomes)
    mean_val = sum(outcomes) / n
    means = [sum(rng.choices(outcomes, k=n)) / n for _ in range(n_resamples)]
    means.sort()
    lo = means[max(0, int(math.floor(0.025 * n_resamples)))]
    hi = means[min(n_resamples - 1, int(math.ceil(0.975 * n_resamples)) - 1)]
    return (round(mean_val, 4), round(lo, 4), round(hi, 4))
