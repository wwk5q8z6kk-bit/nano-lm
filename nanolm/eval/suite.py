"""Independent 6-Suite General-Purpose Capability Evaluation Framework for NanoLM.

All held-out evaluation tasks use disjoint identifier vocabularies, unseen parameter
ranges, and adversarial distractors that never appear in `nanolm.data.corpus`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from nanolm.eval.validators import (
    verify_chart_spec,
    verify_math_or_logic,
    verify_mermaid_flowchart,
    verify_python_solution,
    verify_structured_json,
    verify_summary_faithfulness,
)


@dataclass(frozen=True, slots=True)
class SuiteScore:
    suite: str
    total: int
    passed: int
    pass_rate: float
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    model_id: str
    overall_pass_rate: float
    suites: dict[str, SuiteScore]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "overall_pass_rate": self.overall_pass_rate,
            "suites": {k: asdict(v) for k, v in self.suites.items()},
        }


def build_heldout_eval_tasks() -> dict[str, list[dict[str, Any]]]:
    """Construct held-out, zero-overlap evaluation items across all 6 capability suites."""
    coding_tasks = [
        {
            "id": "code_01_dedup_preserve",
            "prompt": (
                "Write a Python function `stable_unique_positive(items: list[int]) -> list[int]` "
                "that keeps only strictly positive integers (> 0) and preserves first-seen order without duplicates."
            ),
            "harness": (
                "assert stable_unique_positive([4, -2, 4, 0, 7, 1, 7]) == [4, 7, 1]\n"
                "assert stable_unique_positive([-5, 0]) == []\n"
            ),
        },
        {
            "id": "code_02_coalesce_ranges",
            "prompt": (
                "Write a Python function `coalesce_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]` "
                "that merges overlapping or touching closed intervals `[start, end]` and returns them sorted."
            ),
            "harness": (
                "assert coalesce_windows([(5, 8), (1, 3), (3, 6), (10, 12)]) == [(1, 8), (10, 12)]\n"
                "assert coalesce_windows([]) == []\n"
            ),
        },
        {
            "id": "code_03_balanced_delims",
            "prompt": (
                "Write a Python function `is_balanced_brackets(text: str) -> bool` "
                "that checks whether `()`, `[]`, and `{}` are properly nested and balanced, ignoring other characters."
            ),
            "harness": (
                "assert is_balanced_brackets('fn(a[0], {k: (1 + 2)})') is True\n"
                "assert is_balanced_brackets('([)]') is False\n"
                "assert is_balanced_brackets('(((') is False\n"
            ),
        },
        {
            "id": "code_04_top_k_freq",
            "prompt": (
                "Write a Python function `top_two_tokens(words: list[str]) -> list[str]` "
                "returning up to 2 most frequent strings ordered by descending count, breaking ties alphabetically."
            ),
            "harness": (
                "assert top_two_tokens(['zeta', 'beta', 'zeta', 'alpha', 'beta', 'gamma']) == ['beta', 'zeta']\n"
                "assert top_two_tokens(['solo']) == ['solo']\n"
            ),
        },
    ]

    math_logic_tasks = [
        {
            "id": "math_01_cluster_budget",
            "prompt": (
                "A data center deploys 14 enclosures with 12 blades each. Note that serial #9941 is just a rack label. "
                "During maintenance, 19 blades are powered down. How many active blades remain? "
                "End your response with `FINAL_ANSWER: <number>`."
            ),
            "expected": "149",
        },
        {
            "id": "math_02_rate_synthesis",
            "prompt": (
                "Worker pool Alpha processes 35 jobs/min and pool Beta processes 25 jobs/min. "
                "How many minutes are required together to finish 420 jobs? "
                "End your response with `FINAL_ANSWER: <number>`."
            ),
            "expected": "7",
        },
        {
            "id": "logic_03_transitive_order",
            "prompt": (
                "Four services ( Kronos, Lyra, Mira, Nyx ) start sequentially with no ties. "
                "Lyra starts before Kronos. Mira starts before Lyra. Nyx starts after Kronos. "
                "Which service starts first? End your response with `FINAL_ANSWER: <name>`."
            ),
            "expected": "Mira",
        },
        {
            "id": "logic_04_constraint_elimination",
            "prompt": (
                "Each of three caches (C1, C2, C3) uses a distinct eviction policy: LRU, LFU, or FIFO. "
                "C1 does not use LRU or FIFO. C3 does not use FIFO. Which policy does C2 use? "
                "End your response with `FINAL_ANSWER: <policy>`."
            ),
            "expected": "FIFO",
        },
    ]

    summarization_tasks = [
        {
            "id": "sum_01_exception_retention",
            "prompt": (
                "Summarize the release bulletin in one concise sentence (<= 45 words), preserving exact metrics "
                "and any exception:\n"
                "Bulletin: Enabling zero-copy serialization in `velox_stream` cut memory usage from 64 GB to 38 GB "
                "across all Linux workers, except on `arm64-debug` nodes where bounds-checking remains mandatory."
            ),
            "required_facts": ["velox_stream", "64", "38"],
            "required_qualifications": ["except", "arm64-debug"],
            "forbidden_claims": ["all nodes without exception", "windows", "16 gb"],
        },
        {
            "id": "sum_02_conditional_threshold",
            "prompt": (
                "Summarize the policy note in one concise sentence (<= 45 words), preserving exact thresholds "
                "and conditions:\n"
                "Policy: Automated failover triggers when replica lag exceeds 250 ms for 3 consecutive probes, "
                "unless manual freeze mode is active."
            ),
            "required_facts": ["250", "3"],
            "required_qualifications": ["unless", "freeze"],
            "forbidden_claims": ["immediately on 1 probe", "500 ms"],
        },
    ]

    structured_json_tasks = [
        {
            "id": "json_01_service_manifest",
            "prompt": (
                "Output ONLY valid JSON with keys `cluster` (string), `nodes` (list of objects with `host` and `cores`), "
                "and `total_cores` (integer) for cluster `zephyr` containing `n1` (16 cores) and `n2` (32 cores)."
            ),
            "required_fields": {"cluster": str, "nodes": list, "total_cores": int},
            "predicate": lambda o: (
                o["cluster"].lower() == "zephyr"
                and len(o["nodes"]) == 2
                and o["total_cores"] == 48
            ),
        },
        {
            "id": "json_02_budget_ledger",
            "prompt": (
                "Output ONLY valid JSON with keys `project` (string), `items` (list of numbers), and `remaining_usd` (number) "
                "for project `solaris` with starting budget 500 and expenses 120, 85, and 95."
            ),
            "required_fields": {"project": str, "items": list, "remaining_usd": (int, float)},
            "predicate": lambda o: (
                o["project"].lower() == "solaris"
                and sum(o["items"]) == 300
                and abs(float(o["remaining_usd"]) - 200.0) < 1e-6
            ),
        },
    ]

    charts_diagrams_tasks = [
        {
            "id": "diag_01_mermaid_router",
            "kind": "mermaid",
            "prompt": (
                "Output a ```mermaid flowchart LR``` diagram showing `api_ingress` routing `authenticated` "
                "traffic to `core_engine` and `rejected` traffic to `audit_sink`."
            ),
            "expected_edges": [
                ("api_ingress", "core_engine", "authenticated"),
                ("api_ingress", "audit_sink", "rejected"),
            ],
        },
        {
            "id": "chart_02_timeseries_spec",
            "kind": "chart",
            "prompt": (
                "Convert this quarterly throughput trend into a JSON chart spec with keys "
                "`chart_type` ('line' or 'area' for time series), `x_label`, `y_label`, `x_values`, `y_values`:\n"
                "Q1: 120 req/s, Q2: 155 req/s, Q3: 190 req/s, Q4: 240 req/s."
            ),
            "allowed_types": {"line", "area"},
            "expected_x": ["Q1", "Q2", "Q3", "Q4"],
            "expected_y": [120.0, 155.0, 190.0, 240.0],
        },
    ]

    filler = (
        "Subsystem heartbeat nominal; ECC scrubber reported zero uncorrectable bit flips across "
        "channels 0 through 7 during the 60-second telemetry window."
    )
    long_context_tasks = [
        {
            "id": "long_01_distant_needle_sum",
            "prompt": (
                "Read the multi-section log carefully and compute the sum of `vault_quota` for `QUASAR` and `PULSAR`. "
                "End with `FINAL_ANSWER: <number>`.\n\n"
                + f"[SECTION QUASAR] vault_quota=315. {filler}\n"
                + "\n".join(f"[LOG {i}] {filler}" for i in range(18))
                + f"\n[SECTION VORTEX] vault_quota=810. {filler}\n"
                + "\n".join(f"[LOG {i+20}] {filler}" for i in range(18))
                + f"\n[SECTION PULSAR] vault_quota=460. {filler}"
            ),
            "expected": "775",
        },
        {
            "id": "long_02_contradiction_detection",
            "prompt": (
                "Identify which parameter has contradictory values between the header and footer of this report. "
                "Reply with `FINAL_ANSWER: <parameter_name>`.\n\n"
                + "[HEADER] max_retries=5, timeout_sec=30, tls_mode=strict.\n"
                + "\n".join(f"[TRACE {i}] {filler}" for i in range(20))
                + "\n[FOOTER] Verified active config: max_retries=5, timeout_sec=90, tls_mode=strict."
            ),
            "expected": "timeout_sec",
        },
    ]

    return {
        "coding": coding_tasks,
        "math_logic": math_logic_tasks,
        "summarization": summarization_tasks,
        "structured_json": structured_json_tasks,
        "charts_diagrams": charts_diagrams_tasks,
        "long_context": long_context_tasks,
    }


def collect_all_eval_prompts() -> list[str]:
    """Return every prompt string in the held-out suite so `decontaminate_corpus` can enforce 0% leakage."""
    suites = build_heldout_eval_tasks()
    return [item["prompt"] for tasks in suites.values() for item in tasks]


def run_capability_evaluation(
    model_id: str,
    generate_fn: Callable[[str], str],
) -> EvaluationReport:
    """Run all 6 independent capability suites against any model `prompt -> completion` callable."""
    tasks = build_heldout_eval_tasks()
    suite_scores: dict[str, SuiteScore] = {}
    grand_total = 0
    grand_passed = 0

    # 1. Coding (executable Python unit tests)
    c_pass = 0
    c_items = []
    for item in tasks["coding"]:
        resp = generate_fn(item["prompt"])
        ok, reason = verify_python_solution(resp, item["harness"])
        c_pass += int(ok)
        c_items.append({"id": item["id"], "passed": ok, "reason": reason})
    suite_scores["coding"] = SuiteScore(
        suite="coding",
        total=len(tasks["coding"]),
        passed=c_pass,
        pass_rate=round(c_pass / len(tasks["coding"]), 4),
        details={"items": c_items},
    )

    # 2. Math & Logic
    m_pass = 0
    m_items = []
    for item in tasks["math_logic"]:
        resp = generate_fn(item["prompt"])
        ok = verify_math_or_logic(resp, item["expected"])
        m_pass += int(ok)
        m_items.append({"id": item["id"], "passed": ok})
    suite_scores["math_logic"] = SuiteScore(
        suite="math_logic",
        total=len(tasks["math_logic"]),
        passed=m_pass,
        pass_rate=round(m_pass / len(tasks["math_logic"]), 4),
        details={"items": m_items},
    )

    # 3. Summarization & Factual Consistency
    s_pass = 0
    s_items = []
    for item in tasks["summarization"]:
        resp = generate_fn(item["prompt"])
        res = verify_summary_faithfulness(
            resp,
            required_facts=item["required_facts"],
            required_qualifications=item["required_qualifications"],
            forbidden_claims=item["forbidden_claims"],
        )
        ok = bool(res["passed"])
        s_pass += int(ok)
        s_items.append({"id": item["id"], **res})
    suite_scores["summarization"] = SuiteScore(
        suite="summarization",
        total=len(tasks["summarization"]),
        passed=s_pass,
        pass_rate=round(s_pass / len(tasks["summarization"]), 4),
        details={"items": s_items},
    )

    # 4. Structured JSON
    j_pass = 0
    j_valid = 0
    j_items = []
    for item in tasks["structured_json"]:
        resp = generate_fn(item["prompt"])
        valid_syntax, ok, reason = verify_structured_json(resp, item["required_fields"], item["predicate"])
        j_valid += int(valid_syntax)
        j_pass += int(ok)
        j_items.append({"id": item["id"], "valid_json": valid_syntax, "passed": ok, "reason": reason})
    suite_scores["structured_json"] = SuiteScore(
        suite="structured_json",
        total=len(tasks["structured_json"]),
        passed=j_pass,
        pass_rate=round(j_pass / len(tasks["structured_json"]), 4),
        details={"syntax_valid_rate": round(j_valid / len(tasks["structured_json"]), 4), "items": j_items},
    )

    # 5. Charts & Diagrams
    d_pass = 0
    d_items = []
    for item in tasks["charts_diagrams"]:
        resp = generate_fn(item["prompt"])
        if item["kind"] == "mermaid":
            syn_ok, sem_ok, reason = verify_mermaid_flowchart(resp, item["expected_edges"])
        else:
            syn_ok, sem_ok, reason = verify_chart_spec(
                resp, item["allowed_types"], item["expected_x"], item["expected_y"]
            )
        d_pass += int(sem_ok)
        d_items.append({"id": item["id"], "syntax_valid": syn_ok, "passed": sem_ok, "reason": reason})
    suite_scores["charts_diagrams"] = SuiteScore(
        suite="charts_diagrams",
        total=len(tasks["charts_diagrams"]),
        passed=d_pass,
        pass_rate=round(d_pass / len(tasks["charts_diagrams"]), 4),
        details={"items": d_items},
    )

    # 6. Long-Context Reasoning
    l_pass = 0
    l_items = []
    for item in tasks["long_context"]:
        resp = generate_fn(item["prompt"])
        ok = verify_math_or_logic(resp, item["expected"])
        l_pass += int(ok)
        l_items.append({"id": item["id"], "passed": ok})
    suite_scores["long_context"] = SuiteScore(
        suite="long_context",
        total=len(tasks["long_context"]),
        passed=l_pass,
        pass_rate=round(l_pass / len(tasks["long_context"]), 4),
        details={"items": l_items},
    )

    for sc in suite_scores.values():
        grand_total += sc.total
        grand_passed += sc.passed

    return EvaluationReport(
        model_id=model_id,
        overall_pass_rate=round(grand_passed / max(1, grand_total), 4),
        suites=suite_scores,
    )
