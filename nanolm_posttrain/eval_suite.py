"""Deterministic multi-domain evaluation and verifier suite for NanoLM (2B-4B).

Covers NanoLM's core general-purpose foundation capabilities without relying on
subjective LLM judges:
1. ``CODING`` — isolated Python AST + unit-test execution verification;
2. ``REASONING`` — multi-step mathematical & logical exact-answer verification;
3. ``SUMMARIZATION`` — compression-ratio, key-fact recall, and zero-hallucinated-number checks;
4. ``STRUCTURED_EXTRACTION`` — strict JSON schema + exact source-value conservation;
5. ``DIAGRAMS_AND_CHARTS`` — Mermaid syntax/topology/quoted-label validation + JSON chart series conservation;
6. ``LONG_CONTEXT`` — multi-needle synthesis and ordering across long documents.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any


class FoundationDomain(str, Enum):
    CODING = "coding"
    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    STRUCTURED_EXTRACTION = "structured_extraction"
    DIAGRAMS_AND_CHARTS = "diagrams_and_charts"
    LONG_CONTEXT = "long_context"


@dataclass(frozen=True, slots=True)
class TaskVerdict:
    passed: bool
    score: float
    reason: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FoundationTask:
    task_id: str
    domain: FoundationDomain
    system_prompt: str
    user_prompt: str
    spec: dict[str, Any]
    reference_response: str = ""


_CODE_FENCE_RE = re.compile(r"```(?:python|py|json|mermaid)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_MERMAID_HEADERS = (
    "flowchart ",
    "graph ",
    "sequenceDiagram",
    "stateDiagram-v2",
    "stateDiagram",
    "classDiagram",
    "erDiagram",
    "xychart-beta",
)


def _extract_fenced_or_raw(text: str) -> str:
    matches = _CODE_FENCE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def _verify_coding(response: str, spec: dict[str, Any]) -> TaskVerdict:
    from nanolm_posttrain.code_guest import run_guest

    checks = tuple(spec["assertions"])
    passed, error = run_guest(_extract_fenced_or_raw(response), checks)
    if not passed:
        return TaskVerdict(False, 0.0, f"guest_error:{error}", {})
    return TaskVerdict(True, 1.0, "all_assertions_passed", {"assertions": len(checks)})


def _verify_reasoning(response: str, spec: dict[str, Any]) -> TaskVerdict:
    expected = str(spec["expected_answer"]).strip()
    boxed = re.findall(r"FINAL_ANSWER\s*:\s*([^\n]+)", response, flags=re.IGNORECASE)
    candidate = boxed[-1].strip().rstrip(".") if boxed else ""
    if not candidate:
        nums = re.findall(r"-?\d+(?:\.\d+)?", response)
        candidate = nums[-1] if nums else ""
    norm_cand = candidate.replace(",", "").strip().lower()
    norm_exp = expected.replace(",", "").strip().lower()
    has_steps = len([ln for ln in response.strip().splitlines() if ln.strip()]) >= int(spec.get("min_lines", 2))
    if norm_cand == norm_exp and has_steps:
        return TaskVerdict(True, 1.0, "exact_reasoning_match", {"answer": candidate})
    if norm_cand == norm_exp:
        return TaskVerdict(False, 0.5, "answer_without_reasoning_steps", {"answer": candidate})
    return TaskVerdict(False, 0.0, f"wrong_answer:{candidate!r}!={expected!r}", {"answer": candidate})


def _verify_summarization(response: str, spec: dict[str, Any]) -> TaskVerdict:
    source = str(spec["source"])
    max_words = int(spec["max_words"])
    required_terms: list[str] = list(spec["required_terms"])
    summary = response.strip()
    words = summary.split()
    if not words:
        return TaskVerdict(False, 0.0, "empty_summary", {})
    if len(words) > max_words:
        return TaskVerdict(False, 0.25, f"exceeds_word_budget:{len(words)}>{max_words}", {"words": len(words)})

    src_nums = set(re.findall(r"(?<![\w.])[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)%?(?!\w)", source))
    sum_nums = set(re.findall(r"(?<![\w.])[+\-−]?(?:\d+(?:\.\d+)?|\.\d+)%?(?!\w)", summary))
    hallucinated_nums = sorted(sum_nums - src_nums)
    if hallucinated_nums:
        return TaskVerdict(False, 0.0, f"hallucinated_numbers:{hallucinated_nums}", {})

    low = summary.casefold()
    hits = sum(1 for term in required_terms if term.casefold() in low)
    recall = hits / len(required_terms) if required_terms else 1.0
    passed = recall == 1.0
    return TaskVerdict(
        passed,
        round(recall, 4),
        "verified_summary" if passed else f"missing_salient_terms:{hits}/{len(required_terms)}",
        {"words": len(words), "term_recall": recall},
    )


def _json_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_json_equal(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_json_equal(x, y) for x, y in zip(actual, expected))
    return actual == expected


def _verify_structured_extraction(response: str, spec: dict[str, Any]) -> TaskVerdict:
    raw_json = _extract_fenced_or_raw(response)
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        return TaskVerdict(False, 0.0, f"invalid_json:{exc.msg}", {})

    expected: dict[str, Any] = spec["expected_json"]
    if not isinstance(parsed, dict):
        return TaskVerdict(False, 0.0, "json_not_object", {})
    if set(parsed.keys()) != set(expected.keys()):
        return TaskVerdict(
            False,
            0.25,
            f"schema_keys_mismatch:{sorted(parsed.keys())}!={sorted(expected.keys())}",
            {},
        )
    matches = sum(1 for k, v in expected.items() if _json_equal(parsed[k], v))
    score = matches / len(expected) if expected else 1.0
    return TaskVerdict(
        matches == len(expected),
        round(score, 4),
        "exact_structured_json" if matches == len(expected) else f"field_mismatch:{matches}/{len(expected)}",
        {"matched_fields": matches},
    )


_NODE = re.compile(r'(?P<id>\[\*\]|[\w-]+?)(?:\["(?P<quoted>[^"]+)"\]|\[(?P<label>[^\]()]+)\])?(?=\s|$|-->|-\.->|->>)')
_ARROW = re.compile(r'\s*(?:-->>|-->|-\.->|->>)(?:\|[^|]+\|)?\s*')


def _mermaid_chain(lines: list[str], required: list[str], min_edges: int) -> TaskVerdict:
    labels: dict[str, str] = {}
    edges: set[tuple[str, str]] = set()
    for line in lines:
        for statement in line.split("%%", 1)[0].split(";"):
            rest = statement.strip()
            previous = None
            while rest:
                node = _NODE.match(rest)
                if node is None:
                    return TaskVerdict(False, 0.0, "invalid_mermaid_endpoint", {})
                identifier = node["id"]
                label = node["quoted"] or node["label"]
                if label:
                    labels[identifier] = label
                if previous is not None:
                    edges.add((previous, identifier))
                rest = rest[node.end():].strip()
                if not rest or rest.startswith(":"):  # state transition annotation
                    break
                arrow = _ARROW.match(rest)
                if arrow is None or not rest[arrow.end():].strip():
                    return TaskVerdict(False, 0.0, "invalid_mermaid_edge", {})
                previous, rest = identifier, rest[arrow.end():].strip()

    def matches(identifier: str, concept: str) -> bool:
        label = labels.get(identifier, identifier)
        return re.search(r"(?<!\w)" + re.escape(concept) + r"(?!\w)", label, re.I) is not None

    # Walk the same connected vertices even when a label appears on two nodes.
    reachable = {src for src, _ in edges if required and matches(src, required[0])}
    for concept in required[1:]:
        reachable = {dst for src, dst in edges if src in reachable and matches(dst, concept)}
    passed = bool(reachable) and len(edges) >= min_edges
    return TaskVerdict(passed, float(passed), "verified_mermaid" if passed else "incomplete_topology",
                       {"edges": len(edges)})


def _verify_diagrams_and_charts(response: str, spec: dict[str, Any]) -> TaskVerdict:
    mode = spec["subkind"]
    body = _extract_fenced_or_raw(response)
    if mode == "mermaid":
        lines = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("%%")]
        if not lines or not any(lines[0].startswith(h) for h in _MERMAID_HEADERS):
            return TaskVerdict(False, 0.0, "invalid_mermaid_header", {})
        if not lines[0].startswith(str(spec["expected_header"])):
            return TaskVerdict(False, 0.2, f"wrong_diagram_type:{lines[0]}", {})
        # Reject unquoted parentheses inside square-bracket node labels e.g. id[Foo (Bar)].
        for ln in lines[1:]:
            for bracket_content in re.findall(r"\b[A-Za-z0-9_]+\[(?!\")([^\]]+)\]", ln):
                if "(" in bracket_content or ")" in bracket_content:
                    return TaskVerdict(False, 0.3, "unquoted_special_chars_in_node_label", {"line": ln})
        return _mermaid_chain(lines[1:], list(spec["required_concepts"]), int(spec["min_edges"]))

    # Quantitative JSON chart specification
    try:
        chart = json.loads(body)
    except json.JSONDecodeError as exc:
        return TaskVerdict(False, 0.0, f"invalid_chart_json:{exc.msg}", {})
    req_keys = {"chart_type", "x_labels", "series", "total"}
    if not isinstance(chart, dict) or req_keys != chart.keys():
        return TaskVerdict(False, 0.1, "missing_chart_schema_keys", {})
    if chart["chart_type"] != spec["expected_chart_type"]:
        return TaskVerdict(False, 0.2, "chart_type_mismatch", {})
    if chart.get("x_labels") != spec["expected_x_labels"]:
        return TaskVerdict(False, 0.3, "x_labels_mismatch", {})
    if not _json_equal(chart["series"], spec["expected_series"]):
        return TaskVerdict(False, 0.4, "series_values_not_conserved", {})
    if not _json_equal(chart["total"], spec["expected_total"]):
        return TaskVerdict(False, 0.7, "computed_total_mismatch", {})
    return TaskVerdict(True, 1.0, "verified_quantitative_chart", {})


def _verify_long_context(response: str, spec: dict[str, Any]) -> TaskVerdict:
    raw = _extract_fenced_or_raw(response)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return TaskVerdict(False, 0.0, "invalid_json_output", {})
    expected = spec["expected_json"]
    if parsed == expected:
        return TaskVerdict(True, 1.0, "exact_long_context_synthesis", {})
    if isinstance(parsed, dict):
        hits = sum(1 for k, v in expected.items() if parsed.get(k) == v)
        return TaskVerdict(False, round(hits / max(len(expected), 1), 4), "partial_long_context_match", {})
    return TaskVerdict(False, 0.0, "long_context_mismatch", {})


def verify_foundation_response(task: FoundationTask, response: str) -> TaskVerdict:
    dispatch = {
        FoundationDomain.CODING: _verify_coding,
        FoundationDomain.REASONING: _verify_reasoning,
        FoundationDomain.SUMMARIZATION: _verify_summarization,
        FoundationDomain.STRUCTURED_EXTRACTION: _verify_structured_extraction,
        FoundationDomain.DIAGRAMS_AND_CHARTS: _verify_diagrams_and_charts,
        FoundationDomain.LONG_CONTEXT: _verify_long_context,
    }
    return dispatch[task.domain](response, task.spec)


def _haystack_document(needles: list[tuple[int, str]], total_paragraphs: int = 28, filler_topic: str = "orbital thermal telemetry") -> str:
    needle_map = dict(needles)
    blocks: list[str] = []
    for idx in range(total_paragraphs):
        if idx in needle_map:
            blocks.append(f"[Section {idx + 1}] {needle_map[idx]}")
        else:
            blocks.append(
                f"[Section {idx + 1}] Routine {filler_topic} check nominal at cycle {100 + idx}; "
                f"radiator loop {idx % 4} steady at baseline pressure without anomalies."
            )
    return "\n\n".join(blocks)


def build_foundation_eval_suite() -> tuple[FoundationTask, ...]:
    """Held-out 12-task general-purpose NanoLM evaluation suite (2 per pillar)."""

    doc_sum_1 = (
        "Project Helios migrated its storage engine from B-tree pages to an append-only LSM tree in Q3 2026. "
        "Write amplification dropped by 64%, tail p99 read latency improved from 18.4 ms to 6.2 ms, and "
        "snapshot recovery time fell to 11 seconds across a 4.8 TB shard. However, compaction CPU utilization "
        "increased by 12% during peak ingestion windows."
    )
    doc_sum_2 = (
        "During the North Atlantic autonomous glider survey, vehicle Nereus-4 completed a 19-day transect "
        "covering 412 kilometers. Conductivity-temperature sensors recorded a 1.7 C thermocline shoaling "
        "at 85 meters depth while acoustic telemetry packet loss stayed at 2.3%."
    )
    long_doc_1 = _haystack_document(
        [
            (3, "Subsystem A17 recorded calibration code 'ZEPHYR-41' with drift 14 ppm."),
            (14, "Subsystem B09 recorded calibration code 'KESTREL-88' with drift 29 ppm."),
            (24, "Override rule: total drift equals Subsystem A17 drift plus Subsystem B09 drift."),
        ],
        total_paragraphs=30,
    )
    long_doc_2 = _haystack_document(
        [
            (2, "Vault shard alpha holds 130 active keys under custodian 'Lin'."),
            (11, "Vault shard beta holds 245 active keys under custodian 'Okafor'."),
            (26, "Decommission notice: 35 keys in shard beta were revoked at 04:00 UTC."),
        ],
        total_paragraphs=30,
        filler_topic="cluster heartbeat audit",
    )

    return (
        FoundationTask(
            task_id="eval-code-01",
            domain=FoundationDomain.CODING,
            system_prompt="Return only valid Python code inside a ```python``` block. No prose.",
            user_prompt=(
                "Write `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` "
                "that merges overlapping or touching closed intervals `[start, end]` and returns them sorted."
            ),
            spec={
                "assertions": [
                    "assert merge_intervals([]) == []",
                    "assert merge_intervals([(1, 3), (2, 6), (8, 10), (10, 12)]) == [(1, 6), (8, 12)]",
                    "assert merge_intervals([(5, 7), (1, 2)]) == [(1, 2), (5, 7)]",
                ]
            },
        ),
        FoundationTask(
            task_id="eval-code-02",
            domain=FoundationDomain.CODING,
            system_prompt="Return only valid Python code inside a ```python``` block. No prose.",
            user_prompt=(
                "Write `topological_order(graph: dict[str, list[str]]) -> list[str]` where `graph[u]` lists "
                "nodes `v` with a directed edge `u -> v`. Raise `ValueError` if the graph has a cycle. "
                "Break ties lexicographically."
            ),
            spec={
                "assertions": [
                    "assert topological_order({'a': ['c'], 'b': ['c'], 'c': []}) == ['a', 'b', 'c']",
                    "assert topological_order({'b': [], 'a': []}) == ['a', 'b']",
                    "assert topological_order({'z': [], 'a': ['b'], 'b': []}) == ['a', 'b', 'z']",
                    "assert topological_order({'x': []}) == ['x']",
                    "try:\n    topological_order({'a': ['b'], 'b': ['a']})\n    assert False\nexcept ValueError:\n    pass",
                ]
            },
        ),
        FoundationTask(
            task_id="eval-reason-01",
            domain=FoundationDomain.REASONING,
            system_prompt="Show concise step-by-step reasoning and end with exact line `FINAL_ANSWER: <number>`.",
            user_prompt=(
                "A cluster processes 360 jobs/hour with 6 workers. Two workers are taken offline for maintenance, "
                "while each remaining worker receives a kernel update boosting its per-worker throughput by 25%. "
                "How many jobs does the cluster process in 3 hours now?"
            ),
            spec={"expected_answer": "900", "min_lines": 2},
        ),
        FoundationTask(
            task_id="eval-reason-02",
            domain=FoundationDomain.REASONING,
            system_prompt="Show concise step-by-step reasoning and end with exact line `FINAL_ANSWER: <number>`.",
            user_prompt=(
                "A sequence satisfies a_1 = 3 and a_{n+1} = 2*a_n - 1 for all n >= 1. "
                "Compute the sum a_1 + a_2 + a_3 + a_4 + a_5."
            ),
            spec={"expected_answer": "67", "min_lines": 2},
        ),
        FoundationTask(
            task_id="eval-sum-01",
            domain=FoundationDomain.SUMMARIZATION,
            system_prompt="Summarize faithfully in at most 32 words. Preserve key metrics and trade-offs; invent no numbers.",
            user_prompt=f"Summarize this technical report:\n\n{doc_sum_1}",
            spec={
                "source": doc_sum_1,
                "max_words": 32,
                "required_terms": ["LSM", "64%", "6.2", "12%"],
            },
        ),
        FoundationTask(
            task_id="eval-sum-02",
            domain=FoundationDomain.SUMMARIZATION,
            system_prompt="Summarize faithfully in at most 28 words. Preserve key metrics; invent no numbers.",
            user_prompt=f"Summarize this field log:\n\n{doc_sum_2}",
            spec={
                "source": doc_sum_2,
                "max_words": 28,
                "required_terms": ["Nereus-4", "412", "1.7", "2.3%"],
            },
        ),
        FoundationTask(
            task_id="eval-json-01",
            domain=FoundationDomain.STRUCTURED_EXTRACTION,
            system_prompt="Return only a valid JSON object matching the requested keys. No markdown prose.",
            user_prompt=(
                "Extract into JSON with exact keys `service`, `incident_id`, `duration_minutes`, `root_cause`, `mitigated` (boolean):\n"
                "'Incident INC-4092 hit billing-gateway for 37 minutes due to expired TLS certificate; traffic is now mitigated.'"
            ),
            spec={
                "expected_json": {
                    "service": "billing-gateway",
                    "incident_id": "INC-4092",
                    "duration_minutes": 37,
                    "root_cause": "expired TLS certificate",
                    "mitigated": True,
                }
            },
        ),
        FoundationTask(
            task_id="eval-json-02",
            domain=FoundationDomain.STRUCTURED_EXTRACTION,
            system_prompt="Return only a valid JSON object matching the requested keys. No markdown prose.",
            user_prompt=(
                "Extract into JSON with exact keys `molecule`, `boiling_point_c`, `soluble_in_water` (boolean), `hazard_class`:\n"
                "'Compound acetone boils at 56 C, mixes completely in water, and carries hazard class flammable-liquid.'"
            ),
            spec={
                "expected_json": {
                    "molecule": "acetone",
                    "boiling_point_c": 56,
                    "soluble_in_water": True,
                    "hazard_class": "flammable-liquid",
                }
            },
        ),
        FoundationTask(
            task_id="eval-diag-01",
            domain=FoundationDomain.DIAGRAMS_AND_CHARTS,
            system_prompt=(
                "Return only a valid ```mermaid``` diagram block using `flowchart LR`. "
                "Always double-quote node labels containing parentheses, e.g. `A[\"Parser (AST)\"]`."
            ),
            user_prompt=(
                "Create a `flowchart LR` diagram for a compiler pipeline from `Lexer` to `Parser (AST)` "
                "to `Optimizer (SSA)` to `Codegen`."
            ),
            spec={
                "subkind": "mermaid",
                "expected_header": "flowchart LR",
                "required_concepts": ["Lexer", "Parser", "Optimizer", "Codegen"],
                "min_edges": 3,
            },
        ),
        FoundationTask(
            task_id="eval-chart-02",
            domain=FoundationDomain.DIAGRAMS_AND_CHARTS,
            system_prompt="Return only a valid JSON chart specification with exact keys `chart_type`, `x_labels`, `series`, `total`.",
            user_prompt=(
                "Compile a bar chart JSON (`chart_type`: `'bar'`) for quarterly compute spend: "
                "Q1=120, Q2=145, Q3=110, Q4=165. Include `x_labels`, integer `series` list, and exact `total`."
            ),
            spec={
                "subkind": "chart_json",
                "expected_chart_type": "bar",
                "expected_x_labels": ["Q1", "Q2", "Q3", "Q4"],
                "expected_series": [120, 145, 110, 165],
                "expected_total": 540,
            },
        ),
        FoundationTask(
            task_id="eval-long-01",
            domain=FoundationDomain.LONG_CONTEXT,
            system_prompt="Read the entire document and return only JSON with exact keys `a17_code`, `b09_code`, `total_drift_ppm`.",
            user_prompt=long_doc_1,
            spec={
                "expected_json": {
                    "a17_code": "ZEPHYR-41",
                    "b09_code": "KESTREL-88",
                    "total_drift_ppm": 43,
                }
            },
        ),
        FoundationTask(
            task_id="eval-long-02",
            domain=FoundationDomain.LONG_CONTEXT,
            system_prompt="Read the entire document and return only JSON with exact keys `alpha_custodian`, `beta_custodian`, `remaining_active_keys`.",
            user_prompt=long_doc_2,
            spec={
                "expected_json": {
                    "alpha_custodian": "Lin",
                    "beta_custodian": "Okafor",
                    "remaining_active_keys": 340,
                }
            },
        ),
    )


def build_foundation_train_curriculum() -> tuple[FoundationTask, ...]:
    """Disjoint 12-example verified training curriculum (zero overlap with eval suite)."""

    sum_src_a = (
        "AtlasDB replaced mutex lock tables with optimistic versioned latches in release 4.2, cutting "
        "contention stalls by 71% and raising 32-core throughput from 140k to 235k tx/s while adding "
        "8% memory overhead per index leaf."
    )
    sum_src_b = (
        "Solar array flight unit Falcon-7 endured 500 thermal vacuum cycles between -110 C and +125 C, "
        "retaining 98.4% conversion efficiency with zero delamination faults."
    )
    long_train_a = _haystack_document(
        [
            (1, "Node R12 assigned token 'ORION-19' with latency 12 ms."),
            (9, "Node R19 assigned token 'VEGA-52' with latency 27 ms."),
            (16, "Combined hop latency equals Node R12 latency plus Node R19 latency."),
        ],
        total_paragraphs=20,
        filler_topic="mesh packet routing",
    )
    long_train_b = _haystack_document(
        [
            (3, "Warehouse east stores 180 crates managed by 'Patel'."),
            (12, "Warehouse west stores 95 crates managed by 'Gomez'."),
            (18, "Dispatch manifest: 25 crates shipped out from warehouse east."),
        ],
        total_paragraphs=20,
        filler_topic="cold-chain sensor check",
    )

    examples = (
        FoundationTask(
            task_id="train-code-01",
            domain=FoundationDomain.CODING,
            system_prompt="Return only valid Python code inside a ```python``` block. No prose.",
            user_prompt="Write `run_length_encode(s: str) -> list[tuple[str, int]]` grouping consecutive identical characters.",
            spec={"assertions": ["assert run_length_encode('') == []", "assert run_length_encode('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]"]},
            reference_response=(
                "```python\n"
                "def run_length_encode(s: str) -> list[tuple[str, int]]:\n"
                "    if not s:\n"
                "        return []\n"
                "    out: list[tuple[str, int]] = []\n"
                "    cur, count = s[0], 1\n"
                "    for ch in s[1:]:\n"
                "        if ch == cur:\n"
                "            count += 1\n"
                "        else:\n"
                "            out.append((cur, count))\n"
                "            cur, count = ch, 1\n"
                "    out.append((cur, count))\n"
                "    return out\n"
                "```"
            ),
        ),
        FoundationTask(
            task_id="train-code-02",
            domain=FoundationDomain.CODING,
            system_prompt="Return only valid Python code inside a ```python``` block. No prose.",
            user_prompt="Write `min_window_sum(nums: list[int], k: int) -> int` returning the minimum sum of any contiguous subarray of length `k` (raise `ValueError` if `k <= 0` or `k > len(nums)`).",
            spec={"assertions": ["assert min_window_sum([4, 2, 1, 7, 3], 2) == 3"]},
            reference_response=(
                "```python\n"
                "def min_window_sum(nums: list[int], k: int) -> int:\n"
                "    if k <= 0 or k > len(nums):\n"
                "        raise ValueError('invalid window size')\n"
                "    cur = sum(nums[:k])\n"
                "    best = cur\n"
                "    for i in range(k, len(nums)):\n"
                "        cur += nums[i] - nums[i - k]\n"
                "        if cur < best:\n"
                "            best = cur\n"
                "    return best\n"
                "```"
            ),
        ),
        FoundationTask(
            task_id="train-reason-01",
            domain=FoundationDomain.REASONING,
            system_prompt="Show concise step-by-step reasoning and end with exact line `FINAL_ANSWER: <number>`.",
            user_prompt="A pipeline encodes 240 frames/sec across 4 GPUs. Adding 2 more identical GPUs and increasing per-GPU clock efficiency by 10% yields how many frames in 5 seconds?",
            spec={"expected_answer": "1980", "min_lines": 2},
            reference_response=(
                "1. Baseline per-GPU rate = 240 / 4 = 60 frames/sec.\n"
                "2. Upgraded per-GPU rate = 60 * 1.10 = 66 frames/sec.\n"
                "3. With 6 GPUs for 5 seconds: 6 * 66 * 5 = 1980.\n"
                "FINAL_ANSWER: 1980"
            ),
        ),
        FoundationTask(
            task_id="train-reason-02",
            domain=FoundationDomain.REASONING,
            system_prompt="Show concise step-by-step reasoning and end with exact line `FINAL_ANSWER: <number>`.",
            user_prompt="Given b_1 = 2 and b_{n+1} = 3*b_n + 1, compute b_1 + b_2 + b_3 + b_4.",
            spec={"expected_answer": "88", "min_lines": 2},
            reference_response=(
                "1. b_1 = 2, b_2 = 7, b_3 = 22, b_4 = 67.\n"
                "2. Wait: 2 + 7 + 22 + 67 = 98.\n"
                "FINAL_ANSWER: 98"
            ),
        ),
        FoundationTask(
            task_id="train-sum-01",
            domain=FoundationDomain.SUMMARIZATION,
            system_prompt="Summarize faithfully in at most 32 words. Preserve key metrics and trade-offs; invent no numbers.",
            user_prompt=f"Summarize this release note:\n\n{sum_src_a}",
            spec={"source": sum_src_a, "max_words": 32, "required_terms": ["AtlasDB", "71%", "235k", "8%"]},
            reference_response=(
                "AtlasDB 4.2 adopted optimistic versioned latches, reducing contention stalls by 71% and "
                "boosting throughput from 140k to 235k tx/s at an 8% leaf memory cost."
            ),
        ),
        FoundationTask(
            task_id="train-sum-02",
            domain=FoundationDomain.SUMMARIZATION,
            system_prompt="Summarize faithfully in at most 28 words. Preserve key metrics; invent no numbers.",
            user_prompt=f"Summarize this test report:\n\n{sum_src_b}",
            spec={"source": sum_src_b, "max_words": 28, "required_terms": ["Falcon-7", "500", "98.4%"]},
            reference_response=(
                "Flight array Falcon-7 survived 500 thermal vacuum cycles (-110 C to +125 C), preserving "
                "98.4% conversion efficiency with zero delamination faults."
            ),
        ),
        FoundationTask(
            task_id="train-json-01",
            domain=FoundationDomain.STRUCTURED_EXTRACTION,
            system_prompt="Return only a valid JSON object matching the requested keys. No markdown prose.",
            user_prompt=(
                "Extract into JSON with exact keys `job_name`, `build_number`, `test_count`, `status`, `retried` (boolean):\n"
                "'Nightly job tensor-core-ci build 814 passed all 1920 tests on first try without retries.'"
            ),
            spec={"expected_json": {"job_name": "tensor-core-ci", "build_number": 814, "test_count": 1920, "status": "passed", "retried": False}},
            reference_response='{"job_name": "tensor-core-ci", "build_number": 814, "test_count": 1920, "status": "passed", "retried": false}',
        ),
        FoundationTask(
            task_id="train-diag-01",
            domain=FoundationDomain.DIAGRAMS_AND_CHARTS,
            system_prompt=(
                "Return only a valid ```mermaid``` diagram block using `flowchart LR`. "
                "Always double-quote node labels containing parentheses, e.g. `A[\"Parser (AST)\"]`."
            ),
            user_prompt="Create a `flowchart LR` diagram from `Ingest` to `Tokenizer (BPE)` to `Encoder (GQA)` to `Verifier`.",
            spec={"subkind": "mermaid", "expected_header": "flowchart LR", "required_concepts": ["Ingest", "Tokenizer", "Encoder", "Verifier"], "min_edges": 3},
            reference_response=(
                "```mermaid\n"
                "flowchart LR\n"
                "    A[\"Ingest\"] --> B[\"Tokenizer (BPE)\"]\n"
                "    B --> C[\"Encoder (GQA)\"]\n"
                "    C --> D[\"Verifier\"]\n"
                "```"
            ),
        ),
        FoundationTask(
            task_id="train-chart-02",
            domain=FoundationDomain.DIAGRAMS_AND_CHARTS,
            system_prompt="Return only a valid JSON chart specification with exact keys `chart_type`, `x_labels`, `series`, `total`.",
            user_prompt="Compile a line chart JSON (`chart_type`: `'line'`) for monthly active nodes: Jan=40, Feb=55, Mar=65, Apr=80. Include `x_labels`, `series`, and exact `total`.",
            spec={"subkind": "chart_json", "expected_chart_type": "line", "expected_x_labels": ["Jan", "Feb", "Mar", "Apr"], "expected_series": [40, 55, 65, 80], "expected_total": 240},
            reference_response='{"chart_type": "line", "x_labels": ["Jan", "Feb", "Mar", "Apr"], "series": [40, 55, 65, 80], "total": 240}',
        ),
        FoundationTask(
            task_id="train-long-01",
            domain=FoundationDomain.LONG_CONTEXT,
            system_prompt="Read the entire document and return only JSON with exact keys `r12_token`, `r19_token`, `combined_latency_ms`.",
            user_prompt=long_train_a,
            spec={"expected_json": {"r12_token": "ORION-19", "r19_token": "VEGA-52", "combined_latency_ms": 39}},
            reference_response='{"r12_token": "ORION-19", "r19_token": "VEGA-52", "combined_latency_ms": 39}',
        ),
        FoundationTask(
            task_id="train-long-02",
            domain=FoundationDomain.LONG_CONTEXT,
            system_prompt="Read the entire document and return only JSON with exact keys `east_manager`, `west_manager`, `remaining_crates`.",
            user_prompt=long_train_b,
            spec={"expected_json": {"east_manager": "Patel", "west_manager": "Gomez", "remaining_crates": 250}},
            reference_response='{"east_manager": "Patel", "west_manager": "Gomez", "remaining_crates": 250}',
        ),
    )
    # Fix train-reason-02 spec to match exact 2 + 7 + 22 + 67 = 98!
    corrected: list[FoundationTask] = []
    for item in examples:
        if item.task_id == "train-reason-02":
            item = FoundationTask(
                task_id=item.task_id,
                domain=item.domain,
                system_prompt=item.system_prompt,
                user_prompt=item.user_prompt,
                spec={"expected_answer": "98", "min_lines": 2},
                reference_response="1. b_1 = 2, b_2 = 7, b_3 = 22, b_4 = 67.\n2. Sum = 2 + 7 + 22 + 67 = 98.\nFINAL_ANSWER: 98",
            )
        verdict = verify_foundation_response(item, item.reference_response)
        if not verdict.passed:
            raise RuntimeError(f"curriculum reference failed self-verification on {item.task_id}: {verdict}")
        corrected.append(item)
    return tuple(corrected)
