"""NanoLM Comprehensive Deterministic Capability, Tool-Calling, RULER & Reliability Suite (v1).

Preserves and expands NanoLM's 6-domain foundation evaluation into an 8-pillar,
132-item deterministic benchmark with zero LLM-judge subjectivity, bootstrap 95% CIs,
failure-taxonomy classification, and 3-leaderboard (Intelligence / Reliability / Efficiency)
reporting:
  1. tool_calling          (32 items: single, parallel, abstention/When2Call, error_recovery)
  2. coding                (16 items: algorithms, stateful classes, debugging, parsers; subprocess exec)
  3. math_logic            (16 items: algebra, combinatorics, Bayes, modular, formal deduction)
  4. structured_json       (16 items: strict nested schemas, nulls, enums, arithmetic conservation)
  5. summarization         (16 items: technical synthesis, exception retention, zero hallucinated numbers)
  6. diagrams_charts       (12 items: Mermaid flowchart/sequence/state + chart conservation)
  7. long_context          (12 items: programmatic 4k/8k/16k multi-needle + multi-hop variable tracking)
  8. epistemic_reliability (12 items: false premises, unanswerable context QA, contradiction detection)
"""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nanolm.eval.validators import (
    verify_chart_spec,
    verify_math_or_logic,
    verify_mermaid_flowchart,
    verify_python_solution,
    verify_structured_json,
    verify_summary_faithfulness,
)

_THINK_RE = re.compile(r"(?:<\|think\|>|<think>).*?(?:<\|end_think\|>|</think>)", re.DOTALL)
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TOOL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def strip_thinking(text: str) -> str:
    s = _THINK_RE.sub("", text)
    # If an unclosed <think> block truncated at max_tokens, drop the unfinished prefix
    if "<think>" in s and "</think>" not in s:
        s = s[: s.find("<think>")]
    return s.strip()


def parse_tool_calls(response: str) -> tuple[list[dict[str, Any]] | None, str]:
    """Parse tool calls from JSON array/object, <tool_call> tags, or NO_TOOL_CALL abstention.

    Canonical prompt contract asks the model to output EITHER:
      {"tool_calls": [{"name": "...", "arguments": {...}}, ...]}
    OR (when no tool should be invoked):
      {"tool_calls": [], "response": "..."}
    Also accepts Qwen/Hermes <tool_call>{...}</tool_call> tags so raw checkpoints are evaluated fairly.
    """
    cleaned = strip_thinking(response)
    tags = _TOOL_TAG_RE.findall(cleaned)
    if tags:
        calls = []
        for t in tags:
            try:
                obj = json.loads(t)
                if isinstance(obj, dict) and "name" in obj:
                    calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
            except Exception:
                return None, "malformed_tool_tag_json"
        return calls, "ok"

    m = _JSON_BLOCK_RE.search(cleaned)
    cand = m.group(1).strip() if m else cleaned
    if not cand.startswith(("{", "[")):
        i1 = cand.find("{")
        if i1 != -1 and "}" in cand:
            cand = cand[i1 : cand.rfind("}") + 1]
    try:
        obj = json.loads(cand)
    except Exception:
        # If no JSON and no tool mentions, treat plain-text answer as 0 tool calls
        if '"tool_calls"' not in cleaned and "<tool_call>" not in cleaned:
            return [], "plain_text_no_tool"
        return None, "invalid_tool_json"

    if isinstance(obj, dict) and "tool_calls" in obj and isinstance(obj["tool_calls"], list):
        out = []
        for c in obj["tool_calls"]:
            if not isinstance(c, dict) or "name" not in c:
                return None, "invalid_call_entry"
            args = c.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    return None, "invalid_stringified_arguments"
            out.append({"name": c["name"], "arguments": args if isinstance(args, dict) else {}})
        return out, "ok"
    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
        return [{"name": obj["name"], "arguments": obj["arguments"]}], "ok"
    if isinstance(obj, list):
        if all(isinstance(c, dict) and "name" in c for c in obj):
            return [{"name": c["name"], "arguments": c.get("arguments", {})} for c in obj], "ok"
    return None, "unrecognized_json_structure"


TOOL_SYSTEM_INSTRUCTION = (
    "You have access to the provided tools. Respond ONLY with a valid JSON object of the form:\n"
    '{"tool_calls": [{"name": "<tool_name>", "arguments": {<args>}}]}\n'
    "If no tool is appropriate, or if required information is missing from the user's request, "
    'do NOT guess or fabricate arguments; instead return: {"tool_calls": [], "response": "<brief explanation or question>"}.'
)


def _tool_prompt(tools: list[dict[str, Any]], user_msg: str, history: str = "") -> str:
    schema_txt = json.dumps(tools, indent=2)
    hist_block = f"\nPrevious interaction:\n{history}\n" if history else ""
    return f"{TOOL_SYSTEM_INSTRUCTION}\n\nAvailable tools:\n```json\n{schema_txt}\n```{hist_block}\nUser: {user_msg}"


# ---------------------------------------------------------------------------
# Deterministic Task Builders (132 items total, zero overlap with training)
# ---------------------------------------------------------------------------

def build_tool_calling_suite() -> list[dict[str, Any]]:
    common_tools = [
        {
            "name": "query_metrics_tsdb",
            "description": "Query time-series infrastructure metrics for a service and region.",
            "parameters": {
                "type": "object",
                "required": ["service", "metric", "window_minutes"],
                "properties": {
                    "service": {"type": "string"},
                    "metric": {"type": "string", "enum": ["p99_latency_ms", "error_rate", "cpu_util", "rss_mb"]},
                    "region": {"type": "string", "enum": ["us-east", "eu-central", "ap-south"]},
                    "window_minutes": {"type": "integer"},
                },
            },
        },
        {
            "name": "scale_worker_pool",
            "description": "Update replica count for a Kubernetes deployment.",
            "parameters": {
                "type": "object",
                "required": ["deployment", "replicas", "dry_run"],
                "properties": {
                    "deployment": {"type": "string"},
                    "replicas": {"type": "integer"},
                    "dry_run": {"type": "boolean"},
                },
            },
        },
        {
            "name": "create_incident_ticket",
            "description": "Open a paged incident ticket for on-call engineering.",
            "parameters": {
                "type": "object",
                "required": ["title", "severity", "team"],
                "properties": {
                    "title": {"type": "string"},
                    "severity": {"type": "string", "enum": ["SEV1", "SEV2", "SEV3"]},
                    "team": {"type": "string"},
                },
            },
        },
        {
            "name": "lookup_invoice_ledger",
            "description": "Retrieve invoice status and line items by exact invoice_id.",
            "parameters": {
                "type": "object",
                "required": ["invoice_id", "include_line_items"],
                "properties": {
                    "invoice_id": {"type": "string"},
                    "include_line_items": {"type": "boolean"},
                },
            },
        },
    ]

    items: list[dict[str, Any]] = []

    # 1. Single-tool selection & argument accuracy (8 items)
    single_specs = [
        ("tc_single_01", "Check the p99 latency over the last 30 minutes for `auth-gateway` in `eu-central`.",
         "query_metrics_tsdb", lambda a: a.get("service") == "auth-gateway" and a.get("metric") == "p99_latency_ms" and a.get("region") == "eu-central" and a.get("window_minutes") == 30),
        ("tc_single_02", "Perform a dry-run scale of `ingest-consumer` to 18 replicas.",
         "scale_worker_pool", lambda a: a.get("deployment") == "ingest-consumer" and a.get("replicas") == 18 and a.get("dry_run") is True),
        ("tc_single_03", "Scale `billing-cron` down to 2 replicas immediately (not a dry run).",
         "scale_worker_pool", lambda a: a.get("deployment") == "billing-cron" and a.get("replicas") == 2 and a.get("dry_run") is False),
        ("tc_single_04", "Open a SEV2 incident for the `storage-sre` team titled `WAL replication lag > 120s`.",
         "create_incident_ticket", lambda a: a.get("severity") == "SEV2" and a.get("team") == "storage-sre" and "WAL" in str(a.get("title", ""))),
        ("tc_single_05", "Look up invoice `INV-2026-8841` including all line items.",
         "lookup_invoice_ledger", lambda a: a.get("invoice_id") == "INV-2026-8841" and a.get("include_line_items") is True),
        ("tc_single_06", "Fetch `rss_mb` memory usage for `vector-indexer` in `ap-south` over the past 120 minutes.",
         "query_metrics_tsdb", lambda a: a.get("service") == "vector-indexer" and a.get("metric") == "rss_mb" and a.get("region") == "ap-south" and a.get("window_minutes") == 120),
        ("tc_single_07", "Check summary status ONLY (no line items) for invoice `INV-2026-1092`.",
         "lookup_invoice_ledger", lambda a: a.get("invoice_id") == "INV-2026-1092" and a.get("include_line_items") is False),
        ("tc_single_08", "Pull the 15-minute `error_rate` for `checkout-api` in `us-east`.",
         "query_metrics_tsdb", lambda a: a.get("service") == "checkout-api" and a.get("metric") == "error_rate" and a.get("region") == "us-east" and a.get("window_minutes") == 15),
    ]
    for tid, umsg, exp_name, pred in single_specs:
        items.append({
            "id": tid, "sub": "single_tool", "prompt": _tool_prompt(common_tools, umsg),
            "check": lambda calls, en=exp_name, pr=pred: (
                (True, "passed") if (len(calls) == 1 and calls[0]["name"] == en and pr(calls[0]["arguments"]))
                else (False, "missed_tool_call" if len(calls) == 0 else ("wrong_tool_selected" if calls[0]["name"] != en else "argument_value_error"))
            ),
        })

    # 2. Multi-tool parallel orchestration (8 items)
    parallel_specs = [
        ("tc_par_01", "Check 60-minute `cpu_util` for `search-ranker` in `us-east` AND do a dry-run scale of `search-ranker` to 12 replicas.",
         [("query_metrics_tsdb", lambda a: a.get("service") == "search-ranker" and a.get("metric") == "cpu_util" and a.get("window_minutes") == 60),
          ("scale_worker_pool", lambda a: a.get("deployment") == "search-ranker" and a.get("replicas") == 12 and a.get("dry_run") is True)]),
        ("tc_par_02", "Compare 15-minute `error_rate` for `edge-proxy` across both `us-east` and `eu-central`.",
         [("query_metrics_tsdb", lambda a: a.get("service") == "edge-proxy" and a.get("region") == "us-east" and a.get("window_minutes") == 15),
          ("query_metrics_tsdb", lambda a: a.get("service") == "edge-proxy" and a.get("region") == "eu-central" and a.get("window_minutes") == 15)]),
        ("tc_par_03", "Look up `INV-9001` with line items and `INV-9002` without line items.",
         [("lookup_invoice_ledger", lambda a: a.get("invoice_id") == "INV-9001" and a.get("include_line_items") is True),
          ("lookup_invoice_ledger", lambda a: a.get("invoice_id") == "INV-9002" and a.get("include_line_items") is False)]),
        ("tc_par_04", "Open a SEV1 ticket for `net-eng` titled `BGP flap us-east` and fetch 10-minute `error_rate` for `core-router` in `us-east`.",
         [("create_incident_ticket", lambda a: a.get("severity") == "SEV1" and a.get("team") == "net-eng"),
          ("query_metrics_tsdb", lambda a: a.get("service") == "core-router" and a.get("metric") == "error_rate" and a.get("window_minutes") == 10)]),
        ("tc_par_05", "Scale `etl-shard-a` to 6 replicas (live, dry_run=false) and `etl-shard-b` to 6 replicas (dry_run=true).",
         [("scale_worker_pool", lambda a: a.get("deployment") == "etl-shard-a" and a.get("replicas") == 6 and a.get("dry_run") is False),
          ("scale_worker_pool", lambda a: a.get("deployment") == "etl-shard-b" and a.get("replicas") == 6 and a.get("dry_run") is True)]),
        ("tc_par_06", "Query 45-minute `p99_latency_ms` and 45-minute `rss_mb` for `fraud-scorer` in `ap-south`.",
         [("query_metrics_tsdb", lambda a: a.get("service") == "fraud-scorer" and a.get("metric") == "p99_latency_ms" and a.get("window_minutes") == 45),
          ("query_metrics_tsdb", lambda a: a.get("service") == "fraud-scorer" and a.get("metric") == "rss_mb" and a.get("window_minutes") == 45)]),
        ("tc_par_07", "Retrieve `INV-7710` with line items and open a SEV3 ticket for `finops` titled `Audit INV-7710`.",
         [("lookup_invoice_ledger", lambda a: a.get("invoice_id") == "INV-7710" and a.get("include_line_items") is True),
          ("create_incident_ticket", lambda a: a.get("severity") == "SEV3" and a.get("team") == "finops")]),
        ("tc_par_08", "Dry-run scale `cache-tier` to 10 replicas, `queue-tier` to 8 replicas, and `api-tier` to 14 replicas.",
         [("scale_worker_pool", lambda a: a.get("deployment") == "cache-tier" and a.get("replicas") == 10 and a.get("dry_run") is True),
          ("scale_worker_pool", lambda a: a.get("deployment") == "queue-tier" and a.get("replicas") == 8 and a.get("dry_run") is True),
          ("scale_worker_pool", lambda a: a.get("deployment") == "api-tier" and a.get("replicas") == 14 and a.get("dry_run") is True)]),
    ]
    for tid, umsg, reqs in parallel_specs:
        def _chk_par(calls: list[dict[str, Any]], rq=reqs) -> tuple[bool, str]:
            if len(calls) != len(rq):
                return False, ("missed_tool_call" if len(calls) < len(rq) else "extra_tool_call")
            used = set()
            for en, pr in rq:
                found = False
                for idx, c in enumerate(calls):
                    if idx not in used and c["name"] == en and pr(c["arguments"]):
                        used.add(idx)
                        found = True
                        break
                if not found:
                    return False, "parallel_argument_or_name_mismatch"
            return True, "passed"
        items.append({"id": tid, "sub": "multi_parallel", "prompt": _tool_prompt(common_tools, umsg), "check": _chk_par})

    # 3. Abstention / When2Call (8 items: irrelevant request or missing required parameter)
    abstain_msgs = [
        ("tc_abs_01", "Explain the difference between TCP congestion window and receive window in two sentences."),
        ("tc_abs_02", "Translate 'Distributed consensus requires a quorum' into French."),
        ("tc_abs_03", "Scale the worker pool immediately! (I haven't decided which deployment or how many replicas yet.)"),
        ("tc_abs_04", "Look up my latest invoice please — I don't know the invoice ID."),
        ("tc_abs_05", "Send a Slack message to `@alex` saying the deploy succeeded."),
        ("tc_abs_06", "Delete the production Postgres database `users_primary`."),
        ("tc_abs_07", "What is 19 * 23 + 14?"),
        ("tc_abs_08", "Book a flight from Boston to Zurich next Tuesday."),
    ]
    for tid, umsg in abstain_msgs:
        items.append({
            "id": tid, "sub": "abstention", "prompt": _tool_prompt(common_tools, umsg),
            "check": lambda calls: (True, "passed") if len(calls) == 0 else (False, "false_positive_tool_call"),
        })

    # 4. Multi-turn Execution Feedback & Error Recovery (8 items)
    recovery_specs = [
        ("tc_rec_01",
         'Assistant called: {"name": "query_metrics_tsdb", "arguments": {"service": "auth-gateway", "metric": "p95_latency", "region": "us-east", "window_minutes": 20}}\n'
         'Tool error: {"error": "InvalidEnumError", "message": "metric \'p95_latency\' not supported; use \'p99_latency_ms\'."}',
         "Please proceed with the closest supported latency metric for that exact request.",
         lambda calls: len(calls) == 1 and calls[0]["name"] == "query_metrics_tsdb" and calls[0]["arguments"].get("metric") == "p99_latency_ms" and calls[0]["arguments"].get("service") == "auth-gateway" and calls[0]["arguments"].get("window_minutes") == 20),
        ("tc_rec_02",
         'Assistant called: {"name": "scale_worker_pool", "arguments": {"deployment": "pay-svc", "replicas": 24, "dry_run": false}}\n'
         'Tool error: {"error": "EntityNotFound", "message": "Deployment \'pay-svc\' not found. Did you mean \'payment-service\'?"}',
         "Yes, run it against `payment-service` instead, but make it a dry run first.",
         lambda calls: len(calls) == 1 and calls[0]["name"] == "scale_worker_pool" and calls[0]["arguments"].get("deployment") == "payment-service" and calls[0]["arguments"].get("replicas") == 24 and calls[0]["arguments"].get("dry_run") is True),
        ("tc_rec_03",
         'Assistant called: {"name": "query_metrics_tsdb", "arguments": {"service": "ledger-db", "metric": "cpu_util", "region": "eu-west", "window_minutes": 60}}\n'
         'Tool error: {"error": "InvalidRegion", "message": "Region \'eu-west\' does not exist; EU workloads run in \'eu-central\'."}',
         "Retry in the valid EU region.",
         lambda calls: len(calls) == 1 and calls[0]["name"] == "query_metrics_tsdb" and calls[0]["arguments"].get("region") == "eu-central" and calls[0]["arguments"].get("service") == "ledger-db"),
        ("tc_rec_04",
         'Assistant called: {"name": "create_incident_ticket", "arguments": {"title": "Disk full", "severity": "CRITICAL", "team": "infra"}}\n'
         'Tool error: {"error": "ValidationError", "message": "severity must be one of [\'SEV1\', \'SEV2\', \'SEV3\']."}',
         "Map CRITICAL to the highest severity tier and retry.",
         lambda calls: len(calls) == 1 and calls[0]["name"] == "create_incident_ticket" and calls[0]["arguments"].get("severity") == "SEV1" and calls[0]["arguments"].get("team") == "infra"),
        ("tc_rec_05",
         'Assistant called: {"name": "lookup_invoice_ledger", "arguments": {"invoice_id": "8821", "include_line_items": true}}\n'
         'Tool error: {"error": "FormatError", "message": "invoice_id must have prefix \'INV-2026-\' (e.g. \'INV-2026-8821\')."}',
         "Fix the ID format and retry.",
         lambda calls: len(calls) == 1 and calls[0]["name"] == "lookup_invoice_ledger" and calls[0]["arguments"].get("invoice_id") == "INV-2026-8821" and calls[0]["arguments"].get("include_line_items") is True),
        ("tc_rec_06",
         'Assistant called: {"name": "scale_worker_pool", "arguments": {"deployment": "ml-infer", "replicas": 50, "dry_run": false}}\n'
         'Tool error: {"error": "QuotaExceeded", "message": "Namespace max_replicas ceiling is 32."}',
         "Scale to the maximum allowed ceiling instead (live, not dry run).",
         lambda calls: len(calls) == 1 and calls[0]["name"] == "scale_worker_pool" and calls[0]["arguments"].get("deployment") == "ml-infer" and calls[0]["arguments"].get("replicas") == 32 and calls[0]["arguments"].get("dry_run") is False),
        ("tc_rec_07",
         'Assistant called: {"name": "query_metrics_tsdb", "arguments": {"service": "feed-api", "metric": "error_rate", "region": "us-east", "window_minutes": 10080}}\n'
         'Tool error: {"error": "RangeLimitError", "message": "window_minutes cannot exceed 1440 (24h)."}',
         "Use the maximum supported window.",
         lambda calls: len(calls) == 1 and calls[0]["name"] == "query_metrics_tsdb" and calls[0]["arguments"].get("window_minutes") == 1440 and calls[0]["arguments"].get("service") == "feed-api"),
        ("tc_rec_08",
         'Assistant called: {"name": "lookup_invoice_ledger", "arguments": {"invoice_id": "INV-2026-0000", "include_line_items": true}}\n'
         'Tool error: {"error": "PermissionDenied", "message": "Ledger read access revoked for service account; do not retry."}',
         "Understood — let me know why it failed without calling any more tools.",
         lambda calls: len(calls) == 0),
    ]
    for tid, hist, umsg, ok_fn in recovery_specs:
        items.append({
            "id": tid, "sub": "error_recovery", "prompt": _tool_prompt(common_tools, umsg, history=hist),
            "check": lambda calls, fn=ok_fn: (True, "passed") if fn(calls) else (False, "error_recovery_failed"),
        })
    return items


def build_coding_suite() -> list[dict[str, Any]]:
    return [
        {"id": "code_01_dedup", "prompt": "Write a Python function `stable_unique_positive(items: list[int]) -> list[int]` keeping strictly positive integers (>0) in first-seen order without duplicates.",
         "harness": "assert stable_unique_positive([4,-2,4,0,7,1,7]) == [4,7,1]\nassert stable_unique_positive([-5,0]) == []"},
        {"id": "code_02_intervals", "prompt": "Write `coalesce_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]` merging overlapping or touching closed intervals `[start, end]` and returning them sorted.",
         "harness": "assert coalesce_windows([(5,8),(1,3),(3,6),(10,12)]) == [(1,8),(10,12)]\nassert coalesce_windows([]) == []"},
        {"id": "code_03_brackets", "prompt": "Write `is_balanced_brackets(text: str) -> bool` checking whether `()`, `[]`, and `{}` are properly nested and balanced, ignoring other characters.",
         "harness": "assert is_balanced_brackets('fn(a[0], {k: (1+2)})') is True\nassert is_balanced_brackets('([)]') is False\nassert is_balanced_brackets('(((') is False"},
        {"id": "code_04_top2", "prompt": "Write `top_two_tokens(words: list[str]) -> list[str]` returning up to 2 most frequent strings ordered by descending count, breaking ties alphabetically.",
         "harness": "assert top_two_tokens(['zeta','beta','zeta','alpha','beta','gamma']) == ['beta','zeta']\nassert top_two_tokens(['solo']) == ['solo']"},
        {"id": "code_05_topo", "prompt": "Write `topo_order(num_nodes: int, edges: list[tuple[int, int]]) -> list[int]` for nodes `0..num_nodes-1` where `(u, v)` means `u` before `v`. Break ties by smallest node ID first (lexicographically smallest topological sort). Raise `ValueError` if a cycle exists.",
         "harness": "assert topo_order(4, [(0,2),(1,2),(2,3)]) == [0,1,2,3]\ntry:\n    topo_order(2, [(0,1),(1,0)])\n    assert False\nexcept ValueError:\n    pass"},
        {"id": "code_06_lru", "prompt": "Implement a Python class `TinyLRU(capacity: int)` with `get(key: str) -> int` (return `-1` if missing) and `put(key: str, val: int) -> None` evicting the least-recently-used key when full.",
         "harness": "def _t():\n c=TinyLRU(2); c.put('a',1); c.put('b',2); assert c.get('a')==1; c.put('c',3); assert c.get('b')==-1; assert c.get('c')==3\n_t()"},
        {"id": "code_07_rle", "prompt": "Write `rle_encode(s: str) -> str` encoding runs as `<char><count>` (e.g. `'aaabbc' -> 'a3b2c1'`, `'' -> ''`) and `rle_decode(enc: str) -> str` inverting it for multi-digit counts.",
         "harness": "assert rle_encode('aaabbc') == 'a3b2c1'\nassert rle_decode('x12y1z3') == 'x'*12 + 'y' + 'zzz'\nassert rle_encode('') == '' and rle_decode('') == ''"},
        {"id": "code_08_lis", "prompt": "Write `longest_increasing_len(nums: list[int]) -> int` returning the length of the longest strictly increasing subsequence in $O(n \\log n)$ time.",
         "harness": "assert longest_increasing_len([10,9,2,5,3,7,101,18]) == 4\nassert longest_increasing_len([7,7,7]) == 1\nassert longest_increasing_len([]) == 0"},
        {"id": "code_09_median_stream", "prompt": "Write `running_medians(nums: list[int]) -> list[float]` returning the median after each element is inserted.",
         "harness": "assert running_medians([2, 4, 1, 5]) == [2.0, 3.0, 2.0, 3.0]"},
        {"id": "code_10_semver", "prompt": "Write `sort_versions(versions: list[str]) -> list[str]` sorting `'MAJOR.MINOR.PATCH'` strings numerically (e.g. `'1.10.0'` after `'1.2.9'`).",
         "harness": "assert sort_versions(['1.10.0','1.2.9','0.9.1','1.2.10']) == ['0.9.1','1.2.9','1.2.10','1.10.0']"},
        {"id": "code_11_min_window", "prompt": "Write `shortest_subarray_ge(nums: list[int], target: int) -> int` for positive integers `nums`, returning minimum contiguous length with sum >= `target`, or `0` if impossible.",
         "harness": "assert shortest_subarray_ge([2,3,1,2,4,3], 7) == 2\nassert shortest_subarray_ge([1,1,1], 10) == 0"},
        {"id": "code_12_dijkstra", "prompt": "Write `min_latency(n: int, flights: list[tuple[int,int,int]], src: int, dst: int) -> int` for directed weighted edges `(u,v,w)` (`w >= 0`), returning shortest distance or `-1` if unreachable.",
         "harness": "assert min_latency(4, [(0,1,4),(0,2,1),(2,1,2),(1,3,1)], 0, 3) == 4\nassert min_latency(3, [(0,1,5)], 0, 2) == -1"},
        {"id": "code_13_expr_eval", "prompt": "Write `eval_arith(expr: str) -> int` evaluating non-negative integers, `+`, `-`, `*`, and parentheses `()` with standard operator precedence without using `eval`.",
         "harness": "assert eval_arith('3 + 4 * (2 - 5) + 20') == 11\nassert eval_arith('(10 - 2) * (3 + 1)') == 32"},
        {"id": "code_14_trie_prefix", "prompt": "Write `count_with_prefix(words: list[str], queries: list[str]) -> list[int]` returning for each query how many words in `words` start with that prefix.",
         "harness": "assert count_with_prefix(['apple','app','apricot','banana'], ['ap','app','b','cat']) == [3,2,1,0]"},
        {"id": "code_15_bipartite", "prompt": "Write `is_two_colorable(n: int, edges: list[tuple[int, int]]) -> bool` checking whether an undirected graph on `0..n-1` is bipartite.",
         "harness": "assert is_two_colorable(4, [(0,1),(1,2),(2,3),(3,0)]) is True\nassert is_two_colorable(3, [(0,1),(1,2),(2,0)]) is False"},
        {"id": "code_16_bugfix_retry", "prompt": "Fix this buggy function so `backoff_delays(base_ms, attempts, cap_ms)` returns `[min(cap_ms, base_ms * (2**i)) for i in range(attempts)]` and returns `[]` when `attempts <= 0`:\n```python\ndef backoff_delays(base_ms: int, attempts: int, cap_ms: int) -> list[int]:\n    return [max(cap_ms, base_ms * 2**i) for i in range(1, attempts)]\n```",
         "harness": "assert backoff_delays(100, 4, 500) == [100, 200, 400, 500]\nassert backoff_delays(100, 0, 500) == []"},
    ]


def build_math_logic_suite() -> list[dict[str, Any]]:
    sfx = " End your response with `FINAL_ANSWER: <answer>`."
    return [
        {"id": "ml_01", "prompt": "A cluster has 14 racks with 12 servers each (asset tag #8820 is just a label). 19 servers are offline. How many are online?" + sfx, "expected": "149"},
        {"id": "ml_02", "prompt": "Pool A processes 35 tasks/min and Pool B processes 25 tasks/min. Together how many minutes to finish 420 tasks?" + sfx, "expected": "7"},
        {"id": "ml_03", "prompt": "Find the remainder when $7^{100}$ is divided by $13$." + sfx, "expected": "9"},
        {"id": "ml_04", "prompt": "How many 4-letter codes can be formed from `{A,B,C,D,E,F}` without repeating any letter?" + sfx, "expected": "360"},
        {"id": "ml_05", "prompt": "A test for a rare fault (base rate 1%) has 99% sensitivity (true positive rate) and 2% false positive rate. If an alert fires, what is the exact percentage probability (rounded to nearest integer) that the fault is genuinely present?" + sfx, "expected": "33"},
        {"id": "ml_06", "prompt": "Solve for $x$: $3(2x - 5) + 4 = 2(x + 7) + 9$." + sfx, "expected": "6"},
        {"id": "ml_07", "prompt": "The sum of the first $n$ positive integers is $1275$. What is $n$?" + sfx, "expected": "50"},
        {"id": "ml_08", "prompt": "Four services (Kronos, Lyra, Mira, Nyx) boot sequentially with no ties. Lyra boots before Kronos. Mira boots before Lyra. Nyx boots after Kronos. Which boots third?" + sfx, "expected": "Kronos"},
        {"id": "ml_09", "prompt": "Three caches (C1, C2, C3) use distinct policies: LRU, LFU, FIFO. C1 uses neither LRU nor FIFO. C3 does not use FIFO. Which policy does C2 use?" + sfx, "expected": "FIFO"},
        {"id": "ml_10", "prompt": "A batch job's runtime drops by 20% in Stage 1 and then drops by 25% of the new runtime in Stage 2, reaching 45 minutes. What was the original runtime in minutes?" + sfx, "expected": "75"},
        {"id": "ml_11", "prompt": "In a group of 90 engineers, 58 use Rust, 47 use Go, and 12 use neither. How many use BOTH Rust and Go?" + sfx, "expected": "27"},
        {"id": "ml_12", "prompt": "What is $\\gcd(2520, 3960)$?" + sfx, "expected": "360"},
        {"id": "ml_13", "prompt": "A rectangle has perimeter 54 and area 180. What is the length of its longer side?" + sfx, "expected": "15"},
        {"id": "ml_14", "prompt": "If all `Quarks` are `Leptons` is FALSE, and in our toy ontology every `Baryon` is a `Glino` and NO `Glino` is a `Zeta`, can any `Baryon` be a `Zeta`? Answer `YES` or `NO`." + sfx, "expected": "NO"},
        {"id": "ml_15", "prompt": "How many trailing zeros are at the end of $125!$ in base 10?" + sfx, "expected": "31"},
        {"id": "ml_16", "prompt": "Train X leaves at 08:00 at 60 km/h. Train Y leaves the same station on a parallel track at 09:30 at 90 km/h in the same direction. How many hours after 09:30 does Y catch X?" + sfx, "expected": "3"},
    ]


def build_structured_json_suite() -> list[dict[str, Any]]:
    tasks = []
    specs = [
        ("json_01", "Extract JSON with keys `service` (str), `p99_ms` (int), `healthy` (bool), `owners` (list[str]) from: 'Service `atlas-kv` p99 hit 142 ms; status unhealthy; owned by `chen` and `patel`.' Output JSON only.",
         {"service": str, "p99_ms": int, "healthy": bool, "owners": list},
         lambda o: o["service"] == "atlas-kv" and o["p99_ms"] == 142 and o["healthy"] is False and set(o["owners"]) == {"chen", "patel"}),
        ("json_02", "Convert into JSON `{\"items\": [{\"sku\": str, \"qty\": int, \"unit_usd\": int}], \"total_usd\": int}` computing exact `total_usd`: 'Order: 3x NVMe-4TB at $220 each and 2x NIC-100G at $350 each.' Output JSON only.",
         {"items": list, "total_usd": int},
         lambda o: len(o["items"]) == 2 and o["total_usd"] == 1360),
        ("json_03", "Extract JSON `{\"cve\": str, \"cvss\": float, \"patched_version\": str | None}` (use JSON `null` if unpatched): 'Advisory CVE-2026-4410 (CVSS 8.7) affects `libtls` 2.1; no patched release exists yet.' Output JSON only.",
         {"cve": str, "cvss": float},
         lambda o: o["cve"] == "CVE-2026-4410" and abs(o["cvss"] - 8.7) < 1e-6 and o.get("patched_version") is None and "patched_version" in o),
        ("json_04", "Normalize to JSON `{\"region\": str, \"memory_mib\": int, \"gpu_count\": int}` converting GiB to MiB (`1 GiB = 1024 MiB`): 'Node in `ca-central` has 4 GPUs and 64 GiB RAM.' Output JSON only.",
         {"region": str, "memory_mib": int, "gpu_count": int},
         lambda o: o["region"] == "ca-central" and o["memory_mib"] == 65536 and o["gpu_count"] == 4),
    ]
    # Programmatically generate 12 additional deterministic parameterized extraction items (diverse schemas)
    rng = random.Random(20261117)
    for idx in range(5, 17):
        svc = f"svc-{['aurora','borealis','cobalt','delta','ember','fjord','granite','helios','ion','jade','kestrel','lumen'][idx-5]}"
        req_ok = rng.randint(1200, 9500)
        req_err = rng.randint(15, 240)
        lat = round(rng.uniform(11.5, 98.5), 1)
        rollback = (idx % 2 == 0)
        tasks.append({
            "id": f"json_{idx:02d}",
            "prompt": (
                f"Return ONLY a valid JSON object with exact keys `name` (str), `total_requests` (int), "
                f"`error_requests` (int), `latency_ms` (float), and `rollback_triggered` (bool) from:\n"
                f"Canary `{svc}` processed {req_ok} successful requests plus {req_err} failed requests "
                f"at {lat} ms latency; rollback was {'triggered' if rollback else 'not triggered'}."
            ),
            "fields": {"name": str, "total_requests": int, "error_requests": int, "latency_ms": (int, float), "rollback_triggered": bool},
            "pred": lambda o, s=svc, tot=req_ok+req_err, er=req_err, lt=lat, rb=rollback: (
                o["name"] == s and o["total_requests"] == tot and o["error_requests"] == er
                and abs(float(o["latency_ms"]) - lt) < 1e-4 and o["rollback_triggered"] is rb
            ),
        })
    out = [{"id": i, "prompt": p, "fields": f, "pred": pr} for i, p, f, pr in specs] + tasks
    return out


def build_summarization_suite() -> list[dict[str, Any]]:
    base = [
        ("sum_01", "Summarize in <= 45 words preserving all numbers and the exception:\n'Enabling zero-copy serialization in `velox_stream` cut RAM from 64 GB to 38 GB and boosted throughput by 22%, except on `arm64` nodes where misaligned pages caused a 4% regression.'",
         ["64", "38", "22%"], ["arm64", "4%"], ["50%", "zero regression", "x86 regression"], 45),
        ("sum_02", "Summarize in <= 50 words preserving exact metrics and scope:\n'Q3 migration moved 140 microservices to PostgreSQL 17, saving $42,000/month. However, 6 legacy billing tables remained on Oracle due to XA transaction locks.'",
         ["140", "17", "42,000"], ["6", "oracle"], ["all microservices", "zero remaining", "mysql"], 50),
        ("sum_03", "Synthesize both notes in <= 50 words with exact figures:\n'Note A: Model v4.1 reduced TTFT from 310 ms to 185 ms on H100.\nNote B: Accuracy held at 84.2% on MMLU-Pro, but multilingual recall dipped 1.8% on low-resource scripts.'",
         ["310", "185", "84.2%"], ["1.8%"], ["improved multilingual", "90%"], 50),
        ("sum_04", "Summarize in <= 45 words preserving numbers and caveats:\n'Solar array Block-7 generated 1,280 MWh in October (+14% YoY), although inverter #3 was derated for 9 days during transformer maintenance.'",
         ["1,280", "14%"], ["9", "inverter"], ["no downtime", "1,500"], 45),
    ]
    items = []
    for sid, pr, rf, rq, fc, mw in base:
        items.append({"id": sid, "prompt": pr, "facts": rf, "quals": rq, "forbidden": fc, "max_words": mw})
    rng = random.Random(9912)
    for idx in range(5, 17):
        nodes = rng.randint(24, 96)
        p99_old = rng.randint(210, 490)
        p99_new = rng.randint(85, 175)
        fail_pct = rng.randint(2, 9)
        arch = "riscv64" if idx % 2 == 0 else "s390x"
        items.append({
            "id": f"sum_{idx:02d}",
            "prompt": (
                f"Summarize in one or two sentences (<= 45 words), keeping all numbers and the caveat:\n"
                f"'Rollout across {nodes} clusters lowered p99 latency from {p99_old} ms to {p99_new} ms, "
                f"except on `{arch}` hosts where cold-cache misses increased tail errors by {fail_pct}%.'"
            ),
            "facts": [str(nodes), str(p99_old), str(p99_new)],
            "quals": [arch, f"{fail_pct}%"],
            "forbidden": ["0%", "all hosts succeeded", str(p99_old + 100)],
            "max_words": 45,
        })
    return items


def build_diagrams_charts_suite() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    flows = [
        ("diag_01", "Output ONLY a `mermaid` code block (`flowchart LR`) with edges `Client --> Gateway`, `Gateway -->|valid| Worker`, `Gateway -->|invalid| Reject`.",
         [("Client", "Gateway", None), ("Gateway", "Worker", "valid"), ("Gateway", "Reject", "invalid")]),
        ("diag_02", "Output ONLY a `mermaid` `flowchart TD` with edges `Ingest --> Parse`, `Parse -->|ok| Index`, `Parse -->|err| DLQ`, `Index --> Serve`.",
         [("Ingest", "Parse", None), ("Parse", "Index", "ok"), ("Parse", "DLQ", "err"), ("Index", "Serve", None)]),
        ("diag_03", "Output ONLY a `mermaid` `flowchart LR` with edges `Source --> Validate`, `Validate --> Transform`, `Transform -->|clean| Warehouse`, `Transform -->|dirty| Quarantine`.",
         [("Source", "Validate", None), ("Validate", "Transform", None), ("Transform", "Warehouse", "clean"), ("Transform", "Quarantine", "dirty")]),
        ("diag_04", "Output ONLY a `mermaid` `flowchart TD` with edges `Commit --> Build`, `Build -->|pass| Canary`, `Build -->|fail| Notify`, `Canary -->|healthy| Prod`, `Canary -->|rollback| Notify`.",
         [("Commit", "Build", None), ("Build", "Canary", "pass"), ("Build", "Notify", "fail"), ("Canary", "Prod", "healthy"), ("Canary", "Notify", "rollback")]),
        ("diag_05", "Output ONLY a `mermaid` `flowchart LR` with edges `Prompt --> Router`, `Router -->|fast| SmallLM`, `Router -->|deep| Verifier`, `SmallLM --> Output`, `Verifier --> Output`.",
         [("Prompt", "Router", None), ("Router", "SmallLM", "fast"), ("Router", "Verifier", "deep"), ("SmallLM", "Output", None), ("Verifier", "Output", None)]),
        ("diag_06", "Output ONLY a `mermaid` `flowchart TD` with edges `Sensor --> Buffer`, `Buffer --> FFT`, `FFT -->|anomaly| Alert`, `FFT -->|normal| Archive`.",
         [("Sensor", "Buffer", None), ("Buffer", "FFT", None), ("FFT", "Alert", "anomaly"), ("FFT", "Archive", "normal")]),
    ]
    for did, pr, edges in flows:
        items.append({"id": did, "kind": "mermaid", "prompt": pr, "edges": edges})

    charts = [
        ("chart_01", {"Compute": 420, "Storage": 185, "Network": 95}),
        ("chart_02", {"Q1": 1250, "Q2": 1480, "Q3": 1610, "Q4": 1920}),
        ("chart_03", {"Python": 54, "Rust": 28, "Go": 18}),
        ("chart_04", {"North": 312, "South": 288, "East": 405, "West": 195}),
        ("chart_05", {"Prefill": 68, "Decode": 214, "SpecVerify": 39}),
        ("chart_06", {"TierA": 910, "TierB": 640, "TierC": 250}),
    ]
    for cid, kv in charts:
        parts = ", ".join(f"{k}={v}" for k, v in kv.items())
        items.append({
            "id": cid,
            "kind": "chart",
            "prompt": (
                f'Output ONLY a JSON chart specification `{{"chart_type": "bar", "x_label": str, "y_label": str, '
                f'"x_values": list[str], "y_values": list[number]}}` preserving exact order and values for: {parts}.'
            ),
            "x_values": list(kv.keys()),
            "y_values": [float(v) for v in kv.values()],
        })
    return items


def build_long_context_suite() -> list[dict[str, Any]]:
    """12 programmatic RULER-style multi-needle & state-tracking tasks (2k–6k words -> ~3k–8k tokens)."""
    rng = random.Random(314159)
    filler_sentences = [
        "Background telemetry confirms nominal thermal headrooms across secondary optical transceivers during routine polling.",
        "Standard maintenance windows allocate forty-five minutes for non-disruptive kernel live-patching across leaf switches.",
        "Archival object buckets enforce erasure coding across six availability zones with asynchronous metadata compaction.",
        "Periodic scrubbers verify CRC32C checksums on cold parquet segments without impacting foreground read latency.",
    ]
    items = []
    for idx in range(12):
        n_sections = 45 if idx < 4 else (90 if idx < 8 else 150)  # ~2k, ~4k, ~6.8k tokens
        s1, s2 = rng.sample(range(3, n_sections - 3), 2)
        s1, s2 = min(s1, s2), max(s1, s2)
        v1, v2 = rng.randint(110, 490), rng.randint(110, 490)
        distractor = rng.randint(500, 900)
        key_tag = f"VAULT-{idx:02d}-OMEGA"
        decoy_tag = f"VAULT-{idx:02d}-DECOY"
        sections = []
        for s in range(n_sections):
            body = " ".join(rng.choices(filler_sentences, k=3))
            if s == s1:
                body += f" RECORD [{key_tag} PART_A]: baseline_delta = {v1}."
            elif s == s2:
                body += f" RECORD [{key_tag} PART_B]: increment_delta = {v2}."
            elif s == n_sections // 2:
                body += f" RECORD [{decoy_tag}]: decoy_value = {distractor} (ignore this decoy)."
            sections.append(f"[Section {s:03d}] {body}")
        doc = "\n".join(sections)
        items.append({
            "id": f"lc_{idx+1:02d}_sec{n_sections}",
            "prompt": (
                f"Read the log below and compute `baseline_delta + increment_delta` strictly for `{key_tag}` "
                f"(ignoring any decoy tags). End your response with `FINAL_ANSWER: <integer>`.\n\n{doc}"
            ),
            "expected": str(v1 + v2),
        })
    return items


def build_epistemic_reliability_suite() -> list[dict[str, Any]]:
    """12 adversarial hallucination-resistance, false-premise, and unanswerable context-QA prompts."""
    instr = (
        "Answer ONLY using facts explicitly stated in the provided context (or reject false premises). "
        "If the context does not contain the answer or the premise is false, output `FINAL_ANSWER: INSUFFICIENT_EVIDENCE`; "
        "otherwise output `FINAL_ANSWER: <exact_answer>`."
    )
    cases = [
        ("rel_01_unans", "Context: Cluster `nyx-01` runs 64 nodes on kernel 6.8 in `us-east`.\nQuestion: What kernel version does `nyx-01` run in `ap-south`?", "INSUFFICIENT_EVIDENCE"),
        ("rel_02_ans", "Context: Cluster `nyx-01` runs 64 nodes on kernel 6.8 in `us-east`.\nQuestion: How many nodes does `nyx-01` run in `us-east`?", "64"),
        ("rel_03_unans", "Context: Service `hermes` reduced memory to 12 GB using jemalloc.\nQuestion: How much CPU percentage did `hermes` save?", "INSUFFICIENT_EVIDENCE"),
        ("rel_04_false_prem", "Context: Python's GIL was made optional in free-threaded builds starting in Python 3.13 (PEP 703).\nQuestion: According to the context, which PEP removed the GIL in Python 2.7?", "INSUFFICIENT_EVIDENCE"),
        ("rel_05_ans", "Context: Python's GIL was made optional in free-threaded builds starting in Python 3.13 (PEP 703).\nQuestion: Which PEP is cited in the context?", "PEP 703"),
        ("rel_06_unans", "Context: Quarterly revenue hit $18.4M, up from $15.1M.\nQuestion: What was the net profit margin after tax?", "INSUFFICIENT_EVIDENCE"),
        ("rel_07_contradict", "Context: Doc 1 states backup retention is 30 days. Doc 2 states backup retention is 90 days. Neither supersedes the other.\nQuestion: If the documents conflict and no single authoritative retention period is established, output `INSUFFICIENT_EVIDENCE`.", "INSUFFICIENT_EVIDENCE"),
        ("rel_08_ans", "Context: Shard-4 replication factor is 3, quorum write requires 2 acks.\nQuestion: How many acks are required for a quorum write?", "2"),
        ("rel_09_unans", "Context: Model `borealis-2b` was trained on 1.8T tokens with RoPE theta 500,000.\nQuestion: What learning rate was used to train `borealis-2b`?", "INSUFFICIENT_EVIDENCE"),
        ("rel_10_false_prem", "Context: HTTP/3 runs over QUIC, which uses UDP rather than TCP.\nQuestion: According to the context, which TCP congestion algorithm does QUIC use?", "INSUFFICIENT_EVIDENCE"),
        ("rel_11_ans", "Context: Batch `B-77` processed 4,200 records with 12 schema rejections.\nQuestion: How many schema rejections occurred in `B-77`?", "12"),
        ("rel_12_unans", "Context: Failover from primary `db-a` to replica `db-b` completed in 14 seconds.\nQuestion: Who approved the failover ticket?", "INSUFFICIENT_EVIDENCE"),
    ]
    return [{"id": cid, "prompt": f"{instr}\n\n{q}", "expected": exp} for cid, q, exp in cases]


# ---------------------------------------------------------------------------
# Bootstrap CI, Failure Taxonomy & 3-Leaderboard Evaluation Engine
# ---------------------------------------------------------------------------

def bootstrap_ci(binary_outcomes: list[int], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not binary_outcomes:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(binary_outcomes)
    means = sorted(sum(rng.choices(binary_outcomes, k=n)) / n for _ in range(n_boot))
    return round(means[int(0.025 * n_boot)], 4), round(means[int(0.975 * n_boot)], 4)


def evaluate_all_pillars(
    generate_batch: Callable[[list[str]], list[str]],
    model_id: str,
    model_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all 8 pillars (132 items) using batched generation and return full report + 3 leaderboards."""
    t0 = time.perf_counter()
    tc = build_tool_calling_suite()
    co = build_coding_suite()
    ml = build_math_logic_suite()
    js = build_structured_json_suite()
    su = build_summarization_suite()
    dc = build_diagrams_charts_suite()
    lc = build_long_context_suite()
    rl = build_epistemic_reliability_suite()

    all_groups = [
        ("tool_calling", tc),
        ("coding", co),
        ("math_logic", ml),
        ("structured_json", js),
        ("summarization", su),
        ("diagrams_charts", dc),
        ("long_context", lc),
        ("epistemic_reliability", rl),
    ]
    flat_prompts = [it["prompt"] for _, grp in all_groups for it in grp]
    flat_outputs = generate_batch(flat_prompts)
    elapsed = time.perf_counter() - t0
    out_chars = sum(len(x) for x in flat_outputs)

    idx = 0
    pillars: dict[str, Any] = {}
    failures = Counter()

    for pname, grp in all_groups:
        outcomes: list[int] = []
        records: list[dict[str, Any]] = []
        for it in grp:
            resp = strip_thinking(flat_outputs[idx])
            idx += 1
            passed = False
            reason = "failed"
            if pname == "tool_calling":
                calls, parse_status = parse_tool_calls(resp)
                if calls is None:
                    passed, reason = False, parse_status
                else:
                    passed, reason = it["check"](calls)
            elif pname == "coding":
                passed, reason = verify_python_solution(resp, it["harness"])
            elif pname in ("math_logic", "long_context", "epistemic_reliability"):
                passed = verify_math_or_logic(resp, it["expected"])
                reason = "passed" if passed else f"wrong_answer_{pname}"
            elif pname == "structured_json":
                v_json, sem_ok, reason = verify_structured_json(resp, it["fields"], it["pred"])
                passed = v_json and sem_ok
            elif pname == "summarization":
                s_res = verify_summary_faithfulness(resp, it["facts"], it["quals"], it["forbidden"], it["max_words"])
                passed = bool(s_res["passed"])
                reason = "passed" if passed else (
                    "hallucinated_claim" if s_res.get("hallucinated")
                    else ("missing_qualification" if not s_res.get("qualification_preserved")
                          else ("word_limit_exceeded" if not s_res.get("concise") else "incomplete_facts"))
                )
            elif pname == "diagrams_charts":
                if it["kind"] == "mermaid":
                    syn_ok, sem_ok, reason = verify_mermaid_flowchart(resp, it["edges"])
                    passed = syn_ok and sem_ok
                else:
                    syn_ok, sem_ok, reason = verify_chart_spec(resp, {"bar", "column"}, it["x_values"], it["y_values"])
                    passed = syn_ok and sem_ok

            outcomes.append(1 if passed else 0)
            if not passed:
                failures[f"{pname}:{reason}"] += 1
            records.append({"id": it["id"], "passed": passed, "reason": reason})

        lo, hi = bootstrap_ci(outcomes)
        pillars[pname] = {
            "total": len(outcomes),
            "passed": sum(outcomes),
            "score": round(sum(outcomes) / len(outcomes), 4),
            "ci95": [lo, hi],
            "items": records,
        }

    # Three Independent Competitive Leaderboards
    intel_keys = ["coding", "math_logic", "tool_calling", "long_context", "diagrams_charts"]
    rel_keys = ["epistemic_reliability", "structured_json", "summarization"]
    intel_score = round(sum(pillars[k]["score"] for k in intel_keys) / len(intel_keys), 4)
    # Include tool-abstention accuracy explicitly in reliability too
    tc_abs = [r["passed"] for r, spec in zip(pillars["tool_calling"]["items"], tc) if spec["sub"] == "abstention"]
    tc_rec = [r["passed"] for r, spec in zip(pillars["tool_calling"]["items"], tc) if spec["sub"] == "error_recovery"]
    rel_score = round(
        (sum(pillars[k]["score"] for k in rel_keys) + (sum(tc_abs) / max(1, len(tc_abs)))) / (len(rel_keys) + 1),
        4,
    )

    meta = model_meta or {}
    total_params = meta.get("total_params")
    active_params = meta.get("active_params", total_params)
    approx_out_toks = max(1, out_chars // 4)

    return {
        "schema": "nanolm.competitive_eval_8pillar.v1",
        "model_id": model_id,
        "meta": meta,
        "total_items": idx,
        "overall_pass_rate": round(sum(p["passed"] for p in pillars.values()) / idx, 4),
        "leaderboards": {
            "A_intelligence": {
                "composite": intel_score,
                "coding": pillars["coding"]["score"],
                "math_logic": pillars["math_logic"]["score"],
                "tool_calling": pillars["tool_calling"]["score"],
                "tool_error_recovery": round(sum(tc_rec) / max(1, len(tc_rec)), 4),
                "long_context": pillars["long_context"]["score"],
                "diagrams_charts": pillars["diagrams_charts"]["score"],
            },
            "B_reliability": {
                "composite": rel_score,
                "epistemic_reliability": pillars["epistemic_reliability"]["score"],
                "tool_abstention_when2call": round(sum(tc_abs) / max(1, len(tc_abs)), 4),
                "structured_json": pillars["structured_json"]["score"],
                "summarization_faithfulness": pillars["summarization"]["score"],
            },
            "C_efficiency": {
                "total_params": total_params,
                "active_params": active_params,
                "eval_wall_seconds": round(elapsed, 2),
                "approx_output_tok_per_s": round(approx_out_toks / max(0.01, elapsed), 1),
                "intelligence_per_B_active_params": (
                    round(intel_score / (active_params / 1e9), 4) if active_params else None
                ),
            },
        },
        "failure_taxonomy_top15": failures.most_common(15),
        "pillars": pillars,
    }
