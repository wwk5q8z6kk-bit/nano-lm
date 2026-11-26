"""Unit tests for the 5-Engine `NanoLMSystem` and `LAB-FRONTIER-HARD` (Competition 1 vs Competition 2)."""

from __future__ import annotations

import json

from lab.system.controller import NanoLMSystem
from lab.system.frontier_bench import (
    build_frontier_100_suite,
    evaluate_competition_1_naked,
    evaluate_competition_2_system,
)
from lab.system.verification_engine import detect_circular_reasoning


def test_frontier_100_suite_integrity_and_anti_circular():
    mem_store, tasks = build_frontier_100_suite()
    assert len(tasks) == 100
    assert len(mem_store.docs) == 50

    # Active v2 retrieval beats deprecated v1 and flags conflict
    ret = mem_store.retrieve("What is max_inflight_ops for HeliosKV (`helios-kv`)?")
    assert ret.triggered and ret.confidence >= 0.30
    assert ret.has_conflict_or_deprecation
    assert "SPEC-HELIOS-KV-V2" in ret.formatted_evidence

    # Circular reasoning detector catches repeated n-gram loops and unclosed <think>
    looping = "<think>" + (" wait let us check the equation again step by step" * 5) + "</think> 42"
    assert detect_circular_reasoning(looping) is True
    assert detect_circular_reasoning("<think>unclosed thought trace") is True
    assert detect_circular_reasoning("<think>Step 1: x = 3 + 4 = 7.</think> \\boxed{7}") is False


def test_system_recovers_code_json_and_stateful_tools():
    mem_store, tasks = build_frontier_100_suite()
    # Select 1 coding task, 1 stateful tool workflow, and 1 conservation JSON task
    code_item = next(t for t in tasks if t["id"] == "fcod_topo_lex")
    tool_item = next(t for t in tasks if t["id"] == "tagent_mitigate_0")
    json_item = next(t for t in tasks if t["id"] == "fjson_cons_0")

    call_counter = {"code": 0, "json": 0}

    def mock_llm(msgs, think, max_tok):
        txt = msgs[-1]["content"]
        if "assert" in txt and "do NOT redefine" in txt:
            return "```python\nassert lex_toposort(2, [(0, 1), (1, 0)]) is None\n```"
        if "lex_toposort" in msgs[0]["content"]:
            call_counter["code"] += 1
            if call_counter["code"] == 1:
                # First try fails on cycle returning [] instead of None!
                return "```python\ndef lex_toposort(n, edges):\n    return []\n```"
            # Second try after Verification Engine feedback fixes it!
            return (
                "```python\nimport heapq\ndef lex_toposort(n, edges):\n"
                "    g = [[] for _ in range(n)]; ind = [0]*n\n"
                "    for u, v in edges: g[u].append(v); ind[v] += 1\n"
                "    pq = [i for i in range(n) if ind[i] == 0]; heapq.heapify(pq); out = []\n"
                "    while pq:\n"
                "        u = heapq.heappop(pq); out.append(u)\n"
                "        for v in g[u]:\n"
                "            ind[v] -= 1\n"
                "            if ind[v] == 0: heapq.heappush(pq, v)\n"
                "    return out if len(out) == n else None\n```"
            )
        if "INC-7100" in txt:
            if "Interaction Log:" not in txt:
                return json.dumps({"tool_calls": [{"name": "lookup_incident_context", "arguments": {"incident_id": "INC-7100"}}]})
            if "StaleLeaseToken" not in txt and "fresh_lease_token" not in txt:
                return json.dumps({"tool_calls": [{"name": "apply_shard_mitigation", "arguments": {"shard_id": "shard-eu-20", "lease_token": "LEASE-STALE-100", "target_rps": 500}}]})
            if "fresh_lease_token" not in txt:
                return json.dumps({"tool_calls": [{"name": "refresh_lease_token", "arguments": {"shard_id": "shard-eu-20"}}]})
            if "BelowMinimumFloor" not in txt and "MITIGATED" not in txt:
                return json.dumps({"tool_calls": [{"name": "apply_shard_mitigation", "arguments": {"shard_id": "shard-eu-20", "lease_token": "LEASE-LIVE-900", "target_rps": 500}}]})
            if "MITIGATED" not in txt:
                return json.dumps({"tool_calls": [{"name": "apply_shard_mitigation", "arguments": {"shard_id": "shard-eu-20", "lease_token": "LEASE-LIVE-900", "target_rps": 650}}]})
            return json.dumps({"tool_calls": [], "response": "Mitigated shard-eu-20 to 650 RPS."})
        if "FY26-Q1-R1" in msgs[0]["content"]:
            call_counter["json"] += 1
            if call_counter["json"] == 1:
                return '{"report_id": "bad"}'  # deliberately invalid schema on Pass 1
            return json.dumps(json_item["expected_obj"])
        return "0"

    subset = [code_item, tool_item, json_item]
    c1 = evaluate_competition_1_naked(mock_llm, subset, think=False)
    c2 = evaluate_competition_2_system(mock_llm, mem_store, subset)
    assert c1["passed"] == 1  # only Turn-1 tool lookup passes in single-pass naked mode
    assert c2["passed"] == 3  # System repairs code, recovers from 2 live API errors, and fixes JSON!
