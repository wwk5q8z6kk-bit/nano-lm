"""Expanded Decontaminated Multi-Domain Training & RLVR Curriculum v2 for Experiment C2-X4.

Targets general structural/algorithmic patterns across all 14 capability dimensions
without sharing any prompt hash or 8-gram shingle with ``benchmark_15d`` (enforced
via fail-closed ``verify_zero_contamination``).
"""

from __future__ import annotations

import json
import math
import random

from nanolm_frontier.benchmark_15d import BenchmarkDim
from nanolm_frontier.curriculum import CurriculumExample, build_decontaminated_curriculum


def _long_train_haystack(needles: list[tuple[int, str]], n_blocks: int = 70, tag: str = "voltaic telemetry") -> str:
    nmap = dict(needles)
    out = []
    for i in range(n_blocks):
        if i in nmap:
            out.append(f"[Entry {i:02d}] {nmap[i]}")
        else:
            out.append(
                f"[Entry {i:02d}] Nominal {tag} sweep {800 + i}: inverter rail {i % 5} held 48.{i % 8} V steady."
            )
    return "\n".join(out)


def build_curriculum_v2(seed: int = 31415) -> tuple[CurriculumExample, ...]:
    """Return verified training/RLVR examples strictly disjoint from benchmark_15d."""
    rng = random.Random(seed)
    base = list(build_decontaminated_curriculum(seed=2026))
    extra: list[CurriculumExample] = []

    # 1. Richer Python algorithms (heaps/graphs, order statistics, postfix evaluation)
    extra.extend([
        CurriculumExample(
            "v2-cod-heap",
            BenchmarkDim.CODING,
            "Return only valid Python inside a ```python``` block. No prose.",
            "Write `shortest_grid_cost(n: int, arcs: list[tuple[int, int, int]], start: int, goal: int) -> int` using `heapq` on nodes `0..n-1` with non-negative weights `(u, v, w)`, returning `-1` if unreachable.",
            "```python\nimport heapq\n\ndef shortest_grid_cost(n: int, arcs: list[tuple[int, int, int]], start: int, goal: int) -> int:\n    adj = [[] for _ in range(n)]\n    for u, v, w in arcs:\n        adj[u].append((v, w))\n    dist = [10**18] * n\n    dist[start] = 0\n    pq = [(0, start)]\n    while pq:\n        d, u = heapq.heappop(pq)\n        if u == goal:\n            return d\n        if d > dist[u]:\n            continue\n        for v, w in adj[u]:\n            nd = d + w\n            if nd < dist[v]:\n                dist[v] = nd\n                heapq.heappush(pq, (nd, v))\n    return -1\n```",
            {"assertions": [
                "assert shortest_grid_cost(4, [(0,1,5),(0,2,2),(2,1,1),(1,3,3)], 0, 3) == 6",
                "assert shortest_grid_cost(3, [(0,1,2)], 0, 2) == -1",
            ]},
        ),
        CurriculumExample(
            "v2-cod-rank",
            BenchmarkDim.CODING,
            "Return only valid Python inside a ```python``` block. No prose.",
            "Write `nth_ordered_item(items: list[int], rank: int) -> int` returning the 1-based `rank`-th smallest element via `sorted(items)` without mutating `items`.",
            "```python\ndef nth_ordered_item(items: list[int], rank: int) -> int:\n    ordered = sorted(items)\n    return ordered[rank - 1]\n```",
            {"assertions": [
                "arr = [9, 3, 7, 1]",
                "assert nth_ordered_item(arr, 2) == 3",
                "assert arr == [9, 3, 7, 1]",
            ]},
        ),
        CurriculumExample(
            "v2-cod-postfix",
            BenchmarkDim.CODING,
            "Return only valid Python inside a ```python``` block. No prose.",
            "Write `compute_postfix(ops: list[str]) -> int` evaluating postfix tokens with `+`, `-`, `*`, and `/` (`int(a / b)`).",
            "```python\ndef compute_postfix(ops: list[str]) -> int:\n    st = []\n    for tok in ops:\n        if tok in ('+', '-', '*', '/'):\n            b = st.pop()\n            a = st.pop()\n            if tok == '+': st.append(a + b)\n            elif tok == '-': st.append(a - b)\n            elif tok == '*': st.append(a * b)\n            else: st.append(int(a / b))\n        else:\n            st.append(int(tok))\n    return st[-1]\n```",
            {"assertions": [
                "assert compute_postfix(['8', '3', '-', '2', '*']) == 10",
                "assert compute_postfix(['-9', '2', '/']) == -4",
            ]},
        ),
    ])

    # 2. Multi-call Tool Use `{"calls": [...]}` & Fallback Recovery
    extra.extend([
        CurriculumExample(
            "v2-tol-chain-1",
            BenchmarkDim.TOOL_USE,
            'Return only JSON `{"calls": [...]}`.',
            "Available tools: `lookup_user(uid: int)`, `grant_badge(uid: int, badge: str)`. First lookup uid 88, then grant badge `'verified'` to uid 88.",
            json.dumps({"calls": [
                {"tool": "lookup_user", "args": {"uid": 88}},
                {"tool": "grant_badge", "args": {"uid": 88, "badge": "verified"}},
            ]}),
            {"expected_calls": [
                {"tool": "lookup_user", "args": {"uid": 88}},
                {"tool": "grant_badge", "args": {"uid": 88, "badge": "verified"}},
            ]},
        ),
        CurriculumExample(
            "v2-tol-chain-2",
            BenchmarkDim.TOOL_USE,
            'Return only JSON `{"calls": [...]}`.',
            "Call `open_archive(name: '/var/log/app.tar')` failed with `EACCES`. Recovery rule: call `mkdir_safe(dir: '/tmp/stage')`, then call `extract_copy(src: '/var/log/app.tar', chmod: 600)`.",
            json.dumps({"calls": [
                {"tool": "mkdir_safe", "args": {"dir": "/tmp/stage"}},
                {"tool": "extract_copy", "args": {"src": "/var/log/app.tar", "chmod": 600}},
            ]}),
            {"expected_calls": [
                {"tool": "mkdir_safe", "args": {"dir": "/tmp/stage"}},
                {"tool": "extract_copy", "args": {"src": "/var/log/app.tar", "chmod": 600}},
            ]},
        ),
        CurriculumExample(
            "v2-tol-chain-3",
            BenchmarkDim.TOOL_USE,
            'Return only JSON `{"calls": [...]}`.',
            "Tools: `query_quota(project_id: int)`, `scale_workers(count: int, tier: str)`. Query quota for project 19, then scale 6 workers on tier `'gpu-small'`.",
            json.dumps({"calls": [
                {"tool": "query_quota", "args": {"project_id": 19}},
                {"tool": "scale_workers", "args": {"count": 6, "tier": "gpu-small"}},
            ]}),
            {"expected_calls": [
                {"tool": "query_quota", "args": {"project_id": 19}},
                {"tool": "scale_workers", "args": {"count": 6, "tier": "gpu-small"}},
            ]},
        ),
    ])

    # 3. DAG Planning & Critical-Path Makespan (6 diverse diamond/serial DAGs)
    for i in range(6):
        d1, d2, d3, d4, d5 = (rng.randint(2, 7) for _ in range(5))
        f1 = d1
        f2 = f1 + d2
        f3 = f1 + d3
        f4 = f3 + d4
        ms = max(f2, f4) + d5
        dag = {
            "N1": {"dur": d1, "deps": []},
            "N2": {"dur": d2, "deps": ["N1"]},
            "N3": {"dur": d3, "deps": ["N1"]},
            "N4": {"dur": d4, "deps": ["N3"]},
            "N5": {"dur": d5, "deps": ["N2", "N4"]},
        }
        extra.append(
            CurriculumExample(
                f"v2-pln-{i}",
                BenchmarkDim.PLANNING_DECOMPOSITION,
                'Return only valid JSON `{"order": list[str], "makespan": int}`.',
                f"DAG schedule: `N1` (dur {d1}, deps []), `N2` (dur {d2}, deps [N1]), "
                f"`N3` (dur {d3}, deps [N1]), `N4` (dur {d4}, deps [N3]), `N5` (dur {d5}, deps [N2, N4]). "
                "Output valid topological order and minimum parallel makespan.",
                json.dumps({"order": ["N1", "N2", "N3", "N4", "N5"], "makespan": ms}),
                {"dag": dag},
            )
        )

    # 4. 10-Turn Memory Retention with ADD/REMOVE/RESET/DELETE & Percentage Floor (6 examples)
    for i in range(3):
        c1, c2, c3 = rng.randint(2, 5), rng.randint(3, 6), rng.randint(4, 7)
        extra.append(
            CurriculumExample(
                f"v2-mem-inv-{i}",
                BenchmarkDim.MEMORY_RETENTION,
                "Return only valid JSON representing the FINAL inventory.",
                "\n".join([
                    f"Step 1: ADD cpu={c1}",
                    f"Step 2: ADD ram={c2}",
                    f"Step 3: ADD nic={c3}",
                    "Step 4: REMOVE nic=1",
                    "Step 5: ADD cpu=2",
                    "Step 6: RESET ram=2",
                    "Step 7: DELETE nic",
                    "Step 8: ADD psu=4",
                    "Step 9: REMOVE psu=1",
                    'Step 10: Output JSON `{"cpu": int, "ram": int, "psu": int}`.',
                ]),
                json.dumps({"cpu": c1 + 2, "ram": 2, "psu": 3}),
                {"expected_json": {"cpu": c1 + 2, "ram": 2, "psu": 3}},
            )
        )
        p, q = rng.randint(80, 140), rng.randint(40, 90)
        cur_p = p - 20 + 15
        cur_q = q + 20 - 10
        cur_p += cur_p // 10
        cur_p -= 15
        cur_q += 15
        extra.append(
            CurriculumExample(
                f"v2-mem-acct-{i}",
                BenchmarkDim.MEMORY_RETENTION,
                "Return only valid JSON.",
                "\n".join([
                    f"T1: INIT P={p}, Q={q}",
                    "T2: TRANSFER P->Q 20",
                    "T3: DEPOSIT P 15",
                    "T4: WITHDRAW Q 10",
                    "T5: BONUS P +10% (integer floor)",
                    "T6: TRANSFER P->Q 15",
                    'T7: Output JSON `{"P": int, "Q": int}`.',
                ]),
                json.dumps({"P": cur_p, "Q": cur_q}),
                {"expected_json": {"P": cur_p, "Q": cur_q}},
            )
        )

    # 5. Long-Context Multi-Needle Arithmetic & Errata (2 examples)
    h1 = _long_train_haystack([
        (6, "Depot R1 logged 90 MWh on Tuesday under batch BAT-44."),
        (33, "Depot R1 logged 55 MWh on Thursday under batch BAT-44."),
        (59, "Correction credit: subtract 25 MWh from batch BAT-44."),
    ])
    h2 = _long_train_haystack([
        (8, "Vessel Argo loaded 24 containers weighing 480 tons."),
        (41, "Port call unloaded 6 containers weighing 130 tons."),
        (63, "Manifest code finalized as 'PELAGIC-508'."),
    ])
    extra.extend([
        CurriculumExample(
            "v2-lng-1",
            BenchmarkDim.LONG_CONTEXT,
            "Return only valid JSON matching exact keys.",
            h1 + '\n\nReturn `{"batch": "BAT-44", "net_mwh": int}`.',
            json.dumps({"batch": "BAT-44", "net_mwh": 120}),
            {"expected_json": {"batch": "BAT-44", "net_mwh": 120}},
        ),
        CurriculumExample(
            "v2-lng-2",
            BenchmarkDim.LONG_CONTEXT,
            "Return only valid JSON matching exact keys.",
            h2 + '\n\nReturn `{"containers_left": int, "tons_left": int, "code": str}`.',
            json.dumps({"containers_left": 18, "tons_left": 350, "code": "PELAGIC-508"}),
            {"expected_json": {"containers_left": 18, "tons_left": 350, "code": "PELAGIC-508"}},
        ),
    ])

    # 6. Modular CRT, Recurrences, Quoted Mermaid & Structured JSON (7 examples)
    for i in range(4):
        m1, m2 = (5, 9) if i % 2 == 0 else (7, 13)
        if math.gcd(m1, m2) != 1:
            m2 = 11
        x_true = rng.randint(15, m1 * m2 - 2)
        r1, r2 = x_true % m1, x_true % m2
        best_x = next(v for v in range(1, m1 * m2 + 1) if v % m1 == r1 and v % m2 == r2)
        extra.append(
            CurriculumExample(
                f"v2-crt-{i}",
                BenchmarkDim.MATH_LOGIC,
                "Solve step by step and end with `FINAL_ANSWER: <integer>`.",
                f"Find the smallest positive integer `n` satisfying `n ≡ {r1} (mod {m1})` and `n ≡ {r2} (mod {m2})`.",
                f"Checking residues modulo {m1} and {m2} yields {best_x}.\nFINAL_ANSWER: {best_x}",
                {"accepted_answers": [str(best_x)]},
            )
        )

    extra.extend([
        CurriculumExample(
            "v2-rec-1",
            BenchmarkDim.MATH_LOGIC,
            "Solve step by step and end with `FINAL_ANSWER: <integer>`.",
            "Sequence `u_0 = 2, u_1 = 3, u_n = 2*u_{n-1} + u_{n-2}`. Compute `u_5`.",
            "u_0=2, u_1=3, u_2=8, u_3=19, u_4=46, u_5=111.\nFINAL_ANSWER: 111",
            {"accepted_answers": ["111"]},
        ),
        CurriculumExample(
            "v2-dia-lr",
            BenchmarkDim.DIAGRAMS_CHARTS,
            "Return only Mermaid inside ```mermaid``` fence. Quote labels containing parentheses.",
            'Create a `flowchart LR` linking `Sensor["Ingress (TCP)"]` --> `Filter` --> `Sink["Store (NVMe)"]` --> `Notify`.',
            '```mermaid\nflowchart LR\n    S["Ingress (TCP)"] --> F["Filter"]\n    F --> K["Store (NVMe)"]\n    K --> N["Notify"]\n```',
            {"subkind": "mermaid", "expected_header": "flowchart LR", "required_tokens": ["Ingress", "Filter", "Store", "Notify"], "min_edges": 3},
        ),
        CurriculumExample(
            "v2-jsn-rec",
            BenchmarkDim.STRUCTURED_JSON,
            "Return only valid JSON matching exact schema keys.",
            'Extract `{"units": int, "amount_usd": int, "ccy": "USD"}` from: "Ordered 4 drives at 60 USD each and 3 trays at 10 USD each." (units = 4+3).',
            json.dumps({"units": 7, "amount_usd": 270, "ccy": "USD"}),
            {"expected_json": {"units": 7, "amount_usd": 270, "ccy": "USD"}},
        ),
    ])

    return tuple(base + extra)
