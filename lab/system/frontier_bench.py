"""`LAB-FRONTIER-HARD` (100-Item Adversarial Suite) + Separated Competition 1 & Competition 2 Harness.

Strictly separates:
  - **Competition 1 (`naked`)**: Single forward generation from the foundation model (`think=0` or `think=1`)
    with ZERO Python execution, ZERO persistent memory index, ZERO schema critics, and single-shot tool JSON.
  - **Competition 2 (`system`)**: The 5-engine `NanoLMSystem` (Adaptive Controller + Reasoning Engine +
    Persistent/Working Memory + Verification Engine + Tool Execution Engine) applied under IDENTICAL
    system conditions to both `<4B` NanoLM and larger (`8B`, `32B`, `120B`) models!

5 Tracks (20 adversarial tasks each = 100 tasks, zero overlap with any training generator):
  1. `frontier_math` (20): Exact CRT systems, Diophantine counts, derangement/inclusion-exclusion,
     Markov absorption rationals, modular towers, and recurrence determinants (exact ground truth).
  2. `frontier_coding` (20): Multi-edge-case algorithmic & stateful data structures verified against
     adversarial held-out unit tests (never shown to the model during Competition 2 repair!).
  3. `stateful_tool_agents` (20): 3-to-4-turn chained API workflows where Turn 2 depends on opaque IDs
     returned in Turn 1, Turn 2 triggers live API errors (`StaleToken`, `QuotaCap`, `DeprecatedRegion`,
     `RevokedScope`), and scoring checks actual final environment state + grounded final report!
  4. `memory_spec_qa` (20): Multi-hop & version-superseded engineering specifications over a 50-doc
     persistent technical corpus (including deprecated v1 vs active v2 conflicts + unanswerable decoys).
  5. `adversarial_conservation_json` (20): Nested financial/telemetry tables with 5 simultaneous
     row/column conservation equations, weighted averages, enum normalization, and strict key sets.
"""

from __future__ import annotations

import json
import math
import random
from fractions import Fraction
from typing import Any, Callable

from lab.system.controller import GenerateFn, NanoLMSystem
from lab.system.memory_engine import KnowledgeDoc, PersistentKnowledgeStore
from lab.system.tool_engine import StatefulToolEnvironment
from lab.system.verification_engine import extract_canonical_answer
from lab.trackb.eval_capabilities import TOOL_SYSTEM_INSTRUCTION, parse_tool_calls, strip_thinking
from nanolm.eval.validators import verify_math_or_logic, verify_python_solution


# ---------------------------------------------------------------------------
# 1. Persistent Technical Corpus & Memory QA Track (20 tasks + 50 docs)
# ---------------------------------------------------------------------------

def build_technical_memory_corpus() -> tuple[PersistentKnowledgeStore, list[dict[str, Any]]]:
    docs: list[KnowledgeDoc] = []
    tasks: list[dict[str, Any]] = []

    subsystems = [
        ("helios-kv", "HeliosKV Storage Engine", 4096, 16384, 37, "blake3-256"),
        ("aurora-rpc", "AuroraRPC Transport", 128, 512, 19, "chacha20-poly1305"),
        ("kronos-sched", "Kronos Gang Scheduler", 64, 256, 42, "ed25519-hmac"),
        ("borealis-fs", "Borealis Distributed FS", 8192, 65536, 23, "xxh3-128"),
        ("zephyr-mq", "Zephyr Message Broker", 256, 2048, 31, "aes-256-gcm"),
        ("titan-wal", "Titan Write-Ahead Log", 512, 4096, 14, "crc64-nvme"),
        ("vulcan-rdma", "Vulcan RDMA Fabric", 32, 96, 11, "siphash-2-4"),
        ("orion-cdc", "Orion Change Data Capture", 1024, 8192, 27, "sha3-256"),
    ]
    # 8 subsystems * 2 revisions (superseded v1 + active v2) = 16 docs, plus 34 distractor specs = 50 docs
    for slug, title, old_q, new_q, heartbeat_ms, digest_alg in subsystems:
        docs.append(KnowledgeDoc(
            doc_id=f"SPEC-{slug.upper()}-V1",
            title=f"{title} Specification v1",
            version=1,
            status="superseded",
            content=(
                f"In v1 (superseded), `{slug}` enforced `max_inflight_ops = {old_q}`, "
                f"`heartbeat_interval_ms = {heartbeat_ms * 2}`, and legacy checksum `md5`."
            ),
        ))
        docs.append(KnowledgeDoc(
            doc_id=f"SPEC-{slug.upper()}-V2",
            title=f"{title} Specification v2",
            version=2,
            status="active",
            content=(
                f"Active v2 specification for `{slug}` (supersedes v1): `max_inflight_ops = {new_q}`, "
                f"`heartbeat_interval_ms = {heartbeat_ms}`, and mandatory integrity cipher `{digest_alg}`."
            ),
        ))
        # Conflict-resolution task (8 items): must report ACTIVE v2 value, NOT v1!
        tasks.append({
            "id": f"mem_conflict_{slug}",
            "track": "memory_spec_qa",
            "prompt": (
                f"According to the current ACTIVE `{title}` (`{slug}`) specification, what is "
                f"`max_inflight_ops` multiplied by `heartbeat_interval_ms`? End with `\\boxed{{<integer>}}`."
            ),
            "expected": str(new_q * heartbeat_ms),
        })

    # 6 multi-hop cross-document tasks combining two active specs
    for i in range(6):
        s1 = subsystems[i]
        s2 = subsystems[(i + 2) % len(subsystems)]
        tasks.append({
            "id": f"mem_multihop_{i}",
            "track": "memory_spec_qa",
            "prompt": (
                f"Compute the sum of active v2 `max_inflight_ops` for `{s1[0]}` and active v2 "
                f"`max_inflight_ops` for `{s2[0]}`. Output the exact integer in `\\boxed{{}}`."
            ),
            "expected": str(s1[3] + s2[3]),
        })

    # Add 34 realistic distractor protocol documents
    for d_idx in range(34):
        p_name = f"proto-mesh-{100 + d_idx}"
        docs.append(KnowledgeDoc(
            doc_id=f"NOTE-MESH-{100 + d_idx}",
            title=f"Mesh Node `{p_name}` Telemetry Profile",
            version=1,
            status="active",
            content=f"Node `{p_name}` uses MTU {1400 + (d_idx % 5) * 20} bytes and retry backoff {50 + d_idx * 3} ms.",
        ))

    # 6 unanswerable / missing-attribute tasks (must output INSUFFICIENT_EVIDENCE!)
    missing_qs = [
        ("helios-kv", "TLS client certificate rotation period in days"),
        ("aurora-rpc", "maximum BGP AS-path prepending count"),
        ("kronos-sched", "GPU thermal throttling junction temperature in Celsius"),
        ("borealis-fs", "erasure-coding Reed-Solomon parity shard count"),
        ("zephyr-mq", "dead-letter queue S3 lifecycle archival delay in hours"),
        ("nonexistent-hyperion-db", "default write quorum size"),
    ]
    for idx, (slug, attr) in enumerate(missing_qs):
        tasks.append({
            "id": f"mem_abstain_{idx}",
            "track": "memory_spec_qa",
            "prompt": (
                f"Using only verified `{slug}` specifications, what is its `{attr}`? "
                "If the specification does not define this parameter, output `FINAL_ANSWER: INSUFFICIENT_EVIDENCE`."
            ),
            "expected": "INSUFFICIENT_EVIDENCE",
        })

    return PersistentKnowledgeStore(docs), tasks


# ---------------------------------------------------------------------------
# 2. Hard Competition Math & Combinatorics Track (20 tasks, exact ground truth)
# ---------------------------------------------------------------------------

def build_frontier_math_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []

    # 2a. Pairwise coprime 3-congruence Chinese Remainder Theorem + quadratic transform (5 items)
    crt_specs = [
        ((17, 5), (19, 11), (23, 7)),
        ((13, 8), (17, 12), (29, 4)),
        ((11, 9), (23, 15), (31, 6)),
        ((19, 14), (29, 10), (37, 21)),
        ((17, 9), (25, 18), (27, 13)),
    ]
    for i, ((m1, r1), (m2, r2), (m3, r3)) in enumerate(crt_specs):
        M = m1 * m2 * m3
        x_sol = next(x for x in range(1, M + 1) if x % m1 == r1 and x % m2 == r2 and x % m3 == r3)
        ans = x_sol + M  # second smallest positive integer!
        tasks.append({
            "id": f"fmath_crt_{i}",
            "track": "frontier_math",
            "prompt": (
                f"Find the **second smallest** positive integer $n$ satisfying all three congruences:\n"
                f"$n \\equiv {r1} \\pmod{{{m1}}}$, $n \\equiv {r2} \\pmod{{{m2}}}$, and $n \\equiv {r3} \\pmod{{{m3}}}$.\n"
                "Put your exact integer answer in `\\boxed{}`."
            ),
            "expected": str(ans),
        })

    # 2b. Constrained Derangements & Inclusion-Exclusion (5 items)
    for i, (n_tot, k_fix) in enumerate([(8, 2), (9, 3), (9, 2), (10, 3), (10, 4)]):
        # Number of permutations of {1..n_tot} with EXACTLY k_fix fixed points = C(n, k) * D(n - k)
        m = n_tot - k_fix
        der_m = sum(((-1) ** j) * (math.factorial(m) // math.factorial(j)) for j in range(m + 1))
        exact = math.comb(n_tot, k_fix) * der_m
        tasks.append({
            "id": f"fmath_derange_{i}",
            "track": "frontier_math",
            "prompt": (
                f"How many permutations $\\pi$ of $\\{{1, 2, \\dots, {n_tot}\\}}$ have **exact** fixed-point count "
                f"$|\\{{i : \\pi(i) = i\\}}| = {k_fix}$? Put your exact integer answer in `\\boxed{{}}`."
            ),
            "expected": str(exact),
        })

    # 2c. Asymmetric Gambler's Ruin / Biased Random Walk Absorption Numerator+Denominator (5 items)
    walk_specs = [(2, 5, 3, 2), (3, 7, 3, 2), (2, 6, 4, 3), (3, 6, 5, 3), (4, 7, 4, 3)]
    for i, (start, top, p_num, q_num) in enumerate(walk_specs):
        # Move +1 with prob p/(p+q), -1 with prob q/(p+q); absorb at 0 or `top`.
        # Prob(hit `top` before 0 | X_0 = start) = (1 - (q/p)^start) / (1 - (q/p)^top)
        r = Fraction(q_num, p_num)
        prob = (1 - r ** start) / (1 - r ** top)
        ans = prob.numerator + prob.denominator
        tasks.append({
            "id": f"fmath_markov_{i}",
            "track": "frontier_math",
            "prompt": (
                f"A particle starts at integer position $X_0 = {start}$ on $\\{{0, 1, \\dots, {top}\\}}$ with absorbing barriers "
                f"at $0$ and ${top}$. Each step it moves $+1$ with probability $\\frac{{{p_num}}}{{{p_num + q_num}}}$ and "
                f"$-1$ with probability $\\frac{{{q_num}}}{{{p_num + q_num}}}$. If the probability of reaching ${top}$ before $0$ "
                "is written in lowest terms as $\\frac{a}{b}$ with $\\gcd(a,b)=1$, compute $a + b$ and put it in `\\boxed{}`."
            ),
            "expected": str(ans),
        })

    # 2d. Frobenius Coin / Non-negative Linear Diophantine Solution Counts & Divisors (5 items)
    dio_specs = [(7, 11, 13, 180), (5, 9, 14, 150), (6, 11, 17, 195), (8, 13, 19, 210), (9, 14, 23, 240)]
    for i, (a, b, c, target) in enumerate(dio_specs):
        cnt = sum(
            1
            for x in range(target // a + 1)
            for y in range((target - a * x) // b + 1)
            if (target - a * x - b * y) % c == 0
        )
        tasks.append({
            "id": f"fmath_dioph_{i}",
            "track": "frontier_math",
            "prompt": (
                f"How many ordered triples of **non-negative** integers $(x, y, z)$ satisfy "
                f"${a}x + {b}y + {c}z = {target}$? Put the exact count in `\\boxed{{}}`."
            ),
            "expected": str(cnt),
        })

    return tasks


# ---------------------------------------------------------------------------
# 3. Multi-Function & Edge-Case Software Engineering Track (20 tasks)
# ---------------------------------------------------------------------------

def build_frontier_coding_tasks() -> list[dict[str, Any]]:
    base_specs = [
        (
            "fcod_calc_expr",
            "Implement `eval_arith_expr(expr: str) -> int` evaluating an integer arithmetic string supporting `+`, `-` (both binary and unary negation like `-3 * (-2 + 5)`), `*`, integer truncating division toward zero `//_zero` written as `/`, parentheses `( )`, and arbitrary spaces.",
            "assert eval_arith_expr('-3 * (-2 + 5)') == -9\nassert eval_arith_expr('14 - 3 / 2') == 13\nassert eval_arith_expr('-7 / 2 + 10') == 7\nassert eval_arith_expr('((2 + 3) * -(4 - 6)) / 3') == 3",
        ),
        (
            "fcod_lru_ttl",
            "Implement `class TTLLRUCache` with `__init__(self, capacity: int)`, `get(self, key: str, now: int) -> int` (returns `-1` if missing or expired `now >= expiry`, else updates recency and returns value), and `put(self, key: str, value: int, ttl: int, now: int) -> None` (stores `expiry = now + ttl`; before capacity eviction, purge any expired keys at `now`; if still full, evict least-recently-used valid key).",
            "c = TTLLRUCache(2)\nc.put('a', 10, 5, now=0)\nc.put('b', 20, 10, now=1)\nassert c.get('a', now=2) == 10\nc.put('c', 30, 10, now=6)\nassert c.get('a', now=6) == -1\nassert c.get('b', now=6) == 20\nassert c.get('c', now=6) == 30",
        ),
        (
            "fcod_topo_lex",
            "Implement `lex_toposort(n: int, edges: list[tuple[int, int]]) -> list[int] | None` returning the lexicographically smallest topological ordering of nodes `0..n-1` for directed edges `u -> v`, or `None` if a directed cycle exists.",
            "assert lex_toposort(4, [(0, 2), (1, 2), (2, 3)]) == [0, 1, 2, 3]\nassert lex_toposort(3, [(2, 0), (1, 0)]) == [1, 2, 0]\nassert lex_toposort(3, [(0, 1), (1, 2), (2, 0)]) is None",
        ),
        (
            "fcod_interval_cov",
            "Implement `covered_integer_points(intervals: list[tuple[int, int]], k: int) -> int` returning how many integer points `x` are covered by **at least `k`** closed intervals `[l, r]` (`l <= r`, coordinates up to `10^9`, so sweep-line coordinate compression is required; O(max_coord) will time out!).",
            "assert covered_integer_points([(1, 10**9), (500, 10**9 + 500)], 2) == 10**9 - 499\nassert covered_integer_points([(1, 3), (2, 5), (3, 4)], 3) == 1\nassert covered_integer_points([], 1) == 0",
        ),
        (
            "fcod_min_window_subseq",
            "Implement `min_window_subsequence(s: str, t: str) -> str` returning the shortest contiguous substring of `s` that contains `t` as an in-order subsequence (if ties in length, return the leftmost one; if none exists, return `''`).",
            "assert min_window_subsequence('abcdebdde', 'bde') == 'bcde'\nassert min_window_subsequence('fgrqsqsnodwmxzkzxwqegkndaa', 'kzed') == 'kzxwqegknd'\nassert min_window_subsequence('abc', 'd') == ''",
        ),
        (
            "fcod_median_stream",
            "Implement `sliding_medians(nums: list[int], k: int) -> list[float]` returning the exact median of every contiguous window of length `k` (`1 <= k <= len(nums)`).",
            "assert sliding_medians([1, 3, -1, -3, 5, 3, 6, 7], 3) == [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]\nassert sliding_medians([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]",
        ),
        (
            "fcod_semver_resolve",
            "Implement `best_matching_version(versions: list[str], major: int, include_prerelease: bool = False) -> str | None` for strings `'X.Y.Z'` or `'X.Y.Z-rc.N'` (`N` int). Release `'X.Y.Z'` ranks higher than any `'X.Y.Z-rc.N'` of the same `X.Y.Z`; compare numeric components `(X, Y, Z, is_release, rc_num)`. Filter to `X == major` (and exclude `-rc.` when `include_prerelease=False`), returning the highest version string or `None`.",
            "vs = ['2.1.0', '2.10.0-rc.2', '2.10.0-rc.10', '2.9.5', '3.0.0']\nassert best_matching_version(vs, 2, False) == '2.9.5'\nassert best_matching_version(vs, 2, True) == '2.10.0-rc.10'\nassert best_matching_version(['2.10.0', '2.10.0-rc.9'], 2, True) == '2.10.0'\nassert best_matching_version(vs, 1, True) is None",
        ),
        (
            "fcod_bipartite_min_flip",
            "Implement `min_vertex_weight_independent_conflict(n: int, weights: list[int], tree_edges: list[tuple[int, int]]) -> int` returning the **maximum** total weight of an independent set of vertices in an undirected tree on `0..n-1` (`n >= 1`).",
            "assert min_vertex_weight_independent_conflict(4, [10, 5, 8, 20], [(0, 1), (1, 2), (2, 3)]) == 30\nassert min_vertex_weight_independent_conflict(1, [42], []) == 42",
        ),
        (
            "fcod_cron_next",
            "Implement `next_matching_minute(start_hhmm: str, allowed_hours: set[int], allowed_mins: set[int]) -> tuple[str, int]` taking `'HH:MM'` (`00:00`..`23:59`) and non-empty valid sets (`0..23`, `0..59`), searching strictly *after* `start_hhmm` (wrapping past midnight) and returning `('HH:MM', delta_minutes)`.",
            "assert next_matching_minute('23:50', {0, 12}, {0, 30}) == ('00:00', 10)\nassert next_matching_minute('12:00', {12}, {0, 15}) == ('12:15', 15)\nassert next_matching_minute('12:30', {12}, {15}) == ('12:15', 1425)",
        ),
        (
            "fcod_json_diff",
            "Implement `flat_dict_diff(old: dict, new: dict) -> dict[str, tuple]` flattening nested `dict` keys by `'.'` (non-dict values including lists are leaves) and returning `{path: (old_val_or_MISSING, new_val_or_MISSING)}` using string `'__MISSING__'` where a leaf was added or removed, ONLY for paths whose value changed.",
            "d = flat_dict_diff({'a': {'x': 1, 'y': 2}, 'b': [1, 2]}, {'a': {'x': 1, 'z': 3}, 'b': [1, 3]})\nassert d == {'a.y': (2, '__MISSING__'), 'a.z': ('__MISSING__', 3), 'b': ([1, 2], [1, 3])}",
        ),
    ]
    tasks: list[dict[str, Any]] = []
    # 10 distinct algorithm families + 10 parameterized stateful/parser variants = 20 unique tasks
    for tid, desc, har in base_specs:
        tasks.append({
            "id": tid,
            "track": "frontier_coding",
            "prompt": f"{desc}\nProvide your complete implementation inside a ```python``` code block.",
            "test_harness": har,
        })
    for j in range(10):
        mod_p = 1_000_000_007 if j % 2 == 0 else 998_244_353
        step_set = [1, 2, 4] if j < 5 else [1, 3, 5]
        fname = f"count_paths_avoiding_{j}"
        # DP verification calculated directly in harness
        s_tuple = tuple(step_set)
        har = (
            f"def _ref(n, blocked):\n"
            f"    b = set(blocked); dp = [0]*(n+1); dp[0] = 0 if 0 in b else 1\n"
            f"    for i in range(1, n+1):\n"
            f"        if i in b: continue\n"
            f"        dp[i] = sum(dp[i-s] for s in {s_tuple} if i-s >= 0) % {mod_p}\n"
            f"    return dp[n]\n"
            f"assert {fname}(15, [3, 7]) == _ref(15, [3, 7])\n"
            f"assert {fname}(0, [0]) == 0\n"
            f"assert {fname}(40, [{2 + j}, {11 + j}, {23 + j}]) == _ref(40, [{2 + j}, {11 + j}, {23 + j}])"
        )
        tasks.append({
            "id": f"fcod_dp_blocked_{j}",
            "track": "frontier_coding",
            "prompt": (
                f"Implement `def {fname}(n: int, blocked: list[int]) -> int:` returning the number of ways to reach "
                f"position `n >= 0` starting at `0` using step sizes `{list(step_set)}`, without ever landing on any "
                f"index in `blocked` (note: if `0` or `n` is in `blocked`, the count is `0`), modulo `{mod_p}`.\n"
                "Provide your complete implementation inside a ```python``` code block."
            ),
            "test_harness": har,
        })
    return tasks


# ---------------------------------------------------------------------------
# 4. Stateful Multi-Turn Agentic Tool Workflows Track (20 scenarios)
# ---------------------------------------------------------------------------

WORKFLOW_SCHEMAS = [
    {
        "name": "lookup_incident_context",
        "description": "Fetch live service shard ID, current rate_limit_rps, and lease_token for an incident ticket.",
        "parameters": {
            "type": "object",
            "required": ["incident_id"],
            "properties": {"incident_id": {"type": "string"}},
        },
    },
    {
        "name": "refresh_lease_token",
        "description": "Obtain a fresh lease_token for a shard when an existing lease_token is expired/stale.",
        "parameters": {
            "type": "object",
            "required": ["shard_id"],
            "properties": {"shard_id": {"type": "string"}},
        },
    },
    {
        "name": "apply_shard_mitigation",
        "description": "Update rate_limit_rps on a shard using an active lease_token.",
        "parameters": {
            "type": "object",
            "required": ["shard_id", "lease_token", "target_rps"],
            "properties": {
                "shard_id": {"type": "string"},
                "lease_token": {"type": "string"},
                "target_rps": {"type": "integer"},
            },
        },
    },
]


def _make_incident_env(inc_id: str, shard_id: str, cur_rps: int, min_floor_rps: int, fresh_tok: str, locked: bool = False):
    init_state = {
        "incident_id": inc_id,
        "shard_id": shard_id,
        "stale_token": f"LEASE-STALE-{inc_id[-3:]}",
        "fresh_token": fresh_tok,
        "current_rps": cur_rps,
        "min_floor_rps": min_floor_rps,
        "locked": locked,
        "mitigated": False,
    }

    def _lookup(st: dict, args: dict) -> dict:
        if args.get("incident_id") != st["incident_id"]:
            return {"ok": False, "error": "UnknownIncident", "message": f"Incident {args.get('incident_id')} not found."}
        return {
            "ok": True,
            "shard_id": st["shard_id"],
            "current_rps": st["current_rps"],
            "lease_token": st["stale_token"],
        }

    def _refresh(st: dict, args: dict) -> dict:
        if args.get("shard_id") != st["shard_id"]:
            return {"ok": False, "error": "UnknownShard", "message": "Invalid shard_id."}
        if st["locked"]:
            return {"ok": False, "error": "ComplianceFreeze", "message": "Shard is under compliance freeze; do NOT mitigate or retry."}
        return {"ok": True, "shard_id": st["shard_id"], "fresh_lease_token": st["fresh_token"]}

    def _mitigate(st: dict, args: dict) -> dict:
        if st["locked"]:
            return {"ok": False, "error": "ComplianceFreeze", "message": "Shard is under compliance freeze; do NOT mitigate."}
        if args.get("shard_id") != st["shard_id"]:
            return {"ok": False, "error": "UnknownShard", "message": "Invalid shard_id."}
        if args.get("lease_token") == st["stale_token"]:
            return {
                "ok": False,
                "error": "StaleLeaseToken",
                "message": "Provided lease_token is expired; call `refresh_lease_token` first to get `fresh_lease_token`.",
            }
        if args.get("lease_token") != st["fresh_token"]:
            return {"ok": False, "error": "InvalidLeaseToken", "message": "Unrecognized lease_token."}
        rps = int(args.get("target_rps", 0))
        if rps < st["min_floor_rps"]:
            return {
                "ok": False,
                "error": "BelowMinimumFloor",
                "message": f"target_rps={rps} violates safety floor; clamp `target_rps` to at least {st['min_floor_rps']}.",
            }
        st["current_rps"] = rps
        st["mitigated"] = True
        return {"ok": True, "applied_rps": rps, "status": "MITIGATED"}

    return lambda: StatefulToolEnvironment(
        WORKFLOW_SCHEMAS,
        init_state,
        {"lookup_incident_context": _lookup, "refresh_lease_token": _refresh, "apply_shard_mitigation": _mitigate},
    )


def build_stateful_tool_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    # 14 multi-failure recovery workflows (StaleLeaseToken + BelowMinimumFloor clamping!)
    for i in range(14):
        inc = f"INC-{7100 + i * 17}"
        shd = f"shard-eu-{20 + i}"
        cur = 2000 + i * 200
        req_rps = cur // 4           # e.g. 500
        floor_rps = req_rps + 150    # triggers BelowMinimumFloor on first try -> must clamp to floor_rps!
        ftok = f"LEASE-LIVE-{900 + i}"
        tasks.append({
            "id": f"tagent_mitigate_{i}",
            "track": "stateful_tool_agents",
            "prompt": (
                f"Resolve incident `{inc}`: look up its shard context, mitigate the shard to `25%` of its `current_rps` "
                "(handling any stale lease tokens or minimum RPS safety floors returned by the APIs), and report the "
                "final applied RPS and shard_id once mitigated."
            ),
            "make_env": _make_incident_env(inc, shd, cur, floor_rps, ftok, locked=False),
            "expected_rps": floor_rps,
            "expected_shard": shd,
            "expect_locked": False,
            # For Competition 1 (single-turn naked check): Turn-1 must invoke `lookup_incident_context({"incident_id": inc})`!
            "naked_turn1_tool": ("lookup_incident_context", {"incident_id": inc}),
        })
    # 6 compliance-freeze abort workflows (must stop without mitigating and report freeze!)
    for i in range(6):
        inc = f"INC-{8400 + i * 19}"
        shd = f"shard-pci-{10 + i}"
        tasks.append({
            "id": f"tagent_freeze_{i}",
            "track": "stateful_tool_agents",
            "prompt": (
                f"Resolve incident `{inc}` by halving its shard's `current_rps`; if any API reports a `ComplianceFreeze`, "
                "halt immediately without further mitigation retries and explain the freeze."
            ),
            "make_env": _make_incident_env(inc, shd, 3000, 500, f"LEASE-X-{i}", locked=True),
            "expected_rps": 3000,
            "expected_shard": shd,
            "expect_locked": True,
            "naked_turn1_tool": ("lookup_incident_context", {"incident_id": inc}),
        })
    return tasks


# ---------------------------------------------------------------------------
# 5. Adversarial Multi-Constraint Conservation JSON Track (20 tasks)
# ---------------------------------------------------------------------------

def build_adversarial_json_tasks(seed: int = 31415) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []
    for i in range(20):
        q_code = f"FY26-Q{(i % 4) + 1}-R{i + 1}"
        r_na = rng.randint(1200, 4800)
        r_eu = rng.randint(900, 3900)
        r_apac = rng.randint(700, 3400)
        c_na = rng.randint(400, r_na - 150)
        c_eu = rng.randint(350, r_eu - 120)
        c_apac = rng.randint(300, r_apac - 100)
        tot_rev = r_na + r_eu + r_apac
        tot_cost = c_na + c_eu + c_apac
        net_inc = tot_rev - tot_cost
        margin_pct = round(100.0 * net_inc / tot_rev, 2)
        tier = "PLATINUM" if margin_pct >= 55.0 else ("GOLD" if margin_pct >= 40.0 else "SILVER")

        expected_obj = {
            "report_id": q_code,
            "total_revenue": tot_rev,
            "total_cost": tot_cost,
            "net_income": net_inc,
            "operating_margin_pct": margin_pct,
            "margin_tier": tier,
            "region_nets": {"NA": r_na - c_na, "EU": r_eu - c_eu, "APAC": r_apac - c_apac},
        }

        def _make_inv(exp: dict[str, Any]) -> Callable[[dict[str, Any]], tuple[bool, str]]:
            def _inv(obj: dict[str, Any]) -> tuple[bool, str]:
                if set(obj.keys()) != set(exp.keys()):
                    return False, f"exact keys must be {sorted(exp.keys())}, got {sorted(obj.keys())}"
                for k in ("report_id", "total_revenue", "total_cost", "net_income", "margin_tier"):
                    if obj.get(k) != exp[k]:
                        return False, f"mismatch on `{k}` (check sum/tier conservation)"
                if abs(float(obj.get("operating_margin_pct", -999)) - exp["operating_margin_pct"]) > 0.02:
                    return False, "`operating_margin_pct` must equal round(100 * net_income / total_revenue, 2)"
                rn = obj.get("region_nets")
                if not isinstance(rn, dict) or rn != exp["region_nets"]:
                    return False, "`region_nets` must map NA, EU, APAC to exact `revenue - cost`"
                return True, "ok"
            return _inv

        prompt = (
            "Output ONLY a valid JSON object with EXACT keys:\n"
            "`report_id` (str), `total_revenue` (int), `total_cost` (int), `net_income` (int = total_revenue - total_cost), "
            "`operating_margin_pct` (float rounded to 2 decimals = `100 * net_income / total_revenue`), "
            "`margin_tier` (`'PLATINUM'` if margin >= 55.0, `'GOLD'` if 40.0 <= margin < 55.0, else `'SILVER'`), and "
            "`region_nets` (object with exact keys `'NA'`, `'EU'`, `'APAC'` giving each region's `revenue - cost`).\n\n"
            f"Raw Telemetry (`{q_code}`):\n"
            f"- North America (`NA`): revenue = {r_na}, cost = {c_na}\n"
            f"- Europe (`EU`): revenue = {r_eu}, cost = {c_eu}\n"
            f"- Asia-Pacific (`APAC`): revenue = {r_apac}, cost = {c_apac}"
        )
        tasks.append({
            "id": f"fjson_cons_{i}",
            "track": "adversarial_conservation_json",
            "prompt": prompt,
            "required_keys": set(expected_obj.keys()),
            "invariant_fn": _make_inv(expected_obj),
            "expected_obj": expected_obj,
        })
    return tasks


def build_frontier_100_suite() -> tuple[PersistentKnowledgeStore, list[dict[str, Any]]]:
    mem_store, mem_tasks = build_technical_memory_corpus()
    all_tasks = (
        build_frontier_math_tasks()
        + build_frontier_coding_tasks()
        + build_stateful_tool_tasks()
        + mem_tasks
        + build_adversarial_json_tasks()
    )
    assert len(all_tasks) == 100
    return mem_store, all_tasks


# ---------------------------------------------------------------------------
# Separated Competition 1 (Naked) vs Competition 2 (System) Scorer
# ---------------------------------------------------------------------------

def evaluate_competition_1_naked(
    generate_fn: GenerateFn,
    tasks: list[dict[str, Any]],
    think: bool = False,
) -> dict[str, Any]:
    """Competition 1: Naked Foundation Model (0 tools, 0 Python exec, 0 external memory)."""
    by_track: dict[str, list[int]] = {}
    total_chars_out = 0
    for item in tasks:
        tr = item["track"]
        if tr == "stateful_tool_agents":
            # In naked single-pass mode, evaluate whether the model selects the exact valid initial tool & arguments
            p = f"{TOOL_SYSTEM_INSTRUCTION}\n\nAvailable tools:\n```json\n{json.dumps(WORKFLOW_SCHEMAS, indent=2)}\n```\nUser Objective: {item['prompt']}"
            raw = generate_fn([{"role": "user", "content": p}], think, 1024 if not think else 2560)
            total_chars_out += len(raw)
            calls, _ = parse_tool_calls(raw)
            exp_name, exp_args = item["naked_turn1_tool"]
            ok = bool(calls and len(calls) == 1 and calls[0]["name"] == exp_name and calls[0]["arguments"] == exp_args)
        else:
            raw = generate_fn([{"role": "user", "content": item["prompt"]}], think, 1280 if not think else 3072)
            total_chars_out += len(raw)
            vis = strip_thinking(raw)
            if tr in ("frontier_math", "memory_spec_qa"):
                ans = extract_canonical_answer(raw)
                ok = verify_math_or_logic(f"FINAL_ANSWER: {ans}", item["expected"]) or verify_math_or_logic(vis, item["expected"])
            elif tr == "frontier_coding":
                ok, _ = verify_python_solution(vis, item["test_harness"])
            elif tr == "adversarial_conservation_json":
                from lab.system.verification_engine import verify_json_candidate
                rep = verify_json_candidate(raw, required_keys=item["required_keys"], invariant_fn=item["invariant_fn"])
                ok = rep.passed
            else:
                ok = False
        by_track.setdefault(tr, []).append(1 if ok else 0)

    summary_tracks = {k: round(sum(v) / len(v), 4) for k, v in by_track.items()}
    total_passed = sum(sum(v) for v in by_track.values())
    return {
        "competition": "Competition_1_Naked_Foundation",
        "think": think,
        "total_items": len(tasks),
        "passed": total_passed,
        "overall_score": round(total_passed / len(tasks), 4),
        "tracks": summary_tracks,
        "avg_llm_calls_per_task": 1.0,
        "avg_output_chars_per_task": round(total_chars_out / len(tasks), 1),
    }


def evaluate_competition_2_system(
    generate_fn: GenerateFn,
    mem_store: PersistentKnowledgeStore,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Competition 2: Complete 5-Engine Intelligent System (identical harness for 2B, 8B, 32B, 120B)."""
    sys_agent = NanoLMSystem(generate_fn, knowledge_store=mem_store, max_repairs=2, max_tool_turns=5)
    by_track: dict[str, list[int]] = {}
    llm_calls_tot = 0
    out_chars_tot = 0
    sbx_tot = 0
    mem_tot = 0

    for item in tasks:
        tr = item["track"]
        if tr == "frontier_math":
            ep = sys_agent.solve(item["prompt"], task_hint="math_hard")
            ok = verify_math_or_logic(f"FINAL_ANSWER: {ep.extracted_answer}", item["expected"]) or verify_math_or_logic(ep.final_response, item["expected"])
        elif tr == "frontier_coding":
            # IMPORTANT FAIRNESS: System NEVER sees held-out `item["test_harness"]` during solving!
            # It must synthesize its own self-tests inside `NanoLMSystem.solve`!
            ep = sys_agent.solve(item["prompt"], task_hint="coding", self_test_harness="")
            ok, _ = verify_python_solution(ep.final_response, item["test_harness"])
        elif tr == "stateful_tool_agents":
            env = item["make_env"]()
            ep = sys_agent.solve(item["prompt"], task_hint="tool_workflow", tool_env=env)
            if item["expect_locked"]:
                # Must NOT have mitigated, must have seen ComplianceFreeze, and must mention freeze
                saw_freeze = any(c["observation"].get("error") == "ComplianceFreeze" for c in env.call_history)
                ok = (not env.state["mitigated"]) and saw_freeze and ("freeze" in ep.final_response.lower() or "compliance" in ep.final_response.lower())
            else:
                ok = (
                    env.state["mitigated"]
                    and env.state["current_rps"] == item["expected_rps"]
                    and str(item["expected_rps"]) in ep.final_response
                    and item["expected_shard"] in ep.final_response
                )
        elif tr == "memory_spec_qa":
            ep = sys_agent.solve(item["prompt"], task_hint="memory_qa")
            ok = verify_math_or_logic(f"FINAL_ANSWER: {ep.extracted_answer}", item["expected"]) or verify_math_or_logic(ep.final_response, item["expected"])
        elif tr == "adversarial_conservation_json":
            # System checks generic structural/numeric conservation keys without peeking at ground-truth numbers!
            def _generic_accounting_check(obj: dict[str, Any]) -> tuple[bool, str]:
                req = {"report_id", "total_revenue", "total_cost", "net_income", "operating_margin_pct", "margin_tier", "region_nets"}
                if set(obj.keys()) != req:
                    return False, f"keys must be {sorted(req)}"
                try:
                    tr_v = int(obj["total_revenue"])
                    tc_v = int(obj["total_cost"])
                    ni_v = int(obj["net_income"])
                    mp_v = float(obj["operating_margin_pct"])
                    rn = obj["region_nets"]
                except Exception as e:
                    return False, f"type error: {e}"
                if ni_v != tr_v - tc_v:
                    return False, f"net_income ({ni_v}) != total_revenue - total_cost ({tr_v - tc_v})"
                if not isinstance(rn, dict) or set(rn.keys()) != {"NA", "EU", "APAC"} or sum(rn.values()) != ni_v:
                    return False, "sum(region_nets.values()) must equal net_income with keys NA, EU, APAC"
                exp_mp = round(100.0 * ni_v / max(1, tr_v), 2)
                if abs(mp_v - exp_mp) > 0.02:
                    return False, f"operating_margin_pct ({mp_v}) != round(100*net_income/total_revenue, 2) ({exp_mp})"
                exp_t = "PLATINUM" if exp_mp >= 55.0 else ("GOLD" if exp_mp >= 40.0 else "SILVER")
                if obj["margin_tier"] != exp_t:
                    return False, f"margin_tier must be {exp_t} for margin {exp_mp}"
                return True, "ok"

            ep = sys_agent.solve(
                item["prompt"],
                task_hint="json_strict",
                json_required_keys=item["required_keys"],
                json_invariant_fn=_generic_accounting_check,
            )
            from lab.system.verification_engine import verify_json_candidate
            ok = verify_json_candidate(ep.final_response, required_keys=item["required_keys"], invariant_fn=item["invariant_fn"]).passed
        else:
            continue

        llm_calls_tot += ep.llm_calls
        out_chars_tot += ep.output_chars
        sbx_tot += ep.sandbox_runs
        mem_tot += ep.memory_retrievals
        by_track.setdefault(tr, []).append(1 if ok else 0)

    summary_tracks = {k: round(sum(v) / len(v), 4) for k, v in by_track.items()}
    total_passed = sum(sum(v) for v in by_track.values())
    return {
        "competition": "Competition_2_Complete_System",
        "total_items": len(tasks),
        "passed": total_passed,
        "overall_score": round(total_passed / len(tasks), 4),
        "tracks": summary_tracks,
        "avg_llm_calls_per_task": round(llm_calls_tot / len(tasks), 2),
        "avg_output_chars_per_task": round(out_chars_tot / len(tasks), 1),
        "total_sandbox_runs": sbx_tot,
        "total_memory_retrievals": mem_tot,
    }
