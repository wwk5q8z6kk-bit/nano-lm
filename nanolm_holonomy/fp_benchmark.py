"""Held-Out First-Principles 5-Axis Benchmark (FP-5Axis-30, Zero PHI, Zero Overlap).

Tests five distinct scientific dimensions requested by the First-Principles directive:
1. relational_generalization (6 tasks): entity renaming, permutation, 4x scale, DAG critical path.
2. bitemporal_belief_revision (6 tasks): valid/record/as-of clocks, corrections, retractions, report vs verified.
3. exact_information_transport (6 tasks): high-entropy opaque IDs/dosages, cross-tenant denial, explicit UNKNOWN.
4. causal_vs_associational (6 tasks): Simpson's paradox do(a) intervention, port-Hamiltonian invariants, observed vs simulated.
5. cross_domain_transfer (6 tasks): isomorphic rules over synthetic clinical, software API, distributed lock, and Hamiltonian domains.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any

from nanolm_frontier.benchmark_15d import _extract_final_answer, _verify_exact_answer, _verify_json_invariant
from nanolm_posttrain.eval_suite import TaskVerdict


class FPAxis(str, Enum):
    RELATIONAL = "relational_generalization"
    BITEMPORAL = "bitemporal_belief_revision"
    EXACT_TRANSPORT = "exact_information_transport"
    CAUSAL = "causal_vs_associational"
    CROSS_DOMAIN = "cross_domain_transfer"


@dataclass(frozen=True, slots=True)
class FPBenchmarkTask:
    task_id: str
    axis: FPAxis
    difficulty: str
    system_prompt: str
    user_prompt: str
    spec: dict[str, Any]
    max_new_tokens: int = 256


def build_fp_5axis_benchmark() -> tuple[FPBenchmarkTask, ...]:
    """Return 30 deterministic held-out tasks (6 per axis, strictly zero PHI)."""
    tasks: tuple[FPBenchmarkTask, ...] = (
        # ---------------- Axis 1: Relational Generalization ----------------
        FPBenchmarkTask(
            "fp-rel-01",
            FPAxis.RELATIONAL,
            "medium",
            "Solve step by step and end with `FINAL_ANSWER: <str>`.",
            "In graph Alpha, rename vertices (A->V_91, B->V_44, C->V_17, D->V_03). "
            "Edges: V_91 -> V_44, V_44 -> V_17, V_17 -> V_03, and direct shortcut V_91 -> V_17. "
            "Every edge weight is 1. What is the shortest hop distance from V_91 to V_03?",
            {"accepted_answers": ["2"]},
        ),
        FPBenchmarkTask(
            "fp-rel-02",
            FPAxis.RELATIONAL,
            "hard",
            "Compute the parallel DAG critical-path makespan and end with `FINAL_ANSWER: <int>`.",
            "Parallel DAG tasks with durations and dependencies:\n"
            "- T_k1: duration=4, deps=[]\n"
            "- T_m8: duration=9, deps=[T_k1]\n"
            "- T_p3: duration=6, deps=[T_k1]\n"
            "- T_r5: duration=7, deps=[T_m8, T_p3]\n"
            "- T_z9: duration=3, deps=[T_p3]\n"
            "What is the minimum total parallel makespan?",
            {"accepted_answers": ["20"]},  # 4 + max(9,6) + 7 = 20
        ),
        FPBenchmarkTask(
            "fp-rel-03",
            FPAxis.RELATIONAL,
            "hard",
            "Return only valid JSON.",
            "Track variable bindings under 12 entity swaps:\n"
            "Init: k9=14, m2=27, q5=33, z1=41.\n"
            "Step 1: copy q5 -> x4; set k9 = x4 + 6.\n"
            "Step 2: swap m2 <-> z1.\n"
            "Step 3: set z1 = k9 - 5.\n"
            "Return JSON `{\"k9\": int, \"m2\": int, \"z1\": int}`.",
            {"expected_json": {"k9": 39, "m2": 41, "z1": 34}},
        ),
        FPBenchmarkTask(
            "fp-rel-04",
            FPAxis.RELATIONAL,
            "hard",
            "Compute the parallel DAG critical-path makespan and end with `FINAL_ANSWER: <int>`.",
            "Renamed pipeline stages:\n"
            "- stage_x0: duration=5, deps=[]\n"
            "- stage_x1: duration=8, deps=[]\n"
            "- stage_x2: duration=6, deps=[stage_x0, stage_x1]\n"
            "- stage_x3: duration=11, deps=[stage_x0]\n"
            "- stage_x4: duration=4, deps=[stage_x2, stage_x3]\n"
            "Compute total makespan.",
            {"accepted_answers": ["20"]},  # x1(8)->x2(6)=14, x0(5)->x3(11)=16; max=16 + 4 = 20
        ),
        FPBenchmarkTask(
            "fp-rel-05",
            FPAxis.RELATIONAL,
            "medium",
            "End with `FINAL_ANSWER: <str>`.",
            "Symmetric permutation group S_4 transitions starting from identity tuple [0,1,2,3]. "
            "Apply operator S_swap01 ([1,0,2,3]), then S_swap23 ([0,1,3,2]), then S_rotL ([1,2,3,0]) where composition is next[i] = op[cur[i]]. "
            "After these 3 operations, what is the resulting 4-digit string without spaces (e.g., 1230)?",
            {"accepted_answers": ["2103", "[2,1,0,3]", "[2, 1, 0, 3]"]},  # cur after 01->[1,0,2,3], after 23->[1,0,3,2], rotL[cur]: cur=[1,0,3,2] -> op[c]=[0,1,2,3]? wait! Check S_rotL([1,0,3,2]): op=[1,2,3,0] => [op[1], op[0], op[3], op[2]] = [2, 1, 0, 3]!
        ),
        FPBenchmarkTask(
            "fp-rel-06",
            FPAxis.RELATIONAL,
            "hard",
            "End with `FINAL_ANSWER: <str>`.",
            "12 entities sit around a directed cycle node_0 -> node_1 -> ... -> node_11 -> node_0. "
            "Rule: a token moves +5 steps on token 'A', -2 steps on token 'B'. "
            "Start at node_3. Sequence: A, B, A, A, B. Which node index (0..11) holds the token?",
            {"accepted_answers": ["node_2", "2"]},  # (3 + 5 - 2 + 5 + 5 - 2) % 12 = 14 % 12 = 2
        ),

        # ---------------- Axis 2: Bitemporal Belief Revision ----------------
        FPBenchmarkTask(
            "fp-bit-01",
            FPAxis.BITEMPORAL,
            "medium",
            "Return ONLY JSON `{\"state\": str, \"value\": str}` where state is one of VERIFIED_VALUE, REPORTED_UNVERIFIED, CONFLICTING, UNKNOWN_NOT_FOUND, ISOLATED_DENIED.",
            "Bitemporal synthetic medication log (zero real patient data):\n"
            "Query: entity=syn_p10, tenant=clinic_a, authorized_tenant=clinic_a, slot=med_x, as_of_time=3.0 (Wednesday).\n"
            "Records:\n"
            "- rec_1: entity=syn_p10, tenant=clinic_a, slot=med_x, raw_value=10_mg, valid_time=1.0, record_time=1.0 (Monday), source_kind=unverified_report\n"
            "- rec_2: entity=syn_p10, tenant=clinic_a, slot=med_x, raw_value=20_mg, valid_time=2.5, record_time=3.0 (Wednesday), source_kind=unverified_report\n"
            "- rec_3: entity=syn_p10, tenant=clinic_a, slot=med_x, raw_value=15_mg, valid_time=2.5, record_time=5.0 (Friday), source_kind=correction, target_record_id=rec_2\n"
            "Determine the epistemic state and value strictly as_of_time=3.0 (without leaking record_time > 3.0).",
            {"expected_json": {"state": "CONFLICTING", "value": "20_mg"}},
        ),
        FPBenchmarkTask(
            "fp-bit-02",
            FPAxis.BITEMPORAL,
            "medium",
            "Return ONLY JSON `{\"state\": str, \"value\": str}`.",
            "Bitemporal synthetic medication log:\n"
            "Query: entity=syn_p10, tenant=clinic_a, authorized_tenant=clinic_a, slot=med_x, as_of_time=5.0 (Friday).\n"
            "Records:\n"
            "- rec_1: entity=syn_p10, tenant=clinic_a, slot=med_x, raw_value=20_mg, valid_time=2.5, record_time=3.0, source_kind=unverified_report\n"
            "- rec_2: entity=syn_p10, tenant=clinic_a, slot=med_x, raw_value=15_mg, valid_time=2.5, record_time=5.0, source_kind=correction, target_record_id=rec_1\n"
            "Determine the epistemic state and value as_of_time=5.0.",
            {"expected_json": {"state": "VERIFIED_VALUE", "value": "15_mg"}},
        ),
        FPBenchmarkTask(
            "fp-bit-03",
            FPAxis.BITEMPORAL,
            "medium",
            "Return ONLY JSON `{\"state\": str, \"value\": str}`.",
            "Bitemporal software contract log:\n"
            "Query: entity=svc_billing, tenant=org_west, authorized_tenant=org_west, slot=timeout_ms, as_of_time=4.0.\n"
            "Records:\n"
            "- rec_10: entity=svc_billing, tenant=org_west, slot=timeout_ms, raw_value=4500ms, valid_time=1.0, record_time=1.0, source_kind=unverified_report\n"
            "Only this informal chat report exists; no verified deploy order was signed. Return state and value.",
            {"expected_json": {"state": "REPORTED_UNVERIFIED", "value": "4500ms"}},
        ),
        FPBenchmarkTask(
            "fp-bit-04",
            FPAxis.BITEMPORAL,
            "hard",
            "Return ONLY JSON `{\"state\": str, \"value\": str}`.",
            "Bitemporal retraction test:\n"
            "Query: entity=svc_auth, tenant=org_1, authorized_tenant=org_1, slot=key_mode, as_of_time=6.0.\n"
            "Records:\n"
            "- rec_a: entity=svc_auth, tenant=org_1, slot=key_mode, raw_value=ed25519_v1, valid_time=2.0, record_time=2.0, source_kind=verified_order\n"
            "- rec_b: entity=svc_auth, tenant=org_1, slot=key_mode, raw_value=ed25519_v1, valid_time=4.0, record_time=5.0, source_kind=retraction, target_record_id=rec_a\n"
            "Use value='UNKNOWN' if no active value remains after retraction.",
            {"expected_json": {"state": "UNKNOWN_NOT_FOUND", "value": "UNKNOWN"}},
        ),
        FPBenchmarkTask(
            "fp-bit-05",
            FPAxis.BITEMPORAL,
            "hard",
            "End with `FINAL_ANSWER: <str>`.",
            "Out-of-order record_time vs valid_time ingestion:\n"
            "Event E1 claims valid_time=10:00, record_time=14:00: valve_pressure=42_psi (verified_order).\n"
            "Event E2 claims valid_time=11:00, record_time=12:00: valve_pressure=38_psi (verified_order).\n"
            "If an audit queries STRICTLY at reasoning/as_of_time=13:00 (when record_time=14:00 had not arrived yet), which pressure value was visible?",
            {"accepted_answers": ["38_psi", "38"]},
        ),
        FPBenchmarkTask(
            "fp-bit-06",
            FPAxis.BITEMPORAL,
            "hard",
            "End with `FINAL_ANSWER: <str>`.",
            "Wait-free audit preservation rule: Mon record m1 says threshold=80 (verified). Wed record m2 amends threshold=95 (correction superseding m1). "
            "Should historical log entry m1 be erased/mutated in the immutable ledger, or appended/superseded? Reply YES_IMMUTABLE_APPEND or OVERWRITE_IN_PLACE.",
            {"accepted_answers": ["YES_IMMUTABLE_APPEND"]},
        ),

        # ---------------- Axis 3: Exact Information Transport ----------------
        FPBenchmarkTask(
            "fp-ext-01",
            FPAxis.EXACT_TRANSPORT,
            "medium",
            "End with `FINAL_ANSWER: <exact_token>` preserving exact spelling and hyphens.",
            "Ledger excerpt with distractors:\n"
            "[noise_01] deploy=canary hash=aa11-bb22-99\n"
            "[target_slot] release=main artifact_checksum=kx94-v7m2-08f1 status=verified\n"
            "[noise_02] fallback=legacy hash=cc33-dd44-11\n"
            "Return the EXACT `artifact_checksum` string.",
            {"accepted_answers": ["kx94-v7m2-08f1"]},
        ),
        FPBenchmarkTask(
            "fp-ext-02",
            FPAxis.EXACT_TRANSPORT,
            "medium",
            "Return ONLY JSON `{\"state\": str, \"value\": str}`.",
            "Permission isolation check:\n"
            "Query: entity=ent_77, tenant=tenant_red, authorized_tenant=tenant_blue, slot=api_secret, as_of_time=5.0.\n"
            "Records:\n"
            "- rec_99: entity=ent_77, tenant=tenant_red, slot=api_secret, raw_value=sec_9918_az, valid_time=1.0, record_time=1.0, source_kind=verified_order\n"
            "Use value='DENIED' when caller tenant != authorized_tenant.",
            {"expected_json": {"state": "ISOLATED_DENIED", "value": "DENIED"}},
        ),
        FPBenchmarkTask(
            "fp-ext-03",
            FPAxis.EXACT_TRANSPORT,
            "medium",
            "End with `FINAL_ANSWER: <str>`.",
            "Observed ledger slots for node_42: cpu_limit=8_cores, ram_limit=32Gi, disk_zone=us-central1-b. "
            "What is `gpu_count`? Note: never guess or default; if absent from observed evidence, output EXACTLY `UNKNOWN`.",
            {"accepted_answers": ["UNKNOWN"]},
        ),
        FPBenchmarkTask(
            "fp-ext-04",
            FPAxis.EXACT_TRANSPORT,
            "hard",
            "End with `FINAL_ANSWER: <exact_token>`.",
            "10 noisy records:\n"
            "k1: token_r01=p8z; k2: token_r02=q7x; k3: token_r03=m4n; k4: token_r04=w2c; k5: target_salt=z9Q4-77Lm-x2P1;\n"
            "k6: token_r06=h5k; k7: token_r07=b1v; k8: token_r08=d6y; k9: token_r09=j3f; k10: token_r10=n8t.\n"
            "Retrieve the EXACT case-sensitive `target_salt`.",
            {"accepted_answers": ["z9Q4-77Lm-x2P1"]},
        ),
        FPBenchmarkTask(
            "fp-ext-05",
            FPAxis.EXACT_TRANSPORT,
            "medium",
            "Return ONLY JSON `{\"state\": str, \"value\": str}`.",
            "Query: entity=bot_delta, tenant=ops_2, authorized_tenant=ops_2, slot=rate_cap, as_of_time=5.0.\n"
            "Records:\n"
            "- rec_1: entity=bot_delta, tenant=ops_2, slot=rate_cap, raw_value=240_rpm, valid_time=1.0, record_time=2.0, source_kind=verified_order\n"
            "- rec_2: entity=bot_delta, tenant=ops_1, slot=rate_cap, raw_value=999_rpm, valid_time=3.0, record_time=3.0, source_kind=verified_order\n"
            "Notice rec_2 belongs to different tenant `ops_1`. Return active state and value for `ops_2`.",
            {"expected_json": {"state": "VERIFIED_VALUE", "value": "240_rpm"}},
        ),
        FPBenchmarkTask(
            "fp-ext-06",
            FPAxis.EXACT_TRANSPORT,
            "hard",
            "End with `FINAL_ANSWER: <exact_string>`.",
            "Numeric precision conservation: Source ledger states `concentration=0.0625_mmol` (never round to 0.06 or 0.1). "
            "Repeat the EXACT numeric token before `_mmol` (i.e. 0.0625).",
            {"accepted_answers": ["0.0625"]},
        ),

        # ---------------- Axis 4: Causal vs Associational ----------------
        FPBenchmarkTask(
            "fp-cau-01",
            FPAxis.CAUSAL,
            "hard",
            "Compute the exact interventional probability and end with `FINAL_ANSWER: <float>` (2 decimals, e.g., 0.65).",
            "Confounded causal system with binary confounder Z (P(Z=1)=0.50, P(Z=0)=0.50):\n"
            "- Given Z=0: P(Y=1 | do(X=1), Z=0) = 0.80\n"
            "- Given Z=1: P(Y=1 | do(X=1), Z=1) = 0.40\n"
            "Observational P(Y=1 | X=1) looks higher (0.72) due to selection bias on Z. "
            "Compute the true causal backdoor-adjusted P(Y=1 | do(X=1)).",
            {"accepted_answers": ["0.60", "0.6"]},  # 0.5*0.8 + 0.5*0.4 = 0.60
        ),
        FPBenchmarkTask(
            "fp-cau-02",
            FPAxis.CAUSAL,
            "medium",
            "End with `FINAL_ANSWER: <str>`.",
            "Counterfactual rollout distinction:\n"
            "Statement A comes from a learned transition model predicting tomorrow's score (+12 points) without observation.\n"
            "Which `DerivationMode` MUST tag Statement A so it is never confused with a historical fact: OBSERVED, DERIVED, or PREDICTED?",
            {"accepted_answers": ["PREDICTED"]},
        ),
        FPBenchmarkTask(
            "fp-cau-03",
            FPAxis.CAUSAL,
            "hard",
            "End with `FINAL_ANSWER: <int_or_float>`.",
            "Port-Hamiltonian autonomous system (external drive u = 0, skew exchange J=-J^T, strictly positive dissipation R > 0):\n"
            "Initial state energy H(t_0) = 18.4 Joules. After K=5 autonomous relaxation steps with u=0, "
            "can exact H(t_5) strictly exceed 18.4 under this theorem? Answer CANNOT_EXCEED or CAN_EXCEED.",
            {"accepted_answers": ["CANNOT_EXCEED"]},
        ),
        FPBenchmarkTask(
            "fp-cau-04",
            FPAxis.CAUSAL,
            "hard",
            "Solve numerically and end with `FINAL_ANSWER: <float>` (2 decimals).",
            "Causal do-calculus check with P(Z=0)=0.30, P(Z=1)=0.70.\n"
            "Outcome table: P(Y=1 | X=1, Z=0)=0.90; P(Y=1 | X=1, Z=1)=0.50.\n"
            "P(Y=1 | X=0, Z=0)=0.70; P(Y=1 | X=0, Z=1)=0.30.\n"
            "Compute Average Treatment Effect ATE = P(Y=1 | do(X=1)) - P(Y=1 | do(X=0)).",
            {"accepted_answers": ["0.20", "0.2"]},  # do(1)=0.3*0.9+0.7*0.5=0.62, do(0)=0.3*0.7+0.7*0.3=0.42 -> ATE=0.20
        ),
        FPBenchmarkTask(
            "fp-cau-05",
            FPAxis.CAUSAL,
            "medium",
            "End with `FINAL_ANSWER: <str>`.",
            "Observational correlation shows umbrella sales correlate (r=0.92) with wet sidewalks. "
            "Under causal intervention `do(buy_umbrella=1)`, does wetness of sidewalks change? Reply NO_EFFECT or INCREASES_WETNESS.",
            {"accepted_answers": ["NO_EFFECT"]},
        ),
        FPBenchmarkTask(
            "fp-cau-06",
            FPAxis.CAUSAL,
            "hard",
            "Solve cleanly and end with `FINAL_ANSWER: <int>`.",
            "Symplectic invariant step check: vector q = [3, 4], J rotates (3,4) by 90 degrees preserving ||q||^2 = 25, "
            "followed by damping division by 1 + eta*gamma = 1 (zero damping, gamma=0). What is ||q_next||^2?",
            {"accepted_answers": ["25"]},
        ),

        # ---------------- Axis 5: Cross-Domain Transfer ----------------
        FPBenchmarkTask(
            "fp-xdom-01",
            FPAxis.CROSS_DOMAIN,
            "medium",
            "Return ONLY JSON `{\"state\": str, \"value\": str}`.",
            "Software dependency contract ledger (same bitemporal + epistemic schema):\n"
            "Query: entity=pkg_http_core, tenant=repo_main, authorized_tenant=repo_main, slot=semver, as_of_time=5.0.\n"
            "Records:\n"
            "- rec_s1: entity=pkg_http_core, tenant=repo_main, slot=semver, raw_value=2.1.0, valid_time=1.0, record_time=1.0, source_kind=verified_order\n"
            "- rec_s2: entity=pkg_http_core, tenant=repo_main, slot=semver, raw_value=2.4.0, valid_time=2.0, record_time=2.0, source_kind=verified_order\n"
            "Two active un-superseded verified orders conflict. Return state and latest raw_value.",
            {"expected_json": {"state": "CONFLICTING", "value": "2.4.0"}},
        ),
        FPBenchmarkTask(
            "fp-xdom-02",
            FPAxis.CROSS_DOMAIN,
            "medium",
            "Return ONLY JSON `{\"state\": str, \"value\": str}`.",
            "Distributed cluster lock lease ledger:\n"
            "Query: entity=shard_09, tenant=cluster_eu, authorized_tenant=cluster_eu, slot=lease_holder, as_of_time=10.0.\n"
            "Records:\n"
            "- rec_l1: entity=shard_09, tenant=cluster_eu, slot=lease_holder, raw_value=worker_a, valid_time=2.0, record_time=2.0, source_kind=unverified_report\n"
            "- rec_l2: entity=shard_09, tenant=cluster_eu, slot=lease_holder, raw_value=worker_b, valid_time=4.0, record_time=8.0, source_kind=correction, target_record_id=rec_l1\n"
            "Return active state and holder as_of_time=10.0.",
            {"expected_json": {"state": "VERIFIED_VALUE", "value": "worker_b"}},
        ),
        FPBenchmarkTask(
            "fp-xdom-03",
            FPAxis.CROSS_DOMAIN,
            "hard",
            "Solve carefully and end with `FINAL_ANSWER: <int>`.",
            "Conservation invariant transfer (financial double-entry ledger): Accounts [A, B, C] start with [120, 85, 95]. "
            "Transfer 35 from A->B, 20 from B->C, 15 from C->A. What is the conserved sum A + B + C?",
            {"accepted_answers": ["300"]},
        ),
        FPBenchmarkTask(
            "fp-xdom-04",
            FPAxis.CROSS_DOMAIN,
            "hard",
            "Write pure Python defining `resolve_as_of(events: list[dict], cutoff: float) -> str` returning the latest raw_value with record_time <= cutoff or 'UNKNOWN'.",
            "Implement `def resolve_as_of(events: list[dict], cutoff: float) -> str:` where each dict has "
            "`{'record_time': float, 'valid_time': float, 'value': str, 'superseded': bool}`. "
            "Filter `record_time <= cutoff` and not `superseded`; return the `value` of the item maximizing `(valid_time, record_time)`, else `'UNKNOWN'`.",
            {
                "assertions": [
                    "assert resolve_as_of([{'record_time': 1.0, 'valid_time': 1.0, 'value': 'v1', 'superseded': False}, {'record_time': 5.0, 'valid_time': 2.0, 'value': 'v2', 'superseded': False}], 3.0) == 'v1'",
                    "assert resolve_as_of([{'record_time': 1.0, 'valid_time': 1.0, 'value': 'v1', 'superseded': False}, {'record_time': 5.0, 'valid_time': 2.0, 'value': 'v2', 'superseded': False}], 6.0) == 'v2'",
                    "assert resolve_as_of([{'record_time': 1.0, 'valid_time': 1.0, 'value': 'old', 'superseded': True}, {'record_time': 2.0, 'valid_time': 1.5, 'value': 'new', 'superseded': False}], 4.0) == 'new'",
                    "assert resolve_as_of([], 1.0) == 'UNKNOWN'",
                ]
            },
        ),
        FPBenchmarkTask(
            "fp-xdom-05",
            FPAxis.CROSS_DOMAIN,
            "hard",
            "End with `FINAL_ANSWER: <int>`.",
            "Modular group conservation invariant: state s starts at 7. Step multiplies by 5 mod 19, repeated 3 times (7 * 5^3 mod 19). Compute final s.",
            {"accepted_answers": ["1"]},  # 7 * 125 = 875; 19 * 46 = 874; 875 mod 19 = 1!
        ),
        FPBenchmarkTask(
            "fp-xdom-06",
            FPAxis.CROSS_DOMAIN,
            "medium",
            "Return ONLY JSON `{\"domain\": str, \"verifiable\": bool, \"state\": str}`.",
            "Cross-domain schema consistency check: Classify record `{slot: 'voltage_mv', raw_value: '3300', status: 'PATIENT_REPORTED_EQUIV_UNVERIFIED', valid_time: 1.0}` as "
            "domain='physics_telemetry', verifiable=False, state='REPORTED_UNVERIFIED'. Return that exact JSON.",
            {"expected_json": {"domain": "physics_telemetry", "verifiable": False, "state": "REPORTED_UNVERIFIED"}},
        ),
    )
    return tasks


def verify_fp_response(task: FPBenchmarkTask, response: str) -> TaskVerdict:
    """Deterministically grade one FP-5Axis-30 response with zero LLM judges."""
    spec = task.spec
    if "expected_json" in spec:
        return _verify_json_invariant(response, spec)
    if "accepted_answers" in spec:
        return _verify_exact_answer(response, spec)
    if "assertions" in spec:
        from nanolm_posttrain.eval_suite import _verify_coding
        return _verify_coding(response, spec)
    raise ValueError(f"Unknown spec format for {task.task_id}: {spec}")


def deterministic_fp_symbolic_baseline(task: FPBenchmarkTask) -> str:
    """Honest zero-parameter symbolic baseline over FP-5Axis-30 tasks (no LLM)."""
    from nanolm_holonomy.first_principles import (
        AppendOnlySourceLedger,
        DeterministicTypedRuleSolver,
        LedgerRecord,
    )
    from nano.contracts import DerivationMode, EpistemicStatus

    prompt = task.user_prompt
    # 1. Can parse explicit Query + Records blocks deterministically
    q_m = re.search(
        r"Query:\s*entity=([^\s,;]+),\s*tenant=([^\s,;]+),\s*authorized_tenant=([^\s,;]+),\s*slot=([^\s,;]+),\s*as_of_time=([0-9]+(?:\.[0-9]+)?)",
        prompt,
    )
    rec_matches = re.findall(
        r"-\s*([^\s:]+):\s*entity=([^\s,;]+),\s*tenant=([^\s,;]+),\s*slot=([^\s,;]+),\s*raw_value=([^\s,;]+),\s*valid_time=([0-9]+(?:\.[0-9]+)?),\s*record_time=([0-9]+(?:\.[0-9]+)?)[^,]*,\s*source_kind=([^\s,;\n]+)(?:,\s*target_record_id=([^\s,;\n]+))?",
        prompt,
    )
    if q_m and rec_matches:
        q_ent, q_ten, auth_ten, q_slot, as_of_s = (s.rstrip(".,;") for s in q_m.groups())
        ledger = AppendOnlySourceLedger()
        for rid, ent, ten, slot, val, vt, rt, kind, tgt in rec_matches:
            rid = rid.rstrip(".,;")
            ent = ent.rstrip(".,;")
            ten = ten.rstrip(".,;")
            slot = slot.rstrip(".,;")
            val = val.rstrip(".,;")
            kind = kind.rstrip(".,;")
            tgt = (tgt or "").rstrip(".,;")
            ep = EpistemicStatus.CLINICIAN_ASSERTED if kind in ("verified_order", "correction") else EpistemicStatus.PATIENT_REPORTED
            txt = f"[{kind}]" + (f" supersedes={tgt}" if kind == "correction" and tgt else (f" cancels={tgt}" if kind == "retraction" and tgt else ""))
            ledger.append(
                LedgerRecord(
                    record_id=rid,
                    entity_id=ent,
                    tenant_id=ten,
                    slot=slot,
                    raw_value=val,
                    polarity=(-1 if kind == "retraction" else (1 if kind in ("verified_order", "correction") else 0)),
                    valid_time=float(vt),
                    record_time=float(rt),
                    source_kind=kind,
                    epistemic_status=ep,
                    derivation_mode=DerivationMode.OBSERVED,
                    text=txt,
                    target_record_id=tgt or "",
                )
            )
        res = DeterministicTypedRuleSolver.resolve_ledger(
            ledger,
            entity_id=q_ent,
            tenant_id=q_ten,
            authorized_tenant_id=auth_ten,
            slot=q_slot,
            as_of_time=float(as_of_s),
        )
        ans = res["raw_value"] or ("DENIED" if res["state"] == "ISOLATED_DENIED" else "UNKNOWN")
        return json.dumps({"state": res["state"], "value": ans})

    # 2. Can parse strict Parallel DAG duration/deps syntax deterministically
    dag_rows = re.findall(r"-\s*([A-Za-z0-9_]+):\s*duration=(\d+),\s*deps=\[([^\]]*)\]", prompt)
    if dag_rows:
        dur = {tid: int(d) for tid, d, _ in dag_rows}
        deps = {tid: [x.strip() for x in dp.split(",") if x.strip()] for tid, _, dp in dag_rows}
        finish: dict[str, int] = {}

        def _ef(u: str) -> int:
            if u not in finish:
                finish[u] = dur[u] + max((_ef(p) for p in deps[u] if p in dur), default=0)
            return finish[u]

        ms = max(_ef(u) for u in dur)
        return f"FINAL_ANSWER: {ms}"

    # Unstructured natural language / causal / Python tasks cannot be reliably parsed by fixed regexes
    return "FINAL_ANSWER: UNKNOWN"


def verify_fp_zero_contamination(
    corpus_texts: list[str],
    fp_tasks: tuple[FPBenchmarkTask, ...] | list[FPBenchmarkTask],
    *,
    ngram: int = 13,
) -> dict[str, Any]:
    """Assert zero n-gram overlap between training texts and held-out FP-5Axis-30 tasks."""
    def _ngrams(txt: str) -> set[str]:
        toks = re.findall(r"[a-z0-9_]+", txt.casefold())
        if len(toks) < ngram:
            return {" ".join(toks)} if len(toks) >= 6 else set()
        return {" ".join(toks[i : i + ngram]) for i in range(len(toks) - ngram + 1)}

    eval_grams: dict[str, set[str]] = {}
    for t in fp_tasks:
        eval_grams[t.task_id] = _ngrams(t.user_prompt)

    overlaps = 0
    hits: list[dict[str, str]] = []
    for idx, c_txt in enumerate(corpus_texts):
        cg = _ngrams(c_txt)
        for tid, eg in eval_grams.items():
            shared = cg & eg
            if shared:
                overlaps += len(shared)
                hits.append({"train_idx": str(idx), "task_id": tid})
    if overlaps != 0:
        raise ValueError(f"Contamination detected against FP-5Axis-30: {hits[:5]}")
    return {
        "overlaps": overlaps,
        "train_examples_checked": len(corpus_texts),
        "fp_eval_tasks_checked": len(fp_tasks),
    }


__all__ = [
    "FPAxis",
    "FPBenchmarkTask",
    "build_fp_5axis_benchmark",
    "deterministic_fp_symbolic_baseline",
    "verify_fp_response",
    "verify_fp_zero_contamination",
]
