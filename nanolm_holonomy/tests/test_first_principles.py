"""Unit, Stability & Verification Tests for Cycle-4 First-Principles Modules."""

from __future__ import annotations

from dataclasses import replace
import inspect
import random

import pytest
import torch

from nano.contracts import DerivationMode, EpistemicStatus
from nanolm_holonomy.architecture import (
    HSPA_GEN5_2B_CONFIG,
    HSPA_GEN5_4B_CONFIG,
    HSPAFamily,
    HSPAUnifiedLM,
    PortHamiltonianFFNCell,
    build_suite_configs,
)
from nanolm_holonomy.compute_planner import plan_fp_v3_campaign
from nanolm_holonomy.curriculum_v2 import build_curriculum_v2
from nanolm_holonomy.cycle4_first_principles import (
    run_exp_fp1_relational_microcosm,
    run_exp_fp2_gen5_sequence_evolution,
    run_exp_fp3_primitives_and_voi,
)
from nanolm_holonomy.first_principles import (
    AppendOnlySourceLedger,
    DeterministicTypedRuleSolver,
    LedgerRecord,
    PortHamiltonianInteractionNet,
    VoIHaltingPolicy,
    WorldEpistemicState,
    bitemporal_admissibility_mask,
    collate_microcosm_batch,
    generate_microcosm_case,
)
from nanolm_holonomy.fp_benchmark import (
    build_fp_5axis_benchmark,
    deterministic_fp_symbolic_baseline,
    verify_fp_response,
    verify_fp_zero_contamination,
)
from nanolm_holonomy.omega_system import (
    classify_fp_task_mode,
    extract_required_python_symbol,
    omega_fp_generate,
    omega_generate,
    solve_dag_makespan_from_json,
    verify_python_smoke_zero_oracle,
)
from nanolm_holonomy.sidecar_evolution import build_challenging_rlvr_curriculum
from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS


def test_append_only_ledger_immutability_and_bitemporal_non_leakage() -> None:
    ledger = AppendOnlySourceLedger()
    r1 = LedgerRecord(
        record_id="rec_mon",
        entity_id="ent_p1",
        tenant_id="tenant_a",
        slot="med_x",
        raw_value="10_mg",
        polarity=0,
        valid_time=1.0,
        record_time=1.0,
        source_kind="unverified_report",
        epistemic_status=EpistemicStatus.PATIENT_REPORTED,
        derivation_mode=DerivationMode.OBSERVED,
        text="[unverified_report] slot=med_x val=10_mg",
    )
    r2 = LedgerRecord(
        record_id="rec_wed",
        entity_id="ent_p1",
        tenant_id="tenant_a",
        slot="med_x",
        raw_value="20_mg",
        polarity=0,
        valid_time=2.5,
        record_time=3.0,
        source_kind="unverified_report",
        epistemic_status=EpistemicStatus.PATIENT_REPORTED,
        derivation_mode=DerivationMode.OBSERVED,
        text="[unverified_report] slot=med_x val=20_mg",
    )
    r3 = LedgerRecord(
        record_id="rec_fri",
        entity_id="ent_p1",
        tenant_id="tenant_a",
        slot="med_x",
        raw_value="15_mg",
        polarity=1,
        valid_time=2.5,
        record_time=5.0,
        source_kind="correction",
        epistemic_status=EpistemicStatus.CLINICIAN_ASSERTED,
        derivation_mode=DerivationMode.OBSERVED,
        text="[correction] supersedes=rec_wed slot=med_x val=15_mg",
        target_record_id="rec_wed",
    )
    r_other = LedgerRecord(
        record_id="rec_other_ten",
        entity_id="ent_p1",
        tenant_id="tenant_b",
        slot="med_x",
        raw_value="99_mg",
        polarity=1,
        valid_time=1.0,
        record_time=1.0,
        source_kind="verified_order",
        epistemic_status=EpistemicStatus.CLINICIAN_ASSERTED,
        derivation_mode=DerivationMode.OBSERVED,
        text="[verified_order] val=99_mg",
    )
    for rec in (r1, r2, r3, r_other):
        ledger.append(rec)

    # Attempting to overwrite an existing record_id must raise ValueError
    with pytest.raises(ValueError, match="Append-only violation"):
        ledger.append(r1)

    p_mask, q_mask = bitemporal_admissibility_mask(
        ledger.records,
        query_entity_id="ent_p1",
        query_tenant_id="tenant_a",
        as_of_time=3.0,
    )
    assert p_mask == [1, 1, 1, 0]
    assert q_mask == [1, 1, 0, 1]

    wed_res = DeterministicTypedRuleSolver.resolve_ledger(
        ledger, entity_id="ent_p1", tenant_id="tenant_a", authorized_tenant_id="tenant_a", slot="med_x", as_of_time=3.0
    )
    assert wed_res["state"] == WorldEpistemicState.CONFLICTING.value

    fri_res = DeterministicTypedRuleSolver.resolve_ledger(
        ledger, entity_id="ent_p1", tenant_id="tenant_a", authorized_tenant_id="tenant_a", slot="med_x", as_of_time=5.0
    )
    assert fri_res["state"] == WorldEpistemicState.VERIFIED_VALUE.value
    assert fri_res["raw_value"] == "15_mg"

    denied = DeterministicTypedRuleSolver.resolve_ledger(
        ledger, entity_id="ent_p1", tenant_id="tenant_a", authorized_tenant_id="tenant_b", slot="med_x", as_of_time=5.0
    )
    assert denied["state"] == WorldEpistemicState.ISOLATED_DENIED.value


def test_port_hamiltonian_lyapunov_energy_monotone_decay() -> None:
    torch.manual_seed(17)
    rng = random.Random(17)
    cases = [generate_microcosm_case(rng, n_nodes=16, canonical=(i % 2 == 0)) for i in range(4)]
    batch = collate_microcosm_batch(cases, torch.device("cpu"))
    for variant in ("sheaf_port_hamiltonian", "dynamic_rewire_ph"):
        net = PortHamiltonianInteractionNet(variant=variant, d_model=48, steps=6)
        out = net.forward_trajectory(
            batch["text_ids"],
            batch["meta_feats"],
            batch["edge_types"],
            batch["p_mask"],
            batch["q_mask"],
            batch["slot_mask"],
            n_steps=12,
            autonomous_drive_zero=True,
        )
        en = out["energies"]
        diffs = en[:, 1:] - en[:, :-1]
        assert torch.all(diffs <= 1e-5), f"{variant} violated autonomous energy decay: max_diff={diffs.max()}"

    ffn = PortHamiltonianFFNCell(32, 64, iters=1)
    c = torch.randn(3, 6, 64)
    prev = ffn.energy(c)
    for _ in range(8):
        c = ffn(torch.zeros(3, 6, 32), autonomous_drive_zero=True, initial_c=c)
        cur = ffn.energy(c)
        assert torch.all(cur <= prev + 1e-6)
        prev = cur


def test_deltanet_and_gen5_param_counts_and_step_parity() -> None:
    assert HSPA_GEN5_2B_CONFIG.parameter_count <= ABSOLUTE_MAX_PARAMETERS
    assert HSPA_GEN5_4B_CONFIG.parameter_count <= ABSOLUTE_MAX_PARAMETERS

    cfgs = build_suite_configs(preset="small_x1", vocab_size=256, max_seq_len=128)
    for cfg in cfgs:
        model = HSPAUnifiedLM(cfg)
        assert cfg.parameter_count == model.num_parameters, f"{cfg.name}: {cfg.parameter_count} != {model.num_parameters}"
        if cfg.family == HSPAFamily.GEN5_FP_PH_HOLONOMY:
            nocopy = replace(cfg, use_copy_gate=False)
            assert nocopy.parameter_count == HSPAUnifiedLM(nocopy).num_parameters

    for fam in (HSPAFamily.B2_GATED_DELTANET, HSPAFamily.GEN5_FP_PH_HOLONOMY):
        cfg = next(c for c in cfgs if c.family == fam)
        model = HSPAUnifiedLM(cfg).eval()
        x = torch.randint(0, 256, (2, 20))
        with torch.no_grad():
            full = model(x)
            _, st = model(x[:, :10], use_state=True)
            step_outs = []
            for t in range(10, 20):
                lt, st = model(x[:, t : t + 1], past_states=st, use_state=True, past_len=t)
                step_outs.append(lt)
            joined = torch.cat(step_outs, dim=1)
        diff = float((full[:, 10:] - joined).abs().max().item())
        assert diff < 1e-4, f"Step parity mismatch for {fam.value}: {diff}"


def test_controller_v3_zero_oracle_contract_and_fp_benchmark() -> None:
    for fn in (omega_generate, omega_fp_generate):
        sig = inspect.signature(fn)
        pnames = set(sig.parameters)
        assert "verifier_spec" not in pnames
        assert "task" not in pnames
        assert {"system_prompt", "user_prompt", "max_new_tokens", "device"}.issubset(pnames)

    # Required Python symbol contract detection (catches NameError: build_bst class of bugs)
    sym = extract_required_python_symbol("Write pure Python defining `def build_bst(nums: list[int]):` returning balanced root.")
    assert sym == "build_bst"
    ok_missing, err_missing = verify_python_smoke_zero_oracle("def helper(x):\n    return x\n", required_symbol=sym)
    assert not ok_missing and "ContractError" in err_missing

    # Topological DAG critical-path solver
    dag_json = '{"tasks": [{"id": "A", "duration": 4, "deps": []}, {"id": "B", "duration": 9, "deps": ["A"]}, {"id": "C", "duration": 6, "deps": ["A"]}, {"id": "D", "duration": 7, "deps": ["B", "C"]}]}'
    assert solve_dag_makespan_from_json(dag_json) == 20.0
    assert solve_dag_makespan_from_json('{"tasks": [{"id": "A", "duration": 2, "deps": ["B"]}, {"id": "B", "duration": 2, "deps": ["A"]}]}') is None

    # FP-5Axis-30 construction + zero contamination check + deterministic baseline
    fp_tasks = build_fp_5axis_benchmark()
    assert len(fp_tasks) == 30
    cur_texts = [e.user_prompt + "\n" + e.reference_response for e in build_curriculum_v2(seed=31415)]
    rl_texts = [e.user_prompt + "\n" + e.reference_response for e in build_challenging_rlvr_curriculum(seed=777)]
    dec = verify_fp_zero_contamination(cur_texts + rl_texts, fp_tasks)
    assert dec["overlaps"] == 0

    assert classify_fp_task_mode("Return JSON", fp_tasks[6].user_prompt) == "WORLD_LEDGER_OR_CAUSAL"
    # Deterministic baseline passes explicit ledger & explicit DAG tasks (and refuses unstructured ones)
    det_passed = sum(1 for t in fp_tasks if verify_fp_response(t, deterministic_fp_symbolic_baseline(t)).passed)
    assert det_passed >= 7

    plan = plan_fp_v3_campaign(allow_kaggle=True, live_runpod_balance_usd=23.0448)
    assert plan["financial_guard"]["enforce_zero_runpod_spend"] is True
    assert plan["selection"]["recommended"]["estimated_usd"] == 0.0


def test_quick_smoke_fp1_fp2_fp3_cpu() -> None:
    dev = torch.device("cpu")
    r1 = run_exp_fp1_relational_microcosm(device=dev, seeds=(101,), train_steps=2, eval_cases=6, ood_cases=2, ood_nodes=24)
    assert r1["arms"][0]["clean_canonical"]["joint_acc"] >= 0.80

    r2 = run_exp_fp2_gen5_sequence_evolution(
        device=dev,
        s4_seeds=(101,),
        s4_steps=2,
        lm_seeds=(42,),
        lm_steps=2,
        muon_steps=2,
        batch_size=2,
        lm_preset="small_x1",
        train_seqs=32,
        val_seqs_per_dom=4,
    )
    assert len(r2["s4_results"]) == 5 and len(r2["lm_results"]) == 6

    r3 = run_exp_fp3_primitives_and_voi(device=dev, seeds=(101,), train_steps=2, eval_cases=6)
    assert len(r3["knockouts"]) == 9 and len(r3["voi_pareto"]) >= 4
    pol = VoIHaltingPolicy()
    assert bool(pol.should_halt(torch.tensor(0.01), torch.tensor(0.99), step_idx=1).item()) is True
