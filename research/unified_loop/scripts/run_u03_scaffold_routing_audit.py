#!/usr/bin/env python3
"""EXP-U03: Factorial Scaffold Attribution (E23 C7-Omega) & Routing Paradox Audit (E24 HEM-ATRD)."""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
MAIN_REPO = Path("/Users/mac/Projects/nano-lm")
V3_ROOT = MAIN_REPO / ".worktrees" / "nanolm-v3"
for p in (str(MAIN_REPO), str(V3_ROOT), str(MAIN_REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_e23_dynamic_debruijn_cognition as e23  # noqa: E402
import run_e24_hybrid_entropy_margin_routing as e24  # noqa: E402
from nanolm.omega.debruijn_symbolic_workspace import (  # noqa: E402
    DynamicDeBruijnBeliefWorkspace,
    SymbolicBeliefState,
)
from nanolm.omega.margin_shield_refiner import topk_renorm_entropy  # noqa: E402

OUT_DIR = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U03_scaffold_routing_audit"


def mean_std(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / max(1, n - 1)
    return {"mean": round(m, 4), "std": round(math.sqrt(var), 4)}


def render_with_sym_mode(
    ep: e23.DeepCausalChainEpisode,
    ws: DynamicDeBruijnBeliefWorkspace,
    *,
    training: bool,
    rng_perm: list[int] | None,
    use_sym_feats: bool,
):
    toks, s_feats, v_pos, v_lits, fmap, rmap, sym = e23.render_state_for_workspace(
        ep, ws, training=training, rng_perm=rng_perm
    )
    if not use_sym_feats:
        s_feats = torch.zeros_like(s_feats)
    return toks, s_feats, v_pos, v_lits, fmap, rmap, sym


def eval_part_a_policy(
    ws: DynamicDeBruijnBeliefWorkspace | None,
    *,
    use_sym_feats: bool,
    use_shield: bool,
    n_episodes: int,
    n_modules: int,
    chain_len: int,
    ood_high_ids: bool,
    seed: int,
    act_logit_noise: float = 0.0,
) -> dict[str, float]:
    rng = random.Random(seed)
    torch.manual_seed(seed + 7)
    verified_ok = val_ok = contra_total = contra_ok = unsafe_cnt = 0

    if ws is not None:
        ws.eval()

    with torch.no_grad():
        for _ in range(n_episodes):
            ep = e23.DeepCausalChainEpisode(
                n_modules=n_modules, chain_len=chain_len, rng=rng, ood_high_ids=ood_high_ids
            )
            max_turns = chain_len + 9
            for _t in range(max_turns):
                if ep.done:
                    break
                if ws is None:
                    # 0-parameter deterministic rule using identical SymbolicBeliefState observable fields
                    chron = [e for _, e, _ in ep.events if e is not None]
                    sym = SymbolicBeliefState(max_entities=16)
                    for ev_tok, raw_e, _ in ep.events:
                        if ev_tok == e23.TOK_AMBIG:
                            sym.ambiguous_spec = True
                        elif ev_tok == e23.TOK_CLARIFIED:
                            sym.clarified = True
                        elif ev_tok == e23.TOK_FAULT_AT and raw_e is not None:
                            sym.inspected_entities.add(raw_e)
                            sym.verified_fault_entity = raw_e
                        elif ev_tok == e23.TOK_HEALTHY_AT and raw_e is not None:
                            sym.inspected_entities.add(raw_e)
                        elif ev_tok == e23.TOK_CONTRA_VAL:
                            sym.tests_verified_pass = False
                        elif ev_tok == e23.TOK_PATCH_UNTESTED and raw_e is not None:
                            sym.patched_entity = raw_e
                            sym.tests_verified_pass = False
                        elif ev_tok == e23.TOK_TESTS_PASS:
                            sym.tests_verified_pass = True
                    v_lits = [val for _, _, val in ep.events if val is not None]
                    leaf_confirmed = any(ev == e23.TOK_LEAF_SUSPECT for ev, _, _ in ep.events)
                    latest_e = chron[-1]
                    chosen_val = v_lits[-1] if v_lits else ep.init_val
                    if sym.ambiguous_spec and not sym.clarified:
                        act, raw_ent = DynamicDeBruijnBeliefWorkspace.ACT_CLARIFY, chron[0]
                    elif not leaf_confirmed:
                        act, raw_ent = DynamicDeBruijnBeliefWorkspace.ACT_TRACE, latest_e
                    elif sym.verified_fault_entity is None:
                        act, raw_ent = DynamicDeBruijnBeliefWorkspace.ACT_INSPECT, latest_e
                    elif not sym.tests_verified_pass and sym.patched_entity != sym.verified_fault_entity:
                        act, raw_ent = DynamicDeBruijnBeliefWorkspace.ACT_PATCH, sym.verified_fault_entity
                    elif sym.tests_verified_pass:
                        act, raw_ent = DynamicDeBruijnBeliefWorkspace.ACT_STOP, sym.verified_fault_entity
                    else:
                        # Handle post-contradiction re-patch if patched before or after contra
                        last_patch_idx = max((i for i, (ev, _, _) in enumerate(ep.events) if ev == e23.TOK_PATCH_UNTESTED), default=-1)
                        last_contra_idx = max((i for i, (ev, _, _) in enumerate(ep.events) if ev == e23.TOK_CONTRA_VAL), default=-1)
                        if last_contra_idx > last_patch_idx:
                            act, raw_ent = DynamicDeBruijnBeliefWorkspace.ACT_PATCH, sym.verified_fault_entity
                        else:
                            act, raw_ent = DynamicDeBruijnBeliefWorkspace.ACT_TESTS, sym.verified_fault_entity
                else:
                    toks, s_feats, v_pos, v_lits, _, rmap, sym = render_with_sym_mode(
                        ep, ws, training=False, rng_perm=None, use_sym_feats=use_sym_feats
                    )
                    out_raw = ws.forward_decision(
                        toks,
                        s_feats,
                        v_pos,
                        valid_canonical_slots=ws.num_slots,
                        symbolic_state=sym,
                        apply_shield=False,
                    )
                    act_lg = out_raw["act_logits"].clone()
                    if act_logit_noise > 0.0:
                        act_lg = act_lg + torch.randn_like(act_lg) * act_logit_noise
                    if use_shield:
                        if sym.ambiguous_spec and not sym.clarified:
                            act_lg[ws.ACT_PATCH] = -1e9
                            act_lg[ws.ACT_STOP] = -1e9
                            act_lg[ws.ACT_TESTS] = -1e9
                        if sym.verified_fault_entity is None:
                            act_lg[ws.ACT_PATCH] = -1e9
                        if not sym.tests_verified_pass:
                            act_lg[ws.ACT_STOP] = -1e9
                        if sym.tests_verified_pass:
                            act_lg[ws.ACT_PATCH] = -1e9
                            act_lg[ws.ACT_CLARIFY] = -1e9
                    act = int(act_lg.argmax().item())
                    s_logits = out_raw["slot_logits"].clone()
                    for s_idx in range(ws.num_slots):
                        if s_idx not in rmap:
                            s_logits[s_idx] = -1e9
                    c_slot = int(s_logits.argmax().item())
                    raw_ent = rmap.get(c_slot, 0)
                    chosen_val = (
                        v_lits[min(int(out_raw["ptr_logits"].argmax().item()), len(v_lits) - 1)]
                        if v_lits
                        else ep.init_val
                    )
                ep.step(act, raw_ent, chosen_val)

            full_ok = (
                ep.done
                and ep.tests_passed
                and not ep.unsafe_regression
                and ep.patched_entity == ep.fault_entity
                and ep.patched_val == ep.active_target_val
            )
            if full_ok:
                verified_ok += 1
            if ep.patched_val == ep.active_target_val:
                val_ok += 1
            if ep.has_contradiction:
                contra_total += 1
                if ep.patched_val == ep.revised_val:
                    contra_ok += 1
            if ep.unsafe_regression:
                unsafe_cnt += 1

    return {
        "verified_resolution": verified_ok / float(n_episodes),
        "value_custody_acc": val_ok / float(n_episodes),
        "contradiction_revision_acc": contra_ok / float(max(1, contra_total)),
        "unsafe_regression_rate": unsafe_cnt / float(n_episodes),
    }


def run_part_a_arm(canonical_mode: str, use_sym_feats: bool, use_shield: bool, seed: int) -> dict[str, float]:
    if canonical_mode == "classical":
        id_res = eval_part_a_policy(None, use_sym_feats=True, use_shield=True, n_episodes=80, n_modules=6, chain_len=2, ood_high_ids=False, seed=50_000 + seed)
        ood_res = eval_part_a_policy(None, use_sym_feats=True, use_shield=True, n_episodes=100, n_modules=14, chain_len=4, ood_high_ids=True, seed=70_000 + seed)
        stress_res = eval_part_a_policy(None, use_sym_feats=True, use_shield=True, n_episodes=100, n_modules=14, chain_len=4, ood_high_ids=True, seed=80_000 + seed, act_logit_noise=3.5)
        return {
            "params": 0.0,
            "id_verified": id_res["verified_resolution"],
            "deep_ood_verified": ood_res["verified_resolution"],
            "deep_ood_value_custody": ood_res["value_custody_acc"],
            "deep_ood_contra_rev": ood_res["contradiction_revision_acc"],
            "stress_verified": stress_res["verified_resolution"],
            "stress_unsafe": stress_res["unsafe_regression_rate"],
        }

    torch.manual_seed(seed * 199 + 11)
    rng = random.Random(10_000 + seed * 83)
    ws = DynamicDeBruijnBeliefWorkspace(
        e23.VOCAB_SIZE, d_model=48, num_slots=16, canonical_mode=canonical_mode, use_symbolic_shield=use_shield
    )
    opt = torch.optim.AdamW(ws.parameters(), lr=6e-3, weight_decay=1e-4)

    ws.train()
    for ep_idx in range(160):
        c_len = 2 if (ep_idx % 10 < 7) else 3
        ep = e23.DeepCausalChainEpisode(n_modules=6, chain_len=c_len, rng=rng, ood_high_ids=False)
        ep_perm = torch.randperm(ws.num_slots).tolist() if canonical_mode == "random_sym" else None
        loss_acc = torch.tensor(0.0)
        steps_taken = 0
        while not ep.done and steps_taken < 11:
            ora_act, ora_raw_ent, ora_val = ep.oracle_action()
            toks, s_feats, v_pos, v_lits, fmap, _, sym = render_with_sym_mode(
                ep, ws, training=True, rng_perm=ep_perm, use_sym_feats=use_sym_feats
            )
            out = ws.forward_decision(
                toks, s_feats, v_pos, valid_canonical_slots=ws.num_slots, symbolic_state=sym, apply_shield=False
            )
            loss_act = F.cross_entropy(out["act_logits"].unsqueeze(0), torch.tensor([ora_act]))
            ora_c_slot = fmap[ora_raw_ent]
            loss_slot = (
                F.cross_entropy(out["slot_logits"].unsqueeze(0), torch.tensor([ora_c_slot]))
                if ora_act in (ws.ACT_TRACE, ws.ACT_INSPECT, ws.ACT_PATCH)
                else torch.tensor(0.0)
            )
            if ora_act == ws.ACT_PATCH and ora_val in v_lits:
                target_ptr = max(i for i, v in enumerate(v_lits) if v == ora_val)
                loss_ptr = F.cross_entropy(out["ptr_logits"].unsqueeze(0), torch.tensor([target_ptr]))
            else:
                loss_ptr = torch.tensor(0.0)
            loss_acc = loss_acc + loss_act + loss_slot + loss_ptr
            ep.step(ora_act, ora_raw_ent, ora_val)
            steps_taken += 1

        opt.zero_grad(set_to_none=True)
        (loss_acc / max(1, steps_taken)).backward()
        torch.nn.utils.clip_grad_norm_(ws.parameters(), 1.0)
        opt.step()

    id_res = eval_part_a_policy(ws, use_sym_feats=use_sym_feats, use_shield=use_shield, n_episodes=80, n_modules=6, chain_len=2, ood_high_ids=False, seed=50_000 + seed)
    ood_res = eval_part_a_policy(ws, use_sym_feats=use_sym_feats, use_shield=use_shield, n_episodes=100, n_modules=14, chain_len=4, ood_high_ids=True, seed=70_000 + seed)
    stress_res = eval_part_a_policy(ws, use_sym_feats=use_sym_feats, use_shield=use_shield, n_episodes=100, n_modules=14, chain_len=4, ood_high_ids=True, seed=80_000 + seed, act_logit_noise=3.5)
    return {
        "params": float(sum(p.numel() for p in ws.parameters())),
        "id_verified": id_res["verified_resolution"],
        "deep_ood_verified": ood_res["verified_resolution"],
        "deep_ood_value_custody": ood_res["value_custody_acc"],
        "deep_ood_contra_rev": ood_res["contradiction_revision_acc"],
        "stress_verified": stress_res["verified_resolution"],
        "stress_unsafe": stress_res["unsafe_regression_rate"],
    }


@torch.no_grad()
def force_pass2_all_logits(model: e24.Vocab128RefinedLM, x: Tensor) -> tuple[Tensor, Tensor]:
    """Return `(logits1, unconditional_pass2_logits)` without gating or Lyapunov rejection."""
    bsz, seq_len = x.shape
    pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, seq_len)
    h = model.backbone.embed_tokens(x)
    cos, sin = model.backbone.rotary_emb(pos, dtype=h.dtype)
    h1, _, _, _ = model.backbone.layers[0](h, cos, sin, v0=None, depth_bank=None)
    logits1 = model.unembed(h1)
    if model.refiner is None:
        return logits1, logits1
    ref = model.refiner
    x_sel = h1 + ref.step_emb.to(h1.dtype)
    q = ref.q_proj(ref.norm1(x_sel)).view(bsz, seq_len, ref.num_heads, ref.head_dim).transpose(1, 2)
    kv_full = ref.kv_proj(ref.norm1(h1))
    k_f, v_f = kv_full.chunk(2, dim=-1)
    k = k_f.view(bsz, seq_len, ref.num_heads, ref.head_dim).transpose(1, 2)
    v = v_f.view(bsz, seq_len, ref.num_heads, ref.head_dim).transpose(1, 2)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)).view(1, 1, seq_len, seq_len)
    y_attn = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)
    dx_attn = ref.o_proj(y_attn.transpose(1, 2).reshape(bsz, seq_len, -1))
    g_ffn, u_ffn = ref.ffn_gu(ref.norm2(x_sel + dx_attn)).chunk(2, dim=-1)
    dx_tot = dx_attn + ref.ffn_down(F.silu(g_ffn) * u_ffn)
    return logits1, model.unembed(h1 + dx_tot)


def run_part_b_arm(arm: str, seed: int, steps: int = 200) -> dict[str, Any]:
    """Instrumented E24 run isolating training-starvation vs eval Lyapunov-rejection."""
    base_arm = {
        "H0_pass1_unrouted": "H0_pass1_unrouted",
        "H1_x5_full_vocab_shannon": "H1_x5_full_vocab_shannon",
        "H2_x8_top2_margin_quantile": "H2_x8_top2_margin_quantile",
        "H3_mm_atrd_top2_shield": "H3_mm_atrd_top2_shield",
        "H4_hem_atrd_topk_shield": "H4_hem_atrd_topk_shield",
        "H5_hem_train_warmup": "H4_hem_atrd_topk_shield",
        "H6_hem_warmup_decoupled_eval": "H4_hem_atrd_topk_shield",
    }[arm]

    torch.manual_seed(seed * 131 + 17)
    rng = random.Random(30_000 + seed * 71)
    model = e24.Vocab128RefinedLM(base_arm)
    opt = torch.optim.AdamW(model.parameters(), lr=1.1e-2, weight_decay=1e-4)

    train_qry_routed_last50: list[float] = []
    model.train()
    for st in range(steps):
        bx_c, by_c, m_c = e24.make_copy_batch(16, rng)
        bx_h, by_h, m_h = e24.make_8way_relational_batch(16, rng, eval_uncorr=False)
        bx = torch.cat([bx_c, bx_h], dim=0)
        by = torch.cat([by_c, by_h], dim=0)
        lmask1 = torch.cat([m_c, m_h], dim=0)
        lmask2 = lmask1.clone()
        lmask2[16:, 15] = 14.0

        # For H5/H6: during first 100 steps keep target_route_frac higher (0.60 -> 0.30)
        # so ema_tau doesn't starve uncertain relational queries before Pass-2 learns!
        if arm in ("H5_hem_train_warmup", "H6_hem_warmup_decoupled_eval") and model.refiner is not None:
            model.refiner.target_route_frac = 0.65 if st < 120 else 0.30

        lg1, lg2, stats_tr = model.forward_pass(bx)
        if st >= steps - 50:
            train_qry_routed_last50.append(float(stats_tr["hard_mask"][16:, 15].float().mean().item()))

        ce1 = (F.cross_entropy(lg1.reshape(-1, e24.VOCAB_SIZE), by.reshape(-1), reduction="none").view_as(by) * lmask1).sum() / lmask1.sum()
        ce2 = (F.cross_entropy(lg2.reshape(-1, e24.VOCAB_SIZE), by.reshape(-1), reduction="none").view_as(by) * lmask2).sum() / lmask2.sum()
        loss = 0.45 * ce1 + 0.55 * ce2 if model.refiner is not None else ce1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if model.refiner is not None and st >= steps - 25:
            with torch.no_grad():
                _ = model.forward_pass(bx_c)

    model.eval()
    eval_rng = random.Random(80_000 + seed * 97)
    copy_cols = list(range(3, 16))
    with torch.no_grad():
        cx, cy, _ = e24.make_copy_batch(128, eval_rng)
        _, lg_copy, st_copy = model.forward_pass(cx, ood_pass2_noise=0.0)
        copy_acc = float(((lg_copy[:, copy_cols, 4:16].argmax(dim=-1) + 4) == cy[:, copy_cols]).float().mean().item())
        copy_waste = float(st_copy["hard_mask"][:, copy_cols].float().mean().item())

        torch.manual_seed(900 + seed)
        _, lg_shift, _ = model.forward_pass(cx, ood_pass2_noise=5.0)
        ood_copy_ret = float(((lg_shift[:, copy_cols, 4:16].argmax(dim=-1) + 4) == cy[:, copy_cols]).float().mean().item())

        _ = e24.make_8way_relational_batch(128, eval_rng, eval_uncorr=False)  # consume same RNG stream as run_e24
        hx_cf, hy_cf, _ = e24.make_8way_relational_batch(128, eval_rng, eval_uncorr=True)
        lg1_cf, lg_cf, st_cf = model.forward_pass(hx_cf, ood_pass2_noise=0.0)
        _, lg2_raw_cf = force_pass2_all_logits(model, hx_cf)

        p1_acc = float(((lg1_cf[:, 15, 28:36].argmax(-1) + 28) == hy_cf[:, 15]).float().mean().item())
        raw_p2_acc = float(((lg2_raw_cf[:, 15, 28:36].argmax(-1) + 28) == hy_cf[:, 15]).float().mean().item())

        if arm == "H6_hem_warmup_decoupled_eval" and model.refiner is not None:
            # In H6: inspect whether skipping the certainty shield ONLY when u_hem > min_ambiguity_floor
            # (meaning Pass-1 itself is overtly dispersed/uncertain!) removes false rejections
            u1_cf, _ = topk_renorm_entropy(lg1_cf, topk=8)
            routed_cf = u1_cf[:, 15] > max(float(model.refiner.ema_tau.item()), 0.46)
            final_preds = torch.where(routed_cf, lg2_raw_cf[:, 15, 28:36].argmax(-1) + 28, lg1_cf[:, 15, 28:36].argmax(-1) + 28)
            final_cf_acc = float((final_preds == hy_cf[:, 15]).float().mean().item())
            qry_routed = float(routed_cf.float().mean().item())
            qry_accepted = qry_routed
        else:
            final_cf_acc = float(((lg_cf[:, 15, 28:36].argmax(-1) + 28) == hy_cf[:, 15]).float().mean().item())
            qry_routed = float(st_cf["hard_mask"][:, 15].float().mean().item())
            qry_accepted = float(st_cf["accepted_frac"])

    return {
        "seed": seed,
        "params": float(sum(p.numel() for p in model.parameters())),
        "final_cf_acc": round(final_cf_acc, 4),
        "pass1_cf_acc": round(p1_acc, 4),
        "raw_pass2_cf_acc": round(raw_p2_acc, 4),
        "train_qry_routed_last50": round(sum(train_qry_routed_last50) / len(train_qry_routed_last50), 4),
        "eval_qry_routed": round(qry_routed, 4),
        "copy_waste": round(copy_waste, 4),
        "clean_copy_acc": round(copy_acc, 4),
        "ood_copy_ret": round(ood_copy_ret, 4),
    }


def main() -> int:
    torch.set_num_threads(2)
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Part A: Factorial E23 Attribution
    part_a_specs = [
        ("A0_c0_raw_sym0", "none", False, False),
        ("A1_c0_raw_symON", "none", True, False),
        ("A2_root_levels_sym0", "root_levels", False, False),
        ("A3_root_levels_symON", "root_levels", True, False),
        ("A4_frontier_sym0", "frontier_debruijn", False, False),
        ("A5_frontier_symON", "frontier_debruijn", True, False),
        ("A6_frontier_sym0_shield", "frontier_debruijn", False, True),
        ("A7_frontier_symON_shield", "frontier_debruijn", True, True),
        ("A8_classical_rule_0param", "classical", True, True),
    ]
    part_a_res: dict[str, Any] = {}
    for arm_name, mode, sym_on, sh_on in part_a_specs:
        runs = [run_part_a_arm(mode, sym_on, sh_on, s) for s in (0, 1, 2)]
        agg: dict[str, Any] = {"params": int(runs[0]["params"])}
        for k in ("id_verified", "deep_ood_verified", "deep_ood_value_custody", "deep_ood_contra_rev", "stress_verified", "stress_unsafe"):
            agg[k] = mean_std([r[k] for r in runs])
        part_a_res[arm_name] = agg
        print(
            f"[U03-A] {arm_name:26s} params={agg['params']:5d} ID={agg['id_verified']['mean']:.3f} "
            f"DeepOOD={agg['deep_ood_verified']['mean']:.3f}±{agg['deep_ood_verified']['std']:.3f} "
            f"Stress={agg['stress_verified']['mean']:.3f} Unsafe={agg['stress_unsafe']['mean']:.3f}",
            flush=True,
        )

    # Part B: E24 Routing Paradox Per-Seed Forensics + Reconciliation (5 seeds)
    part_b_arms = [
        "H0_pass1_unrouted",
        "H1_x5_full_vocab_shannon",
        "H2_x8_top2_margin_quantile",
        "H3_mm_atrd_top2_shield",
        "H4_hem_atrd_topk_shield",
        "H5_hem_train_warmup",
        "H6_hem_warmup_decoupled_eval",
    ]
    part_b_res: dict[str, Any] = {}
    for arm in part_b_arms:
        runs = [run_part_b_arm(arm, s) for s in (0, 1, 2, 3, 4)]
        agg: dict[str, Any] = {"params": int(runs[0]["params"]), "per_seed": runs}
        for k in ("final_cf_acc", "pass1_cf_acc", "raw_pass2_cf_acc", "train_qry_routed_last50", "eval_qry_routed", "copy_waste", "ood_copy_ret"):
            agg[k] = mean_std([r[k] for r in runs])
        part_b_res[arm] = agg
        print(
            f"[U03-B] {arm:30s} cf={agg['final_cf_acc']['mean']:.3f}±{agg['final_cf_acc']['std']:.3f} "
            f"rawP2={agg['raw_pass2_cf_acc']['mean']:.3f} tr_rt={agg['train_qry_routed_last50']['mean']:.3f} "
            f"ev_rt={agg['eval_qry_routed']['mean']:.3f} waste={agg['copy_waste']['mean']:.3f} "
            f"oodRet={agg['ood_copy_ret']['mean']:.3f}",
            flush=True,
        )

    elapsed = round(time.perf_counter() - t0, 2)
    payload = {
        "schema": "nanolm.unified_loop.u03.v1",
        "elapsed_s": elapsed,
        "paid_compute_usd": 0.0,
        "part_a_e23_attribution": part_a_res,
        "part_b_e24_routing_audit": part_b_res,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[U-03] Wrote {OUT_DIR / 'results.json'} in {elapsed}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
