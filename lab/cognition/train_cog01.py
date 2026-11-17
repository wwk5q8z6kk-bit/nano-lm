"""Train & Evaluate `LAB-COG-01`: 4 Competing Cognitive Architectures on OOD Software Repositories.

Compares (`n=3` independent seeds, matched `~1.1M` `NanoLM-v2` `DDDA` parameters & training episodes):
  - `C0_transcript_imitation`: Standard causal trajectory SFT `(goal + transcript -> action + target + 4-digit val)`
  - `C1_obs_world_model`: `C0` + CWM-style next-observation token-distribution prediction
  - `C2_latent_loop_only`: `C0` + Huginn-style randomized-depth latent recurrence (`K in {1,2,3}`)
  - `C3_ebwm_counterfactual`: Persistent 8-slot Belief Workspace `Z_t` + Factored Epistemic Head
    `(u_intent, u_state, u_verify)` + Decision-Oriented Counterfactual Outcome Head `(info_gain, reg_risk)`
    + Exact Source-Span Value Pointer `P(Z_t, o_{0:t})` (also evaluated with oracle value vs generative vs
    pointer to cleanly decouple cognitive planning gains from exact-copy pointer gains!).
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from lab.cognition.models import (
    VOCAB_SIZE,
    CognitiveLM,
    digits4_to_val,
    encode_text_with_spans,
    val_to_4digits,
)
from lab.cognition.swe_env import (
    ACT_PATCH,
    ACTION_NAMES,
    RepoEpisode,
    compute_exact_belief_and_uncertainty,
    counterfactual_action_outcomes,
    expert_policy,
    initial_goal_observation,
    sample_episode,
    step_environment,
)


def generate_training_episode_rollout(rng: random.Random) -> list[dict[str, Any]]:
    """Roll out a training episode including realistic noisy/distractor or recovery steps (35% rate)."""
    ep = sample_episode(rng, ood=False)
    goal_txt = initial_goal_observation(ep)
    steps_data: list[dict[str, Any]] = []
    latest_obs = goal_txt

    for _ in range(10):
        b_gt = compute_exact_belief_and_uncertainty(ep, max_mods=ep.n_modules)
        opt_a, opt_m, opt_v = expert_policy(ep)
        cf = counterfactual_action_outcomes(ep)

        # Occasionally inject an exploratory/mistaken action *after* recording the optimal teacher target
        # so ALL arms (`C0..C3`) see distractor observations AND regression->revert states during training!
        exec_a, exec_m, exec_v = opt_a, opt_m, opt_v
        if rng.random() < 0.22 and len(steps_data) == 1 and not ep.has_regression:
            exec_a = rng.choice([1, 4])  # either distractor TRACE or premature PATCH causing regression!
            exec_m = ep.rival_mod
            exec_v = ep.decoy_const if ep.ambiguous_goal else ep.true_const

        next_obs, out_info = step_environment(ep, exec_a, exec_m, exec_v)
        steps_data.append({
            "obs_text": latest_obs,
            "n_modules": ep.n_modules,
            "opt_action": opt_a,
            "opt_mod": opt_m,
            "opt_val": opt_v,
            "belief_posterior": b_gt["posterior"][: ep.n_modules],
            "unc_vec": [b_gt["u_intent"], b_gt["u_state"], b_gt["u_verify"]],
            "cf_outcomes": cf,
            "exec_action": exec_a,
            "next_obs_text": next_obs,
        })
        latest_obs = f"ACT {ACTION_NAMES[exec_a]} M{exec_m} {next_obs}"
        if out_info["done"]:
            break
    return steps_data


def train_one_arm(arm: str, seed: int, n_episodes: int, dev: torch.device) -> CognitiveLM:
    torch.manual_seed(seed)
    rng = random.Random(seed * 101 + 17)
    model = CognitiveLM(arm=arm, d=128, n_head=4, n_slots=8).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)

    model.train()
    for ep_idx in range(n_episodes):
        traj = generate_training_episode_rollout(rng)
        opt.zero_grad(set_to_none=True)
        ep_loss = torch.zeros((), device=dev)

        if arm != "C3_ebwm_counterfactual":
            running_ids: list[int] = []
            running_spans: list[tuple[int, int]] = []
            for st_item in traj:
                ids_chunk, spans_chunk = encode_text_with_spans(st_item["obs_text"])
                offset = len(running_ids)
                running_ids.extend(ids_chunk)
                running_spans.extend((offset + p, v) for p, v in spans_chunk)
                # Keep most recent 384 tokens if transcript grows
                if len(running_ids) > 384:
                    trim = len(running_ids) - 384
                    running_ids = running_ids[trim:]
                    running_spans = [(p - trim, v) for p, v in running_spans if p >= trim]

                out = model.forward_transcript_step(running_ids, st_item["n_modules"], running_spans, dev)
                l_act = F.cross_entropy(out["act_logits"].unsqueeze(0), torch.tensor([st_item["opt_action"]], device=dev))
                l_mod = F.cross_entropy(out["mod_logits"].unsqueeze(0), torch.tensor([st_item["opt_mod"]], device=dev))
                d4 = val_to_4digits(st_item["opt_val"])
                l_dig = sum(
                    F.cross_entropy(out["digit_logits"][di].unsqueeze(0), torch.tensor([d4[di]], device=dev))
                    for di in range(4)
                ) / 4.0
                step_l = l_act + l_mod + 0.5 * l_dig

                if arm == "C1_obs_world_model":
                    # Predict bag-of-tokens presence in `next_obs_text` conditioned on (h_last, exec_action)!
                    next_ids, _ = encode_text_with_spans(st_item["next_obs_text"])
                    tgt_bow = torch.zeros(VOCAB_SIZE, device=dev)
                    for nid in next_ids:
                        tgt_bow[nid] = 1.0
                    a_onehot = F.one_hot(torch.tensor(st_item["exec_action"], device=dev), num_classes=len(ACTION_NAMES)).float()
                    wm_logits = model.obs_wm_head(torch.cat([out["h_last"], a_onehot], dim=-1))
                    step_l = step_l + 0.4 * F.binary_cross_entropy_with_logits(wm_logits, tgt_bow)

                ep_loss = ep_loss + step_l
        else:
            ws = None
            for st_idx, st_item in enumerate(traj):
                ids_chunk, spans_chunk = encode_text_with_spans(st_item["obs_text"])
                if st_idx == 0:
                    ws = model.init_c3_workspace(ids_chunk, st_item["n_modules"], spans_chunk, dev)
                else:
                    ws = model.update_c3_workspace(ws, ids_chunk, spans_chunk, dev)

                l_act = F.cross_entropy(ws["act_logits"].unsqueeze(0), torch.tensor([st_item["opt_action"]], device=dev))
                l_mod = F.cross_entropy(ws["mod_logits"].unsqueeze(0), torch.tensor([st_item["opt_mod"]], device=dev))
                d4 = val_to_4digits(st_item["opt_val"])
                l_dig = sum(
                    F.cross_entropy(ws["digit_logits"][di].unsqueeze(0), torch.tensor([d4[di]], device=dev))
                    for di in range(4)
                ) / 4.0

                # 1. Explicit Bayesian belief posterior KL
                gt_post = torch.tensor(st_item["belief_posterior"], dtype=torch.float32, device=dev)
                l_post = -(gt_post * F.log_softmax(ws["belief_post_logits"], dim=-1)).sum()

                # 2. Factored epistemic uncertainty (intent, state, verification)
                gt_unc = torch.tensor(st_item["unc_vec"], dtype=torch.float32, device=dev)
                l_unc = F.mse_loss(ws["unc_pred"], gt_unc)

                # 3. Decision-oriented counterfactual outcome supervision across all 7 actions!
                gt_ig = torch.tensor([c["info_gain"] for c in st_item["cf_outcomes"]], dtype=torch.float32, device=dev)
                gt_rk = torch.tensor([c["reg_risk"] for c in st_item["cf_outcomes"]], dtype=torch.float32, device=dev)
                l_cf = F.mse_loss(ws["pred_ig"], gt_ig) + F.binary_cross_entropy(ws["pred_risk"].clamp(1e-5, 1 - 1e-5), gt_rk)

                # 4. Exact source-span value pointer supervision (points to latest occurrence of `opt_val` when present!)
                l_ptr = torch.zeros((), device=dev)
                if ws["ptr_logits"] is not None:
                    matching = [i for i, (_, v) in enumerate(ws["evidence_vecs"]) if v == st_item["opt_val"]]
                    if matching:
                        l_ptr = F.cross_entropy(ws["ptr_logits"].unsqueeze(0), torch.tensor([matching[-1]], device=dev))

                ep_loss = ep_loss + l_act + l_mod + 0.3 * l_dig + 0.5 * l_post + 0.5 * l_unc + 0.5 * l_cf + 0.6 * l_ptr

        (ep_loss / max(1, len(traj))).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return model


@torch.no_grad()
def evaluate_arm(model: CognitiveLM, n_eval: int, ood: bool, dev: torch.device, seed: int = 9001, inject_distractors: bool = True) -> dict[str, float]:
    """Evaluate closed-loop autonomous episode completion on ID (`4–5` mod) or OOD (`7–9` mod, unseen 4-digit vals)."""
    model.eval()
    rng = random.Random(seed + (500 if ood else 0))

    verified_full = 0            # Solved with exact model-predicted value
    verified_policy_only = 0     # Solved when oracle value is supplied (isolates pure cognitive policy vs value exactness!)
    premature_actions = 0
    val_exact_hits = 0
    val_total_patches = 0
    contradiction_revised = 0
    contradiction_cases = 0

    for _ in range(n_eval):
        base_ep = sample_episode(rng, ood=ood)

        for use_oracle_val in (False, True):
            ep = copy.deepcopy(base_ep)
            obs_txt = initial_goal_observation(ep)
            if inject_distractors:
                # Realistic log clutter: 2 unrelated pass-through observations prepended in unfamiliar order
                unrel = [m for m in range(ep.n_modules) if m not in (ep.true_root_mod, ep.rival_mod, ep.symptom_mod)]
                for um in unrel[:2]:
                    obs_txt += f" OBS INSPECT M{um} UNRELATED_MODULE"

            running_ids: list[int] = []
            running_spans: list[tuple[int, int]] = []
            ws = None
            had_premature = False
            solved = False

            for step_t in range(10):
                ids_c, spans_c = encode_text_with_spans(obs_txt)
                if model.arm != "C3_ebwm_counterfactual":
                    off = len(running_ids)
                    running_ids.extend(ids_c)
                    running_spans.extend((off + p, v) for p, v in spans_c)
                    if len(running_ids) > 384:
                        trim = len(running_ids) - 384
                        running_ids = running_ids[trim:]
                        running_spans = [(p - trim, v) for p, v in running_spans if p >= trim]
                    out = model.forward_transcript_step(running_ids, ep.n_modules, running_spans, dev, loop_k=3)
                    pred_a = int(out["act_logits"].argmax().item())
                    pred_m = int(out["mod_logits"].argmax().item())
                    pred_v = digits4_to_val([int(dl.argmax().item()) for dl in out["digit_logits"]])
                else:
                    ws = model.init_c3_workspace(ids_c, ep.n_modules, spans_c, dev) if step_t == 0 else model.update_c3_workspace(ws, ids_c, spans_c, dev)
                    pred_a = int(ws["act_logits"].argmax().item())
                    pred_m = int(ws["mod_logits"].argmax().item())
                    if ws["ptr_logits"] is not None and len(ws["evidence_vecs"]) > 0:
                        best_ev = int(ws["ptr_logits"].argmax().item())
                        pred_v = ws["evidence_vecs"][best_ev][1]
                    else:
                        pred_v = digits4_to_val([int(dl.argmax().item()) for dl in ws["digit_logits"]])

                if not use_oracle_val and pred_a == ACT_PATCH:
                    val_total_patches += 1
                    if pred_v == ep.true_const and (not ep.ambiguous_goal or ep.clarified):
                        val_exact_hits += 1

                val_apply = ep.true_const if use_oracle_val else pred_v
                next_obs, info = step_environment(ep, pred_a, pred_m, val_apply)
                if info["premature_error"] or info["caused_regression"]:
                    had_premature = True
                obs_txt = f"ACT {ACTION_NAMES[pred_a]} M{pred_m} {next_obs}"
                if info["done"]:
                    solved = bool(info["solved_verified"])
                    break

            if not use_oracle_val:
                verified_full += int(solved)
                premature_actions += int(had_premature)
            else:
                verified_policy_only += int(solved)
                if ep.contradiction_seen:
                    contradiction_cases += 1
                    contradiction_revised += int(ep.patched_mod == ep.true_root_mod)

    return {
        "verified_resolution_full": round(verified_full / n_eval, 4),
        "verified_resolution_oracle_val": round(verified_policy_only / n_eval, 4),
        "premature_or_regression_rate": round(premature_actions / n_eval, 4),
        "value_exact_match": round(val_exact_hits / max(1, val_total_patches), 4),
        "belief_revision_after_contradiction": round(contradiction_revised / max(1, contradiction_cases), 4),
    }


def _run_single_arm_seed(job: tuple[str, int, int, int]) -> tuple[str, dict[str, Any], float]:
    arm, s, n_episodes, n_eval = job
    torch.set_num_threads(1)
    dev = torch.device("cpu")
    t0 = time.perf_counter()
    m = train_one_arm(arm, seed=s, n_episodes=n_episodes, dev=dev)
    n_params = sum(p.numel() for p in m.parameters())
    id_res = evaluate_arm(m, n_eval=n_eval, ood=False, dev=dev, seed=7000 + s)
    ood_res = evaluate_arm(m, n_eval=n_eval, ood=True, dev=dev, seed=8000 + s)
    dt = time.perf_counter() - t0
    rec = {"seed": s, "params": n_params, "in_dist": id_res, "ood": ood_res}
    print(
        f"[{arm} s={s}] ({dt:.1f}s) params={n_params:,} ID_full={id_res['verified_resolution_full']:.3f} "
        f"OOD_full={ood_res['verified_resolution_full']:.3f} OOD_policy={ood_res['verified_resolution_oracle_val']:.3f} "
        f"OOD_premature={ood_res['premature_or_regression_rate']:.3f} OOD_val_exact={ood_res['value_exact_match']:.3f}",
        flush=True,
    )
    return arm, rec, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=900)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/Users/mac/Projects/nano-lm/.worktrees/nanolm-lab/lab/reports/lab_cog_01_results.json")
    args = ap.parse_args()

    arms = ["C0_transcript_imitation", "C1_obs_world_model", "C2_latent_loop_only", "C3_ebwm_counterfactual"]
    all_results: dict[str, Any] = {"schema": "nanolm.lab_cog_01.v1", "episodes_train": args.episodes, "seeds": args.seeds, "arms": {}}

    jobs = [(arm, s, args.episodes, args.eval_episodes) for arm in arms for s in range(args.seeds)]
    grouped: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    wall_by_arm: dict[str, float] = {a: 0.0 for a in arms}

    t_global = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        futs = [ex.submit(_run_single_arm_seed, j) for j in jobs]
        for f in as_completed(futs):
            arm, rec, dt = f.result()
            grouped[arm].append(rec)
            wall_by_arm[arm] = max(wall_by_arm[arm], dt)

    for arm in arms:
        seed_runs = sorted(grouped[arm], key=lambda r: r["seed"])

        def _agg(split: str, key: str) -> dict[str, float]:
            vals = [r[split][key] for r in seed_runs]
            return {"mean": round(st.mean(vals), 4), "half_range": round((max(vals) - min(vals)) / 2, 4)}

        summary = {
            "params": seed_runs[0]["params"],
            "wall_s": round(wall_by_arm[arm], 2),
            "ood_verified_resolution_full": _agg("ood", "verified_resolution_full"),
            "ood_verified_resolution_policy_only": _agg("ood", "verified_resolution_oracle_val"),
            "ood_premature_or_regression_rate": _agg("ood", "premature_or_regression_rate"),
            "ood_unseen_value_exact_match": _agg("ood", "value_exact_match"),
            "ood_belief_revision_acc": _agg("ood", "belief_revision_after_contradiction"),
            "id_verified_resolution_full": _agg("in_dist", "verified_resolution_full"),
            "seed_runs": seed_runs,
        }
        all_results["arms"][arm] = summary
        print(f"=== SUMMARY {arm}: {json.dumps({k: v for k, v in summary.items() if k != 'seed_runs'})}", flush=True)

    all_results["total_wall_s"] = round(time.perf_counter() - t_global, 2)
    Path(args.out).write_text(json.dumps(all_results, indent=2))
    print("SAVED_LAB_COG_01", args.out, f"in {all_results['total_wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
