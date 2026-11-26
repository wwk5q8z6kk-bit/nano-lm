"""Partially Observable Multi-Module Software State-Transition & Counterfactual Environment (`LAB-COG-01`).

Models core software-engineering cognition challenges in miniature with exact Bayesian ground truth:
  - Partially hidden dependency DAG `M_0 -> M_1 -> ... -> M_{N-1}` (`N=4..5` in train; `N=7..9` in OOD test).
  - Misleading surface symptoms: an alert in leaf `M_{N-1}` favors closer suspect `rival_mod` (`0.60` prior),
    but inspecting `rival_mod` falsifies it (`65%` of episodes) requiring belief revision to `true_root_mod`!
  - Intent ambiguity (`ambiguous_goal=True` on `30%` of episodes): spec lists two competing constants
    (`true_const` and `decoy_const`); patching before `ASK_CLARIFY` triggers an immediate regression!
  - Open-vocabulary exactness (`true_const` in `[100..999]` in train; **unseen `[1000..9999]`** in OOD test).
  - Verification sufficiency: `STOP_VERIFIED` succeeds ONLY if `RUN_TESTS` passed *after* the latest patch!
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any

ACT_INSPECT = 0
ACT_TRACE = 1
ACT_RUN_TESTS = 2
ACT_ASK_CLARIFY = 3
ACT_PATCH = 4
ACT_REVERT = 5
ACT_STOP = 6

ACTION_NAMES = ["INSPECT", "TRACE", "RUN_TESTS", "ASK_CLARIFY", "PATCH", "REVERT", "STOP_VERIFIED"]
TRAIN_MOD_NAMES = ["auth", "gateway", "cache", "ledger", "worker", "router"]
OOD_MOD_NAMES = ["chronos", "nebula", "vortex", "quasar", "pulsar", "aegis", "sentinel", "hydra", "zenith"]
EXTREME_OOD_MOD_NAMES = OOD_MOD_NAMES + ["titan", "eclipse", "hyperion"]


@dataclass
class RepoEpisode:
    n_modules: int
    mod_names: list[str]
    symptom_mod: int
    true_root_mod: int
    rival_mod: int
    ambiguous_goal: bool
    true_const: int
    decoy_const: int
    ood: bool
    clarified: bool = False
    inspected: set[int] = field(default_factory=set)
    traced: bool = False
    patched_mod: int | None = None
    patched_val: int | None = None
    has_regression: bool = False
    verified_after_patch: bool = False
    contradiction_seen: bool = False
    hidden_hop: bool = False
    steps: int = 0


def sample_episode(rng: random.Random, ood: bool = False, extreme_ood: bool = False, hidden_hop: bool = False) -> RepoEpisode:
    if extreme_ood:
        n_mod = rng.choice([10, 11, 12])
        names = rng.sample(EXTREME_OOD_MOD_NAMES, n_mod)
        c_true = rng.randint(1000, 9999)
        c_decoy = rng.randint(1000, 9999)
        while c_decoy == c_true:
            c_decoy = rng.randint(1000, 9999)
        ood = True
    elif not ood:
        n_mod = rng.choice([5, 6] if hidden_hop else [4, 5])
        names = rng.sample(TRAIN_MOD_NAMES, n_mod)
        c_true = rng.randint(100, 999)
        c_decoy = rng.randint(100, 999)
        while c_decoy == c_true:
            c_decoy = rng.randint(100, 999)
    else:
        n_mod = rng.choice([7, 8, 9])
        names = rng.sample(OOD_MOD_NAMES, n_mod)
        c_true = rng.randint(1000, 9999)
        c_decoy = rng.randint(1000, 9999)
        while c_decoy == c_true:
            c_decoy = rng.randint(1000, 9999)

    symptom = n_mod - 1
    if hidden_hop:
        # 2-Hop Hidden Root Cause: Turn-0 GOAL only knows immediate caller `rival_mod`;
        # `true_root_mod` lies deeper upstream in `0 .. rival_mod - 2` and is ONLY revealed upon inspecting `rival_mod`!
        rival = max(2, symptom - 1)
        true_root = rng.randint(0, max(0, rival - 2))
    else:
        cand_a = max(0, symptom - 2)
        cand_b = max(0, symptom - 1) if symptom >= 1 else 0
        if cand_a == cand_b:
            cand_a, cand_b = 0, min(1, n_mod - 1)
        true_root, rival = (cand_a, cand_b) if rng.random() < 0.65 else (cand_b, cand_a)
    ambig = rng.random() < 0.30
    return RepoEpisode(
        n_modules=n_mod, mod_names=names, symptom_mod=symptom,
        true_root_mod=true_root, rival_mod=rival, ambiguous_goal=ambig,
        true_const=c_true, decoy_const=c_decoy, ood=ood, hidden_hop=hidden_hop,
    )


def initial_goal_observation(ep: RepoEpisode) -> str:
    if ep.ambiguous_goal:
        # Order randomized so position heuristic cannot guess which constant is active without ASK_CLARIFY!
        pair = [ep.true_const, ep.decoy_const] if (ep.true_const + ep.symptom_mod) % 2 == 0 else [ep.decoy_const, ep.true_const]
        amb_clause = f"AMBIGUOUS_SPEC candidates VAL_{pair[0]} or VAL_{pair[1]}"
    else:
        amb_clause = f"CLEAR_SPEC required VAL_{ep.true_const}"
    mods_str = " ".join(f"M{i}:{n}" for i, n in enumerate(ep.mod_names))
    if ep.hidden_hop:
        # True root cause is completely hidden upstream (not listed in GOAL suspects)!
        return f"GOAL alert M{ep.symptom_mod} suspects M{ep.rival_mod} {amb_clause} REPO {mods_str}"
    # Order suspects by index so token order does NOT leak which suspect is true_root vs rival!
    s_lo, s_hi = sorted([ep.rival_mod, ep.true_root_mod])
    return f"GOAL alert M{ep.symptom_mod} suspects M{s_lo} M{s_hi} {amb_clause} REPO {mods_str}"


def compute_exact_belief_and_uncertainty(ep: RepoEpisode, max_mods: int = 9) -> dict[str, Any]:
    probs = [0.0] * max(max_mods, ep.n_modules)
    if ep.true_root_mod in ep.inspected:
        probs[ep.true_root_mod] = 1.0
    elif ep.rival_mod in ep.inspected:
        probs[ep.true_root_mod] = 0.95
        probs[ep.rival_mod] = 0.05
    else:
        probs[ep.rival_mod] = 0.60
        probs[ep.true_root_mod] = 0.40

    ent = -sum(p * math.log(p + 1e-9) for p in probs if p > 0)
    u_intent = 1.0 if (ep.ambiguous_goal and not ep.clarified) else 0.0
    u_state = min(1.0, ent / math.log(2.0))
    clean_fix = (
        ep.patched_mod == ep.true_root_mod
        and ep.patched_val == ep.true_const
        and not ep.has_regression
        and (not ep.ambiguous_goal or ep.clarified)
    )
    u_verify = 0.0 if (clean_fix and ep.verified_after_patch) else 1.0
    return {
        "posterior": probs,
        "u_intent": u_intent,
        "u_state": round(u_state, 4),
        "u_verify": u_verify,
        "is_cleanly_fixed": clean_fix,
    }


def step_environment(ep: RepoEpisode, action: int, target_mod: int, chosen_val: int) -> tuple[str, dict[str, Any]]:
    ep.steps += 1
    b_before = compute_exact_belief_and_uncertainty(ep)
    target_mod = max(0, min(ep.n_modules - 1, target_mod))
    premature_error = False
    caused_regression = False
    episode_done = False
    solved_verified = False

    if action == ACT_ASK_CLARIFY:
        if ep.ambiguous_goal and not ep.clarified:
            ep.clarified = True
            obs = f"OBS CLARIFIED active contract is VAL_{ep.true_const}"
        else:
            obs = f"OBS REDUNDANT_CLARIFY contract already VAL_{ep.true_const}"
    elif action == ACT_TRACE:
        ep.traced = True
        s_lo, s_hi = sorted([ep.rival_mod, ep.true_root_mod])
        obs = f"OBS TRACE M{ep.symptom_mod} upstream M{s_lo} M{s_hi}"
    elif action == ACT_INSPECT:
        ep.inspected.add(target_mod)
        if target_mod == ep.true_root_mod:
            obs = f"OBS INSPECT M{target_mod} FAULT_FOUND needs VAL_{ep.true_const}"
        elif target_mod == ep.rival_mod:
            ep.contradiction_seen = True
            other = ep.true_root_mod
            obs = f"OBS INSPECT M{target_mod} HEALTHY_MODULE fault must be M{other}"
        else:
            obs = f"OBS INSPECT M{target_mod} UNRELATED_MODULE"
    elif action == ACT_PATCH:
        ep.verified_after_patch = False
        if ep.ambiguous_goal and not ep.clarified:
            premature_error = True
            caused_regression = True
            ep.has_regression = True
            ep.patched_mod, ep.patched_val = target_mod, chosen_val
            obs = f"OBS PATCH M{target_mod} REGRESSION_AMBIGUOUS"
        elif target_mod != ep.true_root_mod or chosen_val != ep.true_const:
            premature_error = (ep.true_root_mod not in ep.inspected and ep.rival_mod not in ep.inspected)
            caused_regression = True
            ep.has_regression = True
            ep.patched_mod, ep.patched_val = target_mod, chosen_val
            obs = f"OBS PATCH M{target_mod} REGRESSION_WRONG_TARGET_OR_VAL"
        else:
            ep.patched_mod, ep.patched_val = target_mod, chosen_val
            ep.has_regression = False
            obs = f"OBS PATCH M{target_mod} APPLIED_CLEAN_NEEDS_TEST"
    elif action == ACT_REVERT:
        ep.patched_mod = ep.patched_val = None
        ep.has_regression = False
        ep.verified_after_patch = False
        obs = "OBS REVERT CLEAN_STATE"
    elif action == ACT_RUN_TESTS:
        if ep.has_regression:
            obs = f"OBS TESTS FAIL_REGRESSION on M{ep.patched_mod}"
        elif ep.patched_mod == ep.true_root_mod and ep.patched_val == ep.true_const and (not ep.ambiguous_goal or ep.clarified):
            ep.verified_after_patch = True
            obs = f"OBS TESTS ALL_PASS verified M{ep.true_root_mod} VAL_{ep.true_const}"
        else:
            obs = "OBS TESTS FAIL_UNFIXED"
    elif action == ACT_STOP:
        episode_done = True
        b_now = compute_exact_belief_and_uncertainty(ep)
        if b_now["is_cleanly_fixed"] and ep.verified_after_patch:
            solved_verified = True
            obs = "OBS STOP VERIFIED_SUCCESS"
        else:
            premature_error = True
            obs = "OBS STOP FAILED_UNVERIFIED"
    else:
        obs = "OBS NOOP"

    b_after = compute_exact_belief_and_uncertainty(ep)
    ig = max(0.0, (b_before["u_intent"] + b_before["u_state"] + b_before["u_verify"]) - (b_after["u_intent"] + b_after["u_state"] + b_after["u_verify"]))
    return obs, {
        "done": episode_done,
        "solved_verified": solved_verified,
        "premature_error": premature_error,
        "caused_regression": caused_regression,
        "info_gain": round(ig, 4),
        "belief_after": b_after,
    }


def expert_policy(ep: RepoEpisode) -> tuple[int, int, int]:
    if ep.has_regression:
        return (ACT_REVERT, 0, ep.true_const)
    if ep.ambiguous_goal and not ep.clarified:
        return (ACT_ASK_CLARIFY, 0, ep.true_const)
    if ep.true_root_mod not in ep.inspected and ep.rival_mod not in ep.inspected:
        # In hidden_hop mode, only `rival_mod` is known at Turn 0 (true_root_mod is undiscovered upstream!)
        first_probe = ep.rival_mod if ep.hidden_hop else min(ep.rival_mod, ep.true_root_mod)
        return (ACT_INSPECT, first_probe, ep.true_const)
    if ep.true_root_mod not in ep.inspected:
        return (ACT_INSPECT, ep.true_root_mod, ep.true_const)
    if ep.patched_mod != ep.true_root_mod or ep.patched_val != ep.true_const:
        return (ACT_PATCH, ep.true_root_mod, ep.true_const)
    if not ep.verified_after_patch:
        return (ACT_RUN_TESTS, ep.true_root_mod, ep.true_const)
    return (ACT_STOP, ep.true_root_mod, ep.true_const)


def counterfactual_action_outcomes(ep: RepoEpisode) -> list[dict[str, float]]:
    """Simulate all 7 actions counterfactually from state `ep` without mutating `ep`."""
    opt_a, opt_m, _ = expert_policy(ep)
    outcomes: list[dict[str, float]] = []
    for a in range(len(ACTION_NAMES)):
        sim = copy.deepcopy(ep)
        cand_m = opt_m if a == opt_a else (min(sim.rival_mod, sim.true_root_mod) if a in (ACT_INSPECT, ACT_PATCH) else 0)
        cand_v = sim.true_const if (not sim.ambiguous_goal or sim.clarified) else sim.decoy_const
        _, info = step_environment(sim, a, cand_m, cand_v)
        outcomes.append({
            "info_gain": float(info["info_gain"]),
            "reg_risk": 1.0 if (info["caused_regression"] or info["premature_error"]) else 0.0,
            "is_optimal": 1.0 if a == opt_a else 0.0,
        })
    return outcomes
