"""Unit tests for `LAB-COG-01` (`swe_env.py` & `models.py`)."""

from __future__ import annotations

import random
import torch

from lab.cognition.models import CognitiveLM, encode_text_with_spans
from lab.cognition.swe_env import (
    expert_policy,
    initial_goal_observation,
    sample_episode,
    step_environment,
)


def test_expert_policy_solves_all_id_and_ood_episodes_cleanly():
    rng = random.Random(123)
    for ood in (False, True):
        for _ in range(40):
            ep = sample_episode(rng, ood=ood)
            solved = False
            for _ in range(8):
                a, m, v = expert_policy(ep)
                _, info = step_environment(ep, a, m, v)
                assert not info["premature_error"], "Expert policy should never take premature unclarified actions!"
                if info["done"]:
                    solved = info["solved_verified"]
                    break
            assert solved, f"Expert policy failed episode: {ep}"


def test_all_four_cognitive_arms_forward_on_ood_9module_repo():
    rng = random.Random(7)
    ep = sample_episode(rng, ood=True)
    ep.n_modules = 9
    ep.mod_names = ["chronos", "nebula", "vortex", "quasar", "pulsar", "aegis", "sentinel", "hydra", "zenith"]
    txt = initial_goal_observation(ep)
    ids, spans = encode_text_with_spans(txt)
    assert len(spans) >= 1

    dev = torch.device("cpu")
    for arm in (
        "C0_transcript_imitation",
        "C1_obs_world_model",
        "C2_latent_loop_only",
        "C3_ebwm_counterfactual",
        "C5_de_bruijn_canonical",
        "C6_causal_suffix_canonical",
    ):
        m = CognitiveLM(arm=arm, d=64, n_head=4, n_slots=6).to(dev)
        if not m.is_c3_family:
            out = m.forward_transcript_step(ids, 9, spans, dev)
            assert out["mod_logits"].shape == (9,)
        else:
            ws = m.init_c3_workspace(ids, 9, spans, dev)
            assert ws["mod_logits"].shape == (9,)
            assert ws["unc_pred"].shape == (3,)

    # Verify exact iso-parameter match across C5, C6, and C7 (1,018,648 params)!
    p_c5 = sum(p.numel() for p in CognitiveLM("C5_de_bruijn_canonical", d=128, n_head=4, n_slots=8).parameters())
    p_c6 = sum(p.numel() for p in CognitiveLM("C6_causal_suffix_canonical", d=128, n_head=4, n_slots=8).parameters())
    p_c7 = sum(p.numel() for p in CognitiveLM("C7_dynamic_register_canonical", d=128, n_head=4, n_slots=8).parameters())
    assert p_c5 == p_c6 == p_c7, f"Iso-parameter mismatch: C5={p_c5}, C6={p_c6}, C7={p_c7}"


def test_hidden_hop_expert_and_dynamic_register_allocation():
    rng = random.Random(42)
    for ext in (False, True):
        for _ in range(25):
            ep = sample_episode(rng, ood=True, extreme_ood=ext, hidden_hop=True)
            goal = initial_goal_observation(ep)
            assert f"suspects M{ep.true_root_mod} " not in goal
            solved = False
            for _ in range(8):
                a, m, v = expert_policy(ep)
                _, info = step_environment(ep, a, m, v)
                if info["done"]:
                    solved = info["solved_verified"]
                    break
            assert solved
