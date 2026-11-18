"""OMEGA-EXP-02: Open-Entity Span-Graph Belief & Learned Expected Free-Energy (`EFE`) Planning.

Challenges `C4b/C5/C6` (`lab/cognition/models.py`) by removing:
  1. The hardcoded `12`-module `MOD_TOKEN_IDS` ceiling (evaluating Super-OOD `N = 16..24` repos),
  2. The fixed `symptom = N-1 / cand = N-2` first-mention layout (using arbitrary multi-hop DAGs
     with shuffled distractor-first goal prompts), and
  3. The hardcoded `suff_pen[4]` / `suff_pen[6]` action-index shields (learning Expected Free-Energy
     pragmatic value + epistemic info-gain - regression risk end-to-end).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import statistics
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ACT_INSPECT = 0
ACT_TRACE = 1
ACT_RUN_TESTS = 2
ACT_ASK_CLARIFY = 3
ACT_PATCH = 4
ACT_REVERT = 5
ACT_STOP = 6

ACTION_NAMES = ("INSPECT", "TRACE", "RUN_TESTS", "ASK_CLARIFY", "PATCH", "REVERT", "STOP_VERIFIED")

TRAIN_NAMES = ("auth", "gateway", "cache", "ledger", "worker", "router", "session", "billing")
OOD_NAMES = (
    "chronos", "nebula", "vortex", "quasar", "pulsar", "aegis", "sentinel", "hydra",
    "zenith", "helios", "orion", "lyra", "draco", "vega", "altair", "sirius",
)
SUPER_OOD_NAMES = (
    "hyperion", "callisto", "ganymede", "europa", "oberon", "titania", "miranda", "triton",
    "proteus", "nereid", "charon", "nix", "kerberos", "styx", "phobos", "deimos",
    "ceres", "pallas", "vesta", "hygiea", "eris", "makemake", "haumea", "sedna",
    "quaoar", "orcus", "salacia", "varda",
)

# Compact token + character vocabulary (only 12 fixed slot IDs SLOT_0..SLOT_11 exist for legacy baselines!)
BASE_VOCAB = [
    "<PAD>", "<SEP>", "GOAL", "OBS", "REPO", "ALERT", "EDGES", "ARROW", "AMBIGUOUS", "CLEAR",
    "OR", "CLARIFIED", "TRUE_SPEC", "TRACED", "INSPECTED", "FAULT_AT", "HEALTHY", "UNRELATED",
    "PATCHED", "REGRESSION_AMBIG", "REGRESSION_WRONG", "NEEDS_TEST", "REVERTED", "TESTS_PASS",
    "TESTS_FAIL", "VERIFIED", "FAILED", "VAL",
] + list(ACTION_NAMES) + [f"SLOT_{i}" for i in range(12)] + [f"D{d}" for d in range(10)] + [
    f"C_{c}" for c in "abcdefghijklmnopqrstuvwxyz_"
]
TOK_INDEX = {t: i for i, t in enumerate(BASE_VOCAB)}
SLOT_TOKEN_IDS = [TOK_INDEX[f"SLOT_{i}"] for i in range(12)]


@dataclass
class HardRepoEpisode:
    n_modules: int
    mod_names: list[str]
    symptom_mod: int
    mid_mod: int
    rival_mod: int
    true_root_mod: int
    edges: list[tuple[int, int]]  # directed (upstream -> downstream)
    ambiguous_goal: bool
    true_const: int
    decoy_const: int
    regime: str
    clarified: bool = False
    traced_mid: bool = False
    inspected: set[int] = field(default_factory=set)
    discovered_edges: list[tuple[int, int]] = field(default_factory=list)
    patched_mod: int | None = None
    patched_val: int | None = None
    has_regression: bool = False
    verified_after_patch: bool = False
    contradiction_seen: bool = False
    steps: int = 0


class HardCausalSWEEnv:
    """Multi-hop arbitrary-DAG software diagnosis environment scaling from N=4..6 to N=16..24."""

    MAX_STEPS = 10

    @staticmethod
    def sample_episode(rng: random.Random, regime: str = "id") -> HardRepoEpisode:
        if regime == "id":
            n_mod = rng.choice([4, 5, 6])
            names = rng.sample(TRAIN_NAMES, n_mod)
            c_true = rng.randint(100, 999)
            c_decoy = rng.randint(100, 999)
        elif regime == "ood":
            n_mod = rng.choice([9, 10, 12])
            names = rng.sample(OOD_NAMES, n_mod)
            c_true = rng.randint(1000, 9999)
            c_decoy = rng.randint(1000, 9999)
        elif regime == "super_ood":
            n_mod = rng.choice([16, 20, 24])
            names = rng.sample(SUPER_OOD_NAMES, n_mod)
            c_true = rng.randint(1000, 9999)
            c_decoy = rng.randint(1000, 9999)
        else:
            raise ValueError(f"Unknown regime: {regime}")

        while c_decoy == c_true:
            c_decoy = rng.randint(1000, 9999) if regime != "id" else rng.randint(100, 999)

        # Pick 4 distinct random indices anywhere in [0..n_mod-1] so distractors appear first in mention order!
        chain = rng.sample(range(n_mod), 4)
        symptom, rival, mid, hidden_root = chain[0], chain[1], chain[2], chain[3]
        rival_is_false = rng.random() < 0.65
        true_root = hidden_root if rival_is_false else rival

        visible_edges = [(rival, symptom), (mid, symptom)]
        all_edges = list(visible_edges) + [(hidden_root, mid)]
        others = [m for m in range(n_mod) if m not in (symptom, rival, mid, hidden_root)]
        for idx_o in range(len(others) - 1):
            if rng.random() < 0.45:
                visible_edges.insert(0, (others[idx_o], others[idx_o + 1]))
                all_edges.append((others[idx_o], others[idx_o + 1]))

        return HardRepoEpisode(
            n_modules=n_mod,
            mod_names=names,
            symptom_mod=symptom,
            mid_mod=mid,
            rival_mod=rival,
            true_root_mod=true_root,
            edges=all_edges,
            ambiguous_goal=(rng.random() < 0.30),
            true_const=c_true,
            decoy_const=c_decoy,
            regime=regime,
            discovered_edges=list(visible_edges),
        )

    @staticmethod
    def initial_observation(ep: HardRepoEpisode) -> dict[str, Any]:
        spec_vals = [ep.true_const, ep.decoy_const] if ep.ambiguous_goal else [ep.true_const]
        if ep.ambiguous_goal and (ep.true_const + ep.decoy_const) % 2 == 1:
            spec_vals.reverse()
        return {
            "kind": "GOAL",
            "alert_mod": ep.symptom_mod,
            "ambiguous": ep.ambiguous_goal,
            "spec_vals": spec_vals,
            "edges": list(ep.discovered_edges),
            "mention_order": list(range(ep.n_modules)),
        }

    @staticmethod
    def oracle_action(ep: HardRepoEpisode) -> tuple[int, int, int]:
        if ep.has_regression:
            return ACT_REVERT, ep.patched_mod or 0, ep.true_const
        if ep.ambiguous_goal and not ep.clarified:
            return ACT_ASK_CLARIFY, ep.symptom_mod, ep.true_const
        if ep.rival_mod not in ep.inspected and ep.true_root_mod not in ep.inspected:
            return ACT_INSPECT, ep.rival_mod, ep.true_const
        if ep.true_root_mod != ep.rival_mod and not ep.traced_mid:
            return ACT_TRACE, ep.mid_mod, ep.true_const
        if ep.true_root_mod not in ep.inspected:
            return ACT_INSPECT, ep.true_root_mod, ep.true_const
        if ep.patched_mod != ep.true_root_mod or ep.patched_val != ep.true_const:
            return ACT_PATCH, ep.true_root_mod, ep.true_const
        if not ep.verified_after_patch:
            return ACT_RUN_TESTS, ep.true_root_mod, ep.true_const
        return ACT_STOP, ep.true_root_mod, ep.true_const

    @staticmethod
    def step(ep: HardRepoEpisode, act: int, target_mod: int, val: int) -> tuple[dict[str, Any], bool, bool]:
        ep.steps += 1
        target_mod = max(0, min(ep.n_modules - 1, int(target_mod)))
        obs: dict[str, Any] = {"kind": "OBS", "act": act, "mod": target_mod, "spec_vals": []}

        if act == ACT_ASK_CLARIFY:
            ep.clarified = True
            obs["status"] = "CLARIFIED"
            obs["spec_vals"] = [ep.true_const]
        elif act == ACT_INSPECT:
            ep.inspected.add(target_mod)
            if target_mod == ep.true_root_mod:
                obs["status"] = "FAULT_AT"
                obs["spec_vals"] = [ep.true_const] if (not ep.ambiguous_goal or ep.clarified) else []
            elif target_mod == ep.rival_mod:
                ep.contradiction_seen = True
                obs["status"] = "HEALTHY"
            else:
                obs["status"] = "UNRELATED"
        elif act == ACT_TRACE:
            if target_mod in (ep.mid_mod, ep.symptom_mod):
                ep.traced_mid = True
                hidden_edge = ep.edges[-1] if (ep.true_root_mod != ep.rival_mod) else (ep.rival_mod, ep.symptom_mod)
                for u, v in ep.edges:
                    if v == ep.mid_mod and (u, v) not in ep.discovered_edges:
                        ep.discovered_edges.append((u, v))
                        hidden_edge = (u, v)
                obs["status"] = "TRACED"
                obs["new_edge"] = hidden_edge
            else:
                obs["status"] = "UNRELATED"
        elif act == ACT_PATCH:
            if ep.ambiguous_goal and not ep.clarified:
                ep.has_regression = True
                ep.patched_mod = target_mod
                ep.verified_after_patch = False
                obs["status"] = "REGRESSION_AMBIG"
            elif target_mod == ep.true_root_mod and val == ep.true_const:
                ep.has_regression = False
                ep.patched_mod = target_mod
                ep.patched_val = val
                ep.verified_after_patch = False
                obs["status"] = "NEEDS_TEST"
            else:
                ep.has_regression = True
                ep.patched_mod = target_mod
                ep.patched_val = val
                ep.verified_after_patch = False
                obs["status"] = "REGRESSION_WRONG"
        elif act == ACT_REVERT:
            ep.has_regression = False
            ep.patched_mod = None
            ep.patched_val = None
            ep.verified_after_patch = False
            obs["status"] = "REVERTED"
        elif act == ACT_RUN_TESTS:
            if not ep.has_regression and ep.patched_mod == ep.true_root_mod and ep.patched_val == ep.true_const:
                ep.verified_after_patch = True
                obs["status"] = "TESTS_PASS"
            else:
                ep.verified_after_patch = False
                obs["status"] = "TESTS_FAIL"
        elif act == ACT_STOP:
            ok = (
                not ep.has_regression
                and ep.patched_mod == ep.true_root_mod
                and ep.patched_val == ep.true_const
                and ep.verified_after_patch
            )
            obs["status"] = "VERIFIED" if ok else "FAILED"
            return obs, True, ok

        done = ep.steps >= HardCausalSWEEnv.MAX_STEPS
        return obs, done, False


def _encode_name_chars(name: str, max_len: int = 10) -> list[int]:
    out = [TOK_INDEX.get(f"C_{ch}", 0) for ch in name.lower()[:max_len]]
    while len(out) < max_len:
        out.append(0)
    return out


def _val_digits(v: int) -> list[int]:
    s = f"{max(0, min(9999, int(v))):04d}"
    return [int(c) for c in s]


class CognitiveAgentModel(nn.Module):
    """Matched-parameter testbed comparing C0, C4b(Sym12), C5(shield/unshield), and OmegaEFEPlanner."""

    ARMS = (
        "c0_transcript_policy",
        "c4b_sym12_equivariant",
        "c5_first_mention_shielded",
        "c5_first_mention_unshielded",
        "omega_efe_planner",
        "omega_no_efe_risk",
        "omega_no_graph_prop",
    )

    def __init__(self, arm: str, d: int = 96) -> None:
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"Unsupported cognitive arm: {arm}")
        self.arm = arm
        self.d = d
        self.is_omega = arm in ("omega_efe_planner", "omega_no_efe_risk", "omega_no_graph_prop")
        self.use_efe = arm in ("omega_efe_planner", "omega_no_graph_prop")
        self.use_graph = arm in ("omega_efe_planner", "omega_no_efe_risk")

        self.tok_emb = nn.Embedding(len(BASE_VOCAB), d)
        self.char_conv = nn.Conv1d(d, d, kernel_size=3, padding=1)
        self.node_feat_proj = nn.Linear(8, d)
        self.global_proj = nn.Linear(7, d)

        self.msg_up = nn.Linear(2 * d, d, bias=False)
        self.msg_down = nn.Linear(2 * d, d, bias=False)
        self.node_gru = nn.GRUCell(d, d)
        self.ws_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.ws_ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))
        self.ws_norm = nn.LayerNorm(d)

        self.unc_head = nn.Linear(d, 3)
        self.prag_head = nn.Linear(d, len(ACTION_NAMES))
        self.ig_head = nn.Linear(d, len(ACTION_NAMES))
        self.risk_head = nn.Linear(d, len(ACTION_NAMES))
        self.efe_meta = nn.Linear(3, 2)

        self.mod_q = nn.Linear(d, d, bias=False)
        self.mod_k = nn.Linear(d, d, bias=False)
        self.span_proj = nn.Linear(4, d)
        self.ptr_q = nn.Linear(d, d, bias=False)
        self.ptr_k = nn.Linear(d, d, bias=False)
        self.digit_heads = nn.ModuleList(nn.Linear(d, 10) for _ in range(4))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_state(
        self,
        ep: HardRepoEpisode,
        val_memory: list[tuple[int, list[float]]],
        device: torch.device,
        *,
        slot_perm: list[int] | None = None,
    ) -> dict[str, Any]:
        n_mod = ep.n_modules
        if self.is_omega or self.arm == "c0_transcript_policy":
            char_ids = torch.tensor([_encode_name_chars(nm) for nm in ep.mod_names], dtype=torch.long, device=device)
            ch_emb = self.tok_emb(char_ids).transpose(1, 2)
            name_vecs = F.silu(self.char_conv(ch_emb)).mean(dim=-1)
        else:
            perm = slot_perm if slot_perm is not None else list(range(n_mod))
            slot_ids = torch.tensor([SLOT_TOKEN_IDS[perm[m] % 12] for m in range(n_mod)], dtype=torch.long, device=device)
            name_vecs = self.tok_emb(slot_ids)

        in_deg = {m: 0 for m in range(n_mod)}
        out_deg = {m: 0 for m in range(n_mod)}
        for u, v in ep.discovered_edges:
            if 0 <= u < n_mod and 0 <= v < n_mod:
                out_deg[u] += 1
                in_deg[v] += 1

        n_feats = []
        for m in range(n_mod):
            is_sym = 1.0 if m == ep.symptom_mod else 0.0
            insp_fault = 1.0 if (m in ep.inspected and m == ep.true_root_mod) else 0.0
            insp_hlth = 1.0 if (m in ep.inspected and m == ep.rival_mod and ep.true_root_mod != ep.rival_mod) else 0.0
            insp_oth = 1.0 if (m in ep.inspected and m not in (ep.true_root_mod, ep.rival_mod)) else 0.0
            traced_f = 1.0 if (ep.traced_mid and m == ep.mid_mod) else 0.0
            patched_f = 1.0 if ep.patched_mod == m else 0.0
            n_feats.append([is_sym, insp_fault, insp_hlth, insp_oth, traced_f, patched_f, min(1.0, in_deg[m] / 2.0), min(1.0, out_deg[m] / 2.0)])

        feat_t = torch.tensor(n_feats, dtype=torch.float32, device=device)
        if not self.is_omega and self.arm != "c0_transcript_policy" and n_mod > 12:
            # Fixed 12-slot bottleneck aliases modules >= 12 modulo 12
            perm = slot_perm if slot_perm is not None else list(range(n_mod))
            slot_buf = torch.zeros(12, self.d, device=device)
            slot_cnt = torch.zeros(12, 1, device=device)
            raw_nodes = name_vecs + self.node_feat_proj(feat_t)
            for m in range(n_mod):
                s = perm[m] % 12
                slot_buf[s] = slot_buf[s] + raw_nodes[m]
                slot_cnt[s] = slot_cnt[s] + 1.0
            slot_buf = slot_buf / slot_cnt.clamp(min=1.0)
            h_nodes = torch.stack([slot_buf[perm[m] % 12] for m in range(n_mod)], dim=0)
        else:
            h_nodes = name_vecs + self.node_feat_proj(feat_t)

        if self.use_graph and ep.discovered_edges:
            for _ in range(3):
                agg = torch.zeros_like(h_nodes)
                for u, v in ep.discovered_edges:
                    if 0 <= u < n_mod and 0 <= v < n_mod:
                        agg[u] = agg[u] + torch.tanh(self.msg_up(torch.cat([h_nodes[u], h_nodes[v]], dim=-1)))
                        agg[v] = agg[v] + torch.tanh(self.msg_down(torch.cat([h_nodes[v], h_nodes[u]], dim=-1)))
                h_nodes = self.node_gru(agg, h_nodes)
        elif self.arm == "c0_transcript_policy":
            h_nodes = h_nodes + 0.2 * h_nodes.mean(dim=0, keepdim=True)

        g_vec = self.global_proj(
            torch.tensor(
                [[
                    1.0 if ep.ambiguous_goal else 0.0,
                    1.0 if ep.clarified else 0.0,
                    1.0 if ep.has_regression else 0.0,
                    1.0 if ep.verified_after_patch else 0.0,
                    1.0 if ep.contradiction_seen else 0.0,
                    ep.steps / 10.0,
                    min(1.0, len(val_memory) / 4.0),
                ]],
                dtype=torch.float32,
                device=device,
            )
        )

        joint = torch.cat([g_vec, h_nodes], dim=0).unsqueeze(0)
        if self.arm != "c0_transcript_policy":
            att_out, _ = self.ws_attn(joint, joint, joint, need_weights=False)
            joint = self.ws_norm(joint + att_out + self.ws_ffn(joint + att_out))
        z_global = joint[0, 0]
        mod_repr = joint[0, 1:]

        unc = torch.sigmoid(self.unc_head(z_global))
        prag = self.prag_head(z_global)
        ig = F.softplus(self.ig_head(z_global))
        risk = torch.sigmoid(self.risk_head(z_global))

        if self.use_efe:
            meta = F.softplus(self.efe_meta(unc))
            act_logits = prag + meta[0] * ig - meta[1] * risk
        elif self.arm == "c5_first_mention_shielded":
            act_logits = prag + ig - 1.5 * risk
            shield = torch.zeros_like(act_logits)
            shield[ACT_PATCH] = 2.2 * (unc[0] + unc[1])
            shield[ACT_STOP] = 2.2 * unc[2]
            act_logits = act_logits - shield
        else:
            act_logits = prag

        mod_logits = (self.mod_q(z_global) @ self.mod_k(mod_repr).T) / math.sqrt(self.d)
        ptr_logits = None
        if val_memory:
            sp_t = torch.tensor([m[1] for m in val_memory], dtype=torch.float32, device=device)
            sp_emb = self.span_proj(sp_t)
            ptr_logits = (self.ptr_q(z_global) @ self.ptr_k(sp_emb).T) / math.sqrt(self.d)
        dig_logits = [dh(z_global) for dh in self.digit_heads]

        return {
            "act_logits": act_logits,
            "mod_logits": mod_logits,
            "ptr_logits": ptr_logits,
            "digit_logits": dig_logits,
            "unc": unc,
            "ig": ig,
            "risk": risk,
        }


def _collect_val_span(obs: dict[str, Any], val_memory: list[tuple[int, list[float]]], step_idx: int) -> None:
    is_clar = 1.0 if obs.get("status") == "CLARIFIED" else 0.0
    is_flt = 1.0 if obs.get("status") == "FAULT_AT" else 0.0
    for idx_v, v in enumerate(obs.get("spec_vals", [])):
        val_memory.append((int(v), [is_clar, is_flt, step_idx / 10.0, idx_v * 0.5]))


def _counterfactual_labels(ep: HardRepoEpisode) -> tuple[list[float], list[float], list[float]]:
    u_intent = 1.0 if (ep.ambiguous_goal and not ep.clarified) else 0.0
    u_root = 0.0 if (ep.true_root_mod in ep.inspected) else 1.0
    u_ver = (
        0.0
        if (ep.patched_mod == ep.true_root_mod and ep.patched_val == ep.true_const and ep.verified_after_patch and not ep.has_regression)
        else 1.0
    )

    ig = [0.0] * len(ACTION_NAMES)
    risk = [0.0] * len(ACTION_NAMES)
    for a in range(len(ACTION_NAMES)):
        if a == ACT_ASK_CLARIFY and u_intent > 0.5:
            ig[a] = 1.0
        elif a == ACT_INSPECT and u_root > 0.5:
            ig[a] = 0.8
        elif a == ACT_TRACE and not ep.traced_mid and u_root > 0.5:
            ig[a] = 0.9
        elif a == ACT_RUN_TESTS and ep.patched_mod is not None and not ep.verified_after_patch:
            ig[a] = 0.9
        if a == ACT_PATCH and (u_intent > 0.5 or u_root > 0.5):
            risk[a] = 1.0
        if a == ACT_STOP and (u_ver > 0.5 or ep.has_regression):
            risk[a] = 1.0
    return [u_intent, u_root, u_ver], ig, risk


def train_and_eval_cognitive_arm(
    arm: str,
    *,
    seed: int,
    device: torch.device,
    train_episodes: int = 420,
    eval_episodes: int = 120,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = random.Random(seed * 997 + 13)
    model = CognitiveAgentModel(arm, d=96).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=1e-2)
    t0 = time.perf_counter()

    model.train()
    for ep_idx in range(train_episodes):
        ep = HardCausalSWEEnv.sample_episode(rng, regime="id")
        val_mem: list[tuple[int, list[float]]] = []
        _collect_val_span(HardCausalSWEEnv.initial_observation(ep), val_mem, 0)
        slot_perm = rng.sample(range(12), 12) if arm == "c4b_sym12_equivariant" else list(range(ep.n_modules))

        opt.zero_grad(set_to_none=True)
        ep_loss = torch.zeros((), device=device)
        for st in range(HardCausalSWEEnv.MAX_STEPS):
            oracle_a, oracle_m, oracle_v = HardCausalSWEEnv.oracle_action(ep)
            u_tgt, ig_tgt, rk_tgt = _counterfactual_labels(ep)
            out = model.forward_state(ep, val_mem, device, slot_perm=slot_perm)

            loss_a = F.cross_entropy(out["act_logits"].unsqueeze(0), torch.tensor([oracle_a], device=device))
            loss_m = F.cross_entropy(out["mod_logits"].unsqueeze(0), torch.tensor([oracle_m], device=device))
            loss_u = F.binary_cross_entropy(out["unc"], torch.tensor(u_tgt, dtype=torch.float32, device=device))
            loss_cf = (
                F.mse_loss(out["ig"], torch.tensor(ig_tgt, dtype=torch.float32, device=device))
                + F.binary_cross_entropy(out["risk"], torch.tensor(rk_tgt, dtype=torch.float32, device=device))
            )
            loss_v = torch.zeros((), device=device)
            if val_mem and out["ptr_logits"] is not None:
                best_idx = max((i for i, (vv, _) in enumerate(val_mem) if vv == oracle_v), default=None)
                if best_idx is not None:
                    loss_v = F.cross_entropy(out["ptr_logits"].unsqueeze(0), torch.tensor([best_idx], device=device))
            for dh_l, dg in zip(out["digit_logits"], _val_digits(oracle_v)):
                loss_v = loss_v + 0.15 * F.cross_entropy(dh_l.unsqueeze(0), torch.tensor([dg], device=device))

            ep_loss = ep_loss + loss_a + loss_m + 0.6 * loss_u + 0.8 * loss_cf + 0.8 * loss_v

            if ep_idx < (train_episodes // 2) and rng.random() < 0.20:
                exec_a = rng.choice([ACT_PATCH, ACT_INSPECT, oracle_a])
                exec_m = rng.randrange(ep.n_modules)
                exec_v = ep.decoy_const if ep.ambiguous_goal else oracle_v
            else:
                exec_a, exec_m, exec_v = oracle_a, oracle_m, oracle_v

            obs, done, _ = HardCausalSWEEnv.step(ep, exec_a, exec_m, exec_v)
            _collect_val_span(obs, val_mem, st + 1)
            if done:
                break

        ep_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    train_s = time.perf_counter() - t0
    model.eval()
    regimes_out: dict[str, Any] = {}
    with torch.no_grad():
        for reg_idx, regime in enumerate(("id", "ood", "super_ood")):
            eval_rng = random.Random(50_000 + seed * 101 + reg_idx * 37)
            succ = 0
            premature_reg = 0
            val_exact = 0
            val_total = 0
            rev_ok = 0
            rev_tot = 0
            for _ in range(eval_episodes):
                ep = HardCausalSWEEnv.sample_episode(eval_rng, regime=regime)
                val_mem = []
                _collect_val_span(HardCausalSWEEnv.initial_observation(ep), val_mem, 0)
                had_premature = False
                for st in range(HardCausalSWEEnv.MAX_STEPS):
                    out = model.forward_state(ep, val_mem, device)
                    pred_a = int(out["act_logits"].argmax().item())
                    pred_m = int(out["mod_logits"].argmax().item())
                    if val_mem and out["ptr_logits"] is not None:
                        p_idx = int(out["ptr_logits"].argmax().item())
                        pred_v = val_mem[p_idx][0]
                    else:
                        ds = [int(dl.argmax().item()) for dl in out["digit_logits"]]
                        pred_v = ds[0] * 1000 + ds[1] * 100 + ds[2] * 10 + ds[3]
                    if pred_a == ACT_PATCH:
                        val_total += 1
                        if pred_v == ep.true_const:
                            val_exact += 1
                    obs, done, ok = HardCausalSWEEnv.step(ep, pred_a, pred_m, pred_v)
                    if ep.has_regression or (pred_a == ACT_STOP and not ok):
                        had_premature = True
                    _collect_val_span(obs, val_mem, st + 1)
                    if done:
                        if ok:
                            succ += 1
                        if ep.contradiction_seen:
                            rev_tot += 1
                            if ep.patched_mod == ep.true_root_mod:
                                rev_ok += 1
                        break
                if had_premature:
                    premature_reg += 1
            regimes_out[regime] = {
                "verified_resolution_full": succ / float(eval_episodes),
                "premature_or_regression_rate": premature_reg / float(eval_episodes),
                "unseen_value_exact": val_exact / float(max(1, val_total)),
                "contradiction_revision_acc": rev_ok / float(max(1, rev_tot)),
            }

    return {
        "arm": arm,
        "seed": seed,
        "params": model.parameter_count(),
        "train_seconds": train_s,
        "regimes": regimes_out,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }


def run_efe_cognition_experiment(
    seeds: list[int],
    device: torch.device,
    *,
    train_episodes: int = 420,
    eval_episodes: int = 120,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    summary: dict[str, Any] = {}
    best_ckpt: dict[str, Tensor] = {}
    for arm in CognitiveAgentModel.ARMS:
        runs = [
            train_and_eval_cognitive_arm(
                arm, seed=sd, device=device, train_episodes=train_episodes, eval_episodes=eval_episodes
            )
            for sd in seeds
        ]
        if arm == "omega_efe_planner":
            best_ckpt = runs[0]["state_dict"]
        reg_agg: dict[str, Any] = {}
        for reg in ("id", "ood", "super_ood"):
            reg_agg[reg] = {}
            for metric in ("verified_resolution_full", "premature_or_regression_rate", "unseen_value_exact", "contradiction_revision_acc"):
                vals = [r["regimes"][reg][metric] for r in runs]
                reg_agg[reg][metric] = {
                    "mean": float(statistics.fmean(vals)),
                    "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                    "n": len(vals),
                }
        summary[arm] = {
            "params": runs[0]["params"],
            "train_seconds_mean": float(statistics.fmean([r["train_seconds"] for r in runs])),
            "regimes": reg_agg,
        }
        print(
            f"[OMEGA-EXP-02] {arm:28s} params={runs[0]['params']:,} "
            f"ID={reg_agg['id']['verified_resolution_full']['mean']:.3f} "
            f"OOD(9-12)={reg_agg['ood']['verified_resolution_full']['mean']:.3f} "
            f"SuperOOD(16-24)={reg_agg['super_ood']['verified_resolution_full']['mean']:.3f} "
            f"Premature={reg_agg['super_ood']['premature_or_regression_rate']['mean']:.3f}",
            flush=True,
        )
    return summary, best_ckpt
