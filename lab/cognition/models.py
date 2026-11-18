"""Four Matched-Parameter Neural Cognitive Architectures (`LAB-COG-01`).

All four arms share the exact same `NanoLM-v2` (`DDDA`: Sharp-Delta + Gated-VRes + QK-Canon)
sequence primitives (`d=128, H=4, L=4`), relational module-entity pointer scoring (enabling
zero-shot generalization from `4–5` training modules to `7–9` OOD modules!), and 4-digit
base-10 value emission heads, isolating ONLY the cognitive mechanism under test:

  - `C0_transcript_imitation`: Standard causal LM policy over raw concatenated episode transcript.
  - `C1_obs_world_model`: `C0` + next-observation token-distribution world-model head (CWM-style).
  - `C2_latent_loop_only`: `C0` + Huginn-style randomized-depth latent recurrence (`K in {1,2,3}`)
    without explicit belief/counterfactual/pointer objectives.
  - `C3_ebwm_counterfactual`: Persistent 8-slot Latent Belief Workspace `Z_t = U(Z_{t-1}, o_t, a_{t-1})`
    + Factored Epistemic Head `(u_intent, u_state, u_verify)` + Decision-Oriented Counterfactual
    Outcome Head `(info_gain, reg_risk)` + Exact Source-Span Value Pointer `P(Z_t, o_{0:t})`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab.arch.model import RMSNorm, rope_cache
from lab.arch.nextgen import NextGenBlock, NextGenCfg
from lab.cognition.swe_env import ACTION_NAMES

# Compact open-capable tokenizer mapping structural tokens, M0..M11, digits, and byte fallback
SPECIAL_TOKENS = [
    "<PAD>", "<SEP>", "GOAL", "OBS", "ACT", "alert", "suspects", "AMBIGUOUS_SPEC", "CLEAR_SPEC",
    "candidates", "or", "required", "REPO", "CLARIFIED", "active", "contract", "is",
    "REDUNDANT_CLARIFY", "already", "TRACE", "upstream", "INSPECT", "FAULT_FOUND", "needs",
    "HEALTHY_MODULE", "fault", "must", "be", "UNRELATED_MODULE", "PATCH", "REGRESSION_AMBIGUOUS",
    "REGRESSION_WRONG_TARGET_OR_VAL", "APPLIED_CLEAN_NEEDS_TEST", "REVERT", "CLEAN_STATE",
    "TESTS", "FAIL_REGRESSION", "on", "ALL_PASS", "verified", "FAIL_UNFIXED", "STOP",
    "VERIFIED_SUCCESS", "FAILED_UNVERIFIED", "NOOP", "VAL_MARKER",
] + ACTION_NAMES + [f"M{i}" for i in range(12)] + [f"D{d}" for d in range(10)] + [f"CH_{c}" for c in "abcdefghijklmnopqrstuvwxyz_:"]

TOK2ID = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
VOCAB_SIZE = len(SPECIAL_TOKENS)
MOD_TOKEN_IDS = [TOK2ID[f"M{i}"] for i in range(12)]


def encode_text_with_spans(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Encode environment text into token IDs AND record `(token_pos, int_val)` for every `VAL_<int>`!

    Notice `VAL_4829` is encoded as `[VAL_MARKER, D0, D4, D8, D2, D9]` (always 4 zero-padded digits!)
    so unseen 4-digit OOD values `[1000..9999]` are 100% representable in input tokens WITHOUT UNKs,
    while `VAL_MARKER` marks the exact source-pointer anchor!
    """
    ids: list[int] = []
    val_spans: list[tuple[int, int]] = []
    for raw_w in text.strip().split():
        # Strip `M0:chronos` into `M0` + character-spelled module name so unseen OOD names never UNK!
        if re.match(r"^M\d+:[a-z]+$", raw_w):
            m_part, name_part = raw_w.split(":", 1)
            ids.append(TOK2ID.get(m_part, 0))
            for ch in (":" + name_part[:6]):
                ids.append(TOK2ID.get(f"CH_{ch}", 0))
        elif raw_w.startswith("VAL_"):
            num = int(raw_w.split("_", 1)[1])
            pos = len(ids)
            ids.append(TOK2ID["VAL_MARKER"])
            for ch_d in f"{num:04d}":
                ids.append(TOK2ID[f"D{ch_d}"])
            val_spans.append((pos, num))
        elif raw_w in TOK2ID:
            ids.append(TOK2ID[raw_w])
        else:
            for ch in raw_w.lower()[:8]:
                if f"CH_{ch}" in TOK2ID:
                    ids.append(TOK2ID[f"CH_{ch}"])
    return ids, val_spans


def val_to_4digits(val: int) -> list[int]:
    s = f"{max(0, min(9999, val)):04d}"
    return [int(c) for c in s]


def digits4_to_val(d4: list[int]) -> int:
    return d4[0] * 1000 + d4[1] * 100 + d4[2] * 10 + d4[3]


class CognitiveLM(nn.Module):
    """Unified testbed supporting C0, C1, C2, and C3 on identical NanoLM-v2 `DDDA` blocks."""

    def __init__(self, arm: str, d: int = 128, n_head: int = 4, n_slots: int = 8):
        super().__init__()
        assert arm in (
            "C0_transcript_imitation",
            "C1_obs_world_model",
            "C2_latent_loop_only",
            "C3_ebwm_counterfactual",
            "C4a_ulbp_latent_plan",
            "C4b_ulbp_equivariant",
        )
        self.arm = arm
        self.is_c3_family = arm in ("C3_ebwm_counterfactual", "C4a_ulbp_latent_plan", "C4b_ulbp_equivariant")
        self.is_c4 = arm in ("C4a_ulbp_latent_plan", "C4b_ulbp_equivariant")
        self.d = d
        self.n_slots = n_slots
        ffn_dim = 256 if self.is_c4 else 288
        ng_cfg = NextGenCfg(
            vocab=VOCAB_SIZE, d=d, n_head=n_head, n_kv=2, ffn=ffn_dim,
            layout="DDDA", seq=512, gated_vres=True, sharp_v=True, qk_canon_attn=True,
        )
        self.ng_cfg = ng_cfg
        self.emb = nn.Embedding(VOCAB_SIZE, d)
        self.blocks = nn.ModuleList(NextGenBlock(ng_cfg, k) for k in ng_cfg.layout)
        self.nf = RMSNorm(d)

        # Shared policy & relational module-pointer + 4-digit generative heads (present in ALL 4 arms!)
        self.act_head = nn.Linear(d, len(ACTION_NAMES))
        self.mod_q = nn.Linear(d, d, bias=False)
        self.mod_k = nn.Linear(d, d, bias=False)
        self.digit_heads = nn.ModuleList(nn.Linear(d, 10) for _ in range(4))

        # C1 auxiliary next-observation bag-of-tokens world model head
        if arm == "C1_obs_world_model":
            self.obs_wm_head = nn.Sequential(nn.Linear(d + len(ACTION_NAMES), d), nn.SiLU(), nn.Linear(d, VOCAB_SIZE))

        # C2 Huginn-style recurrent latent loop embeddings & stability norm
        if arm == "C2_latent_loop_only":
            self.loop_step_emb = nn.Embedding(4, d)
            self.loop_norm = RMSNorm(d)

        # C3 / C4 Persistent 8-Slot Belief Workspace + Epistemic + Counterfactual + Source-Pointer heads
        if self.is_c3_family:
            self.init_belief = nn.Parameter(torch.randn(1, n_slots, d) * 0.02)
            self.slot_gate = nn.Linear(2 * d, d)
            self.belief_post_q = nn.Linear(d, d, bias=False)
            self.unc_head = nn.Linear(d, 3)                    # (u_intent, u_state, u_verify)
            self.cf_head = nn.Linear(d, len(ACTION_NAMES) * 2) # per-action (info_gain, reg_risk)
            self.ptr_q = nn.Linear(d, d, bias=False)
            self.ptr_k = nn.Linear(d, d, bias=False)

        # C4 Uncertainty-Gated Latent Belief-Space Planner (O((S+N)^2 d)) + Predicate-Context Span Encoder
        if self.is_c4:
            self.plan_step_emb = nn.Embedding(3, d)
            self.plan_norm1 = RMSNorm(d)
            self.plan_attn = nn.MultiheadAttention(d, n_head, batch_first=True)
            self.plan_norm2 = RMSNorm(d)
            self.plan_ffn = nn.Sequential(nn.Linear(d, 2 * d, bias=False), nn.SiLU(), nn.Linear(2 * d, d, bias=False))
            self.plan_unc_gate = nn.Linear(3, 1)
            self.span_ctx_proj = nn.Linear(3 * d, d, bias=False)
            self.turn_recency_emb = nn.Embedding(12, d)

    def _encode_seq(self, ids: torch.Tensor, loop_k: int = 1) -> torch.Tensor:
        B, T = ids.shape
        x = self.emb(ids)
        hd = self.d // self.ng_cfg.n_head
        rope = rope_cache(T, hd, 10000.0, ids.device)
        if self.arm == "C2_latent_loop_only":
            for k in range(loop_k):
                x_in = self.loop_norm(x + self.loop_step_emb.weight[min(k, 3)])
                v0 = None
                for b in self.blocks:
                    x_in, v0, _ = b(x_in, rope, v0)
                x = 0.5 * x + 0.5 * x_in
            return self.nf(x)

        v0 = None
        for b in self.blocks:
            x, v0, _ = b(x, rope, v0)
        return self.nf(x)

    def forward_transcript_step(
        self,
        seq_ids: list[int],
        n_modules: int,
        val_spans: list[tuple[int, int]],
        dev: torch.device,
        loop_k: int = 3,
    ) -> dict[str, Any]:
        """Forward pass for C0, C1, C2 over concatenated episode transcript `seq_ids`."""
        inp = torch.tensor([seq_ids], dtype=torch.long, device=dev)
        k_use = (torch.randint(1, 4, (1,)).item() if self.training else loop_k) if self.arm == "C2_latent_loop_only" else 1
        h_seq = self._encode_seq(inp, loop_k=int(k_use))[0]   # (T, d)
        h_last = h_seq[-1]                                    # (d,)

        # Relational module scoring: pool contextual hidden states of M0..M_{n_modules-1}
        mod_vecs = []
        for m in range(n_modules):
            tid = MOD_TOKEN_IDS[m]
            positions = [idx for idx, tok in enumerate(seq_ids) if tok == tid]
            if positions:
                mod_vecs.append(h_seq[positions].mean(dim=0))
            else:
                mod_vecs.append(self.emb.weight[tid])
        M_mat = torch.stack(mod_vecs, dim=0)                  # (n_modules, d)
        mod_logits = (self.mod_q(h_last) @ self.mod_k(M_mat).T) / (self.d ** 0.5)
        act_logits = self.act_head(h_last)
        dig_logits = [dh(h_last) for dh in self.digit_heads]
        return {
            "h_last": h_last,
            "act_logits": act_logits,
            "mod_logits": mod_logits,
            "digit_logits": dig_logits,
        }

    def init_c3_workspace(self, goal_ids: list[int], n_modules: int, val_spans: list[tuple[int, int]], dev: torch.device) -> dict[str, Any]:
        """Initialize C3/C4 persistent 8-slot belief workspace from Goal + Repo declaration."""
        z = self.init_belief.to(dev)                          # (1, S, d)
        # C4b Sym(12) alpha-renaming equivariance: permute bound module symbols uniformly during training!
        if self.arm == "C4b_ulbp_equivariant" and self.training:
            perm = torch.randperm(12).tolist()
        else:
            perm = list(range(12))
        ws = {
            "z": z,
            "mod_mem": None,
            "evidence_vecs": [],   # list[(Tensor(d,), int_val)]
            "n_modules": n_modules,
            "perm": perm,
            "turn_idx": 0,
        }
        return self.update_c3_workspace(ws, goal_ids, val_spans, dev)

    def update_c3_workspace(
        self,
        ws: dict[str, Any],
        obs_ids: list[int],
        obs_val_spans: list[tuple[int, int]],
        dev: torch.device,
    ) -> dict[str, Any]:
        """Recurrent belief update `Z_t = U_theta(Z_{t-1}, obs_t)` (+ C4 subspace latent planning)."""
        z_prev = ws["z"]                                      # (1, S, d)
        perm: list[int] = ws.get("perm", list(range(12)))
        turn_idx: int = int(ws.get("turn_idx", 0))
        if self.arm == "C4b_ulbp_equivariant":
            id_remap = {MOD_TOKEN_IDS[m]: MOD_TOKEN_IDS[perm[m]] for m in range(12)}
            mapped_ids = [id_remap.get(t, t) for t in obs_ids]
        else:
            mapped_ids = obs_ids

        obs_t = torch.tensor([mapped_ids], dtype=torch.long, device=dev)
        x_obs = self.emb(obs_t)                               # (1, L, d)
        cat_in = torch.cat([z_prev, x_obs], dim=1)            # (1, S + L, d)
        hd = self.d // self.ng_cfg.n_head
        rope = rope_cache(cat_in.size(1), hd, 10000.0, dev)
        v0 = None
        h = cat_in
        for b in self.blocks:
            h, v0, _ = b(h, rope, v0)
        h = self.nf(h)
        z_cand = h[:, : self.n_slots, :]
        h_obs = h[0, self.n_slots :, :]                       # (L, d)

        g = torch.sigmoid(self.slot_gate(torch.cat([z_prev, z_cand], dim=-1)))
        z_next = g * z_cand + (1.0 - g) * z_prev

        n_mod = ws["n_modules"]
        prev_mods = ws["mod_mem"]
        new_mods = []
        for m in range(n_mod):
            orig_tid = MOD_TOKEN_IDS[m]
            emb_tid = MOD_TOKEN_IDS[perm[m]]
            pos = [idx for idx, tok in enumerate(obs_ids) if tok == orig_tid]
            if pos:
                v_m = h_obs[pos].mean(dim=0)
                new_mods.append(0.5 * prev_mods[m] + 0.5 * v_m if prev_mods is not None else v_m)
            else:
                new_mods.append(prev_mods[m] if prev_mods is not None else self.emb.weight[emb_tid])
        mod_mat = torch.stack(new_mods, dim=0)                # (n_mod, d)

        # C4: L=3 Uncertainty-Gated Subspace Belief-Graph Planning over [Z_t; M_mat] (<= 20 tokens!)
        if self.is_c4:
            joint = torch.cat([z_next, mod_mat.unsqueeze(0)], dim=1)  # (1, S + n_mod, d)
            for l_step in range(3):
                b_cur = joint[0, : self.n_slots].mean(dim=0)
                u_cur = torch.sigmoid(self.unc_head(b_cur))
                think_g = torch.sigmoid(self.plan_unc_gate(u_cur))    # scalar in (0, 1)
                step_tag = torch.zeros_like(joint)
                step_tag[:, : self.n_slots, :] = self.plan_step_emb.weight[l_step]
                qn = self.plan_norm1(joint + step_tag)
                attn_out, _ = self.plan_attn(qn, qn, qn, need_weights=False)
                mid = joint + think_g * attn_out
                joint = mid + think_g * self.plan_ffn(self.plan_norm2(mid))
            z_next = joint[:, : self.n_slots, :]
            mod_mat = joint[0, self.n_slots :, :]

        ev_list = list(ws["evidence_vecs"])
        for pos_i, val_int in obs_val_spans:
            if 0 <= pos_i < h_obs.size(0):
                if self.is_c4:
                    # 3-token left predicate context [pos-2, pos-1, pos] + turn recency embedding!
                    h_m2 = h_obs[max(0, pos_i - 2)]
                    h_m1 = h_obs[max(0, pos_i - 1)]
                    h_0 = h_obs[pos_i]
                    span_v = self.span_ctx_proj(torch.cat([h_m2, h_m1, h_0], dim=-1)) + self.turn_recency_emb.weight[min(turn_idx, 11)]
                    ev_list.append((span_v, val_int))
                else:
                    ev_list.append((h_obs[pos_i], val_int))

        b_summary = z_next[0].mean(dim=0)                     # (d,)
        unc_pred = torch.sigmoid(self.unc_head(b_summary))    # (3,) in (0, 1): u_intent, u_state, u_verify
        cf_raw = self.cf_head(b_summary).view(len(ACTION_NAMES), 2)
        pred_ig = F.softplus(cf_raw[:, 0])
        pred_risk = torch.sigmoid(cf_raw[:, 1])

        act_logits = self.act_head(b_summary) + 1.2 * pred_ig - 2.0 * pred_risk
        if self.is_c4:
            # Differentiable Epistemic Sufficiency Shield: suppress premature PATCH & unverified STOP!
            suff_pen = torch.zeros_like(act_logits)
            suff_pen[4] = 2.2 * (unc_pred[0] + unc_pred[1])   # ACT_PATCH
            suff_pen[6] = 2.2 * unc_pred[2]                   # ACT_STOP
            act_logits = act_logits - suff_pen

        mod_logits = (self.mod_q(z_next[0, 0]) @ self.mod_k(mod_mat).T) / (self.d ** 0.5)
        belief_post_logits = (self.belief_post_q(z_next[0, 1]) @ self.mod_k(mod_mat).T) / (self.d ** 0.5)
        dig_logits = [dh(z_next[0, 2]) for dh in self.digit_heads]

        ptr_logits = None
        if ev_list:
            ev_mat = torch.stack([e[0] for e in ev_list], dim=0)
            ptr_logits = (self.ptr_q(z_next[0, 3]) @ self.ptr_k(ev_mat).T) / (self.d ** 0.5)

        return {
            "z": z_next,
            "mod_mem": mod_mat,
            "evidence_vecs": ev_list,
            "n_modules": n_mod,
            "perm": perm,
            "turn_idx": turn_idx + 1,
            "act_logits": act_logits,
            "mod_logits": mod_logits,
            "belief_post_logits": belief_post_logits,
            "unc_pred": unc_pred,
            "pred_ig": pred_ig,
            "pred_risk": pred_risk,
            "digit_logits": dig_logits,
            "ptr_logits": ptr_logits,
        }
