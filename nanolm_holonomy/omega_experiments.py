"""NANOLM Omega Experimental Validation Suite (EXP-Omega1, EXP-Omega2, EXP-Omega3)."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_frontier.benchmark_15d import (
    CAPABILITY_DIMENSIONS,
    BenchmarkDim,
    BenchmarkTask,
    bootstrap_accuracy_ci,
    build_benchmark_15d_suite,
    compute_efficiency_score,
    verify_benchmark_response,
)
from nanolm_frontier.curriculum import verify_zero_contamination
from nanolm_frontier.surgery_and_rlvr import (
    GRPOConfig,
    _token_logprobs,
    apply_chat_template_safe,
    train_sft_curriculum,
)
from nanolm_frontier.tournament import _load_hf_model_and_tokenizer
from nanolm_holonomy.architecture import HSPAArchConfig, HSPAFamily, HSPAUnifiedLM
from nanolm_holonomy.curriculum_v2 import build_curriculum_v2
from nanolm_holonomy.cycle2_experiments import _evaluate_15d_extended, _reset_all_cuda_peaks
from nanolm_holonomy.experiments import (
    _GEN_PERMS,
    _PERM_TO_ID,
    _autocast,
    _build_x2_corpus,
    _compose_perm,
    _cuda_amp_dtype,
    _peak_mb,
    _sync,
    MicroByteWordTokenizer,
)
from nanolm_holonomy.omega_architecture import (
    HSPAOmegaSidecar,
    SpectralNullspaceLoRALinear,
    _bypass_omega_adapters,
    apply_nullspace_lora,
    apply_omega_graft,
    so3_projective_adjoint,
    su2_lie_exponential,
)
from nanolm_holonomy.omega_system import omega_generate
from nanolm_holonomy.sidecar_evolution import build_challenging_rlvr_curriculum
from nanolm_holonomy.spinor_algebra import cayley_rotor
from nanolm_holonomy.triton_holonomy import holonomy_scan_fast
from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS, ParameterBudgetExceededError
from nanolm_posttrain.lora import LoRAConfig, apply_lora, lora_state_dict


# ---------------------------------------------------------------------------
# EXP-Omega1: Non-Abelian S_4 Double-Cover Holonomy & 16x Length Extrapolation
# ---------------------------------------------------------------------------

def _s4_stream_batch(batch_size: int, seq_len: int, rng: random.Random, device: torch.device) -> tuple[Tensor, Tensor]:
    xs = [[0] * seq_len for _ in range(batch_size)]
    ys = [[0] * seq_len for _ in range(batch_size)]
    n_gens = len(_GEN_PERMS)
    for b in range(batch_size):
        st = (0, 1, 2, 3)
        for t in range(seq_len):
            gi = rng.randrange(n_gens)
            st = _compose_perm(_GEN_PERMS[gi], st)
            xs[b][t] = gi
            ys[b][t] = _PERM_TO_ID[st]
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)


class _HolonomicProbeModel(nn.Module):
    def __init__(self, arm_type: str, *, n_fibers: int = 24) -> None:
        super().__init__()
        self.arm_type = arm_type
        self.G = n_fibers
        if arm_type == "b0_transformer_rope":
            cfg = HSPAArchConfig(
                name="probe-tx",
                family=HSPAFamily.B0_TRANSFORMER,
                vocab_size=32,
                d_model=192,
                n_layers=2,
                n_heads=6,
                n_kv_heads=2,
                d_head=32,
                d_ff=384,
                max_seq_len=1_024,
            )
            self.tx = HSPAUnifiedLM(cfg)
        else:
            in_dim = n_fibers if arm_type == "b1_abelian_u1" else 3 * n_fibers
            self.emb = nn.Embedding(8, in_dim)
            nn.init.uniform_(self.emb.weight, -1.2, 1.2)
            feat_dim = n_fibers * 2 if arm_type == "b1_abelian_u1" else (
                n_fibers * 4 if arm_type == "h1_su2_lie_linear" else n_fibers * 9
            )
            self.head = nn.Linear(feat_dim, 24)

    def forward(self, x: Tensor) -> Tensor:
        bsz, slen = x.shape
        if self.arm_type == "b0_transformer_rope":
            return self.tx(x)[:, :, :24]
        if self.arm_type == "b1_abelian_u1":
            th = self.emb(x).cumsum(dim=1)
            feats = torch.cat((torch.cos(th), torch.sin(th)), dim=-1)
            return self.head(feats)

        w = self.emb(x).reshape(bsz, slen, self.G, 3).permute(0, 2, 1, 3).unsqueeze(1).contiguous()
        g = cayley_rotor(w) if self.arm_type == "h2_su2_cayley_adjoint" else su2_lie_exponential(w)
        s = torch.zeros(bsz, 1, self.G, slen, 4, device=x.device, dtype=w.dtype)
        init = torch.zeros(bsz, 1, self.G, 4, device=x.device, dtype=w.dtype)
        init[..., 0] = 1.0
        omega, _ = holonomy_scan_fast(g, s, init_state=init, chunk_size=None)
        omega = omega.squeeze(1).permute(0, 2, 1, 3)  # [B, T, G, 4]
        if self.arm_type == "h1_su2_lie_linear":
            feats = omega.reshape(bsz, slen, -1)
        else:
            feats = so3_projective_adjoint(omega).reshape(bsz, slen, -1)
        return self.head(feats)


def run_exp_omega1_s4_holonomy(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (101, 202, 303),
    steps: int = 600,
    batch_size: int = 64,
    train_len: int = 32,
) -> dict[str, Any]:
    """Verify Z_2 Abelian Ceiling vs Faithful Lie-Exp SO(3) Projective Holonomy up to 16x OOD."""
    arm_types = (
        "b0_transformer_rope",
        "b1_abelian_u1",
        "h1_su2_lie_linear",
        "h2_su2_cayley_adjoint",
        "h3_omega_so3_lie_adjoint",
    )
    lengths = (32, 64, 128, 256, 512)
    eval_rng = random.Random(20261118)
    probes = {L: _s4_stream_batch(128, L, eval_rng, device) for L in lengths}

    arms_out: list[dict[str, Any]] = []
    for arm_name in arm_types:
        per_seed: list[dict[str, Any]] = []
        param_cnt = 0
        for seed in seeds:
            torch.manual_seed(seed)
            rng = random.Random(seed)
            model = _HolonomicProbeModel(arm_name, n_fibers=24).to(device)
            param_cnt = sum(p.numel() for p in model.parameters())
            opt = torch.optim.AdamW(model.parameters(), lr=1.2e-2 if "b0_" not in arm_name else 2e-3, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
            model.train()
            for _ in range(steps):
                x, y = _s4_stream_batch(batch_size, train_len, rng, device)
                opt.zero_grad(set_to_none=True)
                lg = model(x)
                loss = F.cross_entropy(lg.reshape(-1, 24), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()

            model.eval()
            accs: dict[str, float] = {}
            with torch.no_grad():
                for L, (px, py) in probes.items():
                    plg = model(px)
                    accs[f"L_{L}"] = round(float((plg[:, -1].argmax(dim=-1) == py[:, -1]).float().mean().item()), 4)
            per_seed.append({"seed": seed, "accuracy": accs})
            del model, opt, sch

        summary: dict[str, Any] = {}
        for L in lengths:
            k = f"L_{L}"
            vs = [r["accuracy"][k] for r in per_seed]
            mu = sum(vs) / len(vs)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vs) / len(vs))
            summary[k] = {"mean": round(mu, 4), "std": round(sd, 4)}

        print(
            f"[EXP-Omega1] {arm_name:24s} S4(32/64/128/256/512)="
            + "/".join(f"{summary[f'L_{L}']['mean']:.3f}" for L in lengths),
            flush=True,
        )
        arms_out.append({"arm": arm_name, "params": param_cnt, "summary": summary, "per_seed": per_seed})
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "experiment": "EXP_Omega1_S4_LieExp_SO3_Holonomy",
        "seeds": list(seeds),
        "train_len": train_len,
        "lengths": list(lengths),
        "arms": arms_out,
    }


# ---------------------------------------------------------------------------
# EXP-Omega2: Spectral Nullspace-LoRA Anti-Forgetting Under Continual Adaptation
# ---------------------------------------------------------------------------

def _build_domain_b_json_sequences(
    tok: MicroByteWordTokenizer, *, seq_len: int = 256, n_seqs: int = 96, seed: int = 9191
) -> dict[str, Any]:
    """Synthetic Domain-B structured JSON record adaptation corpus disjoint from Domain-A."""
    rng = random.Random(seed)
    tr, val = [], []
    tot_tok = 0
    tot_bytes = 0
    for i in range(n_seqs):
        lines = []
        for _ in range(18):
            k1 = rng.choice(["alpha", "beta", "gamma", "delta", "zeta", "theta"])
            v1 = rng.randint(10, 999)
            ok = rng.choice(["true", "false"])
            lines.append(f'{{"entity": "{k1}", "score": {v1}, "verified": {ok}}}')
        ids, nb = tok.encode("\n".join(lines))
        tot_tok += len(ids)
        tot_bytes += nb
        if len(ids) < seq_len:
            ids = (ids * ((seq_len // max(len(ids), 1)) + 2))[:seq_len]
        else:
            ids = ids[:seq_len]
        if i < int(0.8 * n_seqs):
            tr.append(ids)
        else:
            val.append(ids)
    bpt = tot_bytes / max(tot_tok, 1)
    return {"train": tr, "val": val, "bpt": bpt}


def _eval_lm_bpb(
    model: nn.Module, seqs: list[list[int]], vocab_size: int, bpt: float, device: torch.device, batch_size: int = 8
) -> float:
    model.eval()
    tot_loss = 0.0
    tot_tok = 0
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            b = torch.tensor(seqs[i : i + batch_size], dtype=torch.long, device=device)
            with _autocast(device):
                lg = model(b[:, :-1])
            tot_loss += float(F.cross_entropy(lg.float().reshape(-1, vocab_size), b[:, 1:].reshape(-1), reduction="sum").item())
            tot_tok += int(b[:, 1:].numel())
    return round((tot_loss / max(tot_tok, 1)) / max(bpt * math.log(2.0), 1e-6), 4)


def run_exp_omega2_nullspace_continual(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (42, 137, 271),
    pretrain_steps: int = 400,
    adapt_steps: int = 250,
    batch_size: int = 12,
) -> dict[str, Any]:
    """Compare Standard LoRA, Proximal-L2 LoRA, and Spectral Nullspace LoRA on Domain A -> B continual shift."""
    tok = MicroByteWordTokenizer(vocab_size=4_096)
    dom_a = _build_x2_corpus(tok)
    dom_b = _build_domain_b_json_sequences(tok, seq_len=256, n_seqs=96)
    tr_a = dom_a["train"]
    val_a_pairs = [(vseqs[:6], vbpt) for _, (vseqs, vbpt, _) in dom_a["val"].items()]
    tr_b = dom_b["train"]
    val_b = dom_b["val"]
    val_b_bpt = float(dom_b["bpt"])

    def _eval_dom_a(mdl: nn.Module, vocab_sz: int) -> float:
        scores = [_eval_lm_bpb(mdl, vs, vocab_sz, vbpt, device) for vs, vbpt in val_a_pairs]
        return round(sum(scores) / max(len(scores), 1), 4)

    base_cfg = HSPAArchConfig(
        name="omega2-base-hspa",
        family=HSPAFamily.GEN1_H2_HSPA_PREDCODE,
        vocab_size=tok.vocab_size,
        d_model=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        d_head=32,
        d_ff=512,
        n_registers=8,
        d_fiber=128,
        max_seq_len=512,
    )

    modes = ("Standard_LoRA", "Proximal_L2_LoRA", "Spectral_Nullspace_LoRA")
    per_mode: dict[str, list[dict[str, float]]] = {m: [] for m in modes}

    for seed in seeds:
        torch.manual_seed(seed)
        base = HSPAUnifiedLM(base_cfg).to(device)
        opt = torch.optim.AdamW(base.parameters(), lr=1e-3, weight_decay=0.05)
        use_scaler = device.type == "cuda" and _cuda_amp_dtype() == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        base.train()
        for step in range(pretrain_steps):
            s_i = (step * batch_size) % len(tr_a)
            raw = tr_a[s_i : s_i + batch_size]
            if len(raw) < batch_size:
                raw = raw + tr_a[: batch_size - len(raw)]
            b = torch.tensor(raw, dtype=torch.long, device=device)
            opt.zero_grad(set_to_none=True)
            with _autocast(device):
                lg = base(b[:, :-1])
                loss = F.cross_entropy(lg.float().reshape(-1, base_cfg.vocab_size), b[:, 1:].reshape(-1))
            if use_scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

        pre_a_bpb = _eval_dom_a(base, base_cfg.vocab_size)
        pre_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        del base, opt, scaler

        for mode in modes:
            torch.manual_seed(seed + 1000)
            m = HSPAUnifiedLM(base_cfg).to(device)
            m.load_state_dict({k: v.to(device) for k, v in pre_state.items()})
            lora_cfg = LoRAConfig(
                rank=16,
                alpha=32.0,
                target_suffixes=("phi_proj", "o_proj", "w_enc", "w_dec", "w_err"),
            )
            if mode == "Spectral_Nullspace_LoRA":
                apply_nullspace_lora(m, lora_cfg, nullspace_k=32)
            else:
                apply_lora(m, lora_cfg)

            wd = 0.25 if mode == "Proximal_L2_LoRA" else 0.01
            trainable = [p for p in m.parameters() if p.requires_grad]
            opt_m = torch.optim.AdamW(trainable, lr=8e-4, weight_decay=wd)
            m.train()
            for step in range(adapt_steps):
                s_i = (step * batch_size) % len(tr_b)
                raw = tr_b[s_i : s_i + batch_size]
                if len(raw) < batch_size:
                    raw = raw + tr_b[: batch_size - len(raw)]
                b = torch.tensor(raw, dtype=torch.long, device=device)
                opt_m.zero_grad(set_to_none=True)
                with _autocast(device):
                    lg = m(b[:, :-1])
                    loss = F.cross_entropy(lg.float().reshape(-1, base_cfg.vocab_size), b[:, 1:].reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt_m.step()

            post_a_bpb = _eval_dom_a(m, base_cfg.vocab_size)
            post_b_bpb = _eval_lm_bpb(m, val_b, base_cfg.vocab_size, val_b_bpt, device)
            forgetting = round(post_a_bpb - pre_a_bpb, 4)
            per_mode[mode].append(
                {
                    "seed": float(seed),
                    "pre_dom_a_bpb": pre_a_bpb,
                    "post_dom_a_bpb": post_a_bpb,
                    "forgetting_delta_bpb": forgetting,
                    "dom_b_bpb": post_b_bpb,
                    "joint_mean_bpb": round(0.5 * (post_a_bpb + post_b_bpb), 4),
                }
            )
            del m, opt_m
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries: list[dict[str, Any]] = []
    for mode in modes:
        rows = per_mode[mode]
        agg: dict[str, Any] = {"arm": mode}
        for key in ("pre_dom_a_bpb", "post_dom_a_bpb", "forgetting_delta_bpb", "dom_b_bpb", "joint_mean_bpb"):
            vs = [r[key] for r in rows]
            mu = sum(vs) / len(vs)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vs) / len(vs))
            agg[key] = {"mean": round(mu, 4), "std": round(sd, 4)}
        print(
            f"[EXP-Omega2] {mode:24s} forget={agg['forgetting_delta_bpb']['mean']:+.4f} "
            f"domB={agg['dom_b_bpb']['mean']:.4f} joint={agg['joint_mean_bpb']['mean']:.4f}",
            flush=True,
        )
        summaries.append({**agg, "per_seed": rows})

    return {"experiment": "EXP_Omega2_Spectral_Nullspace_Continual", "seeds": list(seeds), "arms": summaries}


# ---------------------------------------------------------------------------
# EXP-Omega3: Unified 1.75B Nullspace-Holonomic Model + Zero-Oracle Controller
# ---------------------------------------------------------------------------

def _grpo_with_omega_ref(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[Any],
    *,
    device: torch.device,
    config: GRPOConfig = GRPOConfig(rounds=1, group_size=4, max_new_tokens=224, lr=2.5e-5, kl_beta=0.10),
) -> dict[str, Any]:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=0.01)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    total_updates = 0
    skipped_zero_var = 0
    round_rewards: list[float] = []
    mean_kls: list[float] = []

    for _ in range(config.rounds):
        r_vals: list[float] = []
        for ex in prompts:
            msgs = [{"role": "system", "content": ex.system_prompt}, {"role": "user", "content": ex.user_prompt}]
            ptxt = apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True)
            pids = tokenizer(ptxt, add_special_tokens=False)["input_ids"]
            plen = len(pids)
            pt = torch.tensor([pids], dtype=torch.long, device=device)

            model.eval()
            rollouts: list[Tensor] = []
            rewards: list[float] = []
            tview = BenchmarkTask(ex.example_id, ex.dim, "hard", ex.system_prompt, ex.user_prompt, ex.verifier_spec)
            with torch.no_grad():
                bp = pt.repeat(config.group_size, 1)
                seqs = model.generate(
                    input_ids=bp,
                    attention_mask=torch.ones_like(bp),
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=config.temperature,
                    top_p=0.95,
                    pad_token_id=pad_id,
                )
                for gi in range(config.group_size):
                    cids = seqs[gi, plen:].tolist()
                    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in cids:
                        cids = cids[: cids.index(tokenizer.eos_token_id) + 1]
                    if not cids:
                        rewards.append(0.0)
                        rollouts.append(seqs[gi : gi + 1, : plen + 1])
                        continue
                    txt = tokenizer.decode(cids, skip_special_tokens=True)
                    v = verify_benchmark_response(tview, txt)
                    rewards.append(1.0 if v.passed else 0.35 * float(v.score))
                    rollouts.append(torch.tensor([pids + cids], dtype=torch.long, device=device))

            mu = sum(rewards) / len(rewards)
            r_vals.append(mu)
            std = math.sqrt(sum((r - mu) ** 2 for r in rewards) / len(rewards))
            if std < 1e-4:
                skipped_zero_var += 1
                continue

            model.train()
            optimizer.zero_grad(set_to_none=True)
            for gi, fg in enumerate(rollouts):
                if fg.shape[1] <= plen:
                    continue
                adv = max(-2.5, min(2.5, (rewards[gi] - mu) / (std + 1e-5)))
                out_pi = model(input_ids=fg)
                logits_pi = out_pi.logits if hasattr(out_pi, "logits") else out_pi
                logp_pi = _token_logprobs(logits_pi, fg, plen)
                with torch.no_grad(), _bypass_omega_adapters(model):
                    out_ref = model(input_ids=fg)
                    logits_ref = out_ref.logits if hasattr(out_ref, "logits") else out_ref
                    logp_ref = _token_logprobs(logits_ref, fg, plen)
                lr_t = (logp_ref - logp_pi).clamp(-10.0, 10.0)
                kl = torch.exp(lr_t) - lr_t - 1.0
                loss_g = (-adv * logp_pi.mean() + config.kl_beta * kl.mean()) / float(len(rollouts))
                loss_g.backward()
                mean_kls.append(float(kl.mean().detach().cpu().item()))
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            optimizer.step()
            total_updates += 1
        round_rewards.append(round(sum(r_vals) / max(len(r_vals), 1), 4))

    model.eval()
    return {
        "rounds": config.rounds,
        "total_prompt_groups": config.rounds * len(prompts),
        "gradient_updates": total_updates,
        "skipped_zero_variance_groups": skipped_zero_var,
        "round_mean_rewards": round_rewards,
        "mean_kl": round(sum(mean_kls) / max(len(mean_kls), 1), 6),
    }


def _evaluate_15d_omega_cognitive(
    model: Any,
    tokenizer: Any,
    tasks: tuple[BenchmarkTask, ...],
    device: torch.device,
    *,
    arm_name: str,
) -> dict[str, Any]:
    model.eval()
    _reset_all_cuda_peaks(device)
    total_params = sum(p.numel() for p in model.parameters())
    if total_params > ABSOLUTE_MAX_PARAMETERS:
        raise ParameterBudgetExceededError(f"{arm_name} has {total_params:,} > 4B limit")

    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "num_hidden_layers"):
        n_l = int(cfg.num_hidden_layers)
        n_kv = int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 8)))
        d_h = int(getattr(cfg, "head_dim", int(cfg.hidden_size) // int(cfg.num_attention_heads)))
        kv_bytes_per_tok = 2 * n_l * n_kv * d_h * 2
    else:
        kv_bytes_per_tok = 0
    kv_mb_4k = (kv_bytes_per_tok * 4096) / (1024.0 * 1024.0)

    per_task: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    total_tokens = 0
    total_seconds = 0.0
    mode_counts: Counter[str] = Counter()
    repairs = 0

    for t in tasks:
        _sync(device)
        t0 = time.perf_counter()
        # Strict Zero-Oracle boundary: pass ONLY system_prompt + user_prompt
        reply, meta = omega_generate(
            model,
            tokenizer,
            system_prompt=t.system_prompt,
            user_prompt=t.user_prompt,
            max_new_tokens=t.max_new_tokens,
            device=device,
        )
        _sync(device)
        dt = max(time.perf_counter() - t0, 1e-6)
        mode_counts[meta["mode"]] += 1
        if meta["repaired"]:
            repairs += 1
        total_tokens += int(meta.get("tokens_generated", 0))
        total_seconds += dt
        latencies_ms.append(dt * 1000.0)

        verdict = verify_benchmark_response(t, reply)
        per_task.append(
            {
                "task_id": t.task_id,
                "dim": t.dim.value,
                "difficulty": t.difficulty,
                "passed": bool(verdict.passed),
                "score": float(verdict.score),
                "reason": verdict.reason,
                "controller_mode": meta["mode"],
                "controller_calls": meta["calls"],
                "repaired": meta["repaired"],
                "repair_reason": meta["reason"],
                "latency_ms": round(dt * 1000.0, 2),
                "response_preview": reply[:200],
            }
        )

    peak_mb = _peak_mb(device)
    lat_sorted = sorted(latencies_ms)
    p50_ms = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
    p95_ms = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))] if lat_sorted else 0.0
    tps = total_tokens / max(total_seconds, 1e-6)

    outcomes = [1.0 if r["passed"] else 0.0 for r in per_task]
    mean_acc, ci_lo, ci_hi = bootstrap_accuracy_ci(outcomes)
    by_dim: dict[str, Any] = {}
    for d in CAPABILITY_DIMENSIONS:
        sub = [r for r in per_task if r["dim"] == d.value]
        p_n = sum(1 for r in sub if r["passed"])
        by_dim[d.value] = {
            "passed": p_n,
            "total": len(sub),
            "accuracy": round(p_n / max(len(sub), 1), 4),
            "mean_score": round(sum(r["score"] for r in sub) / max(len(sub), 1), 4),
        }

    eff = compute_efficiency_score(
        capability_accuracy=mean_acc,
        total_parameters=total_params,
        tokens_per_sec=tps,
        kv_cache_mb_at_4k=kv_mb_4k,
    )
    by_dim[BenchmarkDim.INFERENCE_EFFICIENCY.value] = {
        **eff,
        "p50_ms": round(p50_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "peak_vram_mb": peak_mb,
        "total_parameters": total_params,
    }
    failures = Counter(r["reason"] for r in per_task if not r["passed"])
    return {
        "arm": arm_name,
        "is_external_opponent": False,
        "total_parameters": total_params,
        "passed": int(sum(outcomes)),
        "total_tasks": len(per_task),
        "accuracy": mean_acc,
        "bootstrap_ci95": [ci_lo, ci_hi],
        "controller_modes": dict(mode_counts),
        "controller_repairs": repairs,
        "by_dimension": by_dim,
        "failure_taxonomy": dict(failures.most_common(12)),
        "tasks": per_task,
    }


def run_exp_omega3_unified_1p7b(
    *,
    device: torch.device,
    student_id: str = "Qwen/Qwen3-1.7B",
    out_dir: Path,
) -> dict[str, Any]:
    """Train & evaluate NanoLM-Omega-1.7B (Nullspace-LoRA + LieExp Sidecar + Zero-Oracle Controller)."""
    eval_tasks = build_benchmark_15d_suite()
    cur_v2 = build_curriculum_v2(seed=31415)
    rl_cur = build_challenging_rlvr_curriculum(seed=777)

    dec_sft = verify_zero_contamination(cur_v2, eval_tasks)
    dec_rl = verify_zero_contamination(rl_cur, eval_tasks)

    model, tok = _load_hf_model_and_tokenizer(student_id, device)
    sample = tok("def dijkstra(n, edges, src, dst):\n    import", return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        pre_lg = model(**sample).logits.float().clone()

    graft_info = apply_omega_graft(
        model,
        lora_config=LoRAConfig(rank=16, alpha=32.0),
        nullspace_k=48,
        n_sidecar_layers=8,
        n_registers=4,
        d_fiber=128,
    )
    model.eval()
    with torch.no_grad():
        post_lg = model(**sample).logits.float()
    step0_diff = float((pre_lg - post_lg).abs().max().item())
    if step0_diff > 1e-4:
        raise RuntimeError(f"Step-0 Omega graft equivalence broken: {step0_diff}")
    graft_info["step0_max_abs_logit_diff"] = step0_diff

    # Early-stop 2-epoch SFT preserves pretrained code & syntax while learning holonomy routing
    sft_stats = train_sft_curriculum(model, tok, cur_v2, device=device, epochs=2, lr=1.0e-4)
    rl_focus = [
        ex for ex in cur_v2
        if ex.dim in (BenchmarkDim.TOOL_USE, BenchmarkDim.PLANNING_DECOMPOSITION, BenchmarkDim.MEMORY_RETENTION, BenchmarkDim.CODING)
    ] + list(rl_cur)[:16]
    grpo_stats = _grpo_with_omega_ref(
        model,
        tok,
        rl_focus,
        device=device,
        config=GRPOConfig(rounds=1, group_size=4, max_new_tokens=224, lr=2.5e-5, kl_beta=0.10),
    )

    gate_norms = [
        float(m.sidecar_gate.detach().abs().mean().cpu().item())
        for m in model.modules()
        if isinstance(m, HSPAOmegaSidecar)
    ]

    # Measure adapter manifest + SHA256 reproducibility digest
    adapter_sd = lora_state_dict(model)
    h = hashlib.sha256()
    total_l2 = 0.0
    for k_name in sorted(adapter_sd):
        t = adapter_sd[k_name].float()
        total_l2 += float(t.pow(2).sum().item())
        h.update(k_name.encode("utf-8"))
        h.update(t.numpy().tobytes())
    manifest = {
        "model_id": student_id,
        "total_parameters": graft_info["total_parameters"],
        "trainable_parameters": graft_info["trainable_parameters"],
        "nullspace_k": 48,
        "adapter_tensor_count": len(adapter_sd),
        "adapter_l2_norm": round(math.sqrt(total_l2), 4),
        "adapter_sha256_prefix": h.hexdigest()[:16],
    }
    (out_dir / "nanolm_omega_1.7b_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    arm_greedy = _evaluate_15d_extended(
        model,
        tok,
        eval_tasks,
        device,
        arm_name="NanoLM_Omega_1.7B_Naked_Greedy",
        is_external_opponent=False,
        ttc_syntax_retries=0,
    )
    arm_cognitive = _evaluate_15d_omega_cognitive(
        model,
        tok,
        eval_tasks,
        device,
        arm_name="NanoLM_Omega_1.7B_Cognitive_System",
    )

    tmeta = {
        "graft": graft_info,
        "sft_early2": sft_stats,
        "grpo_omega": grpo_stats,
        "mean_abs_sidecar_gate": round(sum(gate_norms) / max(len(gate_norms), 1), 6),
        "manifest": manifest,
    }
    arm_greedy["training_stats"] = tmeta
    arm_cognitive["training_stats"] = tmeta

    print(
        f"[EXP-Omega3] Naked={arm_greedy['passed']}/56 ({arm_greedy['accuracy']:.4f}, "
        f"coding={arm_greedy['by_dimension']['coding']['passed']}/4) "
        f"| Cognitive={arm_cognitive['passed']}/56 ({arm_cognitive['accuracy']:.4f}, "
        f"coding={arm_cognitive['by_dimension']['coding']['passed']}/4, "
        f"planning={arm_cognitive['by_dimension']['planning_decomposition']['passed']}/4, "
        f"repairs={arm_cognitive['controller_repairs']})",
        flush=True,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": "EXP_Omega3_Unified_1.7B_Nullspace_Cognitive",
        "student_id": student_id,
        "decontamination": {"sft_v2": dec_sft, "rl": dec_rl},
        "arms": [arm_greedy, arm_cognitive],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NANOLM Omega Autonomous Discovery Runner")
    parser.add_argument("--mode", choices=("all", "o1", "o2", "o3"), default="all")
    parser.add_argument("--out-dir", type=str, default="artifacts/nanolm_omega_v1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    res: dict[str, Any] = {
        "meta": {
            "device": str(dev),
            "cuda_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "non-cuda",
            "torch_version": torch.__version__,
        }
    }

    if args.mode in ("all", "o1"):
        r1 = run_exp_omega1_s4_holonomy(device=dev)
        res["exp_omega1"] = r1
        (out_dir / "exp_omega1_s4_holonomy.json").write_text(json.dumps(r1, indent=2), encoding="utf-8")

    if args.mode in ("all", "o2"):
        r2 = run_exp_omega2_nullspace_continual(device=dev)
        res["exp_omega2"] = r2
        (out_dir / "exp_omega2_nullspace_continual.json").write_text(json.dumps(r2, indent=2), encoding="utf-8")

    if args.mode in ("all", "o3"):
        r3 = run_exp_omega3_unified_1p7b(device=dev, out_dir=out_dir)
        res["exp_omega3"] = r3
        (out_dir / "exp_omega3_unified_1p7b.json").write_text(json.dumps(r3, indent=2), encoding="utf-8")

    (out_dir / "omega_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"OMEGA_CAMPAIGN_COMPLETE out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
