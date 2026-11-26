"""Cycle-4 First-Principles Interaction Intelligence Suite (EXP-FP1 through EXP-FP4).

Executed under strict $0.00 RunPod spend guard on Free Kaggle 2x Tesla T4 GPU
(or local CPU/MPS smoke with ``--quick``):

1. ``EXP-FP1`` (Test A): Relational & Bitemporal Event/Slot Microcosm comparing
   ``D0_Deterministic``, ``B0_Flat_Transformer``, ``B1_Unsigned_GAT``, ``M1_Signed_Typed_Mesh``,
   ``M2_Signed_Sheaf_PH`` (Port-Hamiltonian SO(3) transport), and ``M3_Dynamic_Rewire_PH``
   across canonical, paraphrase, 6x OOD scale+renaming, tenant isolation, and u=0 Lyapunov decay.
2. ``EXP-FP2`` (Test B): Full-LM S_4 OOD + 3-seed ~14M multi-task sequence evolution +
   Copy-Gate ablation + AdamW vs Newton-Schulz Muon cross across ``B0_Transformer``,
   ``B2_Gated_DeltaNet``, ``Gen1_H2_PredCode`` (incumbent BEST), ``Gen4_Omega_SO3_PredCode``,
   and ``Gen5_FP_PH_Holonomy``.
3. ``EXP-FP3``: Quantitative 8-Primitive single-knockout matrix + Port-Hamiltonian VoI
   adaptive-compute Pareto frontier.
4. ``EXP-FP4`` (Test C): 1.747B (<= 4B) Compact LM integration & decontamination-verified
   comparison across ``D0_Deterministic``, ``Naked_Merged_Greedy``,
   ``EqualCompute_SelfConsistency``, and Zero-Oracle ``WorldController_v3`` on both
   ``benchmark_15d`` (56 tasks) and ``FP-5Axis-30`` (30 tasks).
"""

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
import re
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_frontier.benchmark_15d import (
    BenchmarkDim,
    CAPABILITY_DIMENSIONS,
    bootstrap_accuracy_ci,
    build_benchmark_15d_suite,
    compute_efficiency_score,
    verify_benchmark_response,
)
from nanolm_frontier.curriculum import verify_zero_contamination
from nanolm_frontier.surgery_and_rlvr import GRPOConfig, apply_chat_template_safe, train_sft_curriculum
from nanolm_frontier.tournament import _load_hf_model_and_tokenizer
from nanolm_holonomy.architecture import HSPAFamily, HSPAUnifiedLM, build_suite_configs
from nanolm_holonomy.compute_planner import plan_fp_v3_campaign
from nanolm_holonomy.curriculum_v2 import build_curriculum_v2
from nanolm_holonomy.cycle2_experiments import _evaluate_15d_extended, _reset_all_cuda_peaks, _unbroken_s4_batch
from nanolm_holonomy.experiments import MicroByteWordTokenizer, _autocast, _build_x2_corpus, _cuda_amp_dtype, _peak_mb, _sync
from nanolm_holonomy.first_principles import (
    DeterministicTypedRuleSolver,
    PRIMITIVES_8,
    PortHamiltonianInteractionNet,
    VoIHaltingPolicy,
    collate_microcosm_batch,
    evaluate_case_with_ledger,
    generate_microcosm_case,
)
from nanolm_holonomy.fp_benchmark import (
    FPAxis,
    FPBenchmarkTask,
    build_fp_5axis_benchmark,
    deterministic_fp_symbolic_baseline,
    verify_fp_response,
    verify_fp_zero_contamination,
)
from nanolm_holonomy.omega_architecture import HSPAOmegaSidecar, apply_omega_graft, merged_omega_inference
from nanolm_holonomy.omega_experiments import _grpo_with_omega_ref
from nanolm_holonomy.omega_system import _run_model_chat_with_margin, omega_fp_generate
from nanolm_holonomy.registry import ExperimentRegistry, RegistryVersion
from nanolm_holonomy.sidecar_evolution import build_challenging_rlvr_curriculum
from nanolm_posttrain.lora import LoRAConfig, lora_state_dict


# ---------------------------------------------------------------------------
# Newton-Schulz Muon + AdamW Hybrid Optimizer (Zero External Dependencies)
# ---------------------------------------------------------------------------

def _newton_schulz5(grad: Tensor, steps: int = 5) -> Tensor:
    """5-step Newton-Schulz polar orthogonalization for 2D weight updates."""
    assert grad.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.t()
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        aat = x @ x.t()
        b_mat = b * aat + c * (aat @ aat)
        x = a * x + b_mat @ x
    return x.t().to(grad.dtype) if transposed else x.to(grad.dtype)


class HybridMuonOptimizer:
    """Apply Newton-Schulz Muon to >=2D internal matrices and AdamW to embeddings/1D/scalars."""

    def __init__(self, model: nn.Module, *, lr: float = 8e-4, momentum: float = 0.92, weight_decay: float = 0.02) -> None:
        muon_params: list[Tensor] = []
        adam_params: list[Tensor] = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 2 and "tok_emb" not in name and min(p.shape) >= 8:
                muon_params.append(p)
            else:
                adam_params.append(p)
        self.muon_params = muon_params
        self.muon_lr = lr * 1.4
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._bufs: dict[int, Tensor] = {}
        self.adamw = torch.optim.AdamW(adam_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.05) if adam_params else None

    def zero_grad(self) -> None:
        for p in self.muon_params:
            p.grad = None
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self) -> None:
        for idx, p in enumerate(self.muon_params):
            if p.grad is None:
                continue
            g = p.grad.float()
            buf = self._bufs.get(idx)
            if buf is None:
                buf = torch.zeros_like(g)
                self._bufs[idx] = buf
            buf.mul_(self.momentum).add_(g)
            upd = g + self.momentum * buf
            orth = _newton_schulz5(upd, steps=5)
            scale = math.sqrt(max(1.0, p.shape[0] / max(p.shape[1], 1)))
            p.mul_(1.0 - self.muon_lr * self.weight_decay)
            p.add_(orth.to(p.dtype), alpha=-self.muon_lr * scale)
        if self.adamw is not None:
            self.adamw.step()


# ---------------------------------------------------------------------------
# EXP-FP1: Relational & Bitemporal Interaction Microcosm (Test A)
# ---------------------------------------------------------------------------

def _eval_deterministic_microcosm(cases: list[Any]) -> dict[str, float]:
    from nanolm_holonomy.first_principles import LABEL_TO_ID

    st_hits = 0
    joint_hits = 0
    for c in cases:
        res = DeterministicTypedRuleSolver.resolve_ledger(
            c.ledger,
            entity_id=c.query_entity,
            tenant_id=c.query_tenant,
            authorized_tenant_id=c.authorized_tenant,
            slot=c.query_slot,
            as_of_time=c.as_of_time,
        )
        st_id = LABEL_TO_ID[res["state"]]
        st_ok = st_id == c.target_state_id
        val_ok = (res["raw_value"] == c.target_raw_value) if c.target_raw_value else (res["raw_value"] == "")
        st_hits += int(st_ok)
        joint_hits += int(st_ok and val_ok)
    n = max(len(cases), 1)
    return {"state_acc": round(st_hits / n, 4), "joint_acc": round(joint_hits / n, 4)}


def _eval_neural_microcosm(
    net: PortHamiltonianInteractionNet,
    cases: list[Any],
    device: torch.device,
    *,
    batch_size: int = 8,
    n_steps: int | None = None,
    ablation: str | None = None,
) -> dict[str, float]:
    net.eval()
    st_hits = 0
    ptr_hits = 0
    joint_hits = 0
    with torch.no_grad():
        for cs in range(0, len(cases), batch_size):
            chunk = cases[cs : cs + batch_size]
            batch = collate_microcosm_batch(chunk, device)
            out = net.forward_trajectory(
                batch["text_ids"],
                batch["meta_feats"],
                batch["edge_types"],
                batch["p_mask"],
                batch["q_mask"],
                batch["slot_mask"],
                n_steps=n_steps,
                ablation=ablation,
            )
            p_st = out["state_logits"].argmax(dim=-1).cpu().tolist()
            p_ptr = out["ptr_logits"].argmax(dim=-1).cpu().tolist()
            for case, s_id, ptr_id in zip(chunk, p_st, p_ptr):
                ev = evaluate_case_with_ledger(case, int(s_id), int(ptr_id))
                st_hits += int(ev["state_ok"])
                ptr_hits += int(ev["ptr_ok"])
                joint_hits += int(ev["joint_ok"])
    n = max(len(cases), 1)
    return {
        "state_acc": round(st_hits / n, 4),
        "ptr_acc": round(ptr_hits / n, 4),
        "joint_acc": round(joint_hits / n, 4),
    }


def _train_microcosm_net(
    variant: str,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    device: torch.device,
) -> PortHamiltonianInteractionNet:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = PortHamiltonianInteractionNet(variant=variant, d_model=48, steps=4).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2.5e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    net.train()
    for step in range(steps):
        n_nodes = 16 if (step % 2 == 0) else 28
        cases = [generate_microcosm_case(rng, n_nodes=n_nodes, canonical=(i % 3 == 0), rename_scramble=(i % 2 == 1)) for i in range(batch_size)]
        batch = collate_microcosm_batch(cases, device)
        opt.zero_grad(set_to_none=True)
        st_lg, ptr_lg = net(batch)
        loss = F.cross_entropy(st_lg, batch["target_state"]) + 0.85 * F.cross_entropy(ptr_lg, batch["target_ptr"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
    return net


def run_exp_fp1_relational_microcosm(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (101, 202, 303),
    train_steps: int = 400,
    eval_cases: int = 70,
    ood_cases: int = 35,
    ood_nodes: int = 160,
) -> dict[str, Any]:
    """Compare D0 deterministic rules and 5 matched neural interaction kernels."""
    probe_rng = random.Random(20261120)
    canon_eval = [generate_microcosm_case(probe_rng, n_nodes=24, canonical=True, rename_scramble=True) for _ in range(eval_cases)]
    para_eval = [generate_microcosm_case(probe_rng, n_nodes=24, canonical=False, rename_scramble=True) for _ in range(eval_cases)]
    ood_eval = [generate_microcosm_case(probe_rng, n_nodes=ood_nodes, canonical=False, rename_scramble=True) for _ in range(ood_cases)]

    det_canon = _eval_deterministic_microcosm(canon_eval)
    det_para = _eval_deterministic_microcosm(para_eval)
    det_ood = _eval_deterministic_microcosm(ood_eval)
    print(f"[EXP-FP1] D0_Deterministic (0 params) canon={det_canon['joint_acc']:.3f} para={det_para['joint_acc']:.3f} ood={det_ood['joint_acc']:.3f}", flush=True)

    arms_out: list[dict[str, Any]] = [
        {
            "arm": "D0_Deterministic_Rules",
            "parameters": 0,
            "clean_canonical": det_canon,
            "semantic_paraphrase": det_para,
            "ood_scale_rename": det_ood,
        }
    ]

    variants = ("flat_transformer", "unsigned_gat", "signed_mesh", "sheaf_port_hamiltonian", "dynamic_rewire_ph")
    for variant in variants:
        seed_rows: list[dict[str, Any]] = []
        max_lyap_delta = 0.0
        for seed in seeds:
            net = _train_microcosm_net(variant, seed=seed, steps=train_steps, batch_size=24, device=device)
            p_count = net.num_parameters
            _sync(device)
            t0 = time.perf_counter()
            ev_can = _eval_neural_microcosm(net, canon_eval, device)
            ev_par = _eval_neural_microcosm(net, para_eval, device)
            ev_ood = _eval_neural_microcosm(net, ood_eval, device, batch_size=4, n_steps=8)
            _sync(device)
            eval_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            if variant in ("sheaf_port_hamiltonian", "dynamic_rewire_ph"):
                b_lyap = collate_microcosm_batch(para_eval[:8], device)
                with torch.no_grad():
                    tr = net.forward_trajectory(
                        b_lyap["text_ids"], b_lyap["meta_feats"], b_lyap["edge_types"],
                        b_lyap["p_mask"], b_lyap["q_mask"], b_lyap["slot_mask"],
                        n_steps=10, autonomous_drive_zero=True,
                    )
                    de = float((tr["energies"][:, 1:] - tr["energies"][:, :-1]).max().item())
                    max_lyap_delta = max(max_lyap_delta, de)

            seed_rows.append({
                "seed": seed,
                "parameters": p_count,
                "eval_ms": eval_ms,
                "clean_canonical": ev_can,
                "semantic_paraphrase": ev_par,
                "ood_scale_rename": ev_ood,
            })
            del net
            if device.type == "cuda":
                torch.cuda.empty_cache()

        def _agg(reg_k: str, metric: str = "joint_acc") -> dict[str, float]:
            vals = [r[reg_k][metric] for r in seed_rows]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
            return {"mean": round(mu, 4), "std": round(sd, 4)}

        summary = {
            "arm": variant,
            "parameters": seed_rows[0]["parameters"],
            "clean_canonical": {"joint_acc": _agg("clean_canonical", "joint_acc"), "state_acc": _agg("clean_canonical", "state_acc")},
            "semantic_paraphrase": {"joint_acc": _agg("semantic_paraphrase", "joint_acc"), "state_acc": _agg("semantic_paraphrase", "state_acc")},
            "ood_scale_rename": {"joint_acc": _agg("ood_scale_rename", "joint_acc"), "state_acc": _agg("ood_scale_rename", "state_acc")},
            "autonomous_lyapunov_max_energy_delta": round(max_lyap_delta, 7) if variant in ("sheaf_port_hamiltonian", "dynamic_rewire_ph") else None,
            "per_seed": seed_rows,
        }
        print(
            f"[EXP-FP1] {variant:24s} ({summary['parameters']:,}p) "
            f"canon={summary['clean_canonical']['joint_acc']['mean']:.3f} "
            f"para={summary['semantic_paraphrase']['joint_acc']['mean']:.3f} "
            f"ood({ood_nodes}n)={summary['ood_scale_rename']['joint_acc']['mean']:.3f}",
            flush=True,
        )
        arms_out.append(summary)

    return {
        "experiment": "EXP_FP1_Relational_And_Bitemporal_Microcosm",
        "seeds": list(seeds),
        "train_steps": train_steps,
        "ood_nodes": ood_nodes,
        "arms": arms_out,
    }


# ---------------------------------------------------------------------------
# EXP-FP2: Gen5 Architecture Sequence Evolution + DeltaNet + Muon Cross (Test B)
# ---------------------------------------------------------------------------

def _build_fp2_corpus(
    tok: MicroByteWordTokenizer,
    *,
    train_seqs: int = 2_400,
    val_seqs_per_dom: int = 72,
    seq_len: int = 256,
) -> dict[str, Any]:
    """Extend _build_x2_corpus with open-vocab copy and bitemporal probes."""
    base = _build_x2_corpus(tok, train_seqs=train_seqs, val_seqs_per_dom=val_seqs_per_dom, seq_len=seq_len)
    rng = random.Random(90901)

    extra_train: list[list[int]] = []
    copy_probes: list[tuple[list[int], int]] = []
    bit_probes: list[tuple[list[int], int]] = []

    key_marker, eq_marker, sep_marker, qry_marker = 3, 4, 5, 6
    asof_marker, verified_marker, unverified_marker = 7, 8, 9
    n_copy_tr = max(8, int(round(train_seqs * 0.145)))
    n_copy_pr = min(70, max(8, val_seqs_per_dom))

    for i in range(n_copy_tr + n_copy_pr):
        seq = [tok.bos_id]
        pairs: list[tuple[int, int]] = []
        for _ in range(8):
            k_id = rng.randint(1200, tok.vocab_size - 1)
            v_id = rng.randint(1200, tok.vocab_size - 1)
            pairs.append((k_id, v_id))
            seq.extend([key_marker, k_id, eq_marker, v_id, sep_marker])
        qk, qv = rng.choice(pairs)
        seq.extend([qry_marker, qk, eq_marker])
        if i < n_copy_tr:
            seq.append(qv)
            while len(seq) < seq_len + 1:
                seq.extend([sep_marker, rng.randint(1200, tok.vocab_size - 1)])
            extra_train.append(seq[: seq_len + 1])
        else:
            copy_probes.append((seq, qv))

    for i in range(n_copy_tr + n_copy_pr):
        val_old = rng.randint(100, 900)
        val_new = rng.randint(100, 900)
        while val_new == val_old:
            val_new = rng.randint(100, 900)
        slot_id = rng.randint(20, 60)
        is_future_hidden = (i % 2 == 0)
        as_of_tok = 13 if is_future_hidden else 16
        seq = [
            tok.bos_id, slot_id, verified_marker, 12, eq_marker, val_old, sep_marker,
            slot_id, (unverified_marker if is_future_hidden else verified_marker), 15, eq_marker, val_new,
            asof_marker, as_of_tok, eq_marker,
        ]
        target = val_old if is_future_hidden else val_new
        if i < n_copy_tr:
            seq.append(target)
            while len(seq) < seq_len + 1:
                seq.extend([sep_marker, rng.randint(100, 900)])
            extra_train.append(seq[: seq_len + 1])
        else:
            bit_probes.append((seq, target))

    base["train"] = list(base["train"]) + extra_train
    random.Random(12345).shuffle(base["train"])
    base["copy_probes"] = copy_probes
    base["bit_probes"] = bit_probes
    return base


def _eval_lm_sequence_model(
    model: HSPAUnifiedLM,
    cfg: Any,
    corpus: dict[str, Any],
    device: torch.device,
    *,
    batch_size: int = 16,
) -> dict[str, Any]:
    model.eval()
    dom_bpb: dict[str, float] = {}
    probe_acc: dict[str, float] = {}
    confs: list[float] = []
    hits_all: list[float] = []

    with torch.no_grad():
        for d_name, (vseqs, vbpt, vprobes) in corpus["val"].items():
            tot_l = 0.0
            tot_tok = 0
            for j in range(0, len(vseqs), batch_size):
                vb = torch.tensor(vseqs[j : j + batch_size], dtype=torch.long, device=device)
                with _autocast(device):
                    vl = model(vb[:, :-1])
                l_sum = F.cross_entropy(vl.float().reshape(-1, cfg.vocab_size), vb[:, 1:].reshape(-1), reduction="sum")
                tot_l += float(l_sum.item())
                tot_tok += int(vb[:, 1:].numel())
            dom_bpb[d_name] = round((tot_l / max(tot_tok, 1)) / (vbpt * math.log(2.0)), 4)

            if vprobes:
                hits = 0
                for pr_ids, tgt_id in vprobes:
                    px = torch.tensor([pr_ids], dtype=torch.long, device=device)
                    with _autocast(device):
                        pl = model(px)
                    probs = F.softmax(pl[0, -1, :].float(), dim=-1)
                    pred = int(probs.argmax().item())
                    ok = int(pred == tgt_id)
                    hits += ok
                    confs.append(float(probs[pred].item()))
                    hits_all.append(float(ok))
                probe_acc[d_name] = round(hits / len(vprobes), 4)

        for p_name in ("copy_probes", "bit_probes"):
            plist = corpus[p_name]
            if not plist:
                probe_acc[p_name] = 0.0
                continue
            px_all = torch.tensor([p[0] for p in plist], dtype=torch.long, device=device)
            tgt_all = [p[1] for p in plist]
            with _autocast(device):
                pl_all = model(px_all)
            probs_all = F.softmax(pl_all[:, -1, :].float(), dim=-1)
            preds_all = probs_all.argmax(dim=-1).cpu().tolist()
            p_confs = probs_all.max(dim=-1).values.cpu().tolist()
            hits = 0
            for pred, tgt_id, conf in zip(preds_all, tgt_all, p_confs):
                ok = int(pred == tgt_id)
                hits += ok
                confs.append(float(conf))
                hits_all.append(float(ok))
            probe_acc[p_name] = round(hits / len(plist), 4)

    # Compute 10-bin Expected Calibration Error (ECE) over probes
    ece = 0.0
    if confs:
        n_c = len(confs)
        for b in range(10):
            lo, hi = b / 10.0, (b + 1) / 10.0
            idxs = [i for i, c in enumerate(confs) if (lo <= c < hi or (b == 9 and c == 1.0))]
            if idxs:
                avg_c = sum(confs[i] for i in idxs) / len(idxs)
                avg_a = sum(hits_all[i] for i in idxs) / len(idxs)
                ece += (len(idxs) / n_c) * abs(avg_a - avg_c)

    return {
        "mean_val_bpb": round(sum(dom_bpb.values()) / len(dom_bpb), 4),
        "domain_bpb": dom_bpb,
        "probe_acc": probe_acc,
        "ece": round(ece, 4),
    }


def _train_sequence_arm(
    cfg: Any,
    corpus: dict[str, Any],
    *,
    seed: int,
    train_steps: int,
    batch_size: int,
    lr: float,
    optimizer_kind: str,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = HSPAUnifiedLM(cfg).to(device)
    use_scaler = device.type == "cuda" and _cuda_amp_dtype() == torch.float16 and optimizer_kind == "adamw"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if optimizer_kind == "muon":
        opt: Any = HybridMuonOptimizer(model, lr=lr)
        sch = None
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(train_steps, 1))

    tr_seqs = corpus["train"]
    n_tr = len(tr_seqs)
    _reset_all_cuda_peaks(device)
    _sync(device)
    t0 = time.perf_counter()
    tokens_seen = 0
    model.train()
    for step in range(train_steps):
        s_i = (step * batch_size) % n_tr
        raw = tr_seqs[s_i : s_i + batch_size]
        if len(raw) < batch_size:
            raw = raw + tr_seqs[: batch_size - len(raw)]
        b = torch.tensor(raw, dtype=torch.long, device=device)
        x, y = b[:, :-1], b[:, 1:]
        opt.zero_grad()
        with _autocast(device):
            lg = model(x)
            loss = F.cross_entropy(lg.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
        if use_scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if sch is not None:
            sch.step()
        tokens_seen += int(y.numel())

    _sync(device)
    train_s = max(time.perf_counter() - t0, 1e-6)
    peak_mb = _peak_mb(device)
    ev = _eval_lm_sequence_model(model, cfg, corpus, device, batch_size=batch_size)
    del model, opt, sch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "seed": seed,
        "optimizer": optimizer_kind,
        **ev,
        "tok_per_sec": round(tokens_seen / train_s, 1),
        "peak_vram_mb": peak_mb,
    }


def run_exp_fp2_gen5_sequence_evolution(
    *,
    device: torch.device,
    s4_seeds: tuple[int, ...] = (101, 202),
    s4_steps: int = 800,
    lm_seeds: tuple[int, ...] = (42, 137, 271),
    lm_steps: int = 600,
    muon_steps: int = 400,
    batch_size: int = 16,
    lm_preset: str = "micro",
    train_seqs: int = 2_400,
    val_seqs_per_dom: int = 72,
) -> dict[str, Any]:
    """Run S_4 full-LM comparison, multi-seed ~14M LM evolution + CopyGate ablation, and Muon cross."""
    # Part 2A: S_4 unbroken causal LM check
    s4_cfgs = {c.family: c for c in build_suite_configs(preset="small_x1", vocab_size=512, max_seq_len=512)}
    s4_arms = (
        s4_cfgs[HSPAFamily.B0_TRANSFORMER],
        s4_cfgs[HSPAFamily.B2_GATED_DELTANET],
        s4_cfgs[HSPAFamily.GEN1_H2_HSPA_PREDCODE],
        s4_cfgs[HSPAFamily.GEN4_OMEGA_SO3_PREDCODE],
        s4_cfgs[HSPAFamily.GEN5_FP_PH_HOLONOMY],
    )
    lengths = (32, 64, 128, 256)
    eval_rng = random.Random(20261121)
    s4_probes = {L: _unbroken_s4_batch(64, L, eval_rng, device) for L in lengths}
    s4_results: list[dict[str, Any]] = []

    for cfg in s4_arms:
        srows: list[dict[str, Any]] = []
        for seed in s4_seeds:
            torch.manual_seed(seed)
            rng = random.Random(seed)
            model = HSPAUnifiedLM(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(s4_steps, 1))
            model.train()
            for _ in range(s4_steps):
                x, y = _unbroken_s4_batch(48, 32, rng, device)
                opt.zero_grad(set_to_none=True)
                lg = model(x)
                loss = F.cross_entropy(lg.float().view(-1, cfg.vocab_size), y.view(-1), ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
            model.eval()
            acc = {}
            with torch.no_grad():
                for L, (px, py) in s4_probes.items():
                    preds = model(px)[:, -1, :].argmax(dim=-1)
                    acc[f"L_{L}"] = round(float((preds == py[:, -1]).float().mean().item()), 4)
            srows.append({"seed": seed, "accuracy": acc})
            del model, opt, sch
        summary = {
            f"L_{L}": round(sum(r["accuracy"][f"L_{L}"] for r in srows) / len(srows), 4)
            for L in lengths
        }
        print(f"[EXP-FP2A S4] {cfg.name:30s} " + "/".join(f"L{L}={summary[f'L_{L}']:.3f}" for L in lengths), flush=True)
        s4_results.append({"name": cfg.name, "params": cfg.parameter_count, "summary": summary, "per_seed": srows})
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Part 2B: 3-seed ~14M Multi-Task LM Evolution + ExperimentRegistry + CopyGate Ablation
    tok = MicroByteWordTokenizer(vocab_size=4_096)
    corpus = _build_fp2_corpus(tok, train_seqs=train_seqs, val_seqs_per_dom=val_seqs_per_dom)
    micro_cfgs = {c.family: c for c in build_suite_configs(preset=lm_preset, vocab_size=tok.vocab_size, max_seq_len=1_024)}
    gen5_cfg = micro_cfgs[HSPAFamily.GEN5_FP_PH_HOLONOMY]
    gen5_nocopy = replace(gen5_cfg, name="gen5-fp-nocopy-micro", use_copy_gate=False)
    lm_arms = (
        micro_cfgs[HSPAFamily.B0_TRANSFORMER],
        micro_cfgs[HSPAFamily.B2_GATED_DELTANET],
        micro_cfgs[HSPAFamily.GEN1_H2_HSPA_PREDCODE],
        micro_cfgs[HSPAFamily.GEN4_OMEGA_SO3_PREDCODE],
        gen5_cfg,
        gen5_nocopy,
    )

    registry = ExperimentRegistry(higher_is_better=False)
    lm_reports: list[dict[str, Any]] = []
    for idx, cfg in enumerate(lm_arms):
        seed_runs = [
            _train_sequence_arm(
                cfg, corpus, seed=sd, train_steps=lm_steps, batch_size=batch_size, lr=8e-4, optimizer_kind="adamw", device=device
            )
            for sd in lm_seeds
        ]
        bpbs = [r["mean_val_bpb"] for r in seed_runs]
        mu_bpb = round(sum(bpbs) / len(bpbs), 4)
        sd_bpb = round(math.sqrt(sum((v - mu_bpb) ** 2 for v in bpbs) / len(bpbs)), 5)

        def _mean_p(pk: str) -> float:
            vals = [r["probe_acc"].get(pk, 0.0) for r in seed_runs]
            return round(sum(vals) / len(vals), 4)

        ver = RegistryVersion(
            name=cfg.name,
            family=cfg.family.value,
            total_parameters=cfg.parameter_count,
            mean_primary_metric=mu_bpb,
            seed_scores=bpbs,
            metrics={
                "std_val_bpb": sd_bpb,
                "open_vocab_copy_acc": _mean_p("copy_probes"),
                "bitemporal_probe_acc": _mean_p("bit_probes"),
                "math_probe_acc": _mean_p("math"),
                "state_probe_acc": _mean_p("state"),
                "ece": round(sum(r["ece"] for r in seed_runs) / len(seed_runs), 4),
                "tok_per_sec": round(sum(r["tok_per_sec"] for r in seed_runs) / len(seed_runs), 1),
                "peak_vram_mb": round(max(r["peak_vram_mb"] for r in seed_runs), 1),
                "seed_runs": seed_runs,
            },
        )
        if idx == 0:
            registry.register_baseline(ver)
            dec = {"promoted": True, "reason": "registered_baseline"}
        else:
            dec = registry.evaluate_candidate(ver, min_seeds=len(lm_seeds), min_delta=0.0002)

        print(
            f"[EXP-FP2B LM] {cfg.name:30s} bpb={mu_bpb:.4f}+/-{sd_bpb:.5f} "
            f"copy={ver.metrics['open_vocab_copy_acc']:.3f} bit={ver.metrics['bitemporal_probe_acc']:.3f} "
            f"math={ver.metrics['math_probe_acc']:.3f} reg={dec['reason']}",
            flush=True,
        )
        lm_reports.append({"config": cfg.parameter_breakdown(8192), "metrics": ver.metrics, "bpb": mu_bpb, "registry_decision": dec})

    # Part 2C: Optimizer Cross (Newton-Schulz Muon, 2 seeds, matched steps)
    m_steps = max(4, min(lm_steps, muon_steps))
    muon_seeds = lm_seeds[:2]
    muon_reports: list[dict[str, Any]] = []
    for cfg in (
        micro_cfgs[HSPAFamily.B0_TRANSFORMER],
        micro_cfgs[HSPAFamily.B2_GATED_DELTANET],
        micro_cfgs[HSPAFamily.GEN1_H2_HSPA_PREDCODE],
        gen5_cfg,
    ):
        m_runs = [
            _train_sequence_arm(
                cfg, corpus, seed=sd, train_steps=m_steps, batch_size=batch_size, lr=8e-4, optimizer_kind="muon", device=device
            )
            for sd in muon_seeds
        ]
        mu_bpb = round(sum(r["mean_val_bpb"] for r in m_runs) / len(m_runs), 4)
        mu_copy = round(sum(r["probe_acc"].get("copy_probes", 0.0) for r in m_runs) / len(m_runs), 4)
        print(f"[EXP-FP2C Muon] {cfg.name:30s} bpb={mu_bpb:.4f} copy={mu_copy:.3f}", flush=True)
        muon_reports.append({"name": cfg.name, "optimizer": "muon_hybrid", "mean_val_bpb": mu_bpb, "open_vocab_copy_acc": mu_copy, "per_seed": m_runs})

    return {
        "experiment": "EXP_FP2_Gen5_Sequence_Evolution",
        "s4_results": s4_results,
        "lm_results": lm_reports,
        "registry": registry.to_dict(),
        "muon_cross": muon_reports,
    }


# ---------------------------------------------------------------------------
# EXP-FP3: 8-Primitive Single-Knockout Matrix & VoI Compute Pareto Curve
# ---------------------------------------------------------------------------

def run_exp_fp3_primitives_and_voi(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (101, 202, 303),
    train_steps: int = 350,
    eval_cases: int = 64,
) -> dict[str, Any]:
    """Quantify each of the 8 computational primitives and calibrated VoI halting."""
    probe_rng = random.Random(20261122)
    mixed_eval = [generate_microcosm_case(probe_rng, n_nodes=24, canonical=(i % 3 == 0), rename_scramble=True) for i in range(eval_cases)]

    knockout_names = (
        "full_u_t",
        "no_state",
        "no_identity_p_ij",
        "no_relation_edges",
        "no_transition_clock_q_ij",
        "no_info_source_ledger",
        "no_constraint_dissipation",
        "no_adaptation_weights",
        "no_objective_voi",
    )
    ko_scores: dict[str, list[dict[str, float]]] = {k: [] for k in knockout_names}
    voi_rows: list[dict[str, Any]] = []

    for seed in seeds:
        net = _train_microcosm_net("sheaf_port_hamiltonian", seed=seed, steps=train_steps, batch_size=24, device=device)
        untrained = PortHamiltonianInteractionNet(variant="sheaf_port_hamiltonian", d_model=48, steps=4).to(device)

        for ko in knockout_names:
            if ko == "full_u_t":
                res = _eval_neural_microcosm(net, mixed_eval, device)
            elif ko == "no_adaptation_weights":
                res = _eval_neural_microcosm(untrained, mixed_eval, device)
            elif ko == "no_objective_voi":
                res = _eval_neural_microcosm(net, mixed_eval, device, n_steps=1)
            elif ko == "no_constraint_dissipation":
                res = _eval_neural_microcosm(net, mixed_eval, device, n_steps=12, ablation=ko)
            else:
                res = _eval_neural_microcosm(net, mixed_eval, device, ablation=ko)
            ko_scores[ko].append(res)

        # VoI Pareto evaluation on first seed + averaged across seeds
        batch = collate_microcosm_batch(mixed_eval, device)
        with torch.no_grad():
            for fixed_k in (1, 2, 4, 8):
                tr_k = net.forward_trajectory(
                    batch["text_ids"], batch["meta_feats"], batch["edge_types"],
                    batch["p_mask"], batch["q_mask"], batch["slot_mask"], n_steps=fixed_k,
                )
                p_st = tr_k["state_logits"].argmax(dim=-1)
                p_pt = tr_k["ptr_logits"].argmax(dim=-1)
                j_acc = float(((p_st == batch["target_state"]) & (p_pt == batch["target_ptr"])).float().mean().item())
                voi_rows.append({"seed": seed, "policy": f"fixed_k={fixed_k}", "mean_steps": float(fixed_k), "joint_acc": round(j_acc, 4)})

            for lam in (0.18, 0.28, 0.42):
                pol = VoIHaltingPolicy(compute_lambda=lam)
                tr_v = net.forward_trajectory(
                    batch["text_ids"], batch["meta_feats"], batch["edge_types"],
                    batch["p_mask"], batch["q_mask"], batch["slot_mask"], n_steps=8, voi_policy=pol,
                )
                p_st = tr_v["state_logits"].argmax(dim=-1)
                p_pt = tr_v["ptr_logits"].argmax(dim=-1)
                j_acc = float(((p_st == batch["target_state"]) & (p_pt == batch["target_ptr"])).float().mean().item())
                m_steps = float(tr_v["used_steps"].float().mean().item())
                voi_rows.append({"seed": seed, "policy": f"voi_lambda={lam}", "mean_steps": round(m_steps, 3), "joint_acc": round(j_acc, 4)})

        del net, untrained
        if device.type == "cuda":
            torch.cuda.empty_cache()

    knockout_summary: dict[str, Any] = {}
    full_joint = sum(r["joint_acc"] for r in ko_scores["full_u_t"]) / len(seeds)
    for ko in knockout_names:
        vals = [r["joint_acc"] for r in ko_scores[ko]]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
        delta = mu - full_joint
        knockout_summary[ko] = {
            "joint_acc_mean": round(mu, 4),
            "joint_acc_std": round(sd, 4),
            "delta_vs_full": round(delta, 4),
        }
        print(f"[EXP-FP3 KO] {ko:28s} joint={mu:.4f}+/-{sd:.4f} (delta={delta:+.4f})", flush=True)

    # Aggregate VoI Pareto across seeds
    voi_by_pol: dict[str, list[dict[str, Any]]] = {}
    for row in voi_rows:
        voi_by_pol.setdefault(row["policy"], []).append(row)
    voi_summary = [
        {
            "policy": pol,
            "mean_steps": round(sum(x["mean_steps"] for x in rows) / len(rows), 3),
            "joint_acc": round(sum(x["joint_acc"] for x in rows) / len(rows), 4),
        }
        for pol, rows in voi_by_pol.items()
    ]
    print("[EXP-FP3 VoI] " + " | ".join(f"{r['policy']}: {r['joint_acc']:.3f}@{r['mean_steps']:.2f}st" for r in voi_summary), flush=True)

    return {
        "experiment": "EXP_FP3_Eight_Primitives_And_VoI_Pareto",
        "primitives_8": list(PRIMITIVES_8),
        "seeds": list(seeds),
        "knockouts": knockout_summary,
        "voi_pareto": voi_summary,
    }


# ---------------------------------------------------------------------------
# EXP-FP4: 1.747B Compact LM Integration (benchmark_15d + FP-5Axis-30)
# ---------------------------------------------------------------------------

def _reset_sidecar_caches(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, HSPAOmegaSidecar):
            mod.reset_cache()


def _evaluate_fp_5axis(
    tasks: tuple[FPBenchmarkTask, ...],
    *,
    arm_name: str,
    solver_fn: Any,
    total_parameters: int,
    device: torch.device,
) -> dict[str, Any]:
    _reset_all_cuda_peaks(device)
    per_task: list[dict[str, Any]] = []
    tot_tokens = 0
    tot_calls = 0
    tot_s = 0.0

    for t in tasks:
        _sync(device)
        t0 = time.perf_counter()
        reply, meta = solver_fn(t)
        _sync(device)
        dt = max(time.perf_counter() - t0, 1e-6)
        verdict = verify_fp_response(t, reply)
        tot_tokens += int(meta.get("tokens_generated", 0))
        tot_calls += int(meta.get("calls", 1))
        tot_s += dt
        per_task.append({
            "task_id": t.task_id,
            "axis": t.axis.value,
            "passed": bool(verdict.passed),
            "score": float(verdict.score),
            "reason": verdict.reason,
            "calls": int(meta.get("calls", 1)),
            "mode": str(meta.get("mode", "direct")),
            "response_preview": reply[:160],
        })

    outcomes = [1.0 if r["passed"] else 0.0 for r in per_task]
    mean_acc, ci_lo, ci_hi = bootstrap_accuracy_ci(outcomes)
    by_axis: dict[str, Any] = {}
    for ax in FPAxis:
        sub = [r for r in per_task if r["axis"] == ax.value]
        p_n = sum(1 for r in sub if r["passed"])
        by_axis[ax.value] = {"passed": p_n, "total": len(sub), "accuracy": round(p_n / max(len(sub), 1), 4)}

    return {
        "arm": arm_name,
        "total_parameters": total_parameters,
        "passed": int(sum(outcomes)),
        "total_tasks": len(per_task),
        "accuracy": mean_acc,
        "bootstrap_ci95": [ci_lo, ci_hi],
        "total_model_calls": tot_calls,
        "total_tokens": tot_tokens,
        "wall_seconds": round(tot_s, 2),
        "peak_vram_mb": _peak_mb(device),
        "by_axis": by_axis,
        "failure_taxonomy": dict(Counter(r["reason"] for r in per_task if not r["passed"]).most_common(12)),
        "tasks": per_task,
    }


def run_exp_fp4_compact_lm_integration(
    *,
    device: torch.device,
    student_id: str = "Qwen/Qwen3-1.7B",
    out_dir: Path,
) -> dict[str, Any]:
    """Evaluate D0, Naked Greedy, Equal-Compute Self-Consistency, and Controller-v3 on 86 total tasks."""
    eval_15d = build_benchmark_15d_suite()
    eval_fp30 = build_fp_5axis_benchmark()
    cur_v2 = build_curriculum_v2(seed=31415)
    rl_cur = build_challenging_rlvr_curriculum(seed=777)

    dec_15d_sft = verify_zero_contamination(cur_v2, eval_15d)
    dec_15d_rl = verify_zero_contamination(rl_cur, eval_15d)
    corpus_texts = [ex.system_prompt + "\n" + ex.user_prompt + "\n" + ex.reference_response for ex in list(cur_v2) + list(rl_cur)]
    dec_fp30 = verify_fp_zero_contamination(corpus_texts, eval_fp30)

    # 1. Zero-parameter deterministic baseline on FP-5Axis-30
    fp_det = _evaluate_fp_5axis(
        eval_fp30,
        arm_name="D0_Deterministic_Symbolic",
        solver_fn=lambda t: (deterministic_fp_symbolic_baseline(t), {"calls": 0, "tokens_generated": 0, "mode": "deterministic"}),
        total_parameters=0,
        device=device,
    )

    model, tok = _load_hf_model_and_tokenizer(student_id, device)
    sample = tok("def resolve_as_of(events, cutoff):\n    return", return_tensors="pt").to(device)
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
    _reset_sidecar_caches(model)
    model.eval()
    with torch.no_grad():
        post_lg = model(**sample).logits.float()
    step0_diff = float((pre_lg - post_lg).abs().max().item())
    if step0_diff > 1e-4:
        raise RuntimeError(f"Step-0 Omega graft equivalence broken: {step0_diff}")
    graft_info["step0_max_abs_logit_diff"] = step0_diff

    model_cfg = getattr(model, "config", None)
    prev_use_cache = getattr(model_cfg, "use_cache", True) if model_cfg is not None else True
    if model_cfg is not None:
        model_cfg.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    sft_stats = train_sft_curriculum(model, tok, cur_v2, device=device, epochs=2, lr=1.0e-4)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if model_cfg is not None:
        model_cfg.use_cache = True
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
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if model_cfg is not None:
        model_cfg.use_cache = prev_use_cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
    (out_dir / "nanolm_fp_v3_1.7b_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with merged_omega_inference(model) as merged_layers:
        # --- Benchmark 1: FP-5Axis-30 ---
        def _naked_solver(t: FPBenchmarkTask) -> tuple[str, dict[str, Any]]:
            _reset_sidecar_caches(model)
            msgs = [{"role": "system", "content": t.system_prompt}, {"role": "user", "content": t.user_prompt}]
            rep, n_tok, _ = _run_model_chat_with_margin(model, tok, msgs, max_new_tokens=t.max_new_tokens, sample=False)
            return rep, {"calls": 1, "tokens_generated": n_tok, "mode": "naked_greedy"}

        def _self_consistency_solver(t: FPBenchmarkTask) -> tuple[str, dict[str, Any]]:
            cands: list[str] = []
            tot_n = 0
            msgs = [{"role": "system", "content": t.system_prompt}, {"role": "user", "content": t.user_prompt}]
            for k_i in range(3):
                _reset_sidecar_caches(model)
                rep, n_tok, _ = _run_model_chat_with_margin(
                    model, tok, msgs, max_new_tokens=t.max_new_tokens, sample=(k_i > 0), temperature=0.35
                )
                cands.append(rep)
                tot_n += n_tok
            counts = Counter(c.strip() for c in cands)
            winner = counts.most_common(1)[0][0]
            return winner, {"calls": 3, "tokens_generated": tot_n, "mode": "self_consistency_3"}

        def _ctrl_v3_fp_solver(t: FPBenchmarkTask) -> tuple[str, dict[str, Any]]:
            _reset_sidecar_caches(model)
            return omega_fp_generate(
                model, tok, system_prompt=t.system_prompt, user_prompt=t.user_prompt, max_new_tokens=t.max_new_tokens, device=device
            )

        fp_naked = _evaluate_fp_5axis(eval_fp30, arm_name="NanoLM_1.7B_Naked_Merged_Greedy", solver_fn=_naked_solver, total_parameters=graft_info["total_parameters"], device=device)
        fp_sc3 = _evaluate_fp_5axis(eval_fp30, arm_name="NanoLM_1.7B_EqualCompute_SC3", solver_fn=_self_consistency_solver, total_parameters=graft_info["total_parameters"], device=device)
        fp_v3 = _evaluate_fp_5axis(eval_fp30, arm_name="NanoLM_1.7B_WorldController_v3", solver_fn=_ctrl_v3_fp_solver, total_parameters=graft_info["total_parameters"], device=device)

        # --- Benchmark 2: Full 56-task 15D suite with Controller-v3 (CODE_PYTHON symbol contract + DAG critical path) ---
        _reset_sidecar_caches(model)
        _reset_all_cuda_peaks(device)
        t_rows_15d: list[dict[str, Any]] = []
        tot_tok_15 = 0
        tot_s_15 = 0.0
        lats_15: list[float] = []
        repairs_15 = 0
        modes_15: Counter[str] = Counter()
        for t in eval_15d:
            _reset_sidecar_caches(model)
            _sync(device)
            t0 = time.perf_counter()
            reply, meta = omega_fp_generate(
                model, tok, system_prompt=t.system_prompt, user_prompt=t.user_prompt, max_new_tokens=t.max_new_tokens, device=device
            )
            _sync(device)
            dt = max(time.perf_counter() - t0, 1e-6)
            modes_15[meta["mode"]] += 1
            if meta["repaired"]:
                repairs_15 += 1
            tot_tok_15 += int(meta.get("tokens_generated", 0))
            tot_s_15 += dt
            lats_15.append(dt * 1000.0)
            v = verify_benchmark_response(t, reply)
            t_rows_15d.append({
                "task_id": t.task_id,
                "dim": t.dim.value,
                "passed": bool(v.passed),
                "score": float(v.score),
                "reason": v.reason,
                "mode": meta["mode"],
                "repaired": meta["repaired"],
                "repair_reason": meta["reason"],
            })

        out_15 = [1.0 if r["passed"] else 0.0 for r in t_rows_15d]
        acc15, ci_lo15, ci_hi15 = bootstrap_accuracy_ci(out_15)
        by_dim15 = {}
        for d in CAPABILITY_DIMENSIONS:
            sub = [r for r in t_rows_15d if r["dim"] == d.value]
            p_n = sum(1 for r in sub if r["passed"])
            by_dim15[d.value] = {"passed": p_n, "total": len(sub), "accuracy": round(p_n / max(len(sub), 1), 4)}
        eff15 = compute_efficiency_score(
            capability_accuracy=acc15,
            total_parameters=graft_info["total_parameters"],
            tokens_per_sec=tot_tok_15 / max(tot_s_15, 1e-6),
            kv_cache_mb_at_4k=448.0,
        )
        arm_15d_v3 = {
            "arm": "NanoLM_FP_v3_1.7B_Controller_v3",
            "total_parameters": graft_info["total_parameters"],
            "passed": int(sum(out_15)),
            "total_tasks": len(t_rows_15d),
            "accuracy": acc15,
            "bootstrap_ci95": [ci_lo15, ci_hi15],
            "controller_modes": dict(modes_15),
            "controller_repairs": repairs_15,
            "by_dimension": {**by_dim15, BenchmarkDim.INFERENCE_EFFICIENCY.value: {**eff15, "peak_vram_mb": _peak_mb(device)}},
            "failure_taxonomy": dict(Counter(r["reason"] for r in t_rows_15d if not r["passed"]).most_common(12)),
            "tasks": t_rows_15d,
        }

    print(
        f"[EXP-FP4 FP-5Axis-30] D0={fp_det['passed']}/30 | Naked={fp_naked['passed']}/30 | "
        f"EqualSC3={fp_sc3['passed']}/30 | WorldCtrl_v3={fp_v3['passed']}/30 ({fp_v3['accuracy']:.4f})",
        flush=True,
    )
    print(
        f"[EXP-FP4 15D-56] Controller_v3={arm_15d_v3['passed']}/56 ({acc15:.4f}, "
        f"coding={by_dim15['coding']['passed']}/4, planning={by_dim15['planning_decomposition']['passed']}/4, "
        f"diagrams={by_dim15['diagrams_charts']['passed']}/4, repairs={repairs_15})",
        flush=True,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": "EXP_FP4_1.7B_WorldController_v3",
        "student_id": student_id,
        "merged_layers": merged_layers,
        "decontamination": {"15d_sft": dec_15d_sft, "15d_rl": dec_15d_rl, "fp30": dec_fp30},
        "training_stats": {"graft": graft_info, "sft_early2": sft_stats, "grpo": grpo_stats, "manifest": manifest},
        "fp_5axis_30_arms": [fp_det, fp_naked, fp_sc3, fp_v3],
        "benchmark_15d_56": arm_15d_v3,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NANOLM FP-v3 Cycle-4 First-Principles Experiment Suite")
    parser.add_argument("--mode", choices=("all", "fp1", "fp2", "fp3", "fp4", "fp123", "fp23"), default="all")
    parser.add_argument("--out-dir", type=str, default="artifacts/nanolm_fp_v3")
    parser.add_argument("--quick", action="store_true", help="Fast smoke test profile for local M3/CPU verification")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    plan = plan_fp_v3_campaign(allow_kaggle=True)
    (out_dir / "preflight_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    res: dict[str, Any] = {
        "meta": {
            "device": str(dev),
            "cuda_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "non-cuda",
            "torch_version": torch.__version__,
            "quick_smoke": bool(args.quick),
        }
    }

    if args.mode in ("all", "fp1", "fp123"):
        r1 = (
            run_exp_fp1_relational_microcosm(device=dev, seeds=(101,), train_steps=16, eval_cases=16, ood_cases=8, ood_nodes=48)
            if args.quick
            else run_exp_fp1_relational_microcosm(device=dev)
        )
        res["exp_fp1"] = r1
        (out_dir / "exp_fp1_microcosm.json").write_text(json.dumps(r1, indent=2), encoding="utf-8")

    if args.mode in ("all", "fp2", "fp123", "fp23"):
        r2 = (
            run_exp_fp2_gen5_sequence_evolution(device=dev, s4_seeds=(101,), s4_steps=6, lm_seeds=(42,), lm_steps=4, muon_steps=4, batch_size=2, lm_preset="small_x1", train_seqs=32, val_seqs_per_dom=4)
            if args.quick
            else run_exp_fp2_gen5_sequence_evolution(device=dev)
        )
        res["exp_fp2"] = r2
        (out_dir / "exp_fp2_gen5_lm.json").write_text(json.dumps(r2, indent=2), encoding="utf-8")

    if args.mode in ("all", "fp3", "fp123", "fp23"):
        r3 = (
            run_exp_fp3_primitives_and_voi(device=dev, seeds=(101,), train_steps=16, eval_cases=16)
            if args.quick
            else run_exp_fp3_primitives_and_voi(device=dev)
        )
        res["exp_fp3"] = r3
        (out_dir / "exp_fp3_primitives_voi.json").write_text(json.dumps(r3, indent=2), encoding="utf-8")

    if args.mode in ("all", "fp4") and not args.quick:
        r4 = run_exp_fp4_compact_lm_integration(device=dev, out_dir=out_dir)
        res["exp_fp4"] = r4
        (out_dir / "exp_fp4_1p7b_integration.json").write_text(json.dumps(r4, indent=2), encoding="utf-8")

    for key, fname in (
        ("exp_fp1", "exp_fp1_microcosm.json"),
        ("exp_fp2", "exp_fp2_gen5_lm.json"),
        ("exp_fp3", "exp_fp3_primitives_voi.json"),
        ("exp_fp4", "exp_fp4_1p7b_integration.json"),
    ):
        fpath = out_dir / fname
        if key not in res and fpath.exists():
            res[key] = json.loads(fpath.read_text(encoding="utf-8"))

    (out_dir / "fp_v3_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"NANOLM_FP_V3_CAMPAIGN_COMPLETE out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
