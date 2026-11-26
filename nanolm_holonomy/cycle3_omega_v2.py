"""Cycle-3 NANOLM Omega-v2 Experiments (EXP-Omega4 through EXP-Omega7).

Validates on Free Kaggle GPU ($0.00) or authorized CUDA hosts:
1. EXP-Omega4: Full-LM unbroken S_4 non-abelian holonomy & 8x length extrapolation
   comparing B0_Transformer, B1_Abelian_SSM, Gen1_H2_PredCode, and Gen4_Omega_SO3_PredCode.
2. EXP-Omega5: 3-seed iso-parameter (~14.0M) multi-domain LM tournament with strict
   ExperimentRegistry promotion gating against Gen1_H2 (Cycle-1/2 best).
3. EXP-Omega6: Static CUDA-Graph stateful decode speedup on ~1.8B & ~3.84B (<=4B)
   Gen4 models + SpectralNullspaceLoRALinear zero-overhead weight folding microbenchmark.
4. EXP-Omega7: Unified NanoLM-Omega-v2-1.7B with merged inference + upgraded Zero-Oracle
   OmegaCognitiveController-v2 (margin gating, schema structure guard, Mermaid normalization,
   multi-probe Python falsification) on all 56 held-out 15D benchmark tasks.
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
import time
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_frontier.benchmark_15d import BenchmarkDim, build_benchmark_15d_suite
from nanolm_frontier.curriculum import verify_zero_contamination
from nanolm_frontier.surgery_and_rlvr import GRPOConfig, apply_chat_template_safe, train_sft_curriculum
from nanolm_frontier.tournament import _load_hf_model_and_tokenizer
from nanolm_holonomy.architecture import (
    HSPA_OMEGA_2B_CONFIG,
    HSPA_OMEGA_4B_CONFIG,
    HSPAFamily,
    HSPAUnifiedLM,
    build_suite_configs,
)
from nanolm_holonomy.cuda_graph_decode import StaticGraphHSPADecoder
from nanolm_holonomy.curriculum_v2 import build_curriculum_v2
from nanolm_holonomy.cycle2_experiments import (
    _evaluate_15d_extended,
    _reset_all_cuda_peaks,
    _unbroken_s4_batch,
)
from nanolm_holonomy.experiments import (
    MicroByteWordTokenizer,
    _autocast,
    _build_x2_corpus,
    _cuda_amp_dtype,
    _peak_mb,
    _sync,
)
from nanolm_holonomy.omega_architecture import (
    HSPAOmegaSidecar,
    SpectralNullspaceLoRALinear,
    apply_omega_graft,
    merged_omega_inference,
)
from nanolm_holonomy.omega_experiments import _evaluate_15d_omega_cognitive, _grpo_with_omega_ref
from nanolm_holonomy.registry import ExperimentRegistry, RegistryVersion
from nanolm_holonomy.sidecar_evolution import build_challenging_rlvr_curriculum
from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS, ParameterBudgetExceededError
from nanolm_posttrain.lora import LoRAConfig, lora_state_dict


# ---------------------------------------------------------------------------
# EXP-Omega4: Full-LM Unbroken S_4 State Tracking & 8x OOD Extrapolation
# ---------------------------------------------------------------------------

def run_exp_omega4_full_lm_s4(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (101, 202, 303),
    steps: int = 1_200,
    batch_size: int = 64,
    train_len: int = 32,
    lr: float = 2.0e-3,
    lengths: tuple[int, ...] = (32, 64, 128, 256),
    probe_batch: int = 96,
) -> dict[str, Any]:
    """Compare full-LM B0, B1, Gen1_H2, and Gen4_Omega_SO3 on unbroken S_4 to 8x OOD."""
    all_cfgs = {c.family: c for c in build_suite_configs(preset="small_x1", vocab_size=512, max_seq_len=512)}
    cfgs = (
        all_cfgs[HSPAFamily.B0_TRANSFORMER],
        all_cfgs[HSPAFamily.B1_ABELIAN_SSM],
        all_cfgs[HSPAFamily.GEN1_H2_HSPA_PREDCODE],
        all_cfgs[HSPAFamily.GEN4_OMEGA_SO3_PREDCODE],
    )
    eval_rng = random.Random(20261119)
    probes = {L: _unbroken_s4_batch(probe_batch, L, eval_rng, device) for L in lengths}

    arms: list[dict[str, Any]] = []
    for cfg in cfgs:
        seed_rows: list[dict[str, Any]] = []
        for seed in seeds:
            torch.manual_seed(seed)
            rng = random.Random(seed)
            model = HSPAUnifiedLM(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
            model.train()
            for _ in range(steps):
                x, y = _unbroken_s4_batch(batch_size, train_len, rng, device)
                opt.zero_grad(set_to_none=True)
                lg = model(x)
                loss = F.cross_entropy(lg.float().view(-1, cfg.vocab_size), y.view(-1), ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()

            model.eval()
            acc: dict[str, float] = {}
            with torch.no_grad():
                for L, (px, py) in probes.items():
                    plg = model(px)
                    preds = plg[:, -1, :].argmax(dim=-1)
                    acc[f"L_{L}"] = round(float((preds == py[:, -1]).float().mean().item()), 4)
            seed_rows.append({"seed": seed, "accuracy": acc})
            del model, opt, sch

        summary: dict[str, Any] = {}
        for L in lengths:
            k = f"L_{L}"
            vals = [r["accuracy"][k] for r in seed_rows]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
            summary[k] = {"mean": round(mu, 4), "std": round(sd, 4)}
        print(
            f"[EXP-Omega4] {cfg.name:32s} S4(32/64/128/256)="
            + "/".join(f"{summary[f'L_{L}']['mean']:.3f}" for L in lengths),
            flush=True,
        )
        arms.append({"config": cfg.parameter_breakdown(256), "summary": summary, "per_seed": seed_rows})
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "experiment": "EXP_Omega4_FullLM_S4_8xOOD",
        "seeds": list(seeds),
        "steps": steps,
        "train_len": train_len,
        "lengths": list(lengths),
        "arms": arms,
    }


# ---------------------------------------------------------------------------
# EXP-Omega5: 3-Seed Iso-Param (~14.0M) Gen4 LM Tournament
# ---------------------------------------------------------------------------

def run_exp_omega5_gen4_lm_evolution(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (42, 137, 271),
    train_steps: int = 800,
    batch_size: int = 16,
    lr: float = 8e-4,
) -> dict[str, Any]:
    """Compare B0_Transformer, Gen1_H2_PredCode, and Gen4_Omega_SO3 across 3 seeds."""
    tok = MicroByteWordTokenizer(vocab_size=4_096)
    corpus = _build_x2_corpus(tok)
    tr_seqs = corpus["train"]
    vals = corpus["val"]
    all_cfgs = {c.family: c for c in build_suite_configs(preset="micro", vocab_size=tok.vocab_size, max_seq_len=1_024)}
    cfgs = (
        all_cfgs[HSPAFamily.B0_TRANSFORMER],
        all_cfgs[HSPAFamily.GEN1_H2_HSPA_PREDCODE],
        all_cfgs[HSPAFamily.GEN4_OMEGA_SO3_PREDCODE],
    )

    registry = ExperimentRegistry(higher_is_better=False)
    reports: list[dict[str, Any]] = []

    for idx, cfg in enumerate(cfgs):
        seed_runs: list[dict[str, Any]] = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = HSPAUnifiedLM(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(train_steps, 1))
            use_scaler = device.type == "cuda" and _cuda_amp_dtype() == torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

            _reset_all_cuda_peaks(device)
            _sync(device)
            t0 = time.perf_counter()
            tokens_seen = 0
            n_tr = len(tr_seqs)
            model.train()
            for step in range(train_steps):
                s_i = (step * batch_size) % n_tr
                raw = tr_seqs[s_i : s_i + batch_size]
                if len(raw) < batch_size:
                    raw = raw + tr_seqs[: batch_size - len(raw)]
                b = torch.tensor(raw, dtype=torch.long, device=device)
                x, y = b[:, :-1], b[:, 1:]
                opt.zero_grad(set_to_none=True)
                with _autocast(device):
                    logits = model(x)
                    loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
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
                sch.step()
                tokens_seen += int(y.numel())
            _sync(device)
            train_s = max(time.perf_counter() - t0, 1e-6)
            peak_mb = _peak_mb(device)

            model.eval()
            dom_bpb: dict[str, float] = {}
            probe_acc: dict[str, float] = {}
            with torch.no_grad():
                for d_name, (vseqs, vbpt, vprobes) in vals.items():
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
                            if int(pl[0, -1, :].argmax().item()) == tgt_id:
                                hits += 1
                        probe_acc[d_name] = round(hits / len(vprobes), 4)

                prompt_tok = torch.tensor([tr_seqs[0][:128]], dtype=torch.long, device=device)
                _, st = model(prompt_tok, use_state=True)
                cur_tok = prompt_tok[:, -1:]
                _sync(device)
                td0 = time.perf_counter()
                for s_off in range(32):
                    with _autocast(device):
                        dl, st = model(cur_tok, past_states=st, use_state=True, past_len=128 + s_off)
                    cur_tok = dl[:, -1:, :].argmax(dim=-1)
                _sync(device)
                decode_ms = round((time.perf_counter() - td0) * 1000.0 / 32.0, 3)

            seed_runs.append(
                {
                    "seed": seed,
                    "mean_val_bpb": round(sum(dom_bpb.values()) / len(dom_bpb), 4),
                    "domain_bpb": dom_bpb,
                    "probe_exact_acc": probe_acc,
                    "train_tok_per_sec": round(tokens_seen / train_s, 1),
                    "decode_ms_per_tok": decode_ms,
                    "peak_vram_mb": peak_mb,
                }
            )
            del model, opt, sch
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        bpbs = [r["mean_val_bpb"] for r in seed_runs]
        mu = round(sum(bpbs) / len(bpbs), 4)
        sd = round(math.sqrt(sum((v - mu) ** 2 for v in bpbs) / len(bpbs)), 5)
        ver = RegistryVersion(
            name=cfg.name,
            family=cfg.family.value,
            total_parameters=cfg.parameter_count,
            mean_primary_metric=mu,
            seed_scores=bpbs,
            metrics={
                "std_val_bpb": sd,
                "mean_tok_per_sec": round(sum(r["train_tok_per_sec"] for r in seed_runs) / len(seed_runs), 1),
                "mean_peak_vram_mb": round(sum(r["peak_vram_mb"] for r in seed_runs) / len(seed_runs), 1),
                "mean_decode_ms_per_tok": round(sum(r["decode_ms_per_tok"] for r in seed_runs) / len(seed_runs), 3),
                "state_kb_at_8k": round(cfg.state_bytes_per_sequence(8192, 2) / 1024.0, 2),
                "seed_runs": seed_runs,
            },
        )
        if idx == 0:
            registry.register_baseline(ver)
            dec = {"promoted": True, "reason": "registered_baseline"}
        else:
            dec = registry.evaluate_candidate(ver, min_seeds=len(seeds), min_delta=0.0002)

        print(
            f"[EXP-Omega5] {cfg.name:30s} bpb={mu:.4f}+/-{sd:.5f} "
            f"tok/s={ver.metrics['mean_tok_per_sec']:.0f} vram={ver.metrics['mean_peak_vram_mb']}MB "
            f"reg={dec['reason']}",
            flush=True,
        )
        reports.append({"config": cfg.parameter_breakdown(8192), "metrics": ver.metrics, "bpb": mu, "std_bpb": sd, "registry_decision": dec})

    return {
        "experiment": "EXP_Omega5_Gen4_3Seed_LM",
        "seeds": list(seeds),
        "train_steps": train_steps,
        "arms": reports,
        "registry": registry.to_dict(),
    }


# ---------------------------------------------------------------------------
# EXP-Omega6: Static CUDA-Graph 2B/4B Decode + Nullspace LoRA Weight Merge
# ---------------------------------------------------------------------------

def run_exp_omega6_cudagraph_and_merge_speedup(
    *,
    device: torch.device,
    seq_len: int = 512,
    n_decode: int = 32,
    configs: tuple[Any, ...] | None = None,
    merge_shapes: tuple[tuple[int, int], ...] = ((2048, 2048), (2048, 6144)),
) -> dict[str, Any]:
    """Measure eager vs Static CUDA Graph decode at 2B/4B and NullspaceLoRA weight folding."""
    dtype = _cuda_amp_dtype() if device.type == "cuda" else torch.float32
    cfgs = configs if configs is not None else (
        replace(HSPA_OMEGA_2B_CONFIG, scan_chunk_size=512),
        replace(HSPA_OMEGA_4B_CONFIG, scan_chunk_size=512),
    )

    decode_rows: list[dict[str, Any]] = []
    for cfg in cfgs:
        if cfg.parameter_count > ABSOLUTE_MAX_PARAMETERS:
            raise ParameterBudgetExceededError(f"{cfg.name} exceeds 4B cap")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _reset_all_cuda_peaks(device)

        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                model = HSPAUnifiedLM(cfg)
        finally:
            torch.set_default_dtype(prev_dtype)
        model.eval()
        weight_mb = round(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024.0 * 1024.0), 1)
        prompt = torch.randint(0, cfg.vocab_size, (1, seq_len), device=device)

        # Warmup eager once before timing
        with torch.no_grad():
            _w_lg, _w_st = model(prompt[:, :min(32, seq_len)], use_state=True, logits_to_keep=1)
            model(_w_lg[:, -1:, :].argmax(dim=-1), past_states=_w_st, use_state=True, past_len=32, logits_to_keep=1)
            del _w_lg, _w_st

            _sync(device)
            st = None
            for cs in range(0, seq_len, 512):
                ce = min(seq_len, cs + 512)
                lg_c, st = model(prompt[:, cs:ce], past_states=st, use_state=True, past_len=cs, logits_to_keep=1)
            cur_e = lg_c[:, -1:, :].argmax(dim=-1)
            eager_toks: list[int] = [int(cur_e.item())]

            _sync(device)
            t0 = time.perf_counter()
            for step_i in range(n_decode):
                lg_d, st = model(cur_e, past_states=st, use_state=True, past_len=seq_len + step_i, logits_to_keep=1)
                cur_e = lg_d[:, -1:, :].argmax(dim=-1)
                eager_toks.append(int(cur_e.item()))
            _sync(device)
            eager_ms = round((time.perf_counter() - t0) * 1000.0 / max(n_decode, 1), 3)
            del st

        decoder = StaticGraphHSPADecoder(model, batch_size=1, device=device)
        graph_captured = decoder.capture_graph(warmup_steps=3)
        cur_g = decoder.prefill(prompt, chunk_size=512)
        graph_toks: list[int] = [int(cur_g.item())]

        _sync(device)
        tg0 = time.perf_counter()
        for _ in range(n_decode):
            lg_g = decoder.step(cur_g)
            cur_g = lg_g[:, -1:, :].argmax(dim=-1).clone()
            graph_toks.append(int(cur_g.item()))
        _sync(device)
        graph_ms = round((time.perf_counter() - tg0) * 1000.0 / max(n_decode, 1), 3)

        greedy_parity = (eager_toks == graph_toks)
        peak_mb = _peak_mb(device)
        speedup = round(eager_ms / max(graph_ms, 1e-6), 2)
        print(
            f"[EXP-Omega6] {cfg.name:24s} captured={graph_captured} "
            f"eager={eager_ms:.2f}ms/tok -> graph={graph_ms:.2f}ms/tok ({speedup}x) "
            f"parity={greedy_parity} vram={peak_mb}MB",
            flush=True,
        )
        decode_rows.append(
            {
                "config": cfg.parameter_breakdown(seq_len),
                "weight_mb": weight_mb,
                "seq_len": seq_len,
                "decode_steps": n_decode,
                "cuda_graph_captured": graph_captured,
                "eager_ms_per_tok": eager_ms,
                "cudagraph_ms_per_tok": graph_ms,
                "speedup_x": speedup,
                "greedy_parity": greedy_parity,
                "peak_vram_mb": peak_mb,
            }
        )
        del decoder, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Sub-benchmark B: SpectralNullspaceLoRALinear merge parity & latency
    merge_rows: list[dict[str, Any]] = []
    for in_f, out_f in merge_shapes:
        base = nn.Linear(in_f, out_f, bias=False).to(device=device, dtype=dtype)
        ns = SpectralNullspaceLoRALinear(base, rank=16, alpha=32.0, nullspace_k=48)
        with torch.no_grad():
            nn.init.normal_(ns.lora_b, std=0.02)
        x = torch.randn(1, 1, in_f, device=device, dtype=dtype)
        with torch.no_grad():
            y_unm = ns(x).clone()
            for _ in range(25):
                _ = ns(x)
            _sync(device)
            tu0 = time.perf_counter()
            for _ in range(250):
                _ = ns(x)
            _sync(device)
            unm_us = round((time.perf_counter() - tu0) * 1e6 / 250.0, 2)

            vk = ns.v_proj_subspace.float()
            ortho_err = float((ns.effective_delta_weight() @ vk).abs().max().item())

            ns.merge()
            y_mrg = ns(x).clone()
            for _ in range(25):
                _ = ns(x)
            _sync(device)
            tm0 = time.perf_counter()
            for _ in range(250):
                _ = ns(x)
            _sync(device)
            mrg_us = round((time.perf_counter() - tm0) * 1e6 / 250.0, 2)
            max_diff = round(float((y_unm.float() - y_mrg.float()).abs().max().item()), 6)
            ns.unmerge()
            speedup_gemm = round(unm_us / max(mrg_us, 1e-6), 2)
        merge_rows.append(
            {
                "shape": [in_f, out_f],
                "unmerged_us": unm_us,
                "merged_us": mrg_us,
                "speedup_x": speedup_gemm,
                "max_logit_diff": max_diff,
                "nullspace_orthogonality_error": round(ortho_err, 7),
            }
        )
        del ns, base
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": "EXP_Omega6_CUDAGraph_And_NullspaceMerge",
        "device": str(device),
        "decode_benchmarks": decode_rows,
        "nullspace_merge_benchmarks": merge_rows,
    }


# ---------------------------------------------------------------------------
# EXP-Omega7: Unified 1.75B Nullspace-Merged + Cognitive Controller-v2 (56 Tasks)
# ---------------------------------------------------------------------------

def _reset_sidecar_caches(model: nn.Module) -> None:
    for mod in model.modules():
        if isinstance(mod, HSPAOmegaSidecar):
            mod.reset_cache()


def run_exp_omega7_unified_v2_1p7b(
    *,
    device: torch.device,
    student_id: str = "Qwen/Qwen3-1.7B",
    out_dir: Path,
) -> dict[str, Any]:
    """Train NanoLM-Omega-v2-1.7B, audit merged vs unmerged speedup, and evaluate all 56 tasks."""
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

    gate_norms = [
        float(m.sidecar_gate.detach().abs().mean().cpu().item())
        for m in model.modules()
        if isinstance(m, HSPAOmegaSidecar)
    ]

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
    (out_dir / "nanolm_omega_v2_1.7b_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Audit merged vs unmerged throughput & greedy output parity on 6 short tasks
    audit_ids = {"cod-01", "mth-01", "sum-01", "jsn-01", "tol-01", "ins-01"}
    audit_tasks = [t for t in eval_tasks if t.task_id in audit_ids]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _run_audit_subset() -> tuple[list[str], float]:
        outs: list[str] = []
        tot_tok = 0
        tot_s = 0.0
        for t in audit_tasks:
            _reset_sidecar_caches(model)
            msgs = [{"role": "system", "content": t.system_prompt}, {"role": "user", "content": t.user_prompt}]
            ptxt = apply_chat_template_safe(tok, msgs, add_generation_prompt=True)
            enc = tok(ptxt, return_tensors="pt", add_special_tokens=False).to(device)
            plen = int(enc["input_ids"].shape[1])
            _sync(device)
            tg0 = time.perf_counter()
            with torch.no_grad():
                oid = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=t.max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                )
            _sync(device)
            tot_s += max(time.perf_counter() - tg0, 1e-6)
            gid = oid[0, plen:]
            tot_tok += int(gid.numel())
            outs.append(tok.decode(gid, skip_special_tokens=True).strip())
        return outs, round(tot_tok / max(tot_s, 1e-6), 2)

    unm_replies, unm_tps = _run_audit_subset()
    with merged_omega_inference(model) as merged_count:
        mrg_replies, mrg_tps = _run_audit_subset()
        exact_audit_matches = sum(1 for a, b in zip(unm_replies, mrg_replies) if a == b)
        merge_audit = {
            "merged_layers": merged_count,
            "audit_tasks": len(audit_tasks),
            "exact_greedy_output_matches": exact_audit_matches,
            "unmerged_tok_per_sec": unm_tps,
            "merged_tok_per_sec": mrg_tps,
            "throughput_speedup_x": round(mrg_tps / max(unm_tps, 1e-6), 2),
        }

        _reset_sidecar_caches(model)
        arm_greedy = _evaluate_15d_extended(
            model,
            tok,
            eval_tasks,
            device,
            arm_name="NanoLM_Omega_v2_1.7B_Merged_Greedy",
            is_external_opponent=False,
            ttc_syntax_retries=0,
        )
        _reset_sidecar_caches(model)
        arm_cognitive = _evaluate_15d_omega_cognitive(
            model,
            tok,
            eval_tasks,
            device,
            arm_name="NanoLM_Omega_v2_1.7B_Cognitive_v2",
        )

    tmeta = {
        "graft": graft_info,
        "sft_early2": sft_stats,
        "grpo_omega": grpo_stats,
        "mean_abs_sidecar_gate": round(sum(gate_norms) / max(len(gate_norms), 1), 6),
        "manifest": manifest,
        "merge_audit": merge_audit,
    }
    arm_greedy["training_stats"] = tmeta
    arm_cognitive["training_stats"] = tmeta

    print(
        f"[EXP-Omega7] MergeSpeedup={merge_audit['throughput_speedup_x']}x ({unm_tps}->{mrg_tps} tok/s, match={exact_audit_matches}/{len(audit_tasks)}) "
        f"| MergedGreedy={arm_greedy['passed']}/56 ({arm_greedy['accuracy']:.4f}) "
        f"| Cognitive_v2={arm_cognitive['passed']}/56 ({arm_cognitive['accuracy']:.4f}, "
        f"coding={arm_cognitive['by_dimension']['coding']['passed']}/4, "
        f"planning={arm_cognitive['by_dimension']['planning_decomposition']['passed']}/4, "
        f"mermaid={arm_cognitive['by_dimension']['diagrams_charts']['passed']}/4, "
        f"repairs={arm_cognitive['controller_repairs']})",
        flush=True,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": "EXP_Omega7_Unified_v2_1.7B_Merged_Cognitive",
        "student_id": student_id,
        "decontamination": {"sft_v2": dec_sft, "rl": dec_rl},
        "merge_audit": merge_audit,
        "arms": [arm_greedy, arm_cognitive],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NANOLM Omega-v2 Cycle-3 Experiment Suite")
    parser.add_argument("--mode", choices=("all", "o4", "o5", "o6", "o7"), default="all")
    parser.add_argument("--out-dir", type=str, default="artifacts/nanolm_omega_v2")
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

    if args.mode in ("all", "o4"):
        r4 = run_exp_omega4_full_lm_s4(device=dev)
        res["exp_omega4"] = r4
        (out_dir / "exp_omega4_full_lm_s4.json").write_text(json.dumps(r4, indent=2), encoding="utf-8")

    if args.mode in ("all", "o5"):
        r5 = run_exp_omega5_gen4_lm_evolution(device=dev)
        res["exp_omega5"] = r5
        (out_dir / "exp_omega5_gen4_lm.json").write_text(json.dumps(r5, indent=2), encoding="utf-8")

    if args.mode in ("all", "o6"):
        r6 = run_exp_omega6_cudagraph_and_merge_speedup(device=dev)
        res["exp_omega6"] = r6
        (out_dir / "exp_omega6_cudagraph_merge.json").write_text(json.dumps(r6, indent=2), encoding="utf-8")

    if args.mode in ("all", "o7"):
        r7 = run_exp_omega7_unified_v2_1p7b(device=dev, out_dir=out_dir)
        res["exp_omega7"] = r7
        (out_dir / "exp_omega7_unified_v2_1p7b.json").write_text(json.dumps(r7, indent=2), encoding="utf-8")

    # Preserve/merge any already-computed experiment JSONs in out_dir
    for key, fname in (
        ("exp_omega4", "exp_omega4_full_lm_s4.json"),
        ("exp_omega5", "exp_omega5_gen4_lm.json"),
        ("exp_omega6", "exp_omega6_cudagraph_merge.json"),
        ("exp_omega7", "exp_omega7_unified_v2_1p7b.json"),
    ):
        fpath = out_dir / fname
        if key not in res and fpath.exists():
            res[key] = json.loads(fpath.read_text(encoding="utf-8"))

    (out_dir / "omega_v2_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"OMEGA_V2_CAMPAIGN_COMPLETE out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
