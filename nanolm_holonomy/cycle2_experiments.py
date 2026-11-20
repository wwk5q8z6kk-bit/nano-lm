"""Cycle-2 Experiments: Fused Triton HSPA-v2, Unbroken S4 Grokking, 32K 4B Scaling, and 1.75B vs 8.19B."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch import Tensor
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
from nanolm_frontier.surgery_and_rlvr import GRPOConfig, apply_chat_template_safe, train_sft_curriculum
from nanolm_frontier.tournament import _load_hf_model_and_tokenizer
from nanolm_holonomy.architecture import (
    HSPA_4B_CONFIG,
    TRANSFORMER_4B_CONFIG,
    HSPAArchConfig,
    HSPAFamily,
    HSPAUnifiedLM,
    build_suite_configs,
)
from nanolm_holonomy.curriculum_v2 import build_curriculum_v2
from nanolm_holonomy.experiments import (
    _BOS_TOK,
    _GEN_OFFSET,
    _GEN_PERMS,
    _PERM_TO_ID,
    _STATE_OFFSET,
    _autocast,
    _build_x2_corpus,
    _measure_scaling_row,
    _compose_perm,
    _cuda_amp_dtype,
    _peak_mb,
    _sync,
    MicroByteWordTokenizer,
)
from nanolm_holonomy.registry import ExperimentRegistry, RegistryVersion
from nanolm_holonomy.sidecar_evolution import (
    HolonomicRegisterSidecar,
    _grpo_with_holonomy_ref,
    apply_holonomic_graft,
    build_challenging_rlvr_curriculum,
)
from nanolm_holonomy.spinor_algebra import cayley_rotor, parallel_holonomy_scan
from nanolm_holonomy.triton_holonomy import HAS_TRITON, holonomy_scan_fast
from nanolm_posttrain.architecture import ABSOLUTE_MAX_PARAMETERS, ParameterBudgetExceededError
from nanolm_posttrain.eval_suite import _extract_fenced_or_raw
from nanolm_posttrain.lora import LoRAConfig, lora_state_dict


def _reset_all_cuda_peaks(device: torch.device) -> None:
    if device.type == "cuda":
        for di in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(di)


# ---------------------------------------------------------------------------
# C2-X0: Direct Kernel Microbenchmark (PyTorch Blelloch vs Fused Triton Scan)
# ---------------------------------------------------------------------------

def run_c2_x0_kernel_microbench(
    *,
    device: torch.device,
    seq_lengths: tuple[int, ...] = (256, 512, 1024, 2048, 4096),
    batch_size: int = 4,
    n_registers: int = 8,
    n_spinors: int = 48,
    iters: int = 6,
) -> dict[str, Any]:
    """Compare Cycle-1 PyTorch Blelloch scan vs Cycle-2 Fused Triton + Analytical Adjoint."""
    rows: list[dict[str, Any]] = []
    for slen in seq_lengths:
        if device.type == "cuda":
            torch.cuda.empty_cache()

        def _bench_variant(use_triton: bool) -> dict[str, Any]:
            g_base = cayley_rotor(
                torch.randn(batch_size, n_registers, n_spinors, slen, 3, device=device) * 0.25
            ).detach()
            s_base = torch.randn(batch_size, n_registers, n_spinors, slen, 4, device=device) * 0.1
            init_base = torch.randn(batch_size, n_registers, n_spinors, 4, device=device) * 0.1

            def _step() -> None:
                g = g_base.clone().requires_grad_(True)
                s = s_base.clone().requires_grad_(True)
                init = init_base.clone().requires_grad_(True)
                if use_triton:
                    out, _ = holonomy_scan_fast(g, s, init_state=init, chunk_size=None)
                else:
                    out, _ = parallel_holonomy_scan(g, s, init_state=init, chunk_size=None)
                out.sum().backward()

            try:
                _step()
                _sync(device)
                _reset_all_cuda_peaks(device)
                t0 = time.perf_counter()
                for _ in range(iters):
                    _step()
                _sync(device)
                dt_ms = round((time.perf_counter() - t0) * 1000.0 / iters, 3)
                peak_mb = _peak_mb(device)
                return {"status": "ok", "fwd_bwd_ms": dt_ms, "peak_vram_mb": peak_mb}
            except RuntimeError as exc:
                return {"status": f"err:{str(exc)[:100]}", "fwd_bwd_ms": None, "peak_vram_mb": None}

        pt_res = _bench_variant(False)
        tr_res = _bench_variant(True)
        speedup = (
            round(pt_res["fwd_bwd_ms"] / max(tr_res["fwd_bwd_ms"], 1e-6), 2)
            if pt_res["fwd_bwd_ms"] and tr_res["fwd_bwd_ms"]
            else None
        )
        rows.append(
            {
                "seq_len": slen,
                "pytorch_blelloch": pt_res,
                "triton_holonomy": tr_res,
                "speedup_x": speedup,
            }
        )
        print(
            f"[C2-X0] seq={slen:4d} pt={pt_res['fwd_bwd_ms']}ms ({pt_res['peak_vram_mb']}MB) "
            f"triton={tr_res['fwd_bwd_ms']}ms ({tr_res['peak_vram_mb']}MB) speedup={speedup}x",
            flush=True,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "experiment": "C2_X0_kernel_microbenchmark",
        "has_triton": HAS_TRITON,
        "shape": {"B": batch_size, "R": n_registers, "G": n_spinors, "d_fiber": 4 * n_spinors},
        "iters": iters,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# C2-X1: Unbroken S_4 Non-Abelian Group Word Problem & 8x OOD Extrapolation
# ---------------------------------------------------------------------------

def _unbroken_s4_batch(batch_size: int, n_gens: int, rng: random.Random, device: torch.device) -> tuple[Tensor, Tensor]:
    """Dense supervision on unbroken chains: input `[BOS, g_1, ..., g_T]`, target `[-100, s_1, ..., s_T]`."""
    xs = [[_BOS_TOK] * (n_gens + 1) for _ in range(batch_size)]
    ys = [[-100] * (n_gens + 1) for _ in range(batch_size)]
    for b in range(batch_size):
        state = (0, 1, 2, 3)
        for t in range(1, n_gens + 1):
            gi = rng.randrange(len(_GEN_PERMS))
            state = _compose_perm(_GEN_PERMS[gi], state)
            xs[b][t] = _GEN_OFFSET + gi
            ys[b][t] = _STATE_OFFSET + _PERM_TO_ID[state]
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)


def run_c2_x1_s4_grokking(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (101, 202, 303),
    steps: int = 1_200,
    batch_size: int = 64,
    train_len: int = 32,
) -> dict[str, Any]:
    """Test non-abelian S_4 state tracking & OOD extrapolation up to 8x (256 tokens)."""
    all_cfgs = {c.family: c for c in build_suite_configs(preset="small_x1", vocab_size=512, max_seq_len=512)}
    cfgs = (
        all_cfgs[HSPAFamily.B0_TRANSFORMER],
        all_cfgs[HSPAFamily.B1_ABELIAN_SSM],
        all_cfgs[HSPAFamily.GEN1_H2_HSPA_PREDCODE],
        all_cfgs[HSPAFamily.GEN3_H4_HSPA_WAVEPRED],
    )

    eval_rng = random.Random(202611)
    lengths = (32, 64, 128, 256)
    probes = {L: _unbroken_s4_batch(96, L, eval_rng, device) for L in lengths}

    arms: list[dict[str, Any]] = []
    for cfg in cfgs:
        seed_rows: list[dict[str, Any]] = []
        for seed in seeds:
            torch.manual_seed(seed)
            rng = random.Random(seed)
            model = HSPAUnifiedLM(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
            use_scaler = device.type == "cuda" and _cuda_amp_dtype() == torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
            model.train()
            for _ in range(steps):
                x, y = _unbroken_s4_batch(batch_size, train_len, rng, device)
                opt.zero_grad(set_to_none=True)
                with _autocast(device):
                    lg = model(x)
                    loss = F.cross_entropy(lg.float().view(-1, cfg.vocab_size), y.view(-1), ignore_index=-100)
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

            model.eval()
            acc: dict[str, float] = {}
            with torch.no_grad():
                for L, (px, py) in probes.items():
                    with _autocast(device):
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
            f"[C2-X1] {cfg.name:30s} S4(L32/64/128/256)="
            + "/".join(f"{summary[f'L_{L}']['mean']:.3f}" for L in lengths),
            flush=True,
        )
        arms.append({"config": cfg.parameter_breakdown(256), "summary": summary, "per_seed": seed_rows})
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"experiment": "C2_X1_unbroken_s4_grokking", "seeds": list(seeds), "train_len": train_len, "arms": arms}


# ---------------------------------------------------------------------------
# C2-X2: 3-Seed Iso-Param LM Evolution & Triton Throughput Benchmark
# ---------------------------------------------------------------------------

def run_c2_x2_triton_lm(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (42, 137, 271),
    train_steps: int = 800,
    batch_size: int = 16,
    lr: float = 8e-4,
) -> dict[str, Any]:
    """Compare B0_Transformer, Gen1_H2_PredCode (Cycle-1 BEST), and Gen3_H4_WavePred across 3 seeds."""
    tok = MicroByteWordTokenizer(vocab_size=4_096)
    corpus = _build_x2_corpus(tok)
    tr_seqs = corpus["train"]
    vals = corpus["val"]
    all_cfgs = {c.family: c for c in build_suite_configs(preset="micro", vocab_size=tok.vocab_size, max_seq_len=1_024)}
    cfgs = (
        all_cfgs[HSPAFamily.B0_TRANSFORMER],
        all_cfgs[HSPAFamily.GEN1_H2_HSPA_PREDCODE],
        all_cfgs[HSPAFamily.GEN3_H4_HSPA_WAVEPRED],
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
            f"[C2-X2] {cfg.name:28s} bpb={mu:.4f}+/-{sd:.5f} "
            f"tok/s={ver.metrics['mean_tok_per_sec']:.0f} vram={ver.metrics['mean_peak_vram_mb']}MB "
            f"dec_ms={ver.metrics['mean_decode_ms_per_tok']:.2f} reg={dec['reason']}",
            flush=True,
        )
        reports.append({"config": cfg.parameter_breakdown(8192), "metrics": ver.metrics, "bpb": mu, "std_bpb": sd, "registry_decision": dec})

    return {"experiment": "C2_X2_triton_3seed_lm", "seeds": list(seeds), "train_steps": train_steps, "arms": reports, "registry": registry.to_dict()}


# ---------------------------------------------------------------------------
# C2-X3: 3.845B <= 4B Ultra-Long-Context Scaling up to 32,768 Tokens
# ---------------------------------------------------------------------------

def run_c2_x3_32k_scaling(
    *,
    device: torch.device,
    seq_lengths: tuple[int, ...] = (512, 2048, 8192, 16384, 32768),
) -> dict[str, Any]:
    """Benchmark Transformer-4B vs HSPA-4B (Gen-3 Triton, chunked prefill 512) up to 32,768 tokens."""
    dtype = _cuda_amp_dtype() if device.type == "cuda" else torch.float32
    cfgs = (TRANSFORMER_4B_CONFIG, replace(HSPA_4B_CONFIG, scan_chunk_size=512))
    out: list[dict[str, Any]] = []

    for cfg in cfgs:
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                model = HSPAUnifiedLM(cfg)
        finally:
            torch.set_default_dtype(prev_dtype)
        model.eval()
        weight_mb = round(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024.0 * 1024.0), 1)

        rows: list[dict[str, Any]] = []
        for slen in seq_lengths:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _reset_all_cuda_peaks(device)
            try:
                rows.append(_measure_scaling_row(model, cfg, slen, device, prefill_chunk=512, logits_to_keep=1))
            except RuntimeError as exc:
                rows.append({"seq_len": slen, "status": f"oom_or_err:{str(exc)[:120]}"})

        print(
            f"[C2-X3] {cfg.name:24s} rows={[(r['seq_len'], r.get('decode_ms_per_tok'), r.get('peak_vram_mb')) for r in rows]}",
            flush=True,
        )
        out.append({"config": cfg.parameter_breakdown(max(seq_lengths)), "weight_mb": weight_mb, "scaling": rows})
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"experiment": "C2_X3_32k_4b_scaling", "device": str(device), "models": out}


# ---------------------------------------------------------------------------
# C2-X4: 1.75B NanoLM-Holonomic-v2 vs 8.19B External Giant Qwen3-8B
# ---------------------------------------------------------------------------

def is_syntax_valid_zero_oracle(system_prompt: str, response: str) -> tuple[bool, str]:
    """Check strictly syntactic validity (Python AST or JSON) without touching task ``spec``.

    Only triggers on prompts whose system instruction explicitly mandates Python or JSON.
    """
    low = system_prompt.casefold()
    raw = _extract_fenced_or_raw(response)
    if "python" in low:
        try:
            ast.parse(raw)
            return True, "ok_python"
        except SyntaxError:
            return False, "syntax_python"
    if "json" in low:
        try:
            obj = json.loads(raw)
            return isinstance(obj, dict), "ok_json" if isinstance(obj, dict) else "json_not_dict"
        except json.JSONDecodeError:
            return False, "syntax_json"
    return True, "not_applicable"


def _evaluate_15d_extended(
    model: Any,
    tokenizer: Any,
    tasks: tuple[BenchmarkTask, ...],
    device: torch.device,
    *,
    arm_name: str,
    is_external_opponent: bool = False,
    ttc_syntax_retries: int = 0,
) -> dict[str, Any]:
    """Evaluate any model across all 15 dimensions, optionally enabling zero-oracle syntax TTC."""
    model.eval()
    _reset_all_cuda_peaks(device)
    first_dev = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    total_params = sum(p.numel() for p in model.parameters())
    if not is_external_opponent and total_params > ABSOLUTE_MAX_PARAMETERS:
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
    total_gen_tokens = 0
    total_gen_seconds = 0.0
    ttc_triggers = 0
    ttc_recoveries = 0

    for t in tasks:
        msgs = [{"role": "system", "content": t.system_prompt}, {"role": "user", "content": t.user_prompt}]
        prompt_text = apply_chat_template_safe(tokenizer, msgs, add_generation_prompt=True)
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(first_dev)
        prompt_len = int(enc["input_ids"].shape[1])

        def _gen(sample: bool) -> tuple[str, int, float]:
            _sync(device)
            tg0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=t.max_new_tokens,
                    do_sample=sample,
                    temperature=0.45 if sample else None,
                    top_p=0.95 if sample else None,
                    pad_token_id=pad_id,
                )
            _sync(device)
            dt_s = max(time.perf_counter() - tg0, 1e-6)
            g_ids = out_ids[0, prompt_len:]
            return tokenizer.decode(g_ids, skip_special_tokens=True).strip(), int(g_ids.numel()), dt_s

        reply, n_tok, dt = _gen(sample=False)
        attempts = 1
        if ttc_syntax_retries > 0:
            ok_syn, _ = is_syntax_valid_zero_oracle(t.system_prompt, reply)
            if not ok_syn:
                ttc_triggers += 1
                for _ in range(ttc_syntax_retries):
                    attempts += 1
                    r2, nt2, dt2 = _gen(sample=True)
                    n_tok += nt2
                    dt += dt2
                    ok2, _ = is_syntax_valid_zero_oracle(t.system_prompt, r2)
                    if ok2:
                        reply = r2
                        ttc_recoveries += 1
                        break

        total_gen_tokens += n_tok
        total_gen_seconds += dt
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
                "attempts": attempts,
                "latency_ms": round(dt * 1000.0, 2),
                "response_preview": reply[:200],
            }
        )

    peak_mb = _peak_mb(device)
    lat_sorted = sorted(latencies_ms)
    p50_ms = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0
    p95_ms = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))] if lat_sorted else 0.0
    tps = total_gen_tokens / max(total_gen_seconds, 1e-6)

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

    by_diff = {
        dn: {
            "passed": sum(1 for r in per_task if r["difficulty"] == dn and r["passed"]),
            "total": sum(1 for r in per_task if r["difficulty"] == dn),
            "accuracy": round(
                sum(1 for r in per_task if r["difficulty"] == dn and r["passed"])
                / max(sum(1 for r in per_task if r["difficulty"] == dn), 1),
                4,
            ),
        }
        for dn in ("easy", "medium", "hard")
    }
    failures = Counter(r["reason"] for r in per_task if not r["passed"])
    return {
        "arm": arm_name,
        "is_external_opponent": is_external_opponent,
        "ttc_syntax_retries": ttc_syntax_retries,
        "ttc_triggers": ttc_triggers,
        "ttc_syntax_recoveries": ttc_recoveries,
        "total_parameters": total_params,
        "passed": int(sum(outcomes)),
        "total_tasks": len(per_task),
        "accuracy": mean_acc,
        "bootstrap_ci95": [ci_lo, ci_hi],
        "mean_partial_score": round(sum(r["score"] for r in per_task) / max(len(per_task), 1), 4),
        "by_dimension": by_dim,
        "by_difficulty": by_diff,
        "failure_taxonomy": dict(failures.most_common(12)),
        "tasks": per_task,
    }


def _load_external_hf_opponent(model_id: str, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _cuda_amp_dtype() if device.type == "cuda" else (torch.float16 if device.type == "mps" else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    shard_multi_gpu = (
        device.type == "cuda"
        and (
            torch.cuda.device_count() > 1
            or torch.cuda.get_device_properties(0).total_memory < 22 * (1024**3)
        )
    )
    load_kwargs: dict[str, Any] = {}
    if shard_multi_gpu:
        load_kwargs["device_map"] = "auto"
        load_kwargs["offload_folder"] = "/tmp/hf_offload"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, **load_kwargs)
    if not shard_multi_gpu:
        model = model.to(device)

    if device.type == "cuda" and dtype == torch.float16:
        first_dev = next(model.parameters()).device
        probe = tok("def dijkstra(n, edges, src, dst):\n    import heapq\n    return 0\n" * 4, return_tensors="pt").to(first_dev)
        model.eval()
        with torch.no_grad():
            probe_logits = model(**probe).logits
        if not torch.isfinite(probe_logits).all():
            del model
            gc.collect()
            torch.cuda.empty_cache()
            try:
                model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **load_kwargs)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **load_kwargs)
            if not shard_multi_gpu:
                model = model.to(device)
    return model, tok


def run_c2_x4_giant_killer(
    *,
    device: torch.device,
    student_id: str = "Qwen/Qwen3-1.7B",
    opponent_id: str = "Qwen/Qwen3-8B",
) -> dict[str, Any]:
    """Compare trained <=4B NanoLM-1.7B-Holonomic-v2 (Greedy & Zero-Oracle TTC) vs 8.19B External Qwen3-8B."""
    eval_tasks = build_benchmark_15d_suite()
    cur_v2 = build_curriculum_v2(seed=31415)
    rl_cur = build_challenging_rlvr_curriculum(seed=777)

    dec_sft = verify_zero_contamination(cur_v2, eval_tasks)
    dec_rl = verify_zero_contamination(rl_cur, eval_tasks)

    # 1. External 8.19B Opponent Evaluation (strictly greedy, non-thinking, read-only)
    opp_model, opp_tok = _load_external_hf_opponent(opponent_id, device)
    opp_arm = _evaluate_15d_extended(
        opp_model,
        opp_tok,
        eval_tasks,
        device,
        arm_name=f"External_Opponent_{opponent_id.split('/')[-1]}",
        is_external_opponent=True,
        ttc_syntax_retries=0,
    )
    print(
        f"[C2-X4] Opponent {opponent_id}: passed={opp_arm['passed']}/{opp_arm['total_tasks']} "
        f"acc={opp_arm['accuracy']:.4f} cap/B={opp_arm['by_dimension']['inference_efficiency']['capability_per_billion_params']:.4f} "
        f"vram={opp_arm['by_dimension']['inference_efficiency']['peak_vram_mb']}MB",
        flush=True,
    )
    del opp_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Train NanoLM-1.7B-Holonomic-v2 (< 4B ceiling)
    model, tok = _load_hf_model_and_tokenizer(student_id, device)
    sample = tok("def dijkstra(n, edges, src, dst):\n    import", return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        pre_lg = model(**sample).logits.float().clone()

    graft_info = apply_holonomic_graft(
        model,
        lora_config=LoRAConfig(rank=16, alpha=32.0),
        n_sidecar_layers=8,
        n_registers=4,
        d_fiber=128,
    )
    model.eval()
    with torch.no_grad():
        post_lg = model(**sample).logits.float()
    step0_diff = float((pre_lg - post_lg).abs().max().item())
    if step0_diff > 1e-4:
        raise RuntimeError(f"Step-0 equivalence violated: {step0_diff}")
    graft_info["step0_max_abs_logit_diff"] = step0_diff

    sft_stats = train_sft_curriculum(model, tok, cur_v2, device=device, epochs=3, lr=1.2e-4)
    # Focused challenging RL prompts with higher kl_beta=0.08 to preserve syntax
    rl_focus = [ex for ex in cur_v2 if ex.dim in (BenchmarkDim.TOOL_USE, BenchmarkDim.PLANNING_DECOMPOSITION, BenchmarkDim.MEMORY_RETENTION, BenchmarkDim.CODING)] + list(rl_cur)[:16]
    grpo_stats = _grpo_with_holonomy_ref(
        model,
        tok,
        rl_focus,
        device=device,
        config=GRPOConfig(rounds=2, group_size=4, max_new_tokens=224, lr=3e-5, kl_beta=0.08),
    )

    gate_norms = [
        float(m.sidecar_gate.detach().abs().mean().cpu().item())
        for m in model.modules()
        if isinstance(m, HolonomicRegisterSidecar)
    ]

    # Iso-decoding Greedy pass@1 comparison
    arm_greedy = _evaluate_15d_extended(
        model,
        tok,
        eval_tasks,
        device,
        arm_name="NanoLM_1.7B_Holonomic_v2_Greedy",
        is_external_opponent=False,
        ttc_syntax_retries=0,
    )
    # Zero-Oracle Syntax-Gated TTC (max_k=3 total attempts on Python/JSON syntax error)
    arm_ttc = _evaluate_15d_extended(
        model,
        tok,
        eval_tasks,
        device,
        arm_name="NanoLM_1.7B_Holonomic_v2_SyntaxTTC",
        is_external_opponent=False,
        ttc_syntax_retries=2,
    )
    train_meta = {
        "graft": graft_info,
        "sft_v2": sft_stats,
        "grpo_v2": grpo_stats,
        "mean_abs_sidecar_gate": round(sum(gate_norms) / max(len(gate_norms), 1), 6),
        "trainable_tensor_count": len(lora_state_dict(model)),
    }
    arm_greedy["training_stats"] = train_meta
    arm_ttc["training_stats"] = train_meta

    print(
        f"[C2-X4] NanoLM-1.7B-v2 Greedy={arm_greedy['passed']}/56 ({arm_greedy['accuracy']:.4f}) "
        f"| SyntaxTTC={arm_ttc['passed']}/56 ({arm_ttc['accuracy']:.4f}, recovered={arm_ttc['ttc_syntax_recoveries']}) "
        f"| vs Qwen3-8B={opp_arm['passed']}/56 ({opp_arm['accuracy']:.4f})",
        flush=True,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "experiment": "C2_X4_1.7B_vs_8.2B_benchmark",
        "student_id": student_id,
        "opponent_id": opponent_id,
        "decontamination": {"sft_v2": dec_sft, "rl": dec_rl},
        "arms": [opp_arm, arm_greedy, arm_ttc],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="NanoLM Cycle-2 Breakthrough Suite Runner")
    parser.add_argument("--mode", choices=("all", "x0", "x1", "x2", "x3", "x4"), default="all")
    parser.add_argument("--out-dir", type=str, default="artifacts/nanolm_holonomy_v2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    res: dict[str, Any] = {
        "meta": {
            "device": str(dev),
            "cuda_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "non-cuda",
            "cuda_count": torch.cuda.device_count() if dev.type == "cuda" else 0,
            "cuda_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if dev.type == "cuda" else [],
            "torch_version": torch.__version__,
        }
    }

    if args.mode in ("all", "x0"):
        r0 = run_c2_x0_kernel_microbench(device=dev)
        res["c2_x0"] = r0
        (out_dir / "c2_x0_kernel_microbench.json").write_text(json.dumps(r0, indent=2), encoding="utf-8")

    if args.mode in ("all", "x1"):
        r1 = run_c2_x1_s4_grokking(device=dev)
        res["c2_x1"] = r1
        (out_dir / "c2_x1_s4_grokking.json").write_text(json.dumps(r1, indent=2), encoding="utf-8")

    if args.mode in ("all", "x2"):
        r2 = run_c2_x2_triton_lm(device=dev)
        res["c2_x2"] = r2
        (out_dir / "c2_x2_triton_lm.json").write_text(json.dumps(r2, indent=2), encoding="utf-8")

    if args.mode in ("all", "x3"):
        r3 = run_c2_x3_32k_scaling(device=dev)
        res["c2_x3"] = r3
        (out_dir / "c2_x3_32k_scaling.json").write_text(json.dumps(r3, indent=2), encoding="utf-8")

    if args.mode in ("all", "x4"):
        r4 = run_c2_x4_giant_killer(device=dev)
        res["c2_x4"] = r4
        (out_dir / "c2_x4_giant_killer.json").write_text(json.dumps(r4, indent=2), encoding="utf-8")

    (out_dir / "cycle2_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"CYCLE2_CAMPAIGN_COMPLETE out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
