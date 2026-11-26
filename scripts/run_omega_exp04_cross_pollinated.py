#!/usr/bin/env python3
"""OMEGA-EXP-04: Cross-Pollinated G5 Benchmark, Equal-Wall-Clock Gate & 32k Chunked Prefill ($0.00).

Evaluates:
  - 4A (3 seeds [11, 22, 33], iso-param mech, replace=False): B0_Transformer vs G4_NoSparseRefiner
       vs G5_CrossPollinated on seen ID copy, 0-exposure unseen open-vocab copy, and 2x-OOD unseen copy.
  - 4B (2 seeds [42, 137], scale_8m, Corpus v1 + leak audit): PR #194 Equal-Token AND Equal-Wall-Clock
       comparison of B0_Transformer vs G4_NoSparseRefiner vs G5_CrossPollinated on ID (T=256) and 2x OOD (T=512).
  - 4C (H-Omega-21): Exact C_stream=1024 forward() vs forward_chunked() parity at T=2,048 and
       peak allocation scaling across T in {1024, 4096, 16384, 32768}.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanolm_discovery.config import ParadigmKind, make_iso_param_suite
from nanolm_discovery.eval_benchmarks import CorpusStream
from nanolm_discovery.models import NanoLMDiscoveryModel
from nanolm_omega.backbone import OmegaVariant, build_omega_model, compute_omega_census, make_omega_suite
from nanolm_omega.cross_pollinated import (
    audit_identifier_and_skeleton_leakage,
    sample_decontaminated_open_vocab_kv_batch,
)

DEFAULT_CORPUS_DIR = Path("/Volumes/Express1M2/nanolm-lab/data/v1")


def _sync_device(dev: torch.device) -> None:
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak_mem(dev: torch.device) -> None:
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def _peak_mem_mib(dev: torch.device) -> float:
    if dev.type == "cuda":
        return float(torch.cuda.max_memory_allocated(dev)) / (1024.0 * 1024.0)
    if dev.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
    return 0.0


def _mean_std(vals: list[float]) -> dict[str, Any]:
    return {
        "mean": float(statistics.fmean(vals)) if vals else 0.0,
        "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


def _eval_kv_acc(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    valid = y >= 0
    with torch.no_grad():
        logits = model(x).logits
        pred = logits.argmax(dim=-1)
    denom = int(valid.sum().item())
    return float((pred[valid] == y[valid]).float().sum().item() / max(1, denom))


def run_exp_4a_open_vocab(seeds: list[int] = (11, 22, 33), steps: int = 75) -> dict[str, Any]:
    """3-seed iso-param open-vocabulary associative recall (replace=False, 0-exposure unseen values)."""
    dev = torch.device("cpu")
    vocab_size = 96
    seen_val_hi = 64
    b0_cfg = make_iso_param_suite("mech", vocab_size=vocab_size)[ParadigmKind.B0_TRANSFORMER]
    omega_suite = make_omega_suite("mech", vocab_size=vocab_size)
    arms_spec = {
        "b0_transformer": lambda: NanoLMDiscoveryModel(b0_cfg),
        "g4_no_sparse_refiner": lambda: build_omega_model(omega_suite[OmegaVariant.G4_NO_SPARSE_REFINER]),
        "g5_cross_pollinated": lambda: build_omega_model(omega_suite[OmegaVariant.G5_CROSS_POLLINATED]),
    }

    per_arm: dict[str, Any] = {}
    for arm_name, builder in arms_spec.items():
        seed_rows = []
        for sd in seeds:
            torch.manual_seed(sd)
            model = builder().to(dev).train()
            params = int(model.deployed_parameter_count())
            opt = torch.optim.AdamW(model.parameters(), lr=4e-3, betas=(0.9, 0.95), weight_decay=0.01)
            gen_tr = torch.Generator().manual_seed(10_000 + sd)
            t0 = time.perf_counter()
            for _ in range(steps):
                xb, yb = sample_decontaminated_open_vocab_kv_batch(
                    16, 6, 4, vocab_size=vocab_size, seen_val_hi=seen_val_hi, unseen_mode=False, device=dev, generator=gen_tr
                )
                opt.zero_grad(set_to_none=True)
                if arm_name == "b0_transformer":
                    logits = model(xb).logits
                    valid = yb >= 0
                    loss = F.cross_entropy(logits.float()[valid], yb[valid])
                else:
                    loss = model(xb, targets=yb).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            train_s = time.perf_counter() - t0
            model.eval()
            gen_ev = torch.Generator().manual_seed(90_000 + sd)
            xs, ys = sample_decontaminated_open_vocab_kv_batch(
                64, 6, 4, vocab_size=vocab_size, seen_val_hi=seen_val_hi, unseen_mode=False, device=dev, generator=gen_ev
            )
            xu, yu = sample_decontaminated_open_vocab_kv_batch(
                64, 6, 4, vocab_size=vocab_size, seen_val_hi=seen_val_hi, unseen_mode=True, device=dev, generator=gen_ev
            )
            x2x, y2x = sample_decontaminated_open_vocab_kv_batch(
                64, 12, 8, vocab_size=vocab_size, seen_val_hi=seen_val_hi, unseen_mode=True, device=dev, generator=gen_ev
            )
            seed_rows.append({
                "seed": sd,
                "params": params,
                "train_sec": train_s,
                "seen_id_acc": _eval_kv_acc(model, xs, ys),
                "unseen_open_vocab_acc": _eval_kv_acc(model, xu, yu),
                "unseen_2x_ood_acc": _eval_kv_acc(model, x2x, y2x),
            })
        per_arm[arm_name] = {
            "params": seed_rows[0]["params"],
            "seen_id_acc": _mean_std([r["seen_id_acc"] for r in seed_rows]),
            "unseen_open_vocab_acc": _mean_std([r["unseen_open_vocab_acc"] for r in seed_rows]),
            "unseen_2x_ood_acc": _mean_std([r["unseen_2x_ood_acc"] for r in seed_rows]),
            "train_sec": _mean_std([r["train_sec"] for r in seed_rows]),
            "seeds": seed_rows,
        }
        print(
            f"[4A] {arm_name}: seen={per_arm[arm_name]['seen_id_acc']['mean']:.4f} "
            f"unseen={per_arm[arm_name]['unseen_open_vocab_acc']['mean']:.4f} "
            f"ood2x={per_arm[arm_name]['unseen_2x_ood_acc']['mean']:.4f}",
            flush=True,
        )
    return {
        "name": "OMEGA-EXP-04A_DECONTAMINATED_OPEN_VOCAB_COPY",
        "backend": "cpu float32",
        "steps": steps,
        "replace": False,
        "arms": per_arm,
    }


def run_repo_leak_audit() -> dict[str, Any]:
    """Audit identifier & 6-gram skeleton overlap across disjoint repository source splits."""
    train_files = sorted((ROOT / "nano").glob("*.py"))[:12] + sorted((ROOT / "nanolm_discovery").glob("*.py"))
    eval_files = sorted((ROOT / "nanolm_omega").glob("*.py"))
    train_docs = [p.read_text(encoding="utf-8", errors="ignore")[:3000] for p in train_files if p.is_file()]
    eval_raw = [p.read_text(encoding="utf-8", errors="ignore")[:3000] for p in eval_files if p.is_file()]
    raw_audit = audit_identifier_and_skeleton_leakage(train_docs, eval_raw, skeleton_n=6)
    synthetic_disjoint_eval = [
        "Verify bitemporal holonomy gauge transport on unseen topological manifold coordinates."
    ]
    clean_audit = audit_identifier_and_skeleton_leakage(train_docs, synthetic_disjoint_eval, skeleton_n=6)
    return {"cross_package_repo_split": raw_audit, "disjoint_probe": clean_audit}


def run_exp_4b_real_corpus_equal_time_and_token(
    corpus_dir: Path,
    *,
    seeds: list[int] = (42, 137),
    equal_steps: int = 30,
    batch_size: int = 4,
    seq_len: int = 128,
) -> dict[str, Any]:
    """PR #194 Equal-Token AND Equal-Wall-Clock comparison on Corpus v1 (`scale_8m`)."""
    leak_summary = run_repo_leak_audit()
    if not (corpus_dir / "manifest.json").is_file():
        return {"status": "SKIPPED_CORPUS_NOT_MOUNTED", "leak_audit": leak_summary}

    stream = CorpusStream(corpus_dir)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    use_bf16 = dev.type == "mps"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    b0_cfg = make_iso_param_suite("scale_8m", vocab_size=stream.vocab_size)[ParadigmKind.B0_TRANSFORMER]
    omega_suite = make_omega_suite("scale_8m", vocab_size=stream.vocab_size)
    g4_cfg = omega_suite[OmegaVariant.G4_NO_SPARSE_REFINER]
    g5_cfg = omega_suite[OmegaVariant.G5_CROSS_POLLINATED]

    def _eval_pair(m: torch.nn.Module) -> dict[str, Any]:
        ev_id = stream.evaluate_validation_bpb(
            m, device=dev, seq_len=seq_len, eval_tokens_per_domain=8_192, batch_size=4, use_bf16=use_bf16, amp_dtype=amp_dtype
        )
        ev_2x = stream.evaluate_validation_bpb(
            m, device=dev, seq_len=seq_len * 2, eval_tokens_per_domain=8_192, batch_size=2, use_bf16=use_bf16, amp_dtype=amp_dtype
        )
        return {"id_bpb": ev_id["composite_bpb"], "ood2x_bpb": ev_2x["composite_bpb"], "id_domains": ev_id["domain_bpb"], "ood2x_domains": ev_2x["domain_bpb"]}

    def _train_model(arm_kind: str, sd: int, target_wall_s: float | None = None) -> dict[str, Any]:
        torch.manual_seed(sd)
        np_rng = np.random.default_rng(sd)
        if arm_kind == "b0_transformer":
            model = NanoLMDiscoveryModel(b0_cfg).to(dev)
        elif arm_kind == "g4_no_sparse_refiner":
            model = build_omega_model(g4_cfg).to(dev)
        else:
            model = build_omega_model(g5_cfg).to(dev)

        opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, betas=(0.9, 0.95), weight_decay=0.05)
        model.train()
        total_steps = equal_steps if target_wall_s is None else 10_000
        _sync_device(dev)
        t_start = time.perf_counter()
        wall_elapsed = 0.0
        steps_done = 0
        for step in range(1, total_steps + 1):
            xb, yb = stream.sample_train_batch(batch_size, seq_len, np_rng, dev)
            opt.zero_grad(set_to_none=True)
            if use_bf16:
                with torch.autocast(device_type=dev.type, dtype=amp_dtype):
                    out = model(xb, targets=yb)
                    loss = out.loss
            else:
                out = model(xb, targets=yb)
                loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            steps_done = step
            if target_wall_s is not None and step % 4 == 0:
                _sync_device(dev)
                wall_elapsed = time.perf_counter() - t_start
                if wall_elapsed >= target_wall_s:
                    break
        _sync_device(dev)
        wall_elapsed = time.perf_counter() - t_start
        tokens_seen = steps_done * batch_size * seq_len
        ev = _eval_pair(model)
        params = int(model.deployed_parameter_count())
        del model, opt
        gc.collect()
        return {
            "seed": sd,
            "params": params,
            "steps": steps_done,
            "tokens": tokens_seen,
            "train_wall_s": wall_elapsed,
            "tok_per_s": float(tokens_seen / max(1e-6, wall_elapsed)),
            **ev,
        }

    arm_runs: dict[str, list[dict[str, Any]]] = {
        "g4_no_sparse_refiner": [],
        "g5_cross_pollinated": [],
        "b0_equal_token": [],
        "b0_equal_wall_clock": [],
    }
    for sd in seeds:
        r_g4 = _train_model("g4_no_sparse_refiner", sd)
        print(f"[4B seed={sd}] g4_no_sparse_refiner: steps={r_g4['steps']} wall={r_g4['train_wall_s']:.1f}s id_bpb={r_g4['id_bpb']:.4f} ood2x={r_g4['ood2x_bpb']:.4f}", flush=True)
        r_g5 = _train_model("g5_cross_pollinated", sd)
        print(f"[4B seed={sd}] g5_cross_pollinated: steps={r_g5['steps']} wall={r_g5['train_wall_s']:.1f}s id_bpb={r_g5['id_bpb']:.4f} ood2x={r_g5['ood2x_bpb']:.4f}", flush=True)
        r_b0_tok = _train_model("b0_transformer", sd)
        print(f"[4B seed={sd}] b0_equal_token: steps={r_b0_tok['steps']} wall={r_b0_tok['train_wall_s']:.1f}s id_bpb={r_b0_tok['id_bpb']:.4f} ood2x={r_b0_tok['ood2x_bpb']:.4f}", flush=True)
        target_t = max(r_g4["train_wall_s"], r_g5["train_wall_s"])
        r_b0_time = _train_model("b0_transformer", sd, target_wall_s=target_t)
        print(f"[4B seed={sd}] b0_equal_wall_clock: steps={r_b0_time['steps']} wall={r_b0_time['train_wall_s']:.1f}s id_bpb={r_b0_time['id_bpb']:.4f} ood2x={r_b0_time['ood2x_bpb']:.4f}", flush=True)
        arm_runs["g4_no_sparse_refiner"].append(r_g4)
        arm_runs["g5_cross_pollinated"].append(r_g5)
        arm_runs["b0_equal_token"].append(r_b0_tok)
        arm_runs["b0_equal_wall_clock"].append(r_b0_time)

    summary = {}
    for k, rows in arm_runs.items():
        summary[k] = {
            "params": rows[0]["params"],
            "steps": _mean_std([float(r["steps"]) for r in rows]),
            "tokens": _mean_std([float(r["tokens"]) for r in rows]),
            "train_wall_s": _mean_std([r["train_wall_s"] for r in rows]),
            "tok_per_s": _mean_std([r["tok_per_s"] for r in rows]),
            "id_bpb": _mean_std([r["id_bpb"] for r in rows]),
            "ood2x_bpb": _mean_std([r["ood2x_bpb"] for r in rows]),
            "seeds": rows,
        }

    return {
        "name": "OMEGA-EXP-04B_REAL_CORPUS_EQUAL_TOKEN_AND_WALL_CLOCK",
        "device": f"{dev.type} ({amp_dtype})",
        "equal_steps": equal_steps,
        "batch_size": batch_size,
        "seq_len_id": seq_len,
        "seq_len_ood2x": seq_len * 2,
        "corpus_sha256": stream.manifest.get("train_sha256"),
        "leak_audit": leak_summary,
        "arms": summary,
    }


def run_exp_4c_chunked_prefill_32k() -> dict[str, Any]:
    """H-Omega-21: Verify C_stream=1024 exact parity at T=2,048 and measure 1k..32k streaming scaling."""
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cfg = make_omega_suite("scale_8m")[OmegaVariant.G5_CROSS_POLLINATED]
    census = compute_omega_census(cfg)
    model = build_omega_model(cfg).to(dev).eval()

    # Exact monolithic vs C_stream=1024 parity at T=2,048
    torch.manual_seed(55)
    x_2k = torch.randint(0, cfg.vocab_size, (1, 2048), device=dev)
    y_2k = torch.randint(0, cfg.vocab_size, (1, 2048), device=dev)
    with torch.no_grad():
        out_full = model(x_2k, targets=y_2k)
        out_chunk_full = model.forward_chunked(x_2k, targets=y_2k, chunk_tokens=1024, keep_logits=True)
        out_chunk_stream = model.forward_chunked(x_2k, targets=y_2k, chunk_tokens=1024, keep_logits=False)
    logit_mae = float((out_full.logits.float() - out_chunk_full.logits.float()).abs().max().item())
    loss_diff = float(abs(out_full.loss.item() - out_chunk_stream.loss.item()))

    use_bf16 = dev.type == "mps"
    scaling = []
    for ctx_len in (1024, 4096, 16384, 32768):
        x_ctx = torch.randint(0, cfg.vocab_size, (1, ctx_len), device=dev)
        y_ctx = torch.randint(0, cfg.vocab_size, (1, ctx_len), device=dev)
        _reset_peak_mem(dev)
        _sync_device(dev)
        t0 = time.perf_counter()
        with torch.no_grad():
            if use_bf16:
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                    res = model.forward_chunked(x_ctx, targets=y_ctx, chunk_tokens=1024, keep_logits=False)
            else:
                res = model.forward_chunked(x_ctx, targets=y_ctx, chunk_tokens=1024, keep_logits=False)
            loss_val = float(res.loss.item())
        _sync_device(dev)
        elapsed = time.perf_counter() - t0
        monolithic_logits_fp32_mib = (1 * ctx_len * cfg.vocab_size * 4) / (1024.0 * 1024.0)
        chunked_logits_fp32_mib = (1 * min(1024, ctx_len) * cfg.vocab_size * 4) / (1024.0 * 1024.0)
        alloc_mib = _peak_mem_mib(dev)
        print(
            f"[4C] ctx={ctx_len}: chunked_sec={elapsed:.2f}s tok/s={ctx_len/max(1e-6,elapsed):.1f} alloc_mib={alloc_mib:.1f}",
            flush=True,
        )
        scaling.append({
            "seq_len": ctx_len,
            "chunk_tokens": 1024,
            "ce_loss": loss_val,
            "prefill_sec": elapsed,
            "tok_per_s": float(ctx_len / max(1e-6, elapsed)),
            "allocated_mib": alloc_mib,
            "monolithic_logits_fp32_mib": monolithic_logits_fp32_mib,
            "chunked_logits_fp32_mib": chunked_logits_fp32_mib,
            "logits_tensor_reduction_x": monolithic_logits_fp32_mib / chunked_logits_fp32_mib,
        })

    del model
    gc.collect()
    return {
        "name": "OMEGA-EXP-04C_STREAMING_CHUNKED_PREFILL_32K",
        "device": dev.type,
        "model": cfg.name,
        "params": census.total_deployed_parameters,
        "state_bytes_32k_bf16": census.state_bytes_at_32768_bf16,
        "parity_t2048_logit_max_abs_err": logit_mae,
        "parity_t2048_ce_loss_abs_err": loss_diff,
        "context_scaling": scaling,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OMEGA-EXP-04 cross-pollinated benchmark ($0.00).")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--steps-4b", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "nanolm_omega_v1" / "omega_exp04_cross_pollinated.json",
    )
    args = parser.parse_args()

    src_paths = [
        Path(__file__),
        ROOT / "nanolm_omega" / "backbone.py",
        ROOT / "nanolm_omega" / "cross_pollinated.py",
    ]
    sources_sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src_paths}

    print("[4A] Running 3-seed replace=False decontaminated open-vocab copy...", flush=True)
    res_4a = run_exp_4a_open_vocab()
    print(json.dumps(res_4a["arms"], indent=2), flush=True)

    print(f"[4B] Running 2-seed scale_8m Equal-Token & Equal-Wall-Clock on {args.corpus_dir}...", flush=True)
    res_4b = run_exp_4b_real_corpus_equal_time_and_token(args.corpus_dir, equal_steps=args.steps_4b)
    print(json.dumps(res_4b.get("arms", res_4b), indent=2), flush=True)

    print("[4C] Running C_stream=1024 exact parity (2k) + 32,768 streaming prefill benchmark...", flush=True)
    res_4c = run_exp_4c_chunked_prefill_32k()
    print(json.dumps(res_4c, indent=2), flush=True)

    payload = {
        "schema": "nanolm.omega.exp04_cross_pollinated.v1",
        "paid_cloud_spend_usd": 0.0,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": str(torch.__version__),
        "sources_sha256": sources_sha,
        "exp_4a_open_vocab_copy": res_4a,
        "exp_4b_real_corpus": res_4b,
        "exp_4c_streaming_32k": res_4c,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[DONE] Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
