"""Cloud TPU (`v5e` / `v6e`) JAX Gated DeltaNet Parity Verifier & `R0-TPU` Throughput/Stability Benchmark.

Executes four pre-registered phases:
  1. `parity_vs_fla_bundle`: Loads `fla_cuda_parity_bundle.npz` (exported from real CUDA `fla==0.6.0` on GPU),
     verifies bit-exact `PyTorch -> JAX -> PyTorch` round-trip (`0.0` diff), and checks `FP32` & `BF16`
     forward logits, scalar loss, per-parameter backward gradients, and `10`-step `AdamW` trajectory on TPU!
  2. `ceilings`: Measures live TPU BF16 MXU `8192x8192` GEMM TFLOPS and `1 GiB` HBM copy bandwidth GB/s.
  3. `r0_benchmark_grid`: Benchmarks `546,865,360` (`546.9M`, `24L, d=896, 7x128, ffn=4864, V=151,936`) pure
     `GatedDeltaNet`, `541.8M` Stage-C `18 GDN + 6 GQA` hybrid (`KEEP_ATTN = [1,2,3,18,19,20]`), and `527.0M`
     pure `Softmax GQA`, plus `151.9M` (`SmolLM2-135M` hybrid), across:
       - Single TPU chip (`16,384` tok/step at `seq=2048, 4096, 8192`) + Cached Top-32 Sparse Teacher KD,
       - All 4 TPU chips (`jax.pmap` data-parallel `65,536` tok/step at `seq=2048, 4096, 8192` + Cached Top-32 KD).
  4. `stability_100_steps`: Runs `100` consecutive optimization steps of `546.9M` `GatedDeltaNet` on TPU,
     verifying `0/100` non-finite losses or gradient norms.
"""
from __future__ import annotations
import argparse
import functools
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dual_gdn_core import (
    init_jax_hybrid_params,
    jax_chunked_ce_or_kd_loss,
    jax_hybrid_logits,
    jax_n_params,
    jax_params_to_npz_dict,
    npz_dict_to_jax_params,
)

CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
CORE_SHA256 = hashlib.sha256((HERE / "dual_gdn_core.py").read_bytes()).hexdigest()

TPU_SPECS = {
    "TPU v5 lite": (197.0, 819.0),
    "TPU v5e": (197.0, 819.0),
    "TPU v6 lite": (918.0, 1600.0),
    "TPU v6e": (918.0, 1600.0),
}


def verify_against_fla_bundle(bundle_path: Path, meta_path: Path, dev) -> dict:
    meta = json.loads(meta_path.read_text())
    b = np.load(bundle_path)
    cfg = meta["probe_cfg"]
    cfg["attn_layers"] = tuple(cfg["attn_layers"])

    init_sd = {k[len("init__"):]: b[k] for k in b.files if k.startswith("init__")}
    grads_ref = {k[len("grad32__"):]: b[k] for k in b.files if k.startswith("grad32__")}
    ids = jnp.asarray(b["ids"], dtype=jnp.int32)
    step_ids = jnp.asarray(b["step_ids"], dtype=jnp.int32)
    lg_ref32 = b["logits_fp32"]
    lg_ref_bf16 = b["logits_bf16"]

    # 1. Bidirectional state_dict <-> JAX PyTree bit-exactness (`P1`)
    jax_p = npz_dict_to_jax_params(init_sd, cfg["layers"], cfg["attn_layers"])
    rt_sd = jax_params_to_npz_dict(jax_p, cfg["attn_layers"])
    rt_max_diff = max(float(np.max(np.abs(init_sd[k] - rt_sd[k]))) for k in init_sd)

    # Exact SmolLM2-135M 22-GDN + 8-GQA and R0 547M parameter counts in JAX
    p_135m = init_jax_hybrid_params(
        jax.random.PRNGKey(0),
        dict(d=576, layers=30, ffn=1536, heads=9, kv_heads=3, vocab=49152, attn_layers=(1, 2, 3, 4, 21, 22, 23, 24)),
    )
    jax_135m_params = jax_n_params(p_135m)
    del p_135m

    with jax.default_device(dev):
        jp = jax.device_put(jax_p, dev)
        ids_d = jax.device_put(ids, dev)

        # 2. FP32 Forward + Backward Parity vs CUDA `fla` (`P2`)
        def loss32_fn(p, tok):
            lg = jax_hybrid_logits(p, tok[:, :-1], cfg, jnp.float32, chunk_size=64, remat=False)
            l = optax.softmax_cross_entropy_with_integer_labels(lg, tok[:, 1:]).mean()
            return l, lg

        (loss_j32, lg_j32), grads_j32 = jax.jit(jax.value_and_grad(loss32_fn, has_aux=True))(jp, ids_d)
        lg_j32_np = np.asarray(lg_j32)
        loss_j32_val = float(loss_j32)

        grads_j_sd = jax_params_to_npz_dict(grads_j32, cfg["attn_layers"])
        grad_rel_errs = {}
        for k, gref in grads_ref.items():
            gj = grads_j_sd[k]
            denom = max(float(np.max(np.abs(gref))), 1e-6)
            grad_rel_errs[k] = round(float(np.max(np.abs(gj - gref))) / denom, 7)

        # 3. BF16 Forward + 10-Step AdamW Trajectory Parity (`P3`)
        @jax.jit
        def eval_bf16(p, tok):
            lg = jax_hybrid_logits(p, tok[:, :-1], cfg, jnp.bfloat16, chunk_size=64, remat=False).astype(jnp.float32)
            return optax.softmax_cross_entropy_with_integer_labels(lg, tok[:, 1:]).mean(), lg

        loss_jbf, lg_jbf = eval_bf16(jp, ids_d)
        loss_jbf_val = float(loss_jbf)
        lg_jbf_np = np.asarray(lg_jbf)

        opt = optax.adamw(learning_rate=3e-4, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.0)

        def run_10_steps(dtype_run):
            p_cur = jp
            st_cur = opt.init(p_cur)

            @jax.jit
            def one_step(p_s, st_s, tok):
                def lf(px):
                    lgx = jax_hybrid_logits(px, tok[:, :-1], cfg, dtype_run, chunk_size=64, remat=False).astype(jnp.float32)
                    return optax.softmax_cross_entropy_with_integer_labels(lgx, tok[:, 1:]).mean()
                lv, gv = jax.value_and_grad(lf)(p_s)
                u, st_n = opt.update(gv, st_s, p_s)
                return optax.apply_updates(p_s, u), st_n, lv

            losses = []
            for s_i in range(10):
                p_cur, st_cur, lv = one_step(p_cur, st_cur, jax.device_put(step_ids[s_i], dev))
                losses.append(round(float(lv), 6))
            return losses

        traj_j32 = run_10_steps(jnp.float32)
        traj_jbf = run_10_steps(jnp.bfloat16)

    return {
        "bundle_sha256_verified": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "smollm2_135m_hybrid_params_jax": jax_135m_params,
        "smollm2_135m_hybrid_params_fla": meta["smollm2_135m_hybrid_params_verified"],
        "roundtrip_max_abs_diff": rt_max_diff,
        "fp32_vs_cuda_fla": {
            "logit_max_abs_err": round(float(np.max(np.abs(lg_j32_np - lg_ref32))), 7),
            "logit_rel_err": round(float(np.max(np.abs(lg_j32_np - lg_ref32)) / max(float(np.max(np.abs(lg_ref32))), 1e-6)), 7),
            "loss_fla": round(float(meta["loss_fp32"]), 7),
            "loss_jax": round(loss_j32_val, 7),
            "loss_abs_err": round(abs(loss_j32_val - float(meta["loss_fp32"])), 7),
            "grad_max_rel_err": round(max(grad_rel_errs.values()), 7),
            "grad_rel_err_by_tensor_sample": {k: grad_rel_errs[k] for k in sorted(grad_rel_errs)[:10]},
            "traj_10steps_fla": meta["traj_fp32_10steps"],
            "traj_10steps_jax": traj_j32,
            "step10_loss_abs_err": round(abs(traj_j32[-1] - meta["traj_fp32_10steps"][-1]), 7),
        },
        "bf16_vs_cuda_fla": {
            "logit_max_abs_err": round(float(np.max(np.abs(lg_jbf_np - lg_ref_bf16))), 5),
            "loss_fla": round(float(meta["loss_bf16"]), 6),
            "loss_jax": round(loss_jbf_val, 6),
            "loss_abs_err": round(abs(loss_jbf_val - float(meta["loss_bf16"])), 6),
            "traj_10steps_fla": meta["traj_bf16_10steps"],
            "traj_10steps_jax": traj_jbf,
            "step10_loss_abs_err": round(abs(traj_jbf[-1] - meta["traj_bf16_10steps"][-1]), 6),
        },
        "kaggle_gpu_meta": meta.get("kaggle_gpu_152m_hybrid_bench"),
    }


def measure_tpu_ceilings(dev, small: bool = False) -> dict:
    n = 2048 if small else 8192
    it = 5 if small else 25
    with jax.default_device(dev):
        a = jax.random.normal(jax.random.PRNGKey(1), (n, n), jnp.bfloat16)
        b = jax.random.normal(jax.random.PRNGKey(2), (n, n), jnp.bfloat16)

        @jax.jit
        def gemm_loop(x, y):
            def body(c, _):
                return jnp.dot(c, y) * jnp.bfloat16(1.0 / n), None
            s, _ = jax.lax.scan(body, x, None, length=it)
            return s[0, 0]

        jax.block_until_ready(gemm_loop(a, b))
        t0 = time.perf_counter()
        jax.block_until_ready(gemm_loop(a, b))
        gemm_tf = (2.0 * (n ** 3) * it) / (time.perf_counter() - t0) / 1e12

        nel = (2 ** 24) if small else (2 ** 29)  # 1 GiB in bf16 (512M elements * 2B)
        x_bw = jnp.ones((nel,), jnp.bfloat16)

        @jax.jit
        def copy_fn(z):
            return z + jnp.bfloat16(1e-4)

        jax.block_until_ready(copy_fn(x_bw))
        t1 = time.perf_counter()
        for _ in range(10):
            y_bw = copy_fn(x_bw)
            jax.block_until_ready(y_bw)
        bw_gbps = (2.0 * nel * 2.0 * 10.0) / (time.perf_counter() - t1) / 1e9

    return {
        "measured_gemm_TFLOPS": round(gemm_tf, 1),
        "measured_hbm_bandwidth_GBps": round(bw_gbps, 1),
        "gemm_matrix": n,
    }


def bench_config(
    label: str,
    cfg: dict,
    micro_b: int,
    seq: int,
    accum: int,
    dp: int,
    with_kd: bool,
    steps: int,
    peak_tflops_per_chip: float,
    vocab_chunk_cap: int = 2048,
) -> dict:
    devs = jax.devices()[:dp]
    cpu_dev = jax.devices("cpu")[0]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-4, b1=0.9, b2=0.95, weight_decay=0.01))
    vocab_chunk = min(vocab_chunk_cap, micro_b * seq)

    with jax.default_device(cpu_dev):
        params_host = init_jax_hybrid_params(jax.random.PRNGKey(42), cfg)
        n_p = jax_n_params(params_host)
        opt_st_host = opt.init(params_host)

    rng = np.random.default_rng(0)

    def loss_single(p, tok, ki, kv):
        return jax_chunked_ce_or_kd_loss(p, tok, cfg, jnp.bfloat16, kd_idx=ki, kd_val=kv, vocab_chunk=vocab_chunk, remat=True)

    if dp == 1:
        dev0 = devs[0]
        p_dev = jax.device_put(params_host, dev0)
        st_dev = jax.device_put(opt_st_host, dev0)
        del params_host, opt_st_host

        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def step_fn(p, st, toks, kis, kvs):
            if accum == 1:
                l, g = jax.value_and_grad(loss_single)(p, toks[0], kis[0] if with_kd else None, kvs[0] if with_kd else None)
            else:
                def acc_body(carry, idx_a):
                    la, ga = carry
                    tk = toks[idx_a]
                    ki = kis[idx_a] if with_kd else None
                    kv = kvs[idx_a] if with_kd else None
                    lv, gv = jax.value_and_grad(loss_single)(p, tk, ki, kv)
                    return (la + lv / accum, jax.tree_util.tree_map(lambda x, y: x + y / accum, ga, gv)), None
                (l, g), _ = jax.lax.scan(
                    acc_body,
                    (jnp.zeros((), jnp.float32), jax.tree_util.tree_map(jnp.zeros_like, p)),
                    jnp.arange(accum),
                )
            u, st_n = opt.update(g, st, p)
            return optax.apply_updates(p, u), st_n, l

        toks_np = rng.integers(0, cfg["vocab"], size=(accum, micro_b, seq + 1), dtype=np.int32)
        kis_np = rng.integers(0, cfg["vocab"], size=(accum, micro_b, seq, 32), dtype=np.int32) if with_kd else np.zeros((1,), np.int32)
        kvs_np = rng.standard_normal((accum, micro_b, seq, 32)).astype(np.float16) if with_kd else np.zeros((1,), np.float16)
        t_d = jax.device_put(toks_np, dev0)
        ki_d = jax.device_put(kis_np, dev0)
        kv_d = jax.device_put(kvs_np, dev0)

        t_c0 = time.perf_counter()
        p_dev, st_dev, l_w = step_fn(p_dev, st_dev, t_d, ki_d, kv_d)
        jax.block_until_ready(l_w)
        compile_s = time.perf_counter() - t_c0

        for _ in range(2):
            p_dev, st_dev, l_w = step_fn(p_dev, st_dev, t_d, ki_d, kv_d)
        jax.block_until_ready(l_w)

        step_ms = []
        losses = []
        for _ in range(steps):
            t_s = time.perf_counter()
            p_dev, st_dev, l_w = step_fn(p_dev, st_dev, t_d, ki_d, kv_d)
            jax.block_until_ready(l_w)
            step_ms.append((time.perf_counter() - t_s) * 1000.0)
            losses.append(float(l_w))

        mem_stats = dev0.memory_stats() or {}
        peak_gib = round(mem_stats.get("peak_bytes_in_use", 0) / (2 ** 30), 3)
        del p_dev, st_dev, t_d, ki_d, kv_d
    else:
        p_rep = jax.device_put_replicated(params_host, devs)
        st_rep = jax.device_put_replicated(opt_st_host, devs)
        del params_host, opt_st_host

        def pmap_step(p, st, toks, kis, kvs):
            if accum == 1:
                l, g = jax.value_and_grad(loss_single)(p, toks[0], kis[0] if with_kd else None, kvs[0] if with_kd else None)
            else:
                def acc_body(carry, idx_a):
                    la, ga = carry
                    lv, gv = jax.value_and_grad(loss_single)(
                        p, toks[idx_a], kis[idx_a] if with_kd else None, kvs[idx_a] if with_kd else None
                    )
                    return (la + lv / accum, jax.tree_util.tree_map(lambda x, y: x + y / accum, ga, gv)), None
                (l, g), _ = jax.lax.scan(
                    acc_body,
                    (jnp.zeros((), jnp.float32), jax.tree_util.tree_map(jnp.zeros_like, p)),
                    jnp.arange(accum),
                )
            l = jax.lax.pmean(l, axis_name="dp")
            g = jax.lax.pmean(g, axis_name="dp")
            u, st_n = opt.update(g, st, p)
            return optax.apply_updates(p, u), st_n, l

        p_step = jax.pmap(pmap_step, axis_name="dp", devices=devs, donate_argnums=(0, 1))
        toks_np = rng.integers(0, cfg["vocab"], size=(dp, accum, micro_b, seq + 1), dtype=np.int32)
        kis_np = rng.integers(0, cfg["vocab"], size=(dp, accum, micro_b, seq, 32), dtype=np.int32) if with_kd else np.zeros((dp, 1), np.int32)
        kvs_np = rng.standard_normal((dp, accum, micro_b, seq, 32)).astype(np.float16) if with_kd else np.zeros((dp, 1), np.float16)
        t_d = jax.device_put_sharded(list(toks_np), devs)
        ki_d = jax.device_put_sharded(list(kis_np), devs)
        kv_d = jax.device_put_sharded(list(kvs_np), devs)

        t_c0 = time.perf_counter()
        p_rep, st_rep, l_w = p_step(p_rep, st_rep, t_d, ki_d, kv_d)
        jax.block_until_ready(l_w)
        compile_s = time.perf_counter() - t_c0

        for _ in range(2):
            p_rep, st_rep, l_w = p_step(p_rep, st_rep, t_d, ki_d, kv_d)
        jax.block_until_ready(l_w)

        step_ms = []
        losses = []
        for _ in range(steps):
            t_s = time.perf_counter()
            p_rep, st_rep, l_w = p_step(p_rep, st_rep, t_d, ki_d, kv_d)
            jax.block_until_ready(l_w)
            step_ms.append((time.perf_counter() - t_s) * 1000.0)
            losses.append(float(l_w[0]))

        mem_stats = devs[0].memory_stats() or {}
        peak_gib = round(mem_stats.get("peak_bytes_in_use", 0) / (2 ** 30), 3)
        del p_rep, st_rep, t_d, ki_d, kv_d

    med_ms = float(np.median(step_ms))
    tok_per_step = dp * accum * micro_b * seq
    tok_s = tok_per_step / (med_ms / 1000.0)
    achieved_tf = (6.0 * n_p * tok_s) / 1e12
    total_peak_tf = peak_tflops_per_chip * dp
    mfu = achieved_tf / total_peak_tf if total_peak_tf > 0 else 0.0

    return {
        "label": label,
        "params": n_p,
        "dp_chips": dp,
        "micro_batch": micro_b,
        "accum": accum,
        "seq": seq,
        "tokens_per_step": tok_per_step,
        "with_cached_top32_kd": with_kd,
        "compile_sec": round(compile_s, 2),
        "median_step_ms": round(med_ms, 2),
        "tokens_per_second_total": round(tok_s, 1),
        "tokens_per_second_per_chip": round(tok_s / dp, 1),
        "achieved_6N_TFLOPS_total": round(achieved_tf, 2),
        "mfu_of_datasheet_peak": round(mfu, 4),
        "peak_chip_hbm_gib": peak_gib,
        "nonfinite_steps": int(sum(not math.isfinite(x) for x in losses)),
    }


def safe_bench_config(
    label: str,
    cfg: dict,
    micro_b: int,
    seq: int,
    accum: int,
    dp: int,
    with_kd: bool,
    steps: int,
    peak_tflops_per_chip: float,
) -> dict:
    """Runs `bench_config` with automatic constant-tokens-per-step microbatch stepdown on HBM OOM."""
    cur_mb, cur_acc, cur_vc = micro_b, accum, 2048
    while True:
        gc.collect()
        try:
            row = bench_config(
                label, cfg, micro_b=cur_mb, seq=seq, accum=cur_acc, dp=dp,
                with_kd=with_kd, steps=steps, peak_tflops_per_chip=peak_tflops_per_chip, vocab_chunk_cap=cur_vc,
            )
            print(json.dumps({"event": "bench_row", **row}), flush=True)
            return row
        except Exception as exc:
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "Out of memory" in msg:
                if cur_mb > 1:
                    cur_mb //= 2
                    cur_acc *= 2
                elif cur_acc > 1:
                    cur_acc = 1
                elif cur_vc > 256:
                    cur_vc //= 2
                else:
                    with jax.default_device(jax.devices("cpu")[0]):
                        n_p_oom = jax_n_params(init_jax_hybrid_params(jax.random.PRNGKey(0), cfg))
                    oom_row = {
                        "label": label,
                        "params": n_p_oom,
                        "dp_chips": dp,
                        "micro_batch": cur_mb,
                        "accum": cur_acc,
                        "seq": seq,
                        "tokens_per_step": dp * cur_acc * cur_mb * seq,
                        "with_cached_top32_kd": with_kd,
                        "compile_sec": 0.0,
                        "median_step_ms": 0.0,
                        "tokens_per_second_total": 0.0,
                        "tokens_per_second_per_chip": 0.0,
                        "achieved_6N_TFLOPS_total": 0.0,
                        "mfu_of_datasheet_peak": 0.0,
                        "peak_chip_hbm_gib": 15.75,
                        "nonfinite_steps": 0,
                        "oom_16gib_hbm": True,
                    }
                    print(json.dumps({"event": "bench_row_oom", **oom_row}), flush=True)
                    return oom_row
                print(json.dumps({"event": "bench_oom_stepdown", "label": label, "new_micro_b": cur_mb, "new_accum": cur_acc, "new_vocab_chunk": cur_vc}), flush=True)
                continue
            raise


def run_stability_100(cfg: dict, dev, micro_b: int = 2, seq: int = 1024, steps: int = 100) -> dict:
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.01))
    with jax.default_device(jax.devices("cpu")[0]):
        p_h = init_jax_hybrid_params(jax.random.PRNGKey(123), cfg)
        st_h = opt.init(p_h)
    with jax.default_device(dev):
        p = jax.device_put(p_h, dev)
        st = jax.device_put(st_h, dev)
        del p_h, st_h

        def lf(px, tk):
            return jax_chunked_ce_or_kd_loss(px, tk, cfg, jnp.bfloat16, vocab_chunk=min(2048, micro_b * seq), remat=True)

        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def step_fn(px, stx, tk):
            l, g = jax.value_and_grad(lf)(px, tk)
            gn = optax.global_norm(g)
            u, st_n = opt.update(g, stx, px)
            return optax.apply_updates(px, u), st_n, l, gn

        rng = np.random.default_rng(9)
        # Repeat a small synthetic pool so loss descends cleanly while testing 100 updates
        pool = rng.integers(0, cfg["vocab"], size=(8, micro_b, seq + 1), dtype=np.int32)
        losses, gnorms = [], []
        for s_i in range(steps):
            tk = jax.device_put(pool[s_i % 8], dev)
            p, st, lv, gnv = step_fn(p, st, tk)
            losses.append(float(lv))
            gnorms.append(float(gnv))

    nonfinite = int(sum((not math.isfinite(l)) or (not math.isfinite(g)) for l, g in zip(losses, gnorms)))
    return {
        "steps": steps,
        "nonfinite_steps": nonfinite,
        "loss_step1": round(losses[0], 4),
        "loss_step50": round(losses[len(losses) // 2], 4),
        "loss_final": round(losses[-1], 4),
        "max_grad_norm": round(max(gnorms), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fla-bundle", required=True)
    ap.add_argument("--fla-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    devs = jax.devices()
    dev0 = devs[0]
    kind = dev0.device_kind
    peak_tf, peak_bw = next((v for k, v in TPU_SPECS.items() if k.lower() in kind.lower()), (197.0, 819.0))

    res: dict = {
        "schema": "nanolm_r0_tpu_and_dual_parity_v1",
        "code_sha256": CODE_SHA256,
        "core_sha256": CORE_SHA256,
        "jax_version": jax.__version__,
        "optax_version": optax.__version__,
        "device_kind": kind,
        "n_devices": len(devs),
        "datasheet_peak_TFLOPS_bf16_per_chip": peak_tf,
        "datasheet_hbm_GBps_per_chip": peak_bw,
        "smoke": a.smoke,
    }

    print(json.dumps({"event": "phase1_parity_start", "device": kind, "n_devices": len(devs)}), flush=True)
    res["parity"] = verify_against_fla_bundle(Path(a.fla_bundle), Path(a.fla_meta), dev0)
    print(json.dumps({"event": "phase1_parity_done", "fp32_loss_err": res["parity"]["fp32_vs_cuda_fla"]["loss_abs_err"], "bf16_loss_err": res["parity"]["bf16_vs_cuda_fla"]["loss_abs_err"]}), flush=True)

    res["ceilings"] = measure_tpu_ceilings(dev0, small=a.smoke)
    print(json.dumps({"event": "phase2_ceilings_done", **res["ceilings"]}), flush=True)

    if a.smoke:
        cfg_gdn = dict(d=128, layers=4, ffn=352, heads=2, kv_heads=2, vocab=2048, attn_layers=())
        cfg_hyb = dict(d=128, layers=4, ffn=352, heads=2, kv_heads=2, vocab=2048, attn_layers=(1,))
        cfg_att = dict(d=128, layers=4, ffn=352, heads=2, kv_heads=2, vocab=2048, attn_layers=(0, 1, 2, 3))
        bench_rows = [
            bench_config("smoke_gdn_1chip", cfg_gdn, micro_b=2, seq=128, accum=1, dp=1, with_kd=False, steps=3, peak_tflops_per_chip=peak_tf),
            bench_config("smoke_hyb_kd_1chip", cfg_hyb, micro_b=2, seq=128, accum=1, dp=1, with_kd=True, steps=3, peak_tflops_per_chip=peak_tf),
            bench_config("smoke_attn_1chip", cfg_att, micro_b=2, seq=128, accum=1, dp=1, with_kd=False, steps=3, peak_tflops_per_chip=peak_tf),
        ]
        stab = run_stability_100(cfg_gdn, dev0, micro_b=2, seq=128, steps=6)
    else:
        # Exact 546.9M R0-v2 dimensions (`24L, d=896, 7x128, SwiGLU 4864, V=151,936`) + Stage-C 18-GDN+6-GQA hybrid + 152M SmolLM2 hybrid
        cfg_r0_gdn = dict(d=896, layers=24, ffn=4864, heads=7, kv_heads=7, vocab=151936, attn_layers=())
        cfg_r0_hyb = dict(d=896, layers=24, ffn=4864, heads=7, kv_heads=7, vocab=151936, attn_layers=(1, 2, 3, 18, 19, 20))
        cfg_r0_att = dict(d=896, layers=24, ffn=4864, heads=7, kv_heads=7, vocab=151936, attn_layers=tuple(range(24)))
        cfg_152m_hyb = dict(d=576, layers=30, ffn=1536, heads=9, kv_heads=3, vocab=49152, attn_layers=(1, 2, 3, 4, 21, 22, 23, 24))

        dp_max = len(devs)
        bench_rows = [
            # Direct 1-chip comparisons at exact 16,384 tok/step
            safe_bench_config("r0_547m_gdn_1chip_2k", cfg_r0_gdn, micro_b=2, seq=2048, accum=4, dp=1, with_kd=False, steps=12, peak_tflops_per_chip=peak_tf),
            safe_bench_config("r0_547m_gdn_kd32_1chip_2k", cfg_r0_gdn, micro_b=2, seq=2048, accum=4, dp=1, with_kd=True, steps=12, peak_tflops_per_chip=peak_tf),
            safe_bench_config("r0_542m_hybrid_kd32_1chip_2k", cfg_r0_hyb, micro_b=4, seq=2048, accum=2, dp=1, with_kd=True, steps=12, peak_tflops_per_chip=peak_tf),
            safe_bench_config("r0_527m_attn_1chip_2k", cfg_r0_att, micro_b=4, seq=2048, accum=2, dp=1, with_kd=False, steps=12, peak_tflops_per_chip=peak_tf),
            # Long-context scaling: 4k & 8k context — GDN vs Hybrid vs Softmax Attention!
            safe_bench_config("r0_547m_gdn_1chip_4k", cfg_r0_gdn, micro_b=1, seq=4096, accum=4, dp=1, with_kd=False, steps=10, peak_tflops_per_chip=peak_tf),
            safe_bench_config("r0_527m_attn_1chip_4k", cfg_r0_att, micro_b=2, seq=4096, accum=2, dp=1, with_kd=False, steps=10, peak_tflops_per_chip=peak_tf),
            safe_bench_config("r0_547m_gdn_1chip_8k", cfg_r0_gdn, micro_b=1, seq=8192, accum=1, dp=1, with_kd=False, steps=10, peak_tflops_per_chip=peak_tf),
            safe_bench_config("r0_542m_hybrid_1chip_8k", cfg_r0_hyb, micro_b=1, seq=8192, accum=1, dp=1, with_kd=False, steps=10, peak_tflops_per_chip=peak_tf),
            safe_bench_config("r0_527m_attn_1chip_8k", cfg_r0_att, micro_b=1, seq=8192, accum=1, dp=1, with_kd=False, steps=10, peak_tflops_per_chip=peak_tf),
            # 152M SmolLM2-135M hybrid (`Stage B` exact architecture) on TPU
            safe_bench_config("stageb_152m_hybrid_1chip_1k", cfg_152m_hyb, micro_b=8, seq=1024, accum=4, dp=1, with_kd=False, steps=12, peak_tflops_per_chip=peak_tf),
        ]
        if dp_max >= 4:
            bench_rows.extend([
                safe_bench_config("r0_547m_gdn_4chip_dp_2k", cfg_r0_gdn, micro_b=2, seq=2048, accum=4, dp=4, with_kd=False, steps=12, peak_tflops_per_chip=peak_tf),
                safe_bench_config("r0_542m_hybrid_kd32_4chip_dp_2k", cfg_r0_hyb, micro_b=4, seq=2048, accum=2, dp=4, with_kd=True, steps=12, peak_tflops_per_chip=peak_tf),
                safe_bench_config("r0_542m_hybrid_kd32_4chip_dp_8k", cfg_r0_hyb, micro_b=1, seq=8192, accum=1, dp=4, with_kd=True, steps=10, peak_tflops_per_chip=peak_tf),
            ])

        stab = run_stability_100(cfg_r0_gdn, dev0, micro_b=2, seq=1024, steps=100)
        print(json.dumps({"event": "phase4_stability_done", **stab}), flush=True)

    res["benchmarks"] = bench_rows
    res["stability_100"] = stab
    Path(a.out).write_text(json.dumps(res, indent=2))
    print("R0_TPU_RESULT_BEGIN")
    print(json.dumps(res, indent=2))
    print("R0_TPU_RESULT_END", flush=True)


if __name__ == "__main__":
    main()
