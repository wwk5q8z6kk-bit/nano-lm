#!/usr/bin/env python3
"""EXP-U02 Part B: Real Subword Corpus Transfer on /Volumes/Express1M2/nanolm-lab/data/v1/.

Evaluates 5 non-embedding iso-parameter (~853K non-emb / ~5.05M total) contenders selected
strictly from Part A across 2 seeds x 1,000 steps:
  - M0_Dense_GQA_RoPE      (ffn=384, mps)
  - M1_GDN1_Cone_Closed    (ffn=334, mps, chunked WY C=32)
  - M9_DualChannel_Pointer (ffn=312, mps, chunked WY C=32)
  - M5_HSPA_LieSO3         (ffn=496, cpu threads=1 parallel workers)
  - M7_SkewCayley_Flow     (ffn=254, cpu threads=1 parallel workers)
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from research.unified_loop.mechanisms.zoo import (  # noqa: E402
    CausalSelfAttnMixer,
    GDN1Mixer,
    MECHANISM_SPECS,
    SkewCayleyFlowMixer,
    UnifiedBlockLM,
    _apply_rope,
    _center_l2,
    _rope_angles,
    count_trainable_params,
)

DATA_DIR = Path("/Volumes/Express1M2/nanolm-lab/data/v1")
MANIFEST_PATH = DATA_DIR / "manifest.json"
OUT_DIR = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U02_cross_paradigm_showdown"

CONTENDER_CONFIGS: dict[str, dict[str, Any]] = {
    "M0_Dense_GQA_RoPE": {"ffn": 384, "device": "mps"},
    "M1_GDN1_Cone_Closed": {"ffn": 334, "device": "mps"},
    "M9_DualChannel_Pointer": {"ffn": 312, "device": "mps"},
    "M5_HSPA_LieSO3": {"ffn": 496, "device": "cpu"},
    "M7_SkewCayley_Flow": {"ffn": 254, "device": "cpu"},
}

D_MODEL = 128
N_HEADS = 4
BATCH_SIZE = 8
SEQ_LEN = 128
STEPS = 1000
LR = 1.5e-3
WEIGHT_DECAY = 0.01
WARMUP = 100
SEEDS = [0, 1]


def count_non_emb_params(model: UnifiedBlockLM) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith("emb."))


def _compute_loss(model: UnifiedBlockLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = model(x)
    V = model.vocab_size
    if model.spec.use_pointer:
        return F.nll_loss(out.reshape(-1, V), y.reshape(-1), reduction="mean")
    return F.cross_entropy(out.reshape(-1, V), y.reshape(-1), reduction="mean")


def _cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    prog = float(step - warmup) / float(max(1, total_steps - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))


@torch.no_grad()
def eval_bpb_and_extrap(
    model: UnifiedBlockLM,
    manifest: dict[str, Any],
    dev: torch.device,
    lengths: tuple[int, ...] = (128, 256),
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for L in lengths:
        dom_bpb: dict[str, float] = {}
        for dom in ("web", "textbook", "math", "code"):
            arr = np.memmap(DATA_DIR / f"val_{dom}.bin", dtype=np.uint16, mode="r")
            windows = 24
            bs = 4
            stride = max(1, (len(arr) - L - 1) // windows)
            starts = [i * stride for i in range(windows)]
            tot_nats, tot_tok = 0.0, 0
            for i in range(0, len(starts), bs):
                xb = np.stack([arr[s : s + L + 1] for s in starts[i : i + bs]]).astype(np.int64)
                tx = torch.from_numpy(xb).to(dev)
                out_l = model(tx[:, :-1])
                V = model.vocab_size
                if model.spec.use_pointer:
                    nll = F.nll_loss(out_l.reshape(-1, V), tx[:, 1:].reshape(-1), reduction="sum")
                else:
                    nll = F.cross_entropy(out_l.reshape(-1, V), tx[:, 1:].reshape(-1), reduction="sum")
                tot_nats += float(nll.item())
                tot_tok += int(tx[:, 1:].numel())
            bpt = manifest["domains"][dom]["val"]["bytes_per_token"]
            dom_bpb[dom] = round((tot_nats / tot_tok) / math.log(2.0) / bpt, 5)
        out[f"L_{L}"] = dom_bpb
    return out


@torch.no_grad()
def eval_copy_probes(model: UnifiedBlockLM, vocab: int, dev: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for L_half in (32, 64, 128):
        g = torch.Generator().manual_seed(1234 + L_half)
        xs = torch.randint(64, vocab - 64, (16, L_half), generator=g)
        xc = torch.cat([xs, xs], dim=1)
        corr, tot_loss, tot_c = 0, 0.0, 0
        for i in range(0, len(xc), 4):
            xb = xc[i : i + 4].to(dev)
            pred = model(xb[:, :-1])
            pred_half = pred[:, L_half:, :].float()
            tgt_half = xb[:, L_half + 1 :]
            if model.spec.use_pointer:
                loss = F.nll_loss(pred_half.reshape(-1, vocab), tgt_half.reshape(-1), reduction="sum")
            else:
                loss = F.cross_entropy(pred_half.reshape(-1, vocab), tgt_half.reshape(-1), reduction="sum")
            tot_loss += float(loss.item())
            corr += int((pred_half.argmax(dim=-1) == tgt_half).sum().item())
            tot_c += int(tgt_half.numel())
        out[f"copy_{L_half}"] = {"acc": round(corr / tot_c, 4), "loss": round(tot_loss / tot_c, 4)}
    return out


@torch.no_grad()
def eval_real_haystack_mqar(model: UnifiedBlockLM, vocab: int, dev: torch.device) -> dict[str, Any]:
    web_arr = np.memmap(DATA_DIR / "val_web.bin", dtype=np.uint16, mode="r")
    res: dict[str, Any] = {}
    for band, (lo_id, hi_id) in (("common_ids", (200, 2000)), ("rare_tail_ids", (20000, vocab - 64))):
        band_out: dict[str, float] = {}
        for T_ctx, n_needles in ((128, 4), (256, 4)):
            rng = np.random.default_rng(777 + T_ctx * 13 + n_needles + lo_id)
            n_trials = 24
            seqs, targets = [], []
            for b in range(n_trials):
                offset = (b * 997) % (len(web_arr) - T_ctx - 10)
                ctx = np.array(web_arr[offset : offset + T_ctx], dtype=np.int64)
                used = set(ctx.tolist())
                cand: list[int] = []
                while len(cand) < 2 * n_needles:
                    tok = int(rng.integers(lo_id, hi_id))
                    if tok not in used:
                        used.add(tok)
                        cand.append(tok)
                pairs = [(cand[2 * i], cand[2 * i + 1]) for i in range(n_needles)]
                positions = np.linspace(8, int(T_ctx * 0.6), n_needles, dtype=int)
                for p, (kt, vt) in zip(positions, pairs):
                    ctx[p] = kt
                    ctx[p + 1] = vt
                qk, qv = pairs[int(rng.integers(0, n_needles))]
                ctx[-1] = qk
                seqs.append(ctx)
                targets.append(qv)
            preds_list = []
            for i in range(0, n_trials, 4):
                xb = torch.tensor(np.stack(seqs[i : i + 4]), dtype=torch.long, device=dev)
                out_l = model(xb)[:, -1, :]
                preds_list.append(out_l.argmax(dim=-1).cpu())
            preds = torch.cat(preds_list, dim=0)
            tgt = torch.tensor(targets, dtype=torch.long)
            band_out[f"T{T_ctx}_N{n_needles}"] = round(float((preds == tgt).float().mean().item()), 4)
        res[band] = band_out
    return res


@torch.no_grad()
def eval_key_geometry(model: UnifiedBlockLM, dev: torch.device) -> list[dict[str, Any]]:
    """Measure per-layer key effective rank and mean off-diagonal key cosine (as in EXP-U01)."""
    web_arr = np.memmap(DATA_DIR / "val_web.bin", dtype=np.uint16, mode="r")
    xb = torch.from_numpy(np.stack([web_arr[i * 512 : i * 512 + 128] for i in range(4)]).astype(np.int64)).to(dev)
    h = F.normalize(model.emb(xb).float(), dim=-1)
    B, T, D = h.shape
    stats: list[dict[str, Any]] = []
    for li, blk in enumerate(model.blocks):
        xn = blk.n1(h)
        mix = blk.mix
        if isinstance(mix, CausalSelfAttnMixer):
            _, k, _ = mix.qkv(xn).view(B, T, 3, mix.h, mix.hd).unbind(2)
            kn = F.normalize(k.float(), dim=-1)
            cos, sin = _rope_angles(T, mix.hd, mix.theta, dev)
            k_eff = _apply_rope(kn, cos, sin)
        elif isinstance(mix, GDN1Mixer):
            _, k, _ = mix.conv(mix.qkv(xn)).split(D, dim=-1)
            k_eff = _center_l2(k.view(B, T, mix.h, mix.hd), center=mix.full_sphere)
        elif isinstance(mix, SkewCayleyFlowMixer):
            _, k, _ = mix.qkv(xn).view(B, T, 3, mix.h, mix.hd).unbind(2)
            k_eff = _center_l2(k, center=True)
        else:
            stats.append({"layer": li, "mean_key_eff_rank": None, "mean_offdiag_key_cos": None})
            h = blk(h)
            continue

        # Compute per-head effective rank and mean off-diag cosine on CPU float32
        ranks, cosines = [], []
        k_cpu = k_eff.detach().float().cpu()  # [B, T, H, hd]
        for hi in range(k_cpu.shape[2]):
            K = k_cpu[0, :64, hi, :]  # [64, hd]
            svals = torch.linalg.svdvals(K)
            p = (svals ** 2) / (svals ** 2).sum().clamp_min(1e-12)
            ranks.append(float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()))
            G = K @ K.T
            off = ~torch.eye(G.shape[0], dtype=torch.bool)
            cosines.append(float(G[off].abs().mean().item()))
        stats.append({
            "layer": li,
            "hd": int(k_cpu.shape[3]),
            "mean_key_eff_rank": round(sum(ranks) / len(ranks), 3),
            "mean_offdiag_key_cos": round(sum(cosines) / len(cosines), 4),
        })
        h = blk(h)
    return stats


def run_single_job(args: tuple[str, int, str]) -> tuple[str, int, dict[str, Any]]:
    mech, seed, dev_str = args
    if dev_str == "cpu":
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    vocab = int(manifest["vocab"])
    train_arr = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    n_train = len(train_arr)

    cfg = CONTENDER_CONFIGS[mech]
    dev = torch.device(dev_str)
    torch.manual_seed(1000 + seed * 97 + sum(ord(c) for c in mech))
    rng = np.random.default_rng(70_000 + seed * 131 + sum(ord(c) for c in mech))

    model = UnifiedBlockLM(
        vocab, D_MODEL, mech, n_heads=N_HEADS, ffn_hidden_override=cfg["ffn"]
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_params = count_trainable_params(model)
    non_emb = count_non_emb_params(model)

    loss_curve: dict[int, float] = {}
    t_train0 = time.perf_counter()
    model.train()
    for step in range(STEPS):
        lr_now = _cosine_lr(step, STEPS, WARMUP, LR)
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        starts = rng.integers(0, n_train - SEQ_LEN - 1, size=BATCH_SIZE)
        xb = np.stack([train_arr[s : s + SEQ_LEN + 1] for s in starts]).astype(np.int64)
        tx = torch.from_numpy(xb).to(dev)
        loss = _compute_loss(model, tx[:, :-1], tx[:, 1:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step + 1 in (200, 500, 800, 1000):
            loss_curve[step + 1] = round(float(loss.item()), 4)

    if dev_str == "mps":
        torch.mps.synchronize()
    train_s = time.perf_counter() - t_train0
    tok_per_s = round((STEPS * BATCH_SIZE * SEQ_LEN) / max(1e-6, train_s), 1)

    model.eval()
    bpb = eval_bpb_and_extrap(model, manifest, dev)
    copy_res = eval_copy_probes(model, vocab, dev)
    mqar_res = eval_real_haystack_mqar(model, vocab, dev)
    geom = eval_key_geometry(model, dev)

    return mech, seed, {
        "device": dev_str,
        "ffn_hidden": cfg["ffn"],
        "total_params": total_params,
        "non_emb_params": non_emb,
        "train_s": round(train_s, 2),
        "tok_per_s": tok_per_s,
        "train_loss_curve": loss_curve,
        "val_bpb": bpb,
        "copy_probes": copy_res,
        "real_haystack_mqar": mqar_res,
        "key_geometry": geom,
    }


def _agg_nested(runs: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float]:
    vals = []
    for r in runs:
        cur: Any = r
        for p in path:
            cur = cur[p]
        vals.append(float(cur))
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return {"mean": round(m, 4), "std": round(math.sqrt(var), 4)}


def main() -> int:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mps_jobs = [(m, s, "mps") for m, c in CONTENDER_CONFIGS.items() if c["device"] == "mps" for s in SEEDS]
    cpu_jobs = [(m, s, "cpu") for m, c in CONTENDER_CONFIGS.items() if c["device"] == "cpu" for s in SEEDS]

    all_runs: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in CONTENDER_CONFIGS}

    # Launch CPU workers concurrently while MPS runs its queue (zero resource conflict)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(cpu_jobs), mp_context=ctx) as pool:
        cpu_futs = [pool.submit(run_single_job, j) for j in cpu_jobs]
        for job in mps_jobs:
            mech, seed, res = run_single_job(job)
            all_runs[mech][seed] = res
            print(
                f"  [MPS] {mech:26s} seed={seed} web@128={res['val_bpb']['L_128']['web']:.4f} "
                f"web@256={res['val_bpb']['L_256']['web']:.4f} copy32={res['copy_probes']['copy_32']['acc']:.3f} "
                f"tok/s={res['tok_per_s']:.0f} ({res['train_s']}s)",
                flush=True,
            )
        for fut in cpu_futs:
            mech, seed, res = fut.result()
            all_runs[mech][seed] = res
            print(
                f"  [CPU] {mech:26s} seed={seed} web@128={res['val_bpb']['L_128']['web']:.4f} "
                f"web@256={res['val_bpb']['L_256']['web']:.4f} copy32={res['copy_probes']['copy_32']['acc']:.3f} "
                f"tok/s={res['tok_per_s']:.0f} ({res['train_s']}s)",
                flush=True,
            )

    summary: dict[str, Any] = {}
    for mech in CONTENDER_CONFIGS:
        runs = [all_runs[mech][s] for s in SEEDS]
        summary[mech] = {
            "device": runs[0]["device"],
            "total_params": runs[0]["total_params"],
            "non_emb_params": runs[0]["non_emb_params"],
            "mean_tok_per_s": round(sum(r["tok_per_s"] for r in runs) / len(runs), 1),
            "val_bpb": {
                L: {dom: _agg_nested(runs, ("val_bpb", L, dom)) for dom in ("web", "textbook", "math", "code")}
                for L in ("L_128", "L_256")
            },
            "copy_probes": {
                ck: _agg_nested(runs, ("copy_probes", ck, "acc"))
                for ck in ("copy_32", "copy_64", "copy_128")
            },
            "real_haystack_mqar": {
                band: {tc: _agg_nested(runs, ("real_haystack_mqar", band, tc)) for tc in ("T128_N4", "T256_N4")}
                for band in ("common_ids", "rare_tail_ids")
            },
            "seeds": {str(s): all_runs[mech][s] for s in SEEDS},
        }

    payload = {
        "schema": "nanolm.unified_loop.u02b.v1",
        "paid_compute_usd": 0.0,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "config": {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "steps": STEPS,
            "seeds": SEEDS,
        },
        "models": summary,
    }
    (OUT_DIR / "results_part_b.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[U02-B] Finished all 5 models x 2 seeds in {payload['elapsed_s']}s -> {OUT_DIR / 'results_part_b.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
