#!/usr/bin/env python3
"""EXP-U06: 2x2 Factorial Spinor-Pointer Synthesis (M10-M12 + M0, M5, M9 controls)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
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
    MECHANISM_SPECS,
    UnifiedBlockLM,
    count_trainable_params,
    verify_strict_causality,
)
from research.unified_loop.scripts.run_u02_cross_paradigm_showdown import (  # noqa: E402
    _run_job as _run_u02a_job,
    mean_std,
)

DATA_DIR = Path("/Volumes/Express1M2/nanolm-lab/data/v1")
MANIFEST_PATH = DATA_DIR / "manifest.json"
OUT_DIR = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U06_spinor_pointer_synthesis"

CONTENDERS: dict[str, int] = {
    "M0_Dense_GQA_RoPE": 384,
    "M5_HSPA_LieSO3": 496,
    "M9_DualChannel_Pointer": 312,
    "M10_HSPA_DensePointer": 474,
    "M11_FullSphere_SparseOWPointer": 312,
    "M12_HSPA_SparseOWPointer": 474,
}

TASKS_PART_A = [
    "S4_Holonomy_8x",
    "MQAR_Selective_Overwrite_4x",
    "Unseen_OpenVocab_Copy",
    "Parity_Dyck_4x",
    "Hierarchy_Denom_Rollup_4x",
]

D_MODEL = 128
N_HEADS = 4
BATCH_SIZE = 8
SEQ_LEN = 128
STEPS = 1000
LR = 1.5e-3
WEIGHT_DECAY = 0.01
WARMUP = 100
SEEDS_A = [0, 1, 2]
SEEDS_B = [0, 1]


def count_non_emb_params(model: UnifiedBlockLM) -> int:
    return sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not n.startswith("emb."))


def _compute_loss(model: UnifiedBlockLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = model(x)
    V = model.vocab_size
    if model.spec.has_pointer:
        return F.nll_loss(out.reshape(-1, V), y.reshape(-1), reduction="mean")
    return F.cross_entropy(out.reshape(-1, V), y.reshape(-1), reduction="mean")


def _cosine_lr(step: int, total_steps: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(1, warmup))
    prog = float(step - warmup) / float(max(1, total_steps - warmup))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))


def run_part_a() -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    t0 = time.perf_counter()

    causality = verify_strict_causality(vocab_size=72, d_model=32)
    bad = [k for k in CONTENDERS if causality[k]["causal_leak_max"] > 1e-6 or not causality[k]["finite"]]
    if bad:
        raise RuntimeError(f"Part A causality failure: {bad}")
    print("[U06-A] d=32 causality passed for all 6 contenders.", flush=True)

    mechs = list(CONTENDERS.keys())
    steps_a = 200
    jobs = [(m, t, s, steps_a) for t in TASKS_PART_A for m in mechs for s in SEEDS_A]
    pending: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = {(t, m): [] for t in TASKS_PART_A for m in mechs}
    results: dict[str, Any] = {
        "schema": "nanolm.unified_loop.u06a.v1",
        "paid_compute_usd": 0.0,
        "steps_per_run": steps_a,
        "seeds": SEEDS_A,
        "tasks": {t: {} for t in TASKS_PART_A},
    }

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=5, mp_context=ctx) as pool:
        futs = [pool.submit(_run_u02a_job, job) for job in jobs]
        for fut in as_completed(futs):
            task, mech, _seed, job_s, r = fut.result()
            pending[(task, mech)].append((job_s, r))
            if len(pending[(task, mech)]) == len(SEEDS_A):
                items = pending[(task, mech)]
                runs = [x[1] for x in items]
                wall_s = round(sum(x[0] for x in items), 2)
                agg: dict[str, Any] = {
                    "params": runs[0]["params"],
                    "wall_s": wall_s,
                    "id_acc": mean_std([x["id_acc"] for x in runs]),
                    "ood_acc": mean_std([x["ood_acc"] for x in runs]),
                }
                for opt_k in ("ood_overwritten_acc", "ood_close_rollup_acc"):
                    if opt_k in runs[0]:
                        agg[opt_k] = mean_std([x[opt_k] for x in runs])
                results["tasks"][task][mech] = agg
                extra = f" ov={agg['ood_overwritten_acc']['mean']:.3f}" if "ood_overwritten_acc" in agg else ""
                if "ood_close_rollup_acc" in agg:
                    extra += f" close={agg['ood_close_rollup_acc']['mean']:.3f}"
                print(
                    f"  [U06-A {task[:10]:10s}] {mech:30s} ID={agg['id_acc']['mean']:.3f} "
                    f"OOD={agg['ood_acc']['mean']:.3f}±{agg['ood_acc']['std']:.3f}{extra} (cpu={wall_s}s)",
                    flush=True,
                )

    for t in TASKS_PART_A:
        results["tasks"][t] = {m: results["tasks"][t][m] for m in mechs}
    results["elapsed_s"] = round(time.perf_counter() - t0, 2)
    (OUT_DIR / "results_part_a.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[U06-A] Complete in {results['elapsed_s']}s", flush=True)
    return results


@torch.no_grad()
def eval_bpb_multi_horizon(
    model: UnifiedBlockLM,
    manifest: dict[str, Any],
    dev: torch.device,
    lengths: tuple[int, ...] = (128, 256, 512),
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for L in lengths:
        dom_bpb: dict[str, float] = {}
        bs = 4 if L <= 256 else 2
        windows = 24 if L <= 256 else 16
        for dom in ("web", "textbook", "math", "code"):
            arr = np.memmap(DATA_DIR / f"val_{dom}.bin", dtype=np.uint16, mode="r")
            stride = max(1, (len(arr) - L - 1) // windows)
            starts = [i * stride for i in range(windows)]
            tot_nats, tot_tok = 0.0, 0
            for i in range(0, len(starts), bs):
                xb = np.stack([arr[s : s + L + 1] for s in starts[i : i + bs]]).astype(np.int64)
                tx = torch.from_numpy(xb).to(dev)
                out_l = model(tx[:, :-1])
                V = model.vocab_size
                if model.spec.has_pointer:
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
    for L_half in (32, 64, 128, 256):
        g = torch.Generator().manual_seed(1234 + L_half)
        xs = torch.randint(64, vocab - 64, (16, L_half), generator=g)
        xc = torch.cat([xs, xs], dim=1)
        bs = 4 if L_half <= 128 else 2
        corr, tot_loss, tot_c = 0, 0.0, 0
        for i in range(0, len(xc), bs):
            xb = xc[i : i + bs].to(dev)
            pred = model(xb[:, :-1])
            pred_half = pred[:, L_half:, :].float()
            tgt_half = xb[:, L_half + 1 :]
            if model.spec.has_pointer:
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
        for T_ctx, n_needles in ((128, 4), (256, 4), (512, 4)):
            rng = np.random.default_rng(777 + T_ctx * 13 + n_needles + lo_id)
            n_trials = 24
            bs = 4 if T_ctx <= 256 else 2
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
            for i in range(0, n_trials, bs):
                xb = torch.tensor(np.stack(seqs[i : i + bs]), dtype=torch.long, device=dev)
                preds_list.append(model(xb)[:, -1, :].argmax(dim=-1).cpu())
            preds = torch.cat(preds_list, dim=0)
            tgt = torch.tensor(targets, dtype=torch.long)
            band_out[f"T{T_ctx}_N{n_needles}"] = round(float((preds == tgt).float().mean().item()), 4)
        res[band] = band_out
    return res


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


def run_part_b() -> dict[str, Any]:
    t0 = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    vocab = int(manifest["vocab"])
    train_arr = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    n_train = len(train_arr)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Strict causality gate at d_model=128, vocab=32768 before training
    x1 = torch.randint(0, vocab, (2, 48))
    x2 = x1.clone()
    x2[:, 36:] = (x2[:, 36:] + 11) % vocab
    for mech, ffn in CONTENDERS.items():
        cm = UnifiedBlockLM(vocab, D_MODEL, mech, n_heads=N_HEADS, ffn_hidden_override=ffn).eval()
        with torch.no_grad():
            y1, y2 = cm(x1), cm(x2)
        leak = float((y1[:, :36] - y2[:, :36]).abs().max().item())
        if leak > 1e-6 or not bool(torch.isfinite(y1).all()):
            raise RuntimeError(f"d=128 causality failure for {mech}: leak={leak}")
    print(f"[U06-B] d=128 causality verified for all 6 contenders on {dev}.", flush=True)

    all_runs: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in CONTENDERS}
    for mech, ffn in CONTENDERS.items():
        for seed in SEEDS_B:
            torch.manual_seed(1000 + seed * 97 + sum(ord(c) for c in mech))
            rng = np.random.default_rng(70_000 + seed * 131 + sum(ord(c) for c in mech))

            model = UnifiedBlockLM(vocab, D_MODEL, mech, n_heads=N_HEADS, ffn_hidden_override=ffn).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            total_p = count_trainable_params(model)
            non_emb_p = count_non_emb_params(model)

            loss_curve: dict[int, float] = {}
            tr0 = time.perf_counter()
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

            if dev.type == "mps":
                torch.mps.synchronize()
            train_s = time.perf_counter() - tr0
            tok_s = round((STEPS * BATCH_SIZE * SEQ_LEN) / max(1e-6, train_s), 1)

            model.eval()
            bpb = eval_bpb_multi_horizon(model, manifest, dev, lengths=(128, 256, 512))
            cp = eval_copy_probes(model, vocab, dev)
            mq = eval_real_haystack_mqar(model, vocab, dev)

            all_runs[mech][seed] = {
                "device": str(dev),
                "ffn_hidden": ffn,
                "total_params": total_p,
                "non_emb_params": non_emb_p,
                "train_s": round(train_s, 2),
                "tok_per_s": tok_s,
                "train_loss_curve": loss_curve,
                "val_bpb": bpb,
                "copy_probes": cp,
                "real_haystack_mqar": mq,
            }
            print(
                f"  [U06-B] {mech:30s} s={seed} web={bpb['L_128']['web']:.4f}/{bpb['L_256']['web']:.4f}/{bpb['L_512']['web']:.4f} "
                f"code512={bpb['L_512']['code']:.4f} copy256={cp['copy_256']['acc']:.3f} tok/s={tok_s:.0f} ({train_s:.1f}s)",
                flush=True,
            )
            del model, opt
            if dev.type == "mps":
                torch.mps.empty_cache()

        # Incremental save per finished contender
        summary = {}
        for m_done in CONTENDERS:
            if len(all_runs[m_done]) < len(SEEDS_B):
                continue
            runs = [all_runs[m_done][s] for s in SEEDS_B]
            summary[m_done] = {
                "device": runs[0]["device"],
                "total_params": runs[0]["total_params"],
                "non_emb_params": runs[0]["non_emb_params"],
                "mean_tok_per_s": round(sum(r["tok_per_s"] for r in runs) / len(runs), 1),
                "val_bpb": {
                    L: {dom: _agg_nested(runs, ("val_bpb", L, dom)) for dom in ("web", "textbook", "math", "code")}
                    for L in ("L_128", "L_256", "L_512")
                },
                "copy_probes": {ck: _agg_nested(runs, ("copy_probes", ck, "acc")) for ck in ("copy_32", "copy_64", "copy_128", "copy_256")},
                "real_haystack_mqar": {
                    band: {tc: _agg_nested(runs, ("real_haystack_mqar", band, tc)) for tc in ("T128_N4", "T256_N4", "T512_N4")}
                    for band in ("common_ids", "rare_tail_ids")
                },
                "seeds": {str(s): all_runs[m_done][s] for s in SEEDS_B},
            }
        payload = {
            "schema": "nanolm.unified_loop.u06b.v1",
            "paid_compute_usd": 0.0,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "config": {"d_model": D_MODEL, "n_heads": N_HEADS, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "steps": STEPS, "seeds": SEEDS_B},
            "models": summary,
        }
        (OUT_DIR / "results_part_b.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[U06-B] Finished all 6 contenders x 2 seeds in {payload['elapsed_s']}s", flush=True)
    return payload


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_part_a()
    run_part_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
