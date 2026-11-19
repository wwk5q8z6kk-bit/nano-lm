#!/usr/bin/env python3
"""EXP-U07: First-Principles Enhancements (Triangular WY + Evidential Cosine Pointer + Gauge-SU(2) Holonomy)."""
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

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from research.unified_loop.mechanisms.zoo import (  # noqa: E402
    UnifiedBlockLM,
    count_trainable_params,
    verify_strict_causality,
)
from research.unified_loop.scripts.run_u02_cross_paradigm_showdown import (  # noqa: E402
    _run_job as _run_u02a_job,
    mean_std,
)
from research.unified_loop.scripts.run_u06_spinor_pointer_synthesis import (  # noqa: E402
    BATCH_SIZE,
    D_MODEL,
    DATA_DIR,
    LR,
    MANIFEST_PATH,
    N_HEADS,
    SEEDS_A,
    SEEDS_B,
    SEQ_LEN,
    STEPS,
    TASKS_PART_A,
    WARMUP,
    WEIGHT_DECAY,
    _compute_loss,
    _cosine_lr,
    count_non_emb_params,
    eval_bpb_multi_horizon,
    eval_copy_probes,
    eval_real_haystack_mqar,
)

OUT_DIR = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U07_first_principles_enhancements"

CONTENDERS: dict[str, int] = {
    "M12_HSPA_SparseOWPointer": 474,
    "M13_HSPA_EvidCosSparsePtr": 463,
    "M14_GaugeSU2_Pure": 462,
    "M15_GaugeSU2_EvidCosSparsePtr": 429,
    "M16_FullSphereTri_EvidCosSparsePtr": 301,
}

ALGEBRA_TASKS_A2 = ["S4_Holonomy_8x", "Hierarchy_Denom_Rollup_4x"]
ALGEBRA_MECHS_A2 = [
    "M0_Dense_GQA_RoPE",
    "M5_HSPA_LieSO3",
    "M12_HSPA_SparseOWPointer",
    "M14_GaugeSU2_Pure",
    "M15_GaugeSU2_EvidCosSparsePtr",
]


def _sanitize_json(obj: Any) -> tuple[Any, bool]:
    """Convert non-finite floats to None for strict RFC-8259 JSON and report whether any occurred."""
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj, False
        return None, True
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        bad = False
        for k, v in obj.items():
            sv, b = _sanitize_json(v)
            out[k] = sv
            bad = bad or b
        return out, bad
    if isinstance(obj, list):
        out_l: list[Any] = []
        bad = False
        for v in obj:
            sv, b = _sanitize_json(v)
            out_l.append(sv)
            bad = bad or b
        return out_l, bad
    return obj, False


def _write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    clean, has_nonfinite = _sanitize_json(payload)
    if isinstance(clean, dict):
        clean["has_nonfinite"] = bool(clean.get("has_nonfinite", False) or has_nonfinite)
    path.write_text(json.dumps(clean, indent=2, allow_nan=False), encoding="utf-8")


def _agg_nested_safe(runs: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float | None]:
    vals: list[float] = []
    for r in runs:
        cur: Any = r
        for p in path:
            cur = cur[p]
        val = float(cur)
        if not math.isfinite(val):
            return {"mean": None, "std": None}
        vals.append(val)
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / max(1, len(vals) - 1)
    return {"mean": round(m, 4), "std": round(math.sqrt(var), 4)}


def _run_cpu_task_suite(
    tasks: list[str],
    mechs: list[str],
    seeds: list[int],
    steps: int,
    out_path: Path,
    schema: str,
    tag: str,
) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    t0 = time.perf_counter()

    causality = verify_strict_causality(vocab_size=72, d_model=32)
    bad = [k for k in mechs if causality[k]["causal_leak_max"] > 1e-6 or not causality[k]["finite"]]
    if bad:
        raise RuntimeError(f"{tag} causality failure: {bad}")
    print(f"[{tag}] d=32 causality verified for {len(mechs)} mechanisms.", flush=True)

    jobs = [(m, t, s, steps) for t in tasks for m in mechs for s in seeds]
    pending: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = {(t, m): [] for t in tasks for m in mechs}
    results: dict[str, Any] = {
        "schema": schema,
        "paid_compute_usd": 0.0,
        "steps_per_run": steps,
        "seeds": seeds,
        "tasks": {t: {} for t in tasks},
    }

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=5, mp_context=ctx) as pool:
        futs = [pool.submit(_run_u02a_job, job) for job in jobs]
        for fut in as_completed(futs):
            task, mech, _seed, job_s, r = fut.result()
            pending[(task, mech)].append((job_s, r))
            if len(pending[(task, mech)]) == len(seeds):
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
                    f"  [{tag} {task[:10]:10s}] {mech:34s} ID={agg['id_acc']['mean']:.3f} "
                    f"OOD={agg['ood_acc']['mean']:.3f}±{agg['ood_acc']['std']:.3f}{extra} (cpu={wall_s}s)",
                    flush=True,
                )

    for t in tasks:
        results["tasks"][t] = {m: results["tasks"][t][m] for m in mechs}
    results["elapsed_s"] = round(time.perf_counter() - t0, 2)
    _write_strict_json(out_path, results)
    print(f"[{tag}] Complete in {results['elapsed_s']}s -> {out_path.name}", flush=True)
    return results


def run_part_b() -> dict[str, Any]:
    t0 = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    vocab = int(manifest["vocab"])
    train_arr = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    n_train = len(train_arr)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Strict causality & d=128 non-emb parity check before training
    x1 = torch.randint(0, vocab, (2, 48))
    x2 = x1.clone()
    x2[:, 36:] = (x2[:, 36:] + 11) % vocab
    for mech, ffn in CONTENDERS.items():
        cm = UnifiedBlockLM(vocab, D_MODEL, mech, n_heads=N_HEADS, ffn_hidden_override=ffn).eval()
        with torch.no_grad():
            y1, y2 = cm(x1), cm(x2)
        leak = float((y1[:, :36] - y2[:, :36]).abs().max().item())
        ne = count_non_emb_params(cm)
        if leak > 1e-6 or not bool(torch.isfinite(y1).all()):
            raise RuntimeError(f"d=128 causality failure for {mech}: leak={leak}")
        if not (851_500 <= ne <= 853_500):
            raise RuntimeError(f"d=128 non_emb parity failure for {mech}: {ne}")
    print(f"[U07-B] d=128 causality & parity verified for all 5 contenders on {dev}.", flush=True)

    all_runs: dict[str, dict[int, dict[str, Any]]] = {m: {} for m in CONTENDERS}
    payload: dict[str, Any] = {}
    for mech, ffn in CONTENDERS.items():
        for seed in SEEDS_B:
            torch.manual_seed(1000 + seed * 97 + sum(ord(c) for c in mech))
            rng = np.random.default_rng(70_000 + seed * 131 + sum(ord(c) for c in mech))

            model = UnifiedBlockLM(vocab, D_MODEL, mech, n_heads=N_HEADS, ffn_hidden_override=ffn).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            total_p = count_trainable_params(model)
            non_emb_p = count_non_emb_params(model)

            loss_curve: dict[int, float] = {}
            train_nonfinite = False
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
                if not bool(torch.isfinite(loss).item()):
                    train_nonfinite = True
                opt.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not bool(torch.isfinite(grad_norm).item()):
                    train_nonfinite = True
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

            seed_rec = {
                "device": str(dev),
                "ffn_hidden": ffn,
                "total_params": total_p,
                "non_emb_params": non_emb_p,
                "train_s": round(train_s, 2),
                "tok_per_s": tok_s,
                "train_nonfinite": train_nonfinite,
                "train_loss_curve": loss_curve,
                "val_bpb": bpb,
                "copy_probes": cp,
                "real_haystack_mqar": mq,
            }
            _, eval_bad = _sanitize_json(seed_rec)
            seed_rec["has_nonfinite"] = bool(train_nonfinite or eval_bad)
            all_runs[mech][seed] = seed_rec
            print(
                f"  [U07-B] {mech:34s} s={seed} web={bpb['L_128']['web']:.4f}/{bpb['L_256']['web']:.4f}/{bpb['L_512']['web']:.4f} "
                f"code512={bpb['L_512']['code']:.4f} copy256={cp['copy_256']['acc']:.3f} tok/s={tok_s:.0f} ({train_s:.1f}s) nf={seed_rec['has_nonfinite']}",
                flush=True,
            )
            del model, opt
            if dev.type == "mps":
                torch.mps.empty_cache()

        summary: dict[str, Any] = {}
        for m_done in CONTENDERS:
            if len(all_runs[m_done]) < len(SEEDS_B):
                continue
            runs = [all_runs[m_done][s] for s in SEEDS_B]
            summary[m_done] = {
                "device": runs[0]["device"],
                "total_params": runs[0]["total_params"],
                "non_emb_params": runs[0]["non_emb_params"],
                "mean_tok_per_s": round(sum(r["tok_per_s"] for r in runs) / len(runs), 1),
                "any_nonfinite": any(r["has_nonfinite"] for r in runs),
                "val_bpb": {
                    L: {dom: _agg_nested_safe(runs, ("val_bpb", L, dom)) for dom in ("web", "textbook", "math", "code")}
                    for L in ("L_128", "L_256", "L_512")
                },
                "copy_probes": {ck: _agg_nested_safe(runs, ("copy_probes", ck, "acc")) for ck in ("copy_32", "copy_64", "copy_128", "copy_256")},
                "real_haystack_mqar": {
                    band: {tc: _agg_nested_safe(runs, ("real_haystack_mqar", band, tc)) for tc in ("T128_N4", "T256_N4", "T512_N4")}
                    for band in ("common_ids", "rare_tail_ids")
                },
                "seeds": {str(s): all_runs[m_done][s] for s in SEEDS_B},
            }
        payload = {
            "schema": "nanolm.unified_loop.u07b.v1",
            "paid_compute_usd": 0.0,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "config": {"d_model": D_MODEL, "n_heads": N_HEADS, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "steps": STEPS, "seeds": SEEDS_B},
            "models": summary,
        }
        _write_strict_json(OUT_DIR / "results_part_b.json", payload)

    print(f"[U07-B] Finished all 5 contenders x 2 seeds in {payload['elapsed_s']}s", flush=True)
    return payload


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _run_cpu_task_suite(
        TASKS_PART_A,
        list(CONTENDERS.keys()),
        SEEDS_A,
        200,
        OUT_DIR / "results_part_a.json",
        "nanolm.unified_loop.u07a.v1",
        "U07-A",
    )
    _run_cpu_task_suite(
        ALGEBRA_TASKS_A2,
        ALGEBRA_MECHS_A2,
        [0, 1],
        600,
        OUT_DIR / "results_part_a2_algebra.json",
        "nanolm.unified_loop.u07a2.v1",
        "U07-A2",
    )
    run_part_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
