#!/usr/bin/env python3
"""EXP-U09: Pre-RunPod Hardening Gate — CenteredRMSNorm, Residual Variance Scaling, Block-AttnRes, Cosine-Annealed AuxBase, & ConceptSlotMemory."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
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
    HSPALieSO3Mixer,
    UnifiedBlockLM,
    count_trainable_params,
    verify_strict_causality,
)
from research.unified_loop.scripts.run_u02_cross_paradigm_showdown import (  # noqa: E402
    D_MODEL as MICRO_D_MODEL,
    IGNORE,
    VOCAB as MICRO_VOCAB,
    eval_accuracy,
    make_hierarchy_denom_rollup_batch,
    make_mqar_overwrite_batch,
    make_open_vocab_copy_batch,
    make_parity_dyck_batch,
    make_s4_batch,
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
    WARMUP,
    WEIGHT_DECAY,
    _cosine_lr,
    count_non_emb_params,
    eval_bpb_multi_horizon,
    eval_copy_probes,
    eval_real_haystack_mqar,
)
from research.unified_loop.scripts.run_u07_first_principles_enhancements import (  # noqa: E402
    _agg_nested_safe,
    _sanitize_json,
    _write_strict_json,
)
from research.unified_loop.scripts.run_u08_unified_speed_aux_hybrid import (  # noqa: E402
    TASKS_U08_A,
    TASKS_U08_A2,
    _compute_u08_loss,
    _matrix_health,
    make_mqar_alias_semantic_batch,
)

OUT_DIR = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U09_rank_and_concept_slots"

# contender -> (mechanism_name, d128_ffn_hidden, aux_base_start_weight)
CONTENDERS_U09: dict[str, tuple[str, int, float]] = {
    "M13_AnnealAux": ("M13_HSPA_EvidCosSparsePtr", 463, 0.25),
    "M19_ScaledNorm": ("M19_HSPA_ScaledNorm_EvidPtr", 463, 0.25),
    "M20_ScaledAttnRes": ("M20_HSPA_ScaledAttnRes_EvidPtr", 463, 0.25),
    "M21_ConceptSlot": ("M21_HSPA_ConceptSlot_EvidPtr", 460, 0.25),
    "M22_FullCore": ("M22_HSPA_FullCore_U09", 460, 0.25),
}

TASKS_U09_A = TASKS_U08_A
TASKS_U09_A2 = TASKS_U08_A2


def _annealed_aux_weight(step: int, total_steps: int, start_w: float = 0.25, end_w: float = 0.0) -> float:
    frac = min(1.0, max(0.0, float(step) / float(max(1, total_steps - 1))))
    return end_w + 0.5 * (start_w - end_w) * (1.0 + math.cos(math.pi * frac))


def verify_preflight(vocab_large: int = 32768) -> dict[str, Any]:
    """Verify strict causality and iso-parameter parity at both d=32 (Part A) and d=128 (Part B)."""
    rep_micro = verify_strict_causality(vocab_size=MICRO_VOCAB, d_model=MICRO_D_MODEL)
    out: dict[str, Any] = {"d32": {}, "d128": {}}
    x1 = torch.randint(0, vocab_large, (2, 48))
    x2 = x1.clone()
    x2[:, 36:] = (x2[:, 36:] + 11) % vocab_large
    for label, (mech, ffn, aux_start) in CONTENDERS_U09.items():
        r32 = rep_micro[mech]
        if r32["causal_leak_max"] > 1e-6 or not r32["finite"] or not (54_500 <= r32["params"] <= 56_500):
            raise RuntimeError(f"d=32 preflight failure for {label}: {r32}")
        out["d32"][label] = r32

        cm = UnifiedBlockLM(vocab_large, D_MODEL, mech, n_heads=N_HEADS, ffn_hidden_override=ffn).eval()
        with torch.no_grad():
            y1, y2 = cm(x1), cm(x2)
        leak = float((y1[:, :36] - y2[:, :36]).abs().max().item())
        ne = count_non_emb_params(cm)
        fin = bool(torch.isfinite(y1).all().item())
        if leak > 1e-6 or not fin or not (851_500 <= ne <= 853_500):
            raise RuntimeError(f"d=128 preflight failure for {label}: ne={ne}, leak={leak}, finite={fin}")
        out["d128"][label] = {
            "ffn_hidden": ffn,
            "non_emb_params": ne,
            "causal_leak_max": leak,
            "finite": fin,
            "aux_start": aux_start,
        }
    return out


def _train_and_eval_micro_u09(args: tuple[str, str, int, int, str, float]) -> tuple[str, str, int, float, dict[str, Any]]:
    contender, task, seed, steps, mech, aux_start = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    jt0 = time.perf_counter()

    torch.manual_seed(seed * 97 + 7)
    rng = random.Random(50_000 + seed * 313 + sum(ord(c) for c in task))
    model = UnifiedBlockLM(MICRO_VOCAB, MICRO_D_MODEL, mech)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)

    bs = 32
    model.train()
    for st in range(steps):
        if task == "S4_Holonomy_8x":
            bx, by = make_s4_batch(bs, 24, rng)
        elif task == "MQAR_Selective_Overwrite_4x":
            bx, by, _ = make_mqar_overwrite_batch(bs, 48, n_writes=10, n_queries=6, rng=rng)
        elif task == "MQAR_Alias_Semantic_4x":
            bx, by, _ = make_mqar_alias_semantic_batch(bs, 48, n_writes=10, n_queries=6, rng=rng)
        elif task == "Unseen_OpenVocab_Copy":
            bx, by = make_open_vocab_copy_batch(bs, copy_len=8, prefix_len=8, rng=rng, unseen_vocab=False)
        elif task == "Parity_Dyck_4x":
            bx, by = make_parity_dyck_batch(bs, 40, rng)
        elif task == "Hierarchy_Denom_Rollup_4x":
            bx, by, _ = make_hierarchy_denom_rollup_batch(bs, 32, max_depth=2, rng=rng, min_peak_depth=1)
        else:
            raise ValueError(task)

        cur_aux = _annealed_aux_weight(st, steps, start_w=aux_start, end_w=0.0)
        loss = _compute_u08_loss(model, bx, by, cur_aux, ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    erng = random.Random(90_000 + seed * 101 + sum(ord(c) for c in task))
    res: dict[str, Any] = {"seed": seed, "params": count_trainable_params(model)}
    with torch.no_grad():
        if task == "S4_Holonomy_8x":
            ix, iy = make_s4_batch(64, 24, erng)
            ox, oy = make_s4_batch(64, 192, erng)
            res["id_acc"] = eval_accuracy(model(ix), iy, col_slice=slice(16, 24))
            res["ood_acc"] = eval_accuracy(model(ox), oy, col_slice=slice(96, 192))
        elif task in ("MQAR_Selective_Overwrite_4x", "MQAR_Alias_Semantic_4x"):
            gen = make_mqar_overwrite_batch if task == "MQAR_Selective_Overwrite_4x" else make_mqar_alias_semantic_batch
            ix, iy, _ = gen(64, 48, n_writes=10, n_queries=6, rng=erng)
            nw_o, nq_o = (40, 24) if task == "MQAR_Selective_Overwrite_4x" else (36, 20)
            ox, oy, oov = gen(64, 192, n_writes=nw_o, n_queries=nq_o, rng=erng)
            lg_id, lg_od = model(ix), model(ox)
            res["id_acc"] = eval_accuracy(lg_id, iy)
            res["ood_acc"] = eval_accuracy(lg_od, oy)
            res["ood_overwritten_acc"] = eval_accuracy(lg_od, oy, mask=oov)
        elif task == "Unseen_OpenVocab_Copy":
            ix, iy = make_open_vocab_copy_batch(64, copy_len=8, prefix_len=8, rng=erng, unseen_vocab=False)
            ox, oy = make_open_vocab_copy_batch(64, copy_len=16, prefix_len=16, rng=erng, unseen_vocab=True)
            res["id_acc"] = eval_accuracy(model(ix), iy)
            res["ood_acc"] = eval_accuracy(model(ox), oy)
        elif task == "Parity_Dyck_4x":
            ix, iy = make_parity_dyck_batch(64, 40, erng)
            ox, oy = make_parity_dyck_batch(64, 160, erng)
            res["id_acc"] = eval_accuracy(model(ix), iy, col_slice=slice(20, 40))
            res["ood_acc"] = eval_accuracy(model(ox), oy, col_slice=slice(80, 160))
        elif task == "Hierarchy_Denom_Rollup_4x":
            ix, iy, _ = make_hierarchy_denom_rollup_batch(64, 32, max_depth=2, rng=erng, min_peak_depth=1)
            ox, oy, ocm = make_hierarchy_denom_rollup_batch(64, 128, max_depth=8, rng=erng, min_peak_depth=4)
            lg_id, lg_od = model(ix), model(ox)
            res["id_acc"] = eval_accuracy(lg_id, iy)
            res["ood_acc"] = eval_accuracy(lg_od, oy)
            res["ood_close_rollup_acc"] = eval_accuracy(lg_od, oy, mask=ocm)

    return task, contender, seed, time.perf_counter() - jt0, res


def _run_cpu_suite(
    tasks: list[str],
    contenders: dict[str, tuple[str, int, float]],
    seeds: list[int],
    steps: int,
    out_path: Path,
    schema: str,
    tag: str,
    *,
    save_per_seed: bool = False,
) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    t0 = time.perf_counter()

    causality = verify_strict_causality(vocab_size=MICRO_VOCAB, d_model=MICRO_D_MODEL)
    bad = [
        label
        for label, (mech, _, _) in contenders.items()
        if causality[mech]["causal_leak_max"] > 1e-6 or not causality[mech]["finite"]
    ]
    if bad:
        raise RuntimeError(f"{tag} causality failure: {bad}")
    print(f"[{tag}] d=32 causality verified for {len(contenders)} contenders.", flush=True)

    jobs = [(label, t, s, steps, mech, aux_w) for t in tasks for label, (mech, _, aux_w) in contenders.items() for s in seeds]
    pending: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = {(t, label): [] for t in tasks for label in contenders}
    results: dict[str, Any] = {
        "schema": schema,
        "paid_compute_usd": 0.0,
        "steps_per_run": steps,
        "seeds": seeds,
        "aux_schedule": "cosine_0.25_to_0.0",
        "tasks": {t: {} for t in tasks},
    }

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=5, mp_context=ctx) as pool:
        futs = [pool.submit(_train_and_eval_micro_u09, job) for job in jobs]
        for fut in as_completed(futs):
            task, label, _seed, job_s, r = fut.result()
            pending[(task, label)].append((job_s, r))
            if len(pending[(task, label)]) == len(seeds):
                items = sorted(pending[(task, label)], key=lambda x: x[1]["seed"])
                runs = [x[1] for x in items]
                wall_s = round(sum(x[0] for x in items), 2)
                agg: dict[str, Any] = {
                    "params": runs[0]["params"],
                    "aux_base_start": contenders[label][2],
                    "aux_base_end": 0.0,
                    "wall_s": wall_s,
                    "id_acc": mean_std([x["id_acc"] for x in runs]),
                    "ood_acc": mean_std([x["ood_acc"] for x in runs]),
                }
                for opt_k in ("ood_overwritten_acc", "ood_close_rollup_acc"):
                    if opt_k in runs[0]:
                        agg[opt_k] = mean_std([x[opt_k] for x in runs])
                if save_per_seed:
                    agg["per_seed"] = [
                        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in xr.items()}
                        for xr in runs
                    ]
                results["tasks"][task][label] = agg
                extra = f" ov={agg['ood_overwritten_acc']['mean']:.3f}" if "ood_overwritten_acc" in agg else ""
                if "ood_close_rollup_acc" in agg:
                    extra += f" close={agg['ood_close_rollup_acc']['mean']:.3f}"
                print(
                    f"  [{tag} {task[:12]:12s}] {label:16s} ID={agg['id_acc']['mean']:.3f} "
                    f"OOD={agg['ood_acc']['mean']:.3f}±{agg['ood_acc']['std']:.3f}{extra} (cpu={wall_s}s)",
                    flush=True,
                )

    for t in tasks:
        results["tasks"][t] = {label: results["tasks"][t][label] for label in contenders}
    results["elapsed_s"] = round(time.perf_counter() - t0, 2)
    _write_strict_json(out_path, results)
    print(f"[{tag}] Complete in {results['elapsed_s']}s -> {out_path.name}", flush=True)
    return results


@torch.no_grad()
def audit_layer_ranks_v2(model: UnifiedBlockLM, vocab: int, dev: torch.device) -> dict[str, list[dict[str, Any]]]:
    """Measure uncentered AND time-centered L0->L3 rank, cosine isotropy, h_in RMS, and ||mix||/||h|| on val_code@256 and copy_128@256."""
    code_arr = np.memmap(DATA_DIR / "val_code.bin", dtype=np.uint16, mode="r")
    stride = (len(code_arr) - 257) // 4
    code_xb = torch.tensor(
        np.stack([code_arr[i * stride : i * stride + 256] for i in range(4)]).astype(np.int64),
        dtype=torch.long,
        device=dev,
    )
    g = torch.Generator().manual_seed(4321)
    xs = torch.randint(64, vocab - 64, (4, 128), generator=g)
    copy_xb = torch.cat([xs, xs], dim=1).to(dev)

    out: dict[str, list[dict[str, Any]]] = {}
    for pname, tokens in (("val_code_256", code_xb), ("copy_128_256", copy_xb)):
        B, T = tokens.shape
        D = model.d_model
        anchor = F.normalize(model.emb(tokens).float(), dim=-1).to(model.emb.weight.dtype)
        h = (math.sqrt(float(D)) * anchor) if model.spec.scaled_residual else anchor
        hist = [h]
        layers: list[dict[str, Any]] = []
        for li, blk in enumerate(model.blocks):
            if model.spec.attn_res:
                stacked = torch.stack(hist, dim=0)
                k_norm = model.depth_norm(stacked).float()
                w = torch.softmax((k_norm * model.depth_q[li].float()).sum(dim=-1), dim=0)
                h_in = (w.unsqueeze(-1) * stacked.float()).sum(dim=0).to(h.dtype)
            else:
                h_in = h

            xn = blk.n1(h_in)
            mix = blk.mix
            mix_y = mix(xn)

            hd = D // mix.h
            mix_4d = mix_y.float().view(B, T, mix.h, hd)
            mix_uncent = F.normalize(mix_4d, dim=-1)
            mix_cent = F.normalize(mix_4d - mix_4d.mean(dim=1, keepdim=True), dim=-1)
            mu_r, mu_c, max_r = _matrix_health(mix_uncent)
            mc_r, mc_c, _ = _matrix_health(mix_cent)

            hin_4d = h_in.float().view(B, T, mix.h, hd)
            hin_uncent = F.normalize(hin_4d, dim=-1)
            hin_cent = F.normalize(hin_4d - hin_4d.mean(dim=1, keepdim=True), dim=-1)
            hu_r, hu_c, _ = _matrix_health(hin_uncent)
            hc_r, hc_c, _ = _matrix_health(hin_cent)

            h_norm_mean = float(h_in.float().norm(dim=-1).mean().item())
            mix_norm_mean = float(mix_y.float().norm(dim=-1).mean().item())
            h_rms = round(float(h_in.float().pow(2).mean().sqrt().item()), 4)
            mix_ratio = round(mix_norm_mean / max(1e-8, h_norm_mean), 4)

            entry: dict[str, Any] = {
                "layer": li,
                "max_rank": max_r,
                "state_eff_rank": mu_r,
                "mix_rank_uncent": mu_r,
                "mix_cos_uncent": mu_c,
                "mix_rank_cent": mc_r,
                "mix_cos_cent": mc_c,
                "h_in_rank_uncent": hu_r,
                "h_in_cos_uncent": hu_c,
                "h_in_rank_cent": hc_r,
                "h_in_cos_cent": hc_c,
                "h_in_rms": h_rms,
                "mix_to_h_ratio": mix_ratio,
            }
            if isinstance(mix, HSPALieSO3Mixer):
                xc = (xn + mix.conv(xn)) if mix.conv is not None else xn
                u_4d = mix.w_in(xc).float().view(B, T, mix.h, mix.r * 3)
                uk_r, uk_c, um = _matrix_health(F.normalize(u_4d, dim=-1))
                ck_r, ck_c, _ = _matrix_health(F.normalize(u_4d - u_4d.mean(dim=1, keepdim=True), dim=-1))
                entry.update({
                    "key_kind": "hspa_u_in",
                    "key_rank_uncent": uk_r,
                    "key_cos_uncent": uk_c,
                    "key_rank_cent": ck_r,
                    "key_cos_cent": ck_c,
                    "key_max_rank": um,
                })
            layers.append(entry)
            h = blk(h_in)
            hist.append(h)
        out[pname] = layers
    return out


def run_part_b() -> dict[str, Any]:
    t0 = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    vocab = int(manifest["vocab"])
    train_arr = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    n_train = len(train_arr)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    verify_preflight(vocab_large=vocab)
    print(f"[U09-B] d=128 causality & parity verified for all {len(CONTENDERS_U09)} contenders on {dev}.", flush=True)

    all_runs: dict[str, dict[int, dict[str, Any]]] = {label: {} for label in CONTENDERS_U09}
    payload: dict[str, Any] = {}
    for label, (mech, ffn, aux_start) in CONTENDERS_U09.items():
        for seed in SEEDS_B:
            torch.manual_seed(1000 + seed * 97 + sum(ord(c) for c in label))
            rng = np.random.default_rng(70_000 + seed * 131 + sum(ord(c) for c in label))

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
                cur_aux = _annealed_aux_weight(step, STEPS, start_w=aux_start, end_w=0.0)
                starts = rng.integers(0, n_train - SEQ_LEN - 1, size=BATCH_SIZE)
                xb = np.stack([train_arr[s : s + SEQ_LEN + 1] for s in starts]).astype(np.int64)
                tx = torch.from_numpy(xb).to(dev)
                loss = _compute_u08_loss(model, tx[:, :-1], tx[:, 1:], cur_aux)
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
            ranks = audit_layer_ranks_v2(model, vocab, dev)
            mean_512 = round(sum(bpb["L_512"][dom] for dom in ("web", "textbook", "math", "code")) / 4.0, 4)

            seed_rec = {
                "device": str(dev),
                "mechanism": mech,
                "aux_schedule": f"cosine_{aux_start}_to_0.0",
                "ffn_hidden": ffn,
                "total_params": total_p,
                "non_emb_params": non_emb_p,
                "train_s": round(train_s, 2),
                "tok_per_s": tok_s,
                "train_nonfinite": train_nonfinite,
                "train_loss_curve": loss_curve,
                "val_bpb": bpb,
                "mean_512_bpb": mean_512,
                "copy_probes": cp,
                "real_haystack_mqar": mq,
                "rank_audit": ranks,
            }
            _, eval_bad = _sanitize_json(seed_rec)
            seed_rec["has_nonfinite"] = bool(train_nonfinite or eval_bad)
            all_runs[label][seed] = seed_rec
            vc = ranks["val_code_256"]
            print(
                f"  [U09-B] {label:16s} s={seed} web={bpb['L_128']['web']:.4f}/{bpb['L_256']['web']:.4f}/{bpb['L_512']['web']:.4f} "
                f"mean512={mean_512:.4f} copy256={cp['copy_256']['acc']:.3f} tok/s={tok_s:.0f} "
                f"L0(||m||/||h||={vc[0]['mix_to_h_ratio']:.2f},cent={vc[0]['mix_rank_cent']:.1f})->"
                f"L3(uncent={vc[-1]['mix_rank_uncent']:.1f},cent={vc[-1]['mix_rank_cent']:.1f}) nf={seed_rec['has_nonfinite']}",
                flush=True,
            )
            del model, opt
            if dev.type == "mps":
                torch.mps.empty_cache()

        summary: dict[str, Any] = {}
        for l_done in CONTENDERS_U09:
            if len(all_runs[l_done]) < len(SEEDS_B):
                continue
            runs = [all_runs[l_done][s] for s in SEEDS_B]
            summary[l_done] = {
                "mechanism": runs[0]["mechanism"],
                "aux_schedule": runs[0]["aux_schedule"],
                "device": runs[0]["device"],
                "total_params": runs[0]["total_params"],
                "non_emb_params": runs[0]["non_emb_params"],
                "mean_tok_per_s": round(sum(r["tok_per_s"] for r in runs) / len(runs), 1),
                "any_nonfinite": any(r["has_nonfinite"] for r in runs),
                "mean_512_bpb": _agg_nested_safe(runs, ("mean_512_bpb",)),
                "val_bpb": {
                    L: {dom: _agg_nested_safe(runs, ("val_bpb", L, dom)) for dom in ("web", "textbook", "math", "code")}
                    for L in ("L_128", "L_256", "L_512")
                },
                "copy_probes": {ck: _agg_nested_safe(runs, ("copy_probes", ck, "acc")) for ck in ("copy_32", "copy_64", "copy_128", "copy_256")},
                "real_haystack_mqar": {
                    band: {tc: _agg_nested_safe(runs, ("real_haystack_mqar", band, tc)) for tc in ("T128_N4", "T256_N4", "T512_N4")}
                    for band in ("common_ids", "rare_tail_ids")
                },
                "rank_summary_val_code_256": [
                    {
                        "layer": li,
                        "mix_rank_uncent": _agg_nested_safe(runs, ("rank_audit", "val_code_256", li, "mix_rank_uncent")),
                        "mix_rank_cent": _agg_nested_safe(runs, ("rank_audit", "val_code_256", li, "mix_rank_cent")),
                        "h_in_rank_cent": _agg_nested_safe(runs, ("rank_audit", "val_code_256", li, "h_in_rank_cent")),
                        "h_in_rms": _agg_nested_safe(runs, ("rank_audit", "val_code_256", li, "h_in_rms")),
                        "mix_to_h_ratio": _agg_nested_safe(runs, ("rank_audit", "val_code_256", li, "mix_to_h_ratio")),
                    }
                    for li in range(4)
                ],
                "seeds": {str(s): all_runs[l_done][s] for s in SEEDS_B},
            }
        payload = {
            "schema": "nanolm.unified_loop.u09b.v1",
            "paid_compute_usd": 0.0,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "config": {"d_model": D_MODEL, "n_heads": N_HEADS, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "steps": STEPS, "seeds": SEEDS_B},
            "models": summary,
        }
        _write_strict_json(OUT_DIR / "results_part_b.json", payload)

    print(f"[U09-B] Finished all {len(CONTENDERS_U09)} contenders x {len(SEEDS_B)} seeds in {payload['elapsed_s']}s", flush=True)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EXP-U09 Rank/Norm Repair, Block-AttnRes, Annealed AuxBase, & ConceptSlotMemory.")
    parser.add_argument("--preflight", action="store_true", help="Only verify causality and iso-param parity at d=32 and d=128.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pref = verify_preflight()
    print("[U09] Preflight parity & causality PASS:", json.dumps(pref["d128"]), flush=True)
    if args.preflight:
        return 0

    _run_cpu_suite(
        TASKS_U09_A,
        CONTENDERS_U09,
        SEEDS_A,
        200,
        OUT_DIR / "results_part_a.json",
        "nanolm.unified_loop.u09a.v1",
        "U09-A",
        save_per_seed=False,
    )
    _run_cpu_suite(
        TASKS_U09_A2,
        CONTENDERS_U09,
        [0, 1],
        600,
        OUT_DIR / "results_part_a2_algebra.json",
        "nanolm.unified_loop.u09a2.v1",
        "U09-A2",
        save_per_seed=True,
    )
    run_part_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
