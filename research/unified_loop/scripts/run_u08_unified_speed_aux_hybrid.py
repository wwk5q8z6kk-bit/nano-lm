#!/usr/bin/env python3
"""EXP-U08: Speed-Unified HSPA Body Scan, On-Device Sparse Index, AuxBase Loss, MultiScale M17, Hybrid M18, & Rank Audit."""
from __future__ import annotations

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
    GDN1Mixer,
    HSPALieSO3Mixer,
    MECHANISM_SPECS,
    UnifiedBlockLM,
    _center_l2,
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
    TASKS_PART_A,
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

OUT_DIR = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U08_unified_speed_aux_hybrid"

# contender -> (mechanism_name, d128_ffn_hidden, aux_base_weight)
CONTENDERS_U08: dict[str, tuple[str, int, float]] = {
    "M13_Ctrl": ("M13_HSPA_EvidCosSparsePtr", 463, 0.0),
    "M13_AuxBase": ("M13_HSPA_EvidCosSparsePtr", 463, 0.25),
    "M17_AuxBase": ("M17_HSPA_MultiScale_EvidCosSparsePtr", 461, 0.25),
    "M18_AuxBase": ("M18_HSPA_HybridSemLexPtr", 463, 0.25),
    "M16_AuxBase": ("M16_FullSphereTri_EvidCosSparsePtr", 301, 0.25),
}

ALGEBRA_CONTENDERS_A2: dict[str, tuple[str, int, float]] = {
    "M5_Pure": ("M5_HSPA_LieSO3", 496, 0.0),
    "M13_Ctrl": ("M13_HSPA_EvidCosSparsePtr", 463, 0.0),
    "M13_AuxBase": ("M13_HSPA_EvidCosSparsePtr", 463, 0.25),
    "M17_AuxBase": ("M17_HSPA_MultiScale_EvidCosSparsePtr", 461, 0.25),
    "M18_AuxBase": ("M18_HSPA_HybridSemLexPtr", 463, 0.25),
}

TASKS_U08_A = TASKS_PART_A + ["MQAR_Alias_Semantic_4x"]
TASKS_U08_A2 = ["S4_Holonomy_8x", "Hierarchy_Denom_Rollup_4x", "MQAR_Alias_Semantic_4x"]


def make_mqar_alias_semantic_batch(
    bs: int, seq_len: int, n_writes: int, n_queries: int, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-alias associative recall (`[4..11]` <-> `[12..19]`): writes an entity via one alias,
    overwrites >=40%, and ALWAYS queries using the OPPOSITE alias of the latest write."""
    xs = torch.full((bs, seq_len), 0, dtype=torch.long)
    ys = torch.full((bs, seq_len), IGNORE, dtype=torch.long)
    ov_mask = torch.zeros(bs, seq_len, dtype=torch.bool)
    qry_start = seq_len - 2 * n_queries
    vals_pool = list(range(20, 36))
    for b in range(bs):
        latest_val: dict[int, int] = {}
        last_alias_side: dict[int, int] = {}
        write_count: dict[int, int] = {}
        stream: list[int] = [1]  # BOS
        entities = list(range(8))
        for wi in range(n_writes):
            if latest_val and wi >= 4 and rng.random() < 0.45:
                e = rng.choice(list(latest_val.keys()))
                v = rng.choice([vv for vv in vals_pool if vv != latest_val[e]])
                side = 1 - last_alias_side[e]
            else:
                unseen = [ee for ee in entities if ee not in latest_val]
                e = rng.choice(unseen if unseen else entities)
                v = rng.choice(vals_pool)
                side = rng.randrange(2)
            tok = (4 + e) if side == 0 else (12 + e)
            latest_val[e] = v
            last_alias_side[e] = side
            write_count[e] = write_count.get(e, 0) + 1
            if len(stream) + 2 <= qry_start:
                stream.extend([tok, v])
                if len(stream) < qry_start and wi % 2 == 0:
                    stream.append(rng.randrange(36, 44))
        while len(stream) < qry_start:
            stream.append(rng.randrange(36, 44))
        xs[b, :qry_start] = torch.tensor(stream[:qry_start], dtype=torch.long)

        written_e = list(latest_val.keys())
        ov_e = [ee for ee, c in write_count.items() if c >= 2]
        for qi in range(n_queries):
            pos = qry_start + 2 * qi
            e = rng.choice(ov_e) if (ov_e and qi % 2 == 0) else rng.choice(written_e)
            opp_side = 1 - last_alias_side[e]
            q_tok = (4 + e) if opp_side == 0 else (12 + e)
            xs[b, pos] = 3  # QRY
            xs[b, pos + 1] = q_tok
            ys[b, pos + 1] = latest_val[e]
            if write_count[e] >= 2:
                ov_mask[b, pos + 1] = True
    return xs, ys, ov_mask


def _compute_u08_loss(
    model: UnifiedBlockLM,
    x: torch.Tensor,
    y: torch.Tensor,
    aux_base_weight: float,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    V = model.vocab_size
    if aux_base_weight > 0.0 and model.spec.has_pointer:
        mix_lp, base_lg = model(x, return_base_logits=True)
        loss_mix = F.nll_loss(mix_lp.reshape(-1, V), y.reshape(-1), ignore_index=ignore_index)
        loss_base = F.cross_entropy(base_lg.reshape(-1, V), y.reshape(-1), ignore_index=ignore_index)
        return loss_mix + aux_base_weight * loss_base
    out = model(x)
    if model.spec.has_pointer:
        return F.nll_loss(out.reshape(-1, V), y.reshape(-1), ignore_index=ignore_index)
    return F.cross_entropy(out.reshape(-1, V), y.reshape(-1), ignore_index=ignore_index)


def _train_and_eval_micro_u08(args: tuple[str, str, int, int, str, float]) -> tuple[str, str, int, float, dict[str, Any]]:
    contender, task, seed, steps, mech, aux_w = args
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
    for _st in range(steps):
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

        loss = _compute_u08_loss(model, bx, by, aux_w, ignore_index=IGNORE)
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
    bad = [label for label, (mech, _, _) in contenders.items() if causality[mech]["causal_leak_max"] > 1e-6 or not causality[mech]["finite"]]
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
        "tasks": {t: {} for t in tasks},
    }

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=5, mp_context=ctx) as pool:
        futs = [pool.submit(_train_and_eval_micro_u08, job) for job in jobs]
        for fut in as_completed(futs):
            task, label, _seed, job_s, r = fut.result()
            pending[(task, label)].append((job_s, r))
            if len(pending[(task, label)]) == len(seeds):
                items = sorted(pending[(task, label)], key=lambda x: x[1]["seed"])
                runs = [x[1] for x in items]
                wall_s = round(sum(x[0] for x in items), 2)
                agg: dict[str, Any] = {
                    "params": runs[0]["params"],
                    "aux_base_weight": contenders[label][2],
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
                    f"  [{tag} {task[:12]:12s}] {label:13s} ID={agg['id_acc']['mean']:.3f} "
                    f"OOD={agg['ood_acc']['mean']:.3f}±{agg['ood_acc']['std']:.3f}{extra} (cpu={wall_s}s)",
                    flush=True,
                )

    for t in tasks:
        results["tasks"][t] = {label: results["tasks"][t][label] for label in contenders}
    results["elapsed_s"] = round(time.perf_counter() - t0, 2)
    _write_strict_json(out_path, results)
    print(f"[{tag}] Complete in {results['elapsed_s']}s -> {out_path.name}", flush=True)
    return results


def _effective_rank_cpu(mat: torch.Tensor) -> float:
    """Effective rank exp(H(s^2 / sum(s^2))) for [T, D] matrix."""
    s = torch.linalg.svdvals(mat.detach().cpu().float())
    p = s.pow(2)
    p = p / p.sum().clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum().item()
    return float(math.exp(ent))


def _matrix_health(normed_h: torch.Tensor) -> tuple[float, float, int]:
    """Return (mean_head_eff_rank, mean_offdiag_cos, hd) for normalized [B, T, H, hd]."""
    B, T, H, hd = normed_h.shape
    eye = torch.eye(T, dtype=torch.bool)
    ranks: list[float] = []
    cos_vals: list[float] = []
    for b in range(B):
        xh = normed_h[b].transpose(0, 1).cpu().float()  # [H, T, hd]
        gram = torch.matmul(xh, xh.transpose(-1, -2))
        offdiag = gram.masked_select(~eye.unsqueeze(0).expand_as(gram)).view(H, -1)
        cos_vals.append(float(offdiag.mean().item()))
        for hi in range(H):
            ranks.append(_effective_rank_cpu(xh[hi]))
    return round(sum(ranks) / len(ranks), 2), round(sum(cos_vals) / len(cos_vals), 5), int(hd)


@torch.no_grad()
def audit_layer_ranks(model: UnifiedBlockLM, vocab: int, dev: torch.device) -> dict[str, list[dict[str, Any]]]:
    """Measure L0->L3 mixer output/state & key/input effective ranks on val_code@256 and random copy_128."""
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
        h = F.normalize(model.emb(tokens).float(), dim=-1).to(model.emb.weight.dtype)
        layers: list[dict[str, Any]] = []
        for li, blk in enumerate(model.blocks):
            xn = blk.n1(h)
            mix = blk.mix
            mix_y = mix(xn)
            # Normalized per-head mixer output representation (always [B, T, H=4, hd=32])
            state_norm = F.normalize(mix_y.float().view(B, T, mix.h, D // mix.h), dim=-1)
            st_rank, st_cos, st_max = _matrix_health(state_norm)
            entry: dict[str, Any] = {
                "layer": li,
                "state_eff_rank": st_rank,
                "state_max_rank": st_max,
                "state_offdiag_cos": st_cos,
            }
            if isinstance(mix, GDN1Mixer):
                _, k_c, _ = mix.conv(mix.qkv(xn)).split(D, dim=-1)
                kf = _center_l2(k_c.view(B, T, mix.h, mix.hd), center=mix.full_sphere)
                kr, kc, km = _matrix_health(kf)
                entry.update({"key_kind": "gdn1_k", "key_eff_rank": kr, "key_max_rank": km, "key_offdiag_cos": kc})
            elif isinstance(mix, HSPALieSO3Mixer):
                xc = (xn + mix.conv(xn)) if mix.conv is not None else xn
                uf = F.normalize(mix.w_in(xc).float().view(B, T, mix.h, mix.r * 3), dim=-1)
                ur, uc, um = _matrix_health(uf)
                entry.update({"key_kind": "hspa_u_in", "key_eff_rank": ur, "key_max_rank": um, "key_offdiag_cos": uc})
            layers.append(entry)
            h = blk(h)
        out[pname] = layers
    return out


def run_part_b() -> dict[str, Any]:
    t0 = time.perf_counter()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    vocab = int(manifest["vocab"])
    train_arr = np.memmap(DATA_DIR / "train.bin", dtype=np.uint16, mode="r")
    n_train = len(train_arr)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    x1 = torch.randint(0, vocab, (2, 48))
    x2 = x1.clone()
    x2[:, 36:] = (x2[:, 36:] + 11) % vocab
    for label, (mech, ffn, _) in CONTENDERS_U08.items():
        cm = UnifiedBlockLM(vocab, D_MODEL, mech, n_heads=N_HEADS, ffn_hidden_override=ffn).eval()
        with torch.no_grad():
            y1, y2 = cm(x1), cm(x2)
        leak = float((y1[:, :36] - y2[:, :36]).abs().max().item())
        ne = count_non_emb_params(cm)
        if leak > 1e-6 or not bool(torch.isfinite(y1).all()):
            raise RuntimeError(f"d=128 causality failure for {label}: leak={leak}")
        if not (851_500 <= ne <= 853_500):
            raise RuntimeError(f"d=128 non_emb parity failure for {label}: {ne}")
    print(f"[U08-B] d=128 causality & parity verified for all 5 contenders on {dev}.", flush=True)

    all_runs: dict[str, dict[int, dict[str, Any]]] = {label: {} for label in CONTENDERS_U08}
    payload: dict[str, Any] = {}
    for label, (mech, ffn, aux_w) in CONTENDERS_U08.items():
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
                starts = rng.integers(0, n_train - SEQ_LEN - 1, size=BATCH_SIZE)
                xb = np.stack([train_arr[s : s + SEQ_LEN + 1] for s in starts]).astype(np.int64)
                tx = torch.from_numpy(xb).to(dev)
                loss = _compute_u08_loss(model, tx[:, :-1], tx[:, 1:], aux_w)
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
            ranks = audit_layer_ranks(model, vocab, dev)

            seed_rec = {
                "device": str(dev),
                "mechanism": mech,
                "aux_base_weight": aux_w,
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
                "rank_audit": ranks,
            }
            _, eval_bad = _sanitize_json(seed_rec)
            seed_rec["has_nonfinite"] = bool(train_nonfinite or eval_bad)
            all_runs[label][seed] = seed_rec
            r0 = ranks["val_code_256"][0]["state_eff_rank"]
            r3 = ranks["val_code_256"][-1]["state_eff_rank"]
            print(
                f"  [U08-B] {label:12s} s={seed} web={bpb['L_128']['web']:.4f}/{bpb['L_256']['web']:.4f}/{bpb['L_512']['web']:.4f} "
                f"code512={bpb['L_512']['code']:.4f} copy256={cp['copy_256']['acc']:.3f} tok/s={tok_s:.0f} "
                f"L0->L3_rank={r0}->{r3}/32 nf={seed_rec['has_nonfinite']}",
                flush=True,
            )
            del model, opt
            if dev.type == "mps":
                torch.mps.empty_cache()

        summary: dict[str, Any] = {}
        for l_done in CONTENDERS_U08:
            if len(all_runs[l_done]) < len(SEEDS_B):
                continue
            runs = [all_runs[l_done][s] for s in SEEDS_B]
            summary[l_done] = {
                "mechanism": runs[0]["mechanism"],
                "aux_base_weight": runs[0]["aux_base_weight"],
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
                "seeds": {str(s): all_runs[l_done][s] for s in SEEDS_B},
            }
        payload = {
            "schema": "nanolm.unified_loop.u08b.v1",
            "paid_compute_usd": 0.0,
            "elapsed_s": round(time.perf_counter() - t0, 2),
            "config": {"d_model": D_MODEL, "n_heads": N_HEADS, "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN, "steps": STEPS, "seeds": SEEDS_B},
            "models": summary,
        }
        _write_strict_json(OUT_DIR / "results_part_b.json", payload)

    print(f"[U08-B] Finished all 5 contenders x 2 seeds in {payload['elapsed_s']}s", flush=True)
    return payload


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _run_cpu_suite(
        TASKS_U08_A,
        CONTENDERS_U08,
        SEEDS_A,
        200,
        OUT_DIR / "results_part_a.json",
        "nanolm.unified_loop.u08a.v1",
        "U08-A",
        save_per_seed=False,
    )
    _run_cpu_suite(
        TASKS_U08_A2,
        ALGEBRA_CONTENDERS_A2,
        [0, 1],
        600,
        OUT_DIR / "results_part_a2_algebra.json",
        "nanolm.unified_loop.u08a2.v1",
        "U08-A2",
        save_per_seed=True,
    )
    run_part_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
