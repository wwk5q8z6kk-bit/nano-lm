#!/usr/bin/env python3
"""EXP-U02 Part A: Iso-Parameter Cross-Paradigm Micro-Lab Showdown (n=3 seeds x 10 mechanisms x 5 tasks)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKTREE_ROOT))

from research.unified_loop.mechanisms.zoo import (  # noqa: E402
    MECHANISM_SPECS,
    UnifiedBlockLM,
    count_trainable_params,
    verify_strict_causality,
)

OUT_DIR = WORKTREE_ROOT / "research" / "unified_loop" / "experiments" / "U02_cross_paradigm_showdown"
VOCAB = 72
D_MODEL = 32
IGNORE = -100


def _build_s4_table() -> tuple[list[list[int]], list[int]]:
    perms = list(itertools.permutations(range(4)))
    p2id = {p: i for i, p in enumerate(perms)}
    mul = [[0] * 24 for _ in range(24)]
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            # Compose a then b: res[k] = b[a[k]]
            c = tuple(b[a[k]] for k in range(4))
            mul[i][j] = p2id[c]
    # 4 generators: adjacent swaps (0,1), (1,2), (2,3) and 4-cycle (1,2,3,0)
    gens = [p2id[(1, 0, 2, 3)], p2id[(0, 2, 1, 3)], p2id[(0, 1, 3, 2)], p2id[(1, 2, 3, 0)]]
    return mul, gens


S4_MUL, S4_GENS = _build_s4_table()


def make_s4_batch(bs: int, seq_len: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.zeros(bs, seq_len, dtype=torch.long)
    ys = torch.full((bs, seq_len), IGNORE, dtype=torch.long)
    for b in range(bs):
        state = 0  # identity
        xs[b, 0] = 1  # BOS
        for t in range(1, seq_len):
            gi = rng.randrange(4)
            xs[b, t] = 4 + gi
            state = S4_MUL[state][S4_GENS[gi]]
            ys[b, t] = 8 + state  # 24 output state tokens: 8..31
    return xs, ys


def make_mqar_overwrite_batch(
    bs: int, seq_len: int, n_writes: int, n_queries: int, rng: random.Random
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(x, y, is_overwritten_mask)` with keys [4..19], vals [20..35], distractors [36..43], QRY=3."""
    xs = torch.full((bs, seq_len), 0, dtype=torch.long)
    ys = torch.full((bs, seq_len), IGNORE, dtype=torch.long)
    ov_mask = torch.zeros(bs, seq_len, dtype=torch.bool)
    qry_start = seq_len - 2 * n_queries
    for b in range(bs):
        latest_val: dict[int, int] = {}
        write_count: dict[int, int] = {}
        stream: list[int] = [1]  # BOS
        keys_pool = list(range(4, 20))
        vals_pool = list(range(20, 36))
        for wi in range(n_writes):
            # Force >= 40% overwrites once some keys exist
            if latest_val and (wi >= 4 and rng.random() < 0.45):
                k = rng.choice(list(latest_val.keys()))
                v_choices = [v for v in vals_pool if v != latest_val[k]]
                v = rng.choice(v_choices)
            else:
                unseen_keys = [k for k in keys_pool if k not in latest_val]
                k = rng.choice(unseen_keys if unseen_keys else keys_pool)
                v = rng.choice(vals_pool)
            latest_val[k] = v
            write_count[k] = write_count.get(k, 0) + 1
            if len(stream) + 2 <= qry_start:
                stream.extend([k, v])
                if len(stream) < qry_start and wi % 2 == 0:
                    stream.append(rng.randrange(36, 44))
        while len(stream) < qry_start:
            stream.append(rng.randrange(36, 44))
        xs[b, :qry_start] = torch.tensor(stream[:qry_start], dtype=torch.long)

        # Sample queries (prioritizing overwritten keys too)
        written_keys = list(latest_val.keys())
        ov_keys = [k for k, c in write_count.items() if c >= 2]
        for qi in range(n_queries):
            pos = qry_start + 2 * qi
            if ov_keys and (qi % 2 == 0):
                qk = rng.choice(ov_keys)
            else:
                qk = rng.choice(written_keys)
            xs[b, pos] = 3  # QRY
            xs[b, pos + 1] = qk
            ys[b, pos + 1] = latest_val[qk]
            if write_count[qk] >= 2:
                ov_mask[b, pos + 1] = True
    return xs, ys, ov_mask


def make_open_vocab_copy_batch(
    bs: int, copy_len: int, prefix_len: int, rng: random.Random, *, unseen_vocab: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Train strictly on [4..43]; OOD unseen eval strictly on [44..67]."""
    tok_pool = list(range(44, 68)) if unseen_vocab else list(range(4, 44))
    bg_pool = list(range(4, 44))  # distractors always from train-range!
    total_len = 1 + prefix_len + 1 + copy_len + 1 + copy_len - 1
    # Sequence layout: [BOS, bg..., SEP, s1..sM, QRY, s1..s_{M-1}] -> targets at QRY..end are s1..sM!
    xs = torch.zeros(bs, total_len, dtype=torch.long)
    ys = torch.full((bs, total_len), IGNORE, dtype=torch.long)
    qry_pos = 1 + prefix_len + 1 + copy_len
    for b in range(bs):
        seq = rng.sample(tok_pool, copy_len)
        bg = [rng.choice(bg_pool) for _ in range(prefix_len)]
        row = [1] + bg + [2] + seq + [3] + seq[:-1]
        xs[b] = torch.tensor(row, dtype=torch.long)
        ys[b, qry_pos:] = torch.tensor(seq, dtype=torch.long)
    return xs, ys


def make_parity_dyck_batch(bs: int, seq_len: int, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    """Input symbols {68:'(', 69:')', 70:'1', 71:'0'}; target encodes (parity, clamp(depth, 0..7)) or invalid=24."""
    xs = torch.zeros(bs, seq_len, dtype=torch.long)
    ys = torch.full((bs, seq_len), IGNORE, dtype=torch.long)
    for b in range(bs):
        parity = 0
        depth = 0
        invalid = False
        for t in range(seq_len):
            r = rng.random()
            if r < 0.30:
                tok = 68  # '('
                depth += 1
            elif r < 0.55 and depth > 0:
                tok = 69  # ')'
                depth -= 1
            elif r < 0.62:
                tok = 69  # occasional invalid ')'
                depth -= 1
                if depth < 0:
                    invalid = True
            elif r < 0.81:
                tok = 70  # '1'
                parity ^= 1
            else:
                tok = 71  # '0'
            xs[b, t] = tok
            ys[b, t] = 24 if invalid else (8 + parity * 8 + min(7, depth))
    return xs, ys


_HIER_SCALES: dict[int, int] = {44: 1, 45: 2, 46: 10}
_HIER_SCALE_TOKS: tuple[int, ...] = (44, 45, 46)


def make_hierarchy_denom_rollup_batch(
    bs: int,
    seq_len: int,
    max_depth: int,
    rng: random.Random,
    *,
    min_peak_depth: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hierarchical F_17 accumulation over nested denomination scales `(scale_tok, val_tok)` & `OPEN(68)`/`CLOSE(69)`.

    Modulus 17 ensures base-10 / binary scaling (`10` is a primitive root mod `17`) stays non-degenerate at arbitrary depth.
    """
    xs = torch.zeros(bs, seq_len, dtype=torch.long)
    ys = torch.full((bs, seq_len), IGNORE, dtype=torch.long)
    close_mask = torch.zeros(bs, seq_len, dtype=torch.bool)
    for b in range(bs):
        xs[b, 0] = 1  # BOS
        acc = 0
        cur_scale = 1
        stack: list[tuple[int, int]] = []
        peak_depth = 0
        pos = 1
        while pos < seq_len:
            rem = seq_len - pos
            # Reserve at least 1 token per open frame to close properly
            need_deepen = peak_depth < min_peak_depth and len(stack) < max_depth and rem >= (len(stack) + 4)
            can_open = len(stack) < max_depth and rem >= (len(stack) + 4)
            must_close = bool(stack) and rem <= len(stack) + 1
            if must_close or (stack and not need_deepen and rng.random() < 0.28):
                xs[b, pos] = 69  # CLOSE_NODE
                p_acc, p_scale = stack.pop()
                acc = (p_acc + acc * cur_scale) % 17
                cur_scale = p_scale
                ys[b, pos] = 8 + acc
                close_mask[b, pos] = True
                pos += 1
            elif need_deepen or (can_open and rng.random() < (0.45 if len(stack) == 0 else 0.30)):
                xs[b, pos] = 68  # OPEN_NODE
                s_tok = rng.choice(_HIER_SCALE_TOKS)
                if pos + 1 < seq_len:
                    xs[b, pos + 1] = s_tok
                stack.append((acc, cur_scale))
                peak_depth = max(peak_depth, len(stack))
                acc = 0
                cur_scale = _HIER_SCALES[s_tok]
                pos += 2
            elif rem >= 2:
                s_tok = rng.choice(_HIER_SCALE_TOKS)
                val = rng.randrange(16)
                v_tok = 48 + val
                xs[b, pos] = s_tok
                xs[b, pos + 1] = v_tok
                acc = (acc + val * _HIER_SCALES[s_tok]) % 17
                ys[b, pos + 1] = 8 + acc
                pos += 2
            else:
                xs[b, pos] = 0
                pos += 1
    return xs, ys, close_mask


def eval_accuracy(logits: torch.Tensor, y: torch.Tensor, *, col_slice: slice | None = None, mask: torch.Tensor | None = None) -> float:
    if col_slice is not None:
        logits = logits[:, col_slice]
        y = y[:, col_slice]
        if mask is not None:
            mask = mask[:, col_slice]
    valid = y != IGNORE
    if mask is not None:
        valid = valid & mask
    if not bool(valid.any()):
        return 0.0
    preds = logits.argmax(dim=-1)
    return float((preds[valid] == y[valid]).float().mean().item())


def trm_guard_and_loopcd(loop_logits: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Monotonic Margin Guard (H-31) and LoopCD (H-35) across recursive loop logits."""
    z_curr = loop_logits[0]
    for z_next in loop_logits[1:]:
        p1 = F.softmax(z_curr.float(), dim=-1)
        t1 = p1.topk(2, dim=-1).values
        m1 = t1[..., 0] - t1[..., 1]
        p2 = F.softmax(z_next.float(), dim=-1)
        t2 = p2.topk(2, dim=-1).values
        m2 = t2[..., 0] - t2[..., 1]
        z_curr = torch.where((m2 > m1).unsqueeze(-1), z_next, z_curr)
    z_loopcd = 1.5 * loop_logits[-1] - 0.5 * loop_logits[0]
    return z_curr, z_loopcd


def train_and_eval_one(mech: str, task: str, seed: int, steps: int = 240) -> dict[str, Any]:
    torch.manual_seed(seed * 97 + 7)
    rng = random.Random(50_000 + seed * 313 + sum(ord(c) for c in task))
    model = UnifiedBlockLM(VOCAB, D_MODEL, mech)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    use_ptr_nll = MECHANISM_SPECS[mech].use_pointer

    bs = 32
    model.train()
    for _st in range(steps):
        if task == "S4_Holonomy_8x":
            bx, by = make_s4_batch(bs, 24, rng)
        elif task == "MQAR_Selective_Overwrite_4x":
            bx, by, _ = make_mqar_overwrite_batch(bs, 48, n_writes=10, n_queries=6, rng=rng)
        elif task == "Unseen_OpenVocab_Copy":
            bx, by = make_open_vocab_copy_batch(bs, copy_len=8, prefix_len=8, rng=rng, unseen_vocab=False)
        elif task == "Parity_Dyck_4x":
            bx, by = make_parity_dyck_batch(bs, 40, rng)
        elif task == "Hierarchy_Denom_Rollup_4x":
            bx, by, _ = make_hierarchy_denom_rollup_batch(bs, 32, max_depth=2, rng=rng, min_peak_depth=1)
        else:
            raise ValueError(task)

        if MECHANISM_SPECS[mech].loops > 1:
            _, loop_lgs = model(bx, return_loop_logits=True)
            loss = sum(
                F.cross_entropy(lg.reshape(-1, VOCAB), by.reshape(-1), ignore_index=IGNORE)
                for lg in loop_lgs
            ) / float(len(loop_lgs))
        else:
            out = model(bx)
            if use_ptr_nll:
                loss = F.nll_loss(out.reshape(-1, VOCAB), by.reshape(-1), ignore_index=IGNORE)
            else:
                loss = F.cross_entropy(out.reshape(-1, VOCAB), by.reshape(-1), ignore_index=IGNORE)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    erng = random.Random(90_000 + seed * 101 + sum(ord(c) for c in task))
    res: dict[str, Any] = {"params": count_trainable_params(model)}
    with torch.no_grad():
        if task == "S4_Holonomy_8x":
            ix, iy = make_s4_batch(64, 24, erng)
            ox, oy = make_s4_batch(64, 192, erng)
            if MECHANISM_SPECS[mech].loops > 1:
                lg_id, ll_id = model(ix, return_loop_logits=True)
                lg_od, ll_od = model(ox, return_loop_logits=True)
                g_od, lcd_od = trm_guard_and_loopcd(ll_od)
                res["id_acc"] = eval_accuracy(lg_id, iy, col_slice=slice(16, 24))
                res["ood_acc"] = eval_accuracy(lg_od, oy, col_slice=slice(96, 192))
                res["ood_step1_acc"] = eval_accuracy(ll_od[0], oy, col_slice=slice(96, 192))
                res["ood_guard_acc"] = eval_accuracy(g_od, oy, col_slice=slice(96, 192))
                res["ood_loopcd_acc"] = eval_accuracy(lcd_od, oy, col_slice=slice(96, 192))
            else:
                res["id_acc"] = eval_accuracy(model(ix), iy, col_slice=slice(16, 24))
                res["ood_acc"] = eval_accuracy(model(ox), oy, col_slice=slice(96, 192))

        elif task == "MQAR_Selective_Overwrite_4x":
            ix, iy, iov = make_mqar_overwrite_batch(64, 48, n_writes=10, n_queries=6, rng=erng)
            ox, oy, oov = make_mqar_overwrite_batch(64, 192, n_writes=40, n_queries=24, rng=erng)
            if MECHANISM_SPECS[mech].loops > 1:
                lg_id, _ = model(ix, return_loop_logits=True)
                lg_od, ll_od = model(ox, return_loop_logits=True)
                g_od, lcd_od = trm_guard_and_loopcd(ll_od)
                res["id_acc"] = eval_accuracy(lg_id, iy)
                res["ood_acc"] = eval_accuracy(lg_od, oy)
                res["ood_overwritten_acc"] = eval_accuracy(lg_od, oy, mask=oov)
                res["ood_guard_acc"] = eval_accuracy(g_od, oy)
                res["ood_loopcd_acc"] = eval_accuracy(lcd_od, oy)
            else:
                lg_id, lg_od = model(ix), model(ox)
                res["id_acc"] = eval_accuracy(lg_id, iy)
                res["ood_acc"] = eval_accuracy(lg_od, oy)
                res["ood_overwritten_acc"] = eval_accuracy(lg_od, oy, mask=oov)

        elif task == "Unseen_OpenVocab_Copy":
            ix, iy = make_open_vocab_copy_batch(64, copy_len=8, prefix_len=8, rng=erng, unseen_vocab=False)
            ox, oy = make_open_vocab_copy_batch(64, copy_len=16, prefix_len=16, rng=erng, unseen_vocab=True)
            if MECHANISM_SPECS[mech].loops > 1:
                lg_id, _ = model(ix, return_loop_logits=True)
                lg_od, ll_od = model(ox, return_loop_logits=True)
                g_od, lcd_od = trm_guard_and_loopcd(ll_od)
                res["id_acc"] = eval_accuracy(lg_id, iy)
                res["ood_acc"] = eval_accuracy(lg_od, oy)
                res["ood_guard_acc"] = eval_accuracy(g_od, oy)
                res["ood_loopcd_acc"] = eval_accuracy(lcd_od, oy)
            else:
                res["id_acc"] = eval_accuracy(model(ix), iy)
                res["ood_acc"] = eval_accuracy(model(ox), oy)

        elif task == "Parity_Dyck_4x":
            ix, iy = make_parity_dyck_batch(64, 40, erng)
            ox, oy = make_parity_dyck_batch(64, 160, erng)
            if MECHANISM_SPECS[mech].loops > 1:
                lg_id, _ = model(ix, return_loop_logits=True)
                lg_od, ll_od = model(ox, return_loop_logits=True)
                g_od, lcd_od = trm_guard_and_loopcd(ll_od)
                res["id_acc"] = eval_accuracy(lg_id, iy, col_slice=slice(20, 40))
                res["ood_acc"] = eval_accuracy(lg_od, oy, col_slice=slice(80, 160))
                res["ood_guard_acc"] = eval_accuracy(g_od, oy, col_slice=slice(80, 160))
                res["ood_loopcd_acc"] = eval_accuracy(lcd_od, oy, col_slice=slice(80, 160))
            else:
                res["id_acc"] = eval_accuracy(model(ix), iy, col_slice=slice(20, 40))
                res["ood_acc"] = eval_accuracy(model(ox), oy, col_slice=slice(80, 160))

        elif task == "Hierarchy_Denom_Rollup_4x":
            ix, iy, _ = make_hierarchy_denom_rollup_batch(64, 32, max_depth=2, rng=erng, min_peak_depth=1)
            ox, oy, ocm = make_hierarchy_denom_rollup_batch(64, 128, max_depth=8, rng=erng, min_peak_depth=4)
            if MECHANISM_SPECS[mech].loops > 1:
                lg_id, _ = model(ix, return_loop_logits=True)
                lg_od, ll_od = model(ox, return_loop_logits=True)
                g_od, lcd_od = trm_guard_and_loopcd(ll_od)
                res["id_acc"] = eval_accuracy(lg_id, iy)
                res["ood_acc"] = eval_accuracy(lg_od, oy)
                res["ood_close_rollup_acc"] = eval_accuracy(lg_od, oy, mask=ocm)
                res["ood_guard_acc"] = eval_accuracy(g_od, oy)
                res["ood_loopcd_acc"] = eval_accuracy(lcd_od, oy)
            else:
                lg_id, lg_od = model(ix), model(ox)
                res["id_acc"] = eval_accuracy(lg_id, iy)
                res["ood_acc"] = eval_accuracy(lg_od, oy)
                res["ood_close_rollup_acc"] = eval_accuracy(lg_od, oy, mask=ocm)
    return res


def mean_std(vals: list[float]) -> dict[str, float]:
    n = len(vals)
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / max(1, n - 1)
    return {"mean": round(m, 4), "std": round(math.sqrt(var), 4)}


def _run_job(args: tuple[str, str, int, int]) -> tuple[str, str, int, float, dict[str, Any]]:
    mech, task, seed, steps = args
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    jt0 = time.perf_counter()
    res = train_and_eval_one(mech, task, seed, steps=steps)
    return task, mech, seed, time.perf_counter() - jt0, res


def main() -> int:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    causality = verify_strict_causality(VOCAB, D_MODEL)
    bad_causal = [k for k, v in causality.items() if v["causal_leak_max"] > 1e-6 or not v["finite"]]
    if bad_causal:
        raise RuntimeError(f"Causality check failed for: {bad_causal} -> {causality}")
    print("[U02-A] Strict causality check passed for all 10 mechanisms.", flush=True)

    tasks = [
        "S4_Holonomy_8x",
        "MQAR_Selective_Overwrite_4x",
        "Unseen_OpenVocab_Copy",
        "Parity_Dyck_4x",
        "Hierarchy_Denom_Rollup_4x",
    ]
    mechs = list(MECHANISM_SPECS.keys())
    seeds = [0, 1, 2]
    steps = 200

    results: dict[str, Any] = {
        "schema": "nanolm.unified_loop.u02a.v2",
        "paid_compute_usd": 0.0,
        "steps_per_run": steps,
        "seeds": seeds,
        "causality_and_params": causality,
        "tasks": {t: {} for t in tasks},
    }

    jobs = [(mech, task, s, steps) for task in tasks for mech in mechs for s in seeds]
    pending: dict[tuple[str, str], list[tuple[float, dict[str, Any]]]] = {(t, m): [] for t in tasks for m in mechs}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=5, mp_context=ctx) as pool:
        futs = [pool.submit(_run_job, job) for job in jobs]
        for fut in as_completed(futs):
            task, mech, seed, job_s, r = fut.result()
            pending[(task, mech)].append((job_s, r))
            if len(pending[(task, mech)]) == len(seeds):
                items = pending[(task, mech)]
                runs = [x[1] for x in items]
                wall_s = round(sum(x[0] for x in items), 2)
                agg: dict[str, Any] = {
                    "params": runs[0]["params"],
                    "wall_s": wall_s,
                    "id_acc": mean_std([r["id_acc"] for r in runs]),
                    "ood_acc": mean_std([r["ood_acc"] for r in runs]),
                }
                for opt_k in ("ood_overwritten_acc", "ood_close_rollup_acc", "ood_step1_acc", "ood_guard_acc", "ood_loopcd_acc"):
                    if opt_k in runs[0]:
                        agg[opt_k] = mean_std([r[opt_k] for r in runs])
                results["tasks"][task][mech] = agg
                extra = f" ov={agg['ood_overwritten_acc']['mean']:.3f}" if "ood_overwritten_acc" in agg else ""
                if "ood_close_rollup_acc" in agg:
                    extra += f" close={agg['ood_close_rollup_acc']['mean']:.3f}"
                if "ood_guard_acc" in agg:
                    extra += f" guard={agg['ood_guard_acc']['mean']:.3f} loopcd={agg['ood_loopcd_acc']['mean']:.3f}"
                print(
                    f"  [{task[:10]:10s}] {mech:30s} p={agg['params']:5d} ID={agg['id_acc']['mean']:.3f} "
                    f"OOD={agg['ood_acc']['mean']:.3f}±{agg['ood_acc']['std']:.3f}{extra} (cpu={wall_s}s)",
                    flush=True,
                )
                results["elapsed_s"] = round(time.perf_counter() - t0, 2)
                (OUT_DIR / "results_part_a.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Sort each task's entries cleanly into canonical MECHANISM_SPECS order
    for task in tasks:
        results["tasks"][task] = {m: results["tasks"][task][m] for m in mechs}
    results["elapsed_s"] = round(time.perf_counter() - t0, 2)
    (OUT_DIR / "results_part_a.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[U02-A] Finished all {len(tasks)} tasks x {len(mechs)} mechanisms x {len(seeds)} seeds in {results['elapsed_s']}s wall", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
