"""RECURRENT-OPERATOR-01: Iso-State Sequence Operator Benchmark (3 Tasks x 7 Arms x 3 Seeds).

Compares CausalAttention, GatedLinear, Delta-SiLU, Delta-FullSphere-SFOR,
BSVD-RightValue, SkewCayley, and HDCD (Heterogeneous Multi-Timescale Dual-Action
Cayley-Delta) at matched non-embedding parameter counts (+-3%) and matched
recurrent state bytes (2,304 B/layer fp32 at d=48, H=4, dh=12).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from research.cycle1.primitives import (
    Cycle1SeqModel,
    count_trainable_params,
    make_adamw_no_embed_decay,
    match_d_ff_for_target_params,
    seed_everything,
    summarize_seeds,
)

torch.set_num_threads(1)

D_MODEL = 48
N_HEADS = 4
N_LAYERS = 2
REF_D_FF = 96

ARMS = [
    ("A0_CausalAttn", "attn"),
    ("R1_GatedLinear", "linear"),
    ("R2_Delta_SiLU", "delta_silu"),
    ("R3_Delta_FullSphere_SFOR", "delta_sfor"),
    ("R4_BSVD_RightValue", "bsvd"),
    ("R5_SkewCayley", "cayley"),
    ("R6_HDCD_MultiTimescale", "hdcd"),
]


def _git_sha(repo_dir: Path) -> str:
    res = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() or "unknown"


# ---------------------------------------------------------------------------
# Task 1: Selective Overwrite MQAR (12 keys 0..11, 16 vals 12..27, SEP=28, PAD=29)
# ---------------------------------------------------------------------------
def _gen_overwrite_mqar_batch(
    rng: np.random.Generator,
    batch_size: int,
    n_unique: int = 4,
    n_overwrites: int = 2,
    n_queries: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sep, pad = 28, 29
    events_len = 2 * (n_unique + n_overwrites)
    total_len = events_len + 1 + 2 * n_queries
    xs = np.full((batch_size, total_len), pad, dtype=np.int64)
    ys = np.full((batch_size, total_len), -100, dtype=np.int64)
    is_ov = np.zeros((batch_size, total_len), dtype=bool)

    for b in range(batch_size):
        keys = rng.choice(12, size=n_unique, replace=False)
        vals = rng.integers(12, 28, size=n_unique)
        cur_val = {int(k): int(v) for k, v in zip(keys, vals)}
        was_ov = {int(k): False for k in keys}

        stream: list[tuple[int, int]] = [(int(k), int(v)) for k, v in zip(keys, vals)]
        for ok in rng.choice(keys, size=n_overwrites, replace=True):
            ok_i = int(ok)
            nv = int(rng.integers(12, 28))
            while nv == cur_val[ok_i]:
                nv = int(rng.integers(12, 28))
            cur_val[ok_i] = nv
            was_ov[ok_i] = True
            stream.append((ok_i, nv))

        perm_init = rng.permutation(n_unique)
        ordered = [stream[i] for i in perm_init] + stream[n_unique:]

        pos = 0
        for k, v in ordered:
            xs[b, pos] = k
            xs[b, pos + 1] = v
            pos += 2
        xs[b, pos] = sep
        pos += 1

        q_keys = rng.choice(keys, size=n_queries, replace=(n_queries > n_unique))
        for qk in q_keys:
            qk_i = int(qk)
            ans = cur_val[qk_i]
            xs[b, pos] = qk_i
            xs[b, pos + 1] = ans
            ys[b, pos] = ans
            is_ov[b, pos] = was_ov[qk_i]
            pos += 2

    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(is_ov)


# ---------------------------------------------------------------------------
# Task 2: Non-Abelian Dihedral D_4 Word Tracking (Per-step Dense State Supervision)
# ---------------------------------------------------------------------------
def _d4_mul(s1: int, s2: int) -> int:
    f1, k1 = s1 // 4, s1 % 4
    f2, k2 = s2 // 4, s2 % 4
    if f2 == 0:
        return 4 * f1 + ((k1 + k2) % 4)
    return 4 * (f1 ^ 1) + ((-k1 + k2) % 4)


# Tokens: 0..3 generators {r=(0,1)=1, r_inv=(0,3)=3, s=(1,0)=4, sr=(1,1)=5},
# 4..5 NOOP distractors (identity 0), 8..15=D4 states (0..7). ZERO state tokens in xs!
_D4_GEN_ELEMS = [1, 3, 4, 5, 0, 0]


def _gen_d4_batch(
    rng: np.random.Generator,
    batch_size: int,
    seq_ops: int,
    noop_prob: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs = np.zeros((batch_size, seq_ops), dtype=np.int64)
    ys = np.zeros((batch_size, seq_ops), dtype=np.int64)
    for b in range(batch_size):
        st = 0
        for i in range(seq_ops):
            tok = int(rng.integers(4, 6)) if rng.random() < noop_prob else int(rng.integers(0, 4))
            xs[b, i] = tok
            st = _d4_mul(st, _D4_GEN_ELEMS[tok])
            ys[b, i] = 8 + st
    return torch.from_numpy(xs), torch.from_numpy(ys)


def _gen_d4_commutator_probe(seq_ops: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = []
    targets = []
    rng = np.random.default_rng(999)
    for _ in range(64):
        prefix = [4, 5]  # NOOP
        suffix = list(rng.integers(0, 4, size=seq_ops - 4))
        for pair in ((0, 2), (2, 0)):
            ops = prefix + list(pair) + suffix
            st = 0
            for tok in ops:
                st = _d4_mul(st, _D4_GEN_ELEMS[tok])
            seqs.append(ops)
            targets.append(8 + st)
    return torch.tensor(seqs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


# ---------------------------------------------------------------------------
# Task 3: Mixed Fast-Syntax + Slow-Revision (Timescale Interference Test)
# ---------------------------------------------------------------------------
def _gen_mixed_timescale_batch(
    rng: np.random.Generator,
    batch_size: int,
    n_keys: int = 4,
    n_overwrites: int = 2,
    syntax_len: int = 12,
    n_queries: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Vocab (36): 0..9 KV keys, 10..21 KV vals, 22..33 syntax tokens ((a + 2*b) % 12), 34=SEP, 35=PAD
    sep, pad = 34, 35
    ev_len = 2 * (n_keys + n_overwrites)
    total_len = ev_len + syntax_len + 1 + 2 * n_queries
    xs = np.full((batch_size, total_len), pad, dtype=np.int64)
    ys = np.full((batch_size, total_len), -100, dtype=np.int64)
    syn_mask = np.zeros((batch_size, total_len), dtype=bool)
    kv_mask = np.zeros((batch_size, total_len), dtype=bool)

    for b in range(batch_size):
        keys = rng.choice(10, size=n_keys, replace=False)
        vals = rng.integers(10, 22, size=n_keys)
        cur = {int(k): int(v) for k, v in zip(keys, vals)}
        pos = 0
        for k, v in zip(keys, vals):
            xs[b, pos] = int(k)
            xs[b, pos + 1] = int(v)
            pos += 2
        for ok in rng.choice(keys, size=n_overwrites, replace=True):
            nv = int(rng.integers(10, 22))
            while nv == cur[int(ok)]:
                nv = int(rng.integers(10, 22))
            cur[int(ok)] = nv
            xs[b, pos] = int(ok)
            xs[b, pos + 1] = nv
            pos += 2

        a = int(rng.integers(0, 12))
        b_tok = int(rng.integers(0, 12))
        xs[b, pos] = 22 + a
        xs[b, pos + 1] = 22 + b_tok
        for s_i in range(2, syntax_len):
            nxt = (a + 2 * b_tok) % 12
            xs[b, pos + s_i] = 22 + nxt
            ys[b, pos + s_i - 1] = 22 + nxt
            syn_mask[b, pos + s_i - 1] = True
            a, b_tok = b_tok, nxt
        pos += syntax_len
        xs[b, pos] = sep
        pos += 1

        for qk in rng.choice(keys, size=n_queries, replace=False):
            ans = cur[int(qk)]
            xs[b, pos] = int(qk)
            xs[b, pos + 1] = ans
            ys[b, pos] = ans
            kv_mask[b, pos] = True
            pos += 2

    return torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(syn_mask), torch.from_numpy(kv_mask)


def _build_matched_model(mixer_kind: str, vocab_size: int, ref_non_emb: int) -> Cycle1SeqModel:
    dff = match_d_ff_for_target_params(
        ref_non_emb, vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, mixer_kind=mixer_kind
    )
    return Cycle1SeqModel(
        vocab_size=vocab_size, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=dff, mixer_kind=mixer_kind
    )


def _run_mqar_task(mixer_kind: str, seed: int, ref_non_emb: int) -> dict[str, Any]:
    seed_everything(seed)
    rng = np.random.default_rng(10_000 + seed)
    model = _build_matched_model(mixer_kind, vocab_size=30, ref_non_emb=ref_non_emb)
    opt = make_adamw_no_embed_decay(model, lr=5e-3, weight_decay=0.01)
    steps = 360
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        xb, yb, _ = _gen_overwrite_mqar_batch(rng, batch_size=32, n_unique=4, n_overwrites=2, n_queries=4)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, 30), yb.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    wall_s = time.perf_counter() - t0

    model.eval()
    eval_rng = np.random.default_rng(90_000 + seed)
    metrics: dict[str, Any] = {"wall_s": round(wall_s, 3)}
    with torch.no_grad():
        for tag, nu, nov, nq in (("id_4", 4, 2, 4), ("ood_8", 8, 4, 6)):
            xs, ys, is_ov = _gen_overwrite_mqar_batch(eval_rng, batch_size=256, n_unique=nu, n_overwrites=nov, n_queries=nq)
            logits = model(xs)
            preds = logits.argmax(dim=-1)
            valid = ys != -100
            correct = (preds == ys) & valid
            ov_valid = valid & is_ov
            ret_valid = valid & (~is_ov)
            metrics[f"{tag}_overall"] = float(correct.sum().item() / max(1, valid.sum().item()))
            metrics[f"{tag}_overwrite"] = float((correct & ov_valid).sum().item() / max(1, ov_valid.sum().item()))
            metrics[f"{tag}_retained"] = float((correct & ret_valid).sum().item() / max(1, ret_valid.sum().item()))
        xs_diag, _, _ = _gen_overwrite_mqar_batch(eval_rng, batch_size=32, n_unique=4, n_overwrites=2, n_queries=4)
        model(xs_diag)
        metrics["key_diagnostics"] = model.mixer_key_diagnostics()
    return metrics


def _run_d4_task(mixer_kind: str, seed: int, ref_non_emb: int) -> dict[str, Any]:
    seed_everything(seed)
    rng = np.random.default_rng(20_000 + seed)
    model = _build_matched_model(mixer_kind, vocab_size=16, ref_non_emb=ref_non_emb)
    opt = make_adamw_no_embed_decay(model, lr=5e-3, weight_decay=0.01)
    steps = 300
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    t0 = time.perf_counter()
    model.train()
    for _ in range(steps):
        xb, yb = _gen_d4_batch(rng, batch_size=32, seq_ops=10, noop_prob=0.25)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, 16), yb.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    wall_s = time.perf_counter() - t0

    model.eval()
    eval_rng = np.random.default_rng(91_000 + seed)
    metrics: dict[str, Any] = {"wall_s": round(wall_s, 3)}
    with torch.no_grad():
        for tag, seq_ops, np_prob in (("id_10", 10, 0.25), ("ood_30_noisy", 30, 0.50)):
            xs, ys = _gen_d4_batch(eval_rng, batch_size=256, seq_ops=seq_ops, noop_prob=np_prob)
            logits = model(xs)
            preds = logits[:, -1, 8:16].argmax(dim=-1) + 8
            metrics[tag] = float((preds == ys[:, -1]).float().mean().item())

        xs_c, ys_c = _gen_d4_commutator_probe(seq_ops=10)
        preds_c = model(xs_c)[:, -1, 8:16].argmax(dim=-1) + 8
        both_ok = (preds_c[0::2] == ys_c[0::2]) & (preds_c[1::2] == ys_c[1::2]) & (preds_c[0::2] != preds_c[1::2])
        metrics["commutator_acc"] = float(both_ok.float().mean().item())
    return metrics


def _run_mixed_task(mixer_kind: str, seed: int, ref_non_emb: int) -> dict[str, Any]:
    seed_everything(seed)
    rng = np.random.default_rng(30_000 + seed)
    model = _build_matched_model(mixer_kind, vocab_size=36, ref_non_emb=ref_non_emb)
    opt = make_adamw_no_embed_decay(model, lr=5e-3, weight_decay=0.01)
    steps = 360
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    t0 = time.perf_counter()
    model.train()
    for _ in range(steps):
        xb, yb, _, _ = _gen_mixed_timescale_batch(rng, batch_size=32, n_keys=4, n_overwrites=2, syntax_len=12, n_queries=4)
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, 36), yb.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    wall_s = time.perf_counter() - t0

    model.eval()
    eval_rng = np.random.default_rng(92_000 + seed)
    with torch.no_grad():
        xs, ys, syn_m, kv_m = _gen_mixed_timescale_batch(
            eval_rng, batch_size=256, n_keys=4, n_overwrites=2, syntax_len=18, n_queries=4
        )
        logits = model(xs)
        preds = logits.argmax(dim=-1)
        syn_acc = float(((preds == ys) & syn_m).sum().item() / max(1, syn_m.sum().item()))
        kv_acc = float(((preds == ys) & kv_m).sum().item() / max(1, kv_m.sum().item()))
    return {
        "wall_s": round(wall_s, 3),
        "local_syntax_acc": syn_acc,
        "distant_revision_acc": kv_acc,
        "joint_mean": 0.5 * (syn_acc + kv_acc),
    }


def main() -> None:
    t_start = time.perf_counter()
    seeds = [0, 1, 2]
    wt_root = Path(__file__).resolve().parents[2]

    ref_model = Cycle1SeqModel(vocab_size=30, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=REF_D_FF, mixer_kind="attn")
    ref_non_emb = count_trainable_params(ref_model) - ref_model.embed.weight.numel()

    arm_meta: dict[str, Any] = {}
    for arm_name, kind in ARMS:
        m = _build_matched_model(kind, vocab_size=30, ref_non_emb=ref_non_emb)
        non_emb = count_trainable_params(m) - m.embed.weight.numel()
        state_desc = "O(T): 384*T B/layer fp32" if kind == "attn" else "O(1): 2,304 B/layer fp32"
        arm_meta[arm_name] = {
            "mixer_kind": kind,
            "d_ff": m.blocks[0].ffn.w_gate.out_features,
            "non_emb_params": non_emb,
            "rel_param_diff": round((non_emb - ref_non_emb) / ref_non_emb, 5),
            "state_bytes_per_layer": state_desc,
        }

    raw_mqar: dict[str, list[dict[str, Any]]] = {a: [] for a, _ in ARMS}
    raw_d4: dict[str, list[dict[str, Any]]] = {a: [] for a, _ in ARMS}
    raw_mixed: dict[str, list[dict[str, Any]]] = {a: [] for a, _ in ARMS}

    for arm_name, kind in ARMS:
        arm_t0 = time.perf_counter()
        for s in seeds:
            raw_mqar[arm_name].append(_run_mqar_task(kind, s, ref_non_emb))
            raw_d4[arm_name].append(_run_d4_task(kind, s, ref_non_emb))
            raw_mixed[arm_name].append(_run_mixed_task(kind, s, ref_non_emb))
        print(
            f"[RO-01 {time.perf_counter()-arm_t0:.1f}s] {arm_name:24s} | "
            f"MQAR_ID={np.mean([x['id_4_overall'] for x in raw_mqar[arm_name]]):.3f} "
            f"OV={np.mean([x['id_4_overwrite'] for x in raw_mqar[arm_name]]):.3f} "
            f"OOD8={np.mean([x['ood_8_overall'] for x in raw_mqar[arm_name]]):.3f} | "
            f"D4_ID={np.mean([x['id_10'] for x in raw_d4[arm_name]]):.3f} "
            f"D4_OOD30={np.mean([x['ood_30_noisy'] for x in raw_d4[arm_name]]):.3f} | "
            f"Syn={np.mean([x['local_syntax_acc'] for x in raw_mixed[arm_name]]):.3f} "
            f"DistKV={np.mean([x['distant_revision_acc'] for x in raw_mixed[arm_name]]):.3f}",
            flush=True,
        )

    def agg(rows_by_arm: dict[str, list[dict[str, Any]]], keys: list[str]) -> dict[str, Any]:
        return {
            arm: {k: summarize_seeds([float(r[k]) for r in rows]) for k in keys}
            for arm, rows in rows_by_arm.items()
        }

    summary = {
        "selective_overwrite_mqar": agg(
            raw_mqar, ["id_4_overall", "id_4_overwrite", "id_4_retained", "ood_8_overall", "ood_8_overwrite", "wall_s"]
        ),
        "nonabelian_d4_tracking": agg(raw_d4, ["id_10", "ood_30_noisy", "commutator_acc", "wall_s"]),
        "mixed_fast_slow_timescale": agg(raw_mixed, ["local_syntax_acc", "distant_revision_acc", "joint_mean", "wall_s"]),
    }

    key_collapse_summary = {}
    for arm_name, _ in ARMS:
        l0_ranks, l1_ranks, l1_cos = [], [], []
        for r in raw_mqar[arm_name]:
            kd = r.get("key_diagnostics", [])
            if len(kd) >= 2:
                l0_ranks.append(kd[0]["eff_rank"])
                l1_ranks.append(kd[1]["eff_rank"])
                l1_cos.append(kd[1]["mean_abs_offdiag_cos"])
        if l0_ranks:
            key_collapse_summary[arm_name] = {
                "layer0_eff_rank": summarize_seeds(l0_ranks),
                "layer1_eff_rank": summarize_seeds(l1_ranks),
                "layer1_abs_offdiag_cos": summarize_seeds(l1_cos),
            }

    elapsed = round(time.perf_counter() - t_start, 3)

    hdcd_d4_ood = summary["nonabelian_d4_tracking"]["R6_HDCD_MultiTimescale"]["ood_30_noisy"]["mean"]
    delta_d4_ood = summary["nonabelian_d4_tracking"]["R3_Delta_FullSphere_SFOR"]["ood_30_noisy"]["mean"]
    hdcd_mixed = summary["mixed_fast_slow_timescale"]["R6_HDCD_MultiTimescale"]["joint_mean"]["mean"]
    attn_id_mqar = summary["selective_overwrite_mqar"]["A0_CausalAttn"]["id_4_overall"]["mean"]

    payload = {
        "schema": "nanolm.cycle1.experiment.v1",
        "experiment_id": "RECURRENT-OPERATOR-01",
        "research_question": (
            "At strictly matched non-embedding parameter count (+-3%) and matched 2,304 B/layer recurrent state, "
            "does partitioning heads into Fast-Syntax + Slow-FullSphere-Delta + Orthogonal-Cayley-Householder-RightValue "
            "guilds (HDCD) jointly solve selective overwrite, non-Abelian D_4 group tracking, and mixed-timescale "
            "retention better than homogeneous Delta, BSVD, SkewCayley, and GatedLinear operators?"
        ),
        "hypothesis": (
            "1) R2_Delta_SiLU suffers positive-orthant key rank collapse vs R3_Delta_FullSphere_SFOR; "
            "2) Homogeneous left-key Delta (R2/R3) fails non-Abelian D_4 tracking while right-value orthogonal transport "
            "(R4_BSVD, R5_SkewCayley, R6_HDCD) solves it; "
            "3) R6_HDCD combines fast local syntax, slow overwrite MQAR, and OOD D_4 group tracking in a single O(1)-state mixer."
        ),
        "competing_explanation": "Uniform R3_Delta_FullSphere_SFOR or R4_BSVD_RightValue might match HDCD without multi-timescale head specialization.",
        "code_revision": _git_sha(wt_root),
        "dataset_provenance": "Synthetic Selective-Overwrite MQAR, Dense-Supervision D_4 generator streams, and Mixed Syntax+KV (seeds 0,1,2)",
        "arms": arm_meta,
        "metrics": ["id_4_overall", "id_4_overwrite", "ood_8_overall", "d4_id_10", "d4_ood_30_noisy", "local_syntax_acc", "distant_revision_acc"],
        "success_criteria": "A0_CausalAttn ID MQAR >= 0.85 (instrument sanity); R6_HDCD improves D4 OOD over R3 and matches/beats on mixed joint_mean",
        "failure_criteria": "A0_CausalAttn fails ID MQAR (<0.80) or HDCD degrades on all three tasks vs uniform R3",
        "seeds": seeds,
        "raw_per_seed": {
            "mqar": raw_mqar,
            "d4": raw_d4,
            "mixed": raw_mixed,
        },
        "results": {
            "summary": summary,
            "key_rank_diagnostics": key_collapse_summary,
        },
        "resource_consumption": {
            "paid_compute_usd": 0.0,
            "device": "cpu_float32_1thread",
            "wall_clock_seconds": elapsed,
        },
        "independent_verification": {
            "attn_sanity_id_mqar": attn_id_mqar,
            "hdcd_vs_delta_d4_ood30": round(hdcd_d4_ood - delta_d4_ood, 4),
            "hdcd_mixed_joint_mean": round(hdcd_mixed, 4),
        },
        "final_decision": (
            "INSTRUMENT_SANITY_CHECK_FAILED"
            if attn_id_mqar < 0.80
            else (
                "PROMOTE_HDCD_MULTI_TIMESCALE"
                if hdcd_d4_ood > delta_d4_ood + 0.05
                else "INCONCLUSIVE_MIXED_OPERATOR_TRADEOFF"
            )
        ),
    }

    out = wt_root / "artifacts" / "cycle1" / "RECURRENT-OPERATOR-01.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"RECURRENT-OPERATOR-01 complete in {elapsed}s -> {out} | Decision: {payload['final_decision']}")


if __name__ == "__main__":
    main()
