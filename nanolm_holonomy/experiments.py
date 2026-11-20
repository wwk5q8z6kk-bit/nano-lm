"""Controlled Multi-Seed Experiments (X1 State Tracking, X2 LM Evolution, X3 4B Scaling, X4 Foundation)."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from nanolm_holonomy.architecture import (
    HSPA_4B_CONFIG,
    TRANSFORMER_4B_CONFIG,
    HSPAArchConfig,
    HSPAFamily,
    HSPAUnifiedLM,
    build_suite_configs,
)
from nanolm_holonomy.registry import ExperimentRegistry, RegistryVersion
from nanolm_holonomy.sidecar_evolution import run_x4_foundation_holonomy_evolution


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _cuda_amp_dtype() -> torch.dtype:
    """Use BF16 on Ampere+ (SM >= 8.0) and FP16 on Turing (SM < 8.0, e.g. T4)."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        return torch.bfloat16
    return torch.float16


def _autocast(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=_cuda_amp_dtype())
    return nullcontext()


def _peak_mb(device: torch.device) -> float:
    if device.type == "cuda":
        total = sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        return round(total / (1024.0 * 1024.0), 2)
    return 0.0


# ---------------------------------------------------------------------------
# Experiment X1: Non-Abelian S_4 Group Word Problem & Dynamic Associative Recall
# ---------------------------------------------------------------------------

_S4_PERMS: list[tuple[int, int, int, int]] = sorted(itertools.permutations(range(4)))
_PERM_TO_ID: dict[tuple[int, int, int, int], int] = {p: i for i, p in enumerate(_S4_PERMS)}
# 8 non-commuting generators mapped to token IDs 10..17; states mapped to 100..123
_GEN_PERMS: tuple[tuple[int, int, int, int], ...] = (
    (1, 0, 2, 3),  # (0 1)
    (0, 2, 1, 3),  # (1 2)
    (0, 1, 3, 2),  # (2 3)
    (3, 1, 2, 0),  # (0 3)
    (1, 2, 0, 3),  # (0 1 2)
    (0, 2, 3, 1),  # (1 2 3)
    (1, 2, 3, 0),  # (0 1 2 3)
    (0, 1, 2, 3),  # identity
)
_BOS_TOK = 1
_QUERY_TOK = 2
_GEN_OFFSET = 10
_STATE_OFFSET = 100


def _compose_perm(p: tuple[int, int, int, int], q: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (p[q[0]], p[q[1]], p[q[2]], p[q[3]])


def _generate_s4_batch(
    batch_size: int,
    n_blocks: int,
    block_len: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Build interleaved sequences ``[BOS, g..g, QUERY, state_tok, ...]`` with masked targets except QUERY."""
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    for _ in range(batch_size):
        state = (0, 1, 2, 3)
        seq = [_BOS_TOK]
        tgt = [-100]
        for b in range(n_blocks):
            for _ in range(block_len):
                gi = rng.randrange(len(_GEN_PERMS))
                state = _compose_perm(_GEN_PERMS[gi], state)
                seq.append(_GEN_OFFSET + gi)
                tgt.append(-100)
            seq.append(_QUERY_TOK)
            # Predict state at the position of QUERY
            tgt.append(_STATE_OFFSET + _PERM_TO_ID[state])
            if b < n_blocks - 1:
                seq.append(_STATE_OFFSET + _PERM_TO_ID[state])
                tgt.append(-100)
        xs.append(seq)
        ys.append(tgt)
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)


def _generate_s4_final_probe(
    n_probes: int,
    n_gens: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Pure unbroken sequence of ``n_gens`` generators followed by single ``QUERY`` (no intermediate states)."""
    xs: list[list[int]] = []
    targets: list[int] = []
    for _ in range(n_probes):
        state = (0, 1, 2, 3)
        seq = [_BOS_TOK]
        for _ in range(n_gens):
            gi = rng.randrange(len(_GEN_PERMS))
            state = _compose_perm(_GEN_PERMS[gi], state)
            seq.append(_GEN_OFFSET + gi)
        seq.append(_QUERY_TOK)
        xs.append(seq)
        targets.append(_STATE_OFFSET + _PERM_TO_ID[state])
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


def _generate_override_recall_batch(
    batch_size: int,
    total_events: int,
    rng: random.Random,
    device: torch.device,
    *,
    vocab_size: int = 512,
) -> tuple[Tensor, Tensor]:
    """Dynamic SET/OVERRIDE associative memory sequences with distractors."""
    # Keys: 200..215, Values: 300..363, Distractors: 400..499, SET=3, OVR=4, QUERY=2
    xs: list[list[int]] = []
    ys: list[int] = []
    for _ in range(batch_size):
        state: dict[int, int] = {}
        seq = [_BOS_TOK]
        keys = [rng.randint(200, 215) for _ in range(4)]
        for step in range(total_events):
            k = rng.choice(keys)
            v = rng.randint(300, 363)
            op = 3 if k not in state else 4
            state[k] = v
            seq.extend([op, k, v])
            # Noise distractors
            for _ in range(rng.randint(2, 4)):
                seq.append(rng.randint(400, min(vocab_size - 1, 499)))
        qk = rng.choice(list(state.keys()))
        seq.extend([_QUERY_TOK, qk])
        xs.append(seq)
        ys.append(state[qk])
    max_len = max(len(s) for s in xs)
    padded = []
    last_positions = []
    for s in xs:
        last_positions.append(len(s) - 1)
        padded.append(s + [0] * (max_len - len(s)))
    return torch.tensor(padded, dtype=torch.long, device=device), (
        torch.tensor(ys, dtype=torch.long, device=device),
        torch.tensor(last_positions, dtype=torch.long, device=device),
    )  # type: ignore[return-value]


def run_x1_state_tracking(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (101, 202, 303),
    s4_steps: int = 400,
    recall_steps: int = 350,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Experiment X1: Multi-seed Non-Abelian S_4 & Dynamic Recall across B0, B1, H1, H2, H3."""
    configs = build_suite_configs(preset="small_x1", vocab_size=512, max_seq_len=512)
    eval_rng = random.Random(9001)
    # Pre-generate identical held-out S4 probes for every model/seed
    s4_probes = {
        L: _generate_s4_final_probe(64, L, eval_rng, device)
        for L in (16, 48, 96, 192)
    }
    rec_probes = {
        ev: _generate_override_recall_batch(64, ev, eval_rng, device, vocab_size=512)
        for ev in (10, 20, 40)  # ~65, ~130, ~260 tokens (2x, 4x OOD!)
    }

    summary_arms: list[dict[str, Any]] = []
    for cfg in configs:
        seed_records: list[dict[str, Any]] = []
        for seed in seeds:
            torch.manual_seed(seed)
            rng = random.Random(seed)
            # 1A: Train S_4 State Tracking (up to 48 generators = 12 blocks * 4)
            m_s4 = HSPAUnifiedLM(cfg).to(device)
            opt = torch.optim.AdamW(m_s4.parameters(), lr=1.5e-3, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(s4_steps, 1))
            m_s4.train()
            for st in range(s4_steps):
                n_blk = 12 if st > 80 else 6
                x, y = _generate_s4_batch(batch_size, n_blk, 4, rng, device)
                opt.zero_grad(set_to_none=True)
                with _autocast(device):
                    logits = m_s4(x)
                    loss = F.cross_entropy(logits.float().view(-1, cfg.vocab_size), y.view(-1), ignore_index=-100)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m_s4.parameters(), 1.0)
                opt.step()
                sch.step()

            m_s4.eval()
            s4_acc: dict[str, float] = {}
            with torch.no_grad():
                for L, (px, py) in s4_probes.items():
                    with _autocast(device):
                        p_logits = m_s4(px)
                    preds = p_logits[:, -1, :].argmax(dim=-1)
                    s4_acc[f"L_{L}"] = round(float((preds == py).float().mean().item()), 4)

            del m_s4, opt, sch

            # 1B: Train Dynamic SET/OVERRIDE Recall (trained at 10 events ~65 tokens)
            torch.manual_seed(seed + 1000)
            m_rec = HSPAUnifiedLM(cfg).to(device)
            opt_r = torch.optim.AdamW(m_rec.parameters(), lr=1.5e-3, weight_decay=0.01)
            sch_r = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r, T_max=max(recall_steps, 1))
            m_rec.train()
            for _ in range(recall_steps):
                rx, (ry, rpos) = _generate_override_recall_batch(batch_size, 10, rng, device, vocab_size=512)
                opt_r.zero_grad(set_to_none=True)
                with _autocast(device):
                    logits_r = m_rec(rx)
                    b_idx = torch.arange(batch_size, device=device)
                    last_logits = logits_r[b_idx, rpos, :].float()
                    loss_r = F.cross_entropy(last_logits, ry)
                loss_r.backward()
                torch.nn.utils.clip_grad_norm_(m_rec.parameters(), 1.0)
                opt_r.step()
                sch_r.step()

            m_rec.eval()
            rec_acc: dict[str, float] = {}
            with torch.no_grad():
                for ev, (px, (py, ppos)) in rec_probes.items():
                    with _autocast(device):
                        l_r = m_rec(px)
                    b_idx = torch.arange(px.shape[0], device=device)
                    preds = l_r[b_idx, ppos, :].argmax(dim=-1)
                    rec_acc[f"ev_{ev}"] = round(float((preds == py).float().mean().item()), 4)

            del m_rec, opt_r, sch_r
            seed_records.append({"seed": seed, "s4_accuracy": s4_acc, "override_recall": rec_acc})

        # Aggregate seed means & stds
        def _agg(key: str, subkeys: list[str]) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for sk in subkeys:
                vals = [sr[key][sk] for sr in seed_records]
                mu = sum(vals) / len(vals)
                sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals))
                out[sk] = {"mean": round(mu, 4), "std": round(sd, 4)}
            return out

        agg_s4 = _agg("s4_accuracy", ["L_16", "L_48", "L_96", "L_192"])
        agg_rec = _agg("override_recall", ["ev_10", "ev_20", "ev_40"])
        print(
            f"[X1] {cfg.name:30s} S4(L48/L96/L192)="
            f"{agg_s4['L_48']['mean']:.3f}/{agg_s4['L_96']['mean']:.3f}/{agg_s4['L_192']['mean']:.3f} "
            f"Recall(10/20/40)={agg_rec['ev_10']['mean']:.3f}/{agg_rec['ev_20']['mean']:.3f}/{agg_rec['ev_40']['mean']:.3f}",
            flush=True,
        )
        summary_arms.append(
            {
                "config": cfg.parameter_breakdown(512),
                "s4_summary": agg_s4,
                "recall_summary": agg_rec,
                "per_seed": seed_records,
            }
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "experiment": "X1_non_abelian_state_and_ood_extrapolation",
        "seeds": list(seeds),
        "arms": summary_arms,
    }


# ---------------------------------------------------------------------------
# Experiment X2: Iso-Parameter Multi-Domain LM Evolution (~14M, Multi-Seed)
# ---------------------------------------------------------------------------

class MicroByteWordTokenizer:
    """Fast deterministic hybrid tokenizer preserving UTF-8 byte accounting."""

    def __init__(self, vocab_size: int = 4_096) -> None:
        self.vocab_size = vocab_size
        self.bos_id = 1
        self._byte_off = 16

    def encode(self, text: str) -> tuple[list[int], int]:
        n_bytes = len(text.encode("utf-8"))
        ids = [self.bos_id]
        span = self.vocab_size - (self._byte_off + 256)
        for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,15}|[0-9]{2,6}|\s+|.", text):
            if len(tok) >= 2 and not tok.isspace():
                d = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest(), "little")
                ids.append(self._byte_off + 256 + (d % span))
            else:
                for b in tok.encode("utf-8"):
                    ids.append(self._byte_off + b)
        return ids, n_bytes


def _build_x2_corpus(
    tok: MicroByteWordTokenizer,
    *,
    train_seqs: int = 2_400,
    val_seqs_per_dom: int = 96,
    seq_len: int = 256,
) -> dict[str, Any]:
    def _doc(dom: str, idx: int, rng: random.Random) -> str:
        if dom == "code":
            fn = f"solve_{idx % 60}"
            k1, k2 = rng.randint(2, 19), rng.randint(3, 29)
            return (
                f"def {fn}(arr: list[int]) -> int:\n"
                f"    dp = 0\n"
                f"    for x in arr:\n"
                f"        dp = (dp * {k1} + x + {k2}) % 1009\n"
                f"    return dp\n"
                f"# expected {fn}([1, 2]) == {(((0*k1+1+k2)%1009)*k1+2+k2)%1009}\n"
            )
        if dom == "math":
            a, b, c = rng.randint(12, 85), rng.randint(11, 75), rng.randint(5, 45)
            ans = (a * b + c) % 23
            return f"DERIVE ({a} * {b} + {c}) mod 23: mul={a*b}; add={a*b+c}; FINAL_ANSWER: {ans}\n"
        if dom == "prose":
            layer = f"fiber_{rng.randint(1, 32)}"
            rate = rng.randint(12, 96)
            return (
                f"Architecture note on {layer}: parallel holonomy scan propagates quaternion frames "
                f"in logarithmic depth across {rate} channels without quadratic KV allocation.\n"
            )
        # state domain
        v0 = rng.randint(20, 90)
        d1 = rng.randint(5, 25)
        d2 = rng.randint(3, 18)
        return f"TX INIT val={v0}; ADD {d1}; SUB {d2}; END_STATE val={v0 + d1 - d2}\n"

    def _pack(n: int, doms: list[str], seed: int) -> tuple[list[list[int]], float, list[tuple[list[int], int]]]:
        rng = random.Random(seed)
        seqs: list[list[int]] = []
        probe_pairs: list[tuple[list[int], int]] = []
        tot_b = 0
        tot_t = 0
        buf: list[int] = []
        step = 0
        while len(seqs) < n:
            d = rng.choice(doms)
            txt = _doc(d, step, rng)
            ids, nb = tok.encode(txt)
            if len(probe_pairs) < 48 and d in ("math", "state") and len(ids) >= 6:
                probe_pairs.append((ids[:-2], ids[-2]))
            tot_b += nb
            tot_t += len(ids)
            buf.extend(ids)
            while len(buf) >= seq_len + 1 and len(seqs) < n:
                seqs.append(buf[: seq_len + 1])
                buf = buf[seq_len:]
            step += 1
        return seqs, tot_b / max(tot_t, 1), probe_pairs

    tr_seqs, _, _ = _pack(train_seqs, ["code", "math", "prose", "state"], 1010)
    vals = {}
    for off, d in enumerate(("code", "math", "prose", "state"), start=1):
        v_s, v_bpt, v_pr = _pack(val_seqs_per_dom, [d], 50_000 + off * 111)
        vals[d] = (v_s, v_bpt, v_pr)
    return {"train": tr_seqs, "val": vals}


def run_x2_lm_evolution(
    *,
    device: torch.device,
    seeds: tuple[int, ...] = (42, 137),
    train_steps: int = 800,
    batch_size: int = 16,
    lr: float = 8e-4,
) -> dict[str, Any]:
    """Experiment X2: Multi-seed iso-parameter LM evolution across B0, B1, H1, H2, Gen2-H3."""
    tok = MicroByteWordTokenizer(vocab_size=4_096)
    corpus = _build_x2_corpus(tok)
    tr_seqs = corpus["train"]
    vals = corpus["val"]
    configs = build_suite_configs(preset="micro", vocab_size=tok.vocab_size, max_seq_len=1_024)

    registry = ExperimentRegistry(higher_is_better=False)
    arm_reports: list[dict[str, Any]] = []

    for idx, cfg in enumerate(configs):
        seed_runs: list[dict[str, Any]] = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = HSPAUnifiedLM(cfg).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.05)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(train_steps, 1))

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                tokens_seen += int(y.numel())
            _sync(device)
            train_sec = max(time.perf_counter() - t0, 1e-6)
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

                # Decode latency check (128 prompt -> 32 O(1)/KV decode steps)
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
                decode_ms_per_tok = round((time.perf_counter() - td0) * 1000.0 / 32.0, 3)

            mean_bpb = round(sum(dom_bpb.values()) / len(dom_bpb), 4)
            seed_runs.append(
                {
                    "seed": seed,
                    "mean_val_bpb": mean_bpb,
                    "domain_bpb": dom_bpb,
                    "probe_exact_acc": probe_acc,
                    "train_tok_per_sec": round(tokens_seen / train_sec, 1),
                    "decode_ms_per_tok": decode_ms_per_tok,
                    "peak_vram_mb": peak_mb,
                }
            )
            del model, opt, sch
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        bpb_vals = [r["mean_val_bpb"] for r in seed_runs]
        mean_bpb_all = round(sum(bpb_vals) / len(bpb_vals), 4)
        std_bpb_all = round(math.sqrt(sum((v - mean_bpb_all) ** 2 for v in bpb_vals) / len(bpb_vals)), 5)
        ver = RegistryVersion(
            name=cfg.name,
            family=cfg.family.value,
            total_parameters=cfg.parameter_count,
            mean_primary_metric=mean_bpb_all,
            seed_scores=bpb_vals,
            metrics={
                "std_val_bpb": std_bpb_all,
                "mean_tok_per_sec": round(sum(r["train_tok_per_sec"] for r in seed_runs) / len(seed_runs), 1),
                "mean_decode_ms_per_tok": round(sum(r["decode_ms_per_tok"] for r in seed_runs) / len(seed_runs), 3),
                "state_kb_at_8k": round(cfg.state_bytes_per_sequence(8192, 2) / 1024.0, 2),
                "seed_runs": seed_runs,
            },
        )
        if idx == 0:
            registry.register_baseline(ver)
            decision = {"promoted": True, "reason": "registered_baseline"}
        else:
            decision = registry.evaluate_candidate(ver, min_seeds=len(seeds), min_delta=0.0002)

        print(
            f"[X2] {cfg.name:28s} bpb={mean_bpb_all:.4f}+/-{std_bpb_all:.5f} "
            f"tok/s={ver.metrics['mean_tok_per_sec']:.0f} dec_ms={ver.metrics['mean_decode_ms_per_tok']:.2f} "
            f"state8k={ver.metrics['state_kb_at_8k']}KB reg={decision['reason']}",
            flush=True,
        )
        arm_reports.append({"config": cfg.parameter_breakdown(8192), "version": ver.metrics, "bpb": mean_bpb_all, "std_bpb": std_bpb_all, "registry_decision": decision})

    return {
        "experiment": "X2_lm_multi_seed_evolution",
        "seeds": list(seeds),
        "train_steps": train_steps,
        "arms": arm_reports,
        "registry": registry.to_dict(),
    }


# ---------------------------------------------------------------------------
# Experiment X3: 4B-Class (~3.84B) Context-Scaling Memory & Latency Law
# ---------------------------------------------------------------------------

def _measure_scaling_row(
    model: HSPAUnifiedLM, cfg: HSPAArchConfig, slen: int, device: torch.device,
    *, prefill_chunk: int = 1024, logits_to_keep: int | None = None,
) -> dict[str, Any]:
    inp = torch.randint(0, cfg.vocab_size, (1, slen), device=device)
    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        # Chunk the HSPA scan while keeping only the requested logits for this measurement.
        if not cfg.is_transformer and slen > prefill_chunk:
            st = None
            for cs in range(0, slen, prefill_chunk):
                ce = min(slen, cs + prefill_chunk)
                chunk_logits, st = model(inp[:, cs:ce], past_states=st, use_state=True, past_len=cs, logits_to_keep=logits_to_keep)
                if ce == slen:
                    cur = chunk_logits[:, -1:, :].argmax(dim=-1)
                del chunk_logits
        else:
            lg, st = model(inp, use_state=True, past_len=0, logits_to_keep=logits_to_keep)
            cur = lg[:, -1:, :].argmax(dim=-1)
            del lg
        _sync(device)
        prefill_s = max(time.perf_counter() - t0, 1e-6)

        # Incremental 16-step decode timing
        _sync(device)
        td0 = time.perf_counter()
        for step_i in range(16):
            lg_d, st = model(cur, past_states=st, use_state=True, past_len=slen + step_i, logits_to_keep=logits_to_keep)
            cur = lg_d[:, -1:, :].argmax(dim=-1)
            del lg_d
        _sync(device)
        decode_ms = round((time.perf_counter() - td0) * 1000.0 / 16.0, 3)

    peak_mb = _peak_mb(device)
    state_mb = round(cfg.state_bytes_per_sequence(slen, 2) / (1024.0 * 1024.0), 3)
    return {
        "seq_len": slen,
        "status": "ok",
        "prefill_tok_per_sec": round(slen / prefill_s, 1),
        "decode_ms_per_tok": decode_ms,
        "analytical_state_mb": state_mb,
        "peak_vram_mb": peak_mb,
    }


def run_x3_4b_scaling_benchmark(
    *,
    device: torch.device,
    seq_lengths: tuple[int, ...] = (512, 2048, 8192, 16384),
) -> dict[str, Any]:
    """Benchmark actual BF16 runtime VRAM, KV/State memory, prefill & decode speed at 3.84B parameters."""
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    cfgs = (TRANSFORMER_4B_CONFIG, replace(HSPA_4B_CONFIG, scan_chunk_size=512))
    out: list[dict[str, Any]] = []

    for cfg in cfgs:
        # Allocate 3.84B model directly in BF16 on GPU to fit comfortably inside 48GB A40 (~7.7 GB)
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                model = HSPAUnifiedLM(cfg)
        finally:
            torch.set_default_dtype(prev_dtype)
        model.eval()
        param_count = model.num_parameters
        weight_mb = round(sum(p.numel() * p.element_size() for p in model.parameters()) / (1024.0 * 1024.0), 1)

        rows: list[dict[str, Any]] = []
        for slen in seq_lengths:
            if device.type == "cuda":
                _sync(device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            try:
                rows.append(_measure_scaling_row(model, cfg, slen, device))
            except RuntimeError as exc:
                rows.append({"seq_len": slen, "status": f"oom_or_err:{str(exc)[:120]}"})

        print(f"[X3] {cfg.name:24s} params={param_count:,} weights={weight_mb}MB rows={[(r['seq_len'], r.get('decode_ms_per_tok'), r.get('peak_vram_mb')) for r in rows]}", flush=True)
        out.append({"config": cfg.parameter_breakdown(max(seq_lengths)), "weight_mb": weight_mb, "scaling": rows})
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {"experiment": "X3_4b_class_scaling_law", "device": str(device), "models": out}


def main() -> int:
    parser = argparse.ArgumentParser(description="NanoLM HSPA Autonomous Discovery Campaign Runner")
    parser.add_argument("--mode", choices=("all", "x1", "x2", "x3", "x4"), default="all")
    parser.add_argument("--out-dir", type=str, default="artifacts/nanolm_holonomy_v1")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _select_device()
    results: dict[str, Any] = {
        "meta": {
            "device": str(dev),
            "cuda_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "non-cuda",
            "torch_version": torch.__version__,
        }
    }

    if args.mode in ("all", "x1"):
        r1 = run_x1_state_tracking(device=dev)
        results["x1"] = r1
        (out_dir / "x1_state_tracking.json").write_text(json.dumps(r1, indent=2), encoding="utf-8")

    if args.mode in ("all", "x2"):
        r2 = run_x2_lm_evolution(device=dev)
        results["x2"] = r2
        (out_dir / "x2_lm_evolution.json").write_text(json.dumps(r2, indent=2), encoding="utf-8")

    if args.mode in ("all", "x3"):
        r3 = run_x3_4b_scaling_benchmark(device=dev)
        results["x3"] = r3
        (out_dir / "x3_4b_scaling.json").write_text(json.dumps(r3, indent=2), encoding="utf-8")

    if args.mode in ("all", "x4"):
        r4 = run_x4_foundation_holonomy_evolution(model_id=args.model_id, device=dev)
        results["x4"] = r4
        (out_dir / "x4_foundation_holonomy.json").write_text(json.dumps(r4, indent=2), encoding="utf-8")

    (out_dir / "holonomy_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"HOLONOMY_CAMPAIGN_COMPLETE out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
