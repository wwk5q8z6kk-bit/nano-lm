from __future__ import annotations

import json
import math
import re
import subprocess
import time
from collections import Counter
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

BOS_CH = "\x02"
_SPLIT_RE = re.compile(r"\x02|\n| ?[a-z]+|[A-Z0-9]+|[^\sA-Za-z0-9]+| +")


def _git_sha(repo_dir: Path) -> str:
    res = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() or "unknown"


class AsciiByteTokenizer:
    """Bijective 128-symbol ASCII byte tokenizer."""

    def __init__(self) -> None:
        self.vocab_size = 128

    def encode(self, text: str) -> list[int]:
        raw = text.encode("ascii")
        return [b for b in raw]

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("ascii")


class DeterministicBPETokenizer:
    """Deterministic ASCII BPE tokenizer (vocab=384) with optional protected-span merge constraint.

    Base IDs 0..127 correspond to single ASCII bytes; IDs 128..vocab_size-1 correspond to learned merges.
    When protect_alnum_spans=True, chunks containing digits [0-9] or uppercase letters [A-Z] are never merged
    and always remain atomic byte tokens.
    """

    def __init__(
        self,
        vocab_size: int = 384,
        protect_alnum_spans: bool = False,
    ) -> None:
        self.vocab_size = int(vocab_size)
        self.protect_alnum_spans = bool(protect_alnum_spans)
        self.merges: list[tuple[int, int]] = []
        self.merge_rank: dict[tuple[int, int], int] = {}
        self.id_to_bytes: list[bytes] = [bytes([i]) for i in range(128)]

    @staticmethod
    def _is_protected_chunk(chunk: str) -> bool:
        return any(("0" <= c <= "9") or ("A" <= c <= "Z") for c in chunk)

    def fit(self, texts: list[str]) -> None:
        chunk_counts: Counter[tuple[int, ...]] = Counter()
        for t in texts:
            for chunk in _SPLIT_RE.findall(t):
                if len(chunk) <= 1 or chunk in (BOS_CH, "\n"):
                    continue
                if self.protect_alnum_spans and self._is_protected_chunk(chunk):
                    continue
                chunk_counts[tuple(chunk.encode("ascii"))] += 1

        self.merges = []
        self.merge_rank = {}
        self.id_to_bytes = [bytes([i]) for i in range(128)]

        n_merges = self.vocab_size - 128
        for step in range(n_merges):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for seq, freq in chunk_counts.items():
                if len(seq) < 2:
                    continue
                for i in range(len(seq) - 1):
                    pair_counts[(seq[i], seq[i + 1])] += freq
            if not pair_counts:
                break
            best_pair, _ = pair_counts.most_common(1)[0]
            new_id = 128 + step
            self.merges.append(best_pair)
            self.merge_rank[best_pair] = step
            self.id_to_bytes.append(self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]])

            new_counts: Counter[tuple[int, ...]] = Counter()
            a, b = best_pair
            for seq, freq in chunk_counts.items():
                if len(seq) < 2:
                    new_counts[seq] += freq
                    continue
                merged: list[int] = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                new_counts[tuple(merged)] += freq
            chunk_counts = new_counts

    def encode_chunk(self, chunk: str) -> list[int]:
        raw = list(chunk.encode("ascii"))
        if len(raw) <= 1 or chunk in (BOS_CH, "\n"):
            return raw
        if self.protect_alnum_spans and self._is_protected_chunk(chunk):
            return raw
        seq = raw
        while len(seq) >= 2:
            best_pos = -1
            best_rank = 10**9
            for i in range(len(seq) - 1):
                r = self.merge_rank.get((seq[i], seq[i + 1]))
                if r is not None and r < best_rank:
                    best_rank = r
                    best_pos = i
            if best_pos < 0:
                break
            a, b = seq[best_pos], seq[best_pos + 1]
            new_id = 128 + best_rank
            out: list[int] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == a and seq[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
        return seq

    def encode(self, text: str) -> list[int]:
        chunks = _SPLIT_RE.findall(text)
        assert "".join(chunks) == text, "Regex split did not partition input cleanly"
        out: list[int] = []
        for ch in chunks:
            out.extend(self.encode_chunk(ch))
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.id_to_bytes[i] for i in ids if 0 <= i < len(self.id_to_bytes)).decode("ascii", errors="replace")


# --- Strictly split ASCII corpus generator ---
_PROSE_SUBJECTS = [
    "the resident clinician",
    "the bedside monitor",
    "the automated sensor",
    "the emergency triage team",
    "the cardiology fellow",
    "the ambulatory tracker",
]
_PROSE_VERBS = [
    "recorded stable baseline parameters",
    "observed gradual recovery trends",
    "verified medication dosage logs",
    "documented normal sinus rhythm",
    "confirmed clear pulmonary auscultation",
    "reported steady hydration response",
]
_PROSE_CONTEXTS = [
    "during the morning clinical rounding window",
    "following intravenous saline administration protocol",
    "without any recurrent febrile episodes noted",
    "while maintaining continuous oxygen saturation monitoring",
    "after reviewing previous laboratory biomarker panels",
    "before transferring summary notes to archive",
]
_PROSE_OUTCOMES = [
    "and scheduled routine follow-up assessment tomorrow.",
    "and approved conservative outpatient observation plan.",
    "and cleared the preliminary safety verification checklist.",
    "and updated the chronological chart ledger accordingly.",
]


def _build_strict_splits(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(41_000 + seed)

    # 1. Prose split: hold out unseen (subject, verb, context, outcome) combinations deterministically
    all_prose: list[str] = []
    for si, s in enumerate(_PROSE_SUBJECTS):
        for vi, v in enumerate(_PROSE_VERBS):
            for ci, c in enumerate(_PROSE_CONTEXTS):
                for oi, o in enumerate(_PROSE_OUTCOMES):
                    line = f"{s} {v} {c} {o}"
                    all_prose.append((si, vi, ci, oi, line))
    prose_train: list[str] = []
    prose_eval: list[str] = []
    for si, vi, ci, oi, line in all_prose:
        if (si * 7 + vi * 13 + ci * 17 + oi * 23 + seed) % 6 == 0:
            prose_eval.append(line)
        else:
            prose_train.append(line)

    # 2. Exact Arithmetic: ADD XXX + YYY = ZZZ (3-digit X,Y in 100..499, Z in 200..998)
    # Disjoint train vs eval pairs via modular split
    arith_train: list[tuple[str, str]] = []
    arith_eval: list[tuple[str, str]] = []
    pairs_seen: set[tuple[int, int]] = set()
    while len(arith_train) < 1200 or len(arith_eval) < 120:
        x = int(rng.integers(100, 500))
        y = int(rng.integers(100, 500))
        if (x, y) in pairs_seen:
            continue
        pairs_seen.add((x, y))
        prompt = f"ADD {x:03d} + {y:03d} = "
        ans = f"{x + y:03d}"
        if (x * 31 + y * 97 + seed) % 7 == 0:
            if len(arith_eval) < 120:
                arith_eval.append((prompt, ans))
        else:
            if len(arith_train) < 1200:
                arith_train.append((prompt, ans))

    # 3. Exact Code Reverse: REV <6-char hex/alnum ID> -> <reversed 6-char ID>
    alphabet = list("0123456789ABCDEF")
    rev_train: list[tuple[str, str]] = []
    rev_eval: list[tuple[str, str]] = []
    ids_seen: set[str] = set()
    while len(rev_train) < 1200 or len(rev_eval) < 120:
        code = "".join(alphabet[int(i)] for i in rng.integers(0, len(alphabet), size=6))
        if code in ids_seen:
            continue
        ids_seen.add(code)
        prompt = f"REV {code} -> "
        ans = code[::-1]
        if len(rev_eval) < 120 and len(rev_train) >= 1200:
            rev_eval.append((prompt, ans))
        elif len(rev_train) < 1200 and (hash(code) + seed) % 8 != 0:
            rev_train.append((prompt, ans))
        elif len(rev_eval) < 120:
            rev_eval.append((prompt, ans))

    return {
        "prose_train": prose_train,
        "prose_eval": prose_eval,
        "arith_train": arith_train,
        "arith_eval": arith_eval,
        "rev_train": rev_train,
        "rev_eval": rev_eval,
    }


def _collate_sequences(seqs: list[list[int]], pad_id: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(s) for s in seqs)
    bs = len(seqs)
    x_in = torch.full((bs, max_len - 1), pad_id, dtype=torch.long)
    y_tgt = torch.full((bs, max_len - 1), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        n = len(s) - 1
        x_in[i, :n] = torch.tensor(s[:n], dtype=torch.long)
        y_tgt[i, :n] = torch.tensor(s[1:], dtype=torch.long)
    return x_in, y_tgt, (y_tgt != -100)


def _deterministic_solve_line(prompt: str) -> str:
    """Exact 0-parameter symbolic/deterministic baseline for arithmetic and code reversal."""
    s = prompt.strip()
    m_add = re.match(r"^ADD\s+(\d+)\s+\+\s+(\d+)\s*=$", s)
    if m_add:
        return str(int(m_add.group(1)) + int(m_add.group(2))).zfill(3)
    m_rev = re.match(r"^REV\s+([A-Z0-9]+)\s*->$", s)
    if m_rev:
        return m_rev.group(1)[::-1]
    return ""


def _train_and_eval_tokenizer_arm(
    seed: int,
    arm_name: str,
    tokenizer: AsciiByteTokenizer | DeterministicBPETokenizer,
    splits: dict[str, Any],
    d_ff: int,
    steps: int = 380,
) -> dict[str, float]:
    seed_everything(seed)
    rng = np.random.default_rng(51_000 + seed)

    # Pre-encode all train and eval samples & verify exact bijective decode(encode(s)) == s
    prose_tr_texts = [f"{BOS_CH}{s}\n" for s in splits["prose_train"]]
    prose_ev_texts = [f"{BOS_CH}{s}\n" for s in splits["prose_eval"]]
    arith_tr_full = [f"{BOS_CH}{p}{a}\n" for p, a in splits["arith_train"]]
    rev_tr_full = [f"{BOS_CH}{p}{a}\n" for p, a in splits["rev_train"]]

    roundtrip_ok = True
    for txt in prose_tr_texts[:80] + prose_ev_texts + arith_tr_full[:80] + rev_tr_full[:80]:
        if tokenizer.decode(tokenizer.encode(txt)) != txt:
            roundtrip_ok = False
            break
    for p, a in splits["arith_eval"] + splits["rev_eval"]:
        full = f"{BOS_CH}{p}{a}\n"
        if tokenizer.decode(tokenizer.encode(full)) != full:
            roundtrip_ok = False
            break

    prose_tr_ids = [tokenizer.encode(t) for t in prose_tr_texts]
    arith_tr_ids = [tokenizer.encode(t) for t in arith_tr_full]
    rev_tr_ids = [tokenizer.encode(t) for t in rev_tr_full]
    prose_ev_ids = [tokenizer.encode(t) for t in prose_ev_texts]

    model = Cycle1SeqModel(
        vocab_size=tokenizer.vocab_size,
        d_model=48,
        n_heads=4,
        n_layers=2,
        d_ff=d_ff,
        mixer_kind="attn",
        pointer_mode="none",
    )
    opt = make_adamw_no_embed_decay(model, lr=3.5e-3, weight_decay=0.01)

    model.train()
    for _ in range(steps):
        # Balanced batch: 12 prose, 10 arith, 10 rev
        batch_seqs: list[list[int]] = []
        for idx in rng.integers(0, len(prose_tr_ids), size=12):
            batch_seqs.append(prose_tr_ids[int(idx)])
        for idx in rng.integers(0, len(arith_tr_ids), size=10):
            batch_seqs.append(arith_tr_ids[int(idx)])
        for idx in rng.integers(0, len(rev_tr_ids), size=10):
            batch_seqs.append(rev_tr_ids[int(idx)])

        x_in, y_tgt, _ = _collate_sequences(batch_seqs)
        logits = model(x_in)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    # 1. Exact True Bits-Per-Byte on held-out Prose + measured latency per 1k raw bytes
    total_nll_nats = 0.0
    total_raw_bytes = sum(len(s.encode("ascii")) + 1 for s in splits["prose_eval"])  # includes trailing '\n'
    total_tokens = sum(len(ids) - 1 for ids in prose_ev_ids)  # excludes BOS

    x_ev, y_ev, _ = _collate_sequences(prose_ev_ids)
    with torch.no_grad():
        t_start = time.perf_counter()
        # Run 3 passes over eval prose for stable latency timing on CPU
        for _ in range(3):
            logits = model(x_ev)
        elapsed_ms = ((time.perf_counter() - t_start) / 3.0) * 1000.0
        total_nll_nats = float(
            F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y_ev.reshape(-1), ignore_index=-100, reduction="sum").item()
        )

    prose_bpb = total_nll_nats / (math.log(2.0) * total_raw_bytes)
    tokens_per_byte = total_tokens / total_raw_bytes
    ms_per_1k_bytes = elapsed_ms * (1000.0 / total_raw_bytes)

    # 2. Greedy exact accuracy on held-out Exact_Arithmetic & Exact_Code_Reverse
    def eval_greedy_task(pairs: list[tuple[str, str]], max_gen_tokens: int) -> float:
        correct = 0
        # Batch by exact prompt token length when possible, else group by len
        by_len: dict[int, list[tuple[list[int], str]]] = {}
        for p, a in pairs:
            pids = tokenizer.encode(f"{BOS_CH}{p}")
            by_len.setdefault(len(pids), []).append((pids, a))
        nl_id = ord("\n")
        with torch.no_grad():
            for plen, items in by_len.items():
                cur = torch.tensor([x[0] for x in items], dtype=torch.long)  # [B, plen]
                bs = cur.shape[0]
                gen_ids: list[list[int]] = [[] for _ in range(bs)]
                done = [False] * bs
                for _ in range(max_gen_tokens):
                    logits = model(cur)[:, -1, :]
                    nxt = logits.argmax(dim=-1)
                    cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
                    for i in range(bs):
                        if not done[i]:
                            tid = int(nxt[i].item())
                            if tid == nl_id or ("\n" in tokenizer.decode([tid])):
                                dec = tokenizer.decode([tid])
                                if "\n" in dec:
                                    # append any prefix before newline if multi-char token
                                    before_nl = dec.split("\n", 1)[0]
                                    if before_nl:
                                        gen_ids[i].extend(tokenizer.encode(before_nl))
                                done[i] = True
                            else:
                                gen_ids[i].append(tid)
                    if all(done):
                        break
                for i, (_, target_ans) in enumerate(items):
                    pred = tokenizer.decode(gen_ids[i]).strip()
                    if pred == target_ans:
                        correct += 1
        return float(correct / len(pairs))

    arith_acc = eval_greedy_task(splits["arith_eval"], max_gen_tokens=6)
    rev_acc = eval_greedy_task(splits["rev_eval"], max_gen_tokens=8)

    # Relative FLOPs per byte (approx linear + quadratic attention proxy normalized to byte level tpb=1)
    mean_seq_toks = float(np.mean([len(ids) - 1 for ids in prose_ev_ids]))
    non_emb = count_trainable_params(model) - tokenizer.vocab_size * 48
    return {
        "prose_bpb": float(prose_bpb),
        "arith_exact_acc": arith_acc,
        "code_rev_exact_acc": rev_acc,
        "tokens_per_byte": float(tokens_per_byte),
        "measured_ms_per_1k_bytes": float(ms_per_1k_bytes),
        "relative_flops_per_byte": float(tokens_per_byte),  # scaled cleanly against R0 in summary/post
        "bpe_roundtrip_ok": 1.0 if roundtrip_ok else 0.0,
        "_mean_seq_len": mean_seq_toks,
        "_non_emb_params": float(non_emb),
    }


def main() -> None:
    t0 = time.perf_counter()
    seeds = [0, 1, 2]
    wt_root = Path(__file__).resolve().parents[2]

    # Iso-non-embedding param matching across vocab=128 (R0) and vocab=384 (R1, R2)
    ref_model = Cycle1SeqModel(vocab_size=128, d_model=48, n_heads=4, n_layers=2, d_ff=96, mixer_kind="attn")
    target_non_emb = count_trainable_params(ref_model) - 128 * 48
    dff_128 = match_d_ff_for_target_params(target_non_emb, 128, 48, 4, 2, "attn", "none")
    dff_384 = match_d_ff_for_target_params(target_non_emb, 384, 48, 4, 2, "attn", "none")

    arms = [
        "R0_Byte_Level",
        "R1_Subword_BPE",
        "R2_ProtectedSpan_Hybrid",
        "R3_Deterministic_Exact_Solver",
    ]
    raw: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    tokenizer_stats: dict[str, Any] = {}

    for seed in seeds:
        splits = _build_strict_splits(seed)
        train_corpus = (
            [f"{BOS_CH}{s}\n" for s in splits["prose_train"]]
            + [f"{BOS_CH}{p}{a}\n" for p, a in splits["arith_train"]]
            + [f"{BOS_CH}{p}{a}\n" for p, a in splits["rev_train"]]
        )

        tok_byte = AsciiByteTokenizer()
        tok_bpe = DeterministicBPETokenizer(vocab_size=384, protect_alnum_spans=False)
        tok_bpe.fit(train_corpus)
        tok_prot = DeterministicBPETokenizer(vocab_size=384, protect_alnum_spans=True)
        tok_prot.fit(train_corpus)

        if seed == 0:
            tokenizer_stats["r1_bpe_merges_count"] = len(tok_bpe.merges)
            tokenizer_stats["r2_protected_merges_count"] = len(tok_prot.merges)
            tokenizer_stats["r1_sample_arith_tokens"] = [tok_bpe.id_to_bytes[i].decode("ascii") for i in tok_bpe.encode("ADD 147 + 285 = 432\n")]
            tokenizer_stats["r2_sample_arith_tokens"] = [tok_prot.id_to_bytes[i].decode("ascii") for i in tok_prot.encode("ADD 147 + 285 = 432\n")]

        r0 = _train_and_eval_tokenizer_arm(seed, "R0_Byte_Level", tok_byte, splits, dff_128)
        r1 = _train_and_eval_tokenizer_arm(seed, "R1_Subword_BPE", tok_bpe, splits, dff_384)
        r2 = _train_and_eval_tokenizer_arm(seed, "R2_ProtectedSpan_Hybrid", tok_prot, splits, dff_384)

        # Normalize relative_flops_per_byte wrt R0_Byte_Level
        base_tpb = r0["tokens_per_byte"]
        for r in (r0, r1, r2):
            r["relative_flops_per_byte"] = float(r["tokens_per_byte"] / base_tpb)
            del r["_mean_seq_len"]
            del r["_non_emb_params"]

        # R3_Deterministic_Exact_Solver (0 params)
        t_det = time.perf_counter()
        arith_ok = sum(1 for p, a in splits["arith_eval"] if _deterministic_solve_line(p) == a) / len(splits["arith_eval"])
        rev_ok = sum(1 for p, a in splits["rev_eval"] if _deterministic_solve_line(p) == a) / len(splits["rev_eval"])
        det_ms = (time.perf_counter() - t_det) * 1000.0
        total_exact_bytes = sum(len((p + a).encode("ascii")) for p, a in splits["arith_eval"] + splits["rev_eval"])
        r3 = {
            "prose_bpb": 7.0,  # Uniform 128-symbol ASCII distribution (log2(128) = 7.0) baseline without language model
            "arith_exact_acc": float(arith_ok),
            "code_rev_exact_acc": float(rev_ok),
            "tokens_per_byte": 1.0,
            "measured_ms_per_1k_bytes": float(det_ms * (1000.0 / total_exact_bytes)),
            "relative_flops_per_byte": 0.0,
            "bpe_roundtrip_ok": 1.0,
        }

        raw["R0_Byte_Level"].append(r0)
        raw["R1_Subword_BPE"].append(r1)
        raw["R2_ProtectedSpan_Hybrid"].append(r2)
        raw["R3_Deterministic_Exact_Solver"].append(r3)

    m_keys = list(raw[arms[0]][0].keys())
    summary = {arm: {k: summarize_seeds([r[k] for r in rows]) for k in m_keys} for arm, rows in raw.items()}

    for arm in arms:
        s = summary[arm]
        print(
            f"[BYTE-TOK] {arm:30s} | BPB={s['prose_bpb']['mean']:.3f} | "
            f"Arith={s['arith_exact_acc']['mean']:.3f} Rev={s['code_rev_exact_acc']['mean']:.3f} | "
            f"Tok/B={s['tokens_per_byte']['mean']:.3f} ms/1kB={s['measured_ms_per_1k_bytes']['mean']:.3f} "
            f"RT={s['bpe_roundtrip_ok']['mean']:.1f}",
            flush=True,
        )

    r0_bpb = summary["R0_Byte_Level"]["prose_bpb"]["mean"]
    r1_bpb = summary["R1_Subword_BPE"]["prose_bpb"]["mean"]
    r2_bpb = summary["R2_ProtectedSpan_Hybrid"]["prose_bpb"]["mean"]
    r0_rev = summary["R0_Byte_Level"]["code_rev_exact_acc"]["mean"]
    r1_rev = summary["R1_Subword_BPE"]["code_rev_exact_acc"]["mean"]
    r2_rev = summary["R2_ProtectedSpan_Hybrid"]["code_rev_exact_acc"]["mean"]
    r0_arith = summary["R0_Byte_Level"]["arith_exact_acc"]["mean"]
    r1_arith = summary["R1_Subword_BPE"]["arith_exact_acc"]["mean"]
    r2_arith = summary["R2_ProtectedSpan_Hybrid"]["arith_exact_acc"]["mean"]

    if (
        r2_bpb < r0_bpb - 0.02
        and (r2_rev > r1_rev + 0.10 or r2_arith > r1_arith + 0.10)
        and summary["R2_ProtectedSpan_Hybrid"]["tokens_per_byte"]["mean"] < 0.70
    ):
        decision = "PROMOTE_PROTECTED_SPAN_HYBRID_TOKENIZATION"
    elif r0_rev + r0_arith > r1_rev + r1_arith + 0.10:
        decision = "PROMOTE_BYTE_ATOMIC_NUMERIC_AND_CODE_SPANS"
    else:
        decision = "CHECK_BYTE_VS_BPE_TRADEOFF"

    elapsed = round(time.perf_counter() - t0, 3)
    m_128 = Cycle1SeqModel(vocab_size=128, d_model=48, n_heads=4, n_layers=2, d_ff=dff_128, mixer_kind="attn")
    m_384 = Cycle1SeqModel(vocab_size=384, d_model=48, n_heads=4, n_layers=2, d_ff=dff_384, mixer_kind="attn")

    payload = {
        "schema": "nanolm.cycle2.experiment.v1",
        "experiment_id": "BYTE-VS-TOKEN-01",
        "research_question": (
            "On a single fixed-budget sequence model (L=2, d=48, H=4, matched non-embedding params) trained on mixed "
            "prose, 3-digit exact addition, and 6-char hex reversal, does standard BPE fragment exact spans relative to "
            "byte-level, and can Protected-Span Hybrid tokenization combine prose BPE compression with exact byte alignment?"
        ),
        "hypothesis": (
            "1) R0_Byte_Level preserves character alignment for ADD/REV but pays tokens_per_byte=1.000; "
            "2) R1_Subword_BPE compresses prose and lowers tokens_per_byte but corrupts numeric/code character spans; "
            "3) R2_ProtectedSpan_Hybrid combines prose compression/speedup with byte-level exactness; "
            "4) R3_Deterministic_Exact_Solver solves exact ADD/REV at 1.000 accuracy and 0 params."
        ),
        "competing_explanation": (
            "On small vocab=384, BPE merges might not disrupt short 3-digit/6-char spans severely, OR shared byte budget "
            "at d_model=48 might already reach good accuracy under R1 without span protection."
        ),
        "code_revision": _git_sha(wt_root),
        "dataset_provenance": (
            "Strictly split ASCII corpus across seeds [0, 1, 2]: combinatorial clinical prose templates (train/eval disjoint), "
            "3-digit ADD XXX + YYY = ZZZ (disjoint operand pairs), and 6-char hex REV (disjoint IDs)"
        ),
        "arms": {
            "R0_Byte_Level": {"trainable_params": count_trainable_params(m_128), "non_embedding_params": count_trainable_params(m_128) - 128 * 48, "vocab_size": 128},
            "R1_Subword_BPE": {"trainable_params": count_trainable_params(m_384), "non_embedding_params": count_trainable_params(m_384) - 384 * 48, "vocab_size": 384},
            "R2_ProtectedSpan_Hybrid": {"trainable_params": count_trainable_params(m_384), "non_embedding_params": count_trainable_params(m_384) - 384 * 48, "vocab_size": 384},
            "R3_Deterministic_Exact_Solver": {"trainable_params": 0, "non_embedding_params": 0, "vocab_size": 128},
        },
        "tokenizer_inspection": tokenizer_stats,
        "metrics": m_keys,
        "success_criteria": "Exact BPE round-trip == 1.0, R2 achieves lower prose_bpb & tokens_per_byte < 0.70 than R0 while beating R1 on arith/code_rev",
        "failure_criteria": "R1 BPE fails to degrade exact tasks or R2 fails to compress prose or match R0 on exact tasks",
        "seeds": seeds,
        "raw_per_seed": raw,
        "results": {"summary": summary},
        "resource_consumption": {"paid_compute_usd": 0.0, "device": "cpu_float32", "wall_clock_seconds": elapsed},
        "independent_verification": {
            "r2_vs_r1_code_rev_gain": round(r2_rev - r1_rev, 4),
            "r2_vs_r1_arith_gain": round(r2_arith - r1_arith, 4),
            "r2_vs_r0_prose_bpb_diff": round(r2_bpb - r0_bpb, 4),
            "r2_tokens_per_byte": round(summary["R2_ProtectedSpan_Hybrid"]["tokens_per_byte"]["mean"], 4),
            "r3_det_exact_avg": round(0.5 * (summary["R3_Deterministic_Exact_Solver"]["arith_exact_acc"]["mean"] + summary["R3_Deterministic_Exact_Solver"]["code_rev_exact_acc"]["mean"]), 4),
        },
        "final_decision": decision,
    }
    out = wt_root / "artifacts" / "cycle2" / "BYTE-VS-TOKEN-01.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"BYTE-VS-TOKEN-01 complete in {elapsed}s -> {out} | Decision: {decision}")


if __name__ == "__main__":
    main()
