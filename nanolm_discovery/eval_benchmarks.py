"""Synthetic mechanistic falsifiers (EXP-INV-01) and real-corpus BPB evaluators."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

# Symmetric group S3 multiplication table (6 elements: 0=e, 1=(12), 2=(23), 3=(13), 4=(123), 5=(132))
S3_MUL: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5),
    (1, 0, 4, 5, 2, 3),
    (2, 5, 0, 4, 3, 1),
    (3, 4, 5, 0, 1, 2),
    (4, 3, 1, 2, 5, 0),
    (5, 2, 3, 1, 0, 4),
)

QUERY_TOKEN = 2
BOS_TOKEN = 1


def generate_s3_batch(
    batch_size: int,
    seq_len: int,
    *,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate S3 composition sequences: token g_t in [10..15] -> target state in [10..15]."""
    gens = rng.integers(0, 6, size=(batch_size, seq_len), dtype=np.int64)
    states = np.zeros_like(gens)
    for b in range(batch_size):
        cur = 0
        for t in range(seq_len):
            cur = S3_MUL[cur][int(gens[b, t])]
            states[b, t] = cur
    inputs = torch.from_numpy(gens + 10).to(device=device)
    targets = torch.from_numpy(states + 10).to(device=device)
    loss_mask = torch.ones_like(targets, dtype=torch.bool)
    return inputs, targets, loss_mask


def generate_overwrite_batch(
    batch_size: int,
    seq_len: int,
    *,
    rng: np.random.Generator,
    device: torch.device,
    n_keys: int = 12,
    n_vals: int = 16,
    distractor_max: int = 5,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate noisy-distractor key-value overwrite & intermediate query sequences.

    Keys in [20..20+n_keys), Vals in [50..50+n_vals), Distractors in [100..250).
    Supervises every position immediately following `QUERY_TOKEN, key`.
    """
    inputs = np.full((batch_size, seq_len), BOS_TOKEN, dtype=np.int64)
    targets = np.full((batch_size, seq_len), -1, dtype=np.int64)
    mask = np.zeros((batch_size, seq_len), dtype=bool)
    for b in range(batch_size):
        kv: dict[int, int] = {}
        t = 1
        while t < seq_len - 3:
            if len(kv) >= 3 and rng.random() < 0.25:
                k = int(rng.choice(list(kv.keys())))
                inputs[b, t] = QUERY_TOKEN
                inputs[b, t + 1] = 20 + k
                targets[b, t + 1] = 50 + kv[k]
                mask[b, t + 1] = True
                t += 2
            else:
                k = int(rng.integers(0, n_keys))
                v = int(rng.integers(0, n_vals))
                inputs[b, t] = 20 + k
                inputs[b, t + 1] = 50 + v
                kv[k] = v
                t += 2
                n_dist = int(rng.integers(1, distractor_max + 1))
                for _ in range(n_dist):
                    if t >= seq_len - 3:
                        break
                    inputs[b, t] = int(rng.integers(100, 250))
                    t += 1
        # Final forced query at end
        if len(kv) > 0 and seq_len >= 4:
            k = int(rng.choice(list(kv.keys())))
            inputs[b, seq_len - 2] = QUERY_TOKEN
            inputs[b, seq_len - 1] = 20 + k
            targets[b, seq_len - 1] = 50 + kv[k]
            mask[b, seq_len - 1] = True
    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device),
        torch.from_numpy(mask).to(device),
    )


def generate_multi_needle_batch(
    batch_size: int,
    seq_len: int = 256,
    *,
    n_needles: int = 8,
    rng: np.random.Generator,
    device: torch.device,
    n_keys: int = 24,
    n_vals: int = 24,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate `n_needles` early key-value needles followed by noise gap and shuffled tail queries."""
    inputs = rng.integers(100, 250, size=(batch_size, seq_len), dtype=np.int64)
    inputs[:, 0] = BOS_TOKEN
    targets = np.full((batch_size, seq_len), -1, dtype=np.int64)
    mask = np.zeros((batch_size, seq_len), dtype=bool)
    prefix_span = max(3 * n_needles, (seq_len // 2))
    tail_start = seq_len - 2 * n_needles - 2
    for b in range(batch_size):
        keys = rng.choice(n_keys, size=n_needles, replace=False)
        vals = rng.integers(0, n_vals, size=n_needles)
        stride = max(3, (prefix_span - 4) // n_needles)
        pos = 2
        for i in range(n_needles):
            inputs[b, pos] = 20 + int(keys[i])
            inputs[b, pos + 1] = 50 + int(vals[i])
            pos += stride
        order = rng.permutation(n_needles)
        qpos = tail_start
        for idx in order:
            inputs[b, qpos] = QUERY_TOKEN
            inputs[b, qpos + 1] = 20 + int(keys[idx])
            targets[b, qpos + 1] = 50 + int(vals[idx])
            mask[b, qpos + 1] = True
            qpos += 2
    return (
        torch.from_numpy(inputs).to(device),
        torch.from_numpy(targets).to(device),
        torch.from_numpy(mask).to(device),
    )


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Pure-numpy trapezoidal AUROC (`0.5` if labels are constant)."""
    if labels.size == 0 or np.all(labels == labels[0]):
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_lbl = labels[order]
    n_pos = float(np.sum(sorted_lbl == 1))
    n_neg = float(np.sum(sorted_lbl == 0))
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5
    ranks = np.arange(1, len(sorted_lbl) + 1, dtype=np.float64)
    # Average ranks for ties
    _, inv, counts = np.unique(scores[order], return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        avg_ranks = np.zeros_like(ranks)
        idx = 0
        for c in counts:
            avg_ranks[idx : idx + c] = ranks[idx : idx + c].mean()
            idx += c
        ranks = avg_ranks
    rank_sum_pos = float(np.sum(ranks[sorted_lbl == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(max(0.0, min(1.0, auc)))


def evaluate_epistemic_calibration(
    model: torch.nn.Module,
    *,
    rng: np.random.Generator,
    device: torch.device,
    seq_len: int = 96,
    batch_size: int = 32,
) -> dict[str, float]:
    """Test whether model's internal uncertainty/entropy rises on unseen keys vs seen keys."""
    model.eval()
    inputs, targets, mask = generate_overwrite_batch(batch_size, seq_len, rng=rng, device=device)
    # Half of sequences get an unseen query key `[QUERY_TOKEN, 45]` near the end where no value was ever stored!
    is_unseen = np.zeros((batch_size, seq_len), dtype=np.int64)
    for b in range(batch_size // 2, batch_size):
        inputs[b, seq_len - 2] = QUERY_TOKEN
        inputs[b, seq_len - 1] = 45  # outside trained key range [20..31]!
        is_unseen[b, seq_len - 1] = 1
        mask[b, seq_len - 1] = True

    with torch.no_grad():
        out = model(inputs)
        logits = out.logits.float()
        probs = torch.softmax(logits, dim=-1)
        pred_entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=-1)
        score_tensor = out.token_uncertainty.float() if out.token_uncertainty is not None else pred_entropy

    tail_scores = score_tensor[:, seq_len - 1].detach().cpu().numpy()
    tail_labels = is_unseen[:, seq_len - 1]
    entropy_tail = pred_entropy[:, seq_len - 1].detach().cpu().numpy()

    auroc_unc = compute_auroc(tail_labels, tail_scores)
    auroc_ent = compute_auroc(tail_labels, entropy_tail)
    std_s = float(np.std(tail_scores))
    std_l = float(np.std(tail_labels))
    pearson_r = float(np.corrcoef(tail_scores, tail_labels)[0, 1]) if std_s > 1e-8 and std_l > 1e-8 else 0.0
    return {
        "auroc_internal_unc": auroc_unc,
        "auroc_predictive_entropy": auroc_ent,
        "pearson_r_unc_unseen": pearson_r,
    }


class CorpusStream:
    """Read-only uint16 memmap loader for decontaminated multi-domain token streams."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Corpus manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.vocab_size = int(self.manifest["vocab"])
        self.train_arr = np.memmap(self.data_dir / "train.bin", dtype=np.uint16, mode="r")
        self.val_arrays: dict[str, np.ndarray] = {}
        self.bpt: dict[str, float] = {}
        for dom in ("web", "textbook", "math", "code"):
            self.val_arrays[dom] = np.memmap(self.data_dir / f"val_{dom}.bin", dtype=np.uint16, mode="r")
            self.bpt[dom] = float(self.manifest["domains"][dom]["val"]["bytes_per_token"])

    def sample_train_batch(
        self,
        batch_size: int,
        seq_len: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        max_start = len(self.train_arr) - (seq_len + 1)
        starts = rng.integers(0, max_start, size=batch_size, dtype=np.int64)
        buf = np.empty((batch_size, seq_len + 1), dtype=np.int64)
        for i, s in enumerate(starts):
            buf[i] = self.train_arr[s : s + seq_len + 1]
        x = torch.from_numpy(buf[:, :-1]).to(device=device, non_blocking=False)
        y = torch.from_numpy(buf[:, 1:]).to(device=device, non_blocking=False)
        return x, y

    def evaluate_validation_bpb(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        seq_len: int = 256,
        eval_tokens_per_domain: int = 131_072,
        batch_size: int = 16,
        use_bf16: bool = True,
        amp_dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        """Compute exact held-out Bits-Per-Byte across all 4 decontaminated validation shards."""
        model.eval()
        domain_bpb: dict[str, float] = {}
        domain_nats: dict[str, float] = {}
        ln2 = math.log(2.0)
        resolved_dtype = amp_dtype if amp_dtype is not None else torch.bfloat16
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=resolved_dtype)
            if use_bf16 and device.type in ("mps", "cuda")
            else torch.no_grad()
        )
        with torch.no_grad():
            for dom, arr in self.val_arrays.items():
                n_seqs = min(eval_tokens_per_domain // seq_len, (len(arr) - 1) // seq_len)
                total_loss = 0.0
                total_toks = 0
                for s_idx in range(0, n_seqs, batch_size):
                    e_idx = min(n_seqs, s_idx + batch_size)
                    cur_b = e_idx - s_idx
                    buf = np.empty((cur_b, seq_len + 1), dtype=np.int64)
                    for i in range(cur_b):
                        off = (s_idx + i) * seq_len
                        buf[i] = arr[off : off + seq_len + 1]
                    x = torch.from_numpy(buf[:, :-1]).to(device)
                    y = torch.from_numpy(buf[:, 1:]).to(device)
                    with amp_ctx:
                        out = model(x, targets=y)
                    n_tok = cur_b * seq_len
                    total_loss += float(out.ce_loss.item()) * n_tok
                    total_toks += n_tok
                mean_nats = total_loss / max(1, total_toks)
                domain_nats[dom] = mean_nats
                domain_bpb[dom] = mean_nats / (ln2 * self.bpt[dom])
        composite = float(sum(domain_bpb.values()) / len(domain_bpb))
        return {
            "domain_bpb": domain_bpb,
            "domain_nats": domain_nats,
            "composite_bpb": composite,
            "eval_tokens_per_domain": eval_tokens_per_domain,
        }
