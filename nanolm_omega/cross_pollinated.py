"""Cross-Pollinated Primitives (`OMEGA-EXP-04` / `G5_CROSS_POLLINATED`):

Synthesizes verified multi-agent discoveries across the NanoLM research program into
`nanolm_omega` with strict mathematical invariants:
  1. `StatefulSyncFreeDualCopyHead`:
     - Dual-offset bounded-cosine + 1/2-gram exact-identity copy (`v3-exact-recall` `DA_dual_idcopy`),
     - Sync-free (`SI-001`, zero `.item()` calls) + `1e-12` subnormal-safe log-mixture (`nanolm-autolab` `SI-004`),
     - Explicit `_no_autocast_ctx` float32 guard (`nanolm-v3` Apple M3 Max `bfloat16` Metal fix),
     - Causal relative-lag recency `-(p - j) / W` (strictly independent of total sequence length `T`,
       yielding `0.0` prefix drift and exact `O(W)` ring-buffer equivalence across `forward()`,
       `forward_chunked()`, and single-token `forward_step()`!).
  2. `sample_decontaminated_open_vocab_kv_batch`:
     - Enforces `replace=False` query key sampling (`EXACT-RECALL-03b` leak prevention) and disjoint
       seen `[key_hi .. vocab_seen_hi)` vs unseen `[vocab_seen_hi .. vocab_size)` value vocabularies.
  3. `audit_identifier_and_skeleton_leakage`:
     - Beyond-13-gram distinctive-identifier + 6-word template-skeleton (`<id>`, `<n>`) leakage audit
       (cross-pollinated from peer commit `9408534`).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nanolm_discovery.operators import DiscoveryRMSNorm, _no_autocast_ctx

_STOP_IDENTIFIERS = frozenset({
    "final_answer", "list", "dict", "tuple", "str", "int", "float", "bool", "none", "true", "false",
    "return", "python", "mermaid", "flowchart", "json", "items", "words", "text", "windows",
    "number", "name", "policy", "parameter_name", "chart_type", "x_label", "y_label", "x_values",
    "y_values", "line", "area", "self", "print",
})
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")
_BACKTICK_RE = re.compile(r"`([^`]+)`")


def compute_dual_copy_param_count(d_model: int, copy_dim: int) -> int:
    """Exact parameter count of `StatefulSyncFreeDualCopyHead(d_model, copy_dim)`."""
    # DiscoveryRMSNorm(d) + 4 * Linear(d, dc, bias=False) + Linear(d, 1, bias=True) + s_raw(2) + gamma1(1) + gamma2(1) + beta_rec(1)
    return 2 * d_model + 4 * d_model * copy_dim + 6


def compute_dual_copy_state_bytes_bf16(copy_dim: int, copy_window: int) -> int:
    """Exact persistent O(W) ring-buffer state footprint in bytes (`int64` IDs + `bf16` keys/query)."""
    w_keep = max(2, copy_window) + 1
    return w_keep * (8 + 4 * copy_dim) + 2 * copy_dim


class StatefulSyncFreeDualCopyHead(nn.Module):
    """Causal Dual-Offset Bounded-Cosine + 1/2-Gram Copy Head with Exact O(W) Ring-Buffer Duality."""

    def __init__(
        self,
        d_model: int,
        copy_dim: int = 32,
        copy_window: int = 256,
    ) -> None:
        super().__init__()
        if copy_dim <= 0 or copy_window < 2:
            raise ValueError(f"Invalid copy_dim={copy_dim} or copy_window={copy_window} (must be >= 2)")
        self.d_model = d_model
        self.copy_dim = copy_dim
        self.copy_window = copy_window

        self.norm = DiscoveryRMSNorm(d_model)
        self.w_qc0 = nn.Linear(d_model, copy_dim, bias=False)
        self.w_kc0 = nn.Linear(d_model, copy_dim, bias=False)
        self.w_qc1 = nn.Linear(d_model, copy_dim, bias=False)
        self.w_kc1 = nn.Linear(d_model, copy_dim, bias=False)
        self.gate = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.gate.bias, -1.5)

        self.s_raw = nn.Parameter(torch.tensor([3.98, 3.98]))
        self.gamma1 = nn.Parameter(torch.tensor(5.0))
        self.gamma2 = nn.Parameter(torch.tensor(3.5))
        self.beta_rec = nn.Parameter(torch.tensor(0.8))

    def forward_logprobs(
        self,
        hidden: Tensor,
        input_ids: Tensor,
        vocab_logits: Tensor,
        state: dict[str, Any] | None = None,
        *,
        return_state: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Any] | None]:
        """Mix `vocab_logits` `[B, T, V]` with causal source-token copy probabilities in log-space.

        Satisfies exact 3-way equivalence:
          `forward_logprobs(seq)` == `chunked` == `step-by-step` (`t_len = 1`).
        """
        bsz, t_len, vocab_size = vocab_logits.shape
        dev = hidden.device
        w_max = self.copy_window
        keep_cap = w_max + 1

        with _no_autocast_ctx(dev):
            p_gen = F.softmax(vocab_logits.float(), dim=-1)
            hf = self.norm(hidden).float()
            g_raw = torch.sigmoid(self.gate(hf))  # [B, T, 1]

            s0 = F.softplus(self.s_raw[0].float())
            s1 = F.softplus(self.s_raw[1].float())
            qc0 = F.normalize(self.w_qc0(hf), dim=-1, eps=1e-6)
            kc0 = F.normalize(self.w_kc0(hf), dim=-1, eps=1e-6)
            qc1 = F.normalize(self.w_qc1(hf), dim=-1, eps=1e-6)
            kc1 = F.normalize(self.w_kc1(hf), dim=-1, eps=1e-6)

            if state is None:
                step_offset = 0
                hist_len = 0
                all_ids = input_ids
                all_kc0 = kc0
                all_kc1 = kc1
                if t_len > 1:
                    qc1_lag = F.pad(qc1[:, :-1, :], (0, 0, 1, 0))
                    x_t_lag = F.pad(input_ids[:, :-1], (1, 0), value=-1)
                else:
                    qc1_lag = torch.zeros_like(qc1)
                    x_t_lag = torch.full_like(input_ids, -1)
            else:
                step_offset = int(state["step"])
                h_ids: Tensor = state["ids"]
                h_kc0: Tensor = state["kc0"].float()
                h_kc1: Tensor = state["kc1"].float()
                h_qc1_last: Tensor = state["qc1_last"].float()
                hist_len = int(h_ids.shape[1])

                all_ids = torch.cat([h_ids, input_ids], dim=1)
                all_kc0 = torch.cat([h_kc0, kc0], dim=1)
                all_kc1 = torch.cat([h_kc1, kc1], dim=1)
                if hist_len >= 1:
                    qc1_lag = torch.cat([h_qc1_last, qc1[:, :-1, :]], dim=1) if t_len > 1 else h_qc1_last
                    x_t_lag = torch.cat([h_ids[:, -1:], input_ids[:, :-1]], dim=1) if t_len > 1 else h_ids[:, -1:]
                else:
                    qc1_lag = F.pad(qc1[:, :-1, :], (0, 0, 1, 0)) if t_len > 1 else torch.zeros_like(qc1)
                    x_t_lag = F.pad(input_ids[:, :-1], (1, 0), value=-1) if t_len > 1 else torch.full_like(input_ids, -1)

            total_n = int(all_ids.shape[1])
            next_state: dict[str, Any] | None = None
            if return_state:
                k_keep = min(total_n, keep_cap)
                next_state = {
                    "ids": all_ids[:, -k_keep:].clone(),
                    "kc0": all_kc0[:, -k_keep:, :].to(hidden.dtype).clone(),
                    "kc1": all_kc1[:, -k_keep:, :].to(hidden.dtype).clone(),
                    "qc1_last": qc1[:, -1:, :].to(hidden.dtype).clone(),
                    "step": step_offset + t_len,
                }

            if total_n <= 2:
                return torch.log(p_gen.clamp_min(1e-12)), torch.zeros_like(g_raw), next_state

            m_cand = total_n - 2
            kc0_prev1 = all_kc0[:, 1:-1, :]  # [B, M, dc] -> index j - 1
            kc1_prev2 = all_kc1[:, :-2, :]   # [B, M, dc] -> index j - 2
            x_prev1 = all_ids[:, 1:-1]       # [B, M] -> x_{j-1}
            x_prev2 = all_ids[:, :-2]        # [B, M] -> x_{j-2}
            x_copy = all_ids[:, 2:]          # [B, M] -> x_j

            sem0 = s0 * torch.matmul(qc0, kc0_prev1.transpose(-1, -2))      # [B, T, M]
            sem1 = s1 * torch.matmul(qc1_lag, kc1_prev2.transpose(-1, -2))  # [B, T, M]

            eq1 = (input_ids.unsqueeze(2) == x_prev1.unsqueeze(1)).float()
            eq2 = (x_t_lag.unsqueeze(2) == x_prev2.unsqueeze(1)).float()

            p_idx = torch.arange(step_offset, step_offset + t_len, device=dev).unsqueeze(1)  # [T, 1]
            j_start = step_offset - hist_len + 2
            j_idx = torch.arange(j_start, j_start + m_cand, device=dev).unsqueeze(0)         # [1, M]
            lag = p_idx - j_idx                                                              # [T, M]
            mask = (lag >= 0) & (lag < w_max)                                                # [T, M]

            rel_decay = -self.beta_rec.float() * (lag.float().clamp(min=0.0) / float(w_max)).unsqueeze(0)
            scores = sem0 + sem1 + self.gamma1.float() * eq1 + self.gamma2.float() * eq2 + rel_decay

            masked_scores = scores.masked_fill(~mask.unsqueeze(0), -1e9)
            alpha = F.softmax(masked_scores, dim=-1) * mask.float().unsqueeze(0)

            p_copy = scores.new_zeros((bsz, t_len, vocab_size))
            p_copy.scatter_add_(2, x_copy.unsqueeze(1).expand(bsz, t_len, m_cand), alpha)

            has_ctx = mask.any(dim=-1, keepdim=True).float().unsqueeze(0)  # [1, T, 1]
            g_eff = g_raw * has_ctx
            p_mix = (1.0 - g_eff) * p_gen + g_eff * p_copy
            log_probs = torch.log(p_mix.clamp_min(1e-12))
            return log_probs, g_eff, next_state


def sample_decontaminated_open_vocab_kv_batch(
    batch_size: int,
    n_pairs: int,
    n_queries: int,
    *,
    vocab_size: int,
    seen_val_hi: int,
    unseen_mode: bool,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Generate strict `replace=False` (`EXACT-RECALL-03b`) associative key-value copy sequences.

    Format:
      `[KEY_0, VAL_0a, VAL_0b, ..., KEY_{N-1}, VAL_{N-1,a}, VAL_{N-1,b}, SEP, KEY_{q0}, VAL_{q0,a}, VAL_{q0,b}, ...]`
    Queries sample `n_queries <= n_pairs` distinct keys WITHOUT replacement (`replace=False`) so
    earlier queries never leak answers to later queries in the prompt prefix.
    Targets are `-1` everywhere except the two value positions of each query.
    """
    if n_queries > n_pairs:
        raise ValueError("n_queries must be <= n_pairs to enforce replace=False without query-prefix leakage")
    sep_tok = 1
    key_lo = 4
    n_keys_pool = max(n_pairs + 4, (seen_val_hi - key_lo) // 2)
    key_hi = key_lo + n_keys_pool
    val_seen_lo = key_hi
    if unseen_mode:
        val_lo, val_hi = seen_val_hi, vocab_size
    else:
        val_lo, val_hi = val_seen_lo, seen_val_hi
    if val_hi - val_lo < 4:
        raise ValueError(f"Insufficient value vocabulary range [{val_lo}, {val_hi}) for vocab_size={vocab_size}")

    seq_len = 3 * n_pairs + 1 + 3 * n_queries
    tokens = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    targets = torch.full((batch_size, seq_len), -1, dtype=torch.long, device=device)

    g_cpu = generator
    rand_keys = torch.argsort(torch.rand((batch_size, n_keys_pool), generator=g_cpu), dim=-1)[:, :n_pairs] + key_lo
    vals_a = torch.randint(val_lo, val_hi, (batch_size, n_pairs), generator=g_cpu)
    vals_b = torch.randint(val_lo, val_hi, (batch_size, n_pairs), generator=g_cpu)
    q_perm = torch.argsort(torch.rand((batch_size, n_pairs), generator=g_cpu), dim=-1)[:, :n_queries]

    rand_keys = rand_keys.to(device)
    vals_a = vals_a.to(device)
    vals_b = vals_b.to(device)
    q_perm = q_perm.to(device)

    ctx = torch.stack([rand_keys, vals_a, vals_b], dim=-1).reshape(batch_size, 3 * n_pairs)
    tokens[:, : 3 * n_pairs] = ctx
    tokens[:, 3 * n_pairs] = sep_tok

    q_keys = torch.gather(rand_keys, 1, q_perm)
    q_va = torch.gather(vals_a, 1, q_perm)
    q_vb = torch.gather(vals_b, 1, q_perm)
    qry_triplets = torch.stack([q_keys, q_va, q_vb], dim=-1).reshape(batch_size, 3 * n_queries)
    q_base = 3 * n_pairs + 1
    tokens[:, q_base:] = qry_triplets

    for qi in range(n_queries):
        pos_k = q_base + 3 * qi
        targets[:, pos_k] = q_va[:, qi]
        targets[:, pos_k + 1] = q_vb[:, qi]

    return tokens[:, :-1].contiguous(), targets[:, :-1].contiguous()


def _extract_distinctive_identifiers(text: str, common_words: set[str] | None = None) -> set[str]:
    found: set[str] = set()
    for span in _BACKTICK_RE.findall(text):
        for tok in _IDENT_RE.findall(span):
            t = tok.strip("-").lower()
            if len(t) >= 5:
                found.add(t)
    for tok in _IDENT_RE.findall(text):
        t = tok.strip("-")
        if ("_" in t and len(t) >= 6) or (t.isupper() and t.isalpha() and len(t) >= 4):
            found.add(t.lower())
    out = {t for t in found if t not in _STOP_IDENTIFIERS}
    if common_words:
        out = {t for t in out if not (t.isalpha() and t in common_words)}
    return out


def _skeletonize(text: str) -> list[str]:
    t = text.lower()
    t = _BACKTICK_RE.sub(" <id> ", t)
    t = re.sub(r"\d+(?:\.\d+)?", " <n> ", t)
    t = re.sub(r"\b[a-z0-9]*_[a-z0-9_]*\b", " <id> ", t)
    return re.findall(r"<id>|<n>|[a-z]+", t)


def _shingles(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)} if len(words) >= n else set()


def audit_identifier_and_skeleton_leakage(
    train_texts: Iterable[str],
    eval_texts: Iterable[str],
    *,
    common_words: set[str] | None = None,
    skeleton_n: int = 6,
) -> dict[str, Any]:
    """Audit train/eval overlap beyond 13-grams via distinctive identifiers and N-word template skeletons."""
    train_list = list(train_texts)
    eval_list = list(eval_texts)
    if common_words is None:
        word_counts: dict[str, int] = {}
        for doc in train_list:
            for w in re.findall(r"[a-z]+", doc.lower()):
                word_counts[w] = word_counts.get(w, 0) + 1
        common_words = {w for w, c in word_counts.items() if c >= 2}

    train_idents: set[str] = set()
    train_skeletons: set[tuple[str, ...]] = set()
    for doc in train_list:
        train_idents.update(_extract_distinctive_identifiers(doc, common_words))
        train_skeletons.update(_shingles(_skeletonize(doc), skeleton_n))

    ident_leaked_items = 0
    skeleton_overlap_fracs: list[float] = []
    max_skeleton_overlap = 0.0
    total_eval_idents = 0
    leaked_eval_idents = 0

    for item in eval_list:
        e_idents = _extract_distinctive_identifiers(item, common_words)
        total_eval_idents += len(e_idents)
        matched_idents = e_idents & train_idents
        if matched_idents:
            ident_leaked_items += 1
            leaked_eval_idents += len(matched_idents)
        e_skels = _shingles(_skeletonize(item), skeleton_n)
        frac = (len(e_skels & train_skeletons) / float(len(e_skels))) if e_skels else 0.0
        skeleton_overlap_fracs.append(frac)
        if frac > max_skeleton_overlap:
            max_skeleton_overlap = frac

    n_eval = max(1, len(eval_list))
    return {
        "n_train_docs": len(train_list),
        "n_eval_docs": len(eval_list),
        "skeleton_n": skeleton_n,
        "train_distinctive_identifiers": len(train_idents),
        "eval_distinctive_identifiers": total_eval_idents,
        "leaked_eval_identifiers": leaked_eval_idents,
        "eval_items_with_identifier_leak": ident_leaked_items,
        "identifier_leak_item_rate": ident_leaked_items / float(n_eval),
        "mean_skeleton_overlap": sum(skeleton_overlap_fracs) / float(n_eval),
        "max_skeleton_overlap": max_skeleton_overlap,
    }
