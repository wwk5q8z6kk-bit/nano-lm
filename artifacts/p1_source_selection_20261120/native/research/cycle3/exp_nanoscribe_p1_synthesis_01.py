from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.cycle1.exp_pba_01 import _git_sha
from research.cycle1.primitives import (
    count_trainable_params,
    make_adamw_no_embed_decay,
    seed_everything,
    summarize_seeds,
)
from research.cycle2.exp_byte_vs_token_01 import BOS_CH, DeterministicBPETokenizer
from research.cycle2.exp_hybrid_interference_01 import HybridSeqModel, _match_hybrid_d_ff

torch.set_num_threads(1)

SLOTS = ("MEDICATION", "DOSE", "ALLERGY", "VITAL_BP")
SLOT_TO_ID = {s: i for i, s in enumerate(SLOTS)}
TURN_OPS = ("ASSERTS", "SUPERSEDES", "RETRACTS", "NOISE")
TURN_OP_TO_ID = {o: i for i, o in enumerate(TURN_OPS)}

# Strictly 100% synthetic non-PHI clinical lexicons (disjoint seen vs unseen open-vocab pools)
SEEN_VALUES: dict[str, tuple[str, ...]] = {
    "MEDICATION": ("ATORVASTATIN", "METFORMIN", "LISINOPRIL", "OMEPRAZOLE", "AMLODIPINE", "GABAPENTIN"),
    "DOSE": ("10MG", "20MG", "40MG", "50MG", "100MG", "500MG"),
    "ALLERGY": ("PENICILLIN", "SULFA", "ASPIRIN", "CODEINE", "LATEX", "SHELLFISH"),
    "VITAL_BP": ("118/76", "124/80", "132/84", "138/86", "112/70", "128/78"),
}
UNSEEN_VALUES: dict[str, tuple[str, ...]] = {
    "MEDICATION": ("EMPAGLIFLOZIN", "APIXABAN", "ROSUVASTATIN", "SITAGLIPTIN"),
    "DOSE": ("25MG", "75MG", "150MG", "300MG"),
    "ALLERGY": ("AMOXICILLIN", "CEPHALEXIN", "IODINE", "MORPHINE"),
    "VITAL_BP": ("142/88", "108/68", "146/92", "136/82"),
}


def verify_verbatim_span(transcript: str, quote: str) -> bool:
    """Exact transcript grounding verifier: accepts only non-empty verbatim substrings of transcript."""
    if not quote:
        return False
    return quote in transcript


def _extract_candidate_span(turn_text: str) -> str:
    """Extract the uppercase/numeric clinical value token from a turn (ignoring slot keywords)."""
    cands = re.findall(r"\b[A-Z0-9][A-Z0-9/]{1,}\b", turn_text)
    filtered = [c for c in cands if c not in SLOTS and c not in ("NOTE", "RX", "NO", "OK")]
    return filtered[-1] if filtered else ""


def _make_turn_text(
    rng: np.random.Generator,
    slot: str,
    op: str,
    val: str,
    noisy_paraphrase: bool,
) -> str:
    if op == "NOISE":
        return rng.choice(
            [
                "patient slept well morning comfort steady",
                "nursing checked hydration chart looks fine",
                "routine ambulatory walk without dizziness",
                "waiting for scheduled afternoon discharge review",
            ]
        ).item()

    if not noisy_paraphrase:
        if op == "ASSERTS":
            return f"{slot} is {val} ."
        if op == "SUPERSEDES":
            return f"{slot} updated to {val} ."
        return f"{slot} retracted clear ."

    # Paraphrased / conversational clinical dialogue that breaks strict canonical regexes
    if op == "ASSERTS":
        if slot == "MEDICATION":
            return f"doctor started daily therapy using {val} today ."
        if slot == "DOSE":
            return f"pharmacy confirmed strength of {val} daily ."
        if slot == "ALLERGY":
            return f"chart lists prior reaction to {val} agent ."
        return f"initial cuff reading resting came in at {val} ."
    if op == "SUPERSEDES":
        if slot == "MEDICATION":
            return f"clinician switched rx away from prior drug onto {val} now ."
        if slot == "DOSE":
            return f"scratch old strength change titration amount to {val} ."
        if slot == "ALLERGY":
            return f"corrected sensitivity record now confirmed as {val} ."
        return f"repeat cuff measurement stabilized at {val} ."
    # Paraphrased RETRACTS
    if slot == "MEDICATION":
        return "hold therapy discontinued rx altogether nothing active ."
    if slot == "DOSE":
        return "drop prior dosage instruction wait on hold zero ."
    if slot == "ALLERGY":
        return "false alarm strike previous reaction entry off list ."
    return "discard cuff artifact reading invalid measurement ."


def _generate_encounters(
    rng: np.random.Generator,
    n_encounters: int,
    unseen_mode: bool,
) -> list[dict[str, Any]]:
    pool = UNSEEN_VALUES if unseen_mode else SEEN_VALUES
    encounters: list[dict[str, Any]] = []

    for _ in range(n_encounters):
        # Choose 2 or 3 active slots out of 4 so at least 1 slot remains genuinely ABSENT
        n_active = int(rng.integers(2, 4))
        active_slots = list(rng.choice(SLOTS, size=n_active, replace=False))
        gold_state: dict[str, str] = {s: "ABSTAIN" for s in SLOTS}
        was_revised: dict[str, bool] = {s: False for s in SLOTS}
        was_retracted: dict[str, bool] = {s: False for s in SLOTS}
        turns: list[dict[str, Any]] = []

        # Initial assertions for active slots
        for sl in active_slots:
            val = str(rng.choice(pool[sl]))
            noisy = bool(rng.random() < 0.55)
            txt = _make_turn_text(rng, sl, "ASSERTS", val, noisy)
            gold_state[sl] = val
            turns.append({"text": txt, "slot": sl, "op": "ASSERTS", "span": val, "noisy": noisy})

        # Add 1 distractor turn + 2 revision/retraction updates
        turns.append(
            {
                "text": _make_turn_text(rng, "MEDICATION", "NOISE", "", False),
                "slot": "MEDICATION",
                "op": "NOISE",
                "span": "",
                "noisy": False,
            }
        )
        for _ in range(2):
            sl = str(rng.choice(active_slots))
            r = float(rng.random())
            if gold_state[sl] != "ABSTAIN" and r < 0.62:
                op = "SUPERSEDES"
                candidates = [v for v in pool[sl] if v != gold_state[sl]]
                nv = str(rng.choice(candidates))
                gold_state[sl] = nv
                was_revised[sl] = True
                val_arg = nv
            elif gold_state[sl] != "ABSTAIN" and r < 0.90:
                op = "RETRACTS"
                gold_state[sl] = "ABSTAIN"
                was_revised[sl] = True
                was_retracted[sl] = True
                val_arg = ""
            else:
                op = "NOISE"
                val_arg = ""
            noisy = bool(rng.random() < 0.65)
            turns.append(
                {
                    "text": _make_turn_text(rng, sl, op, val_arg, noisy),
                    "slot": sl,
                    "op": op,
                    "span": val_arg,
                    "noisy": noisy,
                }
            )

        transcript = " ".join(t["text"] for t in turns)
        encounters.append(
            {
                "transcript": transcript,
                "turns": turns,
                "gold_state": gold_state,
                "was_revised": was_revised,
                "was_retracted": was_retracted,
            }
        )
    return encounters


EncounterSplit = list[dict[str, Any]]


def generate_benchmark_splits(seed: int) -> tuple[EncounterSplit, EncounterSplit, EncounterSplit]:
    """Reconstruct the original train, seen-test, and unseen-test encounter sequence."""
    rng = np.random.default_rng(110_000 + seed)
    return (
        _generate_encounters(rng, n_encounters=160, unseen_mode=False),
        _generate_encounters(rng, n_encounters=60, unseen_mode=False),
        _generate_encounters(rng, n_encounters=60, unseen_mode=True),
    )


def _deterministic_regex_ledger(enc: dict[str, Any]) -> dict[str, str]:
    """Tier 3 0-param canonical regex baseline."""
    state = {s: "ABSTAIN" for s in SLOTS}
    for turn in enc["turns"]:
        txt = turn["text"]
        m_set = re.match(r"^(MEDICATION|DOSE|ALLERGY|VITAL_BP)\s+(?:is|updated to)\s+([A-Z0-9/]+)\s*\.", txt)
        if m_set:
            state[m_set.group(1)] = m_set.group(2)
            continue
        m_ret = re.match(r"^(MEDICATION|DOSE|ALLERGY|VITAL_BP)\s+retracted\s+clear\s*\.", txt)
        if m_ret:
            state[m_ret.group(1)] = "ABSTAIN"
    return state


class TurnPerceptionNet(nn.Module):
    """Lightweight learned turn classifier (Byte-level GRU) predicting (slot_id, op_id) without regex rules."""

    def __init__(self, hidden: int = 48) -> None:
        super().__init__()
        self.emb = nn.Embedding(128, 24)
        self.gru = nn.GRU(24, hidden, batch_first=True)
        self.slot_head = nn.Linear(hidden, len(SLOTS))
        self.op_head = nn.Linear(hidden, len(TURN_OPS))

    def forward(self, x_ids: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_seq, _ = self.gru(self.emb(x_ids))
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h_seq.shape[-1])
        h_last = h_seq.gather(1, idx).squeeze(1)
        return self.slot_head(h_last), self.op_head(h_last)


def _encode_turn_strings(texts: list[str], max_len: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    bs = len(texts)
    ids = torch.zeros((bs, max_len), dtype=torch.long)
    lens = torch.ones(bs, dtype=torch.long)
    for i, s in enumerate(texts):
        arr = [min(127, ord(c)) for c in s[:max_len]]
        ids[i, : len(arr)] = torch.tensor(arr, dtype=torch.long)
        lens[i] = max(1, len(arr))
    return ids, lens


def _score_predictions(
    encounters: list[dict[str, Any]],
    preds_by_enc: list[dict[str, str]],
) -> dict[str, float]:
    n_total = 0
    exact_ok = 0
    rev_total = 0
    rev_ok = 0
    ret_total = 0
    stale_poison = 0
    tp_abst = fp_abst = fn_abst = 0
    crit_halluc = 0

    for enc, pred_map in zip(encounters, preds_by_enc):
        gold = enc["gold_state"]
        for sl in SLOTS:
            n_total += 1
            g = gold[sl]
            p = pred_map.get(sl, "ABSTAIN")
            if p == g:
                exact_ok += 1
            if enc["was_revised"][sl]:
                rev_total += 1
                if p == g:
                    rev_ok += 1
            if enc["was_retracted"][sl]:
                ret_total += 1
                if p != "ABSTAIN":
                    stale_poison += 1
            if g == "ABSTAIN":
                if p == "ABSTAIN":
                    tp_abst += 1
                else:
                    fn_abst += 1
                    crit_halluc += 1
            else:
                if p == "ABSTAIN":
                    fp_abst += 1

    f1 = (2.0 * tp_abst) / max(1, 2 * tp_abst + fp_abst + fn_abst)
    return {
        "exact_acc": float(exact_ok / n_total),
        "revision_update_acc": float(rev_ok / max(1, rev_total)),
        "stale_retraction_poison_rate": float(stale_poison / max(1, ret_total)),
        "abstention_f1": float(f1),
        "critical_hallucination_rate": float(crit_halluc / n_total),
    }


def _train_eval_seq_model(
    seed: int,
    mixer_kinds: list[str],
    pointer_mode: str,
    tok: DeterministicBPETokenizer,
    tr_encs: list[dict[str, Any]],
    ev_seen: list[dict[str, Any]],
    ev_unseen: list[dict[str, Any]],
    ref_non_emb: int,
    steps: int = 240,
) -> dict[str, float]:
    seed_everything(seed)
    rng = np.random.default_rng(91_000 + seed)
    dff = _match_hybrid_d_ff(ref_non_emb, tok.vocab_size, 48, 4, mixer_kinds, pointer_mode)
    model = HybridSeqModel(tok.vocab_size, 48, 4, dff, mixer_kinds=mixer_kinds, pointer_mode=pointer_mode)
    opt = make_adamw_no_embed_decay(model, lr=4e-3, weight_decay=0.01)

    # Pre-encode training examples as (prompt_ids, answer_ids) per (encounter, slot)
    tr_seqs: list[list[int]] = []
    for enc in tr_encs:
        tx = enc["transcript"]
        for sl in SLOTS:
            ans = enc["gold_state"][sl]
            full = f"{BOS_CH}{tx} QRY {sl} -> {ans}\n"
            tr_seqs.append(tok.encode(full))

    model.train()
    for _ in range(steps):
        idx = rng.integers(0, len(tr_seqs), size=24)
        batch = [tr_seqs[int(i)] for i in idx]
        max_len = max(len(s) for s in batch)
        x_in = torch.zeros((len(batch), max_len - 1), dtype=torch.long)
        y_tgt = torch.full((len(batch), max_len - 1), -100, dtype=torch.long)
        for bi, s in enumerate(batch):
            n = len(s) - 1
            x_in[bi, :n] = torch.tensor(s[:n], dtype=torch.long)
            y_tgt[bi, :n] = torch.tensor(s[1:], dtype=torch.long)
        lp = model(x_in)
        loss = F.nll_loss(lp.reshape(-1, tok.vocab_size), y_tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    nl_id = ord("\n")

    def eval_split(encs: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
        t_start = time.perf_counter()
        preds: list[dict[str, str]] = [{s: "ABSTAIN" for s in SLOTS} for _ in encs]
        # Group queries by prompt token length for fast exact batched greedy decoding
        items_by_len: dict[int, list[tuple[int, str, list[int]]]] = {}
        for ei, enc in enumerate(encs):
            tx = enc["transcript"]
            for sl in SLOTS:
                pids = tok.encode(f"{BOS_CH}{tx} QRY {sl} -> ")
                items_by_len.setdefault(len(pids), []).append((ei, sl, pids))

        with torch.no_grad():
            for _, group in items_by_len.items():
                cur = torch.tensor([g[2] for g in group], dtype=torch.long)
                bs = cur.shape[0]
                gen: list[list[int]] = [[] for _ in range(bs)]
                done = [False] * bs
                for _ in range(16):
                    nxt = model(cur)[:, -1, :].argmax(dim=-1)
                    cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
                    for bi in range(bs):
                        if not done[bi]:
                            tid = int(nxt[bi].item())
                            dec = tok.decode([tid])
                            if tid == nl_id or ("\n" in dec):
                                pre = dec.split("\n", 1)[0]
                                if pre:
                                    gen[bi].extend(tok.encode(pre))
                                done[bi] = True
                            else:
                                gen[bi].append(tid)
                    if all(done):
                        break
                for bi, (ei, sl, _) in enumerate(group):
                    preds[ei][sl] = tok.decode(gen[bi]).strip()
        ms_per_enc = ((time.perf_counter() - t_start) * 1000.0) / len(encs)
        return _score_predictions(encs, preds), ms_per_enc

    res_seen, ms_s = eval_split(ev_seen)
    res_unseen, _ = eval_split(ev_unseen)
    return {
        "exact_seen_span_acc": res_seen["exact_acc"],
        "unseen_open_vocab_span_acc": res_unseen["exact_acc"],
        "revision_update_acc": res_unseen["revision_update_acc"],
        "stale_retraction_poison_rate": res_unseen["stale_retraction_poison_rate"],
        "abstention_f1": res_unseen["abstention_f1"],
        "critical_hallucination_rate": res_unseen["critical_hallucination_rate"],
        "ms_per_encounter": ms_s,
    }


def _train_eval_ledger_tiers(
    seed: int,
    tr_encs: list[dict[str, Any]],
    ev_seen: list[dict[str, Any]],
    ev_unseen: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float], int]:
    seed_everything(seed + 500)
    rng = np.random.default_rng(95_000 + seed)
    turn_texts = [t["text"] for enc in tr_encs for t in enc["turns"]]
    turn_slots = [SLOT_TO_ID[t["slot"]] for enc in tr_encs for t in enc["turns"]]
    turn_ops = [TURN_OP_TO_ID[t["op"]] for enc in tr_encs for t in enc["turns"]]

    x_tr, l_tr = _encode_turn_strings(turn_texts)
    y_sl = torch.tensor(turn_slots, dtype=torch.long)
    y_op = torch.tensor(turn_ops, dtype=torch.long)

    perc = TurnPerceptionNet(hidden=48)
    opt = torch.optim.AdamW(perc.parameters(), lr=4e-3)
    perc.train()
    for _ in range(280):
        idx = torch.from_numpy(rng.integers(0, len(turn_texts), size=48))
        sl_log, op_log = perc(x_tr[idx], l_tr[idx])
        loss = F.cross_entropy(sl_log, y_sl[idx]) + F.cross_entropy(op_log, y_op[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    perc.eval()

    def run_split(encs: list[dict[str, Any]], complete_verified: bool) -> tuple[dict[str, float], float]:
        t0 = time.perf_counter()
        all_texts = [t["text"] for enc in encs for t in enc["turns"]]
        x_ev, l_ev = _encode_turn_strings(all_texts)
        with torch.no_grad():
            sl_log, op_log = perc(x_ev, l_ev)
            sl_probs = F.softmax(sl_log, dim=-1).numpy()
            op_probs = F.softmax(op_log, dim=-1).numpy()

        preds: list[dict[str, str]] = []
        cursor = 0
        for enc in encs:
            state = {s: "ABSTAIN" for s in SLOTS}
            slot_conf = {s: 1.0 for s in SLOTS}
            tx = enc["transcript"]
            for turn in enc["turns"]:
                sp = sl_probs[cursor]
                op_p = op_probs[cursor]
                cursor += 1
                sl_idx = int(np.argmax(sp))
                op_idx = int(np.argmax(op_p))
                conf = float(sp[sl_idx] * op_p[op_idx])
                sl_name = SLOTS[sl_idx]
                op_name = TURN_OPS[op_idx]

                if op_name in ("ASSERTS", "SUPERSEDES"):
                    cand = _extract_candidate_span(turn["text"])
                    if cand:
                        state[sl_name] = cand
                        slot_conf[sl_name] = conf
                elif op_name == "RETRACTS":
                    # Without VoC confidence gate, any predicted RETRACTS updates state directly
                    if (not complete_verified) or conf >= 0.55:
                        state[sl_name] = "ABSTAIN"
                        slot_conf[sl_name] = conf

            if complete_verified:
                # Section 8 Tier 3: Expected-VoC confidence gate + exact-quote grounding verifier
                for sl_name in SLOTS:
                    q = state[sl_name]
                    if q != "ABSTAIN":
                        if slot_conf[sl_name] < 0.65 or not verify_verbatim_span(tx, q):
                            state[sl_name] = "ABSTAIN"
            preds.append(state)
        ms = ((time.perf_counter() - t0) * 1000.0) / len(encs)
        return _score_predictions(encs, preds), ms

    n3_s, ms3 = run_split(ev_seen, complete_verified=False)
    n3_u, _ = run_split(ev_unseen, complete_verified=False)
    n4_s, ms4 = run_split(ev_seen, complete_verified=True)
    n4_u, _ = run_split(ev_unseen, complete_verified=True)

    res_n3 = {
        "exact_seen_span_acc": n3_s["exact_acc"],
        "unseen_open_vocab_span_acc": n3_u["exact_acc"],
        "revision_update_acc": n3_u["revision_update_acc"],
        "stale_retraction_poison_rate": n3_u["stale_retraction_poison_rate"],
        "abstention_f1": n3_u["abstention_f1"],
        "critical_hallucination_rate": n3_u["critical_hallucination_rate"],
        "ms_per_encounter": ms3,
    }
    res_n4 = {
        "exact_seen_span_acc": n4_s["exact_acc"],
        "unseen_open_vocab_span_acc": n4_u["exact_acc"],
        "revision_update_acc": n4_u["revision_update_acc"],
        "stale_retraction_poison_rate": n4_u["stale_retraction_poison_rate"],
        "abstention_f1": n4_u["abstention_f1"],
        "critical_hallucination_rate": n4_u["critical_hallucination_rate"],
        "ms_per_encounter": ms4,
    }
    return res_n3, res_n4, count_trainable_params(perc)


def main() -> None:
    t0 = time.perf_counter()
    seeds = [0, 1, 2]
    wt_root = Path(__file__).resolve().parents[2]

    arms = [
        "N0_Native_PureAttn",
        "N1_Native_PureHDCD",
        "N2_Native_Hybrid_HDCD_Attn_Ptr",
        "N3_HybridPtr_Plus_RevisionLedger",
        "N4_Complete_NanoScribe_VNext",
        "N5_Deterministic_Regex_Baseline",
    ]
    raw: dict[str, list[dict[str, float]]] = {a: [] for a in arms}
    arm_meta: dict[str, Any] = {}

    for seed in seeds:
        tr_encs, ev_seen, ev_unseen = generate_benchmark_splits(seed)

        # Protected-span hybrid BPE fit strictly on training transcripts + queries
        fit_texts = [f"{BOS_CH}{e['transcript']} QRY {sl} -> {e['gold_state'][sl]}\n" for e in tr_encs for sl in SLOTS]
        tok = DeterministicBPETokenizer(vocab_size=384, protect_alnum_spans=True)
        tok.fit(fit_texts)

        ref_m = HybridSeqModel(tok.vocab_size, 48, 4, 96, ["attn", "attn"], "none")
        ref_non_emb = count_trainable_params(ref_m) - ref_m.embed.weight.numel()

        if seed == 0:
            for a_name, mk, pm in (
                ("N0_Native_PureAttn", ["attn", "attn"], "none"),
                ("N1_Native_PureHDCD", ["hdcd", "hdcd"], "none"),
                ("N2_Native_Hybrid_HDCD_Attn_Ptr", ["hdcd", "attn"], "standard"),
            ):
                dff = _match_hybrid_d_ff(ref_non_emb, tok.vocab_size, 48, 4, mk, pm)
                pm_mod = HybridSeqModel(tok.vocab_size, 48, 4, dff, mk, pm)
                arm_meta[a_name] = {
                    "tier": "Tier_1_Native_Learned_Only",
                    "trainable_params": count_trainable_params(pm_mod),
                    "non_embedding_params": count_trainable_params(pm_mod) - pm_mod.embed.weight.numel(),
                }

        raw["N0_Native_PureAttn"].append(
            _train_eval_seq_model(seed, ["attn", "attn"], "none", tok, tr_encs, ev_seen, ev_unseen, ref_non_emb)
        )
        raw["N1_Native_PureHDCD"].append(
            _train_eval_seq_model(seed + 10, ["hdcd", "hdcd"], "none", tok, tr_encs, ev_seen, ev_unseen, ref_non_emb)
        )
        raw["N2_Native_Hybrid_HDCD_Attn_Ptr"].append(
            _train_eval_seq_model(seed + 20, ["hdcd", "attn"], "standard", tok, tr_encs, ev_seen, ev_unseen, ref_non_emb)
        )

        n3, n4, perc_p = _train_eval_ledger_tiers(seed, tr_encs, ev_seen, ev_unseen)
        raw["N3_HybridPtr_Plus_RevisionLedger"].append(n3)
        raw["N4_Complete_NanoScribe_VNext"].append(n4)
        if seed == 0:
            arm_meta["N3_HybridPtr_Plus_RevisionLedger"] = {"tier": "Tier_2_Model_Plus_Bounded_Memory", "trainable_params": perc_p}
            arm_meta["N4_Complete_NanoScribe_VNext"] = {"tier": "Tier_3_Complete_Verified_System", "trainable_params": perc_p}
            arm_meta["N5_Deterministic_Regex_Baseline"] = {"tier": "Tier_3_Deterministic_Baseline", "trainable_params": 0}

        # N5 Deterministic Regex Baseline
        t_r = time.perf_counter()
        n5_s = _score_predictions(ev_seen, [_deterministic_regex_ledger(e) for e in ev_seen])
        n5_u = _score_predictions(ev_unseen, [_deterministic_regex_ledger(e) for e in ev_unseen])
        ms5 = ((time.perf_counter() - t_r) * 1000.0) / (len(ev_seen) + len(ev_unseen))
        raw["N5_Deterministic_Regex_Baseline"].append(
            {
                "exact_seen_span_acc": n5_s["exact_acc"],
                "unseen_open_vocab_span_acc": n5_u["exact_acc"],
                "revision_update_acc": n5_u["revision_update_acc"],
                "stale_retraction_poison_rate": n5_u["stale_retraction_poison_rate"],
                "abstention_f1": n5_u["abstention_f1"],
                "critical_hallucination_rate": n5_u["critical_hallucination_rate"],
                "ms_per_encounter": ms5,
            }
        )

    m_keys = list(raw[arms[0]][0].keys())
    summary = {arm: {k: summarize_seeds([r[k] for r in rows]) for k in m_keys} for arm, rows in raw.items()}

    for arm in arms:
        s = summary[arm]
        print(
            f"[NANOSCRIBE] {arm:32s} | Seen={s['exact_seen_span_acc']['mean']:.3f} Unseen={s['unseen_open_vocab_span_acc']['mean']:.3f} | "
            f"Rev={s['revision_update_acc']['mean']:.3f} Poison={s['stale_retraction_poison_rate']['mean']:.3f} | "
            f"AbstF1={s['abstention_f1']['mean']:.3f} Halluc={s['critical_hallucination_rate']['mean']:.3f} ms={s['ms_per_encounter']['mean']:.2f}",
            flush=True,
        )

    n0_u = summary["N0_Native_PureAttn"]["unseen_open_vocab_span_acc"]["mean"]
    n2_u = summary["N2_Native_Hybrid_HDCD_Attn_Ptr"]["unseen_open_vocab_span_acc"]["mean"]
    n4_u = summary["N4_Complete_NanoScribe_VNext"]["unseen_open_vocab_span_acc"]["mean"]
    n5_u = summary["N5_Deterministic_Regex_Baseline"]["unseen_open_vocab_span_acc"]["mean"]
    n4_h = summary["N4_Complete_NanoScribe_VNext"]["critical_hallucination_rate"]["mean"]

    if n4_u > max(n5_u, n0_u) + 0.25 and n2_u > n0_u + 0.20 and n4_h <= 0.05:
        decision = "PROMOTE_TIERED_HYBRID_POINTER_AND_VERIFIED_LEDGER"
    elif n4_u > max(n5_u, n0_u) + 0.15:
        decision = "PROMOTE_COMPLETE_VERIFIED_SYSTEM_PARTIAL_NATIVE_COPY"
    else:
        decision = "CHECK_NANOSCRIBE_TIER_TRADEOFFS"

    elapsed = round(time.perf_counter() - t0, 3)
    payload = {
        "schema": "nanolm.cycle3.experiment.v1",
        "experiment_id": "NANOSCRIBE-P1-SYNTHESIS-01",
        "synthetic_non_phi_verified": True,
        "research_question": (
            "In 100% synthetic non-PHI multi-turn clinical scribing encounters with paraphrased mid-encounter "
            "supersessions/retractions, absent slots, and held-out unseen drug/dose/allergy/BP spans, how do Tier 1 "
            "(native Attn/HDCD/Hybrid+Ptr), Tier 2 (bounded revision ledger), Tier 3 (Expected-VoC + verbatim span "
            "verifier), and 0-param deterministic regex compare?"
        ),
        "hypothesis": (
            "1) Tier 1 N0/N1 without copy pointers fail unseen open-vocab spans; 2) Tier 1 N2 (["
            "'hdcd','attn']+Ptr) lifts native unseen span extraction dramatically; 3) Tier 2/3 revision ledger + "
            "Expected-VoC + verify_verbatim_span achieves top unseen accuracy & revision tracking with low critical "
            "hallucination, beating regex N5 on paraphrased turns."
        ),
        "competing_explanation": (
            "Regex N5 might match enough clean turns or native N0 might generalize partially without pointers or ledger."
        ),
        "code_revision": _git_sha(wt_root),
        "dataset_provenance": (
            "100% synthetic non-PHI multi-turn encounters (zero real patient data): 160 train / 60 seen-eval / 60 unseen-eval "
            "per seed across [0, 1, 2], tokenized via DeterministicBPETokenizer(vocab=384, protect_alnum_spans=True)"
        ),
        "arms": arm_meta,
        "metrics": m_keys,
        "success_criteria": "N4 beats N5 regex and N0 native on unseen_open_vocab_span_acc by >= +0.25 with critical_hallucination <= 0.05, and Tier 1 N2 beats N0 on unseen spans by >= +0.20",
        "failure_criteria": "Tier 1 N2 fails unseen copy OR Tier 3 N4 fails to beat regex N5",
        "seeds": seeds,
        "raw_per_seed": raw,
        "results": {
            "summary": summary,
            "section8_tier_attribution": {
                "Tier_1_Native_Only": {
                    "N0_Native_PureAttn": summary["N0_Native_PureAttn"],
                    "N1_Native_PureHDCD": summary["N1_Native_PureHDCD"],
                    "N2_Native_Hybrid_HDCD_Attn_Ptr": summary["N2_Native_Hybrid_HDCD_Attn_Ptr"],
                },
                "Tier_2_Model_Plus_Bounded_Memory": {
                    "N3_HybridPtr_Plus_RevisionLedger": summary["N3_HybridPtr_Plus_RevisionLedger"],
                },
                "Tier_3_Complete_Verified_System_And_Regex": {
                    "N4_Complete_NanoScribe_VNext": summary["N4_Complete_NanoScribe_VNext"],
                    "N5_Deterministic_Regex_Baseline": summary["N5_Deterministic_Regex_Baseline"],
                },
            },
        },
        "resource_consumption": {"paid_compute_usd": 0.0, "device": "cpu_float32", "wall_clock_seconds": elapsed},
        "independent_verification": {
            "n2_vs_n0_unseen_native_gain": round(n2_u - n0_u, 4),
            "n4_vs_n5_unseen_gain": round(n4_u - n5_u, 4),
            "n4_vs_n0_unseen_gain": round(n4_u - n0_u, 4),
            "n4_critical_hallucination_rate": round(n4_h, 4),
        },
        "final_decision": decision,
    }
    out = wt_root / "artifacts" / "cycle3" / "NANOSCRIBE-P1-SYNTHESIS-01.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"NANOSCRIBE-P1-SYNTHESIS-01 complete in {elapsed}s -> {out} | Decision: {decision}")


if __name__ == "__main__":
    main()
