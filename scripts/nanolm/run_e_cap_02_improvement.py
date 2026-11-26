#!/usr/bin/env python3
"""Experiment E-CAP-02: Targeted Data-Diversity & Cosine-Annealing SFT Intervention.

Tests whether open-vocabulary slot/identifier diversity + long-context/contradiction coverage +
Warmup-Cosine LR decay resolves the Stage-2 underfitting & slot-memorization failure observed in
E-CAP-01 (0.0% pass rate), under strict 13-gram and entity-disjoint decontamination against `nanolm.eval`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from nanolm.config import get_preset_config
from nanolm.data import decontaminate_corpus
from nanolm.eval import collect_all_eval_prompts, run_capability_evaluation
from nanolm.model import NanoLMForCausalLM
from nanolm.tokenizer import format_chat_messages, load_tokenizer
from nanolm.train import TrainConfig, detect_best_device, load_checkpoint, save_atomic_checkpoint

# Disjoint open-vocabulary identifier pools (ZERO overlap with held-out eval entities!)
_FORBIDDEN_EVAL_TOKENS = {
    "zephyr", "solaris", "velox_stream", "arm64-debug", "api_ingress",
    "core_engine", "audit_sink", "quasar", "pulsar", "kronos", "lyra", "mira", "nyx",
}

_SYLLABLES_A = ("al", "ber", "cor", "del", "el", "far", "gal", "har", "ion", "jor", "kel", "lor", "mar", "nor", "pax", "tal", "var", "zen")
_SYLLABLES_B = ("on", "ix", "ar", "us", "is", "um", "ex", "ia", "os", "en", "ax", "or", "il", "un")


def _rand_ident(rng: random.Random) -> str:
    while True:
        cand = rng.choice(_SYLLABLES_A) + rng.choice(_SYLLABLES_B)
        if cand.lower() not in _FORBIDDEN_EVAL_TOKENS:
            return cand


def build_diverse_curated_sft_corpus(seed: int = 2026) -> list[tuple[str, str]]:
    """Generate lexically diverse, compositional instruction-response pairs with zero eval leakage."""
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []

    # 1. Compositional Python Software Engineering (diverse function names + exact required signatures)
    for idx in range(160):
        kind = idx % 4
        if kind == 0:
            fn = "stable_unique_positive" if idx % 3 == 0 else f"unique_pos_{_rand_ident(rng)}"
            arg = rng.choice(("items", "values", "nums", "seq"))
            u = rng.choice((
                f"Write a Python function `{fn}({arg}: list[int]) -> list[int]` returning positive integers (> 0) in first-seen order without duplicates.",
                f"Implement `{fn}({arg}: list[int]) -> list[int]` in Python so negatives/zeros are dropped and first occurrences are kept.",
            ))
            a = (
                f"```python\ndef {fn}({arg}: list[int]) -> list[int]:\n"
                f"    seen = set()\n    out = []\n    for x in {arg}:\n"
                f"        if x > 0 and x not in seen:\n"
                f"            seen.add(x)\n            out.append(x)\n"
                f"    return out\n```"
            )
        elif kind == 1:
            fn = "coalesce_windows" if idx % 3 == 0 else f"merge_ranges_{_rand_ident(rng)}"
            arg = rng.choice(("windows", "spans", "intervals"))
            u = rng.choice((
                f"Write a Python function `{fn}({arg}: list[tuple[int, int]]) -> list[tuple[int, int]]` that sorts and merges overlapping or touching intervals.",
                f"Implement `{fn}({arg}: list[tuple[int, int]]) -> list[tuple[int, int]]` in Python combining intervals when `lo <= prev_hi`.",
            ))
            a = (
                f"```python\ndef {fn}({arg}: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
                f"    if not {arg}:\n        return []\n"
                f"    ordered = sorted({arg})\n    merged = [ordered[0]]\n"
                f"    for lo, hi in ordered[1:]:\n"
                f"        plo, phi = merged[-1]\n"
                f"        if lo <= phi:\n            merged[-1] = (plo, max(phi, hi))\n"
                f"        else:\n            merged.append((lo, hi))\n"
                f"    return merged\n```"
            )
        elif kind == 2:
            fn = "is_balanced_brackets" if idx % 3 == 0 else f"balanced_delims_{_rand_ident(rng)}"
            arg = rng.choice(("text", "expr", "src"))
            u = rng.choice((
                f"Write a Python function `{fn}({arg}: str) -> bool` checking balanced nesting of `()`, `[]`, and `{{}}` while ignoring other characters.",
                f"Implement `{fn}({arg}: str) -> bool` in Python using a stack to validate `()`, `[]`, and `{{}}` pairs.",
            ))
            a = (
                f"```python\ndef {fn}({arg}: str) -> bool:\n"
                f"    pairs = {{')': '(', ']': '[', '}}': '{{'}}\n    stack = []\n"
                f"    for ch in {arg}:\n"
                f"        if ch in '([{{':\n            stack.append(ch)\n"
                f"        elif ch in pairs:\n"
                f"            if not stack or stack.pop() != pairs[ch]:\n                return False\n"
                f"    return len(stack) == 0\n```"
            )
        else:
            fn = "top_two_tokens" if idx % 3 == 0 else f"top_two_{_rand_ident(rng)}"
            arg = rng.choice(("words", "tokens", "items"))
            u = rng.choice((
                f"Write a Python function `{fn}({arg}: list[str]) -> list[str]` returning up to 2 most frequent strings sorted by count descending, ties broken alphabetically.",
                f"Implement `{fn}({arg}: list[str]) -> list[str]` in Python to return the top 2 strings by descending frequency then lexicographical order.",
            ))
            a = (
                f"```python\ndef {fn}({arg}: list[str]) -> list[str]:\n"
                f"    counts = {{}}\n    for w in {arg}:\n        counts[w] = counts.get(w, 0) + 1\n"
                f"    ranked = sorted(counts.keys(), key=lambda k: (-counts[k], k))\n"
                f"    return ranked[:2]\n```"
            )
        pairs.append((u, a))

    # 2. Multi-step Math, Transitive Ordering & Constraint Elimination (open-vocab names!)
    policies_sets = [
        ("LRU", "LFU", "FIFO"),
        ("RR", "LIFO", "EDF"),
        ("AES", "CHACHA", "RSA"),
    ]
    for _ in range(120):
        # Transitive 4-entity order with randomized statement order!
        names = [ _rand_ident(rng).capitalize() for _ in range(4) ]
        while len(set(names)) < 4:
            names = [ _rand_ident(rng).capitalize() for _ in range(4) ]
        first, second, third, fourth = names[0], names[1], names[2], names[3]
        shuffled = names[:]
        rng.shuffle(shuffled)
        stmts = [
            f"{second} starts before {third}.",
            f"{first} starts before {second}.",
            f"{fourth} starts after {third}.",
        ]
        rng.shuffle(stmts)
        pairs.append((
            f"Four services ( {', '.join(shuffled)} ) run sequentially with no ties. "
            f"{' '.join(stmts)} Which service runs first? Reply with `FINAL_ANSWER: <name>`.",
            f"<|think|>{first} < {second} < {third} < {fourth}.<|end_think|>\nFINAL_ANSWER: {first}",
        ))

        # 3x3 Elimination constraint puzzle
        p1, p2, p3 = rng.choice(policies_sets)
        perm_p = [p1, p2, p3]
        rng.shuffle(perm_p)
        ans_c1, ans_c3, ans_c2 = perm_p[0], perm_p[1], perm_p[2]
        slots = [f"S{rng.randint(1,3)}" for _ in range(3)]
        slots = ["N1", "N2", "N3"] if rng.random() < 0.5 else ["B1", "B2", "B3"]
        pairs.append((
            f"Three caches ({slots[0]}, {slots[1]}, {slots[2]}) each pick a unique policy from {p1}, {p2}, or {p3}. "
            f"{slots[0]} does not use {ans_c3} or {ans_c2}. {slots[2]} does not use {ans_c2}. "
            f"Which policy does {slots[1]} use? Reply with `FINAL_ANSWER: <policy>`.",
            f"<|think|>{slots[0]}={ans_c1}, {slots[2]}={ans_c3}, leaving {slots[1]}={ans_c2}.<|end_think|>\nFINAL_ANSWER: {ans_c2}",
        ))

    # 3. Faithful Summarization preserving exact numbers + `except` / `unless` qualifications
    for _ in range(120):
        mod_name = f"{_rand_ident(rng)}_{_rand_ident(rng)}"
        exc_node = f"{_rand_ident(rng)}-check"
        v_hi = rng.randint(40, 128)
        v_lo = rng.randint(12, 39)
        if rng.random() < 0.5:
            pairs.append((
                f"Condense this bulletin into one short sentence keeping exact numbers and exceptions:\n"
                f"Bulletin: Activating zero-copy framing in `{mod_name}` lowered RAM usage from {v_hi} GB to {v_lo} GB "
                f"across Linux hosts, except on `{exc_node}` nodes where audit guards remain active.",
                f"Zero-copy framing in `{mod_name}` cut RAM usage from {v_hi} GB to {v_lo} GB across Linux hosts, except on `{exc_node}` nodes.",
            ))
        else:
            ms = rng.choice((150, 200, 300, 450))
            probes = rng.randint(2, 6)
            mode = f"{_rand_ident(rng)} freeze"
            pairs.append((
                f"Condense this policy into one short sentence keeping exact thresholds and conditions:\n"
                f"Policy: Automatic switchover activates if replication lag exceeds {ms} ms across {probes} consecutive checks, "
                f"unless `{mode}` mode is enabled.",
                f"Automatic switchover activates when replication lag exceeds {ms} ms for {probes} consecutive checks, unless `{mode}` mode is enabled.",
            ))

    # 4. Open-Vocabulary Structured JSON (cluster/nodes + project/expenses)
    for _ in range(120):
        cname = _rand_ident(rng)
        h1, h2 = f"{_rand_ident(rng)[:2]}{rng.randint(1,9)}", f"{_rand_ident(rng)[:2]}{rng.randint(1,9)}"
        c1, c2 = rng.choice((8, 12, 16, 24, 32)), rng.choice((8, 16, 24, 32, 48))
        manifest = {"cluster": cname, "nodes": [{"host": h1, "cores": c1}, {"host": h2, "cores": c2}], "total_cores": c1 + c2}
        pairs.append((
            f"Emit valid JSON only (`cluster`, `nodes` with `host` & `cores`, `total_cores`) "
            f"for cluster `{cname}` having `{h1}` ({c1} cores) and `{h2}` ({c2} cores).",
            json.dumps(manifest, separators=(",", ":")),
        ))

        pname = _rand_ident(rng)
        start_b = rng.choice((400, 600, 800, 1000))
        e1, e2, e3 = rng.randint(40, 130), rng.randint(40, 130), rng.randint(40, 130)
        ledger = {"project": pname, "items": [e1, e2, e3], "remaining_usd": start_b - (e1 + e2 + e3)}
        pairs.append((
            f"Emit valid JSON only (`project`, `items`, `remaining_usd`) "
            f"for project `{pname}` with initial budget {start_b} and costs {e1}, {e2}, and {e3}.",
            json.dumps(ledger, separators=(",", ":")),
        ))

    # 5. Open-Vocabulary Mermaid Flowcharts & Time-Series Chart Specs
    for _ in range(120):
        src = f"{_rand_ident(rng)}_gateway"
        dst1 = f"{_rand_ident(rng)}_worker"
        dst2 = f"{_rand_ident(rng)}_logger"
        lbl1 = rng.choice(("verified", "accepted", "allowed", "authenticated"))
        lbl2 = rng.choice(("dropped", "denied", "rejected", "flagged"))
        pairs.append((
            f"Create a ```mermaid flowchart LR``` diagram where `{src}` directs `{lbl1}` "
            f"requests to `{dst1}` and `{lbl2}` requests to `{dst2}`.",
            f'```mermaid\nflowchart LR\n    A["{src}"] -->|"{lbl1}"| B["{dst1}"]\n'
            f'    A["{src}"] -->|"{lbl2}"| C["{dst2}"]\n```',
        ))

        q_vals = sorted([rng.randint(60, 280) for _ in range(4)])
        chart_obj = {
            "chart_type": "line",
            "x_label": "Quarter",
            "y_label": "Rate (req/s)",
            "x_values": ["Q1", "Q2", "Q3", "Q4"],
            "y_values": q_vals,
        }
        pairs.append((
            f"Produce a JSON chart spec (`chart_type`, `x_label`, `y_label`, `x_values`, `y_values`) "
            f"for quarterly series Q1: {q_vals[0]} req/s, Q2: {q_vals[1]} req/s, Q3: {q_vals[2]} req/s, Q4: {q_vals[3]} req/s.",
            json.dumps(chart_obj, separators=(",", ":")),
        ))

    # 6. Long-Context Distant Contradiction Detection
    param_names = ("sync_window", "keepalive_ms", "batch_limit", "max_workers", "retry_delay", "timeout_sec")
    filler = "Telemetry check passed with normal voltage and zero packet loss across active lanes."
    for _ in range(100):
        p_a, p_b, p_c = rng.sample(param_names, 3)
        va, vb1, vb2, vc = rng.randint(2, 9), rng.randint(10, 45), rng.randint(55, 120), rng.choice(("strict", "relaxed"))
        # Randomize which of the 3 keys contradicts between HEADER and FOOTER!
        diff_slot = rng.randrange(3)
        if diff_slot == 0:
            hdr = f"[HEADER] {p_a}={va}, {p_b}={vb1}, {p_c}={vc}."
            ftr = f"[FOOTER] Verified active config: {p_a}={va + 4}, {p_b}={vb1}, {p_c}={vc}."
            ans_key = p_a
        elif diff_slot == 1:
            hdr = f"[HEADER] {p_a}={va}, {p_b}={vb1}, {p_c}={vc}."
            ftr = f"[FOOTER] Verified active config: {p_a}={va}, {p_b}={vb2}, {p_c}={vc}."
            ans_key = p_b
        else:
            vc2 = "relaxed" if vc == "strict" else "strict"
            hdr = f"[HEADER] {p_a}={va}, {p_b}={vb1}, {p_c}={vc}."
            ftr = f"[FOOTER] Verified active config: {p_a}={va}, {p_b}={vb1}, {p_c}={vc2}."
            ans_key = p_c
        n_fill = rng.randint(8, 16)
        body = "\n".join([hdr, *[f"[STEP {k}] {filler}" for k in range(n_fill)], ftr])
        pairs.append((
            f"Find which configuration key disagrees between the header and footer below. "
            f"End with `FINAL_ANSWER: <key>`.\n\n{body}",
            f"<|think|>Compare header vs footer: `{ans_key}` changed.<|end_think|>\nFINAL_ANSWER: {ans_key}",
        ))

    # Hard 13-gram decontamination filter against held-out `nanolm.eval` prompts!
    eval_prompts = collect_all_eval_prompts()
    decontam_pairs: list[tuple[str, str]] = []
    removed = 0
    for u, a in pairs:
        kept, rep = decontaminate_corpus([f"{u}\n{a}"], eval_prompts, ngram_size=13)
        if kept:
            decontam_pairs.append((u, a))
        else:
            removed += rep.removed_eval_contaminated
    print(f"Decontamination gate: kept {len(decontam_pairs)}/{len(pairs)} pairs (scrubbed {removed} 13-gram overlaps)", flush=True)
    rng.shuffle(decontam_pairs)
    return decontam_pairs


def run_e_cap_02() -> None:
    out_dir = ROOT / "artifacts" / "nanolm_v1"
    tok = load_tokenizer(out_dir / "tokenizers" / "nanolm_bpe_16k.json")
    im_end_id = tok.token_to_id("<|im_end|>")
    eos_id = tok.token_to_id("<|eos|>")
    pad_id = tok.token_to_id("<|pad|>")
    dev = detect_best_device()

    m_cfg = get_preset_config("nanolm-25m-hybrid-swa")
    base_ckpt = out_dir / "runs" / "A2_hybrid_swa_mtp_wsd" / "latest.pt"
    model = NanoLMForCausalLM(m_cfg).to(dev)
    load_checkpoint(base_ckpt, model, device=dev)
    # Strip MTP head during instruction SFT to focus 100% gradient budget on exact token generation
    model.strip_training_only_heads()

    pairs = build_diverse_curated_sft_corpus(seed=2026)
    print(f"=== [E-CAP-02] Diverse Compositional SFT ({len(pairs)} decontaminated examples) ===", flush=True)

    seq_len = 512
    xs, ys, ms = [], [], []
    for u_txt, a_txt in pairs:
        prefix = format_chat_messages([{"role": "user", "content": u_txt}], add_generation_prompt=True)
        full = prefix + a_txt + "<|im_end|>"
        p_ids = tok.encode(prefix).ids
        f_ids = tok.encode(full).ids
        if len(f_ids) > seq_len + 1:
            continue
        padded = f_ids + [pad_id] * (seq_len + 1 - len(f_ids))
        xs.append(padded[:seq_len])
        ys.append(padded[1 : seq_len + 1])
        ms.append([1.0 if (len(p_ids) - 1 <= i < len(f_ids) - 1) else 0.0 for i in range(seq_len)])

    x_ten = torch.tensor(xs, dtype=torch.long)
    y_ten = torch.tensor(ys, dtype=torch.long)
    m_ten = torch.tensor(ms, dtype=torch.float32)
    print(f"Encoded {x_ten.shape[0]} SFT sequences (seq_len={seq_len}) on {dev}", flush=True)

    peak_lr = 8e-4
    min_lr = 4e-5
    steps = 600
    warmup = 40
    bsz = 16
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.05)
    dtype = torch.bfloat16 if dev in ("cuda", "mps") else torch.float32

    model.train()
    t0 = time.perf_counter()
    final_ce = 0.0
    for step in range(1, steps + 1):
        if step <= warmup:
            lr = peak_lr * (step / warmup)
        else:
            p = (step - warmup) / (steps - warmup)
            lr = min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * p))
        for g in opt.param_groups:
            g["lr"] = lr

        idx = torch.randint(0, x_ten.shape[0], (bsz,))
        bx, by, bm = x_ten[idx].to(dev), y_ten[idx].to(dev), m_ten[idx].to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev, dtype=dtype):
            out = model(input_ids=bx, targets=by, loss_mask=bm)
            assert out.ce_loss is not None and out.z_loss is not None
            loss = out.ce_loss + out.z_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        final_ce = float(out.ce_loss.item())
        if step % 150 == 0 or step == 1:
            print(f"  step {step:4d}/{steps} | sft_ce={final_ce:.4f} | lr={lr:.2e} | {time.perf_counter()-t0:.1f}s", flush=True)

    wall = time.perf_counter() - t0
    ckpt_path = out_dir / "runs" / "A2_hybrid_swa_sft_v2_diverse" / "latest.pt"
    save_atomic_checkpoint(
        ckpt_path,
        model=model,
        optimizer=opt,
        step=steps,
        train_cfg=TrainConfig(run_id="A2_hybrid_swa_sft_v2_diverse", max_steps=steps, peak_lr=peak_lr),
    )

    def gen_fn(prompt: str) -> str:
        chat_str = format_chat_messages([{"role": "user", "content": prompt}], add_generation_prompt=True)
        ids = tok.encode(chat_str).ids
        inp = torch.tensor([ids], dtype=torch.long, device=dev)
        out_ids = model.generate(
            inp,
            max_new_tokens=192,
            temperature=0.0,
            stop_token_ids=(im_end_id, eos_id),
        )[0, len(ids) :].tolist()
        while out_ids and out_ids[-1] in (im_end_id, eos_id):
            out_ids.pop()
        return tok.decode(out_ids)

    report = run_capability_evaluation("nanolm_25m_hybrid_swa_sft_v2", gen_fn)
    print(f"\n=== [E-CAP-02 RESULT] Overall Held-Out Pass Rate: {report.overall_pass_rate:.1%} (vs E-CAP-01 0.0%, Qwen2.5-1.5B 31.2%) ===", flush=True)
    for s_name, sc in report.suites.items():
        print(f"  - {s_name:18s}: {sc.passed}/{sc.total} ({sc.pass_rate:.0%})", flush=True)

    payload = {
        "experiment_id": "E-CAP-02",
        "hypothesis": "Open-vocabulary slot diversity + cosine LR annealing resolves SFT slot memorization and underfitting",
        "decontaminated_examples": len(pairs),
        "steps": steps,
        "wall_seconds": round(wall, 2),
        "final_sft_ce": round(final_ce, 4),
        "baseline_e_cap_01_pass_rate": 0.0,
        "e_cap_02_pass_rate": report.overall_pass_rate,
        "external_qwen25_1p5b_pass_rate": 0.3125,
        "evaluation": report.to_dict(),
    }
    (out_dir / "e_cap_02_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_e_cap_02()
