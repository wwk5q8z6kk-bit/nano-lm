"""Phase 2 Superior Learning: Information-Dense Curriculum, rsLoRA vs Full Fine-Tuning, & Online RLVR.

Implements three preregistered controlled experiments (`PREREG_PHASE2_SYSTEM_AND_120B.md`):
  1. `--phase probe_and_generate`:
     - Builds a ~1,050-item strictly disjoint multi-domain pool across 4 categories:
       (`math_reasoning`, `code_exec`, `tool_trajectories`, `json_and_constraints`).
     - Probes Student (`Qwen/Qwen3-1.7B`) via vLLM to measure 0-shot pass/fail + predictive token NLL,
       quantifying exact baseline weakness per category.
     - Generates & deterministically verifies trajectories from a strong teacher (`Qwen/Qwen3-32B`),
       producing matched-size `uniform.jsonl` vs `infodense.jsonl` (prioritizing Student failures &
       high-surprise verified items) plus `rlvr_pool.jsonl`.
  2. `--phase distill`:
     - Compares `uniform` vs `infodense` AND `rsLoRA (r=32, ~2% params)` vs `Full Fine-Tuning (100% params)`
       under identical Proper Top-64 + Remainder-Tail Teacher Logit KD (`alpha_kd=0.85, T=1.5`).
  3. `--phase rlvr`:
     - Runs online Verification-Driven GRPO (`G=4` rollouts/prompt, deterministic execution/math/tool/JSON
       reward in {0, 1}, Schulman k3 KL anchor `kl_beta=0.04`) on high-learnability prompts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.trackb.eval_capabilities import TOOL_SYSTEM_INSTRUCTION, strip_thinking
from lab.trackb.posttrain_pipeline import (
    attach_rslora,
    collect_eval_shield_ngrams,
    extract_topk_teacher_targets,
    filter_verified_trajectories,
    grpo_verifiable_loss,
    proper_topk_kd_loss,
)
from lab.trackb.train_nanolm_b1 import save_merged_rslora_checkpoint
from lab.trackb.train_nanolm_b3 import (
    EXTRA_TOOL_BANK,
    TRAIN_TOOL_BANK,
    _fmt_b3_tool_prompt,
    build_b3_prompt_specs,
    encode_dual_mode_example,
)


def build_phase2_candidate_pool(seed: int = 2027) -> list[dict[str, Any]]:
    """Construct ~1,040 unique, non-overlapping candidate problems across 4 tracked categories."""
    rng = random.Random(seed)
    pool: list[dict[str, Any]] = []

    # Start with the 610 diverse specs from B3 (remapped into 4 clean macro-categories)
    for sp in build_b3_prompt_specs(seed=seed):
        cat = {
            "tool": "tool_trajectories",
            "code": "code_exec",
            "math": "math_reasoning",
            "json_exact": "json_and_constraints",
            "constraint": "json_and_constraints",
        }.get(sp["domain"], "json_and_constraints")
        item = dict(sp)
        item["category"] = cat
        pool.append(item)

    # Expand 1: Additional 180 harder GSM8K train + multi-step combinatorics/number-theory problems
    # (strictly disjoint primes/moduli from LAB-FRONTIER-HARD!)
    try:
        ds_gsm = load_dataset("openai/gsm8k", "main", split="train[400:540]")
        for idx, row in enumerate(ds_gsm):
            if "####" not in row["answer"]:
                continue
            _, final_num = row["answer"].split("####", 1)
            final_clean = final_num.strip().replace(",", "")
            pool.append({
                "domain": "math",
                "sub": "gsm8k_hard_slice",
                "category": "math_reasoning",
                "prompt": f"{row['question'].strip()}\nShow clear reasoning and put the final numerical answer in \\boxed{{}}.",
                "think": (idx % 2 == 0),
                "verifier_spec": {"expected": final_clean},
            })
    except Exception as exc:
        print(f"GSM8K slice warning: {exc}", flush=True)

    # Disjoint modular/combinatorics training problems (primes 7, 13, 37, 41 — never 2-congruence CRT!)
    for i in range(50):
        n_items = 6 + (i % 6)
        k_pick = 2 + (i % 3)
        banned = 1 + (i % 2)
        # Subsets of size k_pick from n_items that do NOT contain all `banned` special elements
        ans = math.comb(n_items, k_pick) - (math.comb(n_items - banned, k_pick - banned) if k_pick >= banned else 0)
        pool.append({
            "domain": "math",
            "sub": "comb_exclusion_train",
            "category": "math_reasoning",
            "prompt": (
                f"A committee of {k_pick} engineers is chosen from {n_items} distinct candidates. "
                f"If 2 specific senior candidates cannot both serve on the committee at the same time, "
                "how many valid committees exist? Put the exact integer in \\boxed{}."
                if banned > 1
                else f"From {n_items} distinct servers, how many ways can we choose {k_pick} servers if server #1 is offline and cannot be selected? Put the exact integer in \\boxed{{}}."
            ),
            "think": (i % 2 == 0),
            "verifier_spec": {"expected": str(ans)},
        })

    # Expand 2: Additional 110 MBPP train problems (indices 120..260)
    try:
        mbpp_train = list(load_dataset("google-research-datasets/mbpp", "full", split="train"))
        fn_re = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\))\s*:")
        added = 0
        for row in mbpp_train[120:]:
            if added >= 110:
                break
            m = fn_re.search(row["code"])
            tests = row.get("test_list") or []
            if not m or not tests:
                continue
            sig = m.group(1)
            pool.append({
                "domain": "code",
                "sub": "mbpp_train_ext",
                "category": "code_exec",
                "prompt": f"{row['text'].strip()}\nWrite the Python solution using `def {sig}:` inside a ```python``` block.",
                "think": (added % 3 == 0),
                "verifier_spec": {"harness": "\n".join(tests)},
            })
            added += 1
    except Exception as exc:
        print(f"MBPP ext warning: {exc}", flush=True)

    # Expand 3: 60 Multi-step Tool Result Interpretation + Chained Follow-Up Trajectories
    hubs = ["ORD-1", "DFW-2", "FRA-1", "NRT-3"]
    for i in range(40):
        sku = f"CHAIN-{3000 + i * 19}"
        hub = hubs[i % 4]
        avail = 10 + (i % 15)
        needed = avail + 25
        asset = f"CONV-{50 + i}"
        hist = (
            f'Turn 1 Assistant: {{"tool_calls": [{{"name": "fetch_warehouse_stock", "arguments": {{"sku": "{sku}", "hub_code": "{hub}", "include_reserved": false}}}}]}}\n'
            f'Turn 1 Tool Result: {{"ok": true, "sku": "{sku}", "unreserved_units": {avail}, "shortfall_for_order": {needed - avail}}}'
        )
        p = _fmt_b3_tool_prompt(
            TRAIN_TOOL_BANK,
            f"Since `{sku}` has a shortfall of {needed - avail} units, schedule a 4-hour `HIGH` priority maintenance/restock job on `{asset}` notifying ops.",
            history=hist,
        )
        pool.append({
            "domain": "tool",
            "sub": "tool_chained_followup",
            "category": "tool_trajectories",
            "prompt": p,
            "think": (i % 3 == 0),
            "verifier_spec": {
                "allowed_tools": ["schedule_maintenance_job"],
                "required_arg_keys": {"schedule_maintenance_job": ["asset_id", "duration_hours", "notify_ops"]},
                "expected_arg_values": {"schedule_maintenance_job": {"asset_id": asset, "duration_hours": 4, "priority": "HIGH", "notify_ops": True}},
            },
        })
    for i in range(30):
        svc = f"api-gw-{i}"
        reg = ["us-west-2", "eu-central-1", "ap-northeast-1"][i % 3]
        hist = (
            f'Turn 1 Assistant: {{"tool_calls": [{{"name": "trigger_canary_rollout", "arguments": {{"service": "{svc}", "image_tag": "v4.{i}.0", "traffic_pct": 10, "region": "{reg}"}}}}]}}\n'
            f'Turn 1 Tool Result: {{"ok": true, "rollout_id": "ROL-{900 + i}", "healthy": true, "p99_ms": {18 + i}}}'
        )
        p = _fmt_b3_tool_prompt(
            EXTRA_TOOL_BANK,
            "Canary rollout succeeded and is healthy! Confirm completion to the user with the rollout_id and p99_ms without calling any more tools.",
            history=hist,
        )
        pool.append({
            "domain": "tool",
            "sub": "tool_verify_completion",
            "category": "tool_trajectories",
            "prompt": p,
            "think": (i % 4 == 0),
            "verifier_spec": {"expect_abstain": True},
        })

    return pool


def verify_single_candidate(sp: dict[str, Any], raw_resp: str, shield: set[int], require_think_match: bool = True) -> bool:
    vspec = dict(sp["verifier_spec"])
    if require_think_match and sp.get("think"):
        vspec["expect_think"] = True
    kept, _ = filter_verified_trajectories(
        [{"domain": sp["domain"], "prompt": sp["prompt"], "response": raw_resp, "verifier_spec": vspec}],
        shield,
    )
    return len(kept) == 1


def run_probe_and_generate(args) -> None:
    from vllm import LLM, SamplingParams

    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shield = collect_eval_shield_ngrams()
    pool = build_phase2_candidate_pool(seed=args.seed)
    print(f"Built Phase-2 pool: {len(pool)} items across {dict(Counter(x['category'] for x in pool))}", flush=True)

    # 1. Probe Student (Qwen3-1.7B) 0-shot pass/fail + predictive token NLL!
    s_tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=False)
    s_prompts = [
        s_tok.apply_chat_template([{"role": "user", "content": sp["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for sp in pool
    ]
    print(f"Probing student {args.student} on {len(pool)} candidates...", flush=True)
    s_llm = LLM(model=args.student, dtype="bfloat16", gpu_memory_utilization=0.55, max_model_len=4096, trust_remote_code=False)
    s_outs = s_llm.generate(s_prompts, SamplingParams(temperature=0.0, max_tokens=768, logprobs=1))
    del s_llm
    torch.cuda.empty_cache()

    student_pass: list[bool] = []
    student_nll: list[float] = []
    cat_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"n": 0, "student_pass": 0, "teacher_pass": 0, "nll_sum": 0.0})

    for sp, so in zip(pool, s_outs):
        o = so.outputs[0]
        ok = verify_single_candidate(sp, o.text, shield, require_think_match=False)
        # Mean negative log-probability of generated tokens (predictive uncertainty!)
        tok_lps = []
        if o.logprobs:
            for d_lp in o.logprobs:
                if d_lp:
                    tok_lps.append(max(v.logprob for v in d_lp.values()))
        mean_nll = -sum(tok_lps) / max(1, len(tok_lps)) if tok_lps else 0.5
        student_pass.append(ok)
        student_nll.append(round(float(mean_nll), 4))
        c = cat_stats[sp["category"]]
        c["n"] += 1
        c["student_pass"] += int(ok)
        c["nll_sum"] += float(mean_nll)

    # 2. Generate verified solutions from Strong Frontier Teacher (e.g. Qwen/Qwen3-32B)!
    t_tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=False)
    t_prompts = [
        t_tok.apply_chat_template([{"role": "user", "content": sp["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=bool(sp["think"]))
        for sp in pool
    ]
    print(f"Generating trajectories from teacher {args.teacher}...", flush=True)
    t_llm = LLM(model=args.teacher, dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=4096, trust_remote_code=False)
    idx0 = [i for i, sp in enumerate(pool) if not sp["think"]]
    idx1 = [i for i, sp in enumerate(pool) if sp["think"]]
    t_texts = [""] * len(pool)
    if idx0:
        for i, o in zip(idx0, t_llm.generate([t_prompts[i] for i in idx0], SamplingParams(temperature=0.0, max_tokens=768))):
            t_texts[i] = o.outputs[0].text
    if idx1:
        for i, o in zip(idx1, t_llm.generate([t_prompts[i] for i in idx1], SamplingParams(temperature=0.0, max_tokens=1152))):
            t_texts[i] = o.outputs[0].text
    del t_llm
    torch.cuda.empty_cache()

    verified_pool: list[dict[str, Any]] = []
    rlvr_pool: list[dict[str, Any]] = []
    for i, (sp, raw_t) in enumerate(zip(pool, t_texts)):
        t_ok = verify_single_candidate(sp, raw_t, shield, require_think_match=True)
        cat_stats[sp["category"]]["teacher_pass"] += int(t_ok)
        # RLVR pool: deterministic items where Student failed OR had elevated uncertainty (exclude callable check_fn closures)
        if sp["domain"] in ("math", "code", "tool", "json_exact") and (not student_pass[i] or student_nll[i] > 0.08):
            rlvr_pool.append({
                "domain": sp["domain"],
                "category": sp["category"],
                "prompt": sp["prompt"],
                "verifier_spec": sp["verifier_spec"],
                "student_0shot_pass": student_pass[i],
                "student_nll": student_nll[i],
            })
        if not t_ok:
            continue
        # Information-Density score I(x): prioritizes examples Student FAILED + high Student NLL!
        info_score = (2.0 if not student_pass[i] else 0.0) + min(3.0, student_nll[i] * 4.0)
        verified_pool.append({
            "idx": i,
            "domain": sp["domain"],
            "sub": sp["sub"],
            "category": sp["category"],
            "think": bool(sp["think"]),
            "prompt": sp["prompt"],
            "response": raw_t,
            "student_0shot_pass": student_pass[i],
            "student_nll": student_nll[i],
            "info_score": round(info_score, 4),
        })

    rng = random.Random(args.seed)
    target_n = min(args.subset_size, len(verified_pool))
    uniform_sub = rng.sample(verified_pool, target_n)

    # Info-Dense curriculum: top 80% highest info_score + 20% random solved anchor to prevent format drift!
    hard_sorted = sorted(verified_pool, key=lambda r: r["info_score"], reverse=True)
    n_hard = int(0.80 * target_n)
    chosen_ids = {r["idx"] for r in hard_sorted[:n_hard]}
    rem = [r for r in verified_pool if r["idx"] not in chosen_ids]
    anchors = rng.sample(rem, min(len(rem), target_n - n_hard))
    infodense_sub = hard_sorted[:n_hard] + anchors
    rng.shuffle(infodense_sub)

    rng.shuffle(rlvr_pool)
    rlvr_sub = rlvr_pool[:120]

    for fname, rows in [("uniform.jsonl", uniform_sub), ("infodense.jsonl", infodense_sub), ("rlvr_pool.jsonl", rlvr_sub)]:
        with (out_dir / fname).open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    diag = {
        "pool_total": len(pool),
        "teacher": args.teacher,
        "student": args.student,
        "verified_teacher_kept": len(verified_pool),
        "teacher_rejected_by_verifier": len(pool) - len(verified_pool),
        "subset_size": target_n,
        "uniform_student_fail_frac": round(sum(1 for r in uniform_sub if not r["student_0shot_pass"]) / max(1, len(uniform_sub)), 4),
        "infodense_student_fail_frac": round(sum(1 for r in infodense_sub if not r["student_0shot_pass"]) / max(1, len(infodense_sub)), 4),
        "category_breakdown": {
            k: {
                "total": int(v["n"]),
                "student_0shot_acc": round(v["student_pass"] / max(1, v["n"]), 4),
                "teacher_verified_acc": round(v["teacher_pass"] / max(1, v["n"]), 4),
                "student_mean_nll": round(v["nll_sum"] / max(1, v["n"]), 4),
            }
            for k, v in cat_stats.items()
        },
    }
    (out_dir / "curriculum_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print("PHASE2_CURRICULUM_DIAGNOSTICS:", json.dumps(diag), flush=True)


def run_distill_comparison(args) -> None:
    """Run controlled Top-64+Tail Teacher Logit KD comparing rsLoRA vs Full Fine-Tuning (FFT)."""
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
    rows = [json.loads(ln) for ln in Path(args.data_jsonl).read_text().splitlines() if ln.strip()]
    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=False)

    encoded = [
        x for r in rows
        if (x := encode_dual_mode_example(tok, r["prompt"], r["response"], bool(r["think"]), max_len=1536)) is not None
    ]
    print(f"[{args.mode}] Encoded {len(encoded)}/{len(rows)} trajectories from {args.data_jsonl}", flush=True)

    # Cache or load pre-extracted teacher Top-64 logits
    cache_path = Path(args.kd_cache) if args.kd_cache else None
    if cache_path and cache_path.exists():
        print(f"Loading cached teacher Top-64 logits from {cache_path}...", flush=True)
        cached_kd = torch.load(cache_path, map_location="cpu", weights_only=False)
        kd_s = 0.0
    else:
        t_kd0 = time.perf_counter()
        print(f"Extracting Top-{args.topk}+Tail logits from teacher {args.teacher}...", flush=True)
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
        teacher.eval()
        cached_kd = []
        with torch.no_grad():
            for item in encoded:
                t_logits = teacher(item["input_ids"].unsqueeze(0).to(dev), use_cache=False).logits
                idx_k, lp_k = extract_topk_teacher_targets(t_logits, k=args.topk, temperature=args.temperature)
                cached_kd.append((idx_k.squeeze(0).cpu(), lp_k.squeeze(0).cpu()))
        del teacher
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        kd_s = time.perf_counter() - t_kd0
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cached_kd, cache_path)
        print(f"Extracted teacher Top-{args.topk} logits in {kd_s:.1f}s.", flush=True)

    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    if args.mode == "rslora":
        trainable_params = attach_rslora(student, rank=args.rank, alpha=float(args.rank))
        lr = args.lr or 5e-5
    elif args.mode == "fft":
        # Full Fine-Tuning: 100% of parameters trainable with gradient checkpointing to stay crisp in bf16
        for p in student.parameters():
            p.requires_grad_(True)
        student.gradient_checkpointing_enable()
        trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        lr = args.lr or 8e-6
    else:
        raise ValueError(args.mode)

    unique_params = sum(p.numel() for p in student.parameters())
    print(f"[{args.mode.upper()}] Trainable={trainable_params:,} / {unique_params:,} ({100*trainable_params/unique_params:.2f}%), lr={lr}", flush=True)

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    total_updates = max(1, (len(encoded) * args.epochs) // args.grad_accum)
    warmup_updates = max(1, int(0.12 * total_updates))

    student.train()
    t0 = time.perf_counter()
    step_idx, upd_idx = 0, 0
    rng = random.Random(args.seed)
    indices = list(range(len(encoded)))
    history = []

    for ep in range(args.epochs):
        rng.shuffle(indices)
        ep_ce, ep_kl = 0.0, 0.0
        for idx_item in indices:
            ex = encoded[idx_item]
            tk_i, tk_lp = cached_kd[idx_item]
            s_logits = student(ex["input_ids"].unsqueeze(0).to(dev), use_cache=False).logits
            out = proper_topk_kd_loss(
                s_logits, tk_i.unsqueeze(0).to(dev), tk_lp.unsqueeze(0).to(dev),
                ex["labels"].unsqueeze(0).to(dev), temperature=args.temperature, alpha_kd=args.alpha_kd,
            )
            (out["loss"] / args.grad_accum).backward()
            ep_ce += float(out["ce"])
            ep_kl += float(out["kd_kl"])
            step_idx += 1
            if step_idx % args.grad_accum == 0 or step_idx == len(encoded) * args.epochs:
                upd_idx += 1
                lr_mult = (
                    upd_idx / warmup_updates
                    if upd_idx <= warmup_updates
                    else 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(1.0, (upd_idx - warmup_updates) / max(1, total_updates - warmup_updates))))
                )
                for g in opt.param_groups:
                    g["lr"] = lr * lr_mult
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        rec = {"epoch": ep + 1, "mean_ce": round(ep_ce / len(indices), 4), "mean_kd_kl": round(ep_kl / len(indices), 4), "elapsed_s": round(time.perf_counter() - t0, 1)}
        history.append(rec)
        print("EPOCH", json.dumps(rec), flush=True)

    out_path = Path(args.out_dir)
    if args.mode == "rslora":
        save_merged_rslora_checkpoint(student, tok, out_path)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        student.gradient_checkpointing_disable()
        student.save_pretrained(out_path)
        tok.save_pretrained(out_path)

    meta = {
        "schema": "nanolm.trackb_phase2_distill.v1",
        "mode": args.mode,
        "student": args.student,
        "teacher": args.teacher,
        "data_jsonl": str(args.data_jsonl),
        "alpha_kd": args.alpha_kd,
        "lr": lr,
        "trainable_params": trainable_params,
        "total_params": unique_params,
        "encoded_examples": len(encoded),
        "kd_cache_seconds": round(kd_s, 2),
        "train_wall_seconds": round(time.perf_counter() - t0, 2),
        "history": history,
    }
    (out_path / "nanolm_training_manifest.json").write_text(json.dumps(meta, indent=2))
    print("SAVED_PHASE2_MODEL", str(out_path), flush=True)


def run_online_rlvr(args) -> None:
    """Online Verification-Driven GRPO (`G=4` rollouts/prompt + reference-KL guardrail)."""
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
    prompts = [json.loads(ln) for ln in Path(args.rlvr_jsonl).read_text().splitlines() if ln.strip()][: args.rlvr_steps]
    shield = collect_eval_shield_ngrams()

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    policy = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    trainable = attach_rslora(policy, rank=16, alpha=16.0)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr or 1.5e-5, weight_decay=0.0)

    G = args.group_size
    t0 = time.perf_counter()
    rewards_log: list[float] = []
    nonzero_groups = 0

    for step_i, item in enumerate(prompts):
        p_txt = tok.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        p_ids = tok.encode(p_txt, add_special_tokens=False)
        if len(p_ids) > 1024:
            continue
        inp_prompt = torch.tensor([p_ids] * G, dtype=torch.long, device=dev)

        # 1. Sample G=4 online rollouts from current policy
        policy.eval()
        with torch.no_grad():
            gen_seqs = policy.generate(
                inp_prompt,
                max_new_tokens=384,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        # 2. Score each rollout with deterministic execution/schema/math verifier!
        p_len = len(p_ids)
        comp_ids = gen_seqs[:, p_len:]
        if comp_ids.size(1) < 2:
            continue
        rewards = []
        resp_masks = torch.zeros_like(comp_ids, dtype=torch.float32)
        for g_idx in range(G):
            seq_g = comp_ids[g_idx].tolist()
            # Mask tokens strictly up to and including first EOS
            if tok.eos_token_id in seq_g:
                valid_len = seq_g.index(tok.eos_token_id) + 1
            else:
                valid_len = len(seq_g)
            resp_masks[g_idx, :valid_len] = 1.0
            txt_g = tok.decode(seq_g[:valid_len], skip_special_tokens=True)
            ok = verify_single_candidate(item, txt_g, shield, require_think_match=False)
            rewards.append(1.0 if ok else 0.0)

        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=dev)
        rewards_log.append(float(r_tensor.mean().item()))

        # Only take gradient step when group has reward variance (some pass, some fail -> crisp signal!)
        if float(r_tensor.std(unbiased=False).item()) < 1e-4:
            continue
        nonzero_groups += 1

        # 3. Compute policy & frozen-reference token logprobs and apply GRPO + k3 KL
        policy.train()
        full_in = gen_seqs[:, :-1]
        target_toks = gen_seqs[:, 1:]
        pol_logits = policy(full_in, use_cache=False).logits[:, p_len - 1 :, :]
        pol_lp = F.log_softmax(pol_logits.float(), dim=-1).gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            ref_logits = ref_model(full_in, use_cache=False).logits[:, p_len - 1 :, :]
            ref_lp = F.log_softmax(ref_logits.float(), dim=-1).gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)

        out = grpo_verifiable_loss(pol_lp, ref_lp, resp_masks, r_tensor, kl_beta=args.kl_beta)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()

        if (step_i + 1) % 15 == 0:
            recent_r = sum(rewards_log[-15:]) / len(rewards_log[-15:])
            print(f"RLVR step {step_i+1}/{len(prompts)}: recent_reward={recent_r:.3f}, active_groups={nonzero_groups}, kl={float(out['kl']):.4f}", flush=True)

    out_path = Path(args.out_dir)
    save_merged_rslora_checkpoint(policy, tok, out_path)
    meta = {
        "schema": "nanolm.trackb_rlvr.v1",
        "base_student": args.student,
        "prompts_evaluated": len(prompts),
        "active_variance_groups": nonzero_groups,
        "group_size": G,
        "kl_beta": args.kl_beta,
        "mean_reward_first_quarter": round(sum(rewards_log[: max(1, len(rewards_log)//4)]) / max(1, len(rewards_log)//4), 4) if rewards_log else 0.0,
        "mean_reward_last_quarter": round(sum(rewards_log[-max(1, len(rewards_log)//4) :]) / max(1, len(rewards_log)//4), 4) if rewards_log else 0.0,
        "trainable_params": trainable,
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    (out_path / "nanolm_training_manifest.json").write_text(json.dumps(meta, indent=2))
    print("SAVED_RLVR_MODEL", str(out_path), json.dumps(meta), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["probe_and_generate", "distill", "rlvr"])
    ap.add_argument("--student", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--teacher", default="Qwen/Qwen3-32B")
    ap.add_argument("--data-dir", default="/workspace/phase2_data")
    ap.add_argument("--data-jsonl", default="/workspace/phase2_data/infodense.jsonl")
    ap.add_argument("--rlvr-jsonl", default="/workspace/phase2_data/rlvr_pool.jsonl")
    ap.add_argument("--kd-cache", default="")
    ap.add_argument("--out-dir", default="/workspace/models/nanolm_b4")
    ap.add_argument("--mode", default="rslora", choices=["rslora", "fft"])
    ap.add_argument("--subset-size", type=int, default=420)
    ap.add_argument("--alpha-kd", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--rlvr-steps", type=int, default=100)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--kl-beta", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()

    if args.phase == "probe_and_generate":
        run_probe_and_generate(args)
    elif args.phase == "distill":
        run_distill_comparison(args)
    else:
        run_online_rlvr(args)


if __name__ == "__main__":
    main()
