"""Phase 3 Track B Innovation (`LAB-TRACKB-03`):
  1. Stratified Multi-Teacher (`Qwen3-32B` + `openai/gpt-oss-120b`) Verified Curriculum
     with Multi-Turn Stateful API Error-Recovery Trajectories (strictly disjoint domains).
  2. `NanoLM-B6-2B` (`1.72B` Active Params): Pretrained-Weight L2-Anchored `100%` Full Fine-Tuning
     + Proper Top-64+Tail Logit KD + `G=8` Online Verifiable `GRPO`.
  3. `NanoLM-B6-3B` (`3.08B` Active Params, Stage-3 `<4B` Scaling Test on `HuggingFaceTB/SmolLM3-3B`):
     Stratified `32B+120B` Verified Dual-Mode Distillation + `G=8` Online Verifiable `GRPO`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.trackb.eval_capabilities import TOOL_SYSTEM_INSTRUCTION, strip_thinking
from lab.trackb.phase2_learning import build_phase2_candidate_pool, verify_single_candidate
from lab.trackb.posttrain_pipeline import (
    attach_rslora,
    collect_eval_shield_ngrams,
    extract_topk_teacher_targets,
    filter_verified_trajectories,
    grpo_verifiable_loss,
    proper_topk_kd_loss,
)
from lab.trackb.train_nanolm_b1 import save_merged_rslora_checkpoint
from lab.trackb.train_nanolm_b3 import encode_dual_mode_example

# Disjoint multi-turn API schemas (OAuth / DNS / Warehouse — zero overlap with LAB-FRONTIER-HARD incident/shard!)
DISJOINT_RECOVERY_TOOLS = [
    {
        "name": "fetch_dns_zone_profile",
        "description": "Retrieve zone_code, current_ttl_sec, and oauth_grant for a domain.",
        "parameters": {
            "type": "object",
            "required": ["domain_name"],
            "properties": {"domain_name": {"type": "string"}},
        },
    },
    {
        "name": "rotate_oauth_grant",
        "description": "Issue a fresh_oauth_grant for a zone_code when an existing oauth_grant is expired.",
        "parameters": {
            "type": "object",
            "required": ["zone_code"],
            "properties": {"zone_code": {"type": "string"}},
        },
    },
    {
        "name": "update_dns_record_ttl",
        "description": "Set target_ttl_sec on a DNS zone using a valid oauth_grant.",
        "parameters": {
            "type": "object",
            "required": ["zone_code", "oauth_grant", "target_ttl_sec"],
            "properties": {
                "zone_code": {"type": "string"},
                "oauth_grant": {"type": "string"},
                "target_ttl_sec": {"type": "integer"},
            },
        },
    },
]


def build_phase3_stratified_pool(seed: int = 3031) -> list[dict[str, Any]]:
    """Expand Phase-2 pool with 90 disjoint multi-turn API error-recovery states."""
    pool = build_phase2_candidate_pool(seed=seed)
    schema_txt = json.dumps(DISJOINT_RECOVERY_TOOLS, indent=2)

    # 1. 30 Stale OAuth Grant -> must invoke `rotate_oauth_grant`!
    for i in range(30):
        dom = f"svc-{100 + i}.internal.example"
        zcode = f"ZONE-NA-{300 + i}"
        hist = (
            f'Turn 1 Assistant: {{"tool_calls": [{{"name": "fetch_dns_zone_profile", "arguments": {{"domain_name": "{dom}"}}}}]}}\n'
            f'Turn 1 Observation: [{{"tool": "fetch_dns_zone_profile", "ok": true, "zone_code": "{zcode}", "current_ttl_sec": 3600, "oauth_grant": "GRANT-EXP-{i}"}}]\n'
            f'Turn 2 Assistant: {{"tool_calls": [{{"name": "update_dns_record_ttl", "arguments": {{"zone_code": "{zcode}", "oauth_grant": "GRANT-EXP-{i}", "target_ttl_sec": 600}}}}]}}\n'
            f'Turn 2 Observation: [{{"tool": "update_dns_record_ttl", "ok": false, "error": "ExpiredOAuthGrant", "message": "oauth_grant GRANT-EXP-{i} is expired; call rotate_oauth_grant first."}}]'
        )
        p = (
            f"{TOOL_SYSTEM_INSTRUCTION}\n\nAvailable tools:\n```json\n{schema_txt}\n```\n"
            f"Interaction Log:\n{hist}\n"
            f"User Objective: Lower DNS TTL for `{dom}` to `600` seconds, recovering from any expired grants or bounds."
        )
        pool.append({
            "domain": "tool",
            "sub": "multiturn_rotate_grant",
            "category": "tool_trajectories",
            "prompt": p,
            "think": True,
            "verifier_spec": {
                "allowed_tools": ["rotate_oauth_grant"],
                "required_arg_keys": {"rotate_oauth_grant": ["zone_code"]},
                "expected_arg_values": {"rotate_oauth_grant": {"zone_code": zcode}},
            },
        })

    # 2. 30 Minimum TTL Floor Violation -> must clamp `target_ttl_sec` to `min_ttl_sec` using `fresh_oauth_grant`!
    for i in range(30):
        dom = f"edge-{200 + i}.cdn.example"
        zcode = f"ZONE-EU-{500 + i}"
        fresh_g = f"GRANT-VALID-{700 + i}"
        min_ttl = 300 + (i % 5) * 60
        hist = (
            f'Turn 1 Observation: [{{"tool": "fetch_dns_zone_profile", "ok": true, "zone_code": "{zcode}", "current_ttl_sec": 1800, "oauth_grant": "GRANT-OLD-{i}"}}]\n'
            f'Turn 2 Observation: [{{"tool": "rotate_oauth_grant", "ok": true, "zone_code": "{zcode}", "fresh_oauth_grant": "{fresh_g}"}}]\n'
            f'Turn 3 Assistant: {{"tool_calls": [{{"name": "update_dns_record_ttl", "arguments": {{"zone_code": "{zcode}", "oauth_grant": "{fresh_g}", "target_ttl_sec": 120}}}}]}}\n'
            f'Turn 3 Observation: [{{"tool": "update_dns_record_ttl", "ok": false, "error": "TTLBelowPolicyMinimum", "message": "target_ttl_sec=120 violates policy minimum; clamp target_ttl_sec to {min_ttl}."}}]'
        )
        p = (
            f"{TOOL_SYSTEM_INSTRUCTION}\n\nAvailable tools:\n```json\n{schema_txt}\n```\n"
            f"Interaction Log:\n{hist}\n"
            f"User Objective: Reduce DNS TTL for `{dom}` toward `120` seconds while respecting any policy minimum bounds and active grants."
        )
        pool.append({
            "domain": "tool",
            "sub": "multiturn_clamp_bound",
            "category": "tool_trajectories",
            "prompt": p,
            "think": True,
            "verifier_spec": {
                "allowed_tools": ["update_dns_record_ttl"],
                "required_arg_keys": {"update_dns_record_ttl": ["zone_code", "oauth_grant", "target_ttl_sec"]},
                "expected_arg_values": {
                    "update_dns_record_ttl": {"zone_code": zcode, "oauth_grant": fresh_g, "target_ttl_sec": min_ttl}
                },
            },
        })

    # 3. 30 Permanent Registry Lock / Post-Success Completion -> must emit `{"tool_calls": [], "response": "..."}`!
    for i in range(30):
        dom = f"vault-{400 + i}.sec.example"
        zcode = f"ZONE-SEC-{800 + i}"
        if i % 2 == 0:
            hist = (
                f'Turn 1 Observation: [{{"tool": "fetch_dns_zone_profile", "ok": true, "zone_code": "{zcode}", "current_ttl_sec": 3600, "oauth_grant": "GRANT-S-{i}"}}]\n'
                f'Turn 2 Observation: [{{"tool": "rotate_oauth_grant", "ok": false, "error": "RegistrySecurityHold", "message": "Zone {zcode} is under security hold; abort changes immediately."}}]'
            )
            obj = f"Update TTL on `{dom}` to `900`; if a RegistrySecurityHold is returned, halt immediately without further tool calls and report the hold."
        else:
            hist = (
                f'Turn 3 Observation: [{{"tool": "update_dns_record_ttl", "ok": true, "zone_code": "{zcode}", "applied_ttl_sec": 600, "status": "UPDATED"}}]'
            )
            obj = f"Confirm the completed TTL update on `{dom}` (`{zcode}`, `600` seconds) with zero additional tool calls."
        p = (
            f"{TOOL_SYSTEM_INSTRUCTION}\n"
            'When finished or halted by a security hold, output `{"tool_calls": [], "response": "<concise grounded report>"}`.\n\n'
            f"Available tools:\n```json\n{schema_txt}\n```\nInteraction Log:\n{hist}\nUser Objective: {obj}"
        )
        pool.append({
            "domain": "tool",
            "sub": "multiturn_halt_or_finish",
            "category": "tool_trajectories",
            "prompt": p,
            "think": True,
            "verifier_spec": {"expect_abstain": True},
        })

    return pool


def _normalize_harmony_to_qwen_response(raw_harmony: str, want_think: bool) -> str:
    """Convert OpenAI gpt-oss Harmony output into standard `<think>...</think>\\nfinal` or clean `final`."""
    s = raw_harmony
    analysis = ""
    final = strip_thinking(s)
    if "<|channel|>analysis<|message|>" in s and "<|channel|>final<|message|>" in s:
        a_part = s.split("<|channel|>analysis<|message|>", 1)[1]
        analysis = a_part.split("<|channel|>final<|message|>", 1)[0]
        for tag in ("<|end|>", "<|start|>assistant"):
            analysis = analysis.replace(tag, "")
        analysis = analysis.strip()
    if want_think:
        thought = analysis[:1200] if analysis else "Let us verify the constraints step by step."
        return f"<think>\n{thought}\n</think>\n\n{final}"
    return final


def run_generate_stratified(args) -> None:
    from vllm import LLM, SamplingParams

    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shield = collect_eval_shield_ngrams()
    pool = build_phase3_stratified_pool(seed=args.seed)
    print(f"Built Phase-3 pool: {len(pool)} items across {dict(Counter(x['category'] for x in pool))}", flush=True)

    # 1. Probe Qwen3-1.7B Student surprise & 0-shot pass rate
    s_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=False)
    s_prompts = [
        s_tok.apply_chat_template([{"role": "user", "content": sp["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for sp in pool
    ]
    s_llm = LLM(model="Qwen/Qwen3-1.7B", dtype="bfloat16", gpu_memory_utilization=0.55, max_model_len=4096, trust_remote_code=False)
    s_outs = s_llm.generate(s_prompts, SamplingParams(temperature=0.0, max_tokens=768, logprobs=1))
    del s_llm
    gc.collect()
    torch.cuda.empty_cache()

    student_pass: list[bool] = []
    student_nll: list[float] = []
    for sp, so in zip(pool, s_outs):
        o = so.outputs[0]
        ok = verify_single_candidate(sp, o.text, shield, require_think_match=False)
        tok_lps = [max(v.logprob for v in d_lp.values()) for d_lp in (o.logprobs or []) if d_lp]
        mean_nll = -sum(tok_lps) / max(1, len(tok_lps)) if tok_lps else 0.5
        student_pass.append(ok)
        student_nll.append(round(float(mean_nll), 4))

    # 2. Primary Teacher: Qwen/Qwen3-32B across all items
    t32_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B", trust_remote_code=False)
    t32_prompts = [
        t32_tok.apply_chat_template([{"role": "user", "content": sp["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=bool(sp["think"]))
        for sp in pool
    ]
    t32_llm = LLM(model="Qwen/Qwen3-32B", dtype="bfloat16", gpu_memory_utilization=0.88, max_model_len=4096, trust_remote_code=False)
    idx0 = [i for i, sp in enumerate(pool) if not sp["think"]]
    idx1 = [i for i, sp in enumerate(pool) if sp["think"]]
    best_resp = [""] * len(pool)
    teacher_src = ["none"] * len(pool)
    if idx0:
        for i, o in zip(idx0, t32_llm.generate([t32_prompts[i] for i in idx0], SamplingParams(temperature=0.0, max_tokens=768))):
            if verify_single_candidate(pool[i], o.outputs[0].text, shield, require_think_match=True):
                best_resp[i] = o.outputs[0].text
                teacher_src[i] = "Qwen3-32B"
    if idx1:
        for i, o in zip(idx1, t32_llm.generate([t32_prompts[i] for i in idx1], SamplingParams(temperature=0.0, max_tokens=1408))):
            if verify_single_candidate(pool[i], o.outputs[0].text, shield, require_think_match=True):
                best_resp[i] = o.outputs[0].text
                teacher_src[i] = "Qwen3-32B"
    del t32_llm
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Frontier 120B Teacher (`openai/gpt-oss-120b`) rescues hard items missed by 32B!
    missed_idx = [i for i, src in enumerate(teacher_src) if src == "none"]
    rescued_120b = 0
    if missed_idx and not args.skip_120b_teacher:
        print(f"Invoking openai/gpt-oss-120b on {len(missed_idx)} hard items missed by Qwen3-32B...", flush=True)
        t120_tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b", trust_remote_code=False)
        p120 = []
        for i in missed_idx:
            try:
                p120.append(t120_tok.apply_chat_template([{"role": "user", "content": pool[i]["prompt"]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium"))
            except TypeError:
                p120.append(t120_tok.apply_chat_template([{"role": "user", "content": pool[i]["prompt"]}], tokenize=False, add_generation_prompt=True))
        t120_llm = LLM(model="openai/gpt-oss-120b", gpu_memory_utilization=0.90, max_model_len=8192, trust_remote_code=False)
        outs120 = t120_llm.generate(p120, SamplingParams(temperature=0.0, max_tokens=2048))
        for i, o in zip(missed_idx, outs120):
            norm_txt = _normalize_harmony_to_qwen_response(o.outputs[0].text, bool(pool[i]["think"]))
            if verify_single_candidate(pool[i], norm_txt, shield, require_think_match=True):
                best_resp[i] = norm_txt
                teacher_src[i] = "gpt-oss-120b"
                rescued_120b += 1
        del t120_llm
        gc.collect()
        torch.cuda.empty_cache()

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rlvr_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, sp in enumerate(pool):
        if sp["domain"] in ("math", "code", "tool", "json_exact") and (not student_pass[i] or student_nll[i] > 0.06 or "multiturn_" in sp.get("sub", "")):
            rlvr_by_cat[sp["category"]].append({
                "domain": sp["domain"],
                "sub": sp.get("sub", ""),
                "category": sp["category"],
                "prompt": sp["prompt"],
                "verifier_spec": sp["verifier_spec"],
            })
        if teacher_src[i] == "none":
            continue
        bonus = 1.5 if "multiturn_" in sp.get("sub", "") else 0.0
        info_sc = (2.0 if not student_pass[i] else 0.0) + min(3.0, student_nll[i] * 4.0) + bonus
        by_cat[sp["category"]].append({
            "idx": i,
            "domain": sp["domain"],
            "sub": sp["sub"],
            "category": sp["category"],
            "think": bool(sp["think"]),
            "prompt": sp["prompt"],
            "response": best_resp[i],
            "teacher_source": teacher_src[i],
            "student_0shot_pass": student_pass[i],
            "student_nll": student_nll[i],
            "info_score": round(info_sc, 4),
        })

    # Stratified High-Surprise Selection: guaranteed >= 22% quota per category (`per_cat = subset_size // 4`)!
    rng = random.Random(args.seed)
    per_cat = args.subset_size // 4
    stratified: list[dict[str, Any]] = []
    for cat, items in sorted(by_cat.items()):
        items_sorted = sorted(items, key=lambda r: r["info_score"], reverse=True)
        n_hard = min(len(items_sorted), int(0.85 * per_cat))
        picked = list(items_sorted[:n_hard])
        rem = items_sorted[n_hard:]
        if rem and len(picked) < per_cat:
            picked.extend(rng.sample(rem, min(len(rem), per_cat - len(picked))))
        stratified.extend(picked)
    rng.shuffle(stratified)

    rlvr_strat: list[dict[str, Any]] = []
    for cat, r_items in sorted(rlvr_by_cat.items()):
        rng.shuffle(r_items)
        rlvr_strat.extend(r_items[:25])
    rng.shuffle(rlvr_strat)

    with (out_dir / "stratified_b6.jsonl").open("w") as f:
        for r in stratified:
            f.write(json.dumps(r) + "\n")
    with (out_dir / "rlvr_b6.jsonl").open("w") as f:
        for r in rlvr_strat:
            f.write(json.dumps(r) + "\n")

    diag = {
        "pool_total": len(pool),
        "verified_32b": sum(1 for s in teacher_src if s == "Qwen3-32B"),
        "rescued_by_120b": rescued_120b,
        "stratified_total": len(stratified),
        "stratified_by_category": dict(Counter(r["category"] for r in stratified)),
        "stratified_by_teacher": dict(Counter(r["teacher_source"] for r in stratified)),
        "rlvr_pool_total": len(rlvr_strat),
    }
    (out_dir / "phase3_curriculum_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print("PHASE3_CURRICULUM_DIAGNOSTICS:", json.dumps(diag), flush=True)


def _encode_smollm3_example(tok, prompt: str, response: str, think: bool, max_len: int = 1536):
    sys_msg = "/think" if think else "/no_think"
    vis_resp = response if think else strip_thinking(response)
    msgs_prompt = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
    msgs_full = msgs_prompt + [{"role": "assistant", "content": vis_resp}]
    # transformers>=5 returns BatchEncoding for tokenize=True (len()==2 keys!) -> render text then encode explicitly.
    p_txt = tok.apply_chat_template(msgs_prompt, tokenize=False, add_generation_prompt=True)
    f_txt = tok.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
    p_ids = tok.encode(p_txt, add_special_tokens=False)
    f_ids = tok.encode(f_txt, add_special_tokens=False)
    if f_ids[: len(p_ids)] != p_ids:
        return None
    if len(f_ids) > max_len or len(f_ids) <= len(p_ids) + 2:
        return None
    inp = torch.tensor(f_ids, dtype=torch.long)
    lbl = inp.clone()
    lbl[: len(p_ids)] = -100
    return {"input_ids": inp, "labels": lbl}


def _run_grpo_stage(base_dir: str, rlvr_jsonl: str, out_dir: str, steps: int, group_size: int, kl_beta: float, is_smollm3: bool, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    prompts = [json.loads(ln) for ln in Path(rlvr_jsonl).read_text().splitlines() if ln.strip()][:steps]
    shield = collect_eval_shield_ngrams()
    tok = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    ref_model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    policy = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    trainable = attach_rslora(policy, rank=16, alpha=16.0)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1.2e-5, weight_decay=0.0)

    t0 = time.perf_counter()
    rewards_log: list[float] = []
    nonzero_groups = 0

    for step_i, item in enumerate(prompts):
        use_think = "multiturn_" in item.get("sub", "")
        if is_smollm3:
            msgs = [{"role": "system", "content": "/think" if use_think else "/no_think"}, {"role": "user", "content": item["prompt"]}]
            p_txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            p_txt = tok.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=use_think)
        p_ids = tok.encode(p_txt, add_special_tokens=False)
        if len(p_ids) > 1152:
            continue
        inp_prompt = torch.tensor([p_ids] * group_size, dtype=torch.long, device=dev)

        policy.eval()
        with torch.no_grad():
            gen_seqs = policy.generate(
                inp_prompt,
                max_new_tokens=512 if use_think else 384,
                do_sample=True,
                temperature=0.75,
                top_p=0.95,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        p_len = len(p_ids)
        comp_ids = gen_seqs[:, p_len:]
        if comp_ids.size(1) < 2:
            continue
        rewards = []
        resp_masks = torch.zeros_like(comp_ids, dtype=torch.float32)
        for g_idx in range(group_size):
            seq_g = comp_ids[g_idx].tolist()
            valid_len = (seq_g.index(tok.eos_token_id) + 1) if tok.eos_token_id in seq_g else len(seq_g)
            resp_masks[g_idx, :valid_len] = 1.0
            txt_g = tok.decode(seq_g[:valid_len], skip_special_tokens=False)
            ok = verify_single_candidate(item, txt_g, shield, require_think_match=False)
            rewards.append(1.0 if ok else 0.0)

        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=dev)
        rewards_log.append(float(r_tensor.mean().item()))
        if float(r_tensor.std(unbiased=False).item()) < 1e-4:
            continue
        nonzero_groups += 1

        policy.train()
        full_in = gen_seqs[:, :-1]
        pol_logits = policy(full_in, use_cache=False).logits[:, p_len - 1 :, :]
        pol_lp = F.log_softmax(pol_logits.float(), dim=-1).gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            ref_logits = ref_model(full_in, use_cache=False).logits[:, p_len - 1 :, :]
            ref_lp = F.log_softmax(ref_logits.float(), dim=-1).gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)

        out = grpo_verifiable_loss(pol_lp, ref_lp, resp_masks, r_tensor, kl_beta=kl_beta)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step()

        if (step_i + 1) % 15 == 0:
            print(f"[GRPO G={group_size}] step {step_i+1}/{len(prompts)} reward={sum(rewards_log[-15:])/15:.3f} active={nonzero_groups}", flush=True)

    save_merged_rslora_checkpoint(policy, tok, Path(out_dir))
    return {
        "grpo_steps": len(prompts),
        "group_size": group_size,
        "active_variance_groups": nonzero_groups,
        "mean_reward_first_quarter": round(sum(rewards_log[: max(1, len(rewards_log)//4)]) / max(1, len(rewards_log)//4), 4) if rewards_log else 0.0,
        "mean_reward_last_quarter": round(sum(rewards_log[-max(1, len(rewards_log)//4) :]) / max(1, len(rewards_log)//4), 4) if rewards_log else 0.0,
        "grpo_wall_s": round(time.perf_counter() - t0, 2),
        "grpo_trainable_params": trainable,
    }


def run_train_b6_2b(args) -> None:
    """Train NanoLM-B6-2B: Pretrained-Weight L2-Anchored 100% FFT + Qwen3-32B Top-64+Tail KD + G=8 GRPO."""
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    rows = [json.loads(ln) for ln in Path(args.data_jsonl).read_text().splitlines() if ln.strip()]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=False)
    encoded = [x for r in rows if (x := encode_dual_mode_example(tok, r["prompt"], r["response"], bool(r["think"]), max_len=1536)) is not None]
    print(f"[B6-2B] Encoded {len(encoded)}/{len(rows)} stratified trajectories", flush=True)

    # Extract Top-64+Tail teacher logits from Qwen3-32B
    t0_kd = time.perf_counter()
    teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-32B", torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    teacher.eval()
    cached_kd = []
    with torch.no_grad():
        for ex in encoded:
            tl = teacher(ex["input_ids"].unsqueeze(0).to(dev), use_cache=False).logits
            ik, lpk = extract_topk_teacher_targets(tl, k=64, temperature=1.5)
            cached_kd.append((ik.squeeze(0).cpu(), lpk.squeeze(0).cpu()))
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    kd_s = time.perf_counter() - t0_kd
    print(f"[B6-2B] Extracted Qwen3-32B Top-64+Tail logits in {kd_s:.1f}s", flush=True)

    student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    # Snapshot frozen pretrained reference weights theta_0 (`3.4 GB` in bf16) for proximal L2 anchoring!
    anchor_params = [p.detach().clone() for p in student.parameters()]
    for p in student.parameters():
        p.requires_grad_(True)
    student.gradient_checkpointing_enable()

    lr = 1.0e-5
    anchor_strength = 0.08  # proximal pull toward theta_0 preventing pretrained capability drift!
    opt = torch.optim.AdamW(list(student.parameters()), lr=lr, weight_decay=0.0)
    total_updates = max(1, len(encoded) // 8)
    warmup_updates = max(1, int(0.12 * total_updates))

    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    order = list(range(len(encoded)))
    rng.shuffle(order)
    upd = 0
    student.train()
    for step_i, idx_ex in enumerate(order, 1):
        ex = encoded[idx_ex]
        ik, lpk = cached_kd[idx_ex]
        sl = student(ex["input_ids"].unsqueeze(0).to(dev), use_cache=False).logits
        out = proper_topk_kd_loss(sl, ik.unsqueeze(0).to(dev), lpk.unsqueeze(0).to(dev), ex["labels"].unsqueeze(0).to(dev), temperature=1.5, alpha_kd=0.80)
        (out["loss"] / 8.0).backward()
        if step_i % 8 == 0 or step_i == len(order):
            upd += 1
            lr_mult = upd / warmup_updates if upd <= warmup_updates else 0.15 + 0.425 * (1.0 + math.cos(math.pi * min(1.0, (upd - warmup_updates) / max(1, total_updates - warmup_updates))))
            eff_lr = lr * lr_mult
            for g in opt.param_groups:
                g["lr"] = eff_lr
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            # Proximal L2 weight-space anchoring toward pretrained Qwen3-1.7B theta_0!
            with torch.no_grad():
                pull = min(0.02, anchor_strength * lr_mult * 0.01)
                for p, p0 in zip(student.parameters(), anchor_params):
                    p.lerp_(p0, pull)

    mid_dir = Path(args.out_dir + "_anchored_fft")
    mid_dir.mkdir(parents=True, exist_ok=True)
    student.gradient_checkpointing_disable()
    unique_p = sum(p.numel() for p in student.parameters())
    student.save_pretrained(mid_dir)
    tok.save_pretrained(mid_dir)
    fft_wall = time.perf_counter() - t0
    del student, anchor_params, opt
    gc.collect()
    torch.cuda.empty_cache()

    grpo_stats = _run_grpo_stage(str(mid_dir), args.rlvr_jsonl, args.out_dir, steps=60, group_size=8, kl_beta=0.05, is_smollm3=False, seed=args.seed)
    manifest = {
        "schema": "nanolm.trackb_phase3_b6_2b.v1",
        "base_model": "Qwen/Qwen3-1.7B",
        "total_params": unique_p,
        "active_params": unique_p,
        "method": "Stratified_32B_120B_L2Anchored_100pct_FFT_Top64KD_plus_G8_GRPO",
        "kd_seconds": round(kd_s, 2),
        "anchored_fft_seconds": round(fft_wall, 2),
        **grpo_stats,
    }
    (Path(args.out_dir) / "nanolm_training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SAVED_NANOLM_B6_2B", args.out_dir, json.dumps(manifest), flush=True)


def run_train_b6_3b(args) -> None:
    """Train Stage-3 NanoLM-B6-3B (`3.08B` Active Params on `HuggingFaceTB/SmolLM3-3B`)."""
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    rows = [json.loads(ln) for ln in Path(args.data_jsonl).read_text().splitlines() if ln.strip()]
    base_id = "HuggingFaceTB/SmolLM3-3B"
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    encoded = [x for r in rows if (x := _encode_smollm3_example(tok, r["prompt"], r["response"], bool(r["think"]), max_len=1536)) is not None]
    print(f"[B6-3B] Encoded {len(encoded)}/{len(rows)} stratified 32B+120B trajectories for SmolLM3-3B", flush=True)

    student = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    unique_p = sum(p.numel() for p in student.parameters())
    trainable = attach_rslora(student, rank=32, alpha=32.0)
    lr = 4.5e-5
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)

    epochs = 2
    total_updates = max(1, (len(encoded) * epochs) // 8)
    warmup_updates = max(1, int(0.12 * total_updates))
    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    indices = list(range(len(encoded)))
    step_c, upd = 0, 0
    student.train()
    for ep in range(epochs):
        rng.shuffle(indices)
        ep_loss = 0.0
        for idx_ex in indices:
            ex = encoded[idx_ex]
            inp = ex["input_ids"].unsqueeze(0).to(dev)
            lbl = ex["labels"].unsqueeze(0).to(dev)
            logits = student(inp, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), lbl[:, 1:].contiguous().view(-1), ignore_index=-100)
            (loss / 8.0).backward()
            ep_loss += float(loss.detach().item())
            step_c += 1
            if step_c % 8 == 0 or step_c == len(encoded) * epochs:
                upd += 1
                lr_mult = upd / warmup_updates if upd <= warmup_updates else 0.15 + 0.425 * (1.0 + math.cos(math.pi * min(1.0, (upd - warmup_updates) / max(1, total_updates - warmup_updates))))
                for g in opt.param_groups:
                    g["lr"] = lr * lr_mult
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
        print(f"[B6-3B] Epoch {ep+1}/{epochs} mean_ce={ep_loss/max(1, len(indices)):.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    mid_dir = Path(args.out_dir + "_sft")
    save_merged_rslora_checkpoint(student, tok, mid_dir)
    sft_wall = time.perf_counter() - t0
    del student, opt
    gc.collect()
    torch.cuda.empty_cache()

    grpo_stats = _run_grpo_stage(str(mid_dir), args.rlvr_jsonl, args.out_dir, steps=50, group_size=8, kl_beta=0.05, is_smollm3=True, seed=args.seed)
    manifest = {
        "schema": "nanolm.trackb_phase3_b6_3b.v1",
        "base_model": base_id,
        "total_params": unique_p,
        "active_params": unique_p,
        "rslora_trainable_params": trainable,
        "method": "Stratified_32B_120B_DualMode_rsLoRA32_plus_G8_GRPO",
        "sft_wall_seconds": round(sft_wall, 2),
        **grpo_stats,
    }
    (Path(args.out_dir) / "nanolm_training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SAVED_NANOLM_B6_3B", args.out_dir, json.dumps(manifest), flush=True)


def run_train_b7_3b(args) -> None:
    """Train NanoLM-B7-3B-Fixed (`3.075B` < 4.0B on `HuggingFaceTB/SmolLM3-3B`):
    Causally Shifted (`x_t -> x_{t+1}`) Self-Anchored Top-64+Tail KL (`alpha_kd=0.35`, `T=1.5`,
    effective Jacobian KL weight `0.35 * 1.5^2 = 0.7875` vs `0.65` CE) + Proximal L2 100% FFT + `G=8` `GRPO`.

    Fixes Phase-5A unshifted `_encode_smollm3_example` label alignment (`f_ids[:-1] -> f_ids[1:]`) so
    both `CE` and `Self-KD` supervise true next-token prediction (`x_t -> x_{t+1}`, including `<|im_end|>` EOS).
    """
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    rows = [json.loads(ln) for ln in Path(args.data_jsonl).read_text().splitlines() if ln.strip()]
    base_id = "HuggingFaceTB/SmolLM3-3B"
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    raw_encoded = [x for r in rows if (x := _encode_smollm3_example(tok, r["prompt"], r["response"], bool(r["think"]), max_len=1536)) is not None]
    # Explicit causal shift (`f_ids[:-1]` -> `labels[1:]`) required by `proper_topk_kd_loss`!
    encoded = [{"input_ids": ex["input_ids"][:-1], "labels": ex["labels"][1:]} for ex in raw_encoded]
    print(f"[B7-3B-Fixed] Encoded & causally shifted {len(encoded)}/{len(rows)} stratified 32B+120B trajectories for SmolLM3-3B", flush=True)

    # 1. Extract Self-Anchor Top-64+Tail next-token distributions from frozen pretrained SmolLM3-3B (theta_0)
    t0_kd = time.perf_counter()
    self_ref = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    self_ref.eval()
    cached_self_kd = []
    with torch.no_grad():
        for ex in encoded:
            tl = self_ref(ex["input_ids"].unsqueeze(0).to(dev), use_cache=False).logits
            ik, lpk = extract_topk_teacher_targets(tl, k=64, temperature=1.5)
            cached_self_kd.append((ik.squeeze(0).cpu(), lpk.squeeze(0).cpu()))
    del self_ref
    gc.collect()
    torch.cuda.empty_cache()
    kd_s = time.perf_counter() - t0_kd
    print(f"[B7-3B-Fixed] Extracted Self-Anchor SmolLM3-3B Top-64+Tail next-token logits in {kd_s:.1f}s", flush=True)

    # 2. 100% Full-Parameter Fine-Tuning with Jacobian-Balanced Self-Anchored KL (alpha_kd=0.35) + Proximal L2 Anchor!
    student = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    anchor_params = [p.detach().clone() for p in student.parameters()]
    for p in student.parameters():
        p.requires_grad_(True)
    student.gradient_checkpointing_enable()

    lr = 1.2e-5
    anchor_strength = 0.08
    opt = torch.optim.AdamW(list(student.parameters()), lr=lr, weight_decay=0.0)
    total_updates = max(1, len(encoded) // 8)
    warmup_updates = max(1, int(0.12 * total_updates))

    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    order = list(range(len(encoded)))
    rng.shuffle(order)
    upd = 0
    loss_sum, ce_sum, kl_sum = 0.0, 0.0, 0.0
    student.train()
    for step_i, idx_ex in enumerate(order, 1):
        ex = encoded[idx_ex]
        ik, lpk = cached_self_kd[idx_ex]
        sl = student(ex["input_ids"].unsqueeze(0).to(dev), use_cache=False).logits
        out = proper_topk_kd_loss(
            sl,
            ik.unsqueeze(0).to(dev),
            lpk.unsqueeze(0).to(dev),
            ex["labels"].unsqueeze(0).to(dev),
            temperature=1.5,
            alpha_kd=0.35,
        )
        (out["loss"] / 8.0).backward()
        loss_sum += float(out["loss"].detach().item())
        ce_sum += float(out["ce"].item())
        kl_sum += float(out["kd_kl"].item())
        if step_i % 8 == 0 or step_i == len(order):
            upd += 1
            lr_mult = upd / warmup_updates if upd <= warmup_updates else 0.15 + 0.425 * (1.0 + math.cos(math.pi * min(1.0, (upd - warmup_updates) / max(1, total_updates - warmup_updates))))
            eff_lr = lr * lr_mult
            for g in opt.param_groups:
                g["lr"] = eff_lr
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                pull = min(0.02, anchor_strength * lr_mult * 0.01)
                for p, p0 in zip(student.parameters(), anchor_params):
                    p.lerp_(p0, pull)
    n_ord = max(1, len(order))
    print(
        f"[B7-3B-Fixed] Self-Anchored KD+FFT epoch mean_loss={loss_sum/n_ord:.4f} "
        f"(ce={ce_sum/n_ord:.4f}, self_kl={kl_sum/n_ord:.4f}) elapsed={time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    mid_dir = Path(args.out_dir + "_self_anchored_fft")
    mid_dir.mkdir(parents=True, exist_ok=True)
    student.gradient_checkpointing_disable()
    unique_p = sum(p.numel() for p in student.parameters())
    student.save_pretrained(mid_dir)
    tok.save_pretrained(mid_dir)
    fft_wall = time.perf_counter() - t0
    del student, anchor_params, opt
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Online Verifiable GRPO (`G=8`, `kl_beta=0.05`)
    grpo_stats = _run_grpo_stage(str(mid_dir), args.rlvr_jsonl, args.out_dir, steps=50, group_size=8, kl_beta=0.05, is_smollm3=True, seed=args.seed)
    manifest = {
        "schema": "nanolm.trackb_phase5b_b7_3b_fixed.v1",
        "base_model": base_id,
        "total_params": unique_p,
        "active_params": unique_p,
        "method": "Stratified_32B_120B_CausalShifted_SelfAnchored_Top64KL_L2Anchored_100pct_FFT_plus_G8_GRPO",
        "self_kd_alpha": 0.35,
        "mean_epoch_ce": round(ce_sum / n_ord, 4),
        "mean_epoch_self_kl": round(kl_sum / n_ord, 4),
        "self_kd_seconds": round(kd_s, 2),
        "anchored_fft_seconds": round(fft_wall, 2),
        **grpo_stats,
    }
    (Path(args.out_dir) / "nanolm_training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SAVED_NANOLM_B7_3B_FIXED", args.out_dir, json.dumps(manifest), flush=True)


def build_causal_completion_augmentations() -> list[dict[str, Any]]:
    """Build strictly disjoint, shield-verified intermediate `ok: true -> next_tool` and grounded abstention trajectories.

    Eliminates the Phase-3..5B spurious shortcut where 100% of `ok: true` tool observations in
    `stratified_b6.jsonl` were terminal `{"tool_calls": [], "response": "..."}` examples (which caused
    distilled students to stop after Turn-0 read-only lookups or Turn-2 token refreshes!), and preserves
    `FINAL_ANSWER: INSUFFICIENT_EVIDENCE` epistemic abstention when retrieved specs omit an attribute.
    """
    schema_txt = json.dumps(DISJOINT_RECOVERY_TOOLS, indent=2)
    cands: list[dict[str, Any]] = []

    # A. 24 Read-Lookup (`ok: true`) -> Mutating Action (`update_dns_record_ttl`) transitions
    for i in range(24):
        dom = f"gateway-{600 + i}.mesh.example"
        zcode = f"ZONE-AP-{600 + i}"
        cur_ttl = 1600 + (i % 6) * 400
        frac_label, div = ("25%", 4) if i % 2 == 0 else ("50%", 2)
        tgt_ttl = cur_ttl // div
        grant = f"GRANT-INIT-{600 + i}"
        hist = (
            f'Turn 1 Assistant: {{"tool_calls": [{{"name": "fetch_dns_zone_profile", "arguments": {{"domain_name": "{dom}"}}}}]}}\n'
            f'Turn 1 Tool Observation: [{{"tool": "fetch_dns_zone_profile", "ok": true, "zone_code": "{zcode}", "current_ttl_sec": {cur_ttl}, "oauth_grant": "{grant}"}}]'
        )
        p = (
            f"{TOOL_SYSTEM_INSTRUCTION}\n"
            "Read Working Memory & API error messages carefully: never reuse an invalidated/stale parameter, "
            "clamp numeric parameters if an API specifies a safety bound, and halt immediately without retrying "
            "if an API reports a compliance/permission freeze. "
            "When ALL required tool steps are verified complete (or halted due to freeze), "
            'output `{"tool_calls": [], "response": "<final grounded summary with exact IDs and final values>"}`.\n\n'
            f"Available tools:\n```json\n{schema_txt}\n```\n"
            f"Working Memory:\nActive Verified State: zone_code='{zcode}', current_ttl_sec={cur_ttl}, oauth_grant='{grant}'\n\n"
            f"Interaction Log:\n{hist}\n\n"
            f"User Objective: Lower DNS TTL on `{dom}` to `{frac_label}` of its `current_ttl_sec` and confirm once updated."
        )
        resp = json.dumps({"tool_calls": [{"name": "update_dns_record_ttl", "arguments": {"zone_code": zcode, "oauth_grant": grant, "target_ttl_sec": tgt_ttl}}]})
        cands.append({
            "domain": "tool",
            "sub": "multiturn_post_lookup_act",
            "category": "tool_trajectories",
            "think": False,
            "prompt": p,
            "response": resp,
            "verifier_spec": {
                "allowed_tools": ["update_dns_record_ttl"],
                "required_arg_keys": {"update_dns_record_ttl": ["zone_code", "oauth_grant", "target_ttl_sec"]},
                "expected_arg_values": {"update_dns_record_ttl": {"zone_code": zcode, "oauth_grant": grant, "target_ttl_sec": tgt_ttl}},
            },
        })

    # B. 24 Credential-Refresh (`ok: true`) -> Mutating Action (`update_dns_record_ttl` with `fresh_oauth_grant`) transitions
    for i in range(24):
        dom = f"ingress-{700 + i}.edge.example"
        zcode = f"ZONE-SA-{700 + i}"
        cur_ttl = 2400 + (i % 4) * 400
        tgt_ttl = cur_ttl // 4
        stale_g = f"GRANT-STALE-{700 + i}"
        fresh_g = f"GRANT-FRESH-{700 + i}"
        hist = (
            f'Turn 1 Assistant: {{"tool_calls": [{{"name": "fetch_dns_zone_profile", "arguments": {{"domain_name": "{dom}"}}}}]}}\n'
            f'Turn 1 Tool Observation: [{{"tool": "fetch_dns_zone_profile", "ok": true, "zone_code": "{zcode}", "current_ttl_sec": {cur_ttl}, "oauth_grant": "{stale_g}"}}]\n'
            f'Turn 2 Assistant: {{"tool_calls": [{{"name": "update_dns_record_ttl", "arguments": {{"zone_code": "{zcode}", "oauth_grant": "{stale_g}", "target_ttl_sec": {tgt_ttl}}}}}]}}\n'
            f'Turn 2 Tool Observation: [{{"tool": "update_dns_record_ttl", "ok": false, "error": "ExpiredOAuthGrant", "message": "oauth_grant is expired; call rotate_oauth_grant first."}}]\n'
            f'Turn 3 Assistant: {{"tool_calls": [{{"name": "rotate_oauth_grant", "arguments": {{"zone_code": "{zcode}"}}}}]}}\n'
            f'Turn 3 Tool Observation: [{{"tool": "rotate_oauth_grant", "ok": true, "zone_code": "{zcode}", "fresh_oauth_grant": "{fresh_g}"}}]'
        )
        p = (
            f"{TOOL_SYSTEM_INSTRUCTION}\n"
            "Read Working Memory & API error messages carefully: never reuse an invalidated/stale parameter, "
            "clamp numeric parameters if an API specifies a safety bound, and halt immediately without retrying "
            "if an API reports a compliance/permission freeze. "
            "When ALL required tool steps are verified complete (or halted due to freeze), "
            'output `{"tool_calls": [], "response": "<final grounded summary with exact IDs and final values>"}`.\n\n'
            f"Available tools:\n```json\n{schema_txt}\n```\n"
            f"Working Memory:\nActive Verified State: zone_code='{zcode}', current_ttl_sec={cur_ttl}, fresh_oauth_grant='{fresh_g}'\n"
            f"Superseded/Invalidated State: oauth_grant='{stale_g}' (INVALID/STALE - DO NOT USE)\n\n"
            f"Interaction Log:\n{hist}\n\n"
            f"User Objective: Lower DNS TTL on `{dom}` to `25%` of its `current_ttl_sec` and report final applied TTL and zone_code."
        )
        resp = json.dumps({"tool_calls": [{"name": "update_dns_record_ttl", "arguments": {"zone_code": zcode, "oauth_grant": fresh_g, "target_ttl_sec": tgt_ttl}}]})
        cands.append({
            "domain": "tool",
            "sub": "multiturn_post_refresh_act",
            "category": "tool_trajectories",
            "think": False,
            "prompt": p,
            "response": resp,
            "verifier_spec": {
                "allowed_tools": ["update_dns_record_ttl"],
                "required_arg_keys": {"update_dns_record_ttl": ["zone_code", "oauth_grant", "target_ttl_sec"]},
                "expected_arg_values": {"update_dns_record_ttl": {"zone_code": zcode, "oauth_grant": fresh_g, "target_ttl_sec": tgt_ttl}},
            },
        })

    # C. 18 Disjoint Epistemic Abstention (`INSUFFICIENT_EVIDENCE`) trajectories
    disjoint_modules = [
        ("cobalt-raid", "CobaltRAID Array Controller", "PCIe hot-swap debounce latency in microseconds"),
        ("quicksilver-ipc", "Quicksilver IPC Ring", "NUMA cross-socket migration threshold"),
        ("amber-kv", "AmberKV Compaction Engine", "Zstandard dictionary retrain interval in hours"),
        ("granite-paxos", "GranitePaxos Consensus", "snapshot chunk encryption cipher"),
        ("velvet-cdn", "VelvetCDN Origin Shield", "HTTP/3 QUIC 0-RTT replay window in seconds"),
        ("phantom-sched", "Phantom Gang Dispatcher", "NVLink partition preemption grace period"),
    ]
    for i in range(18):
        slug, title, missing_field = disjoint_modules[i % len(disjoint_modules)]
        if i % 3 == 0:
            p = (
                "No verified document in persistent memory matched this query above the confidence threshold.\n"
                "If the question asks for a specific private/unseen specification fact not in general math/science, "
                "output `FINAL_ANSWER: INSUFFICIENT_EVIDENCE`; otherwise answer carefully.\n\n"
                f"Question: Based strictly on `{slug}` internal specifications, what is its `{missing_field}`?"
            )
        else:
            p = (
                "Use ONLY the retrieved persistent-memory evidence below to answer. "
                "If the retrieved evidence does NOT contain the exact answer requested, output `FINAL_ANSWER: INSUFFICIENT_EVIDENCE`.\n\n"
                f"Retrieved Evidence:\n[DOC-{slug.upper()}-V2 | status=ACTIVE] `{title}` defines `queue_slots = {512 * ((i % 4) + 1)}` and `flush_ms = {15 + i}`.\n\n"
                f"Question: From verified `{slug}` docs, state its `{missing_field}` (or abstain with `FINAL_ANSWER: INSUFFICIENT_EVIDENCE`)."
            )
        cands.append({
            "domain": "general_verified",
            "sub": "epistemic_abstain_preserve",
            "category": "epistemic_abstention",
            "think": False,
            "prompt": p,
            "response": "The retrieved specification does not define that attribute.\nFINAL_ANSWER: INSUFFICIENT_EVIDENCE",
            "verifier_spec": {},
        })

    shield = collect_eval_shield_ngrams()
    kept, stats = filter_verified_trajectories(cands, shield)
    assert stats["contaminated_dropped"] == 0, f"Decontamination shield caught overlap: {stats}"
    assert len(kept) == len(cands), f"Verification failed on synthetic causal completion items: {stats}"
    return kept


def _apply_wise_ft_interpolation(ckpt_dir: str, base_model_id: str, base_weight: float = 0.30) -> None:
    """WiSE-FT weight-space convex combination: theta_final = (1 - base_weight)*theta_post + base_weight*theta_0."""
    dev = torch.device("cpu")
    post_m = AutoModelForCausalLM.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    base_m = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    with torch.no_grad():
        for p_post, p_base in zip(post_m.parameters(), base_m.parameters()):
            p_post.lerp_(p_base, base_weight)
    post_m.save_pretrained(ckpt_dir)
    del post_m, base_m
    gc.collect()


def run_train_b8_3b(args) -> None:
    """Train NanoLM-B8-3B (`3.075B` < 4.0B on `HuggingFaceTB/SmolLM3-3B`):
    Causal-Transition-Completed Stratified `32B+120B` Curriculum + Causally-Shifted Self-Anchored Top-64+Tail KL
    (`alpha_kd=0.40`, `T=1.5`) + Proximal L2 100% FFT + `G=8` Verifiable `GRPO` + `WiSE-FT` (`base_weight=0.30`).
    """
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    rows = [json.loads(ln) for ln in Path(args.data_jsonl).read_text().splitlines() if ln.strip()]
    aug = build_causal_completion_augmentations()
    all_rows = rows + aug
    base_id = "HuggingFaceTB/SmolLM3-3B"
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    raw_encoded = [x for r in all_rows if (x := _encode_smollm3_example(tok, r["prompt"], r["response"], bool(r["think"]), max_len=1536)) is not None]
    encoded = [{"input_ids": ex["input_ids"][:-1], "labels": ex["labels"][1:]} for ex in raw_encoded]
    print(f"[B8-3B] Encoded & causally shifted {len(encoded)}/{len(all_rows)} ({len(aug)} causal-completion aug) trajectories", flush=True)

    t0_kd = time.perf_counter()
    self_ref = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    self_ref.eval()
    cached_self_kd = []
    with torch.no_grad():
        for ex in encoded:
            tl = self_ref(ex["input_ids"].unsqueeze(0).to(dev), use_cache=False).logits
            ik, lpk = extract_topk_teacher_targets(tl, k=64, temperature=1.5)
            cached_self_kd.append((ik.squeeze(0).cpu(), lpk.squeeze(0).cpu()))
    del self_ref
    gc.collect()
    torch.cuda.empty_cache()
    kd_s = time.perf_counter() - t0_kd

    student = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    anchor_params = [p.detach().clone() for p in student.parameters()]
    for p in student.parameters():
        p.requires_grad_(True)
    student.gradient_checkpointing_enable()

    lr = 1.0e-5
    anchor_strength = 0.08
    opt = torch.optim.AdamW(list(student.parameters()), lr=lr, weight_decay=0.0)
    total_updates = max(1, len(encoded) // 8)
    warmup_updates = max(1, int(0.12 * total_updates))

    t0 = time.perf_counter()
    rng = random.Random(args.seed)
    order = list(range(len(encoded)))
    rng.shuffle(order)
    upd = 0
    ce_sum, kl_sum = 0.0, 0.0
    student.train()
    for step_i, idx_ex in enumerate(order, 1):
        ex = encoded[idx_ex]
        ik, lpk = cached_self_kd[idx_ex]
        sl = student(ex["input_ids"].unsqueeze(0).to(dev), use_cache=False).logits
        out = proper_topk_kd_loss(sl, ik.unsqueeze(0).to(dev), lpk.unsqueeze(0).to(dev), ex["labels"].unsqueeze(0).to(dev), temperature=1.5, alpha_kd=0.40)
        (out["loss"] / 8.0).backward()
        ce_sum += float(out["ce"].item())
        kl_sum += float(out["kd_kl"].item())
        if step_i % 8 == 0 or step_i == len(order):
            upd += 1
            lr_mult = upd / warmup_updates if upd <= warmup_updates else 0.15 + 0.425 * (1.0 + math.cos(math.pi * min(1.0, (upd - warmup_updates) / max(1, total_updates - warmup_updates))))
            for g in opt.param_groups:
                g["lr"] = lr * lr_mult
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                pull = min(0.02, anchor_strength * lr_mult * 0.01)
                for p, p0 in zip(student.parameters(), anchor_params):
                    p.lerp_(p0, pull)

    mid_dir = Path(args.out_dir + "_self_anchored_fft")
    mid_dir.mkdir(parents=True, exist_ok=True)
    student.gradient_checkpointing_disable()
    unique_p = sum(p.numel() for p in student.parameters())
    student.save_pretrained(mid_dir)
    tok.save_pretrained(mid_dir)
    fft_wall = time.perf_counter() - t0
    del student, anchor_params, opt
    gc.collect()
    torch.cuda.empty_cache()

    grpo_stats = _run_grpo_stage(str(mid_dir), args.rlvr_jsonl, args.out_dir, steps=50, group_size=8, kl_beta=0.05, is_smollm3=True, seed=args.seed)
    # Apply WiSE-FT weight-space convex combination (0.70 B8_post + 0.30 SmolLM3-3B_base)!
    _apply_wise_ft_interpolation(args.out_dir, base_id, base_weight=0.30)
    n_ord = max(1, len(order))
    manifest = {
        "schema": "nanolm.trackb_phase5c_b8_3b.v1",
        "base_model": base_id,
        "total_params": unique_p,
        "active_params": unique_p,
        "method": "CausalCompleted_Stratified_32B_120B_SelfAnchored_Top64KL_L2FFT_G8_GRPO_plus_WiSE_FT_0p30",
        "self_kd_alpha": 0.40,
        "wise_ft_base_weight": 0.30,
        "causal_completion_aug_count": len(aug),
        "mean_epoch_ce": round(ce_sum / n_ord, 4),
        "mean_epoch_self_kl": round(kl_sum / n_ord, 4),
        "self_kd_seconds": round(kd_s, 2),
        "anchored_fft_seconds": round(fft_wall, 2),
        **grpo_stats,
    }
    (Path(args.out_dir) / "nanolm_training_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("SAVED_NANOLM_B8_3B", args.out_dir, json.dumps(manifest), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["generate_stratified", "train_b6_2b", "train_b6_3b", "train_b7_3b", "train_b8_3b"])
    ap.add_argument("--data-dir", default="/workspace/phase3_data")
    ap.add_argument("--data-jsonl", default="/workspace/phase3_data/stratified_b6.jsonl")
    ap.add_argument("--rlvr-jsonl", default="/workspace/phase3_data/rlvr_b6.jsonl")
    ap.add_argument("--out-dir", default="/workspace/models/nanolm_b6_2b")
    ap.add_argument("--subset-size", type=int, default=480)
    ap.add_argument("--skip-120b-teacher", action="store_true")
    ap.add_argument("--seed", type=int, default=3031)
    args = ap.parse_args()

    if args.phase == "generate_stratified":
        run_generate_stratified(args)
    elif args.phase == "train_b6_2b":
        run_train_b6_2b(args)
    elif args.phase == "train_b6_3b":
        run_train_b6_3b(args)
    elif args.phase == "train_b7_3b":
        run_train_b7_3b(args)
    else:
        run_train_b8_3b(args)


if __name__ == "__main__":
    main()
