"""Track B Verified Dual-Mode On-Policy Teacher Distillation (`B3`: `Qwen/Qwen3-8B -> Qwen/Qwen3-1.7B`).

Fixes the B1/B2 off-policy format-shift and `think=1` regression by:
  1. Phase `generate` (vLLM): Generating natural on-policy `Qwen/Qwen3-8B` trajectories across ~600
     unique, non-repeating prompts spanning 6 disjoint tool-calling API banks (single, parallel,
     When2Call abstention, multi-turn error recovery), strict arithmetic JSON, `mbpp` `train`-split
     Python coding (with explicit function signatures), `gsm8k` `train`-split natural `\\boxed{}` math,
     deterministic instruction-following constraints, Mermaid diagrams, and grounded epistemic QA —
     split 70% `think=0` / 30% bounded `think=1` (`<= 1024` tokens, requiring closed `</think>`).
  2. Deterministic Verifier + 13-gram Benchmark Shield: Rejecting every teacher mistake, unclosed
     `<think>` trace, or 13-gram overlap before training.
  3. Phase `distill` (PyTorch): Distilling `Qwen/Qwen3-1.7B` with reversible `rsLoRA (r=16, alpha=16)`,
     `alpha_kd=0.85` Proper Top-64 + Tail-Bucket KL Divergence (`T=1.5`), `lr=5e-5`, `1` epoch.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lab.trackb.eval_capabilities import TOOL_SYSTEM_INSTRUCTION
from lab.trackb.posttrain_pipeline import (
    attach_rslora,
    collect_eval_shield_ngrams,
    extract_topk_teacher_targets,
    filter_verified_trajectories,
    proper_topk_kd_loss,
)
from lab.trackb.train_nanolm_b1 import TRAIN_TOOL_BANK, save_merged_rslora_checkpoint
from nanolm.eval.validators import verify_mermaid_flowchart

EXTRA_TOOL_BANK = [
    {
        "name": "set_cloud_budget_alert",
        "description": "Configure a monthly spend alert threshold on a cloud project.",
        "parameters": {
            "type": "object",
            "required": ["project_slug", "monthly_usd_cap", "hard_stop"],
            "properties": {
                "project_slug": {"type": "string"},
                "monthly_usd_cap": {"type": "integer"},
                "hard_stop": {"type": "boolean"},
            },
        },
    },
    {
        "name": "trigger_canary_rollout",
        "description": "Deploy a container tag to a canary traffic slice in a region.",
        "parameters": {
            "type": "object",
            "required": ["service", "image_tag", "traffic_pct", "region"],
            "properties": {
                "service": {"type": "string"},
                "image_tag": {"type": "string"},
                "traffic_pct": {"type": "integer"},
                "region": {"type": "string", "enum": ["us-west-2", "eu-central-1", "ap-northeast-1"]},
            },
        },
    },
    {
        "name": "reserve_spectrometer_slot",
        "description": "Book an NMR spectrometer slot for a sample batch.",
        "parameters": {
            "type": "object",
            "required": ["instrument_id", "sample_code", "minutes"],
            "properties": {
                "instrument_id": {"type": "string", "enum": ["NMR-400A", "NMR-600B", "NMR-800C"]},
                "sample_code": {"type": "string"},
                "minutes": {"type": "integer"},
            },
        },
    },
]

ALL_B3_TOOLS = TRAIN_TOOL_BANK + EXTRA_TOOL_BANK


def _fmt_b3_tool_prompt(tools: list[dict[str, Any]], user_msg: str, history: str = "") -> str:
    schema_txt = json.dumps(tools, indent=2)
    hist_block = f"\nPrevious interaction:\n{history}\n" if history else ""
    return f"{TOOL_SYSTEM_INSTRUCTION}\n\nAvailable tools:\n```json\n{schema_txt}\n```{hist_block}\nUser: {user_msg}"


def build_b3_prompt_specs(seed: int = 2026) -> list[dict[str, Any]]:
    """Build ~600 strictly unique prompt + deterministic verifier specs (no synthetic target text)."""
    rng = random.Random(seed)
    specs: list[dict[str, Any]] = []
    hubs = ["ORD-1", "DFW-2", "FRA-1", "NRT-3"]
    regions = ["us-west-2", "eu-central-1", "ap-northeast-1"]
    nmrs = ["NMR-400A", "NMR-600B", "NMR-800C"]
    prios = ["LOW", "MEDIUM", "HIGH"]

    # 1. Single Tool Calling (50 unique prompts across 6 tools)
    for i in range(25):
        sku = f"SKU-{1000 + i * 37}"
        hub = hubs[i % len(hubs)]
        inc = (i % 2 == 0)
        p = _fmt_b3_tool_prompt(
            TRAIN_TOOL_BANK,
            f"Check stock for `{sku}` at distribution hub `{hub}` ({'including' if inc else 'excluding'} reserved stock).",
        )
        specs.append({
            "domain": "tool", "sub": "tool_single", "prompt": p, "think": (i % 4 == 0),
            "verifier_spec": {
                "allowed_tools": ["fetch_warehouse_stock"],
                "required_arg_keys": {"fetch_warehouse_stock": ["sku", "hub_code", "include_reserved"]},
                "expected_arg_values": {"fetch_warehouse_stock": {"sku": sku, "hub_code": hub, "include_reserved": inc}},
            },
        })
    for i in range(25):
        svc = f"svc-{['auth','billing','search','ingest','router'][i % 5]}-{i}"
        tag = f"v2.{i}.4"
        pct = [5, 10, 15, 20, 25][i % 5]
        reg = regions[i % len(regions)]
        p = _fmt_b3_tool_prompt(
            EXTRA_TOOL_BANK,
            f"Roll out `{svc}` image `{tag}` to `{pct}%` canary traffic in `{reg}`.",
        )
        specs.append({
            "domain": "tool", "sub": "tool_single", "prompt": p, "think": (i % 4 == 0),
            "verifier_spec": {
                "allowed_tools": ["trigger_canary_rollout"],
                "required_arg_keys": {"trigger_canary_rollout": ["service", "image_tag", "traffic_pct", "region"]},
                "expected_arg_values": {"trigger_canary_rollout": {"service": svc, "image_tag": tag, "traffic_pct": pct, "region": reg}},
            },
        })

    # 2. Parallel Multi-Tool Calling (45 unique prompts)
    for i in range(25):
        sku = f"PART-{2100 + i * 29}"
        hub = hubs[i % len(hubs)]
        asset = f"SUB-{100 + i}"
        dur = 2 + (i % 9)
        pr = prios[i % 3]
        p = _fmt_b3_tool_prompt(
            TRAIN_TOOL_BANK,
            f"Do both actions in parallel: check warehouse stock for `{sku}` at `{hub}` (include reserved) AND schedule a {dur}-hour `{pr}` maintenance window on `{asset}` with ops notification enabled.",
        )
        specs.append({
            "domain": "tool", "sub": "tool_parallel", "prompt": p, "think": (i % 3 == 0),
            "verifier_spec": {
                "allowed_tools": ["fetch_warehouse_stock", "schedule_maintenance_job"],
                "required_arg_keys": {
                    "fetch_warehouse_stock": ["sku", "hub_code", "include_reserved"],
                    "schedule_maintenance_job": ["asset_id", "duration_hours", "notify_ops"],
                },
                "expected_arg_values": {
                    "fetch_warehouse_stock": {"sku": sku, "hub_code": hub, "include_reserved": True},
                    "schedule_maintenance_job": {"asset_id": asset, "duration_hours": dur, "notify_ops": True},
                },
            },
        })
    for i in range(20):
        proj = f"ml-proj-{30 + i}"
        cap = 400 + i * 50
        inst = nmrs[i % len(nmrs)]
        samp = f"CMPD-{800 + i}"
        mins = 30 + (i % 4) * 15
        p = _fmt_b3_tool_prompt(
            EXTRA_TOOL_BANK,
            f"Set a monthly cloud budget alert on `{proj}` at ${cap} with hard_stop=true, and also reserve `{inst}` for sample `{samp}` for {mins} minutes.",
        )
        specs.append({
            "domain": "tool", "sub": "tool_parallel", "prompt": p, "think": (i % 3 == 0),
            "verifier_spec": {
                "allowed_tools": ["set_cloud_budget_alert", "reserve_spectrometer_slot"],
                "required_arg_keys": {
                    "set_cloud_budget_alert": ["project_slug", "monthly_usd_cap", "hard_stop"],
                    "reserve_spectrometer_slot": ["instrument_id", "sample_code", "minutes"],
                },
                "expected_arg_values": {
                    "set_cloud_budget_alert": {"project_slug": proj, "monthly_usd_cap": cap, "hard_stop": True},
                    "reserve_spectrometer_slot": {"instrument_id": inst, "sample_code": samp, "minutes": mins},
                },
            },
        })

    # 3. When2Call Abstention (45 unique prompts: missing required args, unsupported action, general STEM QA)
    missing_queries = [
        f"Check inventory at `{hubs[i % 4]}` right away — I will look up the SKU code later." for i in range(10)
    ] + [
        f"Trigger a canary rollout for `checkout-v{i}` in `us-west-2`, though we haven't decided the image_tag or traffic_pct yet." for i in range(8)
    ] + [
        f"Reserve spectrometer `{nmrs[i % 3]}` for 45 minutes; sample code is still pending assignment #{i}." for i in range(7)
    ]
    unsupported_and_qa = [
        f"Delete production SQL table `orders_archive_{2015 + i}` permanently to free disk space." for i in range(8)
    ] + [
        f"Explain why eigenvalues of a real symmetric {3 + i}x{3 + i} matrix are guaranteed to be real." for i in range(6)
    ] + [
        f"Send a Slack message to #ops-oncall summarizing outage ticket INC-{4000 + i}." for i in range(6)
    ]
    for idx, q in enumerate(missing_queries + unsupported_and_qa):
        bank = TRAIN_TOOL_BANK if idx % 2 == 0 else EXTRA_TOOL_BANK
        specs.append({
            "domain": "tool", "sub": "tool_abstain", "prompt": _fmt_b3_tool_prompt(bank, q),
            "think": (idx % 4 == 0), "verifier_spec": {"expect_abstain": True},
        })

    # 4. Multi-Turn API Error Recovery (40 unique prompts)
    for i in range(16):
        sku = f"ITEM-{500 + i * 13}"
        bad_hub, good_hub = [("CHI-1", "ORD-1"), ("DAL-9", "DFW-2"), ("FFM-0", "FRA-1"), ("TYO-2", "NRT-3")][i % 4]
        hist = (
            f'Assistant called: {{"name": "fetch_warehouse_stock", "arguments": {{"sku": "{sku}", "hub_code": "{bad_hub}", "include_reserved": false}}}}\n'
            f'Tool error: {{"error": "DeprecatedHub", "message": "Hub {bad_hub} migrated to {good_hub}."}}'
        )
        p = _fmt_b3_tool_prompt(TRAIN_TOOL_BANK, "Retry against the migrated hub and set include_reserved to true.", history=hist)
        specs.append({
            "domain": "tool", "sub": "tool_recovery", "prompt": p, "think": (i % 3 == 0),
            "verifier_spec": {
                "allowed_tools": ["fetch_warehouse_stock"],
                "required_arg_keys": {"fetch_warehouse_stock": ["sku", "hub_code", "include_reserved"]},
                "expected_arg_values": {"fetch_warehouse_stock": {"sku": sku, "hub_code": good_hub, "include_reserved": True}},
            },
        })
    for i in range(14):
        svc = f"edge-proxy-{i}"
        tag = f"v3.1.{i}"
        reg = regions[i % 3]
        hist = (
            f'Assistant called: {{"name": "trigger_canary_rollout", "arguments": {{"service": "{svc}", "image_tag": "{tag}", "traffic_pct": 60, "region": "{reg}"}}}}\n'
            f'Tool error: {{"error": "CanaryLimitExceeded", "message": "Maximum allowed canary traffic_pct is 25."}}'
        )
        p = _fmt_b3_tool_prompt(EXTRA_TOOL_BANK, "Re-issue the canary rollout clamped to the maximum allowed percentage.", history=hist)
        specs.append({
            "domain": "tool", "sub": "tool_recovery", "prompt": p, "think": (i % 3 == 0),
            "verifier_spec": {
                "allowed_tools": ["trigger_canary_rollout"],
                "required_arg_keys": {"trigger_canary_rollout": ["service", "image_tag", "traffic_pct", "region"]},
                "expected_arg_values": {"trigger_canary_rollout": {"service": svc, "image_tag": tag, "traffic_pct": 25, "region": reg}},
            },
        })
    for i in range(10):
        tbl = f"payroll_shard_{i + 1}"
        hist = (
            f'Assistant called: {{"name": "run_sql_readonly_audit", "arguments": {{"table": "{tbl}", "agg_column": "net_pay", "limit_rows": 1000}}}}\n'
            f'Tool error: {{"error": "PermissionDenied", "message": "Access to {tbl} is restricted; do not retry."}}'
        )
        p = _fmt_b3_tool_prompt(TRAIN_TOOL_BANK, "Report the error clearly to the user without invoking any tool.", history=hist)
        specs.append({
            "domain": "tool", "sub": "tool_recovery_stop", "prompt": p, "think": (i % 3 == 0),
            "verifier_spec": {"expect_abstain": True},
        })

    # 5. Strict Arithmetic-in-JSON & Schema Normalization (75 unique prompts across 3 disjoint schemas)
    for i in range(25):
        jid = f"batch-{['vega','altair','deneb','rigel','sirius'][i % 5]}-{100 + i}"
        ok_n = rng.randint(600, 9200)
        err_n = rng.randint(15, 420)
        sec = round(rng.uniform(3.5, 95.4), 1)
        abt = (i % 3 == 0)
        p = (
            "Respond with ONLY a valid JSON object (no prose outside JSON) containing exact keys "
            "`job_id` (string), `processed_total` (int, sum of succeeded and failed), `failed_count` (int), "
            "`duration_sec` (float), and `aborted` (bool):\n"
            f"Batch `{jid}` recorded {ok_n} succeeded tasks and {err_n} failed tasks over {sec}s; "
            f"status={'aborted' if abt else 'completed normally'}."
        )
        specs.append({
            "domain": "json_exact", "sub": "json_arith", "prompt": p, "think": (i % 3 == 0),
            "verifier_spec": {"expected_obj": {"job_id": jid, "processed_total": ok_n + err_n, "failed_count": err_n, "duration_sec": sec, "aborted": abt}},
        })
    for i in range(25):
        acct = f"ACCT-{300 + i * 7}"
        gross = rng.randint(2500, 18000)
        refund = rng.randint(120, 1400)
        tax = rng.randint(90, 950)
        settled = (i % 2 == 1)
        net = gross - refund - tax
        p = (
            "Return ONLY a JSON object with exact keys `account` (string), `gross_usd` (int), "
            "`net_usd` (int, defined as gross minus refunds minus tax), and `settled` (bool):\n"
            f"Ledger `{acct}` booked ${gross} gross revenue, ${refund} in refunds, and ${tax} withholding tax; "
            f"settlement is {'complete' if settled else 'pending'}."
        )
        specs.append({
            "domain": "json_exact", "sub": "json_ledger", "prompt": p, "think": (i % 3 == 0),
            "verifier_spec": {"expected_obj": {"account": acct, "gross_usd": gross, "net_usd": net, "settled": settled}},
        })
    for i in range(25):
        cluster = f"k8s-{['ewr','ams','sin','iad','fra'][i % 5]}-{i + 1}"
        used = rng.randint(24, 192)
        free = rng.randint(16, 128)
        cordon = (i % 4 == 0)
        p = (
            "Output ONLY raw JSON with exact keys `cluster_id` (string), `total_gpus` (int, used + idle), "
            "`idle_gpus` (int), and `cordoned` (bool):\n"
            f"Cluster `{cluster}` currently has {used} busy GPUs and {free} idle GPUs; node pool is {'cordoned' if cordon else 'schedulable'}."
        )
        specs.append({
            "domain": "json_exact", "sub": "json_k8s", "prompt": p, "think": (i % 4 == 0),
            "verifier_spec": {"expected_obj": {"cluster_id": cluster, "total_gpus": used + free, "idle_gpus": free, "cordoned": cordon}},
        })

    # 6. Unique Algorithmic Python Coding from `mbpp` train+validation splits (CC-BY-4.0, strictly non-test!)
    try:
        mbpp_rows = list(load_dataset("google-research-datasets/mbpp", "full", split="train")) + list(
            load_dataset("google-research-datasets/mbpp", "full", split="validation")
        )
        fn_re = re.compile(r"def\s+([A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\))\s*:")
        added_code = 0
        for row in mbpp_rows:
            if added_code >= 115:
                break
            m = fn_re.search(row["code"])
            tests = row.get("test_list") or []
            if not m or not tests:
                continue
            sig = m.group(1)
            prompt = (
                f"{row['text'].strip()}\n"
                f"Implement the solution in Python using the exact function signature `def {sig}:` inside a ```python``` code block."
            )
            harness = "\n".join(tests)
            specs.append({
                "domain": "code", "sub": "mbpp_train", "prompt": prompt,
                "think": (added_code % 4 == 0),
                "verifier_spec": {"harness": harness},
            })
            added_code += 1
    except Exception as exc:
        print(f"MBPP train load warning: {exc}", flush=True)

    # 7. MIT GSM8K `train` Split Natural `\boxed{}` Math Reasoning (145 unique problems!)
    try:
        ds_gsm = load_dataset("openai/gsm8k", "main", split="train[200:360]")
        for idx, row in enumerate(ds_gsm):
            if "####" not in row["answer"]:
                continue
            _, final_num = row["answer"].split("####", 1)
            final_clean = final_num.strip().replace(",", "")
            prompt = f"{row['question'].strip()}\nShow concise step-by-step reasoning and put your final numerical answer in \\boxed{{}}."
            specs.append({
                "domain": "math", "sub": "gsm8k_natural_boxed", "prompt": prompt,
                "think": (idx % 3 == 0),
                "verifier_spec": {"expected": final_clean},
            })
    except Exception as exc:
        print(f"GSM8K train load warning: {exc}", flush=True)

    # 8. Deterministic Instruction-Following Constraints, Mermaid Diagrams & Grounded Epistemic QA (80 unique prompts)
    topics = [
        ("zero-copy networking", "kernel"), ("columnar compression", "disk"), ("B-tree page splits", "mutex"),
        ("consistent hashing", "central"), ("write-ahead logging", "crash"), ("bloom filters", "deletion"),
        ("SIMD vectorization", "scalar"), ("connection pooling", "handshake"), ("raft leader leases", "quorum"),
        ("generational GC", "heap"), ("RDMA verbs", "socket"), ("speculative decoding", "draft"),
        ("KV-cache paging", "fragmentation"), ("flash attention tiling", "HBM"), ("rotary embeddings", "angle"),
        ("int8 quantization", "outlier"), ("ring all-reduce", "bandwidth"), ("token bucket rate limiting", "burst"),
        ("LSM-tree compaction", "tombstone"), ("cuckoo hashing", "eviction"),
    ]
    for i, (topic, forbidden) in enumerate(topics):
        n_bul = 3 + (i % 3)
        p = (
            f"Explain `{topic}` using EXACTLY {n_bul} markdown bullet points starting with `- `. "
            f"Do NOT use the word `{forbidden}` (case-insensitive) anywhere in your response, and end with `P.S. Verified.`"
        )
        def _make_bul_chk(nb: int, bad: str) -> Callable[[str], bool]:
            def _chk(txt: str) -> bool:
                bullets = [ln for ln in txt.splitlines() if ln.strip().startswith("- ")]
                return len(bullets) == nb and (bad.lower() not in txt.lower()) and txt.strip().endswith("P.S. Verified.")
            return _chk
        specs.append({
            "domain": "constraint", "sub": "ifeval_bullets_forbidden", "prompt": p,
            "think": (i % 4 == 0), "verifier_spec": {"check_fn": _make_bul_chk(n_bul, forbidden)},
        })

    for i in range(20):
        subj = [
            "mercury barometer calibration", "tidal turbine blade pitch", "solid-state battery anodes",
            "geostationary orbit stationkeeping", "supercritical CO2 extraction", "ultraviolet lithography pellicles",
            "piezoelectric quartz resonators", "cryogenic nitrogen distillation", "optical fiber dispersion",
            "eddy-current rail braking", "desalination reverse osmosis", "magnetotelluric crustal sounding",
            "synthetic aperture radar", "proton exchange membranes", "acoustic Doppler velocimetry",
            "laser interferometer arms", "phase-change thermal storage", "ion-trap mass spectrometry",
            "induction motor slip", "aerogel thermal conductivity",
        ][i]
        max_w = 45 + (i % 3) * 10
        p = (
            f"Write a brief technical note on `{subj}` that starts with a title wrapped in double angular brackets "
            f"like `<<Title>>` on the first line, contains at most {max_w} words total, and includes no markdown bold (`**`)."
        )
        def _make_title_chk(mw: int) -> Callable[[str], bool]:
            def _chk(txt: str) -> bool:
                lines = [ln.strip() for ln in txt.strip().splitlines() if ln.strip()]
                if not lines or not (lines[0].startswith("<<") and lines[0].endswith(">>") and len(lines[0]) > 5):
                    return False
                return len(txt.split()) <= mw and "**" not in txt
            return _chk
        specs.append({
            "domain": "constraint", "sub": "ifeval_title_wordcap", "prompt": p,
            "think": (i % 4 == 0), "verifier_spec": {"check_fn": _make_title_chk(max_w)},
        })

    for i in range(20):
        s_a, s_b, s_c = f"Ingest{i}", f"Validate{i}", f"Archive{i}"
        p = (
            f"Create a ```mermaid``` `flowchart LR` diagram where `{s_a}` connects to `{s_b}` labeled `clean`, "
            f"and `{s_b}` connects to `{s_c}` labeled `commit`."
        )
        def _make_merm_chk(a: str, b: str, c: str) -> Callable[[str], bool]:
            def _chk(txt: str) -> bool:
                _, sem_ok, _ = verify_mermaid_flowchart(txt, [(a, b, "clean"), (b, c, "commit")])
                return sem_ok
            return _chk
        specs.append({
            "domain": "constraint", "sub": "mermaid_lr", "prompt": p,
            "think": (i % 4 == 0), "verifier_spec": {"check_fn": _make_merm_chk(s_a, s_b, s_c)},
        })

    for i in range(20):
        svc = f"gateway-{chr(65 + (i % 12))}{i}"
        lat1, lat2 = 140 + i * 5, 45 + i * 2
        p = (
            "Answer strictly from the context below. If the context does not state the requested fact, "
            "output `FINAL_ANSWER: INSUFFICIENT_EVIDENCE`.\n\n"
            f"Context: Service `{svc}` cut median latency from {lat1} ms to {lat2} ms after switching to HTTP/3.\n"
            f"Question: How many CPU cores are allocated to `{svc}`?"
        )
        specs.append({
            "domain": "math", "sub": "epistemic_abstain", "prompt": p,
            "think": (i % 3 == 0), "verifier_spec": {"expected": "INSUFFICIENT_EVIDENCE"},
        })

    return specs


def run_generate_phase(teacher_name: str, out_jsonl: Path, seed: int = 2026) -> None:
    from vllm import LLM, SamplingParams

    specs = build_b3_prompt_specs(seed=seed)
    tok = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=False)

    rendered_prompts: list[str] = []
    for sp in specs:
        msgs = [{"role": "user", "content": sp["prompt"]}]
        rendered_prompts.append(
            tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=bool(sp["think"]))
        )

    print(f"Launching vLLM on {teacher_name} for {len(specs)} unique prompts...", flush=True)
    llm = LLM(
        model=teacher_name,
        dtype="bfloat16",
        gpu_memory_utilization=0.88,
        max_model_len=4096,
        trust_remote_code=False,
    )
    # Split think=0 (max_tokens=768) and think=1 (max_tokens=1024) for clean bounded generation
    idx_t0 = [i for i, sp in enumerate(specs) if not sp["think"]]
    idx_t1 = [i for i, sp in enumerate(specs) if sp["think"]]
    raw_outputs = [""] * len(specs)

    t0 = time.perf_counter()
    if idx_t0:
        outs0 = llm.generate([rendered_prompts[i] for i in idx_t0], SamplingParams(temperature=0.0, max_tokens=768))
        for i, o in zip(idx_t0, outs0):
            raw_outputs[i] = o.outputs[0].text
    if idx_t1:
        outs1 = llm.generate([rendered_prompts[i] for i in idx_t1], SamplingParams(temperature=0.0, max_tokens=1024))
        for i, o in zip(idx_t1, outs1):
            raw_outputs[i] = o.outputs[0].text
    gen_s = time.perf_counter() - t0
    print(f"Teacher generated {len(specs)} completions in {gen_s:.1f}s.", flush=True)

    # Verify & 13-gram shield
    shield = collect_eval_shield_ngrams()
    candidates = []
    for sp, raw_resp in zip(specs, raw_outputs):
        vspec = dict(sp["verifier_spec"])
        if sp["think"]:
            vspec["expect_think"] = True
        candidates.append({
            "domain": sp["domain"],
            "sub": sp["sub"],
            "think": bool(sp["think"]),
            "prompt": sp["prompt"],
            "response": raw_resp,
            "verifier_spec": vspec,
        })

    kept, stats = filter_verified_trajectories(candidates, shield)
    by_sub: dict[str, int] = {}
    think_kept = 0
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for ex in kept:
            by_sub[ex["sub"]] = by_sub.get(ex["sub"], 0) + 1
            if ex["think"]:
                think_kept += 1
            f.write(json.dumps({
                "domain": ex["domain"],
                "sub": ex["sub"],
                "think": ex["think"],
                "prompt": ex["prompt"],
                "response": ex["response"],
            }) + "\n")

    summary = {
        "teacher": teacher_name,
        "gen_seconds": round(gen_s, 2),
        "filter_stats": stats,
        "kept_think1": think_kept,
        "kept_think0": len(kept) - think_kept,
        "kept_by_sub": by_sub,
    }
    out_jsonl.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print("B3_TEACHER_DATA_SUMMARY:", json.dumps(summary), flush=True)


def encode_dual_mode_example(tok, prompt: str, response: str, think: bool, max_len: int = 1536) -> dict[str, torch.Tensor] | None:
    """Exact prefix-aligned tokenization for both `enable_thinking=False` and `enable_thinking=True`."""
    p_txt = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think,
    )
    f_txt = p_txt + response + "<|im_end|>\n"
    p_ids = tok.encode(p_txt, add_special_tokens=False)
    f_ids = tok.encode(f_txt, add_special_tokens=False)
    if len(f_ids) > max_len or len(f_ids) <= len(p_ids) + 1:
        return None
    input_ids = torch.tensor(f_ids[:-1], dtype=torch.long)
    labels = torch.tensor(f_ids[1:], dtype=torch.long)
    labels[: max(0, len(p_ids) - 1)] = -100
    return {"input_ids": input_ids, "labels": labels}


def run_distill_phase(args) -> None:
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps")
    rows = [json.loads(line) for line in Path(args.data_jsonl).read_text().splitlines() if line.strip()]
    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=False)

    encoded = [
        x for r in rows
        if (x := encode_dual_mode_example(tok, r["prompt"], r["response"], bool(r["think"]), max_len=args.max_len)) is not None
    ]
    print(f"Encoded {len(encoded)}/{len(rows)} verified on-policy dual-mode trajectories.", flush=True)

    # 1. Extract Top-K + Tail teacher logprobs first while GPU is empty, then free teacher VRAM completely
    t_kd0 = time.perf_counter()
    print(f"Loading teacher {args.teacher} for Top-{args.topk} + Tail logit extraction...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    teacher.eval()
    cached_kd: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for item in encoded:
            inp = item["input_ids"].unsqueeze(0).to(dev)
            t_logits = teacher(inp, use_cache=False).logits
            idx_k, lp_k = extract_topk_teacher_targets(t_logits, k=args.topk, temperature=args.temperature)
            cached_kd.append((idx_k.squeeze(0).cpu(), lp_k.squeeze(0).cpu()))
    del teacher
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    kd_s = time.perf_counter() - t_kd0
    print(f"Cached Top-{args.topk} teacher logprobs for {len(encoded)} items in {kd_s:.1f}s.", flush=True)

    # 2. Load student + reversible rsLoRA and distill
    student = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.bfloat16, trust_remote_code=False).to(dev)
    trainable_params = attach_rslora(student, rank=args.rank, alpha=float(args.rank))
    total_params = sum(p.numel() for p in student.parameters())
    print(f"Student loaded: {total_params:,} unique params ({trainable_params:,} rsLoRA trainable)", flush=True)

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    total_updates = max(1, (len(encoded) * args.epochs) // args.grad_accum)
    warmup_updates = max(1, int(0.10 * total_updates))

    student.train()
    t0 = time.perf_counter()
    step_idx, upd_idx = 0, 0
    history = []
    rng = random.Random(args.seed)
    indices = list(range(len(encoded)))

    for ep in range(args.epochs):
        rng.shuffle(indices)
        ep_ce, ep_kl, ep_n = 0.0, 0.0, 0
        for idx_item in indices:
            ex = encoded[idx_item]
            tk_i, tk_lp = cached_kd[idx_item]
            s_logits = student(ex["input_ids"].unsqueeze(0).to(dev), use_cache=False).logits
            out = proper_topk_kd_loss(
                s_logits,
                tk_i.unsqueeze(0).to(dev),
                tk_lp.unsqueeze(0).to(dev),
                ex["labels"].unsqueeze(0).to(dev),
                temperature=args.temperature,
                alpha_kd=args.alpha_kd,
            )
            (out["loss"] / args.grad_accum).backward()
            ep_ce += float(out["ce"])
            ep_kl += float(out["kd_kl"])
            ep_n += 1
            step_idx += 1

            if step_idx % args.grad_accum == 0 or step_idx == len(encoded) * args.epochs:
                upd_idx += 1
                if upd_idx <= warmup_updates:
                    lr_mult = upd_idx / warmup_updates
                else:
                    prog = (upd_idx - warmup_updates) / max(1, total_updates - warmup_updates)
                    lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
                for g in opt.param_groups:
                    g["lr"] = args.lr * lr_mult
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

        rec = {"epoch": ep + 1, "mean_ce": round(ep_ce / ep_n, 4), "mean_kd_kl": round(ep_kl / ep_n, 4), "elapsed_s": round(time.perf_counter() - t0, 1)}
        history.append(rec)
        print("EPOCH", json.dumps(rec), flush=True)

    out_path = Path(args.out_dir)
    save_merged_rslora_checkpoint(student, tok, out_path)
    gen_summary_path = Path(args.data_jsonl).with_suffix(".summary.json")
    gen_summary = json.loads(gen_summary_path.read_text()) if gen_summary_path.exists() else {}
    meta = {
        "schema": "nanolm.trackb_posttrain_run.v2",
        "arm": "B3_verified_onpolicy_kd8b",
        "student": args.student,
        "teacher": args.teacher,
        "alpha_kd": args.alpha_kd,
        "temperature": args.temperature,
        "topk": args.topk,
        "rslora_rank": args.rank,
        "lr": args.lr,
        "epochs": args.epochs,
        "trainable_params": trainable_params,
        "unique_params": total_params,
        "encoded_examples": len(encoded),
        "kd_cache_seconds": round(kd_s, 2),
        "train_wall_seconds": round(time.perf_counter() - t0, 2),
        "teacher_generation_summary": gen_summary,
        "history": history,
    }
    (out_path / "nanolm_training_manifest.json").write_text(json.dumps(meta, indent=2))
    print("SAVED_MERGED_MODEL", str(out_path), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["generate", "distill"])
    ap.add_argument("--student", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--teacher", default="Qwen/Qwen3-8B")
    ap.add_argument("--data-jsonl", default="/workspace/data/b3_verified_onpolicy.jsonl")
    ap.add_argument("--out-dir", default="/workspace/models/nanolm_b3_onpolicy_kd8b_2b")
    ap.add_argument("--alpha-kd", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=1.5)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if args.phase == "generate":
        run_generate_phase(args.teacher, Path(args.data_jsonl), seed=args.seed)
    else:
        run_distill_phase(args)


if __name__ == "__main__":
    main()
