#!/usr/bin/env python3
"""Execute E-ARCH-01 (6-Arm Controlled Architecture & Objective Ablation) and E-CAP-01 (Multi-Stage Capability & Reference Evaluation)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from nanolm.config import get_preset_config
from nanolm.data import PackedDocumentDataset, build_multidomain_corpus, decontaminate_corpus
from nanolm.eval import collect_all_eval_prompts, run_capability_evaluation
from nanolm.model import NanoLMForCausalLM
from nanolm.tokenizer import audit_tokenizer_efficiency, format_chat_messages, load_tokenizer
from nanolm.train import TrainConfig, detect_best_device, evaluate_dataset_bpb, load_checkpoint, run_training, save_atomic_checkpoint


def prepare_datasets(tok_path: Path):
    tok = load_tokenizer(tok_path)
    eval_prompts = collect_all_eval_prompts()

    train_dom = build_multidomain_corpus(split="train", prose_docs_limit=1500, structured_docs_count=2000, seed=101)
    val_dom = build_multidomain_corpus(split="val", prose_docs_limit=250, structured_docs_count=240, seed=202)

    raw_train = [d for docs in train_dom.values() for d in docs]
    rng = random.Random(42)
    rng.shuffle(raw_train)
    clean_train, decontam = decontaminate_corpus(raw_train, eval_prompts, ngram_size=13)

    val_prose_texts = val_dom["prose"]
    val_multi_texts = [
        d for k, docs in val_dom.items() if k != "prose" for d in docs
    ]

    rep_prose = audit_tokenizer_efficiency(
        "val-prose", lambda s: tok.encode(s).ids, tok.get_vocab_size(), {"prose": val_prose_texts}
    )
    rep_multi = audit_tokenizer_efficiency(
        "val-multi", lambda s: tok.encode(s).ids, tok.get_vocab_size(), {"multi": val_multi_texts}
    )

    train_ids = [tok.encode(d).ids for d in clean_train]
    val_prose_ids = [tok.encode(d).ids for d in val_prose_texts]
    val_multi_ids = [tok.encode(d).ids for d in val_multi_texts]

    seq_len = 256
    ds_train = PackedDocumentDataset(train_ids, seq_len=seq_len, bos_id=0, eos_id=1)
    ds_val_prose = PackedDocumentDataset(val_prose_ids, seq_len=seq_len, bos_id=0, eos_id=1)
    ds_val_multi = PackedDocumentDataset(val_multi_ids, seq_len=seq_len, bos_id=0, eos_id=1)

    return (
        tok,
        decontam,
        ds_train,
        ds_val_prose,
        rep_prose.overall_bytes_per_token,
        ds_val_multi,
        rep_multi.overall_bytes_per_token,
    )


def run_e_arch_01(out_dir: Path) -> tuple[dict[str, object], str, Path]:
    print("\n=== [E-ARCH-01] 6-Arm Controlled Architecture & Training Objective Ablation ===", flush=True)
    tok_path = out_dir / "tokenizers" / "nanolm_bpe_16k.json"
    (
        tok,
        decontam,
        ds_train,
        ds_val_prose,
        bpt_prose,
        ds_val_multi,
        bpt_multi,
    ) = prepare_datasets(tok_path)

    dev = detect_best_device()
    print(
        f"Device={dev} | Packed train blocks={ds_train.num_blocks} ({ds_train.total_tokens:,} unique packed tokens) "
        f"| Val prose blocks={ds_val_prose.num_blocks} | Val multidomain blocks={ds_val_multi.num_blocks}",
        flush=True,
    )

    base_dense = get_preset_config("nanolm-25m-dense")
    arms = [
        (
            "A0_legacy_no_qknorm_cosine",
            replace(base_dense, name="nanolm-25m-no-qknorm", qk_norm=False, mtp_num_heads=0),
            "cosine",
        ),
        (
            "A1_dense_qknorm_wsd",
            replace(base_dense, name="nanolm-25m-dense-wsd", qk_norm=True, mtp_num_heads=0),
            "wsd",
        ),
        (
            "A2_hybrid_swa_mtp_wsd",
            get_preset_config("nanolm-25m-hybrid-swa"),
            "wsd",
        ),
        (
            "A3_mla_latent_kv_wsd",
            get_preset_config("nanolm-25m-mla"),
            "wsd",
        ),
        (
            "A4_moe_finegrained_wsd",
            get_preset_config("nanolm-45m-moe"),
            "wsd",
        ),
        (
            "A5_depth_loop_wsd",
            get_preset_config("nanolm-25m-loop"),
            "wsd",
        ),
    ]

    arm_results: list[dict[str, object]] = []
    best_arm_id = ""
    best_composite_bpb = 1e9
    best_ckpt = Path()

    for arm_id, m_cfg, sched in arms:
        ckpt_dir = out_dir / "runs" / arm_id
        t_cfg = TrainConfig(
            run_id=arm_id,
            max_steps=240,
            micro_batch_size=16,
            grad_accum_steps=1,
            peak_lr=1.2e-3,
            warmup_steps=24,
            schedule=sched,
            decay_fraction=0.25,
            eval_every=60,
            save_every=240,
            seed=42,
            precision="bf16",
            checkpoint_dir=str(ckpt_dir),
        )
        res = run_training(
            model_cfg=m_cfg,
            train_cfg=t_cfg,
            train_dataset=ds_train,
            val_dataset=ds_val_prose,
            val_bytes_per_token=bpt_prose,
            device=dev,
        )

        # Evaluate trained model on BOTH held-out FineWeb-Edu Prose AND held-out Multi-Domain validation sets
        model = NanoLMForCausalLM(m_cfg).to(dev)
        load_checkpoint(res.final_checkpoint_path, model, device=dev)
        dtype = torch.bfloat16 if dev in ("cuda", "mps") else None
        prose_loss, prose_bpb = evaluate_dataset_bpb(
            model, ds_val_prose, bpt_prose, batch_size=16, max_batches=16, device=dev, autocast_dtype=dtype
        )
        multi_loss, multi_bpb = evaluate_dataset_bpb(
            model, ds_val_multi, bpt_multi, batch_size=16, max_batches=16, device=dev, autocast_dtype=dtype
        )
        composite_bpb = 0.5 * prose_bpb + 0.5 * multi_bpb
        acct = m_cfg.parameter_accounting()

        record = {
            "arm_id": arm_id,
            "model_name": m_cfg.name,
            "schedule": sched,
            "deployed_parameters": acct.total_deployed_parameters,
            "active_parameters_per_token": acct.active_parameters_per_token,
            "training_only_parameters": acct.training_only_parameters,
            "kv_cache_gb_at_32k_bf16": acct.kv_cache_gb_at_32k_bf16,
            "tokens_trained": res.tokens_trained,
            "wall_seconds": res.wall_seconds,
            "tokens_per_second": res.tokens_per_second,
            "val_fineweb_prose_loss": round(prose_loss, 4),
            "val_fineweb_prose_bpb": round(prose_bpb, 4),
            "val_multidomain_loss": round(multi_loss, 4),
            "val_multidomain_bpb": round(multi_bpb, 4),
            "composite_bpb": round(composite_bpb, 4),
            "moe_stats": {k: round(v, 4) for k, v in model.moe_load_stats().items()},
            "checkpoint": res.final_checkpoint_path,
        }
        arm_results.append(record)
        print(
            f"  {arm_id:28s} | prose_bpb={prose_bpb:.4f} | multi_bpb={multi_bpb:.4f} "
            f"| comp_bpb={composite_bpb:.4f} | {res.tokens_per_second:6.0f} tok/s | {res.wall_seconds:4.1f}s",
            flush=True,
        )

        if composite_bpb < best_composite_bpb:
            best_composite_bpb = composite_bpb
            best_arm_id = arm_id
            best_ckpt = Path(res.final_checkpoint_path)

        del model
        if dev == "mps":
            torch.mps.empty_cache()

    summary = {
        "experiment_id": "E-ARCH-01",
        "device": dev,
        "decontamination": decontam.to_dict(),
        "best_arm_id": best_arm_id,
        "best_composite_bpb": round(best_composite_bpb, 4),
        "arms": arm_results,
    }
    (out_dir / "e_arch_01_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, best_arm_id, best_ckpt


def build_decontaminated_sft_pairs(seed: int = 777) -> list[tuple[str, str]]:
    """Construct diverse instruction-tuning examples strictly disjoint from held-out eval tasks!

    Notice: Uses completely disjoint function structures, domains, numbers, and names from
    `nanolm.eval.suite.build_heldout_eval_tasks`, then runs 13-gram decontamination as a hard gate.
    """
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []

    # 1. Diverse general algorithmic Python functions (strictly distinct wording from held-out eval!)
    for idx in range(80):
        variant = idx % 4
        if variant == 0:
            thresh = 0 if idx % 2 == 0 else rng.randint(1, 5)
            fn_name = "stable_unique_positive" if thresh == 0 else f"unique_above_{thresh}_{idx}"
            cmp_op = "> 0" if thresh == 0 else f"> {thresh}"
            pairs.append((
                f"Implement `{fn_name}(seq: list[int]) -> list[int]` in Python: filter values `{cmp_op}` "
                f"and retain earliest appearance order with duplicates omitted.",
                f"<|think|>Maintain a `seen` set and keep elements {cmp_op}.<|end_think|>\n"
                f"```python\ndef {fn_name}(seq: list[int]) -> list[int]:\n"
                f"    seen = set()\n    out = []\n    for val in seq:\n"
                f"        if val {cmp_op} and val not in seen:\n"
                f"            seen.add(val)\n            out.append(val)\n"
                f"    return out\n```",
            ))
        elif variant == 1:
            fn_name = "coalesce_windows" if idx % 2 == 0 else f"fuse_spans_{idx}"
            pairs.append((
                f"Create Python helper `{fn_name}(spans: list[tuple[int, int]]) -> list[tuple[int, int]]` "
                f"to sort `[lo, hi]` pairs and combine any intervals where `next_lo <= cur_hi`.",
                f"<|think|>Sort pairs ascending and fuse adjacent intervals when lo <= prev_hi.<|end_think|>\n"
                f"```python\ndef {fn_name}(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:\n"
                f"    if not spans:\n        return []\n"
                f"    ordered = sorted(spans)\n    out = [ordered[0]]\n"
                f"    for lo, hi in ordered[1:]:\n"
                f"        plo, phi = out[-1]\n"
                f"        if lo <= phi:\n            out[-1] = (plo, max(phi, hi))\n"
                f"        else:\n            out.append((lo, hi))\n"
                f"    return out\n```",
            ))
        elif variant == 2:
            fn_name = "is_balanced_brackets" if idx % 2 == 0 else f"valid_nesting_{idx}"
            pairs.append((
                f"Code `{fn_name}(expr: str) -> bool` in Python using a stack to verify matching "
                f"delimiters `()`, `[]`, `{{}}` across arbitrary strings.",
                f"```python\ndef {fn_name}(expr: str) -> bool:\n"
                f"    matching = {{')': '(', ']': '[', '}}': '{{'}}\n    stack = []\n"
                f"    for ch in expr:\n        if ch in '([{{':\n            stack.append(ch)\n"
                f"        elif ch in matching:\n"
                f"            if not stack or stack.pop() != matching[ch]:\n                return False\n"
                f"    return len(stack) == 0\n```",
            ))
        else:
            k = 2 if idx % 2 == 0 else 3
            fn_name = "top_two_tokens" if k == 2 else f"most_common_{k}_{idx}"
            pairs.append((
                f"Provide Python function `{fn_name}(items: list[str]) -> list[str]` returning the {k} "
                f"highest-count strings sorted by count high-to-low then lexicographically.",
                f"```python\ndef {fn_name}(items: list[str]) -> list[str]:\n"
                f"    freq = {{}}\n    for item in items:\n        freq[item] = freq.get(item, 0) + 1\n"
                f"    ordered = sorted(freq.keys(), key=lambda s: (-freq[s], s))\n"
                f"    return ordered[:{k}]\n```",
            ))

    # 2. Multi-step math & deductive logic (completely distinct surface templates!)
    names_pool = [("Orion", "Vega", "Altair", "Deneb"), ("Helio", "Cryo", "Pyro", "Aero")]
    for _ in range(80):
        racks = rng.randint(5, 13)
        per = rng.randint(6, 11)
        off = rng.randint(3, 15)
        ans = racks * per - off
        pairs.append((
            f"Facility inventory shows {racks} cabinets holding {per} GPUs apiece (ticket #{rng.randint(1000,8999)}). "
            f"After retiring {off} GPUs, compute remaining online GPUs. Conclude with `FINAL_ANSWER: <value>`.",
            f"<|think|>{racks} * {per} = {racks*per}; {racks*per} - {off} = {ans}.<|end_think|>\nFINAL_ANSWER: {ans}",
        ))
        a, b, c, d = rng.choice(names_pool)
        pairs.append((
            f"Tasks {a}, {b}, {c}, {d} execute one at a time. {b} precedes {a}. "
            f"{c} precedes {b}. {d} follows {a}. Name the earliest task using `FINAL_ANSWER: <task>`.",
            f"<|think|>Sequence: {c} -> {b} -> {a} -> {d}.<|end_think|>\nFINAL_ANSWER: {c}",
        ))

    # 3. Qualification-preserving summarization, JSON schemas, Mermaid diagrams, Charts
    for i in range(60):
        mb1 = rng.randint(70, 120)
        mb2 = rng.randint(20, 48)
        pairs.append((
            f"Condense this engineering update into a single sentence under 40 words keeping numbers and exceptions:\n"
            f"Update: Deploying SIMD codecs in `photon_bus` lowered RAM footprint from {mb1} GB down to {mb2} GB "
            f"on Linux fleets, except `riscv-sim` targets where debug guards remain required.",
            f"SIMD codecs in `photon_bus` reduced Linux RAM usage from {mb1} GB to {mb2} GB, "
            f"except on `riscv-sim` targets.",
        ))
        c1, c2 = rng.randint(8, 24), rng.randint(12, 36)
        obj = {"cluster": f"nimbus_{i}", "nodes": [{"host": "h1", "cores": c1}, {"host": "h2", "cores": c2}], "total_cores": c1 + c2}
        pairs.append((
            f"Produce raw JSON containing `cluster`, `nodes` (`host`, `cores`), and `total_cores` "
            f"describing `nimbus_{i}` with machines `h1`={c1} cores and `h2`={c2} cores.",
            json.dumps(obj, separators=(",", ":")),
        ))
        pairs.append((
            f"Write a Mermaid `flowchart LR` block where `edge_proxy_{i}` sends `valid` "
            f"packets to `worker_pool` and `blocked` packets to `quarantine_log`.",
            f'```mermaid\nflowchart LR\n    A["edge_proxy_{i}"] -->|"valid"| B["worker_pool"]\n'
            f'    A -->|"blocked"| C["quarantine_log"]\n```',
        ))
        y1, y2, y3, y4 = rng.randint(50, 90), rng.randint(95, 130), rng.randint(135, 170), rng.randint(175, 220)
        chart = {
            "chart_type": "line",
            "x_label": "Quarter",
            "y_label": "Throughput (req/s)",
            "x_values": ["Q1", "Q2", "Q3", "Q4"],
            "y_values": [y1, y2, y3, y4],
        }
        pairs.append((
            f"Build a JSON chart specification (`chart_type`, `x_label`, `y_label`, `x_values`, `y_values`) "
            f"representing quarterly rate measurements Q1={y1}, Q2={y2}, Q3={y3}, Q4={y4} req/s.",
            json.dumps(chart, separators=(",", ":")),
        ))

    # Decontaminate every SFT pair against held-out eval prompts!
    eval_prompts = collect_all_eval_prompts()
    rendered = [f"{u}\n{a}" for u, a in pairs]
    clean_rendered, rep = decontaminate_corpus(rendered, eval_prompts, ngram_size=13)
    assert rep.removed_eval_contaminated == 0, f"Found {rep.removed_eval_contaminated} contaminated SFT items!"
    rng.shuffle(pairs)
    return pairs


def finetune_and_evaluate_capability(
    out_dir: Path,
    best_arm_id: str,
    best_ckpt: Path,
) -> dict[str, object]:
    print(f"\n=== [E-CAP-01] Stage 1 -> Stage 2 Capability Progression & External Reference ({best_arm_id}) ===", flush=True)
    tok = load_tokenizer(out_dir / "tokenizers" / "nanolm_bpe_16k.json")
    im_end_id = tok.token_to_id("<|im_end|>")
    eos_id = tok.token_to_id("<|eos|>")
    pad_id = tok.token_to_id("<|pad|>")
    dev = detect_best_device()

    # Reconstruct winning model config from checkpoint payload
    payload = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    preset_name = payload["model_config"]["name"]
    # Map modified A0/A1 names back if needed
    if preset_name == "nanolm-25m-dense-wsd":
        m_cfg = get_preset_config("nanolm-25m-dense")
    elif preset_name == "nanolm-25m-no-qknorm":
        m_cfg = replace(get_preset_config("nanolm-25m-dense"), name="nanolm-25m-no-qknorm", qk_norm=False)
    else:
        m_cfg = get_preset_config(preset_name)

    model = NanoLMForCausalLM(m_cfg).to(dev)
    load_checkpoint(best_ckpt, model, device=dev)

    def make_nanolm_generator(mdl: NanoLMForCausalLM):
        def _gen(prompt: str) -> str:
            chat_str = format_chat_messages([{"role": "user", "content": prompt}], add_generation_prompt=True)
            ids = tok.encode(chat_str).ids
            inp = torch.tensor([ids], dtype=torch.long, device=dev)
            out_ids = mdl.generate(
                inp,
                max_new_tokens=192,
                temperature=0.0,
                stop_token_ids=(im_end_id, eos_id),
            )[0, len(ids) :].tolist()
            # Trim stop token if trailing
            while out_ids and out_ids[-1] in (im_end_id, eos_id):
                out_ids.pop()
            return tok.decode(out_ids)
        return _gen

    # 1. Stage 1 Base Pretrain evaluation
    base_report = run_capability_evaluation(f"{best_arm_id}_stage1_pretrain", make_nanolm_generator(model))
    print(f"  Stage 1 Base Pretrain overall pass rate: {base_report.overall_pass_rate:.1%}", flush=True)

    # 2. Stage 2 Supervised Instruction & Reasoning Fine-Tuning (masked assistant-only loss)
    sft_pairs = build_decontaminated_sft_pairs(seed=777)
    seq_len = 384
    sft_inputs: list[list[int]] = []
    sft_targets: list[list[int]] = []
    sft_masks: list[list[float]] = []

    for u_txt, a_txt in sft_pairs:
        prefix = format_chat_messages([{"role": "user", "content": u_txt}], add_generation_prompt=True)
        full = prefix + a_txt + "<|im_end|>"
        p_ids = tok.encode(prefix).ids
        f_ids = tok.encode(full).ids
        if len(f_ids) > seq_len + 1:
            continue
        padded = f_ids + [pad_id] * (seq_len + 1 - len(f_ids))
        x_row = padded[:seq_len]
        y_row = padded[1 : seq_len + 1]
        # Assistant tokens start at index len(p_ids) in f_ids -> target position len(p_ids) - 1 in y_row
        m_row = [
            1.0 if (len(p_ids) - 1 <= idx < len(f_ids) - 1) else 0.0
            for idx in range(seq_len)
        ]
        sft_inputs.append(x_row)
        sft_targets.append(y_row)
        sft_masks.append(m_row)

    x_ten = torch.tensor(sft_inputs, dtype=torch.long)
    y_ten = torch.tensor(sft_targets, dtype=torch.long)
    m_ten = torch.tensor(sft_masks, dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05)
    model.train()
    sft_steps = 220
    bsz = 16
    t0 = time.perf_counter()
    dtype = torch.bfloat16 if dev in ("cuda", "mps") else torch.float32

    for step in range(1, sft_steps + 1):
        perm = torch.randint(0, x_ten.shape[0], (bsz,))
        bx = x_ten[perm].to(dev)
        by = y_ten[perm].to(dev)
        bm = m_ten[perm].to(dev)
        opt.zero_grad(set_to_none=True)
        if dev in ("cuda", "mps"):
            with torch.autocast(device_type=dev, dtype=dtype):
                out = model(input_ids=bx, targets=by, loss_mask=bm)
                assert out.ce_loss is not None and out.z_loss is not None
                loss = out.ce_loss + out.z_loss
        else:
            out = model(input_ids=bx, targets=by, loss_mask=bm)
            assert out.ce_loss is not None and out.z_loss is not None
            loss = out.ce_loss + out.z_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    sft_wall = time.perf_counter() - t0
    sft_ckpt = out_dir / "runs" / f"{best_arm_id}_sft" / "latest.pt"
    save_atomic_checkpoint(
        sft_ckpt,
        model=model,
        optimizer=opt,
        step=sft_steps,
        train_cfg=TrainConfig(run_id=f"{best_arm_id}_sft", max_steps=sft_steps),
    )

    sft_report = run_capability_evaluation(f"{best_arm_id}_stage2_sft", make_nanolm_generator(model))
    print(
        f"  Stage 2 Multi-Domain SFT ({sft_wall:.1f}s) overall pass rate: {sft_report.overall_pass_rate:.1%}",
        flush=True,
    )
    for s_name, sc in sft_report.suites.items():
        print(f"    - {s_name:18s}: {sc.passed}/{sc.total} ({sc.pass_rate:.0%})", flush=True)

    del model
    if dev == "mps":
        torch.mps.empty_cache()

    # 3. External 1.54B Pretrained Reference Baseline (Qwen2.5-1.5B-Instruct, offline local cache)
    ref_report_dict = None
    qwen_snaps = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*"))
    if qwen_snaps:
        print("  Evaluating external local reference anchor: Qwen2.5-1.5B-Instruct (1.54B)...", flush=True)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        q_path = str(qwen_snaps[0])
        q_tok = AutoTokenizer.from_pretrained(q_path, local_files_only=True)
        q_mdl = AutoModelForCausalLM.from_pretrained(
            q_path, local_files_only=True, dtype=torch.float16
        ).to(dev).eval()

        @torch.inference_mode()
        def qwen_gen(prompt: str) -> str:
            text = q_tok.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            inp = q_tok(text, return_tensors="pt").to(dev)
            out = q_mdl.generate(
                **inp, do_sample=False, max_new_tokens=220, pad_token_id=q_tok.eos_token_id
            )
            cont = out[0, inp["input_ids"].shape[-1] :]
            return q_tok.decode(cont, skip_special_tokens=True)

        ref_rep = run_capability_evaluation("ref_qwen2.5_1.5b_instruct", qwen_gen)
        ref_report_dict = ref_rep.to_dict()
        print(f"  External Reference Qwen2.5-1.5B-Instruct overall pass rate: {ref_rep.overall_pass_rate:.1%}", flush=True)
        for s_name, sc in ref_rep.suites.items():
            print(f"    - {s_name:18s}: {sc.passed}/{sc.total} ({sc.pass_rate:.0%})", flush=True)
        del q_mdl
        if dev == "mps":
            torch.mps.empty_cache()

    summary = {
        "experiment_id": "E-CAP-01",
        "selected_architecture": best_arm_id,
        "sft_checkpoint": str(sft_ckpt),
        "stage1_pretrain_eval": base_report.to_dict(),
        "stage2_sft_eval": sft_report.to_dict(),
        "external_reference_qwen25_1p5b": ref_report_dict,
    }
    (out_dir / "e_cap_01_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = ROOT / "artifacts" / "nanolm_v1"
    out.mkdir(parents=True, exist_ok=True)
    _, winner_id, winner_ckpt = run_e_arch_01(out)
    finetune_and_evaluate_capability(out, winner_id, winner_ckpt)
