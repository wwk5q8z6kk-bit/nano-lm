#!/usr/bin/env python3
"""Execute Experiment E-TOK-01 (Multi-Domain Tokenizer Study) and E-INFRA-2B4B (Live 2B-4B MPS Profiling)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tokenizers import Tokenizer

from nanolm.config import get_preset_config, list_presets
from nanolm.data import build_multidomain_corpus, decontaminate_corpus
from nanolm.eval import collect_all_eval_prompts
from nanolm.infra import benchmark_live_inference, estimate_full_training_budget
from nanolm.tokenizer import audit_tokenizer_efficiency, save_tokenizer, train_nanolm_tokenizer


def run_tokenizer_experiment(out_dir: Path) -> dict[str, object]:
    print("=== [E-TOK-01] Multi-Domain Tokenizer & Compression Study ===", flush=True)
    train_domains = build_multidomain_corpus(split="train", prose_docs_limit=1500, structured_docs_count=1600, seed=101)
    val_domains = build_multidomain_corpus(split="val", prose_docs_limit=240, structured_docs_count=240, seed=202)

    eval_prompts = collect_all_eval_prompts()
    raw_train_docs = [doc for docs in train_domains.values() for doc in docs]
    clean_train_docs, decontam = decontaminate_corpus(raw_train_docs, eval_prompts, ngram_size=13)
    print(f"Decontaminated training docs: {decontam.kept_documents}/{decontam.total_candidates} "
          f"(exact_dups={decontam.removed_exact_dups}, contam={decontam.removed_eval_contaminated})", flush=True)

    tok_dir = out_dir / "tokenizers"
    tok_dir.mkdir(parents=True, exist_ok=True)

    # 1. Legacy 4098 BPE (no digit split)
    legacy_path = ROOT / "sft" / "tokenizer.json"
    legacy_tok = Tokenizer.from_file(str(legacy_path))
    legacy_sha = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    rep_legacy = audit_tokenizer_efficiency(
        name="legacy-nano-4k",
        encode_fn=lambda s: legacy_tok.encode(s, add_special_tokens=False).ids,
        vocab_size=legacy_tok.get_vocab_size(),
        domain_samples=val_domains,
        sha256=legacy_sha,
    )

    # 2. NanoLM 16k BPE (with digit splitting + 32 special tokens)
    tok_16k = train_nanolm_tokenizer(clean_train_docs, vocab_size=16_384, min_frequency=2, split_digits=True)
    sha_16k = save_tokenizer(tok_16k, tok_dir / "nanolm_bpe_16k.json")
    rep_16k = audit_tokenizer_efficiency(
        name="nanolm-bpe-16k",
        encode_fn=lambda s: tok_16k.encode(s).ids,
        vocab_size=tok_16k.get_vocab_size(),
        domain_samples=val_domains,
        sha256=sha_16k,
    )

    # 3. NanoLM 32k BPE (with digit splitting + 32 special tokens)
    tok_32k = train_nanolm_tokenizer(clean_train_docs, vocab_size=32_768, min_frequency=2, split_digits=True)
    sha_32k = save_tokenizer(tok_32k, tok_dir / "nanolm_bpe_32k.json")
    rep_32k = audit_tokenizer_efficiency(
        name="nanolm-bpe-32k",
        encode_fn=lambda s: tok_32k.encode(s).ids,
        vocab_size=tok_32k.get_vocab_size(),
        domain_samples=val_domains,
        sha256=sha_32k,
    )

    # 4. External reference Qwen2.5 tokenizer (151,936 vocab) if cached locally
    reports = [rep_legacy.to_dict(), rep_16k.to_dict(), rep_32k.to_dict()]
    qwen_snaps = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/tokenizer.json"))
    if qwen_snaps:
        q_tok = Tokenizer.from_file(str(qwen_snaps[0]))
        q_sha = hashlib.sha256(qwen_snaps[0].read_bytes()).hexdigest()
        rep_qwen = audit_tokenizer_efficiency(
            name="qwen2.5-152k-ref",
            encode_fn=lambda s: q_tok.encode(s, add_special_tokens=False).ids,
            vocab_size=q_tok.get_vocab_size(),
            domain_samples=val_domains,
            sha256=q_sha,
        )
        reports.append(rep_qwen.to_dict())

    for r in reports:
        emb_pct_2b = round((int(r["vocab_size"]) * 2048) / 2_041_000_000 * 100.0, 2)
        r["tied_embedding_share_of_2b_pct"] = emb_pct_2b
        print(
            f"  {r['name']:18s} | V={r['vocab_size']:7d} | overall={r['overall_bytes_per_token']:.3f} B/tok "
            f"| code={r['domains']['code']['bytes_per_token']:.3f} B/tok "
            f"| digit_split={r['individual_digit_splitting']} | emb@2B={emb_pct_2b:.1f}%",
            flush=True,
        )

    summary = {
        "experiment_id": "E-TOK-01",
        "decontamination": decontam.to_dict(),
        "tokenizers": reports,
    }
    (out_dir / "e_tok_01_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_infra_2b_4b_experiment(out_dir: Path) -> dict[str, object]:
    print("\n=== [E-INFRA-2B4B] Full-Scale 2.0B-3.9B Live MPS Hardware Benchmark ===", flush=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    prod_presets = [p for p in list_presets() if get_preset_config(p).tier == "production"]

    live_profiles = []
    scaling_plans = []
    for name in prod_presets:
        cfg = get_preset_config(name)
        acct = cfg.parameter_accounting()
        prof = benchmark_live_inference(
            cfg,
            prompt_len=256,
            decode_tokens=16,
            device=device,
            dtype=torch.bfloat16,
        )
        prof["accounting"] = acct.to_dict()
        live_profiles.append(prof)

        est = estimate_full_training_budget(cfg, target_tokens_billions=100.0)
        scaling_plans.append(est.to_dict())

        print(
            f"  {name:22s} | params={prof['deployed_parameters']/1e9:.3f}B "
            f"(active={prof['active_parameters_per_token']/1e9:.3f}B) "
            f"| prefill={prof['prefill_tok_per_s']:6.1f} tok/s "
            f"| decode={prof['decode_tok_per_s']:5.2f} tok/s "
            f"| KV@272={prof['measured_kv_cache_mb']:6.2f} MB "
            f"| KV@32k={acct.kv_cache_gb_at_32k_bf16:4.2f} GB",
            flush=True,
        )

    summary = {
        "experiment_id": "E-INFRA-2B4B",
        "hardware": "Apple M3 Max (64 GB unified memory, MPS bf16)",
        "live_2b_4b_benchmarks": live_profiles,
        "training_100B_token_estimates": scaling_plans,
    }
    (out_dir / "e_infra_2b4b_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    out = ROOT / "artifacts" / "nanolm_v1"
    out.mkdir(parents=True, exist_ok=True)
    run_tokenizer_experiment(out)
    run_infra_2b_4b_experiment(out)
