#!/usr/bin/env python3
"""Unified CLI for NanoLM Pretraining, Preflight Estimation, and Evaluation across Local MPS, Kaggle, and RunPod."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanolm.config import get_preset_config, list_presets
from nanolm.data import PackedDocumentDataset, build_multidomain_corpus, decontaminate_corpus
from nanolm.eval import collect_all_eval_prompts
from nanolm.infra import benchmark_live_inference, estimate_full_training_budget
from nanolm.tokenizer import audit_tokenizer_efficiency, load_tokenizer, save_tokenizer, train_nanolm_tokenizer
from nanolm.train import TrainConfig, detect_best_device, run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="NanoLM Research & Training CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("presets", help="Display all verified NanoLM architecture presets and budgets")
    p_list.add_argument("--tokens-b", type=float, default=100.0, help="Target pretraining tokens in billions")

    p_prof = sub.add_parser("preflight", help="Run live hardware preflight benchmark for a preset")
    p_prof.add_argument("--preset", default="nanolm-2b-hybrid-swa", choices=list_presets())
    p_prof.add_argument("--prompt-len", type=int, default=256)
    p_prof.add_argument("--decode-tokens", type=int, default=32)

    p_train = sub.add_parser("train", help="Run reproducible packed multi-domain training")
    p_train.add_argument("--preset", default="nanolm-25m-hybrid-swa", choices=list_presets())
    p_train.add_argument("--steps", type=int, default=300)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--grad-accum", type=int, default=1)
    p_train.add_argument("--lr", type=float, default=1.2e-3)
    p_train.add_argument("--schedule", default="wsd", choices=["wsd", "cosine"])
    p_train.add_argument("--seq-len", type=int, default=256)
    p_train.add_argument("--out-dir", default="artifacts/nanolm_cli_run")
    p_train.add_argument("--resume", default=None)

    args = parser.parse_args()

    if args.cmd == "presets":
        rows = []
        for name in list_presets():
            cfg = get_preset_config(name)
            acct = cfg.parameter_accounting()
            est = estimate_full_training_budget(cfg, target_tokens_billions=args.tokens_b)
            rows.append({"preset": name, "accounting": acct.to_dict(), "compute_plan": est.to_dict()})
        print(json.dumps(rows, indent=2))
        return 0

    if args.cmd == "preflight":
        cfg = get_preset_config(args.preset)
        dev = detect_best_device()
        res = benchmark_live_inference(
            cfg,
            prompt_len=args.prompt_len,
            decode_tokens=args.decode_tokens,
            device=dev,
        )
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "train":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = get_preset_config(args.preset)

        train_dom = build_multidomain_corpus(split="train", prose_docs_limit=1500, structured_docs_count=1600)
        val_dom = build_multidomain_corpus(split="val", prose_docs_limit=240, structured_docs_count=240)
        raw_train = [d for docs in train_dom.values() for d in docs]
        raw_val = [d for docs in val_dom.values() for d in docs]
        clean_train, _ = decontaminate_corpus(raw_train, collect_all_eval_prompts(), ngram_size=13)

        tok_file = out_dir / "tokenizer.json"
        if tok_file.is_file():
            tok = load_tokenizer(tok_file)
        else:
            tok = train_nanolm_tokenizer(clean_train, vocab_size=cfg.vocab_size, split_digits=True)
            save_tokenizer(tok, tok_file)

        val_rep = audit_tokenizer_efficiency(
            "val", lambda s: tok.encode(s).ids, tok.get_vocab_size(), {"val": raw_val}
        )
        ds_train = PackedDocumentDataset([tok.encode(d).ids for d in clean_train], seq_len=args.seq_len)
        ds_val = PackedDocumentDataset([tok.encode(d).ids for d in raw_val], seq_len=args.seq_len)

        tcfg = TrainConfig(
            run_id=args.preset,
            max_steps=args.steps,
            micro_batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum,
            peak_lr=args.lr,
            schedule=args.schedule,
            warmup_steps=max(10, args.steps // 10),
            checkpoint_dir=str(out_dir / "checkpoints"),
        )
        res = run_training(
            model_cfg=cfg,
            train_cfg=tcfg,
            train_dataset=ds_train,
            val_dataset=ds_val,
            val_bytes_per_token=val_rep.overall_bytes_per_token,
            resume_from=args.resume,
        )
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
