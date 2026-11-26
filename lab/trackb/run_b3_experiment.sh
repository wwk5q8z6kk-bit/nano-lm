#!/bin/bash
# Track B Experiment B3: Verified Dual-Mode On-Policy Teacher Distillation (Qwen3-8B -> Qwen3-1.7B)
set -euo pipefail
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/nl
cd /workspace/nl
mkdir -p /workspace/data /workspace/models /workspace/results/nanolm_b3_onpolicy_kd8b_2b

echo "=== $(date -u +%FT%TZ) PHASE 1: GENERATE & VERIFY ON-POLICY QWEN3-8B TRAJECTORIES ==="
python -m lab.trackb.train_nanolm_b3 \
  --phase generate \
  --teacher Qwen/Qwen3-8B \
  --data-jsonl /workspace/data/b3_verified_onpolicy.jsonl \
  --seed 2026 \
  2>&1 | tee /workspace/b3_generate.log

echo "=== $(date -u +%FT%TZ) PHASE 2: PROPER TOP-64+TAIL LOGIT KD (alpha_kd=0.85, r=16, lr=5e-5) ==="
python -m lab.trackb.train_nanolm_b3 \
  --phase distill \
  --student Qwen/Qwen3-1.7B \
  --teacher Qwen/Qwen3-8B \
  --data-jsonl /workspace/data/b3_verified_onpolicy.jsonl \
  --out-dir /workspace/models/nanolm_b3_onpolicy_kd8b_2b \
  --alpha-kd 0.85 --temperature 1.5 --topk 64 \
  --rank 16 --epochs 1 --lr 5e-5 --grad-accum 8 --seed 2026 \
  2>&1 | tee /workspace/b3_distill.log

MDIR=/workspace/models/nanolm_b3_onpolicy_kd8b_2b
OUT=/workspace/results/nanolm_b3_onpolicy_kd8b_2b

echo "=== $(date -u +%FT%TZ) PHASE 3: EVAL 8-PILLAR (think=0 & think=1) ==="
python -m lab.trackb.run_8pillar_vllm --model "$MDIR" --out "$OUT/pillars8_think0.json" --think 0 --max-tokens 1536 > "$OUT/pillars8_think0.log" 2>&1
python -m lab.trackb.run_8pillar_vllm --model "$MDIR" --out "$OUT/pillars8_think1.json" --think 1 --max-tokens 4096 > "$OUT/pillars8_think1.log" 2>&1

VARGS="pretrained=$MDIR,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,trust_remote_code=False,enable_thinking=False"
echo "=== $(date -u +%FT%TZ) PHASE 4: EVAL LM_EVAL (know_fast, math, instr, code) ==="
lm_eval --model vllm --model_args "$VARGS" --tasks "arc_challenge,winogrande,truthfulqa_mc2" --num_fewshot 5 --batch_size auto --seed 1234 --output_path "$OUT/know_fast" 2>&1 | tail -30 > "$OUT/know_fast.log"
lm_eval --model vllm --model_args "$VARGS" --tasks "gsm8k_cot_llama,minerva_math500" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/math" 2>&1 | tail -30 > "$OUT/math.log"
lm_eval --model vllm --model_args "$VARGS" --tasks "ifeval" --apply_chat_template --gen_kwargs "max_gen_toks=1280,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/instr" 2>&1 | tail -30 > "$OUT/instr.log"
lm_eval --model vllm --model_args "$VARGS" --tasks "humaneval_instruct" --apply_chat_template --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=1024,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/code" 2>&1 | tail -30 > "$OUT/code.log"

cp "$MDIR/nanolm_training_manifest.json" "$OUT/nanolm_training_manifest.json"
cp /workspace/data/b3_verified_onpolicy.summary.json "$OUT/b3_verified_onpolicy.summary.json"
echo "B3_ALL_COMPLETE"
