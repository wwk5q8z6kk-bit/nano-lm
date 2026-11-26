#!/bin/bash
# Phase 5B: Train causally-shifted NanoLM-B7-3B-Fixed (3.08B < 4.0B) + full 8-pillar, lm_eval,
# and identical frozen NanoLMSystem-v4 Competition 1 & 2 evaluation.
set -euo pipefail
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/nl
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache /workspace/phase3_data /workspace/models /workspace/results
cd /workspace/nl

echo "=== $(date -u +%FT%TZ) STEP 0: INSTALL ==="
pip install -q --break-system-packages vllm "lm_eval[vllm,math,ifeval]" datasets accelerate math_verify sympy antlr4-python3-runtime==4.11 langdetect immutabledict nltk

echo "=== $(date -u +%FT%TZ) STEP 1: TRAIN NANOLM-B7-3B-FIXED (CAUSALLY SHIFTED SELF-ANCHORED TOP-64 KL + L2 FFT + G=8 GRPO) ==="
python -m lab.trackb.phase3_innovation --phase train_b7_3b --data-jsonl /workspace/phase3_data/stratified_b6.jsonl \
  --rlvr-jsonl /workspace/phase3_data/rlvr_b6.jsonl --out-dir /workspace/models/nanolm_b7_3b_fixed --seed 3031 \
  > /workspace/results/step1_b7_3b_fixed.log 2>&1
rm -rf /workspace/models/nanolm_b7_3b_fixed_self_anchored_fft

OUT=/workspace/results/nanolm_b7_3b_fixed
mkdir -p "$OUT"
cp /workspace/models/nanolm_b7_3b_fixed/nanolm_training_manifest.json "$OUT/" || true
echo "=== $(date -u +%FT%TZ) STEP 2: EVAL NANOLM-B7-3B-FIXED (8-PILLAR + GSM8K/MATH-500/IFEVAL/HUMANEVAL) ==="
python -m lab.trackb.run_8pillar_vllm --model /workspace/models/nanolm_b7_3b_fixed --out "$OUT/pillars8_think0.json" --think 0 --max-tokens 1536 --gpu-mem 0.86 > "$OUT/p8_t0.log" 2>&1 || true
python -m lab.trackb.run_8pillar_vllm --model /workspace/models/nanolm_b7_3b_fixed --out "$OUT/pillars8_think1.json" --think 1 --max-tokens 4096 --gpu-mem 0.86 > "$OUT/p8_t1.log" 2>&1 || true
VARGS="pretrained=/workspace/models/nanolm_b7_3b_fixed,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,trust_remote_code=False"
lm_eval --model vllm --model_args "$VARGS" --tasks "gsm8k_cot_llama,minerva_math500" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/math" 2>&1 | tail -25 > "$OUT/math.log" || true
lm_eval --model vllm --model_args "$VARGS" --tasks "ifeval" --apply_chat_template --gen_kwargs "max_gen_toks=1280,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/instr" 2>&1 | tail -25 > "$OUT/instr.log" || true
lm_eval --model vllm --model_args "$VARGS" --tasks "humaneval_instruct" --apply_chat_template --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=1024,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/code" 2>&1 | tail -25 > "$OUT/code.log" || true

echo "=== $(date -u +%FT%TZ) STEP 3: FROZEN SYSTEM-V4 COMP 1 & 2 ON NANOLM-B7-3B-FIXED ==="
mkdir -p /workspace/results/nanolm_b7_3b_fixed_v4
python -m lab.system.run_competitions_vllm --model /workspace/models/nanolm_b7_3b_fixed \
  --out /workspace/results/nanolm_b7_3b_fixed_v4/competitions_1_2_v4.json \
  --gpu-mem 0.86 --max-model-len 8192 > /workspace/results/nanolm_b7_3b_fixed_v4/run.log 2>&1 || true

echo "PHASE5B_ALL_COMPLETE"
