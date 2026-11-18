#!/bin/bash
# Phase 3 Flagship Innovation Pipeline on 1x NVIDIA H200 SXM (141 GB):
#   1. Stratified Surprise + Multi-Turn Error-Recovery Curriculum verified from Qwen3-32B + openai/gpt-oss-120b
#   2. Train NanoLM-B6-2B (1.72B: L2-Anchored 100% FFT + Top-64+Tail KD + G=8 GRPO)
#   3. Train NanoLM-B6-3B (3.08B Stage-3 SmolLM3-3B: Stratified 32B+120B Dual-Mode rsLoRA + G=8 GRPO)
#   4. Evaluate B6-2B, B6-3B, Base 1.7B, Base 3.08B, Qwen3-8B, Qwen3-32B, and gpt-oss-120b
#      across 8-Pillar, lm_eval, Competition 1 (Naked), and Competition 2 (NanoLMSystem-v2)!
set -euo pipefail
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/nl
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache /workspace/phase3_data /workspace/models /workspace/results

echo "=== $(date -u +%FT%TZ) STEP 0: INSTALL VLLM, HARMONY & LM_EVAL[MATH,IFEVAL] ==="
pip install -q --break-system-packages vllm "lm_eval[vllm,math,ifeval]" datasets accelerate math_verify sympy antlr4-python3-runtime==4.11 langdetect immutabledict nltk openai-harmony

cd /workspace/nl

echo "=== $(date -u +%FT%TZ) STEP 1: GENERATE STRATIFIED 32B + 120B VERIFIED CURRICULUM ==="
python -m lab.trackb.phase3_innovation \
  --phase generate_stratified \
  --data-dir /workspace/phase3_data \
  --subset-size 480 \
  --seed 3031 \
  2>&1 | tee /workspace/results/step1_stratified_gen.log
cp /workspace/phase3_data/phase3_curriculum_diagnostics.json /workspace/results/phase3_curriculum_diagnostics.json

echo "=== $(date -u +%FT%TZ) STEP 2: TRAIN NANOLM-B6-2B (1.72B L2-ANCHORED FFT + TOP-64 KD + G=8 GRPO) ==="
python -m lab.trackb.phase3_innovation \
  --phase train_b6_2b \
  --data-jsonl /workspace/phase3_data/stratified_b6.jsonl \
  --rlvr-jsonl /workspace/phase3_data/rlvr_b6.jsonl \
  --out-dir /workspace/models/nanolm_b6_2b \
  --seed 3031 \
  2>&1 | tee /workspace/results/step2_b6_2b.log
rm -rf /workspace/models/nanolm_b6_2b_anchored_fft

echo "=== $(date -u +%FT%TZ) STEP 3: TRAIN STAGE-3 NANOLM-B6-3B (3.08B SMOLLM3-3B + G=8 GRPO) ==="
python -m lab.trackb.phase3_innovation \
  --phase train_b6_3b \
  --data-jsonl /workspace/phase3_data/stratified_b6.jsonl \
  --rlvr-jsonl /workspace/phase3_data/rlvr_b6.jsonl \
  --out-dir /workspace/models/nanolm_b6_3b \
  --seed 3031 \
  2>&1 | tee /workspace/results/step3_b6_3b.log
rm -rf /workspace/models/nanolm_b6_3b_sft

eval_full_model() {
  local MDIR=$1 TAG=$2
  local OUT=/workspace/results/$TAG
  mkdir -p "$OUT"
  cp "$MDIR/nanolm_training_manifest.json" "$OUT/nanolm_training_manifest.json" 2>/dev/null || true
  echo "=== $(date -u +%FT%TZ) FULL EVAL ON $TAG ==="
  python -m lab.trackb.run_8pillar_vllm --model "$MDIR" --out "$OUT/pillars8_think0.json" --think 0 --max-tokens 1536 --gpu-mem 0.86 > "$OUT/p8_t0.log" 2>&1
  python -m lab.trackb.run_8pillar_vllm --model "$MDIR" --out "$OUT/pillars8_think1.json" --think 1 --max-tokens 4096 --gpu-mem 0.86 > "$OUT/p8_t1.log" 2>&1
  python -m lab.system.run_competitions_vllm --model "$MDIR" --out "$OUT/competitions_1_2_v2.json" --gpu-mem 0.86 > "$OUT/comp_v2.log" 2>&1

  local CTK=""
  python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$MDIR'); t.apply_chat_template([{'role':'user','content':'hi'}], tokenize=False, enable_thinking=False)" >/dev/null 2>&1 && CTK=",enable_thinking=False"
  local VARGS="pretrained=$MDIR,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,trust_remote_code=False$CTK"
  lm_eval --model vllm --model_args "$VARGS" --tasks "gsm8k_cot_llama,minerva_math500" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/math" 2>&1 | tail -25 > "$OUT/math.log" || true
  lm_eval --model vllm --model_args "$VARGS" --tasks "ifeval" --apply_chat_template --gen_kwargs "max_gen_toks=1280,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/instr" 2>&1 | tail -25 > "$OUT/instr.log" || true
  lm_eval --model vllm --model_args "$VARGS" --tasks "humaneval_instruct" --apply_chat_template --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=1024,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/code" 2>&1 | tail -25 > "$OUT/code.log" || true
}

eval_full_model "/workspace/models/nanolm_b6_2b" "nanolm_b6_2b"
eval_full_model "/workspace/models/nanolm_b6_3b" "nanolm_b6_3b"

rm -rf /workspace/models/nanolm_b6_2b /workspace/models/nanolm_b6_3b

echo "=== $(date -u +%FT%TZ) STEP 5: NANOLMSYSTEM-V2 HEAD-TO-HEAD COMPARATORS (1.7B, 3B, 8B, 32B, 120B) ==="
for spec in \
  "Qwen/Qwen3-1.7B:qwen3_1p7b_v2:0.85:8192" \
  "HuggingFaceTB/SmolLM3-3B:smollm3_3b_v2:0.86:8192" \
  "Qwen/Qwen3-8B:qwen3_8b_v2:0.88:8192" \
  "Qwen/Qwen3-32B:qwen3_32b_v2:0.90:8192" \
  "openai/gpt-oss-120b:gpt_oss_120b_v2:0.92:16384"
do
  IFS=':' read -r MID TAG GMEM MLEN <<< "$spec"
  mkdir -p "/workspace/results/$TAG"
  echo "=== $(date -u +%FT%TZ) COMP 1 & 2 (V2) ON $MID ==="
  python -m lab.system.run_competitions_vllm \
    --model "$MID" \
    --out "/workspace/results/$TAG/competitions_1_2_v2.json" \
    --gpu-mem "$GMEM" --max-model-len "$MLEN" > "/workspace/results/$TAG/run.log" 2>&1 || true
done

echo "PHASE3_ALL_COMPLETE"
