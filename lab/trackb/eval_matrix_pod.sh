#!/bin/bash
# Fast full-set multi-candidate competitive evaluation on A40 ($0.59/h).
# Runs 8-pillar (132-item) tool-calling/RULER/reliability + full GSM8K, MATH-500, HumanEval, MBPP+, IFEval, ARC-C, WinoGrande, TruthfulQA, BBH-CoT.
set -u
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/nl

eval_8pillar() {
  local M=$1 TAG=$2 THINK=$3 MAXTOK=$4
  mkdir -p /workspace/results/$TAG
  python -m lab.trackb.run_8pillar_vllm --model "$M" --out "/workspace/results/$TAG/pillars8_think${THINK}.json" --think "$THINK" --max-tokens "$MAXTOK" 2>&1 | tail -35 > "/workspace/results/$TAG/pillars8_think${THINK}.log" || echo "FAILED pillars8_$THINK" >> /workspace/results/$TAG/failures.txt
}

eval_standard() {
  local M=$1 TAG=$2 THINK=${3:-0}
  local OUT=/workspace/results/$TAG
  mkdir -p $OUT
  local REV
  REV=$(python -c "from huggingface_hub import model_info;print(model_info('$M').sha)")
  # Only pass enable_thinking=False for Qwen3 / Qwen3.5 / SmolLM3 families that accept it; test via python first
  local CTK=""
  if [ "$THINK" = "0" ]; then
    python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$M'); t.apply_chat_template([{'role':'user','content':'hi'}], tokenize=False, enable_thinking=False)" >/dev/null 2>&1 && CTK=",enable_thinking=False"
  fi
  local VARGS="pretrained=$M,revision=$REV,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,trust_remote_code=False$CTK"
  echo "{\"model\":\"$M\",\"revision\":\"$REV\",\"think\":$THINK}" > $OUT/conditions.json

  [ -d "$OUT/know_fast" ] || lm_eval --model vllm --model_args "$VARGS" --tasks "arc_challenge,winogrande,truthfulqa_mc2" --num_fewshot 5 --batch_size auto --seed 1234 --output_path $OUT/know_fast 2>&1 | tail -30 > $OUT/know_fast.log || echo "FAILED know_fast" >> $OUT/failures.txt
  [ -d "$OUT/math" ] || lm_eval --model vllm --model_args "$VARGS" --tasks "gsm8k_cot_llama,minerva_math500" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0" --batch_size auto --seed 1234 --output_path $OUT/math 2>&1 | tail -30 > $OUT/math.log || echo "FAILED math" >> $OUT/failures.txt
  [ -d "$OUT/instr" ] || lm_eval --model vllm --model_args "$VARGS" --tasks "ifeval" --apply_chat_template --gen_kwargs "max_gen_toks=1280,temperature=0" --batch_size auto --seed 1234 --output_path $OUT/instr 2>&1 | tail -30 > $OUT/instr.log || echo "FAILED instr" >> $OUT/failures.txt
  [ -d "$OUT/code" ] || lm_eval --model vllm --model_args "$VARGS" --tasks "humaneval_instruct,mbpp_plus_instruct" --apply_chat_template --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=1024,temperature=0" --batch_size auto --seed 1234 --output_path $OUT/code 2>&1 | tail -30 > $OUT/code.log || echo "FAILED code" >> $OUT/failures.txt
}

# 1. Complete 8-pillar (think=0 and think=1) for Qwen3.5-2B (math/instr/code/knowledge already done!)
eval_8pillar "Qwen/Qwen3.5-2B" "qwen35_2b" 0 1536
eval_8pillar "Qwen/Qwen3.5-2B" "qwen35_2b" 1 4096

# 2. Full evaluation across remaining eligible <=4B Apache-2.0 / MIT candidates + 1.54B reference
for spec in \
  "Qwen/Qwen3-1.7B:qwen3_1p7b" \
  "HuggingFaceTB/SmolLM3-3B:smollm3_3b" \
  "ibm-granite/granite-4.0-h-micro:granite4_h_3b" \
  "microsoft/Phi-4-mini-instruct:phi4_mini_3p8b" \
  "Qwen/Qwen2.5-1.5B-Instruct:qwen25_1p5b"
do
  M=${spec%%:*}; T=${spec##*:}
  echo "=== $(date -u +%FT%TZ) MATRIX $M ==="
  eval_8pillar "$M" "$T" 0 1536
  eval_standard "$M" "$T" 0
done
echo "MATRIX_ALL_DONE"
