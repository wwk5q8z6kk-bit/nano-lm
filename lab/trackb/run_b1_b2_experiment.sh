#!/bin/bash
# Track B Controlled Experiment: B1 (Verified SFT) vs B2 (Proper Top-64+Tail Teacher Logit KD from Qwen3-8B)
set -u
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/nl

while pgrep -f "post_matrix.sh" >/dev/null 2>&1; do sleep 10; done
cd /workspace/nl

echo "=== $(date -u +%FT%TZ) TRAIN B1_verified_sft (alpha_kd=0.0) ==="
python -m lab.trackb.train_nanolm_b1 \
  --student Qwen/Qwen3-1.7B \
  --alpha-kd 0.0 \
  --rank 32 --epochs 2 --lr 1.5e-4 --grad-accum 8 --seed 42 \
  --out-dir /workspace/models/nanolm_b1_sft_2b \
  2>&1 | tee /workspace/b1_train.log

echo "=== $(date -u +%FT%TZ) TRAIN B2_teacher_kd_8b (alpha_kd=0.6, teacher=Qwen/Qwen3-8B) ==="
python -m lab.trackb.train_nanolm_b1 \
  --student Qwen/Qwen3-1.7B \
  --teacher Qwen/Qwen3-8B \
  --alpha-kd 0.6 --temperature 1.5 --topk 64 \
  --rank 32 --epochs 2 --lr 1.5e-4 --grad-accum 8 --seed 42 \
  --out-dir /workspace/models/nanolm_b2_kd8b_2b \
  2>&1 | tee /workspace/b2_train.log

eval_arm() {
  local MDIR=$1 TAG=$2
  local OUT=/workspace/results/$TAG
  mkdir -p $OUT
  echo "=== $(date -u +%FT%TZ) EVAL 8-PILLAR $TAG ==="
  python -m lab.trackb.run_8pillar_vllm --model "$MDIR" --out "$OUT/pillars8_think0.json" --think 0 --max-tokens 1536 > "$OUT/pillars8_think0.log" 2>&1
  python -m lab.trackb.run_8pillar_vllm --model "$MDIR" --out "$OUT/pillars8_think1.json" --think 1 --max-tokens 4096 > "$OUT/pillars8_think1.log" 2>&1

  local VARGS="pretrained=$MDIR,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,trust_remote_code=False,enable_thinking=False"
  echo "=== $(date -u +%FT%TZ) EVAL LM_EVAL $TAG ==="
  lm_eval --model vllm --model_args "$VARGS" --tasks "arc_challenge,winogrande,truthfulqa_mc2" --num_fewshot 5 --batch_size auto --seed 1234 --output_path "$OUT/know_fast" 2>&1 | tail -30 > "$OUT/know_fast.log"
  lm_eval --model vllm --model_args "$VARGS" --tasks "gsm8k_cot_llama,minerva_math500" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/math" 2>&1 | tail -30 > "$OUT/math.log"
  lm_eval --model vllm --model_args "$VARGS" --tasks "ifeval" --apply_chat_template --gen_kwargs "max_gen_toks=1280,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/instr" 2>&1 | tail -30 > "$OUT/instr.log"
  lm_eval --model vllm --model_args "$VARGS" --tasks "humaneval_instruct" --apply_chat_template --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=1024,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/code" 2>&1 | tail -30 > "$OUT/code.log"
  cp "$MDIR/nanolm_training_manifest.json" "$OUT/nanolm_training_manifest.json"
}

eval_arm "/workspace/models/nanolm_b1_sft_2b" "nanolm_b1_sft_2b"
eval_arm "/workspace/models/nanolm_b2_kd8b_2b" "nanolm_b2_kd8b_2b"
echo "B1_B2_ALL_COMPLETE"
