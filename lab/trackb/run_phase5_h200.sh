#!/bin/bash
# Phase 5 (H200):
#   1. NanoLM-B7-3B (3.08B < 4.0B): Self-Anchored Top-64+Tail KL + Proximal L2 100% FFT + G=8 GRPO
#      on verified Stratified 32B+120B curriculum (reusing synced phase3_data when pre-uploaded).
#   2. Full 8-Pillar + External Standard Benchmarks (GSM8K, MATH-500, IFEval, HumanEval) on NanoLM-B7-3B.
#   3. NanoLMSystem-v4 (Multi-line JSON fix + Error-Only Adaptive Thinking + Truncation/Premature-Stop
#      Guards + Differential Code Self-Check + Exact Rational Math) evaluated under IDENTICAL conditions
#      across 1.72B, 3.08B, 8B, 32B, and 120B.
set -euo pipefail
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/nl
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache /workspace/phase3_data /workspace/models /workspace/results
cd /workspace/nl

echo "=== $(date -u +%FT%TZ) STEP 0: INSTALL ==="
pip install -q --break-system-packages vllm "lm_eval[vllm,math,ifeval]" datasets accelerate math_verify sympy antlr4-python3-runtime==4.11 langdetect immutabledict nltk openai-harmony

if [ ! -s /workspace/phase3_data/stratified_b6.jsonl ] || [ ! -s /workspace/phase3_data/rlvr_b6.jsonl ]; then
  echo "=== $(date -u +%FT%TZ) STEP 1A: GENERATE STRATIFIED 32B+120B CURRICULUM ==="
  python -m lab.trackb.phase3_innovation --phase generate_stratified --data-dir /workspace/phase3_data --subset-size 480 --seed 3031 \
    > /workspace/results/step1_gen.log 2>&1
else
  echo "=== $(date -u +%FT%TZ) STEP 1A: USING PRE-VERIFIED STRATIFIED 32B+120B CURRICULUM ($(wc -l < /workspace/phase3_data/stratified_b6.jsonl) rows) ==="
fi

echo "=== $(date -u +%FT%TZ) STEP 1B: TRAIN NANOLM-B7-3B (SELF-ANCHORED TOP-64 KL + L2 FFT + G=8 GRPO) ==="
python -m lab.trackb.phase3_innovation --phase train_b7_3b --data-jsonl /workspace/phase3_data/stratified_b6.jsonl \
  --rlvr-jsonl /workspace/phase3_data/rlvr_b6.jsonl --out-dir /workspace/models/nanolm_b7_3b --seed 3031 \
  > /workspace/results/step1_b7_3b.log 2>&1
grep -q "Encoded 0/" /workspace/results/step1_b7_3b.log && { echo "FATAL: encoder dropped data"; exit 1; }
rm -rf /workspace/models/nanolm_b7_3b_self_anchored_fft

OUT=/workspace/results/nanolm_b7_3b
mkdir -p "$OUT"
cp /workspace/models/nanolm_b7_3b/nanolm_training_manifest.json "$OUT/" || true
echo "=== $(date -u +%FT%TZ) STEP 1C: EVAL NANOLM-B7-3B (8-PILLAR + GSM8K/MATH-500/IFEVAL/HUMANEVAL) ==="
python -m lab.trackb.run_8pillar_vllm --model /workspace/models/nanolm_b7_3b --out "$OUT/pillars8_think0.json" --think 0 --max-tokens 1536 --gpu-mem 0.86 > "$OUT/p8_t0.log" 2>&1 || true
python -m lab.trackb.run_8pillar_vllm --model /workspace/models/nanolm_b7_3b --out "$OUT/pillars8_think1.json" --think 1 --max-tokens 4096 --gpu-mem 0.86 > "$OUT/p8_t1.log" 2>&1 || true
VARGS="pretrained=/workspace/models/nanolm_b7_3b,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,trust_remote_code=False"
lm_eval --model vllm --model_args "$VARGS" --tasks "gsm8k_cot_llama,minerva_math500" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/math" 2>&1 | tail -25 > "$OUT/math.log" || true
lm_eval --model vllm --model_args "$VARGS" --tasks "ifeval" --apply_chat_template --gen_kwargs "max_gen_toks=1280,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/instr" 2>&1 | tail -25 > "$OUT/instr.log" || true
lm_eval --model vllm --model_args "$VARGS" --tasks "humaneval_instruct" --apply_chat_template --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=1024,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/code" 2>&1 | tail -25 > "$OUT/code.log" || true

echo "=== $(date -u +%FT%TZ) STEP 2: REPRODUCE NANOLM-B6-2B FOR SYSTEM-V4 HEAD-TO-HEAD ==="
python -m lab.trackb.phase3_innovation --phase train_b6_2b --data-jsonl /workspace/phase3_data/stratified_b6.jsonl \
  --rlvr-jsonl /workspace/phase3_data/rlvr_b6.jsonl --out-dir /workspace/models/nanolm_b6_2b --seed 3031 \
  > /workspace/results/step2_b6_2b.log 2>&1
rm -rf /workspace/models/nanolm_b6_2b_anchored_fft

echo "=== $(date -u +%FT%TZ) STEP 3: SYSTEM-V4 HEAD-TO-HEAD (IDENTICAL HARNESS ACROSS 1.72B..120B) ==="
for spec in \
  "/workspace/models/nanolm_b6_2b:nanolm_b6_2b_v4:0.86:8192" \
  "/workspace/models/nanolm_b7_3b:nanolm_b7_3b_v4:0.86:8192" \
  "Qwen/Qwen3-1.7B:qwen3_1p7b_v4:0.85:8192" \
  "HuggingFaceTB/SmolLM3-3B:smollm3_3b_v4:0.86:8192" \
  "Qwen/Qwen3-8B:qwen3_8b_v4:0.88:8192" \
  "Qwen/Qwen3-32B:qwen3_32b_v4:0.90:8192" \
  "openai/gpt-oss-120b:gpt_oss_120b_v4:0.92:16384"
do
  IFS=':' read -r MID TAG GMEM MLEN <<< "$spec"
  mkdir -p "/workspace/results/$TAG"
  echo "=== $(date -u +%FT%TZ) COMP 1 & 2 (V4) ON $MID ==="
  python -m lab.system.run_competitions_vllm --model "$MID" --out "/workspace/results/$TAG/competitions_1_2_v4.json" \
    --gpu-mem "$GMEM" --max-model-len "$MLEN" > "/workspace/results/$TAG/run.log" 2>&1 || true
done
echo "PHASE5_ALL_COMPLETE"
