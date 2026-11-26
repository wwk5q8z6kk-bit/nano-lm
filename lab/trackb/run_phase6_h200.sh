#!/bin/bash
# Phase 6 (H200):
#   1. Train NanoLM-B9-2B (1.72B < 2.0B: Causal-Completed Curriculum + Same-Tokenizer Qwen3-32B Top-64+Tail KD
#      + Proximal L2 100% FFT + G=8 GRPO + WiSE-FT 0.25) + full 8-pillar & lm_eval benchmarks.
#   2. Reproduce NanoLM-B8-3B (3.08B < 4.0B) for System-v5 head-to-head.
#   3. Evaluate NanoLMSystem-v5 (Differential Code Oracle + Math Discrepancy Adjudicator + Delimited JSON
#      Python Synthesis + Schema-Aware Read-vs-Mutate Premature-Stop Guard) under IDENTICAL system code
#      across 1.72B, 3.08B, 8B, 32B, and 120B.
set -euo pipefail
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/nl
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache /workspace/phase3_data /workspace/models /workspace/results
cd /workspace/nl

echo "=== $(date -u +%FT%TZ) STEP 0: INSTALL ==="
pip install -q --break-system-packages vllm "lm_eval[vllm,math,ifeval]" datasets accelerate math_verify sympy antlr4-python3-runtime==4.11 langdetect immutabledict nltk openai-harmony

echo "=== $(date -u +%FT%TZ) STEP 1A: TRAIN NANOLM-B9-2B (1.72B CAUSAL-COMPLETED + 32B KD + GRPO + WISE-FT) ==="
python -m lab.trackb.phase3_innovation --phase train_b9_2b --data-jsonl /workspace/phase3_data/stratified_b6.jsonl \
  --rlvr-jsonl /workspace/phase3_data/rlvr_b6.jsonl --out-dir /workspace/models/nanolm_b9_2b --seed 3031 \
  > /workspace/results/step1_b9_2b.log 2>&1
rm -rf /workspace/models/nanolm_b9_2b_anchored_fft

OUT2=/workspace/results/nanolm_b9_2b
mkdir -p "$OUT2"
cp /workspace/models/nanolm_b9_2b/nanolm_training_manifest.json "$OUT2/" || true
echo "=== $(date -u +%FT%TZ) STEP 1B: EVAL NANOLM-B9-2B (8-PILLAR + GSM8K/MATH-500/IFEVAL/HUMANEVAL) ==="
python -m lab.trackb.run_8pillar_vllm --model /workspace/models/nanolm_b9_2b --out "$OUT2/pillars8_think0.json" --think 0 --max-tokens 1536 --gpu-mem 0.86 > "$OUT2/p8_t0.log" 2>&1 || true
python -m lab.trackb.run_8pillar_vllm --model /workspace/models/nanolm_b9_2b --out "$OUT2/pillars8_think1.json" --think 1 --max-tokens 4096 --gpu-mem 0.86 > "$OUT2/p8_t1.log" 2>&1 || true
VARGS2="pretrained=/workspace/models/nanolm_b9_2b,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,trust_remote_code=False"
lm_eval --model vllm --model_args "$VARGS2" --tasks "gsm8k_cot_llama,minerva_math500" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT2/math" 2>&1 | tail -25 > "$OUT2/math.log" || true
lm_eval --model vllm --model_args "$VARGS2" --tasks "ifeval" --apply_chat_template --gen_kwargs "max_gen_toks=1280,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT2/instr" 2>&1 | tail -25 > "$OUT2/instr.log" || true
lm_eval --model vllm --model_args "$VARGS2" --tasks "humaneval_instruct" --apply_chat_template --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=1024,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT2/code" 2>&1 | tail -25 > "$OUT2/code.log" || true

echo "=== $(date -u +%FT%TZ) STEP 2: REPRODUCE NANOLM-B8-3B (3.08B) FOR SYSTEM-V5 HEAD-TO-HEAD ==="
python -m lab.trackb.phase3_innovation --phase train_b8_3b --data-jsonl /workspace/phase3_data/stratified_b6.jsonl \
  --rlvr-jsonl /workspace/phase3_data/rlvr_b6.jsonl --out-dir /workspace/models/nanolm_b8_3b --seed 3031 \
  > /workspace/results/step2_b8_3b.log 2>&1
rm -rf /workspace/models/nanolm_b8_3b_self_anchored_fft

echo "=== $(date -u +%FT%TZ) STEP 3: SYSTEM-V5 HEAD-TO-HEAD (IDENTICAL HARNESS ACROSS 1.72B..120B) ==="
for spec in \
  "/workspace/models/nanolm_b9_2b:nanolm_b9_2b_v5:0.86:8192" \
  "/workspace/models/nanolm_b8_3b:nanolm_b8_3b_v5:0.86:8192" \
  "Qwen/Qwen3-1.7B:qwen3_1p7b_v5:0.85:8192" \
  "HuggingFaceTB/SmolLM3-3B:smollm3_3b_v5:0.86:8192" \
  "Qwen/Qwen3-8B:qwen3_8b_v5:0.88:8192" \
  "Qwen/Qwen3-32B:qwen3_32b_v5:0.90:8192" \
  "openai/gpt-oss-120b:gpt_oss_120b_v5:0.92:16384"
do
  IFS=':' read -r MID TAG GMEM MLEN <<< "$spec"
  mkdir -p "/workspace/results/$TAG"
  echo "=== $(date -u +%FT%TZ) COMP 1 & 2 (V5) ON $MID ==="
  python -m lab.system.run_competitions_vllm --model "$MID" --out "/workspace/results/$TAG/competitions_1_2_v5.json" \
    --gpu-mem "$GMEM" --max-model-len "$MLEN" > "/workspace/results/$TAG/run.log" 2>&1 || true
done
echo "PHASE6_ALL_COMPLETE"
