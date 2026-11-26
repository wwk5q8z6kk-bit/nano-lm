#!/bin/bash
# Phase 2 Flagship GPU Pipeline (runs on 1x H200 141GB or H100 80GB):
#   1. Info-Dense Student Probe + Qwen3-32B Verified Trajectory Generation
#   2. Controlled Ablation: Uniform vs InfoDense AND rsLoRA vs Full Fine-Tuning (FFT)
#   3. Online Verification-Driven RLVR (GRPO) -> NanoLM-B5-RLVR-2B
#   4. Separated Competition 1 (Naked) & Competition 2 (5-Engine System) Head-to-Head
#      across Base 2B, NanoLM-B4/B5, Qwen3-8B, Qwen3-32B, and openai/gpt-oss-120b!
set -euo pipefail
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/nl
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache /workspace/phase2_data /workspace/models /workspace/results

echo "=== $(date -u +%FT%TZ) STEP 0: INSTALL VLLM & LM_EVAL ==="
pip install -q --break-system-packages vllm "lm_eval[vllm]" datasets accelerate math_verify

cd /workspace/nl

echo "=== $(date -u +%FT%TZ) STEP 1: STUDENT SURPRISE PROBE + QWEN3-32B VERIFIED CURRICULUM ==="
python -m lab.trackb.phase2_learning \
  --phase probe_and_generate \
  --student Qwen/Qwen3-1.7B \
  --teacher Qwen/Qwen3-32B \
  --data-dir /workspace/phase2_data \
  --subset-size 420 \
  --seed 2027 \
  2>&1 | tee /workspace/results/step1_probe_gen.log
cp /workspace/phase2_data/curriculum_diagnostics.json /workspace/results/curriculum_diagnostics.json

echo "=== $(date -u +%FT%TZ) STEP 2A: B4a_uniform_rslora_2b (Uniform Curriculum + rsLoRA r=32 + Qwen3-32B KD) ==="
python -m lab.trackb.phase2_learning \
  --phase distill \
  --student Qwen/Qwen3-1.7B \
  --teacher Qwen/Qwen3-32B \
  --data-jsonl /workspace/phase2_data/uniform.jsonl \
  --out-dir /workspace/models/nanolm_b4a_uniform_rslora_2b \
  --mode rslora --rank 32 --alpha-kd 0.85 --epochs 1 --seed 2027 \
  2>&1 | tee /workspace/results/step2a_b4a.log

echo "=== $(date -u +%FT%TZ) STEP 2B: B4b_infodense_rslora_2b (Info-Dense Curriculum + rsLoRA r=32 + Qwen3-32B KD) ==="
python -m lab.trackb.phase2_learning \
  --phase distill \
  --student Qwen/Qwen3-1.7B \
  --teacher Qwen/Qwen3-32B \
  --data-jsonl /workspace/phase2_data/infodense.jsonl \
  --kd-cache /workspace/phase2_data/infodense_kd.pt \
  --out-dir /workspace/models/nanolm_b4b_infodense_rslora_2b \
  --mode rslora --rank 32 --alpha-kd 0.85 --epochs 1 --seed 2027 \
  2>&1 | tee /workspace/results/step2b_b4b.log

echo "=== $(date -u +%FT%TZ) STEP 2C: B4c_infodense_fft_2b (Info-Dense Curriculum + 100% FULL FINE-TUNING + Qwen3-32B KD) ==="
python -m lab.trackb.phase2_learning \
  --phase distill \
  --student Qwen/Qwen3-1.7B \
  --teacher Qwen/Qwen3-32B \
  --data-jsonl /workspace/phase2_data/infodense.jsonl \
  --kd-cache /workspace/phase2_data/infodense_kd.pt \
  --out-dir /workspace/models/nanolm_b4c_infodense_fft_2b \
  --mode fft --alpha-kd 0.85 --epochs 1 --seed 2027 \
  2>&1 | tee /workspace/results/step2c_b4c.log
rm -f /workspace/phase2_data/infodense_kd.pt

eval_2b_variant() {
  local MDIR=$1 TAG=$2 FULL_KNOW=$3
  local OUT=/workspace/results/$TAG
  mkdir -p "$OUT"
  cp "$MDIR/nanolm_training_manifest.json" "$OUT/nanolm_training_manifest.json" 2>/dev/null || true
  echo "=== $(date -u +%FT%TZ) EVAL 8-PILLAR & COMPETITIONS 1+2 ON $TAG ==="
  python -m lab.trackb.run_8pillar_vllm --model "$MDIR" --out "$OUT/pillars8_think0.json" --think 0 --max-tokens 1536 > "$OUT/pillars8_think0.log" 2>&1
  python -m lab.trackb.run_8pillar_vllm --model "$MDIR" --out "$OUT/pillars8_think1.json" --think 1 --max-tokens 4096 > "$OUT/pillars8_think1.log" 2>&1
  python -m lab.system.run_competitions_vllm --model "$MDIR" --out "$OUT/competitions_1_2.json" --gpu-mem 0.85 > "$OUT/competitions_1_2.log" 2>&1

  local VARGS="pretrained=$MDIR,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,trust_remote_code=False,enable_thinking=False"
  lm_eval --model vllm --model_args "$VARGS" --tasks "gsm8k_cot_llama,minerva_math500" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/math" 2>&1 | tail -25 > "$OUT/math.log"
  lm_eval --model vllm --model_args "$VARGS" --tasks "ifeval" --apply_chat_template --gen_kwargs "max_gen_toks=1280,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/instr" 2>&1 | tail -25 > "$OUT/instr.log"
  lm_eval --model vllm --model_args "$VARGS" --tasks "humaneval_instruct" --apply_chat_template --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=1024,temperature=0" --batch_size auto --seed 1234 --output_path "$OUT/code" 2>&1 | tail -25 > "$OUT/code.log"
  if [ "$FULL_KNOW" = "1" ]; then
    lm_eval --model vllm --model_args "$VARGS" --tasks "arc_challenge,winogrande,truthfulqa_mc2" --num_fewshot 5 --batch_size auto --seed 1234 --output_path "$OUT/know_fast" 2>&1 | tail -25 > "$OUT/know_fast.log"
  fi
}

eval_2b_variant "/workspace/models/nanolm_b4a_uniform_rslora_2b" "nanolm_b4a_uniform_rslora_2b" "0"
eval_2b_variant "/workspace/models/nanolm_b4b_infodense_rslora_2b" "nanolm_b4b_infodense_rslora_2b" "0"
eval_2b_variant "/workspace/models/nanolm_b4c_infodense_fft_2b" "nanolm_b4c_infodense_fft_2b" "0"

# Select best B4 model on Competition 1 + 8-pillar mean to seed Online RLVR (B5)
BEST_B4=$(python -c '
import json
cands = ["nanolm_b4b_infodense_rslora_2b", "nanolm_b4c_infodense_fft_2b", "nanolm_b4a_uniform_rslora_2b"]
def sc(t):
    p0 = json.load(open(f"/workspace/results/{t}/pillars8_think0.json"))["overall_pass_rate"]
    c1 = json.load(open(f"/workspace/results/{t}/competitions_1_2.json"))["competition_1_naked_fast"]["overall_score"]
    return 0.5 * p0 + 0.5 * c1
best = max(cands, key=sc)
print(f"/workspace/models/{best}")
')
echo "=== $(date -u +%FT%TZ) STEP 3: ONLINE RLVR (GRPO) SEEDED FROM $BEST_B4 ==="
python -m lab.trackb.phase2_learning \
  --phase rlvr \
  --student "$BEST_B4" \
  --rlvr-jsonl /workspace/phase2_data/rlvr_pool.jsonl \
  --out-dir /workspace/models/nanolm_b5_rlvr_2b \
  --rlvr-steps 100 --group-size 4 --kl-beta 0.04 --seed 2027 \
  2>&1 | tee /workspace/results/step3_b5_rlvr.log

eval_2b_variant "/workspace/models/nanolm_b5_rlvr_2b" "nanolm_b5_rlvr_2b" "1"

# Free intermediate B4a/B4c model dirs to keep plenty of disk for 120B
rm -rf /workspace/models/nanolm_b4a_uniform_rslora_2b /workspace/models/nanolm_b4c_infodense_fft_2b

echo "=== $(date -u +%FT%TZ) STEP 4A: COMPETITIONS 1 & 2 ON BASE QWEN3-1.7B, QWEN3-8B, & QWEN3-32B ==="
mkdir -p /workspace/results/qwen3_1p7b_comp /workspace/results/qwen3_8b_comp /workspace/results/qwen3_32b_comp
python -m lab.system.run_competitions_vllm --model Qwen/Qwen3-1.7B --out /workspace/results/qwen3_1p7b_comp/competitions_1_2.json --gpu-mem 0.85 > /workspace/results/qwen3_1p7b_comp/run.log 2>&1
python -m lab.system.run_competitions_vllm --model Qwen/Qwen3-8B --out /workspace/results/qwen3_8b_comp/competitions_1_2.json --gpu-mem 0.88 > /workspace/results/qwen3_8b_comp/run.log 2>&1
python -m lab.trackb.run_8pillar_vllm --model Qwen/Qwen3-8B --out /workspace/results/qwen3_8b_comp/pillars8_think0.json --think 0 --max-tokens 1536 > /workspace/results/qwen3_8b_comp/p8.log 2>&1
python -m lab.system.run_competitions_vllm --model Qwen/Qwen3-32B --out /workspace/results/qwen3_32b_comp/competitions_1_2.json --gpu-mem 0.92 > /workspace/results/qwen3_32b_comp/run.log 2>&1
python -m lab.trackb.run_8pillar_vllm --model Qwen/Qwen3-32B --out /workspace/results/qwen3_32b_comp/pillars8_think0.json --think 0 --max-tokens 1536 > /workspace/results/qwen3_32b_comp/p8.log 2>&1

# Purge Qwen3-32B from HF cache before pulling ~65GB openai/gpt-oss-120b!
rm -rf /workspace/hf_cache/hub/models--Qwen--Qwen3-32B*

echo "=== $(date -u +%FT%TZ) STEP 4B: HEAD-TO-HEAD AGAINST OPENAI/GPT-OSS-120B (117B MoE) ==="
mkdir -p /workspace/results/gpt_oss_120b_comp
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
if [ "${VRAM_MB:-80000}" -ge 90000 ]; then
  MAXLEN_120B=16384
else
  MAXLEN_120B=8192
fi
echo "Detected VRAM=${VRAM_MB}MB -> using MAXLEN_120B=${MAXLEN_120B}"
if python -m lab.system.run_competitions_vllm --model openai/gpt-oss-120b --out /workspace/results/gpt_oss_120b_comp/competitions_1_2.json --gpu-mem 0.93 --max-model-len "$MAXLEN_120B" --skip-think1-naked > /workspace/results/gpt_oss_120b_comp/run.log 2>&1; then
  echo "gpt-oss-120b Competitions 1 & 2 succeeded!"
  python -m lab.trackb.run_8pillar_vllm --model openai/gpt-oss-120b --out /workspace/results/gpt_oss_120b_comp/pillars8_think0.json --think 0 --max-tokens 2048 --max-model-len "$MAXLEN_120B" --gpu-mem 0.93 > /workspace/results/gpt_oss_120b_comp/p8.log 2>&1 || true
else
  echo "WARNING: gpt-oss-120b encountered runtime issue; check /workspace/results/gpt_oss_120b_comp/run.log"
fi

echo "PHASE2_FLAGSHIP_ALL_COMPLETE"
