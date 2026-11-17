#!/bin/bash
# Step 4 Head-to-Head Comparator Evaluation on 1x H200 SXM (141 GB):
# Evaluates identical Competition 1 (Naked Foundation) & Competition 2 (5-Engine System)
# plus 8-Pillar deterministic suite across:
#   1. Qwen/Qwen3-1.7B   (1.72B base control)
#   2. Qwen/Qwen3-8B     (8.19B medium dense comparator)
#   3. Qwen/Qwen3-32B    (32.8B large dense teacher/comparator)
#   4. openai/gpt-oss-120b (117B total / 5.1B active MoE frontier open-weight comparator)
set -euo pipefail
export HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=/workspace/nl
export HF_HOME=/workspace/hf_cache
mkdir -p /workspace/hf_cache /workspace/results

echo "=== $(date -u +%FT%TZ) INSTALL VLLM + HARMONY + SYMPY ==="
pip install -q --break-system-packages vllm huggingface_hub transformers sympy openai-harmony

cd /workspace/nl

echo "=== $(date -u +%FT%TZ) 1/4: BASE QWEN3-1.7B COMPETITIONS 1 & 2 ==="
mkdir -p /workspace/results/qwen3_1p7b_comp
python -m lab.system.run_competitions_vllm \
  --model Qwen/Qwen3-1.7B \
  --out /workspace/results/qwen3_1p7b_comp/competitions_1_2.json \
  --gpu-mem 0.85 > /workspace/results/qwen3_1p7b_comp/run.log 2>&1
rm -rf /workspace/hf_cache/hub/models--Qwen--Qwen3-1.7B*

echo "=== $(date -u +%FT%TZ) 2/4: QWEN3-8B COMPETITIONS 1 & 2 + 8-PILLAR ==="
mkdir -p /workspace/results/qwen3_8b_comp
python -m lab.system.run_competitions_vllm \
  --model Qwen/Qwen3-8B \
  --out /workspace/results/qwen3_8b_comp/competitions_1_2.json \
  --gpu-mem 0.88 > /workspace/results/qwen3_8b_comp/run.log 2>&1
python -m lab.trackb.run_8pillar_vllm \
  --model Qwen/Qwen3-8B \
  --out /workspace/results/qwen3_8b_comp/pillars8_think0.json \
  --think 0 --max-tokens 1536 --gpu-mem 0.88 > /workspace/results/qwen3_8b_comp/p8_t0.log 2>&1 || true
rm -rf /workspace/hf_cache/hub/models--Qwen--Qwen3-8B*

echo "=== $(date -u +%FT%TZ) 3/4: QWEN3-32B COMPETITIONS 1 & 2 + 8-PILLAR ==="
mkdir -p /workspace/results/qwen3_32b_comp
python -m lab.system.run_competitions_vllm \
  --model Qwen/Qwen3-32B \
  --out /workspace/results/qwen3_32b_comp/competitions_1_2.json \
  --gpu-mem 0.90 > /workspace/results/qwen3_32b_comp/run.log 2>&1
python -m lab.trackb.run_8pillar_vllm \
  --model Qwen/Qwen3-32B \
  --out /workspace/results/qwen3_32b_comp/pillars8_think0.json \
  --think 0 --max-tokens 1536 --gpu-mem 0.90 > /workspace/results/qwen3_32b_comp/p8_t0.log 2>&1 || true
rm -rf /workspace/hf_cache/hub/models--Qwen--Qwen3-32B*

echo "=== $(date -u +%FT%TZ) 4/4: OPENAI/GPT-OSS-120B (117B MoE) COMPETITIONS 1 & 2 + 8-PILLAR ==="
mkdir -p /workspace/results/gpt_oss_120b_comp
python -m lab.system.run_competitions_vllm \
  --model openai/gpt-oss-120b \
  --out /workspace/results/gpt_oss_120b_comp/competitions_1_2.json \
  --gpu-mem 0.92 --max-model-len 16384 > /workspace/results/gpt_oss_120b_comp/run.log 2>&1 || true
python -m lab.trackb.run_8pillar_vllm \
  --model openai/gpt-oss-120b \
  --out /workspace/results/gpt_oss_120b_comp/pillars8_think0.json \
  --think 0 --max-tokens 2560 --max-model-len 16384 --gpu-mem 0.92 > /workspace/results/gpt_oss_120b_comp/p8_t0.log 2>&1 || true

echo "STEP4_ALL_COMPARATORS_COMPLETE"
