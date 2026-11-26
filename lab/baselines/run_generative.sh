#!/bin/zsh
# LAB-BASE-02: generative baselines (chat template, greedy). Usage: run_generative.sh <hf_model> <tag> [limit]
set -e
M=$1; TAG=$2; LIM=${3:-250}
cd "$(dirname $0)/../.."
HF_ALLOW_CODE_EVAL=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
${LM_EVAL_BIN:-lm_eval} --model hf \
  --model_args pretrained=$M,dtype=bfloat16 --device mps --batch_size 8 \
  --tasks gsm8k,ifeval,humaneval_instruct,mbpp_instruct --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs max_gen_toks=512 --confirm_run_unsafe_code \
  --limit $LIM --seed 1234 --output_path lab/baselines/results/${TAG}_gen --log_samples
