#!/bin/bash
# Track B competitive baseline harness (runs ON a GPU pod).  LAB-BASE-03.
# usage: eval_pod.sh <hf_model_id> <tag> [think=0|1]
# Full test sets (no --limit). Greedy decoding. Chat template for instruct models.
# Every result directory records: model revision, dtype, max context, think mode, harness version.
set -euo pipefail
M=$1; TAG=$2; THINK=${3:-0}
OUT=/workspace/results/$TAG; mkdir -p $OUT
export HF_ALLOW_CODE_EVAL=1 HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
REV=$(python -c "from huggingface_hub import model_info;print(model_info('$M').sha)")
CTK=""; [ "$THINK" = "0" ] && CTK=",enable_thinking=False"
VARGS="pretrained=$M,revision=$REV,dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=8192,trust_remote_code=False$CTK"
echo "{\"model\":\"$M\",\"revision\":\"$REV\",\"think\":$THINK,\"lm_eval\":\"$(python -c 'import lm_eval;print(lm_eval.__version__)')\",\"vllm\":\"$(python -c 'import vllm;print(vllm.__version__)')\"}" > $OUT/conditions.json

run() {  # name, tasks, extra args
  local name=$1 tasks=$2; shift 2
  lm_eval --model vllm --model_args "$VARGS" --tasks "$tasks" --batch_size auto --seed 1234 \
    --output_path $OUT/$name --log_samples "$@" 2>&1 | tail -40 > $OUT/$name.log || echo "FAILED $name" >> $OUT/failures.txt
}
# Knowledge / reasoning (log-likelihood, base-style prompts; standard few-shot)
run knowledge "mmlu,arc_challenge,hellaswag,winogrande,truthfulqa_mc2" --num_fewshot 5
# Generative, chat-templated
run math "gsm8k_cot_llama,minerva_math500" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0"
run hard "mmlu_pro,gpqa_diamond_cot_zeroshot,bbh_cot_fewshot" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=2048,temperature=0"
run instr "ifeval" --apply_chat_template --gen_kwargs "max_gen_toks=1280,temperature=0"
run code "humaneval_instruct,mbpp_plus_instruct" --apply_chat_template --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=1024,temperature=0"
run multiling "mgsm_native_cot" --apply_chat_template --fewshot_as_multiturn --gen_kwargs "max_gen_toks=1024,temperature=0"
echo DONE > $OUT/DONE
