#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
cleanup() {
  rm -f -- 'results/.TRAINING_REPORT_SHA256SUMS.tmp'
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

test ! -e results
test ! -e h2_development
test ! -e .h6-venv
test "$(sha256sum h6_data/manifest.json | awk '{print $1}')" = "2569e7b27b53ef741fc92d1545ca323367155d960e77ff0e6852b32db0d32f31"
python -m venv --system-site-packages .h6-venv
export PATH="$PWD/.h6-venv/bin:$PATH"
python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-h4-runpod.txt
python -c 'import platform, torch, tokenizers; assert platform.python_version().startswith("3.12."); assert torch.__version__ == "2.8.0+cu128"; assert torch.version.cuda == "12.8"; assert tokenizers.__version__ == "0.22.2"; assert torch.cuda.is_available(); print("DISCLOSED_GPU:", torch.cuda.get_device_name(0)); assert torch.cuda.get_device_properties(0).total_memory >= 20 * 1024**3'
python -m pytest -q nano_ai/tests/test_state_conditioned_evidence_query_model.py nano_ai/tests/test_replay_mixture_data.py nano_ai/tests/test_train_evidence_query_h6.py nano_ai/tests/test_evaluate_evidence_query_h6.py nano_ai/tests/test_package_evidence_query_h6.py

test ! -e h2_development
mkdir results
python -m nano_ai.training.train_evidence_query_h6 --data-dir h6_data --base-checkpoint checkpoints/anchors/nano_v01_scribe.pt --tokenizer sft/tokenizer.json --output-dir results/seed-20260805 --seed 20260805 --device cuda 2>&1 | tee results/seed-20260805.log
test ! -e h2_development
python -m nano_ai.training.train_evidence_query_h6 --data-dir h6_data --base-checkpoint checkpoints/anchors/nano_v01_scribe.pt --tokenizer sft/tokenizer.json --output-dir results/seed-20260806 --seed 20260806 --device cuda 2>&1 | tee results/seed-20260806.log
test ! -e h2_development

sha256sum results/seed-20260805/training_report.json results/seed-20260806/training_report.json > results/.TRAINING_REPORT_SHA256SUMS.tmp
mv results/.TRAINING_REPORT_SHA256SUMS.tmp results/TRAINING_REPORT_SHA256SUMS
trap - EXIT INT TERM HUP
