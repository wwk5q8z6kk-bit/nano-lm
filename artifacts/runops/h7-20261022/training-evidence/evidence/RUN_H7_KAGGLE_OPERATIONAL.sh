#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONHASHSEED=0
cleanup() {
  rm -f -- 'results/.TRAINING_REPORT_SHA256SUMS.tmp' 'nano-h7-treatment-results.tar.gz.tmp' 'nano-h7-treatment-results.tar.gz.sha256.tmp'
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

test ! -e results
test ! -e h7_screen
test ! -e control
test ! -e ../nano-h7-evaluation
test ! -e .h7-train-venv
test "$(sha256sum h5_data/manifest.json | awk '{print $1}')" = "2569e7b27b53ef741fc92d1545ca323367155d960e77ff0e6852b32db0d32f31"
test "$(sha256sum h5_data/fit.jsonl | awk '{print $1}')" = "79f7581efbd989f48a474bdd2079e7ca163c2a2682bb48fb8398bd99febe22fb"
test "$(sha256sum h5_data/calibration.jsonl | awk '{print $1}')" = "ce0562ccb44ee83963eace0d873773addaee8e49f29499a6b720b16335930e70"
test ! -e .h7-train-deps
python -m pip install --disable-pip-version-check --no-cache-dir --target .h7-train-deps -r requirements-h7-kaggle.txt
export PYTHONPATH="$PWD/.h7-train-deps"
python "$H7_RUNTIME_PROBE"
python -c 'import platform, torch, tokenizers; assert tuple(map(int, platform.python_version_tuple()[:2])) >= (3, 10); assert tokenizers.__version__ == "0.22.2"; assert torch.__version__; print("runtime", platform.python_version(), torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pytest -q nano_ai/tests/test_boundary_margin.py nano_ai/tests/test_state_conditioned_evidence_query_model.py nano_ai/tests/test_replay_mixture_data.py nano_ai/tests/test_train_evidence_query_h7.py

test ! -e h7_screen
mkdir results
python "$H7_EVENT_MARKER" training_seed_20260805
python -m nano_ai.training.train_evidence_query_h7 --data-dir h5_data --base-checkpoint checkpoints/anchors/nano_v01_scribe.pt --tokenizer sft/tokenizer.json --output-dir results/seed-20260805 --seed 20260805 --device cpu 2>&1 | tee results/seed-20260805.log
test ! -e h7_screen
python "$H7_EVENT_MARKER" training_seed_20260806
python -m nano_ai.training.train_evidence_query_h7 --data-dir h5_data --base-checkpoint checkpoints/anchors/nano_v01_scribe.pt --tokenizer sft/tokenizer.json --output-dir results/seed-20260806 --seed 20260806 --device cpu 2>&1 | tee results/seed-20260806.log
test ! -e h7_screen

sha256sum results/seed-20260805/training_report.json results/seed-20260806/training_report.json > results/.TRAINING_REPORT_SHA256SUMS.tmp
mv results/.TRAINING_REPORT_SHA256SUMS.tmp results/TRAINING_REPORT_SHA256SUMS
find results -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > results/SHA256SUMS
tar -czf nano-h7-treatment-results.tar.gz.tmp results
mv nano-h7-treatment-results.tar.gz.tmp nano-h7-treatment-results.tar.gz
sha256sum nano-h7-treatment-results.tar.gz > nano-h7-treatment-results.tar.gz.sha256.tmp
mv nano-h7-treatment-results.tar.gz.sha256.tmp nano-h7-treatment-results.tar.gz.sha256
trap - EXIT INT TERM HUP
