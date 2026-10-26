#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONHASHSEED=0
cleanup() {
  rm -f -- 'results/SHA256SUMS.tmp' 'nano-h7-screen-results.tar.gz.tmp' 'nano-h7-screen-results.tar.gz.sha256.tmp'
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

test -d treatments/results
test -f treatments/results/TRAINING_REPORT_SHA256SUMS
test ! -L treatments/results/TRAINING_REPORT_SHA256SUMS
test ! -e results
test ! -e h5_data
test ! -e checkpoints/anchors/nano_v01_scribe.pt
test ! -e .h7-eval-venv
(cd treatments && sha256sum -c results/TRAINING_REPORT_SHA256SUMS)
test "$(sha256sum artifacts/nano_h7/screen-v1/manifest.json | awk '{print $1}')" = "a766a41ed99bcffcd8e078c652c8a85fe6d796fe44268e03e0e309d9070cd6a6"
test "$(sha256sum artifacts/nano_h7/screen-v1/screen.jsonl | awk '{print $1}')" = "9b18c5eef105d0fa2ff0a1e575e764c0014a4ff62e40e8b8205dc12674c436f2"
test "$(sha256sum control/h6-seed-20260805-epoch-2.pt | awk '{print $1}')" = "07547ec5c01b46fa87e8d53a40497a7c4b4a6bc92b596dddd67f4be4d18c89e0"
test ! -e .h7-eval-deps
python -m pip install --disable-pip-version-check --no-cache-dir --target .h7-eval-deps -r requirements-h7-kaggle.txt
export PYTHONPATH="$PWD/.h7-eval-deps"
python "$H7_RUNTIME_PROBE"
python -c 'import platform, torch, tokenizers; assert tuple(map(int, platform.python_version_tuple()[:2])) >= (3, 10); assert tokenizers.__version__ == "0.22.2"; assert torch.__version__'
python -m pytest -q nano_ai/tests/test_evaluate_evidence_query_h7.py

report_a_sha="$(sha256sum treatments/results/seed-20260805/training_report.json | awk '{print $1}')"
report_b_sha="$(sha256sum treatments/results/seed-20260806/training_report.json | awk '{print $1}')"
mkdir results
python "$H7_EVENT_MARKER" screen_evaluation
python -m nano_ai.training.evaluate_evidence_query_h7 --screen-dir artifacts/nano_h7/screen-v1 --control-checkpoint control/h6-seed-20260805-epoch-2.pt --tokenizer sft/tokenizer.json --treatment-report treatments/results/seed-20260805/training_report.json "$report_a_sha" --treatment-report treatments/results/seed-20260806/training_report.json "$report_b_sha" --output results/h7_screen_evaluation.json --device cpu --batch-size 32 2>&1 | tee results/h7_screen_evaluation.log

find results -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > results/SHA256SUMS.tmp
mv results/SHA256SUMS.tmp results/SHA256SUMS
tar -czf nano-h7-screen-results.tar.gz.tmp results
mv nano-h7-screen-results.tar.gz.tmp nano-h7-screen-results.tar.gz
sha256sum nano-h7-screen-results.tar.gz > nano-h7-screen-results.tar.gz.sha256.tmp
mv nano-h7-screen-results.tar.gz.sha256.tmp nano-h7-screen-results.tar.gz.sha256
trap - EXIT INT TERM HUP
