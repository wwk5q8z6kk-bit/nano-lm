#!/usr/bin/env bash
# Poll Kaggle native30 kernel; on COMPLETE download reval_results and import summary.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KERNEL="hassaneljesr/nano-native30-revalidation-v1"
DOWN="$ROOT/artifacts/campaign/kaggle_native30_download"
RESULTS="$ROOT/artifacts/campaign/reval_results"

cd "$ROOT"

status_line="$(kaggle kernels status "$KERNEL" 2>/dev/null || true)"
echo "$status_line"
if grep -q 'COMPLETE' <<<"$status_line"; then
  : 
elif grep -q 'RUNNING' <<<"$status_line"; then
  echo "Kernel still RUNNING — re-run when COMPLETE"
  exit 2
elif grep -q 'ERROR' <<<"$status_line"; then
  echo "Kernel ERROR — check logs: kaggle kernels logs $KERNEL"
  exit 1
else
  echo "Unknown kernel status"
  exit 1
fi

rm -rf "$DOWN"
mkdir -p "$DOWN"
kaggle kernels output "$KERNEL" -p "$DOWN"

# Kernel writes to /kaggle/working/reval_results — may appear nested in download.
src="$DOWN"
if [[ -d "$DOWN/reval_results" ]]; then
  src="$DOWN/reval_results"
fi

mkdir -p "$RESULTS"
cp -f "$src"/*_train.json "$RESULTS/" 2>/dev/null || true
cp -f "$src"/*_constrained.json "$RESULTS/" 2>/dev/null || true
cp -f "$src"/*_unconstrained.json "$RESULTS/" 2>/dev/null || true

python3 scripts/import_kaggle_native30_results.py --results-dir "$RESULTS"
echo "Done — see artifacts/campaign/native30_revalidation_summary_v1.json"
