#!/usr/bin/env bash
# Remote entrypoint — Wave-1 30M revalidation on the real corpus.
#
# The corpus is NOT in git (66MB of generated JSON does not belong there). It is
# rebuilt here and the content_hash is verified against the committed manifest,
# so the pod provably trains on the same corpus that passed the leakage and
# coverage gates locally.
set -euo pipefail
cd /workspace/nano-lm
export HF_HOME=/workspace/hf_cache

RUN_IDS="${REVAL_RUN_IDS:-}"
if [[ -z "${RUN_IDS}" ]]; then
  RUN_IDS=$(python3 - <<'PY'
from nanoscribe.native.factorial import REVALIDATION_ARMS, revalidation_run_id
print(" ".join(revalidation_run_id(a, s) for a in REVALIDATION_ARMS for s in a.seeds))
PY
)
fi

echo "==> rebuilding corpus"
python3 scripts/build_native_corpus.py --stage screen

echo "==> verifying corpus content_hash against committed manifest"
python3 - <<'PY'
import json, sys
m = json.load(open("artifacts/campaign/native_corpus_screen_v1_manifest.json"))
exp = json.load(open("artifacts/campaign/manifests/native30_revalidation_wave1_v1.json"))
want = exp["dataset"]["content_hash"]
got = m["content_hash"]
if got != want:
    print(f"CORPUS_HASH_MISMATCH want={want} got={got}", file=sys.stderr)
    raise SystemExit(1)
if not (m["leakage"]["pass"] and m["axis_coverage"]["pass"]):
    print("CORPUS_GATES_FAILED", file=sys.stderr)
    raise SystemExit(1)
print(f"CORPUS_OK hash={got} rows={m['statistics']['partition_sizes']['TRAIN']}")
PY

if ! python3 -m nanoscribe.runpod_gpu_preflight; then
  echo "PREFLIGHT_WARN: preflight failed; checking nvidia-smi"
  nvidia-smi || { echo "PREFLIGHT_FATAL: no GPU"; exit 1; }
  echo "PREFLIGHT_OVERRIDE: nvidia-smi ok, continuing"
fi

mkdir -p /workspace/reval_results
for RUN_ID in ${RUN_IDS}; do
  echo "==> train ${RUN_ID}"
  python3 scripts/train_native_nano.py --run-id "${RUN_ID}" 2>&1 | tail -5

  # Evaluate on the pod: the GPU is already paid for, and pulling nine 30M
  # checkpoints over SSH to evaluate on a laptop would cost far more wall clock
  # than the training itself.
  for MODE in constrained unconstrained; do
    FLAG=""
    [[ "${MODE}" == "unconstrained" ]] && FLAG="--unconstrained"
    echo "==> eval ${RUN_ID} ${MODE}"
    python3 scripts/evaluate_native_nano.py \
      --run-id "${RUN_ID}" --suite p1_screening_eval_v1 ${FLAG} \
      --output "/workspace/reval_results/${RUN_ID}_${MODE}.json" >/dev/null 2>&1 \
      || echo "EVAL_WARN ${RUN_ID} ${MODE}"
  done
  echo "RUN_DONE ${RUN_ID}"
done

echo "==> archiving"
mkdir -p /workspace/campaign_native_checkpoints
python3 - <<'PY'
import shutil
from pathlib import Path
src = Path("artifacts/native_checkpoints")
dst = Path("/workspace/campaign_native_checkpoints")
dst.mkdir(parents=True, exist_ok=True)
for run_dir in src.glob("reval30_*"):
    if run_dir.is_dir():
        shutil.copytree(run_dir, dst / run_dir.name, dirs_exist_ok=True)
print("ARCHIVE_OK", dst)
PY
echo NATIVE30_REVALIDATION_DONE
