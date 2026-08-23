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

# CUDA must be usable by TORCH, not merely visible to nvidia-smi. The previous
# "nvidia-smi ok, continuing" override let a run proceed on CPU: the GPU was
# visible (device_count=1) while torch.cuda.is_available() was False, so the pod
# billed at $1.59/hr to train ~30x slower than it should. A pod that cannot use
# its GPU has no valid experiment attached and must stop, not continue.
python3 - <<'PY' || { echo "CUDA_FATAL: torch cannot use the GPU; aborting rather than training on CPU"; exit 1; }
import sys
import torch

ok = torch.cuda.is_available()
print(
    f"TORCH_CUDA available={ok} torch={torch.__version__} "
    f"built_for_cuda={torch.version.cuda} device_count={torch.cuda.device_count()}"
)
if not ok:
    print(
        "DIAGNOSIS: torch is present and the GPU is visible, but CUDA is "
        "unusable — usually a torch wheel built for a newer CUDA than the "
        "driver provides (e.g. torch+cu130 on a CUDA 12.4 driver).",
        file=sys.stderr,
    )
sys.exit(0 if ok else 1)
PY

# Build EVERY arm before training ANY of them. A 9-run job once died three runs
# in because one arm had an invalid head count — the pod billed for the failed
# arm and every arm after it. This costs ~20s and fails the whole job up front.
echo "==> preflight: constructing every arm"
python3 - <<PY || { echo "ARM_PREFLIGHT_FAILED: an arm cannot be constructed; aborting before training"; exit 1; }
import sys
from nanoscribe.native.config import config_for_run
from nanoscribe.native.model import build_native_model

run_ids = "${RUN_IDS}".split()
bad = []
for rid in run_ids:
    try:
        cfg = config_for_run(rid)
        assert cfg.d_model % cfg.n_heads == 0, (
            f"d_model {cfg.d_model} not divisible by n_heads {cfg.n_heads}"
        )
        build_native_model(cfg)
        print(f"  ARM_OK {rid} d_model={cfg.d_model} n_heads={cfg.n_heads}")
    except Exception as exc:
        bad.append((rid, f"{type(exc).__name__}: {exc}"))
        print(f"  ARM_BAD {rid} {type(exc).__name__}: {exc}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY

mkdir -p /workspace/reval_results
FAILED_RUNS=""
for RUN_ID in ${RUN_IDS}; do
  echo "==> train ${RUN_ID}"
  # A single failing run must not abort the remaining eight — the pod is already
  # paid for and the other arms are still worth collecting.
  if ! python3 scripts/train_native_nano.py --run-id "${RUN_ID}" 2>&1 | tail -5; then
    echo "RUN_FAILED ${RUN_ID}"
    FAILED_RUNS="${FAILED_RUNS} ${RUN_ID}"
    continue
  fi

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
if [[ -n "${FAILED_RUNS}" ]]; then
  echo "RUNS_FAILED:${FAILED_RUNS}"
fi
echo NATIVE30_REVALIDATION_DONE
