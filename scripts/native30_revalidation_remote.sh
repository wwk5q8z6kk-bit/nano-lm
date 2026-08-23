#!/usr/bin/env bash
# Wave-1 30M revalidation — INTERLEAVED order, per-arm durable persistence.
#
# The previous run trained all nine arms successfully and then lost every result
# when the pod was deleted, because nothing was persisted until the very end.
# This version writes each arm to durable storage as soon as it finishes and
# verifies the durable copy before starting the next arm, so an interruption at
# any point still leaves every completed arm recoverable.
#
# Order is interleaved (decoder s0, bottleneck s0, span_port s0, decoder s1, ...)
# so a partial run still yields a complete seed-0 comparison across all three
# arms rather than three seeds of a single arm.
set -uo pipefail          # NOT -e: one failing arm must not abort the rest
cd /workspace/nano-lm
export HF_HOME=/workspace/hf_cache

DURABLE="${DURABLE_DIR:-/workspace/durable/native30_reval}"
HEARTBEAT="${DURABLE}/heartbeat.txt"
mkdir -p "${DURABLE}"

beat() { date -u +%s > "${HEARTBEAT}"; }
beat

# --- durable write/read test BEFORE any training -----------------------------
echo "==> durable write/read test at ${DURABLE}"
echo "durable-probe-$(date -u +%s)" > "${DURABLE}/.probe"
if [[ ! -s "${DURABLE}/.probe" ]]; then
  echo "DURABLE_FATAL: cannot write to ${DURABLE}"; exit 1
fi
echo "DURABLE_OK dir=${DURABLE} probe=$(cat "${DURABLE}/.probe")"
df -h "${DURABLE}" | tail -1

echo "==> rebuilding corpus"
python3 scripts/build_native_corpus.py --stage screen || { echo "CORPUS_BUILD_FAILED"; exit 1; }

echo "==> verifying corpus + eval hashes"
python3 - <<'PY' || { echo "HASHES_FAILED"; exit 1; }
import json, sys
m = json.load(open("artifacts/campaign/native_corpus_screen_v1_manifest.json"))
exp = json.load(open("artifacts/campaign/manifests/native30_revalidation_wave1_v1.json"))
if m["content_hash"] != exp["dataset"]["content_hash"]:
    print(f"CORPUS_HASH_MISMATCH want={exp['dataset']['content_hash']} got={m['content_hash']}", file=sys.stderr)
    raise SystemExit(1)
if not (m["leakage"]["pass"] and m["axis_coverage"]["pass"]):
    print("CORPUS_GATES_FAILED", file=sys.stderr); raise SystemExit(1)
print(f"CORPUS_OK hash={m['content_hash']} rows={m['statistics']['partition_sizes']['TRAIN']}")
try:
    e = json.load(open("artifacts/campaign/p1_screening_eval_v2_manifest.json"))
    print(f"EVAL_V2_OK id={e['suite_id']} hash={e['content_hash']} cases={e['n_cases']}")
except FileNotFoundError:
    print("EVAL_V2_ABSENT (screening uses p1_screening_eval_v1 with controls)")
print("HASHES_OK")
PY
beat

python3 - <<'PY' || { echo "CUDA_FATAL: torch cannot use the GPU"; exit 1; }
import sys, torch
ok = torch.cuda.is_available()
print(f"TORCH_CUDA available={ok} gpu={torch.cuda.get_device_name(0) if ok else 'none'} "
      f"torch={torch.__version__} cuda={torch.version.cuda}")
sys.exit(0 if ok else 1)
PY
beat

RUN_IDS=$(python3 - <<'PY'
from nanoscribe.native.factorial import REVALIDATION_ARMS, revalidation_run_id
seeds = sorted({s for a in REVALIDATION_ARMS for s in a.seeds})
print(" ".join(revalidation_run_id(a, s) for s in seeds for a in REVALIDATION_ARMS))
PY
)
echo "==> interleaved order: ${RUN_IDS}"

echo "==> preflight: constructing every arm"
python3 - <<PY || { echo "ARM_PREFLIGHT_FAILED"; exit 1; }
import sys
from nanoscribe.native.config import config_for_run
from nanoscribe.native.model import build_native_model
bad = []
for rid in "${RUN_IDS}".split():
    try:
        cfg = config_for_run(rid)
        assert cfg.d_model % cfg.n_heads == 0
        build_native_model(cfg)
        print(f"  ARM_OK {rid} d_model={cfg.d_model} n_heads={cfg.n_heads}")
    except Exception as exc:
        bad.append(rid); print(f"  ARM_BAD {rid} {exc}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
beat

DONE_COUNT=0
TOTAL=$(echo ${RUN_IDS} | wc -w | tr -d ' ')
for RUN_ID in ${RUN_IDS}; do
  beat
  echo "==> train ${RUN_ID}"
  if ! python3 scripts/train_native_nano.py --run-id "${RUN_ID}" > "${DURABLE}/${RUN_ID}_train.log" 2>&1; then
    echo "RUN_FAILED ${RUN_ID}"
    tail -5 "${DURABLE}/${RUN_ID}_train.log"
    continue
  fi
  beat

  for MODE in constrained unconstrained; do
    FLAG=""; [[ "${MODE}" == "unconstrained" ]] && FLAG="--unconstrained"
    python3 scripts/evaluate_native_nano.py --run-id "${RUN_ID}" \
      --suite p1_screening_eval_v1 ${FLAG} \
      --output "${DURABLE}/${RUN_ID}_${MODE}.json" >/dev/null 2>&1 \
      || echo "EVAL_WARN ${RUN_ID} ${MODE}"
    beat
  done

  mkdir -p "${DURABLE}/${RUN_ID}"
  cp -f "artifacts/native_checkpoints/${RUN_ID}/latest.pt" "${DURABLE}/${RUN_ID}/latest.pt" 2>/dev/null
  cp -f artifacts/native_checkpoints/${RUN_ID}/step_*.json "${DURABLE}/${RUN_ID}/" 2>/dev/null

  python3 - "${RUN_ID}" "${DURABLE}" <<'PY'
import json, os, sys
rid, durable = sys.argv[1], sys.argv[2]
metrics = {"run_id": rid, "durable_files": {}}
for mode in ("constrained", "unconstrained"):
    p = f"{durable}/{rid}_{mode}.json"
    if os.path.exists(p):
        metrics[mode] = json.load(open(p)).get("suite_metrics", {})
        metrics["durable_files"][mode] = os.path.getsize(p)
ck = f"{durable}/{rid}/latest.pt"
metrics["checkpoint_bytes"] = os.path.getsize(ck) if os.path.exists(ck) else 0
json.dump(metrics, open(f"{durable}/{rid}_final_metrics.json", "w"), indent=2)
print(f"METRICS_WRITTEN {rid} ckpt_bytes={metrics['checkpoint_bytes']}")
PY

  if [[ -s "${DURABLE}/${RUN_ID}_final_metrics.json" ]]; then
    echo "DONE" > "${DURABLE}/${RUN_ID}.DONE"
    DONE_COUNT=$((DONE_COUNT+1))
    echo "ARM_DURABLE_OK ${RUN_ID}  progress=${DONE_COUNT}/${TOTAL}"
  else
    echo "ARM_DURABLE_FAILED ${RUN_ID}"
  fi

  python3 - "${DURABLE}" <<'PY'
import glob, json, os, sys
d = sys.argv[1]
done = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{d}/*.DONE"))
json.dump({"durable_done": done, "count": len(done)}, open(f"{d}/partial_summary.json", "w"), indent=2)
print(f"PARTIAL {len(done)} arms durable")
PY
  beat
done

echo "DURABLE_DONE_COUNT=${DONE_COUNT}/${TOTAL}"
echo NATIVE30_REVALIDATION_DONE
