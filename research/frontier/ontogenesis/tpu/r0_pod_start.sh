#!/bin/bash
# Runs inside the RunPod container (R0, RESEARCH-PLAN.md 20.6). Two in-pod shutdown paths, started before anything else:
#   1. a hard timer, R0_HARD_S seconds after container start, independent of the benchmark process;
#   2. normal end: after the benchmark and a short grace period for the laptop to read the logs.
# The laptop controller (r0_run.py) adds a third path (cost guard and deadline). Termination uses the CLI inside the image,
# or the REST API with the pod's own injected credential as a fallback.
set -u
START=$(date +%s)
MODE=${R0_MODE:-r0}
HARD=${R0_HARD_S:-3300}
GRACE=${R0_GRACE_S:-120}
terminate() {
  echo "R0_TERMINATE_CALLED elapsed=$(( $(date +%s) - START ))s"
  for i in 1 2 3 4 5 6; do
    if command -v runpodctl >/dev/null 2>&1; then
      runpodctl pod delete "$RUNPOD_POD_ID" 2>&1 | head -c 200; runpodctl remove pod "$RUNPOD_POD_ID" 2>&1 | head -c 200       # new and old CLI syntax
    fi
    code=$(curl -s -m 20 -o /tmp/term.out -w '%{http_code}' -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY:-none}" "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID:-none}")
    echo "R0_TERMINATE_REST attempt=$i http=$code body=$(head -c 160 /tmp/term.out 2>/dev/null)"
    code2=$(curl -s -m 20 -o /tmp/term2.out -w '%{http_code}' -H "Content-Type: application/json" -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY:-none}" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID:-none}\\\"}) }\"}")
    echo "R0_TERMINATE_GQL attempt=$i http=$code2 body=$(head -c 160 /tmp/term2.out 2>/dev/null)"
    sleep 5
  done
}
( sleep "$HARD"; echo R0_HARD_TIMER_FIRED; terminate ) &
echo "R0_START mode=$MODE pod=${RUNPOD_POD_ID:-unknown} hard_s=$HARD"
echo "R0_ENV_NAMES $(env | grep -o '^RUNPOD_[A-Z_]*' | sort | tr '\n' ' ')"
echo "R0_TOOLS runpodctl=$(command -v runpodctl || echo none) curl=$(command -v curl || echo none) python=$(python3 --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2
if [ "$MODE" = selftest-hang ]; then echo "R0_HANG_TEST: no completion marker; relying on the hard timer"; sleep 100000; exit 0; fi
if [ "$MODE" = selftest ]; then
  sleep "${R0_SELFTEST_S:-20}"; echo R0_SELFTEST_DONE; sleep "${R0_GRACE_S:-30}"; terminate; sleep 900; exit 0
fi
mkdir -p /workspace/out
echo "$R0_SCRIPT_B64" | base64 -d > /workspace/r0_bench.py
R0_OUT=/workspace/out/r0.json timeout "${R0_BENCH_TIMEOUT_S:-2400}" python3 -u /workspace/r0_bench.py 2>&1 | tee /workspace/out/r0.log
echo "R0_BENCH_EXIT ${PIPESTATUS[0]}"
echo R0_DONE
sleep "$GRACE"
terminate
sleep 900
