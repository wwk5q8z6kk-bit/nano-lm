#!/usr/bin/env python3
"""Host-side RunPod controller for Hopper R0-v2 (`r0_bench_hopper.py`) with 3-layer auto-teardown.

Enforces:
  1. Preflight check: 0 existing pods, verified account balance, worst-case price x time <= 80% of cap ($4.00 default).
  2. In-pod independent hard timer (`R0_HARD_S=1800`) + in-pod normal exit self-termination (`runpodctl` + REST + GraphQL).
  3. Host `finally:` block that deletes the pod via `runpodctl pod delete` and verifies `pods_after == []` + exact balance delta.
"""
from __future__ import annotations
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

POD_START_SH = r"""#!/bin/bash
set -u
START=$(date +%s)
HARD=${R0_HARD_S:-1800}
GRACE=${R0_GRACE_S:-90}
terminate() {
  echo "R0_TERMINATE_CALLED elapsed=$(( $(date +%s) - START ))s"
  for i in 1 2 3 4 5 6; do
    if command -v runpodctl >/dev/null 2>&1; then
      runpodctl pod delete "$RUNPOD_POD_ID" 2>&1 | head -c 200
      runpodctl remove pod "$RUNPOD_POD_ID" 2>&1 | head -c 200
    fi
    curl -s -m 20 -o /tmp/term.out -w '%{http_code}' -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY:-none}" "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID:-none}" >/dev/null 2>&1
    curl -s -m 20 -o /tmp/term2.out -w '%{http_code}' -H "Content-Type: application/json" -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY:-none}" -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID:-none}\\\"}) }\"}" >/dev/null 2>&1
    sleep 5
  done
}
( sleep "$HARD"; echo R0_HARD_TIMER_FIRED; terminate ) &
echo "R0_START pod=${RUNPOD_POD_ID:-unknown} hard_s=$HARD"
[ -f /etc/profile ] && . /etc/profile >/dev/null 2>&1 || true
PY=""
for cand in python /venv/main/bin/python /opt/venv/bin/python /opt/conda/bin/python /usr/local/bin/python /usr/local/bin/python3 /usr/bin/python3 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import torch" >/dev/null 2>&1; then
    PY=$(command -v "$cand"); break
  fi
done
PY=${PY:-python3}
echo "R0_PYTHON=$PY version=$($PY -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2
mkdir -p /workspace/out
echo "$R0_SCRIPT_B64" | base64 -d > /workspace/r0_bench_hopper.py
R0_OUT=/workspace/out/r0.json timeout "${R0_BENCH_TIMEOUT_S:-1500}" "$PY" -u /workspace/r0_bench_hopper.py 2>&1 | tee /workspace/out/r0.log
echo "R0_BENCH_EXIT ${PIPESTATUS[0]}"
echo R0_DONE
sleep "$GRACE"
terminate
sleep 600
"""


def run_cmd(args: list[str], timeout: int = 90) -> tuple[int, str, str]:
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def get_balance() -> float | None:
    rc, out, _ = run_cmd(["runpodctl", "user"])
    return float(json.loads(out)["clientBalance"]) if rc == 0 else None


def get_pods() -> list[dict] | None:
    rc, out, _ = run_cmd(["runpodctl", "pod", "list"])
    return json.loads(out) if (rc == 0 and out.strip()) else None


def find_gpu(display_name: str, cloud: str) -> tuple[float | None, str | None]:
    rc, out, _ = run_cmd(["runpodctl", "gpu", "list"])
    if rc != 0:
        return None, None
    key = "communityPricePerHr" if cloud == "COMMUNITY" else "securePricePerHr"
    for g in json.loads(out):
        if g.get("displayName") == display_name and g.get("available"):
            return g.get(key), g.get("gpuId")
    return None, None


def delete_pod(pid: str, ev_fn) -> bool:
    for i in range(6):
        rc, _, _ = run_cmd(["runpodctl", "pod", "delete", pid])
        ev_fn(f"delete attempt {i + 1}: rc={rc}")
        time.sleep(3)
        cur = get_pods() or []
        if not any(p.get("id") == pid for p in cur):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="H100 SXM")
    ap.add_argument("--cloud", default="COMMUNITY", choices=("COMMUNITY", "SECURE"))
    ap.add_argument("--cap", type=float, default=4.0)
    ap.add_argument("--hard-s", type=int, default=1800)
    ap.add_argument("--image", default="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = out_dir / f"r0-hopper-{stamp}.pod.log"
    sum_path = out_dir / f"r0-hopper-{stamp}.summary.json"
    res_path = out_dir / "r0_hopper_result.json"

    cap_used = 0.8 * a.cap
    S: dict = {"cap_usd": a.cap, "gpu": a.gpu, "cloud": a.cloud, "image": a.image, "events": []}
    t_init = time.time()

    def ev(msg: str):
        el = round(time.time() - t_init, 1)
        S["events"].append([el, msg])
        print(f"[{el:7.1f}s] {msg}", flush=True)
        sum_path.write_text(json.dumps(S, indent=2))

    bal0 = get_balance()
    ph, gid = find_gpu(a.gpu, a.cloud)
    if (ph is None or gid is None) and a.gpu == "H100 SXM":
        # Fallback to H100 PCIe if H100 SXM community slot is out of stock
        ph, gid = find_gpu("H100 PCIe", a.cloud)
        if ph is not None:
            S["gpu"] = "H100 PCIe"
    existing = get_pods()
    S.update(balance_before=bal0, price_per_hour=ph, gpu_id=gid)
    if bal0 is None or ph is None or gid is None or existing is None:
        raise SystemExit(f"Preflight failed: bal={bal0}, price={ph}, gid={gid}, existing={existing}")
    if existing:
        raise SystemExit(f"Preflight failed: {len(existing)} pods already active; refusing to launch")

    worst_usd = ph * ((a.hard_s + 480) / 3600.0)
    S.update(hard_s=a.hard_s, worst_case_usd=round(worst_usd, 3))
    if worst_usd > cap_used:
        raise SystemExit(f"Worst-case ${worst_usd:.2f} exceeds 80% of cap (${cap_used:.2f})")

    env = {
        "R0_HARD_S": str(a.hard_s),
        "R0_GRACE_S": "90",
        "R0_BUDGET_S": "1350",
        "R0_BENCH_TIMEOUT_S": "1450",
        "R0_START_B64": base64.b64encode(POD_START_SH.encode()).decode(),
        "R0_SCRIPT_B64": base64.b64encode((HERE / "r0_bench_hopper.py").read_bytes()).decode(),
    }
    cmd = [
        "runpodctl", "pod", "create",
        "--image", a.image,
        "--gpu-id", gid,
        "--cloud-type", a.cloud,
        "--container-disk-in-gb", "30",
        "--name", "nano-r0-hopper-v2",
        "--ssh=false",
        "--env", json.dumps(env),
        "--docker-args", "bash -c 'echo \"$R0_START_B64\" | base64 -d > /tmp/s.sh; bash /tmp/s.sh'",
    ]

    pid = None
    t_create = time.time()
    try:
        ev(f"Creating pod ({S['gpu']} @ ${ph}/hr, worst_case=${worst_usd:.2f})")
        rc, out, err = run_cmd(cmd, 120)
        m = re.search(r'"id"\s*:\s*"([a-z0-9]+)"', out)
        if rc != 0 or not m:
            ev(f"Create failed rc={rc}: {(err or out)[:300]}")
            raise SystemExit(1)
        pid = m.group(1)
        S["pod_id"] = pid
        t_create = time.time()
        ev(f"Pod created: {pid}")

        seen = set()
        lines: list[str] = []
        running_since = None
        last_log_poll = 0.0

        while True:
            el = time.time() - t_create
            if (el / 3600.0) * ph >= cap_used:
                ev("Cost guard reached; terminating")
                break
            if el > a.hard_s + 480:
                ev("Controller deadline reached; terminating")
                break
            rc, out_p, _ = run_cmd(["runpodctl", "pod", "get", pid])
            st = ""
            try:
                st = json.loads(out_p).get("runtimeStatus", "")
            except Exception:
                pass
            if st == "running" and running_since is None:
                running_since = el
                S["seconds_to_running"] = round(el, 1)
                ev(f"Container running after {el:.0f}s")
            if running_since is not None and st in ("terminated", "stopped", "exited"):
                ev(f"Pod transitioned to {st}")
                break
            if st == "error":
                ev("Pod reported error")
                break
            # If waiting for image pull > 420s without running, abort cleanly
            if running_since is None and el > 420:
                ev("Pod stuck pending > 420s; aborting")
                break

            if time.time() - last_log_poll >= 12:
                last_log_poll = time.time()
                rc_l, out_l, _ = run_cmd(["runpodctl", "pod", "logs", pid], 90)
                for ln in out_l.splitlines():
                    try:
                        j = json.loads(ln)
                    except Exception:
                        continue
                    key = (j.get("ts"), j.get("line"))
                    if key not in seen and j.get("source") == "container":
                        seen.add(key)
                        lines.append(j.get("line", ""))
                txt = "\n".join(lines)
                log_path.write_text(txt)
                if "R0_RESULT_BEGIN" in txt and "R0_RESULT_END" in txt:
                    b_idx = txt.index("R0_RESULT_BEGIN") + len("R0_RESULT_BEGIN")
                    e_idx = txt.index("R0_RESULT_END")
                    raw_json = txt[b_idx:e_idx].strip()
                    parsed = json.loads(raw_json)
                    res_path.write_text(json.dumps(parsed, indent=2))
                    S["result_captured"] = True
                    ev(f"Captured R0_RESULT JSON cleanly to {res_path.name}")
                    break
                if "R0_DONE" in txt:
                    ev("R0_DONE seen without result block; terminating pod immediately")
                    break
                if "R0_HARD_TIMER_FIRED" in txt:
                    ev("In-pod hard timer fired")
                    break
            time.sleep(8)
    finally:
        if pid:
            ok = delete_pod(pid, ev)
            S["deleted_confirmed"] = ok
            ev(f"Pod deletion confirmed: {ok}")
        S["total_wall_s"] = round(time.time() - t_create, 1)
        S["estimated_usd"] = round((S["total_wall_s"] / 3600.0) * ph, 4)
        time.sleep(20)
        bal1 = get_balance()
        S["balance_after"] = bal1
        if bal0 is not None and bal1 is not None:
            S["charged_usd_from_balance"] = round(bal0 - bal1, 4)
        S["pods_after"] = get_pods()
        ev(f"Done: charged=${S.get('charged_usd_from_balance')}, est=${S['estimated_usd']}, pods_after={S['pods_after']}")


if __name__ == "__main__":
    main()
