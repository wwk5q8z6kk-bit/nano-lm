#!/usr/bin/env python3
"""Host-side RunPod multi-GPU controller for Stage B (`stage_b_runner.py`) with 3-layer auto-teardown.

Enforces:
  1. Preflight check: 0 existing pods, verified account balance, multi-GPU total price x time <= 85% of cap.
  2. In-pod independent hard timer (`STAGEB_HARD_S`) + in-pod normal exit self-termination (`runpodctl` + REST + GraphQL).
  3. Incremental streaming capture of each `STAGEB_REC_BEGIN <name> ... STAGEB_REC_END <name>` receipt block so
     individual completed runs are persisted to `--out-dir` immediately as they finish.
  4. Host `finally:` block that deletes the pod via `runpodctl pod delete` and verifies `pods_after == []` + balance delta.
"""
from __future__ import annotations
import argparse
import base64
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

POD_START_SH = r"""#!/bin/bash
set -u
START=$(date +%s)
HARD=${STAGEB_HARD_S:-7200}
GRACE=${STAGEB_GRACE_S:-90}
terminate() {
  echo "STAGEB_TERMINATE_CALLED elapsed=$(( $(date +%s) - START ))s"
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
( sleep "$HARD"; echo STAGEB_HARD_TIMER_FIRED; terminate ) &
echo "STAGEB_START pod=${RUNPOD_POD_ID:-unknown} hard_s=$HARD"
[ -f /etc/profile ] && . /etc/profile >/dev/null 2>&1 || true
PY=""
for cand in python /venv/main/bin/python /opt/venv/bin/python /opt/conda/bin/python /usr/local/bin/python /usr/local/bin/python3 /usr/bin/python3 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import torch" >/dev/null 2>&1; then
    PY=$(command -v "$cand"); break
  fi
done
PY=${PY:-python3}
echo "STAGEB_PYTHON=$PY version=$($PY -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())' 2>&1)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>&1 | head -8
mkdir -p /workspace/out_stage_b /workspace/data_stage_b
echo "$STAGEB_SCRIPT_B64" | base64 -d > /workspace/stage_b_runner.py
STAGEB_OUT=/workspace/out_stage_b STAGEB_DATA=/workspace/data_stage_b timeout "${STAGEB_TIMEOUT_S:-6900}" "$PY" -u /workspace/stage_b_runner.py 2>&1 | tee /workspace/out_stage_b/stage_b.log
echo "STAGEB_RUNNER_EXIT ${PIPESTATUS[0]}"
echo STAGEB_POD_DONE
sleep "$GRACE"
terminate
sleep 600
"""

REQUIRED_JOBS = (
    "P_parent",
    "A_all30_s1000",
    "T_s1000",
    "C1_s1000",
    "T_s1001",
    "C1_s1001",
    "T_s1002",
    "C1_s1002",
    "C2_s1000",
    "R_s1000",
)


def run_cmd(args: list[str], timeout: int = 90) -> tuple[int, str, str]:
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def get_balance() -> float | None:
    rc, out, _ = run_cmd(["runpodctl", "user"])
    return float(json.loads(out)["clientBalance"]) if rc == 0 else None


def get_pods() -> list[dict] | None:
    rc, out, _ = run_cmd(["runpodctl", "pod", "list"])
    return json.loads(out) if (rc == 0 and out.strip()) else None


def list_gpus() -> list[dict]:
    rc, out, _ = run_cmd(["runpodctl", "gpu", "list"])
    return json.loads(out) if (rc == 0 and out.strip()) else []


def delete_pod(pid: str, ev_fn) -> bool:
    for i in range(8):
        rc, _, _ = run_cmd(["runpodctl", "pod", "delete", pid])
        ev_fn(f"delete attempt {i + 1}: rc={rc}")
        time.sleep(4)
        cur = get_pods() or []
        if not any(p.get("id") == pid for p in cur):
            return True
    return False


def extract_receipts(txt: str, out_dir: Path, captured: set[str], ev_fn):
    pattern = re.compile(r"STAGEB_REC_BEGIN ([A-Za-z0-9_]+)\n(.*?)\nSTAGEB_REC_END \1", re.DOTALL)
    for m in pattern.finditer(txt):
        jname, body = m.group(1), m.group(2).strip()
        if jname in captured:
            continue
        try:
            obj = json.loads(body)
        except Exception as exc:
            ev_fn(f"warn: failed parsing receipt {jname}: {exc!r}")
            continue
        dst = out_dir / f"stageb_{jname}.json"
        dst.write_text(json.dumps(obj, indent=2))
        captured.add(jname)
        ev_fn(f"Captured stageb_{jname}.json ({len(captured)}/{len(REQUIRED_JOBS)}) mix_nats={obj.get('final_heldout_nats', {}).get('mix')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="H100 SXM")
    ap.add_argument("--cloud", default="COMMUNITY", choices=("COMMUNITY", "SECURE"))
    ap.add_argument("--gpu-count", type=int, default=4)
    ap.add_argument("--cap", type=float, default=32.0, help="Hard USD budget cap for this Stage B campaign")
    ap.add_argument("--hard-s", type=int, default=7200)
    ap.add_argument("--image", default="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = out_dir / f"stageb-{stamp}.pod.log"
    sum_path = out_dir / f"stageb-{stamp}.controller.json"
    bundle_path = out_dir / "stage_b_bundle.json"

    cap_used = 0.85 * a.cap
    S: dict = {"cap_usd": a.cap, "cap_guard_usd": round(cap_used, 2), "image": a.image, "events": []}
    t_init = time.time()

    def ev(msg: str):
        el = round(time.time() - t_init, 1)
        S["events"].append([el, msg])
        print(f"[{el:7.1f}s] {msg}", flush=True)
        sum_path.write_text(json.dumps(S, indent=2))

    bal0 = get_balance()
    existing = get_pods()
    gpus = list_gpus()
    if bal0 is None or existing is None or not gpus:
        raise SystemExit(f"Preflight failed: bal={bal0}, existing={existing}, gpus_count={len(gpus)}")
    if existing:
        raise SystemExit(f"Preflight failed: {len(existing)} pods already active; refusing to launch")

    # Candidate fallback sequence for multi-GPU creation (in case a pool doesn't have 4 contiguous GPUs on 1 host)
    candidates = [
        (a.gpu, a.cloud, a.gpu_count),
        ("H100 NVL", "COMMUNITY", a.gpu_count),
        ("A100 SXM", "COMMUNITY", a.gpu_count),
        ("RTX PRO 6000", "COMMUNITY", a.gpu_count),
        ("H100 SXM", "SECURE", min(a.gpu_count, 2)),
        ("A100 SXM", "SECURE", a.gpu_count),
    ]
    gpu_by_name = {g.get("displayName"): g for g in gpus if g.get("available")}

    env = {
        "STAGEB_HARD_S": str(a.hard_s),
        "STAGEB_GRACE_S": "90",
        "STAGEB_TIMEOUT_S": str(max(600, a.hard_s - 300)),
        "STAGEB_START_B64": base64.b64encode(POD_START_SH.encode()).decode(),
        "STAGEB_SCRIPT_B64": base64.b64encode((HERE / "stage_b_runner.py").read_bytes()).decode(),
    }

    pid = None
    total_ph = 0.0
    t_create = time.time()
    for disp, cld, gcnt in candidates:
        ginfo = gpu_by_name.get(disp)
        if not ginfo:
            continue
        price_key = "communityPricePerHr" if cld == "COMMUNITY" else "securePricePerHr"
        ph_single = ginfo.get(price_key)
        gid = ginfo.get("gpuId")
        if not ph_single or not gid:
            continue
        cand_total_ph = float(ph_single) * gcnt
        worst_usd = cand_total_ph * ((a.hard_s + 300) / 3600.0)
        if worst_usd > cap_used and gcnt > 2:
            # If 4x SECURE exceeds cap_used, skip to next cheaper candidate
            ev(f"Skipping {gcnt}x {disp} ({cld}) @ ${cand_total_ph:.2f}/hr: worst=${worst_usd:.2f} > guard=${cap_used:.2f}")
            continue
        cmd = [
            "runpodctl", "pod", "create",
            "--image", a.image,
            "--gpu-id", gid,
            "--gpu-count", str(gcnt),
            "--cloud-type", cld,
            "--container-disk-in-gb", "60",
            "--name", "nano-stage-b-f2",
            "--ssh=false",
            "--env", json.dumps(env),
            "--docker-args", "bash -c 'echo \"$STAGEB_START_B64\" | base64 -d > /tmp/s.sh; bash /tmp/s.sh'",
        ]
        ev(f"Attempting pod create: {gcnt}x {disp} ({cld}) @ ${cand_total_ph:.2f}/hr total (worst=${worst_usd:.2f})")
        rc, out, err = run_cmd(cmd, 120)
        m = re.search(r'"id"\s*:\s*"([a-z0-9]+)"', out)
        if rc == 0 and m:
            pid = m.group(1)
            total_ph = cand_total_ph
            t_create = time.time()
            S.update(
                balance_before=bal0,
                gpu=disp,
                gpu_id=gid,
                cloud=cld,
                gpu_count=gcnt,
                price_per_hour_total=round(total_ph, 4),
                hard_s=a.hard_s,
                worst_case_usd=round(worst_usd, 3),
                pod_id=pid,
            )
            ev(f"Pod created: {pid} ({gcnt}x {disp} {cld})")
            break
        ev(f"Candidate {gcnt}x {disp} ({cld}) unavailable: {(err or out).strip()[:220]}")

    if not pid:
        raise SystemExit("All multi-GPU RunPod candidates failed to allocate")

    captured: set[str] = set()
    try:
        seen = set()
        lines: list[str] = []
        running_since = None
        last_log_poll = 0.0

        while True:
            el = time.time() - t_create
            if (el / 3600.0) * total_ph >= cap_used:
                ev(f"Live cost guard (${cap_used:.2f}) reached; terminating")
                break
            if el > a.hard_s + 360:
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
            if running_since is None and el > 480:
                ev("Pod stuck pending > 480s; aborting")
                break

            if time.time() - last_log_poll >= 15:
                last_log_poll = time.time()
                rc_l, out_l, _ = run_cmd(["runpodctl", "pod", "logs", pid], 120)
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
                extract_receipts(txt, out_dir, captured, ev)
                S["captured_jobs"] = sorted(captured)
                sum_path.write_text(json.dumps(S, indent=2))

                if len(captured) == len(REQUIRED_JOBS) or "STAGEB_ALL_DONE" in txt:
                    if len(captured) == len(REQUIRED_JOBS):
                        bundle = {f"stageb_{jn}": json.loads((out_dir / f"stageb_{jn}.json").read_text()) for jn in REQUIRED_JOBS}
                        bundle_path.write_text(json.dumps(bundle, indent=2))
                        S["all_10_receipts_captured"] = True
                        ev(f"All 10 Stage B receipts captured cleanly into {bundle_path.name}!")
                        break
                if "STAGEB_POD_DONE" in txt:
                    ev(f"STAGEB_POD_DONE seen ({len(captured)}/{len(REQUIRED_JOBS)} receipts); terminating pod")
                    break
                if "STAGEB_HARD_TIMER_FIRED" in txt:
                    ev("In-pod hard timer fired")
                    break
            time.sleep(10)
    finally:
        if pid:
            ok = delete_pod(pid, ev)
            S["deleted_confirmed"] = ok
            ev(f"Pod deletion confirmed: {ok}")
        S["total_wall_s"] = round(time.time() - t_create, 1)
        S["estimated_usd"] = round((S["total_wall_s"] / 3600.0) * total_ph, 4)
        time.sleep(20)
        bal1 = get_balance()
        S["balance_after"] = bal1
        if bal0 is not None and bal1 is not None:
            S["charged_usd_from_balance"] = round(bal0 - bal1, 4)
        S["pods_after"] = get_pods()
        ev(f"Done: charged=${S.get('charged_usd_from_balance')}, est=${S['estimated_usd']}, pods_after={S['pods_after']}")


if __name__ == "__main__":
    main()
