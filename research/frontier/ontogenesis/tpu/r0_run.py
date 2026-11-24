#!/usr/bin/env python3
"""Laptop-side controller for the R0 measurement (owner-approved 2026-11-24: one run, hard cap US$6 total including compute,
storage and transfer). Run it under `caffeinate -i -s` so the laptop does not sleep.

Shutdown layers (any one is enough): in-pod hard timer; in-pod normal end after a grace period; this controller's cost guard,
deadline and `finally` block. Billing facts (RunPod docs, read 2026-11-24): per-second billing for compute and storage, no fee for
data in or out; the account has a prepaid balance (larger than the cap, so the provider alone does not enforce US$6) and
auto-stops pods only at zero balance. The cap is therefore enforced here by price x time, with the account balance change as the
ground truth for what was actually charged. The controller never prints account details.

usage: r0_run.py --mode selftest|r0 [--cloud COMMUNITY|SECURE]
"""
import argparse, base64, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=("selftest", "selftest-hang", "r0"), required=True)
ap.add_argument("--selfterm", action="store_true", help="test mode: do NOT delete from the laptop; wait for the pod to terminate itself (controller deadline stays as backup)")
ap.add_argument("--hard-s", type=int, default=0, help="test modes: in-pod hard timer in seconds")
ap.add_argument("--cloud", default="COMMUNITY"); ap.add_argument("--cap", type=float, default=6.0)
ap.add_argument("--gpu", default=None, help="GPU display name from `runpodctl gpu list` (defaults: RTX 4090 for selftest, H100 SXM for r0)")
ap.add_argument("--image", default="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404")
ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "r0"))
a = ap.parse_args()
IS_TEST = a.mode.startswith("selftest")
GPU_NAME = a.gpu or ("RTX 4090" if IS_TEST else "H100 SXM")
CAP_USED = 0.8 * a.cap                      # plan so that the worst case stays at 80% of the cap
os.makedirs(a.out, exist_ok=True); STAMP = time.strftime("%Y%m%d-%H%M%S"); LOGF = os.path.join(a.out, f"{a.mode}-{STAMP}.pod.log"); SUMF = os.path.join(a.out, f"{a.mode}-{STAMP}.summary.json")
S = {"mode": a.mode, "cap_usd": a.cap, "gpu": GPU_NAME, "cloud": a.cloud, "image": a.image, "events": []}


def ev(msg):
    S["events"].append([round(time.time() - T0, 1) if "T0" in globals() else 0, msg]); print(f"[{S['events'][-1][0]:7.1f}s] {msg}", flush=True)
    json.dump(S, open(SUMF, "w"), indent=1)


def run(args, timeout=60):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout); return r.returncode, r.stdout, r.stderr


def balance():
    rc, out, _ = run(["runpodctl", "user"]); return float(json.loads(out)["clientBalance"]) if rc == 0 else None


def pods():
    rc, out, _ = run(["runpodctl", "pod", "list"]); return json.loads(out) if rc == 0 and out.strip() else None


def price():
    rc, out, _ = run(["runpodctl", "gpu", "list"])
    for g in json.loads(out):
        if g.get("displayName") == GPU_NAME: return g.get("communityPricePerHr" if a.cloud == "COMMUNITY" else "securePricePerHr"), g.get("gpuId")
    return None, None


def delete(pid):
    for i in range(5):
        rc, out, err = run(["runpodctl", "pod", "delete", pid]); ev(f"delete attempt {i + 1}: rc={rc}")
        time.sleep(3)
        if not any(p.get("id") == pid for p in (pods() or [])): return True
    return False


# ---- preflight ----
bal0 = balance(); ph, gid = price(); existing = pods()
S.update(balance_before=bal0, price_per_hour=ph, gpu_id_listed=gid)
if bal0 is None or ph is None or existing is None: sys.exit("preflight failed: cannot read balance, price or pod list")
if existing: sys.exit(f"preflight failed: {len(existing)} pods already exist; refusing to start")
hard_s = a.hard_s if (IS_TEST and a.hard_s) else (300 if IS_TEST else min(55, int(CAP_USED / ph * 60 - 12)) * 60)      # in-pod lifetime
hard_min = hard_s / 60; startup_allow_min = 4 if IS_TEST else 12; worst = ph * (hard_min + startup_allow_min) / 60
S.update(hard_min=hard_min, startup_allowance_min=startup_allow_min, worst_case_usd=round(worst, 3))
print(json.dumps({k: S[k] for k in ("mode", "gpu", "cloud", "price_per_hour", "hard_min", "worst_case_usd", "cap_usd")}))
if worst > CAP_USED: sys.exit(f"preflight failed: worst case ${worst:.2f} exceeds 80% of the cap (${CAP_USED:.2f}); not launching")
if bal0 < worst + 1: sys.exit("preflight failed: balance too low")

env = {"R0_MODE": a.mode, "R0_HARD_S": str(hard_s), "R0_GRACE_S": "120" if a.mode == "r0" else "20", "R0_BUDGET_S": "1500", "R0_BENCH_TIMEOUT_S": str(max(600, (hard_min - 8) * 60 - 120))}
env["R0_START_B64"] = base64.b64encode(open(os.path.join(HERE, "r0_pod_start.sh"), "rb").read()).decode()
if a.mode == "r0": env["R0_SCRIPT_B64"] = base64.b64encode(open(os.path.join(HERE, "r0_bench.py"), "rb").read()).decode()
cmd = ["runpodctl", "pod", "create", "--image", a.image, "--gpu-id", gid, "--cloud-type", a.cloud, "--container-disk-in-gb", "30", "--name", f"nano-r0-{a.mode}", "--ssh=false",
       "--env", json.dumps(env), "--docker-args", "bash -c 'echo \"$R0_START_B64\" | base64 -d > /tmp/s.sh; bash /tmp/s.sh'"]
T0 = time.time(); pid = None
try:
    ev("creating pod")
    rc, out, err = run(cmd, 120)
    m = re.search(r'"id"\s*:\s*"([a-z0-9]+)"', out)
    if rc != 0 or not m: ev(f"create failed rc={rc}: {(err or out)[:300].replace(chr(10), ' ')}"); sys.exit(1)
    pid = m.group(1); S["pod_id"] = pid; T_CREATE = time.time(); ev(f"pod created {pid}")
    seen = set(); lines = []; done = False; result_ok = False; last_logs = 0; running_since = None
    while True:
        el = time.time() - T_CREATE; cost_now = el / 3600 * ph
        if cost_now >= CAP_USED: ev(f"cost guard: estimated ${cost_now:.2f} reached 80% of the cap"); break
        if el > (hard_min + startup_allow_min) * 60: ev("controller deadline reached"); break
        rc, out, _ = run(["runpodctl", "pod", "get", pid]); st = ""
        try: st = json.loads(out).get("runtimeStatus", "")
        except Exception: pass
        if st == "running" and running_since is None: running_since = el; S["seconds_to_running"] = round(el, 1); ev(f"container running after {el:.0f}s")
        if st in ("terminated", "stopped") and running_since is not None: ev(f"pod status {st}"); break
        if st == "error": ev("pod status error"); break
        if a.selfterm and running_since is not None and not any(p.get("id") == pid for p in (pods() or [])): ev("pod removed itself (in-pod shutdown worked)"); S["self_terminated"] = True; break
        if time.time() - last_logs > 15:
            last_logs = time.time(); rc, out, _ = run(["runpodctl", "pod", "logs", pid], 90)
            for ln in out.splitlines():
                try: j = json.loads(ln)
                except Exception: continue
                key = (j.get("ts"), j.get("line"))
                if key not in seen and j.get("source") == "container": seen.add(key); lines.append(j.get("line", ""))
            open(LOGF, "w").write("\n".join(lines))
            txt = "\n".join(lines)
            if ("R0_DONE" in txt and "R0_RESULT_END" in txt) or "R0_SELFTEST_DONE" in txt or "R0_HARD_TIMER_FIRED" in txt:
                if not done: done = True; ev("completion marker seen in logs")
                if not a.selfterm: break
        time.sleep(10)
finally:
    if pid:
        ok = delete(pid); S["deleted_confirmed"] = ok; ev(f"deletion confirmed: {ok}")
        if not ok: print("WARNING: pod could not be confirmed deleted; delete it manually now", flush=True)
    S["controller_seconds_create_to_delete"] = round(time.time() - T0, 1)
    time.sleep(45); bal1 = balance(); S["balance_after"] = bal1
    if bal0 is not None and bal1 is not None: S["charged_usd_from_balance"] = round(bal0 - bal1, 4)
    S["estimated_usd_price_x_time"] = round(S["controller_seconds_create_to_delete"] / 3600 * ph, 4); S["pods_after"] = pods()
    ev(f"summary: charged ${S.get('charged_usd_from_balance')} (balance change), estimated ${S['estimated_usd_price_x_time']} (price x time), pods left: {S['pods_after']}")
