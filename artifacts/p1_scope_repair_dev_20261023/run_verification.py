import json, os, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
(out / "verification").mkdir(exist_ok=True)
env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
pytest = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts="]
groups = [("nanoscribe", ["nanoscribe"]), ("nano", ["nano"]), ("learned", ["nano_ai/tests"]), ("benchmarks", ["benchmarks/"]), ("foundations", ["fabric", "trajectory", "wedge_v1"]), ("scripts", ["scripts/test_reconcile_p1_comparison.py", "scripts/test_run_h7_kaggle.py", "scripts/test_run_p1_pilot.py", "scripts/test_run_p1_independent_dev.py"])]
commands = [(name, pytest + paths) for name, paths in groups] + [("fabric-direct", [sys.executable, "fabric/test_fabric.py"]), ("trajectory-direct", [sys.executable, "trajectory/test_recompute_c3.py"]), ("benchmark-direct", [sys.executable, "nanoscribe/benchmark.py"])]
report = {"started_utc": datetime.now(timezone.utc).isoformat(), "commands": []}
for name, command in commands:
    start = time.perf_counter()
    log = out / "verification" / (name + ".log")
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text()
    passed = re.findall(r"(\d+) passed", text)
    skipped = re.findall(r"(\d+) skipped", text)
    record = {"command": command, "exit_code": result.returncode, "seconds": time.perf_counter() - start, "log": str(log.relative_to(out)), "passed": int(passed[-1]) if passed else 0, "skipped": int(skipped[-1]) if skipped else 0}
    report["commands"].append(record)
    report["total_passed"] = sum(x["passed"] for x in report["commands"])
    report["total_skipped"] = sum(x["skipped"] for x in report["commands"])
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(name, result.returncode, record["passed"], record["skipped"], flush=True)
report["completed_utc"] = datetime.now(timezone.utc).isoformat()
(out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
sys.exit(any(x["exit_code"] != 0 for x in report["commands"]))
