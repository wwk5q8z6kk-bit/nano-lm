"""Disposable CPU/SRT diagnostic, not a training launcher or security certification.

Usage: python scripts/probe_execution_isolation.py --srt /path/to/srt --output /tmp/new-run
Requires optional torch. No datasets, checkpoints, secrets, or network downloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import shlex
import socket
import subprocess
import sys
import time


def control_probes(srt, root):
    """Startup and bounded process-group cancellation, independently of torch."""
    root.mkdir(parents=True, exist_ok=False)
    settings = root / "settings.json"
    settings.write_text(json.dumps({"network": {"allowedDomains": [], "deniedDomains": []},
        "filesystem": {"denyRead": [], "allowWrite": [str(root)], "denyWrite": []}}))
    env = {"PATH": os.defpath + ":/opt/homebrew/bin", "HOME": str(root), "TMPDIR": str(root)}
    result = {"startup": [], "cancellation": []}
    for mode in ("baseline", "srt"):
        def wrap(command):
            return ([srt, "-s", str(settings), "-c", shlex.join(command)]
                    if mode == "srt" else command)
        for repeat in range(5):
            start = time.perf_counter()
            p = subprocess.run(wrap(["/usr/bin/true"]), env=env, capture_output=True, timeout=10)
            result["startup"].append({"mode": mode, "repeat": repeat,
                "exit_code": p.returncode, "seconds": time.perf_counter() - start,
                "stderr": p.stderr.decode()})
        started = root / (mode + "-started")
        late = root / (mode + "-late")
        child = f"import time,pathlib; time.sleep(2); pathlib.Path({str(late)!r}).touch()"
        code = (f"import pathlib,subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); "
                f"pathlib.Path({str(started)!r}).touch(); time.sleep(20)")
        p = subprocess.Popen(wrap([sys.executable, "-c", code]), env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + 10
        while not started.exists() and p.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        time.sleep(2.5)
        result["cancellation"].append({"mode": mode, "started": started.exists(),
            "exit_code": p.returncode, "descendant_late_write": late.exists()})
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def worker(root, out, port):
    result = {"started": True}
    (out / "started").touch()
    for label, path, mode in [("secret_read", root / "secret", "r"),
                              ("protected_write", root / "protected", "w")]:
        try:
            with path.open(mode) as handle:
                handle.read() if mode == "r" else handle.write("changed")
            result[label] = "allowed"
        except PermissionError:
            result[label] = "denied"
    child = subprocess.run([sys.executable, "-c",
        "import pathlib,sys; pathlib.Path(sys.argv[1]).read_bytes()", str(root / "secret")],
        capture_output=True)
    result["child_secret_read"] = "allowed" if child.returncode == 0 else "failed"
    result["child_stderr"] = child.stderr.decode()
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            result["loopback_connect"] = "allowed"
    except OSError as error:
        result["loopback_connect"] = type(error).__name__
    payload = b"synthetic artifact\n" * 65536
    started = time.perf_counter()
    (out / "checkpoint.bin").write_bytes(payload)
    result["artifact_sha256"] = hashlib.sha256((out / "checkpoint.bin").read_bytes()).hexdigest()
    result["artifact_write_read_seconds"] = time.perf_counter() - started
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.set_num_threads(1)
        torch.manual_seed(123)
        torch.use_deterministic_algorithms(True)
        model = torch.nn.Linear(8, 2)
        x = torch.arange(256, dtype=torch.float32).reshape(32, 8) / 256
        dataset = TensorDataset(x, x[:, :2])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        started = time.perf_counter()
        steps = 0
        for inputs, target in DataLoader(dataset, batch_size=4, num_workers=2,
                                        multiprocessing_context="spawn"):
            optimizer.zero_grad()
            loss = ((model(inputs) - target) ** 2).mean()
            loss.backward()
            optimizer.step()
            steps += 1
        torch.save(model.state_dict(), out / "model.pt")
        restored = torch.load(out / "model.pt", weights_only=True)
        result["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "steps": steps,
            "seconds": time.perf_counter() - started,
            "checkpoint_equal": all(torch.equal(v, restored[k]) for k, v in model.state_dict().items()),
            "state_sha256": hashlib.sha256(b"".join(v.detach().numpy().tobytes()
                for v in model.state_dict().values())).hexdigest()}
    except Exception as error:
        result["torch_error"] = repr(error)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result["worker_cpu_seconds"] = usage.ru_utime + usage.ru_stime
    result["worker_max_rss_bytes"] = usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--srt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output, args.port)
        return
    if not args.srt or not args.output:
        parser.error("--srt and --output required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "secret").write_text("SYNTHETIC_ONLY")
    (root / "protected").write_text("synthetic protected input")
    results = {"platform": platform.platform(), "python": sys.version,
               "srt_version": subprocess.check_output([args.srt, "--version"], text=True).strip(),
               "runs": [], "gpu_test": "not_run"}
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(16)
        port = server.getsockname()[1]
        for repeat in range(3):
            for mode in ("baseline", "srt"):
                out = root / f"{mode}-{repeat}"
                out.mkdir()
                settings = root / f"settings-{repeat}.json"
                settings.write_text(json.dumps({"network": {"allowedDomains": [], "deniedDomains": []},
                    "filesystem": {"denyRead": [str(root / "secret")],
                                   "allowWrite": [str(out)], "denyWrite": [str(root / "protected")]}}))
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(root),
                           "--output", str(out), "--port", str(port)]
                if mode == "srt":
                    command = [args.srt, "-s", str(settings), "-c",
                               shlex.join(["env", "TMPDIR=" + str(out)] + command)]
                # Do not pass the operator's credential environment into the job.
                env = {"PATH": os.defpath + ":/opt/homebrew/bin", "HOME": str(out),
                       "TMPDIR": str(out), "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
                started = time.perf_counter()
                try:
                    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True, start_new_session=True)
                    stdout, stderr = proc.communicate(timeout=60)
                    row = {"mode": mode, "repeat": repeat, "exit_code": proc.returncode,
                           "stdout": stdout, "stderr": stderr}
                except subprocess.TimeoutExpired as error:
                    os.killpg(proc.pid, signal.SIGKILL)
                    stdout, stderr = proc.communicate()
                    row = {"mode": mode, "repeat": repeat, "timeout": True,
                           "stdout": stdout, "stderr": stderr}
                row["wall_seconds"] = time.perf_counter() - started
                row["worker_started"] = (out / "started").exists()
                results["runs"].append(row)
                (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(mode, repeat, row.get("exit_code", "timeout"), row["wall_seconds"], flush=True)


if __name__ == "__main__":
    main()
