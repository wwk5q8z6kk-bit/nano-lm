#!/usr/bin/env python3
"""Operational wrapper for the already-frozen H7 Kaggle packages.

The operator first runs ``reserve`` on the canonical local authority directory,
then embeds that reservation and this exact script in one private Kaggle kernel.
``run`` accepts an expanded, immutable stage dataset, never regenerates a package,
and never releases a reservation. An interrupted/unknown attempt requires review;
changing output directories or kernel names does not authorize another attempt.

Dataset privacy is an operator attestation checked against the submitted kernel
metadata, not something the remote filesystem can establish. The local reservation
is the cross-kernel authority; remote markers also prevent reuse in one runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import uuid


EXPERIMENT = "nano-p1-h7-boundary-margin-screen-v1"
AUTHORITY = Path.home() / ".codex/nano-evaluation-authority" / EXPERIMENT
GIT_SHA = "d6ed212c32384c1d50089e50260918028af88bfd"
MAX_SECONDS = 4 * 60 * 60
PACKAGES = {
    "training": {
        "root": "nano-h7-training", "script": "RUN_H7_TRAIN.sh",
        "archive_sha256": "1098761fd287dea1d106a32383308032d47c1061d3525877d57888bc7a15cb3f",
        "manifest_sha256": "ed5781f7728b4f7401b5a06f15b9760efc2584e0d977600966f1eb62395050dd",
        "result": "nano-h7-treatment-results.tar.gz",
    },
    "evaluation": {
        "root": "nano-h7-evaluation", "script": "RUN_H7_EVALUATE.sh",
        "archive_sha256": "c2c7fbfe49aea183db11c2c2c15ee9705b57d38ed72d5cde5a847c76cc9d16ec",
        "manifest_sha256": "fcee23dcfa4e173bd110e1717c9c5cba554335f2b5eafbb3049ff5018f1a3b7c",
        "result": "nano-h7-screen-results.tar.gz",
    },
}


class H7OperationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise H7OperationError(message)


def digest(path: Path) -> str:
    require(path.is_file() and not path.is_symlink(), f"Not a regular file: {path}")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_new(path: Path, payload: dict) -> None:
    """Create-only, durable markers: a failure must never erase attempt evidence."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and str(path) == name and "\\" not in name


def regular_tree(root: Path) -> set[str]:
    require(root.is_dir() and not root.is_symlink(), f"Not a regular directory: {root}")
    files = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"Symlink forbidden: {path}")
        require(path.is_dir() or path.is_file(), f"Special file forbidden: {path}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def validate_bundle(root: Path, stage: str) -> dict:
    expected = PACKAGES[stage]
    manifest_path = root / "BUNDLE_MANIFEST.json"
    require(digest(manifest_path) == expected["manifest_sha256"], "Frozen manifest hash mismatch")
    manifest = json.loads(manifest_path.read_bytes())
    for key, value in {
        "schema_version": "nano.h7-kaggle-package.v1", "stage": stage,
        "archive_root": expected["root"], "git_sha": GIT_SHA,
        "recipe": "nano-evidence-query-h7-boundary-margin-v1",
        "authorized_cost_usd": 0, "max_runtime_hours": 4,
        "training_seeds": [20260805, 20260806], "venue": "Kaggle free CPU",
    }.items():
        require(manifest.get(key) == value, f"Frozen manifest field mismatch: {key}")
    members = manifest["members"]
    require(all(safe_name(name) for name in members), "Unsafe bundle member")
    require(regular_tree(root) == set(members) | {"BUNDLE_MANIFEST.json"}, "Bundle member inventory mismatch")
    for name, identity in members.items():
        path = root / name
        require(path.stat().st_size == identity["bytes"] and digest(path) == identity["sha256"], f"Bundle member identity mismatch: {name}")
    return manifest


def validate_metadata(metadata: dict, stage: str, stage_dataset: str, treatment_dataset: str | None) -> None:
    require(metadata.get("is_private") is True, "Kernel must be private")
    require(metadata.get("enable_gpu") is False and metadata.get("enable_tpu") is False, "Only the frozen free CPU venue is authorized")
    expected = [stage_dataset]
    if stage == "evaluation":
        require(bool(treatment_dataset) and treatment_dataset != stage_dataset, "Evaluation needs a separate private treatment dataset")
        expected.append(treatment_dataset)
    else:
        require(treatment_dataset is None, "Training cannot mount treatment or evaluation data")
    require(sorted(metadata.get("dataset_sources", [])) == sorted(expected), "Kernel inputs must be exactly the stage-only private datasets")
    for key in ("kernel_sources", "competition_sources", "model_sources"):
        require(not metadata.get(key), f"Unexpected kernel input: {key}")


def reserve(args: argparse.Namespace) -> Path:
    metadata = json.loads(args.kernel_metadata.read_bytes())
    validate_metadata(metadata, args.stage, args.stage_dataset, args.treatment_dataset)
    expected_private = {args.stage_dataset} | ({args.treatment_dataset} if args.treatment_dataset else set())
    require(set(args.verified_private_dataset) == expected_private, "Operator must attest verified privacy of every input dataset")
    require((args.stage == "evaluation") == bool(args.treatment_archive_sha256), "Only evaluation requires a frozen treatment archive hash")
    if args.treatment_archive_sha256:
        require(bool(re.fullmatch("[0-9a-f]{64}", args.treatment_archive_sha256)), "Invalid treatment archive SHA-256")
    remaining, training_terminal = MAX_SECONDS, None
    if args.stage == "evaluation":
        require(args.training_terminal is not None, "Evaluation requires the preserved successful training terminal record")
        training_terminal = json.loads(args.training_terminal.read_bytes())
        training_reservation = json.loads((args.registry / "training.reservation.json").read_bytes())
        require(training_terminal.get("attempt_id") == training_reservation["attempt_id"], "Training terminal does not match the global reservation")
        require(training_terminal.get("status") == "completed" and training_terminal.get("exit_code") == 0 and training_terminal.get("error") is None, "Failed or unknown training cannot admit evaluation")
        elapsed = training_terminal.get("elapsed_seconds")
        require(isinstance(elapsed, (int, float)) and 0 < elapsed < MAX_SECONDS, "Invalid or exhausted cumulative runtime")
        remaining = int(MAX_SECONDS - elapsed)
        require(remaining > 0, "Four-hour cumulative runtime exhausted")
    else:
        require(args.training_terminal is None, "Training cannot reuse a prior terminal record")
    args.registry.mkdir(parents=True, exist_ok=True)
    path = args.registry / f"{args.stage}.reservation.json"
    reservation = {
        "schema": "nano.h7-operational-reservation.v1", "experiment": EXPERIMENT,
        "stage": args.stage, "attempt_id": str(uuid.uuid4()), "reserved_at_utc": utc_now(),
        "canonical_authority": str(args.registry.resolve()), "package": PACKAGES[args.stage],
        "runner_sha256": digest(Path(__file__)), "kernel_metadata": metadata,
        "stage_dataset": args.stage_dataset, "treatment_dataset": args.treatment_dataset,
        "verified_private_datasets": sorted(expected_private),
        "treatment_archive_sha256": args.treatment_archive_sha256,
        "max_runtime_seconds": remaining, "campaign_runtime_cap_seconds": MAX_SECONDS,
        "training_terminal": training_terminal, "authorization_cost_usd": 0,
        "retry_policy": "Never automatically retry or release; terminal or unknown attempts require review.",
    }
    write_new(path, reservation)
    return path


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def validate_reservation(reservation: dict) -> str:
    require(reservation.get("schema") == "nano.h7-operational-reservation.v1" and reservation.get("experiment") == EXPERIMENT, "Wrong experiment reservation")
    stage = reservation.get("stage")
    require(stage in PACKAGES and reservation.get("package") == PACKAGES[stage], "Wrong frozen package reservation")
    require(reservation.get("runner_sha256") == digest(Path(__file__)), "Runner differs from the locally reserved script")
    require(reservation.get("campaign_runtime_cap_seconds") == MAX_SECONDS and reservation.get("authorization_cost_usd") == 0, "Wrong runtime/cost reservation")
    prior = reservation.get("training_terminal")
    if stage == "training":
        require(prior is None and reservation.get("max_runtime_seconds") == MAX_SECONDS, "Wrong training runtime reservation")
    else:
        require(isinstance(prior, dict) and prior.get("status") == "completed" and prior.get("exit_code") == 0 and prior.get("error") is None, "Evaluation requires completed training")
        elapsed = prior.get("elapsed_seconds")
        require(isinstance(elapsed, (int, float)) and 0 < elapsed < MAX_SECONDS, "Invalid prior training runtime")
        require(reservation.get("max_runtime_seconds") == int(MAX_SECONDS - elapsed) > 0, "Evaluation exceeds remaining cumulative runtime")
    validate_metadata(reservation["kernel_metadata"], stage, reservation["stage_dataset"], reservation["treatment_dataset"])
    require(set(reservation["verified_private_datasets"]) == set(reservation["kernel_metadata"]["dataset_sources"]), "Dataset privacy attestation missing")
    require((stage == "evaluation") == bool(reservation.get("treatment_archive_sha256")), "Treatment artifact reservation mismatch")
    return stage


def derive_wrapper(canonical: str, stage: str) -> str:
    suffix = "train" if stage == "training" else "eval"
    old = f'python -m venv --system-site-packages .h7-{suffix}-venv\nexport PATH="$PWD/.h7-{suffix}-venv/bin:$PATH"\npython -m pip install --disable-pip-version-check --no-cache-dir -r requirements-h7-kaggle.txt'
    new = f'test ! -e .h7-{suffix}-deps\npython -m pip install --disable-pip-version-check --no-cache-dir --target .h7-{suffix}-deps -r requirements-h7-kaggle.txt\nexport PYTHONPATH="$PWD/.h7-{suffix}-deps"\npython "$H7_RUNTIME_PROBE"'
    require(canonical.count(old) == 1, "Expected frozen dependency bootstrap was not found exactly once")
    derived = canonical.replace(old, new)
    if stage == "evaluation":
        old_checksum = "sha256sum -c treatments/results/TRAINING_REPORT_SHA256SUMS"
        require(derived.count(old_checksum) == 1, "Expected frozen report checksum command was not found exactly once")
        derived = derived.replace(old_checksum, "(cd treatments && sha256sum -c results/TRAINING_REPORT_SHA256SUMS)")
    # These create-only operational markers precede the original, unchanged
    # scientific command lines. A started marker never implies completion.
    prefix = "python -m nano_ai.training."
    commands = [line for line in derived.splitlines() if line.startswith(prefix)]
    require(len(commands) == (2 if stage == "training" else 1), "Unexpected scientific command inventory")
    for index, command in enumerate(commands):
        event = f"training_seed_{20260805 + index}" if stage == "training" else "screen_evaluation"
        derived = derived.replace(command, f'python "$H7_EVENT_MARKER" {event}\n{command}', 1)
    return derived


RUNTIME_PROBE = '''import json, os, platform
from pathlib import Path
import torch, tokenizers, pytest
runtime = {"python": platform.python_version(), "torch": torch.__version__,
           "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
           "tokenizers": tokenizers.__version__, "pytest": pytest.__version__}
Path(os.environ["H7_RUNTIME_JSON"]).write_text(json.dumps(runtime, sort_keys=True, indent=2) + "\\n")
print("H7_RUNTIME", json.dumps(runtime, sort_keys=True), flush=True)
assert runtime["python"] == "3.13.15", runtime
assert runtime["torch"] == "2.11.0+cpu" and runtime["torch_cuda"] is None and not runtime["cuda_available"], runtime
assert runtime["tokenizers"] == "0.22.2" and runtime["pytest"] == "9.0.2", runtime
'''

EVENT_MARKER = '''import json, os, sys, time
from pathlib import Path
path = Path(os.environ["H7_EVIDENCE_DIR"]) / (sys.argv[1] + ".started.json")
with path.open("x") as handle:
    json.dump({"event": sys.argv[1], "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, handle)
    handle.write("\\n")
    handle.flush()
    os.fsync(handle.fileno())
print("H7_SCIENTIFIC_COMMAND_START", sys.argv[1], flush=True)
'''


def extract_treatments(archive: Path, expected_sha: str, target: Path) -> None:
    require(digest(archive) == expected_sha, "Treatment archive hash mismatch")
    target.mkdir()
    with tarfile.open(archive, "r:gz") as source:
        names = set()
        for member in source.getmembers():
            name = member.name.rstrip("/")
            require(safe_name(name) and PurePosixPath(name).parts[0] == "results", "Unsafe treatment archive member")
            require(name not in names, "Duplicate treatment archive member")
            names.add(name)
            require(member.isfile() or member.isdir(), "Treatment archive links/special files forbidden")
            dest = target / name
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as original, dest.open("xb") as copy:
                    shutil.copyfileobj(original, copy)
    verify_checksum_file(target, target / "results/SHA256SUMS", complete=True)
    report_names = verify_checksum_file(target, target / "results/TRAINING_REPORT_SHA256SUMS")
    require(report_names == {f"results/seed-{seed}/training_report.json" for seed in (20260805, 20260806)}, "Exactly the two frozen training reports are required")


def verify_checksum_file(root: Path, checksum_file: Path, *, complete: bool = False) -> set[str]:
    require(checksum_file.is_file() and not checksum_file.is_symlink(), "Missing regular checksum file")
    names = set()
    for line in checksum_file.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "Malformed checksum entry")
        sha, name = match.groups()
        require(safe_name(name) and PurePosixPath(name).parts[0] == "results" and name not in names, "Unsafe/duplicate checksum path")
        require(digest(root / name) == sha, f"Treatment checksum mismatch: {name}")
        names.add(name)
    if complete:
        require(names == regular_tree(root) - {checksum_file.relative_to(root).as_posix()}, "Treatment checksum inventory mismatch")
    return names


def validate_mounts(input_root: Path, source: Path, treatments: Path | None) -> None:
    """Inventory filenames only; training cannot silently inherit another mount."""
    source = source.resolve()
    allowed_treatment = treatments.resolve() if treatments else None
    require(source.is_relative_to(input_root.resolve()), "Stage dataset is outside the input mount")
    for path in input_root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            continue
        require(not path.is_symlink() and path.is_file(), "Unexpected link/special file in input mounts")
        require(path.resolve().is_relative_to(source) or path.resolve() == allowed_treatment, f"Unexpected mounted input: {path}")


def run_process(command: list[str], cwd: Path, env: dict, log: Path, remaining: float) -> int:
    require(remaining > 0, "Four-hour stage budget expired before launch")
    with log.open("ab", buffering=0) as handle:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        def tee() -> None:
            for line in iter(process.stdout.readline, b""):
                handle.write(line)
                print(line.decode("utf-8", errors="replace"), end="", flush=True)
        pump = threading.Thread(target=tee, daemon=True)
        pump.start()
        try:
            return process.wait(timeout=remaining)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            pump.join(timeout=15)
            handle.flush()
            os.fsync(handle.fileno())


def preserve_evidence(output: Path, bundle: Path, stage: str) -> None:
    files = [path for path in (output / "evidence").rglob("*") if path.is_file() and not path.is_symlink()]
    results = bundle / "results"
    if results.exists():
        regular_tree(results)
        files.extend(path for path in results.rglob("*") if path.is_file())
    for name in (PACKAGES[stage]["result"], PACKAGES[stage]["result"] + ".sha256"):
        path = bundle / name
        if path.is_file() and not path.is_symlink():
            shutil.copyfile(path, output / name)
            files.append(output / name)
    sums = output / "ATTEMPT_SHA256SUMS"
    with sums.open("x") as handle:
        for path in sorted(files):
            handle.write(f"{digest(path)}  {path.relative_to(output).as_posix()}\n")
    archive_path = output / f"nano-h7-{stage}-attempt-evidence.tar.gz"
    with tarfile.open(archive_path, "x:gz") as archive:
        for path in sorted(files + [sums]):
            archive.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    (output / (archive_path.name + ".sha256")).write_text(f"{digest(archive_path)}  {archive_path.name}\n")


def run(args: argparse.Namespace) -> int:
    reservation = json.loads(args.reservation.read_bytes())
    stage = validate_reservation(reservation)
    require((stage == "evaluation") == bool(args.treatment_archive), "Stage/treatment input mismatch")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    args.marker_root.mkdir(parents=True, exist_ok=True)
    marker = args.marker_root / f"{EXPERIMENT}.{stage}.started.json"
    started = {"attempt_id": reservation["attempt_id"], "stage": stage, "started_at_utc": utc_now(), "reservation_sha256": digest(args.reservation)}
    write_new(marker, started)
    write_new(evidence / "started.json", started)
    write_new(evidence / "reservation.json", reservation)
    bundle = output / PACKAGES[stage]["root"]
    start = time.monotonic()
    phase, exit_code, error = "input_validation", 1, None
    def interrupted(signum: int, _frame: object) -> None:
        raise H7OperationError(f"Interrupted by signal {signum}; no retry authorized")
    previous_signals = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        validate_mounts(args.input_root, args.source_dir, args.treatment_archive)
        validate_bundle(args.source_dir, stage)
        shutil.copytree(args.source_dir, bundle)
        validate_bundle(bundle, stage)
        write_new(evidence / "package-validation.json", {"manifest_sha256": PACKAGES[stage]["manifest_sha256"], "full_tree_verified": True, "frozen_archive_sha256_provenance": PACKAGES[stage]["archive_sha256"], "archive_verification": "Expanded dataset authenticated by pinned manifest and every member; compressed archive is not mounted."})
        if args.treatment_archive:
            extract_treatments(args.treatment_archive, reservation["treatment_archive_sha256"], bundle / "treatments")
        canonical = bundle / PACKAGES[stage]["script"]
        wrapper = bundle / "RUN_H7_KAGGLE_OPERATIONAL.sh"
        wrapper.write_text(derive_wrapper(canonical.read_text(), stage))
        shutil.copyfile(wrapper, evidence / wrapper.name)
        probe = evidence / "runtime_probe.py"
        probe.write_text(RUNTIME_PROBE)
        event_marker = evidence / "event_marker.py"
        event_marker.write_text(EVENT_MARKER)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(H7_RUNTIME_PROBE=str(probe), H7_RUNTIME_JSON=str(evidence / "runtime.json"), H7_EVENT_MARKER=str(event_marker), H7_EVIDENCE_DIR=str(evidence), PYTHONUNBUFFERED="1")
        phase = "packaged_stage_execution"
        exit_code = run_process(["bash", wrapper.name], bundle, env, evidence / "runner.log", reservation["max_runtime_seconds"] - (time.monotonic() - start))
        require(exit_code == 0, f"Packaged {stage} exited with status {exit_code}; no retry authorized")
        require((bundle / PACKAGES[stage]["result"]).is_file(), "Stage exited without its required result archive")
        phase = "completed"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        if exit_code == 0:
            exit_code = 1
        print(error, file=sys.stderr, flush=True)
    finally:
        terminal = {**started, "finished_at_utc": utc_now(), "elapsed_seconds": round(time.monotonic() - start, 3), "phase": phase, "exit_code": exit_code, "error": error, "status": "completed" if exit_code == 0 else "failed_or_interrupted", "retry_authorized": False}
        write_new(evidence / "terminal.json", terminal)
        write_new(args.marker_root / f"{EXPERIMENT}.{stage}.terminal.json", terminal)
        preserve_evidence(output, bundle, stage)
        for sig, previous in previous_signals.items():
            signal.signal(sig, previous)
    return exit_code if 0 <= exit_code <= 255 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reservation = sub.add_parser("reserve", help="Locally reserve the sole stage attempt before provider submission")
    reservation.add_argument("--stage", choices=PACKAGES, required=True)
    reservation.add_argument("--registry", type=Path, default=AUTHORITY)
    reservation.add_argument("--kernel-metadata", type=Path, required=True)
    reservation.add_argument("--stage-dataset", required=True)
    reservation.add_argument("--treatment-dataset")
    reservation.add_argument("--treatment-archive-sha256")
    reservation.add_argument("--training-terminal", type=Path)
    reservation.add_argument("--verified-private-dataset", action="append", required=True)
    execution = sub.add_parser("run", help="Execute one reserved stage in a fresh Kaggle CPU kernel")
    execution.add_argument("--reservation", type=Path, required=True)
    execution.add_argument("--source-dir", type=Path, required=True)
    execution.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    execution.add_argument("--output-dir", type=Path, required=True)
    execution.add_argument("--marker-root", type=Path, default=Path("/kaggle/working"))
    execution.add_argument("--treatment-archive", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "reserve":
            print(reserve(args))
            return 0
        return run(args)
    except (H7OperationError, OSError, ValueError) as exc:
        print(f"H7_OPERATION_BLOCKED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
