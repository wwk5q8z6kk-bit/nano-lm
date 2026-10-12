"""Build a deterministic, authenticated RunPod training bundle for Nano H4.

The packager is intentionally separate from training.  It validates the frozen
H4 data, H3 source pins, base checkpoint, and tokenizer before emitting a
content-addressed archive.  It deliberately does not open or include known
development.  The archive's first script trains both fixed seeds; only after
their report hashes are frozen may the separately transferred, digest-pinned
development pair be opened by the second script on the same runtime.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.training.state_span_data import canonical_json_bytes
from nano_ai.training.train_evidence_query_h4 import (
    H4_TRAINING_RECIPE_VERSION,
    TRAINING_SEEDS,
    _require_preserved_h3_sources,
    load_h4_training_bundle,
)

H4_RUNPOD_BUNDLE_SCHEMA_VERSION = "nano.h4-runpod-bundle.v1"
H4_READINESS_SCHEMA_VERSION = "nano.h4-readiness.v1"
H2_DEVELOPMENT_MANIFEST_SHA256 = (
    "47ee157ac037c0771100b8546c90da91dbd2006198700bb642f1561d2124c1a3"
)
H2_DEVELOPMENT_SHA256 = (
    "9c893d8e64110287b433d567e0e9abb42c611ecba33b40de192741324d37e290"
)
BUNDLE_FILENAME = "nano-h4-runpod-input.tar.gz"
READINESS_FILENAME = "READINESS.json"
ARCHIVE_ROOT = "nano-h4-runpod"

# Exact transitive runtime and focused-test inventory for H4.  This is an
# allowlist rather than a package-tree walk so benchmark, sealed-confirmation,
# caches, documentation, and unrelated project files can never enter the cloud
# bundle by accident.
_SOURCE_ALLOWLIST = (
    "nano_ai/__init__.py",
    "nano_ai/contract.py",
    "nano_ai/evaluation.py",
    "nano_ai/fixtures.py",
    "nano_ai/solver.py",
    "nano_ai/adapters/__init__.py",
    "nano_ai/adapters/anchor_checkpoint.py",
    "nano_ai/adapters/deterministic_v0.py",
    "nano_ai/adapters/evidence_query_pointer.py",
    "nano_ai/adapters/legacy_summary.py",
    "nano_ai/adapters/pointer_span.py",
    "nano_ai/adapters/state_checkpoint.py",
    "nano_ai/adapters/state_span.py",
    "nano_ai/training/__init__.py",
    "nano_ai/training/evaluate_evidence_query.py",
    "nano_ai/training/evaluate_evidence_query_h4.py",
    "nano_ai/training/evaluate_pointer.py",
    "nano_ai/training/evaluate_state_span.py",
    "nano_ai/training/evidence_query_inference.py",
    "nano_ai/training/evidence_query_model.py",
    "nano_ai/training/model.py",
    "nano_ai/training/package_evidence_query_h4.py",
    "nano_ai/training/pointer_data.py",
    "nano_ai/training/pointer_model.py",
    "nano_ai/training/state_span_data.py",
    "nano_ai/training/surface_transfer_data.py",
    "nano_ai/training/train_evidence_query.py",
    "nano_ai/training/train_evidence_query_h4.py",
    "nano_ai/training/train_pointer.py",
    "nano_ai/training/train_state_span.py",
    "nano_ai/tests/test_evaluate_evidence_query_h4.py",
    "nano_ai/tests/test_package_evidence_query_h4.py",
    "nano_ai/tests/test_surface_transfer_data.py",
    "nano_ai/tests/test_train_evidence_query_h4.py",
    "pyproject.toml",
    "requirements-h4-runpod.txt",
)


class H4PackagingError(RuntimeError):
    """Raised when an H4 bundle cannot be proven reproducible."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(path: Path, *, role: str) -> bytes:
    try:
        if not path.is_file() or path.is_symlink():
            raise OSError
        return path.read_bytes()
    except OSError as exc:
        raise H4PackagingError(f"{role} is unavailable or not a regular file") from exc


def _file_identity(path: Path, *, role: str) -> dict[str, Any]:
    payload = _read_regular_file(path, role=role)
    return {"bytes": len(payload), "sha256": _sha256(payload)}


def _require_digest(path: Path, expected: str, *, role: str) -> dict[str, Any]:
    identity = _file_identity(path, role=role)
    if identity["sha256"] != expected:
        raise H4PackagingError(f"{role} SHA-256 mismatch")
    return identity


def _source_members(repo_root: Path) -> dict[str, Path]:
    return {name: repo_root / name for name in _SOURCE_ALLOWLIST}


def _training_run_script(*, training_manifest_sha256: str) -> bytes:
    seed_a, seed_b = TRAINING_SEEDS
    script = f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

test ! -e results
test ! -e h2_development
test ! -e .h4-venv
test "$(sha256sum h4_data/manifest.json | awk '{{print $1}}')" = "{training_manifest_sha256}"
python -m venv --system-site-packages .h4-venv
export PATH="$PWD/.h4-venv/bin:$PATH"
python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-h4-runpod.txt
python -c 'import platform, torch, tokenizers; assert platform.python_version().startswith("3.12."); assert torch.__version__ == "2.8.0+cu128"; assert torch.version.cuda == "12.8"; assert tokenizers.__version__ == "0.22.2"; assert torch.cuda.is_available(); assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090"; assert torch.cuda.get_device_properties(0).total_memory >= 20 * 1024**3'
python -m pytest -q nano_ai/tests/test_surface_transfer_data.py nano_ai/tests/test_train_evidence_query_h4.py nano_ai/tests/test_evaluate_evidence_query_h4.py nano_ai/tests/test_package_evidence_query_h4.py

mkdir results
python -m nano_ai.training.train_evidence_query_h4 --data-dir h4_data --base-checkpoint checkpoints/anchors/nano_v01_scribe.pt --tokenizer sft/tokenizer.json --output-dir results/seed-{seed_a} --seed {seed_a} --device cuda 2>&1 | tee results/seed-{seed_a}.log
python -m nano_ai.training.train_evidence_query_h4 --data-dir h4_data --base-checkpoint checkpoints/anchors/nano_v01_scribe.pt --tokenizer sft/tokenizer.json --output-dir results/seed-{seed_b} --seed {seed_b} --device cuda 2>&1 | tee results/seed-{seed_b}.log

sha256sum results/seed-{seed_a}/training_report.json results/seed-{seed_b}/training_report.json > results/TRAINING_REPORT_SHA256SUMS
"""
    if training_manifest_sha256 not in script:
        raise AssertionError("training manifest pin is missing from H4 training script")
    return script.encode("utf-8")


def _evaluation_run_script(*, training_manifest_sha256: str) -> bytes:
    seed_a, seed_b = TRAINING_SEEDS
    script = f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

test -f results/TRAINING_REPORT_SHA256SUMS
test ! -e results/development_evaluation.json
test -x .h4-venv/bin/python
export PATH="$PWD/.h4-venv/bin:$PATH"
sha256sum -c results/TRAINING_REPORT_SHA256SUMS
test "$(sha256sum h2_development/manifest.json | awk '{{print $1}}')" = "{H2_DEVELOPMENT_MANIFEST_SHA256}"
test "$(sha256sum h2_development/dev.jsonl | awk '{{print $1}}')" = "{H2_DEVELOPMENT_SHA256}"

report_a_sha="$(sha256sum results/seed-{seed_a}/training_report.json | awk '{{print $1}}')"
report_b_sha="$(sha256sum results/seed-{seed_b}/training_report.json | awk '{{print $1}}')"
python -m nano_ai.training.evaluate_evidence_query_h4 --training-data-dir h4_data --training-manifest-sha256 {training_manifest_sha256} --development-data-dir h2_development --development-manifest-sha256 {H2_DEVELOPMENT_MANIFEST_SHA256} --tokenizer sft/tokenizer.json --training-report results/seed-{seed_a}/training_report.json "$report_a_sha" --training-report results/seed-{seed_b}/training_report.json "$report_b_sha" --output results/development_evaluation.json --device cuda --batch-size 32 2>&1 | tee results/development_evaluation.log

find results -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.tmp
mv SHA256SUMS.tmp results/SHA256SUMS
tar -czf nano-h4-results.tar.gz results
sha256sum nano-h4-results.tar.gz > nano-h4-results.tar.gz.sha256
"""
    return script.encode("utf-8")


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes, *, mode: int) -> None:
    info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{name}")
    info.size = len(payload)
    info.mode = mode
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    archive.addfile(info, io.BytesIO(payload))


def _archive_bytes(members: Mapping[str, bytes], generated: Mapping[str, bytes]) -> bytes:
    """Return a byte-for-byte deterministic gzip-compressed tar archive."""

    overlap = set(members).intersection(generated)
    if overlap:
        raise H4PackagingError(
            f"generated bundle members collide with inputs: {sorted(overlap)!r}"
        )
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name, payload in sorted({**members, **generated}.items()):
            _add_bytes(
                archive,
                name,
                payload,
                mode=(
                    0o755
                    if name in {"RUN_H4_TRAIN.sh", "RUN_H4_EVALUATE.sh"}
                    else 0o644
                ),
            )
    return buffer.getvalue()


def _verify_archive_bytes(
    archive_bytes: bytes,
    *,
    member_identity: Mapping[str, Mapping[str, Any]],
) -> None:
    """Authenticate every archived payload before publishing readiness."""

    expected_names = {
        f"{ARCHIVE_ROOT}/{name}" for name in member_identity
    } | {f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json"}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            entries = archive.getmembers()
            observed_names = [entry.name for entry in entries]
            if len(observed_names) != len(set(observed_names)):
                raise H4PackagingError("H4 archive contains duplicate members")
            if set(observed_names) != expected_names:
                raise H4PackagingError("H4 archive member inventory mismatch")
            for entry in entries:
                if not entry.isfile() or entry.name.startswith("/"):
                    raise H4PackagingError("H4 archive contains an unsafe member")
                relative = Path(entry.name).relative_to(ARCHIVE_ROOT).as_posix()
                if ".." in Path(relative).parts:
                    raise H4PackagingError("H4 archive contains an unsafe member")
                if relative == "BUNDLE_MANIFEST.json":
                    continue
                identity = member_identity[relative]
                payload_file = archive.extractfile(entry)
                if payload_file is None:
                    raise H4PackagingError("H4 archive member is unreadable")
                payload = payload_file.read()
                if len(payload) != identity["bytes"] or _sha256(payload) != identity["sha256"]:
                    raise H4PackagingError(
                        f"H4 archive member identity mismatch: {relative}"
                    )
    except (OSError, tarfile.TarError, ValueError) as exc:
        raise H4PackagingError("H4 archive could not be authenticated") from exc


def package_h4_runpod_bundle(
    *,
    repo_root: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
) -> Mapping[str, Any]:
    """Validate and package the exact H4 input without overwriting artifacts."""

    root = Path(repo_root).resolve()
    data = Path(data_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise H4PackagingError("H4 packaging output directory already exists")
    if not output.parent.is_dir():
        raise H4PackagingError("H4 packaging output parent must exist")

    preserved = _require_preserved_h3_sources()
    bundle = load_h4_training_bundle(data)
    base_path = root / "checkpoints" / "anchors" / "nano_v01_scribe.pt"
    tokenizer_path = root / "sft" / "tokenizer.json"
    base = _require_digest(
        base_path, FROZEN_NANO_V01.checkpoint_sha256, role="frozen base checkpoint"
    )
    tokenizer = _require_digest(
        tokenizer_path, FROZEN_NANO_V01.tokenizer_sha256, role="frozen tokenizer"
    )
    paths = _source_members(root)
    paths.update(
        {
            "checkpoints/anchors/nano_v01_scribe.pt": base_path,
            "sft/tokenizer.json": tokenizer_path,
            "h4_data/manifest.json": data / "manifest.json",
            "h4_data/fit.jsonl": data / "fit.jsonl",
            "h4_data/calibration.jsonl": data / "calibration.jsonl",
        }
    )
    member_payloads = {
        name: _read_regular_file(path, role=f"bundle member {name}")
        for name, path in sorted(paths.items())
    }
    member_identity = {
        name: {"bytes": len(payload), "sha256": _sha256(payload)}
        for name, payload in sorted(member_payloads.items())
    }
    training_script = _training_run_script(
        training_manifest_sha256=bundle.manifest_sha256
    )
    evaluation_script = _evaluation_run_script(
        training_manifest_sha256=bundle.manifest_sha256
    )
    for name, payload in (
        ("RUN_H4_TRAIN.sh", training_script),
        ("RUN_H4_EVALUATE.sh", evaluation_script),
    ):
        member_identity[name] = {
            "bytes": len(payload),
            "sha256": _sha256(payload),
        }
    bundle_manifest = {
        "schema_version": H4_RUNPOD_BUNDLE_SCHEMA_VERSION,
        "recipe": H4_TRAINING_RECIPE_VERSION,
        "archive_root": ARCHIVE_ROOT,
        "training_seeds": list(TRAINING_SEEDS),
        "training_manifest_sha256": bundle.manifest_sha256,
        "development_manifest_sha256": H2_DEVELOPMENT_MANIFEST_SHA256,
        "expected_runtime": {
            "python": "3.12.x",
            "torch": "2.8.0+cu128",
            "tokenizers": "0.22.2",
            "cublas_workspace_config": ":4096:8",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "minimum_gpu_memory_gib": 20,
            "dependency_install": "requirements-h4-runpod.txt",
            "isolated_environment": ".h4-venv with system-site-packages",
        },
        "members": member_identity,
    }
    bundle_manifest_bytes = canonical_json_bytes(bundle_manifest)
    archive = _archive_bytes(
        member_payloads,
        {
            "BUNDLE_MANIFEST.json": bundle_manifest_bytes,
            "RUN_H4_EVALUATE.sh": evaluation_script,
            "RUN_H4_TRAIN.sh": training_script,
        },
    )
    _verify_archive_bytes(archive, member_identity=member_identity)
    readiness = {
        "schema_version": H4_READINESS_SCHEMA_VERSION,
        "status": "READY_FOR_TWO_SEED_RUN",
        "recipe": H4_TRAINING_RECIPE_VERSION,
        "archive": {
            "filename": BUNDLE_FILENAME,
            "bytes": len(archive),
            "sha256": _sha256(archive),
            "root": ARCHIVE_ROOT,
        },
        "inputs": {
            "training_manifest": {
                "bytes": len(_read_regular_file(data / "manifest.json", role="H4 manifest")),
                "sha256": bundle.manifest_sha256,
            },
            "fit_sha256": bundle.input_sha256["fit"],
            "calibration_sha256": bundle.input_sha256["calibration"],
            "base_checkpoint": base,
            "tokenizer": tokenizer,
            "development_manifest": {
                "sha256": H2_DEVELOPMENT_MANIFEST_SHA256,
                "included_in_training_bundle": False,
            },
            "development": {
                "sha256": H2_DEVELOPMENT_SHA256,
                "included_in_training_bundle": False,
            },
        },
        "preserved_h3_source_sha256": preserved,
        "training_seeds": list(TRAINING_SEEDS),
        "training_command": "tar -xzf nano-h4-runpod-input.tar.gz && cd nano-h4-runpod && ./RUN_H4_TRAIN.sh",
        "evaluation_command_after_development_transfer": "cd nano-h4-runpod && ./RUN_H4_EVALUATE.sh",
        "network_credentials_embedded": False,
        "fresh_v1_included": False,
        "known_development_included": False,
        "development_transfer_allowed_only_after_training_report_freeze": True,
        "quality_precedes_latency": True,
        "quality_precedes_sealed_confirmation": True,
    }

    try:
        output.mkdir(parents=False, exist_ok=False)
        archive_path = output / BUNDLE_FILENAME
        report_path = output / READINESS_FILENAME
        with archive_path.open("xb") as handle:
            handle.write(archive)
            handle.flush()
            os.fsync(handle.fileno())
        with report_path.open("xb") as handle:
            handle.write(canonical_json_bytes(readiness))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise H4PackagingError("H4 packaging outputs could not be created") from exc
    return readiness


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Package Nano H4 for RunPod")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    package_h4_runpod_bundle(
        repo_root=args.repo_root,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_FILENAME",
    "H4_READINESS_SCHEMA_VERSION",
    "H4_RUNPOD_BUNDLE_SCHEMA_VERSION",
    "READINESS_FILENAME",
    "H4PackagingError",
    "package_h4_runpod_bundle",
]
