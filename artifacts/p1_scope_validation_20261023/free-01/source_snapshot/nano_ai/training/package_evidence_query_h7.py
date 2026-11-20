"""Build isolated, deterministic Kaggle archives for Nano H7.

The training archive contains the frozen H5 fit/calibration inputs and no H7
screen material.  The evaluation archive contains the frozen H7 screen and H6
control, but no training data or base checkpoint.  Treatment reports and their
selected checkpoints must be transferred between the two stages only after
both seeds finish and their report hashes are frozen.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import re
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nano_ai.adapters.anchor_checkpoint import FROZEN_NANO_V01
from nano_ai.training.evaluate_evidence_query_h7 import (
    CONTROL_CHECKPOINT_BYTES,
    CONTROL_CHECKPOINT_SHA256,
    H7_SCREEN_MANIFEST_SHA256,
    H7_SCREEN_SHA256,
)
from nano_ai.training.replay_mixture_data import load_replay_mixture_dataset
from nano_ai.training.state_span_data import canonical_json_bytes
from nano_ai.training.train_evidence_query_h7 import (
    H5_DATASET_FILE_IDENTITY,
    H7_TRAINING_RECIPE_VERSION,
    TRAINING_SEEDS,
    _require_frozen_h5_dataset,
    _require_preserved_h3_sources,
)

H7_KAGGLE_PACKAGE_SCHEMA_VERSION = "nano.h7-kaggle-package.v1"
H7_READINESS_SCHEMA_VERSION = "nano.h7-kaggle-readiness.v1"
TRAINING_ARCHIVE_FILENAME = "nano-h7-kaggle-training.tar.gz"
EVALUATION_ARCHIVE_FILENAME = "nano-h7-kaggle-evaluation.tar.gz"
READINESS_FILENAME = "READINESS.json"
TRAINING_ARCHIVE_ROOT = "nano-h7-training"
EVALUATION_ARCHIVE_ROOT = "nano-h7-evaluation"
TREATMENT_RESULTS_FILENAME = "nano-h7-treatment-results.tar.gz"
FINAL_RESULTS_FILENAME = "nano-h7-screen-results.tar.gz"
MAX_RUNTIME_HOURS = 4
AUTHORIZED_COST_USD = 0

_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_REQUIREMENTS = b"pytest==9.0.2\ntokenizers==0.22.2\n"

# Exact runtime closure shared by both stages. Evaluation-only authorities are
# deliberately separate so the training archive cannot generate or open H7.
_COMMON_SOURCE_ALLOWLIST = (
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
    "nano_ai/training/boundary_margin.py",
    "nano_ai/training/evaluate_evidence_query.py",
    "nano_ai/training/evaluate_pointer.py",
    "nano_ai/training/evaluate_state_span.py",
    "nano_ai/training/evidence_query_inference.py",
    "nano_ai/training/evidence_query_model.py",
    "nano_ai/training/model.py",
    "nano_ai/training/pointer_data.py",
    "nano_ai/training/pointer_model.py",
    "nano_ai/training/replay_mixture_data.py",
    "nano_ai/training/state_conditioned_evidence_query_model.py",
    "nano_ai/training/state_span_data.py",
    "nano_ai/training/surface_transfer_data.py",
    "nano_ai/training/train_evidence_query.py",
    "nano_ai/training/train_evidence_query_h4.py",
    "nano_ai/training/train_evidence_query_h6.py",
    "nano_ai/training/train_evidence_query_h7.py",
    "nano_ai/training/train_pointer.py",
    "nano_ai/training/train_state_span.py",
)
_TRAINING_SOURCE_ALLOWLIST = _COMMON_SOURCE_ALLOWLIST + (
    "nano_ai/tests/test_boundary_margin.py",
    "nano_ai/tests/test_replay_mixture_data.py",
    "nano_ai/tests/test_state_conditioned_evidence_query_model.py",
    "nano_ai/tests/test_train_evidence_query_h7.py",
)
_EVALUATION_SOURCE_ALLOWLIST = _COMMON_SOURCE_ALLOWLIST + (
    "nano_ai/training/diagnose_h6.py",
    "nano_ai/training/evaluate_evidence_query_h7.py",
    "nano_ai/training/h7_boundary_screen_data.py",
    "nano_ai/tests/test_evaluate_evidence_query_h7.py",
)


class H7PackagingError(RuntimeError):
    """Raised when the isolated H7 execution package cannot be proven safe."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_file(path: Path, *, role: str) -> bytes:
    try:
        if not path.is_file() or path.is_symlink():
            raise OSError
        return path.read_bytes()
    except OSError as exc:
        raise H7PackagingError(f"{role} is unavailable or not a regular file") from exc


def _require_digest(
    path: Path, expected: str, *, role: str, expected_bytes: int | None = None
) -> dict[str, Any]:
    payload = _read_regular_file(path, role=role)
    identity = {"bytes": len(payload), "sha256": _sha256(payload)}
    if identity["sha256"] != expected or (
        expected_bytes is not None and identity["bytes"] != expected_bytes
    ):
        raise H7PackagingError(f"{role} identity mismatch")
    return identity


def _source_members(repo_root: Path, *, stage: str) -> dict[str, Path]:
    if stage == "training":
        allowlist = _TRAINING_SOURCE_ALLOWLIST
    elif stage == "evaluation":
        allowlist = _EVALUATION_SOURCE_ALLOWLIST
    else:
        raise H7PackagingError("H7 package stage must be training or evaluation")
    return {name: repo_root / name for name in allowlist}


def _termination_prelude(*, temporary_paths: Sequence[str]) -> str:
    quoted = " ".join(f"'{path}'" for path in temporary_paths)
    return f"""cleanup() {{
  rm -f -- {quoted}
}}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
"""


def _training_run_script(*, training_manifest_sha256: str) -> bytes:
    seed_a, seed_b = TRAINING_SEEDS
    prelude = _termination_prelude(
        temporary_paths=(
            "results/.TRAINING_REPORT_SHA256SUMS.tmp",
            f"{TREATMENT_RESULTS_FILENAME}.tmp",
            f"{TREATMENT_RESULTS_FILENAME}.sha256.tmp",
        )
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONHASHSEED=0
{prelude}
test ! -e results
test ! -e h7_screen
test ! -e control
test ! -e ../{EVALUATION_ARCHIVE_ROOT}
test ! -e .h7-train-venv
test "$(sha256sum h5_data/manifest.json | awk '{{print $1}}')" = "{training_manifest_sha256}"
test "$(sha256sum h5_data/fit.jsonl | awk '{{print $1}}')" = "{H5_DATASET_FILE_IDENTITY['fit']['sha256']}"
test "$(sha256sum h5_data/calibration.jsonl | awk '{{print $1}}')" = "{H5_DATASET_FILE_IDENTITY['calibration']['sha256']}"
python -m venv --system-site-packages .h7-train-venv
export PATH="$PWD/.h7-train-venv/bin:$PATH"
python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-h7-kaggle.txt
python -c 'import platform, torch, tokenizers; assert tuple(map(int, platform.python_version_tuple()[:2])) >= (3, 10); assert tokenizers.__version__ == "0.22.2"; assert torch.__version__; print("runtime", platform.python_version(), torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pytest -q nano_ai/tests/test_boundary_margin.py nano_ai/tests/test_state_conditioned_evidence_query_model.py nano_ai/tests/test_replay_mixture_data.py nano_ai/tests/test_train_evidence_query_h7.py

test ! -e h7_screen
mkdir results
python -m nano_ai.training.train_evidence_query_h7 --data-dir h5_data --base-checkpoint checkpoints/anchors/nano_v01_scribe.pt --tokenizer sft/tokenizer.json --output-dir results/seed-{seed_a} --seed {seed_a} --device cpu 2>&1 | tee results/seed-{seed_a}.log
test ! -e h7_screen
python -m nano_ai.training.train_evidence_query_h7 --data-dir h5_data --base-checkpoint checkpoints/anchors/nano_v01_scribe.pt --tokenizer sft/tokenizer.json --output-dir results/seed-{seed_b} --seed {seed_b} --device cpu 2>&1 | tee results/seed-{seed_b}.log
test ! -e h7_screen

sha256sum results/seed-{seed_a}/training_report.json results/seed-{seed_b}/training_report.json > results/.TRAINING_REPORT_SHA256SUMS.tmp
mv results/.TRAINING_REPORT_SHA256SUMS.tmp results/TRAINING_REPORT_SHA256SUMS
find results -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > results/SHA256SUMS
tar -czf {TREATMENT_RESULTS_FILENAME}.tmp results
mv {TREATMENT_RESULTS_FILENAME}.tmp {TREATMENT_RESULTS_FILENAME}
sha256sum {TREATMENT_RESULTS_FILENAME} > {TREATMENT_RESULTS_FILENAME}.sha256.tmp
mv {TREATMENT_RESULTS_FILENAME}.sha256.tmp {TREATMENT_RESULTS_FILENAME}.sha256
trap - EXIT INT TERM HUP
""".encode()


def _evaluation_run_script() -> bytes:
    seed_a, seed_b = TRAINING_SEEDS
    prelude = _termination_prelude(
        temporary_paths=(
            "results/SHA256SUMS.tmp",
            f"{FINAL_RESULTS_FILENAME}.tmp",
            f"{FINAL_RESULTS_FILENAME}.sha256.tmp",
        )
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONHASHSEED=0
{prelude}
test -d treatments/results
test -f treatments/results/TRAINING_REPORT_SHA256SUMS
test ! -L treatments/results/TRAINING_REPORT_SHA256SUMS
test ! -e results
test ! -e h5_data
test ! -e checkpoints/anchors/nano_v01_scribe.pt
test ! -e .h7-eval-venv
sha256sum -c treatments/results/TRAINING_REPORT_SHA256SUMS
test "$(sha256sum artifacts/nano_h7/screen-v1/manifest.json | awk '{{print $1}}')" = "{H7_SCREEN_MANIFEST_SHA256}"
test "$(sha256sum artifacts/nano_h7/screen-v1/screen.jsonl | awk '{{print $1}}')" = "{H7_SCREEN_SHA256}"
test "$(sha256sum control/h6-seed-20260805-epoch-2.pt | awk '{{print $1}}')" = "{CONTROL_CHECKPOINT_SHA256}"
python -m venv --system-site-packages .h7-eval-venv
export PATH="$PWD/.h7-eval-venv/bin:$PATH"
python -m pip install --disable-pip-version-check --no-cache-dir -r requirements-h7-kaggle.txt
python -c 'import platform, torch, tokenizers; assert tuple(map(int, platform.python_version_tuple()[:2])) >= (3, 10); assert tokenizers.__version__ == "0.22.2"; assert torch.__version__'
python -m pytest -q nano_ai/tests/test_evaluate_evidence_query_h7.py

report_a_sha="$(sha256sum treatments/results/seed-{seed_a}/training_report.json | awk '{{print $1}}')"
report_b_sha="$(sha256sum treatments/results/seed-{seed_b}/training_report.json | awk '{{print $1}}')"
mkdir results
python -m nano_ai.training.evaluate_evidence_query_h7 --screen-dir artifacts/nano_h7/screen-v1 --control-checkpoint control/h6-seed-20260805-epoch-2.pt --tokenizer sft/tokenizer.json --treatment-report treatments/results/seed-{seed_a}/training_report.json "$report_a_sha" --treatment-report treatments/results/seed-{seed_b}/training_report.json "$report_b_sha" --output results/h7_screen_evaluation.json --device cpu --batch-size 32 2>&1 | tee results/h7_screen_evaluation.log

find results -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > results/SHA256SUMS.tmp
mv results/SHA256SUMS.tmp results/SHA256SUMS
tar -czf {FINAL_RESULTS_FILENAME}.tmp results
mv {FINAL_RESULTS_FILENAME}.tmp {FINAL_RESULTS_FILENAME}
sha256sum {FINAL_RESULTS_FILENAME} > {FINAL_RESULTS_FILENAME}.sha256.tmp
mv {FINAL_RESULTS_FILENAME}.sha256.tmp {FINAL_RESULTS_FILENAME}.sha256
trap - EXIT INT TERM HUP
""".encode()


def _add_bytes(
    archive: tarfile.TarFile,
    *,
    root: str,
    name: str,
    payload: bytes,
    executable: bool,
) -> None:
    info = tarfile.TarInfo(f"{root}/{name}")
    info.size = len(payload)
    info.mode = 0o755 if executable else 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    archive.addfile(info, io.BytesIO(payload))


def _archive_bytes(
    *,
    root: str,
    members: Mapping[str, bytes],
    generated: Mapping[str, bytes],
) -> bytes:
    overlap = set(members).intersection(generated)
    if overlap:
        raise H7PackagingError(f"generated members collide: {sorted(overlap)!r}")
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name, payload in sorted({**members, **generated}.items()):
            _add_bytes(
                archive,
                root=root,
                name=name,
                payload=payload,
                executable=name.startswith("RUN_H7_") and name.endswith(".sh"),
            )
    return buffer.getvalue()


def _verify_archive_bytes(
    archive_bytes: bytes,
    *,
    root: str,
    member_identity: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {f"{root}/{name}" for name in member_identity} | {
        f"{root}/BUNDLE_MANIFEST.json"
    }
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            entries = archive.getmembers()
            names = [entry.name for entry in entries]
            if len(names) != len(set(names)) or set(names) != expected:
                raise H7PackagingError("H7 archive inventory mismatch")
            for entry in entries:
                if not entry.isfile() or entry.name.startswith("/"):
                    raise H7PackagingError("H7 archive contains an unsafe member")
                relative = Path(entry.name).relative_to(root).as_posix()
                if ".." in Path(relative).parts:
                    raise H7PackagingError("H7 archive contains an unsafe member")
                if relative == "BUNDLE_MANIFEST.json":
                    continue
                payload_file = archive.extractfile(entry)
                if payload_file is None:
                    raise H7PackagingError("H7 archive member is unreadable")
                payload = payload_file.read()
                identity = member_identity[relative]
                if (
                    len(payload) != identity["bytes"]
                    or _sha256(payload) != identity["sha256"]
                ):
                    raise H7PackagingError(
                        f"H7 archive member identity mismatch: {relative}"
                    )
    except (OSError, tarfile.TarError, ValueError) as exc:
        raise H7PackagingError("H7 archive could not be authenticated") from exc


def _payloads(paths: Mapping[str, Path]) -> dict[str, bytes]:
    return {
        name: _read_regular_file(path, role=f"bundle member {name}")
        for name, path in sorted(paths.items())
    }


def _identities(payloads: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(payload), "sha256": _sha256(payload)}
        for name, payload in sorted(payloads.items())
    }


def _build_stage_archive(
    *,
    stage: str,
    root: str,
    payloads: Mapping[str, bytes],
    run_script_name: str,
    run_script: bytes,
    git_sha: str,
) -> tuple[bytes, Mapping[str, Any]]:
    generated_payloads = {
        "requirements-h7-kaggle.txt": _REQUIREMENTS,
        run_script_name: run_script,
    }
    member_identity = _identities({**payloads, **generated_payloads})
    manifest = {
        "schema_version": H7_KAGGLE_PACKAGE_SCHEMA_VERSION,
        "stage": stage,
        "recipe": H7_TRAINING_RECIPE_VERSION,
        "git_sha": git_sha,
        "archive_root": root,
        "venue": "Kaggle free CPU",
        "authorized_cost_usd": AUTHORIZED_COST_USD,
        "max_runtime_hours": MAX_RUNTIME_HOURS,
        "training_seeds": list(TRAINING_SEEDS),
        "members": member_identity,
    }
    archive = _archive_bytes(
        root=root,
        members=payloads,
        generated={
            **generated_payloads,
            "BUNDLE_MANIFEST.json": canonical_json_bytes(manifest),
        },
    )
    _verify_archive_bytes(archive, root=root, member_identity=member_identity)
    return archive, manifest


def package_h7_kaggle_bundles(
    *,
    repo_root: str | Path,
    data_dir: str | Path,
    control_checkpoint: str | Path,
    output_dir: str | Path,
    git_sha: str,
) -> Mapping[str, Any]:
    """Build both isolated stages without overwriting any existing artifact."""

    if not isinstance(git_sha, str) or _GIT_SHA_PATTERN.fullmatch(git_sha) is None:
        raise H7PackagingError("git_sha must be a full lowercase commit SHA")
    root = Path(repo_root).resolve()
    data = Path(data_dir).resolve()
    control_path = Path(control_checkpoint).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise H7PackagingError("H7 packaging output directory already exists")
    if not output.parent.is_dir():
        raise H7PackagingError("H7 packaging output parent must exist")

    preserved = _require_preserved_h3_sources()
    try:
        frozen_data_hashes = _require_frozen_h5_dataset(data)
        bundle = load_replay_mixture_dataset(data)
    except Exception as exc:
        raise H7PackagingError("H7 input is not the byte-exact H5 dataset") from exc
    if bundle.input_sha256 != frozen_data_hashes:
        raise H7PackagingError("H7 packager resolved another dataset identity")

    base_path = root / "checkpoints/anchors/nano_v01_scribe.pt"
    tokenizer_path = root / "sft/tokenizer.json"
    screen_root = root / "artifacts/nano_h7/screen-v1"
    base = _require_digest(
        base_path,
        FROZEN_NANO_V01.checkpoint_sha256,
        role="frozen base checkpoint",
    )
    tokenizer = _require_digest(
        tokenizer_path,
        FROZEN_NANO_V01.tokenizer_sha256,
        role="frozen tokenizer",
    )
    control = _require_digest(
        control_path,
        CONTROL_CHECKPOINT_SHA256,
        role="frozen H6 control checkpoint",
        expected_bytes=CONTROL_CHECKPOINT_BYTES,
    )
    screen_manifest = _require_digest(
        screen_root / "manifest.json",
        H7_SCREEN_MANIFEST_SHA256,
        role="frozen H7 screen manifest",
    )
    screen = _require_digest(
        screen_root / "screen.jsonl",
        H7_SCREEN_SHA256,
        role="frozen H7 screen",
    )

    training_paths = _source_members(root, stage="training")
    training_paths.update(
        {
            "checkpoints/anchors/nano_v01_scribe.pt": base_path,
            "h5_data/calibration.jsonl": data / "calibration.jsonl",
            "h5_data/fit.jsonl": data / "fit.jsonl",
            "h5_data/manifest.json": data / "manifest.json",
            "sft/tokenizer.json": tokenizer_path,
        }
    )
    evaluation_paths = _source_members(root, stage="evaluation")
    evaluation_paths.update(
        {
            "control/h6-seed-20260805-epoch-2.pt": control_path,
            "artifacts/nano_h7/screen-v1/manifest.json": screen_root
            / "manifest.json",
            "artifacts/nano_h7/screen-v1/screen.jsonl": screen_root
            / "screen.jsonl",
            "sft/tokenizer.json": tokenizer_path,
        }
    )
    training_payloads = _payloads(training_paths)
    evaluation_payloads = _payloads(evaluation_paths)
    if any("nano_h7/screen-v1/" in name for name in training_payloads):
        raise H7PackagingError("H7 screen entered the training archive")
    if any(name.startswith("h5_data/") for name in evaluation_payloads):
        raise H7PackagingError("H5 training data entered the evaluation archive")

    training_archive, training_manifest = _build_stage_archive(
        stage="training",
        root=TRAINING_ARCHIVE_ROOT,
        payloads=training_payloads,
        run_script_name="RUN_H7_TRAIN.sh",
        run_script=_training_run_script(
            training_manifest_sha256=bundle.manifest_sha256
        ),
        git_sha=git_sha,
    )
    evaluation_archive, evaluation_manifest = _build_stage_archive(
        stage="evaluation",
        root=EVALUATION_ARCHIVE_ROOT,
        payloads=evaluation_payloads,
        run_script_name="RUN_H7_EVALUATE.sh",
        run_script=_evaluation_run_script(),
        git_sha=git_sha,
    )

    readiness = {
        "schema_version": H7_READINESS_SCHEMA_VERSION,
        "status": "READY_FOR_ISOLATED_KAGGLE_EXECUTION",
        "experiment_id": "nano-p1-h7-boundary-margin-screen-v1",
        "recipe": H7_TRAINING_RECIPE_VERSION,
        "git_sha": git_sha,
        "venue": "Kaggle free CPU",
        "authorized_cost_usd": AUTHORIZED_COST_USD,
        "max_runtime_hours": MAX_RUNTIME_HOURS,
        "artifact_destination": "local artifacts/nano_h7/kaggle/h7-boundary-margin-v1",
        "training_seeds": list(TRAINING_SEEDS),
        "archives": {
            "training": {
                "filename": TRAINING_ARCHIVE_FILENAME,
                "bytes": len(training_archive),
                "sha256": _sha256(training_archive),
                "root": TRAINING_ARCHIVE_ROOT,
                "manifest": training_manifest,
            },
            "evaluation": {
                "filename": EVALUATION_ARCHIVE_FILENAME,
                "bytes": len(evaluation_archive),
                "sha256": _sha256(evaluation_archive),
                "root": EVALUATION_ARCHIVE_ROOT,
                "manifest": evaluation_manifest,
            },
        },
        "inputs": {
            "training_manifest": H5_DATASET_FILE_IDENTITY["manifest"],
            "fit": H5_DATASET_FILE_IDENTITY["fit"],
            "calibration": H5_DATASET_FILE_IDENTITY["calibration"],
            "base_checkpoint": base,
            "tokenizer": tokenizer,
            "control_checkpoint": control,
            "screen_manifest": screen_manifest,
            "screen": screen,
        },
        "preserved_h3_source_sha256": preserved,
        "commands": {
            "training": (
                f"tar -xzf {TRAINING_ARCHIVE_FILENAME} && "
                f"cd {TRAINING_ARCHIVE_ROOT} && ./RUN_H7_TRAIN.sh"
            ),
            "evaluation_after_result_transfer": (
                f"tar -xzf {EVALUATION_ARCHIVE_FILENAME} && "
                f"cd {EVALUATION_ARCHIVE_ROOT} && mkdir treatments && "
                f"tar -xzf ../{TREATMENT_RESULTS_FILENAME} -C treatments && "
                "./RUN_H7_EVALUATE.sh"
            ),
        },
        "isolation": {
            "screen_in_training_archive": False,
            "control_in_training_archive": False,
            "training_data_in_evaluation_archive": False,
            "base_checkpoint_in_evaluation_archive": False,
            "evaluation_transfer_only_after_two_report_hashes_frozen": True,
            "control_scoring_passes": 1,
            "treatment_scoring_passes": 2,
            "third_seed_allowed": False,
            "threshold_sweep_allowed": False,
        },
        "network_credentials_embedded": False,
        "private_or_phi_records_included": False,
        "paid_compute_authorized": False,
    }

    try:
        output.mkdir(parents=False, exist_ok=False)
        for filename, payload in (
            (TRAINING_ARCHIVE_FILENAME, training_archive),
            (EVALUATION_ARCHIVE_FILENAME, evaluation_archive),
            (READINESS_FILENAME, canonical_json_bytes(readiness)),
        ):
            with (output / filename).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
    except OSError as exc:
        raise H7PackagingError("H7 packaging outputs could not be created") from exc
    return readiness


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Package Nano H7 for Kaggle")
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-sha", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    package_h7_kaggle_bundles(
        repo_root=args.repo_root,
        data_dir=args.data_dir,
        control_checkpoint=args.control_checkpoint,
        output_dir=args.output_dir,
        git_sha=args.git_sha,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EVALUATION_ARCHIVE_FILENAME",
    "H7_KAGGLE_PACKAGE_SCHEMA_VERSION",
    "H7_READINESS_SCHEMA_VERSION",
    "READINESS_FILENAME",
    "TRAINING_ARCHIVE_FILENAME",
    "H7PackagingError",
    "package_h7_kaggle_bundles",
]
