"""Wave-1 native30 revalidation — shared orchestration for local, Kaggle, and Pod.

Knowledge baked in (read before changing defaults):

* **Why revalidate?** The original 30M ranking used a 4,516-char fixture with 64-char
  truncation — EXPLORATORY_SCREENING_RANKING only. Promoting to 100M without
  re-screening on the real corpus would promote a successive-halving artefact.

* **Design:** 3 arms × 3 seeds = 9 runs, interleaved by seed so a partial run still
  compares all arms at seed 0. Control arm `reval30_decoder_control` is plain LM
  (span_port=0) — mandatory to separate capacity from mechanism.

* **Corpus:** `native_corpus_screen_v1` (~19k TRAIN rows, ~14.7M char-level tokens).
  Hash must match `native30_revalidation_wave1_v1.json` before any GPU time.

* **Training:** 1800 steps @ batch 32 ≈ 3 epochs on screen corpus (not fixture's 200
  steps ≈ 33 epochs memorisation).

* **Eval:** Each arm gets *constrained* (candidate selection) and *unconstrained*
  (free generation) on `p1_screening_eval_v1`. Scoring uses `analyze_revalidation.py`
  — Wilson intervals vs decoder control AND majority-class baseline.

* **Local Apple Silicon:** MPS ~4× CPU for 30M arms; no paid GPU needed for this screen.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "artifacts/campaign/manifests/native30_revalidation_wave1_v1.json"
CORPUS_MANIFEST_PATH = ROOT / "artifacts/campaign/native_corpus_screen_v1_manifest.json"
DEFAULT_RESULTS_DIR = ROOT / "artifacts/campaign/reval_results"
DEFAULT_SUMMARY_PATH = ROOT / "artifacts/campaign/native30_revalidation_summary_v1.json"
EVAL_SUITE = "p1_screening_eval_v1"
MARKER_DONE = "NATIVE30_REVALIDATION_DONE"


def interleaved_run_ids() -> tuple[str, ...]:
    """Seed-interleaved order: decoder s0, bottleneck s0, span_port s0, decoder s1, …"""
    from nanoscribe.native.factorial import REVALIDATION_ARMS, revalidation_run_id

    seeds = sorted({seed for arm in REVALIDATION_ARMS for seed in arm.seeds})
    return tuple(
        revalidation_run_id(arm, seed) for seed in seeds for arm in REVALIDATION_ARMS
    )


def detect_surface_label() -> str:
    import torch

    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps:apple_silicon"
    return "cpu"


def verify_corpus(root: Path = ROOT) -> dict[str, Any]:
    """Rebuild screen corpus if missing; assert hash + leakage gates match manifest."""
    corpus_manifest = root / "artifacts/campaign/native_corpus_screen_v1_manifest.json"
    if not corpus_manifest.is_file():
        subprocess.run(
            [sys.executable, str(root / "scripts/build_native_corpus.py"), "--stage", "screen"],
            cwd=root,
            check=True,
        )

    manifest = json.loads(corpus_manifest.read_text())
    expected = json.loads((root / MANIFEST_PATH.relative_to(root)).read_text())

    want_hash = expected["dataset"]["content_hash"]
    got_hash = manifest["content_hash"]
    if got_hash != want_hash:
        raise RuntimeError(f"CORPUS_HASH_MISMATCH want={want_hash} got={got_hash}")

    if not manifest.get("leakage", {}).get("pass") or not manifest.get("axis_coverage", {}).get("pass"):
        raise RuntimeError("CORPUS_GATES_FAILED")

    return {
        "content_hash": got_hash,
        "train_rows": manifest["statistics"]["partition_sizes"]["TRAIN"],
        "revision": manifest.get("revision", expected["dataset"]["revision"]),
    }


def preflight_arms(run_ids: tuple[str, ...]) -> None:
    """Construct every arm before training any — catches d_model % n_heads failures early."""
    from nanoscribe.native.config import config_for_run
    from nanoscribe.native.model import build_native_model

    for run_id in run_ids:
        cfg = config_for_run(run_id)
        if cfg.d_model % cfg.n_heads != 0:
            raise RuntimeError(
                f"{run_id}: d_model {cfg.d_model} not divisible by n_heads {cfg.n_heads}"
            )
        build_native_model(cfg)


def train_one_run(
    run_id: str,
    *,
    max_steps: int | None = None,
    cpu_smoke: bool = False,
) -> dict[str, Any]:
    from nanoscribe.native.config import config_for_run
    from nanoscribe.native.train import train_native

    cfg = config_for_run(run_id, cpu_smoke=cpu_smoke)
    if max_steps is not None:
        cfg = replace(cfg, max_steps=max_steps)
    t0 = time.perf_counter()
    result = train_native(cfg)
    payload = result.to_dict()
    payload["wall_seconds"] = round(time.perf_counter() - t0, 1)
    payload["surface"] = detect_surface_label()
    return payload


def eval_one_run(
    run_id: str,
    results_dir: Path,
    *,
    suite: str = EVAL_SUITE,
    eval_cpu: bool = True,
    root: Path = ROOT,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_script = root / "scripts/evaluate_native_nano.py"
    for mode, extra in (("constrained", []), ("unconstrained", ["--unconstrained"])):
        out = results_dir / f"{run_id}_{mode}.json"
        if out.is_file():
            continue
        cmd = [
            sys.executable,
            str(eval_script),
            "--run-id",
            run_id,
            "--suite",
            suite,
            "--output",
            str(out),
        ]
        if eval_cpu:
            cmd.append("--cpu")
        cmd.extend(extra)
        proc = subprocess.run(cmd, cwd=root, check=False, capture_output=True, text=True)
        if proc.returncode != 0 or not out.is_file():
            # Previously this call discarded returncode and stderr, so an eval
            # crash was indistinguishable from an eval that scored zero. Surface
            # it: a missing/failed cell must not silently become a 0-rate row.
            print(
                f"[native30] EVAL FAILED run={run_id} mode={mode} rc={proc.returncode}\n"
                f"  stderr: {(proc.stderr or '').strip()[-2000:]}",
                file=sys.stderr,
                flush=True,
            )


def write_heartbeat(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "heartbeat.txt").write_text(str(int(time.time())))


def run_revalidation_wave(
    results_dir: Path,
    run_ids: tuple[str, ...],
    *,
    max_steps: int | None = None,
    suite: str = EVAL_SUITE,
    skip_train: bool = False,
    skip_eval: bool = False,
    eval_cpu: bool = True,
    cpu_smoke: bool = False,
    root: Path = ROOT,
) -> tuple[list[str], list[str]]:
    """Train + eval each run_id. Returns (failed_train, skipped_complete)."""
    verify_corpus(root)
    preflight_arms(run_ids)
    results_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    skipped: list[str] = []

    for run_id in run_ids:
        write_heartbeat(results_dir)
        train_marker = results_dir / f"{run_id}_train.json"

        if not skip_train:
            if train_marker.is_file():
                skipped.append(run_id)
            else:
                try:
                    payload = train_one_run(
                        run_id,
                        max_steps=max_steps,
                        cpu_smoke=cpu_smoke,
                    )
                    train_marker.write_text(json.dumps(payload, indent=2) + "\n")
                except Exception as exc:
                    failed.append(run_id)
                    err_path = results_dir / f"{run_id}_train_error.txt"
                    err_path.write_text(f"{type(exc).__name__}: {exc}\n")
                    continue

        if not skip_eval:
            eval_one_run(run_id, results_dir, suite=suite, eval_cpu=eval_cpu, root=root)

    return failed, skipped


def import_train_results(
    results_dir: Path,
    out_path: Path = DEFAULT_SUMMARY_PATH,
    *,
    surface: str | None = None,
    run_ids: tuple[str, ...] | None = None,
    root: Path = ROOT,
) -> tuple[int, int]:
    """Merge per-run train/eval JSON into summary stub (pre-analysis)."""
    manifest = json.loads((root / MANIFEST_PATH.relative_to(root)).read_text())
    expected_ids = list(run_ids or [entry["run_id"] for entry in manifest["runs"]])
    runs: list[dict[str, Any]] = []
    missing: list[str] = []

    for run_id in expected_ids:
        train_path = results_dir / f"{run_id}_train.json"
        if not train_path.is_file():
            missing.append(run_id)
            continue
        train = json.loads(train_path.read_text())
        eval_constrained = results_dir / f"{run_id}_constrained.json"
        eval_unconstrained = results_dir / f"{run_id}_unconstrained.json"
        runs.append(
            {
                "run_id": run_id,
                "train": train,
                "eval_constrained": (
                    json.loads(eval_constrained.read_text()) if eval_constrained.is_file() else None
                ),
                "eval_unconstrained": (
                    json.loads(eval_unconstrained.read_text()) if eval_unconstrained.is_file() else None
                ),
            }
        )

    complete = len(runs) == len(expected_ids) and not missing
    summary = {
        "schema": "nano.campaign.native30_revalidation.v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "experiment_id": manifest["experiment_id"],
        "corpus": manifest["dataset"],
        "surface": surface or detect_surface_label(),
        "import_script": "scripts/import_kaggle_native30_results.py",
        "orchestrator": "nanoscribe.campaign.native30_revalidation",
        "results_dir": str(results_dir),
        "runs_expected": len(expected_ids),
        "runs_imported": len(runs),
        "runs_missing": missing,
        "marker": MARKER_DONE if complete else "INCOMPLETE",
        "verdict": "COMPLETE" if complete else "PARTIAL",
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    return len(runs), len(expected_ids)


def analyze_and_write_summary(
    results_dir: Path,
    out_path: Path = DEFAULT_SUMMARY_PATH,
    root: Path = ROOT,
) -> int:
    """Run statistical analysis (Wilson intervals, promotion verdicts)."""
    analyze = root / "scripts/analyze_revalidation.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(analyze),
            "--results-dir",
            str(results_dir),
            "--out",
            str(out_path),
        ],
        cwd=root,
        check=False,
    )
    return proc.returncode
