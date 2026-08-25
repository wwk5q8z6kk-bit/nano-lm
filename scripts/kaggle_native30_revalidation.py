# Wave-1 30M revalidation — runs end-to-end on a FREE Kaggle GPU (T4).
#
# Why Kaggle rather than RunPod for this screen:
#   * the arms are 30M params on 14.7M char-level tokens — a T4 is sufficient;
#   * Kaggle is free, so an orchestration bug costs time instead of money.
#     Four RunPod launches were lost to orchestration defects (torch wheel vs
#     driver, a CLI default overriding the corpus, a non-divisible head count,
#     and a watchdog that killed a healthy pod);
#   * a Kaggle session has its own lifecycle, so there is no idle-burn failure
#     mode when a background script dies.
#
# Usage (Kaggle notebook, Accelerator = GPU T4 x2, Internet = ON):
#     !pip -q install torch --index-url https://download.pytorch.org/whl/cu121
#     !python kaggle_native30_revalidation.py
#
# Resume-safe: completed runs are skipped on re-execution.
import json
import os
import subprocess
import sys
import time

import re

REPO = "https://github.com/wwk5q8z6kk-bit/nano-lm.git"

# BRANCH/COMMIT reach `git` through shell=True (the setup steps need pipes and
# redirects), so constrain them to characters valid in a git ref. Without this a
# crafted env var could inject shell metacharacters.
_REF_RE = re.compile(r"^[A-Za-z0-9._/-]{1,200}$")


def _safe_ref(value: str, label: str, default: str = "") -> str:
    if not value:
        return default
    if not _REF_RE.match(value):
        raise SystemExit(f"refusing unsafe {label}: {value!r}")
    return value


BRANCH = _safe_ref(
    os.environ.get("NANO_BRANCH", ""), "NANO_BRANCH", "frontier/accelerated-research-campaign-v2"
)
COMMIT = _safe_ref(os.environ.get("NANO_COMMIT", ""), "NANO_COMMIT")
WORKDIR = "/kaggle/working/nano-lm"
RESULTS = "/kaggle/working/reval_results"


def sh(cmd, check=True, cwd=None):
    print(f"$ {cmd}", flush=True)
    proc = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout[-2000:], flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr, flush=True)
        if check:
            raise SystemExit(f"command failed: {cmd}")
    return proc


def main() -> int:
    import torch

    assert torch.cuda.is_available(), (
        "No GPU — enable the accelerator (Kaggle: Settings > Accelerator > GPU T4 x2)"
    )
    cap = torch.cuda.get_device_capability()
    assert cap >= (7, 0), (
        f"{torch.cuda.get_device_name()} has CUDA capability {cap}; this build needs >=7.0 "
        "(P100 is 6.0 and unsupported — switch to GPU T4 x2)"
    )
    print(f"GPU: {torch.cuda.get_device_name()} capability {cap}", flush=True)

    if not os.path.isdir(WORKDIR):
        sh(f"git clone -q --branch {BRANCH} {REPO} {WORKDIR}")
    sh(f"git fetch -q origin {BRANCH}", cwd=WORKDIR)
    sh(f"git checkout -q {COMMIT or f'origin/{BRANCH}'}", cwd=WORKDIR)
    sh("git rev-parse --short HEAD", cwd=WORKDIR)

    # Deps EXCEPT torch: Kaggle ships a torch matched to its driver, and
    # installing the pinned wheel over it is what broke a RunPod launch.
    sh("grep -v -E '^torch==' requirements.txt > /tmp/req_no_torch.txt", cwd=WORKDIR)
    sh("pip -q install -r /tmp/req_no_torch.txt", cwd=WORKDIR, check=False)

    sys.path.insert(0, WORKDIR)
    os.chdir(WORKDIR)

    # Rebuild the corpus and verify it matches the committed manifest.
    sh("python3 scripts/build_native_corpus.py --stage screen", cwd=WORKDIR)
    manifest = json.load(open(f"{WORKDIR}/artifacts/campaign/native_corpus_screen_v1_manifest.json"))
    exp = json.load(
        open(f"{WORKDIR}/artifacts/campaign/manifests/native30_revalidation_wave1_v1.json")
    )
    assert manifest["content_hash"] == exp["dataset"]["content_hash"], (
        f"CORPUS_HASH_MISMATCH want={exp['dataset']['content_hash']} got={manifest['content_hash']}"
    )
    assert manifest["leakage"]["pass"] and manifest["axis_coverage"]["pass"], "CORPUS_GATES_FAILED"
    print(
        f"CORPUS_OK hash={manifest['content_hash'][:16]} "
        f"rows={manifest['statistics']['partition_sizes']['TRAIN']}",
        flush=True,
    )

    from nanoscribe.native.config import config_for_run
    from nanoscribe.native.factorial import REVALIDATION_ARMS, revalidation_run_id
    from nanoscribe.native.model import build_native_model
    from nanoscribe.native.train import train_native

    # Interleaved: a partial session still yields a complete seed-0 comparison
    # across all three arms rather than three seeds of a single arm.
    seeds = sorted({s for a in REVALIDATION_ARMS for s in a.seeds})
    run_ids = [revalidation_run_id(a, s) for s in seeds for a in REVALIDATION_ARMS]

    # Construct EVERY arm before training ANY. A 9-run job once died three runs
    # in on an arm with a non-divisible head count, wasting the rest.
    for rid in run_ids:
        cfg = config_for_run(rid)
        assert cfg.d_model % cfg.n_heads == 0, (
            f"{rid}: d_model {cfg.d_model} not divisible by n_heads {cfg.n_heads}"
        )
        build_native_model(cfg)
        print(f"  ARM_OK {rid} d_model={cfg.d_model} n_heads={cfg.n_heads}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    failed = []
    for rid in run_ids:
        done_marker = f"{RESULTS}/{rid}_train.json"
        if os.path.exists(done_marker):
            print(f"SKIP {rid} (already complete)", flush=True)
            continue
        print(f"==> train {rid}", flush=True)
        t0 = time.time()
        try:
            result = train_native(config_for_run(rid))
            payload = result.to_dict() if hasattr(result, "to_dict") else {"result": str(result)}
            payload["wall_seconds"] = round(time.time() - t0, 1)
            json.dump(payload, open(done_marker, "w"), indent=2)
            print(f"RUN_DONE {rid} in {payload['wall_seconds']}s", flush=True)
        except Exception as exc:  # one bad arm must not kill the rest
            failed.append(rid)
            print(f"RUN_FAILED {rid}: {type(exc).__name__}: {exc}", flush=True)
            continue

        for mode, flag in (("constrained", ""), ("unconstrained", "--unconstrained")):
            sh(
                f"python3 scripts/evaluate_native_nano.py --run-id {rid} "
                f"--suite p1_screening_eval_v1 {flag} "
                f"--output {RESULTS}/{rid}_{mode}.json",
                cwd=WORKDIR,
                check=False,
            )

    if failed:
        print(f"RUNS_FAILED: {failed}", flush=True)
    print("NATIVE30_REVALIDATION_DONE", flush=True)
    print(f"results in {RESULTS} — download and commit to artifacts/campaign/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
