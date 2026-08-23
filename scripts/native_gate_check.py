#!/usr/bin/env python3
"""Machine-checkable Native promotion gate.

Every condition is evaluated from live repository/account state, so the verdict
is reproducible rather than asserted. Exits 0 when the gate is OPEN.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS_MANIFEST = ROOT / "artifacts/campaign/native_corpus_screen_v1_manifest.json"
EXPERIMENT_MANIFEST = ROOT / "artifacts/campaign/manifests/native30_revalidation_wave1_v1.json"


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=False)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _wallet() -> float | None:
    rc, out = _run(["runpodctl", "user"])
    if rc != 0:
        return None
    try:
        return float(json.loads(out)["clientBalance"])
    except Exception:
        return None


def check() -> dict:
    conditions: dict[str, dict] = {}

    def add(key: str, ok: bool, detail: str) -> None:
        conditions[key] = {"pass": bool(ok), "detail": detail}

    # 1-2. corpus generator + manifest
    gen = ROOT / "nanoscribe/native/corpus/generators.py"
    add("1_corpus_generator_exists", gen.is_file(), str(gen.relative_to(ROOT)))
    add("2_corpus_manifest_exists", CORPUS_MANIFEST.is_file(), str(CORPUS_MANIFEST.relative_to(ROOT)))

    manifest = json.loads(CORPUS_MANIFEST.read_text()) if CORPUS_MANIFEST.is_file() else {}
    stats = manifest.get("statistics", {})
    tokens = stats.get("training_tokens_char_level", 0)

    # 3. meaningful token volume (>= 10M char-level tokens for a 30M screen)
    add("3_token_volume", tokens >= 10_000_000, f"{tokens:,} char-level training tokens")

    # 4. partition separation
    leak = manifest.get("leakage", {})
    files = manifest.get("partition_files", {})
    add(
        "4_partition_separation",
        bool(leak.get("pass")) and {"TRAIN", "DEV"} <= set(files),
        f"leakage_pass={leak.get('pass')} partition_files={sorted(files)}",
    )

    # 5. axis coverage
    cov = manifest.get("axis_coverage", {})
    add("5_axis_coverage", bool(cov.get("pass")), f"below_floor={cov.get('below_floor')}")

    # 6. dedupe + leakage guards ran
    ded = manifest.get("dedupe", {})
    add(
        "6_dedupe_and_leakage_guards",
        bool(ded) and bool(leak),
        f"duplicate_rate={ded.get('duplicate_rate')} reserved_hits={leak.get('n_reserved_value_hits')}",
    )

    # 7. trainer acceptance tests
    rc, out = _run([sys.executable, "-m", "pytest", "nanoscribe", "-q", "-p", "no:cacheprovider"])
    tail = out.strip().splitlines()[-1] if out.strip() else ""
    add("7_trainer_tests_pass", rc == 0, tail[:120])

    # 8. CI collects nanoscribe
    ci = ROOT / ".github/workflows/ci.yml"
    ci_text = ci.read_text() if ci.is_file() else ""
    add("8_ci_collects_nanoscribe", "pytest nanoscribe" in ci_text, "ci.yml runs the nanoscribe suite")

    # 9. experiment code committed.
    # The gate writes its own result artifact, so that path must be excluded or
    # the check can never pass — it would always observe the file it just wrote.
    rc, out = _run(["git", "status", "--porcelain"])
    self_written = "artifacts/campaign/native_gate_check_v1.json"
    dirty = [
        line
        for line in out.splitlines()
        if line.strip() and self_written not in line
    ]
    add(
        "9_code_committed",
        not dirty,
        f"{len(dirty)} uncommitted path(s)" + (f": {dirty[:3]}" if dirty else ""),
    )

    # 10. exact command + artifact path ready
    exp = json.loads(EXPERIMENT_MANIFEST.read_text()) if EXPERIMENT_MANIFEST.is_file() else {}
    add(
        "10_command_and_artifact_ready",
        bool(exp.get("command")) and bool(exp.get("artifact_dest")) and bool(exp.get("runs")),
        f"{len(exp.get('runs', []))} runs, dest={exp.get('artifact_dest')}",
    )

    # 11. projected cost within live budget
    wallet = _wallet()
    projected = float((exp.get("cost_projection") or {}).get("projected_usd_2x_margin") or 0)
    add(
        "11_cost_within_budget",
        wallet is not None and projected > 0 and projected < wallet,
        f"projected(2x)=${projected} wallet=${wallet}",
    )

    # 12. the corpus backing this experiment may actually support the claim.
    # A unit fixture, or anything under the scientific-token floor, fails here
    # rather than at conclusion time.
    try:
        from nanoscribe.native.corpus.guard import CorpusGuardError, assert_usable

        assert_usable(manifest, "architecture_ranking")
        add("12_corpus_supports_claim", True, "corpus cleared for architecture_ranking")
    except CorpusGuardError as exc:
        add("12_corpus_supports_claim", False, str(exc)[:160])
    except Exception as exc:  # pragma: no cover
        add("12_corpus_supports_claim", False, f"guard unavailable: {exc}")

    failed = [k for k, v in conditions.items() if not v["pass"]]
    return {
        "schema": "nano.campaign.native_gate.v1",
        "checked_at": datetime.now(UTC).isoformat(),
        "verdict": "PROMOTION_GATE_OPEN" if not failed else "PROMOTION_GATE_BLOCKED_DATA",
        "failed_conditions": failed,
        "conditions": conditions,
    }


def main() -> int:
    result = check()
    out = ROOT / "artifacts/campaign/native_gate_check_v1.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "conditions"}, indent=2))
    for key, val in result["conditions"].items():
        print(f"  {'PASS' if val['pass'] else 'FAIL'}  {key:34s} {val['detail']}")
    return 0 if result["verdict"] == "PROMOTION_GATE_OPEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
