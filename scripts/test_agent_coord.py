#!/usr/bin/env python3
"""Tests for cross-worktree coordination, locks, sharded registries, and git hooks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import agent_coord  # noqa: E402
import install_coordination_hooks  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_coord_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cdir = tmp_path / "coord"
    cdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("NANOLM_COORD_DIR", str(cdir))
    return cdir


def _run_coord(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None, stdin_bytes: bytes | None = None):
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "agent_coord.py"), *args],
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        input=stdin_bytes,
        capture_output=True,
        timeout=30,
    )


def _init_repo_with_worktree(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "repo"
    wt = tmp_path / "wt-old"
    main.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "master", str(main)], check=True)
    subprocess.run(["git", "-C", str(main), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(main), "config", "user.name", "Test"], check=True)
    (main / "README.md").write_text("init\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(main), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(main), "commit", "-qm", "init"], check=True)
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", "-b", "feature/old", str(wt)], check=True)
    return main, wt


def test_lease_claim_conflict_and_self_reclaim(tmp_path: Path) -> None:
    main, wt = _init_repo_with_worktree(tmp_path)

    r1 = _run_coord(["claim", "--agent", "agent-a", "--task", "t1", "docs/"], cwd=main)
    assert r1.returncode == 0, r1.stderr.decode()

    # Idempotent self-reclaim on same worktree succeeds (does not conflict with itself)
    r1_again = _run_coord(["claim", "--agent", "agent-a", "--task", "t1b", "docs/"], cwd=main)
    assert r1_again.returncode == 0

    # Overlapping claim from sibling worktree is refused (exit 2)
    r2 = _run_coord(["claim", "--agent", "agent-b", "--task", "t2", "docs/sub/file.md"], cwd=wt)
    assert r2.returncode == 2
    assert b"CONFLICT" in r2.stderr

    # Disjoint claim from sibling worktree succeeds
    r3 = _run_coord(["claim", "--agent", "agent-b", "--task", "t2", "nano/"], cwd=wt)
    assert r3.returncode == 0

    # Check stdin-nul with spaces
    chk = _run_coord(
        ["check", "--stdin-nul"],
        cwd=wt,
        stdin_bytes=b"docs/my file.md\x00other/ok.py\x00",
    )
    assert chk.returncode == 1
    assert b"CLAIMED" in chk.stderr


def test_dead_pid_and_ttl_expiry(tmp_path: Path, isolated_coord_dir: Path) -> None:
    main, wt = _init_repo_with_worktree(tmp_path)

    # Dead PID lease should be ignored and reaped
    r_dead = _run_coord(
        ["claim", "--agent", "dead-agent", "--bind-pid", "99999999", "scripts/"],
        cwd=main,
    )
    assert r_dead.returncode == 0
    r_other = _run_coord(["claim", "--agent", "live-agent", "scripts/foo.py"], cwd=wt)
    assert r_other.returncode == 0

    # TTL=0 expiry
    r_ttl = _run_coord(["claim", "--agent", "short", "--ttl-sec", "0", "nano_ai/"], cwd=main)
    assert r_ttl.returncode == 0
    time.sleep(0.05)
    r_gc = _run_coord(["gc"], cwd=main)
    assert r_gc.returncode == 0
    assert b"reaped stale lease" in r_gc.stdout


def test_lock_run_mutex_and_runpod_budget(tmp_path: Path, isolated_coord_dir: Path) -> None:
    main, _ = _init_repo_with_worktree(tmp_path)

    # Start background lock holder for 1.2s
    holder = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPTS_DIR / "agent_coord.py"),
            "lock-run",
            "--resource",
            "mps",
            "--agent",
            "holder",
            "--reason",
            "bench",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(1.2)",
        ],
        cwd=str(main),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait until holder lock is active
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if (isolated_coord_dir / "locks" / "mps.holder.json").exists():
                break
            time.sleep(0.05)
        assert (isolated_coord_dir / "locks" / "mps.holder.json").exists()

        # Contender with short timeout gets exit 3
        contender = _run_coord(
            [
                "lock-run",
                "--resource",
                "mps",
                "--timeout",
                "0.2",
                "--agent",
                "contender",
                "--",
                sys.executable,
                "-c",
                "print('ok')",
            ],
            cwd=main,
        )
        assert contender.returncode == 3
        assert b"busy" in contender.stderr
    finally:
        assert holder.wait(timeout=10) == 0

    # After holder exits, same lock succeeds
    after = _run_coord(
        [
            "lock-run",
            "--resource",
            "mps",
            "--timeout",
            "2.0",
            "--",
            sys.executable,
            "-c",
            "print('acquired')",
        ],
        cwd=main,
    )
    assert after.returncode == 0
    assert b"acquired" in after.stdout

    # RunPod budget reserve guard
    (isolated_coord_dir / "budget.json").write_text(
        json.dumps({"runpod_balance_usd": 8.93, "reserve_floor_usd": 20.0}),
        encoding="utf-8",
    )
    rp_blocked = _run_coord(
        ["lock-run", "--resource", "runpod", "--", sys.executable, "-c", "print('x')"],
        cwd=main,
    )
    assert rp_blocked.returncode == 4
    assert b"reserve floor" in rp_blocked.stderr

    rp_allowed = _run_coord(
        ["lock-run", "--resource", "runpod", "--", sys.executable, "-c", "print('allowed')"],
        cwd=main,
        env={"AGENT_ALLOW_PAID_RUNPOD": "1"},
    )
    assert rp_allowed.returncode == 0
    assert b"allowed" in rp_allowed.stdout


def test_sharded_registry_and_monolith_guard(tmp_path: Path) -> None:
    main, _ = _init_repo_with_worktree(tmp_path)

    re_res = _run_coord(
        [
            "record-experiment",
            "--id",
            "EXP-TEST-01",
            "--status",
            "completed",
            "--verdict",
            "PASS",
            "--paid-usd",
            "0",
        ],
        cwd=main,
    )
    assert re_res.returncode == 0

    rf_res = _run_coord(
        [
            "record-finding",
            "--id",
            "FIND-TEST-01",
            "--focus",
            "coordination",
            "--finding",
            "shards work",
        ],
        cwd=main,
    )
    assert rf_res.returncode == 0

    chk = _run_coord(
        [
            "compile-registry",
            "--repo-root",
            str(main),
            "--out-experiments",
            str(tmp_path / "exp.json"),
            "--out-ledger",
            str(tmp_path / "led.json"),
            "--check",
        ],
        cwd=main,
    )
    assert chk.returncode == 0, chk.stderr.decode()
    assert json.loads((tmp_path / "exp.json").read_text())["total_entries"] == 1
    assert json.loads((tmp_path / "led.json").read_text())["total_entries"] == 1

    # Monolithic root file triggers failure in --check
    (main / "NANOLM_EXPERIMENT_REGISTRY.json").write_text("{}\n", encoding="utf-8")
    bad = _run_coord(["compile-registry", "--repo-root", str(main), "--check"], cwd=main)
    assert bad.returncode == 1


def test_end_to_end_hooks_across_older_linked_worktree(tmp_path: Path) -> None:
    main, wt_old = _init_repo_with_worktree(tmp_path)

    # Copy coordination scripts and hooks into main (simulating master having them while wt_old does not)
    (main / "scripts").mkdir()
    (main / ".githooks").mkdir()
    for fname in ("agent_coord.py",):
        (main / "scripts" / fname).write_text((SCRIPTS_DIR / fname).read_text(), encoding="utf-8")
    hooks_src = SCRIPTS_DIR.parent / ".githooks"
    for h in ("pre-commit", "pre-push", "post-checkout", "post-merge"):
        dst = main / ".githooks" / h
        dst.write_text((hooks_src / h).read_text(), encoding="utf-8")
        dst.chmod(0o755)

    install_coordination_hooks.install(main)

    # Claim README.md in main
    assert _run_coord(["claim", "--agent", "main-agent", "claimed_dir/"], cwd=main).returncode == 0

    # wt_old does NOT have .githooks/ or scripts/ in its working tree, yet hooks MUST fire!
    # 1. Protected evidence file blocked
    (wt_old / "SHA256SUMS").write_text("abc\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(wt_old), "add", "SHA256SUMS"], check=True)
    c1 = subprocess.run(["git", "-C", str(wt_old), "commit", "-qm", "bad1"], capture_output=True, text=True)
    assert c1.returncode != 0
    assert "evidence-protected" in c1.stderr
    subprocess.run(["git", "-C", str(wt_old), "reset", "-q"], check=True)
    (wt_old / "SHA256SUMS").unlink()

    # 2. Monolithic registry file blocked
    (wt_old / "NANOLM_EXPERIMENT_REGISTRY.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(wt_old), "add", "NANOLM_EXPERIMENT_REGISTRY.json"], check=True)
    c2 = subprocess.run(["git", "-C", str(wt_old), "commit", "-qm", "bad2"], capture_output=True, text=True)
    assert c2.returncode != 0
    assert "monolithic shared JSON" in c2.stderr
    subprocess.run(["git", "-C", str(wt_old), "reset", "-q"], check=True)
    (wt_old / "NANOLM_EXPERIMENT_REGISTRY.json").unlink()

    # 3. File claimed by main worktree's lease blocked
    (wt_old / "claimed_dir").mkdir()
    (wt_old / "claimed_dir" / "x.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(wt_old), "add", "claimed_dir/x.py"], check=True)
    c3 = subprocess.run(["git", "-C", str(wt_old), "commit", "-qm", "bad3"], capture_output=True, text=True)
    assert c3.returncode != 0
    assert "claimed by another live agent lease" in c3.stderr
    subprocess.run(["git", "-C", str(wt_old), "reset", "-q"], check=True)

    # 4. Unclaimed clean file succeeds
    (wt_old / "unclaimed.py").write_text("y = 2\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(wt_old), "add", "unclaimed.py"], check=True)
    c4 = subprocess.run(["git", "-C", str(wt_old), "commit", "-qm", "good"], capture_output=True, text=True)
    assert c4.returncode == 0, c4.stderr
