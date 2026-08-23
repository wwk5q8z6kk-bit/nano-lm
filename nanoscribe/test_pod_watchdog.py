"""Watchdog parsing pins.

A positional parser terminated a HEALTHY pod at 97% GPU: `grep -c` prints "0"
and exits 1, so `|| echo 0` emitted a duplicate line and the process count was
read from the wrong index. These pin the states that decide whether a paid pod
lives or dies.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pod_watchdog import parse_probe


def test_healthy_run_is_not_mistaken_for_death() -> None:
    probe = "WD_DONE=0\nWD_PROCS=1\nWD_UTIL=97\n"
    got = parse_probe(probe)
    assert got == {"done": False, "procs": 1, "util": 97}
    assert got["procs"] > 0, "a running trainer must never read as zero processes"


def test_completed_run_detected() -> None:
    assert parse_probe("WD_DONE=1\nWD_PROCS=0\nWD_UTIL=0\n")["done"] is True


def test_dead_script_detected() -> None:
    got = parse_probe("WD_DONE=0\nWD_PROCS=0\nWD_UTIL=0\n")
    assert got["done"] is False and got["procs"] == 0


def test_unreadable_probe_returns_none_not_zeros() -> None:
    """An SSH failure must not look like a dead process — that terminates a pod."""
    for bad in ("", "Warning: banner text\n", "garbage", "WD_DONE=0\n"):
        assert parse_probe(bad) is None, f"{bad!r} must be unusable, not read as death"


def test_duplicate_zero_line_does_not_shift_fields() -> None:
    """The exact shape that killed the healthy pod."""
    probe = "0\n0\nWD_DONE=0\nWD_PROCS=2\nWD_UTIL=88\n"
    got = parse_probe(probe)
    assert got["procs"] == 2, "labelled parsing must ignore stray positional lines"


def test_banner_noise_tolerated() -> None:
    probe = "Enjoy your Pod\nWD_DONE=0\nWD_PROCS=3\nWD_UTIL=45\nroot@host:~#\n"
    assert parse_probe(probe) == {"done": False, "procs": 3, "util": 45}


def test_fetch_failure_is_reported_not_silently_ignored() -> None:
    """A failed retrieval must be visible, because terminate() follows it.

    The first watchdog terminated on the done-marker with no retrieval step at
    all, deleting a completed nine-arm run along with the pod. --require-fetch
    now refuses to terminate unless the results are safely local.
    """
    import pod_watchdog

    def fake_ssh(pod, cmd, timeout=120):
        return ""  # pod unreachable / empty archive

    original = pod_watchdog._ssh
    pod_watchdog._ssh = fake_ssh
    try:
        got = pod_watchdog.fetch_results("pod", "/workspace/x", Path("/tmp/wd_fetch_test"))
    finally:
        pod_watchdog._ssh = original

    assert got["fetched"] is False
    assert "reason" in got, "a failed fetch must explain why"


def test_require_fetch_flag_exists() -> None:
    """The guard has to be reachable from the CLI to be usable."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "pod_watchdog.py").read_text()
    assert "--require-fetch" in src
    assert "--fetch-remote" in src
    # termination must be guarded by the fetch result
    assert "if args.require_fetch and not got.get(\"fetched\")" in src
