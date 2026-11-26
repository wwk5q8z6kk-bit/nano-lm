#!/usr/bin/env python3
"""Unified cross-worktree coordination engine (stdlib only, race-safe).

Provides:
  - Cooperative per-lane path leases (`claim`, `heartbeat`, `release`, `gc`, `check`, `status`)
  - Session presence/status announcements (`announce`)
  - Hardware & compute mutex (`lock-run`) with RunPod $20 reserve guard
  - Conflict-free sharded experiment & finding records (`record-experiment`, `record-finding`, `compile-registry`)
  - Live coordination board (`board`)

State lives in `NANOLM_COORD_DIR` if set, otherwise `<git-common-dir>/agent-coord`.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as _dt
import fcntl
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any

DEFAULT_TTL_SEC = int(float(os.environ.get("AGENT_LEASE_TTL_H", "6")) * 3600)
DEFAULT_RESERVE_FLOOR_USD = 20.0
VALID_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
VALID_RESOURCES = ("mps", "cpu-benchmark", "runpod", "kaggle", "gcp-tpu")


def _git(*args: str, cwd: Path | None = None) -> tuple[int, str, str]:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    cmd = ["git", "-c", "core.fsmonitor=false"]
    if cwd is not None:
        cmd += ["-C", str(cwd)]
    cmd += list(args)
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def repo_root(start: Path | None = None) -> Path:
    rc, out, _ = _git("rev-parse", "--show-toplevel", cwd=start)
    if rc == 0 and out:
        return Path(os.path.realpath(out))
    return Path(os.path.realpath(str(start or Path.cwd())))


def common_git_dir(root: Path) -> Path:
    rc, out, _ = _git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=root)
    if rc == 0 and out:
        return Path(os.path.realpath(out))
    rc2, out2, _ = _git("rev-parse", "--git-common-dir", cwd=root)
    if rc2 == 0 and out2:
        p = Path(out2)
        return Path(os.path.realpath(str(p if p.is_absolute() else root / p)))
    return root / ".git"


def coord_dir(root: Path | None = None) -> Path:
    override = os.environ.get("NANOLM_COORD_DIR")
    if override:
        base = Path(override).resolve()
    else:
        r = root or repo_root()
        base = common_git_dir(r) / "agent-coord"
    for sub in ("leases", "sessions", "locks", "compiled"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def lane_key(worktree_path: str | Path) -> str:
    real = os.path.realpath(str(worktree_path))
    digest = hashlib.sha1(real.encode("utf-8")).hexdigest()[:8]
    name = Path(real).name or "root"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return f"{safe}-{digest}"


@contextmanager
def exclusive_lock(lock_path: Path, timeout: float = 10.0):
    """Acquire an exclusive flock without ever unlinking the lock inode."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError(f"Lock path is not a regular file: {lock_path}")
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for lock {lock_path.name}")
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def lease_is_live(lease: dict[str, Any], now: float | None = None) -> bool:
    now = time.time() if now is None else now
    try:
        hb = float(lease.get("heartbeat", 0))
        ttl = float(lease.get("ttl_sec", DEFAULT_TTL_SEC))
    except (TypeError, ValueError):
        return False
    if now > hb + ttl:
        return False
    wt = lease.get("worktree", "")
    if not wt or not os.path.isdir(wt):
        return False
    pid = lease.get("pid")
    if pid is not None:
        try:
            if not pid_alive(int(pid)):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _norm_path(p: str) -> str:
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p.rstrip("/") or "."


def paths_overlap(a: str, b: str) -> bool:
    na, nb = _norm_path(a), _norm_path(b)
    if na == nb:
        return True
    if na == "." or nb == ".":
        return True
    if na.startswith(nb + "/") or nb.startswith(na + "/"):
        return True
    has_glob_a = any(ch in na for ch in "*?[")
    has_glob_b = any(ch in nb for ch in "*?[")
    if has_glob_a and fnmatch.fnmatch(nb, na):
        return True
    if has_glob_b and fnmatch.fnmatch(na, nb):
        return True
    if has_glob_a and has_glob_b:
        lit_a = na.split("*", 1)[0].split("?", 1)[0].split("[", 1)[0]
        lit_b = nb.split("*", 1)[0].split("?", 1)[0].split("[", 1)[0]
        if lit_a and lit_b and (lit_a.startswith(lit_b) or lit_b.startswith(lit_a)):
            return True
    return False


def _load_claude_peer_claims(now: float) -> list[dict[str, Any]]:
    env_path = os.environ.get("NANOLM_CLAUDE_PEER_CLAIMS_FILE")
    if env_path is not None:
        claims_file = Path(env_path).expanduser() if env_path else None
    elif os.environ.get("NANOLM_COORD_DIR"):
        claims_file = None  # Hermetic test isolation unless explicitly opted in
    else:
        claims_file = Path.home() / ".claude" / "state" / "peer_claims.json"
    if not claims_file or not claims_file.is_file():
        return []
    raw = read_json(claims_file, default=[])
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            hb = float(item.get("last_heartbeat", 0))
            ttl = float(item.get("ttl_sec", 300))
        except (TypeError, ValueError):
            continue
        if now > hb + ttl:
            continue
        glob_pat = item.get("scope_glob")
        if glob_pat:
            out.append(
                {
                    "agent": f"claude-peer:{item.get('session_id', 'unknown')}",
                    "worktree": "claude-peer-registry",
                    "branch": "-",
                    "paths": [str(glob_pat)],
                }
            )
    return out


def _load_leases(cdir: Path) -> dict[str, dict[str, Any]]:
    leases: dict[str, dict[str, Any]] = {}
    ldir = cdir / "leases"
    if not ldir.is_dir():
        return leases
    for p in sorted(ldir.glob("*.json")):
        data = read_json(p)
        if isinstance(data, dict):
            leases[p.stem] = data
    return leases


def find_conflicts(
    cdir: Path, paths: list[str], skip_key: str | None = None
) -> list[tuple[str, str, str, str]]:
    if not paths:
        return []
    now = time.time()
    conflicts: list[tuple[str, str, str, str]] = []
    for key, lease in _load_leases(cdir).items():
        if key == skip_key or not lease_is_live(lease, now):
            continue
        for p in paths:
            for q in lease.get("paths") or []:
                if paths_overlap(p, str(q)):
                    conflicts.append(
                        (
                            str(lease.get("agent", "unknown")),
                            str(lease.get("worktree", "")),
                            str(lease.get("branch", "")),
                            str(q),
                        )
                    )
    for ext in _load_claude_peer_claims(now):
        for p in paths:
            for q in ext.get("paths") or []:
                if paths_overlap(p, str(q)):
                    conflicts.append(
                        (ext["agent"], ext["worktree"], ext["branch"], str(q))
                    )
    return conflicts


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_claim(args: argparse.Namespace) -> int:
    root = repo_root()
    cdir = coord_dir(root)
    my_key = lane_key(root)
    paths = list(args.paths) + list(args.paths_flag or [])
    with exclusive_lock(cdir / "leases" / ".lock"):
        conflicts = find_conflicts(cdir, paths, skip_key=my_key)
        if conflicts and not args.force:
            for c in conflicts:
                print(
                    f"CONFLICT with agent={c[0]} worktree={c[1]} branch={c[2]} pattern={c[3]}",
                    file=sys.stderr,
                )
            return 2
        _, branch, _ = _git("branch", "--show-current", cwd=root)
        _, base_sha, _ = _git("merge-base", "HEAD", "origin/master", cwd=root)
        now = time.time()
        payload: dict[str, Any] = {
            "lane_key": my_key,
            "agent": args.agent,
            "task": args.task,
            "worktree": str(root),
            "branch": branch or "(detached)",
            "base_sha": base_sha[:12],
            "paths": [_norm_path(p) for p in paths],
            "started": now,
            "heartbeat": now,
            "ttl_sec": int(args.ttl_sec),
        }
        if args.bind_pid is not None:
            payload["pid"] = int(args.bind_pid)
        atomic_write_json(cdir / "leases" / f"{my_key}.json", payload)
        print(f"claimed {root} ({my_key})")
    return 0


def cmd_heartbeat(args: argparse.Namespace) -> int:
    root = repo_root()
    cdir = coord_dir(root)
    my_key = lane_key(root)
    lease_file = cdir / "leases" / f"{my_key}.json"
    with exclusive_lock(cdir / "leases" / ".lock"):
        data = read_json(lease_file)
        if not isinstance(data, dict):
            print(f"no active lease for {my_key}", file=sys.stderr)
            return 1
        data["heartbeat"] = time.time()
        if args.agent and args.agent != "unknown":
            data["agent"] = args.agent
        atomic_write_json(lease_file, data)
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    root = repo_root()
    cdir = coord_dir(root)
    my_key = lane_key(root)
    lease_file = cdir / "leases" / f"{my_key}.json"
    with exclusive_lock(cdir / "leases" / ".lock"):
        if lease_file.exists():
            lease_file.unlink()
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    root = repo_root()
    cdir = coord_dir(root)
    now = time.time()
    reaped = 0
    with exclusive_lock(cdir / "leases" / ".lock"):
        for p in sorted((cdir / "leases").glob("*.json")):
            data = read_json(p)
            if not isinstance(data, dict) or not lease_is_live(data, now):
                p.unlink(missing_ok=True)
                reaped += 1
                print(f"reaped stale lease {p.name}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    root = repo_root()
    cdir = coord_dir(root)
    my_key = lane_key(root)
    paths = list(args.paths)
    if args.stdin_nul:
        raw = sys.stdin.buffer.read()
        for chunk in raw.split(b"\x00"):
            if chunk:
                paths.append(chunk.decode("utf-8", errors="replace"))
    if not paths:
        return 0
    with exclusive_lock(cdir / "leases" / ".lock"):
        conflicts = find_conflicts(cdir, paths, skip_key=my_key)
    for c in conflicts:
        print(
            f"CLAIMED by agent={c[0]} worktree={c[1]} branch={c[2]} pattern={c[3]}",
            file=sys.stderr,
        )
    return 1 if conflicts else 0


def cmd_status(args: argparse.Namespace) -> int:
    root = repo_root()
    cdir = coord_dir(root)
    now = time.time()
    rows = []
    with exclusive_lock(cdir / "leases" / ".lock"):
        for key, lease in _load_leases(cdir).items():
            live = lease_is_live(lease, now)
            age_h = (now - float(lease.get("heartbeat", now))) / 3600.0
            rows.append(
                {
                    "lane_key": key,
                    "live": live,
                    "agent": lease.get("agent", "unknown"),
                    "branch": lease.get("branch", ""),
                    "worktree": lease.get("worktree", ""),
                    "age_h": round(age_h, 2),
                    "paths": lease.get("paths", []),
                    "task": lease.get("task", ""),
                }
            )
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for r in rows:
            tag = "LIVE " if r["live"] else "STALE"
            print(
                f"{tag} {r['agent']:16} {r['branch'][:36]:36} {r['age_h']:5.2f}h  {r['paths']}  {r['task']}"
            )
    return 0


def cmd_announce(args: argparse.Namespace) -> int:
    root = repo_root()
    cdir = coord_dir(root)
    my_key = lane_key(root)
    _, branch, _ = _git("branch", "--show-current", cwd=root)
    _, head_sha, _ = _git("rev-parse", "--short=12", "HEAD", cwd=root)
    payload = {
        "lane_key": my_key,
        "updated_utc": _utc_now(),
        "updated_epoch": time.time(),
        "agent": args.agent,
        "status": args.status,
        "worktree": str(root),
        "branch": branch or "(detached)",
        "head_sha": head_sha,
        "scope": args.scope,
        "findings": args.findings,
    }
    atomic_write_json(cdir / "sessions" / f"{my_key}.json", payload)
    print(f"announced {my_key}: {args.agent} [{args.status}]")
    return 0


def _check_runpod_budget(cdir: Path) -> tuple[bool, str]:
    if os.environ.get("AGENT_ALLOW_PAID_RUNPOD") == "1":
        return True, "override AGENT_ALLOW_PAID_RUNPOD=1"
    budget_file = cdir / "budget.json"
    budget = read_json(budget_file, default={}) if budget_file.is_file() else {}
    env_bal = os.environ.get("NANOLM_RUNPOD_BALANCE_USD")
    balance = None
    if env_bal is not None:
        try:
            balance = float(env_bal)
        except ValueError:
            pass
    elif isinstance(budget, dict) and "runpod_balance_usd" in budget:
        try:
            balance = float(budget["runpod_balance_usd"])
        except (TypeError, ValueError):
            pass
    floor = DEFAULT_RESERVE_FLOOR_USD
    if isinstance(budget, dict) and "reserve_floor_usd" in budget:
        try:
            floor = float(budget["reserve_floor_usd"])
        except (TypeError, ValueError):
            pass
    if balance is not None and balance <= floor:
        return (
            False,
            f"RunPod balance ${balance:.2f} <= ${floor:.2f} owner reserve floor. "
            "Use free MPS/Kaggle/Colab/GCP credits or set AGENT_ALLOW_PAID_RUNPOD=1 only with owner authorization.",
        )
    return True, "ok"


def cmd_lock_run(args: argparse.Namespace) -> int:
    root = repo_root()
    cdir = coord_dir(root)
    cmd_args = list(args.cmd)
    if cmd_args and cmd_args[0] == "--":
        cmd_args = cmd_args[1:]
    if not cmd_args:
        print("lock-run: missing command after --", file=sys.stderr)
        return 2
    if args.resource == "runpod":
        ok, msg = _check_runpod_budget(cdir)
        if not ok:
            print(f"lock-run refused: {msg}", file=sys.stderr)
            return 4
    lock_path = cdir / "locks" / f"{args.resource}.lock"
    holder_path = cdir / "locks" / f"{args.resource}.holder.json"
    try:
        lock_ctx = exclusive_lock(lock_path, timeout=float(args.timeout))
    except Exception as exc:
        print(f"lock-run error: {exc}", file=sys.stderr)
        return 3
    try:
        with lock_ctx:
            _, branch, _ = _git("branch", "--show-current", cwd=root)
            atomic_write_json(
                holder_path,
                {
                    "resource": args.resource,
                    "agent": args.agent,
                    "worktree": str(root),
                    "branch": branch or "(detached)",
                    "pid": os.getpid(),
                    "started_utc": _utc_now(),
                    "reason": args.reason,
                    "cmd": cmd_args,
                },
            )
            try:
                proc = subprocess.run(cmd_args)
                return proc.returncode
            finally:
                holder_path.unlink(missing_ok=True)
    except TimeoutError:
        holder = read_json(holder_path, default={})
        print(
            f"lock-run busy: resource={args.resource} timeout={args.timeout}s holder={holder}",
            file=sys.stderr,
        )
        return 3


def _validate_id(val: str) -> str:
    if not VALID_ID_RE.match(val):
        raise ValueError(f"Invalid ID {val!r}: must match {VALID_ID_RE.pattern}")
    return val


def cmd_record_experiment(args: argparse.Namespace) -> int:
    root = repo_root()
    exp_id = _validate_id(args.id)
    if args.paid_usd < 0:
        print("record-experiment: --paid-usd must be >= 0", file=sys.stderr)
        return 1
    _, branch, _ = _git("branch", "--show-current", cwd=root)
    _, head_sha, _ = _git("rev-parse", "--short=12", "HEAD", cwd=root)
    payload = {
        "schema": "nanolm.coordination.experiment.v1",
        "id": exp_id,
        "status": args.status,
        "verdict": args.verdict,
        "gate": args.gate,
        "title": args.title,
        "seeds": args.seeds,
        "paid_usd": round(float(args.paid_usd), 4),
        "artifact": args.artifact,
        "report": args.report,
        "summary": args.summary,
        "branch": branch or "(detached)",
        "head_sha": head_sha,
        "recorded_utc": _utc_now(),
    }
    out = root / "coordination" / "experiments" / f"{exp_id}.json"
    atomic_write_json(out, payload)
    print(f"recorded {out.relative_to(root)}")
    return 0


def cmd_record_finding(args: argparse.Namespace) -> int:
    root = repo_root()
    fid = _validate_id(args.id)
    _, branch, _ = _git("branch", "--show-current", cwd=root)
    _, head_sha, _ = _git("rev-parse", "--short=12", "HEAD", cwd=root)
    payload = {
        "schema": "nanolm.coordination.finding.v1",
        "id": fid,
        "focus": args.focus,
        "findings": list(args.finding or []),
        "branch": args.branch or branch or "(detached)",
        "head_sha": head_sha,
        "recorded_utc": _utc_now(),
    }
    out = root / "coordination" / "findings" / f"{fid}.json"
    atomic_write_json(out, payload)
    print(f"recorded {out.relative_to(root)}")
    return 0


def compile_and_check(
    root: Path,
    check_mode: bool = False,
    out_experiments: Path | None = None,
    out_ledger: Path | None = None,
) -> tuple[bool, list[str], dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    if check_mode:
        for legacy in ("NANOLM_CROSS_WORKTREE_LEDGER.json", "NANOLM_EXPERIMENT_REGISTRY.json"):
            p = root / legacy
            if p.exists():
                errors.append(
                    f"monolithic shared file present in repo root: {legacy} (use coordination/experiments/*.json or coordination/findings/*.json)"
                )
    exp_dir = root / "coordination" / "experiments"
    find_dir = root / "coordination" / "findings"
    experiments: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    total_paid = 0.0

    if exp_dir.is_dir():
        for p in sorted(exp_dir.glob("*.json")):
            data = read_json(p)
            if not isinstance(data, dict):
                errors.append(f"invalid JSON in {p.relative_to(root)}")
                continue
            for req in ("id", "status", "verdict", "paid_usd"):
                if req not in data:
                    errors.append(f"{p.relative_to(root)} missing required field {req!r}")
            if data.get("id") != p.stem:
                errors.append(f"{p.relative_to(root)} id {data.get('id')!r} != filename stem {p.stem!r}")
            try:
                paid = float(data.get("paid_usd", 0.0))
                if paid < 0:
                    errors.append(f"{p.relative_to(root)} paid_usd < 0")
                else:
                    total_paid += paid
            except (TypeError, ValueError):
                errors.append(f"{p.relative_to(root)} invalid paid_usd")
            experiments.append(data)

    if find_dir.is_dir():
        for p in sorted(find_dir.glob("*.json")):
            data = read_json(p)
            if not isinstance(data, dict):
                errors.append(f"invalid JSON in {p.relative_to(root)}")
                continue
            for req in ("id", "focus"):
                if req not in data:
                    errors.append(f"{p.relative_to(root)} missing required field {req!r}")
            if data.get("id") != p.stem:
                errors.append(f"{p.relative_to(root)} id {data.get('id')!r} != filename stem {p.stem!r}")
            findings.append(data)

    compiled_exp = {
        "schema": "nanolm.compiled_experiment_registry.v1",
        "total_entries": len(experiments),
        "total_paid_usd": round(total_paid, 4),
        "experiments": experiments,
    }
    compiled_ledger = {
        "schema": "nanolm.compiled_findings_ledger.v1",
        "total_entries": len(findings),
        "findings": findings,
    }
    if not errors:
        if out_experiments is not None:
            atomic_write_json(out_experiments, compiled_exp)
        if out_ledger is not None:
            atomic_write_json(out_ledger, compiled_ledger)
    return len(errors) == 0, errors, compiled_exp, compiled_ledger


def cmd_compile_registry(args: argparse.Namespace) -> int:
    root = Path(args.repo_root).resolve() if args.repo_root else repo_root()
    out_exp = Path(args.out_experiments).resolve() if args.out_experiments else None
    out_led = Path(args.out_ledger).resolve() if args.out_ledger else None
    ok, errors, exp_doc, led_doc = compile_and_check(
        root, check_mode=args.check, out_experiments=out_exp, out_ledger=out_led
    )
    if not ok:
        for e in errors:
            print(e, file=sys.stderr)
        return 1
    print(
        f"REGISTRY_SHARDS_OK experiments={exp_doc['total_entries']} findings={led_doc['total_entries']}"
    )
    return 0


def _probe_lock_held(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True
    finally:
        os.close(fd)


def cmd_board(args: argparse.Namespace) -> int:
    root = repo_root()
    cdir = coord_dir(root)
    now = time.time()
    locks = []
    for res in VALID_RESOURCES:
        lp = cdir / "locks" / f"{res}.lock"
        hp = cdir / "locks" / f"{res}.holder.json"
        if _probe_lock_held(lp):
            locks.append(read_json(hp, default={"resource": res}))
    leases = []
    for key, lease in _load_leases(cdir).items():
        if lease_is_live(lease, now):
            leases.append({"lane_key": key, **lease})
    sessions = []
    for p in sorted((cdir / "sessions").glob("*.json")):
        s = read_json(p)
        if isinstance(s, dict):
            sessions.append(s)
    _, _, exp_doc, led_doc = compile_and_check(root, check_mode=False)
    budget = read_json(cdir / "budget.json", default={}) if (cdir / "budget.json").is_file() else {}
    board = {
        "coord_dir": str(cdir),
        "active_locks": locks,
        "live_leases": leases,
        "sessions": sessions,
        "shards": {
            "experiments": exp_doc["total_entries"],
            "findings": led_doc["total_entries"],
        },
        "budget": budget,
    }
    if args.json:
        print(json.dumps(board, indent=2, sort_keys=True))
    else:
        print(
            f"[board] dir={cdir} locks={len(locks)} live_leases={len(leases)} "
            f"sessions={len(sessions)} shards(exp={exp_doc['total_entries']}, find={led_doc['total_entries']})"
        )
        for lk in locks:
            print(f"  LOCK    {lk.get('resource','?'):14} agent={lk.get('agent','?')} reason={lk.get('reason','')}")
        for l in leases:
            print(f"  LEASE   {l.get('agent','?'):14} {l.get('branch',''):34} paths={l.get('paths',[])}")
        for s in sessions:
            print(f"  SESSION {s.get('agent','?'):14} [{s.get('status','')}] {s.get('branch',''):30} {s.get('scope','')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Cross-worktree coordination engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_claim = sub.add_parser("claim")
    p_claim.add_argument("paths", nargs="*")
    p_claim.add_argument("--agent", default=os.environ.get("AGENT_NAME", "unknown"))
    p_claim.add_argument("--task", default="")
    p_claim.add_argument("--ttl-sec", type=int, default=DEFAULT_TTL_SEC)
    p_claim.add_argument("--bind-pid", type=int, default=None)
    p_claim.add_argument("--force", action="store_true")
    p_claim.add_argument("--paths", dest="paths_flag", nargs="*", default=[])
    p_claim.set_defaults(func=cmd_claim)

    p_hb = sub.add_parser("heartbeat")
    p_hb.add_argument("--agent", default=os.environ.get("AGENT_NAME", "unknown"))
    p_hb.set_defaults(func=cmd_heartbeat)

    p_rel = sub.add_parser("release")
    p_rel.set_defaults(func=cmd_release)

    p_gc = sub.add_parser("gc")
    p_gc.set_defaults(func=cmd_gc)

    p_chk = sub.add_parser("check")
    p_chk.add_argument("paths", nargs="*")
    p_chk.add_argument("--stdin-nul", action="store_true")
    p_chk.set_defaults(func=cmd_check)

    p_st = sub.add_parser("status")
    p_st.add_argument("--json", action="store_true")
    p_st.set_defaults(func=cmd_status)

    p_ann = sub.add_parser("announce")
    p_ann.add_argument("--agent", required=True)
    p_ann.add_argument("--status", required=True)
    p_ann.add_argument("--scope", default="")
    p_ann.add_argument("--findings", default="")
    p_ann.set_defaults(func=cmd_announce)

    p_lr = sub.add_parser("lock-run")
    p_lr.add_argument("--resource", required=True, choices=VALID_RESOURCES)
    p_lr.add_argument("--timeout", type=float, default=1800.0)
    p_lr.add_argument("--agent", default=os.environ.get("AGENT_NAME", "unknown"))
    p_lr.add_argument("--reason", default="")
    p_lr.add_argument("cmd", nargs=argparse.REMAINDER)
    p_lr.set_defaults(func=cmd_lock_run)

    p_re = sub.add_parser("record-experiment")
    p_re.add_argument("--id", required=True)
    p_re.add_argument("--status", required=True)
    p_re.add_argument("--verdict", required=True)
    p_re.add_argument("--gate", default="")
    p_re.add_argument("--title", default="")
    p_re.add_argument("--seeds", type=int, default=1)
    p_re.add_argument("--paid-usd", type=float, default=0.0)
    p_re.add_argument("--artifact", default="")
    p_re.add_argument("--report", default="")
    p_re.add_argument("--summary", default="")
    p_re.set_defaults(func=cmd_record_experiment)

    p_rf = sub.add_parser("record-finding")
    p_rf.add_argument("--id", required=True)
    p_rf.add_argument("--focus", required=True)
    p_rf.add_argument("--finding", action="append", default=[])
    p_rf.add_argument("--branch", default="")
    p_rf.set_defaults(func=cmd_record_finding)

    p_cr = sub.add_parser("compile-registry")
    p_cr.add_argument("--repo-root", default=None)
    p_cr.add_argument("--out-experiments", default=None)
    p_cr.add_argument("--out-ledger", default=None)
    p_cr.add_argument("--check", action="store_true")
    p_cr.set_defaults(func=cmd_compile_registry)

    p_bd = sub.add_parser("board")
    p_bd.add_argument("--json", action="store_true")
    p_bd.set_defaults(func=cmd_board)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
