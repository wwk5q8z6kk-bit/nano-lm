#!/usr/bin/env python3
"""Cross-worktree lease + file-claim registry. Stdlib only, race-safe (flock).

Leases live in `$(git rev-parse --git-common-dir)/agent-leases/`, which every
worktree of the repo shares automatically and which is never committed, so
there are no merge conflicts and no shared tracked file.

  agent_lease.py claim --agent NAME --task "text" [--paths a b/ c.json]
  agent_lease.py heartbeat | release | status | gc
  agent_lease.py check PATH...      # exit 1 if a PATH is claimed by another live lease

Exit 2 on claim conflict. A lease is stale after TTL hours without heartbeat.
"""
import argparse, fcntl, json, os, subprocess, sys, time

TTL_H = float(os.environ.get("AGENT_LEASE_TTL_H", "6"))


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


ROOT = sh("git", "rev-parse", "--show-toplevel")
COMMON = os.path.realpath(os.path.join(ROOT, sh("git", "-C", ROOT, "rev-parse", "--git-common-dir"))
                          if not os.path.isabs(sh("git", "-C", ROOT, "rev-parse", "--git-common-dir"))
                          else sh("git", "-C", ROOT, "rev-parse", "--git-common-dir"))
D = os.path.join(COMMON, "agent-leases")
os.makedirs(D, exist_ok=True)
ME = os.path.join(D, os.path.basename(ROOT) + ".json")


class Lock:
    def __enter__(self):
        self.f = open(os.path.join(D, ".lock"), "w"); fcntl.flock(self.f, fcntl.LOCK_EX)
    def __exit__(self, *e):
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()


def load_all():
    out = {}
    for n in os.listdir(D):
        if n.endswith(".json"):
            try: out[n] = json.load(open(os.path.join(D, n)))
            except Exception: pass
    return out


def live(l): return time.time() - l["heartbeat"] < TTL_H * 3600


def overlap(p, q):
    p, q = p.rstrip("/"), q.rstrip("/")
    return p == q or p.startswith(q + "/") or q.startswith(p + "/")


def conflicts(paths, skip=None):
    c = []
    for n, l in load_all().items():
        if n == skip or not live(l): continue
        for p in paths:
            for q in l["paths"]:
                if overlap(p, q): c.append((l["agent"], l["worktree"], q))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["claim", "heartbeat", "release", "status", "gc", "check"])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--agent", default=os.environ.get("AGENT_NAME", "unknown"))
    ap.add_argument("--task", default="")
    ap.add_argument("--paths", dest="paths2", nargs="*", default=[])
    a = ap.parse_args()
    paths = a.paths + a.paths2
    me = os.path.basename(ROOT) + ".json"
    with Lock():
        if a.cmd == "claim":
            c = conflicts(paths, skip=me)
            if c:
                for x in c: print("CONFLICT with", x, file=sys.stderr)
                sys.exit(2)
            base = sh("git", "merge-base", "HEAD", "origin/master")
            json.dump(dict(agent=a.agent, task=a.task, worktree=ROOT, branch=sh("git", "branch", "--show-current"),
                           base=base[:9], paths=paths, started=time.time(), heartbeat=time.time()), open(ME, "w"))
            print("claimed", ROOT)
        elif a.cmd == "heartbeat":
            l = json.load(open(ME)); l["heartbeat"] = time.time(); json.dump(l, open(ME, "w"))
        elif a.cmd == "release":
            if os.path.exists(ME): os.remove(ME)
        elif a.cmd == "gc":
            for n, l in load_all().items():
                if not live(l) or not os.path.isdir(l["worktree"]):
                    os.remove(os.path.join(D, n)); print("reaped stale lease", n)
        elif a.cmd == "check":
            c = conflicts(paths, skip=me)
            for x in c: print("CLAIMED by", x, file=sys.stderr)
            sys.exit(1 if c else 0)
        else:
            for n, l in load_all().items():
                age = (time.time() - l["heartbeat"]) / 3600
                print(f"{'LIVE ' if live(l) else 'STALE'} {l['agent']:16} {l['branch']:36} {age:5.1f}h  {l['paths']}  {l['task']}")


if __name__ == "__main__":
    main()
