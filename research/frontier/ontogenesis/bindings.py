"""BINDINGS: a stream of key->value facts with updates and deletions, queried by chains.

Events (one position each, same stream for every arm):
  SET k v   bind k to v (overwrites any earlier binding of k)
  DEL k     remove the binding of k
  QRY k     start a query at k; if it has h hops it is followed by h-1 HOP events
  HOP       one more hop of thinking room; the answer f^h(k) is scored at the last event
Answers are symbols, or NULL when the chain reaches an unbound key.

Two size axes are controlled independently:
  M  live bindings at the end (facts that must be kept)
  N  total statements (history, including superseded and deleted facts)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

PAD, SET, DEL, QRY, HOP = 0, 1, 2, 3, 4
NULL = 0  # symbol id 0 is the NULL answer; real symbols are 1..S


@dataclass
class Spec:
    symbols: int = 256        # S
    live: int = 32            # M: live bindings at the end
    churn: float = 0.0        # extra superseded SETs per live binding (N grows, M fixed)
    deleted: float = 0.1      # fraction of extra keys that are set and then deleted
    groups: int = 2           # statement block + query group, repeated
    queries: int = 16         # queries per group
    hops: tuple = (1,)        # hop counts sampled per query
    path: int = 8             # edges per chain; must be >= max hops for non-NULL answers
    null_target: float = 0.2  # desired fraction of NULL answers


def _statements(spec: Spec, rng):
    """Returns the per-key ordered statement lists and the final map."""
    S, M = spec.symbols, spec.live
    n_del = int(round(spec.deleted * M))
    need = M + M // spec.path + 2 + n_del
    assert need <= S, "not enough symbols"
    perm = rng.permutation(S)[:need] + 1
    chain, extra = perm[: need - n_del], perm[need - n_del:]
    edges = []
    i = 0
    while len(edges) < M:                       # injective paths: answers are not concentrated on cycles
        seg = chain[i:i + spec.path + 1]
        edges += [(int(seg[j]), int(seg[j + 1])) for j in range(len(seg) - 1)]
        i += spec.path + 1
    edges = edges[:M]
    per_key = []
    for k, v in edges:
        n_old = rng.poisson(spec.churn)
        ops = [(SET, k, int(rng.integers(1, S + 1))) for _ in range(n_old)]
        if n_old and rng.random() < 0.3:        # some histories pass through a deletion
            ops.insert(int(rng.integers(1, n_old + 1)), (DEL, k, NULL))
        per_key.append(ops + [(SET, k, v)])
    for k in extra:                             # keys whose final state is deleted
        per_key.append([(SET, int(k), int(rng.integers(1, S + 1))), (DEL, int(k), NULL)])
    return per_key


def make(spec: Spec, rng):
    """One sequence. Returns structured arrays (for particle/table arms) and a flat event stream."""
    per_key = _statements(spec, rng)
    slots = np.concatenate([np.full(len(o), i) for i, o in enumerate(per_key)])
    rng.shuffle(slots)                          # random interleave, per-key order preserved
    cursor = [0] * len(per_key)
    stream = []
    for s in slots:
        stream.append(per_key[s][cursor[s]]); cursor[s] += 1
    N = len(stream)
    bounds = np.linspace(0, N, spec.groups + 1).astype(int)
    f, blocks, qgroups = {}, [], []
    for g in range(spec.groups):
        block = stream[bounds[g]:bounds[g + 1]]
        for op, k, v in block:
            if op == SET: f[k] = v
            else: f.pop(k, None)
        keys = np.array(sorted(f)) if f else np.array([1])
        qs = []
        for _ in range(spec.queries):
            h = int(rng.choice(spec.hops))
            want_null = rng.random() < spec.null_target
            for _try in range(20):
                k = int(rng.integers(1, spec.symbols + 1)) if want_null else int(rng.choice(keys))
                x = k
                for _ in range(h):
                    x = f.get(x, NULL)
                    if x == NULL: break
                if (x == NULL) == want_null: break
            qs.append((k, h, x))
        blocks.append(block); qgroups.append(qs)
    ev = []                                     # flat stream: op, a, b, target, scored
    for block, qs in zip(blocks, qgroups):
        ev += [(op, k, v, 0, 0) for op, k, v in block]
        for k, h, ans in qs:
            ev.append((QRY, k, 0, ans if h == 1 else 0, int(h == 1)))
            ev += [(HOP, 0, 0, ans if j == h - 2 else 0, int(j == h - 2)) for j in range(h - 1)]
    return {"blocks": blocks, "queries": qgroups, "events": np.array(ev, dtype=np.int64),
            "N": N, "M": len(f)}


def guessers(spec: Spec, rng, n=200):
    """Accuracy of answers that need no chain reasoning, to read results against."""
    null = one_hop = total = 0
    for _ in range(n):
        d = make(spec, rng)
        f = {}
        for block, qs in zip(d["blocks"], d["queries"]):
            for op, k, v in block:
                if op == SET: f[k] = v
                else: f.pop(k, None)
            for k, h, ans in qs:
                total += 1; null += ans == NULL; one_hop += f.get(k, NULL) == ans
    return {"always_null": null / total, "one_hop_only": one_hop / total}


if __name__ == "__main__":
    r = np.random.default_rng(0)
    for sp in (Spec(), Spec(live=64, churn=3.0, hops=(1, 2, 4), symbols=512)):
        d = make(sp, r)
        print(sp, "N", d["N"], "M", d["M"], "events", d["events"].shape, guessers(sp, r, 50))
