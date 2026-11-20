"""Numerical sanity checks for three 'all prefixes at once' claims. No model, no training."""
import numpy as np
rng = np.random.default_rng(0)

def check_bottleneck(n=24, E=70, trials=30):
    """Directed edges arrive at times 1..E. Claim: j reachable from i using edges with time<=t
    iff D[i,j] <= t, where D is the closure of edge times in the (min, max) semiring."""
    for _ in range(trials):
        src, dst = rng.integers(0, n, E), rng.integers(0, n, E)
        T = np.full((n, n), np.inf)
        for t, (a, b) in enumerate(zip(src, dst), 1): T[a, b] = min(T[a, b], t)
        D = T.copy(); np.fill_diagonal(D, 0)
        for _ in range(int(np.ceil(np.log2(n))) + 1):          # repeated squaring
            D = np.minimum(D, np.min(np.maximum(D[:, :, None], D[None, :, :]), axis=1))
        for t in range(0, E + 1, 7):                            # brute force per prefix
            A = np.eye(n, dtype=bool)
            for a, b in zip(src[:t], dst[:t]): A[a, b] = True
            R = A.copy()
            for _ in range(n): R = R | ((R.astype(int) @ A.astype(int)) > 0)
            assert np.array_equal(R, D <= t), "bottleneck closure mismatch"
    return True

def check_last_write(N=2000, C=37, trials=30):
    """Claim: 'most recent earlier position with the same key' for every position is one stable sort + shift."""
    for _ in range(trials):
        key = rng.integers(0, C, N)
        order = np.lexsort((np.arange(N), key))                 # by key, then position
        prev = np.full(N, -1)
        same = key[order][1:] == key[order][:-1]
        prev[order[1:][same]] = order[:-1][same]
        last, ref = {}, np.full(N, -1)
        for t, k in enumerate(key): ref[t] = last.get(k, -1); last[k] = t
        assert np.array_equal(prev, ref), "last-write mismatch"
    return True

def check_merge_tree(n=40, E=60, trials=30):
    """Undirected unify events at times 1..E. Claim: a,b in the same class at time t iff the
    symmetric (min,max) closure M[a,b] <= t (single-linkage / H0 persistence of the time filtration)."""
    for _ in range(trials):
        a, b = rng.integers(0, n, E), rng.integers(0, n, E)
        T = np.full((n, n), np.inf)
        for t, (x, y) in enumerate(zip(a, b), 1): T[x, y] = T[y, x] = min(T[x, y], t)
        M = T.copy(); np.fill_diagonal(M, 0)
        for _ in range(int(np.ceil(np.log2(n))) + 1):
            M = np.minimum(M, np.min(np.maximum(M[:, :, None], M[None, :, :]), axis=1))
        parent = list(range(n))
        def find(x):
            while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for t, (x, y) in enumerate(zip(a, b), 1):
            parent[find(x)] = find(y)
            if t % 9 == 0:
                r = np.array([find(i) for i in range(n)])
                assert np.array_equal(r[:, None] == r[None, :], M <= t), "merge tree mismatch"
    return True

print("bottleneck closure == per-prefix reachability:", check_bottleneck())
print("sort+shift == sequential last-write-wins:", check_last_write())
print("symmetric closure == union-find at every prefix:", check_merge_tree())
