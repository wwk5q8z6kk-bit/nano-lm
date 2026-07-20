# C-3 INDEPENDENT recompute (evidence-hierarchy level 2). Reads ONLY the raw per-item
# logs outputs_c3_seed{0,1,2}.jsonl + the frozen pool manifest; imports NO kernel code,
# so it is a genuine cross-check of run_c3_10m.py's own results_c3_10m.json (level 3).
# Adds two analyses the committed kernel does NOT do, both demanded by the run's own
# balance report (junction-frequency SMD up to 2.42 in B-space) and Handoff-2 discipline:
#   (1) frequency-conditioned T contrast — does flip(T-avail)-flip(T-sep) survive when
#       the junction-frequency confound is removed by median-split matching?
#   (2) type-level bootstrap CI on each frozen contrast (resample TYPES, not items).
# Applies the FROZEN C-3 decision rules verbatim; never moves a threshold.
# Usage: python3 trajectory/recompute_c3.py [dir_with_outputs]   (default: cwd then trajectory/)
import json, os, sys, collections, math

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = json.load(open(os.path.join(HERE, "c3_pools.json")))
MAN = POOL["manifest"]                       # per-type: T,B,L,junction_bigram_count,...
LABEL = {t: {"T": MAN[t]["T"], "B": MAN[t]["B"], "L": MAN[t]["L"],
             "jbig": MAN[t]["junction_bigram_count"]} for t in MAN}

def find_logs():
    for d in (sys.argv[1] if len(sys.argv) > 1 else None, os.getcwd(), HERE):
        if d and all(os.path.exists(os.path.join(d, f"outputs_c3_seed{s}.jsonl")) for s in range(3)):
            return d
    return None

def load(d):
    recs = []
    for s in range(3):
        recs += [json.loads(l) for l in open(os.path.join(d, f"outputs_c3_seed{s}.jsonl"))]
    return recs

# ---- deterministic RNG (no Math.random / time; fixed LCG for the bootstrap) ----
class LCG:
    def __init__(self, seed): self.x = seed & 0xFFFFFFFF
    def rand(self): self.x = (1103515245 * self.x + 12345) & 0x7FFFFFFF; return self.x / 0x7FFFFFFF
    def choice(self, seq): return seq[int(self.rand() * len(seq)) % len(seq)]

def recall_by_type(recs):
    """per (seed,type) recall from raw hits; independent of kernel aggregation."""
    acc = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        if r.get("kind") not in ("cell", "tfull"):  # cell types + T-full control only
            continue
        acc[(r["seed"], r["type"])][0] += int(r["hit"]); acc[(r["seed"], r["type"])][1] += 1
    rt = collections.defaultdict(dict)
    for (seed, t), (h, n) in acc.items():
        rt[t][seed] = h / n if n else 0.0
    return rt

def majority_flip(rt, t):
    votes = [rt[t].get(s, 0.0) >= 0.5 for s in range(3)]
    return sum(votes) >= 2, votes

def cell_rate(types, rt, pred, unstable=frozenset()):
    # PREREG (line 92-93): "3-way-unstable types are reported separately and
    # excluded from contrasts." The kernel's own cell_rate() implements this
    # ("t not in unstable"); this harness must match or it silently computes a
    # different quantity than the frozen rule specifies.
    sel = [t for t in types if pred(t) and t not in unstable]
    if not sel: return None, 0
    return sum(majority_flip(rt, t)[0] for t in sel) / len(sel) * 100, len(sel)

def verdict(delta):
    if delta is None: return "UNRESOLVED (empty cell)"
    return "SUPPORTED" if delta >= 40 else ("REFUTED" if delta <= 15 else "UNRESOLVED")

def contrast(rt, types, split_key, a, b, unstable=frozenset()):
    """mean over the other-two-factor strata of rate(split=a) - rate(split=b)."""
    others = [k for k in ("T", "B", "L") if k != split_key]
    diffs = []
    strata = set((LABEL[t][others[0]], LABEL[t][others[1]]) for t in types)
    for s0, s1 in strata:
        ra, _ = cell_rate(types, rt, lambda t: LABEL[t][split_key] == a and LABEL[t][others[0]] == s0 and LABEL[t][others[1]] == s1, unstable)
        rb, _ = cell_rate(types, rt, lambda t: LABEL[t][split_key] == b and LABEL[t][others[0]] == s0 and LABEL[t][others[1]] == s1, unstable)
        if ra is not None and rb is not None: diffs.append(ra - rb)
    return sum(diffs) / len(diffs) if diffs else None

def main():
    d = find_logs()
    if not d:
        print("outputs_c3_seed{0,1,2}.jsonl not found (run not retrieved yet). "
              "This harness is ready; re-run once the raw logs are local.")
        return
    recs = load(d)
    cap = sum(1 for r in recs if r.get("cap_confound"))
    print(f"loaded {len(recs)} records from {d}; cap-confound (excluded from truncation metric): {cap}")
    rt = recall_by_type(recs)
    cell_types = [t for t in LABEL if MAN[t].get("T") in ("T-avail", "T-sep")
                  and MAN[t]["B"] in ("B-sub", "B-space") and MAN[t]["L"] in ("short", "long")]

    # seed stability computed BEFORE contrasts -- the frozen rule excludes
    # 3-way-unstable types from contrasts, it does not merely report them
    unstable = set(t for t in cell_types if len(set(majority_flip(rt, t)[1])) > 1)

    dT = contrast(rt, cell_types, "T", "T-avail", "T-sep", unstable)
    dB = contrast(rt, cell_types, "B", "B-sub", "B-space", unstable)
    dL = contrast(rt, cell_types, "L", "short", "long", unstable)
    print("\n== FROZEN CONTRASTS (independent recompute) ==")
    for name, dv in (("H-transition (T-avail - T-sep)", dT),
                     ("H-boundary  (B-sub - B-space)", dB),
                     ("H-length    (short - long)", dL)):
        print(f"  {name:34s}: {dv:+.1f} pts -> {verdict(dv)}" if dv is not None else f"  {name}: empty")

    # T-full control (>=90% or run voided)
    tfull = [t for t in LABEL if MAN[t].get("T") == "T-full"]
    if tfull:
        tf_rate, _ = cell_rate(tfull, rt, lambda t: True)
        print(f"  T-full control: {tf_rate:.0f}%  -> {'PASS' if tf_rate >= 90 else 'VOIDS RUN (instrument failure)'}")

    # (1) frequency analysis. T-avail (count>=20) and T-sep (=0) are frequency-DISJOINT
    # BY DEFINITION of the factor — a median-split match is impossible and would be
    # meaningless. The valid analysis is a DOSE-RESPONSE WITHIN T-avail: junction counts
    # there span 25..425; if flip-rate rises with junction count, the effect is graded on
    # transition frequency (supports the transition reading); if flat, it's the binary
    # seen/unseen distinction. (T-sep's individual tokens ARE frequency-matched to
    # T-avail's per the prereg, so the contrast isolates the JUNCTION, not token rarity.)
    print("\n== FREQUENCY DOSE-RESPONSE WITHIN T-avail (junction count vs per-type recall) ==")
    av_types = [t for t in cell_types if LABEL[t]["T"] == "T-avail"]
    pairs = [(MAN[t]["junction_bigram_count"],
              sum(rt[t].get(s, 0.0) for s in range(3)) / 3) for t in av_types]
    # Spearman rank correlation (deterministic; ties averaged)
    def ranks(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i]); r = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]: j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1): r[order[k]] = avg
            i = j + 1
        return r
    xr, yr = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs); mx, my = sum(xr) / n, sum(yr) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xr, yr))
    den = math.sqrt(sum((a - mx) ** 2 for a in xr) * sum((b - my) ** 2 for b in yr))
    rho = num / den if den else 0.0
    print(f"  n={n} T-avail types; Spearman(junction_count, recall) = {rho:+.2f}  "
          + ("(graded on transition frequency)" if abs(rho) >= 0.4 else "(flat: binary seen/unseen, not dose-graded)"))
    overlap = False  # intrinsic to the factor definition; recorded for the output JSON

    # (2) type-level bootstrap CI (resample TYPES within cells, deterministic LCG).
    # Resamples only stable types -- the frozen rule excludes unstable types from
    # contrasts, so a CI on the contrast must be built from the same population.
    print("\n== TYPE-BOOTSTRAP 90% CI on H-transition (resample types, 2000 draws) ==")
    rng = LCG(20260750)
    by_cell = collections.defaultdict(list)
    for t in cell_types:
        if t in unstable: continue
        by_cell[(LABEL[t]["T"], LABEL[t]["B"], LABEL[t]["L"])].append(t)
    boots = []
    for _ in range(2000):
        resampled = {c: [rng.choice(ts) for _ in ts] for c, ts in by_cell.items()}
        flat = [t for ts in resampled.values() for t in ts]
        # rebuild a flip lookup keyed by resampled multiset
        diffs = []
        for (B, Lb) in set((c[1], c[2]) for c in by_cell):
            av = resampled.get(("T-avail", B, Lb), []); sp = resampled.get(("T-sep", B, Lb), [])
            if not av or not sp: continue
            ra = sum(majority_flip(rt, t)[0] for t in av) / len(av) * 100
            rb = sum(majority_flip(rt, t)[0] for t in sp) / len(sp) * 100
            diffs.append(ra - rb)
        if diffs: boots.append(sum(diffs) / len(diffs))
    boots.sort()
    lo, hi = boots[int(0.05 * len(boots))], boots[int(0.95 * len(boots))]
    print(f"  dT = {dT:+.1f}  90% CI [{lo:+.1f}, {hi:+.1f}]  (n={len(boots)} bootstraps)")

    # seed stability (unstable set computed above, before the contrasts that exclude it)
    print(f"\n== SEED STABILITY == {len(unstable)}/{len(cell_types)} cell types seed-unstable "
          f"({len(unstable)/len(cell_types):.0%})")

    all_ref = all(verdict(x) == "REFUTED" for x in (dT, dB, dL) if x is not None)
    if all_ref and len(unstable) / len(cell_types) >= 0.20:
        print("  -> all three lexical contrasts REFUTED + >=20% seed-unstable "
              "=> H-stochastic SUPPORTED (promote representation/attention probe)")

    out = {"independent_recompute": True, "source_dir": d, "n_records": len(recs),
           "cap_confound_excluded": cap,
           "contrasts": {"H_transition": dT, "H_boundary": dB, "H_length": dL},
           "verdicts": {"H_transition": verdict(dT), "H_boundary": verdict(dB), "H_length": verdict(dL)},
           "T_freq_disjoint_by_design": True, "within_Tavail_dose_rho": rho,
           "bootstrap_dT_90ci": [lo, hi],
           "seed_unstable_frac": len(unstable) / len(cell_types)}
    json.dump(out, open(os.path.join(HERE, "results_c3_recompute.json"), "w"), indent=1)
    print("\n-> results_c3_recompute.json (cross-check against kernel's results_c3_10m.json)")

if __name__ == "__main__":
    main()
