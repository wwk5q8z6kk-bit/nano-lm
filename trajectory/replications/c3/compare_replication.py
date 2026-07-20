# C-3 REPLICATION comparison (owner step 8): compare the venue-robustness replication
# against the FROZEN primary using the SAME frozen scorer (recompute_c3's functions) on
# both sides. Reports side-by-side; NEVER merges into a new primary estimate.
# Isolation is by DIRECTORY: replication raw logs live in <pod_dir>/ with canonical
# filenames, so the frozen scorer runs byte-identically on primary vs replication.
# Usage: python3 compare_replication.py <pod_dir>   (dir with the replication's
#        outputs_c3_seed{0,1,2}.jsonl); primary is read from trajectory/.
import json, os, sys, collections

HERE = os.path.dirname(os.path.abspath(__file__))
TRAJ = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, TRAJ)
import recompute_c3 as rc

def analyze(d, tag):
    recs = rc.load(d)
    rt = rc.recall_by_type(recs)
    cell = [t for t in rc.LABEL if rc.MAN[t].get("T") in ("T-avail", "T-sep")
            and rc.MAN[t]["B"] in ("B-sub", "B-space") and rc.MAN[t]["L"] in ("short", "long")]
    unstable = set(t for t in cell if len(set(rc.majority_flip(rt, t)[1])) > 1)
    dT = rc.contrast(rt, cell, "T", "T-avail", "T-sep", unstable)
    dB = rc.contrast(rt, cell, "B", "B-sub", "B-space", unstable)
    dL = rc.contrast(rt, cell, "L", "short", "long", unstable)
    tfull = [t for t in rc.LABEL if rc.MAN[t].get("T") == "T-full"]
    tf, _ = rc.cell_rate(tfull, rt, lambda t: True) if tfull else (None, 0)
    err = collections.Counter(r.get("error_class") for r in recs
                              if r.get("kind") == "cell" and not r.get("hit"))
    flip = {t: rc.majority_flip(rt, t)[0] for t in cell}
    return {"tag": tag, "n_records": len(recs), "dT": dT, "dB": dB, "dL": dL,
            "vT": rc.verdict(dT), "vB": rc.verdict(dB), "vL": rc.verdict(dL),
            "tfull_pct": tf, "n_unstable": len(unstable), "unstable": unstable,
            "flip": flip, "err": dict(err), "cell": cell}

def main():
    if len(sys.argv) < 2:
        print("usage: compare_replication.py <replication_pod_dir>"); return
    rep_dir = sys.argv[1]
    P = analyze(TRAJ, "PRIMARY")
    R = analyze(rep_dir, "REPLICATION")
    print(f"{'metric':28s} {'PRIMARY':>16s} {'REPLICATION':>16s} {'Δ/agree':>12s}")
    for k, lbl in (("dT", "H-transition Δ"), ("dB", "H-boundary Δ"), ("dL", "H-length Δ")):
        pv, rv = P[k], R[k]
        print(f"  {lbl:26s} {pv:>+16.1f} {rv:>+16.1f} {rv-pv:>+12.1f}")
    for k, lbl in (("vT", "H-transition verdict"), ("vB", "H-boundary verdict"), ("vL", "H-length verdict")):
        agree = "MATCH" if P[k] == R[k] else "*** DIFFER ***"
        print(f"  {lbl:26s} {P[k]:>16s} {R[k]:>16s} {agree:>12s}")
    print(f"  {'T-full control %':26s} {P['tfull_pct']:>16.0f} {R['tfull_pct']:>16.0f} "
          f"{'':>12s}")
    print(f"  {'seed-unstable cell types':26s} {P['n_unstable']:>16d} {R['n_unstable']:>16d}")
    # exact types whose flip state changed between primary and replication
    changed = sorted(t for t in P["cell"] if P["flip"][t] != R["flip"].get(t))
    print(f"\n  flip-state changed types ({len(changed)}/{len(P['cell'])}): {changed}")
    print(f"  primary error-class dist:     {P['err']}")
    print(f"  replication error-class dist: {R['err']}")
    verdict_match = all(P[v] == R[v] for v in ("vT", "vB", "vL"))
    print(f"\n  OVERALL: mechanical verdicts {'REPRODUCE' if verdict_match else 'DISCREPANCY'} "
          f"(primary remains primary; this is a robustness check, NOT a new estimate)")
    out = {"primary": {k: P[k] for k in ("dT","dB","dL","vT","vB","vL","tfull_pct","n_unstable","err")},
           "replication": {k: R[k] for k in ("dT","dB","dL","vT","vB","vL","tfull_pct","n_unstable","err")},
           "flip_changed_types": changed, "verdicts_reproduce": verdict_match,
           "note": "compare-without-merge; primary result at 0359c22 is unchanged"}
    json.dump(out, open(os.path.join(rep_dir, "replication_comparison.json"), "w"), indent=1)
    print(f"  -> {rep_dir}/replication_comparison.json")

if __name__ == "__main__":
    main()
