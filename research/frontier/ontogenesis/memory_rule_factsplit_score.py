"""Mechanical reading of PROTOCOL-MEMRULE-FACTSPLIT.md (F1-F3). Committed before any confirmatory result was read.

Also checks, for every rerun, that its registered `late` metrics reproduce the original result file exactly
(results/memory_rule_transport/ for drop and transport, results/memory_rule_recluster/k32-em-* for full re-index);
a mismatch is reported as a failed rerun and that seed is not scored.
usage: memory_rule_factsplit_score.py [--out verdict.json]
"""
import json, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent / "results"
SEEDS = [4101, 4102, 4103, 4104, 4105]
ORIG = {"drop": "memory_rule_transport/drop-s{s}.json", "transport": "memory_rule_transport/transport-s{s}.json",
        "full-k32": "memory_rule_recluster/k32-em-s{s}.json"}
NEW = {"drop": "memory_rule_factsplit/drop-s{s}.json", "transport": "memory_rule_factsplit/transport-s{s}.json",
       "full-k32": "memory_rule_factsplit/full-k32-s{s}.json", "typed-noreindex": "memory_rule_factsplit/typed-noreindex-s{s}.json"}


def main():
    split, bad, missing = {}, [], []
    for arm, pat in NEW.items():
        for s in SEEDS:
            f = HERE / pat.format(s=s)
            if not f.exists() or not f.stat().st_size: missing.append(str(f)); continue
            r = json.load(open(f))
            if arm in ORIG and r["late"] != json.load(open(HERE / ORIG[arm].format(s=s)))["late"]:
                bad.append(f"{arm} s{s}"); continue
            split[(arm, s)] = r["final_split"]
    if missing: raise SystemExit(f"missing {len(missing)} files, e.g. {missing[0]}; nothing is scored")
    ok = [s for s in SEEDS if all((a, s) in split for a in ("drop", "full-k32"))]
    g = lambda a, s, k: split[(a, s)][k]
    f1 = [s for s in ok if g("drop", s, "exception") <= g("full-k32", s, "exception") - 0.20]
    f2 = [s for s in ok if abs(g("drop", s, "regular") - g("full-k32", s, "regular")) <= 0.10]
    f3 = [s for s in ok if g("drop", s, "corrected") <= g("full-k32", s, "corrected")]
    mean = lambda a, k: round(sum(g(a, s, k) for s in SEEDS if (a, s) in split) / max(1, sum((a, s) in split for s in SEEDS)), 4)
    v = {"reproduction_failures": bad,
         "means": {a: {k: mean(a, k) for k in ("regular", "exception", "corrected")} for a in NEW},
         "per_seed": {f"{a} s{s}": split[(a, s)] for (a, s) in sorted(split)},
         "F1_exceptions_drop_le_full_minus_0.20": {"seeds": len(f1), "of": len(ok), "holds": len(f1) >= 4},
         "F2_regular_drop_within_0.10_of_full": {"seeds": len(f2), "of": len(ok), "holds": len(f2) >= 4},
         "F3_corrected_drop_le_full": {"seeds": len(f3), "of": len(ok), "holds": len(f3) >= 4}}
    print(json.dumps(v, indent=1))
    if "--out" in sys.argv: json.dump(v, open(sys.argv[sys.argv.index("--out") + 1], "x"), indent=1)


if __name__ == "__main__":
    main()
