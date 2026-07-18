# Behavioral failure-mode probe (Stage M Q(M), robust/local precursor to circuit patching).
# When a frozen own-stack anchor FAILS to copy an actual held-out value, WHAT does it output?
#   - SUBSTITUTION: a value that was in the v2 TRAINING set (a memorized/seen value) -> the
#     model copies the wrong, memorized thing (prior-override / memorization-over-abstraction).
#   - OMISSION: "none" (it declines rather than fabricates).
#   - OTHER: anything else (garbled / novel).
# This is the behavioral signature of the memorization->abstraction thesis, measurable now
# on the frozen anchors with the validated scorer — no hooks, no patching, no Kaggle.
import json, os, importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location("ra", os.path.join(os.path.dirname(__file__), "rescore_anchors.py"))
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
FIELDS, RE = ra.FIELDS, ra.RE
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HELD = {"cc": {"toothache", "neck pain", "heartburn"}, "med": {"melatonin", "throat lozenges"}, "alg": {"sulfa drugs"}}
VALFIELDS = ["cc", "med", "alg"]

# seen (v2 TRAINING) value sets — exec the v2 generator prefix to get them verbatim
v2 = open(os.path.join(REPO, "scribe", "build_scribe_data_v2.py")).read()
ns = {}; exec(compile(v2.split('tok = Tokenizer.from_file')[0].replace("from tokenizers import Tokenizer", ""),
                       "v2", "exec"), ns)
SEEN = {"cc": {c[1] for c in ns["CC_TRAIN"]} | {c[1] for c in ns["CC_COMP"]},
        "med": set(ns["MED_TRAIN"]), "alg": set(ns["ALG_TRAIN"])}
print("seen-set sizes:", {f: len(SEEN[f]) for f in VALFIELDS}, flush=True)

def classify_failures(m, items):
    # per field: counts of {sub (memorized seen value), omit ("none"), other} among held-value MISSES
    cat = {f: {"sub": 0, "omit": 0, "other": 0, "miss_total": 0, "held_total": 0} for f in VALFIELDS}
    for it in items:
        out = ra.generate(m, ra.prompt_ids(it["convo"][0]["content"]))
        text = ra.tok.decode(out[len(ra.prompt_ids(it["convo"][0]["content"])):]).strip()
        mm = RE.match(text)
        if not mm: continue
        pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        for f in VALFIELDS:
            t = it["tuple"][f]
            if t not in HELD[f]: continue          # only ACTUAL held-out values
            cat[f]["held_total"] += 1
            if pred[f] == t: continue              # correct copy
            cat[f]["miss_total"] += 1
            if pred[f] in SEEN[f]:  cat[f]["sub"]  += 1
            elif pred[f] == "none": cat[f]["omit"] += 1
            else:                    cat[f]["other"] += 1
    return cat

def run(tag):
    m, _ = ra.load(tag)
    agg = {f: {"sub": 0, "omit": 0, "other": 0, "miss_total": 0, "held_total": 0} for f in VALFIELDS}
    for k in range(5):
        c = classify_failures(m, ra.fresh[k])
        for f in VALFIELDS:
            for key in agg[f]: agg[f][key] += c[f][key]
    print(f"\n=== {tag}: failure mode on ACTUAL held-out values (pooled m0-m4) ===")
    tot = {"sub": 0, "omit": 0, "other": 0, "miss_total": 0}
    for f in VALFIELDS:
        a = agg[f]; mt = max(1, a["miss_total"])
        print(f"  {f:4s}  held {a['held_total']:3d}  misses {a['miss_total']:3d}  ->  "
              f"substitution {a['sub']}/{a['miss_total']}={a['sub']/mt:.0%}  "
              f"omission {a['omit']/mt:.0%}  other {a['other']/mt:.0%}")
        for key in tot: tot[key] += a[key]
    mt = max(1, tot["miss_total"])
    print(f"  ALL   misses {tot['miss_total']}  ->  SUBSTITUTION {tot['sub']/mt:.0%}  "
          f"omission {tot['omit']/mt:.0%}  other {tot['other']/mt:.0%}")
    return {"tag": tag, "per_field": agg, "pooled": tot,
            "substitution_rate": tot["sub"] / mt, "omission_rate": tot["omit"] / mt, "other_rate": tot["other"] / mt}

if __name__ == "__main__":
    res = {t: run(t) for t in ("nano", "scale")}
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "results_failure_mode_anchors.json"), "w"), indent=1)
    print("\n-> results_failure_mode_anchors.json")
