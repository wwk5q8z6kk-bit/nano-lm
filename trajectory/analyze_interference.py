# C-1b post-hoc failure-mode classification (mechanical, from the mandatory per-item
# logs). Classifies every held-item miss into modes fixed here BEFORE aggregate
# interpretation; the mode taxonomy is descriptive (observed shapes), the counts are
# the deliverable. Output: results_interference_modes.json + printed table.
import json, collections, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from slot_diversity_pools import ALG_TRAIN_80

def mode(truth, pred):
    if pred is None: return "unparsed"
    p, t = pred.strip(), truth.strip()
    if p == t: return "hit"
    if p in ("none", "None", ""): return "omission"
    tw, pw = t.split(), p.split()
    # format doubling: correct value present but field text repeats/extends with '|'
    if "|" in p and p.split("|")[0].strip() in (t, t + " "): return "format_double_correct"
    if "|" in p: return "format_double_other"
    # word-boundary truncation: pred = strict prefix subsequence of truth's words
    if pw == tw[:len(pw)] and len(pw) < len(tw): return "tail_truncation"
    # morphological near-miss: same word count, each word equal or ±trailing 's'
    if len(pw) == len(tw) and all(a == b or a == b + "s" or b == a + "s"
                                  for a, b in zip(pw, tw)): return "morph_near_miss"
    if p in ALG_TRAIN_80: return "substitution_trained"
    # garble: shares a word or a >=3-char prefix with truth
    if (set(pw) & set(tw)) or p[:3] == t[:3]: return "garble_of_truth"
    return "substitution_other"

recs = [json.loads(l) for f in ("outputs_if_seed0.jsonl", "outputs_if_seed1.jsonl")
        for l in open(os.path.join(HERE, f))]
per_class = collections.defaultdict(collections.Counter)
per_type = collections.defaultdict(collections.Counter)
for r in recs:
    m = mode(r["type"], r["pred_alg"]) if not r["hit"] else "hit"
    per_class[r["class"]][m] += 1
    per_type[r["type"]][m] += 1

MODES = ["hit", "tail_truncation", "morph_near_miss", "format_double_correct",
         "format_double_other", "omission", "substitution_trained",
         "substitution_other", "garble_of_truth", "unparsed"]
print(f"{'class':11s} " + " ".join(f"{m[:9]:>9s}" for m in MODES))
tot = collections.Counter()
for c in sorted(per_class):
    row = per_class[c]; tot.update(row)
    print(f"{c:11s} " + " ".join(f"{row.get(m,0):>9d}" for m in MODES))
print(f"{'TOTAL':11s} " + " ".join(f"{tot.get(m,0):>9d}" for m in MODES))
n_miss = sum(v for k, v in tot.items() if k != "hit")
print(f"\nmisses: {n_miss}; tail_truncation share of misses: "
      f"{tot['tail_truncation']/max(1,n_miss):.0%}; "
      f"value-correct-but-format ({tot['format_double_correct']}) are instrument-"
      f"level, not copy failures; substitution_trained: {tot['substitution_trained']}")
json.dump({"per_class": {c: dict(per_class[c]) for c in per_class},
           "per_type": {t: dict(per_type[t]) for t in per_type},
           "modes": MODES},
          open(os.path.join(HERE, "results_interference_modes.json"), "w"), indent=1)
print("-> results_interference_modes.json")
