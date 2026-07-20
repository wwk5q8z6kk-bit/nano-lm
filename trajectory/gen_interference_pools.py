# PREREG_token_coverage.md AMENDMENT 2 — interference-class pool construction.
# Deterministic; committed pre-run. Classifies candidates into the 5-class taxonomy
# (I-contain > I-sib > I-xslot > I-template > I-iso, word-level, strip-s for template
# matching) and records coverage vs the D80-arm output-token set + token counts as
# covariates. FALSIFICATION GATE: the classifier must reproduce all six sweep types'
# known classes; candidate hygiene rules enforced mechanically. Output:
# trajectory/interference_pools.json (frozen instrument input).
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slot_diversity_pools import ALG_TRAIN_80, HELD_ALG
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(os.path.join(HERE, "..", "sft", "tokenizer.json"))

# --- world vocabularies (exec'd from the v2 recipe; nothing hand-copied) ---------
V2 = open(os.path.join(HERE, "..", "scribe", "build_scribe_data_v2.py")).read()
_ns = {}
exec(compile(V2.split("def ")[0].replace("from tokenizers import Tokenizer", ""), "v2p", "exec"), _ns)
MED_TRAIN = _ns["MED_TRAIN"]
TPL_LISTS = {k: v for k, v in _ns.items() if isinstance(v, list) and v
             and all(isinstance(x, str) for x in v)
             and (k.startswith("D_") or k.startswith("P_") or k == "DISTRACT")}
STOP = set("""a an the any are is it you your do we i to of for on in at or and no not
have has been that what when why how did does was would should can be so but with over
me my this there here now far only just some most many say tell go got had am if then
yes okay well hello hi about ago around out up off from since where which who quite
pretty nearly almost maybe roughly none nothing anything haven't i'd i'm i've it's
what's there's won't you're""".split())
TMPL_WORDS = set()
for lst in TPL_LISTS.values():
    for t in lst:
        t = re.sub(r"\{[a-z_]+\}", " ", t)
        TMPL_WORDS.update(w for w in re.findall(r"[a-zA-Z']+", t.lower()) if w not in STOP)

TRAIN_VALUES = [v.lower() for v in ALG_TRAIN_80]
TRAIN_WORDS = set(w for v in TRAIN_VALUES for w in v.split())
MED_WORDS = set(w for v in MED_TRAIN for w in v.lower().split())
strip_s = lambda w: w[:-1] if w.endswith("s") and len(w) > 3 else w
TMPL_S = {strip_s(w) for w in TMPL_WORDS}

def classify(value):
    """AMENDMENT 2 precedence: contain > sib > xslot > template > iso."""
    v = value.lower(); words = v.split()
    for t in TRAIN_VALUES:                       # full trained value as contiguous word-subseq
        tw = t.split(); n = len(tw)
        if any(words[i:i + n] == tw for i in range(len(words) - n + 1)):
            return "I-contain", f"contains trained {t!r}"
    shared = set(words) & TRAIN_WORDS
    if shared:
        return "I-sib", f"shares word {sorted(shared)} with trained value"
    for t in (m.lower() for m in MED_TRAIN):     # trained med value (ibuprofen pattern)
        tw = t.split(); n = len(tw)
        if any(words[i:i + n] == tw for i in range(len(words) - n + 1)):
            return "I-xslot", f"is/contains trained med {t!r}"
    tshared = {w for w in words if strip_s(w) in TMPL_S}
    if tshared:
        return "I-template", f"shares template word {sorted(tshared)}"
    return "I-iso", "no lexical relation to trained or template vocabulary"

# --- D80-arm output token set for the coverage covariate (as gen_token_coverage) --
ALG_LINE = 'ALG_TRAIN = ["penicillin", "peanuts", "pollen", "latex", "shellfish"]'
assert ALG_LINE in V2
src = V2.replace(ALG_LINE, "ALG_TRAIN = " + repr(list(ALG_TRAIN_80)))
prefix = src.split("tok = Tokenizer.from_file")[0].replace("from tokenizers import Tokenizer", "")
ns2 = {}; exec(compile(prefix, "v2[d80]", "exec"), ns2)
out_tokens = set()
for c in ns2["convos"]:
    out_tokens.update(tok.encode(c[1]["content"], add_special_tokens=False).ids)

def profile(v):
    ids = tok.encode(v, add_special_tokens=False).ids
    cls, why = classify(v)
    return {"value": v, "class": cls, "why": why, "n_tokens": len(ids),
            "coverage": round(sum(1 for i in ids if i in out_tokens) / len(ids), 4),
            "novel_tokens": [tok.decode([i]) for i in ids if i not in out_tokens]}

# --- falsification gate: the six sweep types must classify per AMENDMENT 2 --------
EXPECT = {"ragweed pollen": "I-contain", "sulfa drugs": "I-template",
          "bee stings": "I-sib", "ibuprofen": "I-xslot",
          "wool": "I-iso", "strawberries": "I-iso"}
gate_ok = True
print("== falsification gate (sweep bridges) ==")
for v, want in EXPECT.items():
    got, why = classify(v)
    ok = got == want; gate_ok &= ok
    print(f"  {'PASS' if ok else 'FAIL'} {v!r}: {got} ({why}) — expected {want}")

# --- candidate pools (frozen here; mechanically validated below) ------------------
CANDIDATES = {
    "I-contain": ["willow pollen", "maple pollen", "hazel pollen", "elm pollen",
                  "goat milk", "rice milk"],
    "I-sib": ["blue dye", "pumpkin seeds", "chia seeds", "spider mites",
              "hornet stings", "deer dander"],
    "I-xslot": ["cough syrup", "antacids", "loratadine", "paracetamol"],
    "I-template": ["food additives", "food coloring", "generic drugs", "pet food"],
    "I-iso": ["camphor", "henna", "iodine", "quinoa", "gum arabic", "rose hips",
              "melon rind", "pine resin"],
}
excl = set(v.lower() for v in ALG_TRAIN_80) | set(v.lower() for v in HELD_ALG) \
       | set(v.lower() for v in MED_TRAIN)
hygiene_ok = True
mods_seen = {}
pools = {}
print("\n== candidate validation ==")
for cls, vals in CANDIDATES.items():
    pools[cls] = []
    for v in vals:
        p = profile(v)
        errs = []
        if p["class"] != cls: errs.append(f"classifies {p['class']}")
        if v.lower() in excl and cls != "I-xslot": errs.append("in a training/held pool")
        for w in v.lower().split():
            if w in mods_seen and mods_seen[w] != cls:
                errs.append(f"word {w!r} reused from {mods_seen[w]}")
        if (set(v.lower().split()) & MED_WORDS) and cls != "I-xslot":
            errs.append("shares word with MED_TRAIN")
        for w in v.lower().split(): mods_seen.setdefault(w, cls)
        if errs:
            hygiene_ok = False
            print(f"  FAIL {cls} {v!r}: {'; '.join(errs)} ({p['why']})")
        else:
            pools[cls].append(p)
            print(f"  ok   {cls:11s} {v!r:20s} cov={p['coverage']:.2f} "
                  f"ntok={p['n_tokens']} novel={p['novel_tokens']}")

sizes_ok = all(len(pools[c]) >= 4 for c in pools)
valid = gate_ok and hygiene_ok and sizes_ok
print(f"\nclass sizes: { {c: len(v) for c, v in pools.items()} }")
print(f"iso coverage span (I-cov covariate): "
      f"{sorted(p['coverage'] for p in pools['I-iso'])}")
json.dump({"amendment": 2, "pools": pools,
           "bridges": {v: profile(v) for v in EXPECT},
           "d80_output_token_count": len(out_tokens),
           "valid": valid,
           "classifier": "contain>sib>xslot>template>iso; word-level; strip-s for template"},
          open(os.path.join(HERE, "interference_pools.json"), "w"), indent=1)
print(f"-> interference_pools.json  {'VALID' if valid else 'INVALID — fix before eval-gen'}")
