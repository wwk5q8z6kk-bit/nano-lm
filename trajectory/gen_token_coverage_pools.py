# PREREG_token_coverage.md — band construction (deterministic; committed pre-run).
# Computes each candidate's token-coverage against the ACTUAL D80-arm training-output
# token set (summaries of the D80-patched v2 recipe), then assigns 4 length-matched
# bands of >=4 types. Bridge types (sulfa drugs, ragweed pollen) carried separately.
# Output: trajectory/token_coverage_bands.json (frozen instrument input).
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slot_diversity_pools import ALG_TRAIN_80, HELD_ALG
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(os.path.join(HERE, "..", "sft", "tokenizer.json"))

# --- D80-arm training-output token set (source-patched v2, exec'd verbatim) ---
V2 = open(os.path.join(HERE, "..", "scribe", "build_scribe_data_v2.py")).read()
ALG_LINE = 'ALG_TRAIN = ["penicillin", "peanuts", "pollen", "latex", "shellfish"]'
assert ALG_LINE in V2
src = V2.replace(ALG_LINE, "ALG_TRAIN = " + repr(list(ALG_TRAIN_80)))
prefix = src.split('tok = Tokenizer.from_file')[0].replace("from tokenizers import Tokenizer", "")
ns = {}; exec(compile(prefix, "v2[d80]", "exec"), ns)
convos = ns["convos"]; assert len(convos) == 12000
out_tokens = set()
for c in convos:
    out_tokens.update(tok.encode(c[1]["content"], add_special_tokens=False).ids)
print(f"D80 training-output token set: {len(out_tokens)} distinct tokens")

# --- committed candidate list (plausible allergen strings; frozen here) ---
CANDIDATES = [
    "amoxicillin dust", "apricots", "bananas ripe", "barley", "basil",
    "bleach fumes", "blue dye", "buckwheat", "camphor", "cane sugar",
    "cherry stones", "chili powder", "cinnamon", "cloves", "coconut oil",
    "copper coins", "corn starch", "cotton fibers", "cough drops", "cumin",
    "dandelion", "diesel fumes", "dill", "elm pollen", "eye drops",
    "fig sap", "garlic powder", "ginger root", "grape skins", "green tea",
    "gum arabic", "hazel pollen", "henna", "iodine", "ivy leaves",
    "lamb wool", "lavender oil", "lemon zest", "maple pollen", "melon rind",
    "mint leaves", "nettle", "nylon", "oat bran", "olive pits",
    "onion skins", "orange peel", "papaya", "paprika", "parsley",
    "pear skins", "pepper spray", "pine nuts", "pine resin", "plum stones",
    "polyester", "pumpkin seeds", "quinoa", "rice flour", "rose hips",
    "rubber gloves", "rye bread", "saffron", "sage", "sand flies",
    "silk thread", "silver rings", "spinach", "spray paint", "talc powder",
    "thyme", "tin foil", "tuna fish", "vanilla", "vine pollen",
    "willow bark", "yeast", "zinc cream", "nasal spray", "throat spray",
]
# exclusions: training pool, all sweep held types (bridges handled separately)
excl = set(ALG_TRAIN_80) | set(HELD_ALG)
cands = [c for c in CANDIDATES if c not in excl]

def profile(v):
    ids = tok.encode(v, add_special_tokens=False).ids
    cov = sum(1 for i in ids if i in out_tokens) / len(ids)
    return {"value": v, "n_tokens": len(ids), "coverage": round(cov, 4),
            "tokens": [tok.decode([i]) for i in ids],
            "novel": [tok.decode([i]) for i in ids if i not in out_tokens]}

profs = [profile(c) for c in cands]
LEN_LO, LEN_HI = 3, 6
eligible = [p for p in profs if LEN_LO <= p["n_tokens"] <= LEN_HI]

BANDS = {"B100": (1.0, 1.01), "Bhi": (0.75, 1.0), "Blo": (0.25, 0.501), "B0": (0.0, 0.25)}
bands = {}
for name, (lo, hi) in BANDS.items():
    pool = [p for p in eligible if (lo <= p["coverage"] < hi) or (name == "B100" and p["coverage"] == 1.0)]
    pool.sort(key=lambda p: (abs(p["n_tokens"] - 4.5), p["value"]))   # length-match, deterministic
    bands[name] = pool[:4]
    print(f"{name}: {len(pool)} eligible -> picked {[p['value'] for p in bands[name]]} "
          f"(cov {[p['coverage'] for p in bands[name]]}, len {[p['n_tokens'] for p in bands[name]]})")

bridges = [profile("sulfa drugs"), profile("ragweed pollen")]
print("bridges:", [(b["value"], b["coverage"], b["n_tokens"]) for b in bridges])

ok = all(len(bands[b]) >= 4 for b in bands)
med = {b: sorted(p["n_tokens"] for p in bands[b])[len(bands[b])//2] for b in bands}
print("band median lengths:", med, "| all bands filled:", ok)
json.dump({"bands": {b: bands[b] for b in bands}, "bridges": bridges,
           "d80_output_token_count": len(out_tokens),
           "selection": "len in [3,6], closest to 4.5, alphabetical tiebreak; committed pre-run"},
          open(os.path.join(HERE, "token_coverage_bands.json"), "w"), indent=1)
print("-> token_coverage_bands.json", "VALID" if ok else "INSUFFICIENT — extend candidate list")
