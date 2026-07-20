# PREREG_slot_diversity.md — FROZEN pools (committed before any variant data/runs exist).
# The 6 HELD types are excluded from every training pool; D20 ⊂ D80 by construction.

HELD_ALG = ["sulfa drugs", "ragweed pollen", "bee stings", "ibuprofen", "wool", "strawberries"]
# 'ibuprofen' is the token-novelty probe: present in MED_TRAIN outputs, never as an allergy.

ALG_TRAIN_5 = ["penicillin", "peanuts", "pollen", "latex", "shellfish"]          # original v2

ALG_TRAIN_20 = ALG_TRAIN_5 + [
    "amoxicillin", "aspirin", "cats", "dogs", "dust mites",
    "eggs", "gluten", "grass pollen", "mold", "nickel",
    "sesame", "soy", "tree nuts", "wasp stings", "codeine",
]

ALG_TRAIN_80 = ALG_TRAIN_20 + [
    "milk", "wheat", "fish", "almonds", "walnuts",
    "cashews", "hazelnuts", "pecans", "pistachios", "lobster",
    "crab", "shrimp", "oysters", "clams", "mussels",
    "tomatoes", "citrus fruits", "kiwi", "bananas", "avocado",
    "mango", "peaches", "apples", "celery", "mustard",
    "sunflower seeds", "poppy seeds", "chickpeas", "lentils", "lupin",
    "birch pollen", "cedar pollen", "oak pollen", "hay", "feathers",
    "horse dander", "rabbit dander", "cockroaches", "bed bugs", "chlorine",
    "fragrances", "hair dye", "adhesive bandages", "cobalt", "chromium",
    "formaldehyde", "sulfites", "msg", "red dye", "yellow dye",
    "cephalosporins", "tetracycline", "erythromycin", "clindamycin", "vancomycin",
    "morphine", "tramadol", "gabapentin", "lidocaine", "novocaine",
]

assert len(ALG_TRAIN_5) == 5 and len(ALG_TRAIN_20) == 20 and len(ALG_TRAIN_80) == 80
assert set(ALG_TRAIN_5) <= set(ALG_TRAIN_20) <= set(ALG_TRAIN_80)
assert not (set(HELD_ALG) & set(ALG_TRAIN_80)), "held types must be excluded from all pools"

if __name__ == "__main__":
    import os
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), "..", "sft", "tokenizer.json"))
    print("pool sizes OK: 5/20/80; held excluded")
    for v in HELD_ALG:
        ids = tok.encode(v, add_special_tokens=False).ids
        print(f"held {v!r:18s} -> {len(ids)} tokens: {[tok.decode([i]) for i in ids]}")
