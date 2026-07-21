# Stage T-v2 — powered gap estimator (PREREG_Tv2.md). Re-instruments the gap
# ONLY: same models, same seeds, same finetune as Arm 1 v1 (deterministic —
# the headless-T4 410m reproduced the interactive run byte-for-byte, so
# re-finetuning at the fixed seed regenerates the frozen v1 adapter exactly).
# The final measurement scores that adapter on K=5 larger fresh instances
# (100 held + 100 seen each) plus inst0 and instT, and reports gap_mean ± SD
# across the five fresh instances — the quantity v1 lacked.
#
# The finetune and score() below are copied verbatim from the reviewed
# kaggle_arm1.py (v1). Only the final-measurement block differs. inst0/instT
# gaps are re-reported for a determinism cross-check against the v1 JSONs.
#
# Usage: python trajectory/kaggle_arm1_v2.py EleutherAI/pythia-160m  (then 410m, 1b)
import json, os, random, re, sys, time
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    import peft
    from peft import LoraConfig, get_peft_model
except ImportError:
    sys.exit("peft required")

SEED = 20260717
LR = 1e-4; EPOCHS = 3; MICRO_BATCH = 8; ACCUM = 4
MAX_LEN = 448; MAX_NEW = 64; BOOT = 10_000
LORA = {"r": 16, "alpha": 32, "dropout": 0.0,
        "targets": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]}
DECODING = "greedy, stop on EOS only, max_new 64"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.environ.get("ARM1_OUT", "/kaggle/working")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-160m"
TAG = MODEL.split("/")[-1]
dev = "cuda"; assert torch.cuda.is_available()

# ---- training text: exec v2 generator prefix verbatim (same as v1) ----
v2_src = open(os.path.join(REPO, "scribe", "build_scribe_data_v2.py")).read()
prefix = v2_src.split('tok = Tokenizer.from_file')[0].replace("from tokenizers import Tokenizer", "")
ns = {}; exec(compile(prefix, "v2[prefix]", "exec"), ns)
convos = ns["convos"]; assert len(convos) == 12000
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
EOS = tok.eos_token_id

p_texts = [c[0]["content"] + "\n" for c in convos]
t_texts = [c[1]["content"] for c in convos]

p_ids_batch = tok(p_texts, add_special_tokens=False)["input_ids"]
t_ids_batch = tok(t_texts, add_special_tokens=False)["input_ids"]

examples, dropped = [], 0
for p_ids, t_ids in zip(p_ids_batch, t_ids_batch):
    t_ids = t_ids + [EOS]
    ids = p_ids + t_ids
    if len(ids) > MAX_LEN: dropped += 1; continue
    labels = [-100] * len(p_ids) + t_ids
    examples.append((ids, labels))
print(f"train examples {len(examples)} (dropped {dropped})", flush=True)

def batches(data, bs):
    idx = list(range(len(data))); random.shuffle(idx)
    for i in range(0, len(idx) - bs + 1, bs):
        chunk = [data[j] for j in idx[i:i + bs]]
        L = max(len(x[0]) for x in chunk)
        x = torch.full((bs, L), EOS, dtype=torch.long)
        y = torch.full((bs, L), -100, dtype=torch.long)
        m = torch.zeros((bs, L), dtype=torch.long)
        for r, (ids, labels) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids); y[r, :len(labels)] = torch.tensor(labels)
            m[r, :len(ids)] = 1
        yield x.to(dev), y.to(dev), m.to(dev)

RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FIELDS = ["cc", "dur", "sev", "med", "alg"]

@torch.no_grad()
def score(model, items, label=""):                       # verbatim from reviewed v1
    model.eval(); tok.padding_side = "left"
    per = []
    for i in range(0, len(items), MICRO_BATCH):
        chunk = items[i:i + MICRO_BATCH]
        enc = tok([it["convo"][0]["content"] + "\n" for it in chunk],
                  return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             eos_token_id=EOS, pad_token_id=EOS)
        for r, it in enumerate(chunk):
            text = tok.decode(gen[r, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            mm = RE.match(text)
            if not mm: per.append((it["held_values"], 0, 0, 0, 0)); continue
            pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
            hits = halluc = omission = 0
            for f in FIELDS:
                t, p = it["tuple"][f], pred[f]
                if p == t: hits += 1
                elif p == "none" and t != "none": omission += 1
                else: halluc += 1
            per.append((it["held_values"], hits, halluc, omission, 1))
    n = len(items)
    parsed = sum(d[4] for d in per)
    recall = sum(d[1] for d in per) / (5 * n); hal = sum(d[2] for d in per) / (5 * n)
    held = [d for d in per if d[0] and d[4]]; seen = [d for d in per if (not d[0]) and d[4]]
    if len(held) >= 2 and len(seen) >= 2:
        held_rec = sum(d[1] for d in held) / (5 * len(held))
        seen_rec = sum(d[1] for d in seen) / (5 * len(seen))
        gap = seen_rec - held_rec
        rng = np.random.default_rng(SEED)
        h = np.array([d[1] for d in held]); s = np.array([d[1] for d in seen]); gaps = []
        for _ in range(BOOT):
            hb = rng.choice(h, len(h), replace=True).sum() / (5 * len(h))
            sb = rng.choice(s, len(s), replace=True).sum() / (5 * len(s))
            gaps.append(sb - hb)
        lo, hi = np.percentile(gaps, [2.5, 97.5]); gap_pts, ci = gap * 100, [lo * 100, hi * 100]
        held_pct, seen_pct = held_rec, seen_rec
    else:
        held_pct = seen_pct = gap_pts = ci = None
    print(f"[{label}] parse {parsed}/{n}={parsed/n:.0%} recall {recall:.0%} halluc {hal:.1%} "
          f"gap {('%.1f'%gap_pts) if gap_pts is not None else 'NA'}", flush=True)
    return {"parse": parsed / n, "recall": recall, "halluc": hal,
            "seen_recall": seen_pct, "held_recall": held_pct,
            "held_parsed": len(held), "seen_parsed": len(seen),
            "gap_pts": gap_pts, "gap_ci_pts": ci, "n": n}

# ---- eval instances: inst0, instT, + 5 fresh larger instances ----
inst0 = json.load(open(os.path.join(REPO, "scribe", "scribe_eval.json")))
instT = json.load(open(os.path.join(REPO, "trajectory", "scribe_eval_T.json")))
fresh = [json.load(open(os.path.join(REPO, "trajectory", f"scribe_eval_m{k}.json"))) for k in range(5)]

# ---- base control (same falsifier as v1) ----
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
base = score(model, inst0, label=f"{TAG}/BASE/inst0")
print(f"base fails parse<50%: {base['parse'] < 0.5}", flush=True)

# ---- LoRA finetune (identical to v1 -> regenerates the frozen adapter) ----
model = get_peft_model(model, LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"],
        lora_dropout=LORA["dropout"], target_modules=LORA["targets"], task_type="CAUSAL_LM"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                        lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
total_steps = (len(examples) // (MICRO_BATCH * ACCUM)) * EPOCHS
t0 = time.time(); step = 0
for ep in range(EPOCHS):
    micro = 0; opt.zero_grad(set_to_none=True)
    for x, y, m in batches(examples, MICRO_BATCH):
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=x, attention_mask=m, labels=y)
        scaler.scale(out.loss / ACCUM).backward(); micro += 1
        if micro % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            if step % 200 == 0: print(f"  ep{ep} step {step}/{total_steps} loss {out.loss.item():.3f}", flush=True)
train_secs = time.time() - t0

# ---- powered final measurement ----
print(f"\n=== FINAL v2 ({TAG}) ===", flush=True)
m0 = score(model, inst0, label=f"{TAG}/inst0")
mT = score(model, instT, label=f"{TAG}/instT")
mfresh = [score(model, fresh[k], label=f"{TAG}/m{k}") for k in range(5)]
fresh_gaps = [r["gap_pts"] for r in mfresh]
gap_mean = float(np.mean(fresh_gaps)); gap_sd = float(np.std(fresh_gaps, ddof=1))
print(f"  fresh gaps: {[round(g,1) for g in fresh_gaps]}", flush=True)
print(f"  gap_mean {gap_mean:.2f} ± {gap_sd:.2f} SD  (2SD band [{gap_mean-2*gap_sd:.1f}, {gap_mean+2*gap_sd:.1f}])", flush=True)

# contamination check (direction-aware): flag ONLY if inst0 EASIER than fresh mean
inst0_gap = m0["gap_pts"]
contam = (inst0_gap is not None) and (inst0_gap < gap_mean - 2 * gap_sd)
print(f"  inst0 gap {inst0_gap:.1f} vs fresh {gap_mean:.1f} — contamination(inst0 easier by >2SD): {contam}", flush=True)
# determinism cross-check vs v1 (inst0/instT gaps should match the v1 JSON)
print(f"  determinism check — inst0 gap {inst0_gap} , instT gap {mT['gap_pts']} (compare to v1 JSON)", flush=True)

results = {
    "stage": "T-v2", "model": MODEL, "prereg": "trajectory/PREREG_Tv2.md",
    "seeds": {"harness": SEED, "fresh_instances": [20260720, 20260721, 20260722, 20260723, 20260724]},
    "hparams": {"lr": LR, "epochs": EPOCHS, "micro_batch": MICRO_BATCH, "accum": ACCUM,
                "lora": LORA, "trainable_params": trainable, "precision": "fp16-autocast"},
    "decoding": DECODING, "bootstrap_resamples": BOOT,
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                 "peft": peft.__version__, "python": sys.version.split()[0]},
    "gpu": torch.cuda.get_device_name(0), "train_wall_secs": round(train_secs),
    "base_control": base, "inst0": m0, "instT": mT,
    "fresh_instances": mfresh, "fresh_gaps": fresh_gaps,
    "gap_mean": gap_mean, "gap_sd": gap_sd,
    "gap_2sd_band": [gap_mean - 2 * gap_sd, gap_mean + 2 * gap_sd],
    "contamination_flag": contam,
}
out_path = os.path.join(OUTDIR, f"results_arm1_v2_{TAG}.json")
json.dump(results, open(out_path, "w"), indent=1)
print(f"results -> {out_path}", flush=True)
