# Stage T, Arm 1 — Pythia ladder finetuned on the scribe v2 recipe (Kaggle T4).
# Pre-registered in trajectory/PREREG.md BEFORE any measurement. This script
# performs ONE measurement per cell: no tuning loops, no retries, no post-hoc
# prompt or hyperparameter changes.
#
# Per rung: finetune -> base control -> score instance 0 + instance T (greedy)
# -> bootstrap CI on the held-out gap -> emergence trajectory over retained
# intermediate checkpoints (descriptive secondary observable, no bars).
# The first rung's instance-0-vs-instance-T comparison is the pre-registered
# equivalence check (per-metric |diff| < 5 pts, else STOP per PREREG).
#
# Text-level training data is byte-identical to nano/scale stages: the v2
# generator prefix (through its convos loop, seed 11) is exec'd verbatim;
# only tokenization differs (Pythia BPE vs nano 4096-BPE) — acknowledged in
# PREREG "Design". Run-logging contract: all seeds, hyperparameters, GPU type,
# wall-clock, and checkpoint cadence land in the results JSON.
#
# Usage (Kaggle, repo cloned at /kaggle/working/nano-lm):
#   python trajectory/kaggle_arm1.py EleutherAI/pythia-160m
#   python trajectory/kaggle_arm1.py EleutherAI/pythia-410m
#   python trajectory/kaggle_arm1.py EleutherAI/pythia-1b
import json, math, os, random, re, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 20260717                  # harness seed (recorded); data recipe seed 11 lives in the v2 prefix
LR = 1e-4                        # pre-registered
EPOCHS = 3                       # pre-registered
MICRO_BATCH = 8                  # venue decision, recorded in results JSON
ACCUM = 4                        # effective batch 32 (matches nano scribe finetune)
MAX_LEN = 448
MAX_NEW = 64
N_CKPT = 6                       # intermediate checkpoints retained (emergence trajectory)
BOOT = 10_000                    # pre-registered bootstrap resamples
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.environ.get("ARM1_OUT", "/kaggle/working")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-160m"
TAG = MODEL.split("/")[-1]
dev = "cuda"
assert torch.cuda.is_available(), "Arm 1 runs on CUDA (Kaggle T4) per PREREG venue"

# ---------------- training text: exec the v2 generator prefix verbatim ----------------
v2_src = open(os.path.join(REPO, "scribe", "build_scribe_data_v2.py")).read()
marker = 'tok = Tokenizer.from_file'
assert marker in v2_src, "v2 generator layout changed; re-verify before running"
prefix = v2_src.split(marker)[0]
prefix = prefix.replace("from tokenizers import Tokenizer", "")   # not needed pre-marker; avoids the dep
ns = {}
exec(compile(prefix, "build_scribe_data_v2.py[prefix]", "exec"), ns)
convos = ns["convos"]
assert len(convos) == 12000, f"expected 12000 v2 convos, got {len(convos)}"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- tokenize for Pythia (plain text; loss masked to summary) ----------------
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
EOS = tok.eos_token_id
NL = tok.encode("\n")[-1]

def encode_example(convo):
    # prompt: dialogue + "\nSummarize the visit.\n"   target: one-line summary + EOS
    prompt = convo[0]["content"] + "\n"
    target = convo[1]["content"]
    p_ids = tok.encode(prompt)
    t_ids = tok.encode(target) + [EOS]
    ids = p_ids + t_ids
    labels = [-100] * len(p_ids) + t_ids
    return ids, labels

examples, dropped = [], 0
for c in convos:
    ids, labels = encode_example(c)
    if len(ids) > MAX_LEN:
        dropped += 1; continue
    examples.append((ids, labels))
print(f"train examples {len(examples)} (dropped {dropped} >{MAX_LEN} tok)", flush=True)

def batches(data, bs):
    idx = list(range(len(data)))
    random.shuffle(idx)
    for i in range(0, len(idx) - bs + 1, bs):
        chunk = [data[j] for j in idx[i:i + bs]]
        L = max(len(x[0]) for x in chunk)
        x = torch.full((bs, L), EOS, dtype=torch.long)
        y = torch.full((bs, L), -100, dtype=torch.long)
        m = torch.zeros((bs, L), dtype=torch.long)
        for r, (ids, labels) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids)
            y[r, :len(labels)] = torch.tensor(labels)
            m[r, :len(ids)] = 1
        yield x.to(dev), y.to(dev), m.to(dev)

# ---------------- scorer: gate_scribe.py logic ported (same regex, same field rules) ----------------
RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")
FIELDS = ["cc", "dur", "sev", "med", "alg"]

@torch.no_grad()
def score(model, items, label=""):
    model.eval()
    tok.padding_side = "left"
    per_dialogue = []          # (held_values, hits_out_of_5, halluc, omission, parsed)
    outs = []
    for i in range(0, len(items), MICRO_BATCH):
        chunk = items[i:i + MICRO_BATCH]
        prompts = [it["convo"][0]["content"] + "\n" for it in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             eos_token_id=[EOS, NL], pad_token_id=EOS)
        for r, it in enumerate(chunk):
            text = tok.decode(gen[r, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            outs.append(text)
            mm = RE.match(text)
            if not mm:
                per_dialogue.append((it["held_values"], 0, 0, 0, 0)); continue
            pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
            hits = halluc = omission = 0
            for f in FIELDS:
                t, p = it["tuple"][f], pred[f]
                if p == t: hits += 1
                elif p == "none" and t != "none": omission += 1
                else: halluc += 1
            per_dialogue.append((it["held_values"], hits, halluc, omission, 1))
    n = len(items)
    parsed = sum(d[4] for d in per_dialogue)
    recall = sum(d[1] for d in per_dialogue) / (5 * n)
    hal = sum(d[2] for d in per_dialogue) / (5 * n)
    omi = sum(d[3] for d in per_dialogue)
    held = [d for d in per_dialogue if d[0]]; seen = [d for d in per_dialogue if not d[0]]
    held_rec = sum(d[1] for d in held) / (5 * len(held)) if held else float("nan")
    seen_rec = sum(d[1] for d in seen) / (5 * len(seen)) if seen else float("nan")
    gap = seen_rec - held_rec
    # pre-registered bootstrap: resample dialogues within each stratum, 95% CI on gap
    rng = np.random.default_rng(SEED)
    h_hits = np.array([d[1] for d in held]); s_hits = np.array([d[1] for d in seen])
    gaps = []
    for _ in range(BOOT):
        hb = rng.choice(h_hits, len(h_hits), replace=True).sum() / (5 * len(h_hits))
        sb = rng.choice(s_hits, len(s_hits), replace=True).sum() / (5 * len(s_hits))
        gaps.append(sb - hb)
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"[{label}] parse {parsed}/{n}={parsed/n:.0%}  recall {recall:.0%}  halluc {hal:.1%}  "
          f"omission {omi}", flush=True)
    print(f"          seen {seen_rec:.0%}  held {held_rec:.0%}  GAP {gap*100:.1f} pts "
          f"(95% CI [{lo*100:.1f}, {hi*100:.1f}])", flush=True)
    print(f"          sample out: {outs[0]!r}", flush=True)
    return {"parse": parsed / n, "recall": recall, "halluc": hal, "omission": omi,
            "seen_recall": seen_rec, "held_recall": held_rec,
            "gap_pts": gap * 100, "gap_ci_pts": [lo * 100, hi * 100]}

inst0 = json.load(open(os.path.join(REPO, "scribe", "scribe_eval.json")))
instT = json.load(open(os.path.join(REPO, "trajectory", "scribe_eval_T.json")))

# ---------------- base control (pre-registered: un-finetuned must fail parse < 50%) ----------------
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
model.gradient_checkpointing_enable()
base_metrics = score(model, inst0, label=f"{TAG}/BASE/inst0")
print(f"base control fails parse<50%: {base_metrics['parse'] < 0.5}", flush=True)

# ---------------- finetune (one run; intermediate checkpoints retained) ----------------
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
steps_per_epoch = len(examples) // (MICRO_BATCH * ACCUM)
total_steps = steps_per_epoch * EPOCHS
ckpt_every = max(1, total_steps // N_CKPT)
print(f"{TAG}: {total_steps} optim steps, ckpt every {ckpt_every}", flush=True)

trajectory = []
step = 0
for ep in range(EPOCHS):
    micro = 0
    opt.zero_grad(set_to_none=True)
    for x, y, m in batches(examples, MICRO_BATCH):
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=x, attention_mask=m, labels=y)
        scaler.scale(out.loss / ACCUM).backward()
        micro += 1
        if micro % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep} step {step}/{total_steps} loss {out.loss.item():.3f}", flush=True)
            if step % ckpt_every == 0 or step == total_steps:
                path = os.path.join(OUTDIR, f"{TAG}_scribe_step{step}.pt")
                torch.save(model.state_dict(), path)
                mets = score(model, inst0, label=f"{TAG}/step{step}/inst0")
                trajectory.append({"step": step, **mets})
                model.train()
train_secs = time.time() - t0

# ---------------- final measurement: both instances (equivalence check = |diff| < 5 pts) ----------------
print(f"\n=== FINAL ({TAG}) ===", flush=True)
final0 = score(model, inst0, label=f"{TAG}/FINAL/inst0")
finalT = score(model, instT, label=f"{TAG}/FINAL/instT")
equiv = {k: abs(final0[k] - finalT[k]) < 0.05 for k in ["parse", "recall", "halluc"]}
print(f"equivalence check (per-metric |diff| < 5 pts): {equiv} -> "
      f"{'OK' if all(equiv.values()) else 'FAIL — STOP PER PREREG'}", flush=True)

results = {
    "stage": "T-arm1", "model": MODEL, "prereg": "trajectory/PREREG.md",
    "seeds": {"harness": SEED, "data_recipe": 11, "instanceT": 20260717},
    "hparams": {"lr": LR, "epochs": EPOCHS, "micro_batch": MICRO_BATCH, "accum": ACCUM,
                "max_len": MAX_LEN, "clip": 1.0, "betas": [0.9, 0.95], "wd": 0.0,
                "precision": "fp16-autocast", "lora": None},
    "gpu": torch.cuda.get_device_name(0), "train_wall_secs": round(train_secs),
    "base_control": base_metrics, "final_inst0": final0, "final_instT": finalT,
    "equivalence_ok": all(equiv.values()), "ckpt_trajectory": trajectory,
}
out_path = os.path.join(OUTDIR, f"results_arm1_{TAG}.json")
json.dump(results, open(out_path, "w"), indent=1)
print(f"results -> {out_path}", flush=True)
