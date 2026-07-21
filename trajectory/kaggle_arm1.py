# Stage T, Arm 1 — Pythia ladder finetuned on the scribe v2 recipe (Kaggle T4).
# Pre-registered in trajectory/PREREG.md BEFORE any measurement. This script
# performs ONE measurement per cell: no tuning loops, no retries, no post-hoc
# prompt or hyperparameter changes.
#
# Adversarially reviewed 2026-07-17 before any run; fixes applied:
#   F1  held/seen recall is CONDITIONAL ON PARSE, matching gate_scribe.py
#       (total_fields counts all items; held/seen strata count parsed items
#       only). The nano 22-pt / scale 23-pt anchors were computed with these
#       semantics; unconditional denominators would have doubled the gap
#       wherever parse < 100% and turned the emergence trajectory into a
#       parse-rate curve.
#   F2  LoRA for ALL rungs (prereg all-or-none policy), invoked pre-emptively:
#       pythia-1b full-FT needs ~16.2 GB optimizer+grad state vs ~15 GB usable
#       on T4 — it would OOM after the base control already measured. Recorded
#       as PREREG Amendment 3. Adapter-only checkpoints also fit the disk quota.
#   F3  equivalence check covers ALL five metrics (parse, recall, halluc,
#       omission rate, gap) and hard-exits nonzero on failure.
#   F4  decoding stops on EOS only (no newline stop) — exact parity with
#       gate_scribe stopping semantics. Recorded in results JSON.
#   F5-F8  run-logging contract completed (examples/dropped, versions,
#       decoding rule, cadence); train wall-clock excludes download + base
#       scoring.
#
# Per rung: base control -> single LoRA finetune -> score instance 0 + T
# (greedy) -> bootstrap CI on the held-out gap -> emergence trajectory over
# retained intermediate adapter checkpoints (descriptive, no bars).
# The first rung's instance-0-vs-instance-T comparison is the pre-registered
# equivalence check (per-metric |diff| < 5 pts, else STOP per PREREG).
#
# Usage (Kaggle, repo cloned at /kaggle/working/nano-lm):
#   python trajectory/kaggle_arm1.py EleutherAI/pythia-160m
#   python trajectory/kaggle_arm1.py EleutherAI/pythia-410m
#   python trajectory/kaggle_arm1.py EleutherAI/pythia-1b
import json, os, random, re, sys, time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    import peft
    from peft import LoraConfig, get_peft_model
except ImportError:
    sys.exit("peft required (pip install peft) — LoRA is the registered method for all rungs")

SEED = 20260717                  # harness seed (recorded); data recipe seed 11 lives in the v2 prefix
LR = 1e-4                        # pre-registered (v2 recipe value, retained under LoRA)
EPOCHS = 3                       # pre-registered
MICRO_BATCH = 8                  # venue decision, recorded in results JSON
ACCUM = 4                        # effective batch 32 (matches nano scribe finetune)
MAX_LEN = 448
MAX_NEW = 64
N_CKPT = 6                       # intermediate adapter checkpoints retained (emergence trajectory)
BOOT = 10_000                    # pre-registered bootstrap resamples
LORA = {"r": 16, "alpha": 32, "dropout": 0.0,
        "targets": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]}
DECODING = "greedy, stop on EOS only (no newline stop; parity with gate_scribe), max_new 64"
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
prefix = v2_src.split(marker)[0].replace("from tokenizers import Tokenizer", "")
ns = {}
exec(compile(prefix, "build_scribe_data_v2.py[prefix]", "exec"), ns)
convos = ns["convos"]
assert len(convos) == 12000, f"expected 12000 v2 convos, got {len(convos)}"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------------- tokenize for Pythia (plain text; loss masked to summary) ----------------
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
EOS = tok.eos_token_id

def encode_example(convo):
    # prompt: dialogue + "\nSummarize the visit.\n"   target: one-line summary + EOS
    prompt = convo[0]["content"] + "\n"
    target = convo[1]["content"]
    p_ids = tok.encode(prompt)
    t_ids = tok.encode(target) + [EOS]
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids

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
    per_dialogue = []          # (held_values, hits_of_5, halluc, omission, parsed)
    outs = []
    for i in range(0, len(items), MICRO_BATCH):
        chunk = items[i:i + MICRO_BATCH]
        prompts = [it["convo"][0]["content"] + "\n" for it in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             eos_token_id=EOS, pad_token_id=EOS)
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
    recall = sum(d[1] for d in per_dialogue) / (5 * n)          # unconditional, per gate_scribe
    hal = sum(d[2] for d in per_dialogue) / (5 * n)
    omi = sum(d[3] for d in per_dialogue)
    # held/seen strata are CONDITIONAL ON PARSE — gate_scribe.py:96-106 semantics
    # (the nano/scale anchor gaps were computed this way). (Implements Fix F1 from the header).
    held = [d for d in per_dialogue if d[0] and d[4]]
    seen = [d for d in per_dialogue if (not d[0]) and d[4]]
    if len(held) >= 2 and len(seen) >= 2:
        held_rec = sum(d[1] for d in held) / (5 * len(held))
        seen_rec = sum(d[1] for d in seen) / (5 * len(seen))
        gap = seen_rec - held_rec
        # Bootstrap reseeds default_rng(SEED) on every call DELIBERATELY: the
        # reported CI is a deterministic function of the measurements. Not a bug.
        rng = np.random.default_rng(SEED)
        h_hits = np.array([d[1] for d in held]); s_hits = np.array([d[1] for d in seen])
        gaps = []
        for _ in range(BOOT):
            hb = rng.choice(h_hits, len(h_hits), replace=True).sum() / (5 * len(h_hits))
            sb = rng.choice(s_hits, len(s_hits), replace=True).sum() / (5 * len(s_hits))
            gaps.append(sb - hb)
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        gap_pts, ci = gap * 100, [lo * 100, hi * 100]
        held_pct, seen_pct = held_rec, seen_rec
    else:                       # e.g. base control: nothing parses; None -> JSON null (not NaN)
        held_pct = seen_pct = gap_pts = ci = None
    print(f"[{label}] parse {parsed}/{n}={parsed/n:.0%}  recall {recall:.0%}  halluc {hal:.1%}  "
          f"omission {omi}", flush=True)
    if gap_pts is not None:
        print(f"          seen {seen_pct:.0%} ({len(seen)} parsed)  held {held_pct:.0%} "
              f"({len(held)} parsed)  GAP {gap_pts:.1f} pts (95% CI [{ci[0]:.1f}, {ci[1]:.1f}])", flush=True)
    else:
        print(f"          held/seen strata too small among parsed ({len(held)}/{len(seen)}) — gap undefined", flush=True)
    print(f"          sample out: {outs[0]!r}", flush=True)
    return {"parse": parsed / n, "recall": recall, "halluc": hal,
            "omission": omi, "omission_rate": omi / (5 * n),
            "seen_recall": seen_pct, "held_recall": held_pct,
            "held_parsed": len(held), "seen_parsed": len(seen),
            "gap_pts": gap_pts, "gap_ci_pts": ci}

inst0 = json.load(open(os.path.join(REPO, "scribe", "scribe_eval.json")))
instT = json.load(open(os.path.join(REPO, "trajectory", "scribe_eval_T.json")))

# ---------------- base control (pre-registered: un-finetuned must fail parse < 50%) ----------------
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
base_metrics = score(model, inst0, label=f"{TAG}/BASE/inst0")
print(f"base control fails parse<50%: {base_metrics['parse'] < 0.5}", flush=True)

# ---------------- LoRA finetune (all rungs; PREREG Amendment 3) ----------------
model = get_peft_model(model, LoraConfig(
    r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
    target_modules=LORA["targets"], task_type="CAUSAL_LM"))
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{TAG}: trainable params {trainable:,}", flush=True)

model.train()
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                        lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
steps_per_epoch = len(examples) // (MICRO_BATCH * ACCUM)
total_steps = steps_per_epoch * EPOCHS
ckpt_every = max(1, total_steps // N_CKPT)
print(f"{TAG}: {total_steps} optim steps, adapter ckpt every {ckpt_every}", flush=True)

t0 = time.time()                # training wall-clock only (F6): after download + base scoring
trajectory = []
step = 0                        # scaler-skipped steps still increment; rare after warmup (noted)
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
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"  ep{ep} step {step}/{total_steps} loss {out.loss.item():.3f}", flush=True)
            if step % ckpt_every == 0 and step != total_steps:      # final saved once below (F7)
                path = os.path.join(OUTDIR, f"{TAG}_scribe_step{step}")
                model.save_pretrained(path)                          # adapter-only (disk quota)
                mets = score(model, inst0, label=f"{TAG}/step{step}/inst0")
                trajectory.append({"step": step, **mets})
                model.train()
train_secs = time.time() - t0
model.save_pretrained(os.path.join(OUTDIR, f"{TAG}_scribe_final"))

# ---------------- final measurement: both instances; equivalence = ALL five metrics (F3) ----------------
print(f"\n=== FINAL ({TAG}) ===", flush=True)
final0 = score(model, inst0, label=f"{TAG}/FINAL/inst0")
finalT = score(model, instT, label=f"{TAG}/FINAL/instT")

def rates(m):
    return {"parse": m["parse"], "recall": m["recall"], "halluc": m["halluc"],
            "omission_rate": m["omission_rate"],
            "gap": None if m["gap_pts"] is None else m["gap_pts"] / 100}

equiv = {}
for k, v0 in rates(final0).items():
    vT = rates(finalT)[k]
    equiv[k] = (v0 is not None and vT is not None and abs(v0 - vT) < 0.05)
equiv_ok = all(equiv.values())
print(f"equivalence check (per-metric |diff| < 5 pts, all five): {equiv} -> "
      f"{'OK' if equiv_ok else 'FAIL — STOP PER PREREG'}", flush=True)

results = {
    "stage": "T-arm1", "model": MODEL, "prereg": "trajectory/PREREG.md",
    "seeds": {"harness": SEED, "data_recipe": 11, "instanceT": 20260717, "dev": 20260718},
    "hparams": {"lr": LR, "epochs": EPOCHS, "micro_batch": MICRO_BATCH, "accum": ACCUM,
                "max_len": MAX_LEN, "max_new": MAX_NEW, "clip": 1.0, "betas": [0.9, 0.95],
                "wd": 0.0, "precision": "fp16-autocast", "lora": LORA,
                "trainable_params": trainable},
    "decoding": DECODING, "bootstrap_resamples": BOOT,
    "data": {"examples": len(examples), "dropped": dropped},
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                 "peft": peft.__version__, "python": sys.version.split()[0]},
    "gpu": torch.cuda.get_device_name(0), "train_wall_secs": round(train_secs),
    "ckpt_every": ckpt_every, "n_ckpt_target": N_CKPT,
    "base_control": base_metrics, "final_inst0": final0, "final_instT": finalT,
    "equivalence": equiv, "equivalence_ok": equiv_ok, "ckpt_trajectory": trajectory,
}
out_path = os.path.join(OUTDIR, f"results_arm1_{TAG}.json")
json.dump(results, open(out_path, "w"), indent=1)
print(f"results -> {out_path}", flush=True)

if not equiv_ok:
    sys.exit(1)                 # first-rung gate must not be scrollable-past (F3)
