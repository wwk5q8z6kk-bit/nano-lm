# Stage T-v2 — PER-FIELD (fieldwise) gap for the Pythia rungs (council-recommended
# PYTHIA_FIELDWISE strengthener). Tests whether the paper's titular "field-localized"
# claim — the own-stack gap sits entirely in the open-vocabulary fields (cc/med/alg)
# and is exactly 0 in the closed-value fields (dur/sev) — also holds on the Pythia side.
#
# Same frozen model as kaggle_arm1_v2.py: same SEED, LoRA config, and deterministic
# finetune regenerate the same adapter (headless-T4 reproduces byte-for-byte). Only the
# final measurement differs: per-field held/seen recall over the 5 fresh instances.
#
# Venue: Kaggle T4, Internet on. NOTE: this script lives on `master`, not the frozen
# stage-t-v2 tag, so check out master (push master to origin first — this script is not
# yet on the remote). The finetune inputs (build_scribe_data_v2.py, the m0-m4 eval
# instances, SEED, LoRA config) are byte-identical to stage-t-v2, so the adapter
# regenerates exactly; only added paper/analysis files differ. Cell:
#   %%bash
#   cd /kaggle/working && rm -rf nano-lm
#   git clone -q https://github.com/wwk5q8z6kk-bit/nano-lm
#   cd nano-lm && git checkout -q master
#   pip install -q peft && pip uninstall -y -q torchao
#   python trajectory/kaggle_pythia_fieldwise.py EleutherAI/pythia-160m   # then 410m, 1b
#   # headless: pin "machine_shape": "NvidiaTeslaT4" in kernel-metadata.json
import json, os, random, re, sys, time
import numpy as np
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    import peft
    from peft import LoraConfig, get_peft_model
except ImportError:
    sys.exit("peft required")

SEED = 20260717
LR = 1e-4; EPOCHS = 3; MICRO_BATCH = 8; ACCUM = 4
MAX_LEN = 448; MAX_NEW = 64
LORA = {"r": 16, "alpha": 32, "dropout": 0.0,
        "targets": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]}
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.environ.get("ARM1_OUT", "/kaggle/working")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-160m"
TAG = MODEL.split("/")[-1]
dev = "cuda"; assert torch.cuda.is_available()

# ---- training text: exec v2 generator prefix verbatim (identical to arm1_v2) ----
v2_src = open(os.path.join(REPO, "scribe", "build_scribe_data_v2.py")).read()
prefix = v2_src.split('tok = Tokenizer.from_file')[0].replace("from tokenizers import Tokenizer", "")
ns = {}; exec(compile(prefix, "v2[prefix]", "exec"), ns)
convos = ns["convos"]; assert len(convos) == 12000
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
EOS = tok.eos_token_id

def encode_example(convo):
    p_ids = tok.encode(convo[0]["content"] + "\n")
    t_ids = tok.encode(convo[1]["content"]) + [EOS]
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids

examples, dropped = [], 0
for c in convos:
    ids, labels = encode_example(c)
    if len(ids) > MAX_LEN: dropped += 1; continue
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
def fieldwise(model, items):
    model.eval(); tok.padding_side = "left"
    acc = {f: [0, 0, 0, 0] for f in FIELDS}          # held_correct, held_total, seen_correct, seen_total
    parsed = 0
    for i in range(0, len(items), MICRO_BATCH):
        chunk = items[i:i + MICRO_BATCH]
        enc = tok([it["convo"][0]["content"] + "\n" for it in chunk],
                  return_tensors="pt", padding=True).to(dev)
        gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                             eos_token_id=EOS, pad_token_id=EOS)
        for r, it in enumerate(chunk):
            text = tok.decode(gen[r, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            mm = RE.match(text)
            if not mm: continue
            parsed += 1
            pred = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
            for f in FIELDS:
                hit = int(pred[f] == it["tuple"][f])
                if it["held_values"]: acc[f][0] += hit; acc[f][1] += 1
                else: acc[f][2] += hit; acc[f][3] += 1
    return acc, parsed

fresh = [json.load(open(os.path.join(REPO, "trajectory", f"scribe_eval_m{k}.json"))) for k in range(5)]

# ---- LoRA finetune (identical to arm1_v2 -> regenerates the frozen adapter) ----
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
model = get_peft_model(model, LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"],
        lora_dropout=LORA["dropout"], target_modules=LORA["targets"], task_type="CAUSAL_LM"))
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                        lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
t0 = time.time(); step = 0
total_steps = (len(examples) // (MICRO_BATCH * ACCUM)) * EPOCHS
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

# ---- per-field measurement over the 5 fresh instances ----
print(f"\n=== FIELDWISE ({TAG}) ===", flush=True)
per_inst = []
for k in range(5):
    acc, parsed = fieldwise(model, fresh[k])
    gaps = {f: (acc[f][2] / max(1, acc[f][3]) - acc[f][0] / max(1, acc[f][1])) * 100 for f in FIELDS}
    per_inst.append(gaps)
    print(f"  m{k} parsed {parsed}/{len(fresh[k])}  " +
          " ".join(f"{f}:{gaps[f]:.1f}" for f in FIELDS), flush=True)

out = {}
for f in FIELDS:
    vals = [pi[f] for pi in per_inst]
    out[f] = {"gap_mean": float(np.mean(vals)), "gap_sd": float(np.std(vals, ddof=1)),
              "per_instance": vals}
print("\n  per-field gap (mean +/- SD over m0-m4):", flush=True)
for f in FIELDS:
    print(f"    {f:4s}  {out[f]['gap_mean']:6.1f} +/- {out[f]['gap_sd']:4.1f}", flush=True)

results = {
    "stage": "T-v2-fieldwise", "model": MODEL, "tag": TAG,
    "note": "per-field held-out gap (seen-value recall - held-value recall) over 5 fresh instances",
    "seeds": {"harness": SEED, "fresh_instances": [20260720, 20260721, 20260722, 20260723, 20260724]},
    "hparams": {"lr": LR, "epochs": EPOCHS, "micro_batch": MICRO_BATCH, "accum": ACCUM, "lora": LORA},
    "per_field": out, "per_instance": per_inst,
    "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                 "peft": peft.__version__, "python": sys.version.split()[0]},
    "gpu": torch.cuda.get_device_name(0), "train_wall_secs": round(train_secs),
}
outp = os.path.join(OUTDIR, f"results_fieldwise_pythia_{TAG}.json")
json.dump(results, open(outp, "w"), indent=1)
print(f"\nresults -> {outp}", flush=True)
