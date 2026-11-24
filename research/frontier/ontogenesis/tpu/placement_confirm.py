"""Confirmatory test of attention placement in the converted SmolLM2-135M hybrid (RESEARCH-PLAN.md section 8, Candidate 1).
Registered before the run. Exploratory stage A2 (results/stage-a2) ranked layers by how much the loss rises when only that
layer is converted, and found that keeping the 8 most harmful to convert as attention scored best on the SAME 16 held-out
sequences used for the ranking. That comparison is optimistic (selection on the evaluation tokens). This test uses the
inherit-trained layer weights saved by stage A2 and evaluates on text never used for training, ranking or evaluation.

Fresh evaluation text: 32 sequences x 1,024 SmolLM2 tokens from the corpus slice starting at token 120,000,000 (stage A used
tokens from 5,000,000). Placements, 22 layers converted and 8 kept as attention (the saved layers, no further training):
  most_harmful   layers 1, 2, 3, 4, 21, 22, 23, 24  (fixed from stage A2; not recomputed)
  highest_error  layers 5, 6, 7, 8, 9, 20, 23, 26   (fixed from stage A)
  least_harmful  layers 0, 6, 7, 10, 11, 12, 28, 29 (fixed from stage A2)
  random         20 draws, seeds 200-219, 8 layers each (the baseline spread)
Plus the teacher and the all-converted model. The same six models are also scored on the 16 selection sequences for comparison.

Registered predictions, judged on the fresh text only (paired on the same tokens):
  P1  most_harmful has a lower loss than at least 19 of the 20 random draws AND is at least 0.10 nats below the random mean
  P2  least_harmful has a higher loss than the random mean
  P3  highest_error lies inside the range (min to max) of the 20 random draws
A failed prediction is reported as failed. One seed of layer training underlies all placements; the spread over placements
is measured, the spread over training seeds is not.
"""
import os, sys, json, glob, random, subprocess, traceback
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

LOCAL = os.environ.get("NANO_LOCAL") == "1"
OUT = os.environ.get("NANO_OUT", "/kaggle/working/placement_confirm.json")
DEV = "cpu" if LOCAL else "cuda"
T = 64 if LOCAL else 1024
N_FRESH, N_SEL = (4, 4) if LOCAL else (32, 16)
R = {"config": {"T": T, "n_fresh": N_FRESH, "n_selection": N_SEL, "local": LOCAL}}


def save(): json.dump(R, open(OUT, "w"), indent=1, default=str)


if not LOCAL:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention", "fla-core"], capture_output=True, text=True)


def make_gdn(hidden, heads, head_dim):
    if LOCAL:
        class G(nn.Module):
            def __init__(s):
                super().__init__(); s.lin = nn.Linear(hidden, hidden, bias=False)
            def forward(s, x, **kw): return s.lin(x) * 0.0, None, None
        return G().to(DEV)
    from fla.layers import GatedDeltaNet
    return GatedDeltaNet(hidden_size=hidden, num_heads=heads, head_dim=head_dim, expand_v=1, mode="chunk").to(DEV)


def windows(tok_teacher, start, n_seq):
    from tokenizers import Tokenizer
    root = os.environ.get("NANO_DATA") or (glob.glob("/kaggle/input/**/train.bin", recursive=True) or [""])[0].rsplit("/", 1)[0]
    ours = Tokenizer.from_file(f"{root}/tokenizer_32k.json"); data = np.memmap(f"{root}/train.bin", dtype=np.uint16, mode="r")
    need = n_seq * T; ids = []; pos = start % (len(data) - 10 ** 6)
    while len(ids) < need + T:
        ids += tok_teacher(ours.decode(data[pos:pos + 1500].astype("int64").tolist()), add_special_tokens=False)["input_ids"] + [tok_teacher.eos_token_id]; pos += 1500
    return torch.tensor(ids[:need]).view(n_seq, T)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "HuggingFaceTB/SmolLM2-135M"; tok = AutoTokenizer.from_pretrained(name)
    teacher = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(DEV).eval(); cfg = teacher.config
    H, hd, D, L = cfg.num_attention_heads, cfg.hidden_size // cfg.num_attention_heads, cfg.hidden_size, cfg.num_hidden_layers
    fresh = windows(tok, 120_000_000, N_FRESH)
    sel = windows(tok, 5_000_000, 64 + N_SEL)[-N_SEL:] if not LOCAL else windows(tok, 5_000_000, 8 + N_SEL)[-N_SEL:]
    mods = {}
    files = glob.glob("/kaggle/input/**/inherit_layers.pt", recursive=True)
    sd = torch.load(files[0], map_location=DEV, weights_only=True) if files and not LOCAL else None   # tensors only
    for i in range(L):
        g = make_gdn(D, H, hd)
        if sd is not None: g.load_state_dict(sd[f"inherit_{i}"])
        mods[i] = g.eval()
    R["weights_loaded"] = sd is not None; R["weights_file"] = files[0] if files else None

    class Wrap(nn.Module):
        def __init__(s, gdn): super().__init__(); s.gdn = gdn
        def forward(s, hidden_states=None, *a, **k): return s.gdn(hidden_states)[0], None
    orig = {i: teacher.model.layers[i].self_attn for i in range(L)}

    @torch.no_grad()
    def loss(replace, data):
        for i in range(L): teacher.model.layers[i].self_attn = Wrap(replace[i]) if i in replace else orig[i]
        tot = n = 0.0
        with torch.autocast(DEV, dtype=torch.float16, enabled=not LOCAL):
            for b in range(0, data.shape[0], 4):
                ids = data[b:b + 4].to(DEV); lg = teacher(ids, use_cache=False).logits.float()
                tot += float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1), reduction="sum")); n += ids[:, 1:].numel()
        for i in range(L): teacher.model.layers[i].self_attn = orig[i]
        return tot / n

    named = {"most_harmful": [1, 2, 3, 4, 21, 22, 23, 24], "highest_error": [5, 6, 7, 8, 9, 20, 23, 26], "least_harmful": [0, 6, 7, 10, 11, 12, 28, 29]}
    for d in range(20): named[f"random_{d:02d}"] = sorted(random.Random(200 + d).sample(range(L), 8))
    res = {"teacher": {"fresh": loss({}, fresh), "selection": loss({}, sel)}, "all_converted": {"fresh": loss(mods, fresh), "selection": loss(mods, sel)}}
    for k, keep in named.items():
        rep = {i: m for i, m in mods.items() if i not in keep}
        res[k] = {"kept": keep, "fresh": round(loss(rep, fresh), 4), "selection": round(loss(rep, sel), 4)}
        R["results"] = res; save()
    rnd = [res[k]["fresh"] for k in res if k.startswith("random_")]; mean = float(np.mean(rnd)); m = res["most_harmful"]["fresh"]
    R["predictions"] = {
        "P1": {"held": bool(sum(m < r for r in rnd) >= 19 and m <= mean - 0.10), "most_harmful": m, "random_mean": round(mean, 4), "random_sd": round(float(np.std(rnd)), 4), "draws_beaten": int(sum(m < r for r in rnd)), "of": len(rnd)},
        "P2": {"held": bool(res["least_harmful"]["fresh"] > mean), "least_harmful": res["least_harmful"]["fresh"], "random_mean": round(mean, 4)},
        "P3": {"held": bool(min(rnd) <= res["highest_error"]["fresh"] <= max(rnd)), "highest_error": res["highest_error"]["fresh"], "random_range": [round(min(rnd), 4), round(max(rnd), 4)]}}
    save(); print(json.dumps(R["predictions"]), flush=True)


try:
    main()
except Exception as e:
    R["fatal"] = {"error": repr(e)[:500], "trace": traceback.format_exc()[-1500:]}; save(); print("FATAL", repr(e)[:300])
