#!/usr/bin/env python3
"""RunPod CUDA fp16 official M0 for E1 kill-gate.

Trains + scores on the E1 harness:
  1) Own-stack Chinchilla-160M + LoRA (r=16, FT_SEED=0) — matches kaggle_ownstack_160m_lora.py
  2) Pythia-160M LoRA (SEED=20260717) — matches kaggle_arm1_v2.py

Then writes/updates results_e1_utility.json with KILL/SURVIVE vs M1–M5.

Expects cwd layout:
  /workspace/nano-lm/   (repo root)
  checkpoints/chinchilla-160m/ownstack160m_pretrain.pt
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "CUDA required (RunPod)"
dev = "cuda"

REPO = Path(os.environ.get("E1_REPO", "/workspace/nano-lm")).resolve()
sys.path.insert(0, str(REPO))

from trajectory.e1.common import (  # noqa: E402
    FIELDS,
    FieldPred,
    ItemPred,
    aggregate_decision,
    evaluate_method,
    load_instances,
    pred_from_values,
)

OUT = REPO / "checkpoints" / "e1_official_m0"
OUT.mkdir(parents=True, exist_ok=True)
TRAJ = REPO / "trajectory"
COST = {
    "M0_pythia160m_lora": 1.2,
    "M0_ownstack_chinchilla_lora": 1.1,
}


# ---------------- Own-stack LoRA (CUDA fp16) ----------------

def train_ownstack_lora(ft_seed: int = 0):
    from peft import LoraConfig, inject_adapter_in_model
    from tokenizers import Tokenizer

    out = OUT / f"ownstack160m_chinchilla_lora_seed{ft_seed}.pt"
    if out.exists():
        print(f"[ownstack] reuse {out}", flush=True)
        return out

    torch.manual_seed(ft_seed)
    random.seed(ft_seed)
    tok = Tokenizer.from_file(str(REPO / "sft" / "tokenizer.json"))
    IMS = tok.token_to_id("<|im_start|>")
    IME = tok.token_to_id("<|im_end|>")
    V = 4098
    d, L, H, KV, hd, ff, S = 1024, 14, 16, 4, 64, 2752, 512
    MICRO, ACCUM = 8, 4
    BATCH = MICRO * ACCUM

    def rope(q, k):
        t = torch.arange(S, device=dev, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        f = torch.outer(t, inv)
        cos, sin = f.cos()[None, None], f.sin()[None, None]

        def rot(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

        return rot(q), rot(k)

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
            s.q = nn.Linear(d, H * hd, bias=False)
            s.k = nn.Linear(d, KV * hd, bias=False)
            s.v = nn.Linear(d, KV * hd, bias=False)
            s.o = nn.Linear(H * hd, d, bias=False)
            s.g = nn.Linear(d, ff, bias=False)
            s.u = nn.Linear(d, ff, bias=False)
            s.dn = nn.Linear(ff, d, bias=False)

        def forward(s, x):
            B = x.shape[0]
            h = s.n1(x)
            q = s.q(h).view(B, S, H, hd).transpose(1, 2)
            k = s.k(h).view(B, S, KV, hd).transpose(1, 2)
            v = s.v(h).view(B, S, KV, hd).transpose(1, 2)
            q, k = rope(q, k)
            k = k.repeat_interleave(H // KV, 1)
            v = v.repeat_interleave(H // KV, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + s.o(a.transpose(1, 2).reshape(B, S, H * hd))
            h = s.n2(x)
            return x + s.dn(F.silu(s.g(h)) * s.u(h))

    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(V, d)
            s.blocks = nn.ModuleList(Block() for _ in range(L))
            s.nf = nn.RMSNorm(d)

        def forward(s, x):
            h = s.emb(x)
            for b in s.blocks:
                h = b(h)
            return F.linear(s.nf(h), s.emb.weight)

    ckpt = REPO / "checkpoints" / "chinchilla-160m" / "ownstack160m_pretrain.pt"
    assert ckpt.exists(), ckpt
    m = GPT()
    st = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(st["m"])
    m.to(dev)
    print(f"[ownstack] loaded pretrain step={st.get('step')}", flush=True)

    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q", "k", "v", "o", "g", "u", "dn"],
    )
    m = inject_adapter_in_model(cfg, m)
    for n, p in m.named_parameters():
        p.requires_grad = "lora_" in n
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[ownstack] LoRA trainables {trainable/1e6:.3f}M", flush=True)

    v2 = (REPO / "scribe" / "build_scribe_data_v2.py").read_text()
    prefix = v2.split("tok = Tokenizer.from_file")[0].replace(
        "from tokenizers import Tokenizer", ""
    )
    ns = {}
    exec(compile(prefix, "v2", "exec"), ns)
    convos = ns["convos"]
    assert len(convos) == 12000

    def render_ids(convo):
        ids, mask = [], []
        for msg in convo:
            head = tok.encode(f"{msg['role']}\n", add_special_tokens=False).ids
            body = tok.encode(msg["content"], add_special_tokens=False).ids
            seg = [IMS] + head + body + [IME]
            tf = 1 if msg["role"] == "assistant" else 0
            for j, tkn in enumerate(seg):
                ids.append(tkn)
                mask.append(1 if (tf and j >= 1 + len(head)) else 0)
        return ids, mask

    X, Msk = [], []
    for c in convos:
        ids, mask = render_ids(c)
        if len(ids) > S:
            continue
        X.append(ids + [0] * (S - len(ids)))
        Msk.append(mask + [0] * (S - len(mask)))
    X = torch.tensor(np.array(X, dtype=np.int64))
    Msk = torch.tensor(np.array(Msk, dtype=np.int64))
    N = X.shape[0]
    FSTEPS = (N // BATCH) * 3
    FWARM = int(0.03 * FSTEPS)
    FLR = 1e-4
    opt = torch.optim.AdamW(
        (p for p in m.parameters() if p.requires_grad),
        lr=FLR, betas=(0.9, 0.95), weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda")
    perm = torch.randperm(N)

    def flr_at(t):
        if t < FWARM:
            return FLR * t / max(1, FWARM)
        p = (t - FWARM) / max(1, FSTEPS - FWARM)
        return FLR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    t0 = time.time()
    m.train()
    for step in range(1, FSTEPS + 1):
        for g in opt.param_groups:
            g["lr"] = flr_at(step)
        i0 = (step * BATCH) % (N - BATCH)
        idx = perm[i0 : i0 + BATCH]
        opt.zero_grad(set_to_none=True)
        for a in range(ACCUM):
            sub = idx[a * MICRO : (a + 1) * MICRO]
            x, mt = X[sub].to(dev), Msk[sub].to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = m(x)[:, :-1]
                tgt = x[:, 1:]
                mtgt = mt[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, V), tgt.reshape(-1), reduction="none"
                ).reshape(tgt.shape)
                loss = (ce * mtgt).sum() / mtgt.sum().clamp(min=1)
            scaler.scale(loss / ACCUM).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            (p for p in m.parameters() if p.requires_grad), 1.0
        )
        scaler.step(opt)
        scaler.update()
        if step % 100 == 0 or step == 1:
            print(
                f"[ownstack] {step}/{FSTEPS} loss={loss.item():.3f} "
                f"({(time.time()-t0)/60:.1f} min)",
                flush=True,
            )
    print(f"[ownstack] done {(time.time()-t0)/60:.1f} min", flush=True)
    torch.save({"m": m.state_dict(), "ft_seed": ft_seed, "kind": "ownstack_lora", "venue": "runpod-cuda"}, out)
    return out


def load_ownstack_predictor(ckpt_path: Path):
    from peft import LoraConfig, inject_adapter_in_model
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(REPO / "sft" / "tokenizer.json"))
    IMS = tok.token_to_id("<|im_start|>")
    IME = tok.token_to_id("<|im_end|>")
    V = 4098
    d, L, H, KV, hd, ff, S = 1024, 14, 16, 4, 64, 2752, 512

    def rope(q, k):
        t = torch.arange(S, device=dev, dtype=torch.float32)
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=dev).float() / hd))
        f = torch.outer(t, inv)
        cos, sin = f.cos()[None, None], f.sin()[None, None]

        def rot(x):
            x1, x2 = x[..., 0::2], x[..., 1::2]
            return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

        return rot(q), rot(k)

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.n1, s.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
            s.q = nn.Linear(d, H * hd, bias=False)
            s.k = nn.Linear(d, KV * hd, bias=False)
            s.v = nn.Linear(d, KV * hd, bias=False)
            s.o = nn.Linear(H * hd, d, bias=False)
            s.g = nn.Linear(d, ff, bias=False)
            s.u = nn.Linear(d, ff, bias=False)
            s.dn = nn.Linear(ff, d, bias=False)

        def forward(s, x):
            B = x.shape[0]
            h = s.n1(x)
            q = s.q(h).view(B, S, H, hd).transpose(1, 2)
            k = s.k(h).view(B, S, KV, hd).transpose(1, 2)
            v = s.v(h).view(B, S, KV, hd).transpose(1, 2)
            q, k = rope(q, k)
            k = k.repeat_interleave(H // KV, 1)
            v = v.repeat_interleave(H // KV, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + s.o(a.transpose(1, 2).reshape(B, S, H * hd))
            h = s.n2(x)
            return x + s.dn(F.silu(s.g(h)) * s.u(h))

    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(V, d)
            s.blocks = nn.ModuleList(Block() for _ in range(L))
            s.nf = nn.RMSNorm(d)

        def forward(s, x):
            h = s.emb(x)
            for b in s.blocks:
                h = b(h)
            return F.linear(s.nf(h), s.emb.weight)

    m = GPT()
    m = inject_adapter_in_model(
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                   target_modules=["q", "k", "v", "o", "g", "u", "dn"]),
        m,
    )
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m.load_state_dict(st["m"], strict=True)
    m.to(dev).eval()

    def prompt_ids(user_content):
        return (
            [IMS] + tok.encode("user\n", add_special_tokens=False).ids
            + tok.encode(user_content, add_special_tokens=False).ids + [IME]
            + [IMS] + tok.encode("assistant\n", add_special_tokens=False).ids
        )

    @torch.no_grad()
    def generate(ids, max_new=64):
        ids = list(ids)
        for _ in range(max_new):
            if len(ids) >= S:
                break
            x = torch.tensor([ids + [0] * (S - len(ids))], device=dev)
            nxt = int(m(x)[0, len(ids) - 1].argmax())
            if nxt == IME:
                break
            ids.append(nxt)
        return ids

    RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")

    def predict(item, source_id):
        t0 = time.perf_counter()
        out = generate(prompt_ids(item["convo"][0]["content"]))
        text = tok.decode(out[len(prompt_ids(item["convo"][0]["content"])):]).strip()
        # recompute prompt length cleanly
        pids = prompt_ids(item["convo"][0]["content"])
        text = tok.decode(out[len(pids):]).strip()
        mm = RE.match(text)
        if not mm:
            return ItemPred(
                fields={f: FieldPred("none") for f in FIELDS},
                latency_s=time.perf_counter() - t0,
                raw=text,
                parsed=False,
            )
        return pred_from_values(dict(zip(FIELDS, [g.strip() for g in mm.groups()])),
                                latency_s=time.perf_counter() - t0)

    return predict


# ---------------- Pythia LoRA (CUDA fp16) ----------------

def train_pythia_lora():
    # peft trap on some images
    os.system("pip uninstall -y -q torchao 2>/dev/null")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = OUT / "pythia160m_lora"
    if (out_dir / "adapter_config.json").exists():
        print(f"[pythia] reuse {out_dir}", flush=True)
        return out_dir

    SEED = 20260717
    LR = 1e-4
    EPOCHS = 3
    MICRO_BATCH = 8
    ACCUM = 4
    MAX_LEN = 448
    LORA = {
        "r": 16, "alpha": 32, "dropout": 0.0,
        "targets": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    }
    MODEL = "EleutherAI/pythia-160m"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    v2 = (REPO / "scribe" / "build_scribe_data_v2.py").read_text()
    prefix = v2.split("tok = Tokenizer.from_file")[0].replace(
        "from tokenizers import Tokenizer", ""
    )
    ns = {}
    exec(compile(prefix, "v2", "exec"), ns)
    convos = ns["convos"]
    assert len(convos) == 12000

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.pad_token = tok.eos_token
    EOS = tok.eos_token_id

    def encode_example(convo):
        p_ids = tok.encode(convo[0]["content"] + "\n")
        t_ids = tok.encode(convo[1]["content"]) + [EOS]
        return p_ids + t_ids, [-100] * len(p_ids) + t_ids

    examples, dropped = [], 0
    for c in convos:
        ids, labels = encode_example(c)
        if len(ids) > MAX_LEN:
            dropped += 1
            continue
        examples.append((ids, labels))
    print(f"[pythia] examples {len(examples)} dropped {dropped}", flush=True)

    def batches(data, bs):
        idx = list(range(len(data)))
        random.shuffle(idx)
        for i in range(0, len(idx) - bs + 1, bs):
            chunk = [data[j] for j in idx[i : i + bs]]
            Lmax = max(len(x[0]) for x in chunk)
            x = torch.full((bs, Lmax), EOS, dtype=torch.long)
            y = torch.full((bs, Lmax), -100, dtype=torch.long)
            m = torch.zeros((bs, Lmax), dtype=torch.long)
            for r, (ids, labels) in enumerate(chunk):
                x[r, : len(ids)] = torch.tensor(ids)
                y[r, : len(labels)] = torch.tensor(labels)
                m[r, : len(ids)] = 1
            yield x.to(dev), y.to(dev), m.to(dev)

    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"],
            target_modules=LORA["targets"], task_type="CAUSAL_LM",
        ),
    )
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR, betas=(0.9, 0.95), weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda")
    total_steps = (len(examples) // (MICRO_BATCH * ACCUM)) * EPOCHS
    t0 = time.time()
    step = 0
    model.train()
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
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0 or step == 1:
                    print(
                        f"[pythia] ep{ep} {step}/{total_steps} loss={out.loss.item():.3f} "
                        f"({(time.time()-t0)/60:.1f} min)",
                        flush=True,
                    )
    print(f"[pythia] done {(time.time()-t0)/60:.1f} min", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "e1_train_meta.json").write_text(json.dumps({
        "model": MODEL, "seed": SEED, "lora": LORA, "venue": "runpod-cuda",
        "train_secs": round(time.time() - t0),
    }, indent=2))
    return out_dir


def load_pythia_predictor(adapter_dir: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "EleutherAI/pythia-160m"
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    tok.pad_token = tok.eos_token
    EOS = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(dev).eval()
    RE = re.compile(r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$")

    @torch.no_grad()
    def predict(item, source_id):
        t0 = time.perf_counter()
        enc = tok(item["convo"][0]["content"] + "\n", return_tensors="pt").to(dev)
        gen = model.generate(
            **enc, max_new_tokens=64, do_sample=False,
            eos_token_id=EOS, pad_token_id=EOS,
        )
        text = tok.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        mm = RE.match(text)
        if not mm:
            return ItemPred(
                fields={f: FieldPred("none") for f in FIELDS},
                latency_s=time.perf_counter() - t0,
                raw=text,
                parsed=False,
            )
        return pred_from_values(dict(zip(FIELDS, [g.strip() for g in mm.groups()])),
                                latency_s=time.perf_counter() - t0)

    return predict


def eval_merge(name, predict_fn, cost, util):
    instances = load_instances()
    for verify_on in (False, True):
        print(f"E1 {name} verify={'on' if verify_on else 'off'}", flush=True)
        res = evaluate_method(name, predict_fn, instances, verify_on=verify_on, cost_c=cost)
        out = {k: v for k, v in res.items() if k != "item_logs"}
        out["prereg"] = "trajectory/PREREG_E1_nonlm_baseline.md"
        out["official_m0"] = True
        out["venue"] = "runpod-cuda"
        (TRAJ / f"results_e1_nonlm_{name}_v{'on' if verify_on else 'off'}.json").write_text(
            json.dumps(out, indent=2)
        )
        (TRAJ / f"results_e1_items_{name}_v{'on' if verify_on else 'off'}.json").write_text(
            json.dumps(res["item_logs"])
        )
        s = res["summary"]
        print(
            f"  U={s['U']:.4f} P={s['P']:.3f} rec={s['recall']:.3f} "
            f"gap={s['gap_pts']:.2f} rho={s['rho']:.3f}",
            flush=True,
        )
        util["rows"] = [r for r in util["rows"] if r["method"] != name]
        util["rows"].append({
            "method": name,
            "verify_on": verify_on,
            **{k: s[k] for k in (
                "U", "P", "M", "rho", "L_p50", "C", "recall", "halluc",
                "gap_pts", "held_recall", "seen_recall", "correct_norm_rate",
                "liability_presented_bad",
            )},
            "U_sensitivity": s["U_sensitivity"],
        })
    return util


def finalize(util):
    cands = ["M0_pythia160m_lora", "M0_ownstack_chinchilla_lora"]
    rows = {r["method"]: r for r in util["rows"] if r["verify_on"]}
    present = [c for c in cands if c in rows]
    assert len(present) == 2, present
    m0 = max(present, key=lambda n: rows[n]["U"])
    fake = {
        r["method"]: {
            "verify_on": True,
            "summary": {"U": r["U"], "U_sensitivity": r["U_sensitivity"]},
        }
        for r in util["rows"] if r["verify_on"]
    }
    decision = aggregate_decision(fake, m0)
    decision["official_m0_candidates"] = {n: rows[n]["U"] for n in present}
    decision["provisional_m0_scale"] = rows.get("M0_scale", {}).get("U")
    decision["venue"] = "runpod-cuda"
    util["decision"] = decision
    util["note"] = (
        "Official M0 = argmax U(Pythia-160M LoRA, ownstack Chinchilla+LoRA) "
        "on RunPod CUDA fp16; KILL/SURVIVE under PREREG_E1."
    )
    return util


def main():
    # peft/torchao trap
    os.system("pip install -q transformers==4.44.2 peft==0.12.0 accelerate==0.33.0 tokenizers 2>/dev/null; pip uninstall -y -q torchao 2>/dev/null")
    util = json.loads((TRAJ / "results_e1_utility.json").read_text())

    print("=== ARM 1: ownstack Chinchilla+LoRA ===", flush=True)
    ckpt = train_ownstack_lora(0)
    util = eval_merge(
        "M0_ownstack_chinchilla_lora",
        load_ownstack_predictor(ckpt),
        COST["M0_ownstack_chinchilla_lora"],
        util,
    )
    (TRAJ / "results_e1_utility.json").write_text(json.dumps(util, indent=2))

    print("=== ARM 2: Pythia-160M LoRA ===", flush=True)
    adir = train_pythia_lora()
    util = eval_merge(
        "M0_pythia160m_lora",
        load_pythia_predictor(adir),
        COST["M0_pythia160m_lora"],
        util,
    )
    util = finalize(util)
    (TRAJ / "results_e1_utility.json").write_text(json.dumps(util, indent=2))
    print(json.dumps(util["decision"], indent=2), flush=True)
    print("E1_OFFICIAL_M0_DONE", flush=True)


if __name__ == "__main__":
    main()
