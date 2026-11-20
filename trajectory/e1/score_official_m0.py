#!/usr/bin/env python3
"""Close E1 kill-gate: score official M0 candidates on the E1 harness.

Official M0 = max_U(Pythia-160M LoRA, own-stack Chinchilla-160M + LoRA corner)
under identical utility / instrument as M1–M5.

Regenerates adapters with frozen seeds/hparams from:
  trajectory/kaggle_arm1_v2.py
  trajectory/kaggle_ownstack_160m_lora.py
then evaluates via trajectory.e1.common.evaluate_method.
"""
from __future__ import annotations

import argparse
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

REPO = Path(__file__).resolve().parents[2]
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
from trajectory.e1.methods import COST_C  # noqa: E402

OUT_CKPT = REPO / "checkpoints" / "e1_official_m0"
OUT_CKPT.mkdir(parents=True, exist_ok=True)
TRAJ = REPO / "trajectory"


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Own-stack GPT + LoRA (corner / chinchilla base)
# ---------------------------------------------------------------------------

def train_ownstack_lora(ft_seed: int = 0, force: bool = False):
    from peft import LoraConfig, inject_adapter_in_model
    from tokenizers import Tokenizer

    out = OUT_CKPT / f"ownstack160m_chinchilla_lora_seed{ft_seed}.pt"
    if out.exists() and not force:
        print(f"[ownstack-lora] reuse {out}", flush=True)
        return out

    dev = device()
    torch.manual_seed(ft_seed)
    random.seed(ft_seed)
    tok = Tokenizer.from_file(str(REPO / "sft" / "tokenizer.json"))
    IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    V = 4098
    d, L, H, KV, hd, ff, S = 1024, 14, 16, 4, 64, 2752, 512
    MICRO, ACCUM = 4, 8  # same effective batch 32; smaller micro for MPS memory
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

    ckpt_path = REPO / "checkpoints" / "chinchilla-160m" / "ownstack160m_pretrain.pt"
    assert ckpt_path.exists(), ckpt_path
    m = GPT()
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m.load_state_dict(st["m"])
    m.to(dev)
    print(f"[ownstack-lora] loaded pretrain step={st.get('step')} on {dev}", flush=True)

    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q", "k", "v", "o", "g", "u", "dn"],
    )
    m = inject_adapter_in_model(cfg, m)
    for n, p in m.named_parameters():
        p.requires_grad = "lora_" in n
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[ownstack-lora] trainables {trainable/1e6:.3f}M", flush=True)

    v2_src = (REPO / "scribe" / "build_scribe_data_v2.py").read_text()
    prefix = v2_src.split("tok = Tokenizer.from_file")[0].replace(
        "from tokenizers import Tokenizer", ""
    )
    ns = {}
    exec(compile(prefix, "v2[prefix]", "exec"), ns)
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
            x = X[sub].to(dev)
            mt = Msk[sub].to(dev)
            logits = m(x)[:, :-1]
            tgt = x[:, 1:]
            mtgt = mt[:, 1:]
            ce = F.cross_entropy(
                logits.reshape(-1, V), tgt.reshape(-1), reduction="none"
            ).reshape(tgt.shape)
            loss = (ce * mtgt).sum() / mtgt.sum().clamp(min=1)
            (loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in m.parameters() if p.requires_grad), 1.0
        )
        opt.step()
        if step % 50 == 0 or step == 1:
            elapsed = (time.time() - t0) / 60
            print(
                f"[ownstack-lora] {step}/{FSTEPS} loss={loss.item():.3f} "
                f"({elapsed:.1f} min)",
                flush=True,
            )
    print(f"[ownstack-lora] done in {(time.time()-t0)/60:.1f} min", flush=True)
    # save full state dict (base+lora)
    torch.save({"m": m.state_dict(), "ft_seed": ft_seed, "kind": "ownstack_lora"}, out)
    print(f"[ownstack-lora] wrote {out}", flush=True)
    return out


def load_ownstack_lora_predictor(ckpt_path: Path):
    from peft import LoraConfig, inject_adapter_in_model
    from tokenizers import Tokenizer

    dev = device()
    tok = Tokenizer.from_file(str(REPO / "sft" / "tokenizer.json"))
    IMS, IME = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
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
    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q", "k", "v", "o", "g", "u", "dn"],
    )
    m = inject_adapter_in_model(cfg, m)
    st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m.load_state_dict(st["m"], strict=True)
    m.to(dev).eval()

    def prompt_ids(user_content):
        return (
            [IMS]
            + tok.encode("user\n", add_special_tokens=False).ids
            + tok.encode(user_content, add_special_tokens=False).ids
            + [IME]
            + [IMS]
            + tok.encode("assistant\n", add_special_tokens=False).ids
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

    RE = re.compile(
        r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$"
    )

    def predict(item: dict, source_id: str) -> ItemPred:
        t0 = time.perf_counter()
        content = item["convo"][0]["content"]
        ids = prompt_ids(content)
        out = generate(ids)
        # decode tokens after prompt
        text = tok.decode(out[len(ids) :]).strip()
        mm = RE.match(text)
        if not mm:
            return ItemPred(
                fields={f: FieldPred("none") for f in FIELDS},
                latency_s=time.perf_counter() - t0,
                raw=text,
                parsed=False,
            )
        vals = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        return pred_from_values(vals, latency_s=time.perf_counter() - t0)

    return predict


# ---------------------------------------------------------------------------
# Pythia-160M LoRA
# ---------------------------------------------------------------------------

def train_pythia_lora(force: bool = False):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = OUT_CKPT / "pythia160m_lora"
    marker = out_dir / "adapter_model.safetensors"
    marker2 = out_dir / "adapter_config.json"
    if marker2.exists() and not force:
        print(f"[pythia-lora] reuse {out_dir}", flush=True)
        return out_dir

    SEED = 20260717
    LR = 1e-4
    EPOCHS = 3
    MICRO_BATCH = 4
    ACCUM = 8
    MAX_LEN = 448
    LORA = {
        "r": 16,
        "alpha": 32,
        "dropout": 0.0,
        "targets": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    }
    MODEL = "EleutherAI/pythia-160m"
    dev = device()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    v2_src = (REPO / "scribe" / "build_scribe_data_v2.py").read_text()
    prefix = v2_src.split("tok = Tokenizer.from_file")[0].replace(
        "from tokenizers import Tokenizer", ""
    )
    ns = {}
    exec(compile(prefix, "v2[prefix]", "exec"), ns)
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
    print(f"[pythia-lora] train examples {len(examples)} (dropped {dropped})", flush=True)

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

    print(f"[pythia-lora] loading {MODEL} on {dev}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA["r"],
            lora_alpha=LORA["alpha"],
            lora_dropout=LORA["dropout"],
            target_modules=LORA["targets"],
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=LR, betas=(0.9, 0.95), weight_decay=0.0,
    )
    total_steps = (len(examples) // (MICRO_BATCH * ACCUM)) * EPOCHS
    t0 = time.time()
    step = 0
    model.train()
    for ep in range(EPOCHS):
        micro = 0
        opt.zero_grad(set_to_none=True)
        for x, y, m in batches(examples, MICRO_BATCH):
            out = model(input_ids=x, attention_mask=m, labels=y)
            (out.loss / ACCUM).backward()
            micro += 1
            if micro % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0 or step == 1:
                    print(
                        f"[pythia-lora] ep{ep} step {step}/{total_steps} "
                        f"loss {out.loss.item():.3f} ({(time.time()-t0)/60:.1f} min)",
                        flush=True,
                    )
    print(f"[pythia-lora] done in {(time.time()-t0)/60:.1f} min", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    meta = {
        "model": MODEL,
        "seed": SEED,
        "lora": LORA,
        "trainable": trainable,
        "train_secs": round(time.time() - t0),
    }
    (out_dir / "e1_train_meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def load_pythia_predictor(adapter_dir: Path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "EleutherAI/pythia-160m"
    dev = device()
    tok = AutoTokenizer.from_pretrained(str(adapter_dir))
    tok.pad_token = tok.eos_token
    EOS = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to(dev).eval()
    RE = re.compile(
        r"^CC: (.+?) \| DUR: (.+?) \| SEV: (.+?) \| MED: (.+?) \| ALG: (.+?)$"
    )

    @torch.no_grad()
    def predict(item: dict, source_id: str) -> ItemPred:
        t0 = time.perf_counter()
        prompt = item["convo"][0]["content"] + "\n"
        enc = tok(prompt, return_tensors="pt").to(dev)
        gen = model.generate(
            **enc,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=EOS,
            pad_token_id=EOS,
        )
        text = tok.decode(gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        mm = RE.match(text)
        if not mm:
            return ItemPred(
                fields={f: FieldPred("none") for f in FIELDS},
                latency_s=time.perf_counter() - t0,
                raw=text,
                parsed=False,
            )
        vals = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        return pred_from_values(vals, latency_s=time.perf_counter() - t0)

    return predict


def eval_and_merge(name: str, predict_fn, cost_c: float):
    instances = load_instances()
    util_path = TRAJ / "results_e1_utility.json"
    util = json.loads(util_path.read_text())
    for verify_on in (False, True):
        print(f"E1-eval {name} verify={'on' if verify_on else 'off'}...", flush=True)
        res = evaluate_method(name, predict_fn, instances, verify_on=verify_on, cost_c=cost_c)
        out = {k: v for k, v in res.items() if k != "item_logs"}
        out["prereg"] = "trajectory/PREREG_E1_nonlm_baseline.md"
        out["official_m0"] = True
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
        util["rows"].append(
            {
                "method": name,
                "verify_on": verify_on,
                **{
                    k: s[k]
                    for k in (
                        "U", "P", "M", "rho", "L_p50", "C", "recall", "halluc",
                        "gap_pts", "held_recall", "seen_recall", "correct_norm_rate",
                        "liability_presented_bad",
                    )
                },
                "U_sensitivity": s["U_sensitivity"],
            }
        )
    util_path.write_text(json.dumps(util, indent=2))
    return util


def finalize_decision(util: dict):
    """Official M0 = max U among Pythia LoRA and corner LoRA (verify-on)."""
    candidates = ["M0_pythia160m_lora", "M0_ownstack_chinchilla_lora"]
    rows = {r["method"]: r for r in util["rows"] if r["verify_on"]}
    present = [c for c in candidates if c in rows]
    if len(present) < 2:
        util["decision"] = {
            "verdict": "INCOMPLETE",
            "reason": f"official M0 requires both candidates; have {present}",
        }
        return util
    m0 = max(present, key=lambda n: rows[n]["U"])
    fake = {}
    for r in util["rows"]:
        if not r["verify_on"]:
            continue
        fake[r["method"]] = {
            "verify_on": True,
            "summary": {"U": r["U"], "U_sensitivity": r["U_sensitivity"]},
        }
    decision = aggregate_decision(fake, m0)
    decision["official_m0_candidates"] = {
        n: rows[n]["U"] for n in present
    }
    decision["provisional_m0_scale"] = rows.get("M0_scale", {}).get("U")
    util["decision"] = decision
    util["note"] = (
        "Official M0 = argmax U(Pythia-160M LoRA, ownstack Chinchilla+LoRA). "
        "KILL/SURVIVE under PREREG_E1 decision rule."
    )
    return util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ownstack", "pythia", "all"], default="all")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="Only eval existing ckpts")
    args = ap.parse_args()

    COST_C["M0_pythia160m_lora"] = 1.2
    COST_C["M0_ownstack_chinchilla_lora"] = 1.1

    if args.only in ("ownstack", "all"):
        if not args.skip_train:
            ckpt = train_ownstack_lora(ft_seed=0, force=args.force_train)
        else:
            ckpt = OUT_CKPT / "ownstack160m_chinchilla_lora_seed0.pt"
            assert ckpt.exists(), ckpt
        pred = load_ownstack_lora_predictor(ckpt)
        util = eval_and_merge("M0_ownstack_chinchilla_lora", pred, COST_C["M0_ownstack_chinchilla_lora"])
    else:
        util = json.loads((TRAJ / "results_e1_utility.json").read_text())

    if args.only in ("pythia", "all"):
        if not args.skip_train:
            adir = train_pythia_lora(force=args.force_train)
        else:
            adir = OUT_CKPT / "pythia160m_lora"
            assert (adir / "adapter_config.json").exists(), adir
        pred = load_pythia_predictor(adir)
        util = eval_and_merge("M0_pythia160m_lora", pred, COST_C["M0_pythia160m_lora"])

    util = finalize_decision(util)
    (TRAJ / "results_e1_utility.json").write_text(json.dumps(util, indent=2))
    print(json.dumps(util["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
