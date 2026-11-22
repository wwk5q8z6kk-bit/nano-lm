"""Prepare standard zero-shot benchmarks for our models: download, format, tokenize once, save one small file.

Formats follow the lm-evaluation-harness conventions so that scores can be set beside published small models:
  hellaswag       Rowan/hellaswag, validation      context "<activity>: <ctx_a> <Ctx_b>", 4 endings          acc, acc_norm
  arc_easy        allenai/ai2_arc ARC-Easy, test    "Question: <q>\nAnswer:" + " <choice>"                     acc, acc_norm
  arc_challenge   allenai/ai2_arc ARC-Challenge     same                                                       acc, acc_norm
  piqa            ybisk/piqa (parquet), validation "Question: <goal>\nAnswer:" + " <solution>"                acc, acc_norm
  winogrande      allenai/winogrande xl, validation option substituted into the prefix; score the shared suffix  acc
  openbookqa      allenai/openbookqa main, test     "<question_stem>" + " <choice>"                            acc, acc_norm
  lambada         EleutherAI/lambada_openai, test   last word given the passage: greedy accuracy and perplexity
Licences as stated on the dataset cards: MIT (HellaSwag), CC BY-SA 4.0 (ARC), AFL 3.0 (PIQA), CC BY (WinoGrande),
Apache 2.0 (OpenBookQA), modified MIT (LAMBADA, OpenAI version). Evaluation use only; nothing here is trained on.

usage: eval_prep.py --tokenizer <tokenizer_32k.json> --out <file.json> [--limit N]
Output: {task: [{"ctx": [ids], "conts": [[ids], ...], "cont_chars": [n, ...], "gold": k}, ...]}
"""
from __future__ import annotations
import argparse, json, re
from datasets import load_dataset
from tokenizers import Tokenizer


def hs_pre(t):
    t = t.strip().replace(" [title]", ". "); t = re.sub(r"\[.*?\]", "", t); return t.replace("  ", " ")


def tasks(limit):
    out = {}
    d = load_dataset("Rowan/hellaswag", split="validation")
    out["hellaswag"] = [(hs_pre(x["activity_label"] + ": " + x["ctx_a"] + " " + x["ctx_b"].capitalize()), [" " + hs_pre(e) for e in x["endings"]], int(x["label"])) for x in d]
    for name, cfg in (("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge")):
        d = load_dataset("allenai/ai2_arc", cfg, split="test")
        out[name] = [("Question: " + x["question"] + "\nAnswer:", [" " + c for c in x["choices"]["text"]], x["choices"]["label"].index(x["answerKey"])) for x in d]
    try:
        d = load_dataset("ybisk/piqa", split="validation", revision="refs/convert/parquet")
        out["piqa"] = [("Question: " + x["goal"] + "\nAnswer:", [" " + x["sol1"], " " + x["sol2"]], int(x["label"])) for x in d]
    except Exception as e:
        print("piqa unavailable:", str(e)[:200])
    d = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    wg = []
    for x in d:
        i = x["sentence"].index("_"); suffix = " " + x["sentence"][i + 1:].strip()
        wg.append(([x["sentence"][:i] + x["option1"], x["sentence"][:i] + x["option2"]], suffix, int(x["answer"]) - 1))
    out["winogrande"] = wg
    d = load_dataset("allenai/openbookqa", "main", split="test")
    out["openbookqa"] = [(x["question_stem"], [" " + c for c in x["choices"]["text"]], x["choices"]["label"].index(x["answerKey"].strip())) for x in d]
    d = load_dataset("EleutherAI/lambada_openai", "en", split="test")
    out["lambada"] = [(x["text"].rsplit(" ", 1)[0], [" " + x["text"].rsplit(" ", 1)[1]], 0) for x in d]
    return {k: v[:limit] if limit else v for k, v in out.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); tk = Tokenizer.from_file(a.tokenizer); enc = lambda s: tk.encode(s, add_special_tokens=False).ids
    res = {}
    for name, items in tasks(a.limit).items():
        rows = []
        for ctx, conts, gold in items:
            if name == "winogrande":        # two prefixes, one shared continuation
                rows.append({"ctxs": [enc(c) for c in ctx], "cont": enc(conts), "gold": gold})
            else:
                rows.append({"ctx": enc(ctx), "conts": [enc(c) for c in conts], "cont_chars": [len(c) for c in conts], "gold": gold})
        res[name] = rows; print(name, len(rows))
    json.dump(res, open(a.out, "w"))
