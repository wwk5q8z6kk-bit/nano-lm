# Batched greedy scorer for the own-stack models — OPTIONAL fast path (~10x), adopted
# ONLY if byte-identical to the native token-by-token scorer on the frozen references.
# Correctness argument: the native path right-pads each prompt to S=512 with token 0 and
# reads logits at position len-1; with causal attention, right-padding beyond the read
# position is invisible, so a batched right-padded forward computes the same per-row
# logits. The residual risk is float reduction-order differences across batch shapes
# flipping argmax ties — hence the empirical gate below (exact count match required).
# Validation: nano anchor on inst0 must reproduce gate_scribe_v2.log EXACTLY
# (parse 39/40, recall 162/200, held 68/95, seen 94/100) AND per-dialogue outputs must
# equal the native scorer's, string-for-string.
import json, os, re, sys, time
import importlib.util
import torch

spec = importlib.util.spec_from_file_location("ra", os.path.join(os.path.dirname(__file__), "rescore_anchors.py"))
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
S, IME = 512, ra.IME

@torch.no_grad()
def generate_batched(m, prompt_id_lists, max_new=64, bs=16, dev=None):
    """Greedy-decode a list of prompts; returns list of generated-id lists (post-prompt)."""
    dev = dev or ra.DEV
    outs = []
    for i in range(0, len(prompt_id_lists), bs):
        chunk = [list(p) for p in prompt_id_lists[i:i + bs]]
        gen = [[] for _ in chunk]
        done = [False] * len(chunk)
        for _ in range(max_new):
            if all(done): break
            x = torch.zeros((len(chunk), S), dtype=torch.long, device=dev)
            for r, ids in enumerate(chunk):
                x[r, :len(ids)] = torch.tensor(ids[:S], device=dev)
            logits = m(x)
            for r, ids in enumerate(chunk):
                if done[r] or len(ids) >= S:
                    done[r] = True; continue
                nxt = int(logits[r, len(ids) - 1].argmax())
                if nxt == IME: done[r] = True; continue
                ids.append(nxt); gen[r].append(nxt)
        outs.extend(gen)
    return outs

def score_batched(m, items, bs=16):
    prompts = [ra.prompt_ids(it["convo"][0]["content"]) for it in items]
    gens = generate_batched(m, prompts, bs=bs)
    texts = [ra.tok.decode(g).strip() for g in gens]
    parsed = correct = halluc = omission = 0
    hc = ht = sc = st = 0
    for it, text in zip(items, texts):
        mm = ra.RE.match(text)
        if not mm: continue
        parsed += 1
        pred = dict(zip(ra.FIELDS, [g.strip() for g in mm.groups()]))
        for f in ra.FIELDS:
            t, p = it["tuple"][f], pred[f]
            hit = (p == t)
            if hit: correct += 1
            elif p == "none" and t != "none": omission += 1
            else: halluc += 1
            if it["held_values"]: ht += 1; hc += hit
            else: st += 1; sc += hit
    return {"parse": parsed, "correct": correct, "halluc": halluc, "omission": omission,
            "held": f"{hc}/{ht}", "seen": f"{sc}/{st}", "texts": texts}

if __name__ == "__main__":
    m, _ = ra.load("nano")
    inst0 = ra.inst0
    t0 = time.time()
    rb = score_batched(m, inst0, bs=16)
    tb = time.time() - t0
    print(f"BATCHED: parse {rb['parse']}/40 correct {rb['correct']}/200 halluc {rb['halluc']} "
          f"omission {rb['omission']} held {rb['held']} seen {rb['seen']}  ({tb:.1f}s)")
    print("reference (gate_scribe_v2.log): parse 39/40 correct 162/200 halluc 23 omission 15? held 68/95 seen 94/100")
    # exact per-dialogue comparison vs native scorer
    t0 = time.time()
    native_texts = []
    for it in inst0:
        pids = ra.prompt_ids(it["convo"][0]["content"])
        out = ra.generate(m, pids)
        native_texts.append(ra.tok.decode(out[len(pids):]).strip())
    tn = time.time() - t0
    mismatches = [i for i, (a, b) in enumerate(zip(rb["texts"], native_texts)) if a != b]
    print(f"NATIVE : ({tn:.1f}s)  speedup x{tn/max(tb,0.01):.1f}")
    print(f"per-dialogue string mismatches: {len(mismatches)}/40 -> " +
          ("BYTE-IDENTICAL: ADOPT as fast path" if not mismatches else f"DIVERGENT at {mismatches[:5]}: DISCARD"))
