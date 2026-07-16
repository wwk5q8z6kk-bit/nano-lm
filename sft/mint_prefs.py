# Stage 2 (pair construction) — the mechanic the vault NAMES but never shows:
# "Best-of-n sampling then judge-rank" -> on-policy (prompt, chosen, rejected) triples.
# Judge = a programmatic RLAIF-lite rubric (no strong judge model at nano; the recipe's RLAIF
# path with a rubric/constitution, scaled down). Reward rewards concise, terminated, non-repetitive
# assistant turns; chosen = argmax, rejected = argmin; keep only well-separated pairs (hard pairs).
import json, random, numpy as np
import model_nano as M

random.seed(0)
G = 6                 # rollouts per prompt (best-of-n)
MARGIN = 0.6          # keep only well-separated pairs (preference-datasets: "hard, well-separated")
N_PROMPTS = 300       # minting prompt set A
tok, IMS = M.tok, M.IMS

def rubric(out, stopped):
    """RLAIF-lite reward: concise+terminated+non-repetitive assistant turn."""
    if len(out) == 0: return 0.0
    distinct = len(set(out)) / len(out)
    clean_stop = 1.0 if (stopped and 3 <= len(out) <= 50) else 0.0
    leak = 0.5 if IMS in out else 0.0
    return clean_stop + 0.5 * distinct - leak

def main():
    # prompt pool = SmolTalk user turns (short), disjoint train(A)/gate(B) split
    convos = json.load(open("sft_convos.json"))
    pool = list({c[0]["content"] for c in convos if c[0]["role"] == "user" and 5 <= len(c[0]["content"]) <= 120})
    random.shuffle(pool)
    setA, setB = pool[:N_PROMPTS], pool[N_PROMPTS:N_PROMPTS + 60]
    json.dump(setB, open("gate_prompts.json", "w"))       # held-out prompts for the DPO win-rate gate

    sft = M.load("sft.pt")
    pairs, kept, rmax_sum, rmin_sum = [], 0, 0.0, 0.0
    for i, p in enumerate(setA):
        cands = [M.sample(sft, p, temp=0.9) for _ in range(G)]     # sample G at high temp for spread
        scored = [(rubric(o, s), o) for (o, s) in cands]
        scored.sort(key=lambda t: t[0], reverse=True)
        rmax, best = scored[0]; rmin, worst = scored[-1]
        if rmax - rmin >= MARGIN and len(best) > 0 and len(worst) > 0:
            pairs.append({"prompt": M.chat_ids(p), "chosen": best + [M.IME], "rejected": worst + [M.IME]})
            kept += 1; rmax_sum += rmax; rmin_sum += rmin
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(setA)} prompts, kept {kept} pairs", flush=True)

    json.dump(pairs, open("pref_pairs.json", "w"))
    print(f"minted {kept} preference pairs (margin>={MARGIN}); "
          f"mean chosen r={rmax_sum/max(1,kept):.2f}, rejected r={rmin_sum/max(1,kept):.2f}", flush=True)

if __name__ == "__main__":
    main()
