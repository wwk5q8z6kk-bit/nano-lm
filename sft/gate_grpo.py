# Stage 4 gate per Recipe ③: "Pass@1 on ... with held-out verifier".
# Held-out prompts (disjoint from GRPO's training questions); same verifiable checker.
# pass@1 estimated over K samples/prompt; gate = GRPO pass@1 reliably > SFT pass@1.
import json, numpy as np, torch
import model_nano as M

K = 8
NMAX = 8
held = json.load(open("grpo_gate_prompts.json"))
def make_prompt(q): return f"In a few words, {q}?"
def verify(out, stopped):
    if not stopped: return 0.0
    w = len(M.tok.decode(out).split()); return 1.0 if (1 <= w <= NMAX) else 0.0

def pass_at_1(path, seed):
    m = M.load(path); torch.manual_seed(seed); rs = []
    for q in held:
        for _ in range(K):
            o, s = M.sample(m, make_prompt(q), temp=1.0); rs.append(verify(o, s))
    return np.array(rs)

r_sft = pass_at_1("sft.pt", 11)
r_grp = pass_at_1("grpo.pt", 11)
n = len(r_sft)
p_sft, p_grp = r_sft.mean(), r_grp.mean()
# 95% CI on the difference (paired-ish, treat as independent Bernoulli means — conservative)
se = ((p_sft*(1-p_sft) + p_grp*(1-p_grp)) / n) ** 0.5
diff = p_grp - p_sft
print("Stage-4 GRPO/RLVR gate — held-out pass@1 (verifiable brevity checker)\n" + "-"*60)
print(f"  held-out prompts {len(held)} x K={K} = {n} trials, NMAX={NMAX} words")
print(f"  SFT   pass@1 {p_sft:.1%}")
print(f"  GRPO  pass@1 {p_grp:.1%}")
print(f"  delta {diff:+.1%}   95% CI ~ [{diff-1.96*se:+.1%}, {diff+1.96*se:+.1%}]")
print("-"*60)
passed = diff > 0 and (diff - 1.96*se) > 0
print(f"GATE {'PASS' if passed else 'FAIL / INCONCLUSIVE'}  (need pass@1 delta CI lower bound > 0)")
