# Stage 3c gate per Recipe ③: "Win-rate vs SFT (AlpacaEval/Arena-Hard)".
# Nano scale-down: judge = the same programmatic RLAIF-lite rubric, but evaluated on HELD-OUT prompts
# (gate_prompts.json, disjoint from the minting set) so a win = GENERALIZATION, not the tautology of
# scoring on the exact pairs DPO trained on. K samples/prompt/model; tie-aware win-rate.
import json, numpy as np, torch
import model_nano as M
from mint_prefs import rubric

K = 4
prompts = json.load(open("gate_prompts.json"))
sft, dpo = M.load("sft.pt"), M.load("dpo.pt")

def mean_reward_and_stop(m):
    torch.manual_seed(7)
    rs, stops, lens = [], 0, []
    per_prompt = []
    for p in prompts:
        pr = []
        for _ in range(K):
            out, st = M.sample(m, p, temp=0.8)
            r = rubric(out, st); rs.append(r); per_prompt.append(r)
            stops += (st and 3 <= len(out) <= 50); lens.append(len(out))
    return np.array(rs), stops / (len(prompts) * K), np.mean(lens)

torch.manual_seed(7); rs_sft, stop_sft, len_sft = mean_reward_and_stop(sft)
torch.manual_seed(7); rs_dpo, stop_dpo, len_dpo = mean_reward_and_stop(dpo)

# tie-aware pairwise win-rate over matched (prompt,sample-index) reward comparisons
wins = (rs_dpo > rs_sft).sum(); ties = (rs_dpo == rs_sft).sum(); losses = (rs_dpo < rs_sft).sum()
n = len(rs_sft)
win_rate = (wins + 0.5 * ties) / n

print("Stage-3c DPO gate — held-out win-rate vs SFT (RLAIF-lite rubric judge)\n" + "-"*62)
print(f"  held-out prompts {len(prompts)} x K={K}  = {n} matched comparisons")
print(f"  SFT   mean reward {rs_sft.mean():.3f}  clean-stop {stop_sft:.0%}  mean-len {len_sft:.0f}")
print(f"  DPO   mean reward {rs_dpo.mean():.3f}  clean-stop {stop_dpo:.0%}  mean-len {len_dpo:.0f}")
print(f"  win {wins}  tie {ties}  loss {losses}  -> tie-aware win-rate {win_rate:.1%}")
# 95% normal-approx CI on win-rate to be honest about noise at 3M params
se = (win_rate * (1 - win_rate) / n) ** 0.5
print(f"  win-rate 95% CI ~ [{max(0,win_rate-1.96*se):.1%}, {min(1,win_rate+1.96*se):.1%}]")
print("-"*62)
passed = win_rate > 0.5 and (win_rate - 1.96*se) > 0.5     # gate = win-rate reliably above 50%
print(f"GATE {'PASS' if passed else 'FAIL / INCONCLUSIVE'}  (need win-rate CI lower bound > 50%)")
