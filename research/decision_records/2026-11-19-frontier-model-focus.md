# 2026-11-19 — Program focus moves to the frontier model

**Status:** the owner decided the focus. The agent chose the scope, path and holds under the owner's delegation. Every choice in this record is open to correction by the owner.

**Owner words, verbatim:** "the project now is focused on frontier model". When asked which of three readings was meant, the owner answered "discover and assess and take best decisions".

**What is the owner's and what is the agent's.** Only the refocus on the frontier model is the owner's decision. The ≤ 4B scope, the adapt-first path and putting P1 on hold are the **agent's choices**, made under that delegation, and they stand until the owner corrects them. The delegation covers direction, not spend. No experiment is authorized by this record, and the standing row "explicit authorization required per launch" (`RESEARCH_STATUS.md`, NOT AUTHORIZED) is unchanged.

Position in the authority chain: the refocus is an owner decision in the sense of `PROJECT_AUTHORITY.md` §2, so the charter now carries an amendment banner that points here. The delegated choices carry the same authority only until the owner corrects them.

## 1. Interpretation chosen

**"Frontier model" means NanoLM: an independent, general-purpose small foundation language model, with about 2B parameters preferred and 4B as the absolute ceiling, running locally on Apple Silicon.**

Three readings were considered:

| Reading | Verdict | Why |
|---|---|---|
| A. Restart the dormant `frontier/*` P1-scribing line (Qwen2.5-1.5B span-port) | **Not chosen** | Dormant for 5 weeks to 3 months. No commit or spend in the last 4 days touches it. |
| B. The `nanolm_frontier` package (`nanolm/posttrain-v1`, `813cfab`): a 5-architecture tournament plus post-training of Qwen3-1.7B | **Chosen as the scope**, *not* as evidence | It already states this identity: "independent general-purpose small foundation LM (≤ 4B, ~2B target)", and `artifacts/nanolm_posttrain_v1/REPORT.md` §1 says "Historical Scribe experiments remain preserved provenance only". The owner's sentence matches it. |
| C. A literal frontier-scale model, competitive with the state of the art | **Infeasible, recorded as such** | The budget is about \$75 of RunPod balance plus one M3 Max with 64 GB. *Assumptions, not measurements:* about 1–2T tokens are needed for a competitive 2B model, and the Mac sustains about $1.5\times10^{13}$ FLOP/s. Under those assumptions, $6ND \approx 1.2$–$2.4\times10^{22}$ FLOPs is decades locally, and on rented GPUs it is several orders of magnitude above the balance. |

**Consequence for P1 scribing.** The line is ON HOLD. It is preserved provenance, **not refuted**. Every ESTABLISHED, VOID and PENDING row in `RESEARCH_STATUS.md` keeps its bucket. The NOT-AUTHORIZED items stay unauthorized. They are dormant, not retired.

## 2. The real fork: build from scratch, or adapt an open base

| Path | Status | Evidence |
|---|---|---|
| **Adapt an open-weight base** (Qwen3-1.7B / 4B class) with post-training, grafts and verifiers | **PRIMARY** | The only path that reaches 1–4B quality inside the budget. `nanolm_frontier` E2 already runs this way: 2,257 s on an A40, \$0.37 attributed to the pod. |
| **Novel architecture, pretrained from scratch at 1–4B** | **ON HOLD (budget), not refuted** | The native30 capability-floor result: a 30M model trained from scratch did not clear the floor. The FLOP arithmetic above. |
| **Novel mechanisms tested small** (≤ 40M, multi-seed, measured noise floor), then **grafted** onto the open base | **SECONDARY, gated on the instrument** | The only route by which DMLA-family ideas can earn a place. `NANO_VNEXT_MASTER_SPEC.md` still applies: no mechanism is admitted without a discriminating experiment. |

## 3. Assessment of the existing frontier-line evidence

This covers only the branches that were read: `nanolm/posttrain-v1` (`nanolm_frontier_v1`, `nanolm_posttrain_v1`) and `nanolm/arch-evolution-v3` (`lab/reports/omega_cycle2_results.json`). The following were **not** assessed, and nothing here speaks for them: `research/unified-discovery-loop-v1`, `research/memory-boundary-v2`, `research/v5-distributed-systems`, `research/v3-exact-recall`, and the omega results on `nanolm/architecture-invention-v1`.

- **E1, a 5-architecture tournament at about 38M parameters, 2.87M tokens, 700 steps, one seed. The code default in `tournament.py` is `seed=42`; the artifact does not record a seed.** Mean validation BPB ranges over 0.6316–0.6368, a spread of 0.0052. **NOT SEPARATED: no noise estimate exists for this instrument.** The native30 seed floor cannot be borrowed, because §22 of the spec forbids carrying it across a change of instrument (different scale, an 8192-entry vocabulary, a different corpus). The report's verdicts (C3 RETAIN, C2 REJECT, C4 MODIFY) are therefore **unsupported**, and each one is a HYPOTHESIS. The only metric that does separate is throughput: C0 is 5.3–12.2× faster, a real cost of the pure-PyTorch delta kernels. Exact-copy recall is 0–4% in every arm, so none of the arms reached the capability the long-context mechanisms are built for. **D8 was checked by reading the code.** The head *is* tied: `_head_logits` computes `F.linear(final_norm(h), tok_emb.weight)`. But `tok_emb` is initialized with `normal_(std=0.02)` and the input is RMS-normalized, so the expected logit std at init is about $0.02\sqrt{512} \approx 0.45$, not the $\sqrt{d}$ blow-up D8 describes. **The D8 defect is not present.** This is inferred from the code and not measured at init. A separate minor issue: the parameter count adds `2·d·vocab` as if the head were untied, overstating every arm by about 4.2M. That error is identical across arms (all share `d_model=512`), so the iso-parameter matching is unaffected.
- **E2, Qwen3-1.7B: Base vs LoRA-SFT vs DexDiff+VAnchor+SFT+GRPO, on 56 items.** These numbers are paired and recomputed from `e2_foundation_15d.json` with an exact McNemar test:

  | Pair | Discordant items | p |
  |---|---|---|
  | Base vs SFT | 6 vs 8 | 0.79 |
  | Base vs Frontier | 5 vs 6 | 1.00 |
  | SFT vs Frontier | 4 vs 3 | 1.00 |

  **No arm separates.** The per-dimension cells hold n=4 items each and carry no information. GRPO made **1 update out of 72** groups (71 had zero reward variance). The grafted gates sit around 0.0015, so the "Frontier" arm is, to measurement, SFT plus near-identity adapters. Exact step-0 graft preservation (max logit diff 0.0) is a sound **engineering** result and is kept.
- **`nanolm_posttrain_v1`: 6/12 → 8/12 held-out items.** n=12, and the training loss went to 0.0019. **NOT SEPARATED.**
- **arch-evolution-v3, "Kalman-Lag1 Surprise-Landmark breakthrough, 2.3025 BPB".** There are two different results here:
  - *Recall, 3 seeds:* 0.178 ± 0.090 against the best control at 0.068 ± 0.004, at matched KV retention of about 16%. A Welch test on the summary statistics gives **t = 2.1 on about 2 degrees of freedom, p ≈ 0.17**. That is **not separated at 3 seeds**: the sd is half the mean. The direction is consistent with the lag-1 ablation (0.068 with lag-1 off), which makes this the **first frontier-line candidate for replication**. It is not supported.
  - *Composite BPB, single run:* 2.3025 against a 2.3313 reference, with copy accuracy 0.0. **NOT SEPARATED**, and "breakthrough" is not supported.

**Pattern across the line, and the reason the instrument comes first.** Every headline comparison so far sits below the resolution of the instrument that produced it. The program has more mechanisms than measurements. Until an instrument can separate arms, adding arms adds nothing.

## 4. What comes first: the instrument (none authorized)

1. **F0, a seed-noise floor for the E1 micro-tournament.** Run C0 with 5 seeds on the identical config, locally on MPS, at about \$0. This yields σ(BPB) for C0. That is the first noise estimate on this instrument. It is assumed, not shown, to bound the other arms, and the slower recurrent arms may need their own replicates. It manipulates nothing, so field 7 of the eighteen-field standard is `N/A — measurement study` under the §20 rule.
2. **F1, a powered capability eval for the adapted-base path.** For paired pass/fail items with a discordant rate near E2's observed 0.25, $n \approx (z_{\alpha/2}+z_\beta)^2 \, p_{disc} / \delta^2$ gives **about 200 items for a 10 pp effect and about 780 for 5 pp**. The current suite has 56. Draw items from decontaminated public held-out sets per dimension, not hand-written ones, and verify the count with the method in `artifacts/campaign/eval_power_report_v1.json` (cross-branch, on `frontier/accelerated-research-campaign-v2`) before it is frozen.
3. **Only after F0 and F1 exist:** replicate Kalman-Lag1 recall (≥ 5 seeds, preregistered), then the question of whether SFT and GRPO beat the base on F1.

Each item needs its prereg and an explicit owner authorization before it launches.

## 5. Spend anomaly, open

`artifacts/nanolm_frontier_v1/SPEND_AND_PROVENANCE.json` (on `nanolm/posttrain-v1`) records an account delta of **\$5.09** in a window where the pod is attributed **\$0.37**. The file itself says concurrent peer pods may account for the difference. That has **not been reconciled**: about \$4.72 is unattributed, out of a balance of about \$75.

## 6. What does not change

- Native30 ESTABLISHED rows, E-DELIMIT VOID, D8 PENDING, and the R8 and §22 rules.
- "Do not accumulate mechanisms" governs the frontier line exactly as it governed the scribe line.
- No PHI on cloud, and no clinical claims.
- Peer branches were not modified by this decision.
