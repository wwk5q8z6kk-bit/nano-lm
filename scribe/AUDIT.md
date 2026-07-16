# nano-scribe — ambient-scribe stage on the nano lineage (2026-07-16)

Purpose stage for nano-lm: convert a short clinic-visit dialogue into a structured
summary line, gated on **faithfulness** — the summary may contain nothing that is not
grounded in the dialogue. This is the hallucination-evaluation problem of production
clinical documentation AI, miniaturized to a scale where the full loop runs in minutes.

Base model: `nano-posttrain-2026-07-16/dpo.pt` (gate-passed: DPO win-rate 80.6%,
95% CI [75.6, 85.6] vs SFT; upstream SFT v2 gate 98% stop / 92% held-out refusal).

## Task

Input (ChatML user turn): a synthetic doctor–patient dialogue (3–6 exchanges, with
distractor small-talk) rendered from a **known ground-truth fact tuple**:
complaint, duration, severity, medication-taken (optional), allergy (optional).

Output (assistant turn): `CC: <complaint> | DUR: <n> <unit> | SEV: <severity> | MED: <med|none> | ALG: <allergy|none>`

Because dialogues are procedurally generated from fact tuples, ground truth is exact —
the gate can measure faithfulness by construction, not by judge.

## Pre-registered gate (WRITTEN BEFORE TRAINING — bars may not move after)

Held-out set: 40 dialogues rendered from **template families never used in training**
(both doctor-question and patient-answer phrasings), half containing **slot values
held out of training entirely** (3 complaints, 2 medications, 1 allergy) — those test
copying-from-context vs. reciting memorized priors.

Primary decoding: **greedy** (the deployment path for an extraction task — deviation
from prior stages' sampled-primary is deliberate and logged). Sampled K=4 @ temp 0.7
reported as a robustness diagnostic.

Per-field scoring (5 fields per dialogue, 200 field decisions):
- pred == truth → **correct**
- pred = none but truth ≠ none → **omission** (hurts recall; NOT hallucination)
- pred ≠ none but truth = none → **fabrication** (hallucination)
- pred ≠ truth, both ≠ none → **substitution** (hallucination)

PASS requires ALL of:
1. **Parse rate ≥ 90%** — output matches the summary schema
2. **Fact recall ≥ 80%** — correct / total fields
3. **Hallucination rate ≤ 10%** — (fabrications + substitutions) / total fields
4. **Base control fails** — the un-finetuned dpo.pt must miss these bars
   (discrimination: the capability is caused by this stage's training)
5. Held-out-value subset reported separately; if recall on held-out values is
   > 20 points below seen values, that is a memorization flag — logged even on PASS

Honest-reporting rule (unchanged from prior stages): if v1 fails, the failure and its
diagnosis are published; one pre-specified improvement sweep, re-measured once.

## Execution log
- (to be filled as stages run)
