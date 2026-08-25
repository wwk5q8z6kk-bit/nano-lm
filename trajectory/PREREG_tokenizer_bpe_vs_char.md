# PREREG — own-stack BPE vs character-level hash_tokens

**Status: RECORDED, NOT RUN (2026-08-25).**

The full design, rationale and pre-committed decision rule live in
`artifacts/campaign/ORX_TOKENIZER_WAVE_DESIGN.md`. This file exists because repo
convention keeps preregistrations in `trajectory/`, and a prereg nobody can find
is not a prereg. It deliberately does **not** restate the criteria: two copies of
a decision rule is two decision rules, and the second one drifts.

## Held fixed across arms

Architecture 30M params · vocabulary 4098 · same corpus · same eval suite ·
3 seeds run inside each node · one device type for every arm.

The only thing that varies is the tokenizer. D3.3 showed why this matters: under
character-level `hash_tokens`, 82.7% of eval prompts and 100% of training rows
exceed `max_seq=512`, so any measured difference was confounded with truncation
rather than with tokenization. Varying scale and tokenizer together would repeat
that mistake one level up.

## Why it is recorded rather than run

Nano's substrate work (`nano/`, NANO-CLIN-001, NANO-SLW-001) loads no model and
spends no compute. This experiment would do both. Recording the criteria now
means the rule cannot be chosen after seeing a result; running it is a separate
decision with a separate budget.

## Recheck

    sed -n '/Pre-registered decision rule/,/^## /p' \
        artifacts/campaign/ORX_TOKENIZER_WAVE_DESIGN.md
