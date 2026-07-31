# Revised Discovery Reconciliation (local primary check)

*Generated 2026-07-31T13:49:09.910068+00:00. Responds to the revised public+local discovery pass. Privileges
audits, JSON, and code over narrative.*

## Verdict on the six A1 corrections

| Revised finding | Local primary check | Status |
|-----------------|---------------------|--------|
| Tagged narrative vs uncommitted evidence bundle | HEAD=`0e01d73`=`paper-alpha-v1`; E1/E3 JSON + many locks still untracked | **CONFIRMED** |
| E2 GATED/STOP; “in flight” stale | `PREREG_E2` STOP; no `results_e2_*`; pods empty; FINDINGS updated | **CONFIRMED** |
| Stage 1 = agent-rubric-pass-1 | `results_e3_human.json` rater id; IAA null | **CONFIRMED** |
| E1 KILL numbers + Amendment 1 structural | Local JSON present; **U recomputed component-wise: all match** | **UPGRADED: arithmetic verified locally** (adapters/logs still uninspected) |
| R★/E4 protocol asserted; world absent | Preregs exist locally; no builder/result | **CONFIRMED** |
| ρ = review load | `e1/common.py` `flagged/n_fields`; PREREG_E1; DECISION_P1 corrected | **CONFIRMED** |

## Most consequential new issue — token budgets (CONFIRMED)

| Condition | Params | Pretrain tokens (primary) | tok/param | Diluted gap |
|-----------|-------:|--------------------------:|----------:|------------:|
| Nano | 3.15M | **32.8M** (`pretrain/AUDIT.md`) | ~10.4 | 18.3±1.3 |
| Scale | ~10M | **~200M** (`scale/AUDIT.md`) | ~20 | 18.7±1.5 |
| Own-stack 160M weak | 159.3M | **200M** (JSON `target_tokens`) | ~1.26 | 16.9±1.7 |
| Own-stack 160M Chinchilla | 159.3M | **3.2B** (JSON) | ~20.1 | 7.0±1.0 |
| Corner 3.2B+LoRA | 159.3M | ~3.2B + LoRA | ~20.1 | 4.2±0.9 |

**Manuscript error:** Paper α methods claim both 3.15M and 10M pretrained on ~200M.
That is false for 3.15M.

**Interpretive consequence:** “flat across ~50× within-stack scale” is **not** a
controlled parameter-only scale law. Prefer the unequal-schedule wording in DIFF H.

## E1 local arithmetic (no longer RAW-UNINSPECTED for U table)

From `results_e1_utility.json` verify-on rows, reconstructing
`U = P - 0.5M - 0.3ρ - 0.02L - 0.05C` matches stored U for every method including
official M0. Decision: KILL; M1=0.998999; official M0=0.925217; margin=+0.073782;
sensitivity_flip=false.

**Still uninspected for full auditability:** adapter/checkpoint hashes for official M0,
cost-normalization protocol for C across devices, excluded RunPod logs as L/C source,
public pre-run prereg commit chronology.

**Wording:** Prefer “best non-generative baseline (M1) dominates”; M2 is within δ, not
numerically above M0 (DIFF I).

## E3 local confirmation

- Auto: 0/486 rescues; `EXACT_NOT_OVERSTATING_BY_NORMALIZE`
- Pack: faithful_rate 0.00; rater `agent-rubric-pass-1`; verdict_prereg `EXACT_SURVIVES`
- **Classification:** agent-rubric threshold result ≠ human/clinician validation (DIFF J)

## Layers (revised)

| Layer | Content |
|-------|---------|
| Measurement | Anchored Stage T/factorial/diversity/C1b/C3/pointer/fabric; E1/E3 locally arithmetic-verified; token schedules unequal |
| Decision | E1 KILL under frozen U; E2 STOP; OLD_TASK forbidden; R★/E4 protocol local |
| Aspiration | NanoScribe architecture; R★ product value untested |

## Narrowest invariant (accepted)

On the measured synthetic structured-extraction distribution, several small LM pipelines
show a field- and lexical-type-localized held-vs-seen exact-copy difference. Magnitude
depends on estimator/denominator. Training-value diversity, pretraining amount, and
adaptation regime alter behavior; internal mechanism unidentified. A generator-aligned
deterministic extractor (M1) outperforms tested generative references under the supplied
closed-world utility.

## Freeze implication

Still **AUDIT_REMEDIATION_REQUIRED**, now with **DIFF H (token/50×)** as the
highest-priority owner Paper α correction — ahead of architecture language cleanup.
