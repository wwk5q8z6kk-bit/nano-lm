# Revised discovery pass — acceptance & verification

*Filed 2026-07-31 after owner-supplied revised discovery pass.
Supersedes local-state conclusions of public-repo-only discovery where they conflict.
No E2/E4/fabric/old-task runs. Frozen verdicts not altered.*

## Verification of high-signal claims (local)

| Discovery claim | Local check | Verdict |
|-----------------|-------------|---------|
| Public `master` / `paper-alpha-v1` at `0e01d73` | `git rev-parse HEAD` → `0e01d73…` | **Confirmed** |
| Post-α E1/E2/E3/E4 primary bundle largely untracked | `git status` shows `?? trajectory/PREREG_E*.md`, `?? results_e1_*`, `?? results_e3_*` | **Confirmed** — tagged narrative, uncommitted evidence |
| E2 = GATED/STOP, no RESULT; “in flight” stale | Prereg status GATED/STOP; no `results_e2_*.json`; pods empty; stale GPU sentence **corrected** this pass | **Confirmed** |
| E3 “human” = `agent-rubric-pass-1` | `results_e3_human.json` rater field; 0/100 faithful | **Confirmed** |
| E1 U numbers / margin | `results_e1_utility.json`: M1 0.998999, M0_pythia 0.925217, margin 0.073782, no flip | **Confirmed locally** (JSON present; independent full recompute of every component not re-done this pass) |
| ρ = review load | `e1/common.py` `rho = flagged/nf`; PREREG_E1 review load | **Confirmed**; public sensitivity “hallucination penalty” prose = **terminology error** |
| Nano 3.15M tokens ≈ 32.8M not ~200M | `pretrain/AUDIT.md`: 4000 steps / **32.8M tokens** | **Confirmed** |
| Scale 10M ≈ 200M tokens | `scale/AUDIT.md`: D≈20N → **~200M tokens** | **Confirmed** |
| Paper α methods says both anchors ~200M | `paper1_draft.md` / `paper1.tex`: “pretrained on ~200M FineWeb tokens” for own stack 3.15M **and** 10M | **Confirmed factual methods error** |
| “Flat across ~50×” overclaims parameter-only law | Foundation / draft / RESEARCH_PROGRAM still say ~50× flatness; token/parameter ratios unequal | **Confirmed interpretation defect** |
| “Non-generative baselines dominate” too plural | M1 dominates; M2 0.886 < 0.925 but within δ | **Confirmed wording overbreadth** |
| R★/E4 protocol files exist locally; no world/result | `REGIME_P1_…`, `PREREG_E4_…` present; no builder/result | **Confirmed**: LOCAL PROTOCOL ASSERTED; WORLD AND RESULT ABSENT |
| No E4/E2/fabric authorization | Explicit | **Honored** |

## Accepted evidence-class discipline

Use discovery labels going forward in audit prose:

`PUBLIC-ANCHORED` · `LOCAL-DOCUMENTARY` · `RAW-UNINSPECTED` · `DECISION/GOVERNANCE` · `ASPIRATIONAL` · `STALE/CONTRADICTORY`

## Narrowest invariant claim (adopted)

> On the measured synthetic structured-extraction distribution, several small LM
> pipelines show a field- and lexical-type-localized difference between copying
> values represented during fine-tuning and emitting held-out values present in
> the input. Magnitude depends strongly on estimator/denominator. Training-value
> diversity, pretraining amount, and adaptation regime alter the behavioral
> profile; internal mechanism remains unidentified. A generator-aligned
> deterministic extractor (M1) outperforms the tested generative references under
> the supplied closed-world utility.

## Most consequential new issue (priority)

**Token-budget mismatch:** 3.15M audit = **32.8M** tokens vs manuscript “~200M for
both own-stack anchors.” This undermines “flat across ~50×” as a clean
parameter-only conclusion. Must be corrected in Paper α / lockfile **with owner
approval** before treating scale-vs-stack narrative as frozen.

## Corrected scale reading (evidence-backed, not yet applied to Paper α)

> Increasing parameters under the tested, unequal pretraining schedules did not
> monotonically eliminate the held-value gap; a 159M model trained on only 200M
> tokens remained high-gap, whereas substantially more pretraining data reduced it.

## Immediate remediation queue (docs/archive only)

| Priority | Action | Authority |
|----------|--------|-----------|
| 1 | Owner-approve Paper α methods token correction + soften 50× / scale-flat language | Owner lockfile / α |
| 2 | Commit untracked E1/E3 primary JSON + preregs + manifests (evidence packaging) | Owner commit |
| 3 | Relabel E3 everywhere as **agent-applied rubric audit**; human equivalence UNVALIDATED | Owner α + lockfiles |
| 4 | Narrow E1 prose: **best** non-generative (M1) dominates; M2 within δ only | Owner α + lockfiles |
| 5 | Fix public stale mechanism language (MASTER_PLAN / RESEARCH_PROGRAM / NANOSCRIBE_VNEXT) | Owner |
| 6 | Durable publish raw JSONL via authorized Release/LFS | Owner |
| 7 | Static audit of full R★/E4 protocol text for circularity (no execution) | Optional docs |

## Explicit non-authorization

- No old-task runs under `OLD_TASK_U`
- No E2 intervention
- No fabric expansion / NanoScribe implementation claim
- No Stage 4 / E4 execution
- No new clinician labeling campaign unless separately authorized

## Decision gate (unchanged)

**AUDIT_REMEDIATION_REQUIRED** — then IDLE_AFTER_FREEZE or AUTHORIZE_E4_DESIGN_ONLY.
