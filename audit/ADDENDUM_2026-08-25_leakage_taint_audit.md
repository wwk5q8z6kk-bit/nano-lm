# Addendum — leak-taint audit of the frozen evidence ledger

**Date:** 2026-08-25
**Author scope:** enumeration only. **No frozen tag is modified by this document.**
**Subject:** do any claims carried by `post-alpha-evidence-freeze-2026-07-31`, `post-alpha-reconciled-evidence-freeze-2026-07-31`, or `paper-alpha-v1` depend on either of the two span-port leakage channels now under ablation?

- **C1** — `PROMPT_ANSWER_TEMPLATE_GOLD_VALUE`: the answer template and system-prompt format examples carry the gold value.
- **C2** — `PARSER_RAW_VALUE_FALLBACK`: the scorer substitutes `raw_value` as the model's quote when the model emitted a label with no quote.

## Why this is dated *before* the ablation results

The four cells of the C1×C2 ablation are still idle. This enumeration is written now, deliberately. Performed after the numbers land, the same exercise would be indistinguishable from fitting the audit to the outcome: whichever claims the result embarrassed would be the ones found "tainted". Fixing the classification in advance makes it falsifiable. If the ablation later shows a large C1 effect, nothing in this table may be quietly re-graded — a change requires its own dated addendum saying what new fact forced it.

## Method

Two independent tests per claim, because provenance alone is a weaker argument than provenance plus mechanism.

1. **Provenance.** Did the code implementing either channel exist in the tagged tree at all?
2. **Mechanism.** Independently of naming, did the scorer that produced the claim's artifact contain an *equivalent* channel — gold value in the model's prompt, or gold value substituted for missing model output?

Commands run (reproducible):

```sh
git ls-tree -r --name-only post-alpha-evidence-freeze-2026-07-31 -- nanoscribe/
git grep -l "span_port\|span-port" post-alpha-evidence-freeze-2026-07-31 -- '*.py'
git log --all --oneline -S "candidate_from_span_port_line" --reverse
git log --all --oneline -S "build_span_port_prompt" --reverse
git show post-alpha-evidence-freeze-2026-07-31:scribe/gate_scribe.py
```

## Finding 1 — provenance: both channels postdate every freeze tag

`git ls-tree -r post-alpha-evidence-freeze-2026-07-31 -- nanoscribe/` returns **empty**: the `nanoscribe/` package did not exist at the freeze. `git grep -l 'span_port\|span-port'` over all `*.py` in the tagged tree returns **no files**: no span-port code of any kind existed.

Both channels are introduced in the span-port adapter lineage, whose earliest commits (`ac46908`, `89a7939`, `c7e0e90` for `candidate_from_span_port_line`; `c4822b9` for `build_span_port_prompt`) are all on post-freeze branches. The freeze tags resolve to `a9d12cb` (2026-07-31 10:13), `67bf87b` (2026-07-31 14:29) and `0e01d73` (2026-07-31 01:20).

> **Correction (same day).** An earlier draft of this addendum repeated `nanoscribe/leakage.py`'s docstring claim that "C1 and C2 arrived together in `dc3b310`". That is **false**, and the correction is recorded here rather than silently applied. Verified against `09745ec`: its `prompt.py` already interpolated `{spec.raw_value!r}` into the question (lines 56–59) and already shipped `Example: STATED: "neck"` in the system prompt (line 20). **The prompt gold-value channel predates `dc3b310`; only the parser fallback is new there.** Credit: peer session `v7pagl1v`, independently re-verified here before acceptance.
>
> This does not move any classification below. The load-bearing provenance fact is the stronger one — `nanoscribe/` does not exist in the freeze tree at all, so *both* channels postdate every tag regardless of which post-freeze commit each first appeared in. The corrected date only shifts C1's first appearance from one post-freeze commit to an earlier post-freeze commit.

**No frozen artifact can have been produced by code that did not exist when it was frozen.**

## Finding 2 — mechanism: the pre-freeze scorer contains neither equivalent

Provenance would be a thin argument if the *old* scorer had its own version of the same defect. It does not. From `scribe/gate_scribe.py` at the freeze tag:

- **Prompt channel (C1 equivalent): absent.** The model is prompted with `prompt_ids(it["convo"][0]["content"])` — the dialogue turn only. The gold tuple `it["tuple"]` is read exclusively on the scoring side. Nothing in the prompt names the target value.
- **Parser fallback (C2 equivalent): absent.** Scoring is `hit = (p == t)` on the parsed field string, with the only two alternative branches being `p == "none" and t != "none"` → omission, and everything else → hallucination. A missing or malformed model output falls through `if not mm: continue` and is scored as an unparsed dialogue. **There is no branch that substitutes the gold value for absent model output.** The same `RE`/`FIELDS` scoring block appears identically in `gate_grounded.py` and `gate_absence.py`.

The two pipelines are also structurally different objects: Paper α's evidence is dialogue→summary field extraction scored by exact field-value match, whereas C1/C2 live in per-atom span-port probing scored by span binding. They share no prompt builder and no scorer.

## Classification

All 28 ledger claims (`papers/EVIDENCE_LEDGER.json`). Classification vocabulary as specified: **unaffected** / **needs-re-measurement** / **retracted-pending-rerun**.

### Group A — scored through the pre-freeze scribe pipeline: **unaffected** (22 claims)

`C_GAP_EXISTS`, `C_FIELD_LOC`, `C_DIVERSITY`, `C_SCALE_OBSERVED`, `C_PARAMETER_ONLY_EFFECT`, `C_OWNSTACK_200M_FULLFT_GATE`, `C_ADAPT_DATA_CELLS`, `C_ADAPT_DATA_INTERP`, `C_INTERFERENCE`, `C_C3_TB`, `C_C3_L`, `C_MORPH_DESC`, `C_MORPH`, `C_POINTER_P1`, `C_POINTER_P2`, `C_E1_MEASUREMENT`, `C_E1_GATE`, `C_E1_PRODUCT_THESIS`, `C_E3_NORMALIZE_RESULT`, `C_E3_AGENT_AUDIT`, `C_LORA_GEOM`, `C_RSTAR_VALUE`.

**Establishing code path:** `scribe/gate_scribe.py` at `post-alpha-evidence-freeze-2026-07-31`, lines 81–113 — prompt construction from `convo[0]["content"]`, scoring by `hit = (p == t)` with no fallback branch.
**Establishing artifacts:** `trajectory/results_anchors_v2_{nano,scale}.json`, `trajectory/results_arm1_v2_pythia-{160m,410m,1b}.json`, `trajectory/results_c3_10m.json`, `trajectory/results_interference_10m.json`, `trajectory/results_sweep_10m.json`, `trajectory/results_corner_3p2b_lora_seed{0,1}.json`, `trajectory/results_e1_utility.json`, `trajectory/results_e3_normalize_construct.json` — all present in the tagged tree.

Neither channel exists in that tree (Finding 1) and neither has an equivalent in that scorer (Finding 2).

### Group B — not scored by any model-output scorer: **unaffected** (5 claims)

| Claim | Why the channels cannot apply |
|---|---|
| `C_E3_HUMAN_STATUS` | Asserts a **non-event** ("has not been completed; IAA is absent"). No scorer involved; corroborated by `results_e3_human.json` recording `NOT_RUN`. |
| `C_E2_STATUS` | Asserts E2 produced **no RESULT**. Nothing was scored. |
| `C_NANOSCRIBE_STATE` | An implementation-state claim about what the repo does **not** evidence. Weakened, never strengthened, by a scorer defect. |
| `C_ZERO_HALLUC_OPEN` | `SPECULATION` / open-world; carries no measured number. |
| `C_CLINICAL_DEPLOYMENT` | `SPECULATION`; carries no measured number. |

### Group C — the one claim adjacent to the affected lineage: `C_FABRIC_SLICE`

> *"On closed synthetic inst0, propose→verify→abstain under rules-strong v2 drove presented-error → 0"*

**Classification: unaffected — but flagged, and the flag is not about C1/C2.**

`fabric/slice.py` at the freeze tag contains neither channel (scanned; no `raw_value`, no gold-substitution fallback, no span-port import), and `nanoscribe/` did not exist, so the verified claim as frozen stands on its own code. It is called out here for two reasons that a future reader should not have to rediscover:

1. It is the frozen claim **closest in subject matter** to the span-port verifier work, so it is the one most likely to be misread as implicated. It is not.
2. Its existing scope lock — *scoped existence proof on synthetic v2 under a decidable verifier relation R*, per `papers/RESEARCH_PROGRAM.md` §Claim discipline item 4 — is doing real work. Any restatement of this result **using the span-port harness** would be a new claim under the post-`dc3b310` code and would need its own ID and its own leak audit. The frozen claim may not be silently extended onto that harness.

### Summary

| Classification | Count | Claims |
|---|---|---|
| **unaffected** | **28** | all |
| **needs-re-measurement** | **0** | — |
| **retracted-pending-rerun** | **0** | — |

## What would overturn this

Stated in advance so the audit is falsifiable:

- Any frozen artifact shown to have been regenerated after `dc3b310` while retaining its frozen filename. Check: `git log --follow --oneline <artifact>` crossing a freeze tag.
- Discovery of a gold-value-in-prompt or gold-substitution path in the pre-freeze `scribe/` scorers that this audit's grep missed. Check: re-run the Finding-2 commands against `gate_scribe.py`, `gate_grounded.py`, `gate_absence.py`, `scribe/pointer/gate*.py`.
- A claim being restated on the span-port harness without a new claim ID.

## What this audit does *not* say

It does not say the span-port measurements are sound — that is exactly what the C1×C2 ablation is being run to find out, and the pre-run pilot (`artifacts/pilot_quote_absent.json`) already shows the span-port instrument has problems of its own (`exact_gold_span` on the floor; C2 structurally unable to fire). It says only that **no claim frozen on 2026-07-31 inherits those problems**, because the code that carries them did not exist and its equivalents were absent from the code that did.

**Related:** `research/preregistrations/PREREG_P1_leakage_2x2.md`.
