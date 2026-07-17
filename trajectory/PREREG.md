# Stage T — scaling trajectory of the held-out copying gap (pre-registered 2026-07-17, before any measurement)

## Question under estimation

How does the held-out-value recall gap (seen-value recall − held-out-value recall on
the scribe task) behave as model capability increases? This is an ESTIMATION stage,
not a gate stage: the deliverable is a curve with confidence intervals, interpreted
against bands fixed below. Evidence to date: gap ≈ 22 pts at 3.15M (nano v2), ≈ 23 pts
at 10M (Stage S) — unmoved by curriculum (Stage C) or 3.2× scale (Stage S).

## Design — two arms, confound stated in advance

**Arm 1 (clean scaling, finetuned):** finetune open pretrained checkpoints on the
scribe v2 data recipe (`scribe/build_scribe_data_v2.py`, verbatim: 12,000 examples,
3 epochs, LR 1e-4, no gibberish slice) across a capability ladder:

- Pythia-160M, Pythia-410M, Pythia-1B (EleutherAI suite: consistent pretraining data
  across sizes — isolates scale). Optional extension if budget allows: Pythia-2.8B.
- Own-stack anchors already measured: nano 3.15M, scale 10M (no re-run; prior audited
  numbers carried forward).

Arm 1 changes ONE variable (scale) relative to Stage S up to tokenizer/pretraining-data
differences, which are acknowledged: Pythia uses its own tokenizer and The Pile, not
our 4096-BPE + FineWeb. The within-Pythia trend (160M→410M→1B) is the clean comparison;
the nano/scale anchors are joined with that caveat visible.

**Arm 2 (frontier behavioral probe, prompted):** 3 API models spanning capability
tiers (small / mid / frontier of one provider family, exact IDs recorded at run time
in RESULTS before any output is scored). Few-shot prompt, FROZEN before measurement
(see prompt-freeze protocol). Arm 2 does NOT isolate scale — adaptation method differs
(prompting vs finetuning). It bounds the phenomenon where deployment happens. If Arm 1
and Arm 2 disagree, that disagreement is reported, not reconciled post-hoc.

## Instrument and contamination control

- **The benchmark is the generator, not any single instance.** The scientific object
  is the eval distribution defined by `scribe/build_scribe_data.py` (v1), lines
  119–130: 40 dialogues, first 20 with held-out slot VALUES, all 40 with held-out
  TEMPLATE families. (Corrected in Amendment 1: v2 changes the TRAINING recipe only
  and reuses the v1 eval set byte-identical — the eval generator has been v1 at all
  stages.) Eval sets are seeded samples from this distribution. Changing the
  generator is a benchmark version bump, never a silent patch. Cross-stage
  comparability = same generator version + reported instance seed.
- Instance 0: `scribe/scribe_eval.json` (40 dialogues, 200 fields, seen/held-out
  value split) — the historical instance, byte-identical to all prior stages,
  retained for cross-stage anchoring.
- **Contamination hazard, stated:** the repo became public 2026-07-16/17. Any model
  with training data past that date may have seen instance 0, corrupting the
  seen/held-out distinction — the exact metric under study.
- Control: instance T (`trajectory/scribe_eval_T.json`) — GENERATED 2026-07-17,
  seed **20260717** (date-derived, fixed before generation), via
  `trajectory/gen_eval_instance.py` (execs v1 definitions verbatim, re-runs only
  the eval loop; training build never executed). QA at generation: schema identical
  to instance 0; 40 items, 20 held-value; **0/40 byte-identical dialogue collisions**
  vs instance 0. Equivalence check still required before measurement: one
  cutoff-clean model (any Arm-1 Pythia; The Pile predates the repo) is scored on
  BOTH instances; per-metric difference must be < 5 pts, else the generator is
  seed-sensitive — stop, diagnose, re-register.
- **Residual channel, acknowledged:** publishing the generator publishes the
  distribution. Instance regeneration defends against memorization of instances,
  NOT against a model learning the template family from the public repo. No clean
  fix exists without breaking reproducibility; the standing monitor is the
  instance-0-vs-instance-T discrepancy on post-cutoff models — a large gap is
  reported as evidence of repo contamination, not averaged away.
- Arm 1 models: pretraining data predates the repo → primary set is valid; fresh set
  run as confirmation.
- Arm 2 models: fresh set is primary; original set reported alongside as a
  contamination probe (a large original-vs-fresh discrepancy is itself evidence of
  memorization and will be reported as such).

## Metrics (identical scorer for all models: `scribe/gate_scribe.py` logic)

Per model: parse rate, recall, hallucination rate, omissions — and the primary
quantity: **gap = seen-value recall − held-out-value recall**, with 95% CI by
bootstrap over dialogues (n=40, 10,000 resamples).

Controls: Arm 1 — each base (un-finetuned) checkpoint must fail parse (< 50%);
if a base model passes parse zero-shot, that is reported and the finetuned/base
delta becomes the tracked quantity. Arm 2 — no base control possible; stated.

## Prompt-freeze protocol (Arm 2)

Prompt developed exclusively on a dev split (`trajectory/scribe_dev.json`): 10 NEW
dialogues from the v1 eval distribution (reference corrected in Amendment 2), seed
**20260718** (= instance-T seed + 1), generated by `trajectory/gen_dev_split.py` —
never scored, never overlapping either eval instance (QA: 0/10 collisions).

Dev-split structure (rationale in Amendment 2): items 0–4 are exemplars (SEEN
values, TRAIN templates) — the few-shot pool; items 5–9 are sanity targets (HELD
templates, 2 with held values) — never scored, never embedded in the prompt.
Constraint: held-out values must NEVER appear in the prompt (in-context exposure
would collapse the held/seen distinction for Arm 2); verified at generation.
Exemplars use train templates to mirror Arm 1's training exposure, keeping the
arms analogous.

The frozen prompt is `trajectory/arm2_prompt.txt` (3 exemplars embedded,
`{dialogue}` substitution slot). It is committed BEFORE the first eval-set API
call; any revision before first measurement requires a visible commit; NO edits
after first measurement. One greedy (temperature 0) run per model per eval set.
No retries.

## Interpretation bands (fixed now)

Let G_top = gap of the most capable model in each arm, with 95% CI.

- **PERSISTS:** G_top CI lower bound ≥ 10 pts in both arms → the failure mode spans
  the tested capability range; verification-architecture claim generalizes as stated.
- **THRESHOLD:** Arm-1 gap declines with scale AND Arm-2 frontier G_top CI includes
  0 (upper bound < 5 pts) → capability-threshold interpretation; report the scale
  region where the decline occurs; verification claim narrows to below-threshold
  models + calibration/tail roles above it.
- **DIVERGENT:** arms disagree (e.g., finetuned ladder gap persists, prompted
  frontier gap closes) → adaptation method, not scale alone, is implicated; this
  becomes the headline and the next stage's hypothesis.
- Anything else: reported as measured, no forced classification.

One measurement per cell. No post-hoc band adjustment. The curve is published
regardless of which band it lands in.

## Companion stage (registered as intent, separately pre-registered before running)

**Stage M (mechanism) — ONE question, registered now:** at the point of held-out-value
failure, does the model attend to the source-value tokens and fail to copy them
(use failure), or never retrieve them at all (retrieval failure)? Method sketch:
attention inspection is necessary but not sufficient — activation patching of the
source-value residual stream into failing runs distinguishes "attends but doesn't
use" from "stores elsewhere." Runs on the open-weight ladder (3M, 10M, Pythia rungs)
using the same finetuned checkpoints Arm 1 produces — retain all checkpoints. The
answer ranks the intervention queue: retrieval failure → attention/architectural
interventions (pointer/copy mechanisms); use failure → objective interventions.
M's full protocol and bars are NOT set here; scope is capped at this one question.

## Axis lattice (future structure, not part of Stage T)

Stage T varies capability and holds fixed: training objective, instruction tuning,
retrieval, context length, domain, prompt style, decoding strategy. Each of these is
a candidate axis for characterizing the space in which the failure appears; each
costs its own pre-registered stage and is prioritized by information value, not
enumerated. Recorded here so "scale" does not silently become the only explanatory
variable in the program.

## Budget and venue

Arm 1: Pythia finetunes ≤ 1B fit Kaggle T4 free tier (scribe finetune at 10M took
3.3 min; 1B est. < 2 h with LoRA fallback if full finetune OOMs — if LoRA is used it
is used for ALL Pythia rungs, never mixed). Arm 2: ~6 runs × 40 dialogues, trivial
API cost. Total wall-clock estimate: 1–2 days.

**Run-logging contract (every Arm-1 run):** GPU type and wall-clock hours, all
random seeds, exact hyperparameters (and LoRA rank/alpha/targets if used),
intermediate checkpoint cadence — intermediate checkpoints RETAINED, not just
final (emergence timing of the seen/held-out gap along the training trajectory is
a free secondary observable, reported descriptively, no bars).

## Amendments (all before any measurement)

**Amendment 1 (2026-07-17, pre-measurement):** instrument reference corrected —
the eval generator is `build_scribe_data.py` (v1), not v2; v2's own header states
the eval set is reused byte-identical from v1. Discovered during instance-T
implementation. No bars, bands, arms, or metrics changed. Instance-T seed
(20260717), generation script, and QA results recorded in the instrument section.

**Amendment 2 (2026-07-17, pre-measurement):** (a) the prompt-freeze section's
generator reference carried the same v1/v2 error Amendment 1 fixed in the
instrument section; corrected. (b) Dev-split structure specified: exemplars are
seen-value/train-template because held values must never appear in-context and
train-template exemplars mirror Arm 1's training exposure; sanity targets are
held-template. Seed 20260718 recorded. Frozen prompt file recorded
(`trajectory/arm2_prompt.txt`). No bars, bands, arms, or metrics changed.

## What would falsify the design itself

- Regeneration equivalence check fails (> 5 pt drift) → generator is
  seed-sensitive; stop, diagnose, re-register before measuring.
- Pythia base models pass parse zero-shot → the task is too easy for the ladder;
  redesign eval difficulty before proceeding (reported, not silently patched).
