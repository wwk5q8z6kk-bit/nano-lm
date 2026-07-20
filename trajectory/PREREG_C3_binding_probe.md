# PREREG — C-3: transition/boundary/length factorial (emission-completion probe)

**Pre-registered 2026-07-20, before any candidate pool, eval instance, or kernel
exists.** Promoted by the C-1b mechanical verdict (PREREG_token_coverage.md RESULT:
lexical interference REFUTED, iso−contain = −4 pts, 0/77 containment substitutions,
independent recompute 1000/1000).

**Evidence-scoped framing (fixed wording):** on this held-value copying task, the
dominant observed error is *partial generation ending at lexical boundaries*, and
that pattern is inconsistent with the preregistered lexical-interference prediction.
(NOT claimed: "the model cannot predict tails" — that is broader than the evidence.)
The new candidate account is mechanistic: copy success may depend on whether the
model has learned a sufficiently strong transition from the generated head into the
remaining tail tokens, especially across lexical (word) boundaries. C-3 tests this
with factorial control.

## Q → P → M → D

**Q:** At fixed diversity (D80) and fixed slot (allergy), does P(complete copy) of a
held value depend on (T) head→tail transition availability in the training stream,
(B) boundary type at the critical junction, or (L) total token count — each
manipulated independently of the other two?

**Registered factors and levels:**

- **T — transition availability** at the value's critical junction (the token pair
  spanning its last intra-value boundary), computed mechanically as the bigram count
  of (junction-left token → junction-right token) over the D80-arm training OUTPUT
  stream (loss-bearing tokens; full-stream count recorded as covariate):
  - **T-avail** — junction bigram count ≥ 20;
  - **T-sep** — junction bigram count = 0, but BOTH junction tokens occur ≥ 20 times
    individually in the output stream ("seen separately, never consecutively");
  - **T-full** (positive control, small cell) — the value's ENTIRE token sequence
    occurs in the output stream (the C-1b I-xslot condition, known 100%).
  (T-zero-with-novel-unigrams is deliberately excluded — it confounds T with
  coverage; coverage is recorded as covariate only.)
- **B — boundary type** at the critical junction:
  - **B-sub** — subword boundary inside a single word (multi-token word);
  - **B-space** — whitespace boundary (two words);
  - **B-none** — single-token value (no junction; T undefined → these types anchor
    the intercept only). Constructible only if novel single-token values exist in
    the 4098-token vocab after hygiene; if the cell is empty, that is reported, not
    forced.
  - **B-punct** (optional) — hyphenated junction; included only if ≥4 hygienic
    candidates tokenize with a stable hyphen pattern.
- **L — total token count**: short (3–4) vs long (5–6), matched WITHIN every T×B
  contrast.

**Predictions (named accounts, C-1b anchors):**
- **H-transition:** T-avail ≫ T-sep at matched B and L (anchors: I-xslot 100/100;
  blue dye→"red dye" where the tail was trained).
- **H-boundary:** B-sub ≫ B-space at matched T and L (anchor: 38% of C-1b misses
  truncate exactly at whitespace boundaries; single-word novels largely flipped).
- **H-length:** short ≫ long at matched T and B (anchor: chia seeds 4tok flip vs
  pumpkin seeds 6tok fail).
- **H-stochastic (null):** all three contrasts ≤ 15 pts AND ≥20% of types
  seed-unstable → the residual is per-type stochastic memorization boundary; then a
  lower-level representation/attention probe is promoted — NOT another lexical
  narrative.

## Design (fixed now)

**Cells:** the T×B×L factorial over {T-avail, T-sep} × {B-sub, B-space} × {short,
long} = 8 core cells, ≥5 types each; plus T-full control (≥3), B-none (as
constructible), B-punct (optional). C-1b's 34 types are re-scored as bridges (known
states; excluded from cells).

**Matched-pair obligations (pool generator must emit these explicitly):**
- same token count, different transition class (T-avail vs T-sep pair per L level);
- same transition class, different boundary type (B-sub vs B-space pair per T level);
- same head, different tails; same tail, different heads (≥3 pairs each, to
  separate head-identity from junction effects).

**Orthogonality proof (HARD GATE — pools are INVALID without it):** the pool
generator must print and store a cell-balance table showing, for every registered
contrast, (a) mean token count difference ≤ 0.5 across the contrasted cells,
(b) boundary-count identical across T contrasts, (c) junction bigram count
distributions non-overlapping between T-avail (≥20) and T-sep (=0), (d) coverage
distributions overlapping across T cells (coverage must NOT separate T classes;
if it does, candidates are re-drawn under the same frozen procedure). Per the
owner directive, NO pool, eval, or kernel work proceeds until this table proves
T, B, L independently manipulated.

**Hygiene (mechanical, as C-1b):** no word shared with ALG_TRAIN_80; no full
trained value contained; no MED_TRAIN word (except T-full controls, which are
drawn from med-trained values by definition); no template content word (strip-s);
no modifier reuse across cells; no C-1b candidate reuse; all values excluded from
every training pool; plausible-allergen surface form.

**Arms:** ONE training arm (D80, v2 recipe source-patched, scale-10M frozen base,
full FT — byte-identical to C-1b). **THREE FT seeds (0,1,2)**; C-1b measured 24%
seed-variant types, so flip states are seed-majority (≥2/3) and 3-way-unstable
types are reported separately and excluded from contrasts.

**Eval:** K=5 instances, seeds 20260750–54, generator procedure of
gen_interference_eval.py; held dialogues cycle uniformly over all cell types +
bridges; seen-alg from D80.

**Instrumentation (mandatory):** per-item logs as C-1b PLUS the generated ALG-field
token-id sequence, enabling mechanical computation of: truncation position
(token index where output diverges from the true sequence), head-only / tail-only /
full-copy classification, token-level edit path, wrong-trained-tail,
morphological normalization, omission, garble, per-type seed flips.

## Measurements (fixed now)

Per-type flip table primary (as all prior experiments). Registered outcome:
**P(complete copy | T, B, L)** per cell, seed-majority; **co-primary mechanism
metric: tail-truncation rate per cell** (fraction of misses whose output token
sequence is a strict prefix of the true sequence ending at the critical junction).
Secondary: truncation-position histogram; binding-leakage count (output = fragment
of another slot's value in the same dialogue); morph-normalization rate.

## Decision rules (fixed now, before any pool exists)

All contrasts on seed-majority flip states over cell types, matched per the
orthogonality gate:

- **H-transition SUPPORTED:** flip(T-avail) − flip(T-sep) ≥ 40 pts (matched B, L);
  REFUTED ≤ 15; else UNRESOLVED.
- **H-boundary SUPPORTED:** flip(B-sub) − flip(B-space) ≥ 40 pts (matched T, L);
  REFUTED ≤ 15; else UNRESOLVED.
- **H-length SUPPORTED:** flip(short) − flip(long) ≥ 40 pts (matched T, B);
  REFUTED ≤ 15; else UNRESOLVED.
- Truncation-locus check: ≥ 60% of B-space misses truncating exactly at the
  whitespace junction confirms the boundary as the failure locus (mechanism-level,
  independent of the flip contrasts).
- T-full control must reproduce ≥ 90% (I-xslot retrodiction); failure of this
  control voids the run (instrument assumption failed → UNRESOLVED, investigate).
- **All three refuted + ≥20% seed-unstable → H-stochastic SUPPORTED** → promote a
  representation/attention probe (Stage M direction), not another lexical account.

## Cost

3 FTs (~4 min each, A6000) + 3 × 5-instance scoring (~13 min each) ≈ ~55 min GPU
≈ **$0.45** (RunPod A6000 $0.49/hr). Kernel adapted from run_interference_10m.py.

## Status

Pre-registered (this document). **HARD GATES before any pool/eval/kernel artifact:**
(1) the peer (Sonnet-lane) independent audit of C-1b confirms the design inputs
cited above; (2) the orthogonality proof procedure above is implemented and its
balance table validates. Artifact order after the gates: gen_c3_pools.py +
c3_pools.json (with bridge-reproduction falsification gate + orthogonality table) →
eval instances → kernel → run → RESULT appended here.

## RESULT (2026-07-20, run complete)

Run: RunPod RTX 4090, pinned commit 45531e2, pools 943b446, eval 2e6c0ce, kernel
93db8d1+45531e2 (kernel amended pre-launch: generation-cap safety guard —
`generate_ids` now returns `stop_reason`; asserted an 8-tok margin against the
true max observed response length of 47 tok under `max_new=64`, 17-tok margin —
fired once at runtime, `cap_confound=1`, correctly excluded from the
truncation-locus denominator). Artifacts (immutable, commit a07862d):
results_c3_10m.json (sha256 1db35820...), outputs_c3_seed{0,1,2}.jsonl (sha256
6805b3b1.../a9bb0764.../52966d8b..., 1000 records each), run_c3_log.txt (sha256
6a7de207...). Base checkpoint sha256 892180f0...c2fc88d — independently verified
against the current canonical GitHub release asset (downloaded fresh, not just
kernel self-report). All 5 checksums verified identical between the pod and local
before teardown; pod deleted, billing confirmed stopped. Independent recomputation
from raw per-item JSONL (fresh code, not the kernel's own computation path)
matched every decision-rule number exactly: H-transition/H-boundary/H-length
deltas, T-full rate, truncation-locus rate+n, and the full 23-type unstable set.

**Triple cross-check.** A second, independently-authored recompute harness
(`recompute_c3.py`, peer-built) was found to have a real bug: it computed the
seed-unstable type set but never applied it as an exclusion filter before
computing the three frozen contrasts, contradicting this PREREG's own text
("3-way-unstable types are reported separately and excluded from contrasts")
and the kernel's implementation — silently reporting +0.0/−16.7 instead of the
correct +1.7/−8.3. Verified by toggling the filter on the real run data: with
the filter applied (matching the frozen rule), all three independent
implementations — the kernel, this session's from-scratch recompute, and the
fixed peer harness — agree exactly. Both REFUTED verdicts are unaffected by
which version is used; the point estimates and any CI built on them are not.
Fixed in commit 823e1ca; the peer harness's own 6 fixture tests still pass.

Two supplementary analyses from the peer harness, valid independent of the bug
above (continuous per-type recall, not the binary flip contrast; not subject to
the unstable-exclusion rule): **dose-response within T-avail** — junction bigram
counts there span 25–425; Spearman(count, mean 3-seed recall) = **+0.47** across
n=24 T-avail types, i.e. a moderate graded relationship with transition
frequency even though the binary T-avail-vs-T-sep contrast is REFUTED (worth
noting as a secondary, exploratory signal — not a contradiction of the frozen
verdict, since SUPPORTED requires ≥40pts on the registered binary contrast, not
a nonzero rank correlation). **Type-bootstrap 90% CI on H-transition** (2000
resamples, stable types only): **[−19.2, +23.3]**, centered near zero and
spanning it — consistent with, and quantifying the uncertainty behind, REFUTED.

**Mechanical verdicts under the frozen thresholds:**
- **H-transition: REFUTED.** flip(T-avail) − flip(T-sep) = **+1.67 pts**
  (rule: ≤15 → refuted), matched across B×L strata. Transition-availability (junction
  bigram frequency) does not predict copy completion.
- **H-boundary: REFUTED.** flip(B-sub) − flip(B-space) = **−8.33 pts** (rule: ≤15 →
  refuted). Direction is slightly negative (B-sub marginally worse), within the
  refuted band — not read as a real reverse effect.
- **H-length: UNRESOLVED.** flip(short) − flip(long) = **+25.0 pts** (band: 15–40).
  Point estimate trends toward the predicted direction but is not decisive under
  the frozen rule.
- **T-full control: 100% (n=9).** Retrodiction passes; run is NOT void.
- **Truncation-locus check: 12.36%** of B-space misses (n=178, 1 cap-confounded
  case excluded) truncate exactly at the whitespace junction (rule: ≥60% →
  confirms). **Does not confirm** the boundary-as-failure-locus mechanism.
- **Seed instability: 24.7%** (23/93 types) — consistent with C-1b's 24%.
- **H-stochastic: NOT SUPPORTED.** The registered AND-gate requires all three
  factors REFUTED; H-length is UNRESOLVED, so the gate does not fire, despite the
  seed-instability rate being independently consistent with a stochastic account.
  This is stated as a gate outcome, not a judgment call — no threshold was
  adjusted post-hoc to reach it.

**Uncertainty context (not part of the frozen decision rule, added post-hoc as a
transparency check, does not override the verdicts above):** cell sizes after
unstable-type removal are small (n=2–6/cell). Wilson 95% CIs on cell flip rates
range as wide as [34%, 100%]; every short-vs-long matched pair's 95% band
overlaps. This confirms H-length's +25pt point estimate is noise-dominated at this
n, reinforcing (not contradicting) UNRESOLVED rather than suggesting a near-miss
on SUPPORTED.

**Error-class census** (815 misses across all held types; 396 restricted to the
core T×B×L cell types under test): the dominant class is **morphological_near_miss**
(43.7% of cell-type misses, vs truncation at 11.9%) — a marked departure from
C-1b's descriptive framing (38% of C-1b's misses, across all interference classes,
were word-boundary tail truncations; these are not identical measurements, since
C-1b's population was not restricted to C-3's B-space-only stratum, but the shift
is still notable). Of morphological_near_miss cases, ~60% are clean singular/plural
suffix variation (daisy seed→"daisy seeds", cane sting→"cane stings", rose
hips→"rose hip") — the model defaults to a differently-inflected form rather than
truncating or substituting. The remainder are other small-edit-distance misses
(e.g., a spurious "an" insertion: apigenin→"anpigenin"). Restricted to the B-space
truncation-locus population specifically (n=178), morphological_near_miss is still
dominant at 53.9%, with truncation at only 12.4% — the population the boundary
account specifically predicted should truncate mostly does not; it mostly
re-inflects.

**Frequency-imbalance caveat:** the pre-launch read-only diagnostic
(c3_balance_report.py) flagged 3/4 T-contrast cells with junction-token frequency
|SMD| > 0.8 (direction inconsistent across cells). This is a covariate outside the
frozen orthogonality gate. Because H-transition's verdict is REFUTED (a null
result), a frequency confound cannot manufacture it — an imbalance could only mask
a true effect, not fabricate a false null — so this does not threaten the verdict
above; noted for completeness, not as a limitation requiring a re-run.

**What this closes and what it opens:** none of the three pre-registered
mechanistic accounts (transition-availability, boundary-type, length) reached
SUPPORTED. Two are cleanly REFUTED; the third is UNRESOLVED but noise-dominated at
this sample size, not a suppressed trend. The auxiliary boundary-locus
corroboration also fails to confirm. The dominant failure mode across strata is
morphological re-inflection (chiefly pluralization), not truncation — a genuinely
new descriptive finding this design was not built to explain, since it isn't one
of the three registered factors. Per the frozen rule, H-stochastic does not fire
(the AND-gate is not met), so no probe is promoted automatically by this result;
the next step is an owner-level choice among (a) a length-focused follow-up with a
larger, better-powered pool given the Wilson-band evidence that C-3's length cells
were underpowered, (b) a new probe targeting the morphological re-inflection
finding itself (does the model have a default/majority inflectional form it falls
back to under uncertainty?), or (c) closing the lexical-mechanism line of inquiry
here and moving to a representation-level probe — none of which C-3's own decision
rule mandates, since H-stochastic was not supported.

## INTERPRETIVE NOTE (2026-07-20, pre-result — frozen design property, no threshold changed)

Recorded before C-3 results exist, from the committed pool manifest alone; changes no
threshold and no pool. The T factor is operationalized as junction-bigram count ≥ 20
(T-avail) vs = 0 (T-sep). These are therefore **frequency-disjoint by construction** —
they cannot overlap, so a frequency-matched median-split of the H-transition contrast is
impossible. This is not a confound to remove; it is the definition of "transition
availability" in this world. The design correctly controls the alternative account
(T-sep's individual junction tokens satisfy the frequency floor — only their *adjacency*
is unseen), so the contrast isolates the junction transition rather than token rarity.

Consequences for the eventual verdict language (binding now):
- A SUPPORTED H-transition must be reported as **"held values whose internal head→tail
  junction was observed ≥20× in training outputs copy better than those whose junction
  was never observed (individual tokens frequency-matched)"** — NOT as "the model cannot
  predict unseen tails" (broader than the evidence) and NOT as "junction frequency is
  irrelevant."
- The supporting graded evidence is the **within-T-avail dose-response** (junction counts
  span 25–425; `recompute_c3.py` reports Spearman(count, recall)). A positive rho grades
  the effect on transition frequency; a flat rho supports a binary seen/unseen threshold.
- The run's own `c3_balance_report.json` flags junction-frequency SMD up to 2.42 in the
  B-space cells: this is the same intrinsic disjointness surfacing per-cell, correctly
  marked non-blocking (it cannot invalidate the frozen pool). It is a scoping caveat, not
  a defect.

Independent recompute harness: `trajectory/recompute_c3.py` (reads only raw
outputs_c3_seed{0,1,2}.jsonl + the frozen manifest; imports no kernel code; applies the
frozen rules; adds the dose-response + type-level bootstrap CI the kernel does not). Run
it against the retrieved logs to cross-check `results_c3_10m.json` (evidence level 2 vs 3).
