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
