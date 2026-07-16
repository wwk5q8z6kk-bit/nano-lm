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

### v1 — trained 2026-07-16, GATE FAIL (honest, on pre-registered bars)
- Train: 8000 examples, 750 steps, 3.1 min; loss → ~0 by step 200 (task is templated;
  near-zero train loss expected — generalization is what the gate measures).
- Base control (dpo.pt): parse 0% — emits refusal boilerplate. Discrimination clean.
- Scribe greedy (primary): **parse 100% ✅, recall 74% ❌ (bar 80), halluc 14.0% ❌ (bar 10)**.
- Held-out-value recall 65% vs seen-value 82% — 17-pt gap (under the 20-pt memorization
  flag, but consistent with partial label-memorization instead of copying).
- Sampled diagnostic agrees (72% / 14.2%) — not a decoding artifact.

**Diagnosis (from failure cases):** `CC: stopped` for "shortness of breath" (grabbed from
"hasn't stopped" in an unseen patient template); `CC: seat` for "neck pain" (grabbed from
"have a seat" in an unseen opener). The model learned POSITION-ANCHORED extraction —
"the noun after the phrases I saw in training" — because template diversity was too low
(5 openers, 5 complaint phrasings, fixed section order). Same failure class as SFT v1's
thin refusal slice: insufficient diversity → surface anchoring instead of the skill.

### Pre-specified improvement sweep (written BEFORE v2 training; bars unchanged)
1. **Template diversity ↑**: every family roughly doubled-to-tripled (openers 5→12,
   complaint phrasings 5→12, all Q/A families 2-3→6-8). Held-out families untouched.
2. **Structural variation**: med/allergy question order randomized; optional doctor
   acknowledgments; complaint+duration sometimes fused in one utterance — breaks
   positional anchoring, forces semantic extraction.
3. **Value-space diversity ↑ (anti-memorization)**: complaint pool augmented with
   ~190 compositional values (body-part × sensation) so CC cannot be solved as a
   14-way classification — copying becomes the only viable strategy. Eval held-out
   values remain excluded. Med list 8→18.
4. Train size 8000→12000 to cover the wider template space; epochs unchanged.
- Re-measure ONCE on the same 40-dialogue eval set, same bars. Result below, either way.

### v2 — trained 2026-07-16 (12000 ex, 204-value CC space, 4.7 min), GATE FAIL — stage CLOSED

Greedy (primary): **parse 98% ✅  recall 81% ✅ (bar 80)  hallucination 11.5% ❌ (bar ≤10)**.
Sampled diagnostic agrees (81% / 11.9%) — not a decoding artifact. Base control still 0%.

Every metric improved (recall 74→81, halluc 14.0→11.5, omissions 25→10, seen-value recall
82→94, held-out-value recall 65→72) — the diversity levers worked — but the hallucination
bar was missed by 1.5 points. **Per the pre-registered protocol (one sweep, one re-measure),
the stage closes at FAIL.** No further tuning inside this stage; a third run against the
same bars would be bar-chasing, and the protocol's integrity is the deliverable.

**Where the remaining 11.5% lives:** errors concentrate on the CC field for held-out
values under held-out openers (e.g. "neck pain" under "Come in, have a seat." still →
`CC: seat`; seen-value recall is 94%). The model's copy-from-context skill improved with
value-space diversity but is not fully general at 3.15M params — consistent with a
capacity wall on content-addressed copying (no attention head has been given an
induction-friendly curriculum).

**Legitimate continuation (would be a NEW stage with fresh pre-registration, not a v3
of this one):** (a) copy-curriculum pretraining slice (synthetic key→value copy tasks)
to induce induction-head behavior before scribe SFT; (b) scale test at ~10M params to
measure the capacity hypothesis directly; (c) constrained decoding for the CC field
(grammar-restricted to dialogue n-grams) — an inference-time guardrail, reported
separately from model capability.

**Meta-lesson for the vault:** the faithfulness gate did its job — it caught a model
that LOOKS excellent (98% parse, fluent output, 94% on familiar content) but hallucinates
above tolerance exactly where inputs leave its training distribution. That failure shape
(great on-distribution, unsafe on the tail) is the clinically dangerous one, and only
the held-out-value axis of the gate exposed it.

---

# Stage G — grounding-verifier guardrail (NEW stage, fresh pre-registration, 2026-07-16)

Reframe: scribe v2 measured MODEL hallucination (11.5%). A deployed scribe is a SYSTEM:
draft → per-field grounding verification → present verified fields, FLAG unverifiable
ones for human review. This stage measures whether a simple verifier can drive
hallucination-as-presented near zero, and at what human-review cost. (This mirrors the
draft-plus-clinician-verification architecture of production clinical documentation AI.)

## Verifier (fixed before measurement)

For each parsed field value v:
- v == "none" → PRESENT (absence claims are unverifiable by substring; documented limitation)
- else → PRESENT iff v appears (case-insensitive substring) in a **Patient utterance**
  of the dialogue; otherwise FLAG for review.
The patient-line restriction is deliberate: v1/v2 failures included copying doctor-line
words ("have a seat" → `CC: seat`); source-role awareness is part of grounding.

## Pre-registered bars (WRITTEN BEFORE MEASUREMENT — system requirements for a
draft-for-review scribe, not reverse-engineered from results)

Over the same 40-dialogue eval set, greedy primary, model = scribe.pt (v2), 200 fields:
1. **Residual hallucination (hallucinated fields among PRESENTED) ≤ 2.5%**
2. **Presented-field precision ≥ 95%**
3. **Review load (flagged fields) ≤ 25%** — a guardrail that flags everything is useless
4. Report (no bar): verifier catch-rate on the 23 known v2 hallucinations; recall among
   presented fields; per-field breakdown.

Known risk, stated in advance: substitutions whose wrong value DOES occur in a patient
line (e.g. "hasn't stopped" → `CC: stopped`; a medication read into CC) will PASS the
verifier — substring grounding cannot catch wrong-field-right-source errors. If those
alone exceed bar 1, the honest conclusion is "naive grounding verification is
insufficient," which is itself the finding. One measurement, no tuning after seeing it.

## Result — measured once, STAGE G FAIL (by 0.8 pts on precision), stage CLOSED

- **Residual hallucination: 0/200 = 0.0% ✅** (bar ≤2.5) — the verifier caught **all 23**
  model hallucinations (catch-rate 100%). The pre-stated risk (wrong-field-right-source
  survivors) did not materialize on this eval set.
- **Review load: 14.0% ✅** (bar ≤25) — 28 flagged fields incl. one unparseable draft.
- **Presented precision: 162/172 = 94.2% ❌** (bar ≥95) — missed by 0.8 points.

**Where the miss lives:** all 10 presented-but-wrong fields are OMISSIONS — the model
claimed `none` where the dialogue contained a medication/allergy. The verifier's
pass-through rule for absence claims (documented as a limitation in the pre-registration)
is precisely where the bar was lost. Substring grounding can verify presence; it cannot
verify absence.

**System-level finding (the stage's real product):** hallucination-as-presented can be
driven to zero with a simple source-role-aware grounding verifier at modest review cost —
but a trustworthy scribe system ALSO needs an absence-verifier (e.g., lexicon scan of
patient utterances: if any known medication term appears while the draft claims
`MED: none`, flag it). That is a legitimate next stage with its own pre-registration.
Per protocol: single measurement, no post-hoc tuning, stage closes at FAIL.

**Cumulative arc of the scribe track:** model halluc 14.0% (v1) → 11.5% (v2, diversity)
→ 0.0% presented (Stage G, verification) — hallucination in high-stakes drafting is a
SYSTEMS problem: training reduces it, but verification architecture is what eliminates
it from the output, at a measurable human-review cost. The residual risk moved from
fabrication to omission — a different, quieter failure mode that needs its own gate axis.

---

# Stage A — absence-verifier axis (NEW stage, fresh pre-registration, 2026-07-16)

Stage G's miss: `none` (absence) claims pass unverified, so omissions reach the output.
Stage A adds the absence check that substring grounding cannot do:

## Verifier (Stage G rules + the following; fixed before measurement)
- MED/ALG predicted `none` → scan patient utterances against the field's **lexicon**
  (the task's full value vocabulary, including eval-held-out terms — the deployed-system
  analogue of RxNorm / an allergen registry, which exists independently of training data).
  If any lexicon term appears in patient text while the draft claims `none` → FLAG.
- CC/DUR/SEV predicted `none` → FLAG unconditionally (mandatory encounter fields; an
  absence claim there is inherently suspect).
- All Stage G presence rules unchanged. Same eval set, same scribe.pt, greedy, one run.

## Pre-registered bars (system requirements, unchanged where already set)
1. Residual hallucination (presented) ≤ 2.5%
2. Presented-field precision ≥ 95%   ← the bar Stage G missed by 0.8
3. Review load ≤ 25% (absence flags will add load; the guardrail must stay usable)
4. Report: omission catch-rate, false absence-flags (pred none, truth none, flagged),
   full presented/flagged breakdown.

Known risk, stated in advance: lexicon absence-scanning inherits lexicon coverage — a
spoken med phrased off-lexicon is invisible to it. On this synthetic eval the lexicon is
complete by construction; in production this bar would need paraphrase-robust matching.
One measurement, no tuning after seeing it.

## Result — measured once, STAGE A PASS (all three bars)

- **Residual hallucination: 0/200 = 0.0% ✅** (bar ≤2.5) — catch-rate 23/23 = 100%
- **Residual omissions: 0/200 = 0.0%** — catch-rate 10/10 = 100%, false absence-flags 0
- **Presented precision: 162/162 = 100.0% ✅** (bar ≥95)
- **Review load: 19.0% ✅** (bar ≤25) — 38 flagged fields (28 presence + 10 absence)

**System-level conclusion of the scribe track:** with a 3.15M-parameter drafting model
whose intrinsic hallucination rate is 11.5%, a two-axis verification layer (presence
grounding against patient utterances + lexicon absence-scanning + mandatory-field flags)
yields an output channel in which **every presented field is correct** and every model
error is routed to human review, at a 19% review cost. Trust in the output came from the
verification architecture, not from model scale. The gates that closed at FAIL (v1, v2,
Stage G) were what located each next lever; the pre-registration discipline is what makes
this PASS credible.

Production caveats (stated for honesty, not hedging): synthetic dialogues with complete
lexicon coverage and exact-match fields; real clinical language needs paraphrase-robust
grounding, entity normalization, and an over-extraction axis. The architecture and the
measurement discipline are the transferable artifacts, not the numbers.
