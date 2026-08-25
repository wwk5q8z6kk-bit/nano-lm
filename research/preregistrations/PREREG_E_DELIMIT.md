# PREREG — E-DELIMIT: forced-choice span selection

**Registered 2026-08-25, before any arm ran.**
**Rule source:** §5 of `research/hypotheses/BOTTLENECK_2026-08-25_delimitation.md`,
committed at `23ede77` on `work/leakage-power-analysis` — i.e. before this file
existed and before any E-DELIMIT arm was run. The rule below is reproduced from
that commit unchanged. Everything under "Pre-commitments" is new to this file
and is registered here, before the run, for the reasons given.

---

## Authorization

Experiment-scoped authorization, granted by the owner 2026-08-25, recorded here
verbatim so the run carries its own provenance:

> Experiment-scoped authorization for E-DELIMIT is granted. Three arms, 12
> instances × 16 slots, same model and pinned revision, all leak channels
> closed, local MPS, \$0. Run it unchanged — preregistered arms and the kill
> condition as written, no mid-flight edits. If the kill condition fires, report
> H-delimit REFUTED and hand the span-port line back to the retrieval hypotheses
> without softening it.

The authorization was required — not waived by the run being free. Per
`docs/ACTIVE_NOW.md`: *"The distinction is cost / risk / evidential
significance, not 'free local versus paid cloud.'"* E-DELIMIT is a
pre-registered confirmatory contrast with a kill condition, so
`confirmatory_evidential_run = PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION`
governs; `local_zero_cost_exploratory_training = ALLOWED` does not cover it.

---

## The rule (reproduced from `23ede77`, unchanged)

**E-DELIMIT — forced-choice span selection.** Hold retrieval fixed; make
delimitation trivial; see whether the loss disappears.

Same 12 instances × 16 slots, same model, same greedy decoding, same leak
channels closed, canonical unified question form. One factor, three levels,
scored on the identical slot set so the contrast is paired:

| arm | output the model must produce |
|---|---|
| **A (control)** | current free-form quote |
| **B (menu)** | the located turn's candidate sub-spans enumerated; model picks an index |
| **C (offsets)** | character start/end offsets into the transcript |

Arm B is the discriminator. It removes *generation* of the boundary and leaves
only *selection* of it, while leaving retrieval exactly as hard.

**Pre-registered predictions.**
- **H5 true:** B converts most located slots into grounded ones —
  `asserted_grounded` in B ≥ **60% of LOCATED** (≥ 58/97), against 2/120 in A.
- **H1/retrieval true:** B changes almost nothing; the model cannot pick the
  right sub-span because it never had the value, so grounded stays < 25% of
  LOCATED.

**Pre-registered KILL condition for H5:**

> If arm B's `asserted_grounded` is **< 25% of LOCATED (< 24/97)**, H-delimit is
> **REFUTED for this model**. The boundary is then not merely unexpressed but
> absent from the representation, and the span-port line rejoins the retrieval
> hypotheses rather than standing apart from them.

**Between 25% and 60% → WEAKENED**; report both arms and do not round toward a
verdict.

**Power.** Paired binary over ≥97 informative slots, exact McNemar
(`nanoscribe/mcnemar_c1.py`); d ≥ 6 one-directional gives p < 0.05.

---

## Pre-commitments registered in this file, before the run

Each is a choice the rule above does not determine. Registered now so it cannot
be fitted to a result.

### P1 — Base commit: `9a3ecd4`, not the branch the rule was written on

The rule lives on `work/leakage-power-analysis` @ `23ede77`. That branch
**cannot host this experiment**, and running it there would silently reproduce
the defect that voided the C3 arm:

- `23ede77` carries the **superseded yes/no question form**
  (`"Does the patient mention 'migraines'?"`). The instrument that produced
  arm A's numbers carries the **unified wh form**
  (`"What does the {who} say about {identifier}?"`), added in `b707478`.
- `23ede77` has no `campaign_instances.py` and reports
  `campaign_v2_20260825` — 5 single cases. The 12-instance × 16-slot suite is
  `campaign_v2_multi_20260825`, on the L000 line.
- The two lines diverged at `0f6377c`; neither is an ancestor of the other.

Arm A is run `e04b3016`, produced at `9a3ecd4`. R2 makes a contrast legal only
if the question templates match, so arms B and C are built on `9a3ecd4`. The two
analysis-only scripts (`analyze_span_extent.py`, `mcnemar_c1.py`) are imported
from `23ede77`; they read JSON payloads and touch no part of the measurement
path.

### P2 — Arm A is reused, not re-run

Arm A is `e04b3016` (`asserted_grounded` 2/192, LOCATED 97/120). Re-running it
would, under R2, make every other cell of the finished leakage grid stale. Reuse
also guarantees the identical slot set that `mcnemar_c1.py` asserts on.

### P3 — Menu scope is the whole transcript

§5 says "the located turn's candidate sub-spans". Read literally that enumerates
only the gold-bearing turn, which hands the model the location and collapses
retrieval — contradicting the same paragraph's *"leaving retrieval exactly as
hard"*, and failing guard R5, under which an index-0 parrot must score at chance
rather than ceiling. **The menu is built over the whole transcript.** This is a
departure from the literal text, recorded as such.

### P4 — Menu order is keyed to the slot, never to position

Candidates are ordered by `sha256(slot_id || candidate_text)`. Transcript order
would make a candidate's index correlate with its turn, which is exactly what
would let the R5 parrot pick up real signal and stop being a chance baseline.

### P5 — Candidate generation, and the ceiling it implies

Candidates are every contiguous 1–5 whitespace-token run inside a turn, plus
each run's punctuation-trimmed form, never crossing a turn boundary. Bounds are
pre-committed from the gold distribution, not tuned: the longest gold span in
`campaign_v2_multi` is 5 tokens, and 87 of 120 are a single token. The trimmed
variant is required because gold is often the bare word inside a token carrying
a trailing period ("smoked" in "smoked."); without it 36 of 120 gold spans are
unreachable.

**`gold_in_menu` is emitted per slot.** A slot whose gold is absent from its menu
is `INVALID_NO_SIGNAL` for arm B, not a miss — otherwise a generator bug reads as
H5 REFUTED, which is the expensive wrong conclusion.

### P6 — Arm C is secondary and cannot refute H5

Arm C requires the model to do index arithmetic over transcript offsets, a known
weakness independent of delimitation, and it is the one arm whose transcript
rendering differs (each turn is prefixed with its start offset, so offsets are
well-defined at all). Arms A and B share byte-identical transcripts. **A poor
arm C result is not evidence about H5.** The kill condition is stated on arm B
and is evaluated on arm B alone.

### P7 — Instrument validation, run before any model call

Recorded before the model ran, on the fixture path, over all 192 slots:

| check | result | meaning |
|---|---|---|
| `gold_in_menu` | **120/120** | every gold span is reachable; no slot is `INVALID_NO_SIGNAL` |
| oracle (picks the gold index) | **120/120 exact** | the arm can express a win; nothing downstream blocks it |
| **R5 menu-parrot** (always index 0) | **2/120 = 1.7%** | **PASSES** — at chance, not at ceiling |
| R2 question hash across arms | `d83fd028fd54181f`, equal | the question is byte-identical across all three arms |
| R1 format hash across arms | 3 distinct values | the arms differ where they are supposed to |

Menu size: min 38, median 64, max 120 candidates.

---

## Condition and provenance

- Leak channels: `C1off_C2off_Qon_QSoff` — identical to arm A, unchanged by this
  experiment.
- Suite: `campaign_v2`, revision `campaign_v2_multi_20260825`, 60 encounters,
  192 slots, 12 instances.
- Model: Qwen2.5-1.5B-Instruct, pinned revision `989aa798`.
- Arm branches differ from their shared parent by exactly one line —
  `delimit.OUTPUT_FORMAT`. `git diff` is the R1 proof.

## Recheck

```bash
python3 nanoscribe/run_eval.py --suite campaign_v2          # per arm
python3 nanoscribe/analyze_span_extent.py --log <arm>.json --gold-tree .
python3 nanoscribe/mcnemar_c1.py --ref <armA>.json --trt <armB>.json \
    --ref-extent <armA_extent>.json --trt-extent <armB_extent>.json
```

## Status

**Registered, not resolved.** Append a RESULT section below without editing
anything above it.

---

# RESULT — 2026-08-25: arm B is VOID, the kill condition is not evaluable

Nothing above this line was edited. This section is appended on
`work/e-delimit-result`; the arm branches themselves are frozen by their runs.

**Full writeup:** `research/negative_results/RESULT_2026-08-25_E_DELIMIT.md`

| arm | run | `asserted_grounded` /192 | **LOCATED /120** |
|---|---|---|---|
| A — free-form (published baseline) | `e04b3016` | 2 | **97 (80.8%)** |
| A′ — free-form replication on the harness | `38b12909` | 2 | **97 (80.8%)** |
| B — menu (the discriminator) | `4de84c18` | **0** | **30 (25.0%)** |
| *R5 index-0 parrot on B's own menus* | *(software)* | *2* | ***23 (19.2%)*** |
| C — offsets (secondary, P6) | `aa779aba` | 0 | 2 (1.7%) |

**Verdict: arm B VOID — manipulation failed. H5 is UNTESTED.**

Arm B's `asserted_grounded` is 0, which read naively against the kill condition
(`< 25% of LOCATED`) would fire it. It must not be read that way. The arm's
stated premise — *"leaving retrieval exactly as hard"* — is refuted by its own
data: LOCATED fell 97 → 30, landing 7 slots above a constant index-0 baseline.
The contrast varied two things, so the precondition for the kill condition does
not hold and H-delimit is neither confirmed nor refuted. Reporting REFUTED here
would repeat the C3 error of `ddb5ce6` one level up.

**Mechanism.** The model's index picks are front-biased: median index 13.5
against a median menu size of 64, median relative position 0.211 (uniform ~0.5),
53% of picks below index 20, and indices 1–2 alone accounting for 33% of all
picks. Arm B measured long-list indexing, not boundary selection.

**Guard post-mortem.** P7's R5 check passed and was the wrong yardstick — it
verifies the menu does not leak, not that the model can use it. The catching
statistic is the parrot's **LOCATED** (19.2%), against which arm B's 25.0% is
nearly indistinguishable. **The next version of this prereg must add a blocking
LOCATED-invariance check**, evaluated before the primary endpoint is read.

**Control.** P1/P2 held: arm A′ reproduced the published extent census exactly
(2/95/0/8/15/72, LOCATED 97/120) and the across-instance vector to 4dp, so the
output-format refactor is behaviour-preserving. R1/R2 held: `question_template_hash`
equal across all three arms, `output_format_hash` distinct, arm B's diff against
its parent exactly one line.

**Not run here:** the repaired arm B (two-stage elicitation — ask for the turn,
then enumerate only that turn's sub-spans). That needs its own pre-registration
and its own experiment-scoped authorization.
