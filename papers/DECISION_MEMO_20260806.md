# Decision memo — the next experiment is vocabulary diversity, and it is justified

**2026-08-06.** Subtask 20. Names the single next model experiment and the shape
of its preregistration, on the evidence in `RESULT_PER_STATE_DIAGNOSIS.md`.

---

## 1. Is a model experiment justified at all?

`PLAN_20260805_SURFACE_ROBUSTNESS.md` gated any model work behind "the
instrument is trustworthy." That gate is now partly satisfied and the memo must
be honest about which part.

**Satisfied:** the harness exists, is tested (584 passing), reproduces an
independent measurement to four decimals through a second code path, and has
already overturned two standing diagnoses — H6's `absent` failure (lexical, not
architectural) and `conflicting`'s (lexical, not structural). It has also
refused every arm-level claim it could not support, and caught an error of mine.

**Not satisfied:** two seeds. Every arm-level ordering is withheld. The
instrument can compare *distributions* (familiar vs unfamiliar vocabulary) but
not *phrasings*.

That is enough. The experiment proposed below is a distribution-level
intervention measured by a distribution-level comparison, which is exactly the
resolution the instrument currently has. **A model experiment is justified.**

## 2. The experiment

**Hypothesis (H7-V).** The model's epistemic-state failures are caused by the
narrowness of its training vocabulary, not by its architecture. Widening the
vocabulary — without changing the architecture, the objective, or the parameter
count — will raise held-out surface robustness.

**Why this and nothing else.** Every span-carrying state reaches 95–100% on
familiar wording and collapses on unfamiliar wording. The training pools are
**8 denial phrasings, 6 hedges, 86 values**. Three alternatives are retired on
evidence, not preference:

| retired | why |
|---|---|
| H7 / a state head | the state machinery already hits 95–100% when words are familiar |
| deterministic composition of existential probes | composition is not what fails; and the rule it depends on generalises at ~3% |
| rung-1 scale ($150) | a larger model trained on 8 denial phrasings learns 8 denial phrasings |

**Intervention.** Regenerate the training partition with an expanded surface
vocabulary — target ≥10× the current phrasing count per concept — holding the
world model, field set, document structure, and gold semantics fixed. Only the
wording varies. Architecture, objective, optimiser, and parameter count are
unchanged from H6, so the comparison is clean.

**Falsification.** If held-out surface robustness does not improve while
in-distribution accuracy holds, the lexical account is wrong and the failure is
representational after all. That outcome returns H7 to the table with a much
sharper motivation than it has today.

## 3. Three constraints the preregistration must fix in advance

**(a) Train and evaluation lexicons must be disjoint, and split before use.**
`data/external/negspacy` and `data/external/medspacy` are currently marked
**evaluation-only**. If their phrasings enter training, they stop measuring
generalisation — the project would be grading itself on its own textbook. The
preregistration must partition each external inventory into a training half and
an evaluation half, by a rule fixed before anyone looks at results, and record
the split's hash. This is the single easiest way to accidentally destroy the
instrument, and it would not be visible in any accuracy number.

**(b) ≥3 seeds, specified up front.** `RESULT_SURFACE_HARNESS_RUN1.md` §4b
established that a third H6-family seed is blocked by design — the trainer pins
its own SHA and enforces the frozen seed set. A *new* experiment is not bound by
H6's seed tuple and must declare ≥3 from the outset. The justification is now
measured rather than stylistic: out-of-distribution per-arm accuracy moves 31.5
points between two seeds at Kendall τ = 0.00, and `uncertain` scores 76.0% vs
43.6% on *identical, unmodified* documents.

**(c) Gates denominated in gold, plus a surface clause.** Following the H6 gate
design — which the audit found degeneracy-safe and which
`nano_ai/tests/test_gate_degeneracy_safety.py` now pins — every risk bar must be
ANDed with absolute floors over fixed gold denominators. New for H7-V: the
primary gate must be `surface_robust_accuracy` (min over arm means), **not** a
single held-out number. H6 was decided on two strings; that must not recur.

## 4. What must not be claimed

- **This does not reverse H6.** Its threshold was frozen in advance and measured
  correctly; the verdict stands. What changed is the diagnosis.
- **`uncertain` may not be fixed by this experiment.** It is the one state weak
  *in-distribution* (39.8-point spread on trained phrasings). Vocabulary breadth
  addresses transfer, not a concept never learned. Expect it to lag, and do not
  read that as the hypothesis failing.
- **None of this transfers to real documents yet.** Everything is synthetic
  clinic dialogue. The open-licensed dogfood corpus remains the only route to
  knowing otherwise, and it is the natural successor to H7-V regardless of
  outcome.

## 5. Cost and sequencing

Data regeneration and training are CPU: H6 trained in 1,301 seconds per seed at
$0. Three seeds is roughly an hour of local compute. **No paid provider is
required and none is proposed.** The $150 rung-1 budget stays deferred; §2
retires the argument for it.

Sequencing: (1) write the H7-V preregistration including the lexicon split and
its hash; (2) regenerate data; (3) train ≥3 seeds; (4) evaluate through the
existing harness, which needs no changes to accept new checkpoints.

## 6. The finding worth carrying out of this cycle

Independent of Nano: **a benchmark whose target concept has ten surface
realisations cannot distinguish a model that learned the concept from one that
learned the list** — and every gate built on it inherits that ambiguity. Two
model generations were diagnosed against such a benchmark before anything
measured the confound. The instrument that resolved it is ~200 lines and cost
nothing to run.
