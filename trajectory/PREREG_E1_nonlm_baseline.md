# PREREG — E1 Non-LM / constrained baseline (program kill-gate)

**Pre-registered 2026-07-30, BEFORE any non-LM baseline training, scoring, or
utility evaluation exists.** Highest-ROI experiment in the program after the
Scientific Research Council (2026-07-30). Overrides the previous Phase C residual
queue. Design, utility function, decision rule, and falsifiers fixed here so a
later execution cannot be accused of post-hoc substrate defense.

## Status

**Executed 2026-07-30; official M0 closed 2026-07-31. Program verdict: KILL
(H-substrate).** Design remains frozen. Results: `trajectory/results_e1_nonlm_*.json` +
`trajectory/results_e1_utility.json`.

Decision vs **local M0=scale10m**: **KILL** (best non-LM = M1_template,
margin +0.145 on default \(U\); also M2_dict_span beats M0). Official M0
scored on RunPod CUDA fp16 (frozen recipes): Pythia-160M LoRA
\(U=0.925\) · ownstack Chinchilla-160M+LoRA \(U=0.898\) →
\(\max U(M0)=0.925\). Vs M1 \(U=0.999\) / M2 \(U=0.886\) under
\(\delta=0.05\): **KILL** (margin \(+0.074\) vs max official M0;
sensitivity_flip=false). AMENDMENT 1 dominance still holds. Ecology tag:
**general**. Venue: RunPod RTX 3090 (`e1-official-m0`).

## Entanglement this experiment breaks

Two objects were sold as one story:

1. **Science object:** when/why small transformers fail at exact OOD span emission.
2. **Systems object:** whether propose→verify→abstain yields certified presented precision.

E1 asks the prior question neither object answers: **is a generative small LM the
right substrate for this task under an explicit decision-theoretic utility?**

If a non-generative method matches or beats LM+verify on \(U\), the LM-centric
program thesis loses ROI even if every transformer experiment remains internally
valid (council theory **T5 — wrong substrate**).

## Question → Prediction → Measurement → Decision

**Q:** On the identical scribe extraction task and faithfulness instrument, does a
non-generative or constrained baseline achieve expected utility greater than or
equal to the best measured generative configuration (own-stack or Pythia corner),
with and without the existing deterministic verifiers?

**Predictions (falsifiable, stated before any baseline run):**

- **H-substrate (kill):** at least one non-LM / constrained method achieves
  \(U \ge U(\text{best LM+verify})\) on the frozen utility below → generative-LM
  frame **demoted**; science track may continue as measurement-only; fabric /
  architecture track **gated**.
- **H-LM-necessary:** every serious non-LM baseline has \(U < U(\text{best LM+verify})
  - \delta\) with \(\delta\) pre-registered below → LM-centric frame **survives
  this gate** (does not prove optimality; only fails to kill).
- **H-verify-orthogonal:** verification lifts \(U\) similarly for LM and non-LM
  proposers → systems paper stands independent of substrate; if verify helps only
  the LM, the systems claim is entangled with LM error structure (report loudly).

## Utility function (FOUNDATIONAL — written before metrics)

Decision the system optimizes: **emit a structured clinical summary line that a
downstream chart can trust, under bounded human review budget.**

Define per evaluation instance (then mean over m0–m4):

\[
U = \alpha\, P - \beta\, M - \gamma\, \rho - \lambda\, L - \kappa\, C
\]

where:

| Symbol | Meaning | Unit | Default weight (frozen) |
|---|---|---|---|
| \(P\) | Presented precision under the deployment presenter (verify-on or verify-off arm) | [0,1] | \(\alpha = 1.0\) |
| \(M\) | Miss rate = 1 − recall on fields that should be emitted (omissions + wrong under soft/human policy below) | [0,1] | \(\beta = 0.5\) |
| \(\rho\) | Review load = fraction of fields routed to human | [0,1] | \(\gamma = 0.3\) |
| \(L\) | p50 end-to-end latency per dialogue | seconds | \(\lambda = 0.02\) / s |
| \(C\) | Relative compute cost vs 10M greedy scribe (wall-clock × device class normalized) | dimensionless ≥0 | \(\kappa = 0.05\) |

**Liability proxy (mandatory report, not inside \(U\) v1):** count of
fabrications+substitutions that would have been *presented* without review.
v1 keeps liability out of \(U\) to avoid double-counting with \(P\); v2 may set
a liability weight only via a separate pre-registered amendment.

**Sensitivity (pre-registered):** also report \(U\) under
\((\alpha,\beta,\gamma) \in \{(1,0.5,0.3), (1,1,0.3), (1,0.5,0.6)\}\).
Kill/survive decisions use the **default** row; sensitivity that flips the
decision → graded, no binary kill.

**\(\delta\) margin:** H-LM-necessary requires
\(U(\text{best non-LM}) < U(\text{best LM+verify}) - 0.05\) on default weights.
Within 0.05 → **tie** → demote LM-uniqueness claims; do not claim LM necessary.

## Construct policy for "correct" (attacks T3 simultaneously)

Primary science instrument remains **exact string match** (backward compatible with
immutable JSONs). For \(U\) and the kill decision, also compute:

1. **Exact** — current scorer.
2. **Normalize-then-match** — lowercase, strip punctuation, singularize trivial
   English plurals (frozen rule list committed before scoring).
3. **Human faithfulness** — stratified sample of 60 disagreements between (1) and
   (2), plus 40 random errors, rated by a single frozen rubric:
   `{faithful, unfaithful, unsure}`; majority of two passes if available, else one
   rater with rubric committed here. Human labels **do not** retune methods.

Kill decision uses **exact** \(U\) as primary and reports normalize + human as
robustness. If normalize/human flips the kill while exact does not → **construct
contested**; LM frame not cleared; E3-style follow-up mandatory before architecture
claims.

## Methods under test (minimum set — all required)

Held fixed across methods: same dialogues (m0–m4 or regeneration seeds committed
before run), same schema `CC|DUR|SEV|MED|ALG`, same greedy/deterministic decode
where applicable, same verifier pair (presence+absence) as `scribe` Stage G/A when
the verify-on arm is active.

| ID | Method | Class | Notes |
|---|---|---|---|
| M0 | Best LM reference | Generative | Frozen: Pythia-160M LoRA corner *or* own 3.2B+LoRA corner — pick the higher \(U\) on verify-on; name it in results JSON before comparing |
| M1 | Regex / template slot filler | Symbolic | Hand rules over dialogue turns; no learned weights |
| M2 | Dictionary + fuzzy span match | Symbolic | Closed + open lists from train pools only; held values must be copied from dialogue spans via string search |
| M3 | Linear-chain CRF or CRF-lite BIO tagger | Structured prediction | Train on v2 scribe alignments (source span → field); emit fields from tagged spans |
| M4 | Constrained decoder / finite-state copy | Constrained gen | Schema-constrained output; copy-only for open slots (no free vocab for ALG/MED/CC values) |
| M5 | Span extractor (start–end classifier) | Non-autoregressive | Per-field span head over dialogue tokens |

**Optional (quota only, not required for kill):** M6 pointer-generator (See et al.)
trained from scratch on v2 — historical lineage control.

**Forbidden post-hoc:** adding LLM-based methods to the "non-LM" set; tuning M1–M5
after seeing LM gaps; changing \(\alpha,\beta,\gamma\) after results.

## Causal identification table (required framing)

| Mechanism claim | Intervention | Observable | Competing prediction | Expected effect | Power | Falsifier |
|---|---|---|---|---|---|---|
| T5 wrong substrate | Replace generative proposer with M1–M5 | \(\Delta U\) vs M0 | T1/T2: LM still wins \(U\) after verify | \(\lvert\Delta U\rvert \ge 0.05\) | 5 instances × 200 fields; bootstrap CI on \(U\) | All non-LM \(U < U_{M0}-0.05\) |
| Verify is substrate-agnostic | Cross 2×2: method × {verify-off, verify-on} | \(\Delta U\), \(\Delta P\), \(\Delta\rho\) | Helps only LM | Interaction term | Same | Verify×method interaction ≈ 0 |
| Exact-match overstates LM failure | Score M0 under normalize+human | Gap shrink | Gap stable | ≥10 pt clean-gap shrink | 100 human labels | Gap shrink <5 pt and humans agree exact |
| Review economics dominate | Vary \(\gamma\) sensitivity | Rank(M0…M5) | Rank stable | Rank flip | Pre-registered weight grid | No rank flip under grid |

## Measurement instrument

- **Science metrics (unchanged):** diluted gap, clean gap, per-field, per-type flips,
  hallucination/omission/substitution rates — for M0 and any method that emits
  comparable fields.
- **Systems metrics:** presented precision, review load, provenance completeness
  (span citation required for every presented open-slot value; symbolic methods
  must cite byte offsets).
- **Utility:** \(U\) default + sensitivity grid.
- **Reporting:** per-method JSON; per-item outputs logged (council invariant).

## Decision rule (fixed now)

On default \(U\), mean over instances:

1. **KILL (H-substrate):**
   \(\max_{m\in\{M1..M5\}} U(m) \ge U(M0) - 0.05\)
   → demote generative-LM-centric product/architecture claims; Paper α may proceed
   as measurement; Paper β systems claims must be substrate-agnostic; fabric
   expansion **STOP**.
2. **SURVIVE (H-LM-necessary):**
   \(\max_{m\in\{M1..M5\}} U(m) < U(M0) - 0.05\)
   under default *and* no sensitivity flip → LM frame survives this gate; proceed
   to E3 then E2; fabric remains gated until E3.
3. **GRADED:** sensitivity flips rank, or only verify-off kills while verify-on
   survives (or vice versa) → report interaction; **no architecture punchline**;
   do not claim either polar verdict.

Secondary (does not override kill): if M1–M5 match closed-field accuracy but fail
open-slot copy similarly to weak LMs → phenomenon is **statistical-learning-general**,
not transformer-specific (council model-ecology question). Record as
`ecology: general | transformer-ish | inconclusive`.

## Negative predictions (boundaries)

If H-substrate is false (LM survives), the theory still predicts:

- M1/M2 **should not** beat M0 on open-vocab held types without span copy machinery.
- Raising dictionary coverage to include held types **must be forbidden** (train/test
  leak); if leak is accidentally introduced, run is VOID.
- Verification **should** add less value to M3/M5 if their errors are already span-
  grounded; if verify review load for M3 ≈ M0, LM error structure is not special.

## What this resolves / does not

**Resolves:** whether the program’s product premise (small generative LM + verify)
survives a utility-facing substrate comparison; whether verify is orthogonal;
partial construct pressure via normalize+human.

**Does not resolve:** LoRA mechanism (E2); open-world verifier soundness; clinical
normalization objective (H4) — if ontology coding is later declared the true goal,
amend utility and re-run decision under a new prereg, do not retrofit.

## Compute / effort estimate

- M1/M2: hours (rules).
- M3/M5: ≤1–2 T4-days including alignment export.
- M4: ≤1 T4-day if reusing constrained-decoding scaffolding; else defer with
  explicit VOID for M4 only (M1–M3–M5 still suffice for kill).
- Human labels: ~100 items, few hours.

Total: deliberately cheaper than another diversity continuum or fabric redesign.

## Exit criteria linkage

- KILL → program becomes (α) empirical LM-failure measurement + (β) substrate-
  agnostic verification theory; generative nano core optional.
- SURVIVE → E3 then E2 unlocked; fabric still not primary until E3 construct check.
- Either way → update `papers/EMPIRICAL_FOUNDATION.md` kill-gate section with the
  frozen JSON paths and verdict in an owner commit.

## Honest-reporting rule

Single primary measurement pass on frozen methods and weights. No method fishing.
Failures of M1–M5 to parse schema count as \(U\) disasters (high \(M\), low \(P\)),
not exclusions. VOID only for train/test leak or instrument bugs, declared before
unblinding \(U\) ranks.


## AMENDMENT 1 (2026-07-30) — dominance close for official M0

Official M0 targets (Pythia-160M LoRA; own-3.2B+LoRA corner) are not present as
loadable checkpoints on the execution host (MPS-only; no HF peft adapters).
Rather than leave the kill-gate open indefinitely, record the pre-registered
implication of the measured M1 utility:

\[
U(M1) = 0.998999 \ge 1 - \delta = 0.95
\]

Therefore \(\max_{m\in\{M1..M5\}} U(m) \ge U(M0) - \delta\) holds for **all**
\(U(M0) \le 1\). Measuring official M0 on this harness can refine the numeric
margin relative to local scale10m but **cannot reverse KILL** without amending
methods or weights (forbidden post-hoc). Optional future scoring of official M0
remains informative for systems comparisons, not for reopening the substrate gate.

## RESULT (2026-07-30)

| Method | U verify-on | P | recall | held | gap | C |
|---|---|---|---|---|---|---|
| M1_template | **0.999** | 1.000 | 1.000 | 1.000 | 0.0 | 0.02 |
| M2_dict_span | **0.886** | 0.925 | 0.925 | 0.862 | 12.6 | 0.03 |
| M0_scale (local) | 0.854 | 0.984 | 0.902 | 0.809 | 18.7 | 1.0 |
| M4_constrained | 0.819 | 0.881 | 0.881 | 0.762 | 23.8 | 0.04 |
| M5_span_clf | 0.689 | 0.940 | 0.682 | 0.657 | 5.0 | 0.2 |
| M3_crf_lite | 0.248 | 0.591 | 0.462 | 0.443 | 3.8 | 0.15 |

- **Verdict:** **KILL** (H-substrate). Sensitivity grid: no rank flip.
- **Verify interaction:** lifts U for M0 (+0.057) and M5 (+0.175) and M3 (+0.063);
  near-zero for M1/M2/M4 (already span-grounded / perfect). Partial support for
  H-verify-orthogonal being **false** — verify helps error-prone proposers more.
- **Normalize construct (E1 arm / E3 auto):** `correct_norm_rate == recall` for
  every method (0 rescues) — see `PREREG_E3_faithfulness_construct.md`.
- **Human faithfulness:** deferred to E3 pack (blocked on rater).
- **Official M0 (2026-07-31, RunPod CUDA fp16):** Pythia-160M LoRA
  \(U=0.925\); ownstack Chinchilla-160M+LoRA \(U=0.898\);
  \(\max U=0.925\). Vs M1/M2 under \(\delta=0.05\): **KILL** confirmed
  (decision in `results_e1_utility.json`).
- **Artifacts:** `results_e1_utility.json`, `results_e1_nonlm_*.json`,
  `results_e1_items_*.json`.
