# Review — the proposed DMLA architecture program, against the registered record

**Date:** 2026-08-25
**Subject:** a proposed 8-step architecture research program (A0→A7: multiscale
representation → persistent latent state → iterative inference → memory →
adaptive halting → sparse specialists → energy dynamics), offered as the response
to the native30 capability-floor failure.
**Reviewed against:** `frontier/accelerated-research-campaign-v2` @ `b5c6b2c`
(native line) and `work/leakage-power-analysis` @ `23ede77` (span-port line).

---

## 0. Verdict

**Adopt the discipline. Reject the sequencing. Do not start A1.**

The plan's premises are accurate — every artifact it names is real. The failure
it responds to is real and correctly characterised. But it proposes an eight-rung
architecture ladder whose first three rungs collide with the registered state of
the program, and whose mechanisms (§4 latent state, §5 memory, §6 iterative
inference) are aimed at the bottleneck of the instrument that is *not* the
current capability gate.

There is a $0, ~20-minute experiment already pre-registered that discriminates
whether those mechanisms are aimed at the right target. It should run first.

---

## 1. What the plan gets right — adopt these

- **§16 "don't invent fake physics."** The stated bar — *what mathematical
  property are we importing, and why should it solve a known failure mode?* — is
  the correct one, and it is stricter than most of what gets written under the
  "physics-inspired" label. Keep it verbatim.
- **§10 incremental progression with per-rung ablation.** This is already how
  the program works (Stage C → S → P → P2 each isolated one mechanism and each
  produced a mechanical verdict). The plan restates an existing strength.
- **§2 "the architecture may change, the measuring stick does not."** This is
  already enforced, more sharply than the plan states it: contrast-hygiene rule
  R2 says all arms must carry the same prompt-template hash and *if one arm is
  re-run, all are stale*. The plan's version is weaker than the repo's.
- **§13 Gate B (multi-seed reproducibility).** Vindicated one hour before the
  plan was written: the native30 arm split looked like a clean architecture
  effect at 0/6 vs 6/6 on seed 0, and dissolved to seed noise on seeds 1–2
  <file path="trajectory/PREREG_causalfix_wave_arm_split.md" exp="frontier/accelerated-research-campaign-v2" />.

## 2. Three collisions with the registered record

### C1 — There is no A0 to freeze (plan step 4)

The plan says "tag the current architecture as the baseline." The current native
baseline is void *as an architecture comparison*, and was already void before the
capability floor fired. Defect **D1.2**: the three native30 arm objectives were
scalar multiples of `lm`, so the arms shared one gradient direction and differed
only in effective learning rate — the arm comparison measured LR, not objective.
Defect **D2.3**: the analyzer's verdict fallthrough turned total output collapse
into `NOT_SEPARATED`, fabricating six nulls that were reported as findings
<file path="artifacts/DEFECT_INDEX.md" exp="frontier/accelerated-research-campaign-v2" />.

Both are fixed (`c98e4ad`, and the D2.3 guard is now doing real work). But the
post-fix wave's own verdict is **SEED NOISE**, explicitly *not used to rank
architectures*. A0 is not a baseline; it is a floor reading.

### C2 — Plan step 6 (tokenizer swap first) is scoped out by standing instruction

`CLAUDE-PROGRESS.md` on the native branch: *"Out of scope by instruction:
minbpe/BPE swap (breaks comparability mid-revalidation; own preregistered
change)."*

This is a sequencing collision, not a substantive disagreement. The plan is
right that a 4098-token character-level hash tokenizer is a poor representation
for span-port scribing. The objection is that doing it *first* breaks the
revalidation it is sitting inside, and that it is a large enough change to need
its own pre-registration rather than to ride in as "step 6 of a ladder."

### C3 — Plan §11 contradicts plan §13

§11 says drop to ~5–10M parameters for cheap mechanism tests. §13 Gate A
requires clearing the capability floor. The 30M model **already fired the
floor** — pooled constrained coverage 54/1350 = 4.0% against a 10% threshold,
unconstrained coverage 0/150 in 9/9 runs, every model abstaining on 144 of 150
atoms <file path="trajectory/PREREG_causalfix_wave_arm_split.md" exp="frontier/accelerated-research-campaign-v2" />.

Shrinking to 5–10M makes discrimination strictly worse. At 4% coverage the
pooled n per arm is ~18, and the registered power note states plainly that the
design *detects a large effect and nothing else*. The first question is not
"which architecture" — it is "what config clears the floor at all," which is a
scale/steps/representation question. And the single lever most likely to move it
(representation) is the one C2 scopes out mid-revalidation.

## 3. The deeper problem — the plan treats two instruments as one

The program's own newest analysis says this explicitly, and the plan inherits the
error it corrects:

> The four standing hypotheses are correctly scoped to the scribe line, where
> P2's first-token result confirms a retrieval deficit. The span-port line has a
> different dominant bottleneck — delimitation — and the program has been
> carrying results between the two lines as though they measured one thing.
>
> — <file path="research/hypotheses/BOTTLENECK_2026-08-25_delimitation.md" exp="work/leakage-power-analysis" />

| | **scribe line** (Paper α) | **span-port line** (P1) |
|---|---|---|
| model | nano 3.15M … own-stack 160M | Qwen2.5-1.5B-Instruct |
| measured bottleneck | **retrieval** — teacher-forced held-value first-token top-1 21% held vs 92% seen | **delimitation** — LOCATED 97/120, exact extent 2/120 |
| can an extent error even occur? | No — the field template delimits the answer | Yes — the model chooses the quote's extent |

The plan's core mechanisms — persistent latent state (§4), associative memory
(§5), iterative refinement (§6) — are all **retrieval-flavoured**. They target
the scribe line's bottleneck. But `docs/ACTIVE_NOW.md` puts the capability
frontier on the *span-port* instrument (`capability_frontier = P1_SCRIBING`,
`current_gate = P1_ENCOUNTER_REPRESENTATION_AND_EVIDENCE_TRANSPORT`), and on that
instrument the model already finds the right conversational turn 81% of the time
and then fails to delimit within it: median quote 29 chars against gold median 8,
median quote/enclosing-turn ratio **1.000**, and a pure phrasing edit moved exact
extent 8× (16 → 2) while moving LOCATED not at all (95 → 97).

I re-measured the canonical leakage-free cell independently and it corroborates:
`asserted_grounded` 0.167 ± 0.389 per instance across 12 instances = 2/192, with
`observed_coverage` 0.818 <run id="e04b3016-c00b-4c0d-b328-486c07e9177e" label="L000 unified, grounded 2/192" />.

Building a memory system to fix a model that already located the evidence is
solving the wrong half of the problem.

## 4. Recommended sequence — replacing plan §23

1. **Recommend E-DELIMIT first — not launched here; see §6 for why.** Three arms
   (free-form quote / enumerated sub-span menu / character offsets), same 12
   instances × 16 slots, same model, same leak channels closed. Arm B removes
   *generation* of the boundary and leaves only *selection*, holding retrieval
   exactly as hard. Pre-registered with a kill condition: if arm B's
   `asserted_grounded` is < 25% of LOCATED, H-delimit is REFUTED and the
   span-port line rejoins the retrieval hypotheses. ~6.5 min/arm, local MPS,
   **\$0** <file path="research/hypotheses/BOTTLENECK_2026-08-25_delimitation.md" exp="work/leakage-power-analysis" />.

   This is the cheapest available test of the plan's central premise. If H5
   confirms, most of A2–A7 is aimed at the wrong instrument. If H5 is refuted,
   the plan's retrieval mechanisms become well-motivated *and* inherit an
   evidence base. Either way it costs 20 minutes and settles a direction that
   would otherwise be settled by an eight-rung ladder.

2. **Then decide the floor question on the native line, separately.** Not "which
   architecture" but "what minimum config produces non-degenerate output at all."
   Coverage is the endpoint, not arm separation. This is where the tokenizer
   argument belongs — as its own pre-registered change with its own comparability
   argument, per C2, not as rung A1 of a ladder.

3. **Only then consider an architecture ladder**, and scope each rung to a named
   instrument and a named measured bottleneck. A rung that cannot say which of
   the two lines it targets should not be built.

## 5. On plan steps 2–3, stated precisely

The plan's "apply the parked evaluation suppression" and "promote the completed
instrumentation changes" are vaguer than the record. Precisely:

- The **evaluation suppression is already applied and already fired.** The D2.3
  analyzer guard now distinguishes `NOT_SEPARATED` (real coverage, legitimate
  null) from `INVALID_NO_SIGNAL` (no output, no verdict inferable); under the
  pre-fix analyzer all six cells would have read `NOT_SEPARATED`. Separately,
  `exact_gold_span` in constrained mode is banned as a decision input and is
  being removed from output rather than footnoted (`span_metrics_are_tautological`).
- The **instrumentation is landed**: `nanoscribe/native/integrity.py` with
  `run_startup_gate()` called in `train_native()` before the first optimizer step
  on every run, a reference-oracle differential test against
  `F.scaled_dot_product_attention(is_causal=True)`, and a BPB floor; 12 tests
  green, three of which reconstruct a defect and prove the gate fires
  <file path="nanoscribe/native/integrity.py" exp="frontier/accelerated-research-campaign-v2" />.

Neither is a blocker. Both are done.

## 6. Two open items — both since decided by the owner

> **RESOLVED 2026-08-25.** Both items below were settled against the permissive
> reading. See `research/decision_records/2026-08-25-authorization-clause-and-revalidation-claim.md`.
> 6a: the confirmatory clause governs — `docs/ACTIVE_NOW.md` line 47 forecloses
> the `$0`-settles-it argument in its own text. Experiment-scoped authorization
> for E-DELIMIT was subsequently granted and is recorded in
> `research/preregistrations/PREREG_E_DELIMIT.md`. 6b: keep the floor result and
> the seed-noise null as established; mark the revalidation claim PENDING the
> gated re-run. The original text is kept below unedited.

### 6a — E-DELIMIT sits between two authorization clauses

`docs/ACTIVE_NOW.md` carries both `local_zero_cost_exploratory_training =
ALLOWED` and `confirmatory_evidential_run = PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION`.
E-DELIMIT is simultaneously \$0-on-local-MPS *and* pre-registered-confirmatory
with a kill condition, so it satisfies the first clause and triggers the second.
Which governs is a genuine ambiguity in the posture, not something to resolve by
picking the reading that permits action.

It is therefore **recommended, not launched**. This is the one question in this
review worth answering before anything else moves: does a \$0 local run of a
pre-registered confirmatory contrast need experiment-scoped authorization, or
does the zero-cost clause cover it?

### 6b — which wave the revalidation claim may be made from

The wave behind the capability-floor result wrote to
`artifacts/campaign/reval_results_causalfix/`. That is the pre-assertion wave
(started 01:57) which the progress note describes as having *"the code fixes but
not the runtime gate, so it cannot prove it ran clean and does not satisfy the
revalidation claim"* — with the kill-or-keep decision left explicitly to the
owner. The RESULT was subsequently appended from that wave's output.

**This does not threaten either conclusion**, and the reason is worth stating:
the integrity gate guards against leakage, and leakage inflates results. Both
conclusions here are negative — a floor reading and a null. A missing
anti-leakage gate cannot manufacture 4% coverage or a seed-noise verdict. The
conclusions are robust in the direction that matters.

What is left open is narrower: whether the revalidation *claim* — "the native30
wave ran clean under the gate" — can be made from this wave, or whether it needs
the gated re-run to `reval30_*_fixed_*` dirs that the progress note specifies.
That is an owner call about what the record is allowed to assert, not a question
about what the numbers mean.

---

## Appendix — measured values cited

| quantity | value | source |
|---|---|---|
| native30 pooled constrained coverage | 54/1350 = 4.0% (floor 10%) | `b5c6b2c` |
| native30 unconstrained coverage | 0/150 in 9/9 runs (rule ≥ 8/9) | `b5c6b2c` |
| native30 abstention | 144 of 150 atoms, every run | `b5c6b2c` |
| arm contrast (covered atoms) | control 0/18, span_port 0/18, bottleneck 6/18 (6,0,0) | `b5c6b2c` |
| arm verdict | SEED NOISE — Wilson overlap 0.1628 vs 0.1759; higher in 1/3 seeds | `b5c6b2c` |
| span-port L000 grounded (pre-unification) | 16/192, vs parrot floor 0, perfect-reader ceiling 120 | `ddb5ce6` |
| span-port L000 grounded (unified form) | 2/192 (0.167 ± 0.389 across 12 instances) | run `e04b3016` |
| span-port LOCATED | 97/120 (81%) | `f6d898a` |
| span-port median quote / gold | 29 chars / 8 chars; quote-to-turn ratio 1.000 | `f6d898a` |
| unbound vs correct abstention (L000) | asserted_unbound 39 vs abstained_correct 32 | `ddb5ce6` |
| scribe-line first-token top-1 | 21% held vs 92% seen | Stage P2 |
