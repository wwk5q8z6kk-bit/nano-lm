# Research status

**Updated:** 2026-08-26
**Companion to:** [`NANO_VNEXT_MASTER_SPEC.md`](NANO_VNEXT_MASTER_SPEC.md)

One page, seven claim buckets. A claim lives in exactly one. Moving a claim
upward requires the evidence named in its row, not an argument.

> **Reading rule.** ESTABLISHED means a pre-registered decision rule fired on
> data that survived its own guards. SUPPORTED means the evidence points one way
> but no rule fired. REFUTED means a *valid* experiment tested the claim and its
> prediction failed — the instrument worked, so this **is** information, unlike a
> VOID. HYPOTHESIS means nothing has been measured. VOID means the instrument
> failed and the result carries no information about the hypothesis — a VOID is
> **not** a negative result.
>
> A valid experiment that lands between its registered anchors is **NULL /
> INCONCLUSIVE**: it did not distinguish the alternatives, and the hypothesis
> stays in HYPOTHESES rather than moving. Where a null is itself the claim
> ("the arms do not separate"), it is recorded as an ESTABLISHED claim about
> non-separation — see the native30 rows.
>
> These are *claim* buckets. The *run* verdicts they compose from are in
> [`NANO_VNEXT_MASTER_SPEC.md`](NANO_VNEXT_MASTER_SPEC.md) §22.

---

## ESTABLISHED

| Claim | Evidence | Guard that held |
|---|---|---|
| **The span-port model returns the enclosing turn rather than the minimal span.** With all leak channels closed it selects the correct turn for **97/120** gold-bearing slots — **3.07× the 0.263 chance rate** — and returns a character-identical minimal span in **2**. All 95 non-exact located quotes are *over*-extended; **zero** under-extended. | `e04b3016`, replicated by `38b12909`; `artifacts/span_extent_L000_unified.json`; chance baseline recomputed independently over all 120 gold slots | Replication reproduced the extent census cell-for-cell and the across-instance vector to 4dp |

> **Wording qualified 2026-08-26, after a concurrent session audited its own
> claim.** This row previously read *"locates but does not delimit."* The
> **description survives** — the model does return enclosing turns, corroborated
> three independent ways (containment 81% vs 26% chance; median
> quote/enclosing-turn ratio 1.000; character-overlap profile ≈0.47 against the
> analytic whole-turn prediction 2|g|/(|g|+|q|) ≈ 0.43).
>
> **The word "cannot" does not survive.** `exact_gold_span` demands character
> identity with a gold span the harness authored as the *minimal value string*.
> **Nobody has established that minimal is the target.** For a scribe, quoting
> the sentence carrying the assertion is defensible. If it is acceptable, then on
> ~80% of slots the model is not failing at all and "13% of ceiling" measures
> conformance to an **unvalidated convention**, not a capability limit. See
> `artifacts/ADDENDUM-delimitation-construct-validity.md` on
> `work/edelimit-instrument`.
>
> **Wording policy — HEDGE_REQUIRED (adopted from the source retraction).** For
> the span-port line, **"bottleneck", "cannot delimit" and "delimitation failure"
> are forbidden** until the construct question resolves. *"Bottleneck" is the
> wrong word for anything currently in evidence.* Say what was measured: the
> model returns the enclosing turn rather than the minimal span.
>
> A third survivor, which sharpens the construct concern: **exact-span scoring is
> highly prompt-sensitive — 16/192 → 2/192 on a phrasing edit that left LOCATED
> unmoved.** The primary metric moved 8× on wording while the descriptive finding
> did not move at all.
>
> `span_character_f1` was registered CO-PRIMARY and never reported: mean 0.332,
> median 0.372. It is **not** the convention-robust alternative it looks like —
> it is computed post-binding, and of 38 zero-F1 slots, 21 are `asserted_unbound`
> and 15 demonstrably contain the gold span but failed to bind. Quoting 0.33 as a
> clean substitute for 2/192 would be a second convenient-metric error on top of
> the first.
| **The honest span-port baseline is 2/192, not the 83% coverage once claimed.** The earlier headline was measured with both prompt leak channels open. | `ddb5ce6`, run `e04b3016` | Mechanically verified in the run's own artifact: `gold_in_answer_template` 0/192, `gold_in_question` 0/192 |
| **On the safety property, the model asserts unsupported content about as often as it correctly declines.** `asserted_unbound` 43 vs `abstained_correct` 24. | run `e04b3016` | Same cell, same slot set |
| **C2 (parser gold-value fallback) is inert on the primary endpoint.** `asserted_grounded` identical across all four C2 pairs in all 12 instances (Δ 0.000, sd 0.000). | `ddb5ce6` | Four of eight grid cells are redundant; axis folded |
| **The delimit output-format refactor is behaviour-preserving.** Run one analyzer over both payloads and the extent census is **cell-for-cell identical** — 2 / 95 / 0 / 8 / 15 / 72, LOCATED 97/120 each. | `38b12909` vs `e04b3016`, both through `nanoscribe/analyze_span_extent.py` | R1 control; `question_template_hash` equal across arms, `output_format_hash` distinct |

> **Independently corroborated 2026-08-25.** A concurrent session
> (`work/edelimit-instrument`) built E-DELIMIT separately off the same base
> `9a3ecd4`, stopped before launching its own arms to avoid duplicate compute,
> and re-analysed runs `38b12909` and `4de84c18` with its own code. It reached
> the same verdict — **arm B VOID, H-delimit untested** — and added a finer pick
> census: unrelated 90 (75%), over-extended 27, under-extended 2, exact 1, with
> format compliance ~96%. Its arm-B LOCATED (30/120) matches this ledger exactly.
>
> **One discrepancy, settled.** That session reported arm A′ LOCATED as **98/120**
> against L000's published **97/120** and read it as "agreement to within one
> slot." It is not a run difference: it compared *its* normalization of A′ against
> the *published* (this ledger's) normalization of L000. Running a single
> analyzer over both payloads gives **97/120 for each, identical in every cell**.
> The two runs agree *exactly*; the one-slot gap is analyzer-vs-analyzer, and the
> replication is therefore stronger than either session stated alone.
| **The native30 arms do not separate at this scale.** `evidence_bottleneck` pooled 6/450 (0.0133, Wilson [0.0061, 0.0288]) against decoder control 0/450 (Wilson [0, 0.0085]); per-seed [6, 0, 0]; `seed_spread` 0.04 exceeds the effect. Verdict `NOT_SEPARATED`, `effect_exceeds_seed_spread: false`. | `artifacts/campaign/native30_revalidation_summary_causalfix.json` | Registered decision rule; the split is seed noise, not an architecture effect |
| **The native30 instrument was sound — the gated wave ran clean.** All 9 arms landed under `reval_results_fixed/`; every `*_train.json` carries an integrity block with attention leakage **exactly 0.0**, each block matching the `scaled_dot_product_attention(is_causal=True)` reference to 2.98e-07–5.07e-07, supervised 305, prompt cap 512 == `max_seq`. Nothing bypassed or relaxed. | `326b301`; `artifacts/RESULT-native30-gated-revalidation.md` | **Establishes the instrument was sound — not that anything was learned.** The two claims are kept separate by rule |
| **…and the result replicated exactly against the pre-gate causalfix wave.** Verdicts, effect sizes and pooled coverage all identical: `evidence_bottleneck` constrained NOT_SEPARATED +0.0133 cov 18; `span_port` constrained NOT_SEPARATED +0.0000 cov 18; both unconstrained INVALID_NO_SIGNAL cov 0. | `326b301` | **NOT_SEPARATED here is under-powered, not a null** — recorded as such at source |
| **Converged training loss does not detect leakage** (n=27). Gated `decoder_control_s1` finished at **0.00962** — lower than eight of the nine known-leaking archived runs — with attention leakage proven exactly 0.0. Medians: archived 0.028, causalfix 0.086, gated 0.061. | `326b301`; `artifacts/DEFECT_INDEX.md` D6 | The "loss ≈0.002" signature was a **40-step number retold as convergence**; only two of nine archived runs are near it and the range spans 0.008–0.536. Rule produced: calibrate a threshold against the archived failure, not the anecdote about it. **Consequence: converged training loss is RETIRED as a leakage diagnostic.** Leakage is established by the integrity block — attention-leakage measurement against the `is_causal=True` reference — and by nothing else |
| **The capability floor fired — in its permissible form only.** *30M at 1800 steps **with a tokenizer that cannot fit 83% of eval prompts in its context** is below the floor for `p1_screening_eval_v1`.* Measured: **124/150 = 82.7%** of prompts exceed `max_seq=512` under char-level `hash_tokens` (median 530, max 576); 100% of the training corpus does (median 615, max 694). | `artifacts/campaign/TOKENIZER_CONTEXT_CONFOUND.md`; CONFOUND NOTICE in `trajectory/PREREG_causalfix_wave_arm_split.md` | The prereg names the short form *"30M is below the capability floor"* **impermissible** — parameter count is not the only thing varying |

**Two qualifiers that travel with the native30 rows and must not be dropped:**

1. `span_metrics_are_tautological: True` for every constrained cell. Constrained
   mode selects from candidates, so span metrics cannot fail by construction —
   they qualify every constrained-cell reading.
2. Unconstrained cells are `INVALID_NO_SIGNAL`, **not** a null: pooled coverage 0
   with `malformed` 150/150 per run. No verdict is inferable from an instrument
   that produced nothing.

---

## SUPPORTED BUT NOT CONFIRMED

| Claim | Evidence | Why it is not ESTABLISHED |
|---|---|---|
| **Breadth before specialization.** At matched parameter count a generally-pretrained model reads 3.5 ± 0.7 against a domain-native model's 16.9 ± 1.7; 16× more tokens moves the native model to 7.0 ± 1.0, LoRA on top to 4.2 ± 0.9. | `papers/FINDING_BREADTH_BEFORE_SPECIALIZATION.md`; Stage-T ladder | Layer 2/3 synthesis of frozen evidence plus external studies — **not a new measurement** |
| **Data and method are substitutes.** 200M tokens + LoRA ≈ 3.2B tokens + full FT (7.1 ± 1.2 vs 7.0 ± 1.0). | Chinchilla control, P2 2×2 | Interaction account replaces the earlier 73/27 decomposition; mechanism unidentified (E2 GATED/STOP) |
| **A fitting tokenizer would move the floor.** `sft/tokenizer.json` (own-stack BPE, same vocab 4098) gives measured 2.60× compression on the exact eval prompts — median 204, max 226, **0/150** over context, ~60% headroom, embedding parameter count unchanged. | `artifacts/campaign/TOKENIZER_CONTEXT_CONFOUND.md` | The compression is measured; the capability effect is not. An existing repository asset, not new work |
| **Slots behave differently by value diversity.** Diversity effect 66.7 pts; position innocent. | slot-diversity sweep, H-slot SUPPORTED | Pre-registered and supported; not re-measured under the current instrument |

---

## REFUTED

A valid experiment tested the claim and its prediction failed. Unlike a VOID,
the instrument worked — these rows carry information and must not be reopened
without new evidence.

| Claim, as it was made | Refuting evidence | What survives |
|---|---|---|
| **"Qwen already exhibits the retrieval competence the nano trunk was measured to lack."** ANALYSIS §3, this program's own argument. | Cross-regime selection probe, run `d222465e`, commit `3205a64`. Stage P's own statistic (teacher-forced top-1 at value tokens), task and statistic held fixed, model axis crossed. **HELD first-token 11/28 = 39.3%**, 95% Wilson **[23.6%, 57.6%]** — the interval **excludes the 92% anchor** at which §3 would have stood. Held-vs-seen gap **+41.0 pts** against a SEEN control of 102/127 = 80.3%, two-proportion z = 4.42, p ≈ 9.8e-06. | Qwen is better than the nano trunk (the interval also excludes 21%) and nowhere near solved. **Content-addressed selection does not generalise OOD for Qwen either.** |
| **"A model with the circuit still grounds at 13%, therefore inducing the circuit would not deliver P1."** | Withdrawn with the premise above — the antecedent is false. The downstream conclusion that *"Stage M should not be described as unblocking P1"* is withdrawn with it. | **Stage M may well be on the P1 critical path.** This is now open, not closed. |

> **The dichotomy itself is NULL / INCONCLUSIVE, and that is a separate fact.**
> The probe registered two anchors — *held near 92%* (selection is a small-trunk
> property) and *held near 21%* (selection is task-intrinsic). 39.3% is near
> neither; the interval excludes both. The registered third branch (*report
> as-is beside the control*) is what fired. So the specific claim is refuted
> while the **small-trunk-vs-task-intrinsic question remains untested** — it sits
> in HYPOTHESES below, not here.
>
> **Newly suspect, recorded not resolved.** Treating *"79% located"* and *"39.3%
> selection"* as two measurements of one quantity. The first is a generous
> substring criterion on free-running output; the second is exact next-token
> top-1 under teacher forcing on a different task. Honest reading: **both**
> regimes carry a selection deficit, and delimitation is an additional layer on
> top rather than an alternative to it.
>
> **Interpretation boundary.** This establishes that the selection deficit is not
> confined to the small trunk. It does **not** establish where the deficit comes
> from, and it does **not** touch the delimitation measurement, which is a direct
> observation about span extent given a quote was emitted.

---

## HYPOTHESES

Nothing measured. Each needs instrument → bottleneck → manipulation → invariance
requirement → decision rule before it is an experiment.

- **Is the selection deficit a small-trunk property or task-intrinsic?**
  NULL / INCONCLUSIVE on the cross-regime probe above — 39.3% sits between the
  two registered anchors and the interval excludes both. Untested, not weakened.

- **H5 — delimitation.** The model addresses content at unit granularity and
  cannot resolve sub-unit boundaries. **Untested** (see VOID RESULTS).
- **DMLA** and every mechanism in it — multiscale representation, persistent
  latent state, iterative refinement, retrieval, structured memory, adaptive
  computation, sparse specialists, energy/constraint dynamics, early exits,
  modality experts, retrieval-conditioned computation. *An architecture
  hypothesis, not an architecture commitment.* See `NEURAL_CANDIDATES` in
  `nano/architecture.py` — presence in that tuple implies nothing measured.
- **Retrieval-capacity and induction-circuit accounts** of the scribe OOD gap,
  surviving after curriculum (Stage C), scale (Stage S) and architecture
  (Stage P) all failed to move it.
- **Monolith versus modular Nano-System.** Whether Nano-Core / Reason / Memory /
  Verify / Route / Vision / Audio converge into one model is open in **both**
  directions.

---

## VOID RESULTS

A VOID carries no information about its hypothesis. It must never be cited as a
negative result.

| Arm | Why VOID | What is *not* concluded |
|---|---|---|
| **E-DELIMIT arm B (menu)** — run `4de84c18` | **Invariance precondition failed.** The arm's premise was to leave retrieval exactly as hard; LOCATED collapsed **97/120 → 30/120** — **0.95× the 0.263 chance rate, i.e. indistinguishable from guessing**, and below even the constant index-0 parrot's 23/120. The menu did not make locating *harder*; it reduced span selection to **chance**. Model picks are front-biased: median index 13.5 against median menu size 64, median relative position 0.211 (uniform ≈ 0.5), indices 1–2 alone = 33% of picks. The arm measured long-list indexing, not boundary selection. | `asserted_grounded` 0/192 would fire the pre-registered kill condition. **H5 is not refuted.** The span-port line does **not** rejoin the retrieval hypotheses on this evidence. |
| **E-DELIMIT arm C (offsets)** — run `aa779aba` | LOCATED 2/120. Secondary by pre-registration (P6): requires index arithmetic over transcript offsets, a known weakness independent of delimitation. | Not evidence about H5. Reported for completeness only. |
| **native30 revalidation wave 1** — `artifacts/campaign/native30_revalidation_summary_v1.json` | **False null.** The decoder was non-causal — `nn.MultiheadAttention` with no mask and no `is_causal` in a decoder trained on next-token prediction, so every position could attend to its own label. Measured leakage before the fix: changing tokens at positions 6–7 moved logits at 0–5 by up to **20.1**; after the fix, exactly **0.0**. Compounded by a target truncated out of the loss. | All six arm×mode cells read `NOT_SEPARATED`, effect +0.0000, pooled 0/450. **Do not bank these verdicts. Do not retire `span_port` or `evidence_bottleneck` on them.** |
| **C3 arm (leakage grid)** | The C3-off prompt removed the gold surface *and* changed the question from yes/no to wh-extraction. The measured effect was dominated by form. | Superseded by the unified wh form (`b707478`). The R1 rule exists because of this arm. |
| **Stage P / P1 (unsupervised pointer head)** | Manipulation check failed — the copy pathway never engaged (M = 0.18, p_gen = 0.83). | The check prevented a false REFUTE. |

---

## PENDING REVALIDATION

| Claim | Blocked on |
|---|---|
| ~~**"The native30 wave ran clean under the anti-leakage integrity gate."**~~ | **CLEARED 2026-08-26** — the gated wave landed. Promoted to ESTABLISHED above. |
| **D3 (context/truncation) closure** | D3.1 fixed the 64-char cap; the char-level-vs-512-context mismatch (D3.3) was never addressed. D3 remains only **partly** closed. |
| **1. Construct validity — is a non-minimal span an error at all?** | `exact_gold_span` encodes a minimal-span convention **no clinician has ratified**. If enclosing-turn evidence is acceptable, ~80% of slots are not failures. Needs human/clinician judgement; `E3 = UNRESOLVED`. A concurrent session argues this **gates** E-DELIMIT round 2 rather than following it — surfaced as an owner call below. |
| **2. Causal identification — H-delimit vs H-retrieve** | **Untested.** The one arm built to discriminate them ran **at chance** and is VOID. **Independent of (1):** even if the minimal-span target were ratified tomorrow, this stays open. Conversely, settling this would not settle (1). Earlier framing here blurred the two; they are separate. |

---

## NOT AUTHORIZED

| Item | Status |
|---|---|
| **The repaired two-stage span-port experiment** (retrieval → conditional delimitation) | Designed, not pre-registered, not authorized. Needs its own prereg **and** experiment-scoped authorization. **Do not run.** |
| Any further experiment | Explicit authorization required per launch. |
| E2 GPU · E4 execution reopen · fabric/v2 · NanoScribe build · Stage M / OLD_TASK_U | `EVIDENCE_CURRENT.md` non-authorizations, unchanged |
| PHI / private owner material on cloud | `NOT_AUTHORIZED`, standing |
| Clinical claims | `FORBIDDEN_WITHOUT_EXTERNAL_HUMAN_VALIDATION`, standing |

### Recorded authorization gap — not silently corrected

The eight-cell leakage ablation (`ddb5ce6`) was pre-registered, carried
CONFIRMED/REFUTED decision rules, and produced the program's headline span-port
result. By the clause reading settled 2026-08-25 it was a **confirmatory
evidential run**, so the zero-cost clause did not cover it and it needed
experiment-scoped authorization **it does not have on the record**.

The ablation's *numbers* are not in question; the gap is procedural. Retroactive
ratification remains available to the owner and would close it in one sentence.
Recorded because `ddb5ce6` is load-bearing: the honest baseline, the C2-inert
finding, the C1 UNRESOLVED verdict and the C3 void all rest on it. See
`research/decision_records/2026-08-25-authorization-clause-and-revalidation-claim.md`.

---

## NEXT CANDIDATE EXPERIMENTS

Ranked. None authorized. Each must supply the **twelve-field chain**
(`NANO_VNEXT_MASTER_SPEC.md` §20) and pass the **readiness gate** (§25) before it
is proposed for launch. The gate's fourth question — *what competing explanations
does the experiment distinguish?* — is what produced this ranking.

1. **Tokenizer swap on the native30 instrument.** Highest information per dollar
   and uses an existing asset. Swap `hash_tokens` → `sft/tokenizer.json`; hold
   parameters, steps, corpus and eval fixed. *Instrument:* `p1_screening_eval_v1`.
   *Bottleneck:* 82.7% of eval prompts truncated. *Invariance requirement:*
   embedding parameter count unchanged (it is — same vocab 4098). *Discriminates:*
   capability floor vs context-fit confound — the single largest ambiguity in the
   current record.
2. **Two-stage span-port: retrieval → conditional delimitation.**
   **NOT READY — fails the readiness gate** (SPEC §25). Under HEDGE_REQUIRED
   there is **no measured failure mode** on this line to aim at, so gate
   question 3 is unanswerable and an unanswered question is a stop. Blocked
   behind unresolved (1), the construct question — which is a human/clinician
   judgement, not compute. Kept here for its design, which is sound.
   *Instrument:* `campaign_v2` span-port. *Hypothesised mechanism (H-delimit,
   not a measured bottleneck):* conditional span selection, given the turn
   selection the model already performs at 97/120. *Manipulation:* stage 1 is
   free-form and unchanged — the model quotes a turn. Stage 2 enumerates
   sub-spans **of the turn the model itself chose**, never the gold turn, so no
   location is handed over. Formulation adopted from `work/edelimit-instrument`,
   which sharpened it: because the model has already done the retrieval, this
   holds retrieval fixed **by construction** rather than leaving it to be checked
   afterwards — strictly better than the version first drafted here.
   *Invariance requirement:* **R8**, below, still applies as a backstop.
3. ~~**Gated native30 re-run.**~~ **DONE 2026-08-26** — `326b301`. Gate passed
   *and* result replicated, the two claims kept separate.

### Owner call — does the construct question gate candidate 2?

Not a candidate; a **reordering** that a concurrent session surfaced rather than
acted on, because it changes the success ladder.

The argument: `exact_gold_span` may measure conformance to an unvalidated
convention (minimal span) rather than faithfulness. If quoting the enclosing
utterance is acceptable for a scribe, then the model is not failing on ~80% of
slots, and a sharper delimitation experiment would **spend compute explaining a
convention**. On that reading, E3's faithfulness construct and the P1 span metric
are the same question asked twice, and the construct question should be settled
**before** candidate 2 runs, not after.

Against: settling the construct requires human/clinician judgement about what a
scribe should quote, which is `E3 = UNRESOLVED` and gated on external validation.
Waiting could stall the span-port line indefinitely.

**This ranking is the owner's to make.** Recorded here so it is not resolved by
whoever happens to launch first.

**Do not** answer a delimitation question with a generic memory/reasoning
architecture experiment. Instrument and bottleneck must match.

### R8 — retrieval preservation (registered from the arm-B VOID)

`docs/RUNBOOK_contrast_hygiene.md` R8, registered on `work/edelimit-instrument`
after this VOID and adopted here:

> When an arm manipulates the output format, pre-register a
> **retrieval-preservation** invariant on the capacity the manipulation is
> supposed to leave alone, and make it a **VOID condition**. For the span-port
> line: arm B is VOID if its **LOCATED falls more than 10 points below the
> free-form control's**.

The instructive part is *why the existing guard missed it*. A format-feasibility
gate (`well_formed_rate ≥ 0.80`) would have **passed at ~96%** — the model used
the menu format perfectly well. **Format compliance and task preservation are
different properties**, and only the first was being checked. This is the
concrete instance of the "invariance requirement" field that
`NANO_VNEXT_MASTER_SPEC.md` §25 now requires of every experiment.
