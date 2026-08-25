# Research status

**Updated:** 2026-08-25
**Companion to:** [`NANO_VNEXT_MASTER_SPEC.md`](NANO_VNEXT_MASTER_SPEC.md)

One page, seven buckets. A claim lives in exactly one. Moving a claim upward
requires the evidence named in its row, not an argument.

> **Reading rule.** ESTABLISHED means a pre-registered decision rule fired on
> data that survived its own guards. SUPPORTED means the evidence points one way
> but no rule fired. HYPOTHESIS means nothing has been measured. VOID means the
> instrument failed and the result carries no information about the hypothesis —
> a VOID is **not** a negative result.

---

## ESTABLISHED

| Claim | Evidence | Guard that held |
|---|---|---|
| **The span-port model locates but does not delimit.** With all leak channels closed it selects the correct turn for **97/120** gold-bearing slots and delimits the gold span in **2**. All 95 non-exact located quotes are *over*-extended; **zero** under-extended. | `e04b3016`, replicated by `38b12909`; `artifacts/span_extent_L000_unified.json` | Replication reproduced the extent census cell-for-cell and the across-instance vector to 4dp |
| **The honest span-port baseline is 2/192, not the 83% coverage once claimed.** The earlier headline was measured with both prompt leak channels open. | `ddb5ce6`, run `e04b3016` | Mechanically verified in the run's own artifact: `gold_in_answer_template` 0/192, `gold_in_question` 0/192 |
| **On the safety property, the model asserts unsupported content about as often as it correctly declines.** `asserted_unbound` 43 vs `abstained_correct` 24. | run `e04b3016` | Same cell, same slot set |
| **C2 (parser gold-value fallback) is inert on the primary endpoint.** `asserted_grounded` identical across all four C2 pairs in all 12 instances (Δ 0.000, sd 0.000). | `ddb5ce6` | Four of eight grid cells are redundant; axis folded |
| **The delimit output-format refactor is behaviour-preserving.** | `38b12909` vs `e04b3016` | R1 control; `question_template_hash` equal across arms, `output_format_hash` distinct |
| **The native30 arms do not separate at this scale.** `evidence_bottleneck` pooled 6/450 (0.0133, Wilson [0.0061, 0.0288]) against decoder control 0/450 (Wilson [0, 0.0085]); per-seed [6, 0, 0]; `seed_spread` 0.04 exceeds the effect. Verdict `NOT_SEPARATED`, `effect_exceeds_seed_spread: false`. | `artifacts/campaign/native30_revalidation_summary_causalfix.json` | Registered decision rule; the split is seed noise, not an architecture effect |
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

## HYPOTHESES

Nothing measured. Each needs instrument → bottleneck → manipulation → invariance
requirement → decision rule before it is an experiment.

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
| **E-DELIMIT arm B (menu)** — run `4de84c18` | **Invariance precondition failed.** The arm's premise was to leave retrieval exactly as hard; LOCATED collapsed **97/120 → 30/120**, landing 7 slots above the constant index-0 parrot's 23/120. Model picks are front-biased: median index 13.5 against median menu size 64, median relative position 0.211 (uniform ≈ 0.5), indices 1–2 alone = 33% of picks. The arm measured long-list indexing, not boundary selection. | `asserted_grounded` 0/192 would fire the pre-registered kill condition. **H5 is not refuted.** The span-port line does **not** rejoin the retrieval hypotheses on this evidence. |
| **E-DELIMIT arm C (offsets)** — run `aa779aba` | LOCATED 2/120. Secondary by pre-registration (P6): requires index arithmetic over transcript offsets, a known weakness independent of delimitation. | Not evidence about H5. Reported for completeness only. |
| **native30 revalidation wave 1** — `native30_revalidation_summary_v1.json` | **False null.** The decoder was non-causal — `nn.MultiheadAttention` with no mask and no `is_causal` in a decoder trained on next-token prediction, so every position could attend to its own label. Measured leakage before the fix: changing tokens at positions 6–7 moved logits at 0–5 by up to **20.1**; after the fix, exactly **0.0**. Compounded by a target truncated out of the loss. | All six arm×mode cells read `NOT_SEPARATED`, effect +0.0000, pooled 0/450. **Do not bank these verdicts. Do not retire `span_port` or `evidence_bottleneck` on them.** |
| **C3 arm (leakage grid)** | The C3-off prompt removed the gold surface *and* changed the question from yes/no to wh-extraction. The measured effect was dominated by form. | Superseded by the unified wh form (`b707478`). The R1 rule exists because of this arm. |
| **Stage P / P1 (unsupervised pointer head)** | Manipulation check failed — the copy pathway never engaged (M = 0.18, p_gen = 0.83). | The check prevented a false REFUTE. |

---

## PENDING REVALIDATION

| Claim | Blocked on |
|---|---|
| **"The native30 wave ran clean under the anti-leakage integrity gate."** | The `_causalfix` wave carries the code fixes but **not** the runtime gate, so it cannot prove it ran clean. Pending the gated re-run to `reval30_*_fixed_*`, which **does not exist**. The floor result and the seed-noise null survive the missing gate — the gate guards against leakage and leakage *inflates*, so it cannot manufacture 4% coverage or a null — but *"ran clean under the gate"* is a statement **about the gate** and cannot be made from a wave that did not have one. |
| **D3 (context/truncation) closure** | D3.1 fixed the 64-char cap; the char-level-vs-512-context mismatch was never addressed. D3 is only **partly** closed. |

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

Ranked. None authorized. Each must state instrument, bottleneck, hypothesis,
manipulation, **invariance requirement**, decision rule, authorization, cost, and
artifact before it is proposed for launch.

1. **Tokenizer swap on the native30 instrument.** Highest information per dollar
   and uses an existing asset. Swap `hash_tokens` → `sft/tokenizer.json`; hold
   parameters, steps, corpus and eval fixed. *Instrument:* `p1_screening_eval_v1`.
   *Bottleneck:* 82.7% of eval prompts truncated. *Invariance requirement:*
   embedding parameter count unchanged (it is — same vocab 4098). *Discriminates:*
   capability floor vs context-fit confound — the single largest ambiguity in the
   current record.
2. **Two-stage span-port: retrieval → conditional delimitation.** *Instrument:*
   `campaign_v2` span-port. *Bottleneck:* delimitation, conditional on the
   retrieval the model already performs at 97/120. *Manipulation:* ask for the
   turn, then enumerate only that turn's sub-spans. Concedes that retrieval and
   delimitation cannot both be held fixed in one prompt, and measures the
   conditional quantity H5 is actually about. **Requires a blocking
   LOCATED-invariance check** — the guard whose absence voided arm B.
3. **Gated native30 re-run to `reval30_*_fixed_*`.** Closes the PENDING
   REVALIDATION row. Mechanical, no new hypothesis.

**Do not** answer a delimitation question with a generic memory/reasoning
architecture experiment. Instrument and bottleneck must match.
