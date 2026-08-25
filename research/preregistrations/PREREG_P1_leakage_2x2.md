# PREREG — P1 span-port gold-value leakage, C1 × C2 factorial

**Status:** registered, NOT YET RUN. All four cells idle/provisional.
**Date:** 2026-08-25
**Instrument:** `nanoscribe/run_eval.py --suite campaign_v2`, Qwen2.5-1.5B-Instruct, greedy (`do_sample=False`), local MPS.
**Scope:** zero-cost local evaluation under `local_zero_cost_exploratory_training = ALLOWED` (`docs/ACTIVE_NOW.md`). No PHI, all encounters synthetic. Touches no frozen evidence and no tags.
**Supersedes:** the pre-run decision rule previously held only on the root node. That rule is restated here and attached to all four cells, because a rule that lives on one cell of a factorial cannot bind the other three.

---

## 1. Round question

Which channel carries the P1 span-port score — the model reading the transcript, or the gold value handed to it?

## 2. Design

2×2 over the two *leakage* channels, with the task-specification channel Q pinned ON in every cell a verdict rests on.

|            | C2 on                  | C2 off                  |
|------------|------------------------|-------------------------|
| **C1 on**  | root `C1on_C2on_Qon`   | `C1on_C2off_Qon`        |
| **C1 off** | `C1off_C2on_Qon`       | `C1off_C2off_Qon`       |

- **C1** = `PROMPT_ANSWER_TEMPLATE_GOLD_VALUE` — the answer template and the system-prompt format examples carry the gold value.
- **C2** = `PARSER_RAW_VALUE_FALLBACK` — the scorer substitutes `raw_value` as the model's quote when the model emitted a label with no quote.
- **Q** = `PROMPT_QUESTION_NAMES_CONCEPT` — task specification, deliberately not ablated: measured, 16 slots collapse to 14 distinct prompts with Q off, colliding in enc-1 and enc-4.

**C2 is a scorer flag, not a prompt flag.** It changes no token the model emits. The 2×2 therefore requires only **two** generation passes; the C2 axis is pure post-processing of each pass.

## 3. Estimand — a paired within-item contrast

Decoding is greedy and all four cells score the **same** slots, so re-running a cell reproduces it byte-for-byte. Replication cannot come from re-runs; it comes from more slots. The design is fully within-item, so per slot $i$:

$$c_i = \bigl(Y_i^{11} - Y_i^{01}\bigr) - \bigl(Y_i^{10} - Y_i^{00}\bigr)$$

Under the mechanism in §4, C1-on saturates both C2 cells and $c_i$ collapses to a McNemar discordant-pair indicator. **The reported SD must be the SD of the paired contrast, not the across-instance SD of aggregate `exact_gold_span`** — between-slot difficulty cancels in the pairing, and an unpaired SD would describe a test we are not running. Across-instance variation enters only as the clustering correction.

**Informative denominator.** Only **present-value** slots can produce a span discordance. The campaign_v2 fixture ceiling is `exact_span 10/16` — exactly the 10 present-value slots. The 6 `NOT_MENTIONED` slots (enc-4's five absent + enc-1/medication) have no gold span and can never flip. Power is computed over **10 slots per instance**, not 16. Those 6 slots carry the abstention/commission axis instead and are analysed there.

**Rejected replicate axis.** Stochastic decoding was considered and rejected: the same seed under C1-on and C1-off yields different token streams, so samples are not matched draws. It would break the pairing and abandon greedy decoding, which every other number in this program uses.

## 4. Mechanism, and the predicted direction of every effect

Let $g$ = the model's genuine span-recovery rate, $q$ = P(emits a label with no quote), $\pi_{C2}$ = P(C2 fires and rescues to an exact gold span).

- Under **C1-on**, the answer template hands over the exact string. A parrot emits it *with* a quote, so $q \to 0$ and C2 never fires: $S(1,1) = S(1,0)$.
- Under **C1-off**, the model must locate the span itself; if it returns a bare label, C2 rescues it: $S(0,1) = g + \pi_{C2}$, $S(0,0) = g$.

Therefore
$$\theta = \bigl[S(1,1)-S(0,1)\bigr] - \bigl[S(1,0)-S(0,0)\bigr] = -\pi_{C2}.$$

### Predicted directions

| Effect | Predicted direction | Basis |
|---|---|---|
| **C1 main effect** | Turning C1 **off decreases** the score. | Removing a channel that dictates the answer cannot help a parrot. |
| **C2 main effect** | **≈ 0**, marginally. | C2's effect is $\approx 0$ at C1-on and $\pi_{C2}$ at C1-off; the marginal average is therefore roughly $\pi_{C2}/2$ and is *not* where C2's signal lives. |
| **C1 × C2 interaction** | **Super-additive damage** — removing both channels costs more than the sum of removing each alone, by exactly $\pi_{C2}$. Equivalently $\theta < 0$ on the score scale. | Redundant-rescue (OR-gate) structure with C1 dominant: C1 masks C2 by suppressing C2's trigger. |

**We predict sub-additivity nowhere.** The channels are redundant rescue paths for the same slots, and redundant paths are super-additive under joint removal, never sub-additive.

**Registered consequence:** C2's *entire* signal lives in the interaction. A main-effects-only reading of this 2×2 would conclude "C2 is inert" whether or not it is. That is the specific error this powering requirement exists to prevent.

## 5. Power — the replicate count, and what the pilot did to it

Because $\theta = -\pi_{C2}$, the interaction effect size **is** $\pi_{C2}$. Guessing it would make the replicate count faith-based; measuring it on measurement slots would make this prereg post-hoc. It was therefore estimated on **throwaway encounters disjoint from every measurement instance** (`nanoscribe/pilot_quote_absent.py`, results in `artifacts/pilot_quote_absent.json`) — an internal-pilot / blinded sample-size re-estimation, reporting only `quote_absent` and a span-match rate, never a scored campaign_v2 slot.

**Test.** Exact McNemar. At these sample sizes the normal approximation is not honest: conditional on $d$ discordant slots the null is $\mathrm{Binomial}(d, \tfrac12)$, so a one-directional $d$ needs $d \ge 6$ for $p<0.05$ two-sided ($d=5$ gives $0.0625$). **The design target is a discordant count, not a slot count.**

**Clustering.** Kish design effect $\mathrm{DEFF} = 1 + (m-1)\,\mathrm{ICC}$ with $m = 3.2$ slots/encounter and $\mathrm{ICC}=0.20$, giving $\mathrm{DEFF}=1.44$.

**Note $N$ and $\pi$ trade off.** $N\pi \approx 6$–$8$ is what is needed, so padding the suite with easy slots raises $N$, lowers $\pi$, and buys nothing. Added slots must be quote-ambiguous or the compute is wasted.

### Measured (20 present-value slots × 2 prompt passes, Qwen2.5-1.5B-Instruct, MPS)

| Quantity | Value |
|---|---|
| $\hat\pi_{C2}$ (pooled) | **0 / 40**; 95% Clopper–Pearson upper bound **0.072** |
| $\hat\pi_{C1}$ | **0.100** (2/20 discordant, both C1off-wins) |
| `exact_gold_span` base rate | **0/20** at C1-on, **2/20** at C1-off |

### Required K

| Contrast | $\pi$ used | K for 80% power |
|---|---|---|
| C1 main effect | 0.100 (measured) | **12 instances** |
| C1×C2 interaction | 0 (point estimate) | **none exists** — unidentified |
| C1×C2 interaction | 0.072 (95% upper bound) | 16 instances |

**The interaction is not underpowered — it is unidentified.** The model emitted a quote on 40/40 slots, including on `NOT_MENTIONED` where the format does not ask for one. C2's trigger (`label != NOT_MENTIONED and not quotes`) never occurred. No replicate count repairs a manipulation with no purchase on the model.

## 6. Registered predictions (falsifiable, stated before the run)

- **P1 — cell identity.** `C1on_C2off_Qon` returns **byte-identical** aggregates to the root, and `C1off_C2off_Qon` returns byte-identical aggregates to `C1off_C2on_Qon`. If any C2-off cell differs from its C2-on counterpart, something other than the registered flag moved and the run is **invalid**, not interesting.
- **P2 — precondition failure.** `quote_absent` will be 0 (or near 0) in every cell. The registered reading is then *"the parser channel never fires for this model"*, quoting `quote_absent` as the denominator. It is **not** the claim "C2 does not matter", and it is **not** evidence that the scorer is sound for other models.
- **P3 — floor.** `exact_gold_span` will be at or near floor in all cells, because the model quotes whole sentences rather than spans (`STATED: "My calf has been aching for three days."` for gold span `calf`).

## 7. Decision rule — attached to all four cells

**Primary endpoint (registered change).** `exact_gold_span` **alone cannot carry the verdict**: the pilot puts it on the floor (0/20, 2/20), where the previously-registered threshold "falls by ≥ 3 of 16 slots" is arithmetically unreachable and a null is a floor artifact rather than evidence about leakage. Registered co-primary endpoints:

1. `span_character_f1` — continuous, already computed, does not collapse when the model over-quotes.
2. `assertion_state_correct` — the label axis, independent of quote granularity.

`exact_gold_span` is retained as a **secondary** endpoint and reported as counts.

**H-leak.** The baseline span-port score is substantially carried by gold-value leakage rather than by evidence transport.

- **CONFIRMED** if, at K ≥ 12, the C1 contrast on **either** co-primary endpoint is significant under exact McNemar (d ≥ 6 one-directional) **in the predicted direction** (C1-off worse).
- **REFUTED** if, at K ≥ 12, both co-primary contrasts have d < 6 **and** the observed direction is not C1-off-worse.
- **UNRESOLVED** otherwise — report as such; do not round toward a verdict.
- **DIRECTION REVERSAL** is a reportable outcome in its own right: the pilot's C1 contrast ran *backwards* (C1-off 2/20 vs C1-on 0/20). If that replicates, the registered reading is that the gold-value template degrades span precision by inviting sentence-level quoting, not that leakage is absent.

**Interaction.** No verdict will be issued on C1×C2. Registered as **UNIDENTIFIED** on the pilot evidence, to be reported with $\hat\pi_{C2}$ and its upper bound.

**Secondary observation, not a verdict trigger.** The enc-4 false-positive signal (`unbound_assertion` over 5 absent slots) is reported but decides nothing: n=5 per instance is too thin, and the primary rule must stand alone.

**Blocking manipulation check — currently NOT binding.** `test_leakage_ablation.py::test_pure_echo_model_is_caught` and `test_prompts_stay_distinct_in_every_scoring_cell`. If either fails, a REFUTED verdict is **VOID** rather than reassuring — the Stage P/P1 lesson.

> **Defect, recorded before the run.** The echo adapter builds its answer from `spec.raw_value` (`f'STATED: "{spec.raw_value}"'`), **not from the prompt it was shown**. It is therefore cell-invariant by construction: it emits the same line whether or not the prompt carries the gold value, so it cannot detect a prompt-channel leak and does not bind this ablation. A parrot that binds must be built **from the prompt text**. Until it is, "the manipulation check passes" is not evidence. Credit: peer session `v7pagl1v`; verified here at `b6473cb:nanoscribe/test_leakage_ablation.py` line 201.

## 7b. HOLD — a leak that is open in all four cells

Registered after §5 and before any launch. **The 2×2 as specified cannot detect what it was built to detect.**

Q was split out of C1 as "task specification, not leakage" on the grounds that naming the concept is how a slot is identified at all. That is right in principle but wrong as implemented: `topic_for_spec` identifies the slot by its **gold surface string** (`Does the patient mention 'neck'?`), so with Q pinned ON the answer sits in the prompt in **every cell** — including `C1off_C2off_Qon`, the cell labelled "leakage-free".

Measured on i0, fixture-only, with a parrot that discards the transcript before parsing (perfect-reader ceiling is `exact_span 10/16`):

| Cell | parrot `exact_span` |
|---|---|
| `C1on_C2on` | 10/16 |
| `C1on_C2off` | 10/16 |
| `C1off_C2on` | 9/16 |
| `C1off_C2off` | **9/16** |

A model with **zero transcript access** scores within one slot of ceiling in the leakage-free cell. `exact_gold_span` is saturated, and the previously-registered **REFUTED** branch ("moves ≤ 1 slot across all four cells") is precisely the signature of a channel open everywhere — so it **cannot** be read as evidence of no leakage. That branch is hereby **void as an inference**, not merely underpowered.

This is a *stronger* reason to hold than the power finding in §5, and it explains it: $\hat\pi_{C1}=0.10$ is small because Q carries the answer regardless of C1.

The joint table **does** still discriminate — parrot `state_ok` 6/16 vs ceiling 10/16, `correct_abstention` 0–3/6 vs 6/6, `unbound_assertion` 3–6 vs 0 — which is independent support for the §7 move from a single accuracy scalar to discrimination-based endpoints.

Source: peer session `v7pagl1v` at `b6473cb`, independently reproduced there with a separately-written parrot. **No node launches until `Q_SURFACE` is resolved** — i.e. until the slot can be identified without naming its gold surface string (a slot id, a type-plus-position index, or a paraphrase that does not contain the answer).

## 8. Reporting contract

Results are reported as the joint **(grounded | unbound) × (asserted | abstained)** table, plus **selective risk at matched coverage**.

**Open dependency, stated rather than fudged.** `qwen_inference.py` retains no logprobs (`generate()` is called without `output_scores`), so there is currently no confidence signal to threshold on and matched coverage is **not computable**. Either sequence logprob is captured before the run, or each cell's selective risk is reported at its **natural** coverage with the coverage delta stated explicitly. Reporting unmatched numbers as "matched" is the failure mode being guarded against here.

## 9. Launch gate

This prereg does **not** authorize launch. Launch requires:

1. K ≥ 12 instances in the instrument (currently 5), **or** an explicit registered downgrade of the C1 contrast to descriptive.
2. The co-primary endpoints emitted by the runner.
3. Sequence-logprob capture, or the registered natural-coverage fallback.
4. `git diff <baseline>...orx/<child>` showing **only** the flag flip for each child — anything else confounds the ablation.
