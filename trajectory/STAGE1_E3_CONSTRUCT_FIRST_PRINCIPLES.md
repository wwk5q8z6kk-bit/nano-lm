# Stage 1 — E3 construct validity (first principles)

*Bounded human arm executed (not skipped). Docs + labels only.
No E2, fabric, or old-task substrate runs. 2026-07-31.*

**Artifacts**

| Path | Role |
|------|------|
| `PREREG_E3_faithfulness_construct.md` | Pre-registered thresholds |
| `e3_human_rating_pack.json` | Frozen n=100 exact-error pack |
| `results_e3_normalize_construct.json` | Auto arm (0/486 rescues) |
| `results_e3_human.json` | This Stage 1 labeling + verdict |
| `PIPELINE_GATE_LOG.md` | Gate 1 result |

---

## 1. What question can E3 actually answer?

**Threat T3:** The science metric (exact string match) might be measuring
*string disagreement* rather than *faithfulness failure*.

Three nested evaluation layers (from first principles):

| Layer | Question | Already known / this stage |
|-------|----------|----------------------------|
| **Exact** | Is `pred == truth`? | Pack = 100 exact failures by construction |
| **Normalized** | Same after case/punct/plural strip? | Auto: **0/486** M0 rescues; on pack: **0/100** |
| **Human-acceptable** | Same clinically relevant fact? | **This arm:** 0/100 acceptable under frozen rubric |

If many exact failures were still normalized-equal or human-acceptable, reported
gaps would **overstate** failure. If almost none are, exact-match is a **harsh but
directionally faithful** instrument *for this synthetic world*.

---

## 2. What E3 cannot answer (limitations — do not paper over)

1. **Not clinical validation.** Dialogues are synthetic; “clinically relevant”
   means *task-fact equivalence under the schema*, not real-world medicine.
2. **Single rater pass.** Labels are `agent-rubric-pass-1` applying written pattern
   rules. **No IAA.** A second human clinician pass could move edge cases
   (especially partial strings). Until then, treat human arm as **bounded rubric
   audit**, not gold clinical annotation.
3. **Pack composition bias.** Amendment 1 pack is **exact errors only**, almost
   all `held=True`, fields CC/MED/ALG only, method **M0_scale verify-off**. It
   stresses the failure mass; it does **not** estimate population false-negative
   rate of exact-match on successes.
4. **Normalize rule is thin.** Frozen `normalize_value` handles case, light punct,
   articles, trivial plurals — **not** synonyms (Tylenol/acetaminophen),
   abbreviations, morphology beyond plural, or reordering. Auto arm already
   showed that thin normalize rescues nothing; that fails **formatting** as the
   explanation, not synonymy.
5. **Synonymy / ontology (H4) remains open.** This Stage 1 does **not** test
   whether a coding-style soft objective would shrink gaps. That would need a
   different pre-registered synonym table or human ontology rubric — out of
   Stage 1 scope.
6. **Partial-span edge case.** `throat` vs `throat lozenges` (14/100) is the
   softest class. Rated **unfaithful / not human-acceptable**: incomplete med
   identity. A looser “partial credit” metric could reclassify these; we
   **refuse** to move the goalposts post hoc. Report as sensitivity note: even
   if all 14 were flipped to faithful, faithful-rate = 0.14 **&lt; 0.20** survive
   threshold and **&lt; 0.50** collapse threshold → verdict unchanged.
7. **Does not unlock product or reopen E1.** Construct clearance of exact-match
   as a *science* instrument ≠ generative substrate justification. `OLD_TASK_RUNS`
   stay forbidden under `OLD_TASK_U`.
8. **Does not identify mechanism.** Substitutions vs omissions are descriptive
   error modes, not LoRA/circuit evidence (E2 still gated).

---

## 3. Protocol used (no skip)

**Owner direction:** bounded E3 human — fixed subset, rubric
exact/normalized/human-acceptable, decision: does qualitative open-slot gap
shrink materially?

**Subset:** the **full frozen pack n=100** (already the bounded Amendment-1
study). No new sampling; no method retuning; no eval leakage into rules.

**Decision thresholds (prereg, unchanged):**

| Verdict | Rule |
|---------|------|
| CONSTRUCT_COLLAPSE | auto shrink ≥10 pts **or** faithful-rate ≥0.50 |
| EXACT_SURVIVES | auto shrink &lt;5 pts **and** faithful-rate &lt;0.20 |
| GRADED | otherwise |

**Qualitative gap shrink (Stage 1 one-liner):** material iff
`human_acceptable_rate ≥ 0.50` on the exact-error pack.

---

## 4. Results

| Quantity | Value |
|----------|-------|
| Auto gap shrink (M0) | **0.0** pts (0/486) |
| Pack normalized matches | **0/100** |
| Pack human-acceptable | **0/100** |
| Pack faithful (prereg) | **0/100** (rate **0.00**) |
| Unfaithful | **100/100** |
| Qualitative gap shrinks materially? | **No** |

**Error anatomy (descriptive):**

| Class | n | Reading |
|-------|---|---------|
| ALG omission (`none` vs `sulfa drugs`, gold in dialogue) | 23 | Safe decline / miss — exact correctly flags |
| CC substitution / garble (wrong complaint) | 63 | Different fact — exact correctly flags |
| MED truncation (`throat` vs `throat lozenges`) | 14 | Incomplete med — exact correctly flags under full-value rubric |

**Prereg verdict: `EXACT_SURVIVES`.**

**Gate 1 action row:** *Gap remains large under soft/human → copy-failure
interpretation stands; proceed to Stage 2.*

---

## 5. Sensitivity (pre-committed honesty)

| Alternate soft rule | Faithful-rate | Verdict change? |
|---------------------|---------------|-----------------|
| As rated | 0.00 | EXACT_SURVIVES |
| Count all MED truncations as faithful | 0.14 | Still EXACT_SURVIVES (&lt;0.20) |
| Count MED truncations + any substring overlap as faithful | 0.14 | Same (only MED overlaps) |
| Collapse threshold 0.50 | — | Not approached |

---

## 6. Updated posterior (construct)

| Claim | After Stage 1 |
|-------|----------------|
| Formatting/normalize drives the gap | **Falsified** (auto + pack) |
| Exact-match mostly flags human-acceptable paraphrases on this pack | **Falsified** under stated rubric |
| Exact-match may still overstate vs rich synonym ontology | **Open** (not tested) |
| Paper α must keep exact≠human-equivalence limitation | **Still required** — different reason: metric is strict and human arm is single-pass/synthetic, not because failures are “just formatting” |
| Product path / E4 | **Unaffected** — still needs R★ + P2 |

---

## 7. Gate 1 record

**PASS — gap remains large under soft/human (EXACT_SURVIVES).**  
Proceed Stage 2 (R★ harden; already drafted) → Stage 3 P2.  
No old-task LM runs. No E2/fabric.
