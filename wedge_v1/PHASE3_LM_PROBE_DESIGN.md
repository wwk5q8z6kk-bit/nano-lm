# Wedge v1 Phase 3 — LM probe design (prep only)

**Status:** `DESIGN_READY_AWAITING_AUTH`  
**Auth required:** `AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE` — **NOT GRANTED**  
**Prerequisite RESULT:** Phase 2 classical baseline  
  (`wedge_v1/results_wedge_v1_classical.json`, U≈0.926, 40/40 on clean synthetic track)  
**Companions:** `papers/WEDGE_V1.md`, `papers/FIRST_PRINCIPLES_RISK_MITIGATION.md`,  
  `trajectory/PROGRAM_A_FIRST_PRINCIPLES_MITIGATIONS.md` (science lessons only; Program A **superseded** as product default)

```text
PHASE3_EXECUTE = FORBIDDEN
TRAINING = FORBIDDEN_UNTIL_SEPARATE_AUTH
LM_SOLVERS = FORBIDDEN_UNTIL_PHASE3_AUTH
A_D_REOPEN = FORBIDDEN   # do not put LM on tasks classical already wins
```

---

## 0. Why Phase 3 exists (first principles)

Phase 2 showed: on this **clean synthetic** mini-corpus, classical+verify already delivers high \(U\).

The only honest generative question left on *this* pack is:

> On E-class items where classical **correctly abstains** (no lexical path), does a small LM + verify improve \(U_{\mathrm{dep}}\) vs continued abstention — under information parity and constructive faithfulness?

That is **not** “beat classical on A–D.” A–D classical wins are load-bearing. Reopening them would recreate E1 theater.

---

## 1. Phase 2 facts that bind Phase 3

| Fact | Implication |
|------|-------------|
| U≈0.926, Q=1.0, E=0, R≈0.079, C=1.0 | Classical is the **reference** solver; LM must beat *this* under frozen \(U\) |
| T35 / T36 / T39 = ABSTAIN (by design) | Only these (plus optional T37 if classical saturates) are probe candidates |
| Liability presented_bad = 0 | LM must not raise presented fabrications |
| Probe flags B1–B4 true | World inclusion OK; do not rewrite gold after scores |
| Clean track only | Noisy OCR track stays diagnostic |

**Gold E-class behavior (classical):**

| Task | Classical action | Why |
|------|------------------|-----|
| T35 paraphrastic retrieve | ABSTAIN (`retrieve_no_paraphrase`) | No lexical overlap by construction |
| T36 implicit dose-change | ABSTAIN (`implicit_default_abstain`) | Relation not surface-entailed |
| T39 coref binding | ABSTAIN (`coref_lite_abstain`) | Lite rules refuse ambiguous “It” |

Phase 2 *scores these ABSTAIN as pass* — that is correct for classical. Phase 3 asks whether LM can **present verified claims** on a subset without destroying overall \(U\).

---

## 2. Blocker inventory → mitigations (Phase 3-specific)

| ID | Blocker | Root cause | Mitigation |
|----|---------|------------|------------|
| **P3-R1** | LM on A–D “for completeness” | Scope creep / substrate revival | **Hard allowlist:** T35, T36, T39 only (optional T37 normalize path if needed). A–D stay classical-only. |
| **P3-R2** | Post-rationalized citations | Generate-then-cite | Constructive faithfulness: retrieve/select spans **before** claim text; span-ablation must fail support (WEDGE B13) |
| **P3-R3** | Cost kills fair capability read | Mixing \(U_{\mathrm{dep}}\) and science | Dual estimands: \(U_{\mathrm{dep}}\) official; \(U_{\mathrm{cap}}=Q-0.5E-0.3R\) diagnostic (Program A mitigations §3) |
| **P3-R4** | Undertrained / toy LM confounds regime | Method–task interaction | Frozen **recipe card**: base SHA, steps/tokens, venue, seed; or off-the-shelf encoder/decoder with **no train** if owner chooses zero-train probe |
| **P3-R5** | Information asymmetry | LM sees gold schema / classical misses | Same corpus bytes + same schema; no gold offsets in LM context |
| **P3-R6** | Forced-answer Goodhart | Always-answer raises E/liability | Verify-on mandatory; ABSTAIN allowed; forced-answer ablation must worsen \(Q\) |
| **P3-R7** | Synthetic ceiling | Clean corpus understates real docs | Phase 3 RESULT scoped `clean_synthetic`; noisy track separate (needs auth extension) |
| **P3-R8** | Circular LM judge | Scoring with LM | Official \(U\) = gold atoms + span containment + entity/number lock only |
| **P3-R9** | One-shot information loss | Single probe, many mechanisms | Pre-tag strata: paraphrase / implicit / coref; report per-stratum diagnostic, overall official |
| **P3-R10** | Productizing a probe | U_LM↑ on 3 tasks → “ship NanoScribe” | SURVIVE = registry admit **only** those task classes; no OS claim |

---

## 3. Decision question (frozen form)

Under frozen Phase-2 \(U\) weights (\(\delta=0.05\)):

```text
U(classical + LM_on_E + verify; clean)  ?≥  U(classical_only; clean) + δ
```

on the **same** corpus/gold, with LM eligible **only** on allowlisted E tasks.

| Verdict | Meaning |
|---------|---------|
| **SURVIVE** | LM solver admitted to registry for allowlisted task classes only |
| **KILL** | Keep classical abstention on E; no LM path in product for this wedge v1 |
| **VOID** | Auth/recipe/gold mutation / parity break — no claim |
| **GRADED** | \(U_{\mathrm{cap}}\) gain without \(U_{\mathrm{dep}}\) SURVIVE → science note only; product KILL |

Sensitivities that zero \(C\) **cannot** flip official \(U_{\mathrm{dep}}\) verdict.

---

## 4. Solver cascade (Phase 3)

```text
task ∈ A–D  → classical only (unchanged)
task ∈ {T35,T36,T39}:
    1. classical attempt (may ABSTAIN)
    2. if ABSTAIN and Phase3 auth: LM propose under span-first protocol
    3. verifier (binding + support + entity/number lock)
    4. PRESENT | ABSTAIN | REVIEW
```

No LM call on A–D even if classical is wrong (Phase 2 says it isn’t on this pack).

---

## 5. Constructive faithfulness protocol (mandatory)

For every LM-presented claim:

1. **Retrieve** candidate spans (BM25 / exact) *or* model-proposed offsets that must exist in doc text.  
2. **Lock** entities/numbers appearing in the claim to substrings of those spans.  
3. **Emit** claim text as quote, normalize, or paraphrase-under-lock — never free invent then cite.  
4. **Ablation probe:** remove cited offsets → support must fail.  
5. Empty evidence → reject (T33 law).

---

## 6. Recipe card (fill at execute auth — examples only)

```text
phase3_mode: <zero_train_api_or_local | sft_tiny>
base_model: <id>
base_sha256: <pin or N/A for API>
train_steps: <0 | N>
max_usd: <0 | N>
venue: <local-mps | runpod-cuda | ...>
seed: <int>
allowlist_tasks: [T35, T36, T39]
C_schedule_ack: true
U_dep_only_verdict: true
constructive_faithfulness: true
old_task_u: forbidden
```

**Default recommendation (cheapest falsifier):** `zero_train` local small instruct model, greedy decode, verify-on — answers “does generation help *at all* under verify” before any SFT spend.

---

## 7. Builder checklist (unlocked only by Phase 3 auth)

| Step | Action |
|------|--------|
| 0 | Confirm Phase 2 RESULT hash unchanged |
| 1 | Freeze Phase 3 recipe card + auth record |
| 2 | Implement LM path **only** behind allowlist |
| 3 | Wire constructive faithfulness + ablation probe |
| 4 | Score verify-on / verify-off; write `results_wedge_v1_phase3.json` |
| 5 | Compute \(U_{\mathrm{dep}}\) official + \(U_{\mathrm{cap}}\) diagnostic + per-stratum |
| 6 | Decision.json: SURVIVE / KILL / VOID |
| 7 | SHA256SUMS + no gold edits |

---

## 8. Explicit non-actions under this design doc

- Do not train or call LM APIs  
- Do not edit Phase 2 gold / corpus  
- Do not admit LM on A–D  
- Do not treat synthetic SURVIVE as clinical readiness  
- Do not reopen Program A / E4-prime as the default product path  

---

## 9. Ready state

```text
PHASE2 = DONE
PHASE3_DESIGN = READY
PHASE3_EXECUTE = NOT_AUTHORIZED
NEXT_OWNER_STRING = AUTHORIZE_WEDGE_V1_PHASE3_LM_PROBE
ALTERNATE = IDLE_ON_WEDGE  # classical-only product surface is already strong on clean track
```

Idle is a valid owner choice: Phase 2 already showed classical+verify works on this wedge’s clean synthetic pack. Phase 3 is optional curiosity about E-class residual, not a required unlock.
