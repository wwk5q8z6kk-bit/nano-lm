# Project evolution plan

**Issued:** 2026-08-22  
**Authority:** Owner authorization to think forward and plan evolution  
**Scope:** Planning only — does not authorize spend, tag moves, ledger edits, or public claims  
**Front door for execution:** `papers/ACTIVE_NOW.md`  
**Frontier mandate:** `frontier/ACTIVE_MANDATE.md` · branch `frontier/active-v1`

---

## 1. North star (unchanged)

Build **Nano**: a local-first scribe intelligence that turns a transcript (or document) into a **structured record where every asserted value is bound to evidence — or withheld**.

Operating rule:

```text
smallest sufficient solver → verify → present | abstain | review
```

This is a **system** claim, not a "tiny LM scores better" claim.

---

## 2. Two tracks (do not collapse)

| Track | What it is | Success looks like |
|-------|------------|-------------------|
| **Nano AI** (`nano_ai/`, `sft/`, P4) | The actual scribe model + span contract + real-document transfer | CUAD P4 passes preregistered gates on ≥3 seeds |
| **Wedge v1** (`wedge_v1/`) | Supporting lab: retrieval, verify, abstain, failure galleries, U measurement | Owner-private corpus usefulness sentence + honest kill/pivot |

**Wedge is not Nano.** Green dogfood on `papers/` or fixtures does not validate the scribe product.

**Paper α / Evidence Ledger / freeze tags** are a third layer: historical science — cite, never silently rewrite.

---

## 3. Governance tiers (replace paralysis with boundaries)

### Tier A — Evidence & publication (strict)

Requires explicit owner force + prereg where applicable:

- Scientific verdict changes, ledger row promotion
- Protected tag moves, freeze brand claims
- Public capability claims, clinical readiness
- Large paid compute, destructive ops

### Tier B — Research & prototype (disciplined, not paralyzed)

Allowed under **ACTIVE_MANDATE** / `frontier/active-v1` without per-shell auth:

- Code, tests, branches, local MLX/CUDA experiments under cost ceiling
- Wedge architecture workstreams W1–W6 (LM only after `lm-admit` says INDICATED)
- Nano P4 **after** `PREREG_P4_CUAD.md` frozen
- Failure galleries, contact protocols, benchmark integration

### Tier C — Ordinary engineering (free inside scope)

- Refactors, docs that don't change Layer-1 claims
- CLI/UX for wedge (`report`, `study`, `status`)
- Bug fixes, smoke tests, ingest paths

**Invariant:** Tier C must not smuggle Tier A changes.

---

## 4. Where we are (2026-08-22)

### Nano — engineering center (`ACTIVE_NOW`)

| Done | Implication |
|------|-------------|
| LoRA control | Pretraining supplies lexical transfer; from-scratch path parked |
| Span route (b) + snap + field filter | Accepted inference stack (`RESULT_FIELD_FILTER.md`) |
| 20-seed calibration | Grounding robust; all-six pass rate 0.60 — binding is hard, not random |
| CUAD decision + token-windowed asks (n≈40 pilot) | Sequence-length finding closed; P4 instrument ready |

| Next (gated) | Blocker |
|--------------|---------|
| Freeze `PREREG_P4_CUAD.md` | Must precede any scored CUAD eval |
| Abstention-floor pilot on held-out CUAD slice | Sets P4 floor (67.95% impossible — cannot guess) |
| n=40 token-windowed score, ≥3 seeds | Real-document transfer verdict |

### Wedge — proving ground (`frontier/DEVELOPMENT_PLAN`)

| Done | Implication |
|------|-------------|
| P0–P1 Verified Ask slice | ask/find/scan/compare/report/ingest + CoE binding |
| W1–W3 architecture | BM25 margin, evidence atoms, epistemic merge |
| Fixture + papers contact harness | Useful for lookup/TTL; weak on open NL questions |
| W6 admission harness | **LM_PROBE_NOT_INDICATED** on clean synthetic |

| Blocker | Why it matters |
|---------|----------------|
| **Real `$OWNER_CORPUS`** + usefulness labels | Only path to product judgment (not fixture theater) |

---

## 5. Evolution arc (12 months)

```text
Q3 2026  Nano P4 prereg → pilot → scored CUAD (real-doc verdict)
         Wedge owner-corpus contact (parallel, cheap)
Q4 2026  If P4 passes: narrow Nano product slice (batch inference + binder only)
         If wedge useful: deepen ingest/retrieval; if not: pivot wedge to Nano test harness only
2027 H1  Optional: LM in wedge only if measured ΔU > δ on owner corpus
         Optional: clinical dialogue bridge (ACI-Bench class) — separate prereg
Never    NanoScribe OS, OLD_TASK_U default gen, governance-for-its-own-sake loops
```

---

## 6. Phase plan (sequenced)

### Phase N0 — P4 instrument lock (Nano) · **NOW**

1. Freeze `PREREG_P4_CUAD.md` (gates from `DESIGN_P4_GATES.md` — P2/P3/P5 + abstention pilot)
2. Run abstention-floor pilot; name floors before main score
3. Score n=40 windowed asks, ≥3 seeds, per-seed reporting + Wilson intervals

**Exit:** `RESULT_P4_CUAD.md` with pass/fail per preregistered gate  
**Kill:** Cannot meet grounding floors (P2/P3) or abstention+coverage pair on answerable third

### Phase N1 — Owner corpus contact (Wedge) · **parallel**

```bash
export OWNER_CORPUS=/path/to/private/docs   # never commit contents
./scripts/gate0_contact.sh
python -m wedge_v1 review --corpus "$OWNER_CORPUS" --interactive
```

**Exit:** One honest useful / not-useful sentence + labeled reviews  
**Kill:** Owner would not use weekly; only path to value is unverifiable generation

### Phase N2 — Integration decision (gate)

| If P4 pass + wedge useful | If P4 pass + wedge meh | If P4 fail |
|---------------------------|------------------------|------------|
| Nano inference path + wedge as eval harness for regressions | Ship Nano slice; wedge maintenance mode | Revisit base size / span port / training data — not "more governance" |
| LM probe only on measured wedge failures | No LM in wedge by default | Do not expand product claims |

### Phase N3 — Product slice (only after N0 pass)

Minimal batch scribe: transcript in → structured JSON + spans out → abstain  
No UI, no clinical deployment, no cloud default

### Phase N4 — Optional LM in wedge (only after N1 + W6 INDICATED)

Preregister marginal value: U_classical vs U_hybrid on **owner** task pack  
δ = 0.05 default

---

## 7. What to stop doing

1. **24-task reconciliation loops** — DIFF E committed; tags documented; move on  
2. **Treating every commit like Layer-1 evidence** — use Tier B/C defaults on `frontier/active-v1`  
3. **Reviving from-scratch Nano** because `EXECUTION_QUEUE` Priority 10 still says "Next"  
4. **Using wedge fixture green as Nano validation**  
5. **Scoring CUAD before prereg freeze**  
6. **Governance doc churn as substitute for P4 or owner corpus**

---

## 8. Decision gates (summary)

| Gate | Question | Authority |
|------|----------|-----------|
| G-P4-PREREG | Are P4 floors frozen before data? | Owner + prereg file |
| G-P4-PASS | Does Nano transfer to real legal text? | `RESULT_P4_CUAD.md` |
| G-OWNER | Would owner use wedge weekly on private docs? | Owner labels + sentence |
| G-LM | Does LM beat classical by ΔU on owner tasks? | `lm-admit` + preregistered probe |
| G-SHIP | Any public "Nano works" claim? | Owner + ledger update |

---

## 9. Immediate actions (next 14 days)

| # | Action | Track | Owner |
|---|--------|-------|-------|
| 1 | Draft/freeze `PREREG_P4_CUAD.md` | Nano | Agent + owner sign-off |
| 2 | Abstention pilot on held-out CUAD slice | Nano | Agent (local MLX) |
| 3 | Point `$OWNER_CORPUS` at real folder; run `gate0_contact.sh` | Wedge | **Owner path required** |
| 4 | Keep smoke/dogfood green; no new governance docs | Both | Agent |
| 5 | Do **not** push tags or amend ledger without typed force | Evidence | — |

---

## 10. Success definition (evolution complete)

The project has **evolved** (not just documented) when:

1. **Nano** has a preregistered real-document RESULT (P4) that survives its own gates  
2. **Wedge** has an owner-private usefulness verdict (yes/no with evidence)  
3. **Evidence Core** remains intact and citeable  
4. **One** minimal inference path exists that a developer can run locally without reading 50 status docs  
5. Governance applies **only** where Tier A risk exists

---

## 11. Anti-goals

- General chatbot / NotebookLM clone without verify-first differentiation  
- NanoScribe cognitive OS  
- Clinical deployment claims  
- Unlimited R★ / E4 redesign without revision budget  
- Agent-IDE scope creep on `frontier/active-v1`

---

*Planning artifact. Execution authority remains `ACTIVE_NOW.md` + owner typed forces for Tier A.*
