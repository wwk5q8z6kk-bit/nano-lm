# Accelerated Research Campaign v2

**Status:** Active plan on `frontier/accelerated-research-campaign-v2`  
**Authority:** Complements [ACTIVE_NOW.md](ACTIVE_NOW.md) and [EXECUTION_PLAN.md](EXECUTION_PLAN.md). Paid compute follows [artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md](../artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md).  
**Machine manifest:** [frontier/accelerated_research_campaign_v2.json](../frontier/accelerated_research_campaign_v2.json)  
**Prior campaign:** v1 complete — see [artifacts/campaign/checkpoint_v4.json](../artifacts/campaign/checkpoint_v4.json)

---

## 1. Current state (repo truth)

### Branch and integration posture

| Layer | This branch (`frontier/accelerated-research-campaign-v2`) | Cross-branch / worktree |
|-------|-----------------------------------------------------------|-------------------------|
| **P1 stack** | `nanoscribe/` — encounter schema, evaluator, harness, tool-calling adapters, native training, campaign infra (53+ modules) | Selective primitives from `cursor/span-port-route-b-182e` per [frontier/p1_integration_manifest.json](../frontier/p1_integration_manifest.json) |
| **nano_ai / H6 transfer** | Not integrated wholesale; span-port grammar + evaluator lifted | Full tree on `cursor/span-port-route-b-182e`; weights/logs **rejected** for import |
| **P4 CUAD harness** | Explicitly **parked** — integration manifest `REJECT` | `frontier/nanoscribe-core-v1` @ `/Users/mac/Projects/nano-lm-nanoscribe` (`nano_ai/training/`, `benchmarks/`) |
| **Wedge** | Integrated — `wedge_v1/` classical-first document QA | Supporting subsystem only; LM escalation not indicated |
| **Fabric / Paper α / E1 / E4** | Integrated — frozen evidence, regression harness | Verdicts scoped per [FAILURE_TO_ARCHITECTURE.md](FAILURE_TO_ARCHITECTURE.md) |
| **Note realization** | **Not built** | B3 in EXECUTION_PLAN still open |
| **External medical benchmarks** | **Not wired** — no MTS-Dialog / ACI-Bench / PriMock57 adapters in tree | — |
| **Human clinician eval** | Protocol drafted only — [domains/medical/EVALUATION_PROTOCOL.md](domains/medical/EVALUATION_PROTOCOL.md) | Not executed |

### Campaign v1 empirical snapshot (measured, not aspirational)

Source: `artifacts/campaign/student_gap_v1.json`, `campaign_status.json`, `native_extended_summary.json`.

| Arm | C2 screening (n=128) | Assertion correct | Coverage | Exact gold span | Malformed |
|-----|-------------------|-------------------|----------|-----------------|-----------|
| Managed ref (Qwen3-32B-AWQ) | winner | **1.00** | 0.787 | **0.110** | 11 |
| Student A (Qwen2.5-32B) | distillation target | 0.680 | 0.781 | 0.102 | 6 |
| Native 100M evidence_bottleneck (extended 200 steps) | smoke only (n=3) | 0.00 | 0.00 | 0.00 | **8/8** |

**Interpretation:** The binding bottleneck is **span/evidence transport**, not assertion classification alone. Managed reference achieves perfect assertion-state accuracy on C2 but only ~11% exact gold span rate. Native hash-LM training converges on loss but does not yet produce valid structured decode. Student vs managed ref assertion gap (−0.32) is documented; QLoRA canary is unlocked but not yet decisive for world-class.

### What works today

- Encounter representation v0 in software (`nanoscribe/encounter.py`) with transport/support/state evaluator (`nanoscribe/evaluate.py`)
- Three-track harness: fixture, serverless, managed reference (`nanoscribe/harness.py`)
- Tool-calling / CandidateAtom JSON path with offline smoke pass
- Deterministic verifier hard set: 500 cases @ 100% baseline (`checkpoint_v4.json`)
- Agent canary: 48/48 parse on GPT-OSS-120B; outcome mean 0.5 (tool-selection ceiling, not scribing)
- Campaign control plane, wallet gates, experiment manifests

### What does not work yet (blocks world-class)

1. **Exact evidence span transport** (~10% on best operational model)
2. **Verified record → note rendering** (no implementation)
3. **External OOD medical dialogue evaluation** (datasets not integrated)
4. **Blinded human evaluation** (protocol only)
5. **Native compact model P1 viability** (garbled decode despite training loss convergence)
6. **Real-document transfer probe** (P4 CUAD parked; not a P1 substitute)

---

## 2. World-class P1 scribe — operational definition

"Best among all" means **measurable superiority on faithful, evidence-grounded clinical documentation** under a frozen utility function — not leaderboard rank on a single automatic metric.

### Primary outcome metrics (frozen before confirmatory runs)

| Metric | Definition | World-class directional target* | Current best (campaign v1) |
|--------|------------|--------------------------------|----------------------------|
| **Critical error rate (CER)** | Unsupported or contradicted clinical claims that would harm care if uncorrected | ≤ 0.5% on external held-out; zero tolerance class for allergy/dose/contraindication | Not measured on external set |
| **Exact-value transport** | Numeric/unit/dose strings match gold after normalization | ≥ 95% on eligible atoms | Not separately reported; span transport ~11% |
| **Evidence span correctness** | Predicted span matches gold offsets (transport layer) | ≥ 85% eligible atoms | **0.110** (managed ref C2) |
| **Assertion / negation / temporality** | STATED/DENIED/NOT_MENTIONED + temporal axes | ≥ 95% on eligible | **1.00** assertion (managed ref C2) |
| **Omission severity** | Weighted by clinical criticality (allergy > ROS fluff) | Critical omission ≤ 2%; any-tier recall ≥ 90% | Omission=1 on student C2 (128 cases) |
| **Clinician edit distance** | Token/section edits to reach acceptable note (human protocol) | Median ≤ 15% of note length | **Not measured** |
| **Time to final note** | Wall-clock transcript → clinician-accepted note | Competitive with best commercial scribe on same encounters | **Not measured** |
| **Review burden** | Fraction requiring human review at fixed risk threshold | ≤ 10% at 99% critical-error catch | Verifier hard-set only; not end-to-end |
| **Risk–coverage** | Precision vs coverage under abstention policy | Beat classical+verification baseline at equal coverage | Classical baseline not yet run on same external set |

\*Directional targets are **hypotheses for preregistration**, not claims. Confirmatory runs require [EXPERIMENT_STRATEGY.md](research/EXPERIMENT_STRATEGY.md) prereg + owner authorization.

### Benchmarks and datasets (planned integration)

| Dataset | Role | Status |
|---------|------|--------|
| **Internal C1/C2 screening** | Fast regression, mechanism discrimination | **Active** — `p1_screening_eval_v1` frozen forever |
| **MTS-Dialog** | External medical dialogue → note faithfulness | Not integrated |
| **ACI-Bench** | Clinician-AI dialogue benchmark | Not integrated |
| **PriMock57** | Primary-care mock consultation | Not integrated |
| **p1_distill_train_v1** | Teacher disagreement / failure patterns only | Active for training; disjoint from screening |

### Baseline we must beat

Per E1/E4 and [FAILURE_TO_ARCHITECTURE.md](FAILURE_TO_ARCHITECTURE.md), the comparison arm is **not** raw LLM generation alone:

```text
classical extraction + constrained span selection + independent verification
→ present | abstain | review
```

Any generative or student model must beat this stack on **utility U** at equal or lower review burden before claiming P1 superiority. Historical scribe v1/v2 failures (recall 74%, halluc 14%) set the floor for what "not world-class" looks like.

### Human evaluation protocol (required for exit)

Per [domains/medical/EVALUATION_PROTOCOL.md](domains/medical/EVALUATION_PROTOCOL.md):

1. Blinded pairwise or single-arm review by ≥2 clinicians per encounter slice
2. Critical-error adjudication with severity rubric
3. Edit-distance and time-to-acceptable-note measurement
4. Owner sign-off on P1 exit record

Mock/synthetic PASS alone is explicitly insufficient.

---

## 3. Gap analysis

Mapped from [FAILURE_TO_ARCHITECTURE.md](FAILURE_TO_ARCHITECTURE.md) and campaign v1 results.

| Gap | Evidence | Architectural response | Campaign v2 priority |
|-----|----------|------------------------|-------------------|
| **Span/evidence bottleneck** | exact_gold_span ~11% at best; student support_direct_exact 0.68 | ConstrainedSelector + evidence heads; teacher distillation on failure patterns; native evidence_bottleneck arm | **P0** |
| **Note realization not built** | No `render_note` / NotePlan in codebase; B3 open | Record-first rendering with claim decomposition + verification | **P0** |
| **No external eval** | Zero MTS/ACI/PriMock adapters | Dataset adapters + same evaluator; preregister before confirmatory | **P1** |
| **No human eval** | Protocol draft only | Pilot 20–50 encounters; rubric in EVALUATION_FRAMEWORK | **P1** (after automatic gate) |
| **Native decode failure** | 8/8 malformed smoke post-200-step train | Defer 300M promotion; fix hash-LM → structured line decode or hybrid routing | **P2** |
| **Real-doc transfer (P4)** | CUAD parked on nanoscribe worktree | Reconnaissance only — does not unblock P1 gate | **P3** (deferred) |
| **Assertion gap student vs teacher** | −0.32 assertion on C2 | Optional QLoRA 50-step canary — unlocked, not yet run | **P2** (experiment-scoped) |
| **Over-abstention risk** | wedge lesson | Fail-closed with nearest-evidence surfacing | Design constraint |

**Bottom line:** World-class is blocked primarily by **evidence transport + end-to-end system** (record → verified note → human-acceptable output), not by picking a larger teacher.

---

## 4. Campaign v2 — three loops

### Loop A — Model science

**Question:** Which model mechanisms improve span transport and assertion under fixed software?

| Exp ID | Hypothesis | Surface | Metric | Stop condition |
|--------|------------|---------|--------|----------------|
| A1 | Constrained tool-calling improves span vs free generation | Managed ref + student serverless | Δ exact_gold_span, support_direct_exact on C2 | ±2% span with no assertion regression → continue; else pivot selector |
| A2 | Distillation on failure/disagreement patterns closes assertion gap | Axolotl Hub 50-step canary → short QLoRA | Δ assertion (−0.32 target), malformed | No Δ after canary → stop QLoRA lane |
| A3 | Evidence-bottleneck native arch beats decoder-only at 100M | Native pod (existing weights path) | P1 smoke viable (malformed < 20%) | Still garbled → defer native to Loop B software fixes |
| A4 | Teacher ceiling on span transport | GPT-OSS-120B / Qwen3-32B managed | exact_gold_span upper bound | If ceiling < 50%, problem is task/selector not student size |

**Budget:** Within active experiment envelope (~$149 campaign remaining per `checkpoint_v4.json`). Each paid job requires manifest per CAMPAIGN_AUTONOMOUS_EXECUTION.

### Loop B — System intelligence

**Question:** Does verified encounter record + verification stack beat raw generation on utility U?

| Exp ID | Hypothesis | Deliverable | Metric | Stop condition |
|--------|------------|-------------|--------|----------------|
| B1 | Encounter schema v0 JSON + CI pins | `nanoscribe/encounter.py` + schema artifact | Schema tests pass | Done when exported + docs synced |
| B2 | Span transport on managed student path | Selector + adapter improvements | C2 span ≥ 25% (interim) | Plateau 3 runs → escalate selector constraints |
| B3 | Verified record → note rendering v0 | `nanoscribe/render.py` (new) | Section completeness, unsupported-claim rate | Any unsupported critical claim → fail |
| B4 | End-to-end verifier on rendered note | Fabric + nanoscribe evaluator | Review burden @ fixed risk | Beat classical baseline on U |
| B5 | Verifier hard set expansion | 500 → 2000 deterministic cases | baseline_accuracy ≥ 0.95 | Already at 1.0 @ 500 — expand only if discriminative |

**Compute:** B1, B4, B5 = **local CPU**. B2 smoke = local; C2 full = **RunPod serverless** (routine budget). B3 = local.

### Loop C — Product

**Question:** Do clinicians accept the output with low edit burden?

| Exp ID | Hypothesis | Deliverable | Metric | Stop condition |
|--------|------------|-------------|--------|----------------|
| C1 | External benchmark adapter | MTS-Dialog or ACI-Bench loader (no PHI in repo) | Same evaluator metrics on holdout | Adapter + 50-case pilot |
| C2 | Human eval pilot | Preregistered protocol + rubric | Edit distance, CER, time-to-note | n≥20 encounters |
| C3 | Risk–coverage product curve | Abstention policy sweep | U vs coverage | Dominates classical at ≥80% coverage |
| C4 | Clinician workflow smoke | NanoScribe draft UI or export format | Qualitative blockers | Not a gate for v2 |

**Compute:** C1 preprocessing = local. C2 = **owner-authorized** (may need secure environment; no PHI in git). C4 = local.

---

## 5. Sequencing and gates

```text
NOW (v2 entry)
  │
  ├─ Loop B: B1 schema formalization ────────────────────┐
  ├─ Loop B: B2 span transport on managed student ─────┤
  └─ Loop B: B3 note rendering v0 ─────────────────────┤
                                                        ▼
INTERIM GATE (automatic)
  C2 exact_gold_span ≥ 0.25 AND assertion ≥ 0.90
  AND note render: zero unsupported critical claims on smoke
                                                        │
  ├─ Loop A: A2 QLoRA canary (if assertion gap remains) ─┤
  ├─ Loop C: C1 external benchmark pilot ──────────────┤
  └─ Loop B: B4 end-to-end classical comparison ─────────┤
                                                        ▼
PRE-HUMAN GATE
  External pilot + classical+verify baseline beaten on U
                                                        │
  └─ Loop C: C2 human eval pilot (prereg required) ────┤
                                                        ▼
P1 MASTERY DECISION (owner + human protocol)
```

**Deferred without new evidence:** native 300M promotion, hybrid native–student, learned verifier training, P4 CUAD integration, Kimi-blocking waves.

---

## 6. Next three bounded engineering tasks

Owner can authorize these independently; each has a clear done-when and is mostly local.

### Task 1 — Encounter representation schema v0 (EXECUTION_PLAN B1) — **DONE**

- **Done when:** JSON Schema exported from `nanoscribe/encounter.py`; `nanoscribe/test_encounter_v0.py` + docs synced; `check_docs_integrity` passes
- **Compute:** Local CPU
- **Artifact:** `nanoscribe/schemas/encounter_v0.schema.json` (wire shape; semantic invariants via `EncounterRecord.from_dict`)

### Task 2 — Span/evidence transport on managed student path (EXECUTION_PLAN B2)

- **Done when:** C2 re-run with improved ConstrainedSelector / tool-calling path shows ≥25% exact_gold_span OR documents why ceiling is teacher-limited
- **Compute:** RunPod serverless ephemeral (routine budget); manifest required
- **Artifact:** `artifacts/campaign/span_transport_v2.json`

### Task 3 — Verified record → note rendering v0 (EXECUTION_PLAN B3)

- **Done when:** `EncounterRecord` → sectioned note string with claim IDs; verifier flags unsupported claims; smoke test on 3 contract encounters
- **Compute:** Local CPU
- **Artifact:** `nanoscribe/render.py` + `nanoscribe/test_note_render.py`

---

## 7. Cross-worktree pointers (selective port only)

Do **not** wholesale merge `frontier/nanoscribe-core-v1`. Allowed imports per integration manifest:

| Worktree path | Useful artifact | Port disposition |
|---------------|-----------------|------------------|
| `/Users/mac/Projects/nano-lm-nanoscribe/nano_ai/span_contract_v2/` | Naming, types | Already partially lifted → encounter.py |
| `/Users/mac/Projects/nano-lm-nanoscribe/benchmarks/` | Benchmark constitution, CUAD adapters | **Defer** — P4 reconnaissance only |
| `/Users/mac/Projects/nano-lm-nanoscribe/nano_ai/training/` | CUAD / P4 training | **Reject** for P1 |

---

## 8. Honest distance from "best scribe"

| Dimension | Distance | What would move the needle |
|-----------|----------|----------------------------|
| Evidence fidelity | **Large** — 11% span transport vs ~85% target | Constrained selection + teacher distillation on span failures |
| Clinical note quality | **Unknown** — no note renderer, no human eval | B3 rendering + C2 human pilot |
| External validity | **Large** — no public benchmark runs | C1 adapter + same evaluator |
| Model efficiency | **Early** — native 100M not viable for decode | Fix decode or route through managed student |
| Safety / verification | **Moderate** — hard verifier strong; not end-to-end | B4 classical+verify comparison on full pipeline |

**Realistic assessment:** Campaign v1 established measurement infrastructure and identified the bottleneck. Campaign v2 must convert that into **2–3× span transport improvement** and a **verifiable note output** before external or human eval is worth the cost. Claiming "world-class" today would be false.

---

## Verification

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 -m pytest nanoscribe/test_encounter_v0.py nanoscribe/test_evidence_transport.py -q
```
