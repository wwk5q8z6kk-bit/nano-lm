# Reconciliation Report — Canonical Program Reset

**Date:** 2026-08-22  
**Branch:** `docs/canonical-program-reset`  
**Authority:** Owner top-down program definition + documentation migration instruction

---

## 1. What the project originally was

A **3.15M-parameter from-scratch LM** built end-to-end on Apple Silicon as an **evaluation-gated development testbed**. The immediate task was medical ambient scribing: dialogue → structured summary under faithfulness gates. The repo organized around training stages (pretrain, SFT, DPO, RLVR), scribe gates, and Paper α's held-out copying measurement program.

## 2. What it became (from Git history)

| Era | What happened |
|-----|---------------|
| 2025–2026 Q1 | Full LLM stack at legible scale; scribe v1/v2 honest FAILs |
| Stage G/A/C/S | Verification architecture; scale test; held-out gap persists |
| Paper α | Field-localized OOD copying failure documented; E1 utility kill |
| Post-α freeze | E1 KILL (classical beats LoRA on old closed task); E3 rubric audit; E4 KILL on tested R★ |
| Strategic reset | Product pivot to Wedge v1 — local verified document intelligence |
| H6 / Nano AI | Pretrained span-port on ~1.5B; from-scratch H1–H5 rejected for production path |
| Frontier branch | Active product discovery under wedge mandate |
| **This reset** | **Nano Core + capability ladder (P1–P9)** — medicine-first DomainPack, docs authority in `docs/` |

The repo briefly held **three competing authorities**: README (3.15M testbed), `papers/PROGRAM_AUTHORITY` (Wedge-as-frontier), and frontier mandate docs. This reset unifies them under `docs/`.

## 3. What major failed experiments taught us

See [`FAILURE_TO_ARCHITECTURE.md`](../FAILURE_TO_ARCHITECTURE.md). Summary:

- **Scribe v1/v2:** OOD hallucination and position-anchored extraction — diversity alone insufficient
- **Stage C/S:** Scale moves average bars, not necessarily held-out gap
- **Pointer/copy (P2):** OOD gap is not output-mixture; source selection doesn't generalize
- **E1:** On old closed task under frozen U, generative LM not preferred substrate
- **E4:** On tested R★, generative+verify loses — routing evidence, not program death
- **Wedge over-abstention:** Fail-closed must still surface evidence — product failure class
- **LM probe:** Not indicated on current snapshots — classical sufficient for tested workflows

## 4. Invalid assumptions (retired)

| Assumption | Status |
|------------|--------|
| Nano = 3.15M LM | **Invalid** — experimental foundation only |
| Nano = Wedge product | **Invalid** — Wedge is supporting subsystem |
| One generative model is the system | **Invalid** — model/software co-design |
| E1/E4 kill entire program | **Invalid** — scoped to tested regime/utility |
| Synthetic benchmark pass = clinical validation | **Invalid** |
| `papers/` is current product management | **Invalid** — science only |
| `frontier/` is canonical authority | **Invalid** — branch notes only |
| AZ_EXECUTION_PLAN is master roadmap | **Invalid** — post-E1 historical reaction |

## 5. Architecture that survives

**Model layers:** mechanism (3M–100M scratch) · compact production (pretrained/adapted) · teachers (ceilings, distillation, judging)

**System layers:** retrieval · memory · schemas · constrained decoding · verifiers · routing · human review

**Output contract:** present · abstain · review — with evidence spans and provenance

**Product shape:** Nano Core + DomainPack (medical first)

## 6. What Nano now is

> Build the smallest useful, reliable intelligence system that constructs faithful evidence representations, compresses without corruption, maintains state across time, and eventually reasons, plans, and acts — preserving provenance and uncertainty.

**Current frontier:** P1 Master Scribing (medical DomainPack)

**Sequence:** P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9

## 7. Repository changes

### Created

```text
docs/                          # canonical program truth (full tree)
docs/archive/                  # historical strategy copies
docs/domains/medical/          # DomainPack specs
docs/research/                 # model + system research programs
docs/subsystems/               # WEDGE, VERIFICATION
docs/infrastructure/           # RUNPOD, REPRODUCIBILITY
scripts/check_active_now.py    # ACTIVE_NOW md/json consistency
.github/workflows/ci.yml       # CI gate
papers/PROGRAM_AUTHORITY.md    # superseded stub
papers/PROJECT_EVOLUTION_PLAN.md  # superseded stub
frontier/DEVELOPMENT_PLAN.md   # branch-history pointer
```

### Rewritten

```text
README.md                      # mission-first public overview
AGENTS.md                      # agent ops → docs/ authority
papers/README.md               # science layer index
papers/STRATEGIC_RESET.md      # superseded stub
papers/AMBITION.md             # superseded stub
papers/AZ_EXECUTION_PLAN.md    # superseded stub
papers/WEDGE_V1.md             # superseded stub
papers/EXECUTION_QUEUE.md      # superseded stub
ACTIVE_MANDATE.md              # superseded stub
papers/ACTIVE_MANDATE.md       # superseded stub
CLAUDE-PROGRESS.md             # superseded banner added
```

### Archived (content preserved)

```text
docs/archive/AZ_EXECUTION_PLAN_POST_E1_20260731.md
docs/archive/LEGACY_STRATEGY_INDEX.md
```

### Unchanged (evidence-protected)

```text
papers/EMPIRICAL_FOUNDATION.md
papers/EVIDENCE_LEDGER.md (+ JSON)
papers/PREREG_*, papers/RESULT_*
trajectory/results_*.json (primary record)
freeze tags, SHA manifests
audit/discussion-to-implementation/* (historical audit trail)
```

## 8. Broken references corrected

| Old authority | New authority |
|---------------|---------------|
| `papers/STRATEGIC_RESET.md` | `docs/PROJECT_CHARTER.md` |
| `papers/PROGRAM_AUTHORITY.md` | `docs/PROJECT_AUTHORITY.md` |
| `papers/AZ_EXECUTION_PLAN.md` | `docs/archive/AZ_EXECUTION_PLAN_POST_E1_20260731.md` |
| `papers/AMBITION.md` | `docs/PROJECT_CHARTER.md` |
| `papers/WEDGE_V1.md` | `docs/subsystems/WEDGE.md` |
| `papers/EXECUTION_QUEUE.md` | `docs/EXECUTION_PLAN.md` |
| `papers/PROJECT_EVOLUTION_PLAN.md` | `docs/ROADMAP.md` |
| `ACTIVE_MANDATE.md` | `docs/ACTIVE_NOW.md` |
| `frontier/DEVELOPMENT_PLAN.md` | `docs/EXECUTION_PLAN.md` |
| `CLAUDE-PROGRESS.md` | `docs/ROADMAP.md` (narrative only) |

Live entry points updated: `README.md`, `AGENTS.md`, `papers/README.md`, `frontier/NEXT.md`

## 9. Current executable work (one bounded frontier)

**Owner review of this branch** (file map + README) before merge to `master`.

Then **P1 scribe engineering:**

1. Encounter representation schema v0 (entity/event/evidence refs)
2. Span/evidence bottleneck on held-out medical dialogue (no PHI in repo)
3. Verified record → note rendering (note as view, not truth)

## 10. Future sequence

```text
P1 Scribing mastery (external + human eval)
→ P2 Summarization (hierarchical over verified state)
→ P3 Charting (longitudinal identity)
→ P4 Synthesis → P5 Questioning → P6 Reasoning
→ P7 Planning → P8 Tools/Action → P9 Adaptation
```

P2/P3 interfaces specified in `docs/domains/medical/SUMMARIZATION_AND_CHARTING.md` — not implemented until P1 exit.

## 11. Tests

```bash
python3 scripts/check_active_now.py   # ACTIVE_NOW_OK
python3 -m pytest fabric/test_fabric.py -q   # 8 passed
```

## 12. Remaining owner gates

| Gate | Status |
|------|--------|
| Merge doc reset to `master` | **Pending owner review** |
| Paid RunPod / cloud GPU | NOT_AUTHORIZED |
| PHI / private corpus in git | FORBIDDEN |
| Protected evidence tag moves | NOT_AUTHORIZED |
| Publication / clinical claims | NOT_AUTHORIZED |
| LM training runs | NOT_AUTHORIZED |
| P4 CUAD scoring (Nano AI track) | Requires prereg freeze (separate from this doc reset) |

---

*This report is historical record of the documentation migration. Current execution state: [`ACTIVE_NOW.md`](../ACTIVE_NOW.md).*
