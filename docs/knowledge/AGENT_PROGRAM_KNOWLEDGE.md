# Agent program knowledge (A→Z)

**Read this first** for autonomous engineering on Nano P1. Typed authority: [PROJECT_AUTHORITY.md](../PROJECT_AUTHORITY.md). Live gates: [ACTIVE_NOW.md](../ACTIVE_NOW.md) · [PROGRAM_CHECKPOINTS.json](PROGRAM_CHECKPOINTS.json).

---

## A — Authority (what wins)

```text
tagged artifacts / ledger / RESULT_*     →  empirical truth (wins over docs)
PROJECT_CHARTER / CAPABILITY_LADDER      →  mission (not E1/E4 kill scope)
SYSTEM_ARCHITECTURE + NANOSCRIBE           →  structure
ACTIVE_NOW.json + ACTIVE_NOW.md          →  current status (must agree)
EXECUTION_PLAN                           →  bounded tasks
artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md  →  paid compute (wins over chat)
code + tests                             →  implementation reality
```

**Never:** let charter or ambition docs override measured results. **Never:** modify Evidence Core in doc/campaign PRs.

---

## B — Mission (one paragraph)

Build the **smallest useful reliable intelligence** that constructs a **faithful evidence-grounded representation** of messy input, compresses without corrupting, maintains state across time, and eventually reasons/plans/acts with provenance.

**Near-term product:** P1 **NanoScribe** = Nano Core + Medical DomainPack + **P1 only**. Full medical intelligence is **P2–P9**, earned in order.

**Macro-phases:** P1 Scribing → P2–P3 Summarization/Charting → P4–P9 Intelligence expansion.

---

## C — Campaign (how paid work runs)

**Mental model:** RunPod = multi-surface **research OS** (API → Hub Serverless → Flash → Template → Pod). Not “rent a GPU.”

| Rule | Detail |
|------|--------|
| No manifest | No paid compute |
| Wallet | `runpodctl user` — physical ceiling; `campaign_remaining = min(authorized, balance − $10)` |
| Routine RunPod | **ALLOWED** within active experiment budget |
| Materially costly | Experiment-scoped authorization |
| Confirmatory / evidential | Prereg + experiment-scoped |
| PHI / private data | **NOT AUTHORIZED** |
| Idle burn | Forbidden — scale to zero, delete ephemeral endpoints |
| Pod terminate | Pull artifacts **before** terminate (`torch.load` / adapter verify) |

**Authority chain:** [CAMPAIGN_AUTONOMOUS_EXECUTION.md](../../artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md) → manifest JSON → `campaign_control_plane.py inventory`.

**v2 plan:** [ACCELERATED_RESEARCH_CAMPAIGN_V2.md](../ACCELERATED_RESEARCH_CAMPAIGN_V2.md) · manifest `frontier/accelerated_research_campaign_v2.json`.

---

## D — Data boundaries

| Class | Policy |
|-------|--------|
| Public / licensed / de-identified | OK with registry entry |
| Private owner material | **NOT AUTHORIZED** |
| PHI | **NOT AUTHORIZED** |
| `p1_screening_eval_v1` | **FROZEN** — never train on it |
| `p1_distill_train_v1` | Distillation train — disjoint from screening |

---

## E — Encounter representation (truth object)

**Schema:** `nano.encounter.v0` in `nanoscribe/encounter.py`  
**JSON Schema:** `nanoscribe/schemas/encounter_v0.schema.json`

```text
immutable source → EncounterRecord → CandidateAtom proposals
→ ConstrainedSelector → verified record → note (view, not truth)
```

**Primary model interface:** structured `CandidateAtom` JSON. Tool-call path (`submit_candidate_atoms`) is equivalent after parse. See [TOOL_CALLING.md](../infrastructure/TOOL_CALLING.md).

**Binding bottleneck (measured):** **exact gold span transport** ~11% best (managed ref C2). Assertion state can be 100% while spans fail — do not confuse them.

---

## F — Failure lessons (do not repeat)

| Failure | Lesson |
|---------|--------|
| Scribe v1/v2 faithfulness | Template/OOD limits exposed |
| Paper α held-out copy | Field-localized failure — measure tails |
| E1 classical win | **Scoped** to old closed task — not program kill |
| E4 classical win | **Scoped** to tested R★ — not program kill |
| July-31 IDLE / TRAINING NOT_AUTHORIZED | **Historical** — RunPod is active now |
| Native 100M smoke | Loss ↓ but garbled decode — architecture hypothesis not validated |

Full map: [FAILURE_TO_ARCHITECTURE.md](../FAILURE_TO_ARCHITECTURE.md).

---

## G — Git / branch truth

- **`origin/master`** = development integration truth (`dc4c1f9` — PRs #37–#51).
- **Never** trust stale local `master` if it diverges (may include `9fe5b6b6` evidence reconciliation).
- **Frontier branch** `frontier/accelerated-research-campaign-v2` carries Nano Core (`nano/`) + campaign v2 + tool/agent stack. Local may be ahead of origin.
- **Cross-branch, not integrated:** `nano_ai/`, `artifacts/nano_h6/` — cite as lineage only.
- **Selective port** to master — no wholesale merge of parked worktrees.

---

## H — Harness tracks (P1 measurement)

| Track | Role | Notes |
|-------|------|-------|
| Fixture | CI deterministic | Zero cost |
| Compact 1.5B | Historical continuity | Not the strategic bet |
| Serverless Qwen3.8-27B | Strong control | Primary regression surface |
| Managed ref (Qwen3-32B-AWQ) | Capability ceiling | C1/C2 winner |
| Native 30–100M | Architecture screen | Not Axolotl; decode must work |
| Frontier teacher / API | Ceiling probes | Kimi may 500 — have fallback |

Code: `nanoscribe/tracks.py`, `nanoscribe/harness.py`.

---

## I — Integration map (this tree)

| Path | Role |
|------|------|
| `nano/` | **Nano Core** — contracts, kernel, ontology, capability registry, DomainPack-0 (SLW) |
| `nanoscribe/` | P1 product stack — encounter v0, adapters, harness, campaign |
| `artifacts/campaign/` | Manifests, spend, checkpoints, round summaries |
| `artifacts/nano_clin_001/`, `artifacts/nano_slw_001/` | Core slice measurements (deterministic, no model) |
| `fabric/` | Verification harness |
| `wedge_v1/` | Classical-first document QA (supporting) |
| `pretrain/`, `sft/`, `scribe/` | Mechanism-era experiments |
| `trajectory/` | Frozen result JSONs |
| `papers/` | Science — not current execution authority |
| `docs/` | Current program truth |
| `frontier/` | Branch configs + [NEXT.md](../../frontier/NEXT.md) |

Detail: [NANOSCRIBE.md](../subsystems/NANOSCRIBE.md) · [MODEL_RESEARCH_PROGRAM.md](../research/MODEL_RESEARCH_PROGRAM.md).

---

## J — Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/campaign_control_plane.py inventory` | Pre-wave control plane (no spend) |
| `scripts/campaign_fanout.py` | Structured + tool fan-out |
| `scripts/tool_call_harness.py` | Offline tool-call fixtures |
| `scripts/agent_canary_bench.py` | Agent platform canary |
| `scripts/ci_nanoscribe.sh` | Full nanoscribe CI lane |
| `scripts/check_active_now.py` | ACTIVE_NOW md/json sync |
| `scripts/check_docs_integrity.py` | Docs + knowledge anchors |

---

## K — Anti-patterns (stop doing these)

1. Treating Qwen adapter as “Nano” — Qwen is **control/baseline**, Nano is the program.
2. Claiming P1 mastered from synthetic benchmarks only.
3. Training on `p1_screening_eval_v1`.
4. Raw Pods for baseline inference when Serverless works.
5. Axolotl for Native Nano training.
6. Terminating pods before artifact pull.
7. Stale Hub listing IDs without `discover_hub_catalog`.
8. Reopening doc-reset governance instead of shipping measured increments.
9. Equating NanoScribe with full medical intelligence.
10. Modifying Evidence Core in campaign/doc PRs.

---

## L — Verify before you claim done

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
bash scripts/ci_nanoscribe.sh
```

Optional fast path:

```bash
python3 -m pytest nanoscribe/test_encounter_v0.py nanoscribe/test_evidence_transport.py nanoscribe/test_tool_calling.py -q
```

---

## M — What “done” means (Phase C v2)

See [PROGRAM_CHECKPOINTS.json](PROGRAM_CHECKPOINTS.json). Summary:

- **Done:** control plane, tool calling, agent canary, harness CI.
- **Done (frontier Core):** NANO-CLIN-001 ledger slice; NANO-SLW-001 synthetic world (0 undeclared error vs silent-resolution control); `MTA-EPISTEMIC` ranked next-information need (`nano/needs.py`, KIND beats random-key control).
- **In progress:** native extended viability; student span-transport improvement.
- **Pending:** selective master port, note rendering, external eval, human eval. Do not conflate Core substrate proofs with P1 mastery.

**Campaign success criterion (engineering):** measurable **≥2× improvement** in exact gold span transport **or** documented teacher ceiling + pivot — with committed artifact JSON, not narrative.

---

## N — Naming

```text
Nano Core + Medical DomainPack + P1  =  NanoScribe
P1–P3 earned  =  longitudinal medical documentation intelligence (not yet)
P4–P9  =  synthesis → reasoning → planning → action → adaptation
```

---

## O — Owner gates (standing)

Materially costly runs outside budget, confirmatory without prereg, PHI/private data, protected tag moves, publication/clinical claims, master merge of frontier without review.

---

*This file is agent-optimized narrative. Update when integration base or campaign phase changes. Machine gates: PROGRAM_CHECKPOINTS.json.*
