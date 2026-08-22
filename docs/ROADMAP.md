# Roadmap

One developmental arc — not contradictory projects.

## Narrative arc

```text
3.15M from scratch (evaluation testbed)
→ SFT / DPO / RLVR behavioral gates
→ medical scribing (faithfulness gates)
→ held-out copying failure (Paper α)
→ scale / diversity / adaptation experiments
→ verification architecture (Stage G/A, Fabric)
→ utility kill gates (E1 KILL on old closed task)
→ regime R★ + E4 KILL (generative+verify lost on tested R★)
→ Wedge v1 (local verified document intelligence)
→ pretrained transfer / span grounding (H6 program)
→ Nano Core + capability ladder (this docs reset)
→ P1 Master Scribing (current frontier)
```

## What history taught (compressed)

| Era | Lesson |
|-----|--------|
| Scribe v1/v2 | Position-anchored extraction fails OOD; template diversity insufficient alone |
| Stage G/A | Verification can reach presented precision on synthetic distribution — scoped, not open-world |
| Stage C/S | Scale moves average bars but not necessarily held-out gap |
| Paper α / pointer | OOD value-copying gap is structural for small transformers on low-diversity regimes |
| E1 | On **old closed scribe task** under frozen U, classical beats official LoRA M0 — generative LM not preferred substrate **for that task** |
| E4 | On **tested R★**, classical still wins — generative+verify track STOP for that regime |
| Wedge | Classical-first local document intelligence works; LM probe not indicated on current snapshots; over-abstention is a product failure class |

**Invalid assumption to retire:** "Nano = 3.15M LM" or "Nano = Wedge only" or "one generative model is the system."

**Surviving architecture:** compact models + retrieval + memory + schemas + verifiers + routing + human review.

## Phase map (forward)

| Phase | Focus | Status |
|-------|-------|--------|
| **0 — Evidence foundation** | Paper α, E1/E3/E4, freeze tags | Done (protected) |
| **1 — Doc authority reset** | `docs/` canonical tree, README rewrite | **In progress** |
| **2 — P1 scribe evidence** | Encounter representation, span bottleneck, external eval design | Next engineering |
| **3 — P1 scribe mastery** | Exit gate: metrics + blinded human eval | Gated |
| **4 — P2 summarization** | Hierarchical compression over verified state | Spec only until P1 exit |
| **5 — P3 charting** | Longitudinal identity and timelines | Spec only until P1 exit |
| **6+ — P4–P9** | Synthesis → adaptation | Architectural requirements only |

## What we are not doing now

- Re-running E1 on the old task
- Claiming clinical validation from mock/synthetic benchmarks alone
- Paid RunPod without explicit authority
- Merging doc reset to `master` without owner review

## Historical documents

See [archive/LEGACY_STRATEGY_INDEX.md](archive/LEGACY_STRATEGY_INDEX.md).
