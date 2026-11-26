# Roadmap

The current objective is a general-purpose frontier model.

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
→ P1 scribing application (historical)
→ general-purpose language model research (current)
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

## Current milestones

Recover and measure the native checkpoints, compare candidate architectures at matched budgets, then scale or adapt the candidate that survives replication. The [execution plan](EXECUTION_PLAN.md) defines the measurements and decisions. Model and systems results have separate evidence requirements.

[Earlier scribing program](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/46c65f0044e81ac73251e18a5a7d69a7299a8079/docs/ROADMAP.md)
