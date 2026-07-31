# Technology Roadmap

**Layer 2 — Engineering ambition.** Conditional. Non-evidential.  
**Adopted:** 2026-07-31  
**Constitution:** `LABORATORY_CONSTITUTION.md`

> If unlimited future evidence eventually supported it, what should we be able to build?  
> Nothing here is “next.” Promotion requires `DECISION_GATES.md`.

## North-star system (aspirational)

A factorized cognitive stack:

```
Interfaces ──► Intent / Policy ──► Planning / Routing
                     │                    │
                     ▼                    ▼
              Memory fabric ◄──────► Retrievers / Tools
                     │                    │
                     ▼                    ▼
         Learning cores (classical, LM, hybrid, synthesizers)
                     │
                     ▼
         Verification / Calibration / Abstention
                     │
                     ▼
         Execution sandbox + Observability + Replay
```

## Capability catalog (dream list — unimplemented unless evidenced)

| Capability | Why it belongs in the dream | Evidence today |
|------------|-----------------------------|----------------|
| Distributed reasoning | Scale cognition without single-context monoliths | None as product |
| Typed memory (episodic/semantic/graph/causal/user) | Persist validated state; slot-aware stats | Speculative / partial Fabric provenance |
| Verification & abstention | Trust from checks, not eloquence | Fabric slice (scoped synthetic) |
| Planning & policy engine | Controllable action selection | Unimplemented |
| Tool use & retrieval | Classical strengths + gen where needed | E1 shows classical can dominate closed worlds |
| Provenance & contradiction handling | Auditability | Partial in Fabric schemas |
| Agents & orchestration | Collaboration under contracts | Unimplemented |
| Multimodal I/O | Real-world interfaces | Unimplemented |
| Simulation / world models | Counterfactual testing | Unimplemented |
| Program synthesis / compilers | Lower plans to deterministic runtimes | Unimplemented |
| Observability, debugging, replay | Scientific + production hygiene | Partial offline tests/logs |
| Security, permissions, alignment | Safe writes and tool use | Unimplemented |
| Hardware-aware serving | L/C realism in utilities | Partial E1 L/C schema |

## Explicit non-claims

- This roadmap is **not** NanoScribe product authorization.  
- Fabric ≠ this stack.  
- E1 KILL does **not** remove rows from this catalog; it removes one justification path for generative extraction on one task.  
- Building any row requires a Decision Gate pass and an Execution Queue entry.

## Conditional architecture families

1. **Classical-first IE** — default when E1-like utilities recur.  
2. **Hybrid router** — classical default, generative only in gated failure regimes (needs E4-class evidence).  
3. **Verified generative proposer** — only if SURVIVE under frozen regime utility + feasibility gate.  
4. **Compiler/runtime cognitive OS** — long-horizon; requires repeated wedge wins across programs C/E/I/O.

## Relation to legacy docs

`MASTER_PLAN.md` and `NANOSCRIBE_VNEXT.md` retain historical architecture prose.  
This file is the **living Layer-2 catalog**. Legacy phases 3–4 remain `STOP` as execution, not as imagination.


## Benchmark infrastructure (aspirational; not queued beyond Program 0)

Conditional modules for **this repo’s** harness — not a separate institution:

1. Pinned adapters + content-addressed runs (Program 0).
2. Later (separate auth): checkpoint census, broader suites, multi-axis boards.

Explicit non-claim: `benchmarks/` scaffolding ≠ leadership or Layer-1 evidence.
