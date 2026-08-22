# Failure to Architecture

Major historical failures and **architectural lessons that survive**. Verdicts are not rewritten; consequences are stated for today's Nano Core program.

## Scribe track

| Failure | What happened | Architectural lesson |
|---------|---------------|----------------------|
| **Scribe v1** | Faithfulness gate FAIL (recall 74%, halluc 14%) | Position-anchored extraction; need diversity + verification axis |
| **Scribe v2** | FAIL by 1.5 pts on halluc bar | Template diversity alone insufficient for OOD |
| **Stage G** | Residual hallucination → 0% with review load | Verification layer can catch unverifiable claims — abstain/review is load-bearing |
| **Stage A** | Presented precision 100% on synthetic dist | Scoped verifier relation works; not open-world elimination |
| **Stage C** | Held-out gap unchanged | Curriculum does not fix OOD symbolic emission |
| **Stage S** | First model-side PASS; gap ~23 pts unchanged | Scale does not retire verification; tail failures persist |

## Paper α / copying research

| Failure | Lesson |
|---------|--------|
| **Held-out value copying** | Small transformers converge to closed-set strategies on low-diversity regimes |
| **Pointer/copy head (P2)** | OOD gap is not an output-mixture problem; source selection does not generalize |
| **Slot diversity (supported)** | Diversity can move measured gaps — regime matters |
| **C-1b, C-3 (refuted)** | Specific interference hypotheses closed for tested designs |

## Utility kill gates

| Gate | Verdict | Scoped meaning |
|------|---------|----------------|
| **E1** | KILL H-substrate | On **old closed scribe task** under frozen U, classical beats official LoRA M0 — generative LM not preferred **for that task** |
| **E4** | KILL on R★ v1 | Generative+verify loses on **tested R★** — not a kill of entire Nano program |

Do not cite E1/E4 as "AI doesn't work." Cite as **routing evidence** for smallest sufficient solver.

## Wedge / document intelligence

| Failure class | Lesson |
|---------------|--------|
| **Over-abstention** | Fail-closed must still surface nearest evidence and scope hints — product failure, not virtue |
| **LM probe not indicated** | Classical sufficient on current snapshots — do not default to generative escalation |

## Pretraining / transfer (H6)

| Work | Lesson |
|------|--------|
| **Span-port / transfer curve** | Compact pretrained models are points on capability frontier — not permanent Nano definition |
| **Register / seed distribution** | Mechanism experiments inform design; teachers used for ceilings and distillation |

## Invalid assumptions (retired)

- Nano = one generative model
- Passing a synthetic gate = clinical readiness
- More LM on the old closed task changes the roadmap post-E1
- Wedge replaces scribing — it is a **supporting subsystem** for verified information

## Regression discipline

Each architectural response should add:

- a test or harness pin
- a schema or verifier invariant
- an entry in evaluation docs — not prose-only "lessons learned"
